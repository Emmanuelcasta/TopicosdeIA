"""
Tool Registry: herramientas tipadas, con esquema generado y ejecución segura.

Decisión de diseño: el esquema JSON de cada herramienta NO se escribe a mano.
Se deriva de la firma de la función (tipos) y de su docstring (descripciones),
con el mismo formato que espera la plantilla de chat de Qwen2.5 para tool
calling (`{"type": "function", "function": {...}}`). Así la descripción que ve
el modelo y el código que se ejecuta no pueden divergir, y añadir una
herramienta es escribir UNA función decorada: el núcleo del agente no cambia.

El registro separa tres responsabilidades que el agente nunca mezcla:
  esquemas()  -> lo que se le MUESTRA al modelo
  validar()   -> convertir los argumentos que PROPONE el modelo a los tipos reales
  ejecutar()  -> correr la función y devolver un resultado tipado (nunca lanza)
"""

from __future__ import annotations

import inspect
import json
import re
import time
import typing
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Callable, Literal, get_args, get_origin


class ErrorHerramienta(Exception):
    """Error esperado de dominio (p. ej. división entre cero). Su mensaje se
    devuelve al modelo como observación, para que lo explique o corrija."""


# --------------------------------------------------------------------------
# Tipos de datos del protocolo agente <-> herramientas
# --------------------------------------------------------------------------

@dataclass
class LlamadaHerramienta:
    nombre: str
    argumentos: dict
    formato: str = "nativo"        # nativo (<tool_call>) | recuperado (JSON suelto)


@dataclass
class ResultadoHerramienta:
    herramienta: str
    argumentos: dict
    ok: bool
    resultado: str | None = None   # texto legible para el modelo
    valor: str | None = None       # forma canónica ("73/72", "9.316") para métricas
    error: str | None = None
    tipo_error: str | None = None  # desconocida | argumentos | dominio | interno
    ms: float = 0.0
    extra: dict = field(default_factory=dict)

    def como_observacion(self) -> str:
        if self.ok:
            d = {"ok": True, "resultado": self.resultado}
            if self.valor is not None and self.valor != self.resultado:
                d["valor"] = self.valor
        else:
            d = {"ok": False, "error": self.error}
        return json.dumps(d, ensure_ascii=False)

    def a_dict(self) -> dict:
        return asdict(self)


@dataclass
class Herramienta:
    nombre: str
    descripcion: str
    familia: str
    funcion: Callable
    parametros: dict
    requeridos: list
    comodin_de: tuple = ()          # familias que esta herramienta también puede cubrir

    def esquema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.nombre,
            "description": self.descripcion,
            "parameters": {"type": "object", "properties": self.parametros,
                           "required": self.requeridos},
        }}


# --------------------------------------------------------------------------
# Esquema a partir de tipos y docstring
# --------------------------------------------------------------------------

_TIPOS_JSON = {int: "integer", float: "number", str: "string", bool: "boolean"}


def _esquema_tipo(anotacion) -> dict:
    origen = get_origin(anotacion)
    if origen is Literal:
        return {"type": "string", "enum": list(get_args(anotacion))}
    if origen in (list, typing.List):
        (interno,) = get_args(anotacion) or (float,)
        return {"type": "array", "items": _esquema_tipo(interno)}
    if anotacion in _TIPOS_JSON:
        return {"type": _TIPOS_JSON[anotacion]}
    raise TypeError(f"tipo no soportado en una herramienta: {anotacion!r}")


def _parsear_docstring(doc: str) -> tuple[str, dict]:
    """Formato Google: descripción, y luego una sección 'Args:' con 'nombre: texto'."""
    doc = inspect.cleandoc(doc or "")
    partes = re.split(r"^\s*Args:\s*$", doc, maxsplit=1, flags=re.MULTILINE)
    descripcion = " ".join(partes[0].split())
    args: dict = {}
    if len(partes) > 1:
        actual = None
        for linea in partes[1].splitlines():
            m = re.match(r"^\s*(\w+)\s*:\s*(.*)$", linea)
            if m:
                actual = m.group(1)
                args[actual] = m.group(2).strip()
            elif actual and linea.strip():
                args[actual] += " " + linea.strip()
    return descripcion, args


# --------------------------------------------------------------------------
# Conversión de argumentos: lo que escribe un LLM pequeño no siempre es JSON limpio
# --------------------------------------------------------------------------

def a_fraccion(valor: Any) -> Fraction:
    """Número exacto a partir de lo que proponga el modelo: 12, 2.35, "2,35",
    "18.500"?, "3/4", "1 1/2"... Se trabaja con Fraction para no introducir
    errores de coma flotante en una herramienta cuyo propósito es no equivocarse."""
    if isinstance(valor, bool):
        raise ValueError("un booleano no es un número")
    if isinstance(valor, int):
        return Fraction(valor)
    if isinstance(valor, float):
        return Fraction(repr(valor))
    if isinstance(valor, Fraction):
        return valor
    if not isinstance(valor, str):
        raise ValueError(f"no es un número: {valor!r}")
    t = valor.strip().replace("−", "-").replace(" ", "").replace("%", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):      # 18,500 -> miles
        t = t.replace(",", "")
    elif re.fullmatch(r"-?\d+,\d+", t):                     # 2,35 -> decimal
        t = t.replace(",", ".")
    m = re.fullmatch(r"(-?\d+)\s*\+?\s*(\d+)/(\d+)", valor.strip())  # "1 1/2"
    if m and " " in valor.strip():
        entero, num, den = map(int, m.groups())
        signo = -1 if entero < 0 else 1
        return Fraction(entero) + signo * Fraction(num, den)
    try:
        return Fraction(t)
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError(f"no es un número: {valor!r}") from e


def _convertir(valor: Any, anotacion, nombre: str):
    origen = get_origin(anotacion)
    if origen is Literal:
        opciones = get_args(anotacion)
        v = str(valor).strip().lower()
        if v not in opciones:
            raise ValueError(f"'{nombre}' debe ser uno de {list(opciones)}, no {valor!r}")
        return v
    if origen in (list, typing.List):
        if isinstance(valor, str):
            try:
                valor = json.loads(valor)
            except json.JSONDecodeError:
                valor = [p for p in re.split(r"[;\s]+|,(?=\s)", valor.strip("[] ")) if p]
        if not isinstance(valor, (list, tuple)) or not valor:
            raise ValueError(f"'{nombre}' debe ser una lista no vacía")
        (interno,) = get_args(anotacion) or (float,)
        return [_convertir(v, interno, nombre) for v in valor]
    if anotacion is float:
        return a_fraccion(valor)
    if anotacion is int:
        f = a_fraccion(valor)
        if f.denominator != 1:
            raise ValueError(f"'{nombre}' debe ser entero, no {valor!r}")
        return int(f)
    if anotacion is str:
        if isinstance(valor, (dict, list)):
            raise ValueError(f"'{nombre}' debe ser texto")
        return str(valor)
    if anotacion is bool:
        return bool(valor)
    return valor


# --------------------------------------------------------------------------
# El registro
# --------------------------------------------------------------------------

class RegistroHerramientas:
    def __init__(self):
        self._herramientas: dict[str, Herramienta] = {}

    # ---- alta de herramientas -------------------------------------------
    def herramienta(self, familia: str, comodin_de: tuple = (), nombre: str | None = None):
        """Decorador: `@registro.herramienta(familia="aritmetica")`."""
        def decorar(fn: Callable) -> Callable:
            self.registrar(fn, familia=familia, comodin_de=comodin_de, nombre=nombre)
            return fn
        return decorar

    def registrar(self, fn: Callable, familia: str, comodin_de: tuple = (),
                  nombre: str | None = None) -> Herramienta:
        nombre = nombre or fn.__name__
        if nombre in self._herramientas:
            raise ValueError(f"ya existe una herramienta llamada {nombre!r}")
        descripcion, docs_args = _parsear_docstring(fn.__doc__)
        if not descripcion:
            raise ValueError(f"la herramienta {nombre!r} necesita una descripción (docstring)")
        hints = typing.get_type_hints(fn, include_extras=False)
        parametros, requeridos = {}, []
        for p in inspect.signature(fn).parameters.values():
            if p.name not in hints:
                raise TypeError(f"{nombre}: el parámetro {p.name!r} no tiene tipo")
            esquema = _esquema_tipo(hints[p.name])
            if p.name not in docs_args:
                raise ValueError(f"{nombre}: falta describir {p.name!r} en 'Args:'")
            esquema["description"] = docs_args[p.name]
            if p.default is inspect.Parameter.empty:
                requeridos.append(p.name)
            else:
                esquema["default"] = p.default
            parametros[p.name] = esquema
        h = Herramienta(nombre, descripcion, familia, fn, parametros, requeridos, tuple(comodin_de))
        self._herramientas[nombre] = h
        return h

    # ---- consulta ----------------------------------------------------------
    def __contains__(self, nombre: str) -> bool:
        return nombre in self._herramientas

    def __len__(self) -> int:
        return len(self._herramientas)

    def obtener(self, nombre: str) -> Herramienta:
        return self._herramientas[nombre]

    def nombres(self) -> list[str]:
        return list(self._herramientas)

    def esquemas(self) -> list[dict]:
        return [h.esquema() for h in self._herramientas.values()]

    def familia_de(self, nombre: str) -> str | None:
        h = self._herramientas.get(nombre)
        return h.familia if h else None

    def subconjunto(self, incluir=None, excluir=()) -> "RegistroHerramientas":
        """Vista con parte de las herramientas (p. ej. C4 sin `buscar_documentos`)."""
        nuevo = RegistroHerramientas()
        for n, h in self._herramientas.items():
            if (incluir is None or n in incluir) and n not in excluir:
                nuevo._herramientas[n] = h
        return nuevo

    # ---- validación y ejecución -------------------------------------------
    def validar(self, llamada: LlamadaHerramienta) -> dict:
        """Devuelve los argumentos convertidos o lanza ValueError con un mensaje
        pensado para que el MODELO lo lea y corrija la llamada."""
        h = self._herramientas[llamada.nombre]
        hints = typing.get_type_hints(h.funcion)
        args = llamada.argumentos if isinstance(llamada.argumentos, dict) else {}
        desconocidos = set(args) - set(h.parametros)
        if desconocidos:
            raise ValueError(f"argumentos desconocidos para {h.nombre}: {sorted(desconocidos)}; "
                             f"acepta {list(h.parametros)}")
        faltan = [r for r in h.requeridos if r not in args]
        if faltan:
            raise ValueError(f"faltan argumentos para {h.nombre}: {faltan}")
        return {k: _convertir(v, hints[k], k) for k, v in args.items()}

    def ejecutar(self, llamada: LlamadaHerramienta) -> ResultadoHerramienta:
        t0 = time.perf_counter()
        base = dict(herramienta=llamada.nombre, argumentos=llamada.argumentos)

        def fin(**kw):
            return ResultadoHerramienta(**base, **kw, ms=(time.perf_counter() - t0) * 1000)

        if llamada.nombre not in self._herramientas:
            return fin(ok=False, tipo_error="desconocida",
                       error=f"no existe la herramienta {llamada.nombre!r}. Disponibles: {self.nombres()}")
        try:
            args = self.validar(llamada)
        except (ValueError, TypeError) as e:
            return fin(ok=False, tipo_error="argumentos", error=str(e))
        try:
            salida = self._herramientas[llamada.nombre].funcion(**args)
        except ErrorHerramienta as e:
            return fin(ok=False, tipo_error="dominio", error=str(e))
        except Exception as e:  # noqa: BLE001 — una herramienta nunca tumba al agente
            return fin(ok=False, tipo_error="interno", error=f"{type(e).__name__}: {e}")

        if isinstance(salida, dict):
            return fin(ok=True, resultado=str(salida.get("resultado")), valor=salida.get("valor"),
                       extra={k: v for k, v in salida.items() if k not in ("resultado", "valor")})
        return fin(ok=True, resultado=str(salida), valor=str(salida))
