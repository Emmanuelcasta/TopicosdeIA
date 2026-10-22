"""
Verificador de planteamiento.

Por qué existe: el análisis caso por caso de M3 mostró que, tras entrenar al
agente, **13 de sus 20 fallos ya no son de aritmética**. La herramienta calculó
bien y la respuesta igual salió mal, porque el plan estaba mal o incompleto:

    ari-09  "interés del 1.8% mensual ... en 7 meses"
            multiplicar(2450000, 0.018) -> 44100 ... y respondió 44100.
            El 7 del enunciado NUNCA entró a una herramienta: plan incompleto.

    ari-12  "0.0375 ÷ 0.0015 + √2025"
            calculó solo la raíz (45) y respondió 45: media expresión.

El verificador de `nucleo.py` solo comprueba que la respuesta final coincida con
alguna observación; esos dos casos la pasan. Hace falta mirar el **plan**, no el
último número.

Qué comprueba, y por qué cada cosa:

  ARGUMENTO NO TRAZABLE   Un número que entra a una herramienta debería venir del
                          enunciado, del contexto recuperado o de una observación
                          previa. Si aparece de la nada, el modelo lo inventó.

  DATO SIN USAR           Un número del enunciado que no entró a ninguna
                          herramienta ni aparece en la respuesta. Es la señal de
                          plan incompleto. Severidad media a propósito: hay
                          enunciados con distractores deliberados (`dif-02`), así
                          que esto es una sospecha, no un veredicto.

  RESPUESTA NO TRAZABLE   El valor final no coincide con ninguna observación ni
                          con una derivación de un paso.

  CÁLCULO DE CABEZA       La respuesta muestra una operación (`a × b = c`) que
                          ninguna herramienta ejecutó.

**No es un detector de "respuesta incorrecta".** Un argumento puede ser trazable y
aun así ser el equivocado: en `com-07` el modelo usó 3.5 (que sí estaba en el
documento recuperado) cuando debía usar 1.0. Eso no lo caza ninguna heurística
sintáctica, y se declara como límite en vez de disimularse.

Trabaja sobre `Fraction`, así que no acumula error de coma flotante.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from nucleo import numeros_en, valor_final  # [local]

TOLERANCIA = Fraction(1, 10**6)

# Operaciones que un paso intermedio razonable puede introducir sin que el número
# esté escrito en ninguna parte: pasar a porcentaje, su complemento, redondear.
import math

_UNARIAS = {
    "×100": lambda v: v * 100,
    "÷100": lambda v: v / 100,
    "1−x": lambda v: 1 - v,
    "1+x": lambda v: 1 + v,
    "−x": lambda v: -v,
    # El redondeo contextual es un paso legítimo del dominio: 1250 ÷ 180 = 6.94
    # significa 7 camiones. Sin esto, el verificador se queja de respuestas buenas.
    # El factor de un descuento o un aumento: "20 % menos" -> × 0.80. Son dos
    # operaciones aritméticas pero un solo paso de razonamiento, y el modelo lo
    # escribe así continuamente.
    "1−x÷100": lambda v: 1 - v / 100,
    "1+x÷100": lambda v: 1 + v / 100,
    "⌈⌉": lambda v: Fraction(math.ceil(v)),
    "⌊⌋": lambda v: Fraction(math.floor(v)),
    "≈": lambda v: Fraction(round(float(v), 2)).limit_denominator(10**6),
}


@dataclass
class Hallazgo:
    tipo: str
    detalle: str
    severidad: str = "alta"          # alta | media

    def __str__(self) -> str:
        return f"[{self.severidad}] {self.tipo}: {self.detalle}"


@dataclass
class Revision:
    hallazgos: list = field(default_factory=list)
    disponibles: set = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.hallazgos

    @property
    def graves(self) -> list:
        return [h for h in self.hallazgos if h.severidad == "alta"]

    def mensaje(self) -> str | None:
        """Lo que se le devuelve al modelo para que corrija. None si no hay nada."""
        if not self.hallazgos:
            return None
        lineas = [h.detalle for h in sorted(self.hallazgos, key=lambda h: h.severidad != "alta")]
        return " ".join(lineas[:3])


def _frac(texto: str) -> Fraction | None:
    try:
        return Fraction(texto)
    except (ValueError, ZeroDivisionError):
        return None


def _numeros(texto: str, partes_de_fraccion: bool = False) -> list:
    """Números de un texto como Fraction, sin duplicados y en orden de aparición.

    Con `partes_de_fraccion`, un "2/5" aporta además el 2 y el 5: el enunciado los
    menciona, así que usarlos en un paso intermedio es legítimo.
    """
    vistos, salida = set(), []
    for t in numeros_en(texto or ""):
        f = _frac(t)
        if f is None:
            continue
        candidatos = [f]
        # Solo si la fracción está ESCRITA como a/b en el texto: un decimal como
        # 1.8 es Fraction(9, 5) por dentro, y añadir el 9 y el 5 inventaría
        # números que el enunciado nunca menciona.
        if partes_de_fraccion and "/" in t:
            candidatos += [Fraction(f.numerator), Fraction(f.denominator)]
        for c in candidatos:
            if c not in vistos:
                vistos.add(c)
                salida.append(c)
    return salida


def _cerca(a: Fraction, b: Fraction) -> bool:
    return abs(a - b) <= TOLERANCIA * max(Fraction(1), abs(b))


def derivables(base: set, pasos: int = 1) -> set:
    """Valores alcanzables desde `base` en `pasos` operaciones elementales.

    Un paso = una unaria sobre un valor, o una binaria entre dos valores. Es lo
    que permite aceptar un argumento como `0.018` cuando el enunciado dice `1.8`
    (por ciento), o `0.60` cuando dice `40` de descuento, sin tener que escribirlo.
    """
    alcanzables = set(base)
    for _ in range(pasos):
        nuevos = set()
        for v in alcanzables:
            for fn in _UNARIAS.values():
                try:
                    nuevos.add(fn(v))
                except ZeroDivisionError:
                    pass
        for a in alcanzables:
            for b in alcanzables:
                nuevos.update({a + b, a - b, a * b})
                if b != 0:
                    nuevos.add(a / b)
        alcanzables |= nuevos
    return alcanzables


def _esta(valor: Fraction, conjunto: set) -> bool:
    return any(_cerca(valor, v) for v in conjunto)


# Los delimitadores evitan partir una fracción: en "2/5 × 300 = 120" no debe
# leerse "5 × 300 = 120", que no es una operación que nadie escribiera.
_PAT_OPERACION = re.compile(
    r"(?<![\d/.,])(-?\d+(?:[.,]\d+)?)\s*([+\-×÷*])\s*(-?\d+(?:[.,]\d+)?)(?![\d/])\s*=\s*"
    r"(-?\d+(?:[.,]\d+)?)(?![\d/])")


class VerificadorPlan:
    """Revisa el plan de una respuesta contra lo que el agente observó.

    `pasos_derivacion` controla cuánto margen se le da al modelo para introducir
    un número sin que esté escrito. Con 1 ya se aceptan porcentajes, complementos
    y combinaciones de dos datos; subirlo hace al verificador más permisivo.
    """

    def __init__(self, pasos_derivacion: int = 1, exigir_uso_de_datos: bool = True):
        self.pasos = pasos_derivacion
        self.exigir_uso_de_datos = exigir_uso_de_datos

    # ------------------------------------------------------------------
    def revisar(self, pregunta: str, respuesta: str, observaciones: list,
                contextos=(), conocidos_extra=()) -> Revision:
        """`conocidos_extra` son números que el modelo tiene legítimamente a la
        vista aunque no estén en el enunciado: los de los ejemplos few-shot o los
        del prompt de sistema. Sin esto, el verificador se queja de que el agente
        use el 40 % del SIEE que vio en un ejemplo, que es un falso positivo."""
        del_enunciado = _numeros(pregunta, partes_de_fraccion=True)
        del_contexto = [n for c in contextos for n in _numeros(str(c))]
        del_contexto += [f for v in conocidos_extra for f in _numeros(str(v))]
        exitosas = [o for o in observaciones if o.get("ok")]

        rev = Revision()
        conocidos = set(del_enunciado) | set(del_contexto)
        usados_del_enunciado = set()

        # 1 · Cada argumento debe venir de algún sitio.
        for obs in exitosas:
            argumentos = _numeros(_texto_argumentos(obs.get("argumentos", {})))
            alcanzables = derivables(conocidos, self.pasos)
            for arg in argumentos:
                for n in del_enunciado:
                    if _cerca(arg, n) or _cerca(arg, n / 100) or _cerca(arg, n * 100):
                        usados_del_enunciado.add(n)
                if not _esta(arg, alcanzables):
                    rev.hallazgos.append(Hallazgo(
                        "argumento_no_trazable",
                        f"el número {_fmt(arg)} que le pasaste a {obs['herramienta']} no está en el "
                        f"enunciado ni salió de un paso anterior.",
                    ))
            valor = _frac_de(obs.get("valor"))
            if valor is not None:
                conocidos.add(valor)

        observados = {v for v in (_frac_de(o.get("valor")) for o in exitosas) if v is not None}

        # 2 · Datos del enunciado que nadie usó: plan incompleto.
        if self.exigir_uso_de_datos and exitosas:
            en_respuesta = set(_numeros(respuesta))
            for n in del_enunciado:
                if n in usados_del_enunciado or _esta(n, observados) or _esta(n, en_respuesta):
                    continue
                rev.hallazgos.append(Hallazgo(
                    "dato_sin_usar",
                    f"el enunciado da {_fmt(n)} y tu procedimiento no lo usa en ningún paso; "
                    f"comprueba si el planteamiento está completo.",
                    severidad="media",
                ))

        # 3 · La respuesta final tiene que salir de lo observado.
        if observados:
            vf = valor_final(respuesta)
            final = _frac(vf.replace(",", ".")) if vf else None
            if final is None:
                rev.hallazgos.append(Hallazgo(
                    "respuesta_sin_valor", "tu respuesta no termina en un resultado numérico."))
            elif not _esta(final, derivables(observados | set(del_enunciado), self.pasos)):
                rev.hallazgos.append(Hallazgo(
                    "respuesta_no_trazable",
                    f"tu respuesta final ({_fmt(final)}) no coincide con ningún resultado que "
                    f"hayan devuelto las herramientas ({', '.join(_fmt(v) for v in list(observados)[:3])}).",
                ))

        # 4 · Operaciones escritas que ninguna herramienta ejecutó.
        for a, op, b, c in _PAT_OPERACION.findall(respuesta or ""):
            res = _frac(c.replace(",", "."))
            if res is not None and observados and not _esta(res, observados):
                rev.hallazgos.append(Hallazgo(
                    "calculo_de_cabeza",
                    f"escribiste {a} {op} {b} = {c} pero ninguna herramienta calculó ese resultado.",
                    severidad="media",
                ))
                break

        rev.disponibles = conocidos
        return rev


def _texto_argumentos(argumentos) -> str:
    import json
    return json.dumps(argumentos, ensure_ascii=False) if not isinstance(argumentos, str) else argumentos


def _frac_de(valor) -> Fraction | None:
    if valor is None:
        return None
    nums = _numeros(str(valor))
    return nums[0] if len(nums) == 1 else None


def _fmt(v: Fraction) -> str:
    if v.denominator == 1:
        return str(v.numerator)
    f = float(v)
    return f"{f:.6f}".rstrip("0").rstrip(".") if abs(f) >= 1e-4 else str(v)
