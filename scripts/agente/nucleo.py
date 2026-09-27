"""
Núcleo del agente: normalizador de consulta, parser de acciones y bucle ReAct.

El núcleo no conoce ninguna herramienta concreta ni ningún modelo concreto:

  llm(mensajes, esquemas) -> texto      cualquier función (Qwen, un simulador de tests)
  RegistroHerramientas                  qué herramientas hay y cómo se ejecutan

Flujo de una pregunta (la arquitectura de la propuesta M3):

  Usuario
    -> Normalizador        (reglas deterministas + reescritura LLM opcional, con salvaguarda de números)
    -> Razonador (LLM)     pensamiento + propuesta de acción, en el formato nativo <tool_call> de Qwen2.5
    -> Parser/Validador    texto -> LlamadaHerramienta (o error de formato que se devuelve al modelo)
    -> Tool Registry       validación de tipos + ejecución segura -> ResultadoHerramienta
    -> Observación         vuelve al modelo como <tool_response>
    -> ... repetir hasta que el modelo responda o se agote el tope de pasos
    -> Verificador         ¿la respuesta final usa algún resultado de las herramientas?
    -> Respuesta final + traza completa (para métricas y RAGAS)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from registro import LlamadaHerramienta, RegistroHerramientas, ResultadoHerramienta  # [local]


# --------------------------------------------------------------------------
# Utilidades numéricas (independientes de CODIGO_METRICAS)
# --------------------------------------------------------------------------

_PAT_NUM = re.compile(r"-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?")


def numeros_en(texto: str) -> list[str]:
    """Números de un texto con punto decimal. Como en todo el proyecto, el punto
    es SIEMPRE decimal (9.316) y la coma solo es de miles si le siguen 3 cifras."""
    t, previo = texto or "", None
    while t != previo:                      # 1,138,712 necesita dos pasadas
        previo, t = t, re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", t)
    return [re.sub(r"\s+", "", n).replace(",", ".") for n in _PAT_NUM.findall(t)]


def a_numero(texto: str | None) -> float | None:
    if texto is None:
        return None
    try:
        if "/" in texto:
            a, b = texto.split("/")
            return float(a) / float(b)
        return float(texto)
    except (ValueError, ZeroDivisionError):
        return None


def mismo_numero(a, b, tol: float = 1e-6) -> bool:
    x, y = a_numero(str(a)) if a is not None else None, a_numero(str(b)) if b is not None else None
    return x is not None and y is not None and abs(x - y) <= tol * max(1.0, abs(y))


_PAT_RESPUESTA = re.compile(r"Respuesta\s+final\s*:\s*(.+)", re.IGNORECASE)


def valor_final(texto: str) -> str | None:
    m = _PAT_RESPUESTA.search(texto or "")
    fuente = m.group(1).splitlines()[0] if m else (texto or "")
    nums = numeros_en(fuente)
    if not nums:
        return None
    return nums[0] if m else nums[-1]


# --------------------------------------------------------------------------
# Normalizador de la consulta (preguntas mal escritas)
# --------------------------------------------------------------------------
# Decisión: dos capas.
#  1) REGLAS deterministas y auditables para lo frecuente y seguro: espacios,
#     signos repetidos, abreviaturas de chat y operadores escritos en palabras
#     ENTRE números ("17/24 mas 11/36", "3847 x 296").
#  2) REESCRITURA con el LLM (opcional, solo en la arquitectura agéntica) para
#     faltas de ortografía que ninguna regla cubre ("bale", "evaluasion").
# Salvaguarda común: si la versión corregida no conserva EXACTAMENTE los mismos
# números, se descarta. Una corrección ortográfica que cambia un dato es peor
# que la pregunta mal escrita.

_ABREVIATURAS = [
    (r"\bq\b", "que"), (r"\bxq\b", "porque"), (r"\bpq\b", "porque"), (r"\bpa\b", "para"),
    # "x" solo significa "por" delante de palabras típicas; "halla x si..." debe quedar igual.
    (r"\bx\b(?=\s+(?:pagar|persona|personas|cada|dia|día|mes|periodo|período|ciento|favor|hora|kilo)\b)", "por"),
    (r"\bai\b", "hay"), (r"\bk\b", "que"),
    (r"\btmb\b", "también"), (r"\bdl\b", "del"),
]
_OPERADORES_ENTRE_NUMEROS = [
    (r"(?<=\d)\s+(?:mas|más)\s+(?=\d)", " + "),
    (r"(?<=\d)\s+menos\s+(?=\d)", " - "),
    (r"(?<=\d)\s*[x×*]\s*(?=\d)", " × "),
    (r"(?<=\d)\s+por\s+(?=\d)", " × "),
    (r"(?<=\d)\s+(?:dividido\s+(?:en|entre|por)|entre)\s+(?=\d)", " ÷ "),
]


@dataclass
class ConsultaNormalizada:
    original: str
    normalizada: str
    cambios: list = field(default_factory=list)
    reescrita_llm: bool = False
    descartada: str | None = None      # motivo si la reescritura LLM se rechazó


def parece_informal(texto: str) -> bool:
    """Disparador de la reescritura LLM: sin tildes, sin eñes y sin '¿'. Una
    pregunta bien escrita en español casi siempre tiene alguno de los tres; así
    no se gasta una generación (ni se arriesga el sentido) en las que no lo necesitan."""
    return not re.search(r"[¿áéíóúñÁÉÍÓÚÑ]", texto)


class Normalizador:
    def __init__(self, reescribir: Callable[[str], str] | None = None,
                 condicion: Callable[[str], bool] | None = parece_informal):
        self.reescribir = reescribir
        self.condicion = condicion

    @staticmethod
    def _mismos_numeros(a: str, b: str) -> bool:
        return sorted(numeros_en(a)) == sorted(numeros_en(b))

    def reglas(self, texto: str) -> tuple[str, list]:
        cambios, t = [], " ".join(texto.split())
        for patron, reemplazo in _OPERADORES_ENTRE_NUMEROS + _ABREVIATURAS:
            nuevo = re.sub(patron, reemplazo, t, flags=re.IGNORECASE)
            if nuevo != t:
                cambios.append(patron)
                t = nuevo
        t2 = re.sub(r"([?!.])\1+", r"\1", t)
        if t2 != t:
            cambios.append("signos repetidos")
        t = t2
        if "?" in t and "¿" not in t:
            t = "¿" + t[0].upper() + t[1:] if t else t
        elif t:
            t = t[0].upper() + t[1:]
        return t, cambios

    def __call__(self, texto: str) -> ConsultaNormalizada:
        t, cambios = self.reglas(texto)
        if not self._mismos_numeros(texto, t):     # las reglas nunca deben tocar números
            return ConsultaNormalizada(texto, texto, [], descartada="las reglas alteraban números")
        res = ConsultaNormalizada(texto, t, cambios)
        if self.reescribir is not None and (self.condicion is None or self.condicion(texto)):
            candidata = (self.reescribir(t) or "").strip().strip('"').splitlines()
            candidata = candidata[0].strip() if candidata else ""
            if not candidata:
                res.descartada = "reescritura vacía"
            elif not self._mismos_numeros(t, candidata):
                res.descartada = f"la reescritura cambió los números: {candidata!r}"
            elif not (0.5 <= len(candidata) / max(1, len(t)) <= 2.0):
                res.descartada = "la reescritura cambió demasiado la longitud"
            else:
                res.normalizada, res.reescrita_llm = candidata, candidata != t
        return res


# --------------------------------------------------------------------------
# Parser: texto del modelo -> pensamiento + llamadas | respuesta
# --------------------------------------------------------------------------

@dataclass
class SalidaModelo:
    texto: str
    pensamiento: str = ""
    llamadas: list = field(default_factory=list)
    respuesta: str | None = None
    error_formato: str | None = None


_PAT_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


def _json_de(texto: str):
    """Primer objeto JSON balanceado del texto (tolera texto alrededor)."""
    inicio = texto.find("{")
    while inicio != -1:
        nivel, en_cadena, escape = 0, False, False
        for i in range(inicio, len(texto)):
            c = texto[i]
            if en_cadena:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    en_cadena = False
                continue
            if c == '"':
                en_cadena = True
            elif c == "{":
                nivel += 1
            elif c == "}":
                nivel -= 1
                if nivel == 0:
                    try:
                        return json.loads(texto[inicio:i + 1])
                    except json.JSONDecodeError:
                        break
        inicio = texto.find("{", inicio + 1)
    return None


def _a_llamada(obj, formato: str) -> LlamadaHerramienta | None:
    if not isinstance(obj, dict):
        return None
    nombre = obj.get("name") or obj.get("tool") or obj.get("nombre")
    args = obj.get("arguments", obj.get("args", obj.get("argumentos", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(nombre, str):
        return None
    return LlamadaHerramienta(nombre=nombre, argumentos=args if isinstance(args, dict) else {},
                              formato=formato)


def analizar_salida(texto: str) -> SalidaModelo:
    s = SalidaModelo(texto=texto)
    bloques = _PAT_TOOL_CALL.findall(texto)
    if "<tool_call>" in texto:
        s.pensamiento = texto.split("<tool_call>")[0].strip()
        for b in bloques:
            llamada = _a_llamada(_json_de(b), "nativo")
            if llamada is None:
                s.error_formato = f"no pude leer la llamada a herramienta: {b[:120]!r}"
            else:
                s.llamadas.append(llamada)
        if not s.llamadas and s.error_formato is None:
            s.error_formato = "etiqueta <tool_call> vacía"
    else:
        # Recuperación: el modelo escribió el JSON sin las etiquetas.
        obj = _json_de(texto)
        llamada = _a_llamada(obj, "recuperado") if obj else None
        if llamada is not None and ("arguments" in obj or "args" in obj):
            s.pensamiento = texto[:texto.find("{")].strip()
            s.llamadas.append(llamada)
        else:
            s.respuesta = texto.strip()
    s.pensamiento = re.sub(r"^\s*(Pensamiento|Thought)\s*:\s*", "", s.pensamiento, flags=re.IGNORECASE)
    return s


# --------------------------------------------------------------------------
# Traza
# --------------------------------------------------------------------------

@dataclass
class EventoTraza:
    paso: int
    tipo: str        # normalizacion | contexto | pensamiento | llamada | observacion |
                     # error_formato | verificacion | respuesta | tope_pasos
    datos: dict


@dataclass
class RespuestaAgente:
    pregunta: str
    respuesta: str
    pregunta_normalizada: str
    traza: list = field(default_factory=list)
    contextos: list = field(default_factory=list)       # chunks usados (fijos o buscados)
    observaciones: list = field(default_factory=list)   # ResultadoHerramienta como dict
    pasos_llm: int = 0
    terminacion: str = "respuesta"                      # respuesta | tope_pasos
    segundos: float = 0.0

    @property
    def llamadas(self) -> list:
        return self.observaciones

    def a_dict(self) -> dict:
        d = asdict(self)
        d["traza"] = [asdict(e) if not isinstance(e, dict) else e for e in self.traza]
        return d


# --------------------------------------------------------------------------
# El agente
# --------------------------------------------------------------------------

class AgenteTutor:
    """Una sola clase configurable para las configuraciones C2–C5.

    contexto_fijo : función pregunta -> [chunks] que se INYECTA siempre (RAG clásico).
                    None = el agente decide si buscar (vía herramienta `buscar_documentos`).
    registro      : herramientas disponibles; None = sin herramientas.
    max_rondas    : cuántas veces puede pedir herramientas (1 = function calling de una
                    ronda, C3; >1 = ReAct, C4/C5).
    verificar     : si la respuesta final no coincide con ningún resultado numérico
                    observado, se le pide UNA revisión.
    normalizador  : Normalizador o None.
    few_shot      : lista de conversaciones de ejemplo (mensajes) que van antes de la pregunta.
    """

    def __init__(self, llm: Callable, system: str, registro: RegistroHerramientas | None = None,
                 contexto_fijo: Callable | None = None, max_rondas: int = 1, max_pasos: int = 6,
                 verificar: bool = False, normalizador: Normalizador | None = None,
                 few_shot: list | None = None, nombre: str = "agente",
                 extraer_contextos: Callable | None = None):
        self.llm, self.system, self.registro = llm, system, registro
        self.contexto_fijo, self.max_rondas, self.max_pasos = contexto_fijo, max_rondas, max_pasos
        self.verificar, self.normalizador, self.few_shot = verificar, normalizador, few_shot or []
        self.nombre = nombre
        # Cómo sacar chunks de una observación de `buscar_documentos` (para RAGAS).
        self.extraer_contextos = extraer_contextos

    # ------------------------------------------------------------------
    def _mensajes_iniciales(self, pregunta: str, contextos: list) -> list:
        msgs = [{"role": "system", "content": self.system}]
        for conversacion in self.few_shot:
            msgs.extend(conversacion)
        if contextos:
            bloque = "\n\n".join(contextos)
            msgs.append({"role": "user", "content": f"Contexto:\n{bloque}\n\nPregunta: {pregunta}"})
        else:
            msgs.append({"role": "user", "content": pregunta})
        return msgs

    @staticmethod
    def _msg_llamadas(pensamiento: str, llamadas: list) -> dict:
        return {"role": "assistant", "content": pensamiento,
                "tool_calls": [{"type": "function",
                                "function": {"name": l.nombre, "arguments": l.argumentos}}
                               for l in llamadas]}

    # ------------------------------------------------------------------
    def __call__(self, pregunta: str) -> RespuestaAgente:
        t0 = time.perf_counter()
        traza: list = []
        paso = 0

        norm = self.normalizador(pregunta) if self.normalizador else None
        q = norm.normalizada if norm else pregunta
        if norm:
            traza.append(EventoTraza(0, "normalizacion", asdict(norm)))

        contextos = list(self.contexto_fijo(q)) if self.contexto_fijo else []
        if contextos:
            traza.append(EventoTraza(0, "contexto", {"chunks": contextos}))

        msgs = self._mensajes_iniciales(q, contextos)
        esquemas = self.registro.esquemas() if self.registro is not None and len(self.registro) else None
        observaciones: list = []
        rondas, revisado, pasos_llm = 0, False, 0

        while pasos_llm < self.max_pasos:
            paso += 1
            herramientas_activas = esquemas if (esquemas and rondas < self.max_rondas) else None
            texto = self.llm(msgs, herramientas_activas)
            pasos_llm += 1
            salida = analizar_salida(texto)

            if salida.llamadas and herramientas_activas:
                if salida.pensamiento:
                    traza.append(EventoTraza(paso, "pensamiento", {"texto": salida.pensamiento}))
                msgs.append(self._msg_llamadas(salida.pensamiento, salida.llamadas))
                for llamada in salida.llamadas:
                    traza.append(EventoTraza(paso, "llamada", asdict(llamada)))
                    res = self.registro.ejecutar(llamada)
                    d = res.a_dict()
                    d["formato"] = llamada.formato
                    observaciones.append(d)
                    traza.append(EventoTraza(paso, "observacion", d))
                    if res.ok and self.extraer_contextos and llamada.nombre == "buscar_documentos":
                        contextos.extend(self.extraer_contextos(res))
                    msgs.append({"role": "tool", "name": llamada.nombre,
                                 "content": res.como_observacion()})
                rondas += 1
                continue

            if salida.llamadas and not herramientas_activas:
                # Pidió herramientas cuando ya no le quedan rondas: se le pide responder.
                traza.append(EventoTraza(paso, "error_formato",
                                         {"motivo": "llamada sin rondas disponibles", "texto": texto}))
                msgs.append({"role": "assistant", "content": texto})
                msgs.append({"role": "user", "content": "Ya no puedes usar más herramientas. "
                             "Responde con lo que tienes y termina con 'Respuesta final:'."})
                continue

            if salida.error_formato:
                traza.append(EventoTraza(paso, "error_formato", {"motivo": salida.error_formato, "texto": texto}))
                msgs.append({"role": "assistant", "content": texto})
                msgs.append({"role": "user", "content": f"Error: {salida.error_formato}. Escribe la llamada "
                             "como JSON válido dentro de <tool_call></tool_call> o responde directamente."})
                continue

            respuesta = salida.respuesta or ""
            if self.verificar and not revisado:
                problema = self._verificar(respuesta, observaciones)
                if problema:
                    revisado = True
                    traza.append(EventoTraza(paso, "verificacion", {"problema": problema, "respuesta": respuesta}))
                    msgs.append({"role": "assistant", "content": respuesta})
                    msgs.append({"role": "user", "content": f"Verificación: {problema} Revisa tu respuesta "
                                 "usando los resultados de las herramientas y vuelve a escribirla completa, "
                                 "terminando con 'Respuesta final:'."})
                    continue
            traza.append(EventoTraza(paso, "respuesta", {"texto": respuesta}))
            return RespuestaAgente(pregunta, respuesta, q, traza, contextos, observaciones,
                                   pasos_llm, "respuesta", time.perf_counter() - t0)

        # Tope de pasos: una última generación SIN herramientas.
        traza.append(EventoTraza(paso, "tope_pasos", {"max_pasos": self.max_pasos}))
        msgs.append({"role": "user", "content": "Se acabaron los pasos. Responde ya con lo que tienes "
                     "y termina con 'Respuesta final:'."})
        respuesta = analizar_salida(self.llm(msgs, None)).respuesta or ""
        pasos_llm += 1
        traza.append(EventoTraza(paso + 1, "respuesta", {"texto": respuesta}))
        return RespuestaAgente(pregunta, respuesta, q, traza, contextos, observaciones,
                               pasos_llm, "tope_pasos", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    @staticmethod
    def _verificar(respuesta: str, observaciones: list) -> str | None:
        """Consistencia respuesta <-> herramientas. Solo interviene si hubo
        resultados numéricos y la respuesta final no usa NINGUNO de ellos."""
        valores = [o.get("valor") for o in observaciones if o.get("ok") and o.get("valor")]
        valores = [v for v in valores if a_numero(v) is not None]
        if not valores:
            return None
        v = valor_final(respuesta)
        if v is None:
            return "tu respuesta no tiene un resultado numérico final."
        if any(mismo_numero(v, x) for x in valores):
            return None
        return (f"tu respuesta final ({v}) no coincide con ningún resultado obtenido con las "
                f"herramientas ({', '.join(valores[-3:])}).")
