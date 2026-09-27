"""
RAGAS adaptado al tutor: las cuatro métricas de S10, calculadas "por dentro"
con un LLM evaluador local, más una versión OBJETIVA de cada una que sirve para
calibrarlo (como se calibró el juez en M2 contra la verdad numérica).

Qué mide cada una, en este dominio:

  faithfulness       ¿cada afirmación de la respuesta se apoya en lo que el sistema
                     OBSERVÓ? Para un agente, lo observado son los documentos
                     recuperados Y los resultados de las herramientas. Un cálculo
                     escrito "de cabeza", sin herramienta ni documento que lo
                     respalde, no es fiel aunque sea correcto: es la alerta de S10
                     ("acertó por la razón equivocada").
  context_precision  De los chunks recuperados, ¿los útiles quedaron arriba?
                     (average precision sobre el ranking)
  context_recall     ¿Lo recuperado contiene todo lo necesario para la referencia?
  answer_relevancy   ¿La respuesta va al grano? Se generan preguntas a partir de la
                     respuesta y se compara su similitud con la pregunta real.

Versiones objetivas (sin LLM):
  faithfulness_numerica   fracción de números de la respuesta que aparecen en la
                          pregunta, los documentos o las herramientas.
  context_precision_oro   relevancia = el chunk contiene la clave de oro.
  context_recall_oro      la clave de oro está en algún chunk recuperado.

El evaluador es cualquier función `llm(system, user, max_new_tokens) -> texto`.
"""

from __future__ import annotations

import re
import unicodedata

from nucleo import a_numero, mismo_numero, numeros_en  # [local]


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", (texto or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------

def afirmaciones(respuesta: str) -> list[str]:
    """Descompone la respuesta en afirmaciones verificables (una por paso/oración).
    Se quita el rótulo 'Paso N:' para que su número no cuente como dato."""
    texto = re.sub(r"Paso\s*\d+\s*:", "\n", respuesta or "", flags=re.IGNORECASE)
    partes = re.split(r"\n+|(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿])", texto)
    return [p.strip(" -•") for p in partes if len(p.strip().split()) >= 3]


def veredictos(texto: str, n: int) -> list[int] | None:
    """Lee 'k: sí/no' para k = 1..n. None si el formato no cuadra."""
    encontrados = {}
    for linea in (texto or "").splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)-]\s*(s[ií]|no|yes)\b", _norm(linea))
        if m:
            encontrados[int(m.group(1))] = 0 if m.group(2) == "no" else 1
    if all(k in encontrados for k in range(1, n + 1)):
        return [encontrados[k] for k in range(1, n + 1)]
    return None


def _si_no(texto: str) -> int:
    t = _norm(texto).strip()
    return 0 if re.match(r"^\W*no\b", t) else int(bool(re.match(r"^\W*(si|yes)\b", t)))


def _lote_si_no(llm, system: str, encabezado: str, items: list[str], max_tokens_por_item: int = 6) -> list[int]:
    """Un solo prompt con todos los ítems; si el evaluador no respeta el formato,
    se cae a una llamada por ítem (más caro, pero siempre da un veredicto)."""
    if not items:
        return []
    lista = "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))
    user = (f"{encabezado}\n\n{lista}\n\nResponde una línea por ítem con el formato "
            f"'número: sí' o 'número: no'. Nada más.")
    v = veredictos(llm(system, user, max_tokens_por_item * len(items) + 8), len(items))
    if v is not None:
        return v
    return [_si_no(llm(system, f"{encabezado}\n\nÍtem: {t}\n\nResponde solo 'sí' o 'no'.", 4)) for t in items]


def average_precision(relevancias: list[int]) -> float | None:
    """Context precision de RAGAS: media de precision@k en las posiciones relevantes."""
    if not relevancias or not any(relevancias):
        return 0.0 if relevancias else None
    acumulado, suma = 0, 0.0
    for k, r in enumerate(relevancias, 1):
        acumulado += r
        if r:
            suma += acumulado / k
    return suma / sum(relevancias)


# --------------------------------------------------------------------------
# Las cuatro métricas (con evaluador LLM)
# --------------------------------------------------------------------------

SYSTEM_EVALUADOR = ("Eres un evaluador riguroso de un sistema de preguntas y respuestas en español. "
                    "Juzgas SOLO con la información que se te da.")


def faithfulness(llm, pregunta: str, respuesta: str, contextos: list[str], observaciones: list[str]) -> float | None:
    fuentes = [c for c in contextos if c] + [o for o in observaciones if o]
    if not fuentes:
        return None                      # sin nada observado, la métrica no aplica
    claims = afirmaciones(respuesta)
    if not claims:
        return None
    material = "\n\n".join(f"[Fuente {i}] {f}" for i, f in enumerate(fuentes, 1))
    encabezado = (f"PREGUNTA:\n{pregunta}\n\nFUENTES (documentos y resultados de herramientas):\n{material}\n\n"
                  "Para cada afirmación, ¿se puede deducir DIRECTAMENTE de las fuentes o de los datos de la "
                  "pregunta? Un cálculo cuyo resultado no aparece en las fuentes NO se puede deducir.")
    v = _lote_si_no(llm, SYSTEM_EVALUADOR, encabezado, claims)
    return sum(v) / len(v)


def context_precision(llm, pregunta: str, referencia: str, contextos: list[str]) -> tuple[float | None, list[int]]:
    if not contextos:
        return None, []
    encabezado = (f"PREGUNTA:\n{pregunta}\n\nRESPUESTA DE REFERENCIA:\n{referencia}\n\n"
                  "Para cada fragmento recuperado, ¿fue útil para llegar a la respuesta de referencia?")
    rel = _lote_si_no(llm, SYSTEM_EVALUADOR, encabezado, [c[:600] for c in contextos])
    return average_precision(rel), rel


def context_recall(llm, referencia: str, contextos: list[str]) -> float | None:
    if not contextos:
        return None
    frases = [f for f in afirmaciones(referencia) if not re.match(r"respuesta final", _norm(f))]
    if not frases:
        return None
    material = "\n\n".join(c[:600] for c in contextos)
    encabezado = (f"CONTEXTO RECUPERADO:\n{material}\n\n"
                  "Para cada frase de la respuesta de referencia, ¿la información que contiene "
                  "(el dato o la regla, no el cálculo) está en el contexto?")
    v = _lote_si_no(llm, SYSTEM_EVALUADOR, encabezado, frases)
    return sum(v) / len(v)


def answer_relevancy(llm, similitud, pregunta: str, respuesta: str, n: int = 3) -> float | None:
    if not (respuesta or "").strip():
        return 0.0
    salida = llm(SYSTEM_EVALUADOR,
                 f"RESPUESTA:\n{respuesta}\n\nEscribe {n} preguntas DISTINTAS que esta respuesta contestaría. "
                 "Una por línea, sin numerar ni explicar.", 40 * n)
    preguntas = [re.sub(r"^\s*[\d\-.•)]+\s*", "", l).strip() for l in salida.splitlines()]
    preguntas = [p for p in preguntas if len(p.split()) >= 3][:n]
    if not preguntas:
        return 0.0
    return sum(similitud(pregunta, p) for p in preguntas) / len(preguntas)


# --------------------------------------------------------------------------
# Versiones objetivas (para calibrar al evaluador)
# --------------------------------------------------------------------------

def faithfulness_numerica(pregunta: str, respuesta: str, contextos: list[str], observaciones: list[str]) -> float | None:
    texto = re.sub(r"Paso\s*\d+\s*:", " ", respuesta or "", flags=re.IGNORECASE)
    nums = numeros_en(texto)
    if not nums:
        return None
    disponibles = numeros_en(pregunta) + [n for c in contextos for n in numeros_en(c)] \
        + [n for o in observaciones for n in numeros_en(o)]
    # Los porcentajes de un documento aparecen como "40 por ciento" y se usan como 0.4.
    extendidos = disponibles + [str(a_numero(d) / 100) for d in disponibles if a_numero(d) is not None]
    respaldados = sum(1 for n in nums if any(mismo_numero(n, d) for d in extendidos))
    return respaldados / len(nums)


def relevancias_oro(contextos: list[str], clave: str | None) -> list[int]:
    if not clave:
        return []
    return [int(_norm(clave) in _norm(c)) for c in contextos]


def context_precision_oro(contextos: list[str], clave: str | None) -> float | None:
    if not contextos or not clave:
        return None
    return average_precision(relevancias_oro(contextos, clave))


def context_recall_oro(contextos: list[str], clave: str | None) -> float | None:
    if not clave:
        return None
    return float(any(relevancias_oro(contextos, clave))) if contextos else 0.0


# --------------------------------------------------------------------------
# Calibración
# --------------------------------------------------------------------------

def kappa_cohen(a: list[int], b: list[int]) -> float | None:
    if not a or len(a) != len(b):
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
