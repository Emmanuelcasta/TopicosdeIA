"""
Métricas propias del agente, complementarias al harness de M2.

El harness mide la RESPUESTA (acierto, juez, similitud). Estas métricas miden
el PROCESO a partir de la traza, que es donde un agente puede fallar sin que
la respuesta lo delate:

  Uso correcto      ¿las llamadas son válidas (formato + tipos) y se ejecutan sin error?
  Selección         ¿usó las familias de herramientas que el caso necesita, y no otras?
  Fidelidad         ¿la respuesta final usa el resultado que devolvió la herramienta?
  Proceso           pasos del LLM, llamadas, tope de pasos alcanzado, latencia
  Robustez          ¿acierta igual con la pregunta mal escrita que con la original?
  Consistencia      ¿da la misma respuesta (mal escrita vs original; varias muestras)?
"""

from __future__ import annotations

from collections import Counter
from math import comb

from nucleo import a_numero, mismo_numero, valor_final  # [local]


# --------------------------------------------------------------------------
# Selección y uso de herramientas, por caso
# --------------------------------------------------------------------------

def metricas_herramientas_caso(esperadas: dict, observaciones: list, respuesta: str,
                               familia_de, comodines: dict) -> dict:
    """
    esperadas     : {"familias": [...], "opcionales": [...]} del eval set
    observaciones : ResultadoHerramienta como dicts (de la traza)
    familia_de    : nombre de herramienta -> familia
    comodines     : nombre de herramienta -> familias que también cubre (evaluar_expresion)
    """
    requeridas = set(esperadas.get("familias", []))
    opcionales = set(esperadas.get("opcionales", []))
    llamadas = len(observaciones)

    usadas_estrictas: set = set()     # familia propia de cada herramienta usada
    usadas_amplias: set = set()       # + lo que cubren los comodines
    innecesarias = 0
    for o in observaciones:
        fam = familia_de(o["herramienta"])
        if fam is None:
            continue                  # herramienta inexistente: cuenta como uso incorrecto, no como selección
        cubre = {fam} | set(comodines.get(o["herramienta"], ()))
        usadas_estrictas.add(fam)
        usadas_amplias |= cubre
        if not (cubre & (requeridas | opcionales)):
            innecesarias += 1

    validas = sum(1 for o in observaciones if o.get("tipo_error") not in ("desconocida", "argumentos"))
    sin_error = sum(1 for o in observaciones if o.get("ok"))
    errores_dominio = sum(1 for o in observaciones if o.get("tipo_error") == "dominio")

    recall = len(requeridas & usadas_amplias) / len(requeridas) if requeridas else None
    recall_estricto = len(requeridas & usadas_estrictas) / len(requeridas) if requeridas else None

    # Fidelidad: si hubo resultados numéricos, ¿la respuesta final usa alguno?
    valores = [o.get("valor") for o in observaciones if o.get("ok") and a_numero(o.get("valor")) is not None]
    vf = valor_final(respuesta)
    fidelidad = None if not valores else any(mismo_numero(vf, v) for v in valores)

    return {
        "llamadas": llamadas,
        "llamadas_validas": validas,
        "llamadas_sin_error": sin_error,
        "errores_dominio": errores_dominio,
        "llamadas_nativas": sum(1 for o in observaciones if o.get("formato") == "nativo"),
        "uso_herramienta": llamadas > 0,
        "necesitaba_herramienta": bool(requeridas),
        # Selección: cubrió lo requerido (con comodines) y no llamó nada fuera de lugar.
        "recall_seleccion": recall,
        "recall_seleccion_estricto": recall_estricto,
        "seleccion_correcta": (recall == 1.0 if requeridas else llamadas == 0 or innecesarias == 0)
                              and innecesarias == 0,
        "innecesarias": innecesarias,
        "fidelidad_herramienta": fidelidad,
    }


def _media(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def resumen_agente(filas: list[dict]) -> dict:
    """Agrega las métricas por caso de UN sistema (filas = detalle enriquecido)."""
    llam = sum(f["llamadas"] for f in filas)
    necesitaban = [f for f in filas if f["necesitaba_herramienta"]]
    no_necesitaban = [f for f in filas if not f["necesitaba_herramienta"]]
    return {
        "n": len(filas),
        "llamadas_totales": llam,
        "llamadas_por_caso": llam / len(filas) if filas else 0.0,
        "tasa_llamadas_validas": (sum(f["llamadas_validas"] for f in filas) / llam) if llam else None,
        "tasa_llamadas_sin_error": (sum(f["llamadas_sin_error"] for f in filas) / llam) if llam else None,
        "tasa_formato_nativo": (sum(f["llamadas_nativas"] for f in filas) / llam) if llam else None,
        # Decidir CUÁNDO usar herramientas
        "uso_cuando_hace_falta": _media([f["uso_herramienta"] for f in necesitaban]),
        "abstencion_cuando_no_hace_falta": _media([not f["uso_herramienta"] for f in no_necesitaban]),
        # Decidir CUÁL
        "recall_seleccion": _media([f["recall_seleccion"] for f in filas]),
        "recall_seleccion_estricto": _media([f["recall_seleccion_estricto"] for f in filas]),
        "seleccion_correcta": _media([f["seleccion_correcta"] for f in filas]),
        "llamadas_innecesarias": sum(f["innecesarias"] for f in filas),
        "fidelidad_herramienta": _media([f["fidelidad_herramienta"] for f in filas]),
        # Proceso
        "pasos_llm_medios": _media([f.get("pasos_llm") for f in filas]),
        "tope_pasos": sum(1 for f in filas if f.get("terminacion") == "tope_pasos"),
        "errores_formato": sum(f.get("errores_formato", 0) for f in filas),
        "verificaciones": sum(f.get("verificaciones", 0) for f in filas),
        "segundos_medios": _media([f.get("segundos") for f in filas]),
    }


def eventos(traza: list, tipo: str) -> list:
    return [e for e in traza if (e["tipo"] if isinstance(e, dict) else e.tipo) == tipo]


# --------------------------------------------------------------------------
# Robustez y consistencia
# --------------------------------------------------------------------------

def robustez_mal_escrito(detalle: list[dict], detalle_referencia: list[dict]) -> dict:
    """Pares (mal escrito, original). `detalle_referencia` contiene los originales
    evaluados con EL MISMO sistema (pueden venir del eval set de M2 o del M3)."""
    por_id = {d["id"]: d for d in detalle_referencia}
    pares = [(d, por_id[d["original"]]) for d in detalle
             if d.get("original") and d["original"] in por_id]
    if not pares:
        return {"pares": 0}
    ambos = sum(1 for m, o in pares if m["acierto"] and o["acierto"])
    solo_original = sum(1 for m, o in pares if o["acierto"] and not m["acierto"])
    solo_mal = sum(1 for m, o in pares if m["acierto"] and not o["acierto"])
    misma_resp = sum(1 for m, o in pares
                     if valor_final(m["respuesta"]) is not None
                     and mismo_numero(valor_final(m["respuesta"]), valor_final(o["respuesta"])))
    return {
        "pares": len(pares),
        "aciertos_original": sum(o["acierto"] for _, o in pares),
        "aciertos_mal_escrito": sum(m["acierto"] for m, _ in pares),
        "ambos": ambos,
        "perdidos_por_escritura": solo_original,
        "ganados": solo_mal,
        "misma_respuesta": misma_resp / len(pares),
    }


def autoconsistencia(muestras: dict[str, list[str]]) -> dict:
    """{id: [respuesta muestreada 1, 2, ...]} -> fracción de casos en que TODAS
    las muestras dan el mismo valor final, y acuerdo medio con la moda."""
    unanimes, acuerdo = 0, []
    for resp in muestras.values():
        valores = [valor_final(r) for r in resp]
        canon = [None if v is None else round(a_numero(v), 6) if a_numero(v) is not None else v
                 for v in valores]
        moda, n = Counter(canon).most_common(1)[0]
        unanimes += int(n == len(canon) and moda is not None)
        acuerdo.append(n / len(canon))
    return {"casos": len(muestras), "unanimes": unanimes,
            "tasa_unanimidad": unanimes / len(muestras) if muestras else None,
            "acuerdo_medio": _media(acuerdo)}


def mcnemar_exacto(aciertos_x: list[bool], aciertos_y: list[bool]) -> tuple[int, int, float]:
    """(mejora, empeora, p) de pasar de X a Y sobre los mismos casos (binomial exacta, dos colas)."""
    mejora = sum(1 for a, b in zip(aciertos_x, aciertos_y) if not a and b)
    empeora = sum(1 for a, b in zip(aciertos_x, aciertos_y) if a and not b)
    n = mejora + empeora
    if n == 0:
        return mejora, empeora, 1.0
    p = sum(comb(n, i) for i in range(0, min(mejora, empeora) + 1)) / 2 ** n
    return mejora, empeora, min(1.0, 2 * p)
