"""Análisis EXPLORATORIO de los fallos del tutor. No es una medición.

Qué es y qué no es, porque la diferencia cambia lo que se puede afirmar:

  * Lo **medido** son `acierto`, `abstuvo` y `alucino`, que salen del arnés de M2
    con sus tres dimensiones y su criterio por caso. Esas cifras están en los
    scorecards y no se recalculan aquí.
  * Lo que hace este script es **etiquetar con heurísticas** las respuestas ya
    guardadas para orientar dónde mirar. Una etiqueta es una hipótesis sobre la
    causa, no un veredicto: nadie anotó estos 371 casos a mano.

Tres reglas para que las cifras no engañen:

  1. **Una etiqueta principal por caso fallado.** La suma de las principales es
     exactamente el número de fallos. Las secundarias se informan aparte y no se
     suman, porque un caso con tres señales sigue siendo un fallo.
  2. **`sin_evidencia` cuando los datos no alcanzan.** Hay categorías que piden
     información que no se guardó (pasajes de oro por caso, legibilidad anotada).
     Ahí se dice, en vez de rellenar con una estimación.
  3. **Las etiquetas condicionadas se marcan.** `calculo_sin_herramienta` solo
     puede aplicarse a una respuesta que no llamó a ninguna herramienta, así que
     una configuración que siempre llama no puede recibirla. Ver cero ahí **no
     significa** que haya dejado de equivocarse en las cuentas.

Fuentes, todas del repositorio:

    `M3_rag_agentes/resultados/respuestas_s10.jsonl`   7 configuraciones × 53 casos
    `M3_rag_agentes/resultados/detalle_s10_v2.csv`     la calificación del arnés
    `m4/scorecard_m4.csv`                              12 casos visuales

    python scripts/taxonomia_errores.py
    python scripts/taxonomia_errores.py --config "C5-FT · entrenado"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts" / "agente"))

RESP_M3 = RAIZ / "M3_rag_agentes" / "resultados" / "respuestas_s10.jsonl"
DETALLE_M3 = RAIZ / "M3_rag_agentes" / "resultados" / "detalle_s10_v2.csv"
SCORECARD_M4 = RAIZ / "m4" / "scorecard_m4.csv"
SALIDA = RAIZ / "docs" / "taxonomia_errores.md"

# Cada etiqueta: cómo se decide, y de qué tipo de afirmación se trata.
#   comprobable  la señal está en los datos guardados
#   heuristica   es una inferencia razonable, no una comprobación
#   condicionada solo puede aplicarse a cierto tipo de respuesta (ver regla 3)
#   sin_evidencia los datos guardados no permiten decidirla
ETIQUETAS = {
    "aritmetica_falsa": ("comprobable",
        "el verificador aritmético encuentra una igualdad que no cuadra en la respuesta"),
    "calculo_sin_herramienta": ("condicionada",
        "falló un caso numérico sin llamar a ninguna herramienta; solo aplicable "
        "a configuraciones que pueden no llamarlas"),
    "final_no_coincide_con_pasos": ("comprobable",
        "ninguna observación de herramienta aparece en el número final"),
    "planteamiento_incorrecto": ("heuristica",
        "las herramientas devolvieron bien y la respuesta igual falló"),
    "herramienta_innecesaria": ("comprobable",
        "llamó a una herramienta en un caso adversarial"),
    "pregunta_mal_escrita": ("comprobable",
        "el caso pertenece al bloque `mal_escrito` del eval set"),
    "explicacion_confusa": ("comprobable",
        "el juez puntuó <= 2 pese a que el caso se contó como acierto"),
    "negativa_innecesaria": ("comprobable",
        "se abstuvo en un caso que tenía respuesta"),
    "no_consulto_documentos": ("comprobable",
        "falló un caso de conocimiento sin recuperar ningún pasaje"),
    "no_reconocio_el_limite": ("comprobable",
        "respondió un adversarial en vez de abstenerse"),
    "respuesta_a_imagen_ilegible": ("comprobable",
        "respondió un caso visual cuyo `esperado` del scorecard es «ilegible»"),
    "respuesta_fuera_de_dominio": ("comprobable",
        "respondió un caso visual cuyo `esperado` es «fuera de dominio»"),
    "dato_no_disponible": ("comprobable",
        "respondió un caso visual cuyo `esperado` es «dato no disponible»"),
    "sin_conclusion_formato": ("comprobable",
        "no emitió la línea `Diagnóstico:`; es un fallo de formato, no de criterio"),
    "contexto_irrelevante": ("sin_evidencia",
        "haría falta el pasaje de oro de cada caso de S10, y no se guardó por caso"),
    "mala_eleccion_de_herramienta": ("sin_evidencia",
        "haría falta la herramienta correcta anotada por caso; `recall_seleccion` "
        "solo existe para el subconjunto con oro"),
    "lectura_visual_erronea": ("sin_evidencia",
        "haría falta la transcripción correcta de cada hoja anotada a mano"),
}


def _numeros(texto: str) -> set:
    return set(re.findall(r"-?\d+(?:[.,]\d+)?", texto or ""))


def _aritmetica_falsa(respuesta: str):
    """Evidencia dura: una igualdad escrita en la respuesta que no se cumple."""
    from verificador_aritmetico import revisar_hoja

    pasos = [l for l in (respuesta or "").splitlines() if l.strip()]
    h = revisar_hoja(pasos)
    return h if h is not None and h.tipo in ("calculo", "traslado") else None


BLOQUES_NUMERICOS = ("calculo", "aritmetica", "compuesto", "mal_escrito")


def etiquetar_m3(reg: dict, fila: dict) -> tuple:
    """-> (etiqueta principal, [secundarias]). La principal es una sola."""
    obs = reg.get("observaciones") or []
    exitosas = [o for o in obs if o.get("ok") and o.get("valor")]
    bloque = fila.get("bloque", "")
    acierto = fila.get("acierto", "").lower() == "true"
    abstuvo = fila.get("abstuvo", "").lower() == "true"
    juez = int(fila["juez"]) if str(fila.get("juez", "")).isdigit() else None

    if acierto:
        if juez is not None and juez <= 2:
            return "explicacion_confusa", []
        return "acierto", []
    if abstuvo:
        if bloque == "adversarial":
            return "abstencion_correcta", []     # no es un fallo
        return "negativa_innecesaria", []

    # Falló y no se abstuvo. Secundarias primero: describen el contexto.
    secundarias = []
    if bloque == "mal_escrito":
        secundarias.append("pregunta_mal_escrita")
    if bloque == "adversarial" and obs:
        secundarias.append("herramienta_innecesaria")

    # La principal, de la más comprobable a la más conjetural.
    h = _aritmetica_falsa(reg.get("respuesta", ""))
    if h is not None:
        return "aritmetica_falsa", secundarias
    if bloque == "adversarial":
        return "no_reconocio_el_limite", secundarias
    if bloque == "conocimiento":
        return ("no_consulto_documentos" if not reg.get("contextos")
                else "fallo_con_contexto_recuperado"), secundarias
    if exitosas:
        valores = {str(o["valor"]) for o in exitosas}
        if valores and not (valores & _numeros(reg.get("respuesta", ""))):
            return "final_no_coincide_con_pasos", secundarias
        return "planteamiento_incorrecto", secundarias
    if bloque in BLOQUES_NUMERICOS and not obs:
        return "calculo_sin_herramienta", secundarias
    return "fallo_sin_etiqueta", secundarias


def analizar_m3(config: str | None = None) -> tuple:
    respuestas = {}
    for linea in RESP_M3.read_text(encoding="utf-8").splitlines():
        r = json.loads(linea)
        respuestas[(r["config"], r["id"])] = r

    principales = defaultdict(Counter)
    secundarias = defaultdict(Counter)
    llamo_siempre = defaultdict(lambda: [0, 0])      # [casos, casos con llamadas]
    for fila in csv.DictReader(DETALLE_M3.open(encoding="utf-8")):
        if config and fila["config"] != config:
            continue
        reg = respuestas.get((fila["config"], fila["id"]))
        if reg is None:
            continue
        c = fila["config"]
        llamo_siempre[c][0] += 1
        if reg.get("observaciones"):
            llamo_siempre[c][1] += 1
        p, s = etiquetar_m3(reg, fila)
        principales[c][p] += 1
        for e in s:
            secundarias[c][e] += 1
    return principales, secundarias, llamo_siempre


# El `esperado` del scorecard dice qué clase de válvula es cada adversarial.
# Sin esto los tres salían como «imagen ilegible», y solo vis-10 lo es.
POR_ESPERADO = {
    "ilegible": "respuesta_a_imagen_ilegible",
    "fuera de dominio": "respuesta_fuera_de_dominio",
    "dato no disponible": "dato_no_disponible",
}


def analizar_m4() -> Counter:
    c = Counter()
    if not SCORECARD_M4.exists():
        return c
    for fila in csv.DictReader(SCORECARD_M4.open(encoding="utf-8")):
        if fila.get("estado") != "evaluado":
            continue
        if fila.get("acierto", "").lower() == "true":
            c["acierto"] += 1
        elif fila.get("abstuvo", "").lower() == "true":
            c["abstencion_correcta" if fila.get("abstuvo_bien", "").lower() == "true"
              else "negativa_innecesaria"] += 1
        elif fila.get("sin_conclusion", "").lower() == "true":
            c["sin_conclusion_formato"] += 1
        else:
            esperado = (fila.get("esperado") or "").strip().lower()
            c[POR_ESPERADO.get(esperado, "diagnostico_incorrecto")] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    args = ap.parse_args()

    principales, secundarias, llamo = analizar_m3(args.config)
    m4 = analizar_m4()
    configs = sorted(principales)

    L = ["# Fallos del tutor · análisis exploratorio", "",
         "Generado con `scripts/taxonomia_errores.py`.", "",
         "> **Esto no es una medición.** Las cifras del arnés (`acierto`, "
         "`abstuvo`, `alucino`) están en los scorecards y no se recalculan aquí. "
         "Lo de abajo son etiquetas heurísticas sobre respuestas ya guardadas, "
         "pensadas para orientar dónde mirar. Nadie anotó estos casos a mano.", "",
         "Tres reglas de lectura:", "",
         "1. **Una etiqueta principal por caso.** La columna suma exactamente el "
         "número de casos, así que un caso con varias señales no se cuenta como "
         "varios fallos. Las secundarias van en su propia tabla y no se suman.",
         "2. **`sin_evidencia`** marca las categorías que los datos guardados no "
         "permiten decidir. No se estiman.",
         "3. **Las etiquetas condicionadas** solo pueden aplicarse a cierto tipo "
         "de respuesta. Un cero ahí no es una mejora.", "",
         "`explicacion_confusa` se asigna a casos que el arnés contó como "
         "**acierto**: es una pega de calidad, no un fallo, y no se suma a los "
         "fallos.", "",
         "## Qué significa cada etiqueta", "",
         "| Etiqueta | Tipo | Cómo se decide |", "|---|---|---|"]
    for k, (tipo, como) in ETIQUETAS.items():
        L.append(f"| `{k}` | {tipo} | {como} |")

    L += ["", "## M3 · 53 casos por configuración · etiqueta principal", "",
          "Fuente: `respuestas_s10.jsonl` + `detalle_s10_v2.csv`.", "",
          "| Etiqueta principal | " + " | ".join(configs) + " |",
          "|---" * (len(configs) + 1) + "|"]
    orden = ["acierto", "abstencion_correcta", "aritmetica_falsa",
             "calculo_sin_herramienta", "final_no_coincide_con_pasos",
             "planteamiento_incorrecto", "fallo_con_contexto_recuperado",
             "no_consulto_documentos", "no_reconocio_el_limite",
             "negativa_innecesaria", "explicacion_confusa", "fallo_sin_etiqueta"]
    vistas = {e for c in principales.values() for e in c}
    for e in orden + sorted(vistas - set(orden)):
        if e not in vistas:
            continue
        L.append(f"| `{e}` | " + " | ".join(str(principales[c].get(e, 0))
                                            for c in configs) + " |")
    L.append("| **total** | " + " | ".join(str(sum(principales[c].values()))
                                           for c in configs) + " |")

    L += ["", "### Señales secundarias (no suman fallos)", "",
          "| Señal | " + " | ".join(configs) + " |", "|---" * (len(configs) + 1) + "|"]
    for e in sorted({e for c in secundarias.values() for e in c}):
        L.append(f"| `{e}` | " + " | ".join(str(secundarias[c].get(e, 0))
                                            for c in configs) + " |")

    L += ["", "### Por qué `calculo_sin_herramienta` no se puede comparar entre "
          "configuraciones", "",
          "La etiqueta exige que la respuesta no haya llamado a ninguna "
          "herramienta. Una configuración que llama casi siempre no puede "
          "recibirla, de modo que su cero no dice nada sobre si sus cuentas son "
          "correctas. Proporción de casos con al menos una llamada:", "",
          "| Configuración | casos con llamada | de | % |", "|---|---|---|---|"]
    for c in configs:
        n, con = llamo[c]
        L.append(f"| {c} | {con} | {n} | {100 * con / n:.0f}% |")
    L += ["", "Para la aritmética sí hay evidencia independiente de las "
          "llamadas: `aritmetica_falsa` la da el verificador determinista "
          "(`scripts/agente/verificador_aritmetico.py`) leyendo las igualdades "
          "escritas en la respuesta. Esa fila sí es comparable, con el límite de "
          "que solo ve las igualdades que el tutor escribió explícitamente."]

    if m4:
        L += ["", "## M4 · 12 casos visuales", "",
              "Fuente: `m4/scorecard_m4.csv`. La etiqueta de un adversarial la "
              "fija su columna `esperado`: solo `vis-10` es ilegible; `vis-11` es "
              "fuera de dominio y `vis-12` es un dato que la hoja no tiene.", "",
              "| Etiqueta | n |", "|---|---|"]
        for k, v in m4.most_common():
            L.append(f"| `{k}` | {v} |")
        L += ["", "`sin_conclusion_formato` **no es un fallo de criterio**: el "
              "tutor no llegó a emitir un veredicto. Se cuenta aparte porque se "
              "arregla de otra manera.", "",
              "Lo que estos 12 casos **no** permiten decir es si la lectura de la "
              "hoja fue correcta: haría falta la transcripción anotada a mano, que "
              "no existe. Por eso `lectura_visual_erronea` queda `sin_evidencia`."]

    SALIDA.parent.mkdir(exist_ok=True)
    SALIDA.write_text("\n".join(L) + "\n", encoding="utf-8")

    # `explicacion_confusa` se asigna a casos que el arnés contó como acierto:
    # es una pega de calidad, no un fallo. Sumarla infla el recuento.
    no_fallo = ("acierto", "abstencion_correcta", "explicacion_confusa")
    for c in configs:
        fallos = sum(n for e, n in principales[c].items() if e not in no_fallo)
        print(f"\n{c}  ({fallos} casos fallados de {sum(principales[c].values())}; "
              f"{principales[c].get('explicacion_confusa', 0)} aciertos con "
              "explicación confusa)")
        for e, n in principales[c].most_common():
            if e not in no_fallo:
                print(f"    {n:3d}  {e}")
    if m4:
        print("\nM4 visual:", dict(m4))
    print(f"\n-> {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
