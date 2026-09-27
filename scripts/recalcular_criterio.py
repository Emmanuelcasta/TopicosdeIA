"""
Re-evaluación de las corridas con el criterio corregido de `adv-02`.

Por qué hace falta: S08 (Colab) y S10 (Lightning) se ejecutaron con la versión
ANTIGUA de `data/eval_set_m2.jsonl`, en la que al caso `adv-02` (división entre
cero) le faltaban las claves "no definido" y "no tiene sentido". Los notebooks lo
avisaron imprimiendo `SHA coincide: False`. Respuestas correctas como
*"Dividir entre cero no tiene sentido en matemáticas"* se contaron como fallo.

Por qué se puede corregir sin GPU: el criterio de la Dimensión 3 es una función
pura del texto de la respuesta, y las respuestas están guardadas en los CSV de
detalle. La similitud y la nota del juez no dependen del criterio, así que no
cambian. Aquí se re-evalúa cada fila con el eval set corregido y se reescriben
los agregados afectados.

Salidas (sufijo `_v2`, no se sobrescribe nada):
    resultados/detalle_rag_avanzado_v2.csv   · resultados/scorecard_rag_avanzado_v2.csv
    resultados/detalle_s10_v2.csv            · resultados/scorecard_agente_v2.csv
    resultados/correccion_adv02.md           · el informe de qué cambió

Uso:
    python scripts/recalcular_criterio.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from nb_comun import CODIGO_METRICAS  # noqa: E402
from nb_comun_eval import CODIGO_EVALUACION  # noqa: E402

RAIZ = AQUI.parent
RES = RAIZ / "M3_rag_agentes" / "resultados"

_ns: dict = {}
exec(CODIGO_METRICAS, _ns)      # noqa: S102 — el mismo código que embeben los notebooks
exec(CODIGO_EVALUACION, _ns)    # noqa: S102
evaluar_caso = _ns["evaluar_caso"]


def cargar_casos() -> dict:
    casos = {}
    for archivo in ("eval_set_m2.jsonl", "eval_set_m3_agente.jsonl"):
        for linea in (RAIZ / "data" / archivo).read_text(encoding="utf-8").splitlines():
            caso = json.loads(linea)
            casos[caso["id"]] = caso
    return casos


CASOS = cargar_casos()


def reevaluar(ruta: Path, col_sistema: str) -> tuple[list[dict], list[dict]]:
    filas = list(csv.DictReader(ruta.open(encoding="utf-8")))
    cambios = []
    for f in filas:
        ev = evaluar_caso(CASOS[f["id"]], f["respuesta"])
        antes = f["acierto"].lower() == "true"
        if ev["acierto"] != antes or f["motivo"] != ev["motivo"]:
            if ev["acierto"] != antes:
                cambios.append({"sistema": f[col_sistema], "id": f["id"], "bloque": f["bloque"],
                                "antes": antes, "despues": ev["acierto"], "motivo": ev["motivo"]})
            f["acierto"], f["abstuvo"], f["alucino"], f["motivo"] = (
                str(ev["acierto"]), str(ev["abstuvo"]), str(ev["alucino"]), ev["motivo"])
    return filas, cambios


def escribir(ruta: Path, filas: list[dict]) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def _b(v) -> bool:
    return str(v).lower() == "true"


def scorecard_s08(filas: list[dict]) -> list[list]:
    """Reconstruye `scorecard_rag_avanzado.csv` desde el detalle re-evaluado."""
    sistemas = list(dict.fromkeys(f["sistema"] for f in filas))
    por = {s: [f for f in filas if f["sistema"] == s] for s in sistemas}
    bloques = {"calculo": 12, "conocimiento": 4, "adversarial": 5}

    def fila(nombre, fn):
        return [nombre] + [fn(por[s]) for s in sistemas]

    salida = [["dimension"] + sistemas]
    salida.append(fila("exactitud_calculo", lambda g: round(
        sum(_b(f["acierto"]) for f in g if f["bloque"] == "calculo") / bloques["calculo"], 3)))
    salida.append(fila("sim_embeddings_prom", lambda g: round(sum(float(f["sim"]) for f in g) / len(g), 3)))
    salida.append(fila("llm_juez_prom", lambda g: round(sum(float(f["juez"]) for f in g) / len(g), 3)))
    salida.append(fila("aciertos_total", lambda g: f"{sum(_b(f['acierto']) for f in g)}/{len(g)}"))
    for b, n in bloques.items():
        salida.append(fila(f"aciertos_{b}", lambda g, b=b, n=n:
                           f"{sum(_b(f['acierto']) for f in g if f['bloque'] == b)}/{n}"))
    salida.append(fila("alucinaciones", lambda g: sum(_b(f["alucino"]) for f in g)))
    salida.append(fila("abstenciones", lambda g: sum(_b(f["abstuvo"]) for f in g)))
    return salida


def scorecard_s10(filas: list[dict], original: Path) -> list[list]:
    """Parchea en `scorecard_agente.csv` solo las filas que dependen del criterio."""
    tabla = list(csv.reader(original.open(encoding="utf-8")))
    cabecera, cuerpo = tabla[0], tabla[1:]
    configs = cabecera[1:]
    por = {c: [f for f in filas if f["config"] == c] for c in configs}

    def valor(metrica, g):
        m2 = [f for f in g if f["conjunto"] == "M2"]
        m3 = [f for f in g if f["conjunto"] == "M3"]
        bloque = lambda b, fs: sum(_b(f["acierto"]) for f in fs if f["bloque"] == b)  # noqa: E731
        return {
            "Exactitud en cálculo": bloque("calculo", m2) / 12,
            "Aciertos totales /21": float(sum(_b(f["acierto"]) for f in m2)),
            "Conocimiento /4": float(bloque("conocimiento", m2)),
            "Adversarial /5": float(bloque("adversarial", m2)),
            "Alucinaciones": float(sum(_b(f["alucino"]) for f in m2)),
            "Abstenciones": float(sum(_b(f["abstuvo"]) for f in m2)),
            "Aritmética /13": float(bloque("aritmetica", m3)),
            "Compuesto /8": float(bloque("compuesto", m3)),
            "Mal escrito /11": float(bloque("mal_escrito", m3)),
            "Alucinaciones M3": float(sum(_b(f["alucino"]) for f in m3)),
        }.get(metrica)

    for fila in cuerpo:
        nuevo = [valor(fila[0], por[c]) for c in configs]
        if nuevo[0] is not None:
            fila[1:] = [str(v) for v in nuevo]
    return [cabecera] + cuerpo


def main() -> None:
    informe = ["# Corrección del criterio de `adv-02`", "",
               "Las corridas de S08 (Colab) y S10 (Lightning) usaron la versión antigua de",
               "`data/eval_set_m2.jsonl`, sin las claves \"no definido\" y \"no tiene sentido\".",
               "Los notebooks lo avisaron con `SHA coincide: False`.", "",
               "Las respuestas no cambian: solo se vuelve a aplicar el criterio de la Dimensión 3",
               "sobre el texto ya generado. Similitud y juez quedan iguales.", ""]

    filas_s08, cambios_s08 = reevaluar(RES / "detalle_rag_avanzado.csv", "sistema")
    escribir(RES / "detalle_rag_avanzado_v2.csv", filas_s08)
    with (RES / "scorecard_rag_avanzado_v2.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(scorecard_s08(filas_s08))

    filas_s10, cambios_s10 = reevaluar(RES / "detalle_s10.csv", "config")
    escribir(RES / "detalle_s10_v2.csv", filas_s10)
    with (RES / "scorecard_agente_v2.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(scorecard_s10(filas_s10, RES / "scorecard_agente.csv"))

    for titulo, cambios in (("S08", cambios_s08), ("S10", cambios_s10)):
        informe += [f"## {titulo}: {len(cambios)} caso(s) cambian", ""]
        if cambios:
            informe += ["| sistema | caso | antes | después | motivo |", "|---|---|---|---|---|"]
            informe += [f"| {c['sistema']} | {c['id']} | {'OK' if c['antes'] else 'FALLA'} | "
                        f"{'OK' if c['despues'] else 'FALLA'} | {c['motivo']} |" for c in cambios]
        informe.append("")
        print(f"{titulo}: {len(cambios)} cambios")
        for c in cambios:
            print(f"   {c['sistema']:20s} {c['id']:8s} -> {'OK' if c['despues'] else 'FALLA'}")

    (RES / "correccion_adv02.md").write_text("\n".join(informe), encoding="utf-8")
    print("\nEscritos: detalle_*_v2.csv, scorecard_*_v2.csv, correccion_adv02.md")


if __name__ == "__main__":
    main()
