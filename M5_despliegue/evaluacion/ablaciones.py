"""Las ablaciones de §5, preparadas para correr en Colab con GPU.

Qué hace. Toma el **eval set fijo de M2** sin tocarlo, monta cada configuración
declarada en `app/configuraciones.py` y pasa cada una por el **mismo harness** de
`nb_comun_eval.CODIGO_HARNESS` —las tres dimensiones, el mismo criterio por
caso—. Nada de evaluación se reimplementa aquí: si se reimplementara, las cifras
nuevas no serían comparables con las de M1–M4.

La regla del diseño. Cada fila cambia **una sola pieza** respecto a su fila de
referencia. Una tabla donde dos filas difieren en tres cosas no permite atribuir
la diferencia a ninguna.

    fila  configuración        única diferencia          referencia  medido
    0     s08d                 — (reproducción)          —           18/21
    1     solo_lora            sin RAG ni enrutador      0           14/21
    2     s08c                 sin enrutador             0           13/21
    3     s08d_herramientas    + 18 herramientas         0           no
    4     s08d_verificador     + verificador (1 reintento)  3        no
    5     s08d_normalizador    + normalizador            0           no

Las filas 0 a 2 son **controles**: sus cifras ya están medidas. Si no las
reproducen, el entorno no es el que midió y las filas 3 a 5 no se interpretan.
El script lo dice y devuelve código 2.

**Una excepción honesta a la regla de un cambio.** La fila 3 cambia dos cosas:
las herramientas y el prompt (`SYSTEM_RAG` → `SYSTEM_C5_AGENTE`). No hay manera
de separarlas: sin un prompt que explique el formato de llamada, el modelo no
llama a nada, así que «herramientas con el prompt de RAG» no es una condición
medible, es una configuración que no funciona. Si la fila 3 baja respecto a la 0,
la causa puede ser cualquiera de las dos, y así hay que reportarlo. Las filas 4 y
5 sí cambian una sola pieza.

    python M5_despliegue/evaluacion/ablaciones.py --plan      # sin modelos
    python M5_despliegue/evaluacion/ablaciones.py --solo-controles
    python M5_despliegue/evaluacion/ablaciones.py --filas 0,1,2,3,4,5

Requiere GPU y el modelo base descargado; sin ellos se queja y no finge nada. El
juez LLM es parte del harness, así que cada fila gasta su propia generación.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "M5_despliegue" / "app"), str(RAIZ / "scripts"),
                str(RAIZ / "scripts" / "agente")]

EVAL_SET = RAIZ / "data" / "eval_set_m2.jsonl"
SALIDA = RAIZ / "M5_despliegue" / "evaluacion" / "ablaciones.json"

# fila -> (config, única diferencia, fila de referencia, resultado medido)
FILAS = [
    ("s08d", "— (reproducción de la línea base)", None, "18/21"),
    ("solo_lora", "se quita RAG y enrutador", 0, "14/21"),
    ("s08c", "se quita solo el enrutador", 0, "13/21"),
    ("s08d_herramientas",
     "se añaden las 18 herramientas (y, forzosamente, el prompt del agente)",
     0, None),
    ("s08d_verificador", "se añade el verificador (1 reintento)", 3, None),
    ("s08d_normalizador", "se añade el normalizador de preguntas", 0, None),
]
CONTROLES = [0, 1, 2]


def cargar_harness():
    """El harness canónico, ejecutado desde su única copia en el repositorio.

    El juez y la similitud cargan sus propios modelos al definirse, así que esto
    no funciona sin red ni sin el modelo en caché. Se dice con una frase en vez
    de con una traza de 20 líneas.
    """
    from nb_comun_eval import (
        CODIGO_EVALUACION, CODIGO_HARNESS, CODIGO_JUEZ, CODIGO_SIMILITUD,
    )

    ns: dict = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for codigo in (CODIGO_EVALUACION, CODIGO_SIMILITUD, CODIGO_JUEZ,
                           CODIGO_HARNESS):
                exec(codigo, ns)      # noqa: S102 — fuente propia del repositorio
    except Exception as e:      # noqa: BLE001
        raise SystemExit(
            f"no se pudo montar el harness ({type(e).__name__}): necesita el "
            "modelo base y el de embeddings descargados, y GPU para que esto "
            "termine en un tiempo razonable. Usa --plan para revisar las "
            "configuraciones sin cargar nada.") from e
    return ns["harness"]


def imprimir_plan() -> int:
    """Monta cada configuración sin modelo y enseña qué cambiaría cada fila."""
    import pipeline
    from configuraciones import obtener

    print(f"{'fila':5s} {'configuración':20s} {'medido':8s} única diferencia")
    for i, (config, diferencia, referencia, medido) in enumerate(FILAS):
        cfg = obtener(config)
        ref = "—" if referencia is None else f"fila {referencia}"
        print(f"{i:<5d} {config:20s} {medido or 'no':8s} {diferencia} "
              f"(vs {ref})")
        print(f"      adaptador={cfg.adaptador} retrieval={cfg.retrieval} "
              f"enrutador={cfg.enrutador} herramientas={cfg.herramientas} "
              f"verificador={cfg.verificador} normalizador={cfg.normalizador} "
              f"system={cfg.system}")
        e = pipeline.Tutor(config).estado
        print(f"      {e.resumen()}")
    print("\nControles:", CONTROLES, "· si no reproducen su cifra medida, las "
          "filas experimentales no se interpretan.")
    return 0


def correr(fila: int, harness, eval_set) -> dict:
    import pipeline

    config, diferencia, referencia, medido = FILAS[fila]
    tutor = pipeline.Tutor(config)
    print(f"\n=== fila {fila} · {config} · {diferencia} ===")
    tutor._cargar_modelo()
    print("  ", tutor.estado.resumen())
    if not tutor.estado.listo:
        return {"fila": fila, "config": config, "error": tutor.estado.motivos}

    # Se guarda la traza de cada caso al pasar, para no volver a generar: una
    # segunda llamada por caso duplicaría el coste y podría dar otra respuesta.
    trazas: dict = {}

    def sistema(pregunta):
        r = tutor.responder(pregunta)
        trazas[pregunta] = {"herramientas": r["herramientas"],
                            "contextos": r["contextos"],
                            "pasos_llm": r["pasos_llm"],
                            "latencia_s": r["latencia_s"],
                            "verificador": r.get("verificador")}
        return r["respuesta"]

    t0 = time.perf_counter()
    sc = harness(eval_set, sistema, nombre=config)
    for d in sc["detalle"]:
        d.update(trazas.get(d["input"], {}))
    casos_con_llamada = sum(1 for d in sc["detalle"] if d.get("herramientas"))
    casos_con_contexto = sum(1 for d in sc["detalle"] if d.get("contextos"))
    sc.update({"fila": fila, "config": config, "diferencia_unica": diferencia,
               "fila_referencia": referencia, "medido_previamente": medido,
               "casos_con_llamada": casos_con_llamada,
               "casos_con_contexto": casos_con_contexto,
               "segundos_total": round(time.perf_counter() - t0, 1),
               "segundos_por_caso": round((time.perf_counter() - t0) / len(eval_set), 2),
               "estado": tutor.estado.resumen()})
    return sc


def revisar_controles(hechas: dict) -> list:
    """Un control que no reproduce su cifra invalida el resto de la tabla."""
    problemas = []
    for fila in CONTROLES:
        sc = hechas.get(fila)
        if sc is None or "aciertos" not in sc:
            continue
        esperado = int(FILAS[fila][3].split("/")[0])
        if sc["aciertos"] != esperado:
            problemas.append(
                f"fila {fila} ({FILAS[fila][0]}): {sc['aciertos']}/21, se midió "
                f"{FILAS[fila][3]}")
    return problemas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--filas", default="all",
                    help="por ejemplo 0,1,2 · 'all' corre las seis")
    ap.add_argument("--solo-controles", action="store_true",
                    help="atajo para --filas 0,1,2")
    ap.add_argument("--plan", action="store_true",
                    help="enseña las filas y las configuraciones sin cargar "
                         "ningún modelo; no mide nada")
    args = ap.parse_args()

    if args.plan:
        return imprimir_plan()

    filas = (CONTROLES if args.solo_controles else
             list(range(len(FILAS))) if args.filas == "all" else
             [int(x) for x in args.filas.split(",")])

    eval_set = [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines()]
    print(f"eval set fijo de M2: {len(eval_set)} casos · {EVAL_SET}")
    print("NO se modifica. Los 12 casos visuales de M4 no se usan aquí.")
    harness = cargar_harness()

    hechas = {f: correr(f, harness, eval_set) for f in filas}

    print("\n" + "=" * 78)
    print(f"{'fila':5s} {'configuración':20s} {'aciertos':9s} {'alucin.':8s} "
          f"{'cálc.':6s} {'conoc.':7s} {'advers.':8s} {'s/caso':7s} {'llam.':6s} medido")
    for f in filas:
        sc = hechas[f]
        if "aciertos" not in sc:
            print(f"{f:<5d} {sc['config']:20s} NO SE PUDO MONTAR: {sc['error']}")
            continue
        print(f"{f:<5d} {sc['config']:20s} {sc['aciertos']:>2d}/21     "
              f"{sc['alucinaciones']:<8d} {sc['aciertos_calculo']:>2d}/12  "
              f"{sc['aciertos_conocimiento']:>2d}/4   "
              f"{sc['aciertos_adversarial']:>2d}/5    "
              f"{sc['segundos_por_caso']:>5.1f}   "
              f"{sc['casos_con_llamada']:>2d}/21  "
              f"{sc['medido_previamente'] or '—'}")

    problemas = revisar_controles(hechas)
    print()
    if problemas:
        print("LOS CONTROLES NO REPRODUCEN LO MEDIDO:")
        for p in problemas:
            print("  -", p)
        print("\nLas filas experimentales no se interpretan: el entorno no es el "
              "que midió esas cifras. Hay que arreglar eso antes de seguir.")
    else:
        print("Controles correctos. Recordatorio para leer las filas 3 a 5: con "
              "21 casos, una diferencia de uno o dos casos no distingue nada.")

    SALIDA.write_text(json.dumps(
        {"eval_set": str(EVAL_SET.relative_to(RAIZ)), "n": len(eval_set),
         "controles_ok": not problemas, "problemas_de_control": problemas,
         "filas": [hechas[f] for f in filas]},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {SALIDA.relative_to(RAIZ)}")
    return 2 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
