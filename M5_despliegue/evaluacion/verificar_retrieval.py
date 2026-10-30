"""¿La app monta de verdad el retrieval y el enrutador de S08-D?

Qué comprueba y qué no. **No** mide aciertos: no carga el LLM. Comprueba que las
piezas que la app monta reproducen las del notebook, comparando contra las cifras
que `nb_rag_avanzado` dejó en `m3_rag_avanzado.json`:

    pesos de la regresión del enrutador      margen, numeros, rerank
    exactitud leave-one-out sobre dev        33 consultas
    ruta decidida para cada caso del eval    21 casos de M2

Si los pesos y las 21 rutas coinciden, el enrutador servido es el medido. Si no
coinciden, lo que se despliegue bajo el nombre «S08-D» no es S08-D, y esto lo
dice antes de la demo en vez de después.

Por qué hace falta. Dos discrepancias se encontraron exactamente así:
`rerank_max=None` dejaba el tercer rasgo en 0,0, y `st.encode()` sin
`normalize_embeddings=True` cambiaba la escala del rasgo `margen`. Con cualquiera
de las dos, la exactidud leave-one-out bajaba de 0,970 a 0,939: parecido, y
distinto.

    python M5_despliegue/evaluacion/verificar_retrieval.py
    python M5_despliegue/evaluacion/verificar_retrieval.py --guardar

Necesita `sentence-transformers`, `rank-bm25` y `scikit-learn` (están en
`config/requirements.txt`) y descarga dos modelos la primera vez. No necesita GPU:
en CPU tarda unos dos minutos.
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

REFERENCIA = RAIZ / "M3_rag_agentes" / "resultados" / "m3_rag_avanzado.json"
EVAL_SET = RAIZ / "data" / "eval_set_m2.jsonl"
SALIDA = RAIZ / "M5_despliegue" / "evaluacion" / "verificacion_retrieval.json"
TOL = 1e-3      # los pesos de referencia están guardados con todos sus decimales


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guardar", action="store_true",
                    help="escribe el resultado en verificacion_retrieval.json")
    args = ap.parse_args()

    import pipeline
    from enrutador import Enrutador
    from nb_rag_avanzado import CODIGO_CONSULTAS_DEV

    ref = json.loads(REFERENCIA.read_text(encoding="utf-8"))["enrutador"]

    chunks, metadatos, _ = pipeline.cargar_corpus()
    t0 = time.perf_counter()
    buscar = pipeline.buscador_hibrido(chunks)
    if buscar is None:
        print("FALLA: no se pudo montar el retrieval híbrido. Faltan "
              "dependencias de config/requirements.txt o no hay red para los "
              "modelos. La app se replegaría a búsqueda léxica, que NO es S08-D.")
        return 2
    print(f"retrieval híbrido + reranker montado en {time.perf_counter() - t0:.1f}s "
          f"sobre {len(chunks)} chunks")

    ns: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(CODIGO_CONSULTAS_DEV, ns)      # noqa: S102 — fuente del repositorio
    e = Enrutador(buscar.encode, buscar.rerank_max, ns["consultas_dev"])
    loo = e.leave_one_out()

    fallos = []

    print("\npesos de la regresión logística")
    for rasgo, esperado in ref["pesos"].items():
        obtenido = e.pesos[rasgo]
        ok = abs(obtenido - esperado) <= TOL
        fallos += [] if ok else [f"peso {rasgo}: {obtenido:.4f} != {esperado:.4f}"]
        print(f"  {rasgo:8s} {obtenido:+.4f}  referencia {esperado:+.4f}  "
              f"{'ok' if ok else 'DISTINTO'}")

    print("\nvalidación leave-one-out sobre las 33 consultas de desarrollo")
    for clave, nombre in (("loo_dev", "exactitud"),
                          ("loo_dev_solo_centroides", "solo_centroides")):
        obtenido, esperado = loo[nombre], ref[clave]
        ok = abs(obtenido - esperado) <= TOL
        fallos += [] if ok else [f"{clave}: {obtenido:.4f} != {esperado:.4f}"]
        print(f"  {clave:24s} {obtenido:.4f}  referencia {esperado:.4f}  "
              f"{'ok' if ok else 'DISTINTO'}")
    if loo["n"] != ref["n_dev"]:
        fallos.append(f"n de desarrollo: {loo['n']} != {ref['n_dev']}")

    print("\nruta de cada caso del eval set fijo de M2")
    eval_set = [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines()]
    rutas, distintas = {}, 0
    for c in eval_set:
        d = e.clasificar(c["input"])
        rutas[c["id"]] = d.ruta
        esperada = ref["rutas_eval"].get(c["id"])
        if esperada is not None and d.ruta != esperada:
            distintas += 1
            print(f"  {c['id']:8s} {d.ruta:9s} referencia {esperada:9s} DISTINTA")
    if distintas:
        fallos.append(f"{distintas} de {len(eval_set)} rutas distintas")
    print(f"  {len(eval_set) - distintas} de {len(eval_set)} coinciden con la "
          "referencia")

    print()
    if fallos:
        print("NO reproduce S08-D:")
        for f in fallos:
            print("  -", f)
    else:
        print("Reproduce el enrutador de S08-D: pesos, leave-one-out y las 21 "
              "rutas coinciden con m3_rag_avanzado.json.")

    if args.guardar:
        SALIDA.write_text(json.dumps({
            "reproduce": not fallos,
            "fallos": fallos,
            "pesos": {k: round(v, 6) for k, v in e.pesos.items()},
            "pesos_referencia": ref["pesos"],
            "loo_dev": round(loo["exactitud"], 6),
            "loo_dev_referencia": ref["loo_dev"],
            "loo_dev_solo_centroides": round(loo["solo_centroides"], 6),
            "loo_dev_solo_centroides_referencia": ref["loo_dev_solo_centroides"],
            "n_dev": loo["n"],
            "rutas_eval": rutas,
            "rutas_eval_referencia": ref["rutas_eval"],
            "nota": "No mide aciertos: el LLM no se carga. Solo comprueba que el "
                    "retrieval y el enrutador servidos son los medidos en S08.",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"-> {SALIDA.relative_to(RAIZ)}")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
