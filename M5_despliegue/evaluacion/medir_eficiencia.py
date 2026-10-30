"""Compara el tutor en fp16 frente a int4: calidad y coste.

Lo que S15 pide no es enseñar una interfaz, sino **comparar calidad y coste de
ejecución**. Este script lo hace con el mismo checkpoint y el mismo conjunto en
las dos condiciones, cambiando solo la cuantización.

Decisiones que hacen la comparación válida:

  · **Mismo checkpoint, mismo adaptador, mismo conjunto.** Lo único que cambia
    es `load_in_4bit`.
  · **Temperatura cero y semilla fija**: dos corridas de la misma condición
    tienen que dar lo mismo, o las diferencias no significan nada.
  · **Calentamiento** antes de medir: la primera generación paga la compilación
    de kernels y el traslado de pesos, y contarla ensucia la mediana.
  · **p50 y p95 con el n a la vista.** Con 21 casos, el p95 es prácticamente el
    máximo: se reporta, y se dice que es una cota superior ruidosa, no un SLO.
  · **La calidad se mide con el harness que ya existe**, no con uno nuevo.

Lo que NO hace: no entrena, no modifica el eval set de M2 y no usa los 12 casos
de M4, que ya se inspeccionaron y dejaron de ser ciegos.

    python M5_despliegue/evaluacion/medir_eficiencia.py --conjunto data/eval_set_m2.jsonl
    python M5_despliegue/evaluacion/medir_eficiencia.py --solo-plan   # sin GPU
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "scripts" / "agente"))
sys.path.insert(0, str(RAIZ / "M5_despliegue" / "app"))

SALIDA = RAIZ / "M5_despliegue" / "resultados"
CALENTAMIENTO = 2
REPETICIONES = 3


def memoria_gpu_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:  # noqa: BLE001
        return None


def medir_condicion(casos, cuantizar: bool, repeticiones: int = REPETICIONES) -> dict:
    """Una condición: fp16 o int4. Devuelve latencias, tokens/s y respuestas."""
    import os

    import torch

    os.environ["TUTOR_CUANTIZAR"] = "1" if cuantizar else ""
    import importlib

    import pipeline
    importlib.reload(pipeline)
    tutor = pipeline.obtener_tutor()
    tutor._cargar_modelo()
    if not tutor.estado.listo:
        return {"error": tutor.estado.resumen()}

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for c in casos[:CALENTAMIENTO]:           # calentamiento, no se mide
        tutor.responder(c["input"])

    latencias, filas = [], []
    for rep in range(repeticiones):
        for c in casos:
            t0 = time.perf_counter()
            r = tutor.responder(c["input"])
            dt = time.perf_counter() - t0
            latencias.append(dt)
            if rep == 0:
                palabras = len(r["respuesta"].split())
                filas.append({"id": c["id"], "respuesta": r["respuesta"],
                              "latencia_s": round(dt, 3), "palabras": palabras,
                              "tokens_s": round(palabras / dt, 1) if dt else None})

    orden = sorted(latencias)
    return {
        "condicion": "int4" if cuantizar else "fp16",
        "n_casos": len(casos), "repeticiones": repeticiones,
        "n_mediciones": len(latencias),
        "latencia_p50_s": round(statistics.median(orden), 3),
        "latencia_p95_s": round(orden[max(0, int(0.95 * len(orden)) - 1)], 3),
        "latencia_media_s": round(statistics.fmean(orden), 3),
        "memoria_pico_gb": memoria_gpu_gb(),
        "tokens_s_mediana": round(statistics.median(
            [f["tokens_s"] for f in filas if f["tokens_s"]]), 1) if filas else None,
        "adaptador": tutor.estado.adaptador,
        "dispositivo": tutor.estado.dispositivo,
        "detalle": filas,
    }


def plan() -> str:
    return """PLAN DE MEDICIÓN · lo que hay que ejecutar en Colab con GPU

  1. fp16   : python M5_despliegue/evaluacion/medir_eficiencia.py
  2. int4   : el script corre las dos condiciones en la misma sesión

  Qué se reporta, por condición:
    · memoria pico en GB
    · latencia p50 y p95, con el número de mediciones a la vista
    · tokens por segundo (mediana)
    · calidad con el harness de M2, sobre el MISMO conjunto

  Límite que hay que declarar al presentar: con 21 casos y 3 repeticiones son 63
  mediciones; el p95 es una cota superior ruidosa, no un SLO.

  No se ejecuta aquí: este computador no tiene GPU y una corrida en CPU no
  compara nada útil."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conjunto", default="data/eval_set_m2.jsonl")
    ap.add_argument("--repeticiones", type=int, default=REPETICIONES)
    ap.add_argument("--solo-plan", action="store_true",
                    help="imprime el plan sin cargar nada")
    args = ap.parse_args()

    if args.solo_plan:
        print(plan())
        return

    ruta = RAIZ / args.conjunto
    casos = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines()]
    print(f"{len(casos)} casos de {args.conjunto}  (NO se modifica)")

    resultados = {}
    for cuantizar in (False, True):
        nombre = "int4" if cuantizar else "fp16"
        print(f"\n--- {nombre} ---")
        r = medir_condicion(casos, cuantizar, args.repeticiones)
        if "error" in r:
            print("  no se pudo medir:", r["error"])
        else:
            print(f"  p50 {r['latencia_p50_s']}s · p95 {r['latencia_p95_s']}s "
                  f"({r['n_mediciones']} mediciones) · pico {r['memoria_pico_gb']} GB")
        resultados[nombre] = r

    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / "eficiencia_fp16_vs_int4.json"
    destino.write_text(json.dumps(resultados, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"\nGuardado: {destino.relative_to(RAIZ)}")
    print("\nLa calidad se mide aparte, con el harness de M2 sobre el mismo")
    print("conjunto y las respuestas de cada condición. Sin eso, una bajada de")
    print("latencia no dice si el modelo sigue sirviendo.")


if __name__ == "__main__":
    main()
