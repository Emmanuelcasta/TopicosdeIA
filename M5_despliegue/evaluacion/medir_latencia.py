"""Latencia del tutor: arranque en frío aparte de las consultas calientes.

Por qué separadas. La primera petición paga la carga del modelo, del adaptador y
del retrieval —decenas de segundos— y promediarla con las demás da un número que
no describe ninguna de las dos cosas: ni lo que tarda arrancar, ni lo que tarda
responder. Son dos cifras y se reportan como dos.

**Lo que este script se niega a contar.** Durante la depuración de Kaggle las
peticiones fallaban en 5,71–5,83 s. Eso es el tiempo hasta la excepción, no la
latencia del modelo. Aquí solo entran en la estadística las respuestas con
`ok=True`; las fallidas se cuentan aparte y, si las hay, el informe lo dice en
vez de esconderlas en una mediana.

Con n pequeño el p95 es prácticamente el máximo: se reporta, con el número de
muestras al lado, y se dice que es una cota superior ruidosa y no un compromiso
de servicio.

    python M5_despliegue/evaluacion/medir_latencia.py --repeticiones 8
    python M5_despliegue/evaluacion/medir_latencia.py --solo-plan   # sin GPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "M5_despliegue" / "app"),
                str(RAIZ / "M5_despliegue" / "evaluacion"),
                str(RAIZ / "scripts"), str(RAIZ / "scripts" / "agente")]

SALIDA = RAIZ / "M5_despliegue" / "resultados" / "latencia.json"

# Preguntas de las tres vías que `s08d`/`revision_foto` enrutan distinto, para
# que la mediana no describa solo un tipo de consulta.
PREGUNTAS = [
    ("calculo", "¿Cuánto es 3/4 + 1/8?"),
    ("calculo", "Un taller compra 3 cajas a 18500 pesos cada una, ¿cuánto paga?"),
    ("conocimiento", "¿Cuál es la nota mínima aprobatoria del colegio?"),
    ("adversarial", "Explica por qué 9 es un número primo."),
]


def percentiles(muestras) -> dict:
    import statistics

    orden = sorted(muestras)
    if not orden:
        return {}
    return {
        "n": len(orden),
        "p50_s": round(statistics.median(orden), 2),
        "p95_s": round(orden[max(0, int(0.95 * len(orden)) - 1)], 2),
        "min_s": round(orden[0], 2),
        "max_s": round(orden[-1], 2),
    }


def plan() -> str:
    return """Qué mide y cómo, para poder leerlo sin malinterpretarlo:

  arranque en frío   una sola muestra: la primera pregunta, que carga modelo,
                     adaptador y retrieval. No se promedia con nada.
  consultas calientes con el modelo ya cargado. De aquí salen p50 y p95.
  por bloque         cálculo, conocimiento y adversarial por separado: el
                     enrutador los trata distinto y mezclarlos esconde cuál
                     cuesta.
  fallidas           contadas aparte y NUNCA en la estadística. Una petición
                     que lanza a los 5,8 s no es una latencia de 5,8 s.

Requiere GPU y el modelo cargado. Sin eso no se ejecuta ni se simula."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeticiones", type=int, default=8,
                    help="consultas calientes, repartidas entre las preguntas")
    ap.add_argument("--solo-plan", action="store_true")
    args = ap.parse_args()

    if args.solo_plan:
        print(plan())
        return 0

    os.environ.setdefault("USE_TF", "0")
    import compat_torch

    info = compat_torch.asegurar_distributed_tensor()
    print("Entorno: " + " | ".join(
        f"{k} {v}" for k, v in compat_torch.versiones().items()))
    print(f"torch.distributed.tensor: {info['accion'] or info['error'] or '—'}\n")

    from pipeline import Tutor

    config = os.environ.get("TUTOR_CONFIG")
    tutor = Tutor(config)
    print(f"Configuración: {tutor.estado.config_nombre}")

    # --- arranque en frío: la primera pregunta de verdad ------------------
    t0 = time.perf_counter()
    primera = tutor.responder(PREGUNTAS[0][1])
    frio = time.perf_counter() - t0
    print("\nEtapas de la carga:")
    for e in tutor.estado.etapas:
        print(f"  {'ok   ' if e['ok'] else 'FALLA'} {e['segundos']:>7.2f}s  "
              f"{e['etapa']}")
    if not primera["ok"]:
        print(f"\nLa primera pregunta FALLÓ: {primera['respuesta'][:300]}")
        print("No hay latencia que medir. Arregla la carga antes de volver aquí.")
        return 2
    print(f"\nArranque en frío (incluye la carga): {frio:.1f} s  · n=1")

    # --- consultas calientes ---------------------------------------------
    calientes, fallidas, por_bloque = [], [], {}
    for i in range(args.repeticiones):
        bloque, pregunta = PREGUNTAS[i % len(PREGUNTAS)]
        t0 = time.perf_counter()
        r = tutor.responder(pregunta)
        dt = time.perf_counter() - t0
        if not r["ok"]:
            fallidas.append({"pregunta": pregunta, "segundos": round(dt, 2),
                             "motivo": r["respuesta"][:160]})
            continue
        calientes.append(dt)
        por_bloque.setdefault(bloque, []).append(dt)
        print(f"  [{i + 1}/{args.repeticiones}] {bloque:12s} {dt:6.2f}s  "
              f"{r['respuesta'][:60]!r}")

    resumen = {
        "configuracion": tutor.estado.config_nombre,
        "estado": tutor.estado.resumen(),
        "entorno": compat_torch.versiones(),
        "arranque_frio_s": round(frio, 2),
        "carga_por_etapas": tutor.estado.etapas,
        "calientes": percentiles(calientes),
        "por_bloque": {b: percentiles(v) for b, v in por_bloque.items()},
        "fallidas": fallidas,
        "aviso": ("el p95 sobre pocas muestras es prácticamente el máximo: cota "
                  "superior ruidosa, no un compromiso de servicio. Las peticiones "
                  "fallidas no entran en la estadística"),
    }

    print("\n" + "=" * 62)
    print(f"Arranque en frío : {resumen['arranque_frio_s']} s (n=1)")
    c = resumen["calientes"]
    if c:
        print(f"Calientes        : p50 {c['p50_s']} s · p95 {c['p95_s']} s "
              f"· n={c['n']} (min {c['min_s']} / max {c['max_s']})")
    for b, v in resumen["por_bloque"].items():
        print(f"  {b:12s} p50 {v['p50_s']} s · n={v['n']}")
    if fallidas:
        print(f"Fallidas         : {len(fallidas)} · NO cuentan como latencia")
        for f in fallidas[:3]:
            print(f"  {f['segundos']}s · {f['motivo']}")
    print(resumen["aviso"])

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"-> {SALIDA.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
