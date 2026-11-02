"""Por qué el retrieval híbrido no se monta, en cuatro niveles separados.

**El error que esto corrige.** En Kaggle, la validación informó que
`sentence-transformers`, `rank_bm25` y `sklearn` estaban instalados, y de ahí se
concluyó que el retrieval estaba listo. No lo estaba: el tutor respondió `7/8`
correctamente con la pestaña Estado diciendo `degradado: True` y
`faltantes: retrieval híbrido + reranker`. Un `import` que funciona no prueba
nada sobre la cadena de S08.

Los cuatro niveles, que son cuatro cosas distintas y fallan por motivos
distintos:

    1 · paquete instalado      `importlib` lo encuentra y la versión cuadra
    2 · modelo cargable        los pesos están en caché o se pueden descargar
    3 · componente montado     `buscador_hibrido` devuelve un buscador con
                               `encode` y `rerank_max`
    4 · componente efectivo    una consulta real devuelve chunks y el enrutador
                               decide buscar o no buscar

Para el nivel 1 basta con tener el paquete; para el 2 hace falta red o caché;
para el 3, que las versiones encajen entre sí; para el 4, que la cadena
completa produzca resultados. Se para en el primero que falle y se imprime su
traceback entero: adivinar cuál de los cuatro era es justo lo que costó la
sesión anterior.

No necesita GPU ni modelo de lenguaje: no carga Qwen. Sí descarga los dos
modelos de la cadena de S08 si no están en caché (unos 500 MB), que es
precisamente lo que hay que comprobar.

    python M5_despliegue/evaluacion/diagnostico_retrieval.py
    python M5_despliegue/evaluacion/diagnostico_retrieval.py --solo-entorno
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "M5_despliegue" / "app"), str(RAIZ / "scripts"),
                str(RAIZ / "scripts" / "agente")]

SALIDA = RAIZ / "M5_despliegue" / "resultados" / "diagnostico_retrieval.json"

# Lo que de verdad tiene que encajar junto. `transformers`, `tokenizers` y
# `huggingface_hub` se rompen entre sí con una facilidad que no tiene el resto:
# en Kaggle, el hub 1.29.0 preinstalado con transformers 4.45.2 impedía hasta
# el `import`. Los rangos son los que pasaron una importación limpia allí.
ESPERADO = {
    "transformers": "4.45.2",
    "tokenizers": ">=0.20,<0.21",
    "huggingface_hub": "0.26.5",
    "sentence_transformers": "3.x (requiere transformers>=4.41)",
    "rank_bm25": "0.2.2",
    "sklearn": "1.5.x",
    "numpy": "1.26.x",
    "peft": ">=0.13.2,<0.21",
    "torch": "el de Kaggle; NO reinstalar",
}

# Presente = el adaptador no carga. Reaparece tras cada reinicio del runtime
# porque viene en la imagen de Kaggle, así que se comprueba, no se supone.
PROHIBIDO = ("torchao",)


def _version(modulo: str) -> str | None:
    """La versión instalada, sin importar el paquete si se puede evitar."""
    import importlib.metadata as md
    for nombre in (modulo, modulo.replace("_", "-"),
                   {"sklearn": "scikit-learn"}.get(modulo, modulo)):
        try:
            return md.version(nombre)
        except Exception:      # noqa: BLE001, PERF203
            continue
    try:
        return getattr(importlib.import_module(modulo), "__version__", "?")
    except Exception:      # noqa: BLE001
        return None


def entorno() -> dict:
    """Versiones, flags de red y paquetes prohibidos. No monta nada."""
    datos = {"python": sys.version.split()[0], "paquetes": {}, "esperado": ESPERADO}
    for m in ESPERADO:
        datos["paquetes"][m] = _version(m) or "NO INSTALADO"
    datos["prohibidos_presentes"] = [
        p for p in PROHIBIDO if importlib.util.find_spec(p) is not None]
    # Si el hub está en modo offline y los modelos no están en caché, el nivel 2
    # falla por configuración y no por red. Verlo aquí ahorra buscarlo abajo.
    datos["flags_hub"] = {
        v: os.environ.get(v) for v in
        ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "HF_HUB_CACHE",
         "USE_TF", "HF_DATASETS_OFFLINE")
        if os.environ.get(v) is not None}
    try:
        import torch
        datos["torch"] = {"version": torch.__version__,
                          "cuda": torch.cuda.is_available(),
                          "dispositivo": (torch.cuda.get_device_name(0)
                                          if torch.cuda.is_available() else "CPU")}
    except Exception as e:      # noqa: BLE001
        datos["torch"] = {"error": f"{type(e).__name__}: {e}"}
    return datos


def _nivel(n: int, titulo: str, funcion) -> dict:
    """Corre un nivel y lo registra con su traceback completo si falla."""
    print(f"\n--- nivel {n} · {titulo} ---", flush=True)
    t0 = time.perf_counter()
    try:
        detalle = funcion()
    except BaseException as e:      # noqa: BLE001 — se informa, no se oculta
        traza = traceback.format_exc()
        print(traza, flush=True)
        return {"nivel": n, "titulo": titulo, "ok": False,
                "segundos": round(time.perf_counter() - t0, 2),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traza}
    ok = detalle.get("ok", True)
    for linea in detalle.get("lineas", []):
        print(f"  {linea}", flush=True)
    return {"nivel": n, "titulo": titulo, "ok": ok,
            "segundos": round(time.perf_counter() - t0, 2), **detalle}


# --------------------------------------------------------------------------
# Los cuatro niveles
# --------------------------------------------------------------------------

def nivel_1_paquetes() -> dict:
    """Instalado e importable. Es lo ÚNICO que comprobaba la validación vieja."""
    lineas, faltan = [], []
    for m in ("sentence_transformers", "rank_bm25", "sklearn", "numpy"):
        v = _version(m)
        try:
            importlib.import_module(m)
            estado = f"importa · {v}"
        except Exception as e:      # noqa: BLE001
            estado = f"NO IMPORTA ({type(e).__name__}: {e})"
            faltan.append(m)
        lineas.append(f"{m:24s} {estado}")
    lineas.append("ojo: esto NO dice que la cadena de S08 funcione, "
                  "solo que los paquetes están")
    d = {"ok": not faltan, "faltan": faltan, "lineas": lineas}
    if faltan:
        d["error"] = "no importan: " + ", ".join(faltan)
    return d


def nivel_2_modelos() -> dict:
    """Los dos modelos de la cadena: en caché o descargables.

    Es el nivel que la red y `HF_HUB_OFFLINE` deciden, y el único que tarda:
    sin caché son ~500 MB.
    """
    from pipeline import MODELO_EMBEDDINGS, MODELO_RERANKER
    from sentence_transformers import CrossEncoder, SentenceTransformer

    lineas = []
    t0 = time.perf_counter()
    st = SentenceTransformer(MODELO_EMBEDDINGS)
    lineas.append(f"embeddings {MODELO_EMBEDDINGS} · "
                  f"{time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    reranker = CrossEncoder(MODELO_RERANKER, max_length=512)
    lineas.append(f"reranker   {MODELO_RERANKER} · "
                  f"{time.perf_counter() - t0:.1f}s")
    v = st.encode(["prueba"], normalize_embeddings=True)
    p = reranker.predict([("¿cuánto cuesta?", "El precio es 10 pesos.")],
                         show_progress_bar=False)
    lineas.append(f"encode -> vector de {len(v[0])} dimensiones · "
                  f"predict -> {float(p[0]):.3f}")
    return {"ok": True, "lineas": lineas,
            "dimensiones": int(len(v[0])), "rerank_prueba": float(p[0])}


def nivel_3_montaje() -> dict:
    """`buscador_hibrido` con el corpus real, y su diagnóstico por etapas."""
    from pipeline import buscador_hibrido, cargar_corpus

    chunks, _metadatos, _ids = cargar_corpus()
    diag: list = []
    buscar = buscador_hibrido(chunks, diag)
    lineas = [f"{'ok   ' if x['ok'] else 'FALLA'} {x['segundos']:>6.2f}s  "
              f"{x['etapa']}" + ("" if x["ok"] else f"\n        -> {x['error']}")
              for x in diag]
    if buscar is None:
        fallo = next((x for x in diag if not x["ok"]), None)
        if fallo:
            lineas.append(f"causa probable: {fallo['causa_probable']}")
            print(fallo.get("traceback", ""), flush=True)
        return {"ok": False, "lineas": lineas, "etapas": diag,
                "error": fallo["error"] if fallo else "sin etapas registradas"}
    for attr in ("encode", "rerank_max"):
        if getattr(buscar, attr, None) is None:
            return {"ok": False, "lineas": lineas,
                    "error": f"el buscador no expone {attr}: el enrutador de "
                             "S08-D no se puede montar sin él"}
    lineas.append(f"chunks {len(chunks)} · encode y rerank_max expuestos")
    return {"ok": True, "lineas": lineas, "etapas": diag, "chunks": len(chunks)}


def nivel_4_efectivo(config: str) -> dict:
    """La cadena entera decidiendo, sin modelo de lenguaje.

    `Tutor.resolver_componentes()` monta corpus, retrieval, enrutador y
    herramientas sin cargar Qwen, que es para lo que se separó de
    `_armar_agente`. Lo que se comprueba aquí es lo que la pestaña Estado tiene
    que acabar diciendo: `degradado: False` y ningún componente faltante.
    """
    from pipeline import Tutor

    tutor = Tutor(config)
    piezas = tutor.resolver_componentes()
    e = tutor.estado
    lineas = [f"configuración solicitada: {e.config_nombre}",
              f"configuración efectiva  : {e.efectiva()}",
              f"retrieval activo        : {e.retrieval}",
              f"enrutador activo        : {e.enrutador}",
              f"degradado               : {e.degradado}",
              f"faltantes               : {', '.join(e.faltantes) or 'ninguno'}"]
    for m in e.motivos:
        lineas.append(f"aviso: {m}")

    # Las dos vías del enrutador, que es lo que distingue S08-D de S08-C.
    rutas = {}
    contexto = piezas.get("contexto_fijo")
    if contexto is not None:
        for etiqueta, pregunta in (
                ("conocimiento", "¿Cuál es la nota mínima aprobatoria del colegio?"),
                ("calculo", "¿Cuánto es 3/4 + 1/8?")):
            ctx = contexto(pregunta)
            rutas[etiqueta] = {"pregunta": pregunta, "chunks": len(ctx or [])}
            lineas.append(f"enrutador · {etiqueta:12s} -> "
                          f"{len(ctx or [])} chunks recuperados")
        # Criterio observable, no una corazonada: la factual tiene que traer
        # contexto y la aritmética no debería necesitarlo.
        bien = (rutas["conocimiento"]["chunks"] > 0
                and rutas["calculo"]["chunks"] == 0)
        lineas.append("decisión del enrutador: " + (
            "coherente (busca en la factual, no en la aritmética)" if bien else
            "NO coherente: revisar antes de presentar el enrutador como activo"))
        rutas["coherente"] = bien

    return {"ok": not e.degradado, "lineas": lineas, "rutas": rutas,
            "error": ("degradado, faltan: " + ", ".join(e.faltantes)
                      if e.degradado else ""),
            "degradado": e.degradado, "faltantes": e.faltantes,
            "configuracion_efectiva": e.efectiva(), "avisos": e.motivos}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.environ.get("TUTOR_CONFIG", "s08d"),
                    help="configuración a montar en el nivel 4")
    ap.add_argument("--solo-entorno", action="store_true",
                    help="versiones y flags, sin montar nada")
    args = ap.parse_args()

    env = entorno()
    print("=" * 68)
    print(f"Entorno · Python {env['python']}")
    for m, v in env["paquetes"].items():
        print(f"  {m:24s} {v:20s} esperado: {ESPERADO[m]}")
    if env["flags_hub"]:
        print("  flags del hub: " + ", ".join(
            f"{k}={v}" for k, v in env["flags_hub"].items()))
    print(f"  torch: {env['torch']}")
    if env["prohibidos_presentes"]:
        print("  AVISO · paquetes que rompen la carga del adaptador y siguen "
              f"instalados: {', '.join(env['prohibidos_presentes'])}")
        print("          pip uninstall -y " + " ".join(env["prohibidos_presentes"]))
    print("=" * 68)

    if args.solo_entorno:
        return 0

    niveles = [
        _nivel(1, "paquete instalado", nivel_1_paquetes),
        _nivel(2, "modelo descargado o en caché", nivel_2_modelos),
        _nivel(3, "componente montado", nivel_3_montaje),
        _nivel(4, f"componente efectivo ({args.config})",
               lambda: nivel_4_efectivo(args.config)),
    ]
    # Se informan todos los que se corrieron, pero el primero que falla es el
    # que hay que arreglar: los de abajo fallarán por arrastre.
    primero = next((n for n in niveles if not n["ok"]), None)

    print("\n" + "=" * 68)
    for n in niveles:
        print(f"nivel {n['nivel']} · {'OK   ' if n['ok'] else 'FALLA'} · "
              f"{n['segundos']:>6.2f}s · {n['titulo']}")
    if primero is None:
        print("Los cuatro niveles pasan: la cadena de S08 está montada y "
              "responde. `degradado` debe ser False en la pestaña Estado.")
    else:
        print(f"\nPrimer nivel que falla: {primero['nivel']} "
              f"({primero['titulo']}).")
        print(f"  {primero.get('error', 'sin error registrado')}")
        print("Los niveles posteriores fallan por arrastre; arregla este "
              "primero. NO presentes la configuración como lista.")

    resumen = {"entorno": env, "config": args.config, "niveles": niveles,
               "primer_fallo": primero["nivel"] if primero else None,
               "que_no_prueba": ("ninguno de los cuatro niveles mide exactitud "
                                 "del tutor: comprueban montaje, no resultados")}
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"-> {SALIDA.relative_to(RAIZ)}")
    return 0 if primero is None else primero["nivel"]


if __name__ == "__main__":
    raise SystemExit(main())
