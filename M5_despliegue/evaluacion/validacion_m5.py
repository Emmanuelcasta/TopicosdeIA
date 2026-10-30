"""La validación de M5, separada en cinco experimentos que no se mezclan.

Por qué separados. Un único porcentaje del tutor oculta de qué falló. M4 ya lo
dijo al cerrar: hay que medir por separado la lectura, la conclusión, el formato
y la explicación, porque una respuesta puede transcribir bien la foto y
diagnosticar mal, o explicar de forma convincente señalando el paso equivocado.
Aquí se extiende a las cinco capas del sistema desplegado.

    E1  texto · baseline            ¿cuántos casos acierta S08-D?
    E2  recuperación y enrutamiento ¿busca cuando debe y trae lo que debe?
    E3  herramientas matemáticas    ¿elige la herramienta y usa su resultado?
    E4  lectura de fotos            ¿transcribe bien, y detecta lo ilegible?
    E5  flujo completo de foto      ¿localiza el paso y lo explica sin reescribir?

**Lo que este archivo NO hace, y es deliberado.**
`verificar_retrieval.py` comprueba que los pesos del enrutador y las 21 rutas
coinciden con los del notebook. Eso es una **comprobación de reproducción**: dice
que la pieza desplegada es la medida, y **no** dice nada sobre cuántos casos
acierta el tutor. Mezclar las dos cosas —«el enrutador reproduce, luego el tutor
acierta 18/21»— es el error que esta separación existe para impedir. E2 incluye
esa comprobación como requisito previo, nunca como resultado.

Qué se registra en cada experimento que lo permita:

    exactitud por caso          id, bloque, acierto, abstuvo, alucino, motivo
    selección de herramienta    qué herramienta llamó y si era la esperada
    verificación correcta       si el verificador acertó el paso, y sus falsos positivos
    abstenciones apropiadas     abstuvo en adversarial (bien) o en un caso con respuesta (mal)
    errores                     excepciones, formato inválido, tope de pasos
    latencia fría y caliente    la primera petición del proceso, aparte de las demás
    p50 y p95                   con el n a la vista, siempre

    python M5_despliegue/evaluacion/validacion_m5.py                # estado
    python M5_despliegue/evaluacion/validacion_m5.py --correr E2    # lo que se pueda
    python M5_despliegue/evaluacion/validacion_m5.py --informe      # a markdown
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "M5_despliegue" / "app"), str(RAIZ / "scripts"),
                str(RAIZ / "scripts" / "agente")]

SALIDA = RAIZ / "M5_despliegue" / "resultados"
INFORME = RAIZ / "M5_despliegue" / "docs" / "VALIDACION_M5.md"

EVAL_M2 = RAIZ / "data" / "eval_set_m2.jsonl"
EVAL_M5_VISION = RAIZ / "data" / "eval_set_m5_vision.jsonl"
IMAGENES_M5 = RAIZ / "m4" / "eval_set" / "imagenes_m5"
HERRAMIENTAS_ESPERADAS = RAIZ / "data" / "herramientas_esperadas_m2.jsonl"


# --------------------------------------------------------------------------
# Los cinco experimentos, declarados
# --------------------------------------------------------------------------

EXPERIMENTOS = [
    {
        "id": "E1",
        "nombre": "Texto · baseline S08-D y sus controles",
        "mide": "exactitud por caso, alucinaciones, abstenciones, latencia",
        "no_mide": ("nada visual, y nada sobre herramientas: S08-D no las monta"),
        "conjunto": "data/eval_set_m2.jsonl · 21 casos, sin modificar",
        "como": "python M5_despliegue/evaluacion/ablaciones.py --solo-controles",
        "necesita": ["GPU", "modelo base", "adaptador qwen-lora"],
        "criterio": ("los tres controles reproducen su cifra: s08d 18/21, "
                     "solo_lora 14/21, s08c 13/21. Si no, el entorno no es el "
                     "que midió y nada más se interpreta"),
        "registra": ["exactitud por caso", "alucinaciones", "abstenciones",
                     "latencia p50/p95", "errores de formato"],
    },
    {
        "id": "E2",
        "nombre": "Recuperación y enrutamiento",
        "mide": ("si enruta cada caso como el notebook, y si los pasajes que "
                 "trae contienen la clave de oro"),
        "no_mide": ("la exactitud del tutor. Un enrutador perfecto con un "
                    "generador malo da 0 aciertos, y al contrario"),
        "conjunto": ("33 consultas de desarrollo + los 21 casos de M2 para las "
                     "rutas + las 4 claves de oro de conocimiento"),
        "como": ("python M5_despliegue/evaluacion/verificar_retrieval.py "
                 "--guardar   (requisito previo, no resultado)"),
        "necesita": ["sentence-transformers", "rank-bm25", "scikit-learn",
                     "los dos modelos de retrieval en caché"],
        "criterio": ("pesos, leave-one-out y las 21 rutas idénticas a "
                     "m3_rag_avanzado.json. Recall de la clave de oro en los 4 "
                     "casos de conocimiento ≥ 3/4, como en S08"),
        "registra": ["ruta por caso", "pesos de la regresión", "leave-one-out",
                     "recall de la clave de oro", "latencia del retrieval"],
    },
    {
        "id": "E3",
        "nombre": "Herramientas matemáticas",
        "mide": ("si el modelo ELIGE la herramienta correcta al leer la "
                 "pregunta, y si su respuesta final usa el valor devuelto"),
        "no_mide": ("si las herramientas calculan bien: eso está en "
                    "scripts/tests/test_calculo.py, son deterministas y se "
                    "comprueban sin modelo"),
        "conjunto": ("data/eval_set_m2.jsonl con "
                     "data/herramientas_esperadas_m2.jsonl, más preguntas de "
                     "cálculo en español que todavía NO existen en ningún "
                     "eval set"),
        "como": ("python M5_despliegue/evaluacion/ablaciones.py --filas 3,4"),
        "necesita": ["GPU", "casos de cálculo en un eval set (FALTAN)"],
        "criterio": ("selección correcta ≥ la de C5-FT (0,585 en "
                     "scorecard_agente_v2.csv, fila `Selección correcta`) y "
                     "fidelidad a la herramienta ≥ 0,84 (misma fuente)"),
        "registra": ["herramienta llamada por caso", "selección correcta",
                     "fidelidad a la herramienta", "llamadas con error",
                     "innecesarias en adversarial"],
    },
    {
        "id": "E4",
        "nombre": "Lectura de fotos y salvaguarda de ilegibles",
        "mide": ("si la transcripción coincide con la de referencia, y si las "
                 "borrosas se marcan no fiables sin rechazar las nítidas"),
        "no_mide": ("el diagnóstico. Una lectura correcta no implica un "
                    "diagnóstico correcto, y en M4 fue justo así"),
        "conjunto": ("data/eval_set_m5_vision.jsonl · 12 hojas NUEVAS, 11 pares "
                     "legible/borrosa + 1 solo borrosa"),
        "como": "pendiente de notebook; el eval set ya está generado",
        "necesita": ["GPU", "las 23 fotografías de m4/eval_set/HOJAS_M5.md"],
        "criterio": ("≥ 9/11 nítidas transcritas sin error de número; "
                     "≥ 10/12 borrosas marcadas no fiables; ≤ 1/11 nítidas "
                     "rechazadas por error"),
        "registra": ["transcripción por caso frente a la de referencia",
                     "no_fiable por caso y su motivo", "falsos positivos",
                     "latencia de lectura"],
    },
    {
        "id": "E5",
        "nombre": "Flujo completo de revisión de foto",
        "mide": ("si con la lectura confirmada localiza el paso, si el "
                 "verificador acierta, y si explica sin reescribir el "
                 "procedimiento del estudiante"),
        "no_mide": ("nada que se pueda comparar con M4: configuración distinta "
                    "y hojas distintas. No es una mejora de 2/12"),
        "conjunto": "el mismo de E4, con la transcripción de referencia confirmada",
        "como": "pendiente de notebook; TUTOR_CONFIG=revision_foto",
        "necesita": ["GPU", "las 23 fotografías", "E4 aprobado"],
        "criterio": ("0 errores inventados en los 3 casos `sin_error` —es el "
                     "criterio que manda—; el paso señalado correcto en los 4 de "
                     "localización y los 2 compensados; abstención en los 3 "
                     "adversariales; 0 falsos positivos del verificador"),
        "registra": ["paso señalado por caso", "hallazgo del verificador",
                     "falso error", "abstenciones apropiadas",
                     "si la respuesta reescribió los números del estudiante",
                     "latencia fría y caliente del flujo de dos pasos"],
    },
    {
        "id": "E6",
        "nombre": "Muestra de la app desplegada",
        "mide": ("el harness sobre interacciones reales, con el protocolo de "
                 "etiquetado ciego de docs/REFINAMIENTO_TUTOR_M5.md §6"),
        "no_mide": ("nada todavía. No hay URL pública, y sin ella una «muestra "
                    "real» no existe"),
        "conjunto": "50 interacciones consecutivas de una ventana con ≥ 50",
        "como": ("python M5_despliegue/evaluacion/validacion_m5.py "
                 "--muestra-url <URL>"),
        "necesita": ["una URL pública del Space", "50 interacciones reales",
                     "dos personas para el etiquetado ciego"],
        "criterio": ("se reporta n, ventana temporal, versión y commit. Sin esas "
                     "cuatro cosas el número no significa nada"),
        "registra": ["exactitud etiquetada a mano", "acuerdo entre etiquetadores "
                     "(Cohen κ)", "latencia fría y caliente reales",
                     "p50/p95 con n", "errores y caídas"],
    },
]


# --------------------------------------------------------------------------
# Qué se puede comprobar desde aquí, sin GPU y sin URL
# --------------------------------------------------------------------------

def estado_requisitos() -> dict:
    """Qué hay y qué falta. No mide nada: comprueba disponibilidad."""
    def tiene(modulo):
        import importlib.util
        return importlib.util.find_spec(modulo) is not None

    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        pass

    fotos = sorted(IMAGENES_M5.glob("*.jpg")) if IMAGENES_M5.is_dir() else []
    esperadas = []
    if EVAL_M5_VISION.exists():
        for linea in EVAL_M5_VISION.read_text(encoding="utf-8").splitlines():
            c = json.loads(linea)
            esperadas += [f for f in (c["imagen_nitida"], c["imagen_borrosa"]) if f]

    adaptadores = {n: (RAIZ / "adaptadores" / n).is_dir()
                   for n in ("qwen-lora", "qwen-lora-agente",
                             "qwen-lora-diagnostico")}
    return {
        "GPU": gpu,
        "retrieval (sentence-transformers, rank_bm25, sklearn)":
            all(tiene(m) for m in ("sentence_transformers", "rank_bm25", "sklearn")),
        "transformers + peft": tiene("transformers") and tiene("peft"),
        "adaptadores": adaptadores,
        "eval set de M2 (21 casos)": EVAL_M2.exists(),
        "herramientas esperadas de M2": HERRAMIENTAS_ESPERADAS.exists(),
        "eval set visual de M5 (12 hojas)": EVAL_M5_VISION.exists(),
        "fotografías de M5": f"{len(fotos)} de {len(esperadas)}",
        "verificación del retrieval":
            (SALIDA.parent / "evaluacion" / "verificacion_retrieval.json").exists(),
        "URL pública del Space": bool(
            __import__("os").environ.get("TUTOR_URL_PUBLICA")),
    }


def latencias(muestras) -> dict:
    """Fría aparte de las calientes. La primera petición paga la carga del
    modelo y promediarla con el resto esconde las dos cifras que importan."""
    if not muestras:
        return {}
    fria, calientes = muestras[0], muestras[1:]
    orden = sorted(calientes)
    d = {"n_total": len(muestras), "latencia_fria_s": round(fria, 3),
         "n_calientes": len(calientes)}
    if orden:
        d.update({
            "caliente_p50_s": round(statistics.median(orden), 3),
            "caliente_p95_s": round(orden[max(0, int(0.95 * len(orden)) - 1)], 3),
            "caliente_min_s": round(orden[0], 3),
            "caliente_max_s": round(orden[-1], 3),
            "aviso": (f"p95 sobre {len(orden)} mediciones: es prácticamente el "
                      "máximo, una cota superior ruidosa, no un compromiso de "
                      "servicio"),
        })
    return d


def correr_e2() -> dict:
    """E2 sin GPU: la comprobación de reproducción del enrutador.

    Se delega en `verificar_retrieval.py` en vez de repetir su lógica, y se
    devuelve marcado como lo que es: un requisito previo, no una exactitud.
    """
    guion = RAIZ / "M5_despliegue" / "evaluacion" / "verificar_retrieval.py"
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, str(guion), "--guardar"],
                       capture_output=True, text=True, cwd=RAIZ, check=False)
    return {
        "experimento": "E2 (parcial)",
        "que_es": ("comprobación de REPRODUCCIÓN del enrutador y el retrieval. "
                   "NO es una medición de exactitud del tutor"),
        "codigo_salida": p.returncode,
        "reproduce": p.returncode == 0,
        "segundos": round(time.perf_counter() - t0, 1),
        "salida": p.stdout.strip().splitlines()[-6:],
        "pendiente": ("el recall de la clave de oro y la exactitud del tutor "
                      "necesitan GPU"),
    }


def informe() -> str:
    req = estado_requisitos()
    L = ["# Validación de M5 · cinco experimentos que no se mezclan", "",
         "Generado por `M5_despliegue/evaluacion/validacion_m5.py`.", "",
         "> **Una comprobación de reproducción no es una medición de exactitud.** "
         "Que el enrutador desplegado tenga los mismos pesos y las mismas 21 "
         "rutas que el del notebook dice que la pieza servida es la medida. No "
         "dice cuántos casos acierta el tutor hoy. Las dos cosas van en "
         "experimentos distintos, y E2 incluye la primera como requisito previo.",
         "",
         "## Estado de los requisitos", "", "| Requisito | Estado |", "|---|---|"]
    for k, v in req.items():
        if isinstance(v, dict):
            v = ", ".join(f"{n}: {'sí' if ok else 'NO'}" for n, ok in v.items())
        elif isinstance(v, bool):
            v = "sí" if v else "**NO**"
        L.append(f"| {k} | {v} |")

    L += ["", "## Los experimentos", ""]
    for e in EXPERIMENTOS:
        L += [f"### {e['id']} · {e['nombre']}", "",
              f"**Mide:** {e['mide']}", "",
              f"**NO mide:** {e['no_mide']}", "",
              f"**Conjunto:** {e['conjunto']}", "",
              f"**Cómo:** `{e['como']}`", "",
              "**Necesita:** " + ", ".join(e["necesita"]), "",
              f"**Criterio de aprobación:** {e['criterio']}", "",
              "**Registra:** " + ", ".join(e["registra"]), "", "---", ""]

    L += ["## Lo que falta ejecutar", "",
          "| Experimento | Bloqueado por |", "|---|---|"]
    bloqueos = {
        "E1": "GPU",
        "E2": "parcial: la reproducción se puede correr sin GPU; el recall y la "
              "exactitud, no",
        "E3": "GPU **y** casos de cálculo en un eval set, que no existen",
        "E4": "GPU y las 23 fotografías de `m4/eval_set/HOJAS_M5.md`",
        "E5": "lo de E4, más que E4 esté aprobado",
        "E6": "una URL pública del Space y 50 interacciones reales",
    }
    for i, b in bloqueos.items():
        L.append(f"| {i} | {b} |")
    L += ["", "Ninguno de estos resultados se simula. Mientras no se ejecuten, de "
          "M5 solo se puede decir que el sistema está montado y que la pieza de "
          "retrieval servida es la que se midió en S08."]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--correr", help="id del experimento, por ejemplo E2")
    ap.add_argument("--informe", action="store_true",
                    help="escribe docs/VALIDACION_M5.md")
    ap.add_argument("--muestra-url",
                    help="URL pública del Space para E6; sin ella E6 no existe")
    args = ap.parse_args()

    if args.muestra_url:
        print("E6 necesita, además de la URL, 50 interacciones reales y dos "
              "personas para el etiquetado ciego (protocolo en "
              "docs/REFINAMIENTO_TUTOR_M5.md §6). No se simula: cuando el Space "
              "exista, se recogen las interacciones y se etiquetan a mano.")
        return 1

    print("Requisitos:")
    for k, v in estado_requisitos().items():
        if isinstance(v, dict):
            v = ", ".join(f"{n}: {'sí' if ok else 'NO'}" for n, ok in v.items())
        print(f"  {k:52s} {v}")

    if args.correr:
        if args.correr.upper() != "E2":
            print(f"\n{args.correr.upper()} necesita GPU o fotografías: no se "
                  "puede correr aquí. Lo que falta está en --informe.")
            return 1
        print("\nE2 (parcial) · comprobación de reproducción:")
        r = correr_e2()
        for linea in r["salida"]:
            print("  " + linea)
        print(f"\n  reproduce: {r['reproduce']}  ·  {r['segundos']}s")
        print(f"  {r['que_es']}")
        SALIDA.mkdir(parents=True, exist_ok=True)
        (SALIDA / "validacion_e2.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  -> {(SALIDA / 'validacion_e2.json').relative_to(RAIZ)}")
        return 0 if r["reproduce"] else 2

    if args.informe:
        INFORME.parent.mkdir(parents=True, exist_ok=True)
        INFORME.write_text(informe() + "\n", encoding="utf-8")
        print(f"\n-> {INFORME.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
