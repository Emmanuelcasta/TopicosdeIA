"""Prueba de humo de la vía visual. NO es el eval set de visión de M5.

Qué comprueba: que un JPG del dataset de Kaggle se abre, que el VLM se carga en
el dispositivo correcto, que `leer_imagen` devuelve JSON estructurado y que las
salvaguardas (acuerdo entre dos lecturas, `lectura_no_fiable`) se pronuncian.

Qué NO comprueba, y no se puede deducir de su salida: exactitud de lectura,
localización del paso con error, ni la salvaguarda de ilegibles. Las imágenes
son fotos de texto tipografiado en pantalla con el identificador del caso a la
vista, no hojas manuscritas, y `caso-10.jpg` se lee bien aunque `m5v-10` exija
lo contrario. Por eso este script no calcula ninguna tasa de acierto: si
imprimiera un «9/12» alguien lo copiaría al informe.

El eval set oficial sigue siendo `data/eval_set_m5_vision.jsonl` con sus 23
tomas de nombre exacto en `m4/eval_set/imagenes_m5/`. Este paquete no las
sustituye y sus nombres no se tocan para que parezcan esas.

Uso:
    python M5_despliegue/evaluacion/humo_vision.py                 # caso-01,07,10
    HUMO_CASOS=caso-03,caso-11 python .../humo_vision.py
    HUMO_DIR=/kaggle/input/eval-set-m5-humo python .../humo_vision.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "M5_despliegue" / "app"), str(RAIZ / "scripts"),
                str(RAIZ / "scripts" / "agente")]

PAQUETE_LOCAL = RAIZ / "M5_despliegue" / "evalset-m5" / "kaggle_dataset_humo"
SALIDA = RAIZ / "M5_despliegue" / "resultados" / "humo_vision.json"
# Los tres que pidió la profesora para el humo del día antes, en su equivalente
# de este paquete: uno de localización, uno sin error y el adversarial.
CASOS_POR_DEFECTO = "caso-01,caso-07,caso-10"
LADO_MAYOR = 1024


def localizar_paquete() -> tuple[Path, Path | None]:
    """-> (directorio de imágenes, manifiesto) sin adivinar.

    Prioridad: `HUMO_DIR`, luego cualquier dataset montado en /kaggle/input que
    contenga `caso-01.jpg`, luego la copia local del repositorio.
    """
    candidatos = []
    if os.environ.get("HUMO_DIR"):
        candidatos.append(Path(os.environ["HUMO_DIR"]))
    entrada = Path("/kaggle/input")
    if entrada.is_dir():
        candidatos += sorted(p.parent for p in entrada.glob("*/**/caso-01.jpg"))
    candidatos.append(PAQUETE_LOCAL)

    for base in candidatos:
        for imgs in (base, base / "imagenes"):
            if (imgs / "caso-01.jpg").is_file():
                # El manifiesto puede estar junto a las imágenes o un nivel
                # arriba, según cómo Kaggle haya desempaquetado el ZIP.
                for m in (imgs / "manifest.json", imgs.parent / "manifest.json",
                          PAQUETE_LOCAL / "manifest.json"):
                    if m.is_file():
                        return imgs, m
                return imgs, None
    raise SystemExit(
        "no encuentro caso-01.jpg. Adjunta el dataset al notebook o pasa "
        "HUMO_DIR=/kaggle/input/<slug>. Candidatos probados: "
        + ", ".join(str(c) for c in candidatos))


def comprobar_integridad(imgs: Path, manifiesto: Path | None) -> list:
    """Compara el SHA-256 de cada archivo con el del manifiesto del paquete.

    Sin esto, un fallo de lectura no se puede separar de un archivo que Kaggle
    recomprimió o truncó al subirlo.
    """
    if manifiesto is None:
        return [{"archivo": "(sin manifiesto)", "estado": "no comprobado"}]
    declarados = json.loads(manifiesto.read_text(encoding="utf-8"))["archivos"]
    filas = []
    for d in declarados:
        ruta = imgs / d["archivo"]
        if not ruta.is_file():
            filas.append({"archivo": d["archivo"], "estado": "AUSENTE"})
            continue
        real = hashlib.sha256(ruta.read_bytes()).hexdigest()
        filas.append({
            "archivo": d["archivo"], "caso": d["caso"],
            "estado": "ok" if real == d["sha256"] else "SHA DISTINTO",
            "sha256": real[:16],
        })
    return filas


def cargar(ruta: Path):
    """JPG -> PIL RGB, orientación corregida y lado mayor acotado.

    `exif_transpose` porque la orientación de una foto vive en los metadatos: sin
    corregirla el bloque adversarial mediría rotación en vez de legibilidad. El
    tope del lado mayor es el mismo de M4, para que la latencia por caso no
    dependa de cómo salió la foto.
    """
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
    if max(im.size) > LADO_MAYOR:
        e = LADO_MAYOR / max(im.size)
        im = im.resize((int(im.width * e), int(im.height * e)), Image.LANCZOS)
    return im


def main() -> int:
    imgs, manifiesto = localizar_paquete()
    casos = [c.strip() for c in
             os.environ.get("HUMO_CASOS", CASOS_POR_DEFECTO).split(",") if c.strip()]

    print("=" * 78)
    print("PRUEBA DE HUMO DE LA VÍA VISUAL · no es el eval set de visión de M5")
    print("=" * 78)
    print(f"imágenes:   {imgs}")
    print(f"manifiesto: {manifiesto}")
    print(f"casos:      {', '.join(casos)}")
    print()

    integridad = comprobar_integridad(imgs, manifiesto)
    malas = [f for f in integridad if f["estado"] not in ("ok", "no comprobado")]
    print(f"integridad: {len(integridad) - len(malas)}/{len(integridad)} ok")
    for f in malas:
        print(f"  {f['estado']}  {f['archivo']}")
    print()

    import pipeline

    t = pipeline.Tutor("revision_foto")
    print(f"configuración: {t.cfg.nombre}  ·  vision={t.cfg.vision}")

    filas = []
    for nombre in casos:
        ruta = imgs / f"{nombre}.jpg"
        if not ruta.is_file():
            print(f"\n{nombre}: AUSENTE en {imgs}")
            filas.append({"caso": nombre, "estado": "ausente"})
            continue
        im = cargar(ruta)
        t0 = time.perf_counter()
        r = t.leer_imagen(im)
        seg = round(time.perf_counter() - t0, 2)

        print(f"\n--- {nombre}  ({im.width}x{im.height})  {seg}s")
        print(f"  ok:                 {r.get('ok')}")
        print(f"  etapa:              {r.get('etapa')}")
        print(f"  confirmable:        {r.get('confirmable')}")
        print(f"  lecturas_coinciden: {r.get('lecturas_coinciden')}")
        print(f"  motivo_no_fiable:   {r.get('motivo_no_fiable') or '(ninguno)'}")
        print(f"  pasos leídos:       {len(r.get('pasos') or [])}")
        print(f"  enunciado:          {(r.get('enunciado') or '')[:90]}")
        for i, p in enumerate(r.get("pasos") or [], 1):
            print(f"    {i}. {str(p)[:88]}")
        if not r.get("ok"):
            print(f"  respuesta:          {r.get('respuesta')}")

        filas.append({
            "caso": nombre, "estado": "leido" if r.get("ok") else "fallo",
            "px": [im.width, im.height], "segundos": seg,
            "confirmable": r.get("confirmable"),
            "lecturas_coinciden": r.get("lecturas_coinciden"),
            "motivo_no_fiable": r.get("motivo_no_fiable"),
            "n_pasos": len(r.get("pasos") or []),
            "transcripcion": r.get("transcripcion"),
        })

    # Lo que esta salida NO autoriza a decir. Va en el JSON, no solo en pantalla:
    # el archivo es lo que alguien abrirá dentro de un mes.
    informe = {
        "tipo": "prueba de humo de la vía visual",
        "es_eval_set_de_vision": False,
        "no_mide": [
            "exactitud de lectura: las imágenes son texto tipografiado en "
            "pantalla, no hojas manuscritas",
            "localización del paso con error: no hay transcripción de referencia "
            "comparable, porque el identificador del caso está impreso en la foto",
            "la salvaguarda de ilegibles: caso-10.jpg es legible y m5v-10 exige "
            "que solo exista borrosa",
        ],
        "eval_set_oficial": "data/eval_set_m5_vision.jsonl · 23 tomas pendientes",
        "origen_imagenes": str(imgs),
        "integridad": integridad,
        "casos": filas,
    }
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"\nescrito: {SALIDA.relative_to(RAIZ)}")
    print("Ninguna tasa de acierto: este conjunto no la soporta.")
    return 0 if all(f["estado"] == "leido" for f in filas) else 1


if __name__ == "__main__":
    raise SystemExit(main())
