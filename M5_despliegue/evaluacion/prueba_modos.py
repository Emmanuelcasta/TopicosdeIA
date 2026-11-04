"""Prueba manual de los dos modos de foto, de punta a punta, sin interfaz.

NO es el scorecard. El scorecard es `evaluar_vision_m5.py` sobre las 12 hojas
del eval set; esto es una sonda exploratoria para ver qué pasa con UNA foto:
las dos lecturas crudas, en qué difieren si difieren, el propósito que sugiere,
y la resolución o la revisión que sale si el estudiante confirma lo leído tal
cual. Escribe en `resultados/pruebas_manuales.jsonl`, aparte del scorecard.

    python M5_despliegue/evaluacion/prueba_modos.py FOTO [--recorte 0,0.42]
           [--png] [--modo auto|resolver|revisar] [--etiqueta texto]

`--recorte a,b` corta la franja vertical [a, b] de la foto (fracciones del
alto) ANTES de leerla. Sirve para obtener una hoja con solo el enunciado a
partir de una foto real, sin dibujar ni inventar nada: la franja es la misma
tinta, solo que sin los pasos debajo. `--png` guarda ese recorte como PNG y
lee el PNG, para comprobar que el formato no cambia el resultado.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "M5_despliegue" / "app"), str(RAIZ / "scripts"),
                str(RAIZ / "scripts" / "agente")]
SALIDA = RAIZ / "M5_despliegue" / "resultados" / "pruebas_manuales.jsonl"


def preparar(ruta: Path, recorte: str | None, png: bool):
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
    origen = {"archivo": ruta.name, "formato": Image.open(ruta).format,
              "px": list(im.size)}
    if recorte:
        a, b = (float(x) for x in recorte.split(","))
        im = im.crop((0, int(im.height * a), im.width, int(im.height * b)))
        origen["recorte"] = [a, b]
    if png:
        destino = Path("/tmp") / (ruta.stem + "-prueba.png")
        im.save(destino, format="PNG")
        im = Image.open(destino)
        origen["leido_como"] = "PNG"
    else:
        origen["leido_como"] = origen["formato"]
    return im, origen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("foto")
    ap.add_argument("--recorte")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--modo", default="auto", choices=["auto", "resolver", "revisar"])
    ap.add_argument("--etiqueta", default="")
    a = ap.parse_args()

    import pipeline

    tutor = pipeline.Tutor(os.environ.get("TUTOR_CONFIG", "revision_foto"))
    carga = tutor.precargar_vision()
    im, origen = preparar(Path(a.foto), a.recorte, a.png)

    t0 = time.perf_counter()
    lec = tutor.leer_imagen(im)
    s_lectura = round(time.perf_counter() - t0, 2)
    print(f"=== {origen}")
    print(f"lectura {s_lectura}s · ok={lec.get('ok')} · confirmable={lec.get('confirmable')}"
          f" · coinciden={lec.get('lecturas_coinciden')} · propósito={lec.get('proposito')}")
    if lec.get("diferencias"):
        print(f"  difieren en: {lec['diferencias']}")
    print(f"  enunciado: {lec.get('enunciado')!r}")
    print(f"  pasos    : {lec.get('pasos')}")

    fila = {"etiqueta": a.etiqueta, "origen": origen, "carga_vlm": carga,
            "lectura_s": s_lectura, "ok": lec.get("ok"),
            "confirmable": lec.get("confirmable"),
            "coinciden": lec.get("lecturas_coinciden"),
            "diferencias": lec.get("diferencias"),
            "proposito_sugerido": lec.get("proposito"),
            "enunciado": lec.get("enunciado"), "pasos": lec.get("pasos"),
            "motivo_no_fiable": lec.get("motivo_no_fiable")}

    if lec.get("confirmable"):
        modo = a.modo if a.modo != "auto" else lec.get("proposito") or "revisar"
        fila["modo"] = modo
        # El estudiante confirma la transcripción TAL CUAL: si estaba mal, lo
        # que sale está mal, y eso es lo que hay que ver.
        t1 = time.perf_counter()
        if modo == "resolver":
            r = tutor.resolver_enunciado(lec.get("enunciado") or "", confirmado=True)
        else:
            r = tutor.revisar_pasos(lec.get("pasos") or [], tutor.PREGUNTA_REVISION,
                                    enunciado=lec.get("enunciado"), confirmado=True)
        fila.update({"respuesta_s": round(time.perf_counter() - t1, 2),
                     "respuesta": r.get("respuesta"), "camino": r.get("camino"),
                     "herramientas": r.get("herramientas"),
                     "verificador": r.get("verificador_aritmetico"),
                     "pasos_llm": r.get("pasos_llm"), "ok_respuesta": r.get("ok")})
        print(f"--- {modo} · {fila['respuesta_s']}s · camino={r.get('camino')} "
              f"· herramientas={r.get('herramientas')}")
        print(f"  verificador: {r.get('verificador_aritmetico')}")
        print("  " + (r.get("respuesta") or "").replace("\n", "\n  "))
    else:
        print(f"--- se detiene: {lec.get('motivo_no_fiable') or lec.get('respuesta')}")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fila, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
