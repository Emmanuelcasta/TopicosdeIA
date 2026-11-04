"""Compara redacciones de la instrucción de lectura con el VLM cargado UNA vez.

Exploratorio. No toca el scorecard ni elige nada solo: imprime, por foto y por
instrucción, la salida cruda del VLM y cómo la estructura `estructurar_lineas`.
Para elegir una redacción sin ajustarla al eval set, se usa sobre fotos que NO
son del eval set de M5: las `vis-*` de M4 (ya gastadas, conjunto de
regresión) y franjas de enunciado recortadas con `ARCHIVO@a,b`.

    python M5_despliegue/evaluacion/sonda_prompts.py FOTO [FOTO@0,0.4 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "M5_despliegue" / "app"), str(RAIZ / "scripts"),
                str(RAIZ / "scripts" / "agente")]

VARIANTES = {
    "copiar_actual": None,            # vision.INSTRUCCION_COPIAR
    "independiente_actual": None,     # vision.INSTRUCCION_COPIAR_INDEPENDIENTE
    "transcribe_lineas": (
        "Transcribe línea por línea el texto manuscrito de esta hoja, en el mismo "
        "orden y tal como está escrito. No resuelvas el ejercicio ni corrijas "
        "ninguna cuenta.\n"
        'Devuelve SOLO un JSON: {"lineas": ["...", "..."]}\n'
        'Si no hay texto legible, devuelve {"ilegible": true, "motivo": "..."}.'),
    "lee_de_arriba_abajo": (
        "Lee la hoja de arriba abajo y escribe cada línea manuscrita exactamente "
        "como aparece, sin añadir ni arreglar nada.\n"
        'Devuelve SOLO un JSON: {"lineas": ["...", "..."]}\n'
        'Solo si no puedes distinguir el texto, devuelve {"ilegible": true, '
        '"motivo": "..."}.'),
}


def main() -> int:
    from PIL import Image, ImageOps

    import pipeline
    import vision

    VARIANTES["copiar_actual"] = vision.INSTRUCCION_COPIAR
    VARIANTES["independiente_actual"] = vision.INSTRUCCION_COPIAR_INDEPENDIENTE
    tutor = pipeline.Tutor("revision_foto")
    print("carga VLM:", tutor.precargar_vision())
    vlm, proc, torch = tutor._vlm

    for arg in sys.argv[1:]:
        ruta, _, franja = arg.partition("@")
        im = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
        if franja:
            a, b = (float(x) for x in franja.split(","))
            im = im.crop((0, int(im.height * a), im.width, int(im.height * b)))
        im, _ = pipeline.normalizar_foto(im)
        print(f"\n######## {Path(ruta).name} {franja or ''}")
        for nombre, instr in VARIANTES.items():
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": instr}]}]
            prompt = proc.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
            with torch.no_grad():
                ent = proc(text=[prompt], images=[im], return_tensors="pt").to(vlm.device)
                out = vlm.generate(**ent, max_new_tokens=320, do_sample=False)
            crudo = proc.batch_decode(out[:, ent["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()
            js = vision.json_de(crudo)
            est = (vision.estructurar_lineas(js["lineas"])
                   if isinstance(js, dict) and isinstance(js.get("lineas"), list) else js)
            print(f"-- {nombre}: {crudo[:300]!r}")
            print(f"   => {est}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
