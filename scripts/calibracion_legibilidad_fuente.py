"""Conjunto de calibración para la detección de hojas ilegibles.

Por qué hace falta un conjunto aparte. La corrida del 24-10 dejó claro que la
coincidencia entre dos lecturas no sirve: ante la foto desenfocada de `vis-10` el
VLM produjo **dos transcripciones idénticas carácter a carácter**, las dos
inventando `24 x 13 = 306` donde la hoja dice 302. No era inestabilidad de
muestreo; era determinismo.

La detección que la sustituye tiene señales con umbral (la confianza en los
tokens numéricos, marcada con mediana − k·MAD). Calibrar ese `k` mirando
`vis-10` convertiría un caso de evaluación en conjunto de ajuste, y entonces el
0/3 adversarial dejaría de significar nada. De ahí este archivo: un conjunto de
pares **legible / ilegible** que no sale del eval set.

Lo que genera, y lo que NO vale:

  · Tríos sintéticos: cada hoja se dibuja con texto **impreso** y se producen dos
    copias degradadas, una leve y otra fuerte. La leve es la que importa: es el
    análogo de `vis-10`, donde el VLM todavía alcanza a leer parte y rellena el
    resto. La fuerte es el control: si la salvaguarda no caza esa, no funciona.

  · **No valida el rendimiento sobre letra manuscrita.** Texto impreso borroso y
    letra a mano borrosa no son el mismo problema para un VLM: la segunda es más
    difícil incluso nítida. Un resultado bueno aquí NO autoriza a afirmar que la
    salvaguarda funciona con las fotos reales.

  · La validación de verdad necesita **6 fotos**: las 3 hojas `dev-*` escritas a
    mano, cada una fotografiada nítida y desenfocada. Mientras no existan, lo
    honesto es decir que la salvaguarda está sin validar, y el notebook lo dice.

Los enunciados son los de los casos `dev-*` más tres inventados. Ninguno sale del
eval set de M4, y los `dev-*` no se fotografían nunca, así que usarlos aquí no
toca ningún caso de evaluación.

Uso:
    python calibracion_legibilidad_fuente.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

RAIZ = AQUI.parent
DESTINO = RAIZ / "m4" / "eval_set" / "calibracion"
MANIFIESTO = DESTINO / "manifiesto.json"
EVAL_M4 = RAIZ / "data" / "eval_set_m4_vision.jsonl"

# Tres hojas inventadas, además de las `dev-*`. Suficientes para que el criterio
# robusto tenga "resto del conjunto" contra el que comparar.
EXTRA = [
    {"id": "cal-01",
     "enunciado": "Un ramo trae 13 flores. ¿Cuántas flores hay en 6 ramos?",
     "pasos": ["Multiplicamos: 13 × 6.", "13 × 6 = 78.", "Respuesta final: 78 flores"]},
    {"id": "cal-02",
     "enunciado": "Se reparten 540 fichas entre 18 cajas. ¿Cuántas por caja?",
     "pasos": ["Dividimos: 540 ÷ 18.", "540 ÷ 18 = 30.", "Respuesta final: 30 fichas"]},
    {"id": "cal-03",
     "enunciado": "Un bidón tiene 1250 mililitros y se usan 480. ¿Cuánto queda?",
     "pasos": ["Restamos: 1250 − 480.", "1250 − 480 = 770.", "Respuesta final: 770 ml"]},
]


def _hojas() -> list:
    """Las `dev-*` del eval set más las inventadas. Nunca un caso `vis-*`."""
    hojas = list(EXTRA)
    if EVAL_M4.exists():
        for linea in EVAL_M4.read_text(encoding="utf-8").splitlines():
            c = json.loads(linea)
            if c["split"] == "dev" and c["pasos"]:
                hojas.append({"id": c["id"], "enunciado": c["enunciado"],
                              "pasos": list(c["pasos"])})
    return hojas


def _dibujar(hoja: dict, ancho: int = 1024, alto: int = 768):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (ancho, alto), (250, 249, 245))
    d = ImageDraw.Draw(im)
    try:
        grande = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 34)
        normal = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    except OSError:
        grande = normal = ImageFont.load_default()

    y = 70
    d.multiline_text((70, y), hoja["enunciado"], fill=(30, 30, 40), font=grande,
                     spacing=10)
    y += 130
    for i, paso in enumerate(hoja["pasos"], 1):
        d.text((70, y), f"Paso {i}: {paso}", fill=(30, 30, 40), font=normal)
        y += 70
    return im


# Dos niveles de degradación, y el motivo importa. Un primer intento con
# desenfoque fuerte dejó las hojas casi en blanco: más ilegibles que `vis-10`,
# donde el VLM todavía alcanza a leer "24 x 13" y rellena el resultado. Una
# calibración más fácil que el caso real daría una lectura falsamente optimista
# de la salvaguarda.
#
#   leve   -> análogo a una foto trepidada de mano: se intuye el texto. Es el
#             nivel que de verdad pone a prueba la detección.
#   fuerte -> ilegible sin discusión. Sirve de control: si la salvaguarda no caza
#             ESTE, es que no funciona en absoluto.
NIVELES = {"ilegible_leve": dict(radio=3.0, desplazamiento=5, contraste=0.72),
           "ilegible_fuerte": dict(radio=6.0, desplazamiento=9, contraste=0.45)}


def _degradar(im, radio, desplazamiento, contraste):
    """Desenfoque, movimiento y bajo contraste: lo que hace ilegible una foto."""
    from PIL import Image, ImageEnhance, ImageFilter

    out = im.filter(ImageFilter.GaussianBlur(radius=radio))
    # Movimiento: se superpone la imagen desplazada, como una foto trepidada.
    movida = out.transform(out.size, Image.AFFINE,
                           (1, 0, desplazamiento, 0, 1, desplazamiento * 0.6),
                           resample=Image.BILINEAR)
    out = Image.blend(out, movida, 0.5)
    out = ImageEnhance.Contrast(out).enhance(contraste)
    out = ImageEnhance.Brightness(out).enhance(1.08)
    return out


def main() -> None:
    hojas = _hojas()
    if not hojas:
        raise SystemExit("no hay hojas de calibración")

    DESTINO.mkdir(parents=True, exist_ok=True)
    registros = []
    for hoja in hojas:
        nitida = _dibujar(hoja)
        versiones = [("legible", nitida)]
        versiones += [(n, _degradar(nitida, **cfg)) for n, cfg in NIVELES.items()]
        for etiqueta, im in versiones:
            nombre = f"{hoja['id']}-{etiqueta}.png"
            im.save(DESTINO / nombre)
            registros.append({"archivo": nombre, "hoja": hoja["id"],
                              "etiqueta": etiqueta,
                              "legible": etiqueta == "legible",
                              "enunciado": hoja["enunciado"],
                              "pasos": hoja["pasos"]})

    MANIFIESTO.write_text(json.dumps({
        "aviso": ("Conjunto de calibración SINTÉTICO, con texto impreso. Sirve para "
                  "comprobar que la detección funciona, NO para afirmar que funciona "
                  "con letra manuscrita. La validación real necesita fotos de las "
                  "hojas dev-* escritas a mano, nítidas y desenfocadas."),
        "no_es_eval_set": ("Ninguna hoja sale de los 12 casos vis-*. Las dev-* no se "
                           "fotografían nunca, así que usarlas aquí no toca ningún "
                           "caso de evaluación."),
        "pares": registros,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(registros)} imágenes ({len(hojas)} hojas × 3) -> {DESTINO}")
    for r in registros:
        print(f"  {r['archivo']:26s} {r['etiqueta']}")
    print(f"\nManifiesto: {MANIFIESTO}")
    print("\nAVISO: es texto impreso, no letra manuscrita. Un buen resultado aquí no")
    print("autoriza a decir que la salvaguarda funciona con las fotos reales.")


if __name__ == "__main__":
    main()
