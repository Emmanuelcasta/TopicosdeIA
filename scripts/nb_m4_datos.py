"""La celda del eval set de M4, embebida igual que el corpus y los eval sets anteriores.

Mismo patrón que `nb_comun_agente.celda_eval_m3`: el archivo viaja dentro del
notebook como blob gzip+base64 y se verifica por SHA-256, para que la corrida en
Colab no dependa de que Drive tenga la copia correcta — el problema que ya costó
reejecutar S08 y S10 con las claves viejas de `adv-02`.

Lo propio de M4 es el **alias de imagen**: `vis-12` usa a propósito la foto de
`vis-07` con otra pregunta. El eval set guarda `vis-12.jpg` como nombre lógico y
quien lo resuelve es `archivo_de`, en `scripts/agente/vision.py`, donde está
probado. Así se fotografían 11 hojas para 12 casos sin tocar el eval set.
"""

from __future__ import annotations

from nb_comun import code, md
from nb_comun_agente import _codigo_blob


CODIGO_EVAL_M4 = '''
# Se separan los dos splits. `dev` existe para mirar y afinar; `eval` solo se
# ejecuta, y sus 12 casos son los que se reportan.
eval_m4 = [c for c in eval_m4_todo if c["split"] == "eval"]
dev_m4 = [c for c in eval_m4_todo if c["split"] == "dev"]
CASOS_M4 = {c["id"]: c for c in eval_m4_todo}

# `archivo_de` resuelve el alias de `vis-12` -> `vis-07.jpg` y viene de vision.py.
from collections import Counter
print("Casos de evaluación:", len(eval_m4), "| de desarrollo:", len(dev_m4))
print("Por bloque:", dict(Counter(c["bloque"] for c in eval_m4)))
print("Por criterio:", dict(Counter(c["criterio_s13"] for c in eval_m4)))
print("Archivos de imagen distintos que se necesitan:",
      len({archivo_de(c) for c in eval_m4}))
for c in eval_m4:
    alias = "" if archivo_de(c) == c["imagen"] else f"  (alias de {archivo_de(c)})"
    print(f"  {c['id']:8s} {c['bloque']:13s} {c['criterio_s13']:9s} {c['imagen']}{alias}")
'''


def celda_eval_m4(prefijo: str = "3.2") -> list[dict]:
    return [
        md(f"""
### {prefijo} · El eval set de M4

`data/eval_set_m4_vision.jsonl`, generado y validado por
`scripts/eval_set_m4_fuente.py`. Se escribió **antes** de tener las fotos y antes
de escribir el pipeline: el criterio de cada caso está fijado de antemano, que es
la única forma de que medir signifique algo.

| Bloque | n | Qué mide |
|---|---|---|
| **localización** | 4 | Nombrar el paso donde está el error |
| **compensado** | 2 | Un paso mal con el resultado final **bien**: lo que un corrector de respuestas no ve |
| **sin error** | 3 | **No** inventar un error donde no lo hay |
| **adversarial** | 3 | Abstenerse: hoja ilegible, hoja que no es matemáticas, dato que no está escrito |

Cada caso trae `evaluacion` con las **claves aceptadas** y los **textos
prohibidos**, y un `criterio_s13` (`exacto` / `valvula`) que lo liga a la
plantilla de S13. La calificación la hace el mismo `evaluar_caso` de M2 y M3.

Hay además 3 casos de **desarrollo** (`dev-*`) que no se reportan: existen para
mirar sin gastar los de evaluación, y no se fotografían.

> **`vis-12` reutiliza la hoja de `vis-07`.** La foto es la misma y lo que cambia
> es la pregunta: se pide un dato que no está escrito en la hoja. Por eso son
> **12 casos y 11 fotos**.
"""),
        code("import base64, gzip, hashlib, json\nfrom pathlib import Path\n"
             + _codigo_blob("data/eval_set_m4_vision.jsonl", "eval_m4_todo")
             + CODIGO_EVAL_M4),
    ]
