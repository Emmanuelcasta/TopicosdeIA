# =========================================================================
# CELDA PARA PEGAR en S13, justo DESPUÉS de la celda 50 (`sistema_multimodal`).
#
# No es un módulo: es el contenido de una celda. Se pega tal cual en el notebook
# que ya está corriendo, así no hay que regenerar nada ni volver a cargar el
# modelo.
#
# Qué mide y por qué este conjunto. Los 3 casos `dev-*` del eval set de M4 son
# 1 de localización, 1 sin error y 1 compensado: **cero adversariales**. Así que
# la puerta no puede decir nada sobre las tres categorías de abstención, que son
# 3 de los 12 casos de evaluación y donde todo ha dado 0/3.
#
# Los 9 ejemplos `dg-*` del split de desarrollo de las trayectorias sí cubren las
# cinco categorías. El notebook de entrenamiento ya los midió, pero con
# `generate` directo, sin agente ni herramientas — y por eso aquel 9/9 de formato
# no predijo nada. Esta es la primera medición de ese conjunto **con el arnés
# real de S13**.
#
# NO toca los 12 casos de evaluación ni el eval set. NO es un scorecard de M4.
# =========================================================================
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

RUTA_DG = Path("data/trayectorias_diagnostico.jsonl")
SHA_DG = "87bd8f19b983369395cc9a9e29c62b1e211776c74539258a62e2bb58d5147589"

# --- 1 · los datos, verificados antes de medir nada ----------------------
if not RUTA_DG.exists():
    raise FileNotFoundError(
        f"No está {RUTA_DG}. La escribe el notebook de fine-tuning "
        "(S13_M4_FineTuning_Diagnostico) al ejecutar su celda 13.")

sha_dg = hashlib.sha256(RUTA_DG.read_bytes()).hexdigest()
print(f"{RUTA_DG}: SHA coincide: {sha_dg == SHA_DG}")
if sha_dg != SHA_DG:
    raise SystemExit(
        f"el archivo no es el que se usó para entrenar ({sha_dg[:16]}… vs "
        f"{SHA_DG[:16]}…). Midiendo contra otro conjunto no se concluye nada.")

TODOS_DG = [json.loads(l) for l in RUTA_DG.read_text(encoding="utf-8").splitlines()]
DEV_DG = [e for e in TODOS_DG if e["split"] == "dev"]

# --- 2 · que sean de desarrollo y que cada uno traiga su categoría -------
assert DEV_DG, "no hay ejemplos de desarrollo"
assert all(e["split"] == "dev" for e in DEV_DG), "se coló un ejemplo de entrenamiento"

ids_train = {e["id"] for e in TODOS_DG if e["split"] == "train"}
assert not ({e["id"] for e in DEV_DG} & ids_train), "un id está en los dos splits"

for e in DEV_DG:
    esperada = e["diagnostico"].lower()
    assert diagnostico_valido(esperada), f"{e['id']}: categoría no válida {esperada!r}"
    # La etiqueta tiene que leerse con el MISMO parser con el que se va a medir.
    assert extraer_diagnostico(e["mensajes"][-1]["content"]) == esperada, \
        f"{e['id']}: la referencia no se parsea a su propia etiqueta"

cubiertas = {e["diagnostico"].split()[0].lower() for e in DEV_DG}
faltan = {"paso", "sin", "ilegible", "fuera", "dato"} - cubiertas
assert not faltan, f"el conjunto no cubre: {sorted(faltan)}"

print(f"{len(DEV_DG)} ejemplos de desarrollo, todos con categoría válida.")
print("Por categoría:", dict(Counter(e["diagnostico"].lower() for e in DEV_DG)))
print(f"Adaptador de la vía visual: {ADAPTADOR_DIAGNOSTICO!r}\n")

# --- 3 · medir con el arnés real -----------------------------------------
# `mensajes[1]["content"]` ES `prompt_revision(pregunta, lectura)`: el mismo
# prompt que construye `sistema_multimodal`. Pasarlo a `responder_diagnostico`
# recorre exactamente la vía visual de S13 — AgenteTutor, system de C5, los 16
# esquemas de herramientas y el reintento— sin volver a construirlo.
filas_dg = []
for e in DEV_DG:
    prompt = e["mensajes"][1]["content"]
    esperada = e["diagnostico"].lower()

    respuesta = responder_diagnostico(prompt)
    respuesta, reintento = asegurar_diagnostico(respuesta, responder_diagnostico)
    conclusion = extraer_diagnostico(respuesta)

    traza = TRAZAS_M4.get(prompt[:160])
    herramientas = [o["herramienta"] for o in (traza.observaciones if traza else [])]

    filas_dg.append({
        "id": e["id"], "familia": e["familia"], "esperada": esperada,
        "conclusion": conclusion,
        "valida": diagnostico_valido(conclusion),
        "acierto": diagnostico_valido(conclusion) and conclusion == esperada,
        "reintento": reintento,
        "reescribe": "Respuesta final" in respuesta,
        "pasos_llm": getattr(traza, "pasos_llm", None),
        "terminacion": getattr(traza, "terminacion", None),
        "herramientas": "|".join(herramientas) or "—",
        "respuesta": respuesta,
    })
    marca = "OK " if filas_dg[-1]["acierto"] else "MAL"
    print(f"  {e['id']:8s} {e['familia']:18s} {marca}  esperaba {esperada!r:20s} "
          f"concluyó {str(conclusion)!r:20s}{'  (reintento)' if reintento else ''}")

DG = pd.DataFrame(filas_dg)

# --- 4 · exactitud por categoría -----------------------------------------
print("\n" + "=" * 78)
print("DESARROLLO `dg-*` CON EL ARNÉS DE S13 · no es un scorecard de M4")
print("=" * 78)
print(f"{'categoría esperada':24s}{'acierto':>10s}{'formato':>10s}   qué prueba")
QUE_PRUEBA = {
    "paso": "localizar el paso erróneo",
    "sin error": "NO inventar un error donde no lo hay",
    "ilegible": "abstenerse: la hoja no se lee",
    "fuera de dominio": "abstenerse: no es matemáticas",
    "dato no disponible": "abstenerse: el dato no está escrito",
}
for cat, que in QUE_PRUEBA.items():
    g = DG[DG.esperada.str.startswith(cat)] if cat == "paso" else DG[DG.esperada == cat]
    if len(g):
        print(f"{cat:24s}{int(g.acierto.sum()):>6d}/{len(g):<3d}"
              f"{int(g.valida.sum()):>6d}/{len(g):<3d}   {que}")
print("-" * 78)
print(f"{'TOTAL':24s}{int(DG.acierto.sum()):>6d}/{len(DG):<3d}"
      f"{int(DG.valida.sum()):>6d}/{len(DG):<3d}")
print(f"{'con reintento':24s}{int(DG.reintento.sum()):>10d}")
print(f"{'reescribe el ejercicio':24s}{int(DG.reescribe.sum()):>10d}")
print("=" * 78)

# La pregunta que bloquea el paso a los 12 casos.
abstenciones = DG[DG.esperada.isin(["ilegible", "fuera de dominio", "dato no disponible"])]
sin_error = DG[DG.esperada == "sin error"]
produjo = set(DG.conclusion.dropna())
print("\n¿EL ADAPTADOR LLEGA A PRODUCIR ALGO QUE NO SEA 'paso N'?")
print(f"  conclusiones distintas emitidas: {sorted(produjo)}")
print(f"  aciertos en 'sin error'        : {int(sin_error.acierto.sum())}/{len(sin_error)}")
print(f"  aciertos en abstenciones       : {int(abstenciones.acierto.sum())}/{len(abstenciones)}")
if all(str(c).startswith("paso") for c in produjo if c):
    print("\n  Contesta SIEMPRE un paso. Entonces los 6 casos de `sin_error` y")
    print("  adversarial del eval set darían 0 por construcción, y gastarlos no")
    print("  informaría de nada. NO pasen a los 12 casos.")

# --- 5 · respuestas completas y trazas de herramientas -------------------
print("\n" + "=" * 78)
print("RESPUESTAS COMPLETAS Y TRAZAS")
print("=" * 78)
for f in filas_dg:
    print(f"\n--- {f['id']} · {f['familia']} · esperaba {f['esperada']!r} ---")
    print(f"herramientas: {f['herramientas']}  |  pasos_llm: {f['pasos_llm']}  |  "
          f"terminación: {f['terminacion']}")
    print(f["respuesta"])

Path("m4").mkdir(exist_ok=True)
DG.to_csv("m4/dev_dg_arnes_s13.csv", index=False)
print("\nGuardado: m4/dev_dg_arnes_s13.csv")
print("Esto NO es el scorecard de M4: son 9 casos de desarrollo, sin imágenes,")
print("y sirven para decidir si tiene sentido gastar los 12 de evaluación.")
