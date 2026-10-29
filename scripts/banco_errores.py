"""
Banco de errores: cada fallo medido, con su causa y el cambio que lo atacaría.

El problema que resuelve: los fallos del proyecto están repartidos entre varios
CSV de detalle, las trazas del agente y los comentarios de la profesora. Para
decidir qué arreglar hay que cruzarlos a mano cada vez, y la trazabilidad entre
"este error" → "este cambio" → "esta prueba" se pierde.

Cada entrada del banco responde seis preguntas:

    pregunta · respuesta del modelo · respuesta esperada · tipo de error ·
    causa probable · cambio propuesto

y además guarda de dónde salió (`origen`), qué herramientas se usaron y si el
verificador de planteamiento lo habría detectado. Ese último campo es el que
convierte el banco en una prueba: si un cambio dice atacar el planteamiento, el
banco dice exactamente sobre qué 13 casos debería notarse.

La clasificación es automática y conservadora. Donde no hay evidencia para
decidir la causa, se marca `por_revisar` en vez de inventar una.

Uso:
    python scripts/banco_errores.py                 # escribe el banco y el resumen
    python scripts/banco_errores.py --tipo planteamiento
    python scripts/banco_errores.py --sistema "C5-FT · entrenado"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "agente"))

from verificador import VerificadorPlan  # noqa: E402

RAIZ = AQUI.parent
RES_M3 = RAIZ / "M3_rag_agentes" / "resultados"
SALIDA = RAIZ / "data" / "errores_observados.jsonl"
RESUMEN = RAIZ / "M3_rag_agentes" / "resultados" / "banco_errores.md"

# Qué cambio ataca cada tipo de error. Es la tabla que convierte el diagnóstico
# en trabajo, y la que hay que actualizar cuando un cambio se implementa.
CAMBIO_PROPUESTO = {
    "planteamiento": ("verificador de planteamiento (scripts/agente/verificador.py): revisa que "
                      "los argumentos vengan del enunciado y que no quede ningún dato sin usar"),
    "no consultó documentos": ("enrutador de S08 integrado al agente (scripts/agente/enrutador.py): "
                               "decide buscar antes de responder"),
    "leyó el documento y no lo usó": ("prompt de cita textual en la vía documental; si no basta, el "
                                      "límite es el tamaño del modelo"),
    "aritmética de cabeza": ("adaptador entrenado con trayectorias de herramientas (ya existe: "
                             "adaptadores/qwen-lora-agente)"),
    "herramienta falló": "revisar el esquema y la conversión de argumentos del registro",
    "robustez": ("ejemplos de abstención en el corpus y válvula de escape explícita en el prompt"),
    "falso acierto": "ninguno: acertó pero sin apoyo verificable; vigilar en RAGAS",
}

CAUSA_PROBABLE = {
    "planteamiento": "el modelo eligió mal qué calcular o dejó el plan a medias; la herramienta respondió bien",
    "no consultó documentos": "el agente no decide buscar: el dato no está en los pesos y nadie lo trajo",
    "leyó el documento y no lo usó": "el chunk correcto llegó al prompt y la generación lo ignoró",
    "aritmética de cabeza": "sin herramientas, un modelo de 1.5B se equivoca en la cuenta",
    "herramienta falló": "argumentos mal formados o fuera del dominio de la función",
    "robustez": "la pregunta induce una respuesta y el modelo no se abstiene",
}


def clasificar(reg: dict, fila: dict) -> str:
    obs = reg["observaciones"]
    exitosas = [o for o in obs if o["ok"] and o.get("valor")]
    busco = any(o["herramienta"] == "buscar_documentos" for o in obs)
    if fila["bloque"] == "adversarial":
        return "robustez"
    if fila["bloque"] in ("conocimiento", "compuesto"):
        if not (busco or reg["contextos"]):
            return "no consultó documentos"
        if not exitosas:
            return "leyó el documento y no lo usó"
    if any(not o["ok"] for o in obs):
        return "herramienta falló"
    if exitosas:
        return "planteamiento"
    return "aritmética de cabeza"


def esperado_de(caso: dict) -> str:
    ev = caso.get("evaluacion", {})
    if ev.get("tipo", "").startswith("numerico"):
        return ev.get("valor", "")
    claves = ev.get("claves") or []
    return " | ".join(claves[:3])


def construir(verificador: VerificadorPlan) -> list[dict]:
    casos = {}
    for archivo in ("eval_set_m2.jsonl", "eval_set_m3_agente.jsonl"):
        ruta = RAIZ / "data" / archivo
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            c = json.loads(linea)
            casos[c["id"]] = c

    respuestas = {}
    for linea in (RES_M3 / "respuestas_s10.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(linea)
        respuestas[(r["config"], r["id"])] = r

    entradas = []
    for fila in csv.DictReader((RES_M3 / "detalle_s10_v2.csv").open(encoding="utf-8")):
        if fila["acierto"].lower() == "true":
            continue
        clave = (fila["config"], fila["id"])
        reg = respuestas.get(clave)
        if reg is None:
            continue
        tipo = clasificar(reg, fila)
        rev = verificador.revisar(reg["pregunta"], reg["respuesta"], reg["observaciones"],
                                  reg.get("contextos", ()))
        caso = casos.get(fila["id"], {})
        entradas.append({
            "id": f"{fila['id']}@{fila['config'].split(' · ')[0]}",
            "caso": fila["id"], "sistema": fila["config"],
            "bloque": fila["bloque"], "subtipo": fila["subtipo"],
            "pregunta": reg["pregunta"],
            "respuesta_modelo": reg["respuesta"],
            "respuesta_esperada": esperado_de(caso) or caso.get("esperado", ""),
            "motivo_harness": fila["motivo"],
            "tipo_error": tipo,
            "causa_probable": CAUSA_PROBABLE.get(tipo, "por_revisar"),
            "cambio_propuesto": CAMBIO_PROPUESTO.get(tipo, "por_revisar"),
            "herramientas_usadas": [o["herramienta"] for o in reg["observaciones"]],
            "detectado_por_verificador": bool(rev.graves),
            "hallazgo_verificador": rev.graves[0].detalle if rev.graves else None,
            "origen": "M3_rag_agentes/resultados/detalle_s10_v2.csv",
            "estado": "abierto",
        })
    return entradas


def resumen_md(entradas: list[dict]) -> str:
    por_tipo = Counter(e["tipo_error"] for e in entradas)
    sistemas = sorted({e["sistema"] for e in entradas})
    lin = ["# Banco de errores", "",
           f"{len(entradas)} fallos catalogados sobre {len(sistemas)} configuraciones, "
           "generados desde las trazas de S10 con `scripts/banco_errores.py`.", "",
           "Cada entrada trae pregunta, respuesta del modelo, respuesta esperada, tipo de error,",
           "causa probable y el cambio que lo ataca. El campo `detectado_por_verificador` dice si",
           "el verificador de planteamiento lo habría señalado: ese es el conjunto sobre el que",
           "debería notarse ese cambio concreto.", "",
           "## Por tipo de error", "", "| Tipo | n | Cambio propuesto |", "|---|---|---|"]
    for tipo, n in por_tipo.most_common():
        lin.append(f"| {tipo} | {n} | {CAMBIO_PROPUESTO.get(tipo, 'por revisar')} |")

    lin += ["", "## Por sistema", "", "| Sistema | fallos | detectados por el verificador |", "|---|---|---|"]
    for s in sistemas:
        de_s = [e for e in entradas if e["sistema"] == s]
        det = sum(e["detectado_por_verificador"] for e in de_s)
        lin.append(f"| {s} | {len(de_s)} | {det} |")

    foco = [e for e in entradas if e["sistema"].startswith("C5-FT")]
    lin += ["", "## El sistema que se va a mejorar · C5-FT", "",
            "| caso | tipo de error | herramientas | ¿lo ve el verificador? |", "|---|---|---|---|"]
    for e in sorted(foco, key=lambda e: e["tipo_error"]):
        h = "|".join(e["herramientas_usadas"])[:34] or "—"
        lin.append(f"| `{e['caso']}` | {e['tipo_error']} | {h} | {'sí' if e['detectado_por_verificador'] else 'no'} |")
    return "\n".join(lin) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tipo")
    p.add_argument("--sistema")
    args = p.parse_args()

    entradas = construir(VerificadorPlan())
    SALIDA.parent.mkdir(exist_ok=True)
    with SALIDA.open("w", encoding="utf-8") as fh:
        for e in entradas:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    RESUMEN.write_text(resumen_md(entradas), encoding="utf-8")

    print(f"{len(entradas)} errores catalogados")
    print(f"  {SALIDA}")
    print(f"  {RESUMEN}")
    print("\nPor tipo:")
    for tipo, n in Counter(e["tipo_error"] for e in entradas).most_common():
        det = sum(e["detectado_por_verificador"] for e in entradas if e["tipo_error"] == tipo)
        print(f"  {n:3d}  {tipo:30s} (verificador detecta {det})")

    filtradas = [e for e in entradas
                 if (not args.tipo or e["tipo_error"] == args.tipo)
                 and (not args.sistema or e["sistema"] == args.sistema)]
    if args.tipo or args.sistema:
        print(f"\n{len(filtradas)} entradas que coinciden:")
        for e in filtradas:
            print(f"  {e['id']:22s} {e['motivo_harness'][:46]}")


if __name__ == "__main__":
    main()
