"""Empaqueta la entrega de M3 desde una sola fuente canónica.

El problema que resuelve. `M3_rag_agentes/entrega/` contiene 26 archivos, y 24 de
ellos son copias **byte a byte** de archivos que ya viven en
`M3_rag_agentes/notebooks/`, `M3_rag_agentes/resultados/` y
`M2_harness/documentacion/`. Son 3,3 MB repetidos que hasta ahora se copiaron a
mano, sin forma de saber si una copia se había quedado atrás.

La decisión es conservar el paquete, no eliminarlo: es lo que se entregó y tiene
que poder abrirse solo, sin depender del resto del repositorio. Lo que cambia es
que deja de ser un duplicado a mano y pasa a ser un **artefacto derivado**, con
una fuente declarada y verificable.

Dos archivos NO se derivan de nada: `M2_M3_Documentacion.md` y
`M3_Agente_Documentacion.md` se escribieron para la entrega y su copia canónica
está ahí mismo. El script los comprueba pero no los toca.

Uso:
    python scripts/empaquetar_entrega.py              # verifica (no escribe nada)
    python scripts/empaquetar_entrega.py --reconstruir
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ENTREGA = RAIZ / "M3_rag_agentes" / "entrega"

# destino dentro de entrega/  ->  fuente canónica en el repositorio
DERIVADOS = {
    "documentacion/SCOREBOARD.md": "M2_harness/documentacion/SCOREBOARD.md",
}
for _n in ("S08_Lab_RAG_Avanzado_Tutor_Matematicas",
           "S10_Evaluacion_RAGAS_Tutor_Matematicas",
           "S10_FineTuning_Agente_Herramientas",
           "S10_Lab_Agente_ReAct_Tutor_Matematicas",
           "S10_Lab_Herramientas_Tutor_Matematicas"):
    DERIVADOS[f"notebooks/{_n}.ipynb"] = f"M3_rag_agentes/notebooks/{_n}.ipynb"
for _n in ("correccion_adv02.md", "deltas_s08.csv", "deltas_s08_v2.csv",
           "deltas_s10.csv", "deltas_s10_v2.csv", "detalle_rag_avanzado_v2.csv",
           "detalle_s10_v2.csv", "finetuning_agente.json", "m3_agente.json",
           "m3_rag_avanzado.json", "ragas_s10.json", "ragas_scorecard_s10.csv",
           "respuestas_s10.jsonl", "retrieval_s08.csv", "scorecard_agente_v2.csv",
           "scorecard_final_m3.csv", "scorecard_final_m3_v2.csv",
           "scorecard_rag_avanzado_v2.csv"):
    DERIVADOS[f"resultados/{_n}"] = f"M3_rag_agentes/resultados/{_n}"

# Propios de la entrega: su copia canónica es la que está en entrega/.
PROPIOS = ("documentacion/M2_M3_Documentacion.md",
           "documentacion/M3_Agente_Documentacion.md")


def sha(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def verificar() -> list:
    """-> lista de problemas. No escribe nada."""
    problemas = []
    for destino, fuente in sorted(DERIVADOS.items()):
        d, f = ENTREGA / destino, RAIZ / fuente
        if not f.exists():
            problemas.append(f"falta la FUENTE {fuente}")
            continue
        if not d.exists():
            problemas.append(f"falta en la entrega: {destino}")
            continue
        if sha(d) != sha(f):
            problemas.append(f"DESFASADO: {destino} ya no coincide con {fuente}")
    for propio in PROPIOS:
        if not (ENTREGA / propio).exists():
            problemas.append(f"falta el documento propio de la entrega: {propio}")
    return problemas


def reconstruir() -> int:
    copiados = 0
    for destino, fuente in sorted(DERIVADOS.items()):
        d, f = ENTREGA / destino, RAIZ / fuente
        if not f.exists():
            print(f"  SIN FUENTE  {fuente}")
            continue
        if d.exists() and sha(d) == sha(f):
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, d)
        print(f"  copiado  {destino}")
        copiados += 1
    return copiados


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reconstruir", action="store_true",
                    help="copia desde las fuentes canónicas; por defecto solo verifica")
    args = ap.parse_args()

    print(f"Entrega M3: {len(DERIVADOS)} archivos derivados + {len(PROPIOS)} propios")
    if args.reconstruir:
        n = reconstruir()
        print(f"\n{n} archivos actualizados." if n else "\nNada que actualizar.")

    problemas = verificar()
    if problemas:
        print(f"\n{len(problemas)} PROBLEMAS:")
        for p in problemas:
            print("  -", p)
        sys.exit(1)
    print("\nLa entrega coincide con sus fuentes canónicas. Es autocontenida y"
          "\nreproducible: se puede regenerar con --reconstruir.")


if __name__ == "__main__":
    main()
