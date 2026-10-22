"""
Genera los notebooks del proyecto a partir de las definiciones en
`nb_*.py`.

Por qué generarlos con un script en lugar de editarlos a mano:

1. Las celdas compartidas (carga de datos, métricas, W&B) se escriben una sola
   vez en `nb_comun.py`. Editarlas a mano en cinco archivos garantizaría que
   tarde o temprano dejaran de coincidir, y con ello la comparabilidad del
   experimento.
2. El corpus embebido en cada notebook proviene del mismo `data/*.jsonl`, así
   que los cinco comparten el mismo hash SHA-256 por construcción.
3. Regenerar tras un cambio en el dataset es un solo comando.

Uso:
    python dataset_fuente.py      # 1. valida el corpus y escribe el JSONL
    python generar_notebooks.py   # 2. reconstruye los notebooks

El script NO ejecuta ningún notebook: solo escribe el JSON de cada uno.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nb_bert
import nb_comparacion
import nb_flan_t5
import nb_harness
import nb_pipeline
import nb_qwen
import nb_rag
import nb_rag_avanzado
import nb_herramientas
import nb_finetuning_agente
import nb_agente
import nb_ragas
import nb_m4
from nb_comun import RAIZ, notebook

NOTEBOOKS = [
    ("S04_Lab_Fine_tuning_Qwen.ipynb", nb_qwen),
    ("S04_Lab_Fine_tuning_BERT.ipynb", nb_bert),
    ("S04_Lab_Fine_tuning_Qwen_BERT.ipynb", nb_pipeline),
    ("S04_Lab_Fine_tuning_FLAN_T5.ipynb", nb_flan_t5),
    ("S04_Comparacion_Arquitecturas.ipynb", nb_comparacion),
    ("S06_Lab_Harness_Tutor_Matematicas.ipynb", nb_harness),
    ("S07_Lab_RAG_Tutor_Matematicas.ipynb", nb_rag),
    ("S08_Lab_RAG_Avanzado_Tutor_Matematicas.ipynb", nb_rag_avanzado),
    ("S10_Lab_Herramientas_Tutor_Matematicas.ipynb", nb_herramientas),
    ("S10_FineTuning_Agente_Herramientas.ipynb", nb_finetuning_agente),
    ("S10_Lab_Agente_ReAct_Tutor_Matematicas.ipynb", nb_agente),
    ("S10_Evaluacion_RAGAS_Tutor_Matematicas.ipynb", nb_ragas),
    ("S13_M4_Pipeline_Visual.ipynb", nb_m4),
]


# Cada notebook vive en la carpeta de su módulo. M4 no usa la subcarpeta
# `notebooks/`: su pipeline vive junto al eval set y a las fotos, en `m4/`.
CARPETA_DE = {"S04": "M1_arquitecturas/notebooks", "S06": "M2_harness/notebooks",
              "S07": "M3_rag_agentes/notebooks", "S08": "M3_rag_agentes/notebooks",
              "S10": "M3_rag_agentes/notebooks", "S13": "m4/pipeline"}


def validar_celdas(celdas: list[dict], nombre: str) -> list[str]:
    """Comprobaciones básicas de que el notebook está bien formado."""
    problemas = []
    for i, celda in enumerate(celdas):
        if celda["cell_type"] not in ("markdown", "code"):
            problemas.append(f"{nombre}[{i}]: tipo de celda inválido")
        if not celda["source"] or not any(s.strip() for s in celda["source"]):
            problemas.append(f"{nombre}[{i}]: celda vacía")
        if celda["cell_type"] == "code":
            fuente = "".join(celda["source"])
            # Las celdas con magias (%pip, %%) no son Python válido, se saltan.
            if not fuente.lstrip().startswith("%"):
                try:
                    compile(fuente, f"{nombre}[{i}]", "exec")
                except SyntaxError as e:
                    problemas.append(f"{nombre}[{i}]: SyntaxError linea {e.lineno}: {e.msg}")
    return problemas


def main(filtros: list[str] | None = None) -> None:
    """Sin `filtros` regenera los notebooks de la lista; con ellos, solo los que
    contienen alguno de los textos dados (`python generar_notebooks.py S13`).

    El filtro existe para poder trabajar en un módulo sin reescribir los otros
    once notebooks, que ya están ejecutados y entregados.
    """
    seleccion = [(n, m) for n, m in NOTEBOOKS
                 if not filtros or any(f in n for f in filtros)]
    if not seleccion:
        raise SystemExit(f"Ningún notebook coincide con {filtros}. "
                         f"Disponibles: {[n for n, _ in NOTEBOOKS]}")

    for archivo, generador in [("math_tutor_dataset.jsonl", "dataset_fuente.py"),
                               ("eval_set_m2.jsonl", "eval_set_fuente.py"),
                               ("eval_set_m3_agente.jsonl", "eval_set_m3_fuente.py"),
                               ("trayectorias_herramientas.jsonl", "trayectorias_fuente.py"),
                               ("eval_set_m4_vision.jsonl", "eval_set_m4_fuente.py")]:
        if not (RAIZ / "data" / archivo).exists():
            raise SystemExit(f"Falta data/{archivo}. Ejecuten antes: python {generador}")

    todos_problemas = []
    for nombre, modulo in seleccion:
        celdas = modulo.construir()
        problemas = validar_celdas(celdas, nombre)
        todos_problemas.extend(problemas)

        nb = notebook(celdas, nombre)
        destino = RAIZ / CARPETA_DE[nombre[:3]] / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        n_md = sum(1 for c in celdas if c["cell_type"] == "markdown")
        n_code = len(celdas) - n_md
        print(f"  {nombre:42s} {len(celdas):3d} celdas "
              f"({n_md} markdown, {n_code} código)  "
              f"{destino.stat().st_size/1024:6.1f} KB")

    if todos_problemas:
        print("\nPROBLEMAS DETECTADOS:")
        for p in todos_problemas:
            print("  -", p)
        raise SystemExit(1)

    print(f"\n{len(seleccion)} notebooks generados en {RAIZ}")
    print("Ninguno fue ejecutado: todas las celdas están sin salidas.")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
