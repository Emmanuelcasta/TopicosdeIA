"""
Genera los cinco notebooks del proyecto a partir de las definiciones en
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
from nb_comun import RAIZ, notebook

NOTEBOOKS = [
    ("S04_Lab_Fine_tuning_Qwen.ipynb", nb_qwen),
    ("S04_Lab_Fine_tuning_BERT.ipynb", nb_bert),
    ("S04_Lab_Fine_tuning_Qwen_BERT.ipynb", nb_pipeline),
    ("S04_Lab_Fine_tuning_FLAN_T5.ipynb", nb_flan_t5),
    ("S04_Comparacion_Arquitecturas.ipynb", nb_comparacion),
    ("S06_Lab_Harness_Tutor_Matematicas.ipynb", nb_harness),
    ("S07_Lab_RAG_Tutor_Matematicas.ipynb", nb_rag),
]


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


def main() -> None:
    for archivo, generador in [("math_tutor_dataset.jsonl", "dataset_fuente.py"),
                               ("eval_set_m2.jsonl", "eval_set_fuente.py")]:
        if not (RAIZ / "data" / archivo).exists():
            raise SystemExit(f"Falta data/{archivo}. Ejecuten antes: python {generador}")

    todos_problemas = []
    for nombre, modulo in NOTEBOOKS:
        celdas = modulo.construir()
        problemas = validar_celdas(celdas, nombre)
        todos_problemas.extend(problemas)

        nb = notebook(celdas, nombre)
        destino = RAIZ / nombre
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

    print(f"\n{len(NOTEBOOKS)} notebooks generados en {RAIZ}")
    print("Ninguno fue ejecutado: todas las celdas están sin salidas.")


if __name__ == "__main__":
    main()
