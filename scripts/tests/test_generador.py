"""
La guarda que impide regenerar encima de un notebook ejecutado.

    python -m pytest scripts/tests/test_generador.py -q

Por qué existe: dos veces se regeneró un notebook que tenía salidas y se perdió
la evidencia de una corrida. Las dos con el mismo mecanismo —un filtro que
coincidía con más archivos de los previstos— y la segunda sin haberlo archivado
antes. Un `.ipynb` con salidas no es un archivo generado: es un resultado.

Estas pruebas NO regeneran ningún notebook del proyecto: trabajan sobre archivos
temporales.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts"))

from generar_notebooks import revisar_destinos, tiene_salidas  # noqa: E402


def _nb(tmp_path, nombre, *, salidas=False, exec_count=None):
    celda = {"cell_type": "code", "source": ["print(1)\n"], "metadata": {},
             "execution_count": exec_count,
             "outputs": ([{"output_type": "stream", "name": "stdout", "text": ["1\n"]}]
                         if salidas else [])}
    ruta = tmp_path / nombre
    ruta.write_text(json.dumps({"cells": [celda], "metadata": {},
                                "nbformat": 4, "nbformat_minor": 5}),
                    encoding="utf-8")
    return ruta


def test_un_notebook_limpio_se_puede_sobrescribir(tmp_path):
    assert tiene_salidas(_nb(tmp_path, "limpio.ipynb")) is False


def test_un_notebook_con_salidas_queda_protegido(tmp_path):
    assert tiene_salidas(_nb(tmp_path, "ejecutado.ipynb", salidas=True)) is True


def test_el_execution_count_solo_tambien_protege(tmp_path):
    """El notebook de la corrida de desarrollo llegó sin `outputs` en algunas
    celdas pero con `execution_count`: cualquiera de las dos marcas basta."""
    assert tiene_salidas(_nb(tmp_path, "contado.ipynb", exec_count=7)) is True


def test_un_destino_que_no_existe_no_estorba(tmp_path):
    assert tiene_salidas(tmp_path / "todavia-no.ipynb") is False


def test_un_archivo_ilegible_no_se_declara_protegido(tmp_path):
    """Si no se puede leer, no se afirma nada sobre sus salidas."""
    roto = tmp_path / "roto.ipynb"
    roto.write_text("{esto no es json", encoding="utf-8")
    assert tiene_salidas(roto) is False


def test_revisar_destinos_devuelve_solo_los_protegidos(tmp_path):
    limpio = _nb(tmp_path, "a.ipynb")
    ejecutado = _nb(tmp_path, "b.ipynb", salidas=True)
    nuevo = tmp_path / "c.ipynb"
    assert revisar_destinos([limpio, ejecutado, nuevo]) == [ejecutado]


def test_sin_destinos_protegidos_la_lista_esta_vacia(tmp_path):
    assert revisar_destinos([_nb(tmp_path, "a.ipynb")]) == []


def test_el_caso_real_que_costo_la_evidencia(tmp_path):
    """El filtro `S13` coincidía con dos notebooks: el pipeline (limpio) y el de
    fine-tuning (ejecutado). La guarda tiene que detener el lote ENTERO, no
    regenerar el limpio y quejarse del otro después."""
    pipeline = _nb(tmp_path, "S13_M4_Pipeline_Visual.ipynb")
    finetuning = _nb(tmp_path, "S13_M4_FineTuning_Diagnostico.ipynb", salidas=True)
    protegidos = revisar_destinos([pipeline, finetuning])
    assert protegidos == [finetuning]
    assert pipeline.exists(), "el limpio sigue ahí, sin tocar"
