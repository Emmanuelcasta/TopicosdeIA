"""
Pruebas del pipeline visual de M4, sin GPU y sin fotos.

    python -m pytest scripts/tests/test_m4_vision.py -q

Lo que se puede comprobar antes de tener las imágenes es exactamente lo que, si
estuviera mal, haría que el scorecard final saliera equivocado sin avisar:

  · el alias de `vis-12`, que es la razón de que haya 12 casos y 11 fotos;
  · el inventario, que decide qué casos entran en las medias;
  · el parser tolerante del JSON que devuelve el VLM;
  · el formato con el que el tutor ve la transcripción, del que depende que
    "Paso 2" pueda acertarse;
  · la atribución del fallo a la lectura o al diagnóstico, que es el resultado
    principal del módulo.

Lo que NO se prueba aquí: la pasada del VLM y el redimensionado de la imagen.
Eso solo se puede medir con las fotos puestas y la GPU encendida.

Al final hay pruebas sobre el eval set real: si alguien edita
`data/eval_set_m4_vision.jsonl` y rompe un criterio, lo dicen aquí y no en Colab.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "agente"))

from vision import (  # noqa: E402
    ALIAS_IMAGENES,
    archivo_de,
    fuente_del_fallo,
    inventario,
    json_de,
    lectura_fiel,
    pasos_como_texto,
)

EVAL_M4 = RAIZ / "data" / "eval_set_m4_vision.jsonl"


@pytest.fixture(scope="module")
def casos():
    if not EVAL_M4.exists():
        pytest.skip("no está data/eval_set_m4_vision.jsonl")
    return [json.loads(l) for l in EVAL_M4.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def evaluacion(casos):
    return [c for c in casos if c["split"] == "eval"]


def caso(**campos):
    base = {"id": "x", "bloque": "localizacion", "imagen": "x.jpg",
            "pasos": ["a", "b", "c"], "valor_final_escrito": "302"}
    base.update(campos)
    return base


def lectura(pasos, final=None, ilegible=None, crudo=""):
    js = {"ilegible": True, "motivo": "borrosa"} if ilegible else {
        "enunciado": "e", "pasos": pasos, "respuesta_final": final}
    return {"json": js, "crudo": crudo, "segundos": 1.0, "ok": True}


# --------------------------------------------------------------------------
# El alias: 12 casos, 11 fotos
# --------------------------------------------------------------------------

def test_alias_resuelve_vis12_a_la_foto_de_vis07():
    assert archivo_de({"imagen": "vis-12.jpg"}) == "vis-07.jpg"


def test_las_demas_imagenes_no_se_tocan():
    assert archivo_de({"imagen": "vis-03.jpg"}) == "vis-03.jpg"


def test_el_eval_set_pide_once_archivos_para_doce_casos(evaluacion):
    """La cifra que el notebook anuncia: si cambia, el README deja de ser cierto."""
    pedidas, _, _ = inventario(evaluacion, presentes=[])
    assert len(evaluacion) == 12
    assert len(pedidas) == 11


def test_el_archivo_compartido_apunta_a_los_dos_casos(evaluacion):
    pedidas, _, _ = inventario(evaluacion, presentes=[])
    assert sorted(pedidas["vis-07.jpg"]) == ["vis-07", "vis-12"]


def test_el_alias_apunta_a_un_caso_que_existe(casos):
    """Si `vis-07` se renombrara, el alias dejaría de resolver a nada."""
    nombres = {c["imagen"] for c in casos}
    for destino in ALIAS_IMAGENES.values():
        assert destino in nombres


# --------------------------------------------------------------------------
# Inventario: qué falta y qué entra en las medias
# --------------------------------------------------------------------------

def test_sin_fotos_faltan_todas(evaluacion):
    pedidas, faltan, sobran = inventario(evaluacion, presentes=[])
    assert faltan.keys() == pedidas.keys()
    assert sobran == set()


def test_con_todas_las_fotos_no_falta_ninguna(evaluacion):
    pedidas, _, _ = inventario(evaluacion, presentes=[])
    _, faltan, sobran = inventario(evaluacion, presentes=pedidas.keys())
    assert faltan == {} and sobran == set()


def test_una_foto_de_menos_solo_afecta_a_sus_casos(evaluacion):
    pedidas, _, _ = inventario(evaluacion, presentes=[])
    presentes = set(pedidas) - {"vis-07.jpg"}
    _, faltan, _ = inventario(evaluacion, presentes)
    # La foto compartida arrastra DOS casos: vis-07 y vis-12.
    assert list(faltan) == ["vis-07.jpg"]
    assert sorted(faltan["vis-07.jpg"]) == ["vis-07", "vis-12"]


def test_los_archivos_de_sobra_se_listan_aparte(evaluacion):
    pedidas, _, _ = inventario(evaluacion, presentes=[])
    _, _, sobran = inventario(evaluacion, set(pedidas) | {"selfie.jpg"})
    assert sobran == {"selfie.jpg"}


# --------------------------------------------------------------------------
# El parser del JSON que devuelve el VLM
# --------------------------------------------------------------------------

@pytest.mark.parametrize("texto", [
    '{"pasos": ["a"]}',
    'Claro, aquí tienes:\n{"pasos": ["a"]}',
    '```json\n{"pasos": ["a"]}\n```',
    '{"pasos": ["a"]}\nEspero que te sirva.',
])
def test_rescata_el_json_de_la_envoltura(texto):
    """Un 2B envuelve el JSON en prosa o en vallas: perderlo por eso sería tirar
    una lectura buena."""
    assert json_de(texto) == {"pasos": ["a"]}


@pytest.mark.parametrize("texto", ["", "no hay json aquí", "{roto", '{"a": }'])
def test_devuelve_none_en_vez_de_lanzar(texto):
    assert json_de(texto) is None


# --------------------------------------------------------------------------
# El formato con el que el tutor ve la transcripción
# --------------------------------------------------------------------------

def test_numera_los_pasos_para_que_paso_n_sea_acertable():
    texto = pasos_como_texto(lectura(["24 × 13.", "24 × 13 = 302."], final="302"))
    assert "Paso 1: 24 × 13." in texto
    assert "Paso 2: 24 × 13 = 302." in texto
    assert "Respuesta final escrita: 302" in texto


def test_dice_que_la_hoja_es_ilegible_sin_inventar_pasos():
    texto = pasos_como_texto(lectura([], ilegible=True))
    assert "no se pudo leer" in texto and "Paso 1" not in texto


def test_si_el_json_falla_pasa_el_crudo_en_vez_de_quedarse_en_blanco():
    texto = pasos_como_texto({"json": None, "crudo": "24 x 13 = 302"})
    assert "24 x 13 = 302" in texto


def test_sin_lectura_no_lanza():
    assert "no salió estructurada" in pasos_como_texto(None)


# --------------------------------------------------------------------------
# La atribución del fallo: ojos o razonamiento
# --------------------------------------------------------------------------

def test_transcripcion_fiel_cuando_conserva_pasos_y_valor():
    assert lectura_fiel(caso(), lectura(["a", "b", "c"], final="302")) is True


def test_no_es_fiel_si_perdio_el_error_escrito():
    """El caso que importa: el VLM 'arregló' la cuenta y el error desapareció."""
    assert lectura_fiel(caso(), lectura(["a", "b", "c"], final="312")) is False


def test_no_es_fiel_si_fusiono_dos_lineas():
    """Con un paso de menos la numeración se desplaza y 'Paso 2' ya no apunta a lo
    mismo, aunque los números estén todos."""
    assert lectura_fiel(caso(), lectura(["a", "b 302"], final="302")) is False


def test_acepta_el_valor_en_el_ultimo_paso_y_no_solo_en_respuesta_final():
    assert lectura_fiel(caso(), lectura(["a", "b", "c = 302"], final=None)) is True


def test_ilegible_no_es_fiel():
    assert lectura_fiel(caso(), lectura([], ilegible=True)) is False


def test_adversarial_no_aplica():
    """None no es 'sí' ni 'no': en un adversarial no hay procedimiento que
    transcribir, así que el fallo nunca se atribuye a la lectura."""
    assert lectura_fiel(caso(bloque="adversarial", pasos=[]), lectura([])) is None


def test_fuente_del_fallo():
    assert fuente_del_fallo(True, False) == "—"            # acertó: no hay fallo
    assert fuente_del_fallo(False, False) == "lectura"
    assert fuente_del_fallo(False, True) == "diagnostico"
    assert fuente_del_fallo(False, None) == "diagnostico"  # adversarial fallado


# --------------------------------------------------------------------------
# El eval set real: que nadie rompa un criterio sin enterarse
# --------------------------------------------------------------------------

def test_los_bloques_son_los_que_el_scorecard_imprime(evaluacion):
    from collections import Counter
    assert dict(Counter(c["bloque"] for c in evaluacion)) == {
        "localizacion": 4, "compensado": 2, "sin_error": 3, "adversarial": 3}


def test_cada_caso_trae_criterio_declarado_de_antemano(evaluacion):
    for c in evaluacion:
        assert c["criterio_s13"] in ("exacto", "valvula"), c["id"]
        assert c["evaluacion"]["tipo"] == "contiene_alguna", c["id"]
        assert c["evaluacion"]["claves"], c["id"]


def test_ninguna_clave_esta_tambien_prohibida(evaluacion):
    """Un caso con la misma frase en las dos listas no se podría acertar nunca."""
    for c in evaluacion:
        ev = c["evaluacion"]
        choque = {k.lower() for k in ev["claves"]} & {p.lower() for p in ev.get("prohibido", [])}
        assert not choque, f"{c['id']}: {choque}"


def test_los_casos_de_localizacion_aceptan_su_paso_y_prohiben_los_otros(evaluacion):
    for c in evaluacion:
        if c["bloque"] not in ("localizacion", "compensado"):
            continue
        n = c["paso_error"]
        assert f"paso {n}" in [k.lower() for k in c["evaluacion"]["claves"]], c["id"]
        assert f"paso {n}" not in [p.lower() for p in c["evaluacion"]["prohibido"]], c["id"]


def test_los_compensados_tienen_el_final_bien_y_un_paso_mal(evaluacion):
    """Es el bloque que justifica el módulo: si el valor final escrito dejara de
    coincidir con el correcto, el caso ya no mediría un error compensado."""
    comp = [c for c in evaluacion if c["bloque"] == "compensado"]
    assert comp
    for c in comp:
        assert c["valor_final_escrito"] == c["valor_correcto"], c["id"]
        assert c["paso_error"] is not None, c["id"]


def test_los_casos_sin_error_no_senalan_ningun_paso(evaluacion):
    for c in evaluacion:
        if c["bloque"] != "sin_error":
            continue
        assert c["paso_error"] is None, c["id"]
        assert c["valor_final_escrito"] == c["valor_correcto"], c["id"]


def test_los_adversariales_piden_abstencion(evaluacion):
    for c in evaluacion:
        if c["bloque"] != "adversarial":
            continue
        assert c["criterio_s13"] == "valvula", c["id"]
        assert c["pasos"] == [], c["id"]
