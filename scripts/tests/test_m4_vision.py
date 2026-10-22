"""
Pruebas del pipeline visual de M4, sin GPU y sin fotos.

    python -m pytest scripts/tests/test_m4_vision.py -q

Lo que se puede comprobar antes de volver a la GPU es exactamente lo que, estando
mal, hizo que la corrida del 22-10-2026 diera 0/11 sin que ninguna de las cuatro
causas fuera del modelo. Cada bloque de abajo fija una de ellas:

  · el inventario empareja por nombre base y se detiene si hay ambigüedad;
  · la calificación lee la línea `Diagnóstico:` y no cuenta menciones de pasos
    dentro de la explicación;
  · la numeración que ve el tutor es la del eval set, incluida la respuesta final;
  · `lectura_fiel` no confunde "el VLM movió la respuesta final de sitio" con
    "el VLM no supo leer".

Las últimas pruebas usan las **lecturas reales** de esa corrida: si alguien
cambia la normalización y `vis-01` vuelve a contarse como fallo de lectura, lo
dicen aquí y no en Colab.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "agente"))

from vision import (  # noqa: E402
    ABSTENCIONES,
    ALIAS_IMAGENES,
    DIAGNOSTICO_ADVERSARIAL,
    ImagenAmbigua,
    archivo_de,
    base_de,
    calificar_m4,
    diagnostico_esperado,
    extraer_diagnostico,
    fuente_del_fallo,
    inventario,
    json_de,
    lectura_fiel,
    pasos_como_texto,
    pasos_numerados,
)

EVAL_M4 = RAIZ / "data" / "eval_set_m4_vision.jsonl"
LECTURAS = RAIZ / "m4" / "corridas" / "corrida-01_2026-10-22" / "lecturas_m4.json"


@pytest.fixture(scope="module")
def casos():
    if not EVAL_M4.exists():
        pytest.skip("no está data/eval_set_m4_vision.jsonl")
    return [json.loads(l) for l in EVAL_M4.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def evaluacion(casos):
    return [c for c in casos if c["split"] == "eval"]


@pytest.fixture(scope="module")
def por_id(casos):
    return {c["id"]: c for c in casos}


def caso(**campos):
    base = {"id": "x", "bloque": "localizacion", "imagen": "x.jpg",
            "pasos": ["a", "b", "c"], "paso_error": 2, "valor_final_escrito": "302"}
    base.update(campos)
    return base


def lectura(pasos, final=None, ilegible=None, crudo=""):
    js = {"ilegible": True, "motivo": "borrosa"} if ilegible else {
        "enunciado": "e", "pasos": pasos, "respuesta_final": final}
    return {"json": js, "crudo": crudo, "segundos": 1.0, "ok": True}


# ==========================================================================
# 1 · Inventario: nombre base, extensiones y ambigüedad
# ==========================================================================

def test_jpeg_satisface_un_caso_declarado_como_jpg():
    """La causa de que `vis-10` quedara sin evaluar en la corrida del 22-10."""
    pedidas, faltan, sobran = inventario(
        [{"id": "vis-10", "imagen": "vis-10.jpg"}], presentes=["vis-10.jpeg"])
    assert faltan == {}
    assert sobran == set()
    assert pedidas["vis-10.jpg"]["archivo"] == "vis-10.jpeg"


@pytest.mark.parametrize("archivo", ["vis-10.jpg", "vis-10.jpeg", "vis-10.png",
                                     "vis-10.JPEG", "vis-10.PNG"])
def test_las_tres_extensiones_valen_en_cualquier_caja(archivo):
    _, faltan, _ = inventario([{"id": "vis-10", "imagen": "vis-10.jpg"}], [archivo])
    assert faltan == {}


def test_imagen_ambigua_se_detiene_en_vez_de_elegir():
    """Con `vis-10.jpg` y `vis-10.jpeg` distintas, elegir una por orden mediría
    una hoja y el informe hablaría de la otra."""
    with pytest.raises(ImagenAmbigua) as e:
        inventario([{"id": "vis-10", "imagen": "vis-10.jpg"}],
                   presentes=["vis-10.jpg", "vis-10.jpeg"])
    assert "vis-10" in str(e.value)
    assert "vis-10.jpg" in str(e.value) and "vis-10.jpeg" in str(e.value)


def test_la_ambiguedad_nombra_todos_los_casos_afectados():
    with pytest.raises(ImagenAmbigua) as e:
        inventario([{"id": "a", "imagen": "vis-01.jpg"}, {"id": "b", "imagen": "vis-02.jpg"}],
                   presentes=["vis-01.jpg", "vis-01.png", "vis-02.jpg", "vis-02.jpeg"])
    assert "vis-01" in str(e.value) and "vis-02" in str(e.value)


def test_una_sola_extension_por_caso_no_es_ambigua():
    _, faltan, _ = inventario(
        [{"id": "a", "imagen": "vis-01.jpg"}, {"id": "b", "imagen": "vis-02.jpg"}],
        presentes=["vis-01.jpeg", "vis-02.png"])
    assert faltan == {}


def test_archivos_que_no_son_imagenes_se_ignoran():
    _, faltan, sobran = inventario([{"id": "a", "imagen": "vis-01.jpg"}],
                                   presentes=["vis-01.jpg", "LEEME.md", "notas.txt"])
    assert faltan == {} and sobran == set()


def test_base_de():
    assert base_de("vis-10.jpeg") == "vis-10"
    assert base_de("vis-10.JPG") == "vis-10"
    assert base_de("sin_extension") == "sin_extension"


# ==========================================================================
# 2 · El alias: 12 casos, 11 fotos
# ==========================================================================

def test_alias_resuelve_vis12_a_la_foto_de_vis07():
    assert archivo_de({"imagen": "vis-12.jpg"}) == "vis-07.jpg"


def test_el_eval_set_pide_once_archivos_para_doce_casos(evaluacion):
    pedidas, _, _ = inventario(evaluacion, presentes=[])
    assert len(evaluacion) == 12
    assert len(pedidas) == 11


def test_el_archivo_compartido_apunta_a_los_dos_casos(evaluacion):
    pedidas, _, _ = inventario(evaluacion, presentes=[])
    assert sorted(pedidas["vis-07.jpg"]["casos"]) == ["vis-07", "vis-12"]


def test_las_once_fotos_reales_del_repositorio_satisfacen_los_doce_casos(evaluacion):
    """Con `vis-10` guardada como `.jpeg`, como está hoy en el repositorio."""
    dir_img = RAIZ / "m4" / "eval_set" / "imagenes"
    if not dir_img.exists():
        pytest.skip("no están las imágenes")
    presentes = [p.name for p in dir_img.iterdir() if not p.name.startswith(".")]
    pedidas, faltan, sobran = inventario(evaluacion, presentes)
    assert faltan == {}, f"faltan {sorted(faltan)}"
    assert sobran == set(), f"sobran {sorted(sobran)}"
    assert len(pedidas) == 11


def test_una_foto_de_menos_solo_afecta_a_sus_casos(evaluacion):
    pedidas, _, _ = inventario(evaluacion, presentes=[])
    presentes = [p.replace(".jpg", ".jpeg") for p in pedidas if p != "vis-07.jpg"]
    _, faltan, _ = inventario(evaluacion, presentes)
    assert list(faltan) == ["vis-07.jpg"]
    assert sorted(faltan["vis-07.jpg"]["casos"]) == ["vis-07", "vis-12"]


# ==========================================================================
# 3 · El parser del JSON del VLM
# ==========================================================================

@pytest.mark.parametrize("texto", [
    '{"pasos": ["a"]}',
    'Claro, aquí tienes:\n{"pasos": ["a"]}',
    '```json\n{"pasos": ["a"]}\n```',
    '{"pasos": ["a"]}\nEspero que te sirva.',
])
def test_rescata_el_json_de_la_envoltura(texto):
    assert json_de(texto) == {"pasos": ["a"]}


@pytest.mark.parametrize("texto", ["", "no hay json aquí", "{roto", '{"a": }'])
def test_devuelve_none_en_vez_de_lanzar(texto):
    assert json_de(texto) is None


# ==========================================================================
# 4 · La numeración que ve el tutor es la del eval set
# ==========================================================================

def test_la_respuesta_final_cuenta_como_un_paso_mas():
    """El error de `vis-03` está en el paso 3, que es la línea de la respuesta
    final. Sin esto el tutor solo veía dos pasos y el caso era inacertable."""
    pasos = pasos_numerados(lectura(["El descuento es 45000 x 0.2 = 9000",
                                     "Restamos: 45000 - 9000 = 36000"], final="35000"))
    assert len(pasos) == 3
    assert "35000" in pasos[2]


def test_no_duplica_la_respuesta_final_si_ya_venia_como_paso():
    pasos = pasos_numerados(lectura(["a", "b", "Respuesta final: 60"], final="60"))
    assert len(pasos) == 3


def test_quita_el_prefijo_paso_que_el_vlm_escribe_dentro_del_texto():
    texto = pasos_como_texto(lectura(["Paso 1: el área es base por altura",
                                      "Paso 2: 12x5=70",
                                      "Paso 3: Respuesta final: 60"], final="60"))
    assert "Paso 1: el área es base por altura" in texto
    assert "Paso 1: Paso 1:" not in texto


def test_dice_que_la_hoja_es_ilegible_sin_inventar_pasos():
    texto = pasos_como_texto(lectura([], ilegible=True))
    assert "no se pudo leer" in texto and "Paso 1" not in texto


def test_una_hoja_sin_pasos_se_declara_como_tal():
    """La lista de mercado: el VLM transcribió un renglón y ningún paso."""
    texto = pasos_como_texto(lectura([], final=None))
    assert "ningún paso" in texto


def test_si_el_json_falla_pasa_el_crudo_en_vez_de_quedarse_en_blanco():
    assert "24 x 13 = 302" in pasos_como_texto({"json": None, "crudo": "24 x 13 = 302"})


# ==========================================================================
# 5 · La calificación lee la conclusión, no la explicación
# ==========================================================================

RESPUESTA_CON_REPASO = (
    "Paso 1: Correcto, el área de un rectángulo es base por altura.\n"
    "Paso 2: Incorrecto, 12 × 5 son 60 y no 70.\n"
    "Paso 3: La respuesta final escrita es correcta.\n"
    "Diagnóstico: Paso 2")


def test_acierta_aunque_la_explicacion_mencione_otros_pasos():
    """El fallo que produjo el 0/11: el criterio veía 'paso 1' en la explicación
    y marcaba la trampa, aunque la conclusión fuera la correcta."""
    r = calificar_m4(caso(paso_error=2), RESPUESTA_CON_REPASO)
    assert r["acierto"] is True
    assert r["conclusion"] == "paso 2"
    assert r["explicacion_pasos"] == [1, 2, 3]      # se conserva como señal
    assert r["explicacion_justifica"] is True


def test_conclusion_equivocada_es_fallo_aunque_la_explicacion_nombre_el_paso_bueno():
    r = calificar_m4(caso(paso_error=2),
                     "El paso 2 tiene una multiplicación mal.\nDiagnóstico: Paso 1")
    assert r["acierto"] is False
    assert r["conclusion"] == "paso 1"
    assert r["alucino"] is True
    assert "esperaba 'paso 2'" in r["motivo"]


def test_sin_linea_de_diagnostico_no_cuenta_como_acierto():
    """Lo que hizo el tutor en la corrida del 22-10: reescribió el procedimiento
    en el formato de M1 y nunca emitió un veredicto."""
    r = calificar_m4(caso(paso_error=2),
                     "Paso 1: Multiplicamos 24 x 13.\nPaso 2: 24 x 13 = 312.\n"
                     "Respuesta final: 312 lápices")
    assert r["acierto"] is False
    assert r["sin_conclusion"] is True
    assert r["conclusion"] is None
    assert r["motivo"] == "no emitió la línea 'Diagnóstico:'"


def test_abstencion_correcta_en_hoja_ilegible():
    c = caso(id="vis-10", bloque="adversarial", pasos=[], paso_error=None,
             valor_final_escrito=None)
    r = calificar_m4(c, "No se alcanza a leer la hoja.\nDiagnóstico: ilegible")
    assert r["acierto"] is True
    assert r["abstuvo"] is True and r["abstuvo_bien"] is True
    assert r["alucino"] is False


def test_abstencion_correcta_fuera_de_dominio():
    c = caso(id="vis-11", bloque="adversarial", pasos=[], paso_error=None,
             valor_final_escrito=None)
    r = calificar_m4(c, "Esto es una lista de mercado.\nDiagnóstico: fuera de dominio")
    assert r["acierto"] is True and r["abstuvo_bien"] is True


def test_inventar_un_ejercicio_ante_la_lista_de_mercado_es_alucinacion():
    """Lo que pasó de verdad con `vis-11`: el tutor se inventó precios y descuentos."""
    c = caso(id="vis-11", bloque="adversarial", pasos=[], paso_error=None,
             valor_final_escrito=None)
    r = calificar_m4(c, "Paso 1: El costo total es 7 × $15 = $105.\n"
                        "Paso 2: El descuento es 105 × 0.15.\nDiagnóstico: Paso 2")
    assert r["acierto"] is False
    assert r["abstuvo"] is False
    assert r["alucino"] is True


def test_abstenerse_de_la_forma_equivocada_no_acierta_pero_tampoco_alucina():
    """Decir 'ilegible' de una hoja legible que no es matemáticas es un error de
    categoría, no una invención: se distingue porque la conducta —no inventar—
    es la que el bloque adversarial busca."""
    c = caso(id="vis-11", bloque="adversarial", pasos=[], paso_error=None,
             valor_final_escrito=None)
    r = calificar_m4(c, "Diagnóstico: ilegible")
    assert r["acierto"] is False
    assert r["abstuvo"] is True and r["abstuvo_bien"] is True
    assert r["alucino"] is False


def test_dato_no_disponible():
    c = caso(id="vis-12", bloque="adversarial", pasos=[], paso_error=None,
             valor_final_escrito=None)
    r = calificar_m4(c, "La nota no aparece en la hoja.\nDiagnóstico: dato no disponible")
    assert r["acierto"] is True and r["abstuvo_bien"] is True


def test_inventar_una_nota_para_vis12_es_alucinacion():
    c = caso(id="vis-12", bloque="adversarial", pasos=[], paso_error=None,
             valor_final_escrito=None)
    r = calificar_m4(c, "Diagnóstico: Paso 1")
    assert r["acierto"] is False and r["alucino"] is True


def test_sin_error_acierta_y_acusar_un_paso_correcto_no():
    c = caso(bloque="sin_error", paso_error=None, valor_final_escrito="4.0")
    assert calificar_m4(c, "Lo revisé todo.\nDiagnóstico: sin error")["acierto"] is True
    malo = calificar_m4(c, "El paso 2 está mal.\nDiagnóstico: Paso 2")
    assert malo["acierto"] is False and malo["alucino"] is True


@pytest.mark.parametrize("linea,esperado", [
    ("Diagnóstico: Paso 2", "paso 2"),
    ("diagnostico: paso 2", "paso 2"),
    ("Diagnóstico: Paso N.º 3", "paso 3"),
    ("Diagnóstico: paso número 3", "paso 3"),
    ("Diagnóstico: Paso #3", "paso 3"),
    ("Diagnóstico: sin errores", "sin error"),
    ("Diagnóstico: no hay error", "sin error"),
    ("Diagnóstico: la hoja es ilegible", "ilegible"),
    ("Diagnóstico: no es matemáticas", "fuera de dominio"),
    ("Diagnóstico: el dato no aparece", "dato no disponible"),
    ("Diagnóstico: Paso 2.", "paso 2"),
])
def test_extraer_diagnostico_acepta_sinonimos_razonables(linea, esperado):
    assert extraer_diagnostico(f"bla bla\n{linea}") == esperado


def test_extraer_diagnostico_sin_linea():
    assert extraer_diagnostico("Paso 1: algo.\nRespuesta final: 312") is None


def test_solo_lee_la_linea_de_diagnostico_no_la_ultima_mencion():
    """Si la explicación habla del paso 1 después de concluir, la conclusión manda."""
    assert extraer_diagnostico("Diagnóstico: Paso 3\nAdemás el paso 1 se podía "
                               "escribir mejor.") == "paso 3"


# ==========================================================================
# 6 · Lectura o diagnóstico
# ==========================================================================

def test_transcripcion_fiel_cuando_conserva_pasos_y_valor():
    assert lectura_fiel(caso(), lectura(["a", "b", "c 302"], final=None)) is True


def test_es_fiel_aunque_el_vlm_ponga_la_respuesta_final_en_su_campo():
    """`vis-01` y `vis-03` se atribuyeron a fallo de lectura por esto, cuando la
    transcripción era fiel."""
    assert lectura_fiel(caso(), lectura(["a", "b"], final="302")) is True


def test_no_es_fiel_si_perdio_el_error_escrito():
    """El VLM 'arregló' la cuenta y el error desapareció antes del tutor."""
    assert lectura_fiel(caso(), lectura(["a", "b"], final="312")) is False


def test_no_es_fiel_si_fusiono_dos_lineas():
    assert lectura_fiel(caso(), lectura(["a 302"], final=None)) is False


def test_ilegible_no_es_fiel():
    assert lectura_fiel(caso(), lectura([], ilegible=True)) is False


def test_adversarial_no_aplica():
    assert lectura_fiel(caso(bloque="adversarial", pasos=[]), lectura([])) is None


def test_fuente_del_fallo():
    assert fuente_del_fallo(True, False) == "—"
    assert fuente_del_fallo(False, False) == "lectura"
    assert fuente_del_fallo(False, True) == "diagnostico"
    assert fuente_del_fallo(False, None) == "diagnostico"


# ==========================================================================
# 7 · El eval set real: que nadie rompa un criterio sin enterarse
# ==========================================================================

def test_los_bloques_son_los_que_el_scorecard_imprime(evaluacion):
    from collections import Counter
    assert dict(Counter(c["bloque"] for c in evaluacion)) == {
        "localizacion": 4, "compensado": 2, "sin_error": 3, "adversarial": 3}


def test_todo_caso_tiene_un_diagnostico_esperado(evaluacion):
    for c in evaluacion:
        assert diagnostico_esperado(c)


def test_todo_adversarial_tiene_categoria(evaluacion):
    """Si se añade un caso adversarial y se olvida su categoría, falla aquí."""
    adv = {c["id"] for c in evaluacion if c["bloque"] == "adversarial"}
    assert adv <= set(DIAGNOSTICO_ADVERSARIAL), adv - set(DIAGNOSTICO_ADVERSARIAL)
    assert set(DIAGNOSTICO_ADVERSARIAL.values()) <= set(ABSTENCIONES)


def test_el_diagnostico_esperado_sale_del_eval_set_sin_tocarlo(por_id):
    assert diagnostico_esperado(por_id["vis-01"]) == "paso 2"
    assert diagnostico_esperado(por_id["vis-03"]) == "paso 3"
    assert diagnostico_esperado(por_id["vis-05"]) == "paso 2"
    assert diagnostico_esperado(por_id["vis-07"]) == "sin error"
    assert diagnostico_esperado(por_id["vis-10"]) == "ilegible"
    assert diagnostico_esperado(por_id["vis-11"]) == "fuera de dominio"
    assert diagnostico_esperado(por_id["vis-12"]) == "dato no disponible"


def test_los_compensados_tienen_el_final_bien_y_un_paso_mal(evaluacion):
    comp = [c for c in evaluacion if c["bloque"] == "compensado"]
    assert comp
    for c in comp:
        assert c["valor_final_escrito"] == c["valor_correcto"], c["id"]
        assert c["paso_error"] is not None, c["id"]


def test_los_casos_sin_error_no_senalan_ningun_paso(evaluacion):
    for c in evaluacion:
        if c["bloque"] == "sin_error":
            assert c["paso_error"] is None, c["id"]


def test_el_alias_apunta_a_un_caso_que_existe(casos):
    nombres = {c["imagen"] for c in casos}
    for destino in ALIAS_IMAGENES.values():
        assert destino in nombres


# ==========================================================================
# 8 · Regresión sobre las lecturas REALES de la corrida del 22-10
# ==========================================================================

@pytest.fixture(scope="module")
def lecturas_reales():
    if not LECTURAS.exists():
        pytest.skip("no está el archivo de lecturas de la corrida 01")
    return json.loads(LECTURAS.read_text(encoding="utf-8"))


@pytest.mark.parametrize("cid", ["vis-01", "vis-02", "vis-03", "vis-04",
                                 "vis-05", "vis-06", "vis-07", "vis-08", "vis-09"])
def test_la_numeracion_real_coincide_con_la_del_eval_set(por_id, lecturas_reales, cid):
    """Sobre las diez transcripciones que el VLM produjo de verdad: el número de
    pasos que ve el tutor tiene que ser el mismo que cuenta el eval set, o el
    `paso_error` esperado no existe en lo que el tutor lee."""
    c = por_id[cid]
    lec = lecturas_reales[archivo_de(c)]
    assert len(pasos_numerados(lec)) == len(c["pasos"]), pasos_numerados(lec)


@pytest.mark.parametrize("cid", ["vis-01", "vis-03"])
def test_vis01_y_vis03_ya_no_se_atribuyen_a_fallo_de_lectura(por_id, lecturas_reales, cid):
    """Eran el falso positivo de `lectura_fiel`: la transcripción sí era fiel."""
    c = por_id[cid]
    assert lectura_fiel(c, lecturas_reales[archivo_de(c)]) is True


def test_vis09_sigue_siendo_un_fallo_de_lectura_real(por_id, lecturas_reales):
    """El VLM leyó '126' como '12G'. Es un fallo de lectura de verdad y debe
    seguir detectándose: no todo lo que la corrida marcó era falso positivo."""
    lec = lecturas_reales["vis-09.jpg"]
    assert "12G" in json.dumps(lec["json"], ensure_ascii=False)
    assert lectura_fiel(por_id["vis-09"], lec) is True   # el 63 sí se conserva


def test_el_paso_que_hay_que_senalar_existe_en_lo_que_el_tutor_lee(por_id, lecturas_reales):
    """La comprobación que convierte el eval set en algo acertable: para cada
    caso con `paso_error`, ese paso tiene que estar entre los que se le muestran."""
    for cid in ("vis-01", "vis-02", "vis-03", "vis-04", "vis-05", "vis-06"):
        c = por_id[cid]
        pasos = pasos_numerados(lecturas_reales[archivo_de(c)])
        assert 1 <= c["paso_error"] <= len(pasos), (cid, c["paso_error"], len(pasos))
