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
    assert "NO se pudo leer" in texto and "Paso 1" not in texto


def test_una_hoja_sin_pasos_se_declara_como_tal():
    """La lista de mercado: el VLM transcribió un renglón y ningún paso."""
    texto = pasos_como_texto(lectura([], final=None))
    assert "ningún paso" in texto


def test_una_transcripcion_sin_estructura_no_llega_al_tutor():
    """Cambio deliberado de comportamiento. Antes el texto crudo se le pasaba al
    tutor "por si acaso"; ahora una transcripción que no salió estructurada se
    trata como lectura no fiable y el crudo NO se le muestra.

    El motivo es el de `vis-10`: si el VLM no pudo leer la hoja, lo que produjo no
    es la hoja, y dárselo al tutor hace que juzgue un invento como si fuera el
    procedimiento del estudiante."""
    texto = pasos_como_texto({"ok": False, "json": None, "crudo": "24 x 13 = 302"})
    assert "NO se pudo leer" in texto
    assert "24 x 13 = 302" not in texto


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


# ==========================================================================
# 9 · Los fallos de la corrida del 22-10 por la tarde (exec 71–105)
#
# Cuatro problemas distintos, cada uno con su prueba. Ninguna de estas reglas se
# ajustó a un caso concreto: la de la hoja sin pasos es estructural, la de la
# lectura inventada compara el modelo consigo mismo, y la de la conclusión
# inválida separa dos tipos de fallo que antes se sumaban.
# ==========================================================================

from vision import (  # noqa: E402
    PETICION_CONCLUSION,
    RECORDATORIO_FORMATO,
    asegurar_diagnostico,
    diagnostico_valido,
    huella_lectura,
    lectura_no_fiable,
    lecturas_consistentes,
    numeros_de,
    prompt_revision,
)


# --- 9.1 · una hoja sin procedimiento no se convierte en un paso -----------

def test_una_hoja_sin_pasos_no_fabrica_un_paso():
    """`vis-11`: el VLM devolvió `pasos: []` y `respuesta_final: "Arroz"`, y el
    render fabricaba "Paso 1: Respuesta final: Arroz". Al tutor se le presentaba
    un procedimiento de un paso donde la hoja no tenía ninguno."""
    lec = lectura([], final="Arroz")
    assert pasos_numerados(lec) == []
    texto = pasos_como_texto(lec)
    assert "Paso 1" not in texto
    assert "ningún paso" in texto


def test_con_pasos_la_respuesta_final_si_se_numera():
    """La regla anterior no debe romper el caso normal (`vis-01`, `vis-03`)."""
    assert len(pasos_numerados(lectura(["a", "b"], final="302"))) == 3


# --- 9.2 · la lectura inventada de una hoja ilegible ----------------------

def test_dos_lecturas_iguales_son_consistentes():
    a = lectura(["24 x 13", "24 x 13 = 302"], final="302 lápices")
    b = lectura(["Multiplicamos: 24 x 13", "24 x 13 = 302"], final="302 lapices")
    assert lecturas_consistentes(a, b), "la redacción puede variar; los números no"


def test_dos_lecturas_con_numeros_distintos_no_son_consistentes():
    """El modo de fallo de `vis-10`: el VLM inventó `306` donde la hoja dice `302`.
    Una invención no se repite igual al volver a muestrear."""
    a = lectura(["24 x 13 = 302"], final="302")
    b = lectura(["24 x 13 = 306"], final="306")
    assert not lecturas_consistentes(a, b)


def test_dos_lecturas_con_distinto_numero_de_pasos_no_son_consistentes():
    assert not lecturas_consistentes(lectura(["a 1", "b 2"]), lectura(["a 1"]))


def test_la_lectura_inconsistente_llega_al_tutor_como_ilegible():
    lec = lectura(["24 x 13 = 306"], final="306")
    lec["consistente"] = False
    no_fiable, motivo = lectura_no_fiable(lec)
    assert no_fiable and "no coinciden" in motivo
    texto = pasos_como_texto(lec)
    assert "NO se pudo leer" in texto
    assert "306" not in texto, "el invento del VLM no debe llegar al tutor"


def test_la_lectura_consistente_si_se_le_muestra():
    lec = lectura(["24 x 13 = 302"], final="302")
    lec["consistente"] = True
    assert lectura_no_fiable(lec)[0] is False
    assert "Paso 1: 24 x 13 = 302" in pasos_como_texto(lec)


def test_una_lectura_no_fiable_se_atribuye_a_la_lectura_no_al_tutor():
    """`vis-10` se contaba en `diagnostico` cuando el fallo fue de ver."""
    c = caso(id="vis-10", bloque="adversarial", pasos=[], paso_error=None,
             valor_final_escrito=None)
    lec = lectura(["24 x 13 = 306"], final="306")
    lec["consistente"] = False
    assert lectura_fiel(c, lec) is False
    assert fuente_del_fallo(False, lectura_fiel(c, lec)) == "lectura"


def test_una_lectura_no_fiable_no_acredita_nada():
    """No se marca como correcto sin evidencia: sigue siendo fallo si no concluye
    lo que el caso espera."""
    c = caso(id="vis-10", bloque="adversarial", pasos=[], paso_error=None,
             valor_final_escrito=None)
    r = calificar_m4(c, "Paso 1: parece correcto.\nDiagnóstico: sin error")
    assert r["acierto"] is False


def test_el_tutor_puede_acertar_ilegible_cuando_se_le_dice_que_no_se_leyo():
    """El final deseable del camino: con la salvaguarda activa, concluir
    'ilegible' es alcanzable, y por eso el caso vuelve a medir algo."""
    c = caso(id="vis-10", bloque="adversarial", pasos=[], paso_error=None,
             valor_final_escrito=None)
    r = calificar_m4(c, "La hoja no se puede leer.\nDiagnóstico: ilegible")
    assert r["acierto"] is True and r["abstuvo_bien"] is True


def test_huella_de_una_ilegible_declarada():
    assert huella_lectura(lectura([], ilegible=True)) == ("ilegible",)


def test_numeros_de():
    assert numeros_de("24 x 13 = 302 y 3/4") == ["24", "13", "302", "3/4"]
    assert numeros_de("2,35 pesos") == ["2.35"]


# --- 9.3 · la conclusión en texto libre no es una alucinación -------------

def test_conclusion_en_texto_libre_se_cuenta_aparte():
    """`vis-04` emitió "Diagnóstico: equivoco el producto 1248…": señalaba el
    error sin nombrar el paso. No es una conclusión válida, pero tampoco una
    invención, y sumarla a las alucinaciones exageraba el fallo."""
    r = calificar_m4(caso(paso_error=2),
                     "Revisé la división.\nDiagnóstico: equivoco el producto 1248 entre 8")
    assert r["acierto"] is False
    assert r["conclusion_invalida"] is True
    assert r["alucino"] is False, "no inventó un veredicto: lo escribió mal"
    assert r["sin_conclusion"] is False
    assert "no válida" in r["motivo"]


def test_una_conclusion_valida_y_equivocada_si_es_alucinacion():
    r = calificar_m4(caso(paso_error=2), "Diagnóstico: Paso 1")
    assert r["alucino"] is True and r["conclusion_invalida"] is False


@pytest.mark.parametrize("c,ok", [
    ("paso 2", True), ("sin error", True), ("ilegible", True),
    ("fuera de dominio", True), ("dato no disponible", True),
    ("equivoco el producto", False), (None, False), ("", False),
])
def test_diagnostico_valido(c, ok):
    assert diagnostico_valido(c) is ok


# --- 9.4 · el prompt y el reintento --------------------------------------

def test_la_exigencia_del_formato_va_al_final_del_prompt():
    """La causa de que 8 de 11 respuestas terminaran copiando 'Respuesta final':
    era lo último que el modelo leía."""
    pr = prompt_revision("¿en qué paso me equivoqué?",
                         lectura(["24 x 13 = 302"], final="302"))
    assert pr.rstrip().endswith(RECORDATORIO_FORMATO.rstrip())
    assert pr.index("Diagnóstico: sin error") > pr.index("Paso 1:")


def test_el_prompt_prohibe_la_etiqueta_que_descarrilaba_al_modelo():
    assert "NO escribas 'Respuesta final'" in prompt_revision("x", lectura(["a"]))


def test_el_prompt_no_trae_ningun_ejercicio_resuelto(evaluacion):
    """Cambio deliberado tras la corrida del 24-10. Antes el prompt llevaba un
    ejemplo resuelto (una panadería) y `vis-11` lo copió entero, con su
    `Diagnóstico: Paso 2`, en lugar de responder sobre la lista de mercado.

    Ahora del ejemplo queda solo la FORMA de la última línea: enseñar el formato
    no exige enseñar un ejercicio, y un ejercicio en el prompt es una respuesta
    que el modelo puede copiar."""
    pr = prompt_revision("x", lectura(["a"]))
    assert "panader" not in pr.lower()
    # Antes de "Ahora la hoja de verdad" no puede haber un procedimiento.
    preambulo = pr.split("Ahora la hoja de verdad")[0]
    assert preambulo.count("Paso 1:") == 0, "hay un ejercicio resuelto en el preámbulo"
    for c in evaluacion:
        if c["enunciado"] and not c["enunciado"].startswith("("):
            assert c["enunciado"] not in pr, c["id"]


def test_sin_reintento_si_la_conclusion_ya_es_valida():
    llamadas = []
    r, reintento = asegurar_diagnostico(
        "Revisé todo.\nDiagnóstico: Paso 2", lambda p: llamadas.append(p) or "x")
    assert reintento is False and llamadas == []
    assert r.endswith("Diagnóstico: Paso 2")


def test_el_reintento_conserva_la_explicacion_y_añade_la_linea():
    explicacion = "Paso 1: correcto. Paso 2: la multiplicación está mal."
    r, reintento = asegurar_diagnostico(explicacion, lambda p: "Diagnóstico: Paso 2")
    assert reintento is True
    assert explicacion in r, "la explicación original no se pierde"
    assert extraer_diagnostico(r) == "paso 2"


def test_el_reintento_recibe_la_propia_explicacion_del_modelo():
    """No se le deja recalcular: se le pide resumir lo que ya dijo."""
    vistos = []
    asegurar_diagnostico("El paso 3 está mal.",
                         lambda p: vistos.append(p) or "Diagnóstico: Paso 3")
    assert "El paso 3 está mal." in vistos[0]
    assert "sin ninguna explicación" in vistos[0]


def test_el_reintento_se_pide_una_sola_vez():
    """Si el reintento tampoco concluye, no se insiste: queda sin conclusión y se
    reporta, que es más honesto que seguir preguntando hasta acertar."""
    llamadas = []
    r, reintento = asegurar_diagnostico(
        "Paso 1: bien.", lambda p: llamadas.append(p) or "no sé qué decir")
    assert reintento is True and len(llamadas) == 1
    assert diagnostico_valido(extraer_diagnostico(r)) is False


def test_la_peticion_del_reintento_ofrece_las_cinco_categorias():
    for etiqueta in ("Paso N", "sin error", "ilegible", "fuera de dominio",
                     "dato no disponible"):
        assert etiqueta in PETICION_CONCLUSION


# ==========================================================================
# 10 · Los fallos de la corrida del 24-10 (Tesla T4, exec 5–37)
#
# Dos: la puerta `dev-*` no detenía nada, y la consistencia entre dos
# generaciones no detecta una invención estable.
# ==========================================================================

from vision import (  # noqa: E402
    PuertaNoSuperada,
    atipicos_por_mad,
    exigir_puerta_dev,
    revisar_puerta_dev,
)


# --- 10.1 · la puerta detiene de verdad ----------------------------------

def test_la_puerta_se_supera_con_tres_conclusiones_validas():
    pu = revisar_puerta_dev([("dev-01", "paso 2"), ("dev-02", "sin error"),
                             ("dev-03", "paso 2")])
    assert pu["ok"] is True and pu["validos"] == 3
    exigir_puerta_dev(pu)          # no lanza


def test_la_puerta_no_exige_acertar_solo_el_formato():
    """`dev-01` espera 'paso 2'; concluir 'paso 1' es un fallo de criterio, no de
    formato, y no debe bloquear la evaluación."""
    pu = revisar_puerta_dev([("dev-01", "paso 1"), ("dev-02", "sin error"),
                             ("dev-03", "ilegible")])
    assert pu["ok"] is True
    exigir_puerta_dev(pu)


def test_cero_de_tres_detiene():
    """El caso real del 24-10: 0/3 y la ejecución siguió igualmente."""
    pu = revisar_puerta_dev([("dev-01", None), ("dev-02", None), ("dev-03", None)])
    assert pu["ok"] is False and pu["validos"] == 0
    with pytest.raises(PuertaNoSuperada) as e:
        exigir_puerta_dev(pu)
    assert "0/3" in str(e.value)
    assert "dev-01" in str(e.value)


def test_una_sola_conclusion_invalida_tambien_detiene():
    pu = revisar_puerta_dev([("dev-01", "paso 2"), ("dev-02", "sin error"),
                             ("dev-03", "equivoco el producto")])
    assert pu["ok"] is False
    with pytest.raises(PuertaNoSuperada):
        exigir_puerta_dev(pu)


def test_no_haber_ejecutado_el_ensayo_tambien_detiene():
    """Ejecutar la celda de los 12 casos directamente no debe colarse."""
    with pytest.raises(PuertaNoSuperada) as e:
        exigir_puerta_dev(None)
    assert "no se ha ejecutado" in str(e.value)


def test_la_salida_deliberada_existe_y_queda_registrada():
    pu = revisar_puerta_dev([("dev-01", None)])
    exigir_puerta_dev(pu, permitir_sin_puerta=True)      # no lanza
    assert pu["ok"] is False, "el estado de la puerta no se falsea al permitirla"


def test_la_celda_de_evaluacion_no_se_ejecuta_si_la_puerta_falla():
    """La prueba que pedía el encargo: se ejecuta **la celda real** del notebook
    con la puerta en 0/3 y se comprueba que no califica ni un solo caso.

    Se mira que no quede `M4` definido: si la celda hubiera corrido, existiría.
    """
    nb = RAIZ / "m4" / "pipeline" / "S13_M4_Pipeline_Visual.ipynb"
    if not nb.exists():
        pytest.skip("no está el notebook generado")
    celdas = [("".join(c["source"])) for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]
              if c["cell_type"] == "code"]
    fuente = next(s for s in celdas if "filas = []" in s and "calificar_m4" in s)

    llamadas = []
    entorno = {
        "exigir_puerta_dev": exigir_puerta_dev,
        "PUERTA_DEV": revisar_puerta_dev([("dev-01", None), ("dev-02", None),
                                          ("dev-03", None)]),
        # Si la celda llegara a correr, estas se usarían y lo sabríamos.
        "eval_m4": [{"id": "vis-01"}],
        "sistema_multimodal": lambda *a, **k: llamadas.append(a) or "",
    }
    with pytest.raises(PuertaNoSuperada):
        exec(fuente, entorno)
    assert llamadas == [], "no se llamó al tutor ni una vez"
    assert "M4" not in entorno, "la celda no llegó a construir el scorecard"


def test_la_celda_de_evaluacion_pasa_la_puerta_cuando_se_supera():
    """La otra mitad: con la puerta superada, la celda sí arranca (y falla más
    adelante por las variables que este entorno mínimo no tiene, lo que prueba
    que pasó de la comprobación)."""
    nb = RAIZ / "m4" / "pipeline" / "S13_M4_Pipeline_Visual.ipynb"
    if not nb.exists():
        pytest.skip("no está el notebook generado")
    celdas = [("".join(c["source"])) for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]
              if c["cell_type"] == "code"]
    fuente = next(s for s in celdas if "filas = []" in s and "calificar_m4" in s)
    entorno = {"exigir_puerta_dev": exigir_puerta_dev,
               "PUERTA_DEV": revisar_puerta_dev([("dev-01", "paso 2")])}
    with pytest.raises(Exception) as e:
        exec(fuente, entorno)
    assert not isinstance(e.value, PuertaNoSuperada)


# --- 10.2 · la invención estable que la consistencia no detecta ----------

def test_la_consistencia_no_detecta_una_invencion_estable():
    """El fallo del 24-10, fijado como prueba: las dos lecturas de `vis-10`
    coincidieron y las dos inventaron `306`. Esta señal, sola, no sirve."""
    inventada = lectura(["Multiplicamos: 24 x 13 = 306"], final="306 piezas")
    assert lecturas_consistentes(inventada, inventada) is True
    inventada["consistente"] = True
    assert lectura_no_fiable(inventada)[0] is False, (
        "con solo esta señal la lectura inventada pasa: por eso hacen falta más")


def test_la_pregunta_de_legibilidad_aparte_si_la_detecta():
    """Señal 3: preguntado sin transcribir, el modelo dice que no se lee."""
    lec = lectura(["24 x 13 = 306"], final="306")
    lec.update(consistente=True, legible_declarada=False,
               motivo_legibilidad="la foto está desenfocada")
    no_fiable, motivo = lectura_no_fiable(lec)
    assert no_fiable and "desenfocada" in motivo
    assert "306" not in pasos_como_texto(lec)


def test_la_confianza_atipica_tambien_la_detecta():
    """Señal 4, independiente de las otras dos."""
    lec = lectura(["24 x 13 = 306"], final="306")
    lec.update(consistente=True, legible_declarada=True, confianza_atipica=True)
    assert lectura_no_fiable(lec)[0] is True


def test_una_lectura_limpia_pasa_las_cuatro_señales():
    """El falso positivo es un coste real: marcar ilegible una hoja legible hace
    fallar un caso que el tutor podía acertar."""
    lec = lectura(["24 x 13 = 302"], final="302")
    lec.update(consistente=True, legible_declarada=True, confianza_atipica=False)
    assert lectura_no_fiable(lec)[0] is False
    assert "Paso 1: 24 x 13 = 302" in pasos_como_texto(lec)


def test_legibilidad_desconocida_no_marca_nada():
    """Si la pregunta de legibilidad no se pudo parsear (`None`), no se inventa un
    veredicto en ninguna dirección."""
    lec = lectura(["24 x 13 = 302"], final="302")
    lec.update(consistente=True, legible_declarada=None, confianza_atipica=False)
    assert lectura_no_fiable(lec)[0] is False


# --- 10.3 · el criterio robusto de atípicos ------------------------------

def test_atipicos_por_mad_marca_el_valor_muy_bajo():
    v = {"a": -0.5, "b": -0.52, "c": -0.48, "d": -0.51, "e": -0.49, "f": -4.0}
    assert atipicos_por_mad(v) == {"f"}


def test_atipicos_por_mad_no_marca_un_conjunto_homogeneo():
    assert atipicos_por_mad({c: -0.5 - i * 0.01 for i, c in enumerate("abcdef")}) == set()


def test_atipicos_por_mad_no_opina_con_pocos_datos():
    """Con tres o menos no hay 'resto del conjunto' contra el que comparar."""
    assert atipicos_por_mad({"a": -0.5, "b": -0.5, "c": -9.0}) == set()


def test_atipicos_por_mad_tolera_mad_cero():
    assert atipicos_por_mad({"a": -1.0, "b": -1.0, "c": -1.0, "d": -1.0}) == set()


def test_atipicos_por_mad_no_marca_los_valores_altos():
    """Solo la confianza BAJA es sospechosa; la alta no."""
    v = {"a": -0.5, "b": -0.52, "c": -0.48, "d": -0.51, "e": -0.49, "f": -0.01}
    assert atipicos_por_mad(v) == set()


# ==========================================================================
# 11 · El experimento de diagnóstico: datos, etiquetas y separación
# ==========================================================================

TRAYECTORIAS_DG = RAIZ / "data" / "trayectorias_diagnostico.jsonl"


@pytest.fixture(scope="module")
def diagnostico():
    if not TRAYECTORIAS_DG.exists():
        pytest.skip("no están las trayectorias de diagnóstico")
    return [json.loads(l) for l in TRAYECTORIAS_DG.read_text(encoding="utf-8").splitlines()]


def test_hay_train_y_dev_y_no_comparten_enunciado(diagnostico):
    tr = [e for e in diagnostico if e["split"] == "train"]
    dv = [e for e in diagnostico if e["split"] == "dev"]
    assert len(tr) >= 25 and len(dv) >= 6
    assert not ({e["enunciado"] for e in tr if e["enunciado"]} &
                {e["enunciado"] for e in dv if e["enunciado"]})


def test_cada_ejemplo_termina_en_una_conclusion_valida(diagnostico):
    """La etiqueta se lee con el MISMO parser del evaluador: si el generador y el
    evaluador se separaran, se entrenaría hacia un formato que no se mide."""
    for e in diagnostico:
        salida = e["mensajes"][-1]["content"]
        conclusion = extraer_diagnostico(salida)
        assert diagnostico_valido(conclusion), (e["id"], conclusion)
        assert conclusion == e["diagnostico"].lower(), e["id"]
        assert salida.strip().splitlines()[-1].startswith("Diagnóstico:"), e["id"]


def test_las_cinco_categorias_estan_en_los_dos_splits(diagnostico):
    for split in ("train", "dev"):
        cats = {e["diagnostico"].split()[0].lower()
                for e in diagnostico if e["split"] == split}
        assert {"paso", "sin", "ilegible", "fuera", "dato"} <= cats, (split, cats)


def test_el_paso_señalado_no_lo_domina_un_indice(diagnostico):
    """El eval set está sesgado a 'Paso 2' (4 de 6). Si el entrenamiento lo está
    igual, el modelo aprende el atajo y acertaría sin localizar nada."""
    from collections import Counter
    pasos = Counter(e["paso_error"] for e in diagnostico
                    if e["split"] == "train" and e["paso_error"])
    assert len(pasos) >= 3
    assert max(pasos.values()) <= 0.5 * sum(pasos.values()), dict(pasos)


def test_la_aritmetica_respalda_cada_etiqueta(diagnostico):
    """Una etiqueta mal puesta enseñaría lo contrario de lo que se busca."""
    from fractions import Fraction
    for e in diagnostico:
        op = e.get("op")
        if not op:
            continue
        a, b = Fraction(str(op["a"])), Fraction(str(op["b"]))
        bueno = {"x": a * b, "+": a + b, "-": a - b, "/": a / b}[op["op"]]
        escrito = Fraction(str(op["escrito"]))
        if e["paso_error"] in (2, 3):
            assert bueno != escrito, f"{e['id']}: se marca error y la cuenta está bien"
        if e["diagnostico"] == "sin error":
            assert bueno == escrito, f"{e['id']}: dice 'sin error' y la cuenta está mal"


def test_ningun_enunciado_sale_del_eval_set_de_m4(diagnostico, casos):
    """La filtración que invalidaría el experimento entero."""
    import re as _re
    import unicodedata as _ud

    def norm(s):
        s = _ud.normalize("NFKD", (s or "").lower())
        return " ".join(_re.findall(r"\w+", "".join(c for c in s if not _ud.combining(c))))

    def jac(x, y):
        a, b = set(norm(x).split()), set(norm(y).split())
        return len(a & b) / len(a | b) if a | b else 0.0

    evaluacion = [c["enunciado"] for c in casos
                  if c.get("enunciado") and not c["enunciado"].startswith("(")]
    for e in diagnostico:
        if not e["enunciado"]:
            continue
        peor = max(evaluacion, key=lambda x: jac(e["enunciado"], x))
        assert jac(e["enunciado"], peor) < 0.6, (e["id"], peor)


def test_la_explicacion_no_reescribe_el_procedimiento(diagnostico):
    """Es la conducta que se quiere quitar: en la corrida del 24-10 el tutor
    rehacía el ejercicio en vez de dictaminar."""
    for e in diagnostico:
        explicacion = e["mensajes"][-1]["content"].rsplit("\n", 1)[0]
        assert "Respuesta final" not in explicacion, e["id"]
        assert explicacion.count("Paso ") <= 2, e["id"]


def test_el_prompt_de_entrenamiento_es_el_de_inferencia(diagnostico):
    """Si se separaran, el entrenamiento no se trasladaría a S13."""
    from vision import INSTRUCCION_REVISION, RECORDATORIO_FORMATO
    for e in diagnostico:
        usuario = e["mensajes"][1]["content"]
        assert INSTRUCCION_REVISION in usuario, e["id"]
        assert usuario.rstrip().endswith(RECORDATORIO_FORMATO.rstrip()), e["id"]


def test_ningun_prompt_de_entrenamiento_trae_un_ejercicio_resuelto(diagnostico):
    """`vis-11` copió el ejemplo del prompt entero, con su `Diagnóstico: Paso 2`.
    Si los prompts de entrenamiento lo llevaran, se estaría enseñando a copiarlo.

    Se mira el preámbulo, no palabras sueltas: `dg-a13` usa "bandejas" en su
    propio ejercicio y eso es legítimo."""
    for e in diagnostico:
        preambulo = e["mensajes"][1]["content"].split("Ahora la hoja de verdad")[0]
        assert "panader" not in preambulo.lower(), e["id"]
        assert preambulo.count("Paso 1:") == 0, e["id"]


# --- el conjunto de calibración, separado del eval set --------------------

CALIBRACION = RAIZ / "m4" / "eval_set" / "calibracion" / "manifiesto.json"


@pytest.fixture(scope="module")
def calibracion():
    if not CALIBRACION.exists():
        pytest.skip("no está el conjunto de calibración")
    return json.loads(CALIBRACION.read_text(encoding="utf-8"))


def test_la_calibracion_no_usa_ninguna_hoja_del_eval_set(calibracion, evaluacion):
    """Si usara una foto `vis-*`, calibrar el umbral la convertiría en dato de
    ajuste y su resultado dejaría de medir nada."""
    ids_eval = {c["id"] for c in evaluacion}
    for par in calibracion["pares"]:
        assert par["hoja"] not in ids_eval, par["hoja"]
        assert not par["hoja"].startswith("vis-"), par["hoja"]


def test_la_calibracion_trae_pares_etiquetados(calibracion):
    from collections import Counter
    cuenta = Counter(p["etiqueta"] for p in calibracion["pares"])
    assert cuenta["legible"] >= 4
    assert cuenta["ilegible_leve"] >= 4, "falta el nivel que de verdad prueba la detección"
    assert cuenta["ilegible_fuerte"] >= 4, "falta el control"


def test_el_manifiesto_declara_su_limite(calibracion):
    """Que el aviso esté en los datos, no solo en la conversación."""
    assert "manuscrita" in calibracion["aviso"]


# ==========================================================================
# 12 · Carga y selección del adaptador de diagnóstico
#
# La corrida del 25-10 imprimió "No está adaptadores/qwen-lora-diagnostico" con
# los dos archivos presentes en Drive: la condición miraba si estaba *cargado* y
# nadie lo cargaba. Un aviso que no verifica lo que afirma manda a buscar el
# fallo donde no está.
# ==========================================================================

from vision import (  # noqa: E402
    ARCHIVOS_ADAPTADOR,
    elegir_adaptador,
    revisar_adaptador,
)


def _adaptador(tmp_path, *archivos):
    d = tmp_path / "qwen-lora-diagnostico"
    d.mkdir()
    for nombre in archivos:
        (d / nombre).write_text("x", encoding="utf-8")
    return d


def test_adaptador_completo_pasa(tmp_path):
    d = _adaptador(tmp_path, *ARCHIVOS_ADAPTADOR)
    ok, motivo = revisar_adaptador(d)
    assert ok is True and "completo" in motivo


def test_directorio_inexistente_lo_dice(tmp_path):
    ok, motivo = revisar_adaptador(tmp_path / "no-existe")
    assert ok is False and "no existe el directorio" in motivo


def test_un_archivo_no_es_un_directorio(tmp_path):
    f = tmp_path / "qwen-lora-diagnostico"
    f.write_text("x", encoding="utf-8")
    ok, motivo = revisar_adaptador(f)
    assert ok is False and "no es un directorio" in motivo


def test_falta_el_config_y_el_motivo_lo_nombra(tmp_path):
    d = _adaptador(tmp_path, "adapter_model.safetensors")
    ok, motivo = revisar_adaptador(d)
    assert ok is False
    assert "adapter_config.json" in motivo
    assert "adapter_model.safetensors" in motivo, "el motivo lista lo que SÍ hay"


def test_faltan_los_pesos(tmp_path):
    d = _adaptador(tmp_path, "adapter_config.json")
    ok, motivo = revisar_adaptador(d)
    assert ok is False and "adapter_model.safetensors" in motivo


def test_directorio_vacio(tmp_path):
    d = tmp_path / "qwen-lora-diagnostico"
    d.mkdir()
    ok, motivo = revisar_adaptador(d)
    assert ok is False and "vacío" in motivo


def test_acepta_el_formato_antiguo_bin(tmp_path):
    """Un adaptador guardado por una versión vieja de peft trae .bin."""
    d = _adaptador(tmp_path, "adapter_config.json", "adapter_model.bin")
    ok, motivo = revisar_adaptador(d)
    assert ok is True and "adapter_model.bin" in motivo


def test_el_motivo_nunca_afirma_una_ausencia_que_no_comprobo(tmp_path):
    """El fallo exacto del 25-10: decir 'no está' sin haber mirado."""
    d = _adaptador(tmp_path, *ARCHIVOS_ADAPTADOR)
    ok, motivo = revisar_adaptador(d)
    assert ok and "no existe" not in motivo and "falta" not in motivo


# --- la selección ---------------------------------------------------------

def test_elige_diagnostico_cuando_esta_cargado():
    assert elegir_adaptador(["m1", "agente", "diagnostico"], "diagnostico", "agente") == "diagnostico"


def test_cae_al_alternativo_cuando_no_esta():
    assert elegir_adaptador(["m1", "agente"], "diagnostico", "agente") == "agente"


def test_la_seleccion_no_inventa_un_tercero():
    assert elegir_adaptador([], "diagnostico", "agente") == "agente"


# --- el notebook generado -------------------------------------------------

@pytest.fixture(scope="module")
def celdas_pipeline():
    nb = RAIZ / "m4" / "pipeline" / "S13_M4_Pipeline_Visual.ipynb"
    if not nb.exists():
        pytest.skip("no está el notebook generado")
    return [("".join(c["source"]))
            for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"]


def test_el_notebook_carga_el_adaptador_de_diagnostico(celdas_pipeline):
    carga = [s for s in celdas_pipeline
             if "load_adapter" in s and "qwen-lora-diagnostico" in s]
    assert carga, "ninguna celda carga adaptadores/qwen-lora-diagnostico"
    assert 'adapter_name="diagnostico"' in carga[0]


def test_la_carga_comprueba_el_disco_antes(celdas_pipeline):
    carga = next(s for s in celdas_pipeline
                 if "load_adapter" in s and "qwen-lora-diagnostico" in s)
    assert "revisar_adaptador" in carga
    assert carga.index("revisar_adaptador") < carga.index("load_adapter")


def test_la_carga_no_toca_m1_ni_agente(celdas_pipeline):
    carga = next(s for s in celdas_pipeline
                 if "load_adapter" in s and "qwen-lora-diagnostico" in s)
    assert 'assert "m1" in modelo.peft_config' in carga
    assert "from_pretrained" not in carga, "no debe recargar el modelo base"
    for otro in ("qwen-lora-agente", '"adaptadores/qwen-lora"'):
        assert otro not in carga, f"la celda toca {otro}"


def test_la_carga_va_antes_de_la_seleccion(celdas_pipeline):
    i_carga = next(i for i, s in enumerate(celdas_pipeline)
                   if "load_adapter" in s and "qwen-lora-diagnostico" in s)
    i_sel = next(i for i, s in enumerate(celdas_pipeline)
                 if "ADAPTADOR_DIAGNOSTICO = elegir_adaptador" in s)
    assert i_carga < i_sel


def test_el_mensaje_de_la_seleccion_usa_el_motivo_comprobado(celdas_pipeline):
    """Ya no puede afirmar una ausencia por su cuenta."""
    sel = next(s for s in celdas_pipeline if "ADAPTADOR_DIAGNOSTICO = elegir_adaptador" in s)
    assert "DIAGNOSTICO_MOTIVO" in sel
    assert "No está adaptadores" not in sel


# ==========================================================================
# 13 · Verificador aritmético determinista (EXPERIMENTO)
#
# Comprueba las cuentas de una hoja con aritmética exacta, reutilizando
# `parsear` de `herramientas.py`. Mide el verificador, no al tutor.
# ==========================================================================

from verificador_aritmetico import (  # noqa: E402
    igualdades_de,
    pista_para_el_tutor,
    revisar_hoja,
)


def test_marca_el_paso_cuya_cuenta_no_cuadra():
    h = revisar_hoja(["Multiplicamos: 26 × 14.", "26 × 14 = 354.",
                      "Respuesta final: 354"])
    assert h.paso == 2 and h.tipo == "calculo"
    assert "364" in h.detalle


def test_marca_el_traslado_cuando_los_pasos_cuadran():
    """`dg-a06` y `vis-03`: las cuentas están bien y lo que falla es copiar el
    resultado a la línea final."""
    h = revisar_hoja(["Multiplicamos: 26 × 14.", "26 × 14 = 364.",
                      "Respuesta final: 354"])
    assert h.paso == 3 and h.tipo == "traslado"


def test_no_marca_nada_si_todo_cuadra():
    h = revisar_hoja(["Restamos: 1420 − 385.", "1420 − 385 = 1035.",
                      "Respuesta final: 1035"])
    assert h.paso is None and h.tipo == "ninguno"


def test_marca_el_primero_y_no_todos():
    """Si el paso 2 está mal, los siguientes arrastran el error."""
    h = revisar_hoja(["10 + 5 = 16.", "16 × 2 = 32.", "Respuesta final: 32"])
    assert h.paso == 1


def test_se_abstiene_sin_pasos():
    h = revisar_hoja([])
    assert h.paso is None and h.tipo == "no_evaluable"


def test_se_abstiene_sin_aritmetica():
    h = revisar_hoja(["Lista: leche, pan", "Respuesta final: huevos"])
    assert h.tipo == "no_evaluable"


def test_entiende_fracciones_exactas():
    """Sin coma flotante: 3/4 + 1/8 son 7/8, no 0.875 aproximado."""
    assert revisar_hoja(["3/4 + 1/8 = 7/8.", "Respuesta final: 7/8"]).tipo == "ninguno"
    assert revisar_hoja(["3/4 + 1/8 = 8/8.", "Respuesta final: 8/8"]).paso == 1


def test_entiende_los_simbolos_de_la_hoja():
    for expr in ("24 × 13 = 312", "24 x 13 = 312", "1248 ÷ 8 = 156",
                 "1500 − 847 = 653"):
        assert revisar_hoja([expr + "."]).tipo == "ninguno", expr


def test_no_opina_de_lo_que_no_puede_interpretar():
    """Una línea sin igualdad legible no se marca: no opinar es mejor que
    inventar un veredicto."""
    assert revisar_hoja(["El área es base por altura.",
                         "Respuesta final: 60"]).tipo == "no_evaluable"


def test_igualdades_de():
    assert igualdades_de("Paso 2: 24 × 13 = 302.") == [("24 × 13", "302")]
    assert igualdades_de("El área de un rectángulo es base por altura.") == []


def test_la_pista_no_emite_veredicto():
    """La conclusión la sigue dando el tutor: si la pista dijera `Diagnóstico:`,
    el scorecard mediría esta función y no al modelo."""
    for pasos in (["24 × 13 = 302."], ["24 × 13 = 312."], []):
        assert "Diagnóstico" not in pista_para_el_tutor(revisar_hoja(pasos))


def test_sobre_los_nueve_casos_de_desarrollo(diagnostico):
    """El resultado que el README reporta: 4/4 pasos erróneos evaluables con el
    índice correcto y 0 falsos positivos en los `sin error`."""
    dev = [e for e in diagnostico if e["split"] == "dev"]
    con_paso = [e for e in dev if e["diagnostico"].lower().startswith("paso ")]
    sin_error = [e for e in dev if e["diagnostico"].lower() == "sin error"]

    aciertos = sum(1 for e in con_paso
                   if revisar_hoja(e["pasos"]).paso ==
                   int(e["diagnostico"].split()[1]))
    fp = sum(1 for e in sin_error if revisar_hoja(e["pasos"]).marca_un_paso)

    assert aciertos == len(con_paso) == 4, f"{aciertos}/{len(con_paso)}"
    assert fp == 0


def test_el_prefijo_paso_n_no_entra_en_la_cuenta():
    """Fallo real: `:` se trataba como división, así que "Paso 2: 24 × 13 = 302"
    se leía como "2 / 24 × 13". Las transcripciones del VLM traen ese prefijo
    (`vis-05`, `vis-06`, `vis-09` lo tienen)."""
    assert igualdades_de("Paso 2: 24 × 13 = 302.") == [("24 × 13", "302")]
    assert revisar_hoja(["Paso 1: 24 × 13.", "Paso 2: 24 × 13 = 302.",
                         "Paso 3: Respuesta final: 302"]).paso == 2
    assert revisar_hoja(["Paso 1: 24 × 13.", "Paso 2: 24 × 13 = 312.",
                         "Paso 3: Respuesta final: 312"]).tipo == "ninguno"


# ==========================================================================
# 11 · Por qué la salvaguarda no marcó `vis-10` el 25-10, y la señal 4 con UNA
#      sola foto (revisión M4)
# ==========================================================================

from vision import (  # noqa: E402
    LLAVES_LECTURA_VIGENTE,
    confianza_atipica_frente_a,
    lectura_vigente,
    referencia_de_confianza,
    revisar_foto,
)


def test_la_cache_entregada_es_de_un_esquema_anterior_y_no_se_reusa():
    """`m4/lecturas_m4.json` trae `consistente` pero no las señales 3 y 4: con el
    criterio anterior (`"consistente" in previa`) se reusaba y la salvaguarda
    quedaba ciega en 8 de 11 fotos."""
    cache = json.loads((RAIZ / "m4" / "lecturas_m4.json").read_text(encoding="utf-8"))
    assert all("consistente" in v for v in cache.values())
    assert not any(lectura_vigente(v) for v in cache.values())


def test_una_lectura_con_las_cuatro_senales_si_se_reusa():
    lec = {k: None for k in LLAVES_LECTURA_VIGENTE}
    assert lectura_vigente(lec) is True
    assert lectura_vigente(None) is False


def test_con_tres_confianzas_la_senal_4_no_decide():
    """Lo que pasó el 25-10: solo las 3 fotos del piloto tenían confianza."""
    assert atipicos_por_mad({"vis-01": -0.004, "vis-07": -0.0003, "vis-10": -0.101}) == set()


# Confianzas de lecturas reales (Qwen2-VL-2B, Apple MPS fp16) de las 10 fotos
# legibles del eval set. Solo sirven para probar la mecánica: la referencia de
# verdad se construye con fotos que NO son del eval set.
_CONF_LEGIBLES = [-0.0043, -0.0161, -0.0270, -0.0131, -0.0092, -0.0357,
                  -0.0003, -0.0020, -0.0025, -0.0034]


def test_la_referencia_exige_al_menos_cuatro_fotos():
    with pytest.raises(ValueError):
        referencia_de_confianza([-0.01, -0.02, None])


def test_la_foto_movida_cae_bajo_el_umbral_de_la_referencia():
    ref = referencia_de_confianza(_CONF_LEGIBLES)
    assert confianza_atipica_frente_a(-0.101, ref) is True
    assert confianza_atipica_frente_a(-0.0043, ref) is False


def test_sin_valor_o_con_mad_nula_la_senal_4_no_se_dispara():
    assert confianza_atipica_frente_a(None, referencia_de_confianza(_CONF_LEGIBLES)) is None
    assert confianza_atipica_frente_a(-0.5, referencia_de_confianza([-0.01] * 5)) is None
    assert confianza_atipica_frente_a(-0.5, None) is None


def _lectura(conf, legible=True, consistente=True):
    return {"ok": True, "consistente": consistente, "legible_declarada": legible,
            "confianza_numeros": conf,
            "json": {"pasos": ["24 x 13", "24 x 13 = 306"], "respuesta_final": "306"}}


def test_revisar_foto_no_diagnostica_una_lectura_no_fiable():
    """`vis-10`: el VLM dice que se lee y lo lee igual dos veces; solo la señal 4,
    contra una referencia, la detiene. El tutor no llega a ver el invento."""
    llamadas = []
    r = revisar_foto("img", "¿en qué paso me equivoqué?",
                     leer=lambda img, a: _lectura(-0.101),
                     diagnosticar=lambda p, l: llamadas.append(p) or "Diagnóstico: paso 1",
                     referencia=referencia_de_confianza(_CONF_LEGIBLES), archivo="vis-10.jpg")
    assert r["estado"] == "no_pude_leer" and r["diagnostico"] == "ilegible"
    assert r["senal_4"] == "atípica" and llamadas == []
    assert "No pude leer tu hoja" in r["respuesta"]


def test_revisar_foto_sin_referencia_deja_pasar_la_foto_movida_y_lo_dice():
    r = revisar_foto("img", "p", leer=lambda img, a: _lectura(-0.101),
                     diagnosticar=lambda p, l: "Diagnóstico: sin error")
    assert r["estado"] == "diagnostico" and r["senal_4"] == "sin referencia"


def test_revisar_foto_diagnostica_una_lectura_fiable():
    r = revisar_foto("img", "p", leer=lambda img, a: _lectura(-0.004),
                     diagnosticar=lambda p, l: "Explicación.\nDiagnóstico: paso 2",
                     referencia=referencia_de_confianza(_CONF_LEGIBLES))
    assert r["estado"] == "diagnostico" and r["diagnostico"] == "paso 2"
    assert r["lectura"][-1] == "Respuesta final: 306"


def test_revisar_foto_respeta_las_otras_senales():
    r = revisar_foto("img", "p", leer=lambda img, a: _lectura(-0.004, legible=False),
                     diagnosticar=lambda p, l: "Diagnóstico: paso 2")
    assert r["estado"] == "no_pude_leer"


# ---------------------------------------------------------------------------
# 12 · `numeros_fieles`: todos los números de la hoja, no solo el final
# ---------------------------------------------------------------------------
from vision import lectura_fiel, numeros_fieles  # noqa: E402

_VIS09 = {"id": "vis-09", "bloque": "localizacion", "valor_final_escrito": "63",
          "pasos": ["El área del triángulo es base por altura dividido entre dos.",
                    "14 × 9 = 126.", "126 ÷ 2 = 63.",
                    "Respuesta final: 63 centímetros cuadrados"]}


def _leida(pasos, final):
    return {"json": {"pasos": pasos, "respuesta_final": final}, "ok": True,
            "consistente": True, "legible_declarada": True,
            "confianza_numeros": -0.003, "confianza_atipica": False}


def test_numeros_fieles_detecta_el_126_leido_como_12G():
    """Lectura real de vis-09 (la misma en su corrida y en la reproducción)."""
    lec = _leida(["El área de triángulo es base por altura dividido entre 2.",
                  "14 x 9 = 12G", "12G / 2 = 63", "Respuesta Final: 63 cm"], "63 cm")
    assert lectura_fiel(_VIS09, lec) is True       # la regla actual no lo ve
    assert numeros_fieles(_VIS09, lec) is False


def test_numeros_fieles_acepta_la_misma_cifra_escrita_distinto():
    caso = {"bloque": "localizacion",
            "pasos": ["El descuento es 45000 × 0.20 = 9000.", "Respuesta final: 3/4"]}
    lec = _leida(["El descuento es 45000 x 0.2 = 9000"], "6/8")
    assert numeros_fieles(caso, lec) is False      # 3/4 escrito, 6/8 leído
    lec = _leida(["El descuento es 45000 x 0.2 = 9000"], "3/4")
    assert numeros_fieles(caso, lec) is True


def test_numeros_fieles_no_aplica_a_adversariales():
    assert numeros_fieles({"bloque": "adversarial", "pasos": []}, _leida([], "")) is None
