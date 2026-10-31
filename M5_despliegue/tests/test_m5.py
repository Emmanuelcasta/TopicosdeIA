"""Pruebas de M5 que corren sin GPU y sin modelo.

Qué cubren: las rutas que pueden romperse en silencio —el estado degradado, el
health check, el registro que no debe filtrar texto— y que la aplicación no
duplique la lógica del tutor.

Qué NO cubren: la calidad de las respuestas. Eso es el harness, y necesita GPU.

    python -m pytest M5_despliegue/tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "M5_despliegue" / "app"))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "scripts" / "agente"))

import pipeline  # noqa: E402
from telemetria import registrar  # noqa: E402
from salud import salud, texto_salud  # noqa: E402


# --- el corpus sale de la fuente canónica, no de una copia ---------------

def test_el_corpus_se_carga_de_la_fuente_del_repositorio():
    chunks, metadatos, ids = pipeline.cargar_corpus()
    assert len(chunks) == len(metadatos) == len(ids) > 0
    assert all("fuente" in m for m in metadatos)


def test_no_hay_una_segunda_copia_del_corpus_en_m5():
    """Si alguien pega el corpus dentro de la app, las dos se separan."""
    fuente = (RAIZ / "M5_despliegue" / "app" / "pipeline.py").read_text(encoding="utf-8")
    assert "CODIGO_CORPUS" in fuente, "debe importar la fuente canónica"
    assert "sistema institucional de evaluación" not in fuente, "corpus duplicado"


def test_la_app_no_reimplementa_el_tutor():
    app = (RAIZ / "M5_despliegue" / "app" / "app.py").read_text(encoding="utf-8")
    for pieza in ("AgenteTutor(", "RegistroHerramientas(", "SYSTEM_C5"):
        assert pieza not in app, f"la interfaz reimplementa {pieza}"


# --- retrieval de repliegue ----------------------------------------------

def test_el_buscador_lexico_encuentra_lo_evidente():
    chunks, _, _ = pipeline.cargar_corpus()
    buscar = pipeline.buscador_lexico(chunks)
    idx = buscar("nota mínima aprobatoria del colegio", k=3)
    assert idx, "no encontró nada para una consulta del corpus"
    assert all(0 <= i < len(chunks) for i in idx)


def test_el_buscador_no_lanza_con_consulta_vacia():
    buscar = pipeline.buscador_lexico(["algo"])
    assert buscar("", k=3) == []


# --- estado degradado ----------------------------------------------------

def test_el_tutor_arranca_sin_modelo_y_lo_dice():
    """Sin torch/transformers el pipeline no debe caerse: debe reportar."""
    t = pipeline.Tutor()
    assert t.estado.corpus is True
    assert isinstance(t.estado.listo, bool)
    assert t.estado.resumen()


def test_responder_sin_modelo_devuelve_un_fallo_legible(monkeypatch):
    t = pipeline.Tutor()
    monkeypatch.setattr(t, "_cargar_modelo", lambda: None)
    t.estado.modelo = False
    t.estado.motivos = ["modelo: falta torch"]
    r = t.responder("¿cuánto es 2+2?")
    assert r["ok"] is False
    assert "modelo" in r["respuesta"]
    assert isinstance(r["latencia_s"], float)


def test_pregunta_vacia_no_carga_el_modelo():
    t = pipeline.Tutor()
    r = t.responder("   ")
    assert r["ok"] is False and "pregunta" in r["respuesta"]


def test_la_via_visual_no_existe_si_la_configuracion_no_la_pide():
    """S08-D es texto. Ofrecer la pestaña de visión ahí sería ofrecer M4 como si
    formara parte de lo que midió 18/21, y no lo es."""
    for metodo, args in (("leer_imagen", (None,)),
                         ("revisar_pasos", (["3 x 24 = 82"], "", None, True)),
                         ("revisar_imagen", (None,))):
        r = getattr(pipeline.Tutor("s08d"), metodo)(*args)
        assert r["ok"] is False, metodo
        assert "no incluye la vía visual" in r["respuesta"], metodo
        assert "NO está medida" in r["respuesta"], metodo


def test_imagen_ausente_se_reporta():
    r = pipeline.Tutor("revision_foto").leer_imagen(None)
    assert r["ok"] is False and "foto" in r["respuesta"]


# --- la configuración se elige, no se deduce del disco --------------------

def test_la_configuracion_por_defecto_es_la_medida():
    from configuraciones import POR_DEFECTO, obtener
    cfg = obtener(None)
    assert POR_DEFECTO == "s08d"
    assert cfg.medido == "18/21" and not cfg.es_experimental
    # S08-D no es una configuración agéntica: ni herramientas ni prompt de agente.
    assert cfg.adaptador == "m1" and cfg.system == "rag"
    assert cfg.herramientas is False and cfg.vision is False


def test_cada_variante_experimental_se_declara_sin_medir():
    from configuraciones import CONFIGURACIONES
    for nombre in ("s08d_herramientas", "s08d_verificador", "s08d_vision"):
        cfg = CONFIGURACIONES[nombre]
        assert cfg.es_experimental and cfg.medido is None and cfg.fuente is None
        assert "HIPÓTESIS" in cfg.notas


def test_toda_configuracion_medida_cita_su_fuente():
    from configuraciones import CONFIGURACIONES
    for cfg in CONFIGURACIONES.values():
        if cfg.medido:
            assert cfg.fuente and ".csv" in cfg.fuente, cfg.nombre


def test_una_configuracion_inexistente_se_dice_en_vez_de_adivinarse():
    from configuraciones import obtener
    with pytest.raises(KeyError) as e:
        obtener("la_mejor")
    assert "s08d" in str(e.value)


def test_el_adaptador_lo_pide_la_configuracion_no_el_disco():
    """El fallo original: `qwen-lora-agente` se cargaba por estar presente."""
    assert pipeline.Tutor("s08d").estado.adaptador_pedido == "m1"
    assert pipeline.Tutor("solo_lora").estado.adaptador_pedido == "m1"
    assert pipeline.Tutor("agente_c5ft").estado.adaptador_pedido == "agente"
    # Y nada se da por cargado antes de cargarlo.
    assert pipeline.Tutor("s08d").estado.adaptador_cargado is None


def test_la_salud_distingue_lo_declarado_de_lo_verificado():
    from salud import salud
    s = salud(pipeline.Tutor("s08d"))
    c = s["componentes"]
    assert s["montaje"].startswith("declarado")
    assert c["retrieval_pedido"] == "hibrido"
    assert c["herramientas_activas"] == 0            # S08-D no las usa
    assert c["herramientas_en_el_registro"] == 18     # pero existen
    assert "18/21" in s["resultado_medido"]
    assert salud(pipeline.Tutor("s08d_verificador"))["resultado_medido"].startswith(
        "ninguno")


# --- health check --------------------------------------------------------

def test_la_salud_responde_sin_cargar_el_modelo():
    d = salud()
    assert d["estado"] == "arriba"
    assert "version" in d and "activo_s" in d
    assert "componentes" not in d, "sin tutor no debe inventar estado del modelo"


def test_la_salud_con_tutor_expone_lo_necesario():
    d = salud(pipeline.Tutor())
    for clave in ("version", "activo_s", "listo", "degradado",
                  "configuracion_solicitada", "configuracion_efectiva",
                  "componentes_faltantes", "resultado_medido", "componentes"):
        assert clave in d, clave
    for clave in ("modelo_cargado", "dispositivo", "corpus", "herramientas",
                  "adaptador_pedido", "adaptador_cargado", "retrieval_activo",
                  "enrutador_activo", "cuantizado"):
        assert clave in d["componentes"], clave
    assert json.dumps(d)          # serializable


def test_texto_salud_es_legible():
    assert "version:" in texto_salud()


# --- privacidad en el registro -------------------------------------------

def test_no_se_registra_el_texto_del_usuario_por_defecto(monkeypatch):
    monkeypatch.setattr("telemetria.LOG_TEXTO", False)
    fila = registrar("consulta", "me llamo Ana y saqué 2.1", ok=True, latencia_s=1.0)
    assert "pregunta" not in fila, "se filtró el texto del usuario"
    assert fila["pregunta_len"] == len("me llamo Ana y saqué 2.1")
    assert len(fila["pregunta_hash"]) == 12


def test_el_texto_se_registra_solo_si_se_activa(monkeypatch):
    monkeypatch.setattr("telemetria.LOG_TEXTO", True)
    fila = registrar("consulta", "hola", ok=True)
    assert fila["pregunta"] == "hola"


def test_wandb_apagado_no_rompe_nada(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    import telemetria
    telemetria._wandb, telemetria._avisado = None, False
    assert telemetria._iniciar_wandb() is None
    assert registrar("consulta", "x", ok=True)["evento"] == "consulta"


# --- configuración sin secretos ------------------------------------------

def test_el_env_de_ejemplo_no_trae_secretos():
    txt = (RAIZ / "M5_despliegue" / "config" / ".env.example").read_text(encoding="utf-8")
    assert "WANDB_API_KEY=" not in txt.replace("# WANDB_API_KEY", "")
    for linea in txt.splitlines():
        if "=" in linea and not linea.startswith("#"):
            valor = linea.split("=", 1)[1].strip()
            assert len(valor) < 60, f"valor sospechosamente largo: {linea}"


def test_requirements_tiene_versiones_fijadas():
    """Cada paquete con versión exacta, o con los DOS extremos acotados.

    El rango se admite solo con tope: `pydantic>=2.0` a secas fue justamente el
    agujero por el que entró pydantic 2.11 y rompió la interfaz en Kaggle.
    """
    txt = (RAIZ / "M5_despliegue" / "config" / "requirements.txt").read_text(
        encoding="utf-8")
    paquetes = [l for l in txt.splitlines() if l.strip() and not l.startswith("#")]
    assert paquetes
    for p in paquetes:
        especificacion = p.split(";")[0]          # sin el marcador de plataforma
        acotado = "==" in especificacion or (
            ">=" in especificacion and "<" in especificacion)
        assert acotado, f"sin versión fijada ni rango con tope: {p}"


def test_pydantic_excluye_las_versiones_que_rompen_la_interfaz():
    """Desde pydantic 2.11, un campo `dict` pelado se describe con
    `additionalProperties: true`, y `gradio_client 1.3.0` revienta al recorrerlo:
    `TypeError: argument of type 'bool' is not iterable`. Sin este tope, el
    enlace público de Gradio responde Internal Server Error."""
    txt = (RAIZ / "M5_despliegue" / "config" / "requirements.txt").read_text(
        encoding="utf-8")
    linea = next(l for l in txt.splitlines()
                 if l.strip().startswith("pydantic") and "pydantic_core" not in l)
    assert "<2.11" in linea.replace(" ", ""), linea
    # Y gradio_client fijado explícitamente: en Kaggle el pin transitivo de
    # gradio no basta, porque pip conserva lo preinstalado cuando puede.
    assert "gradio-client==1.3.0" in txt or "gradio_client==1.3.0" in txt


# --- la reproducción de S08-D queda registrada ----------------------------

def test_la_verificacion_del_retrieval_esta_registrada_y_reproduce():
    """Comprueba el REGISTRO, no el sistema vivo.

    Montar el retrieval de verdad necesita `sentence-transformers` y descargar
    dos modelos, así que no cabe en esta suite. Lo que esta prueba vigila es que
    el registro de la última verificación siga diciendo que reproduce: si alguien
    toca `buscador_hibrido` o `_montar_enrutador`, hay que volver a correr
    `M5_despliegue/evaluacion/verificar_retrieval.py --guardar` y el registro
    cambiará con él.
    """
    import json

    ruta = RAIZ / "M5_despliegue" / "evaluacion" / "verificacion_retrieval.json"
    assert ruta.exists(), "falta la verificación: corre verificar_retrieval.py --guardar"
    v = json.loads(ruta.read_text(encoding="utf-8"))
    assert v["reproduce"] is True and not v["fallos"]
    # Los tres rasgos del enrutador, no dos.
    assert set(v["pesos"]) == {"margen", "numeros", "rerank"}
    assert abs(v["pesos"]["rerank"]) > 0.1, "el rasgo del reranker no puede ser 0"
    assert v["loo_dev"] == v["loo_dev_referencia"] or \
        abs(v["loo_dev"] - v["loo_dev_referencia"]) < 1e-3
    assert v["rutas_eval"] == v["rutas_eval_referencia"]
    assert len(v["rutas_eval"]) == 21


# --- las ablaciones: una pieza por fila -----------------------------------

def test_cada_fila_de_ablacion_cambia_una_sola_pieza():
    """Si dos filas difieren en tres cosas, la diferencia no se atribuye.

    La excepción declarada es la fila de herramientas: añadirlas obliga a cambiar
    el prompt, porque sin el formato de llamada el modelo no llama a nada. El
    test exige que esa fila sea la ÚNICA con dos diferencias y que su descripción
    lo diga.
    """
    import sys as _sys
    _sys.path.insert(0, str(RAIZ / "M5_despliegue" / "evaluacion"))
    from ablaciones import FILAS
    from configuraciones import obtener

    piezas = ("adaptador", "retrieval", "enrutador", "herramientas",
              "verificador", "normalizador", "vision", "system")
    for i, (config, diferencia, referencia, _) in enumerate(FILAS):
        if referencia is None:
            continue
        a, b = obtener(FILAS[referencia][0]), obtener(config)
        distintas = [p for p in piezas if getattr(a, p) != getattr(b, p)]
        if config == "s08d_herramientas":
            assert set(distintas) == {"herramientas", "system"}, distintas
            assert "prompt" in diferencia
        elif config == "solo_lora":
            # Quitar el RAG quita también el enrutador: no hay nada que enrutar.
            assert set(distintas) == {"retrieval", "enrutador"}, distintas
        else:
            assert len(distintas) == 1, f"fila {i} ({config}) cambia {distintas}"


def test_los_controles_de_la_ablacion_tienen_cifra_medida():
    import sys as _sys
    _sys.path.insert(0, str(RAIZ / "M5_despliegue" / "evaluacion"))
    from ablaciones import CONTROLES, FILAS
    from configuraciones import obtener

    for i in CONTROLES:
        config, _, _, medido = FILAS[i]
        assert medido, f"la fila de control {i} no tiene cifra que reproducir"
        assert obtener(config).medido == medido
        assert obtener(config).fuente


def test_el_eval_set_fijo_de_m2_no_se_toca():
    """21 casos: 12 cálculo, 4 conocimiento, 5 adversarial. Si esto cambia,
    ninguna cifra de M1 a M4 sigue siendo comparable."""
    import collections
    import json

    casos = [json.loads(l) for l in
             (RAIZ / "data" / "eval_set_m2.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(casos) == 21
    assert collections.Counter(c["bloque"] for c in casos) == {
        "calculo": 12, "conocimiento": 4, "adversarial": 5}


# =========================================================================
# Degradación · la cifra medida se apaga cuando falta una pieza
#
# Por qué estas pruebas y no otras. `18/21` describe un sistema concreto: LoRA
# de M1 + híbrida con reranker + enrutador. Si en una ejecución falta cualquiera
# de los tres, lo que responde es otro sistema, y enseñar esa cifra sería
# atribuirle el resultado de uno distinto. Cada prueba quita un componente y
# comprueba tres cosas: que el sistema queda marcado degradado, que la cifra
# desaparece, y que la interfaz dice qué pieza falta.
# =========================================================================

def _degradar(monkeypatch, config="s08d", **sustituciones):
    """Monta los componentes con una pieza inutilizada. Sin modelo ni GPU."""
    for nombre, valor in sustituciones.items():
        monkeypatch.setattr(pipeline, nombre, valor)
    t = pipeline.Tutor(config)
    t.resolver_componentes()
    return t


def test_sin_recuperador_hibrido_el_sistema_queda_degradado(monkeypatch):
    t = _degradar(monkeypatch, buscador_hibrido=lambda chunks: None)
    assert t.estado.degradado
    assert "retrieval híbrido + reranker" in t.estado.faltantes
    assert t.estado.medido_aplicable is None
    # Responde, pero por la vía léxica, y lo dice.
    assert "REPLIEGUE" in t.estado.retrieval
    assert "18/21" not in t.estado.resumen()
    assert "S08-D ·" not in t.estado.efectiva()


def test_sin_reranker_tampoco_hay_enrutador_y_los_dos_se_informan(monkeypatch):
    """El enrutador usa el puntaje del reranker como tercer rasgo: sin él, no es
    el enrutador que midió nada, y caen las dos piezas."""
    t = _degradar(monkeypatch, buscador_hibrido=lambda chunks: None)
    assert t.estado.faltantes == ["retrieval híbrido + reranker",
                                  "enrutador consulta/problema"]
    assert t.estado.enrutador_pedido is True
    assert t.estado.enrutador is False


def test_un_enrutador_que_no_arranca_se_informa_aunque_el_retrieval_este(monkeypatch):
    """Caso distinto del anterior: la híbrida se monta y el enrutador falla."""
    def hibrido_sin_enrutador(chunks):
        def buscar(consulta, k=3):
            return list(range(min(k, len(chunks))))
        buscar.encode = lambda textos: [[0.0]] * len(list(textos))
        buscar.rerank_max = lambda consulta: 0.0
        return buscar

    monkeypatch.setattr(pipeline, "buscador_hibrido", hibrido_sin_enrutador)
    monkeypatch.setattr(pipeline.Tutor, "_montar_enrutador",
                        lambda self, buscar: None)
    t = pipeline.Tutor("s08d")
    t.resolver_componentes()
    assert t.estado.faltantes == ["enrutador consulta/problema"]
    assert t.estado.degradado and t.estado.medido_aplicable is None
    assert "hibrido" in t.estado.retrieval        # el retrieval SÍ está
    assert "SIN enrutador (se pidió)" in t.estado.piezas()


def test_el_adaptador_que_falta_en_el_disco_marca_la_ejecucion_degradada():
    """Caso real, no hipotético: en este repositorio `qwen-lora-agente` y
    `qwen-lora-diagnostico` no están descargados, solo `qwen-lora`."""
    t = pipeline.Tutor("agente_c5ft")
    ruta = t.comprobar_adaptador()
    if ruta is None:                       # el adaptador no está en el disco
        assert t.estado.degradado
        assert "adaptador LoRA 'agente'" in t.estado.faltantes
        assert t.estado.medido_aplicable is None
        assert any("modelo BASE" in m for m in t.estado.motivos)
    else:                                   # alguien lo descargó: entonces no falta
        assert ruta.is_dir() and not t.estado.faltantes


def test_el_adaptador_presente_no_marca_nada():
    t = pipeline.Tutor("s08d")
    ruta = t.comprobar_adaptador()
    if ruta is not None:
        assert ruta.name == "qwen-lora"
        assert t.estado.faltantes == []


def test_sin_adaptador_el_sistema_no_se_llama_s08d():
    """Sin el LoRA el modelo base marcó 9/21 (C0). No es una variante de S08-D."""
    e = pipeline.Estado(config="S08-D", config_nombre="S08-D · completa",
                        medido="18/21", adaptador_pedido="m1",
                        corpus=True, modelo=True, montado=True)
    e.faltantes.append("adaptador LoRA 'm1'")
    assert e.degradado and e.medido_aplicable is None
    assert "MODO DEGRADADO" in e.efectiva()
    assert "18/21" not in e.resumen()
    assert "NO MEDIDO" in e.resumen()
    assert "adaptador NINGUNO (modelo base)" in e.piezas()


def test_con_todas_las_piezas_la_cifra_historica_si_se_informa():
    e = pipeline.Estado(config="S08-D", config_nombre="S08-D · completa",
                        medido="18/21", adaptador_pedido="m1",
                        adaptador_cargado="m1", retrieval="hibrido",
                        enrutador_pedido=True, enrutador=True,
                        corpus=True, modelo=True, montado=True)
    assert not e.degradado
    assert e.medido_aplicable == "18/21"
    assert "18/21" in e.resumen()


def test_la_salud_degradada_no_atribuye_la_cifra_a_esta_ejecucion(monkeypatch):
    from salud import salud, texto_salud

    t = _degradar(monkeypatch, buscador_hibrido=lambda chunks: None)
    s = salud(t)
    assert s["degradado"] is True
    assert s["componentes_faltantes"]
    assert s["resultado_medido"].startswith("NO MEDIDO")
    assert "no describe lo que responde ahora" in s["resultado_medido"]
    assert s["configuracion_solicitada"] != s["configuracion_efectiva"]
    # Y el texto plano, que es lo que se pega en un informe, dice lo mismo.
    assert "NO MEDIDO" in texto_salud(t)


def test_la_cabecera_de_la_interfaz_marca_el_modo_degradado(monkeypatch):
    import app as interfaz

    t = _degradar(monkeypatch, buscador_hibrido=lambda chunks: None)
    cab = interfaz._cabecera(t)
    assert "MODO DEGRADADO" in cab
    assert "NO MEDIDO" in cab
    assert "retrieval híbrido + reranker" in cab
    # La cifra puede aparecer, pero solo para decir que NO es de esta ejecución.
    if "18/21" in cab:
        assert "no describe lo que responde ahora" in cab
    # Y se ve, pieza por pieza, qué está funcionando.
    assert "Qué está funcionando" in cab
    assert "lexico" in cab


# =========================================================================
# El flujo de revisión de una foto, en dos pasos
#
# Lo que estas pruebas protegen es una decisión de producto, no un detalle de
# implementación: en M4 el VLM inventó «24 x 13 = 306» —dos lecturas idénticas,
# así que ni la doble lectura lo detectó— y el tutor diagnosticó sobre ese
# invento. Un diagnóstico sobre una transcripción sin confirmar es una
# afirmación sobre una hoja que puede no existir.
# =========================================================================

LECTURA_OK = {
    "ok": True, "consistente": True,
    "json": {"enunciado": "3 cajas a 24 pesos cada una",
             "pasos": ["Multiplico 3 x 24", "3 x 24 = 82"],
             "respuesta_final": "82"},
}
LECTURA_ILEGIBLE = {"ok": True, "consistente": True,
                    "json": {"ilegible": True, "motivo": "la foto está movida"}}


def _tutor_con_lectura(monkeypatch, lectura):
    t = pipeline.Tutor("revision_foto")
    monkeypatch.setattr(pipeline.Tutor, "_leer", lambda self, imagen: lectura)
    return t


def test_la_lectura_no_diagnostica_solo_pregunta(monkeypatch):
    t = _tutor_con_lectura(monkeypatch, LECTURA_OK)
    r = t.leer_imagen(object())
    assert r["ok"] and r["etapa"] == "lectura"
    assert r["confirmable"] is True
    assert r["pasos"] == ["Multiplico 3 x 24", "3 x 24 = 82",
                          "Respuesta final: 82"]
    assert "¿Es exactamente lo que escribiste?" in r["pregunta"]
    # Y NO hay veredicto: ni paso señalado ni diagnóstico.
    assert "Diagnóstico" not in r["respuesta"]
    assert "paso 2" not in r["respuesta"].lower()


def test_una_hoja_ilegible_detiene_el_flujo(monkeypatch):
    t = _tutor_con_lectura(monkeypatch, LECTURA_ILEGIBLE)
    r = t.leer_imagen(object())
    assert r["confirmable"] is False
    assert "movida" in r["motivo_no_fiable"]
    assert "No puedo leer bien esta hoja" in r["respuesta"]
    assert "Diagnóstico" not in r["respuesta"]
    # Ofrece una salida concreta en vez de dejar al estudiante sin nada.
    assert "otra foto" in r["respuesta"] or "escribir los pasos" in r["respuesta"]


def test_una_hoja_sin_pasos_tampoco_se_diagnostica(monkeypatch):
    t = _tutor_con_lectura(monkeypatch, {
        "ok": True, "consistente": True,
        "json": {"enunciado": "Arroz, leche, huevos", "pasos": []}})
    r = t.leer_imagen(object())
    assert r["confirmable"] is False
    assert "no encuentro los pasos" in r["respuesta"]


def test_sin_confirmacion_no_hay_diagnostico():
    t = pipeline.Tutor("revision_foto")
    r = t.revisar_pasos(["3 x 24 = 82"], confirmado=False)
    assert r["ok"] is False
    assert "no está confirmada" in r["respuesta"]


def test_el_verificador_determinista_corre_antes_del_tutor():
    """La aritmética la decide `Fraction`, no el modelo de 1.5B."""
    t = pipeline.Tutor("revision_foto")
    h, evidencia = t._comprobar_aritmetica(["Multiplico 3 x 24", "3 x 24 = 82"])
    assert h["tipo"] == "calculo" and h["paso"] == 2
    assert "72" in h["detalle"] and "82" in h["detalle"]
    assert evidencia and "72" in evidencia


def test_cuando_las_cuentas_cuadran_se_le_dice_al_tutor():
    """Inventar un error donde no lo hay es el fallo más dañino de los dos."""
    t = pipeline.Tutor("revision_foto")
    h, evidencia = t._comprobar_aritmetica(["Multiplico 3 x 24", "3 x 24 = 72"])
    assert h["tipo"] == "ninguno"
    assert "SÍ cuadran" in evidencia


def test_la_evidencia_se_pasa_como_observacion_no_como_veredicto():
    from vision import prompt_revision_confirmada

    p = prompt_revision_confirmada(
        "¿dónde me equivoqué?", "3 cajas a 24",
        ["Multiplico 3 x 24", "3 x 24 = 82"],
        evidencia="Comprobación automática: el paso 2 no cuadra: 3 x 24 da 72.")
    assert "NO el veredicto" in p
    assert "decide tú qué paso señalar" in p
    # La regla de no reescribir el procedimiento del estudiante viaja en el prompt.
    assert "No cambies los números del estudiante" in p
    # Y la numeración la pone el sistema, no el modelo de visión.
    assert "Paso 1: Multiplico 3 x 24" in p
    assert "Paso 2: 3 x 24 = 82" in p


def test_el_prompt_sin_evidencia_no_menciona_comprobacion():
    from vision import prompt_revision_confirmada

    p = prompt_revision_confirmada("¿dónde?", None, ["2 + 2 = 4"])
    assert "Comprobación aritmética automática" not in p


def test_revisar_imagen_se_queda_en_la_lectura(monkeypatch):
    """La firma vieja de M4 sigue existiendo, pero ya no diagnostica de una."""
    t = _tutor_con_lectura(monkeypatch, LECTURA_OK)
    r = t.revisar_imagen(object(), "¿en qué paso me equivoqué?")
    assert r["etapa"] == "lectura"
    assert r["confirmable"] is True
    assert "pasos_confirmados" not in r


def test_la_configuracion_de_producto_es_experimental_y_no_cita_cifras_de_m4():
    from configuraciones import obtener

    cfg = obtener("revision_foto")
    assert cfg.es_experimental and cfg.medido is None and cfg.fuente is None
    assert cfg.vision and cfg.verificador
    assert "SIN MEDIR" in cfg.notas
    # S08-D sigue siendo el baseline textual, intacto.
    base = obtener("s08d")
    assert base.medido == "18/21" and not base.vision and not base.verificador
    assert base.system == "rag" and base.herramientas is False


def test_los_doce_casos_de_m4_no_se_describen_como_holdout_ciego():
    """La profesora señaló esta contradicción: `m4/README.md` ya decía que se
    habían gastado, y los documentos de M5 seguían llamándolos reservados."""
    for ruta in (RAIZ / "M5_despliegue" / "app" / "app.py",
                 RAIZ / "M5_despliegue" / "app" / "pipeline.py",
                 RAIZ / "M5_despliegue" / "config" / "README_SPACE.md",
                 RAIZ / "M5_despliegue" / "docs" / "GUION_DEMO.md",
                 RAIZ / "docs" / "REFINAMIENTO_TUTOR_M5.md",
                 RAIZ / "docs" / "ORGANIZACION_REPO_Y_PLAN_M5.md"):
        texto = ruta.read_text(encoding="utf-8")
        for frase in ("siguen sin gastarse", "siguen reservados",
                      "no se han\ngastado"):
            assert frase not in texto, f"{ruta.name}: «{frase}»"


# =========================================================================
# El plan de validación · cada experimento dice qué NO mide
# =========================================================================

def _experimentos():
    import sys as _sys
    _sys.path.insert(0, str(RAIZ / "M5_despliegue" / "evaluacion"))
    from validacion_m5 import EXPERIMENTOS
    return EXPERIMENTOS


def test_cada_experimento_declara_que_no_mide():
    """Un experimento sin límites declarados se cita fuera de contexto."""
    for e in _experimentos():
        for campo in ("mide", "no_mide", "conjunto", "como", "necesita",
                      "criterio", "registra"):
            assert e.get(campo), f"{e['id']}: falta {campo}"
        assert len(e["no_mide"]) > 20, e["id"]


def test_las_cinco_capas_estan_separadas():
    ids = [e["id"] for e in _experimentos()]
    assert ids[:5] == ["E1", "E2", "E3", "E4", "E5"]
    textos = " ".join(e["nombre"].lower() for e in _experimentos())
    for capa in ("texto", "enrutamiento", "herramientas", "lectura", "flujo"):
        assert capa in textos, capa


def test_la_reproduccion_del_retrieval_no_se_presenta_como_exactitud():
    """La confusión que el encargo pidió evitar: pesos y rutas no son aciertos."""
    e2 = next(e for e in _experimentos() if e["id"] == "E2")
    assert "exactitud del tutor" in e2["no_mide"]
    assert "requisito previo" in e2["como"] or "requisito previo" in e2["criterio"]

    import sys as _sys
    _sys.path.insert(0, str(RAIZ / "M5_despliegue" / "evaluacion"))
    import validacion_m5
    # Y el informe generado lo dice en la cabecera, no en una nota al pie.
    cabecera = validacion_m5.informe().split("## Estado")[0]
    assert "no es una medición de exactitud" in cabecera.lower()


def test_la_latencia_separa_la_fria_de_las_calientes():
    import sys as _sys
    _sys.path.insert(0, str(RAIZ / "M5_despliegue" / "evaluacion"))
    from validacion_m5 import latencias

    d = latencias([9.5, 1.1, 1.2, 1.0, 1.3, 1.15])
    assert d["latencia_fria_s"] == 9.5        # la carga del modelo, aparte
    assert d["n_calientes"] == 5
    assert 1.0 <= d["caliente_p50_s"] <= 1.3   # la fría no contamina la mediana
    assert "cota superior ruidosa" in d["aviso"]
    assert latencias([]) == {}


def test_el_eval_set_visual_de_m5_no_reutiliza_enunciados():
    """Reutilizar un caso y llamarlo generalización es el error que esto evita."""
    import sys as _sys
    _sys.path.insert(0, str(RAIZ / "scripts"))
    import eval_set_m5_vision_fuente as m5v

    assert m5v.validar(m5v.CASOS) == []
    assert len(m5v.CASOS) == 12
    # Pares legible/borrosa: sin ellos no se puede hablar de falsos positivos.
    assert sum(1 for c in m5v.CASOS if c["imagen_borrosa"]) >= 11
    # El ilegible real NO tiene versión nítida, a propósito.
    ileg = next(c for c in m5v.CASOS if c["id"] == "m5v-10")
    assert ileg["imagen_nitida"] is None and ileg["imagen_borrosa"]
    # Y cada caso trae su transcripción de referencia.
    assert all(c["pasos"] for c in m5v.CASOS)


def test_el_eval_set_visual_de_m4_sigue_intacto():
    """Evidencia histórica: el conjunto nuevo se añade, no sustituye."""
    assert (RAIZ / "m4" / "eval_set" / "eval_set_m4.csv").exists()
    assert (RAIZ / "m4" / "eval_set" / "HOJAS_PARA_ESCRIBIR.md").exists()
    assert (RAIZ / "m4" / "scorecard_m4.csv").exists()
    assert (RAIZ / "m4" / "resultados_m4.json").exists()


def test_el_notebook_de_m4_fija_las_versiones():
    """La causa declarada de la irreproducibilidad: un rango sin tope."""
    import json as _json

    nb = _json.loads((RAIZ / "m4" / "pipeline" /
                      "S13_M4_Pipeline_Visual.ipynb").read_text(encoding="utf-8"))
    codigo = "".join("".join(c["source"]) for c in nb["cells"]
                     if c["cell_type"] == "code")
    assert "transformers==4.45.2" in codigo
    assert '"transformers>=4.45"' not in codigo
    # Y se comprueba que el pip install surtió efecto.
    assert "VERSIONES_ESPERADAS" in codigo
    # La caché queda a la vista, que fue la otra causa.
    assert "FORZAR_DESCARGA" in codigo
    # La plantilla no lleva salidas: la evidencia vive en los *_ejecutado*.
    assert not any(c.get("outputs") for c in nb["cells"])


def test_la_version_fijada_coincide_con_la_del_despliegue():
    """Si el notebook y el Space usan versiones distintas, medir en uno no dice
    nada del otro."""
    req = (RAIZ / "M5_despliegue" / "config" /
           "requirements.txt").read_text(encoding="utf-8")
    assert "transformers==4.45.2" in req
    assert "peft==0.13.2" in req and "accelerate==0.34.2" in req


# =========================================================================
# Gradio con pydantic >= 2.11 · el fallo del enlace público en Kaggle
#
# Síntoma: la URL *.gradio.live responde Internal Server Error. Causa: `gr.Image`
# publica el esquema de `FileData`, cuyo campo `meta` es un `dict` pelado;
# pydantic >= 2.11 lo describe con `additionalProperties: true`, y
# `gradio_client 1.3.0` recorre ese booleano como si fuera un subesquema:
#
#   if "additionalProperties" in schema:
#       ... _json_schema_to_python_type(schema["additionalProperties"], defs)
#   -> get_type(True) -> "const" in True -> TypeError
#
# Reproducido en local con gradio 4.44.1 + gradio_client 1.3.0 + pydantic 2.13.5.
# =========================================================================

def test_el_parche_de_compatibilidad_arregla_el_subesquema_booleano():
    """Lo que importa: la función que fallaba deja de fallar, con la semántica
    correcta de JSON Schema (`true` acepta cualquier valor, `false` ninguno)."""
    import compat_gradio
    pytest.importorskip("gradio_client")
    from gradio_client import utils

    compat_gradio.aplicar()
    assert utils._json_schema_to_python_type(True, {}) == "Any"
    assert utils._json_schema_to_python_type(False, {}) == "None"
    # Y un objeto con `additionalProperties: true`, que es el caso real.
    esquema = {"type": "object", "properties": {},
               "additionalProperties": True}
    assert "Any" in utils._json_schema_to_python_type(esquema, {})


def test_aplicar_es_idempotente():
    import compat_gradio
    pytest.importorskip("gradio_client")

    primera = compat_gradio.aplicar()
    segunda = compat_gradio.aplicar()
    # La segunda ve el guardia ya puesto y no vuelve a envolver nada.
    assert segunda["parche_aplicado"] is False
    assert "ya acepta" in segunda["motivo"]
    assert primera["versiones"] == segunda["versiones"]


def test_el_diagnostico_separa_las_dos_mitades_del_problema():
    """Una cosa es que la librería no acepte booleanos; otra, que la pydantic
    instalada los produzca. Decir solo «parche aplicado» no distingue si el
    fallo estaba ocurriendo."""
    import compat_gradio

    info = compat_gradio.aplicar()
    assert "se_disparaba" in info and isinstance(info["se_disparaba"], bool)
    v = info["versiones"]
    for clave in ("python", "gradio", "gradio_client", "pydantic"):
        assert clave in v, clave


def test_la_app_puede_describir_su_api():
    """La comprobación de verdad: `get_api_info()` es lo que la ruta principal
    necesita para servir la página. Si lanza, el enlace público da error 500."""
    pytest.importorskip("gradio")
    import compat_gradio

    resultado = compat_gradio.comprobar_api()
    assert resultado.startswith("get_api_info(): OK"), resultado


def test_la_app_aplica_el_parche_antes_de_construir_la_interfaz():
    """El orden importa: parchear después de que Gradio haya resuelto los
    esquemas no serviría de nada."""
    fuente = (RAIZ / "M5_despliegue" / "app" / "app.py").read_text(encoding="utf-8")
    pos_compat = fuente.index("import compat_gradio")
    pos_pipeline = fuente.index("from pipeline import")
    assert pos_compat < pos_pipeline
    assert "COMPAT = compat_gradio.aplicar()" in fuente


def test_un_adaptador_ausente_se_ve_antes_de_terminar_el_montaje(monkeypatch):
    """Al lanzar en Kaggle el adaptador viene de un dataset adjunto y es fácil
    apuntarlo mal. Si `efectiva()` preguntara primero por `montado`, el hecho ya
    conocido quedaría tapado detrás de «sin montar todavía»."""
    monkeypatch.setitem(pipeline.RUTAS_ADAPTADOR, "m1", "ruta/que/no/existe")
    t = pipeline.Tutor("s08d")
    assert t.comprobar_adaptador() is None
    assert t.estado.montado is False          # el montaje no ha terminado
    assert t.estado.degradado                 # y aun así se sabe que falta
    assert "MODO DEGRADADO" in t.estado.efectiva()
    assert t.estado.medido_aplicable is None
