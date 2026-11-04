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


def test_requirements_tiene_tope_superior():
    """Todo paquete acotado por arriba: con `==` o con `<`.

    El invariante es el tope, no el piso. Los dos fallos de Kaggle fueron de lo
    mismo: `pydantic>=2.0` sin tope dejó entrar 2.11 y rompió la interfaz, y la
    `starlette` nueva eliminó la firma de `TemplateResponse` que usa Gradio
    4.44. Un piso de más nunca ha roto nada aquí; un techo de menos, dos veces.
    """
    txt = (RAIZ / "M5_despliegue" / "config" / "requirements.txt").read_text(
        encoding="utf-8")
    paquetes = [l for l in txt.splitlines() if l.strip() and not l.startswith("#")]
    assert paquetes
    for p in paquetes:
        especificacion = p.split(";")[0]          # sin el marcador de plataforma
        assert "==" in especificacion or "<" in especificacion, (
            f"sin tope superior, puede romperse con la próxima versión: {p}")


def test_ninguna_version_fijada_es_inexistente():
    """`peft==0.20.0` no existe en PyPI (la última publicada es 0.17.1) y hacía
    que `pip install -r` fallara entero en cualquier entorno limpio, incluido el
    Space. Se comprueba la forma del pin, no la red: un `==` a una versión que
    nadie puede instalar es un error de escritura, no una decisión."""
    txt = (RAIZ / "M5_despliegue" / "config" / "requirements.txt").read_text(
        encoding="utf-8")
    linea = next(l for l in txt.splitlines()
                 if l.strip().startswith("peft"))
    assert "peft==0.20.0" not in linea, linea
    assert "<" in linea, f"peft necesita tope superior: {linea}"


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
    # Gradio 4.44 importa HfFolder, retirado en versiones recientes del Hub.
    assert 'huggingface_hub==0.26.5' in txt
    assert 'starlette<1.0' in txt


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
    t = _degradar(monkeypatch, buscador_hibrido=lambda chunks, diag=None: None)
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
    t = _degradar(monkeypatch, buscador_hibrido=lambda chunks, diag=None: None)
    assert t.estado.faltantes == ["retrieval híbrido + reranker",
                                  "enrutador consulta/problema"]
    assert t.estado.enrutador_pedido is True
    assert t.estado.enrutador is False


def test_un_enrutador_que_no_arranca_se_informa_aunque_el_retrieval_este(monkeypatch):
    """Caso distinto del anterior: la híbrida se monta y el enrutador falla."""
    def hibrido_sin_enrutador(chunks, diag=None):
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

    t = _degradar(monkeypatch, buscador_hibrido=lambda chunks, diag=None: None)
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

    t = _degradar(monkeypatch, buscador_hibrido=lambda chunks, diag=None: None)
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
    """Antes una hoja sin pasos no era confirmable. Desde que existe el modo
    «resolver», un enunciado suelto sí lo es —es la otra cosa que un estudiante
    fotografía—, pero sigue sin haber DIAGNÓSTICO: no hay pasos que juzgar, se
    sugiere resolver y el estudiante confirma. Si el texto no es matemáticas
    (esta lista de mercado), lo dice la resolución, no la lectura."""
    t = _tutor_con_lectura(monkeypatch, {
        "ok": True, "consistente": True,
        "json": {"enunciado": "Arroz, leche, huevos", "pasos": []}})
    r = t.leer_imagen(object())
    assert r["pasos"] == []
    assert r["proposito"] == "resolver"
    assert "paso" not in (r["respuesta"] or "").lower().replace("pasos resueltos", "")


def test_una_hoja_sin_enunciado_ni_pasos_no_es_confirmable(monkeypatch):
    t = _tutor_con_lectura(monkeypatch, {
        "ok": True, "consistente": True, "json": {"enunciado": "", "pasos": []}})
    r = t.leer_imagen(object())
    assert r["confirmable"] is False


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
    nada del otro.

    Se leen los PINES, no el texto del archivo. Esta prueba pasaba sola:
    buscaba `peft==0.20.0` como subcadena y lo encontraba en el comentario que
    explica que esa versión no existe en PyPI, así que seguía en verde después
    de cambiar el pin a un rango.
    """
    req = (RAIZ / "M5_despliegue" / "config" /
           "requirements.txt").read_text(encoding="utf-8")
    pins = [l.split("#")[0].strip() for l in req.splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    assert "transformers==4.45.2" in pins
    assert "accelerate==0.34.2" in pins
    peft = next(p for p in pins if p.startswith("peft"))
    assert peft != "peft==0.20.0", "no existe en PyPI"
    assert "<0.21" in peft, peft


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
    # Esta app carga modelos PyTorch; no debe activar integraciones TensorFlow
    # incompatibles del runtime Kaggle al importar Transformers.
    assert 'os.environ.setdefault("USE_TF", "0")' in fuente
    assert fuente.index('os.environ.setdefault("USE_TF", "0")') < pos_pipeline


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


# =========================================================================
# Carga del modelo por etapas · el AttributeError de Kaggle
#
# `AttributeError: module 'torch.distributed' has no attribute 'tensor'` llegaba
# a pantalla recortado a 160 caracteres y sin decir en qué paso ocurría: puede
# venir del tokenizer, del modelo base, del `.to("cuda")`, del adaptador o del
# montaje del retrieval, y cada uno se arregla distinto.
# =========================================================================

def test_una_etapa_que_falla_queda_identificada_con_su_traceback(capsys):
    t = pipeline.Tutor("s08d")
    with pytest.raises(ZeroDivisionError):
        t._etapa("mover a cuda", lambda: 1 / 0)

    assert t.estado.etapa_fallida == "mover a cuda"
    fallo = t.estado.etapas[-1]
    assert fallo["ok"] is False
    assert "ZeroDivisionError" in fallo["error"]
    assert "Traceback" in fallo["traceback"]
    # Y el traceback COMPLETO va al log del proceso, que en Kaggle es la celda.
    assert "FALLO EN LA ETAPA: mover a cuda" in capsys.readouterr().out


def test_una_etapa_que_pasa_se_cronometra():
    t = pipeline.Tutor("s08d")
    assert t._etapa("tokenizer", lambda: "ok") == "ok"
    etapa = t.estado.etapas[-1]
    assert etapa["ok"] and etapa["error"] is None
    assert isinstance(etapa["segundos"], float)
    assert t.estado.etapa_fallida == ""


def test_la_salud_muestra_la_etapa_que_fallo():
    from salud import salud

    t = pipeline.Tutor("s08d")
    with pytest.raises(RuntimeError):
        t._etapa("adaptador LoRA (PeftModel.from_pretrained)",
                 lambda: (_ for _ in ()).throw(RuntimeError("sin memoria")))
    s = salud(t)
    assert s["etapa_fallida"].startswith("adaptador LoRA")
    assert any("FALLA" in linea for linea in s["carga_por_etapas"])
    assert any("sin memoria" in linea for linea in s["traceback"])


def test_cargar_modelo_no_reintenta_tras_fallar_una_etapa(monkeypatch):
    """Sin esto, cada pregunta volvería a intentar la carga y el usuario vería
    la misma espera larga una y otra vez sin más información."""
    t = pipeline.Tutor("s08d")
    t.estado.etapa_fallida = "modelo base"
    llamadas = []
    monkeypatch.setattr(t, "_etapa",
                        lambda n, f: llamadas.append(n))
    t._cargar_modelo()
    assert llamadas == []


# --- torch.distributed.tensor --------------------------------------------

def test_asegurar_distributed_tensor_hace_que_el_atributo_exista():
    """La forma del error lo dice todo: `AttributeError`, no `ModuleNotFound`.
    El subpaquete existe pero `import torch` no lo carga, así que el acceso
    `torch.distributed.tensor.X` falla hasta que alguien lo importa."""
    pytest.importorskip("torch")
    import compat_torch

    info = compat_torch.asegurar_distributed_tensor()
    assert info["torch"] != "ausente"
    if info["distributed_disponible"]:
        assert info["tensor_despues"] is True, info
        assert not info["error"], info["error"]
        import torch.distributed as dist
        assert hasattr(dist, "tensor")


def test_el_diagnostico_de_torch_reporta_lo_necesario():
    pytest.importorskip("torch")
    import compat_torch

    v = compat_torch.versiones()
    for clave in ("python", "torch", "transformers", "peft", "accelerate"):
        assert clave in v, clave
    texto = compat_torch.diagnostico()
    assert "torch.distributed.tensor" in texto
    assert "Accesos por atributo" in texto


def test_el_buscador_de_culpables_ignora_las_cadenas():
    """El propio torch lleva 'torch.distributed.tensor.parallel…' como texto en
    sus registros de API; reportarlo sería ruido justo donde hace falta señal."""
    import compat_torch

    hallazgos = compat_torch.quien_usa_distributed_tensor()
    assert all("_log_api_usage_once" not in h for h in hallazgos), hallazgos


def test_la_app_asegura_torch_antes_de_importar_pipeline():
    fuente = (RAIZ / "M5_despliegue" / "app" / "app.py").read_text(encoding="utf-8")
    assert fuente.index("import compat_torch") < fuente.index("from pipeline import")
    assert "COMPAT_TORCH = compat_torch.asegurar_distributed_tensor()" in fuente


# =========================================================================
# El adaptador lo guardó una PEFT más nueva que la instalable
#
# `adaptadores/qwen-lora/adapter_config.json` trae campos que NINGUNA PEFT
# publicada en PyPI (hasta 0.17.1) conoce: `alora_invocation_tokens`,
# `use_qalora`, `qalora_group_size`, `trainable_token_indices`. Con PEFT 0.13.2
# eso da `TypeError: __init__() got an unexpected keyword argument
# 'alora_invocation_tokens'`. Es lo que obligó a una PEFT muy nueva en Kaggle y
# lo que rompería el Space.
# =========================================================================

def _config_adaptador():
    import json
    ruta = RAIZ / "adaptadores" / "qwen-lora" / "adapter_config.json"
    if not ruta.exists():
        pytest.skip("el adaptador no está en este disco")
    return json.loads(ruta.read_text(encoding="utf-8"))


def test_los_campos_que_sobran_valen_su_valor_por_defecto():
    """La premisa del arreglo: omitirlos no cambia ningún peso. Si alguno
    llevara un valor activo, omitirlo SÍ cambiaría el comportamiento y habría
    que exigir la versión de PEFT, no parchear."""
    import inspect
    pytest.importorskip("peft")
    from peft import LoraConfig

    datos = _config_adaptador()
    admitidos = set(inspect.signature(LoraConfig.__init__).parameters)
    sobrantes = {k: v for k, v in datos.items() if k not in admitidos}
    activos = {k: v for k, v in sobrantes.items()
               if v not in (None, False, 0, "", [])}
    # `peft_version` es metadato: dice con qué versión se guardó, no configura
    # nada. De hecho es el dato que explica todo esto —el adaptador lo escribió
    # PEFT 0.19.1, más nueva que cualquiera publicada en PyPI—.
    assert datos.get("peft_version"), "el adaptador no dice con qué PEFT se guardó"
    activos.pop("peft_version", None)
    # `qalora_group_size` solo cuenta si `use_qalora` está encendido.
    if not datos.get("use_qalora", False):
        activos.pop("qalora_group_size", None)
    assert not activos, (
        f"estos campos desconocidos NO valen su valor por defecto: {activos}. "
        "Omitirlos cambiaría el adaptador; hay que exigir la PEFT que los "
        "entienda en vez de filtrarlos")


def test_el_adaptador_carga_aunque_la_peft_sea_mas_vieja():
    """La verificación de verdad: cargar el adaptador real."""
    pytest.importorskip("peft")
    pytest.importorskip("torch")
    _config_adaptador()

    t = pipeline.Tutor("solo_lora")
    if t.comprobar_adaptador() is None:
        pytest.skip("el adaptador no está donde la configuración lo pide")
    t._cargar_modelo()
    assert t.estado.modelo, t.estado.motivos
    assert t.estado.adaptador_cargado == "m1"
    assert not t.estado.degradado
    assert all(e["ok"] for e in t.estado.etapas), t.estado.etapas


def test_si_hubo_que_filtrar_campos_queda_dicho():
    """Un filtrado silencioso sería indistinguible de que todo encajara."""
    import inspect
    pytest.importorskip("peft")
    from peft import LoraConfig

    datos = _config_adaptador()
    admitidos = set(inspect.signature(LoraConfig.__init__).parameters)
    if set(datos) <= admitidos:
        pytest.skip("esta PEFT entiende el adapter_config entero")

    t = pipeline.Tutor("solo_lora")
    if t.comprobar_adaptador() is None:
        pytest.skip("el adaptador no está donde la configuración lo pide")
    t._cargar_modelo()
    avisos = [m for m in t.estado.motivos if "se omitieron" in m]
    assert avisos, t.estado.motivos
    # El aviso dice con qué versión se guardó y con cuál se leyó: sin las dos,
    # no se sabe si el filtrado fue inevitable o un pin mal puesto.
    assert "guardado con PEFT" in avisos[0] and "leído con" in avisos[0]


# =========================================================================
# El retrieval que no se monta tiene que decir POR QUÉ
#
# Lo que pasó en Kaggle el 1 de noviembre de 2026: el tutor respondió
# correctamente `7/8` a «¿Cuánto es 3/4 + 1/8?» y la pestaña Estado decía
# `degradado: True`, `faltantes: retrieval híbrido + reranker`. Nada más. El
# aviso era «falta rank_bm25, sentence-transformers o los modelos en caché», y
# los tres paquetes estaban instalados: el mensaje era falso en sus dos primeras
# opciones y mudo en la tercera.
#
# La causa quedaba oculta porque `buscador_hibrido` tenía un `except ImportError`
# y un `except Exception` que devolvían la misma None. Estas pruebas fijan que
# ahora se registre la etapa, el error, la causa probable y el traceback.
# =========================================================================

def test_el_retrieval_que_no_se_monta_dice_en_que_etapa_y_con_que_error(monkeypatch):
    """Una None sin explicación obliga a adivinar entre cinco causas distintas."""
    # Con None en sys.modules, el `from rank_bm25 import ...` lanza ImportError
    # tanto si el paquete está instalado como si no: la prueba vale en los dos
    # entornos, y no depende de que falte una dependencia.
    monkeypatch.setitem(sys.modules, "rank_bm25", None)

    t = pipeline.Tutor("s08d")
    t.resolver_componentes()

    assert t.estado.degradado
    assert "retrieval híbrido + reranker" in t.estado.faltantes
    diag = t.estado.retrieval_diagnostico
    assert diag, "sin diagnóstico no se sabe en qué etapa se rompió la cadena"
    fallo = next(x for x in diag if not x["ok"])
    assert "importar" in fallo["etapa"]
    assert "rank_bm25" in fallo["error"]
    assert "dependencia" in fallo["causa_probable"]
    assert "ImportError" in fallo["traceback"] or "rank_bm25" in fallo["traceback"]
    # Y el aviso que se enseña ya no es la lista de sospechosos.
    aviso = next(m for m in t.estado.motivos if m.startswith("retrieval:"))
    assert "rank_bm25" in aviso and "causa probable" in aviso


def test_la_salud_publica_el_diagnostico_del_retrieval(monkeypatch):
    """La pestaña Estado es donde se mira en Kaggle; si no sale ahí, no sirve."""
    monkeypatch.setitem(sys.modules, "rank_bm25", None)
    t = pipeline.Tutor("s08d")
    t.resolver_componentes()

    d = salud(t)
    assert "retrieval_por_etapas" in d
    assert any("FALLA" in linea for linea in d["retrieval_por_etapas"])
    assert "dependencia" in d["retrieval_causa_probable"]
    assert d["retrieval_traceback"], "el traceback es lo único accionable"
    assert "retrieval_por_etapas" in texto_salud(t)


def test_el_diagnostico_no_publica_credenciales():
    """Los errores del hub citan URLs y cabeceras, y esto se pega en informes."""
    sucio = ("OSError: 401 en https://huggingface.co/api/models · "
             "Authorization: Bearer hf_abcdefghijklmnop1234 · token=hf_otracosa99")
    limpio = pipeline._redactar(sucio)
    assert "hf_abcdefghijklmnop1234" not in limpio
    assert "hf_otracosa99" not in limpio
    assert "[REDACTADO]" in limpio
    assert "huggingface.co/api/models" in limpio, "la URL sí hace falta verla"


def test_la_causa_probable_separa_la_red_de_la_version():
    """Son dos arreglos opuestos: con una, reinstalar no sirve; con la otra,
    tener internet tampoco. Confundirlas cuesta una sesión de GPU."""
    casos = {
        ImportError("No module named 'rank_bm25'"): "dependencia",
        MemoryError(): "memoria",
        OSError("Max retries exceeded with url: https://huggingface.co"): "red",
        OSError("We couldn't connect to huggingface.co"): "red",
        TypeError("__init__() got an unexpected keyword argument"): "versión",
        AttributeError("module 'torch.distributed' has no attribute 'tensor'"):
            "versión",
    }
    for excepcion, esperado in casos.items():
        assert esperado in pipeline._causa_probable(excepcion), excepcion


def test_el_buscador_hibrido_prueba_una_consulta_antes_de_declararse_montado():
    """Que los constructores no lancen no significa que la cadena responda: sin
    esta etapa, la primera pregunta del estudiante era la que lo descubría."""
    fuente = (RAIZ / "M5_despliegue" / "app" / "pipeline.py").read_text(
        encoding="utf-8")
    inicio = fuente.index("def buscador_hibrido(")
    cuerpo = fuente[inicio:fuente.index("def buscador_lexico(")]
    assert "consulta de prueba contra el corpus" in cuerpo
    # Y codificar el corpus ocurre DENTRO de una etapa: antes estaba fuera de
    # todo manejo de errores y un fallo ahí no degradaba el retrieval, tumbaba
    # la carga entera del tutor.
    assert 'etapa(f"codificar los' in cuerpo
    assert "except Exception:      # noqa: BLE001" not in cuerpo


def test_el_enrutador_no_se_monta_sobre_la_busqueda_lexica():
    """El tercer rasgo de S08-D es el puntaje del cross-encoder. Un enrutador
    sin `encode` ni `rerank_max` no es el que se midió, así que no se monta."""
    chunks, _m, _i = pipeline.cargar_corpus()
    lexico = pipeline.buscador_lexico(chunks)
    assert getattr(lexico, "encode", None) is None
    assert getattr(lexico, "rerank_max", None) is None

    t = pipeline.Tutor("s08d")
    assert t._montar_enrutador(lexico) is None
    assert any("cross-encoder" in m for m in t.estado.motivos)


# --- la validación distingue instalado / cargable / montado / efectivo ----

def test_la_validacion_no_da_el_retrieval_por_montado_con_solo_los_imports():
    """El error de la sesión anterior: «están los tres paquetes, luego el
    retrieval está listo». Los cuatro niveles existen para que no se repita."""
    sys.path.insert(0, str(RAIZ / "M5_despliegue" / "evaluacion"))
    import validacion_m5

    req = validacion_m5.estado_requisitos()
    nivel1 = next(k for k in req if "nivel 1" in k)
    assert "NO implica montado" in nivel1
    niveles = next(k for k in req if "niveles 2-4" in k)
    # Sin una corrida del diagnóstico no se afirma nada.
    assert isinstance(req[niveles], str)
    assert ("NO COMPROBADO" in req[niveles] or "nivel" in req[niveles]
            or "cuatro niveles pasan" in req[niveles])
    # Y la clave antigua, que mezclaba los cuatro en un booleano, ya no está.
    assert "retrieval (sentence-transformers, rank_bm25, sklearn)" not in req


def test_el_perfil_de_kaggle_no_baja_ninguna_version():
    """Comprobado en la T4 el 2/11/2026: la imagen de Kaggle importa
    `sentence_transformers` sin un fallo, y lo que rompía el retrieval eran
    nuestras propias bajadas. `numpy==1.26.4` deja a jax sin
    `numpy.dtypes.StringDType`, que existe desde numpy 2.0; el fallo sube por
    tensorflow y `transformers.modeling_tf_utils` hasta el import. Así que en
    Kaggle solo se instala lo que de verdad falta."""
    kaggle = (RAIZ / "M5_despliegue" / "config" /
              "requirements-kaggle.txt").read_text(encoding="utf-8")
    pins = [l.split("#")[0].strip() for l in kaggle.splitlines()
            if l.strip() and not l.lstrip().startswith("#")]
    assert pins == ["rank-bm25==0.2.2"], (
        "el perfil de Kaggle instala de más: " + ", ".join(pins))
    # El error exacto queda escrito, no resumido: es la prueba del perfil.
    assert "StringDType" in kaggle
    assert "pip uninstall -y torchao" in kaggle


def test_el_perfil_del_space_no_se_usa_en_kaggle():
    """`requirements.txt` es correcto para el Space —torch 2.4.1 con numpy
    1.26.4— y es justo el que rompe Kaggle. Los dos perfiles existen porque
    describen dos máquinas distintas, no por duplicación."""
    space = (RAIZ / "M5_despliegue" / "config" / "requirements.txt").read_text(
        encoding="utf-8")
    kaggle = (RAIZ / "M5_despliegue" / "config" /
              "requirements-kaggle.txt").read_text(encoding="utf-8")
    pins_space = [l.split("#")[0].strip() for l in space.splitlines()
                  if l.strip() and not l.lstrip().startswith("#")]
    assert any(p.startswith("numpy==1.26") for p in pins_space), (
        "el Space sí fija numpy; si eso cambia, este análisis hay que rehacerlo")
    assert "numpy" not in [p.split("==")[0] for p in
                           kaggle.splitlines() if p and not p.startswith("#")]


def test_el_guion_de_kaggle_dice_como_distinguir_completo_de_degradado():
    """Sin ese criterio escrito, una respuesta correcta se lee como un sistema
    completo: es literalmente lo que pasó con el `7/8`."""
    guion = (RAIZ / "M5_despliegue" / "docs" / "GUION_KAGGLE.md").read_text(
        encoding="utf-8")
    assert "diagnostico_retrieval.py" in guion
    assert "degradado" in guion
    assert "requirements-kaggle.txt" in guion
    # Y el paso 0 ya no pide un commit que ya está hecho.
    assert "git commit" not in guion.split("## Contexto")[0]


# --------------------------------------------------------------------------
# La vía aritmética directa
#
# Regresión de la fase 2. `s08d_verificador` monta las 18 herramientas, pero el
# adaptador `m1` se entrenó para RAG, no para emitir `<tool_call>`: ante
# «¿Cuánto es 3/4 + 1/8?» contestaba `0.75` de cabeza y la traza registraba
# cero llamadas. Estas pruebas corren sin GPU: la vía directa responde antes de
# cargar ningún modelo.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pregunta, cuenta", [
    ("¿Cuánto es 3/4 + 1/8?", "3/4 + 1/8"),
    ("347 × 28", "347 × 28"),
    ("Calcula 1764 ÷ 7", "1764 ÷ 7"),
    ("cuanto es 2+2?", "2+2"),
    ("resultado de (5+3)*2", "(5+3)*2"),
])
def test_reconoce_una_cuenta(pregunta, cuenta):
    assert pipeline.cuenta_de(pregunta) == cuenta


@pytest.mark.parametrize("pregunta", [
    "Un bus recorre 68 km cada día. ¿Cuántos kilómetros recorre en 14 días?",
    "¿Cuál es la nota mínima aprobatoria del colegio?",
    "Explica por qué 9 es un número primo.",
    "3/4",
    "9716",
    "",
])
def test_no_confunde_un_enunciado_con_una_cuenta(pregunta):
    assert pipeline.cuenta_de(pregunta) is None


@pytest.mark.parametrize("pregunta, valor, herramienta", [
    ("¿Cuánto es 3/4 + 1/8?", "7/8", "operar_fracciones"),
    ("347 × 28", "9716", "evaluar_expresion"),
])
def test_la_cuenta_la_ejecuta_una_herramienta_de_verdad(pregunta, valor, herramienta):
    r = pipeline.Tutor("s08d_verificador").responder(pregunta)
    assert r["ok"]
    assert r["herramientas"] == [herramienta]
    assert r["observaciones"][0]["ok"]
    assert r["observaciones"][0]["valor"] == valor
    assert valor in r["respuesta"]
    assert r["pasos_llm"] == 0


def test_la_linea_base_medida_no_cambia_de_camino():
    """`s08d` se midió 18/21 SIN herramientas. Aquí no entra la vía directa."""
    assert pipeline.Tutor("s08d")._aritmetica_directa("347 × 28") is None


def test_dos_preguntas_seguidas_no_comparten_respuesta():
    """Regresión: la segunda consulta mostraba la respuesta de la primera."""
    t = pipeline.Tutor("s08d_verificador")
    a = t.responder("347 × 28")
    b = t.responder("¿Cuánto es 3/4 + 1/8?")
    assert "9716" in a["respuesta"] and "7/8" not in a["respuesta"]
    assert "7/8" in b["respuesta"] and "9716" not in b["respuesta"]


def test_el_boton_vacia_las_cajas_antes_de_responder():
    """Si el evento de respuesta muere, queda vacío, no lo de la vez anterior."""
    app = pytest.importorskip("app")
    assert app._vaciar_texto() == ("", "")


def test_la_respuesta_no_ensena_el_razonamiento_interno():
    app = pytest.importorskip("app")
    crudo = "Pensamiento: Necesitamos multiplicar el recorrido por 14.\nRespuesta final: 952"
    assert app.presentar(crudo) == "Respuesta final: 952"
    assert "Pensamiento" not in app.presentar(crudo)


def test_la_foto_se_normaliza_igual_que_en_el_evaluador():
    """La app pasaba el PIL crudo de Gradio y el evaluador reducía a 1024: la
    misma hoja se rechazaba en una y se leía en el otro."""
    pytest.importorskip("PIL")
    from PIL import Image

    grande = Image.new("RGB", (4160, 1345), "white")
    chica, info = pipeline.normalizar_foto(grande)
    assert max(chica.size) == pipeline.LADO_MAYOR_FOTO
    assert info["px_antes"] == [4160, 1345]
    assert info["px_despues"] == list(chica.size)
    # Una imagen que ya cabe no se toca.
    pequena, info2 = pipeline.normalizar_foto(Image.new("RGB", (800, 600), "white"))
    assert pequena.size == (800, 600)
    assert info2["px_despues"] == [800, 600]


def test_la_revision_ensena_la_comprobacion_y_avisa_si_no_coincide():
    """En m5v-03 el tutor dijo «Paso 2» y el verificador, «paso 3». El informe
    guardaba las dos cosas y la pantalla solo enseñaba la primera."""
    app = pytest.importorskip("app")
    r = {"etapa": "diagnostico", "respuesta": "Diagnóstico: Paso 2",
         "verificador_aritmetico": {"paso": 3, "tipo": "traslado",
                                    "detalle": "el paso 3 copia mal el resultado: "
                                               "era 39000 y está escrito 38000"}}
    salida = app.presentar_revision(r)
    assert "Paso 2" in salida
    assert "39000" in salida and "38000" in salida
    assert "paso 3" in salida
    # y lo dice: las dos no coinciden
    assert "no se equivoca sumando" in salida


def test_sin_hallazgo_aritmetico_lo_dice_en_vez_de_callarse():
    app = pytest.importorskip("app")
    r = {"etapa": "diagnostico", "respuesta": "Diagnóstico: Paso 1",
         "verificador_aritmetico": {"paso": None, "tipo": "ninguno",
                                    "detalle": "todas las cuentas escritas cuadran"}}
    salida = app.presentar_revision(r)
    assert "planteamiento" in salida


# --------------------------------------------------------------------------
# Foto con solo el enunciado: resolver, no revisar
# --------------------------------------------------------------------------

def test_explica_la_suma_de_fracciones_como_lo_haria_un_profesor():
    pasos, obs = pipeline.explicar_cuenta("3/4 + 1/8")
    texto = "\n".join(pasos)
    assert "3/4 = 6/8" in texto
    assert "6/8 + 1/8 = 7/8" in texto
    assert pasos[-1] == "Respuesta final: 7/8"
    assert obs[0]["herramienta"] == "operar_fracciones" and obs[0]["valor"] == "7/8"


def test_explica_un_producto_descomponiendo_y_lo_confirma_la_herramienta():
    pasos, obs = pipeline.explicar_cuenta("347 × 28")
    assert "347 × 20 = 6940." in pasos and "347 × 8 = 2776." in pasos
    assert "6940 + 2776 = 9716." in pasos
    assert obs[0]["herramienta"] == "evaluar_expresion" and obs[0]["valor"] == "9716"


def test_resolver_sin_confirmar_se_niega():
    r = pipeline.Tutor("revision_foto").resolver_enunciado("¿Cuánto es 3/4 + 1/8?")
    assert not r["ok"] and "confirm" in r["respuesta"]


def test_resolver_una_cuenta_no_carga_el_modelo_y_registra_la_herramienta():
    r = pipeline.Tutor("revision_foto").resolver_enunciado(
        "¿Cuánto es 3/4 + 1/8?", confirmado=True)
    assert r["ok"] and r["camino"] == "herramienta" and r["pasos_llm"] == 0
    assert r["herramientas"] == ["operar_fracciones"]
    assert r["verificador_aritmetico"]["tipo"] == "ninguno"


def test_la_solucion_dice_como_se_comprobo():
    app = pytest.importorskip("app")
    r = pipeline.Tutor("revision_foto").resolver_enunciado(
        "¿Cuánto es 3/4 + 1/8?", confirmado=True)
    salida = app.presentar_solucion(r)
    assert "6/8 + 1/8 = 7/8" in salida and "operar_fracciones" in salida
    assert "Pensamiento" not in salida


def test_si_el_modelo_escribe_una_cuenta_mal_la_solucion_lo_avisa():
    app = pytest.importorskip("app")
    r = {"etapa": "solucion", "camino": "modelo", "enunciado": "bus",
         "pasos_solucion": ["Paso 1: 68 × 14 = 942", "Respuesta final: 942"],
         "verificador_aritmetico": {"paso": 1, "tipo": "calculo",
                                    "detalle": "el paso 1 no cuadra: 68 × 14 da 952"}}
    salida = app.presentar_solucion(r)
    assert "NO debes darla por buena" in salida and "952" in salida


def test_dos_lecturas_de_un_enunciado_con_otra_operacion_no_coinciden():
    """Sin pasos, la huella antigua quedaba vacía: «3/4 + 1/8» y «3/4 - 1/8»
    pasaban por la misma lectura."""
    from vision import lecturas_consistentes
    a = {"json": {"enunciado": "¿Cuánto es 3/4 + 1/8?", "pasos": [], "respuesta_final": ""}}
    b = {"json": {"enunciado": "¿Cuánto es 3/4 - 1/8?", "pasos": [], "respuesta_final": ""}}
    c = {"json": {"enunciado": "¿cuanto es 3/4 + 1/8 ?", "pasos": [], "respuesta_final": ""}}
    assert not lecturas_consistentes(a, b)
    assert lecturas_consistentes(a, c)


def test_la_traza_dice_en_que_difieren_dos_lecturas_sin_ensenar_el_texto():
    from vision import diferencias_de_lectura
    a = {"json": {"enunciado": "3/4 + 1/8", "pasos": [], "respuesta_final": ""}}
    b = {"json": {"enunciado": "3/4 - 1/8", "pasos": [], "respuesta_final": ""}}
    d = diferencias_de_lectura(a, b)
    assert d == ["operaciones del enunciado"]
    assert diferencias_de_lectura(a, a) == []


# --------------------------------------------------------------------------
# Lectura por líneas: el VLM copia, el código separa enunciado y pasos
# --------------------------------------------------------------------------

def test_una_hoja_con_solo_enunciado_queda_sin_pasos():
    from vision import estructurar_lineas
    js = estructurar_lineas(["7. Un paquete trae 18 galletas.",
                             "¿Cuántas galletas hay en 15 paquetes?"])
    assert js["pasos"] == [] and js["respuesta_final"] is None
    assert js["enunciado"] == ("Un paquete trae 18 galletas. "
                               "¿Cuántas galletas hay en 15 paquetes?")


def test_una_hoja_con_procedimiento_separa_enunciado_y_pasos():
    from vision import estructurar_lineas
    js = estructurar_lineas(["Una camiseta cuesta 52000 pesos",
                             "Paso 1: El descuento es 52000 x 0.25 = 13000",
                             "Paso 2: Resto 52000 - 13000 = 39000",
                             "Paso 3: Respuesta Final: 38000 pesos"])
    assert js["enunciado"] == "Una camiseta cuesta 52000 pesos"
    assert len(js["pasos"]) == 3 and js["respuesta_final"] == "38000 pesos"


def test_las_instrucciones_de_m4_no_cambiaron():
    """`nb_m4.py` reproduce M4 con ellas: tocarlas cambia un resultado ya medido."""
    import vision
    assert '"pasos": []' not in vision.INSTRUCCION_VER
    assert '"pasos": []' not in vision.INSTRUCCION_VER_INDEPENDIENTE
    # Y las de M5 no le piden al VLM que decida qué es un paso.
    assert "pasos" not in vision.INSTRUCCION_COPIAR
    assert "pasos" not in vision.INSTRUCCION_COPIAR_INDEPENDIENTE


def test_recupera_las_lineas_de_un_json_sin_cerrar():
    """Caso real de la sonda: una copia correcta llegó sin el `]` final."""
    from vision import lineas_de
    crudo = '```json\n{\n  "lineas": ["7. un paquete trae 18 galletas; cuantas galletas hay en 15 paquetes?"\n}\n```'
    assert lineas_de(crudo) == ["7. un paquete trae 18 galletas; cuantas galletas hay en 15 paquetes?"]
    assert lineas_de('{"ilegible": true}') is None


def test_las_lineas_de_relleno_no_cuentan():
    from vision import estructurar_lineas, lecturas_consistentes
    a = {"json": {"lineas": ["8. De un rollo de 7/8 de metro se usan 3/8 ¿cuánto queda?"]}}
    b = {"json": {"lineas": ["8. De un rollo de 7/8 de metro se usan 3/8 cuánto queda?", "...", "..."]}}
    assert lecturas_consistentes(a, b)
    assert estructurar_lineas(b["json"]["lineas"])["pasos"] == []


def test_partir_una_linea_en_dos_no_es_otra_lectura_pero_otro_numero_si():
    """`vis-10`, foto borrosa: una copia leyó 306 y la otra 312. Eso tiene que
    parar la hoja; que una copia parta una línea en dos, no."""
    from vision import lecturas_consistentes
    a = {"json": {"lineas": ["Paso 1: lo que queda se obtiene sumando 1500 + 847"]}}
    b = {"json": {"lineas": ["Paso 1: lo que queda se obtiene", "sumando 1500 + 847"]}}
    assert lecturas_consistentes(a, b)
    c = {"json": {"lineas": ["24 x 13 = 306"]}}
    d = {"json": {"lineas": ["24 x 13 = 312"]}}
    assert not lecturas_consistentes(c, d)


def test_el_verificador_divide_fracciones_bien():
    """Caso real de la sonda: el verificador dijo «7/8 ÷ 3/8 da 7/192» y marcó
    como error un paso aritméticamente correcto (7/3)."""
    pytest.importorskip("sympy")
    from verificador_aritmetico import revisar_hoja
    h = revisar_hoja(["Dividimos: 7/8 ÷ 3/8 = 7/3."])
    assert h.tipo == "ninguno", h.detalle
    h = revisar_hoja(["7/8 ÷ 3/8 = 7/192"])
    assert h.tipo == "calculo" and "7/3" in h.detalle


def test_recupera_lineas_envueltas_en_objetos_mal_formados():
    """Caso real (m5v-03, segunda lectura): claves repetidas que `json` colapsa."""
    from vision import lineas_de
    crudo = ('{"lineas": [{"texto": "Una camiseta cuesta 52000 pesos", '
             '"numero": "Paso 1: El descuento es 52000 x 0.25 = 13000", '
             '"numero": "Paso 2: Resto 52000 - 13000 = 39000"}]}')
    assert lineas_de(crudo) == ["Una camiseta cuesta 52000 pesos",
                                "Paso 1: El descuento es 52000 x 0.25 = 13000",
                                "Paso 2: Resto 52000 - 13000 = 39000"]



def test_una_palabra_distinta_para_la_hoja_igual_que_un_numero():
    """Antes (1bab37e) las palabras no se comparaban. Ahora sí, por decisión del
    responsable del proyecto: una lectura en la que cambia una palabra no es la
    misma lectura. Números, operaciones y cantidad de pasos, igual que antes."""
    from vision import lecturas_consistentes
    L = lambda *x: {"json": {"lineas": list(x)}}
    assert not lecturas_consistentes(L("¿Cuánto se paga?", "Paso 1: 52000 x 0.25 = 13000"),
                                     L("Cuantos se paga?", "Paso 1: 52000 x 0.25 = 13000"))
    assert lecturas_consistentes(L("¿Cuánto se paga?", "Paso 1: 52000 x 0.25 = 13000"),
                                 L("cuanto se paga", "Paso 1: 52000 × 0,25 = 13000."))
    assert not lecturas_consistentes(L("Paso 1: 52000 x 0.25 = 13000"),
                                     L("Paso 1: 52000 x 0.35 = 13000"))
    assert not lecturas_consistentes(L("Paso 1: 3 + 4 = 7"), L("Paso 1: 3 - 4 = 7"))
    assert not lecturas_consistentes(L("Paso 1: 3 + 4 = 7", "Paso 2: 7 x 2 = 14"),
                                     L("Paso 1: 3 + 4 = 7"))


def test_sin_ningun_numero_no_se_resuelve_nada():
    """m5v-11, lista de mercado: el modelo inventó «68 ÷ 2 = 34»."""
    r = pipeline.Tutor("revision_foto").resolver_enunciado(
        "Pan Queso Dos litros de leche Jabón", confirmado=True)
    assert r["ok"] and r["camino"] == "ninguno" and r["pasos_llm"] == 0
    assert "no es" in r["respuesta"].lower() or "no veo" in r["respuesta"].lower()


# --------------------------------------------------------------------------
# Doble lectura estricta: palabras Y contenido matemático
# --------------------------------------------------------------------------

def _lectura_lineas(*lineas):
    from vision import estructurar_lineas
    return {"json": estructurar_lineas(list(lineas)) | {"lineas": list(lineas)}}


_HOJA = ("Una camiseta cuesta 52000 pesos y tiene 25% de descuento. ¿Cuánto se paga?",
         "Paso 1: El descuento es 52000 x 0.25 = 13000",
         "Paso 2: Resto el descuento 52000 - 13000 = 39000",
         "Paso 3: Respuesta Final: 38000 pesos")


def test_doble_lectura_identicas_coinciden():
    from vision import diferencias_de_lectura
    assert diferencias_de_lectura(_lectura_lineas(*_HOJA), _lectura_lineas(*_HOJA)) == []


def test_doble_lectura_puntuacion_acentos_y_saltos_no_cuentan():
    from vision import diferencias_de_lectura
    otra = ("una camiseta cuesta 52000 pesos y tiene 25 % de descuento cuanto se paga",
            "Paso 1: el descuento es 52000 × 0,25 = 13000.",
            "Paso 2: resto el descuento 52000 - 13000 = 39000",
            "Paso 3: respuesta final: 38000 pesos")
    assert diferencias_de_lectura(_lectura_lineas(*_HOJA), _lectura_lineas(*otra)) == []
    # Toda la hoja en una sola línea es la misma lectura.
    assert diferencias_de_lectura(_lectura_lineas(*_HOJA),
                                  _lectura_lineas(" ".join(_HOJA))) == []


def test_doble_lectura_una_palabra_distinta_para_la_hoja():
    from vision import diferencias_de_lectura
    otra = (_HOJA[0].replace("¿Cuánto", "Cuantos"),) + _HOJA[1:]
    assert diferencias_de_lectura(_lectura_lineas(*_HOJA),
                                  _lectura_lineas(*otra)) == ["palabras del enunciado"]


def test_doble_lectura_un_numero_distinto_para_la_hoja():
    from vision import diferencias_de_lectura
    otra = (_HOJA[0], _HOJA[1].replace("0.25", "0.35")) + _HOJA[2:]
    assert diferencias_de_lectura(_lectura_lineas(*_HOJA),
                                  _lectura_lineas(*otra)) == ["números de los pasos"]


# --------------------------------------------------------------------------
# Verificador: igualdades encadenadas, cada una por separado
# --------------------------------------------------------------------------

def test_verificador_cadena_de_fracciones_correcta():
    pytest.importorskip("sympy")
    from verificador_aritmetico import igualdades_de, revisar_hoja
    paso = "Dividimos: 7/8 ÷ 3/8 = 7/8 × 8/3 = 56/24"
    assert igualdades_de(paso) == [("7/8 ÷ 3/8", "7/8 × 8/3"), ("7/8 × 8/3", "56/24")]
    assert revisar_hoja([paso]).tipo == "ninguno"
    assert revisar_hoja(["7/8 ÷ 3/8 = 7/3"]).tipo == "ninguno"


def test_verificador_cadena_con_una_igualdad_falsa_dice_cual():
    pytest.importorskip("sympy")
    from verificador_aritmetico import revisar_hoja
    h = revisar_hoja(["3/4 + 1/8 = 6/8 + 1/8 = 8/8"])
    assert h.tipo == "calculo" and h.paso == 1
    assert "6/8 + 1/8" in h.detalle and "7/8" in h.detalle and "8/8" in h.detalle


def test_verificador_cadena_equivalente_de_otra_forma():
    pytest.importorskip("sympy")
    from verificador_aritmetico import revisar_hoja
    assert revisar_hoja(["3/4 + 1/8 = 0.75 + 0.125 = 0,875 = 7/8"]).tipo == "ninguno"


def test_verificador_no_marca_un_paso_correcto():
    pytest.importorskip("sympy")
    from verificador_aritmetico import revisar_hoja
    for paso in ["Paso 2: 18 × 15 = 270 galletas", "El descuento es 52000 x 0.25 = 13000.",
                 "40 × 0,35 = 14", "Llevamos a octavos: 3/4 = 6/8"]:
        assert revisar_hoja([paso]).tipo == "ninguno", paso


# --------------------------------------------------------------------------
# Diagnóstico combinado: tutor + verificador
# --------------------------------------------------------------------------

def _decidir(pasos, conclusion_modelo):
    pytest.importorskip("sympy")
    from verificador_aritmetico import pasos_comprobados, revisar_hoja
    from vision import decidir_diagnostico
    h = revisar_hoja(pasos)
    return decidir_diagnostico(conclusion_modelo, {"paso": h.paso, "tipo": h.tipo,
                                                   "detalle": h.detalle},
                               pasos_comprobados(pasos))


def test_el_verificador_manda_sobre_la_aritmetica():
    d = _decidir(["52000 × 0.25 = 13000", "52000 − 13000 = 39000",
                  "Respuesta final: 38000 pesos"], "paso 2")
    assert d["conclusion"] == "paso 3" and d["desacuerdo"] and d["fuente"] == "verificador"


def test_un_planteamiento_anterior_sin_cuentas_lo_decide_el_tutor():
    d = _decidir(["Para saber lo que queda, sumo: 1240 + 365", "1240 + 365 = 1605",
                  "Respuesta final: 1605 sillas"], "paso 1")
    assert d["conclusion"] == "paso 1" and d["fuente"] == "modelo"


def test_no_se_inventa_un_error_en_un_paso_comprobado():
    d = _decidir(["Multiplico 18 × 15", "18 × 15 = 270", "Respuesta final: 270"], "paso 2")
    assert d["conclusion"] == "sin error" and d["desacuerdo"]


def test_las_abstenciones_del_tutor_se_respetan():
    d = _decidir(["8500 + 12300 = 20800", "Respuesta final: 20800 pesos"],
                 "dato no disponible")
    assert d["conclusion"] == "dato no disponible"
