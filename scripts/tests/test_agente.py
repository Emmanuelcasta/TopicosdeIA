"""
Tests del paquete `scripts/agente` (sin GPU, sin modelos).

    python -m pytest scripts/tests -q

El LLM se sustituye por un guion: una lista de salidas que el "modelo" devuelve
en orden. Así se prueba el bucle agéntico completo (llamadas, observaciones,
errores de formato, verificación, tope de pasos) de forma determinista.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "agente"))

from herramientas import REGISTRO  # noqa: E402
from metricas import autoconsistencia, mcnemar_exacto, metricas_herramientas_caso, resumen_agente, robustez_mal_escrito  # noqa: E402
from nucleo import AgenteTutor, Normalizador, analizar_salida, numeros_en, valor_final  # noqa: E402
from ragas_tutor import (  # noqa: E402
    afirmaciones, average_precision, context_precision_oro, context_recall_oro,
    faithfulness, faithfulness_numerica, kappa_cohen, veredictos,
)
from registro import LlamadaHerramienta, RegistroHerramientas  # noqa: E402


def tool_call(nombre, **args):
    return f"<tool_call>\n{json.dumps({'name': nombre, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"


class Guion:
    """LLM falso: devuelve las salidas en orden y registra qué herramientas vio."""
    def __init__(self, *salidas):
        self.salidas, self.vistas, self.mensajes = list(salidas), [], []

    def __call__(self, mensajes, esquemas):
        self.vistas.append(None if esquemas is None else [e["function"]["name"] for e in esquemas])
        self.mensajes.append(list(mensajes))
        return self.salidas.pop(0)


# --------------------------------------------------------------------------
# Registro y herramientas
# --------------------------------------------------------------------------

def test_esquemas_generados_desde_tipos_y_docstring():
    e = REGISTRO.obtener("dividir").esquema()["function"]
    assert e["parameters"]["required"] == ["dividendo", "divisor"]
    assert e["parameters"]["properties"]["redondeo"]["enum"] == ["ninguno", "arriba", "abajo"]
    assert all(p["description"] for p in e["parameters"]["properties"].values())
    assert REGISTRO.obtener("sumar").esquema()["function"]["parameters"]["properties"]["numeros"]["type"] == "array"


def test_anadir_herramienta_sin_tocar_el_nucleo():
    reg = RegistroHerramientas()

    @reg.herramienta(familia="aritmetica")
    def mcd(a: int, b: int) -> dict:
        """Máximo común divisor de dos enteros.

        Args:
            a: primer entero.
            b: segundo entero.
        """
        import math
        return {"resultado": str(math.gcd(a, b)), "valor": str(math.gcd(a, b))}

    r = reg.ejecutar(LlamadaHerramienta("mcd", {"a": "1386", "b": 2310}))
    assert r.ok and r.valor == "462"
    llm = Guion(tool_call("mcd", a=12, b=18), "Respuesta final: 6")
    out = AgenteTutor(llm, "sys", registro=reg, max_rondas=3)("mcd de 12 y 18")
    assert out.respuesta.endswith("6") and llm.vistas[0] == ["mcd"]


def test_herramienta_sin_docstring_de_argumentos_se_rechaza():
    reg = RegistroHerramientas()
    with pytest.raises(ValueError):
        @reg.herramienta(familia="aritmetica")
        def mala(a: int) -> int:
            """Hace algo."""
            return a


@pytest.mark.parametrize("nombre,args,valor", [
    ("multiplicar", {"numeros": [3847, 296]}, "1138712"),
    ("evaluar_expresion", {"expresion": "3847 x 296"}, "1138712"),
    ("operar_fracciones", {"fraccion_a": "17/24", "fraccion_b": "11/36", "operacion": "suma"}, "73/72"),
    ("operar_fracciones", {"fraccion_a": "5/6", "fraccion_b": "15/28", "operacion": "division"}, "14/9"),
    ("raiz", {"radicando": 7056}, "84"),
    ("evaluar_expresion", {"expresion": "2.35 × 4.8 − 1.964"}, "9.316"),
    ("resolver_ecuacion", {"ecuacion": "7x − 23 = 4x + 58"}, "27"),
    ("evaluar_expresion", {"expresion": "15³ − 12^4"}, "-17361"),
    ("dividir", {"dividendo": 98532, "divisor": 12}, "8211"),
    ("simplificar_fraccion", {"fraccion": "1386/2310"}, "3/5"),
    ("evaluar_expresion", {"expresion": "0.0375 ÷ 0.0015 + √2025"}, "70"),
    ("operar_expresion", {"expresion": "(2x + 3)(x − 5)", "operacion": "expandir"}, "2x^2 - 7x - 15"),
    ("dividir", {"dividendo": 1250, "divisor": 180, "redondeo": "arriba"}, "7"),
    ("valor_antes_de_porcentaje", {"valor_final": 92000, "porcentaje": 15, "operacion": "aumento"}, "80000"),
    ("aplicar_porcentaje", {"cantidad": "200000", "porcentaje": "20", "operacion": "descuento"}, "160000"),
    ("promedio_ponderado", {"valores": [4.0, 3.5, 3.0, 4.5], "pesos": [40, 25, 20, 15]}, "3.75"),
    ("evaluar_expresion", {"expresion": "(5/8)*(4/7)"}, "5/14"),
    ("restar", {"minuendo": "3,0", "sustraendo": 2.65}, "0.35"),
    ("porcentaje_de", {"porcentaje": 1.8, "cantidad": 2450000}, "44100"),
])
def test_herramientas_resuelven_los_casos_del_eval_set(nombre, args, valor):
    r = REGISTRO.ejecutar(LlamadaHerramienta(nombre, args))
    assert r.ok, r.error
    assert r.valor == valor


@pytest.mark.parametrize("nombre,args,tipo", [
    ("dividir", {"dividendo": 15, "divisor": 0}, "dominio"),
    ("evaluar_expresion", {"expresion": "15/0"}, "dominio"),
    ("raiz", {"radicando": -4}, "dominio"),
    ("evaluar_expresion", {"expresion": "__import__('os').system('ls')"}, "dominio"),
    ("evaluar_expresion", {"expresion": "open('x')"}, "dominio"),
    ("sumar", {"nums": [1, 2]}, "argumentos"),
    ("dividir", {"dividendo": "abc", "divisor": 2}, "argumentos"),
    ("no_existe", {}, "desconocida"),
])
def test_errores_se_devuelven_como_observacion(nombre, args, tipo):
    r = REGISTRO.ejecutar(LlamadaHerramienta(nombre, args))
    assert not r.ok and r.tipo_error == tipo
    assert json.loads(r.como_observacion())["ok"] is False


# --------------------------------------------------------------------------
# Normalizador
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entrada,contiene", [
    ("cuanto es 3847 x 296", "3847 × 296"),
    ("cuanto es 17/24 mas 11/36", "17/24 + 11/36"),
    ("cuanto da 15 dividido en 0", "15 ÷ 0"),
    ("cual es la nota minima pa pasar y cuantas recuperaciones ai x periodo", "para pasar"),
    ("raiz cuadrada de 7056??", "7056?"),
])
def test_reglas_del_normalizador(entrada, contiene):
    n = Normalizador()(entrada)
    assert contiene in n.normalizada
    assert sorted(numeros_en(n.normalizada)) == sorted(numeros_en(entrada))


def test_x_como_incognita_no_se_toca():
    assert "x" in Normalizador()("halla x si 2x = 6").normalizada.split()


def test_reescritura_que_cambia_numeros_se_descarta():
    n = Normalizador(reescribir=lambda t: "¿Cuánto vale el examen final si saqué 5.0?")(
        "cuanto bale el examen si saque 4.5")
    assert n.descartada and "4.5" in n.normalizada and not n.reescrita_llm


def test_reescritura_solo_si_parece_informal():
    llamadas = []
    norm = Normalizador(reescribir=lambda t: llamadas.append(t) or t)
    norm("¿Cuánto es 17/24 + 11/36?")
    norm("cuanto es 17/24 mas 11/36")
    assert len(llamadas) == 1


def test_mcnemar():
    assert mcnemar_exacto([False] * 5, [True] * 5) == (5, 0, 0.0625)
    assert mcnemar_exacto([True, False], [True, False])[2] == 1.0


def test_reescritura_valida_se_acepta():
    n = Normalizador(reescribir=lambda t: "¿Cuánto vale el examen final en la nota del periodo?")(
        "cuanto bale el examen final en la nota del periodo")
    assert n.reescrita_llm and "vale" in n.normalizada


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

def test_parser_llamada_nativa_con_pensamiento():
    s = analizar_salida("Pensamiento: necesito multiplicar.\n" + tool_call("multiplicar", numeros=[2, 3]))
    assert s.pensamiento == "necesito multiplicar." and s.llamadas[0].nombre == "multiplicar"
    assert s.llamadas[0].formato == "nativo" and s.respuesta is None


def test_parser_varias_llamadas_y_argumentos_como_cadena():
    t = tool_call("sumar", numeros=[1, 2]) + '\n<tool_call>{"name": "raiz", "arguments": "{\\"radicando\\": 9}"}</tool_call>'
    s = analizar_salida(t)
    assert [l.nombre for l in s.llamadas] == ["sumar", "raiz"] and s.llamadas[1].argumentos == {"radicando": 9}


def test_parser_recupera_json_sin_etiquetas():
    s = analizar_salida('Voy a usar: {"name": "raiz", "arguments": {"radicando": 49}}')
    assert s.llamadas[0].formato == "recuperado"


def test_parser_json_invalido_es_error_de_formato():
    s = analizar_salida('<tool_call>{"name": "raiz", "arguments": {radicando: 49</tool_call>')
    assert s.error_formato and not s.llamadas


def test_parser_respuesta_directa_con_llaves_no_es_llamada():
    s = analizar_salida('El conjunto {1, 2} tiene dos elementos. Respuesta final: 2')
    assert s.respuesta and not s.llamadas


def test_parser_comillas_escapadas():
    s = analizar_salida('<tool_call>{"name": "evaluar_expresion", "arguments": {"expresion": "2 \\"x\\" 3"}}</tool_call>')
    assert s.llamadas[0].argumentos["expresion"] == '2 "x" 3'


def test_valor_final_punto_decimal():
    assert valor_final("Paso 1: 11.28 - 1.964 = 9.316.\nRespuesta final: 9.316") == "9.316"
    assert valor_final("Respuesta final: 1,138,712") == "1138712"


# --------------------------------------------------------------------------
# Bucle del agente
# --------------------------------------------------------------------------

def test_react_multipaso_con_observaciones():
    llm = Guion(
        "Pensamiento: primero las cajas que quedan.\n" + tool_call("restar", minuendo=48, sustraendo=23),
        "Pensamiento: ahora el total.\n" + tool_call("multiplicar", numeros=[25, 1275]),
        "Paso 1: Quedan 25 cajas.\nPaso 2: 25 × 1275 = 31875.\nRespuesta final: 31875 tornillos",
    )
    out = AgenteTutor(llm, "sys", registro=REGISTRO, max_rondas=5)("pregunta")
    assert out.terminacion == "respuesta" and out.pasos_llm == 3
    assert [o["valor"] for o in out.observaciones] == ["25", "31875"]
    assert [e.tipo for e in out.traza].count("observacion") == 2
    # La observación vuelve al modelo como mensaje 'tool'
    assert llm.mensajes[1][-1]["role"] == "tool"


def test_function_calling_de_una_ronda_no_ofrece_herramientas_despues():
    llm = Guion(tool_call("raiz", radicando=7056), "Respuesta final: 84")
    AgenteTutor(llm, "sys", registro=REGISTRO, max_rondas=1)("raiz de 7056")
    assert llm.vistas[0] is not None and llm.vistas[1] is None


def test_error_de_formato_se_devuelve_y_el_agente_se_recupera():
    llm = Guion('<tool_call>{"name": "raiz", "arguments": {</tool_call>',
                tool_call("raiz", radicando=49), "Respuesta final: 7")
    out = AgenteTutor(llm, "sys", registro=REGISTRO, max_rondas=3)("raiz de 49")
    assert out.respuesta == "Respuesta final: 7"
    assert any(e.tipo == "error_formato" for e in out.traza)


def test_error_de_dominio_llega_al_modelo():
    llm = Guion(tool_call("dividir", dividendo=15, divisor=0),
                "La división entre cero no está definida. Respuesta final: No definido")
    out = AgenteTutor(llm, "sys", registro=REGISTRO, max_rondas=3)("15 entre 0")
    assert out.observaciones[0]["tipo_error"] == "dominio"
    assert "no está definida" in llm.mensajes[1][-1]["content"]


def test_verificador_pide_una_sola_revision():
    llm = Guion(tool_call("multiplicar", numeros=[3847, 296]),
                "Respuesta final: 1138000",            # no usa el resultado
                "Respuesta final: 1138000")            # insiste: no se vuelve a pedir
    out = AgenteTutor(llm, "sys", registro=REGISTRO, max_rondas=3, verificar=True)("3847 por 296")
    assert sum(e.tipo == "verificacion" for e in out.traza) == 1 and out.pasos_llm == 3


def test_verificador_no_interviene_si_coincide():
    llm = Guion(tool_call("multiplicar", numeros=[3847, 296]), "Respuesta final: 1,138,712")
    out = AgenteTutor(llm, "sys", registro=REGISTRO, max_rondas=3, verificar=True)("3847 por 296")
    assert not any(e.tipo == "verificacion" for e in out.traza)


def test_tope_de_pasos_fuerza_respuesta_sin_herramientas():
    llm = Guion(*[tool_call("raiz", radicando=4)] * 3, "Respuesta final: 2")
    out = AgenteTutor(llm, "sys", registro=REGISTRO, max_rondas=10, max_pasos=3)("raiz de 4")
    assert out.terminacion == "tope_pasos" and llm.vistas[-1] is None


def test_contexto_fijo_y_normalizador():
    llm = Guion("Respuesta final: 40")
    out = AgenteTutor(llm, "sys", contexto_fijo=lambda q: ["El examen final vale 40 por ciento."],
                      normalizador=Normalizador())("cuanto bale el examen??")
    assert "Contexto:" in llm.mensajes[0][-1]["content"] and out.contextos
    assert out.pregunta_normalizada.endswith("?") and not out.pregunta_normalizada.endswith("??")


def test_buscar_documentos_alimenta_los_contextos():
    reg = REGISTRO.subconjunto()

    @reg.herramienta(familia="documentos")
    def buscar_documentos(consulta: str) -> dict:
        """Busca en los documentos del colegio.

        Args:
            consulta: qué buscar.
        """
        return {"resultado": "[SIEE] El examen final vale 40 por ciento.", "valor": None,
                "chunks": ["El examen final vale 40 por ciento."]}

    llm = Guion(tool_call("buscar_documentos", consulta="examen final"),
                tool_call("porcentaje_de", porcentaje=40, cantidad=4.5), "Respuesta final: 1.8")
    out = AgenteTutor(llm, "sys", registro=reg, max_rondas=4,
                      extraer_contextos=lambda r: r.extra.get("chunks", []))("pregunta")
    assert out.contextos == ["El examen final vale 40 por ciento."]
    assert "buscar_documentos" not in REGISTRO          # el subconjunto no modifica el original


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------

def _obs(nombre, ok=True, valor=None, tipo_error=None):
    return {"herramienta": nombre, "ok": ok, "valor": valor, "tipo_error": tipo_error, "formato": "nativo"}


COMODINES = {n: REGISTRO.obtener(n).comodin_de for n in REGISTRO.nombres()}


def test_seleccion_con_comodin_y_estricta():
    m = metricas_herramientas_caso({"familias": ["fracciones"], "opcionales": []},
                                   [_obs("evaluar_expresion", valor="73/72")], "Respuesta final: 73/72",
                                   REGISTRO.familia_de, COMODINES)
    assert m["recall_seleccion"] == 1.0 and m["recall_seleccion_estricto"] == 0.0
    assert m["seleccion_correcta"] and m["fidelidad_herramienta"]


def test_herramienta_innecesaria():
    m = metricas_herramientas_caso({"familias": [], "opcionales": []},
                                   [_obs("sumar", valor="3")], "Gabriel García Márquez",
                                   REGISTRO.familia_de, COMODINES)
    assert m["innecesarias"] == 1 and not m["seleccion_correcta"]


def test_resumen_y_robustez():
    filas = [dict(metricas_herramientas_caso({"familias": ["aritmetica"]}, [_obs("sumar", valor="5")],
                                             "Respuesta final: 5", REGISTRO.familia_de, COMODINES),
                  pasos_llm=2, terminacion="respuesta"),
             dict(metricas_herramientas_caso({"familias": []}, [], "No lo sé", REGISTRO.familia_de, COMODINES),
                  pasos_llm=1, terminacion="respuesta")]
    r = resumen_agente(filas)
    assert r["uso_cuando_hace_falta"] == 1.0 and r["abstencion_cuando_no_hace_falta"] == 1.0
    rob = robustez_mal_escrito(
        [{"id": "mal-11", "original": "ari-01", "acierto": False, "respuesta": "Respuesta final: 1138000"}],
        [{"id": "ari-01", "acierto": True, "respuesta": "Respuesta final: 1138712"}])
    assert rob["perdidos_por_escritura"] == 1 and rob["misma_respuesta"] == 0.0
    ac = autoconsistencia({"a": ["Respuesta final: 7", "Respuesta final: 7.0"], "b": ["R: 1", "R: 2"]})
    assert ac["unanimes"] == 1


# --------------------------------------------------------------------------
# RAGAS
# --------------------------------------------------------------------------

def test_ragas_objetivas():
    assert average_precision([1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([0, 0]) == 0.0
    ctx = ["La nota mínima aprobatoria es 3.0.", "Otro texto."]
    assert context_recall_oro(ctx, "nota minima aprobatoria es 3.0") == 1.0
    assert context_precision_oro(list(reversed(ctx)), "nota minima aprobatoria es 3.0") == 0.5
    # 4.5 (pregunta) y 0.40 (documento) están respaldados; 1.8 (dos veces) solo si hubo herramienta:
    assert faithfulness_numerica("examen con 4.5", "4.5 × 0.40 = 1.8. Respuesta final: 1.8",
                                 ["vale el 40 por ciento"], []) == pytest.approx(2 / 4)
    assert faithfulness_numerica("examen con 4.5", "4.5 × 0.40 = 1.8. Respuesta final: 1.8",
                                 ["vale el 40 por ciento"], ['{"resultado": "1.8"}']) == 1.0
    assert kappa_cohen([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_veredictos_y_faithfulness_con_evaluador_falso():
    assert veredictos("1: sí\n2: No\n3: si", 3) == [1, 0, 1]
    assert veredictos("1: sí", 2) is None
    assert afirmaciones("Paso 1: El examen vale 40%.\nPaso 2: 4.5 × 0.4 = 1.8.\nRespuesta final: 1.8")
    evaluador = lambda system, user, max_tokens: "1: sí\n2: no\n3: no"  # noqa: E731
    f = faithfulness(evaluador, "p", "Paso 1: El examen vale 40%.\nPaso 2: 4.5 × 0.4 = 1.8 puntos.\nRespuesta final: 1.8 puntos",
                     ["vale 40"], [])
    assert f == pytest.approx(1 / 3)
    assert faithfulness(evaluador, "p", "algo largo aquí", [], []) is None
