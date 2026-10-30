"""Las herramientas de cálculo: derivadas e integrales.

Por qué existen. La auditoría de temas de M5 encontró que el tutor no tenía nada
para cálculo —cero herramientas, cero casos en el corpus de M1, cero en los eval
sets— y aun así respondía: generaba una derivada plausible sin forma de
comprobarla. Son deterministas, así que cubren el tema sin entrenar nada.

    python -m pytest scripts/tests/test_calculo.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "agente"))

from herramientas import REGISTRO  # noqa: E402
from registro import LlamadaHerramienta  # noqa: E402


def ejecutar(nombre, **args):
    return REGISTRO.ejecutar(LlamadaHerramienta(nombre, args))


# --- derivadas ------------------------------------------------------------

@pytest.mark.parametrize("expr,esperado", [
    ("3*x**2 + 5*x - 2", "6*x + 5"),
    ("x**3", "3*x**2"),
    ("5", "0"),
    ("x**2 + 2*x + 1", "2*x + 2"),
])
def test_derivadas_de_polinomios(expr, esperado):
    r = ejecutar("derivar", expresion=expr)
    assert r.ok, r.error
    assert r.valor == esperado


def test_derivada_de_una_raiz():
    r = ejecutar("derivar", expresion="sqrt(x)")
    assert r.ok and "sqrt(x)" in r.valor


def test_derivada_respecto_a_otra_variable():
    r = ejecutar("derivar", expresion="t**2", variable="t")
    assert r.ok and r.valor == "2*t"


# --- integrales -----------------------------------------------------------

def test_primitiva_lleva_constante():
    r = ejecutar("integrar", expresion="2*x")
    assert r.ok and "+ C" in r.resultado


def test_integral_definida_es_exacta():
    """∫₀³ x² dx = 9, no 8.999999."""
    r = ejecutar("integral_definida", expresion="x**2", desde=0, hasta=3)
    assert r.ok and r.valor == "9"


def test_integral_definida_con_resultado_fraccionario():
    """∫₀¹ x² dx = 1/3: con coma flotante saldría 0.333…"""
    r = ejecutar("integral_definida", expresion="x**2", desde=0, hasta=1)
    assert r.ok and "1/3" in str(r.valor)


def test_derivar_e_integrar_son_inversas():
    d = ejecutar("derivar", expresion="x**3 + x")
    assert d.ok
    i = ejecutar("integrar", expresion=d.valor)
    assert i.ok and "x" in i.valor


# --- lo que deben rechazar ------------------------------------------------

def test_rechaza_una_variable_fuera_de_la_lista_blanca():
    r = ejecutar("derivar", expresion="x**2", variable="w")
    assert not r.ok and "no permitida" in r.error


def test_no_ejecuta_codigo():
    """El parser usa lista blanca de símbolos, nunca `eval` de Python."""
    for expr in ("__import__(1)", "open('x')", "exec(1)"):
        r = ejecutar("derivar", expresion=expr)
        assert not r.ok, expr


def test_una_integral_que_no_tiene_forma_cerrada_se_declara():
    """Mejor decir que no se sabe que devolver una `Integral` sin resolver."""
    r = ejecutar("integrar", expresion="sqrt(1 + x**3)")
    assert not r.ok or "Integral" not in str(r.valor)


def test_los_errores_no_tumban_al_agente():
    """`ejecutar` nunca lanza: todo error vuelve como observación."""
    r = ejecutar("integral_definida", expresion="1/x", desde=-1, hasta=1)
    assert r.ok is False or r.valor is not None


# --- registro -------------------------------------------------------------

def test_las_tres_estan_registradas_en_su_familia():
    nombres = set(REGISTRO.nombres())
    assert {"derivar", "integrar", "integral_definida"} <= nombres
    for n in ("derivar", "integrar", "integral_definida"):
        assert REGISTRO.obtener(n).familia == "calculo"


def test_el_esquema_llega_al_modelo_con_su_descripcion():
    esquema = REGISTRO.obtener("derivar").esquema
    texto = str(esquema)
    assert "deriva" in texto.lower() and "expresion" in texto


# =========================================================================
# Exactitud: el resultado, no algo que lo contenga
#
# La primera versión de estos tests comprobaba `"sqrt(x)" in r.valor`, que pasa
# también con una derivada equivocada que mencione la raíz. Aquí se compara el
# resultado completo.
# =========================================================================

@pytest.mark.parametrize("expr,esperado", [
    ("sqrt(x)", "1/(2*sqrt(x))"),
    ("1/x", "-1/x**2"),
    ("x*sqrt(x)", "3*sqrt(x)/2"),
    ("pi*x**2", "2*pi*x"),
])
def test_la_derivada_es_exactamente_la_esperada(expr, esperado):
    r = ejecutar("derivar", expresion=expr)
    assert r.ok, r.error
    assert r.valor == esperado


@pytest.mark.parametrize("expr,esperado", [
    ("2*x", "x**2"),
    ("x**2", "x**3/3"),
    ("1/x", "log(x)"),
    ("1", "x"),
])
def test_la_primitiva_es_exactamente_la_esperada(expr, esperado):
    r = ejecutar("integrar", expresion=expr)
    assert r.ok, r.error
    assert r.valor == esperado


@pytest.mark.parametrize("expr,a,b,esperado", [
    ("x**2", 0, 1, "1/3"),
    ("x", 0, 2, "2"),
    ("1/x", 1, 2, "log(2)"),
])
def test_la_integral_definida_no_se_convierte_en_decimal(expr, a, b, esperado):
    """2/3 no es 0.666667. El harness de M2 compara valores exactos."""
    r = ejecutar("integral_definida", expresion=expr, desde=a, hasta=b)
    assert r.ok, r.error
    assert r.valor == esperado or esperado in r.resultado


# =========================================================================
# Ambigüedad: preguntar en vez de elegir en silencio
# =========================================================================

@pytest.mark.parametrize("expr", ["a*x**2 + b", "x*y", "n*t"])
def test_varias_variables_sin_decir_cual_pide_aclaracion(expr):
    r = ejecutar("derivar", expresion=expr)
    assert not r.ok
    assert "varias variables" in r.error and "variable" in r.error


def test_al_indicar_la_variable_la_ambiguedad_desaparece():
    assert ejecutar("derivar", expresion="a*x**2 + b", variable="x").valor == "2*a*x"
    assert ejecutar("derivar", expresion="a*x**2 + b", variable="a").valor == "x**2"


def test_una_sola_variable_no_hace_falta_nombrarla():
    """`deriva t**2` no es ambiguo: exigir `variable` ahí sería ruido."""
    assert ejecutar("derivar", expresion="t**2").valor == "2*t"
    assert ejecutar("integrar", expresion="2*n").valor == "n**2"


def test_una_constante_no_es_ambigua():
    assert ejecutar("derivar", expresion="5").valor == "0"


# =========================================================================
# Integrales que no tienen respuesta
# =========================================================================

@pytest.mark.parametrize("expr,a,b", [
    ("1/x", 0, 1),        # diverge en 0
    ("1/x", -1, 1),       # valor principal, no integral de Riemann
    ("1/x**2", 0, 1),     # diverge
    ("1/x**2", -1, 1),    # diverge
])
def test_una_integral_divergente_se_declara_sin_respuesta(expr, a, b):
    """Devolver `oo`, `zoo` o un número cualquiera aquí es alucinar."""
    r = ejecutar("integral_definida", expresion=expr, desde=a, hasta=b)
    assert not r.ok
    assert "no converge" in r.error
    assert r.valor is None


def test_una_primitiva_sin_forma_cerrada_se_declara():
    r = ejecutar("integrar", expresion="x**x")
    assert not r.ok


# =========================================================================
# Topes de tamaño y complejidad
#
# Sin ellos `integrate` sobre una expresión larga bloquea el proceso: una demo
# colgada es peor que una herramienta que dice que no puede.
# =========================================================================

def test_una_expresion_demasiado_larga_se_rechaza_por_longitud():
    r = ejecutar("derivar", expresion="x+" * 120 + "x")
    assert not r.ok and "tope es 200" in r.error


def test_un_exponente_enorme_se_rechaza():
    r = ejecutar("derivar", expresion="x**200")
    assert not r.ok and "exponente" in r.error
    assert not ejecutar("integral_definida", expresion="x**9999",
                        desde=0, hasta=1).ok


def test_demasiadas_operaciones_se_rechazan():
    from herramientas import MAX_OPERACIONES
    expr = "+".join(f"{i}*x**{i}" for i in range(1, 18))   # 134 caracteres, 48 ops
    r = ejecutar("integrar", expresion=expr)
    assert len(expr) <= 200, "este caso debe fallar por operaciones, no por largo"
    assert not r.ok
    assert str(MAX_OPERACIONES) in r.error


def test_el_tope_no_estorba_a_un_ejercicio_normal():
    """Los topes tienen que dejar pasar lo que un estudiante realmente escribe."""
    for expr in ("3*x**2 + 5*x - 2", "x**3 - 4*x**2 + 7*x - 1",
                 "(x**2 + 1)*(x - 3)", "sqrt(x) + 1/x"):
        assert ejecutar("derivar", expresion=expr).ok, expr


def test_el_tope_de_tiempo_se_cumple_en_la_practica():
    """El tope solo sirve si de verdad acota el tiempo. Se mide."""
    import time
    t0 = time.perf_counter()
    for expr in ("x**50", "+".join(f"x**{i}" for i in range(1, 12))):
        ejecutar("integrar", expresion=expr)
    assert time.perf_counter() - t0 < 10.0


# =========================================================================
# De punta a punta: la pregunta en español entra por el agente
#
# LÍMITE DE ESTOS TESTS. El LLM está sustituido por un guion, así que prueban la
# fontanería completa —esquemas visibles, argumentos validados, observación
# devuelta, respuesta final apoyada en el valor de la herramienta— pero NO que
# Qwen2.5-1.5B elija la herramienta al leer la pregunta. Eso requiere GPU y
# todavía no se ha medido.
# =========================================================================

def _agente(*salidas):
    import json as _json

    from nucleo import AgenteTutor

    class Guion:
        def __init__(self, salidas):
            self.salidas, self.vistas = list(salidas), []

        def __call__(self, mensajes, esquemas):
            self.vistas.append(None if esquemas is None
                               else [e["function"]["name"] for e in esquemas])
            return self.salidas.pop(0)

    def tool_call(nombre, **args):
        return ("<tool_call>\n"
                + _json.dumps({"name": nombre, "arguments": args}, ensure_ascii=False)
                + "\n</tool_call>")

    llm = Guion([s if isinstance(s, str) else tool_call(*s[0:1], **s[1])
                 for s in salidas])
    return AgenteTutor(llm, "sys", registro=REGISTRO, max_rondas=3, verificar=True), llm


def test_de_punta_a_punta_deriva_un_polinomio_en_espanol():
    agente, llm = _agente(("derivar", {"expresion": "3*x**2 + 5*x - 2"}),
                          "Respuesta final: 6*x + 5")
    out = agente("deriva 3x² + 5x − 2 respecto a x")
    assert "derivar" in llm.vistas[0]
    assert out.observaciones[0]["herramienta"] == "derivar"
    assert out.observaciones[0]["valor"] == "6*x + 5"
    assert "6*x + 5" in out.respuesta


def test_de_punta_a_punta_integral_definida_en_espanol():
    agente, llm = _agente(("integral_definida",
                           {"expresion": "x**2", "desde": 0, "hasta": 1}),
                          "Respuesta final: 1/3")
    out = agente("¿cuánto vale la integral de x² entre 0 y 1?")
    assert out.observaciones[0]["valor"] == "1/3"
    assert "1/3" in out.respuesta


def test_de_punta_a_punta_la_ambiguedad_llega_al_modelo_como_observacion():
    """El error de la herramienta vuelve al bucle: el modelo puede preguntar."""
    agente, llm = _agente(("derivar", {"expresion": "a*x**2 + b"}),
                          "Respuesta final: necesito saber respecto a qué "
                          "variable quieres derivar: hay a, b y x.")
    out = agente("deriva a x² + b")
    assert out.observaciones[0]["ok"] is False
    assert "varias variables" in out.observaciones[0]["error"]
    assert "variable" in out.respuesta


def test_de_punta_a_punta_una_integral_divergente_no_produce_numero():
    agente, llm = _agente(("integral_definida",
                           {"expresion": "1/x", "desde": 0, "hasta": 1}),
                          "Respuesta final: esa integral no converge, no hay un "
                          "número que darte.")
    out = agente("integra 1/x entre 0 y 1")
    assert out.observaciones[0]["ok"] is False
    assert "no converge" in out.observaciones[0]["error"]
    assert not any(c.isdigit() and c != "1" for c in out.respuesta.split("final:")[-1])


def test_las_tres_herramientas_de_calculo_le_llegan_al_modelo():
    """Si no están en los esquemas, el modelo no puede elegirlas nunca."""
    agente, llm = _agente("Respuesta final: ya está")
    agente("deriva x²")
    assert {"derivar", "integrar", "integral_definida"} <= set(llm.vistas[0])
