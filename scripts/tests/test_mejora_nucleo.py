"""
Tests del verificador de planteamiento y del enrutador integrado.

    python -m pytest scripts/tests/test_mejora_nucleo.py -q

Los casos sintéticos fijan el comportamiento; los de regresión al final usan las
**trazas reales** de S10, así que si alguien toca el verificador y deja de ver
`ari-09` (el plan incompleto que motivó todo esto), el test lo dice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "agente"))

import numpy as np  # noqa: E402
from enrutador import Enrutador, contexto_enrutado  # noqa: E402
from verificador import VerificadorPlan, derivables  # noqa: E402
from fractions import Fraction  # noqa: E402


def obs(herramienta, argumentos, valor, ok=True):
    return {"herramienta": herramienta, "argumentos": argumentos, "ok": ok, "valor": valor}


@pytest.fixture
def v():
    return VerificadorPlan()


# --------------------------------------------------------------------------
# Verificador · lo que debe aceptar
# --------------------------------------------------------------------------

def test_plan_completo_no_se_queja(v):
    rev = v.revisar(
        "Una caja trae 24 lápices. ¿Cuántos hay en 13 cajas?",
        "Paso 1: 24 × 13 = 312.\nRespuesta final: 312",
        [obs("multiplicar", {"numeros": [24, 13]}, "312")])
    assert rev.ok, [str(h) for h in rev.hallazgos]


def test_acepta_porcentaje_derivado(v):
    """El enunciado dice 1.8 por ciento; usar 0.018 es un paso legítimo."""
    rev = v.revisar(
        "Un préstamo de 2450000 genera un interés del 1.8% mensual en 7 meses.",
        "Paso 1: 2450000 × 0.018 × 7 = 308700.\nRespuesta final: 308700",
        [obs("evaluar_expresion", {"expresion": "2450000 × 0.018 × 7"}, "308700")])
    assert not [h for h in rev.graves if h.tipo == "argumento_no_trazable"]


def test_acepta_redondeo_contextual(v):
    """1250 ÷ 180 = 6.94 y la respuesta son 7 camiones: redondear es del dominio."""
    rev = v.revisar(
        "Hay 1250 cajas y cada camión lleva 180. ¿Cuántos camiones se necesitan?",
        "Paso 1: 1250 ÷ 180 = 6.94.\nPaso 2: no hay camiones partidos.\nRespuesta final: 7 camiones",
        [obs("dividir", {"dividendo": 1250, "divisor": 180}, "6.944444")])
    assert not [h for h in rev.graves if h.tipo == "respuesta_no_trazable"]


def test_acepta_partes_de_una_fraccion_del_enunciado(v):
    """Si el enunciado dice 2/5, usar el 2 o el 5 en un paso es legítimo."""
    rev = v.revisar(
        "Ana leyó 2/5 de un libro de 300 páginas.",
        "Paso 1: 2 × 300 = 600.\nPaso 2: 600 ÷ 5 = 120.\nRespuesta final: 120",
        [obs("multiplicar", {"numeros": [2, 300]}, "600"), obs("dividir", {"dividendo": 600, "divisor": 5}, "120")])
    assert not [h for h in rev.graves if h.tipo == "argumento_no_trazable"]


def test_encadena_observaciones_previas(v):
    """El resultado de un paso alimenta el siguiente sin estar en el enunciado."""
    rev = v.revisar(
        "Un taller compra 3 cajas a 18500 cada una y paga con 150000. ¿Cuánto le devuelven?",
        "Paso 1: 3 × 18500 = 55500.\nPaso 2: 150000 - 55500 = 94500.\nRespuesta final: 94500",
        [obs("multiplicar", {"numeros": [3, 18500]}, "55500"),
         obs("restar", {"minuendo": 150000, "sustraendo": 55500}, "94500")])
    assert not [h for h in rev.graves if h.tipo == "argumento_no_trazable"]


def test_conocidos_extra_silencia_el_few_shot(v):
    """Un número que el modelo vio en un ejemplo del prompt no es inventado."""
    sin = v.revisar("¿Cuántos puntos aporta un examen de 4.5?",
                    "Paso 1: 4.5 × 0.4 = 1.8.\nRespuesta final: 1.8",
                    [obs("multiplicar", {"numeros": [4.5, 0.4]}, "1.8")])
    con = v.revisar("¿Cuántos puntos aporta un examen de 4.5?",
                    "Paso 1: 4.5 × 0.4 = 1.8.\nRespuesta final: 1.8",
                    [obs("multiplicar", {"numeros": [4.5, 0.4]}, "1.8")],
                    conocidos_extra=["el examen final vale el 40 por ciento"])
    assert any(h.tipo == "argumento_no_trazable" for h in sin.graves)
    assert not [h for h in con.graves if h.tipo == "argumento_no_trazable"]


# --------------------------------------------------------------------------
# Verificador · lo que debe detectar
# --------------------------------------------------------------------------

def test_detecta_plan_incompleto(v):
    """ari-09: calculó el interés de UN mes y la pregunta decía 7."""
    rev = v.revisar(
        "Un préstamo de 2450000 pesos genera un interés simple del 1.8% mensual. "
        "¿Cuánto interés se paga en 7 meses?",
        "Paso 1: 2450000 × 0.018 = 44100.\nRespuesta final: 44100 pesos",
        [obs("multiplicar", {"numeros": [2450000, 0.018]}, "44100")])
    sin_usar = [h for h in rev.hallazgos if h.tipo == "dato_sin_usar"]
    assert sin_usar and "7" in sin_usar[0].detalle


def test_detecta_argumento_inventado(v):
    rev = v.revisar("¿Cuánto es 24 × 13?",
                    "Paso 1: 24 × 17 = 408.\nRespuesta final: 408",
                    [obs("multiplicar", {"numeros": [24, 17]}, "408")])
    assert any(h.tipo == "argumento_no_trazable" for h in rev.graves)


def test_detecta_respuesta_que_no_sale_de_las_herramientas(v):
    rev = v.revisar("¿Cuánto es 3847 × 296?",
                    "Paso 1: lo calculé.\nRespuesta final: 1138000",
                    [obs("multiplicar", {"numeros": [3847, 296]}, "1138712")])
    assert any(h.tipo == "respuesta_no_trazable" for h in rev.graves)


def test_detecta_operacion_escrita_que_nadie_calculo(v):
    rev = v.revisar("¿Cuánto es 24 × 13 y cuánto es eso más 10?",
                    "Paso 1: 24 × 13 = 312.\nPaso 2: 312 + 10 = 999.\nRespuesta final: 312",
                    [obs("multiplicar", {"numeros": [24, 13]}, "312")])
    assert any(h.tipo == "calculo_de_cabeza" for h in rev.hallazgos)


def test_sin_observaciones_no_inventa_quejas(v):
    """Si el agente no llamó herramientas, el verificador no tiene nada que decir."""
    rev = v.revisar("¿Qué es un número primo?", "Es el que solo se divide entre 1 y él mismo.", [])
    assert rev.ok


def test_derivables_incluye_porcentaje_y_complemento():
    base = {Fraction(20)}
    d = derivables(base, 1)
    assert Fraction(1, 5) in d          # 20 ÷ 100
    assert Fraction(4, 5) in d          # 1 − 0.20


# --------------------------------------------------------------------------
# Enrutador
# --------------------------------------------------------------------------

PALABRAS_DOC = ("colegio", "siee", "reglamento", "plan de área", "nota mínima",
                "recuperación", "protocolo", "examen final vale")


def encode_falso(textos):
    """Codificador determinista: dos dimensiones, 'suena a documento' y 'trae números'."""
    filas = []
    for t in textos:
        b = t.lower()
        doc = sum(p in b for p in PALABRAS_DOC)
        num = sum(ch.isdigit() for ch in b)
        v = np.array([doc + 0.1, num * 0.1 + 0.1], dtype=float)
        filas.append(v / np.linalg.norm(v))
    return np.array(filas)


def rerank_falso(pregunta):
    return 5.0 if any(p in pregunta.lower() for p in PALABRAS_DOC) else -5.0


CONSULTAS = (
    [{"q": f"¿Qué dice el reglamento del colegio sobre {t}?", "tipo": "consulta"}
     for t in ("la nota mínima", "las recuperaciones", "el examen final vale", "el protocolo")]
    + [{"q": "¿Cuál es la nota mínima del colegio?", "tipo": "consulta"},
       {"q": "¿Qué dice el plan de área de octavo?", "tipo": "consulta"}]
    + [{"q": f"¿Cuánto es {a} × {b}?", "tipo": "problema"} for a, b in ((24, 13), (7, 9), (125, 4))]
    + [{"q": "Resuelve la ecuación 3x + 7 = 25.", "tipo": "problema"},
       {"q": "Un tren recorre 120 km en 2 horas, ¿a qué velocidad va?", "tipo": "problema"},
       {"q": "¿Cuánto es 3/4 + 1/8?", "tipo": "problema"}]
)


@pytest.fixture(scope="module")
def enrutador():
    return Enrutador(encode_falso, rerank_falso, CONSULTAS)


def test_enruta_consulta_documental(enrutador):
    d = enrutador.clasificar("¿Cuál es la nota mínima aprobatoria del colegio?")
    assert d.es_consulta and d.probabilidad > 0.5


def test_enruta_problema_de_calculo(enrutador):
    d = enrutador.clasificar("¿Cuánto es 3847 × 296?")
    assert not d.es_consulta


def test_leave_one_out_es_honesto(enrutador):
    r = enrutador.leave_one_out()
    assert r["n"] == len(CONSULTAS)
    assert r["exactitud"] >= 0.8, r["errores"]


def test_funciona_sin_reranker():
    e = Enrutador(encode_falso, None, CONSULTAS)
    assert e.clasificar("¿Qué dice el reglamento del colegio sobre la nota mínima?").es_consulta


def test_contexto_enrutado_solo_busca_cuando_toca(enrutador):
    llamadas = []
    ctx = contexto_enrutado(enrutador, lambda q: ["[Fuente: SIEE] la nota mínima es 3.0"],
                            registrar=lambda q, d: llamadas.append(d.ruta))
    assert ctx("¿Cuál es la nota mínima del colegio?")          # trae contexto
    assert ctx("¿Cuánto es 24 × 13?") == []                     # no gasta en buscar
    assert llamadas == ["consulta", "problema"]


# --------------------------------------------------------------------------
# Regresión sobre las trazas reales de S10
# --------------------------------------------------------------------------

TRAZAS = RAIZ / "M3_rag_agentes" / "resultados" / "respuestas_s10.jsonl"


@pytest.fixture(scope="module")
def trazas():
    if not TRAZAS.exists():
        pytest.skip("no están las trazas de S10")
    out = {}
    for linea in TRAZAS.read_text(encoding="utf-8").splitlines():
        r = json.loads(linea)
        out[(r["config"], r["id"])] = r
    return out


@pytest.mark.parametrize("caso", ["ari-09", "ari-12"])
def test_regresion_detecta_planes_incompletos_reales(v, trazas, caso):
    r = trazas[("C5-FT · entrenado", caso)]
    rev = v.revisar(r["pregunta"], r["respuesta"], r["observaciones"], r.get("contextos", ()))
    assert rev.hallazgos, f"{caso} dejó de detectarse"


def test_regresion_no_se_queja_de_los_aciertos_limpios(v, trazas):
    """dif-01 lo resolvió bien encadenando dos herramientas: no debe haber queja grave."""
    r = trazas[("C5-FT · entrenado", "dif-02")]
    rev = v.revisar(r["pregunta"], r["respuesta"], r["observaciones"], r.get("contextos", ()))
    assert not [h for h in rev.graves if h.tipo == "respuesta_no_trazable"]
