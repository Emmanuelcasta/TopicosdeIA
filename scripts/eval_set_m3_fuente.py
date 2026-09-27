"""
Eval set de M3 (S10) — lo que el eval set de M2 no puede medir de un agente.

`eval_set_m2.jsonl` NO se toca: es la vara fija que compara M1, M2, S07 y S08.
Este archivo añade tres bloques que se reportan APARTE, porque miden
capacidades que solo aparecen al dar herramientas y un bucle agéntico:

  D · ARITMETICA (13)   Cuentas donde un modelo de 1.5B se equivoca calculando:
                        números grandes, fracciones con denominadores no
                        triviales, raíces, decimales, ecuaciones y una expansión
                        algebraica. Es donde una herramienta DEBERÍA notarse.
                        (El bloque de cálculo de M2 ya está en 10/12 y sus dos
                        fallos son de PLANTEAMIENTO, no de aritmética: allí una
                        calculadora casi no tiene margen.)

  E · COMPUESTO (8)     Hay que BUSCAR un dato del colegio Y CALCULAR con él.
                        Ningún caso de M2 lo exige, y es exactamente el tipo de
                        pregunta donde S10 dice que un agente se justifica.

  F · MAL_ESCRITO (11)  Versiones mal escritas (sin tildes, faltas, abreviaturas,
                        "x" como multiplicación, sin signos) de casos existentes.
                        Cada una apunta a su `original`: mide ROBUSTEZ (¿acierta
                        igual?) y CONSISTENCIA (¿da la misma respuesta?).

Además escribe `herramientas_esperadas_m2.jsonl`: la anotación de qué familias
de herramientas debería usar cada caso de M2, en un archivo separado para no
modificar el eval set de M2.

Familias de herramientas (definidas en scripts/agente/herramientas.py):
  aritmetica, fracciones, potencias_raices, porcentajes, algebra,
  estadistica, documentos

Uso:
    python eval_set_m3_fuente.py            # valida y escribe los JSONL
    python eval_set_m3_fuente.py --check    # solo valida
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_set_fuente import TOLERANCIA, a_numero, normalizar  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "data" / "eval_set_m3_agente.jsonl"
SALIDA_ANOTACION_M2 = RAIZ / "data" / "herramientas_esperadas_m2.jsonl"
EVAL_M2 = RAIZ / "data" / "eval_set_m2.jsonl"
CORPUS_ENTRENAMIENTO = RAIZ / "data" / "math_tutor_dataset.jsonl"

FAMILIAS = {"aritmetica", "fracciones", "potencias_raices", "porcentajes",
            "algebra", "estadistica", "documentos"}
BLOQUES = ("aritmetica", "compuesto", "mal_escrito")


def E(id, bloque, subtipo, input, esperado, criterio, evaluacion,
      familias, opcionales=(), check=None, dato=None, original=None):
    return {
        "id": id,
        "bloque": bloque,
        "subtipo": subtipo,
        "input": input,
        "esperado": esperado,
        "criterio": criterio,
        "evaluacion": evaluacion,
        # Herramientas: familias REQUERIDAS (el caso no se resuelve bien sin
        # ellas) y OPCIONALES (usarlas no es un error, pero no hace falta).
        "herramientas": {"familias": list(familias), "opcionales": list(opcionales)},
        "original": original,       # solo bloque F: id del caso bien escrito
        "dato": dato,               # solo bloque E: texto que DEBE estar en el corpus
        "check": check,
    }


# --------------------------------------------------------------------------
# BLOQUE D · Aritmética exigente (13)
# --------------------------------------------------------------------------

ARITMETICA = [
    E("ari-01", "aritmetica", "multiplicacion_grande",
      "Calcula 3847 × 296.",
      "Paso 1: Multiplicamos 3847 × 296 = 1138712.\nRespuesta final: 1138712",
      "Producto de 4 por 3 cifras: los errores de acarreo son el fallo típico de un modelo pequeño.",
      {"tipo": "numerico", "valor": "1138712"}, ["aritmetica"], check="3847*296"),

    E("ari-02", "aritmetica", "multipaso_grande",
      "Una bodega recibe 48 cajas con 1275 tornillos cada una y despacha 23 cajas completas. ¿Cuántos tornillos quedan en la bodega?",
      "Paso 1: Quedan 48 - 23 = 25 cajas.\nPaso 2: 25 × 1275 = 31875 tornillos.\nRespuesta final: 31875 tornillos",
      "Dos operaciones con números de 4 cifras.",
      {"tipo": "numerico", "valor": "31875"}, ["aritmetica"], check="(48-23)*1275"),

    E("ari-03", "aritmetica", "suma_fracciones",
      "¿Cuánto es 17/24 + 11/36?",
      "Paso 1: El mínimo común múltiplo de 24 y 36 es 72.\nPaso 2: 17/24 = 51/72 y 11/36 = 22/72.\nPaso 3: 51/72 + 22/72 = 73/72.\nRespuesta final: 73/72",
      "Denominadores con MCM no evidente. Sumar numeradores y denominadores (28/60) es el error clásico.",
      {"tipo": "numerico", "valor": "73/72"}, ["fracciones"], ["aritmetica"],
      check="Fraction(17,24)+Fraction(11,36)"),

    E("ari-04", "aritmetica", "division_fracciones",
      "Calcula (5/6) ÷ (15/28) y simplifica el resultado.",
      "Paso 1: Dividir es multiplicar por la inversa: (5/6) × (28/15) = 140/90.\nPaso 2: Simplificamos entre 10: 14/9.\nRespuesta final: 14/9",
      "División de fracciones con simplificación final.",
      {"tipo": "numerico", "valor": "14/9"}, ["fracciones"], ["aritmetica"],
      check="Fraction(5,6)/Fraction(15,28)"),

    E("ari-05", "aritmetica", "raiz_cuadrada",
      "¿Cuál es la raíz cuadrada de 7056?",
      "Paso 1: Buscamos el número que al cuadrado da 7056: 84 × 84 = 7056.\nRespuesta final: 84",
      "Raíz exacta de un número de 4 cifras que no se memoriza.",
      {"tipo": "numerico", "valor": "84"}, ["potencias_raices"], check="math.isqrt(7056)"),

    E("ari-06", "aritmetica", "decimales",
      "Calcula 2.35 × 4.8 − 1.964.",
      "Paso 1: Primero la multiplicación: 2.35 × 4.8 = 11.28.\nPaso 2: Luego la resta: 11.28 - 1.964 = 9.316.\nRespuesta final: 9.316",
      "Jerarquía de operaciones y alineación de decimales.",
      {"tipo": "numerico", "valor": "9.316"}, ["aritmetica"], check="2.35*4.8-1.964"),

    E("ari-07", "aritmetica", "ecuacion_lineal",
      "Resuelve la ecuación 7x − 23 = 4x + 58.",
      "Paso 1: Pasamos los términos con x a un lado: 7x - 4x = 58 + 23.\nPaso 2: 3x = 81.\nPaso 3: x = 81 ÷ 3 = 27.\nRespuesta final: x = 27",
      "Incógnita en ambos lados: el error típico es cambiar mal un signo al transponer.",
      {"tipo": "numerico", "valor": "27"}, ["algebra"], ["aritmetica"], check="(58+23)/(7-4)"),

    E("ari-08", "aritmetica", "potencias",
      "Calcula 15³ − 12⁴.",
      "Paso 1: 15³ = 3375.\nPaso 2: 12⁴ = 20736.\nPaso 3: 3375 - 20736 = -17361.\nRespuesta final: -17361",
      "Dos potencias grandes y un resultado negativo.",
      {"tipo": "numerico", "valor": "-17361"}, ["potencias_raices"], ["aritmetica"], check="15**3-12**4"),

    E("ari-09", "aritmetica", "interes_simple",
      "Un préstamo de 2450000 pesos genera un interés simple del 1.8% mensual. ¿Cuánto interés se paga en 7 meses?",
      "Paso 1: Interés de un mes: 2450000 × 0.018 = 44100.\nPaso 2: En 7 meses: 44100 × 7 = 308700.\nRespuesta final: 308700 pesos",
      "Porcentaje decimal sobre una cantidad grande, luego multiplicado.",
      {"tipo": "numerico", "valor": "308700"}, ["porcentajes"], ["aritmetica"], check="2450000*0.018*7"),

    E("ari-10", "aritmetica", "division_grande",
      "Se reparten 98532 pesos en partes iguales entre 12 personas. ¿Cuánto recibe cada una?",
      "Paso 1: Dividimos 98532 ÷ 12 = 8211.\nRespuesta final: 8211 pesos",
      "División larga exacta.",
      {"tipo": "numerico", "valor": "8211"}, ["aritmetica"], check="98532/12"),

    E("ari-11", "aritmetica", "simplificar_fraccion",
      "Simplifica la fracción 1386/2310 hasta su mínima expresión.",
      "Paso 1: El máximo común divisor de 1386 y 2310 es 462.\nPaso 2: 1386 ÷ 462 = 3 y 2310 ÷ 462 = 5.\nRespuesta final: 3/5",
      "MCD grande: simplificar a medias (por ejemplo 693/1155) es el fallo esperado.",
      {"tipo": "numerico", "valor": "3/5"}, ["fracciones"], check="Fraction(1386,2310)"),

    E("ari-12", "aritmetica", "decimales_y_raiz",
      "Calcula 0.0375 ÷ 0.0015 + √2025.",
      "Paso 1: 0.0375 ÷ 0.0015 = 25.\nPaso 2: √2025 = 45.\nPaso 3: 25 + 45 = 70.\nRespuesta final: 70",
      "División entre decimales pequeños y una raíz exacta.",
      {"tipo": "numerico", "valor": "70"}, ["aritmetica", "potencias_raices"], check="0.0375/0.0015+math.sqrt(2025)"),

    E("ari-13", "aritmetica", "expansion_algebraica",
      "Expande y simplifica la expresión (2x + 3)(x − 5).",
      "Paso 1: Aplicamos la propiedad distributiva: 2x·x + 2x·(-5) + 3·x + 3·(-5).\nPaso 2: 2x² - 10x + 3x - 15.\nPaso 3: Reducimos términos semejantes: 2x² - 7x - 15.\nRespuesta final: 2x² - 7x - 15",
      "Operación algebraica: el término -7x es donde se equivoca quien suma mal los signos.",
      {"tipo": "contiene_alguna",
       "claves": ["2x^2 - 7x - 15", "2x² - 7x - 15", "2x**2 - 7x - 15", "2x^2-7x-15", "2x²-7x-15",
                  "2*x**2 - 7*x - 15", "2x² − 7x − 15", "2x^2 − 7x − 15"]},
      ["algebra"]),
]


# --------------------------------------------------------------------------
# BLOQUE E · Compuesto: dato del colegio + cálculo (8)
# --------------------------------------------------------------------------
# `dato` debe aparecer literalmente (normalizado) en el corpus documental de
# S07/S08: si el corpus cambia y el dato desaparece, el validador falla.

COMPUESTO = [
    E("com-01", "compuesto", "ponderacion_examen",
      "Según el sistema de evaluación del colegio, ¿cuántos puntos aporta a la nota del periodo un examen final calificado con 4.5?",
      "Paso 1: Según el SIEE, el examen final vale el 40% de la nota del periodo.\nPaso 2: 4.5 × 0.40 = 1.8.\nRespuesta final: 1.8",
      "Buscar el 40% y multiplicar. Sin el dato, el modelo inventa el porcentaje.",
      {"tipo": "numerico", "valor": "1.8"}, ["documentos", "porcentajes"], ["aritmetica"],
      check="4.5*0.4", dato="examen final del periodo equivale al 40 por ciento"),

    E("com-02", "compuesto", "nota_periodo_completa",
      "Un estudiante sacó 4.0 en el examen final, 3.5 en talleres, 3.0 en quices y 4.5 en trabajo en clase. ¿Cuál es su nota del periodo según la ponderación del colegio?",
      "Paso 1: Según el SIEE: examen 40%, talleres 25%, quices 20%, trabajo en clase 15%.\nPaso 2: 4.0 × 0.40 + 3.5 × 0.25 + 3.0 × 0.20 + 4.5 × 0.15 = 1.6 + 0.875 + 0.6 + 0.675.\nPaso 3: La suma es 3.75.\nRespuesta final: 3.75",
      "Cuatro porcentajes del documento y un promedio ponderado.",
      {"tipo": "numerico", "valor": "3.75"}, ["documentos", "estadistica"], ["aritmetica", "porcentajes"],
      check="4.0*0.4+3.5*0.25+3.0*0.2+4.5*0.15", dato="los talleres al 25 por ciento, los quices al 20 por ciento"),

    E("com-03", "compuesto", "tope_recuperacion",
      "Un estudiante tenía 2.4 en un examen, fue a recuperación y sacó 4.8. ¿Qué nota le queda registrada según el reglamento del colegio?",
      "Paso 1: Según el SIEE, la nota máxima en una recuperación es 3.5.\nPaso 2: Como 4.8 supera ese tope, la nota registrada es 3.5.\nRespuesta final: 3.5",
      "El cálculo es aplicar un tope que solo está en el documento. Responder 4.8 es no haberlo buscado.",
      {"tipo": "numerico", "valor": "3.5"}, ["documentos"], ["aritmetica"],
      check="min(4.8,3.5)", dato="la nota maxima que puede obtenerse en una recuperacion es 3.5"),

    E("com-04", "compuesto", "distancia_a_aprobar",
      "Con la nota mínima aprobatoria del colegio, ¿cuánto le falta a un estudiante que tiene 2.65 para aprobar?",
      "Paso 1: Según el SIEE, la nota mínima aprobatoria es 3.0.\nPaso 2: 3.0 - 2.65 = 0.35.\nRespuesta final: 0.35",
      "Dato institucional más una resta con decimales.",
      {"tipo": "numerico", "valor": "0.35"}, ["documentos", "aritmetica"], [],
      check="3.0-2.65", dato="la nota minima aprobatoria es 3.0"),

    E("com-05", "compuesto", "grados_plan_area",
      "Según el plan de área, ¿cuántos grados escolares hay entre el grado en que se introducen las ecuaciones de primer grado y el grado en que se trabaja la trigonometría?",
      "Paso 1: Según el plan de área, las ecuaciones de primer grado se introducen en octavo (8.º).\nPaso 2: La trigonometría se trabaja en décimo (10.º).\nPaso 3: 10 - 8 = 2.\nRespuesta final: 2 grados",
      "Dos datos del mismo documento y una resta.",
      {"tipo": "numerico", "valor": "2"}, ["documentos"], ["aritmetica"],
      check="10-8", dato="en grado decimo se trabajan la trigonometria"),

    E("com-06", "compuesto", "suma_porcentajes_doc",
      "Según el sistema de evaluación del colegio, ¿qué porcentaje de la nota del periodo suman juntos el examen final y los talleres?",
      "Paso 1: Según el SIEE, el examen final vale 40% y los talleres 25%.\nPaso 2: 40 + 25 = 65.\nRespuesta final: 65%",
      "Dos porcentajes del documento.",
      {"tipo": "numerico", "valor": "65"}, ["documentos"], ["aritmetica", "porcentajes"],
      check="40+25", dato="los talleres al 25 por ciento"),

    E("com-07", "compuesto", "inasistencia_ponderada",
      "Un estudiante faltó sin excusa al examen final del periodo. Según el reglamento, ¿cuántos puntos aporta ese examen a su nota del periodo?",
      "Paso 1: Según el SIEE, la inasistencia no justificada se califica con 1.0.\nPaso 2: El examen final vale el 40%: 1.0 × 0.40 = 0.4.\nRespuesta final: 0.4",
      "Dos datos del SIEE (la nota por inasistencia y el peso del examen) y un producto.",
      {"tipo": "numerico", "valor": "0.4"}, ["documentos", "porcentajes"], ["aritmetica"],
      check="1.0*0.4", dato="se califica con 1.0"),

    E("com-08", "compuesto", "recuperaciones_anio",
      "Según el sistema de evaluación del colegio, ¿cuántas oportunidades de recuperación tiene en total un estudiante a lo largo de 4 periodos?",
      "Paso 1: Según el SIEE, hay dos oportunidades de recuperación por periodo.\nPaso 2: 2 × 4 = 8.\nRespuesta final: 8 oportunidades",
      "Dato por periodo multiplicado por el número de periodos.",
      {"tipo": "numerico", "valor": "8"}, ["documentos"], ["aritmetica"],
      check="2*4", dato="dos oportunidades de recuperacion por periodo"),
]


# --------------------------------------------------------------------------
# BLOQUE F · Mal escrito (11)
# --------------------------------------------------------------------------
# Se escriben como las escribiría un estudiante con prisa: sin tildes ni signos
# de apertura, con faltas (b/v, s/c/z, h), abreviaturas ("x", "q", "pa") y
# números pegados. La evaluación y las herramientas se COPIAN del original al
# construir el archivo, para que la única diferencia sea la escritura.

MAL_ESCRITO = [
    ("mal-01", "dif-06", "sin_tildes_abreviaturas",
     "una tienda da 20% de descuento y despues otro 10% sobre lo ya rebajado x pagar en efectivo. si el precio era 200000 cuanto se paga al final"),
    ("mal-02", "dif-08", "faltas_ortograficas",
     "si 6 obreros acen un muro en 12 dias cuantos dias se demorarian 9 obreros trabajando al mismo ritmo"),
    ("mal-03", "con-01", "faltas_ortograficas",
     "cuanto bale el examen final en la nota del periodo segun el sistema de evaluasion del colejio"),
    ("mal-04", "con-03", "coloquial_abreviado",
     "cual es la nota minima pa pasar en el colegio y cuantas recuperaciones ai x periodo"),
    ("mal-05", "ari-03", "operador_en_palabras",
     "cuanto es 17/24 mas 11/36"),
    ("mal-06", "ari-07", "faltas_y_espacios",
     "resuelbe la ecuasion 7x -23 = 4x+58"),
    ("mal-07", "com-02", "sin_puntuacion",
     "saque 4.0 en el examen final 3.5 en talleres 3.0 en quices y 4.5 en trabajo en clase cual es mi nota del periodo con la ponderacion del colegio"),
    ("mal-08", "adv-02", "coloquial",
     "cuanto da 15 dividido en 0"),
    ("mal-09", "dif-12", "faltas_ortograficas",
     "en una caja ai 5 volas rojas y 3 azules. se sacan 2 bolas una despues de otra sin debolver la primera, cual es la probabilidad de q las dos sean rojas"),
    ("mal-10", "ari-05", "signos_repetidos",
     "raiz cuadrada de 7056??"),
    ("mal-11", "ari-01", "x_como_multiplicacion",
     "cuanto es 3847 x 296"),
]


# --------------------------------------------------------------------------
# Anotación de herramientas para el eval set de M2 (archivo aparte)
# --------------------------------------------------------------------------

HERRAMIENTAS_M2 = {
    "dif-01": (["aritmetica"], []),
    "dif-02": (["aritmetica"], []),
    "dif-03": (["aritmetica"], []),
    "dif-04": (["aritmetica"], []),
    "dif-05": (["aritmetica"], []),
    "dif-06": (["porcentajes"], ["aritmetica"]),
    "dif-07": (["fracciones"], ["aritmetica"]),
    "dif-08": (["aritmetica"], []),
    "dif-09": (["aritmetica"], []),
    "dif-10": (["porcentajes"], ["aritmetica"]),
    "dif-11": (["estadistica"], ["aritmetica", "porcentajes"]),
    "dif-12": (["fracciones"], ["aritmetica"]),
    "con-01": (["documentos"], []),
    "con-02": (["documentos"], []),
    "con-03": (["documentos"], []),
    "con-04": (["documentos"], []),
    # Adversariales: ninguna herramienta es NECESARIA. Llamar a `dividir(15, 0)`
    # y explicar el error que devuelve es legítimo, por eso es opcional.
    "adv-01": ([], ["aritmetica"]),
    "adv-02": ([], ["aritmetica"]),
    "adv-03": ([], []),
    "adv-04": ([], []),
    "adv-05": (["aritmetica"], []),
}


# --------------------------------------------------------------------------
# Construcción y validación
# --------------------------------------------------------------------------

def _texto_corpus_documental() -> str:
    """El corpus de S07/S08, ejecutando el MISMO código que embeben los notebooks."""
    from nb_comun_rag import CODIGO_CORPUS
    ns: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(CODIGO_CORPUS, ns)  # noqa: S102
    return normalizar(" ".join(d["texto"] for d in ns["corpus"]))


def construir() -> list[dict]:
    eval_m2 = {json.loads(l)["id"]: json.loads(l) for l in EVAL_M2.read_text(encoding="utf-8").splitlines()}
    limpios = {r["id"]: r for r in ARITMETICA + COMPUESTO}
    registros = ARITMETICA + COMPUESTO

    for mid, original, subtipo, texto in MAL_ESCRITO:
        if original in limpios:
            base = limpios[original]
            herr = base["herramientas"]
        else:
            base = eval_m2[original]
            fam, opc = HERRAMIENTAS_M2[original]
            herr = {"familias": fam, "opcionales": opc}
        registros.append(E(
            mid, "mal_escrito", subtipo, texto, base["esperado"],
            f"Versión mal escrita de {original}: debe acertar igual que el original. " + base["criterio"],
            base["evaluacion"], herr["familias"], herr["opcionales"], original=original))
    return registros


def validar(registros: list[dict]) -> list[str]:
    errores = []
    ids = [r["id"] for r in registros]
    for i, n in Counter(ids).items():
        if n > 1:
            errores.append(f"id duplicado: {i}")

    eval_m2 = [json.loads(l) for l in EVAL_M2.read_text(encoding="utf-8").splitlines()]
    ids_m2 = {c["id"] for c in eval_m2}
    if set(HERRAMIENTAS_M2) != ids_m2:
        errores.append("la anotación de herramientas de M2 no cubre exactamente los casos de M2")
    for cid, (fam, opc) in HERRAMIENTAS_M2.items():
        for f in fam + opc:
            if f not in FAMILIAS:
                errores.append(f"[{cid}] familia desconocida: {f}")

    corpus_doc = _texto_corpus_documental()
    ids_todos = {r["id"] for r in registros} | ids_m2
    entradas_limpias = {normalizar(c["input"]) for c in eval_m2} | {
        normalizar(r["input"]) for r in registros if r["bloque"] != "mal_escrito"}

    for r in registros:
        rid = r["id"]
        if r["bloque"] not in BLOQUES:
            errores.append(f"[{rid}] bloque inválido: {r['bloque']}")
        for f in r["herramientas"]["familias"] + r["herramientas"]["opcionales"]:
            if f not in FAMILIAS:
                errores.append(f"[{rid}] familia desconocida: {f}")

        ev = r["evaluacion"]
        if ev["tipo"].startswith("numerico") and a_numero(ev["valor"]) is None:
            errores.append(f"[{rid}] 'valor' no numérico: {ev['valor']}")

        if r["check"]:
            esperado = eval(r["check"], {"__builtins__": {"min": min}},  # noqa: S307
                            {"Fraction": Fraction, "math": math})
            obtenido = a_numero(ev["valor"])
            if obtenido is None or abs(float(esperado) - obtenido) > TOLERANCIA:
                errores.append(f"[{rid}] ARITMÉTICA: check={r['check']} da {esperado}, valor={ev['valor']}")

        if ev["tipo"] == "numerico" and r["bloque"] != "mal_escrito" and ev["valor"] not in r["esperado"]:
            errores.append(f"[{rid}] el valor {ev['valor']} no aparece en 'esperado'")

        if r["bloque"] == "compuesto":
            if not r["dato"] or normalizar(r["dato"]) not in corpus_doc:
                errores.append(f"[{rid}] el dato no está en el corpus documental: {r['dato']!r}")
            if "documentos" not in r["herramientas"]["familias"]:
                errores.append(f"[{rid}] un caso compuesto debe requerir 'documentos'")

        if r["bloque"] == "mal_escrito":
            if r["original"] not in ids_todos:
                errores.append(f"[{rid}] original inexistente: {r['original']}")
            if normalizar(r["input"]) in entradas_limpias:
                errores.append(f"[{rid}] no está mal escrito: coincide con un caso limpio")
        elif normalizar(r["input"]) in {normalizar(c["input"]) for c in eval_m2}:
            errores.append(f"[{rid}] repite un caso del eval set de M2")

    # No contaminación con el corpus de entrenamiento.
    entrenamiento = {normalizar(json.loads(l)["entrada"])
                     for l in CORPUS_ENTRENAMIENTO.read_text(encoding="utf-8").splitlines()}
    for r in registros:
        if normalizar(r["input"]) in entrenamiento:
            errores.append(f"[{r['id']}] CONTAMINACIÓN: está en el corpus de entrenamiento")
    return errores


def reporte(registros: list[dict]) -> str:
    lineas = ["", "=" * 70, "EVAL SET M3 (AGENTE) — REPORTE", "=" * 70]
    conteo = Counter(r["bloque"] for r in registros)
    lineas.append(f"Total de casos: {len(registros)}")
    for b in BLOQUES:
        lineas.append(f"  {b:12s} {conteo[b]:2d}")
    lineas.append("")
    for r in registros:
        fam = ",".join(r["herramientas"]["familias"]) or "(ninguna)"
        extra = f"  <- {r['original']}" if r["original"] else ""
        lineas.append(f"  {r['id']:7s} {r['bloque']:12s} {r['subtipo']:26s} {fam}{extra}")
    lineas.append("=" * 70)
    return "\n".join(lineas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="solo validar")
    args = parser.parse_args()

    registros = construir()
    errores = validar(registros)
    if errores:
        print("VALIDACIÓN FALLIDA:", file=sys.stderr)
        for e in errores:
            print("  -", e, file=sys.stderr)
        raise SystemExit(1)

    print(reporte(registros))
    if args.check:
        print("\nValidación superada (no se escribió ningún archivo).")
        return

    with SALIDA.open("w", encoding="utf-8") as fh:
        for r in registros:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "check"}, ensure_ascii=False) + "\n")
    with SALIDA_ANOTACION_M2.open("w", encoding="utf-8") as fh:
        for cid, (fam, opc) in HERRAMIENTAS_M2.items():
            fh.write(json.dumps({"id": cid, "herramientas": {"familias": fam, "opcionales": opc}},
                                ensure_ascii=False) + "\n")
    print(f"\nEscrito: {SALIDA}  ({len(registros)} casos)")
    print(f"Escrito: {SALIDA_ANOTACION_M2}  ({len(HERRAMIENTAS_M2)} anotaciones)")


if __name__ == "__main__":
    main()
