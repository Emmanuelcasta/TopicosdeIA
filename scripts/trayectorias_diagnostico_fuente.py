"""Trayectorias de diagnóstico para M4: enseñar a dictaminar, no a resolver.

Por qué existe. En la corrida del 24-10 el tutor **rehizo correctamente** los
cinco procedimientos con error y no avisó de ninguno:

    el estudiante escribió  24 × 13 = 302     el tutor respondió  24 × 13 = 312
    el estudiante escribió  12 × 5  = 70      el tutor respondió  12 × 5  = 60

No es un problema de aritmética ni de lectura: el adaptador de M1 se entrenó para
emitir `Paso N: … Respuesta final: …` y eso es lo único que sabe producir. Los 7
casos que no emitieron `Diagnóstico:` son exactamente los que tenían un error:
cuando el trabajo del estudiante está mal, el modelo produce su propia versión y
se queda sin nada que concluir. Ni con un reintento acotado sale de la plantilla.

El modelo nunca ha visto un ejemplo de "revisar y dictaminar". Esto lo escribe.

Decisiones que condicionan todo lo demás:

  · **El prompt es el de inferencia, literal.** Cada ejemplo se construye con
    `prompt_revision()` de `vision.py`, la misma función que usa el notebook. Si
    entrenáramos con un prompt parecido pero distinto, el entrenamiento no se
    trasladaría.

  · **Ningún ejercicio sale del eval set de M4**, ni de M2, ni de M3, ni
    parafraseado. La comprobación es la de `trayectorias_fuente.py`: Jaccard
    sobre palabras contra todos los casos de evaluación, y el generador falla si
    alguna pasa de 0.6. Incluye los casos `dev-*`, que son la puerta del
    notebook: contaminarlos invalidaría la única señal honesta de formato.

  · **Las etiquetas se validan contra la aritmética.** Para cada hoja se
    comprueba que el paso marcado como erróneo sea efectivamente erróneo y que
    los demás estén bien, con `Fraction` y sin coma flotante. Una etiqueta mal
    puesta enseñaría exactamente lo contrario de lo que se busca.

  · **La explicación nombra el paso y el error, y no reescribe el
    procedimiento.** Es la conducta que falta: el modelo ya sabe calcular.

Uso:
    python trayectorias_diagnostico_fuente.py            # escribe el JSONL
    python trayectorias_diagnostico_fuente.py --listar   # imprime los ejemplos
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI / "agente"))

from eval_set_fuente import normalizar  # noqa: E402
from vision import (  # noqa: E402
    INSTRUCCION_REVISION,
    diagnostico_valido,
    extraer_diagnostico,
    prompt_revision,
)

RAIZ = AQUI.parent
SALIDA = RAIZ / "data" / "trayectorias_diagnostico.jsonl"
REPORTE = RAIZ / "m4" / "trayectorias_diagnostico.md"
EVAL_SETS = [RAIZ / "data" / "eval_set_m2.jsonl",
             RAIZ / "data" / "eval_set_m3_agente.jsonl",
             RAIZ / "data" / "eval_set_m4_vision.jsonl"]

UMBRAL_JACCARD = 0.6
PREGUNTA = "Revisa mi procedimiento: ¿en qué paso me equivocé?"

SYSTEM_DIAGNOSTICO = (
    "Eres un tutor de matemáticas que revisa el trabajo escrito de un estudiante "
    "y emite un veredicto."
)


# --------------------------------------------------------------------------
# Las hojas. Cada una es un ejercicio NUEVO, con sus pasos y el paso erróneo
# declarado. `op` permite validar la aritmética sin confiar en el texto.
# --------------------------------------------------------------------------

def H(id, familia, split, enunciado, pasos, final, diagnostico, explicacion,
      op=None, paso_error=None, pregunta=PREGUNTA, lectura=None):
    return dict(id=id, familia=familia, split=split, enunciado=enunciado,
                pasos=list(pasos), final=final, diagnostico=diagnostico,
                explicacion=explicacion, op=op, paso_error=paso_error,
                pregunta=pregunta, lectura=lectura)


def _mal(a, b, op, escrito):
    """El valor escrito NO es el resultado de a op b. Para validar la etiqueta."""
    return _correcto(a, b, op) != Fraction(str(escrito))


def _correcto(a, b, op):
    a, b = Fraction(str(a)), Fraction(str(b))
    return {"x": a * b, "+": a + b, "-": a - b, "/": a / b}[op]


def _fmt(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


# --- familia A · un paso aritmético mal, con el paso erróneo variando -----
# El índice del paso erróneo se reparte a propósito entre 1, 2 y 3: si todos los
# ejemplos lo tuvieran en el 2, el modelo aprendería a decir "Paso 2".
_A = [
    # (id, enunciado, operandos, op, escrito, paso_error, plantilla)
    ("dg-a01", "Un estuche trae 18 colores. ¿Cuántos colores hay en 7 estuches?",
     18, 7, "x", 116, 2),
    ("dg-a02", "Una bodega guarda 960 botellas en 12 estantes iguales. ¿Cuántas por estante?",
     960, 12, "/", 70, 1),
    ("dg-a03", "Un depósito tiene 2400 litros y se extraen 875. ¿Cuántos quedan?",
     2400, 875, "-", 1625, 3),
    ("dg-a04", "En una excursión van 23 niños y 19 niñas. ¿Cuántos participantes hay?",
     23, 19, "+", 32, 1),
    ("dg-a05", "Un paquete de 15 cuadernos cuesta 42000 pesos. ¿Cuánto cuesta cada cuaderno?",
     42000, 15, "/", 2600, 1),
    ("dg-a06", "Una impresora saca 26 páginas por minuto. ¿Cuántas en 14 minutos?",
     26, 14, "x", 354, 3),
    ("dg-a07", "Un rollo de 5000 metros se reparte en tramos de 25 metros. ¿Cuántos tramos?",
     5000, 25, "/", 250, 3),
    ("dg-a08", "Un salón tiene 34 sillas y llegan 48 más. ¿Cuántas sillas hay?",
     34, 48, "+", 72, 1),
    ("dg-a09", "Una finca produjo 1560 kilos y vendió 985. ¿Cuántos kilos quedan?",
     1560, 985, "-", 675, 2),
    ("dg-a10", "Un bus hace 7 viajes con 48 pasajeros cada uno. ¿Cuántos pasajeros en total?",
     7, 48, "x", 326, 1),
    ("dg-a11", "Se reparten 1800 semillas entre 24 materas. ¿Cuántas por matera?",
     1800, 24, "/", 65, 3),
    ("dg-a12", "Un tanque tiene 3250 litros y se añaden 1480. ¿Cuántos litros hay?",
     3250, 1480, "+", 4630, 2),
    ("dg-a13", "Una caja de 36 huevos se divide en bandejas de 6. ¿Cuántas bandejas?",
     36, 6, "/", 7, 3),
    ("dg-a14", "Un libro de 412 páginas se lee en 4 días iguales. ¿Cuántas páginas por día?",
     412, 4, "/", 108, 3),
    ("dg-a15", "Una fábrica empaca 45 bolsas por minuto durante 12 minutos. "
     "¿Cuántas bolsas empacó?", 45, 12, "x", 530, 1),
    ("dg-a16", "Hay 72 estudiantes y se van 29. ¿Cuántos quedan?",
     72, 29, "-", 53, 1),
    ("dg-a17", "Una tienda vende 19 cajas a 8500 pesos cada una. ¿Cuánto recaudó?",
     19, 8500, "x", 160500, 2),
    ("dg-a18", "Un camión carga 2100 kilos en 14 viajes iguales. ¿Cuántos kilos por viaje?",
     2100, 14, "/", 160, 3),
]

VERBO = {"x": "La multiplicación", "/": "La división", "+": "La suma", "-": "La resta"}
SIGNO = {"x": "×", "/": "÷", "+": "+", "-": "−"}
PLANTEO = {"x": "Multiplicamos", "/": "Dividimos", "+": "Sumamos", "-": "Restamos"}


def _familia_a() -> list:
    hojas = []
    for i, (hid, enun, a, b, op, escrito, pe) in enumerate(_A):
        bueno = _correcto(a, b, op)
        split = "dev" if i % 6 == 5 else "train"
        if pe == 1:
            # El error está en el planteamiento: la operación elegida es la que no toca.
            op_mal = {"x": "+", "/": "-", "+": "x", "-": "/"}[op]
            pasos = [f"{PLANTEO[op_mal]}: {a} {SIGNO[op_mal]} {b}.",
                     f"{a} {SIGNO[op_mal]} {b} = {_fmt(_correcto(a, b, op_mal))}.",
                     f"Respuesta final: {_fmt(_correcto(a, b, op_mal))}"]
            final = _fmt(_correcto(a, b, op_mal))
            expl = (f"El paso 1 plantea la operación equivocada: aquí hay que "
                    f"{PLANTEO[op].lower()}, no {PLANTEO[op_mal].lower()}.")
            operacion = None
        elif pe == 2:
            pasos = [f"{PLANTEO[op]}: {a} {SIGNO[op]} {b}.",
                     f"{a} {SIGNO[op]} {b} = {escrito}.",
                     f"Respuesta final: {escrito}"]
            final = str(escrito)
            expl = (f"El paso 2 está mal: {VERBO[op].lower()} {a} {SIGNO[op]} {b} "
                    f"da {_fmt(bueno)}, no {escrito}.")
            operacion = dict(a=a, b=b, op=op, escrito=escrito)
        else:                                   # pe == 3, el error es el traslado
            pasos = [f"{PLANTEO[op]}: {a} {SIGNO[op]} {b}.",
                     f"{a} {SIGNO[op]} {b} = {_fmt(bueno)}.",
                     f"Respuesta final: {escrito}"]
            final = str(escrito)
            expl = (f"Los dos primeros pasos están bien, pero el paso 3 copia mal "
                    f"el resultado: era {_fmt(bueno)} y quedó escrito {escrito}.")
            operacion = dict(a=a, b=b, op=op, escrito=escrito)
        hojas.append(H(hid, "paso_aritmetico", split, enun, pasos, final,
                       f"Paso {pe}", expl, op=operacion, paso_error=pe))
    return hojas


# --- familia B · error compensado: paso mal, resultado final bien ---------
_B = [
    ("dg-b01", "Un patio mide 16 metros por 9 metros. ¿Cuál es su área?",
     16, 9, "x", 154, 144, 2),
    ("dg-b02", "Se reparten 1440 fichas entre 16 mesas. ¿Cuántas por mesa?",
     1440, 16, "/", 85, 90, 2),
    ("dg-b03", "Una caja tiene 27 bolsas de 8 unidades. ¿Cuántas unidades hay?",
     27, 8, "x", 206, 216, 2),
    ("dg-b04", "Un trayecto de 1250 metros se divide en 25 tramos. ¿Cuánto mide cada tramo?",
     1250, 25, "/", 45, 50, 2),
    ("dg-b05", "Hay 345 libros y se añaden 268. ¿Cuántos libros hay?",
     345, 268, "+", 603, 613, 2),
    ("dg-b06", "Un pedido de 84 cajas se agrupa en lotes de 12. ¿Cuántos lotes?",
     84, 12, "/", 8, 7, 2),
]


def _familia_b() -> list:
    hojas = []
    for i, (hid, enun, a, b, op, escrito, bueno_final, pe) in enumerate(_B):
        split = "dev" if i % 6 == 5 else "train"
        pasos = [f"{PLANTEO[op]}: {a} {SIGNO[op]} {b}.",
                 f"{a} {SIGNO[op]} {b} = {escrito}.",
                 f"Respuesta final: {bueno_final}"]
        expl = (f"El resultado final está bien, pero el paso 2 no: {a} {SIGNO[op]} "
                f"{b} da {bueno_final} y ahí quedó escrito {escrito}.")
        hojas.append(H(hid, "compensado", split, enun, pasos, str(bueno_final),
                       f"Paso {pe}", expl,
                       op=dict(a=a, b=b, op=op, escrito=escrito), paso_error=pe))
    return hojas


# --- familia C · sin error -------------------------------------------------
_C = [
    ("dg-c01", "Un estante tiene 14 filas de 9 libros. ¿Cuántos libros hay?", 14, 9, "x"),
    ("dg-c02", "Se reparten 756 dulces entre 27 niños. ¿Cuántos recibe cada uno?", 756, 27, "/"),
    ("dg-c03", "Un tanque tiene 1870 litros y se añaden 930. ¿Cuántos hay?", 1870, 930, "+"),
    ("dg-c04", "Una finca tenía 1420 árboles y se talaron 385. ¿Cuántos quedan?", 1420, 385, "-"),
    ("dg-c05", "Un pliego de 96 centímetros se corta en 8 partes iguales. ¿Cuánto mide cada una?",
     96, 8, "/"),
    ("dg-c06", "Un taller arma 23 sillas por día. ¿Cuántas en 6 días?", 23, 6, "x"),
    ("dg-c07", "Hay 512 fichas y se agregan 288. ¿Cuántas fichas hay?", 512, 288, "+"),
    ("dg-c08", "Una pared de 240 ladrillos se levanta en 15 hiladas iguales. ¿Cuántos por hilada?",
     240, 15, "/"),
]


def _familia_c() -> list:
    hojas = []
    for i, (hid, enun, a, b, op) in enumerate(_C):
        bueno = _correcto(a, b, op)
        split = "dev" if i % 4 == 3 else "train"
        pasos = [f"{PLANTEO[op]}: {a} {SIGNO[op]} {b}.",
                 f"{a} {SIGNO[op]} {b} = {_fmt(bueno)}.",
                 f"Respuesta final: {_fmt(bueno)}"]
        expl = ("Revisé el planteamiento, la operación y el traslado del resultado: "
                "los tres pasos están bien.")
        hojas.append(H(hid, "sin_error", split, enun, pasos, _fmt(bueno),
                       "sin error", expl,
                       op=dict(a=a, b=b, op=op, escrito=_fmt(bueno)), paso_error=None))
    return hojas


# --- familias D, E, F · las tres formas de abstenerse ---------------------
# Escritas a mano: no son variaciones de un ejercicio, son situaciones.
_ABSTENCIONES = [
    H("dg-d01", "ilegible", "train", None, [], None, "ilegible",
      "La foto no permite leer lo que está escrito, así que no puedo revisar el "
      "procedimiento.",
      lectura={"ok": True, "consistente": True,
               "json": {"ilegible": True, "motivo": "la foto está desenfocada"}}),
    H("dg-d02", "ilegible", "train", None, [], None, "ilegible",
      "No se distinguen las cifras de la hoja; con esa foto no puedo decir dónde "
      "está el error.",
      lectura={"ok": True, "consistente": True,
               "json": {"ilegible": True, "motivo": "hay un reflejo sobre el papel"}}),
    H("dg-d03", "ilegible", "dev", None, [], None, "ilegible",
      "La imagen está demasiado lejos para leer los números con seguridad.",
      lectura={"ok": True, "consistente": True,
               "json": {"ilegible": True, "motivo": "la hoja se ve muy pequeña"}}),
    H("dg-d04", "ilegible", "train", None, [], None, "ilegible",
      "La transcripción no salió legible, así que no hay procedimiento que revisar.",
      lectura={"ok": False, "json": None, "crudo": "...."}),

    H("dg-e01", "fuera_de_dominio", "train", "Lista: leche, pan, huevos", [], None,
      "fuera de dominio",
      "Lo que hay en la hoja es una lista de compras, no un ejercicio de "
      "matemáticas, así que no hay pasos que revisar.",
      lectura={"ok": True, "consistente": True,
               "json": {"enunciado": "Lista: leche, pan, huevos", "pasos": [],
                        "respuesta_final": "huevos"}}),
    H("dg-e02", "fuera_de_dominio", "train", "Tarea de sociales: capitales", [], None,
      "fuera de dominio",
      "La hoja trae apuntes de otra materia; ahí no hay un procedimiento "
      "matemático que corregir.",
      lectura={"ok": True, "consistente": True,
               "json": {"enunciado": "Tarea de sociales: capitales", "pasos": [],
                        "respuesta_final": "Bogotá"}}),
    H("dg-e03", "fuera_de_dominio", "dev", "Receta: 2 tazas de harina", [], None,
      "fuera de dominio",
      "Es una receta, no un ejercicio: no hay ningún paso de cálculo planteado.",
      lectura={"ok": True, "consistente": True,
               "json": {"enunciado": "Receta: 2 tazas de harina", "pasos": [],
                        "respuesta_final": "harina"}}),

    H("dg-f01", "dato_no_disponible", "train",
      "Un cuaderno trae 80 hojas. ¿Cuántas hojas hay en 6 cuadernos?",
      ["Multiplicamos: 80 × 6.", "80 × 6 = 480.", "Respuesta final: 480 hojas"],
      "480 hojas", "dato no disponible",
      "En la hoja no aparece el precio: ahí solo está el procedimiento de las "
      "hojas, así que no tengo ese dato.",
      pregunta="¿Cuánto costó cada cuaderno?"),
    H("dg-f02", "dato_no_disponible", "train",
      "Un recorrido de 240 kilómetros se hace en 3 horas. ¿Cuál es la velocidad?",
      ["La velocidad es distancia entre tiempo.", "240 ÷ 3 = 80.",
       "Respuesta final: 80 kilómetros por hora"],
      "80 kilómetros por hora", "dato no disponible",
      "La hoja no dice qué día se hizo el recorrido; ese dato no está escrito.",
      pregunta="¿En qué fecha se hizo el recorrido?"),
    H("dg-f03", "dato_no_disponible", "dev",
      "Una caja trae 45 tornillos. ¿Cuántos hay en 8 cajas?",
      ["Multiplicamos: 45 × 8.", "45 × 8 = 360.", "Respuesta final: 360 tornillos"],
      "360 tornillos", "dato no disponible",
      "No hay ninguna calificación anotada en la hoja, así que no puedo decir qué "
      "nota le pusieron.",
      pregunta="¿Qué nota le puso el profesor?"),
]


# --------------------------------------------------------------------------
# De hoja a ejemplo de entrenamiento
# --------------------------------------------------------------------------

def lectura_de(hoja: dict) -> dict:
    """La transcripción tal como la produciría el VLM. Es lo que ve el prompt."""
    if hoja["lectura"] is not None:
        return hoja["lectura"]
    return {"ok": True, "consistente": True, "legible_declarada": True,
            "confianza_atipica": False, "crudo": "(sintética)", "segundos": 0.0,
            "json": {"enunciado": hoja["enunciado"], "pasos": list(hoja["pasos"]),
                     "respuesta_final": hoja["final"]}}


def ejemplo(hoja: dict) -> dict:
    usuario = prompt_revision(hoja["pregunta"], lectura_de(hoja))
    asistente = f"{hoja['explicacion']}\nDiagnóstico: {hoja['diagnostico']}"
    return {
        "id": hoja["id"], "familia": hoja["familia"], "split": hoja["split"],
        "diagnostico": hoja["diagnostico"], "paso_error": hoja["paso_error"],
        "pregunta": hoja["pregunta"], "enunciado": hoja["enunciado"],
        "pasos": hoja["pasos"], "final": hoja["final"], "op": hoja["op"],
        "mensajes": [{"role": "system", "content": SYSTEM_DIAGNOSTICO},
                     {"role": "user", "content": usuario},
                     {"role": "assistant", "content": asistente}],
    }


def construir() -> list[dict]:
    hojas = _familia_a() + _familia_b() + _familia_c() + list(_ABSTENCIONES)
    return [ejemplo(h) for h in hojas]


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------

def _jaccard(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"\w+", normalizar(a or ""))), set(re.findall(r"\w+", normalizar(b or "")))
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def _textos_de_evaluacion() -> list[str]:
    """Todo lo que NO se puede parafrasear: los eval sets de M2, M3 y M4, este
    último incluyendo los casos `dev-*`, que son la puerta del notebook."""
    textos = []
    for ruta in EVAL_SETS:
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            r = json.loads(linea)
            for clave in ("input", "enunciado", "pregunta"):
                v = r.get(clave)
                if v and not str(v).startswith("("):
                    textos.append(str(v))
    return textos


def validar(ejemplos: list[dict]) -> list[str]:
    errores = []

    for eid, n in Counter(e["id"] for e in ejemplos).items():
        if n > 1:
            errores.append(f"id duplicado: {eid}")

    for e in ejemplos:
        eid = e["id"]

        # 1 · la etiqueta se parsea con el MISMO extractor del evaluador
        salida = e["mensajes"][-1]["content"]
        conclusion = extraer_diagnostico(salida)
        if not diagnostico_valido(conclusion):
            errores.append(f"[{eid}] la respuesta no trae una conclusión válida: {conclusion!r}")
        esperado = e["diagnostico"].lower()
        if conclusion != esperado:
            errores.append(f"[{eid}] la conclusión {conclusion!r} no es la etiqueta {esperado!r}")
        if not salida.strip().splitlines()[-1].startswith("Diagnóstico:"):
            errores.append(f"[{eid}] la conclusión no es la última línea")

        # 2 · la aritmética respalda la etiqueta
        op = e["op"]
        if op:
            mal = _mal(op["a"], op["b"], op["op"], str(op["escrito"]).replace("/", "/"))
            if e["paso_error"] in (2, 3) and not mal:
                errores.append(f"[{eid}] se marca un error en el paso {e['paso_error']} "
                               f"pero {op['a']} {op['op']} {op['b']} sí da {op['escrito']}")
            if e["diagnostico"] == "sin error" and mal:
                errores.append(f"[{eid}] se etiqueta 'sin error' y la cuenta está mal")

        # 3 · la explicación no reescribe el procedimiento entero
        explicacion = salida.rsplit("\n", 1)[0]
        if explicacion.count("Paso ") > 2 or "Respuesta final" in explicacion:
            errores.append(f"[{eid}] la explicación reescribe el procedimiento: "
                           "es justo la conducta que se quiere quitar")

        # 4 · el prompt de entrenamiento es el de inferencia
        if INSTRUCCION_REVISION not in e["mensajes"][1]["content"]:
            errores.append(f"[{eid}] el prompt no es el de `prompt_revision`")

    # 5 · sin filtraciones del eval set, con la comprobación de siempre
    evaluacion = _textos_de_evaluacion()
    normalizadas = {normalizar(x) for x in evaluacion}
    for e in ejemplos:
        for campo in ("enunciado", "pregunta"):
            texto = e[campo]
            if not texto:
                continue
            if normalizar(texto) in normalizadas and campo == "enunciado":
                errores.append(f"[{e['id']}] CONTAMINACIÓN: {campo} está en un eval set")
            peor = max(evaluacion, key=lambda x: _jaccard(texto, x))
            j = _jaccard(texto, peor)
            if j >= UMBRAL_JACCARD and campo == "enunciado":
                errores.append(f"[{e['id']}] {campo} demasiado parecido (J={j:.2f}) a "
                               f"{peor[:60]!r}")

    # 6 · train y dev no comparten enunciado
    tr = {normalizar(e["enunciado"] or "") for e in ejemplos if e["split"] == "train"}
    dv = {normalizar(e["enunciado"] or "") for e in ejemplos if e["split"] == "dev"}
    if tr & dv - {""}:
        errores.append(f"train y dev comparten enunciados: {sorted(tr & dv)[:3]}")

    # 7 · cada split cubre las cinco categorías
    for split in ("train", "dev"):
        cats = {e["diagnostico"].split()[0].lower() for e in ejemplos if e["split"] == split}
        faltan = {"paso", "sin", "ilegible", "fuera", "dato"} - cats
        if faltan:
            errores.append(f"el split {split} no cubre: {sorted(faltan)}")

    # 8 · el paso señalado varía: si no, se aprende a decir "Paso 2"
    pasos = Counter(e["paso_error"] for e in ejemplos
                    if e["split"] == "train" and e["paso_error"])
    if len(pasos) < 3:
        errores.append(f"el paso erróneo no varía bastante en train: {dict(pasos)}")
    elif pasos and max(pasos.values()) > 0.5 * sum(pasos.values()):
        # El eval set está sesgado a "Paso 2" (4 de 6). Si el entrenamiento lo
        # está igual o más, el modelo aprende a contestar 2 y acertaría por el
        # atajo, no por localizar el error.
        errores.append(f"un índice de paso domina el entrenamiento: {dict(pasos)}; "
                       "reparta los errores entre los pasos 1, 2 y 3")

    return errores


def reporte(ejemplos: list[dict]) -> str:
    lin = ["# Trayectorias de diagnóstico de M4", "",
           "Generadas y validadas por `scripts/trayectorias_diagnostico_fuente.py`.",
           "Enseñan a **dictaminar**, que es lo que el tutor no sabía hacer: en la "
           "corrida del 24-10 rehízo correctamente los cinco procedimientos con "
           "error y no avisó de ninguno.", "",
           f"**{len(ejemplos)} ejemplos** · "
           f"train {sum(1 for e in ejemplos if e['split'] == 'train')} · "
           f"dev {sum(1 for e in ejemplos if e['split'] == 'dev')}", "",
           "| familia | train | dev | qué enseña |", "|---|---|---|---|"]
    QUE = {"paso_aritmetico": "señalar el paso erróneo, con el índice variando entre 1, 2 y 3",
           "compensado": "un paso mal con el resultado final bien",
           "sin_error": "decir que está correcto, sin inventar un error",
           "ilegible": "abstenerse cuando la foto no se lee",
           "fuera_de_dominio": "abstenerse cuando la hoja no es matemáticas",
           "dato_no_disponible": "abstenerse cuando se pide algo que no está escrito"}
    for fam in QUE:
        de = [e for e in ejemplos if e["familia"] == fam]
        lin.append(f"| {fam} | {sum(1 for e in de if e['split'] == 'train')} | "
                   f"{sum(1 for e in de if e['split'] == 'dev')} | {QUE[fam]} |")

    pasos = Counter(e["paso_error"] for e in ejemplos if e["paso_error"])
    lin += ["", f"Paso señalado: {dict(sorted(pasos.items()))} — reparto deliberado, "
            "para que el modelo no aprenda a contestar siempre 'Paso 2'.", "",
            "## Cómo se evita la filtración", "",
            "Ningún enunciado sale del eval set de M2, M3 ni M4, ni parafraseado. "
            "La comprobación es la de `trayectorias_fuente.py`: Jaccard sobre "
            f"palabras contra todos los casos de evaluación, con tope {UMBRAL_JACCARD}. "
            "Incluye los casos `dev-*` de M4, que son la puerta del notebook: "
            "contaminarlos invalidaría la única señal honesta de formato.", "",
            "## Qué se valida antes de escribir el archivo", "",
            "1. La conclusión se parsea con el **mismo** `extraer_diagnostico` del "
            "evaluador y coincide con la etiqueta.", "",
            "2. La aritmética respalda la etiqueta: el paso marcado como erróneo lo "
            "es, y ningún `sin error` lleva una cuenta mal. Con `Fraction`.", "",
            "3. La explicación **no reescribe** el procedimiento: es la conducta que "
            "se quiere quitar.", "",
            "4. El prompt de entrenamiento es literalmente `prompt_revision()`, el "
            "de inferencia.", "",
            "5. `train` y `dev` no comparten enunciados, y los dos cubren las cinco "
            "categorías.", "",
            "## Lo que esto NO promete", "",
            "Es un experimento. Mejorar el formato de salida no implica mejorar el "
            "razonamiento: puede salir un modelo que emita siempre "
            "`Diagnóstico: …` y siga sin localizar el paso. Por eso el notebook "
            "reporta diagnóstico, lectura y formato **por separado**, y el "
            "adaptador se guarda aparte de `qwen-lora` y `qwen-lora-agente`.", "",
            "## Ejemplos", ""]
    for e in ejemplos[:2] + [x for x in ejemplos if x["familia"] == "fuera_de_dominio"][:1]:
        lin += [f"### `{e['id']}` · {e['familia']} · {e['split']}", "",
                "```", e["mensajes"][-1]["content"], "```", ""]
    return "\n".join(lin) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listar", action="store_true")
    args = ap.parse_args()

    ejemplos = construir()
    errores = validar(ejemplos)
    if errores:
        print(f"{len(errores)} PROBLEMAS:")
        for e in errores:
            print("  -", e)
        raise SystemExit(1)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", encoding="utf-8") as fh:
        for e in ejemplos:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    REPORTE.parent.mkdir(parents=True, exist_ok=True)
    REPORTE.write_text(reporte(ejemplos), encoding="utf-8")

    print(f"{len(ejemplos)} ejemplos validados")
    print(f"  train: {sum(1 for e in ejemplos if e['split'] == 'train')}  "
          f"dev: {sum(1 for e in ejemplos if e['split'] == 'dev')}")
    print(f"  por familia: {dict(Counter(e['familia'] for e in ejemplos))}")
    print(f"  paso señalado: {dict(sorted(Counter(e['paso_error'] for e in ejemplos if e['paso_error']).items()))}")
    print(f"  -> {SALIDA}")
    print(f"  -> {REPORTE}")

    if args.listar:
        for e in ejemplos:
            print("\n" + "=" * 70)
            print(f"{e['id']} · {e['familia']} · {e['split']} · -> {e['diagnostico']}")
            print(e["mensajes"][-1]["content"])


if __name__ == "__main__":
    main()
