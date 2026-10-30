"""
Eval set visual de M5 — hojas NUEVAS, porque las de M4 ya se gastaron.

Por qué hace falta otro. Los 12 casos de M4 se ejecutaron en tres corridas
exploratorias (4/12, 1/12 y 2/12, según `m4/README.md` y `m4/CONCLUSIONES_M4.md`),
y una de esas corridas acertó 4/12 repitiendo «Paso 2» en las 11 respuestas con
conclusión: un sistema que contestara siempre «Paso 2» habría acertado lo mismo.
Esos 12 casos siguen sirviendo de **regresión** —comprobar que un cambio no
rompe lo que antes funcionaba— pero no para afirmar que la configuración nueva
generaliza. Para eso hacen falta hojas que el sistema no haya visto.

Qué añade este conjunto sobre el de M4, y por qué cada cosa:

  * **Pares legible / borrosa.** Cada hoja se fotografía dos veces: nítida y
    movida. Es la única forma de medir la salvaguarda de ilegibles sin inventar
    un umbral: con el par se puede decir cuántas borrosas detecta y cuántas
    nítidas rechaza por error. La calibración de M4 usaba texto IMPRESO generado
    por script, que sirve para ver si el mecanismo corre y para nada más.
  * **Transcripción de referencia.** M4 no la tenía anotada, y por eso
    `lectura_visual_erronea` quedó `sin_evidencia` en la taxonomía: no había
    forma de saber si el tutor falló por leer mal o por razonar mal. Con la
    transcripción escrita a mano se separan las dos cosas.
  * **El paso del error y la respuesta esperada**, como en M4, para que la
    verdad no dependa de ningún juez.

Bloques, que se reportan por separado y nunca promediados:

  A · LOCALIZACION (4)   error en un paso concreto; hay que nombrar el paso
  B · COMPENSADO (2)     paso mal y resultado final CORRECTO
  C · SIN_ERROR (3)      procedimientos correctos; miden el falso error
  D · ADVERSARIAL (3)    ilegible real, fuera de dominio, dato ausente

Uso:
    python scripts/eval_set_m5_vision_fuente.py            # valida y escribe
    python scripts/eval_set_m5_vision_fuente.py --check    # solo valida
    python scripts/eval_set_m5_vision_fuente.py --hojas    # imprime las hojas
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import unicodedata
from collections import Counter
from fractions import Fraction
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_M5 = RAIZ / "m4" / "eval_set"          # junto al de M4, no encima
SALIDA_JSONL = RAIZ / "data" / "eval_set_m5_vision.jsonl"
SALIDA_CSV = DIR_M5 / "eval_set_m5.csv"
SALIDA_HOJAS = DIR_M5 / "HOJAS_M5.md"

# De estos conjuntos NO se puede reutilizar un enunciado: si se repitiera, una
# mejora podría ser memorización del adaptador o del corpus, y no se podría
# distinguir. La validación lo comprueba y falla.
PREVIOS = [RAIZ / "data" / "eval_set_m4_vision.jsonl",
           RAIZ / "data" / "eval_set_m2.jsonl",
           RAIZ / "data" / "eval_set_m3_agente.jsonl"]
CORPUS_ENTRENAMIENTO = RAIZ / "data" / "math_tutor_dataset.jsonl"

PREGUNTA = "Revisa mi procedimiento: ¿en qué paso me equivoqué?"


def normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", (texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def paso(n: int) -> dict:
    """Las claves que acepta una respuesta que señala el paso n."""
    return {"tipo": "contiene_alguna",
            "claves": [f"paso {n}", f"paso {n}:", f"el paso {n}",
                       f"paso número {n}", f"paso n.º {n}"],
            "prohibido": [f"paso {k}" for k in (1, 2, 3, 4, 5) if k != n]}


SIN_ERROR = {"tipo": "contiene_alguna",
             "claves": ["sin error", "no hay error", "está correcto",
                        "es correcto", "todo bien", "ningún paso"],
             "prohibido": ["paso 1", "paso 2", "paso 3", "paso 4"]}

ABSTENCION = {"tipo": "contiene_alguna",
              "claves": ["no puedo", "no se alcanza a leer", "no se lee",
                         "no distingo", "no es un ejercicio", "no aparece",
                         "no está en la foto", "no puedo leer"],
              "prohibido": ["paso 1", "paso 2", "paso 3", "paso 4"]}


def C(id, bloque, enunciado, pasos, paso_error, valor_final_escrito,
      valor_correcto, evaluacion, criterio, check=None, par_borroso=True,
      solo_borrosa=False, nota_foto=None, pregunta=PREGUNTA):
    """Un caso = una hoja manuscrita + dos fotos + su criterio.

    `pasos` es a la vez lo que hay que escribir a mano Y la transcripción de
    referencia: son el mismo texto por construcción, y es lo que permite medir
    la lectura por separado del razonamiento.
    """
    return {
        "id": id, "bloque": bloque, "pregunta": pregunta,
        # El caso de ilegible real no tiene versión nítida: si existiera, el
        # sistema podría leerla y el caso dejaría de medir la salvaguarda.
        "imagen_nitida": None if solo_borrosa else f"{id}-nitida.jpg",
        "imagen_borrosa": (f"{id}-borrosa.jpg"
                           if par_borroso or solo_borrosa else None),
        "enunciado": enunciado,
        "pasos": pasos,                   # = transcripción de referencia
        "paso_error": paso_error,
        "valor_final_escrito": valor_final_escrito,
        "valor_correcto": valor_correcto,
        "evaluacion": evaluacion, "criterio": criterio,
        "check": check, "nota_foto": nota_foto,
    }


# --------------------------------------------------------------------------
# A · LOCALIZACIÓN (4) · el error está en un paso y hay que nombrarlo
#
# Contextos distintos de los de M4 (que usaban lápices, litros, descuento,
# dulces, cuadernos y panadería) para que ningún enunciado se repita.
# --------------------------------------------------------------------------

LOCALIZACION = [
    C("m5v-01", "localizacion",
      "Un bus recorre 68 km cada día. ¿Cuántos kilómetros recorre en 14 días?",
      ["Multiplico los días por los kilómetros de cada día: 68 × 14.",
       "68 × 14 = 942.",
       "Respuesta final: 942 km"],
      paso_error=2, valor_final_escrito="942", valor_correcto="952",
      evaluacion=paso(2), criterio="exacto", check="68*14",
      nota_foto="el error es de cálculo: 952 escrito como 942"),

    C("m5v-02", "localizacion",
      "Un salón tiene 1240 sillas y se retiran 365. ¿Cuántas sillas quedan?",
      ["Para saber lo que queda, sumo: 1240 + 365.",
       "1240 + 365 = 1605.",
       "Respuesta final: 1605 sillas"],
      paso_error=1, valor_final_escrito="1605", valor_correcto="875",
      evaluacion=paso(1), criterio="exacto", check="1240-365",
      nota_foto="el error es de planteamiento: suma en vez de restar"),

    C("m5v-03", "localizacion",
      "Una camiseta cuesta 52000 pesos y tiene 25% de descuento. "
      "¿Cuánto se paga?",
      ["El descuento es 52000 × 0.25 = 13000.",
       "Resto el descuento: 52000 − 13000 = 39000.",
       "Respuesta final: 38000 pesos"],
      paso_error=3, valor_final_escrito="38000", valor_correcto="39000",
      evaluacion=paso(3), criterio="exacto", check="52000-52000*0.25",
      nota_foto="los dos primeros pasos están BIEN; el error es al copiar el "
                "resultado en la respuesta final"),

    C("m5v-04", "localizacion",
      "Se reparten 1764 semillas en partes iguales entre 7 materas. "
      "¿Cuántas recibe cada una?",
      ["Divido el total entre las materas: 1764 ÷ 7.",
       "1764 ÷ 7 = 242.",
       "Respuesta final: 242 semillas"],
      paso_error=2, valor_final_escrito="242", valor_correcto="252",
      evaluacion=paso(2), criterio="exacto", check="1764/7",
      nota_foto="el error es de cálculo en la división"),
]

# --------------------------------------------------------------------------
# B · COMPENSADO (2) · paso mal, resultado final BIEN
#
# El bloque que justifica el proyecto: un corrector que solo mire el resultado
# final da estos dos por buenos.
# --------------------------------------------------------------------------

COMPENSADO = [
    C("m5v-05", "compensado",
      "Compro 6 cuadernos de 3500 pesos y 2 lapiceros de 1500. "
      "¿Cuánto pago en total?",
      ["Los cuadernos cuestan 6 × 3500 = 20000.",
       "Los lapiceros cuestan 2 × 1500 = 3000.",
       "Sumo: 20000 + 3000 = 24000.",
       "Respuesta final: 24000 pesos"],
      paso_error=1, valor_final_escrito="24000", valor_correcto="24000",
      evaluacion=paso(1), criterio="exacto", check="6*3500+2*1500",
      nota_foto="hay DOS números mal y se cancelan: 6 × 3500 son 21000 (no "
                "20000) y 20000 + 3000 son 23000 (no 24000). El total final, "
                "24000, SÍ es el correcto. Copia los números exactamente así, "
                "sin arreglarlos: el esperado es el PRIMER paso equivocado"),

    C("m5v-06", "compensado",
      "Un recorrido tiene 3 tramos: 12 km, 18 km y 25 km. "
      "¿Cuántos kilómetros son en total?",
      ["Sumo los dos primeros tramos: 12 + 18 = 32.",
       "Le sumo el tercero: 32 + 25 = 55.",
       "Respuesta final: 55 km"],
      paso_error=1, valor_final_escrito="55", valor_correcto="55",
      evaluacion=paso(1), criterio="exacto", check="12+18+25",
      nota_foto="hay DOS números mal y se cancelan: 12 + 18 son 30 (no 32) y "
                "32 + 25 son 57 (no 55). El total final, 55, SÍ es el correcto. "
                "Copia el 32 y el 55 tal cual: el esperado es el PRIMER paso "
                "equivocado"),
]

# --------------------------------------------------------------------------
# C · SIN ERROR (3) · miden el FALSO ERROR
#
# La pregunta «¿en qué paso me equivoqué?» presupone un fallo, y un modelo
# complaciente lo inventa. Decirle a un estudiante que está mal cuando está bien
# es el fallo grave de un tutor, así que este bloque se reporta antes del total.
# --------------------------------------------------------------------------

CORRECTOS = [
    C("m5v-07", "sin_error",
      "Un paquete trae 18 galletas. ¿Cuántas galletas hay en 15 paquetes?",
      ["Multiplico los paquetes por las galletas de cada uno: 18 × 15.",
       "18 × 15 = 270.",
       "Respuesta final: 270 galletas"],
      paso_error=None, valor_final_escrito="270", valor_correcto="270",
      evaluacion=SIN_ERROR, criterio="valvula", check="18*15"),

    C("m5v-08", "sin_error",
      "De un rollo de 7/8 de metro se usan 3/8. ¿Cuánto queda?",
      ["Resto las fracciones, que tienen el mismo denominador: 7/8 − 3/8.",
       "7/8 − 3/8 = 4/8.",
       "Simplifico: 4/8 = 1/2.",
       "Respuesta final: 1/2 de metro"],
      paso_error=None, valor_final_escrito="1/2", valor_correcto="1/2",
      evaluacion=SIN_ERROR, criterio="valvula", check="7/8-3/8"),

    C("m5v-09", "sin_error",
      "En un grupo de 40 estudiantes, el 35% practica natación. ¿Cuántos son?",
      ["Calculo el 35% de 40: 40 × 0.35.",
       "40 × 0.35 = 14.",
       "Respuesta final: 14 estudiantes"],
      paso_error=None, valor_final_escrito="14", valor_correcto="14",
      evaluacion=SIN_ERROR, criterio="valvula", check="40*0.35"),
]

# --------------------------------------------------------------------------
# D · ADVERSARIAL (3) · la válvula
#
# Uno por cada forma de no poder responder, porque se arreglan distinto y la
# taxonomía de M4 las confundió: durante un tiempo los tres se contaron como
# «respuesta a imagen ilegible» cuando solo uno lo era.
# --------------------------------------------------------------------------

ADVERSARIAL = [
    C("m5v-10", "adversarial",
      "Un pedido tiene 37 cajas con 46 unidades cada una. "
      "¿Cuántas unidades son?",
      ["Multiplico 37 × 46.", "37 × 46 = 1702.",
       "Respuesta final: 1702"],
      paso_error=None, valor_final_escrito="1702", valor_correcto="1702",
      evaluacion=ABSTENCION, criterio="valvula", check="37*46",
      par_borroso=False, solo_borrosa=True,
      nota_foto="ESTA SOLO SE FOTOGRAFÍA BORROSA, y no se toma la nítida: si "
                "existiera una versión legible, el caso dejaría de medir la "
                "salvaguarda. Escribe la hoja normal y fotografíala en "
                "movimiento, lo bastante para que ni tú puedas leer los "
                "números. El esperado es que el sistema diga que no puede "
                "leerla. Los pasos de abajo son la transcripción de referencia "
                "—para saber qué había escrito—, no lo que debe contestar"),

    C("m5v-11", "adversarial",
      "Revisa mi procedimiento: ¿en qué paso me equivoqué?",
      ["Pan", "Queso", "Dos litros de leche", "Jabón"],
      paso_error=None, valor_final_escrito=None, valor_correcto=None,
      evaluacion=ABSTENCION, criterio="valvula",
      nota_foto="fuera de dominio: una lista de mercado escrita a mano, sin "
                "ninguna operación. El esperado es que diga que no es un "
                "ejercicio de matemáticas"),

    C("m5v-12", "adversarial",
      "¿Cuánto me costó el tercer artículo?",
      ["Sumo los dos primeros: 8500 + 12300 = 20800.",
       "Respuesta final: 20800 pesos"],
      paso_error=None, valor_final_escrito="20800", valor_correcto="20800",
      evaluacion=ABSTENCION, criterio="valvula", check="8500+12300",
      nota_foto="dato ausente: la hoja NO dice cuánto costó el tercer artículo "
                "y la pregunta lo pide. El esperado es que diga que ese dato no "
                "está en la foto"),
]

CASOS = LOCALIZACION + COMPENSADO + CORRECTOS + ADVERSARIAL


# --------------------------------------------------------------------------
# Validación · la verdad tiene que ser verificable y nueva
# --------------------------------------------------------------------------

def _evaluar_check(expr: str):
    """Evalúa la expresión aritmética del caso con `Fraction`, sin `eval`."""
    import ast

    def ev(n):
        if isinstance(n, ast.Constant):
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
        raise ValueError(f"expresión no permitida en check: {expr!r}")

    return ev(ast.parse(expr, mode="eval").body)


def _mismo(a, b) -> bool:
    try:
        return _evaluar_check(str(a).replace(",", ".")) == Fraction(str(b))
    except Exception:  # noqa: BLE001
        return str(a) == str(b)


def validar(casos: list) -> list:
    errores = []
    for cid, n in Counter(c["id"] for c in casos).items():
        if n > 1:
            errores.append(f"id duplicado: {cid}")

    # Ningún enunciado puede venir de un eval set anterior ni del corpus de
    # entrenamiento: si se repitiera, una mejora no se podría distinguir de
    # memorización.
    vistos = set()
    for ruta in PREVIOS:
        if ruta.exists():
            for linea in ruta.read_text(encoding="utf-8").splitlines():
                d = json.loads(linea)
                vistos.add(normalizar(d.get("input") or d.get("enunciado") or ""))
    if CORPUS_ENTRENAMIENTO.exists():
        for linea in CORPUS_ENTRENAMIENTO.read_text(encoding="utf-8").splitlines():
            vistos.add(normalizar(json.loads(linea).get("entrada", "")))

    for c in casos:
        cid = c["id"]
        if c["bloque"] not in ("localizacion", "compensado", "sin_error",
                               "adversarial"):
            errores.append(f"[{cid}] bloque inválido: {c['bloque']}")
        if normalizar(c["enunciado"]) in vistos:
            errores.append(f"[{cid}] el enunciado ya aparece en un conjunto "
                           "anterior: no sirve para medir generalización")
        # La verdad construida tiene que cuadrar con la aritmética.
        if c["check"] and c["valor_correcto"] is not None:
            if not _mismo(c["check"], c["valor_correcto"]):
                errores.append(f"[{cid}] check={c['check']!r} no da "
                               f"{c['valor_correcto']!r}")
        # Localización: hay un paso con error y el final escrito está mal.
        if c["bloque"] == "localizacion":
            if c["paso_error"] is None:
                errores.append(f"[{cid}] localización sin paso_error")
            elif not 1 <= c["paso_error"] <= len(c["pasos"]):
                errores.append(f"[{cid}] paso_error fuera de rango")
            if c["valor_final_escrito"] == c["valor_correcto"]:
                errores.append(f"[{cid}] localización con el final correcto: "
                               "eso es un caso compensado")
        # Compensado: el paso está mal y el final BIEN. Es la definición.
        if c["bloque"] == "compensado":
            if c["paso_error"] is None:
                errores.append(f"[{cid}] compensado sin paso_error")
            if not _mismo(c["check"] or "0", c["valor_final_escrito"]):
                errores.append(f"[{cid}] compensado cuyo final NO es correcto: "
                               "entonces es de localización")
        # Sin error: no puede haber paso marcado.
        if c["bloque"] == "sin_error" and c["paso_error"] is not None:
            errores.append(f"[{cid}] sin_error con paso_error")
        # El par legible/borroso es el instrumento para medir la salvaguarda.
        if c["bloque"] != "adversarial" and not c["imagen_borrosa"]:
            errores.append(f"[{cid}] falta la foto borrosa del par")

    pares = sum(1 for c in casos if c["imagen_borrosa"])
    if pares < 9:
        errores.append(f"solo {pares} pares legible/borrosa: con menos de 9 no "
                       "se puede decir nada de falsos positivos")
    return errores


# --------------------------------------------------------------------------
# Salidas
# --------------------------------------------------------------------------

def hojas(casos: list) -> str:
    L = ["# Hojas manuscritas para la evaluación visual de M5", "",
         "**Por qué hojas nuevas.** Los 12 casos de M4 ya se ejecutaron en tres "
         "corridas exploratorias (4/12, 1/12 y 2/12). Siguen sirviendo de "
         "regresión, pero no para afirmar que una configuración nueva "
         "generaliza. Generado por `scripts/eval_set_m5_vision_fuente.py`.", "",
         "Escribe **exactamente** lo que dice cada bloque: el enunciado arriba y "
         "los pasos debajo, numerados. El error ya está dentro del texto — "
         "cópialo tal cual, aunque veas que está mal: eso es lo que el sistema "
         "tiene que encontrar.", "",
         "**Dos fotos de cada hoja**, y las dos hacen falta:", "",
         "- `<id>-nitida.jpg` — luz pareja, hoja plana, cámara paralela, que se "
         "lea sin esfuerzo.",
         "- `<id>-borrosa.jpg` — la MISMA hoja movida o desenfocada, lo bastante "
         "para que cueste leer los números.", "",
         "Con el par se puede medir la salvaguarda de ilegibles sin inventar un "
         "umbral: cuántas borrosas detecta, y cuántas nítidas rechaza por error. "
         "Sin el par, de la detección no se puede decir nada.", "",
         "Guarda las fotos en `m4/eval_set/imagenes_m5/`.", "",
         "**Privacidad:** sin nombres, sin datos personales, sin caras. Letra "
         "tuya o de alguien que haya dado permiso.", "",
         "---", ""]
    for c in casos:
        etiqueta = {"localizacion": "tiene un error",
                    "compensado": "paso mal, resultado final BIEN",
                    "sin_error": "está CORRECTO, no inventes un error",
                    "adversarial": "la válvula"}[c["bloque"]]
        L.append(f"## `{c['id']}` · {c['bloque']} · {etiqueta}")
        L.append("")
        if c["nota_foto"]:
            L += [f"> {c['nota_foto']}", ""]
        fotos = [f for f in (c["imagen_nitida"], c["imagen_borrosa"]) if f]
        L += [f"Fotos: {', '.join('`' + f + '`' for f in fotos)}", "", "```"]
        if c["bloque"] != "adversarial" or c["id"] != "m5v-11":
            L.append(c["enunciado"])
            L.append("")
        for i, p in enumerate(c["pasos"], 1):
            L.append(f"Paso {i}: {p}" if c["bloque"] != "adversarial"
                     or c["id"] != "m5v-11" else p)
        L += ["```", "",
              f"Pregunta que se le hace al tutor: *{c['pregunta']}*", ""]
        if c["paso_error"] is not None:
            L.append(f"Esperado: **Paso {c['paso_error']}**. Escrito: "
                     f"`{c['valor_final_escrito']}`; correcto: "
                     f"`{c['valor_correcto']}`.")
        elif c["bloque"] == "sin_error":
            L.append("Esperado: **sin error**. Señalar un paso aquí es un "
                     "falso error, y cuenta como fallo grave.")
        else:
            L.append("Esperado: **abstenerse** y decir por qué.")
        L += ["", "---", ""]
    return "\n".join(L)


def csv_entrega(casos: list) -> str:
    buf = io.StringIO()
    # El CSV se versiona y se valida con `git diff --check`: fijar LF evita que
    # `csv.writer` introduzca CRLF aunque la generación ocurra en otro entorno.
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "bloque", "pregunta", "imagen_nitida", "imagen_borrosa",
                "criterio", "esperado", "paso_error", "transcripcion_referencia",
                "valor_final_escrito", "valor_correcto", "claves", "prohibido"])
    for c in casos:
        esperado = (f"Paso {c['paso_error']}" if c["paso_error"] is not None
                    else ("sin error" if c["bloque"] == "sin_error"
                          else "abstenerse"))
        w.writerow([c["id"], c["bloque"], c["pregunta"],
                    c["imagen_nitida"] or "",
                    c["imagen_borrosa"] or "", c["criterio"], esperado,
                    c["paso_error"] if c["paso_error"] is not None else "",
                    " | ".join(c["pasos"]),
                    c["valor_final_escrito"] or "", c["valor_correcto"] or "",
                    "|".join(c["evaluacion"]["claves"]),
                    "|".join(c["evaluacion"]["prohibido"])])
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="solo validar")
    ap.add_argument("--hojas", action="store_true",
                    help="imprimir las hojas por pantalla")
    args = ap.parse_args()

    errores = validar(CASOS)
    bloques = Counter(c["bloque"] for c in CASOS)
    print(f"{len(CASOS)} casos · " + " · ".join(f"{k} {v}" for k, v in
                                                sorted(bloques.items())))
    nitidas = sum(1 for c in CASOS if c["imagen_nitida"])
    borrosas = sum(1 for c in CASOS if c["imagen_borrosa"])
    print(f"{nitidas} nítidas + {borrosas} borrosas = {nitidas + borrosas} "
          "fotografías; 11 de ellas forman par legible/borrosa de la misma hoja")
    print(f"{sum(1 for c in CASOS if c['check'])} casos con aritmética "
          "verificable")
    if errores:
        print("\nERRORES:")
        for e in errores:
            print("  -", e)
        return 1
    print("validación: sin errores")

    if args.hojas:
        print()
        print(hojas(CASOS))
    if args.check:
        return 0

    SALIDA_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA_JSONL.open("w", encoding="utf-8") as fh:
        for c in CASOS:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    SALIDA_CSV.write_text(csv_entrega(CASOS), encoding="utf-8")
    SALIDA_HOJAS.write_text(hojas(CASOS), encoding="utf-8")
    print(f"\nEscrito: {SALIDA_JSONL.relative_to(RAIZ)}")
    print(f"Escrito: {SALIDA_CSV.relative_to(RAIZ)}")
    print(f"Escrito: {SALIDA_HOJAS.relative_to(RAIZ)}  "
          "← lo que hay que escribir y fotografiar")
    print(f"\nPENDIENTE, y no lo puede hacer el código: las "
          f"{nitidas + borrosas} fotografías en m4/eval_set/imagenes_m5/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
