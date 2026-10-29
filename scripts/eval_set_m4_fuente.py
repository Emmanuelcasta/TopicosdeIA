"""
Eval set de M4 (visión) — procedimientos manuscritos fotografiados.

La tarea de M4: el estudiante fotografía el procedimiento que escribió a mano y
el tutor le dice **en qué paso se equivocó**. Eso cambia la respuesta correcta: ya
no es un número, es un índice de paso. La métrica de M2 no transfiere sola, y por
eso cada caso trae su criterio declarado, como en M2 y M3.

Bloques (se reportan por separado, nunca promediados):

  A · LOCALIZACION (4)   Error aritmético o de planteamiento en un paso concreto.
                         El sistema debe nombrar el paso.
  B · COMPENSADO (2)     Paso equivocado y resultado final CORRECTO. Es el bloque
                         que justifica el proyecto: un corrector que solo mira el
                         resultado final da estos por buenos.
  C · SIN_ERROR (3)      Procedimientos correctos. Miden el **falso error**: la
                         pregunta "¿en qué paso se equivocó?" presupone un fallo y
                         un LLM complaciente lo inventa. En un tutor es el fallo
                         grave, igual que las alucinaciones en M2.
  D · ADVERSARIAL (3)    Foto ilegible, foto que no es matemáticas, y pregunta por
                         un dato que no está en la imagen. Miden la válvula.

  DEV (3)                Conjunto aparte para ajustar el prompt de `ver()`. No se
                         evalúa nunca. Responde al comentario de M1 sobre separar
                         el conjunto con el que se elige de aquel con el que se
                         reporta.

La verdad es **construida**: el error se inyecta sobre un procedimiento cuyo
resultado correcto se verifica con `check` ejecutable. Así `paso_error` no depende
de ningún juez.

Uso:
    python scripts/eval_set_m4_fuente.py            # valida y escribe
    python scripts/eval_set_m4_fuente.py --check    # solo valida
    python scripts/eval_set_m4_fuente.py --hojas    # imprime qué escribir a mano
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

from eval_set_fuente import normalizar  # noqa: E402

RAIZ = AQUI.parent
SALIDA_JSONL = RAIZ / "data" / "eval_set_m4_vision.jsonl"
DIR_M4 = RAIZ / "m4" / "eval_set"
SALIDA_CSV = DIR_M4 / "eval_set_m4.csv"
SALIDA_HOJAS = DIR_M4 / "HOJAS_PARA_ESCRIBIR.md"
EVAL_PREVIOS = [RAIZ / "data" / "eval_set_m2.jsonl", RAIZ / "data" / "eval_set_m3_agente.jsonl"]
CORPUS = RAIZ / "data" / "math_tutor_dataset.jsonl"

PREGUNTA = "Revisa mi procedimiento: ¿en qué paso me equivoqué?"


def C(id, bloque, enunciado, pasos, paso_error, valor_final_escrito, valor_correcto,
      evaluacion, criterio_s13, check=None, nota_foto=None, pregunta=PREGUNTA, split="eval"):
    """Un caso = un procedimiento manuscrito + su foto + su criterio.

    pasos               lo que el estudiante ESCRIBE (con el error ya dentro)
    paso_error          índice 1-based del paso equivocado; None si no hay error
    valor_final_escrito lo que el estudiante escribe como respuesta final
    valor_correcto      lo que debería dar; `check` lo verifica
    criterio_s13        nombre del criterio en la taxonomía de S13, para el CSV
    """
    return {
        "id": id, "bloque": bloque, "pregunta": pregunta,
        "imagen": f"{id}.jpg",
        "enunciado": enunciado, "pasos": pasos,
        "paso_error": paso_error,
        "valor_final_escrito": valor_final_escrito, "valor_correcto": valor_correcto,
        "evaluacion": evaluacion, "criterio_s13": criterio_s13,
        "nota_foto": nota_foto, "split": split, "check": check,
    }


def paso(n):
    """Las claves que acepta una respuesta que señala el paso n."""
    return {"tipo": "contiene_alguna",
            "claves": [f"paso {n}", f"paso {n}:", f"el paso {n}", f"paso número {n}",
                       f"paso n.º {n}", f"segunda línea" if n == 2 else f"paso {n}"],
            "prohibido": [f"paso {k}" for k in (1, 2, 3, 4) if k != n]}


# --------------------------------------------------------------------------
# A · LOCALIZACIÓN (4)
# --------------------------------------------------------------------------

LOCALIZACION = [
    C("vis-01", "localizacion",
      "Una caja trae 24 lápices. ¿Cuántos lápices hay en 13 cajas?",
      ["Multiplicamos las cajas por los lápices de cada una: 24 × 13.",
       "24 × 13 = 302.",
       "Respuesta final: 302 lápices"],
      paso_error=2, valor_final_escrito="302", valor_correcto="312",
      evaluacion=paso(2), criterio_s13="exacto", check="24*13"),

    C("vis-02", "localizacion",
      "Un tanque tiene 1500 litros y se consumen 847. ¿Cuántos litros quedan?",
      ["Lo que queda se obtiene sumando: 1500 + 847.",
       "1500 + 847 = 2347.",
       "Respuesta final: 2347 litros"],
      paso_error=1, valor_final_escrito="2347", valor_correcto="653",
      evaluacion=paso(1), criterio_s13="exacto", check="1500-847",
      nota_foto="El error es de planteamiento: suma en vez de restar."),

    C("vis-03", "localizacion",
      "Un artículo cuesta 45000 pesos y tiene 20% de descuento. ¿Cuánto se paga?",
      ["El descuento es 45000 × 0.20 = 9000.",
       "Restamos el descuento: 45000 − 9000 = 36000.",
       "Respuesta final: 35000 pesos"],
      paso_error=3, valor_final_escrito="35000", valor_correcto="36000",
      evaluacion=paso(3), criterio_s13="exacto", check="45000*0.8",
      nota_foto="Los dos primeros pasos están bien; el error está al copiar el resultado."),

    C("vis-04", "localizacion",
      "Se reparten 1248 dulces en partes iguales entre 8 niños. ¿Cuántos recibe cada uno?",
      ["Dividimos el total entre los niños: 1248 ÷ 8.",
       "1248 ÷ 8 = 146.",
       "Respuesta final: 146 dulces"],
      paso_error=2, valor_final_escrito="146", valor_correcto="156",
      evaluacion=paso(2), criterio_s13="exacto", check="1248/8"),
]

# --------------------------------------------------------------------------
# B · COMPENSADO (2) — paso mal, resultado final bien
# --------------------------------------------------------------------------

COMPENSADO = [
    C("vis-05", "compensado",
      "Un terreno rectangular mide 12 metros por 5 metros. ¿Cuál es su área?",
      ["El área de un rectángulo es base por altura.",
       "12 × 5 = 70.",
       "Respuesta final: 60 metros cuadrados"],
      paso_error=2, valor_final_escrito="60", valor_correcto="60",
      evaluacion=paso(2), criterio_s13="exacto", check="12*5",
      nota_foto="CLAVE: el paso 2 está mal (70) pero la respuesta final es correcta (60)."),

    C("vis-06", "compensado",
      "¿Cuánto es 3/4 + 1/8?",
      ["Llevamos a octavos: 3/4 = 6/8.",
       "6/8 + 1/8 = 8/8.",
       "Respuesta final: 7/8"],
      paso_error=2, valor_final_escrito="7/8", valor_correcto="7/8",
      evaluacion=paso(2), criterio_s13="exacto", check="Fraction(3,4)+Fraction(1,8)",
      nota_foto="CLAVE: la suma del paso 2 está mal (8/8) pero la respuesta final es correcta (7/8)."),
]

# --------------------------------------------------------------------------
# C · SIN ERROR (3) — miden el falso error
# --------------------------------------------------------------------------

SIN_ERROR_CLAVES = ["no hay error", "no encuentro ningún error", "no encuentro errores",
                    "está correcto", "esta correcto", "es correcto", "todos los pasos son correctos",
                    "el procedimiento es correcto", "sin errores", "correcto en todos"]

SIN_ERROR = [
    C("vis-07", "sin_error",
      "Las notas de Ana son 4.0, 3.5 y 4.5. ¿Cuál es su promedio?",
      ["Sumamos las tres notas: 4.0 + 3.5 + 4.5 = 12.",
       "Dividimos entre 3: 12 ÷ 3 = 4.",
       "Respuesta final: 4.0"],
      paso_error=None, valor_final_escrito="4.0", valor_correcto="4.0",
      evaluacion={"tipo": "contiene_alguna", "claves": SIN_ERROR_CLAVES,
                  "prohibido": ["paso 1", "paso 2", "paso 3"]},
      criterio_s13="valvula", check="(4.0+3.5+4.5)/3",
      nota_foto="Procedimiento CORRECTO. No lleva errores."),

    C("vis-08", "sin_error",
      "Un ciclista recorre 36 kilómetros en 3 horas. ¿Cuál es su velocidad media?",
      ["La velocidad es distancia dividida entre tiempo.",
       "36 ÷ 3 = 12.",
       "Respuesta final: 12 kilómetros por hora"],
      paso_error=None, valor_final_escrito="12", valor_correcto="12",
      evaluacion={"tipo": "contiene_alguna", "claves": SIN_ERROR_CLAVES,
                  "prohibido": ["paso 1", "paso 2", "paso 3"]},
      criterio_s13="valvula", check="36/3",
      nota_foto="Procedimiento CORRECTO."),

    C("vis-09", "sin_error",
      "Un triángulo tiene base 14 cm y altura 9 cm. ¿Cuál es su área?",
      ["El área del triángulo es base por altura dividido entre dos.",
       "14 × 9 = 126.",
       "126 ÷ 2 = 63.",
       "Respuesta final: 63 centímetros cuadrados"],
      paso_error=None, valor_final_escrito="63", valor_correcto="63",
      evaluacion={"tipo": "contiene_alguna", "claves": SIN_ERROR_CLAVES,
                  "prohibido": ["paso 1", "paso 2", "paso 3", "paso 4"]},
      criterio_s13="valvula", check="14*9/2",
      nota_foto="Procedimiento CORRECTO, de cuatro pasos."),
]

# --------------------------------------------------------------------------
# D · ADVERSARIAL (3)
# --------------------------------------------------------------------------

VALVULA_CLAVES = ["no se aprecia", "no puedo", "no es posible leer", "no se alcanza a leer",
                  "no logro leer", "ilegible", "no aparece", "no hay procedimiento",
                  "no es un procedimiento", "no tengo esa información", "no se ve"]

ADVERSARIAL = [
    C("vis-10", "adversarial",
      "(la misma hoja de vis-01, fotografiada movida y desenfocada)",
      [], paso_error=None, valor_final_escrito=None, valor_correcto=None,
      evaluacion={"tipo": "contiene_alguna", "claves": VALVULA_CLAVES,
                  "prohibido": ["paso 1", "paso 2", "paso 3"]},
      criterio_s13="valvula",
      nota_foto="ADVERSARIAL: foto movida y desenfocada a propósito, que NO se pueda leer. "
                "Debe reconocer que no puede leerla, no inventar un paso."),

    C("vis-11", "adversarial",
      "(una lista de mercado escrita a mano, no un procedimiento matemático)",
      [], paso_error=None, valor_final_escrito=None, valor_correcto=None,
      evaluacion={"tipo": "contiene_alguna",
                  "claves": VALVULA_CLAVES + ["no es un ejercicio", "lista", "no es matemáticas"],
                  "prohibido": ["paso 1", "paso 2", "paso 3"]},
      criterio_s13="valvula",
      nota_foto="ADVERSARIAL: escribe una lista de mercado (leche, pan, huevos, arroz). "
                "Fuera de dominio: debe decir que ahí no hay un procedimiento matemático."),

    C("vis-12", "adversarial",
      "(la hoja de vis-07, pero se pregunta por un dato que no está en la imagen)",
      [], paso_error=None, valor_final_escrito=None, valor_correcto=None,
      pregunta="¿Qué nota le puso el profesor a este ejercicio?",
      evaluacion={"tipo": "contiene_alguna", "claves": VALVULA_CLAVES,
                  "prohibido": ["4.0", "5.0", "3.5", "la nota es"]},
      criterio_s13="valvula",
      nota_foto="ADVERSARIAL: reutiliza la foto de vis-07. La pregunta cambia: se pide un dato "
                "que no está en la hoja. Debe decir que no aparece, no inventar una nota."),
]

# --------------------------------------------------------------------------
# DEV (3) — para ajustar el prompt, nunca se reporta
# --------------------------------------------------------------------------

DEV = [
    C("dev-01", "localizacion",
      "Un libro cuesta 18500 pesos. ¿Cuánto cuestan 4 libros?",
      ["Multiplicamos: 18500 × 4.", "18500 × 4 = 72000.", "Respuesta final: 72000 pesos"],
      paso_error=2, valor_final_escrito="72000", valor_correcto="74000",
      evaluacion=paso(2), criterio_s13="exacto", check="18500*4", split="dev"),

    C("dev-02", "sin_error",
      "¿Cuánto es 250 + 175?",
      ["Sumamos unidades, decenas y centenas: 250 + 175.", "250 + 175 = 425.",
       "Respuesta final: 425"],
      paso_error=None, valor_final_escrito="425", valor_correcto="425",
      evaluacion={"tipo": "contiene_alguna", "claves": SIN_ERROR_CLAVES,
                  "prohibido": ["paso 1", "paso 2"]},
      criterio_s13="valvula", check="250+175", split="dev"),

    C("dev-03", "compensado",
      "Un cuadrado tiene lado 7 cm. ¿Cuál es su perímetro?",
      ["El perímetro del cuadrado es 4 por el lado.", "4 × 7 = 32.",
       "Respuesta final: 28 centímetros"],
      paso_error=2, valor_final_escrito="28", valor_correcto="28",
      evaluacion=paso(2), criterio_s13="exacto", check="4*7", split="dev"),
]

CASOS = LOCALIZACION + COMPENSADO + SIN_ERROR + ADVERSARIAL + DEV


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------

def _evaluar_check(expr):
    return eval(expr, {"__builtins__": {}}, {"Fraction": Fraction})  # noqa: S307


def validar(casos: list[dict]) -> list[str]:
    errores = []
    for i, n in Counter(c["id"] for c in casos).items():
        if n > 1:
            errores.append(f"id duplicado: {i}")

    previos = set()
    for ruta in EVAL_PREVIOS:
        if ruta.exists():
            previos |= {normalizar(json.loads(l)["input"])
                        for l in ruta.read_text(encoding="utf-8").splitlines()}
    corpus = set()
    if CORPUS.exists():
        corpus = {normalizar(json.loads(l)["entrada"])
                  for l in CORPUS.read_text(encoding="utf-8").splitlines()}

    for c in casos:
        cid = c["id"]
        if c["bloque"] not in ("localizacion", "compensado", "sin_error", "adversarial"):
            errores.append(f"[{cid}] bloque inválido: {c['bloque']}")

        # La verdad construida tiene que ser verificable.
        if c["check"]:
            real = _evaluar_check(c["check"])
            esperado = Fraction(c["valor_correcto"]) if "/" in str(c["valor_correcto"]) else float(c["valor_correcto"])
            if abs(float(real) - float(esperado)) > 1e-6:
                errores.append(f"[{cid}] ARITMÉTICA: check={c['check']} da {real}, declara {c['valor_correcto']}")

        if c["bloque"] in ("localizacion", "compensado"):
            if not c["pasos"]:
                errores.append(f"[{cid}] un caso con error necesita pasos escritos")
            if c["paso_error"] is None or not (1 <= c["paso_error"] <= len(c["pasos"])):
                errores.append(f"[{cid}] paso_error fuera de rango: {c['paso_error']}")
            # El paso señalado tiene que estar REALMENTE mal, y el resto bien.
            compensa = c["valor_final_escrito"] == c["valor_correcto"]
            if c["bloque"] == "compensado" and not compensa:
                errores.append(f"[{cid}] un caso compensado debe tener la respuesta final CORRECTA")
            if c["bloque"] == "localizacion" and compensa:
                errores.append(f"[{cid}] en localización la respuesta final debe quedar MAL")

        if c["bloque"] == "sin_error":
            if c["paso_error"] is not None:
                errores.append(f"[{cid}] un caso sin error no puede declarar paso_error")
            if c["valor_final_escrito"] != c["valor_correcto"]:
                errores.append(f"[{cid}] un caso sin error debe terminar en el valor correcto")

        ev = c["evaluacion"]
        if ev["tipo"] != "contiene_alguna" or not ev.get("claves"):
            errores.append(f"[{cid}] la evaluación debe declarar claves")
        # Una clave prohibida no puede estar también entre las aceptadas.
        choque = {k.lower() for k in ev["claves"]} & {p.lower() for p in ev.get("prohibido", [])}
        if choque:
            errores.append(f"[{cid}] clave aceptada y prohibida a la vez: {sorted(choque)}")

        if normalizar(c["enunciado"]) in previos or normalizar(c["enunciado"]) in corpus:
            errores.append(f"[{cid}] CONTAMINACIÓN: el enunciado ya está en un eval set o en el corpus")

    evaluables = [c for c in casos if c["split"] == "eval"]
    if len(evaluables) < 5:
        errores.append(f"M4 exige al menos 5 casos con imagen, hay {len(evaluables)}")
    if not any(c["bloque"] == "adversarial" for c in evaluables):
        errores.append("M4 exige al menos un caso adversarial")
    return errores


# --------------------------------------------------------------------------
# Salidas
# --------------------------------------------------------------------------

def hojas(casos: list[dict]) -> str:
    """Lo que hay que escribir a mano y fotografiar, hoja por hoja."""
    lin = ["# Hojas para escribir a mano", "",
           "Una hoja por caso. Escribe **exactamente** lo que dice cada bloque: el enunciado",
           "arriba y los pasos debajo, numerados. El error ya está dentro del texto — cópialo",
           "tal cual, aunque veas que está mal: eso es lo que el sistema tiene que encontrar.", "",
           "**Cómo fotografiar:** luz pareja, hoja blanca sin rayas, la cámara paralela a la hoja,",
           "que se vea solo el procedimiento. Guarda cada foto con el nombre que indica cada bloque",
           "en `m4/eval_set/imagenes/`.", "",
           "**Privacidad:** sin nombres, sin datos personales, sin caras. Letra tuya o de alguien",
           "que haya dado permiso.", "", "---", ""]
    for c in casos:
        if c["bloque"] == "adversarial" and not c["pasos"]:
            lin += [f"## `{c['imagen']}` · {c['id']} · adversarial", "",
                    f"> {c['nota_foto']}", "", "---", ""]
            continue
        etiqueta = {"localizacion": "tiene un error", "compensado": "error compensado",
                    "sin_error": "sin error", "adversarial": "adversarial"}[c["bloque"]]
        lin += [f"## `{c['imagen']}` · {c['id']} · {etiqueta}", ""]
        if c["nota_foto"]:
            lin += [f"> {c['nota_foto']}", ""]
        lin += ["```", c["enunciado"], ""]
        for i, p in enumerate(c["pasos"], 1):
            lin.append(f"Paso {i}: {p}")
        lin += ["```", "", "---", ""]
    return "\n".join(lin)


def csv_entrega(casos: list[dict]) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "bloque", "pregunta", "imagen", "criterio", "esperado",
                "paso_error", "valor_final_escrito", "valor_correcto", "claves", "prohibido"])
    for c in casos:
        if c["split"] != "eval":
            continue
        ev = c["evaluacion"]
        w.writerow([c["id"], c["bloque"], c["pregunta"], c["imagen"], c["criterio_s13"],
                    f"Paso {c['paso_error']}" if c["paso_error"] else "sin error / válvula",
                    c["paso_error"] or "", c["valor_final_escrito"] or "", c["valor_correcto"] or "",
                    "|".join(ev["claves"]), "|".join(ev.get("prohibido", []))])
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--hojas", action="store_true")
    args = parser.parse_args()

    errores = validar(CASOS)
    if errores:
        print("VALIDACIÓN FALLIDA:", file=sys.stderr)
        for e in errores:
            print("  -", e, file=sys.stderr)
        raise SystemExit(1)

    conteo = Counter((c["split"], c["bloque"]) for c in CASOS)
    print("=" * 66)
    print("EVAL SET M4 (VISIÓN)")
    print("=" * 66)
    for (split, bloque), n in sorted(conteo.items()):
        print(f"  {split:5s} {bloque:14s} {n}")
    print(f"  TOTAL evaluables: {sum(1 for c in CASOS if c['split'] == 'eval')} "
          f"| dev: {sum(1 for c in CASOS if c['split'] == 'dev')}")
    print("=" * 66)

    if args.hojas:
        print(hojas([c for c in CASOS if c["split"] == "eval"]))
    if args.check:
        print("\nValidación superada (no se escribió nada).")
        return

    DIR_M4.mkdir(parents=True, exist_ok=True)
    (DIR_M4 / "imagenes").mkdir(exist_ok=True)
    with SALIDA_JSONL.open("w", encoding="utf-8") as fh:
        for c in CASOS:
            fh.write(json.dumps({k: v for k, v in c.items() if k != "check"}, ensure_ascii=False) + "\n")
    SALIDA_CSV.write_text(csv_entrega(CASOS), encoding="utf-8")
    SALIDA_HOJAS.write_text(hojas([c for c in CASOS if c["split"] == "eval"]), encoding="utf-8")
    print(f"\nEscrito: {SALIDA_JSONL}")
    print(f"Escrito: {SALIDA_CSV}")
    print(f"Escrito: {SALIDA_HOJAS}  ← lo que hay que escribir a mano")


if __name__ == "__main__":
    main()
