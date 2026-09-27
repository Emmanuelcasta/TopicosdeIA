"""
Trayectorias de uso de herramientas: el material para ENSEÑAR al agente
cuándo y cómo usar cada herramienta (few-shot en C3–C5, fine-tuning en C5-FT).

Cada trayectoria es una conversación en el formato nativo de tool calling de
Qwen2.5:

  usuario: problema
  asistente: Pensamiento (decisión y por qué) + <tool_call>
  herramienta: <tool_response> (resultado REAL, ejecutado aquí)
  ... (tantos pasos como haga falta)
  asistente: explicación paso a paso + "Respuesta final: ..."

Dos orígenes:

  CORPUS (165)  Derivadas automáticamente de `registros.py`. Cada paso con una
                operación escrita ("420 × 0.25 = 105") se convierte en la llamada
                a la herramienta adecuada; la llamada SE EJECUTA y solo se conserva
                si su resultado coincide con el del corpus. La respuesta final es
                la salida del corpus, que ya estaba verificada por su `check`.
                Mantienen la partición del corpus (train/validation).

  MANUALES (34) Lo que el corpus no enseña: consultar documentos, casos
                compuestos (buscar + calcular), NO usar herramientas (conceptual,
                fuera de dominio, datos insuficientes), errores de dominio que hay
                que explicar, álgebra y preguntas mal escritas.

Validaciones: toda llamada se ejecuta; el resultado esperado (ok o error de
dominio) debe cumplirse; el valor final debe coincidir con la última
observación numérica; y ninguna pregunta puede coincidir o parecerse demasiado
a un caso de los eval sets de M2 y M3 (no contaminación).

Uso:
    python trayectorias_fuente.py            # valida y escribe data/trayectorias_herramientas.jsonl
    python trayectorias_fuente.py --check
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI / "agente"))

import registros as corpus_registros  # noqa: E402
from documentos import formatear_pasajes  # noqa: E402
from eval_set_fuente import normalizar  # noqa: E402
from herramientas import REGISTRO  # noqa: E402
from nucleo import mismo_numero, valor_final  # noqa: E402
from registro import LlamadaHerramienta, ResultadoHerramienta  # noqa: E402

RAIZ = AQUI.parent
SALIDA = RAIZ / "data" / "trayectorias_herramientas.jsonl"
CORPUS_JSONL = RAIZ / "data" / "math_tutor_dataset.jsonl"
EVAL_SETS = [RAIZ / "data" / "eval_set_m2.jsonl", RAIZ / "data" / "eval_set_m3_agente.jsonl"]

UMBRAL_JACCARD = 0.6


# --------------------------------------------------------------------------
# Corpus documental (el MISMO código que embeben S07/S08)
# --------------------------------------------------------------------------

def _corpus_documental():
    from nb_comun_rag import CODIGO_CHUNKING, CODIGO_CORPUS
    ns: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(CODIGO_CORPUS, ns)    # noqa: S102
        exec(CODIGO_CHUNKING, ns)  # noqa: S102
    return ns["chunks"], ns["metadatos"], ns["ids_chunks"]


CHUNKS, METADATOS, IDS_CHUNKS = _corpus_documental()


def observacion_documentos(clave: str | None, consulta: str) -> ResultadoHerramienta:
    """Resultado de `buscar_documentos` tal como lo devolvería la herramienta real:
    el chunk que contiene la clave primero, completado con chunks del mismo documento."""
    if clave:
        oro = [i for i, c in enumerate(CHUNKS) if normalizar(clave) in normalizar(c)]
        if not oro:
            raise ValueError(f"clave inexistente en el corpus: {clave!r}")
        doc = METADATOS[oro[0]]["doc_id"]
        resto = [i for i, m in enumerate(METADATOS) if m["doc_id"] == doc and i not in oro]
        idxs = (oro + resto)[:3]
    else:
        idxs = [i for i, m in enumerate(METADATOS) if m["doc_id"] == "siee"][:3]
    texto = formatear_pasajes([(METADATOS[i]["fuente"], CHUNKS[i]) for i in idxs])
    return ResultadoHerramienta("buscar_documentos", {"consulta": consulta}, True, texto, None,
                                extra={"ids": [IDS_CHUNKS[i] for i in idxs]})


# --------------------------------------------------------------------------
# Construcción de mensajes
# --------------------------------------------------------------------------

def _msg_asistente_llamada(pensamiento: str, nombre: str, argumentos: dict) -> dict:
    return {"role": "assistant", "content": f"Pensamiento: {pensamiento}",
            "tool_calls": [{"type": "function", "function": {"name": nombre, "arguments": argumentos}}]}


def _msg_herramienta(res: ResultadoHerramienta) -> dict:
    return {"role": "tool", "name": res.herramienta, "content": res.como_observacion()}


def _ejecutar(nombre: str, argumentos: dict, consulta_doc_clave=None) -> ResultadoHerramienta:
    if nombre == "buscar_documentos":
        return observacion_documentos(consulta_doc_clave, argumentos["consulta"])
    return REGISTRO.ejecutar(LlamadaHerramienta(nombre, argumentos))


# --------------------------------------------------------------------------
# 1 · Trayectorias derivadas del corpus
# --------------------------------------------------------------------------

_OP_A_HERRAMIENTA = {"+": "sumar", "-": "restar", "×": "multiplicar", "÷": "dividir"}
_NUM = r"-?\d+(?:\.\d+)?"
_FRAC = r"\d+/\d+"

PENSAMIENTOS = {
    "sumar": "necesito sumar {x}; uso la herramienta para no equivocarme en la cuenta.",
    "restar": "necesito restar {x}; la hago con la herramienta.",
    "multiplicar": "necesito multiplicar {x}; delego el producto a la herramienta.",
    "dividir": "necesito dividir {x}; uso la herramienta de división.",
    "operar_fracciones": "hay que operar las fracciones {x}; uso la herramienta de fracciones, que simplifica el resultado.",
    "potencia": "necesito calcular la potencia {x}.",
    "raiz": "necesito la raíz de {x}.",
    "resolver_ecuacion": "es una ecuación ({x}); la resuelvo con la herramienta de álgebra.",
    "evaluar_expresion": "necesito evaluar {x} respetando la jerarquía de operaciones.",
    "simplificar_fraccion": "hay que simplificar la fracción {x}; la herramienta calcula el máximo común divisor.",
}


def _llamada_para(expr: str):
    """Expresión de un paso del corpus -> (herramienta, argumentos) o None."""
    e = expr.strip().strip("().").strip()
    e = re.sub(r"\s+", " ", e)
    m = re.fullmatch(rf"\(?({_FRAC})\)?\s*([+\-×÷])\s*\(?({_FRAC})\)?", e)
    if m:
        op = {"+": "suma", "-": "resta", "×": "multiplicacion", "÷": "division"}[m.group(2)]
        return "operar_fracciones", {"fraccion_a": m.group(1), "fraccion_b": m.group(3), "operacion": op}
    m = re.fullmatch(rf"({_NUM})\s*([²³])", e)
    if m:
        return "potencia", {"base": float(m.group(1)) if "." in m.group(1) else int(m.group(1)),
                            "exponente": 2 if m.group(2) == "²" else 3}
    m = re.fullmatch(r"√\s*(\d+(?:\.\d+)?)", e)
    if m:
        return "raiz", {"radicando": int(m.group(1)) if "." not in m.group(1) else float(m.group(1))}
    partes = re.split(r"\s*([+\-×÷])\s*", e)
    if len(partes) >= 3 and all(re.fullmatch(_NUM, p) for p in partes[::2]):
        ops = set(partes[1::2])
        numeros = [float(p) if "." in p else int(p) for p in partes[::2]]
        if len(partes) == 3:
            op = partes[1]
            if op == "+":
                return "sumar", {"numeros": numeros}
            if op == "×":
                return "multiplicar", {"numeros": numeros}
            if op == "-":
                return "restar", {"minuendo": numeros[0], "sustraendo": numeros[1]}
            return "dividir", {"dividendo": numeros[0], "divisor": numeros[1]}
        if ops == {"+"}:
            return "sumar", {"numeros": numeros}
        if ops == {"×"}:
            return "multiplicar", {"numeros": numeros}
    if re.search(r"[+\-×÷²³√*/^]", e) and re.search(r"\d", e):
        return "evaluar_expresion", {"expresion": e}
    return None


_PAT_CADENA = re.compile(r"([\d(√][\d\s.,+\-×÷*/()²³√]*?)\s*=\s*(?:[\d\s.+\-×÷/()²³]*=\s*)?(-?\d+(?:\.\d+)?(?:/\d+)?)")


def llamadas_de_paso(paso: str) -> list[tuple[str, dict, str]]:
    """[(herramienta, argumentos, resultado_esperado)] de un paso del corpus."""
    salida = []
    m = re.search(r"ra[ií]z\s+(c[uú]bica|cuadrada)\s+de\s+(\d+)\s+es\s+(\d+)", paso, re.IGNORECASE)
    if m:
        indice = 3 if m.group(1).lower().startswith("c") and "b" in m.group(1).lower() else 2
        return [("raiz", {"radicando": int(m.group(2)), "indice": indice}, m.group(3))]
    for expr, resultado in _PAT_CADENA.findall(paso):
        if not re.search(r"[+\-×÷²³√]", expr) and not re.fullmatch(rf"\s*{_FRAC}\s*", expr):
            continue
        if re.fullmatch(rf"\s*{_FRAC}\s*", expr):         # "3/4 = 6/8": equivalencia, no operación
            continue
        ll = _llamada_para(expr)
        if ll:
            salida.append((*ll, resultado))
    return salida


def trayectoria_corpus(r: dict, salida_texto: str, split: str) -> dict | None:
    mensajes = [{"role": "user", "content": r["entrada"]}]
    usadas, vistos = [], set()

    candidatos = []
    if r["categoria"] == "ecuaciones":
        m = re.search(r"(\d*\s*x(?:\s*[+\-×÷]\s*\d+(?:\.\d+)?)*\s*=\s*\d+(?:\.\d+)?)", r["entrada"] + " " + " ".join(r["pasos"]))
        if m and re.search(r"[+\-×÷]|\dx", m.group(1)):
            ec = m.group(1).strip().replace("×", "*").replace("÷", "/")
            candidatos.append(("resolver_ecuacion", {"ecuacion": ec}, r["valor"]))
    if not candidatos:
        for paso in r["pasos"]:
            candidatos.extend(llamadas_de_paso(paso))

    def respaldo_check():
        """Si ningún paso dio una llamada verificable, se usa el `check` del corpus."""
        if not r.get("check"):
            return []
        if re.fullmatch(r"\d+/\d+", r["check"]):
            return [("simplificar_fraccion", {"fraccion": r["check"]}, r["valor"])]
        expr = r["check"].replace("**0.5", "^(1/2)").replace("**", "^").replace("*", " × ")
        return [("evaluar_expresion", {"expresion": expr}, r["valor"])]

    def agregar(lista):
        for nombre, args, esperado in lista:
            clave = json.dumps([nombre, args], sort_keys=True)
            if clave in vistos:
                continue
            res = _ejecutar(nombre, args)
            if not res.ok or not (mismo_numero(res.valor, esperado) or res.valor == esperado):
                continue                       # la llamada no reproduce el paso: se descarta
            vistos.add(clave)
            x = args.get("expresion") or args.get("ecuacion") or args.get("fraccion") or " y ".join(
                str(v) for v in (args.get("numeros") or [a for k, a in args.items() if k not in ("operacion", "indice")]))
            mensajes.append(_msg_asistente_llamada(PENSAMIENTOS[nombre].format(x=x), nombre, args))
            mensajes.append(_msg_herramienta(res))
            usadas.append(nombre)

    agregar(candidatos)
    if not usadas:
        agregar(respaldo_check())
    if not usadas:
        return None
    mensajes.append({"role": "assistant", "content": salida_texto})
    return {"id": f"tr-{r['id']}", "origen": f"corpus:{r['id']}", "tipo": "calculo",
            "categoria": r["categoria"], "split": split, "pregunta": r["entrada"],
            "herramientas": usadas, "valor": r["valor"], "espera_error": False, "mensajes": mensajes}


# --------------------------------------------------------------------------
# 2 · Trayectorias manuales
# --------------------------------------------------------------------------
# Cada acción: (pensamiento, herramienta, argumentos[, clave_documento]).
# `valor` = valor esperado de la respuesta final (None si no es numérica).

def M(id, tipo, pregunta, acciones, final, valor=None, espera_error=False):
    return {"id": id, "tipo": tipo, "pregunta": pregunta, "acciones": acciones, "final": final,
            "valor": valor, "espera_error": espera_error}


MANUALES = [
    # ---- documentos: dato institucional, se busca y se cita --------------------
    M("man-doc-01", "documentos", "¿Cuál es la escala de calificación que usa el colegio?",
      [("es un dato institucional; no debo inventarlo, lo busco en los documentos del colegio.",
        "buscar_documentos", {"consulta": "escala de calificación institucional"}, "escala institucional va de 1.0 a 5.0")],
      "Paso 1: Según el Sistema Institucional de Evaluación (SIEE), la escala va de 1.0 a 5.0.\nRespuesta final: de 1.0 a 5.0"),
    M("man-doc-02", "documentos", "¿Cuántos días tiene el acudiente para presentar una excusa cuando el estudiante falta a un examen?",
      [("es una norma del colegio; la busco en el reglamento.", "buscar_documentos",
        {"consulta": "plazo para presentar excusa por inasistencia a examen"}, "tres dias habiles")],
      "Paso 1: Según el SIEE, el acudiente debe presentar la excusa dentro de los tres días hábiles siguientes.\nRespuesta final: 3 días hábiles", "3"),
    M("man-doc-03", "documentos", "¿Qué temas se trabajan en grado noveno según el plan de área de matemáticas?",
      [("es información del plan de área; la busco.", "buscar_documentos",
        {"consulta": "contenidos de grado noveno plan de área"}, "noveno se amplia")],
      "Paso 1: Según el plan de área, en grado noveno se trabajan los sistemas de dos ecuaciones lineales y la función lineal.\nRespuesta final: sistemas de dos ecuaciones lineales y función lineal"),
    M("man-doc-04", "documentos", "¿Qué estrategia sugiere el protocolo del colegio cuando un estudiante usa regla de tres directa en problemas de obreros y días?",
      [("es una estrategia del protocolo del colegio; la busco.", "buscar_documentos",
        {"consulta": "protocolo proporcionalidad inversa obreros días"}, "dias-obrero")],
      "Paso 1: Según el protocolo de errores frecuentes, primero se calcula el trabajo total en días-obrero.\nRespuesta final: calcular primero el trabajo total en días-obrero"),
    M("man-doc-05", "documentos", "Según el formulario de geometría del colegio, ¿cómo se calcula el área de un trapecio?",
      [("piden la fórmula tal como está en el formulario; la busco.", "buscar_documentos",
        {"consulta": "fórmula área del trapecio"}, "trapecio")],
      "Paso 1: Según el formulario de geometría, el área del trapecio es la semisuma de las bases multiplicada por la altura.\nRespuesta final: (base mayor + base menor) ÷ 2 × altura"),
    M("man-doc-06", "documentos", "¿Cuántos estudiantes puede haber como máximo en un salón según el reglamento del colegio?",
      [("es una norma institucional; la busco antes de responder.", "buscar_documentos",
        {"consulta": "número máximo de estudiantes por salón"}, None)],
      "Busqué en los documentos del colegio y ninguno indica un número máximo de estudiantes por salón.\nNo tengo esa información en mis fuentes."),

    # ---- compuestos: buscar el dato y calcular con él ------------------------------
    M("man-com-01", "compuesto", "Según la escala de calificación del colegio, ¿cuántos puntos hay entre la nota más baja y la más alta?",
      [("necesito la escala del colegio (documento) y después una resta.", "buscar_documentos",
        {"consulta": "escala de calificación"}, "escala institucional va de 1.0 a 5.0"),
       ("la escala va de 1.0 a 5.0; resto con la herramienta.", "restar", {"minuendo": 5.0, "sustraendo": 1.0})],
      "Paso 1: Según el SIEE, la escala va de 1.0 a 5.0.\nPaso 2: 5.0 - 1.0 = 4.\nRespuesta final: 4 puntos", "4"),
    M("man-com-02", "compuesto", "Un acudiente entregó la excusa 5 días hábiles después del examen. ¿Por cuántos días se pasó del plazo del reglamento?",
      [("necesito el plazo del reglamento y luego restar.", "buscar_documentos",
        {"consulta": "plazo excusa inasistencia"}, "tres dias habiles"),
       ("el plazo es de 3 días hábiles; resto 5 - 3.", "restar", {"minuendo": 5, "sustraendo": 3})],
      "Paso 1: Según el SIEE, el plazo es de tres días hábiles.\nPaso 2: 5 - 3 = 2.\nRespuesta final: 2 días hábiles", "2"),
    M("man-com-03", "compuesto", "El plan de área recomienda un mínimo de sesiones para traducir enunciados verbales. Si cada sesión dura 55 minutos, ¿cuántos minutos son como mínimo?",
      [("necesito el número de sesiones del plan de área y luego multiplicar.", "buscar_documentos",
        {"consulta": "sesiones traducción de enunciados verbales"}, "dos sesiones"),
       ("son al menos dos sesiones; multiplico por 55.", "multiplicar", {"numeros": [2, 55]})],
      "Paso 1: Según el plan de área, se dedican al menos dos sesiones.\nPaso 2: 2 × 55 = 110.\nRespuesta final: 110 minutos", "110"),
    M("man-com-04", "compuesto", "Según el protocolo del colegio, ¿cuántos buses de 40 puestos se necesitan para llevar a 130 estudiantes?",
      [("el protocolo dice cómo redondear en estos casos; lo busco.", "buscar_documentos",
        {"consulta": "redondeo al dividir personas entre vehículos"}, "redondearse hacia arriba"),
       ("hay que redondear hacia arriba; divido 130 entre 40 con redondeo arriba.", "dividir",
        {"dividendo": 130, "divisor": 40, "redondeo": "arriba"})],
      "Paso 1: Según el protocolo, al repartir personas en vehículos se redondea hacia arriba.\nPaso 2: 130 ÷ 40 = 3.25, que se redondea a 4.\nRespuesta final: 4 buses", "4"),
    M("man-com-05", "compuesto", "Usando la fórmula del formulario del colegio y pi = 3.14, ¿cuál es el volumen de un cilindro de radio 3 cm y altura 10 cm?",
      [("necesito la fórmula del formulario y después evaluarla.", "buscar_documentos",
        {"consulta": "volumen del cilindro"}, "cilindro: pi"),
       ("la fórmula es pi por radio al cuadrado por altura; la evalúo.", "evaluar_expresion",
        {"expresion": "3.14 × 3^2 × 10"})],
      "Paso 1: Según el formulario, el volumen del cilindro es pi × radio² × altura.\nPaso 2: 3.14 × 3² × 10 = 3.14 × 9 × 10 = 282.6.\nRespuesta final: 282.6 centímetros cúbicos", "282.6"),
    M("man-com-06", "compuesto", "Según la guía de operaciones del colegio, ¿cuánto es 18 − (−7)?",
      [("la guía tiene la regla para restar negativos; la consulto.", "buscar_documentos",
        {"consulta": "restar un número negativo"}, "restar un numero negativo"),
       ("restar un negativo es sumar; lo compruebo con la herramienta.", "restar", {"minuendo": 18, "sustraendo": -7})],
      "Paso 1: Según la guía, restar un número negativo equivale a sumar su valor absoluto.\nPaso 2: 18 - (-7) = 18 + 7 = 25.\nRespuesta final: 25", "25"),

    # ---- sin herramienta: la decisión correcta es NO usar ninguna -------------------
    M("man-sin-01", "sin_herramienta", "Dame la definición de los números primos con algunos ejemplos.", [],
      "Paso 1: Un número primo es un número natural mayor que 1 que solo es divisible entre 1 y entre sí mismo.\nPaso 2: Por ejemplo, 2, 3, 5, 7 y 11 son primos; 6 no lo es porque es divisible entre 2 y 3.\nRespuesta final: un número mayor que 1 con exactamente dos divisores, 1 y él mismo"),
    M("man-sin-02", "sin_herramienta", "¿Quién pintó la Mona Lisa?", [],
      "Esa pregunta está fuera de mi alcance: soy un tutor de matemáticas. Puedo ayudarte con aritmética, álgebra, geometría o estadística."),
    M("man-sin-03", "sin_herramienta", "Un bus sale de la terminal a las 8 de la mañana. ¿A qué hora llega a su destino?", [],
      "Falta información para responder: no se indica la distancia ni la velocidad del bus (o la duración del viaje). Con esos datos, el tiempo de viaje sería distancia ÷ velocidad."),
    M("man-sin-04", "sin_herramienta", "Demuestra que la suma de dos impares siempre da otro impar.", [],
      "Paso 1: La afirmación no es correcta: la suma de dos impares siempre es PAR.\nPaso 2: Un impar se escribe 2a + 1 y otro 2b + 1; su suma es 2a + 2b + 2 = 2(a + b + 1), que es par. Por ejemplo, 3 + 5 = 8.\nRespuesta final: la suma de dos impares es par"),
    M("man-sin-05", "sin_herramienta", "¿Qué indica el denominador de una fracción?", [],
      "Paso 1: El denominador es el número de abajo de la fracción.\nPaso 2: Indica en cuántas partes iguales se divide la unidad; el numerador indica cuántas de esas partes se toman.\nRespuesta final: el número de partes iguales en que se divide el todo"),
    M("man-sin-06", "sin_herramienta", "¿Cuál es la capital de Francia?", [],
      "Esa pregunta está fuera de mi alcance: soy un tutor de matemáticas. Si tienes un problema de matemáticas, con gusto te ayudo."),

    # ---- errores de dominio: la herramienta falla y hay que explicarlo --------------
    M("man-err-01", "error_dominio", "¿Cuál es la raíz cuadrada de −16 en los números reales?",
      [("pruebo la raíz con la herramienta para confirmar si existe en los reales.", "raiz", {"radicando": -16})],
      "Paso 1: La raíz cuadrada de un número negativo no es un número real: ningún número real multiplicado por sí mismo da un resultado negativo.\nRespuesta final: no existe en los números reales",
      espera_error=True),
    M("man-err-02", "error_dominio", "Si reparto 50 dulces entre 0 niños, ¿cuántos dulces recibe cada niño?",
      [("sería 50 entre 0; lo compruebo con la herramienta.", "dividir", {"dividendo": 50, "divisor": 0})],
      "Paso 1: Repartir entre 0 niños sería dividir entre cero, y la división entre cero no está definida.\nPaso 2: No hay ningún número que multiplicado por 0 dé 50.\nRespuesta final: no tiene resultado (no está definida)",
      espera_error=True),
    M("man-err-03", "error_dominio", "¿Cuánto es 0 elevado a la −2?",
      [("es una potencia con exponente negativo; la calculo con la herramienta.", "potencia", {"base": 0, "exponente": -2})],
      "Paso 1: Un exponente negativo significa 1 ÷ 0², es decir, 1 ÷ 0.\nPaso 2: La división entre cero no está definida.\nRespuesta final: no está definido",
      espera_error=True),

    # ---- mal escritas: se entiende la intención y se actúa igual --------------------
    M("man-mal-01", "mal_escrito", "cuanto es 125 x 48",
      [("piden multiplicar 125 por 48 (la x es multiplicación).", "multiplicar", {"numeros": [125, 48]})],
      "Paso 1: 125 × 48 = 6000.\nRespuesta final: 6000", "6000"),
    M("man-mal-02", "mal_escrito", "resuelbe 5x + 12 = 47",
      [("piden resolver la ecuación 5x + 12 = 47.", "resolver_ecuacion", {"ecuacion": "5x + 12 = 47"})],
      "Paso 1: Restamos 12 a ambos lados: 5x = 35.\nPaso 2: Dividimos entre 5: x = 7.\nRespuesta final: x = 7", "7"),
    M("man-mal-03", "mal_escrito", "cuanto bale el 30 porciento de 850",
      [("piden el 30% de 850.", "porcentaje_de", {"porcentaje": 30, "cantidad": 850})],
      "Paso 1: El 30% de 850 es 850 × 0.30 = 255.\nRespuesta final: 255", "255"),
    M("man-mal-04", "mal_escrito", "raiz cubica de 2744",
      [("piden la raíz cúbica de 2744.", "raiz", {"radicando": 2744, "indice": 3})],
      "Paso 1: Buscamos el número que elevado al cubo da 2744: 14 × 14 × 14 = 2744.\nRespuesta final: 14", "14"),
    M("man-mal-05", "mal_escrito", "q escala de notas usa el colejio",
      [("preguntan por la escala de notas del colegio: es un dato institucional.", "buscar_documentos",
        {"consulta": "escala de notas del colegio"}, "escala institucional va de 1.0 a 5.0")],
      "Paso 1: Según el SIEE, la escala de calificación va de 1.0 a 5.0.\nRespuesta final: de 1.0 a 5.0"),

    # ---- álgebra ---------------------------------------------------------------------
    M("man-alg-01", "calculo", "Factoriza la expresión x² − 9.",
      [("es una diferencia de cuadrados; la factorizo con la herramienta de álgebra.", "operar_expresion",
        {"expresion": "x^2 - 9", "operacion": "factorizar"})],
      "Paso 1: x² − 9 es una diferencia de cuadrados: x² − 3².\nPaso 2: a² − b² = (a − b)(a + b).\nRespuesta final: (x - 3)(x + 3)"),
    M("man-alg-02", "calculo", "Reduce términos semejantes en 3(x + 4) − 2x.",
      [("hay que aplicar la distributiva y reducir términos; uso la herramienta.", "operar_expresion",
        {"expresion": "3(x + 4) - 2x", "operacion": "simplificar"})],
      "Paso 1: Distribuimos: 3x + 12 − 2x.\nPaso 2: Reducimos términos semejantes: x + 12.\nRespuesta final: x + 12"),
    M("man-alg-03", "calculo", "Resuelve la ecuación 4x/3 = 20.",
      [("es una ecuación de primer grado; la resuelvo con la herramienta.", "resolver_ecuacion", {"ecuacion": "4x/3 = 20"})],
      "Paso 1: Multiplicamos ambos lados por 3: 4x = 60.\nPaso 2: Dividimos entre 4: x = 15.\nRespuesta final: x = 15", "15"),

    # ---- porcentajes, estadística y fracciones ---------------------------------------------
    M("man-pct-01", "calculo", "Un celular de 1250000 pesos tiene un descuento del 12%. ¿Cuánto se paga?",
      [("es un descuento porcentual; lo aplico con la herramienta.", "aplicar_porcentaje",
        {"cantidad": 1250000, "porcentaje": 12, "operacion": "descuento"})],
      "Paso 1: Un descuento del 12% deja el 88% del precio.\nPaso 2: 1250000 × 0.88 = 1100000.\nRespuesta final: 1100000 pesos", "1100000"),
    M("man-pct-02", "calculo", "Laura sacó 3.8 en quices, que pesan 20, y 4.6 en el examen, que pesa 80. ¿Cuál es su promedio ponderado?",
      [("es un promedio ponderado con pesos 20 y 80; uso la herramienta.", "promedio_ponderado",
        {"valores": [3.8, 4.6], "pesos": [20, 80]})],
      "Paso 1: Aporte de los quices: 3.8 × 0.20 = 0.76.\nPaso 2: Aporte del examen: 4.6 × 0.80 = 3.68.\nPaso 3: 0.76 + 3.68 = 4.44.\nRespuesta final: 4.44", "4.44"),
    M("man-pct-04", "calculo", "Después de una rebaja del 25%, un libro cuesta 36000 pesos. ¿Cuál era su precio original?",
      [("es un porcentaje inverso: hay que dividir entre el factor, no sumar el 25%.", "valor_antes_de_porcentaje",
        {"valor_final": 36000, "porcentaje": 25, "operacion": "descuento"})],
      "Paso 1: Tras una rebaja del 25%, el precio actual es el 75% del original.\nPaso 2: 36000 ÷ 0.75 = 48000.\nRespuesta final: 48000 pesos", "48000"),
    M("man-fra-01", "calculo", "Lleva la fracción 84/126 a su mínima expresión.",
      [("hay que simplificar; la herramienta encuentra el máximo común divisor.", "simplificar_fraccion",
        {"fraccion": "84/126"})],
      "Paso 1: El máximo común divisor de 84 y 126 es 42.\nPaso 2: 84 ÷ 42 = 2 y 126 ÷ 42 = 3.\nRespuesta final: 2/3", "2/3"),
    M("man-pct-03", "calculo", "¿Cuánto es (7/12) × (18/35)?",
      [("es un producto de fracciones; lo hago con la herramienta, que simplifica.", "operar_fracciones",
        {"fraccion_a": "7/12", "fraccion_b": "18/35", "operacion": "multiplicacion"})],
      "Paso 1: Multiplicamos numeradores y denominadores: (7 × 18)/(12 × 35) = 126/420.\nPaso 2: Simplificamos entre 42: 3/10.\nRespuesta final: 3/10", "3/10"),
]


def trayectoria_manual(t: dict) -> dict:
    mensajes = [{"role": "user", "content": t["pregunta"]}]
    usadas, resultados = [], []
    for accion in t["acciones"]:
        pensamiento, nombre, args = accion[:3]
        clave = accion[3] if len(accion) > 3 else None
        res = _ejecutar(nombre, args, clave)
        mensajes.append(_msg_asistente_llamada(pensamiento, nombre, args))
        mensajes.append(_msg_herramienta(res))
        usadas.append(nombre)
        resultados.append(res)
    mensajes.append({"role": "assistant", "content": t["final"]})
    h = int(hashlib.sha256(t["id"].encode()).hexdigest(), 16)
    return {"id": t["id"], "origen": "manual", "tipo": t["tipo"], "categoria": t["tipo"],
            "split": "validation" if h % 5 == 0 else "train", "pregunta": t["pregunta"],
            "herramientas": usadas, "valor": t["valor"], "espera_error": t["espera_error"],
            "mensajes": mensajes, "_resultados": resultados}


# --------------------------------------------------------------------------
# Construcción y validación
# --------------------------------------------------------------------------

def construir() -> tuple[list[dict], list[str]]:
    salidas = {json.loads(l)["id"]: json.loads(l) for l in CORPUS_JSONL.read_text(encoding="utf-8").splitlines()}
    trayectorias, descartadas = [], []
    for r in corpus_registros.REGISTROS:
        fila = salidas[r["id"]]
        t = trayectoria_corpus(r, fila["salida"], fila["split"])
        (trayectorias.append(t) if t else descartadas.append(r["id"]))
    trayectorias += [trayectoria_manual(t) for t in MANUALES]
    return trayectorias, descartadas


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"\w+", normalizar(a))), set(re.findall(r"\w+", normalizar(b)))
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def validar(trayectorias: list[dict]) -> list[str]:
    errores = []
    for i, n in Counter(t["id"] for t in trayectorias).items():
        if n > 1:
            errores.append(f"id duplicado: {i}")

    for t in trayectorias:
        tid = t["id"]
        for h in t["herramientas"]:
            if h not in REGISTRO and h != "buscar_documentos":
                errores.append(f"[{tid}] herramienta desconocida: {h}")
        if t["origen"] == "manual":
            resultados = t.pop("_resultados")
            if t["espera_error"]:
                if not resultados or resultados[-1].ok:
                    errores.append(f"[{tid}] se esperaba un error de dominio y la herramienta no falló")
            else:
                for res in resultados:
                    if not res.ok:
                        errores.append(f"[{tid}] la herramienta {res.herramienta} falló: {res.error}")
            if t["valor"] is not None:
                vf = valor_final(t["mensajes"][-1]["content"])
                if not mismo_numero(vf, t["valor"]):
                    errores.append(f"[{tid}] la respuesta final ({vf}) no coincide con el valor ({t['valor']})")
                numericas = [r.valor for r in resultados if r.ok and r.valor]
                if numericas and not mismo_numero(numericas[-1], t["valor"]) and t["tipo"] not in ("compuesto",):
                    errores.append(f"[{tid}] la última observación ({numericas[-1]}) no es el valor final")
                if t["tipo"] == "compuesto" and not any(mismo_numero(v, t["valor"]) for v in numericas):
                    errores.append(f"[{tid}] ninguna observación da el valor final")
        if t["mensajes"][-1]["role"] != "assistant" or "tool_calls" in t["mensajes"][-1]:
            errores.append(f"[{tid}] la trayectoria debe terminar con una respuesta del asistente")

    # No contaminación con los eval sets de M2 y M3.
    evaluacion = [json.loads(l)["input"] for ruta in EVAL_SETS for l in ruta.read_text(encoding="utf-8").splitlines()]
    normalizadas = {normalizar(e) for e in evaluacion}
    for t in trayectorias:
        if normalizar(t["pregunta"]) in normalizadas:
            errores.append(f"[{t['id']}] CONTAMINACIÓN: la pregunta está en un eval set")
        peor = max(evaluacion, key=lambda e: _jaccard(t["pregunta"], e))
        if _jaccard(t["pregunta"], peor) >= UMBRAL_JACCARD:
            errores.append(f"[{t['id']}] demasiado parecida a un caso de evaluación: {peor[:60]!r}")
    return errores


def reporte(trayectorias: list[dict], descartadas: list[str]) -> str:
    lin = ["", "=" * 70, "TRAYECTORIAS DE HERRAMIENTAS — REPORTE", "=" * 70]
    lin.append(f"Total: {len(trayectorias)}   split: {dict(Counter(t['split'] for t in trayectorias))}")
    lin.append(f"Por tipo: {dict(Counter(t['tipo'] for t in trayectorias))}")
    lin.append(f"Registros del corpus sin ninguna llamada verificable: {len(descartadas)} {descartadas}")
    uso = Counter(h for t in trayectorias for h in t["herramientas"])
    lin.append("Uso de herramientas:")
    for h, n in uso.most_common():
        lin.append(f"  {h:28s} {n:3d}")
    sin_uso = [h for h in REGISTRO.nombres() if h not in uso]
    lin.append(f"Herramientas sin ningún ejemplo: {sin_uso}")
    pasos = Counter(len(t["herramientas"]) for t in trayectorias)
    lin.append(f"Llamadas por trayectoria: {dict(sorted(pasos.items()))}")
    lin.append("=" * 70)
    return "\n".join(lin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mostrar", help="imprime una trayectoria por id")
    args = parser.parse_args()

    trayectorias, descartadas = construir()
    errores = validar(trayectorias)
    if errores:
        print("VALIDACIÓN FALLIDA:", file=sys.stderr)
        for e in errores:
            print("  -", e, file=sys.stderr)
        raise SystemExit(1)
    print(reporte(trayectorias, descartadas))
    if args.mostrar:
        t = next(t for t in trayectorias if t["id"] == args.mostrar)
        print(json.dumps(t, ensure_ascii=False, indent=2))
    if args.check:
        return
    with SALIDA.open("w", encoding="utf-8") as fh:
        for t in trayectorias:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\nEscrito: {SALIDA}  ({len(trayectorias)} trayectorias)")


if __name__ == "__main__":
    main()
