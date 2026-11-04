"""Verificador aritmético determinista de una hoja transcrita.

**Es un experimento.** Lo que mide y lo que no está en
`m4/README.md`; aquí solo está el mecanismo.

Por qué existe. Tres configuraciones del tutor —adaptador de diagnóstico,
adaptador de S10 y una variante del prompt— dieron 3/9, 2/9 y 2/9 sobre los
mismos casos de desarrollo. Y cambiar un solo carácter del prompt (`Paso 2` →
`Paso N`) desplazó la conclusión en 3 de 9 casos **sin que la hoja cambiara**: el
modelo no compara el valor calculado con el escrito, emite un índice casi
arbitrario.

La idea es no pedirle que compare. Las cuentas de una hoja se comprueban con
aritmética exacta —`parsear` de `herramientas.py`, que es SymPy racional, sin
coma flotante— y al tutor se le entrega el paso marcado. La tarea pasa de
"detecta el error" a "reporta el paso marcado".

Qué comprueba, y en este orden:

  1. **Un paso con una igualdad escrita** (`a op b = c`): se evalúa `a op b` y se
     compara con `c`. El primero que no cuadre se marca.
  2. **La línea de la respuesta final**: se compara con el último valor que las
     igualdades anteriores dejaron calculado. Es el error de `vis-03` y de
     `dg-a06`, donde los pasos están bien y lo que falla es el traslado.

Qué **no** comprueba, y es el límite que define el alcance:

  · **El planteamiento.** Si el estudiante eligió la operación equivocada pero la
    calculó bien, las igualdades cuadran y el verificador no dice nada. Eso sigue
    siendo trabajo del modelo.
  · **Las hojas sin aritmética**: ilegibles, fuera de dominio, o cuando se pide un
    dato que no está escrito. Ahí devuelve "no evaluable" y se abstiene, que es lo
    correcto: una herramienta que opina de lo que no puede ver es peor que una que
    calla.

Solo biblioteca estándar más `herramientas.py`, que ya está probado.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from herramientas import ErrorHerramienta, parsear  # [local]

# Una igualdad escrita dentro de un paso: "24 × 13 = 302", "3/4 + 1/8 = 7/8".
# Los delimitadores evitan partir una fracción por la mitad.
_IGUALDAD = re.compile(
    r"(?<![\w/.,])"
    r"(-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?"                 # operando a
    # Sin `:` entre los operadores: en "Paso 2: 24 × 13 = 302" se leería
    # "2 / 24 × 13", porque `normalizar_expresion` mapea `:` a división.
    r"(?:\s*[+\-−×x*÷/]\s*-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?)+)"   # op b (puede encadenar)
    r"\s*=\s*"
    r"(-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?)"                # resultado c
    r"(?![\w/])")

_MENCIONA_FINAL = re.compile(r"respuesta\s*final", re.IGNORECASE)
# El VLM numera los pasos dentro del texto ("Paso 2: 12x5=70"); se quita para que
# ese número no entre en la igualdad.
_PREFIJO_PASO = re.compile(r"^\s*paso\s*\d+\s*[:.\-]\s*", re.IGNORECASE)
_NUMERO = re.compile(r"-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?")


@dataclass
class Hallazgo:
    """Lo que el verificador puede decirle al tutor."""
    paso: int | None          # 1-based; None si no hay nada que marcar
    tipo: str                 # calculo | traslado | ninguno | no_evaluable
    detalle: str

    @property
    def marca_un_paso(self) -> bool:
        return self.paso is not None


def _valor(texto: str) -> Fraction | None:
    """El texto como número exacto, o None si no se puede interpretar."""
    try:
        expr = parsear(texto, x_es_por=True)
    except (ErrorHerramienta, Exception):      # noqa: BLE001
        return None
    try:
        if not expr.is_Rational:
            return None
        return Fraction(int(expr.p), int(expr.q))
    except (AttributeError, TypeError):
        return None


# Los caracteres de una expresión aritmética escrita. Sin `:` —en «Paso 2: 24 ×
# 13» se leería como división— y sin `%`.
_EXPR = r"[\d\s.,/+\-−×xX*÷()]"
_COLA = re.compile(rf"(-?\(?\d{_EXPR}*?)[\s.,]*$")
_CABEZA = re.compile(rf"^\s*(-?\(?\d{_EXPR}*)")


def _recortar(expr: str) -> str:
    """Quita lo que no es expresión al final: puntos, comas, operadores sueltos."""
    return re.sub(r"[\s.,+\-−×xX*÷/(]+$", "", expr).strip()


def igualdades_de(paso: str) -> list:
    """[(izquierda, derecha)] de CADA igualdad adyacente escrita en un paso.

    «7/8 ÷ 3/8 = 7/8 × 8/3 = 56/24» son DOS igualdades —(7/8 ÷ 3/8, 7/8 × 8/3) y
    (7/8 × 8/3, 56/24)—, y cada una se comprueba por separado. La versión
    anterior, con una sola expresión regular, partía la cadena por el medio:
    leía «7/8 ÷ 3/8 = 7/8» y marcaba como error un paso correcto.

    Se parte el texto por «=»; de cada trozo se toma la expresión pegada al
    signo —la cola del trozo de la izquierda, la cabeza del de la derecha—. Un
    trozo intermedio es a la vez la derecha de una igualdad y la izquierda de la
    siguiente.
    """
    limpio = _PREFIJO_PASO.sub("", paso or "")
    trozos = limpio.split("=")
    pares = []
    for izq_txt, der_txt in zip(trozos, trozos[1:]):
        cola, cabeza = _COLA.search(izq_txt), _CABEZA.match(der_txt)
        if not cola or not cabeza:
            continue
        izq, der = _recortar(cola.group(1)), _recortar(cabeza.group(1))
        if izq and der and any(c.isdigit() for c in izq) and any(c.isdigit() for c in der):
            pares.append((izq, der))
    return pares


def revisar_hoja(pasos) -> Hallazgo:
    """Marca el PRIMER paso cuya aritmética no cuadra.

    El primero y no todos: si el paso 2 está mal, los siguientes arrastran el
    error y señalarlos aparte confundiría al estudiante. Es la misma regla que se
    le pide al tutor en el prompt.
    """
    # El prefijo "Paso N:" se quita UNA vez, aquí: si no, su número entra tanto en
    # las igualdades como en la comparación de la respuesta final.
    pasos = [_PREFIJO_PASO.sub("", str(p)) for p in (pasos or [])]
    if not pasos:
        return Hallazgo(None, "no_evaluable", "la hoja no tiene pasos que revisar")

    ultimo_calculado: Fraction | None = None
    hubo_igualdad = False

    for i, paso in enumerate(pasos, 1):
        es_final = bool(_MENCIONA_FINAL.search(paso))

        for izq, der in igualdades_de(paso):
            a, b = _valor(izq), _valor(der)
            if a is None or b is None:
                continue                      # no interpretable: no se opina
            hubo_igualdad = True
            if a != b:
                return Hallazgo(i, "calculo",
                                f"el paso {i} no cuadra: {izq.strip()} da {_fmt(a)}, "
                                f"y está escrito {der.strip()}")
            ultimo_calculado = a

        # La línea de la respuesta final: ¿coincide con lo último calculado?
        if es_final and ultimo_calculado is not None and not igualdades_de(paso):
            escrito = _primer_numero(paso)
            if escrito is not None and escrito != ultimo_calculado:
                return Hallazgo(i, "traslado",
                                f"el paso {i} copia mal el resultado: era "
                                f"{_fmt(ultimo_calculado)} y está escrito {_fmt(escrito)}")

    if not hubo_igualdad:
        return Hallazgo(None, "no_evaluable",
                        "no hay ninguna igualdad aritmética que comprobar")
    return Hallazgo(None, "ninguno", "todas las cuentas escritas cuadran")


def pasos_comprobados(pasos) -> set:
    """Los pasos (1-based) cuya aritmética el verificador COMPROBÓ y cuadra.

    Un paso entra si tiene al menos una igualdad interpretable y todas se
    cumplen, o si es la línea de respuesta final y coincide con lo último
    calculado. Un paso sin nada que comprobar —«Sumo los dos tramos»— no entra:
    de él el verificador no sabe nada, ni bueno ni malo.
    """
    pasos = [_PREFIJO_PASO.sub("", str(p)) for p in (pasos or [])]
    ok, ultimo = set(), None
    for i, paso in enumerate(pasos, 1):
        pares = [(_valor(a), _valor(b)) for a, b in igualdades_de(paso)]
        pares = [(a, b) for a, b in pares if a is not None and b is not None]
        if pares:
            if all(a == b for a, b in pares):
                ok.add(i)
                ultimo = pares[-1][0]
            else:
                break
        elif _MENCIONA_FINAL.search(paso) and ultimo is not None:
            if _primer_numero(paso) == ultimo:
                ok.add(i)
            else:
                break
    return ok


# --------------------------------------------------------------------------
# Planteamiento y datos (ronda del 2026-11-05)
#
# Lo de arriba comprueba que cada cuenta esté bien HECHA. Lo de abajo compara
# la hoja con el enunciado resuelto por separado, para dos fallos que las
# cuentas no ven: la operación equivocada bien calculada (m5v-02, suma donde
# hay que restar) y un error inventado en una hoja correcta (m5v-08). La
# referencia la produce quien llame —en la app, el tutor resolviendo el
# enunciado CONFIRMADO, comprobado a su vez por `revisar_hoja`—; aquí solo se
# compara, con aritmética exacta.
# --------------------------------------------------------------------------

# Una expresión con al menos una operación: «1240 + 365», «7/8 − 3/8», «40 x 0.35».
_EXPRESION = re.compile(
    r"(?<![\w/.,])(-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?"
    r"(?:\s*[+\-−×xX*÷]\s*-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?)+)(?![\w/])")


def expresiones_de(paso: str) -> list:
    """Las cuentas planteadas en un paso, con o sin «=» detrás."""
    return [m.group(1).strip() for m in _EXPRESION.finditer(_PREFIJO_PASO.sub("", paso or ""))]


def valor_final(pasos) -> Fraction | None:
    """El resultado al que llega una hoja: el número de «Respuesta final», o si
    no la hay, lo último escrito a la derecha de una igualdad."""
    pasos = [_PREFIJO_PASO.sub("", str(p)) for p in (pasos or [])]
    for p in reversed(pasos):
        if _MENCIONA_FINAL.search(p):
            cola = _MENCIONA_FINAL.split(p)[-1]
            # «Respuesta final: 52000 × 0.75 = 39000» es 39000, no 52000; y
            # «Respuesta final: 40 × 0.35» es 14 (corrida de 4825044).
            pares = igualdades_de(cola)
            if pares and _valor(pares[-1][1]) is not None:
                return _valor(pares[-1][1])
            expr = expresiones_de(cola)
            if expr and _valor(expr[-1]) is not None:
                return _valor(expr[-1])
            return _primer_numero(cola)
    for p in reversed(pasos):
        pares = igualdades_de(p)
        if pares:
            return _valor(pares[-1][1])
    return None


def valores_de(pasos) -> set:
    """Todos los valores que aparecen calculados en una solución: cada cuenta
    planteada y cada lado de cada igualdad."""
    salida = set()
    for p in pasos or []:
        for e in expresiones_de(str(p)):
            v = _valor(e)
            if v is not None:
                salida.add(v)
        for a, b in igualdades_de(str(p)):
            for x in (_valor(a), _valor(b)):
                if x is not None:
                    salida.add(x)
    final = valor_final(pasos)
    if final is not None:
        salida.add(final)
    return salida


def _operandos(expr: str) -> tuple:
    """Los datos con que opera una cuenta, sin orden: «365 + 1240» y
    «1240 − 365» operan con los mismos."""
    valores = (_valor(n) for n in _NUMERO.findall(expr.replace("−", "-")))
    return tuple(sorted(str(abs(v)) for v in valores if v is not None))


def operandos_de(pasos) -> set:
    """Los conjuntos de datos con que opera cada cuenta de una solución."""
    return {_operandos(e) for p in (pasos or []) for e in expresiones_de(str(p))}


def revisar_planteamiento(pasos, final_referencia, valores_referencia,
                          hasta: int | None = None,
                          operandos_referencia: set | None = None,
                          esperadas: set | None = None) -> Hallazgo | None:
    """Compara la hoja con el enunciado resuelto aparte. None si no se puede.

      · «coincide»       la hoja llega al mismo resultado que la referencia.
      · «planteamiento»  no llega, y el paso p es el primero que plantea una
                         cuenta cuyo valor no aparece en la referencia.
      · «discrepa»       no llega, pero no hay un paso que lo explique: no se
                         elige uno a ciegas.

    `hasta`: solo se buscan pasos ANTERIORES a ese (el primer error de cálculo):
    un planteamiento posterior no es el primer error.
    """
    if final_referencia is None:
        return None
    final = valor_final(pasos)
    if final is None:
        return None
    referencia = set(valores_referencia or ()) | {final_referencia}
    if final == final_referencia and hasta is None:
        return Hallazgo(None, "coincide",
                        f"tu resultado, {_fmt(final)}, es el mismo que sale al "
                        "resolver el enunciado por separado")
    if final == final_referencia:
        return None
    limpios = [_PREFIJO_PASO.sub("", str(p)) for p in (pasos or [])]
    # Firmas de la referencia: con qué datos opera cada cuenta. Un paso solo se
    # señala como mal planteado si la referencia opera con LOS MISMOS datos y
    # obtiene otra cosa (1240 + 365 frente a 1240 − 365). Si la referencia
    # siguió otro camino, no hay con qué comparar ese paso y no se elige.
    for i, paso in enumerate(limpios, 1):
        if hasta is not None and i >= hasta:
            break
        for e in expresiones_de(paso):
            v = _valor(e)
            # Si el enunciado nombra la operación y el paso la usa, no se le
            # acusa de mal planteado aunque la referencia diga otra cosa.
            if esperadas and operaciones_de(e) & esperadas:
                continue
            if (v is not None and v not in referencia
                    and (operandos_referencia is None
                         or _operandos(e) in operandos_referencia)):
                return Hallazgo(
                    i, "planteamiento",
                    f"en el paso {i} se plantea {e}, que da {_fmt(v)}, pero al "
                    f"resolver el enunciado por separado el resultado es "
                    f"{_fmt(final_referencia)}: la cuenta puede estar bien hecha, "
                    "pero no es la operación que pide el enunciado")
    if hasta is not None:
        return None
    return Hallazgo(None, "discrepa",
                    f"tu resultado, {_fmt(final)}, no coincide con el que sale al "
                    f"resolver el enunciado por separado, {_fmt(final_referencia)}, "
                    "pero no encuentro un paso concreto que lo explique: revisa "
                    "el planteamiento con tu profesor")


def _fmt_legible(v: Fraction) -> str:
    """Entero, decimal exacto si lo hay (13,5 -> «13.5»), o fracción."""
    if v.denominator == 1:
        return str(v.numerator)
    d = v.denominator
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    if d == 1:
        return f"{float(v):.10f}".rstrip("0").rstrip(".")
    return f"{v.numerator}/{v.denominator}"


_TOKEN_NUMERO = r"(?<![\d.,/])({})(?![\d/]|[.,]\d)"


def corregir_cuentas(pasos) -> tuple:
    """-> (pasos con las cuentas rehechas, [qué se corrigió]).

    Para resolver un enunciado el modelo PLANTEA y el código CALCULA: cada
    igualdad «a op b = c» con c mal se reescribe con el valor exacto, y ese
    valor sustituye al equivocado en los pasos siguientes y en la respuesta
    final. El planteamiento —qué operación hacer con qué datos— sigue siendo
    del modelo y no se toca. Corrida del 2026-11-05: el modelo planteaba bien
    1764 ÷ 7 y escribía 246.
    """
    salida, cambios, sustituir = [], [], {}
    ultimo = None
    for paso in pasos or []:
        texto = str(paso)
        for malo, bueno in sustituir.items():
            texto = re.sub(_TOKEN_NUMERO.format(re.escape(malo)), bueno, texto)
        for _ in range(6):                      # cadenas: una igualdad por vuelta
            arreglado = False
            for izq, der in igualdades_de(texto):
                a, b = _valor(izq), _valor(der)
                if a is None or b is None:
                    continue
                ultimo = a
                if a != b:
                    i = texto.find(izq)
                    j = texto.find("=", i + len(izq))
                    k = texto.find(der, j)
                    if min(i, j, k) < 0:
                        continue
                    bueno = _fmt_legible(a)
                    texto = texto[:k] + bueno + texto[k + len(der):]
                    sustituir[der.strip()] = bueno
                    cambios.append(f"{izq} = {bueno} (el borrador decía {der})")
                    arreglado = True
                    break
            if not arreglado:
                break
        cola = _MENCIONA_FINAL.split(texto)[-1] if _MENCIONA_FINAL.search(texto) else ""
        expr = expresiones_de(cola) if not igualdades_de(cola) else []
        if expr and _valor(expr[-1]) is not None:
            # «Respuesta final: 40 × 0.35» -> se escribe el resultado.
            valor = _fmt_legible(_valor(expr[-1]))
            fin = texto.rfind(expr[-1]) + len(expr[-1])
            texto = f"{texto[:fin]} = {valor}{texto[fin:]}"
            cambios.append(f"respuesta final {expr[-1]} = {valor} (el borrador no la calculaba)")
            ultimo = _valor(expr[-1])
        elif _MENCIONA_FINAL.search(texto) and not igualdades_de(texto) and ultimo is not None:
            m = _NUMERO.search(_MENCIONA_FINAL.split(texto)[-1])
            if m and _valor(m.group()) is not None and _valor(m.group()) != ultimo:
                inicio = texto.rfind(m.group())
                bueno = _fmt_legible(ultimo)
                cambios.append(f"respuesta final {bueno} (el borrador decía {m.group()})")
                texto = texto[:inicio] + bueno + texto[inicio + len(m.group()):]
        salida.append(texto)
    return salida, cambios


# --------------------------------------------------------------------------
# ¿Un borrador usa los datos del enunciado y la operación que este pide?
#
# El modelo de 1.5B inventa datos («39000 × 14», «14 ÷ 2») y a veces elige la
# operación contraria a la que dice el enunciado («se usan 3/8, ¿cuánto
# queda?» resuelto como 7/8 ÷ 3/8, dos veces). Dos comprobaciones, ambas de
# vocabulario escolar general y no de estas hojas:
#   · un dato que no está en el enunciado ni sale de una cuenta anterior;
#   · una palabra del enunciado que fija la operación («quedan», «se
#     retiran», «reparten», «en total», «cada», «%»).
# --------------------------------------------------------------------------

_PALABRAS_NUMERO = {"dos": [2], "tres": [3], "cuatro": [4], "cinco": [5], "seis": [6],
                    "siete": [7], "ocho": [8], "nueve": [9], "diez": [10], "doce": [12],
                    "mitad": [2, Fraction(1, 2)], "medio": [2, Fraction(1, 2)],
                    "media": [2, Fraction(1, 2)], "doble": [2], "triple": [3],
                    "tercio": [3, Fraction(1, 3)], "cuarto": [4, Fraction(1, 4)],
                    "docena": [12], "cien": [100], "mil": [1000]}


def datos_del_enunciado(enunciado: str) -> set:
    """Los valores que un procedimiento puede usar sin calcularlos: los números
    del enunciado, sus formas de porcentaje (25 % -> 0.25, 0.75, 75) y las
    cantidades escritas con palabras. 1 y 100 siempre."""
    import unicodedata
    t = unicodedata.normalize("NFKD", (enunciado or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    datos = {Fraction(1), Fraction(100)}
    for m in re.finditer(r"(\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?)\s*(%|por ?ciento)?", t):
        v = _valor(m.group(1))
        if v is None:
            continue
        datos.add(v)
        if m.group(2):
            datos |= {v / 100, 1 - v / 100, 100 - v}
    for palabra, valores in _PALABRAS_NUMERO.items():
        if re.search(rf"\b{palabra}\b", t):
            datos |= {Fraction(x) for x in valores}
    # El inverso de un dato (dividir por 3/8 es multiplicar por 8/3).
    datos |= {1 / v for v in list(datos) if v}
    return datos


def datos_ajenos(pasos, enunciado: str) -> list:
    """Los operandos que un procedimiento usa sin que salgan del enunciado ni
    de una cuenta anterior. [] si todos tienen origen."""
    conocidos = set(datos_del_enunciado(enunciado))
    ajenos = []
    for p in pasos or []:
        texto = _PREFIJO_PASO.sub("", str(p))
        for e in expresiones_de(texto):
            for n in _NUMERO.findall(e.replace("−", "-")):
                v = _valor(n)
                if v is not None and abs(v) not in conocidos and n not in ajenos:
                    ajenos.append(n.strip())
            v = _valor(e)
            if v is not None:
                conocidos.add(v)
        for a, b in igualdades_de(texto):
            for x in (_valor(a), _valor(b)):
                if x is not None:
                    conocidos.add(x)
    return ajenos


_PISTAS_DE_OPERACION = {
    "-": r"\b(queda|quedan|quedaron|sobra|sobran|retir\w*|se usan?|uso|gast\w*|pierd\w*|"
         r"perd\w*|regal\w*|quit\w*|diferencia|faltan?|descuento)\b",
    "+": r"\b(en total|juntos|juntas|suman|anad\w*|agreg\w*)\b",
    "÷": r"\b(repart\w*|partes iguales)\b",
    "×": r"\b(cada|por cada|veces)\b|%|por ?ciento",
}


def operaciones_esperadas(enunciado: str) -> set:
    """Las operaciones que el enunciado pide con palabras. Vacío si no dice."""
    import unicodedata
    t = unicodedata.normalize("NFKD", (enunciado or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return {op for op, pista in _PISTAS_DE_OPERACION.items() if re.search(pista, t)}


def operaciones_de(expr: str) -> set:
    """Las operaciones de una cuenta, normalizadas: × x * -> ×, ÷ : -> ÷."""
    ops = set()
    for o in re.findall(r"(?<=[\d)\s])\s*([+\-−×xX*÷:])\s*(?=[\d(-])", expr or ""):
        ops.add({"x": "×", "X": "×", "*": "×", "×": "×", "÷": "÷", ":": "÷",
                 "−": "-", "-": "-", "+": "+"}[o])
    return ops


def borrador_valido(pasos, enunciado: str) -> tuple:
    """-> (válido, motivo). Un borrador vale si no inventa datos y, cuando el
    enunciado nombra la operación, la usa en algún paso."""
    ajenos = datos_ajenos(pasos, enunciado)
    if ajenos:
        return False, f"usa datos que no están en el enunciado: {', '.join(ajenos[:3])}"
    esperadas = operaciones_esperadas(enunciado)
    usadas = {o for p in pasos or [] for e in expresiones_de(str(p)) for o in operaciones_de(e)}
    if esperadas and usadas and not (esperadas & usadas):
        return False, ("no usa la operación que pide el enunciado "
                       f"({', '.join(sorted(esperadas))})")
    return True, ""


# Cantidades escritas con palabras: si el enunciado las trae, sí tiene datos.
_NUMEROS_EN_PALABRAS = re.compile(
    r"\b(uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|"
    r"veinte|treinta|cuarenta|cincuenta|cien|ciento|mil|media|medio|mitad|"
    r"doble|triple|tercio|cuarto|docena)\b", re.IGNORECASE)


def falta_un_dato(enunciado, pasos) -> Hallazgo | None:
    """El enunciado no trae ningún dato, pero los pasos operan con números.

    Es lo que pasa cuando la pregunta pide algo que no está en la hoja
    (m5v-12: «¿Cuánto me costó el tercer artículo?» y los pasos suman los dos
    primeros). Sin datos en el enunciado no hay forma de saber si las cuentas
    responden a la pregunta, y señalar un paso sería inventar.
    """
    texto = str(enunciado or "")
    if not texto.strip() or any(c.isdigit() for c in texto) or _NUMEROS_EN_PALABRAS.search(texto):
        return None
    usados = [e for p in (pasos or []) for e in expresiones_de(str(p))]
    if not usados:
        return None
    numeros = []
    for e in usados:
        for n in _NUMERO.findall(e):
            if n not in numeros:
                numeros.append(n)
    return Hallazgo(None, "dato_ausente",
                    "el enunciado no trae ningún dato numérico, y los números que "
                    f"usas ({', '.join(numeros[:4])}) no aparecen en él: el dato que "
                    "pide la pregunta no está en la foto. Sin ese dato no puedo "
                    "decir si el procedimiento responde a la pregunta")


def _primer_numero(texto: str) -> Fraction | None:
    m = _NUMERO.search(texto or "")
    return _valor(m.group()) if m else None


def _fmt(v: Fraction) -> str:
    if v.denominator == 1:
        return str(v.numerator)
    return f"{v.numerator}/{v.denominator}"


def pista_para_el_tutor(h: Hallazgo) -> str:
    """Lo que se le añadiría al prompt. Una frase, sin veredicto.

    No dice `Diagnóstico:` a propósito: la conclusión la sigue emitiendo el tutor,
    y así el scorecard sigue midiendo al tutor y no a esta función.
    """
    if h.tipo == "calculo" or h.tipo == "traslado":
        return f"Comprobación automática de las cuentas: {h.detalle}."
    if h.tipo == "ninguno":
        return ("Comprobación automática de las cuentas: todas las igualdades "
                "escritas en la hoja cuadran.")
    return ("Comprobación automática de las cuentas: no hay cuentas que "
            "comprobar en esta hoja.")
