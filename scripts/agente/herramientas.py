"""
Herramientas matemáticas del tutor.

Todas trabajan con aritmética EXACTA (Fraction / SymPy racional): una
calculadora que devuelve 0.30000000000000004 no le ahorra errores a nadie.

Cada herramienta devuelve {"resultado": texto para el modelo, "valor": forma
canónica para las métricas}. Los errores de dominio (división entre cero, raíz
par de un negativo) se lanzan como ErrorHerramienta: el agente los recibe como
observación y debe EXPLICARLOS, que es justo lo que pide el caso adv-02.

Familias (usadas para medir si el agente eligió bien):
  aritmetica, fracciones, potencias_raices, porcentajes, algebra, estadistica
  (+ documentos, que registra el notebook con el retriever de S08)

Para añadir una herramienta: escribir una función tipada con docstring y
decorarla con `@REGISTRO.herramienta(familia=...)`. Nada más cambia.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Literal

from registro import ErrorHerramienta, RegistroHerramientas, a_fraccion  # [local]

REGISTRO = RegistroHerramientas()

FAMILIAS_NUMERICAS = ("aritmetica", "fracciones", "potencias_raices", "porcentajes", "estadistica")


# --------------------------------------------------------------------------
# Formato de salida
# --------------------------------------------------------------------------

def _decimal(q: Fraction, max_decimales: int = 6) -> str:
    """Decimal legible: exacto si termina en <= max_decimales, redondeado si no."""
    if q.denominator == 1:
        return str(q.numerator)
    texto = f"{float(round(q, max_decimales)):.{max_decimales}f}".rstrip("0").rstrip(".")
    return "0" if texto in ("-0", "") else texto


def _es_decimal_exacto(q: Fraction, max_decimales: int = 6) -> bool:
    d = q.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return d == 1 and round(q, max_decimales) == q


def formatear(q: Fraction, preferir_fraccion: bool = False) -> dict:
    """{"resultado", "valor"} para un racional."""
    if q.denominator == 1:
        v = str(q.numerator)
        return {"resultado": v, "valor": v}
    frac = f"{q.numerator}/{q.denominator}"
    if preferir_fraccion or not _es_decimal_exacto(q):
        return {"resultado": f"{frac} (≈ {_decimal(q)})", "valor": frac}
    dec = _decimal(q)
    return {"resultado": dec, "valor": dec}


# --------------------------------------------------------------------------
# Parser seguro de expresiones (SymPy, nunca eval de Python)
# --------------------------------------------------------------------------

_PERMITIDOS = re.compile(r"^[0-9a-z_+\-*/().,\s=]*$")
_IDENTIFICADORES_OK = {"x", "y", "z", "a", "b", "c", "n", "m", "t", "sqrt", "pi"}
_SUPERINDICES = {"²": "**2", "³": "**3", "⁴": "**4", "⁵": "**5"}


def normalizar_expresion(texto: str, x_es_por: bool = False) -> str:
    t = texto.strip().lower()
    t = t.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    t = t.replace("^", "**").replace(":", "/")
    for s, r in _SUPERINDICES.items():
        t = t.replace(s, r)
    t = re.sub(r"√\s*\(", "sqrt(", t)
    t = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", t)
    t = re.sub(r"ra[ií]z\s+(?:cuadrada\s+)?de\s+(\d+(?:\.\d+)?)", r"sqrt(\1)", t)
    if x_es_por:                                   # "3847 x 296": aquí x es "por"
        t = re.sub(r"(?<=[\d)])\s*x\s*(?=[\d(])", "*", t)
    t = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", t)  # 18,500 -> 18500
    t = re.sub(r"(\d),(\d)", r"\1.\2", t)          # 2,35   -> 2.35
    return t


def _decimales_a_racional(t: str) -> str:
    return re.sub(r"\d+\.\d+", lambda m: f"({Fraction(m.group())})", t)


def parsear(texto: str, x_es_por: bool = False):
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor, implicit_multiplication_application, parse_expr, standard_transformations,
    )
    t = normalizar_expresion(texto, x_es_por=x_es_por)
    if not _PERMITIDOS.match(t) or "__" in t:
        raise ErrorHerramienta(f"la expresión contiene símbolos no permitidos: {texto!r}")
    desconocidos = set(re.findall(r"[a-z_]+", t)) - _IDENTIFICADORES_OK
    if desconocidos:
        raise ErrorHerramienta(f"nombres no permitidos en la expresión: {sorted(desconocidos)}")
    locales = {n: sympy.Symbol(n) for n in _IDENTIFICADORES_OK - {"sqrt", "pi"}}
    locales.update(sqrt=sympy.sqrt, pi=sympy.pi)
    transformaciones = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        return parse_expr(_decimales_a_racional(t), local_dict=locales,
                          transformations=transformaciones, evaluate=True)
    except Exception as e:  # noqa: BLE001
        raise ErrorHerramienta(f"no pude interpretar la expresión {texto!r}: {e}") from e


def _sympy_a_salida(expr) -> dict:
    import sympy
    if expr.has(sympy.zoo, sympy.nan) or expr == sympy.oo:
        raise ErrorHerramienta("la operación no está definida (hay una división entre cero)")
    if expr.is_Rational:
        return formatear(Fraction(int(expr.p), int(expr.q)))
    if expr.is_number:
        if not expr.is_real:
            raise ErrorHerramienta("el resultado no es un número real")
        aprox = float(sympy.N(expr, 15))
        texto = f"{aprox:.6f}".rstrip("0").rstrip(".")
        return {"resultado": f"{sympy.sstr(expr)} (≈ {texto})", "valor": texto}
    texto = sympy.sstr(expr)
    return {"resultado": texto, "valor": texto}


# --------------------------------------------------------------------------
# ARITMÉTICA
# --------------------------------------------------------------------------

@REGISTRO.herramienta(familia="aritmetica")
def sumar(numeros: list[float]) -> dict:
    """Suma dos o más números (enteros o decimales) de forma exacta.

    Args:
        numeros: lista de números a sumar, por ejemplo [245, 378.5].
    """
    return formatear(sum(numeros, Fraction(0)))


@REGISTRO.herramienta(familia="aritmetica")
def restar(minuendo: float, sustraendo: float) -> dict:
    """Resta dos números de forma exacta: minuendo menos sustraendo.

    Args:
        minuendo: número del que se resta.
        sustraendo: número que se resta.
    """
    return formatear(minuendo - sustraendo)


@REGISTRO.herramienta(familia="aritmetica")
def multiplicar(numeros: list[float]) -> dict:
    """Multiplica dos o más números de forma exacta.

    Args:
        numeros: lista de factores, por ejemplo [3847, 296].
    """
    producto = Fraction(1)
    for n in numeros:
        producto *= n
    return formatear(producto)


@REGISTRO.herramienta(familia="aritmetica")
def dividir(dividendo: float, divisor: float,
            redondeo: Literal["ninguno", "arriba", "abajo"] = "ninguno") -> dict:
    """Divide dos números. Permite redondear el cociente a entero hacia arriba
    (p. ej. cuántos vehículos hacen falta) o hacia abajo (cuántos grupos completos).

    Args:
        dividendo: número que se divide.
        divisor: número entre el que se divide. No puede ser cero.
        redondeo: 'ninguno' (cociente exacto), 'arriba' o 'abajo' (a entero).
    """
    if divisor == 0:
        raise ErrorHerramienta("la división entre cero no está definida: ningún número "
                               "multiplicado por 0 da el dividendo")
    q = dividendo / divisor
    if redondeo == "arriba":
        return {**formatear(Fraction(math.ceil(q))), "cociente_exacto": _decimal(q)}
    if redondeo == "abajo":
        return {**formatear(Fraction(math.floor(q))), "cociente_exacto": _decimal(q)}
    return formatear(q)


@REGISTRO.herramienta(familia="aritmetica", comodin_de=FAMILIAS_NUMERICAS)
def evaluar_expresion(expresion: str) -> dict:
    """Evalúa una expresión aritmética completa respetando la jerarquía de
    operaciones. Acepta +, -, ×, ÷, paréntesis, potencias (^) y raíces (sqrt o √).

    Args:
        expresion: la expresión, por ejemplo '2.35 × 4.8 - 1.964' o '(5/8)*(4/7)'.
    """
    expr = parsear(expresion, x_es_por=True)
    if expr.free_symbols:
        raise ErrorHerramienta("la expresión tiene incógnitas: usa resolver_ecuacion u operar_expresion")
    return _sympy_a_salida(expr)


# --------------------------------------------------------------------------
# FRACCIONES
# --------------------------------------------------------------------------

@REGISTRO.herramienta(familia="fracciones")
def operar_fracciones(fraccion_a: str, fraccion_b: str,
                      operacion: Literal["suma", "resta", "multiplicacion", "division"]) -> dict:
    """Suma, resta, multiplica o divide dos fracciones y simplifica el resultado.

    Args:
        fraccion_a: primera fracción, por ejemplo '17/24' (también acepta enteros).
        fraccion_b: segunda fracción, por ejemplo '11/36'.
        operacion: 'suma', 'resta', 'multiplicacion' o 'division'.
    """
    a, b = a_fraccion(fraccion_a), a_fraccion(fraccion_b)
    if operacion == "division" and b == 0:
        raise ErrorHerramienta("no se puede dividir entre una fracción igual a cero")
    q = {"suma": a + b, "resta": a - b, "multiplicacion": a * b,
         "division": a / b if b else None}[operacion]
    return formatear(q, preferir_fraccion=True)


@REGISTRO.herramienta(familia="fracciones")
def simplificar_fraccion(fraccion: str) -> dict:
    """Lleva una fracción a su mínima expresión e indica el máximo común divisor usado.

    Args:
        fraccion: la fracción, por ejemplo '1386/2310'.
    """
    m = re.fullmatch(r"\s*(-?\d+)\s*/\s*(-?\d+)\s*", str(fraccion))
    if not m:
        raise ErrorHerramienta(f"'{fraccion}' no tiene la forma numerador/denominador con enteros")
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        raise ErrorHerramienta("el denominador no puede ser cero")
    mcd = math.gcd(num, den)
    return {**formatear(Fraction(num, den), preferir_fraccion=True), "mcd": mcd}


# --------------------------------------------------------------------------
# POTENCIAS Y RAÍCES
# --------------------------------------------------------------------------

@REGISTRO.herramienta(familia="potencias_raices")
def potencia(base: float, exponente: float) -> dict:
    """Eleva una base a un exponente (entero o fraccionario).

    Args:
        base: la base, por ejemplo 15.
        exponente: el exponente, por ejemplo 3.
    """
    import sympy
    if base == 0 and exponente < 0:
        raise ErrorHerramienta("cero elevado a un exponente negativo no está definido")
    if exponente.denominator == 1:
        return formatear(base ** exponente.numerator)
    return _sympy_a_salida(sympy.Rational(base.numerator, base.denominator)
                           ** sympy.Rational(exponente.numerator, exponente.denominator))


@REGISTRO.herramienta(familia="potencias_raices")
def raiz(radicando: float, indice: int = 2) -> dict:
    """Calcula la raíz n-ésima de un número; exacta si existe, aproximada si no.

    Args:
        radicando: el número dentro de la raíz, por ejemplo 7056.
        indice: 2 para raíz cuadrada, 3 para cúbica, etc.
    """
    import sympy
    if indice < 1:
        raise ErrorHerramienta("el índice de la raíz debe ser un entero positivo")
    if radicando < 0 and indice % 2 == 0:
        raise ErrorHerramienta("la raíz de índice par de un número negativo no es un número real")
    r = sympy.Rational(radicando.numerator, radicando.denominator)
    expr = -sympy.root(-r, indice) if r < 0 else sympy.root(r, indice)
    return _sympy_a_salida(sympy.nsimplify(expr))


# --------------------------------------------------------------------------
# PORCENTAJES
# --------------------------------------------------------------------------

@REGISTRO.herramienta(familia="porcentajes")
def porcentaje_de(porcentaje: float, cantidad: float) -> dict:
    """Calcula el porcentaje de una cantidad (p. ej. el 18% de 2450000).

    Args:
        porcentaje: el porcentaje como número, por ejemplo 18 para 18%.
        cantidad: la cantidad sobre la que se calcula.
    """
    return formatear(porcentaje / 100 * cantidad)


@REGISTRO.herramienta(familia="porcentajes")
def aplicar_porcentaje(cantidad: float, porcentaje: float,
                       operacion: Literal["aumento", "descuento"]) -> dict:
    """Aplica un aumento o un descuento porcentual a una cantidad y devuelve el
    valor final. Para descuentos sucesivos, llamarla una vez por descuento.

    Args:
        cantidad: el valor inicial.
        porcentaje: el porcentaje como número, por ejemplo 20 para 20%.
        operacion: 'aumento' o 'descuento'.
    """
    factor = 1 + porcentaje / 100 if operacion == "aumento" else 1 - porcentaje / 100
    return {**formatear(cantidad * factor), "factor": _decimal(factor)}


@REGISTRO.herramienta(familia="porcentajes")
def valor_antes_de_porcentaje(valor_final: float, porcentaje: float,
                              operacion: Literal["aumento", "descuento"]) -> dict:
    """Recupera el valor ORIGINAL a partir del valor después de un aumento o un
    descuento (porcentaje inverso): divide entre el factor, no resta el porcentaje.

    Args:
        valor_final: el valor después del cambio, por ejemplo 92000.
        porcentaje: el porcentaje aplicado, por ejemplo 15.
        operacion: 'aumento' o 'descuento'.
    """
    factor = 1 + porcentaje / 100 if operacion == "aumento" else 1 - porcentaje / 100
    if factor == 0:
        raise ErrorHerramienta("un descuento del 100% no permite recuperar el valor original")
    return {**formatear(valor_final / factor), "factor": _decimal(factor)}


# --------------------------------------------------------------------------
# ESTADÍSTICA
# --------------------------------------------------------------------------

@REGISTRO.herramienta(familia="estadistica")
def promedio_ponderado(valores: list[float], pesos: list[float]) -> dict:
    """Calcula el promedio ponderado de unos valores con sus pesos. Los pesos
    pueden ser porcentajes (30, 70) o fracciones (0.3, 0.7).

    Args:
        valores: las notas o valores, por ejemplo [4.2, 3.5].
        pesos: el peso de cada valor, en el mismo orden, por ejemplo [30, 70].
    """
    if len(valores) != len(pesos):
        raise ErrorHerramienta("debe haber un peso por cada valor")
    total = sum(pesos, Fraction(0))
    if total == 0:
        raise ErrorHerramienta("los pesos no pueden sumar cero")
    return formatear(sum((v * p for v, p in zip(valores, pesos)), Fraction(0)) / total)


# --------------------------------------------------------------------------
# ÁLGEBRA
# --------------------------------------------------------------------------

@REGISTRO.herramienta(familia="algebra")
def resolver_ecuacion(ecuacion: str, incognita: str = "x") -> dict:
    """Resuelve una ecuación con una incógnita, por ejemplo '7x - 23 = 4x + 58'.

    Args:
        ecuacion: la ecuación con un signo '='.
        incognita: la letra de la incógnita (por defecto 'x').
    """
    import sympy
    if ecuacion.count("=") != 1:
        raise ErrorHerramienta("la ecuación debe tener exactamente un signo '='")
    if incognita not in _IDENTIFICADORES_OK - {"sqrt", "pi"}:
        raise ErrorHerramienta(f"incógnita no válida: {incognita!r}")
    izq, der = ecuacion.split("=")
    x = sympy.Symbol(incognita)
    soluciones = sympy.solve(sympy.Eq(parsear(izq), parsear(der)), x)
    if not soluciones:
        raise ErrorHerramienta("la ecuación no tiene solución (o se cumple para todo valor)")
    salidas = [_sympy_a_salida(s) for s in soluciones]
    if len(salidas) == 1:
        return {"resultado": f"{incognita} = {salidas[0]['resultado']}", "valor": salidas[0]["valor"]}
    return {"resultado": ", ".join(f"{incognita} = {s['resultado']}" for s in salidas),
            "valor": ";".join(s["valor"] for s in salidas)}


@REGISTRO.herramienta(familia="algebra")
def operar_expresion(expresion: str,
                     operacion: Literal["simplificar", "expandir", "factorizar"]) -> dict:
    """Simplifica, expande o factoriza una expresión algebraica, por ejemplo
    expandir '(2x + 3)(x - 5)'.

    Args:
        expresion: la expresión algebraica.
        operacion: 'simplificar', 'expandir' o 'factorizar'.
    """
    import sympy
    expr = parsear(expresion)
    resultado = {"simplificar": sympy.simplify, "expandir": sympy.expand,
                 "factorizar": sympy.factor}[operacion](expr)
    texto = sympy.sstr(resultado).replace("**", "^").replace("*", "")
    return {"resultado": texto, "valor": texto}
