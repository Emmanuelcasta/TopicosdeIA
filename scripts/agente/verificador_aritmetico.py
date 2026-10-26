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


def igualdades_de(paso: str) -> list:
    """[(izquierda, derecha)] de las igualdades escritas en un paso."""
    limpio = _PREFIJO_PASO.sub("", paso or "")
    return [(m.group(1), m.group(2)) for m in _IGUALDAD.finditer(limpio)]


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
