"""
Construcción, validación y exportación del dataset del tutor de matemáticas.

Este script es la ÚNICA fuente de verdad del corpus. Los notebooks no editan
datos: consumen el archivo `data/math_tutor_dataset.jsonl` que produce este
script. Esa separación garantiza que las cuatro arquitecturas se entrenen y
evalúen exactamente sobre las mismas particiones.

Uso:
    python dataset_fuente.py            # valida y escribe el JSONL
    python dataset_fuente.py --check    # solo valida, no escribe

Buenas prácticas de curación aplicadas (Hugging Face Datasets, Meta AI Dataset
Curation, guías de preparación de datos para fine-tuning):

1. FORMATO UNIFORME    La salida se genera mecánicamente desde `pasos` +
                       `respuesta_final`. Ningún ejemplo puede desviarse del
                       formato porque nadie lo escribe a mano.
2. CONSISTENCIA        Un único esquema de campos, una única plantilla de
                       prompt, un único marcador de respuesta final
                       ("Respuesta final:") que además hace la evaluación
                       automática posible.
3. REPRESENTATIVIDAD   11 categorías balanceadas a 15 ejemplos cada una, con
                       mezcla deliberada de problemas en lenguaje natural y
                       operaciones directas.
4. DIVERSIDAD          Variación de contextos (dinero, distancias, personas,
                       objetos), de nivel (básico/intermedio) y de longitud.
5. LIMPIEZA            Verificación aritmética automática de cada ejemplo,
                       deduplicación exacta y normalizada, y separación de los
                       ítems originales que hacían dos preguntas a la vez.
6. TRAZABILIDAD        Cada ejemplo declara su `origen` ("base" = uno de los 50
                       ejemplos originales, "nuevo" = añadido en la curación).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registros import REGISTROS  # noqa: E402

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent
SALIDA_JSONL = RAIZ / "data" / "math_tutor_dataset.jsonl"

SEMILLA = 42
EJEMPLOS_VALIDACION_POR_CLASE = 3
TOLERANCIA = 1e-6

CATEGORIAS = [
    "suma",
    "resta",
    "multiplicacion",
    "division",
    "operaciones_combinadas",
    "potencias_raices",
    "fracciones",
    "porcentajes",
    "ecuaciones",
    "geometria",
    "estadistica_probabilidad",
]

TIPOS = {"problema", "operacion"}
NIVELES = {"basico", "intermedio"}
ORIGENES = {"base", "nuevo"}

MARCADOR_RESPUESTA = "Respuesta final:"
INSTRUCCION = (
    "Eres un tutor de matemáticas. Resuelve el siguiente problema explicando "
    "el procedimiento paso a paso y termina con la respuesta final."
)

# Ejemplos fijos para la comparación cualitativa baseline vs fine-tuned.
# DEBEN pertenecer a la partición de validación: si el modelo ya los vio
# durante el entrenamiento, la comparación cualitativa mide memorización, no
# aprendizaje. `validar()` lo verifica explícitamente.
IDS_DEMO = ["div-09", "por-09", "fra-12", "ecu-08", "geo-06"]


# --------------------------------------------------------------------------
# Construcción del texto
# --------------------------------------------------------------------------

def construir_salida(reg: dict) -> str:
    """Genera la salida canónica: pasos numerados + marcador de respuesta.

    El formato es idéntico para los 165 ejemplos. Esto le da al modelo una
    señal de formato inequívoca y, sobre todo, hace que la respuesta final sea
    extraíble con una expresión regular durante la evaluación.
    """
    lineas = [f"Paso {i}: {p}" for i, p in enumerate(reg["pasos"], start=1)]
    lineas.append(f"{MARCADOR_RESPUESTA} {reg['respuesta_final']}")
    return "\n".join(lineas)


def a_registro_final(reg: dict, split: str) -> dict:
    return {
        "id": reg["id"],
        "split": split,
        "categoria": reg["categoria"],
        "categoria_id": CATEGORIAS.index(reg["categoria"]),
        "tipo": reg["tipo"],
        "nivel": reg["nivel"],
        "origen": reg["origen"],
        "entrada": reg["entrada"],
        "salida": construir_salida(reg),
        "pasos": reg["pasos"],
        "respuesta_final": reg["respuesta_final"],
        "valor": reg["valor"],
        "es_demo": reg["id"] in IDS_DEMO,
    }


# --------------------------------------------------------------------------
# Particiones
# --------------------------------------------------------------------------

def asignar_splits(registros: list[dict]) -> dict[str, str]:
    """Split estratificado y determinista: n ejemplos de validación por clase.

    Estratificar por categoría es obligatorio aquí: con 15 ejemplos por clase,
    un split aleatorio simple puede dejar clases sin representación en
    validación y volver el macro-F1 indefinido.
    """
    rng = random.Random(SEMILLA)
    por_categoria = defaultdict(list)
    for r in registros:
        por_categoria[r["categoria"]].append(r["id"])

    splits = {}
    for categoria in CATEGORIAS:
        ids = sorted(por_categoria[categoria])
        rng.shuffle(ids)
        for i, id_ in enumerate(ids):
            splits[id_] = "validation" if i < EJEMPLOS_VALIDACION_POR_CLASE else "train"
    return splits


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------

def a_numero(texto: str) -> float | None:
    """Convierte '3/5', '-3', '4.125' a float. Devuelve None si no aplica."""
    t = texto.strip().replace(",", "")
    try:
        if "/" in t:
            return float(Fraction(t))
        return float(t)
    except (ValueError, ZeroDivisionError):
        return None


def normalizar_texto(texto: str) -> str:
    """Minúsculas, sin tildes ni puntuación: base para detectar duplicados."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def validar(registros: list[dict]) -> list[str]:
    errores: list[str] = []

    vistos_id: set[str] = set()
    vistos_entrada: dict[str, str] = {}

    for r in registros:
        rid = r["id"]

        if rid in vistos_id:
            errores.append(f"[{rid}] id duplicado")
        vistos_id.add(rid)

        if r["categoria"] not in CATEGORIAS:
            errores.append(f"[{rid}] categoría desconocida: {r['categoria']}")
        if r["tipo"] not in TIPOS:
            errores.append(f"[{rid}] tipo inválido: {r['tipo']}")
        if r["nivel"] not in NIVELES:
            errores.append(f"[{rid}] nivel inválido: {r['nivel']}")
        if r["origen"] not in ORIGENES:
            errores.append(f"[{rid}] origen inválido: {r['origen']}")

        clave = normalizar_texto(r["entrada"])
        if clave in vistos_entrada:
            errores.append(f"[{rid}] entrada duplicada respecto a {vistos_entrada[clave]}")
        vistos_entrada[clave] = rid

        if not (1 <= len(r["pasos"]) <= 4):
            errores.append(f"[{rid}] debe tener entre 1 y 4 pasos, tiene {len(r['pasos'])}")
        for i, paso in enumerate(r["pasos"], start=1):
            if not paso.strip().endswith("."):
                errores.append(f"[{rid}] el paso {i} no termina en punto")
            if len(paso.split()) < 3:
                errores.append(f"[{rid}] el paso {i} es demasiado corto")

        if not r["entrada"].strip().endswith(("?", ".", ":")):
            errores.append(f"[{rid}] la entrada no termina en signo de cierre")

        if r["valor"] != r["valor"].strip():
            errores.append(f"[{rid}] 'valor' tiene espacios sobrantes")

        # Verificación aritmética: la red de seguridad del corpus.
        if r["check"]:
            esperado = eval(r["check"], {"__builtins__": {}}, {})  # noqa: S307
            obtenido = a_numero(r["valor"])
            if obtenido is None:
                errores.append(f"[{rid}] 'valor' no es numérico pero tiene check")
            elif abs(float(esperado) - obtenido) > TOLERANCIA:
                errores.append(
                    f"[{rid}] ARITMÉTICA: check={r['check']} da {esperado} "
                    f"pero valor={r['valor']}"
                )

        # La respuesta final legible debe contener el valor canónico.
        if r["valor"] not in r["respuesta_final"]:
            errores.append(
                f"[{rid}] 'valor' ({r['valor']}) no aparece en "
                f"'respuesta_final' ({r['respuesta_final']})"
            )

    conteo = Counter(r["categoria"] for r in registros)
    faltantes = [c for c in CATEGORIAS if c not in conteo]
    if faltantes:
        errores.append(f"categorías sin ejemplos: {faltantes}")
    if len(set(conteo.values())) > 1:
        errores.append(f"corpus desbalanceado: {dict(conteo)}")

    ids = {r["id"] for r in registros}
    splits = asignar_splits(registros)
    for demo in IDS_DEMO:
        if demo not in ids:
            errores.append(f"el id de demostración {demo} no existe")
        elif splits[demo] != "validation":
            errores.append(
                f"el id de demostración {demo} está en train: la comparación "
                f"cualitativa mediría memorización, no generalización"
            )

    return errores


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------

def reporte(finales: list[dict]) -> str:
    lineas = ["", "=" * 62, "REPORTE DE CALIDAD DEL DATASET", "=" * 62]
    lineas.append(f"Total de ejemplos: {len(finales)}")

    for campo in ("split", "tipo", "nivel", "origen"):
        conteo = Counter(f[campo] for f in finales)
        detalle = "  ".join(f"{k}={v}" for k, v in sorted(conteo.items()))
        lineas.append(f"{campo.capitalize():10s}: {detalle}")

    lineas.append("")
    lineas.append(f"{'categoría':26s} {'train':>6s} {'val':>5s} {'total':>6s}")
    lineas.append("-" * 46)
    for categoria in CATEGORIAS:
        tr = sum(1 for f in finales if f["categoria"] == categoria and f["split"] == "train")
        va = sum(1 for f in finales if f["categoria"] == categoria and f["split"] == "validation")
        lineas.append(f"{categoria:26s} {tr:6d} {va:5d} {tr + va:6d}")

    palabras_in = [len(f["entrada"].split()) for f in finales]
    palabras_out = [len(f["salida"].split()) for f in finales]
    lineas.append("")
    lineas.append(
        f"Palabras entrada  min={min(palabras_in)}  media="
        f"{sum(palabras_in) / len(palabras_in):.1f}  max={max(palabras_in)}"
    )
    lineas.append(
        f"Palabras salida   min={min(palabras_out)}  media="
        f"{sum(palabras_out) / len(palabras_out):.1f}  max={max(palabras_out)}"
    )
    lineas.append("=" * 62)
    return "\n".join(lineas)


# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------

def construir() -> list[dict]:
    errores = validar(REGISTROS)
    if errores:
        print("VALIDACIÓN FALLIDA:", file=sys.stderr)
        for e in errores:
            print("  -", e, file=sys.stderr)
        raise SystemExit(1)

    splits = asignar_splits(REGISTROS)
    return [a_registro_final(r, splits[r["id"]]) for r in REGISTROS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="solo validar")
    args = parser.parse_args()

    finales = construir()
    print(reporte(finales))

    if args.check:
        print("\nValidación superada (no se escribió ningún archivo).")
        return

    SALIDA_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA_JSONL.open("w", encoding="utf-8") as fh:
        for r in finales:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nEscrito: {SALIDA_JSONL}  ({len(finales)} líneas)")


if __name__ == "__main__":
    main()
