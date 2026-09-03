"""
Celdas compartidas por los cinco notebooks.

Todo lo que debe ser idéntico entre arquitecturas (datos, particiones,
métricas, semillas, configuración de W&B) se define aquí una sola vez. Si el
código de métricas viviera copiado en cada notebook, cualquier ajuste posterior
rompería la comparabilidad del experimento sin que nos diéramos cuenta.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import textwrap
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
JSONL = RAIZ / "data" / "math_tutor_dataset.jsonl"

CATEGORIAS = [
    "suma", "resta", "multiplicacion", "division", "operaciones_combinadas",
    "potencias_raices", "fracciones", "porcentajes", "ecuaciones",
    "geometria", "estadistica_probabilidad",
]

PROYECTO_WANDB = "tutor-matematicas-arquitecturas"


# --------------------------------------------------------------------------
# Constructores de celdas
# --------------------------------------------------------------------------

def _fuente(texto: str) -> list[str]:
    texto = texto.strip("\n")
    return texto.splitlines(keepends=True) or [""]


def md(texto: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _fuente(texto)}


def code(texto: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _fuente(texto),
    }


def notebook(celdas: list[dict], nombre_corto: str) -> dict:
    # nbformat >= 4.5 exige un `id` por celda. Se asignan por posición para que
    # sean estables entre regeneraciones y los diffs sigan siendo legibles.
    for i, celda in enumerate(celdas):
        celda["id"] = f"celda-{i:03d}"
    return {
        "cells": celdas,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "name": nombre_corto},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------
# Datos embebidos
# --------------------------------------------------------------------------

def _blob() -> tuple[str, str]:
    crudo = JSONL.read_bytes()
    sha = hashlib.sha256(crudo).hexdigest()
    b64 = base64.b64encode(gzip.compress(crudo, mtime=0)).decode()
    return "\n".join(textwrap.wrap(b64, 96)), sha


def celda_datos(prefijo: str = "4") -> list[dict]:
    blob, sha = _blob()
    lineas = "\n".join(f'    "{l}"' for l in blob.split("\n"))
    return [
        md(f"""
### {prefijo}.1 · Materialización del corpus

El corpus vive en `data/math_tutor_dataset.jsonl`, generado por
`scripts/dataset_fuente.py`. Para que el notebook funcione en Colab sin subir
archivos, abajo va una **copia comprimida** de ese mismo archivo.

La celda **no sobrescribe** el JSONL si ya existe: eso permite escalar el
dataset (reemplazar el archivo por uno mayor) sin tocar el notebook. El hash
SHA-256 que se imprime debe ser idéntico en los cinco notebooks; si difiere,
alguno está entrenando con datos distintos y la comparación no sería válida.

Hash esperado de la versión embebida: `{sha[:16]}…`
"""),
        code(f'''
import base64, gzip, hashlib, json
from pathlib import Path

RUTA_DATOS = Path("data/math_tutor_dataset.jsonl")
SHA_ESPERADO = "{sha}"

_BLOB = (
{lineas}
)

if not RUTA_DATOS.exists():
    RUTA_DATOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_DATOS.write_bytes(gzip.decompress(base64.b64decode(_BLOB)))
    print(f"Corpus escrito en {{RUTA_DATOS}} (copia embebida).")
else:
    print(f"Se usará el corpus existente en {{RUTA_DATOS}}.")

sha_real = hashlib.sha256(RUTA_DATOS.read_bytes()).hexdigest()
print("SHA-256:", sha_real[:16], "…")
print("Coincide con la versión embebida:", sha_real == SHA_ESPERADO)
'''),
    ]


def celda_carga_datos(prefijo: str = "4") -> list[dict]:
    return [
        md(f"""
### {prefijo}.2 · Carga y particiones

Las particiones vienen **fijadas en el archivo** (campo `split`), no se
calculan aquí. Es una decisión deliberada: si cada notebook hiciera su propio
`train_test_split`, cuatro arquitecturas estarían evaluándose sobre conjuntos
distintos y las métricas no serían comparables entre sí.

La partición es estratificada por categoría (12 entrenamiento + 3 validación
por clase), generada con semilla 42.
"""),
        code('''
import json
from collections import Counter

registros = [json.loads(l) for l in RUTA_DATOS.read_text(encoding="utf-8").splitlines()]

train = [r for r in registros if r["split"] == "train"]
val   = [r for r in registros if r["split"] == "validation"]
demo  = [r for r in registros if r["es_demo"]]

CATEGORIAS = [
    "suma", "resta", "multiplicacion", "division", "operaciones_combinadas",
    "potencias_raices", "fracciones", "porcentajes", "ecuaciones",
    "geometria", "estadistica_probabilidad",
]
CAT2ID = {c: i for i, c in enumerate(CATEGORIAS)}
ID2CAT = {i: c for c, i in CAT2ID.items()}

print(f"Total: {len(registros)}  |  train: {len(train)}  |  validación: {len(val)}")
print(f"Ejemplos de demostración (todos en validación): {[d['id'] for d in demo]}")
print()
print("Distribución por categoría (train / val):")
ctr, cva = Counter(r["categoria"] for r in train), Counter(r["categoria"] for r in val)
for c in CATEGORIAS:
    print(f"  {c:26s} {ctr[c]:3d} / {cva[c]:2d}")
print()
print("Ejemplo completo:")
print(json.dumps(train[0], ensure_ascii=False, indent=2))
'''),
        md("""
Un ejemplo del corpus se ve así:

```
entrada : "María tiene 48 caramelos y quiere repartirlos por igual entre 6
           amigos. ¿Cuántos caramelos recibirá cada amigo?"
salida  : "Paso 1: Repartir en partes iguales es dividir.
           Paso 2: 48 ÷ 6 = 8.
           Respuesta final: 8 caramelos"
valor   : "8"
```

El campo `valor` es la clave de toda la evaluación automática: es la respuesta
en forma canónica (sin unidades ni texto). Comparar `valor` contra lo que el
modelo escribe después de `Respuesta final:` nos da una métrica objetiva de
**si el modelo resolvió bien el problema**, independiente de cómo lo redactó.
"""),
    ]


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------

CODIGO_METRICAS = '''
import re
import unicodedata
from fractions import Fraction

MARCADOR_RESPUESTA = "Respuesta final:"

_PAT_RESPUESTA = re.compile(r"Respuesta\\s+final\\s*:\\s*(.+)", re.IGNORECASE)
_PAT_PASO1     = re.compile(r"Paso\\s*1\\s*:", re.IGNORECASE)
_PAT_CATEGORIA = re.compile(r"Categor[ií]a\\s*:\\s*([a-zA-Z_]+)", re.IGNORECASE)
# La alternativa de fracción va primero: en "3/5" queremos capturar la fracción
# completa, no el "3" suelto.
_PAT_VALOR = re.compile(r"-?\\d+(?:\\.\\d+)?\\s*/\\s*-?\\d+(?:\\.\\d+)?|-?\\d+(?:\\.\\d+)?")


def _sin_miles(texto):
    """Quita la coma como separador de miles. El corpus usa el punto como
    separador decimal y nunca la coma, así que la conversión no es ambigua."""
    return texto.replace(",", "")


def a_float(valor):
    if valor is None:
        return None
    try:
        return float(Fraction(valor)) if "/" in valor else float(valor)
    except (ValueError, ZeroDivisionError):
        return None


def valor_predicho(texto):
    """Extrae la respuesta del modelo en forma canónica.

    Prioridad 1: el número que sigue al marcador 'Respuesta final:'.
    Prioridad 2: el último número del texto.

    El segundo caso importa para que la comparación sea JUSTA: un modelo sin
    fine-tuning no conoce nuestro formato, y penalizarlo por eso mediría
    obediencia al formato, no capacidad matemática. Con el fallback medimos lo
    segundo; el apego al formato se mide aparte con `formato_valido`.
    """
    m = _PAT_RESPUESTA.search(texto)
    if m:
        linea = m.group(1).splitlines()[0]
        v = _PAT_VALOR.search(_sin_miles(linea))
        if v:
            return v.group(0).replace(" ", "")
    todos = _PAT_VALOR.findall(_sin_miles(texto))
    return todos[-1].replace(" ", "") if todos else None


def respuesta_correcta(generado, valor_oro, tol=1e-6):
    a = a_float(valor_predicho(generado))
    b = a_float(valor_oro)
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def formato_valido(texto):
    """¿El modelo produjo la estructura que le enseñamos?"""
    return bool(_PAT_PASO1.search(texto)) and bool(_PAT_RESPUESTA.search(texto))


def categoria_predicha(texto):
    m = _PAT_CATEGORIA.search(texto)
    return m.group(1).lower() if m else None


def _tokens(texto):
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+|[^\\sa-z0-9]", t)


def _lcs(a, b):
    """Longitud de la subsecuencia común más larga (programación dinámica)."""
    previa = [0] * (len(b) + 1)
    for x in a:
        actual = [0]
        for j, y in enumerate(b):
            actual.append(previa[j] + 1 if x == y else max(previa[j + 1], actual[j]))
        previa = actual
    return previa[-1]


def rouge_l(generado, referencia):
    """ROUGE-L (F1 sobre la subsecuencia común más larga).

    Se implementa a mano en lugar de usar `evaluate` para que el notebook no
    dependa de descargas en tiempo de ejecución y el número sea exactamente
    reproducible.
    """
    p, r = _tokens(generado), _tokens(referencia)
    if not p or not r:
        return 0.0
    l = _lcs(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)


def evaluar_generacion(generados, registros):
    """Métricas de generación sobre un conjunto de ejemplos.

    exactitud       : ¿la respuesta final es numéricamente correcta?  <- la que importa
    formato_valido  : ¿respetó la estructura Paso N / Respuesta final?
    rouge_l         : ¿se parece el procedimiento al de referencia?
    long_media      : longitud media en palabras (detecta divagación)
    """
    assert len(generados) == len(registros)
    n = len(generados)
    correctas = [respuesta_correcta(g, r["valor"]) for g, r in zip(generados, registros)]
    formatos  = [formato_valido(g) for g in generados]
    rouges    = [rouge_l(g, r["salida"]) for g, r in zip(generados, registros)]
    return {
        "n": n,
        "exactitud": sum(correctas) / n,
        "formato_valido": sum(formatos) / n,
        "rouge_l": sum(rouges) / n,
        "long_media": sum(len(g.split()) for g in generados) / n,
        "_correctas": correctas,
    }


def tabla_metricas(antes, despues, titulo="Baseline vs Fine-tuned"):
    """Imprime la comparación en el formato que usaremos en el informe."""
    filas = [
        ("Exactitud de la respuesta", "exactitud", "{:.1%}"),
        ("Formato válido",            "formato_valido", "{:.1%}"),
        ("ROUGE-L del procedimiento", "rouge_l", "{:.3f}"),
        ("Longitud media (palabras)", "long_media", "{:.1f}"),
    ]
    ancho = 30
    print(titulo)
    print("=" * 68)
    print(f"{'Métrica':{ancho}s} {'Baseline':>12s} {'Fine-tuned':>12s} {'Δ':>10s}")
    print("-" * 68)
    for etiqueta, clave, fmt in filas:
        a, d = antes[clave], despues[clave]
        print(f"{etiqueta:{ancho}s} {fmt.format(a):>12s} {fmt.format(d):>12s} "
              f"{d - a:>+10.3f}")
    print("=" * 68)
'''


def celda_metricas() -> list[dict]:
    return [
        md("""
### Métricas de generación

Antes de tocar el modelo definimos **cómo vamos a medirlo**. Fijar las métricas
antes de ver resultados evita el sesgo de elegir después la métrica que mejor
nos deja.

| Métrica | Qué mide | Por qué está |
|---|---|---|
| **Exactitud de la respuesta** | ¿El número final es correcto? | Es la única métrica que responde "¿sirve como tutor?". Un procedimiento bonito con resultado equivocado es un fracaso. |
| **Formato válido** | ¿Usó `Paso N:` y `Respuesta final:`? | Separa *aprender a resolver* de *aprender a formatear*. El fine-tuning con pocos datos suele mejorar mucho lo segundo y poco lo primero; sin esta métrica confundiríamos ambos efectos. |
| **ROUGE-L** | Solapamiento de la explicación con la de referencia | Aproxima si el procedimiento se parece al esperado. Es una métrica débil (premia coincidencia léxica, no razonamiento) y así hay que leerla. |
| **Longitud media** | Palabras generadas | Detecta el modo de fallo típico del baseline: divagar o repetirse hasta agotar `max_new_tokens`. |

Nota metodológica importante: para extraer la respuesta del modelo usamos el
marcador `Respuesta final:` y, si no aparece, **el último número del texto**.
Sin ese respaldo estaríamos castigando al baseline por desconocer un formato
que todavía no le hemos enseñado, y la mejora del fine-tuning se vería
artificialmente enorme.
"""),
        code(CODIGO_METRICAS),
    ]


# --------------------------------------------------------------------------
# Entorno, W&B, resultados
# --------------------------------------------------------------------------

def celda_instalacion() -> list[dict]:
    return [
        md("""
## 0 · Preparación del entorno

Instalamos el ecosistema Hugging Face. Las versiones se fijan por rango mayor
para evitar que un cambio de API rompa el notebook meses después.

> **Antes de empezar:** activen la GPU en `Entorno de ejecución → Cambiar tipo
> de entorno de ejecución → T4 GPU`. Sin GPU el entrenamiento es inviable.

**Sobre la desinstalación de `torchao`.** Colab trae `torchao` preinstalado, y
`peft` comprueba su versión con una función que **lanza `ImportError` en lugar
de devolver `False`** cuando la encuentra más antigua de lo que espera. El
resultado es que `get_peft_model()` falla con un error que no tiene ninguna
relación aparente con LoRA.

Ningún notebook del proyecto usa cuantización de torchao, así que lo quitamos.
La alternativa —actualizarlo— también funcionaría, pero torchao está acoplado a
la versión de torch y actualizarlo puede arrastrar un torch distinto y romper
otras cosas en Colab. Desinstalarlo no afecta a nada de lo que hacemos aquí.
"""),
        code('''
%pip install -q "transformers>=4.44" "datasets>=2.20" "peft>=0.12" \\
    "accelerate>=0.33" "bitsandbytes>=0.43" "scikit-learn>=1.3" wandb

# Ver la nota de arriba: evita que get_peft_model() falle con un ImportError
# de torchao que nada tiene que ver con LoRA.
%pip uninstall -y -q torchao

print("Librerías instaladas. Si Colab pide reiniciar la sesión, reinícienla y sigan desde aquí.")
'''),
        code('''
import os, random, sys
import numpy as np
import torch
import transformers

SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)
transformers.set_seed(SEMILLA)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"transformers  {transformers.__version__}")
print(f"torch         {torch.__version__}")
print(f"Dispositivo   {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU           {torch.cuda.get_device_name(0)}")
else:
    print("AVISO: sin GPU el fine-tuning tardará horas. Activen el runtime T4.")

# Detección temprana del conflicto peft/torchao. Más vale que salte aquí, en la
# celda de entorno, que dentro de get_peft_model() veinte celdas más adelante
# con un mensaje que no menciona LoRA por ninguna parte.
try:
    from peft.import_utils import is_torchao_available
except Exception:
    pass                       # ruta interna de peft cambiada: no es un problema
else:
    try:
        is_torchao_available()
    except ImportError as e:
        print(f"\\nAVISO peft/torchao: {e}")
        print("  Solución: ejecuten  %pip uninstall -y torchao  y reinicien la sesión.")
'''),
    ]


def celda_drive() -> list[dict]:
    return [
        md("""
### Persistencia entre notebooks

Los notebooks 3 y 5 **leen carpetas que producen los notebooks 1, 2 y 4**:

```
1 · Qwen    -> adaptadores/qwen-lora/      ─┐
2 · BERT    -> modelos/bert-clasificador/  ─┤-> el notebook 3 las lee
4 · FLAN-T5 -> adaptadores/flan-t5-lora/    │
1,2,3,4     -> resultados/*.json           ─┴-> el notebook 5 los lee
```

En Colab, `/content` se borra al desconectar el runtime, así que ese trabajo se
perdería entre sesiones. Montando Google Drive y trabajando desde una carpeta
suya, los artefactos sobreviven y cada notebook se puede ejecutar el día que se
pueda.

Con `USAR_DRIVE = False` todo queda en `/content`, lo cual es válido si
ejecutan los notebooks 1, 2 y 3 seguidos sin desconectar.

Fuera de Colab la celda no hace nada: el directorio de trabajo se queda como
está.

> **Los checkpoints intermedios nunca van a Drive.** El `Trainer` guarda en
> `output_dir` el modelo *más el estado del optimizador* en cada época. Para el
> notebook 2, que hace fine-tuning completo de BETO, eso son varios GB que
> además se escribirían por red. Como son desechables —lo que importa es el
> modelo final—, se mandan siempre al disco local del runtime mediante
> `DIR_CHECKPOINTS`.
"""),
        code('''
import os
from pathlib import Path

USAR_DRIVE    = True
CARPETA_DRIVE = "/content/drive/MyDrive/ProyectoIA"

try:
    import google.colab  # noqa: F401
    EN_COLAB = True
except ImportError:
    EN_COLAB = False

if EN_COLAB and USAR_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    Path(CARPETA_DRIVE).mkdir(parents=True, exist_ok=True)
    os.chdir(CARPETA_DRIVE)

# Checkpoints del Trainer: grandes y desechables -> siempre en disco local.
DIR_CHECKPOINTS = "/content/salidas" if EN_COLAB else "salidas"

print(f"En Colab           : {EN_COLAB}")
print(f"Directorio de trabajo: {Path.cwd()}")
print(f"Checkpoints en     : {DIR_CHECKPOINTS}")
'''),
    ]


def celda_wandb(nombre_run: str, notas: str) -> list[dict]:
    return [
        md(f"""
### Weights & Biases

W&B registra automáticamente la curva de pérdida, los hiperparámetros y el
consumo de GPU. Es lo que después nos permitirá comparar las cuatro
arquitecturas sobre los mismos ejes en lugar de sobre capturas de pantalla.

Convención de nombres del proyecto (idéntica en los cinco notebooks):

- **Proyecto:** `{PROYECTO_WANDB}`
- **Run:** `{nombre_run}`
- **Tags:** identifican arquitectura y fase, para poder filtrar en el panel

Si no quieren usar W&B, pongan `USAR_WANDB = False`: el notebook seguirá
funcionando y las métricas se guardarán igual en `resultados/`.
"""),
        code(f'''
import os

USAR_WANDB   = True          # ponlo en False para trabajar sin conexión a W&B
PROYECTO     = "{PROYECTO_WANDB}"
NOMBRE_RUN   = "{nombre_run}"
TAGS         = {notas}

if USAR_WANDB:
    import wandb
    wandb.login()            # pedirá la API key la primera vez
    os.environ["WANDB_PROJECT"] = PROYECTO
    os.environ["WANDB_LOG_MODEL"] = "false"
    REPORTAR_A = "wandb"
else:
    os.environ["WANDB_MODE"] = "disabled"
    REPORTAR_A = "none"

print(f"Registro de experimentos: {{REPORTAR_A}}  |  run: {{NOMBRE_RUN}}")
'''),
    ]


CODIGO_RESULTADOS = '''
import json
from pathlib import Path

DIR_RESULTADOS = Path("resultados")
DIR_RESULTADOS.mkdir(exist_ok=True)


def guardar_resultados(nombre, payload):
    """Persiste las métricas para que el notebook de comparación las agregue.

    Sin este paso, comparar las cuatro arquitecturas obligaría a re-ejecutar
    todo en una sola sesión. Con él, cada notebook se ejecuta cuando se pueda y
    la comparación se hace al final leyendo los JSON.
    """
    limpio = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            limpio[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        elif not k.startswith("_"):
            limpio[k] = v
    ruta = DIR_RESULTADOS / f"{nombre}.json"
    ruta.write_text(json.dumps(limpio, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Métricas guardadas en {ruta}")
    return ruta
'''


def celda_resultados() -> list[dict]:
    return [code(CODIGO_RESULTADOS)]


# --------------------------------------------------------------------------
# Utilidad compartida: TrainingArguments tolerante a versiones
# --------------------------------------------------------------------------

CODIGO_ARGS_COMPATIBLES = '''
import inspect
from transformers import TrainingArguments


def construir_args(clase=TrainingArguments, **kwargs):
    """Crea TrainingArguments filtrando los parámetros que la versión instalada
    de `transformers` no reconoce.

    Motivo: entre versiones recientes `evaluation_strategy` pasó a llamarse
    `eval_strategy`. Sin esta capa, el notebook funciona hoy y falla el
    semestre que viene. Se prefiere `eval_strategy` y se traduce si hace falta.
    """
    admitidos = set(inspect.signature(clase.__init__).parameters)
    if "eval_strategy" in kwargs and "eval_strategy" not in admitidos:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    descartados = [k for k in kwargs if k not in admitidos]
    for k in descartados:
        kwargs.pop(k)
    if descartados:
        print(f"Parámetros no soportados por esta versión, se omiten: {descartados}")
    return clase(**kwargs)
'''


def celda_args_compatibles() -> list[dict]:
    return [code(CODIGO_ARGS_COMPATIBLES)]


# --------------------------------------------------------------------------
# Encabezado común
# --------------------------------------------------------------------------

def encabezado(titulo: str, subtitulo: str) -> dict:
    return md(f"""
# {titulo}

### {subtitulo}

**Proyecto:** Tutor inteligente de matemáticas · Comparación de arquitecturas
mediante fine-tuning
**Curso:** SI4006 · Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT
**Notebook base:** `S04_Lab_Fine_tuning_Qwen.ipynb`

---
""")
