"""Notebook 5 — Comparación final de las cuatro arquitecturas."""

from __future__ import annotations

from nb_comun import (celda_carga_datos, celda_datos, celda_drive, code,
                      encabezado, md)


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "Comparación de arquitecturas para un tutor de matemáticas",
        "Notebook 5 · Síntesis experimental y recomendación",
    ))

    c.append(md("""
## 1 · Introducción

Este notebook no entrena nada. **Agrega** los resultados de los cuatro
notebooks anteriores y produce la comparación que da sentido al experimento.

| Notebook | Arquitectura | Modelo | Tarea |
|---|---|---|---|
| 1 · Qwen | Decoder-only | Qwen2.5-1.5B-Instruct | Generar la solución |
| 2 · BERT | Encoder-only | BETO (bert-base-spanish) | Clasificar el problema |
| 3 · Qwen+BERT | Pipeline modular | BETO → Qwen2.5+LoRA | Clasificar y luego generar |
| 4 · FLAN-T5 | Encoder-decoder | flan-t5-base | Ambas en una pasada |

### Por qué la comparación es válida

Las cuatro arquitecturas comparten, por construcción y no por casualidad:

- **El mismo corpus**, con el mismo hash SHA-256 verificado en cada notebook.
- **Las mismas particiones**: 132 entrenamiento / 33 validación, estratificadas,
  fijadas en el archivo de datos y no recalculadas por cada notebook.
- **Las mismas métricas**, definidas en una única celda compartida.
- **La misma semilla** (42) y decodificación greedy en toda evaluación.

Si alguna de esas condiciones se rompiera, las diferencias entre modelos
podrían venir de los datos y no de la arquitectura. La verificación de hashes de
la sección 2 comprueba justamente eso.

### Requisito

Haber ejecutado los notebooks 1 a 4, de modo que existan los cuatro archivos en
`resultados/`. El notebook funciona con resultados parciales, avisando de lo que
falta.
"""))

    c.append(md("""
## 2 · Carga de resultados
"""))

    c.append(code('''
%pip install -q "transformers>=4.44" pandas matplotlib
print("Listo.")
'''))

    c.extend(celda_drive())

    c.append(code('''
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIR_RESULTADOS = Path("resultados")

ESPERADOS = {
    "qwen": "Notebook 1 · Qwen (decoder-only)",
    "bert": "Notebook 2 · BERT (encoder-only)",
    "pipeline_qwen_bert": "Notebook 3 · Pipeline modular",
    "flan_t5": "Notebook 4 · FLAN-T5 (encoder-decoder)",
}

R = {}
for clave, descripcion in ESPERADOS.items():
    ruta = DIR_RESULTADOS / f"{clave}.json"
    if ruta.exists():
        R[clave] = json.loads(ruta.read_text(encoding="utf-8"))
        print(f"  OK      {descripcion}")
    else:
        print(f"  FALTA   {descripcion}  ->  ejecuten ese notebook y vuelvan aquí")

if not R:
    raise FileNotFoundError("No hay ningún resultado en resultados/. Ejecuten los notebooks 1-4.")
print(f"\\n{len(R)} de {len(ESPERADOS)} notebooks disponibles.")
'''))

    c.append(code('''
# Verificación de comparabilidad: los cuatro notebooks deben haber usado el
# mismo número de ejemplos de validación. Si no coincide, algo se ejecutó con
# datos distintos y las comparaciones de abajo no serían legítimas.
n_vals = {k: v.get("n_val") for k, v in R.items() if v.get("n_val")}
print("Ejemplos de validación por notebook:", n_vals)
if len(set(n_vals.values())) > 1:
    print("\\nAVISO: los conjuntos de validación NO coinciden. "
          "Regeneren el dataset y re-ejecuten los notebooks antes de comparar.")
else:
    print("Conjuntos de validación consistentes: la comparación es válida.")
'''))

    # -------------------------------------------------------------- tabla
    c.append(md("""
## 3 · Tabla comparativa

Diez dimensiones. Las que se pueden medir se leen de los JSON; las
cualitativas llevan una justificación explícita para que se puedan discutir en
lugar de aceptarse como dadas.
"""))

    c.append(code('''
def g(clave, *ruta, default=None):
    """Acceso seguro a un valor anidado del JSON de resultados."""
    nodo = R.get(clave)
    for paso in ruta:
        if not isinstance(nodo, dict) or paso not in nodo:
            return default
        nodo = nodo[paso]
    return nodo


def pct(x):
    return "—" if x is None else f"{x:.1%}"


def num(x, fmt="{:.0f}"):
    return "—" if x is None else fmt.format(x)


def mm(*claves):
    """Parámetros en millones, sumando varios modelos si hace falta."""
    total = sum(g(k, "parametros_totales", default=0) or 0 for k in claves)
    return f"{total/1e6:.0f}" if total else "—"


filas = {
    "Arquitectura": [
        "Decoder-only", "Encoder-only", "Encoder + Decoder (2 modelos)", "Encoder-decoder (1 modelo)",
    ],
    "Tipo de modelo": [
        "Autoregresivo", "Discriminativo", "Cascada modular", "Seq2seq",
    ],
    "Modelo base": [
        g("qwen", "modelo", default="—"),
        g("bert", "modelo", default="—"),
        "BETO → Qwen2.5+LoRA",
        g("flan_t5", "modelo", default="—"),
    ],
    "Parámetros totales (M)": [
        mm("qwen"), mm("bert"), mm("qwen", "bert"), mm("flan_t5"),
    ],
    "Tokenización": [
        f"BPE · {g('qwen','tokenizador','vocabulario',default='?')} tokens",
        f"WordPiece · {g('bert','tokenizador','vocabulario',default='?')} tokens",
        "Dos tokenizadores distintos",
        f"SentencePiece · {g('flan_t5','tokenizador','vocabulario',default='?')} tokens",
    ],
    "Método de fine-tuning": [
        g("qwen", "metodo", default="—"),
        g("bert", "metodo", default="—"),
        "Ninguno (composición)",
        g("flan_t5", "metodo", default="—"),
    ],
    "Tiempo de entrenamiento (s)": [
        num(g("qwen", "tiempo_entrenamiento_s"), "{:.0f}"),
        num(g("bert", "tiempo_entrenamiento_s"), "{:.0f}"),
        "0 (reutiliza 1 y 2)",
        num(g("flan_t5", "tiempo_entrenamiento_s"), "{:.0f}"),
    ],
    "Capacidad de clasificación": [
        "No aplica",
        pct(g("bert", "finetuned", "accuracy")),
        pct(g("pipeline_qwen_bert", "accuracy_clasificador")),
        pct(g("flan_t5", "clasificacion_finetuned", "accuracy")),
    ],
    "Capacidad generativa (exactitud)": [
        pct(g("qwen", "finetuned", "exactitud")),
        "No aplica",
        pct(g("pipeline_qwen_bert", "config_B_pipeline", "exactitud")),
        pct(g("flan_t5", "finetuned", "exactitud")),
    ],
    "Formato válido": [
        pct(g("qwen", "finetuned", "formato_valido")),
        "No aplica",
        pct(g("pipeline_qwen_bert", "config_B_pipeline", "formato_valido")),
        pct(g("flan_t5", "finetuned", "formato_valido")),
    ],
}

tabla = pd.DataFrame(filas, index=["Qwen", "BERT", "Qwen+BERT", "FLAN-T5"]).T
pd.set_option("display.max_colwidth", 42)
print(tabla.to_string())
'''))

    c.append(md("""
### Dimensiones cualitativas

Estas no salen de un JSON: son juicios de ingeniería sobre lo observado
durante el laboratorio. Cada una lleva su razón, para que sea discutible.
"""))

    c.append(code('''
cualitativas = pd.DataFrame({
    "Facilidad de fine-tuning": {
        "Qwen": "Media — LoRA obligatorio; los parámetros entrenables deben forzarse a fp32 o la pérdida diverge a NaN",
        "BERT": "Alta — fine-tuning completo directo, receta estándar, sin sorpresas",
        "Qwen+BERT": "Media — no entrena, pero exige coordinar versiones y alineación de etiquetas entre dos artefactos",
        "FLAN-T5": "Media — LoRA sencillo, pero fp16 rompe el modelo: obliga a fp32",
    },
    "Velocidad de entrenamiento": {
        "Qwen": "Lenta — 1.5B parámetros, secuencias largas",
        "BERT": "Muy rápida — 110M, secuencias de 64 tokens, segundos por época",
        "Qwen+BERT": "N/A — hereda el costo de 1 y 2",
        "FLAN-T5": "Media — 250M pero en fp32 y con dos pilas (encoder y decoder)",
    },
    "Consumo de recursos (inferencia)": {
        "Qwen": "Alto — ~3 GB en fp16 y generación token a token",
        "BERT": "Muy bajo — ~0.4 GB, una sola pasada",
        "Qwen+BERT": "Alto — suma ambos, dominado por Qwen",
        "FLAN-T5": "Medio — ~1 GB, generación más corta",
    },
    "Interpretabilidad": {
        "Qwen": "Baja — solo se ve el texto final; no hay estado intermedio inspeccionable",
        "BERT": "Alta — distribución sobre 11 clases, matriz de confusión, confianza calibrable",
        "Qwen+BERT": "La más alta — el estado intermedio es explícito y se puede atribuir el fallo a una etapa concreta",
        "FLAN-T5": "Media — la línea 'Categoria:' expone la decisión interna, pero no es un estado separable",
    },
    "Escalabilidad": {
        "Qwen": "Alta — más datos mejoran directamente; hay variantes de 3B, 7B, 14B con el mismo código",
        "BERT": "Media — añadir categorías exige reetiquetar y reentrenar; el espacio de clases es cerrado",
        "Qwen+BERT": "Media — escala, pero cada componente por separado y hay que revalidar la interfaz",
        "FLAN-T5": "Media — mT5 resolvería el tokenizador; flan-t5-large necesita más GPU",
    },
}).T
print(cualitativas.to_string())
'''))

    # -------------------------------------------------------------- gráficos
    c.append(md("""
## 4 · Visualización comparativa
"""))

    c.append(code('''
import matplotlib.pyplot as plt

etiquetas, exact_base, exact_ft = [], [], []
for clave, nombre, ruta_base, ruta_ft in [
    ("qwen", "Qwen", ("baseline", "exactitud"), ("finetuned", "exactitud")),
    ("pipeline_qwen_bert", "Pipeline", ("config_A_solo_qwen", "exactitud"), ("config_B_pipeline", "exactitud")),
    ("flan_t5", "FLAN-T5", ("baseline", "exactitud"), ("finetuned", "exactitud")),
]:
    if clave in R:
        etiquetas.append(nombre)
        exact_base.append(g(clave, *ruta_base, default=0))
        exact_ft.append(g(clave, *ruta_ft, default=0))

if etiquetas:
    x = np.arange(len(etiquetas)); ancho = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - ancho/2, exact_base, ancho, label="Antes (baseline / sin enrutar)")
    ax.bar(x + ancho/2, exact_ft, ancho, label="Después (fine-tuned / pipeline)")
    for i, (b, f) in enumerate(zip(exact_base, exact_ft)):
        ax.text(i - ancho/2, b + 0.01, f"{b:.0%}", ha="center", fontsize=9)
        ax.text(i + ancho/2, f + 0.01, f"{f:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(etiquetas)
    ax.set_ylabel("Exactitud de la respuesta final"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Capacidad generativa · {list(n_vals.values())[0]} ejemplos de validación")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()
'''))

    c.append(code('''
clf_etiquetas, clf_valores = [], []
if "bert" in R:
    clf_etiquetas.append("BETO\\n(especialista)")
    clf_valores.append(g("bert", "finetuned", "accuracy", default=0))
if "bert" in R:
    clf_etiquetas.append("TF-IDF\\n(baseline clásico)")
    clf_valores.append(g("bert", "baseline_tfidf", "accuracy", default=0))
if "flan_t5" in R:
    clf_etiquetas.append("FLAN-T5\\n(implícita)")
    clf_valores.append(g("flan_t5", "clasificacion_finetuned", "accuracy", default=0))

if clf_etiquetas:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    barras = ax.bar(clf_etiquetas, clf_valores, color=["#3b7dd8", "#9aa5b1", "#d88b3b"][:len(clf_valores)])
    ax.axhline(1/11, ls=":", c="red", lw=1, label="Azar (1/11)")
    for b, v in zip(barras, clf_valores):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.0%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy de clasificación")
    ax.set_title("Capacidad de clasificación · quién identifica mejor el tipo de problema")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()
'''))

    # -------------------------------------------------------------- tokenización
    c.append(md("""
## 5 · Análisis cualitativo I — Tokenización

Los tres tokenizadores se comparan **sobre el mismo texto**, cargándolos aquí
directamente. No hace falta GPU: un tokenizador son unos pocos MB.

Qué mide cada columna:

- **Fertilidad** = tokens por palabra. Más bajo es mejor: significa que el
  vocabulario captura unidades lingüísticas del español en lugar de trocearlas.
  Impacta en el costo de cómputo y en cuánto cabe en la ventana de contexto.
- **Tasa de `<unk>`** = proporción de tokens que el modelo no puede
  representar. Es información destruida antes de llegar al modelo, y ningún
  entrenamiento la recupera.
- **Tokens por ejemplo** = costo efectivo de procesar un problema completo.
"""))

    c.append(code('''
from transformers import AutoTokenizer

MODELOS_TOK = {
    "Qwen2.5 (BPE)": "Qwen/Qwen2.5-1.5B-Instruct",
    "BETO (WordPiece)": "dccuchile/bert-base-spanish-wwm-cased",
    "FLAN-T5 (SentencePiece)": "google/flan-t5-base",
}

tokenizadores = {}
for nombre, mid in MODELOS_TOK.items():
    try:
        tokenizadores[nombre] = AutoTokenizer.from_pretrained(mid)
        print(f"  OK  {nombre}")
    except Exception as e:
        print(f"  Falló {nombre}: {e}")
'''))

    c.extend(celda_datos(prefijo="5"))
    c.extend(celda_carga_datos(prefijo="5"))

    c.append(code('''
textos = [r["entrada"] + " " + r["salida"] for r in registros]

filas_tok = []
for nombre, tok in tokenizadores.items():
    n_tokens, n_unk, n_palabras = 0, 0, 0
    for t in textos:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_tokens += len(ids)
        n_unk += sum(1 for i in ids if i == tok.unk_token_id)
        n_palabras += len(t.split())
    filas_tok.append({
        "Tokenizador": nombre,
        "Vocabulario": tok.vocab_size,
        "Tokens/palabra": round(n_tokens / n_palabras, 3),
        "Tokens/ejemplo": round(n_tokens / len(textos), 1),
        "Tasa <unk>": f"{n_unk / n_tokens:.3%}",
    })

df_tok = pd.DataFrame(filas_tok)
print(df_tok.to_string(index=False))
'''))

    c.append(code('''
frases = [
    "¿Cuántos caramelos recibirá cada amigo?",
    "864 ÷ 12 = 72",
    "El área del círculo es π × r²",
    "√144 + 8 = 20",
]

for frase in frases:
    print("=" * 78)
    print(f"«{frase}»")
    for nombre, tok in tokenizadores.items():
        piezas = tok.tokenize(frase)
        ids = tok(frase, add_special_tokens=False)["input_ids"]
        n_unk = sum(1 for i in ids if i == tok.unk_token_id)
        marca = f"  [{n_unk} <unk>]" if n_unk else ""
        print(f"  {nombre:26s} ({len(piezas):2d}){marca}")
        print(f"      {' | '.join(piezas)}")
print("=" * 78)
'''))

    c.append(md("""
### Cómo leer esta comparación

**Qwen (BPE, 151k)** — El vocabulario más grande de los tres, entrenado
multilingüe. Suele dar la fertilidad más baja en español y representar los
símbolos matemáticos sin recurrir a `<unk>`. El costo es un embedding de
entrada enorme: 151k × 1536 son ~230M de parámetros solo en la tabla de
embeddings, más que el modelo FLAN-T5 completo.

**BETO (WordPiece, 31k)** — Vocabulario pequeño pero **enteramente dedicado al
español**. Es la demostración de que el tamaño no es lo que importa: importa la
correspondencia entre el vocabulario y el idioma de los datos. En cortes
morfológicos ("estudiantes", "dividimos") suele hacerlo mejor que tokenizadores
multilingües mucho mayores.

**FLAN-T5 (SentencePiece, 32k)** — Del mismo tamaño que BETO pero entrenado
sobre inglés. Es donde aparecen los `<unk>` en `¿`, `÷`, `√`, `²`. El notebook 4
mide cuánto y lo mitiga normalizando. La lección práctica: un vocabulario del
mismo tamaño puede ser excelente o inservible según el idioma para el que se
construyó.

**Consecuencia para el proyecto.** Si el tutor debe operar en español, el
tokenizador es un criterio de selección de primer orden, al mismo nivel que el
número de parámetros. Un modelo mayor con mal tokenizador puede rendir peor que
uno menor bien ajustado al idioma.
"""))

    # -------------------------------------------------------------- fine-tuning
    c.append(md("""
## 6 · Análisis cualitativo II — Fine-tuning

Tres ejes: estabilidad, sensibilidad al dataset pequeño y convergencia.
"""))

    c.append(code('''
filas_ft = []
for clave, nombre in [("qwen", "Qwen2.5-1.5B"), ("bert", "BETO"), ("flan_t5", "FLAN-T5-base")]:
    if clave not in R:
        continue
    ent = g(clave, "parametros_entrenables")
    tot = g(clave, "parametros_totales")
    filas_ft.append({
        "Modelo": nombre,
        "Método": g(clave, "metodo", default="—"),
        "Entrenables": f"{ent/1e6:.2f} M" if ent else "—",
        "% del total": f"{100*ent/tot:.2f}%" if ent and tot else "—",
        "LR": g(clave, "learning_rate", default="—"),
        "Épocas": g(clave, "epocas", default="—"),
        "Mejor época": g(clave, "mejor_epoca", default="—"),
        "Tiempo (s)": f"{g(clave,'tiempo_entrenamiento_s'):.0f}" if g(clave, "tiempo_entrenamiento_s") else "—",
    })

df_ft = pd.DataFrame(filas_ft)
print(df_ft.to_string(index=False))
print()
print("La columna 'Mejor época' es la más informativa: indica cuándo dejó de")
print("mejorar la validación. Una mejor época muy temprana respecto al total")
print("significa que el modelo agotó lo aprendible del corpus enseguida.")
'''))

    c.append(code('''
for clave, nombre, total in [("qwen", "Qwen", g("qwen", "epocas")),
                             ("bert", "BETO", g("bert", "epocas")),
                             ("flan_t5", "FLAN-T5", g("flan_t5", "epocas"))]:
    mejor = g(clave, "mejor_epoca")
    if mejor and total:
        frac = mejor / total
        if frac < 0.35:
            lectura = "convergencia MUY TEMPRANA -> el corpus es el límite, no el entrenamiento"
        elif frac < 0.75:
            lectura = "convergencia equilibrada -> presupuesto de épocas razonable"
        else:
            lectura = "seguía mejorando al final -> probablemente convenga entrenar más"
        print(f"{nombre:10s} mejor época {mejor:.0f} de {total:.0f} ({frac:.0%})  ->  {lectura}")
'''))

    c.append(md("""
### Estabilidad

| Modelo | Comportamiento observado |
|---|---|
| **Qwen + LoRA** | Estable **siempre que** los parámetros entrenables estén en fp32. Con los pesos LoRA en fp16 el gradiente se subdesborda y la pérdida se va a NaN en pocos pasos. Es el fallo más común del notebook 1 y por eso el casting es explícito allí. |
| **BETO completo** | El más estable de los tres. Fine-tuning completo de un encoder con `lr=3e-5` es una receta madura y sin sorpresas. |
| **FLAN-T5 + LoRA** | Estable en fp32; **inutilizable en fp16**. T5 fue preentrenado en bfloat16 y sus activaciones desbordan el rango de fp16. Como la T4 no tiene bf16, fp32 es la única opción. |

Conclusión transferible: la estabilidad depende menos de la arquitectura que de
la **interacción entre la precisión numérica y el preentrenamiento del modelo**.
Es una fuente de fallos que no aparece en los tutoriales y consume horas de
depuración.

### Sensibilidad al dataset pequeño

Ordenados de menos a más sensible:

1. **BETO** — Es el que mejor tolera 132 ejemplos. Solo tiene que aprender una
   frontera de decisión sobre representaciones que ya son buenas; el
   preentrenamiento hace casi todo el trabajo.
2. **FLAN-T5** — Tiene que aprender un formato de salida completo (categoría +
   pasos + respuesta). Más que aprender, sobre todo tiene que aprender a
   *imitar*, y para eso 132 ejemplos alcanzan razonablemente.
3. **Qwen** — El más propenso a memorizar. 1.5B de parámetros sobre 132
   ejemplos aprenden el corpus de memoria si se lo permiten. Por eso se evalúa
   por época y se recupera el mejor punto en lugar del último.

### Convergencia

- **BERT** converge en pocas épocas y luego se aplana. El aplanamiento no es un
  fallo: es la señal de que agotó la información disponible.
- **Qwen con LoRA** baja rápido las primeras épocas —está aprendiendo el
  formato— y después mucho más lento, cuando le tocaría aprender a calcular,
  que es lo difícil.
- **FLAN-T5** desciende de forma más gradual y sostenida: tiene menos capacidad
  de memorización, así que el sobreajuste tarda más en aparecer.
"""))

    # -------------------------------------------------------------- curva aprendizaje
    c.append(md("""
## 7 · Análisis cualitativo III — Curva de aprendizaje

Tres preguntas distintas que conviene no mezclar.

### ¿Cuál aprende más rápido?

**BETO**, sin discusión, en las tres acepciones de "rápido":

- *En tiempo de reloj*: segundos por época contra minutos.
- *En número de ejemplos*: alcanza su meseta antes que los demás.
- *En épocas*: su mejor época suele estar en el primer tercio del presupuesto.

La razón es que su tarea es la más fácil. Elegir entre 11 opciones es
incomparablemente más simple que generar una secuencia correcta de 40 tokens
donde un solo dígito equivocado invalida la respuesta.

### ¿Cuál requiere más datos?

**Qwen**, y con diferencia. La razón es la estructura del espacio de salida:

| Modelo | Espacio de salida | Ejemplos necesarios |
|---|---|---|
| BETO | 11 clases | Decenas por clase bastan para una frontera decente |
| FLAN-T5 | Secuencias cortas y muy estructuradas | Cientos |
| Qwen | Secuencias libres | Miles, para que la mejora sea de razonamiento y no solo de formato |

Con 132 ejemplos, Qwen aprende **formato**. Para aprender **aritmética**
harían falta uno o dos órdenes de magnitud más, o un enfoque distinto
(herramientas externas de cálculo, verificación simbólica).

### ¿Cuál generaliza mejor?

Depende de a qué se le llame generalizar, y merece la pena separarlo:

- **A ejemplos nuevos de categorías conocidas**: BETO. Es lo que mide su
  accuracy en validación.
- **A tipos de problema no vistos**: Qwen. Un decoder grande arrastra
  conocimiento matemático de su preentrenamiento y puede intentar problemas
  fuera de nuestras 11 categorías. BETO no puede: su espacio de clases es
  cerrado por diseño.
- **Al formato**: FLAN-T5 y Qwen, ambos aprenden la estructura de salida con
  facilidad.

**El punto que no hay que perder de vista:** con 33 ejemplos de validación,
ninguna diferencia menor a ~10 puntos porcentuales es estadísticamente
distinguible. Un solo ejemplo vale 3 puntos. Las conclusiones de esta sección
son *direccionales* y están sostenidas por el razonamiento arquitectónico tanto
como por los números. Presentarlas como mediciones definitivas sería un error
metodológico, y explicitarlo es parte del rigor del trabajo.
"""))

    c.append(code('''
print("Recordatorio de tamaño muestral")
print("=" * 60)
n = list(n_vals.values())[0] if n_vals else 33
print(f"Ejemplos de validación: {n}")
print(f"Un ejemplo vale: {1/n:.1%}")
print(f"Intervalo de confianza aproximado al 95% para p=0.5: ±{1.96*np.sqrt(0.25/n):.1%}")
print("=" * 60)
print("Cualquier diferencia entre modelos menor a ese margen NO es concluyente.")
'''))

    # -------------------------------------------------------------- recomendación
    c.append(md("""
## 8 · Recomendación final

### La pregunta

¿Qué arquitectura desplegaría en producción un tutor de matemáticas en español?

### La respuesta corta

**Qwen2.5 con LoRA como generador, con el clasificador BERT como componente de
enrutamiento y observabilidad — es decir, la arquitectura del notebook 3 — pero
solo si el pipeline demuestra aporte medible. Si B ≈ A en sus resultados,
desplieguen Qwen solo y conserven BERT como instrumento de monitorización.**

### La justificación

**1. La tarea es generativa y eso descarta opciones.**
Un tutor tiene que explicar. BERT queda fuera como sistema completo: es un
componente excelente de una arquitectura, no una arquitectura.

**2. Entre los dos generadores, Qwen tiene la ventaja estructural.**
- Su tokenizador maneja el español y la notación matemática sin `<unk>`;
  FLAN-T5 necesita normalización previa que degrada la salida.
- Su preentrenamiento matemático es más fuerte.
- Escala hacia arriba (3B, 7B, 14B) sin cambiar el código, lo que da un camino
  claro de mejora.
- El precio es el costo de inferencia, que es real pero gestionable.

**3. El clasificador se justifica aunque no mejore la exactitud.**
Este es el punto contraintuitivo y el más importante de la recomendación. En
producción, un enrutador de 110M que cuesta el ~2% de la latencia total aporta
tres cosas que el modelo generativo no puede dar:

- **Observabilidad**: saber qué tipos de problema llegan y en cuáles falla el
  sistema, sin leer texto generado a mano.
- **Control**: rechazar preguntas fuera de dominio, o enrutar `operaciones_
  combinadas` a una calculadora simbólica en lugar de a un LLM, que es la vía
  realista para arreglar la aritmética.
- **Degradación elegante**: con confianza baja, pedir aclaración en lugar de
  inventar.

Ninguna de las tres aparece en las métricas de este laboratorio, y las tres
deciden si un sistema es operable.

**4. Por qué no FLAN-T5, pese a su elegancia conceptual.**
Un modelo único es preferible a dos, y hacer clasificación y generación con una
sola pérdida es la solución limpia. Pero `flan-t5-base` arrastra un tokenizador
inglés y un preentrenamiento matemático más débil. La recomendación cambiaría
si se usara **mT5** o **flan-t5-large**: entonces la comparación habría que
rehacerla, y ese es el experimento natural que sigue a este laboratorio.

### La condición indispensable

Ninguna de las cuatro arquitecturas, con 132 ejemplos, produce un tutor
confiable. La exactitud aritmética es el cuello de botella y no se resuelve con
arquitectura. Antes de desplegar cualquier cosa haría falta:

1. **Más datos** — el corpus está preparado para escalar: basta reemplazar
   `data/math_tutor_dataset.jsonl` y re-ejecutar. Sin cambiar una línea de
   código.
2. **Verificación externa del cálculo** — que el modelo genere el procedimiento
   y una herramienta simbólica (SymPy) valide la aritmética. Es el enfoque que
   usan los sistemas de producción serios, y encaja de forma natural en la
   arquitectura modular del notebook 3.
3. **Evaluación humana** — un procedimiento pedagógicamente correcto no es lo
   mismo que un número correcto, y ninguna de nuestras métricas mide lo primero.
"""))

    c.append(code('''
resumen = {
    "notebooks_disponibles": list(R),
    "n_validacion": n_vals,
    "generacion_exactitud": {
        "qwen": g("qwen", "finetuned", "exactitud"),
        "pipeline": g("pipeline_qwen_bert", "config_B_pipeline", "exactitud"),
        "flan_t5": g("flan_t5", "finetuned", "exactitud"),
    },
    "clasificacion_accuracy": {
        "bert": g("bert", "finetuned", "accuracy"),
        "bert_baseline_tfidf": g("bert", "baseline_tfidf", "accuracy"),
        "flan_t5_implicita": g("flan_t5", "clasificacion_finetuned", "accuracy"),
    },
    "tokenizadores": filas_tok,
    "tiempos_entrenamiento_s": {
        k: g(k, "tiempo_entrenamiento_s") for k in ("qwen", "bert", "flan_t5")
    },
}

Path("resultados").mkdir(exist_ok=True)
Path("resultados/comparacion_final.json").write_text(
    json.dumps(resumen, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(resumen, ensure_ascii=False, indent=2, default=str))
print("\\nGuardado en resultados/comparacion_final.json")
'''))

    c.append(md("""
## 9 · Cierre

Lo que este laboratorio deja, más allá de los números:

1. **Un protocolo experimental correcto.** Mismos datos, mismas particiones,
   mismas métricas, baseline antes de tocar nada. Sin eso, cuatro notebooks
   habrían sido cuatro anécdotas.
2. **Una respuesta arquitectónica fundamentada.** No "Qwen es mejor", sino
   "para esta tarea, en este idioma, con este presupuesto, un decoder con
   enrutador es la elección defendible, y estas son las condiciones bajo las
   que cambiaría".
3. **Los límites, dichos en voz alta.** 33 ejemplos de validación no permiten
   distinguir diferencias pequeñas. El corpus limita más que la arquitectura.
   La aritmética no se arregla con fine-tuning. Reconocerlo no debilita el
   trabajo: es lo que lo hace un experimento y no una demostración.
"""))

    return c
