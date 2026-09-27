"""Notebook 2 — BERT (encoder-only) como clasificador de problemas."""

from __future__ import annotations

from nb_comun import (
    celda_args_compatibles, celda_carga_datos, celda_datos, celda_instalacion,
    celda_drive, celda_resultados, celda_wandb, code, encabezado, md,
)


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "Fine-tuning de BERT para clasificar problemas de matemáticas",
        "Notebook 2 de 4 · Arquitectura encoder-only (clasificación)",
    ))

    # ------------------------------------------------------------------ 1
    c.append(md("""
## 1 · Introducción

### Qué problema resolvemos aquí (y cuál no)

**BERT no genera soluciones.** Es importante decirlo desde el principio porque
es el malentendido más común del laboratorio. Un modelo encoder-only no tiene
cabeza de lenguaje autoregresiva: no puede escribir "Paso 1: dividimos 48 entre
6". Lo que hace es *entender* el texto y producir una representación densa que
alimenta un clasificador.

Su papel en el tutor es el de **enrutador**: dada la pregunta de un estudiante,
decidir de qué tipo de problema se trata. Esa decisión es la que el notebook 3
usará para especializar la generación.

### Las 11 categorías

| Categoría | Ejemplo |
|---|---|
| `suma` | "Una biblioteca tenía 145 libros y recibió 89 más" |
| `resta` | "Un tanque contiene 250 litros y se usan 88" |
| `multiplicacion` | "15 cajas con 24 lápices cada una" |
| `division` | "Repartir 48 caramelos entre 6 amigos" |
| `operaciones_combinadas` | "(45 - 9) ÷ 6 + 11" |
| `potencias_raices` | "√144 + 8" |
| `fracciones` | "5/6 + 1/3" |
| `porcentajes` | "El 25% de 420 estudiantes" |
| `ecuaciones` | "2x + 5 = 19" |
| `geometria` | "Área de un círculo de radio 5" |
| `estadistica_probabilidad` | "Probabilidad de sacar un 4 en un dado" |

### El manual de etiquetado

Las categorías se solapan por naturaleza: "3.75 + 2.48" es una suma *y* una
operación con decimales; "el 25% de 420" es un porcentaje *y* una
multiplicación. Sin una regla explícita, el etiquetado sería inconsistente y el
modelo aprendería ruido.

La regla que se aplicó al construir el corpus es una **cascada de precedencia**,
del criterio más específico al más general:

```
1. ¿Pregunta por área, perímetro, ángulos, volumen?  -> geometria
2. ¿Hay que despejar una incógnita?                  -> ecuaciones
3. ¿Es promedio, mediana, moda, rango, probabilidad? -> estadistica_probabilidad
4. ¿Interviene un porcentaje?                        -> porcentajes
5. ¿Los operandos son fracciones?                    -> fracciones
6. ¿Hay potencias o raíces?                          -> potencias_raices
7. ¿Dos o más operaciones distintas / paréntesis?    -> operaciones_combinadas
8. En otro caso, la única operación presente         -> suma|resta|multiplicacion|division
```

Consecuencia deliberada: "área de un cuadrado de lado 12" es `geometria`
aunque implique una potencia, y "3.75 + 2.48" es `suma` porque los decimales
son un formato numérico, no una destreza distinta.

### Ventajas y limitaciones del encoder-only

**Ventajas**
- Atención **bidireccional**: cada token ve el contexto completo, a izquierda y
  derecha. Para entender de qué trata una frase, eso es estrictamente mejor que
  la atención causal de un decoder.
- Muy eficiente: una sola pasada hacia adelante, sin generación token a token.
  La inferencia es dos órdenes de magnitud más rápida que la de Qwen.
- 110M de parámetros: el fine-tuning completo cabe sin problema en una T4.
- La salida es una distribución sobre 11 clases, así que se puede medir con
  métricas cerradas y bien entendidas (F1, matriz de confusión).

**Limitaciones**
- No genera texto. No sirve como tutor por sí solo.
- El conjunto de categorías es cerrado: una pregunta de trigonometría se
  clasificará forzosamente como una de las 11 clases existentes.
- Con 12 ejemplos de entrenamiento por clase, es fácil que aprenda atajos
  léxicos (ver la palabra "porcentaje") en vez de la estructura del problema.
"""))

    # ------------------------------------------------------------------ 2
    c.append(md("""
## 2 · Objetivos

1. Establecer **tres baselines** de dificultad creciente antes de tocar BERT:
   clase mayoritaria, TF-IDF + regresión logística, y BERT con la cabeza sin
   entrenar.
2. Hacer fine-tuning de BETO (BERT en español) para clasificación en 11 clases.
3. Evaluar con accuracy, precision, recall y F1 (macro y ponderado), más el
   reporte por clase y la matriz de confusión.
4. Identificar **qué categorías se confunden entre sí** y explicar por qué.
   Esa información es la que determina si el pipeline del notebook 3 es viable.
5. Registrar todo en W&B y en `resultados/bert.json`.
"""))

    c.extend(celda_instalacion())
    c.extend(celda_drive())

    # ------------------------------------------------------------------ 3
    c.append(md("""
## 3 · Arquitectura del modelo

```
              "¿Cuál es el 25% de 420 estudiantes?"
                              |
                     tokenización WordPiece
                              |
        [CLS]  ¿ cuál es el 25 % de 420 estudiantes ?  [SEP]
                              |
                    +---------v----------+
                    |  Embeddings        |
                    | (token+pos+segm.)  |
                    +---------v----------+
                              |
                   +----------v-----------+
                   |  12 bloques BERT:    |
                   |  - Autoatención      |  <- BIDIRECCIONAL: cada token
                   |    (12 cabezas)      |     ve toda la frase
                   |  - FFN + LayerNorm   |
                   +----------v-----------+
                              |
                  vector [CLS] de 768 dimensiones
                              |
                    +---------v----------+
                    | Cabeza lineal 768→11|  <- esto es lo NUEVO,
                    +---------v----------+      inicializado al azar
                              |
                   logits sobre 11 categorías
```

La diferencia decisiva frente a Qwen está en la máscara de atención: BERT no
tiene ninguna. El token `25` puede atender a `estudiantes`, que está a su
derecha. Un decoder no puede hacerlo, y por eso los encoders siguen siendo la
mejor opción para clasificar.

El vector `[CLS]` es un resumen aprendido de la secuencia completa. Sobre él se
monta una capa lineal de 768×11 que es la única parte del modelo que no viene
preentrenada.
"""))

    c.append(code('''
MODELO_ID = "dccuchile/bert-base-spanish-wwm-cased"   # BETO: BERT entrenado en español
# Alternativa multilingüe (peor en español, útil si BETO no está disponible):
# MODELO_ID = "bert-base-multilingual-cased"

DIR_MODELO   = "modelos/bert-clasificador"
LONGITUD_MAX = 64        # se justifica en la sección 6
'''))

    c.append(md("""
> **Por qué BETO y no `bert-base-multilingual-cased`.** mBERT reparte un
> vocabulario de 119k tokens entre 104 idiomas; BETO dedica sus 31k tokens
> exclusivamente al español. Sobre texto en español, BETO parte las palabras en
> menos piezas y con cortes más lingüísticos ("estudiantes" como una unidad en
> lugar de "estu ##dian ##tes"). Con 132 ejemplos de entrenamiento, esa
> eficiencia de representación importa mucho más que en un corpus grande.
"""))

    # ------------------------------------------------------------------ 4
    c.append(md("""
## 4 · Carga del dataset
"""))
    c.extend(celda_datos())
    c.extend(celda_carga_datos())
    c.extend(celda_resultados())

    # ------------------------------------------------------------------ 5
    c.append(md("""
## 5 · Preprocesamiento

Para clasificación el preprocesamiento es notablemente más simple que para
generación: la entrada es el enunciado, la etiqueta es un entero de 0 a 10.

Un detalle que sí importa: **el modelo solo ve `entrada`, nunca `salida`**.
Sería trivial clasificar leyendo el procedimiento ("Dividimos 48 entre 6" dice
la categoría en voz alta), pero en producción el procedimiento todavía no
existe: es justo lo que queremos generar después. Entrenar con él sería una
fuga de información que inflaría las métricas y no funcionaría en el pipeline.
"""))

    c.append(code('''
from collections import Counter

X_train = [r["entrada"] for r in train]
y_train = [r["categoria_id"] for r in train]
X_val   = [r["entrada"] for r in val]
y_val   = [r["categoria_id"] for r in val]

print(f"Entrenamiento: {len(X_train)}  |  Validación: {len(X_val)}")
print(f"Clases: {len(CATEGORIAS)}")
print(f"Distribución en validación: {dict(Counter(y_val))}")
print()
for i in range(3):
    print(f"  [{CATEGORIAS[y_train[i]]}] {X_train[i]}")
'''))

    # ------------------------------------------------------------------ 6
    c.append(md("""
## 6 · Tokenización

BETO usa **WordPiece**: parte las palabras en subunidades marcando con `##` las
continuaciones. Comparen mentalmente con lo que verán en los otros notebooks
(BPE en Qwen, SentencePiece en FLAN-T5); en el notebook 5 lo cuantificamos.
"""))

    c.append(code('''
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODELO_ID)

longitudes, fertilidades = [], []
for r in registros:
    ids = tokenizer(r["entrada"])["input_ids"]
    longitudes.append(len(ids))
    fertilidades.append((len(ids) - 2) / len(r["entrada"].split()))

longitudes = np.array(longitudes)
print(f"Tokens por enunciado: min={longitudes.min()}  media={longitudes.mean():.1f}  "
      f"p95={np.percentile(longitudes, 95):.0f}  max={longitudes.max()}")
print(f"Ejemplos que excederían LONGITUD_MAX={LONGITUD_MAX}: {(longitudes > LONGITUD_MAX).sum()}")
print(f"Fertilidad (tokens/palabra): {np.mean(fertilidades):.3f}")
print(f"Vocabulario: {tokenizer.vocab_size}")

STATS_TOKENIZADOR = {
    "modelo": MODELO_ID,
    "tokenizador": tokenizer.__class__.__name__,
    "vocabulario": int(tokenizer.vocab_size),
    "fertilidad_media": float(np.mean(fertilidades)),
    "tokens_media": float(longitudes.mean()),
    "tokens_max": int(longitudes.max()),
}
'''))

    c.append(code('''
muestras = [
    "¿Cuál es el 25% de 420 estudiantes?",
    "Resuelve la siguiente operación: 864 ÷ 12.",
    "Calcula el área de un círculo de radio 5 usando π = 3.14.",
]
for m in muestras:
    piezas = tokenizer.tokenize(m)
    print(f"\\n({len(piezas)} tokens) {m}")
    print("   " + " | ".join(piezas))

ids_unk = [i for m in muestras for i in tokenizer(m)["input_ids"] if i == tokenizer.unk_token_id]
print(f"\\nTokens desconocidos: {len(ids_unk)}")
'''))

    c.append(code('''
from datasets import Dataset


def tokenizar(reg):
    tok = tokenizer(reg["entrada"], truncation=True, max_length=LONGITUD_MAX)
    tok["labels"] = reg["categoria_id"]
    return tok


ds_train = Dataset.from_list([tokenizar(r) for r in train])
ds_val   = Dataset.from_list([tokenizar(r) for r in val])
print(ds_train)
print("\\nPrimer ejemplo decodificado:")
print(tokenizer.decode(ds_train[0]["input_ids"]))
print("Etiqueta:", CATEGORIAS[ds_train[0]["labels"]])
'''))

    c.append(code('''
from transformers import DataCollatorWithPadding

collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
print("Collator con padding dinámico listo.")
'''))

    # ------------------------------------------------------------------ métricas clasificación
    c.append(md("""
### Métricas de clasificación

Cuatro métricas, y hay que tener claro qué mide cada una porque con 11 clases
balanceadas es fácil citar la equivocada:

| Métrica | Qué responde |
|---|---|
| **Accuracy** | ¿Qué proporción del total acertó? Con clases balanceadas es informativa; con clases desbalanceadas engaña. |
| **Precision** (por clase) | De todo lo que predije como `porcentajes`, ¿cuánto lo era de verdad? Penaliza los falsos positivos. |
| **Recall** (por clase) | De todos los `porcentajes` reales, ¿cuántos detecté? Penaliza los falsos negativos. |
| **F1-macro** | Media armónica de precision y recall, promediada **dando el mismo peso a cada clase**. Es la métrica principal de este notebook: si el modelo ignora una clase entera, el F1-macro se hunde aunque la accuracy apenas se mueva. |

Se reporta también el F1 ponderado por soporte, pero con nuestro conjunto
balanceado (3 por clase) coincide casi exactamente con la accuracy y aporta
poco.
"""))

    c.append(code('''
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)


def metricas_clasificacion(y_true, y_pred):
    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_pe, r_pe, f1_pe, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_ma),
        "recall_macro": float(r_ma),
        "f1_macro": float(f1_ma),
        "precision_ponderada": float(p_pe),
        "recall_ponderado": float(r_pe),
        "f1_ponderado": float(f1_pe),
    }


def compute_metrics(eval_pred):
    """Callback del Trainer: se ejecuta al final de cada época."""
    logits, labels = eval_pred
    return metricas_clasificacion(labels, np.argmax(logits, axis=-1))


def imprimir_metricas(nombre, m):
    print(f"{nombre:34s} acc={m['accuracy']:.3f}  F1-macro={m['f1_macro']:.3f}  "
          f"P-macro={m['precision_macro']:.3f}  R-macro={m['recall_macro']:.3f}")
'''))

    # ------------------------------------------------------------------ 7
    c.append(md("""
## 7 · Baseline

Aquí la metodología se separa de la del notebook de Qwen, y conviene explicar
por qué.

Un modelo generativo sin entrenar **sí sabe hacer la tarea** (mal, pero la
hace): se le pide que resuelva un problema y produce texto. Un encoder sin
entrenar **no puede clasificar en absoluto**: su cabeza de clasificación acaba
de inicializarse con números aleatorios. Preguntarle es preguntarle a un dado
de 11 caras.

Por eso usamos tres referencias, no una:

1. **Clase mayoritaria** — el mínimo absoluto. Con 11 clases balanceadas está
   en 1/11 ≈ 9%. Cualquier sistema debe superarlo o no sirve para nada.
2. **TF-IDF + regresión logística** — el baseline honesto. Un modelo clásico,
   sin redes neuronales, entrenado en dos segundos. Si BERT no lo supera, no
   hay ninguna razón para pagar el costo de un transformer de 110M de
   parámetros. *Esta es la comparación que de verdad importa.*
3. **BETO con la cabeza sin entrenar** — se incluye para hacer visible que el
   preentrenamiento aporta representaciones, pero no la tarea.
"""))

    c.append(code('''
# Baseline 1: clase mayoritaria
from collections import Counter

clase_mayoritaria = Counter(y_train).most_common(1)[0][0]
pred_mayoritaria = [clase_mayoritaria] * len(y_val)
m_mayoritaria = metricas_clasificacion(y_val, pred_mayoritaria)
imprimir_metricas("1) Clase mayoritaria", m_mayoritaria)
print(f"   (predice siempre '{CATEGORIAS[clase_mayoritaria]}')")
'''))

    c.append(code('''
# Baseline 2: TF-IDF + regresión logística
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

clasico = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
    LogisticRegression(max_iter=2000, C=5.0),
)
clasico.fit(X_train, y_train)
pred_clasico = clasico.predict(X_val)
m_clasico = metricas_clasificacion(y_val, pred_clasico)
imprimir_metricas("2) TF-IDF + LogisticRegression", m_clasico)

# Con solo 3 ejemplos de validación por clase, una única medición es ruidosa.
# La validación cruzada sobre el corpus completo da una estimación mucho más
# estable de lo que el enfoque clásico puede lograr. Es barata: úsenla como
# referencia principal al discutir si BERT vale la pena.
X_todo = [r["entrada"] for r in registros]
y_todo = [r["categoria_id"] for r in registros]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
scores = cross_val_score(clasico, X_todo, y_todo, cv=cv, scoring="f1_macro")
print(f"   Validación cruzada 5-fold sobre los 165 ejemplos: "
      f"F1-macro = {scores.mean():.3f} ± {scores.std():.3f}")
'''))

    c.append(code('''
# Baseline 3: BETO con la cabeza de clasificación sin entrenar
import torch
from transformers import AutoModelForSequenceClassification

modelo = AutoModelForSequenceClassification.from_pretrained(
    MODELO_ID,
    num_labels=len(CATEGORIAS),
    id2label={i: c for i, c in enumerate(CATEGORIAS)},
    label2id={c: i for i, c in enumerate(CATEGORIAS)},
).to(DEVICE)


@torch.no_grad()
def predecir(modelo, textos, batch=16):
    modelo.eval()
    salida = []
    for i in range(0, len(textos), batch):
        lote = tokenizer(textos[i:i + batch], padding=True, truncation=True,
                         max_length=LONGITUD_MAX, return_tensors="pt").to(modelo.device)
        salida.extend(modelo(**lote).logits.argmax(-1).cpu().tolist())
    return salida


pred_sin_entrenar = predecir(modelo, X_val)
m_sin_entrenar = metricas_clasificacion(y_val, pred_sin_entrenar)
imprimir_metricas("3) BETO sin fine-tuning", m_sin_entrenar)
print("   (la cabeza está inicializada al azar: el resultado debe rondar 1/11 = 9%)")
'''))

    # ------------------------------------------------------------------ 8
    c.append(md("""
## 8 · Configuración del fine-tuning

### Por qué fine-tuning completo y no LoRA

En el notebook de Qwen, LoRA es prácticamente obligatorio: 1.500 millones de
parámetros no caben en memoria con sus estados de optimizador. Aquí la
situación es distinta:

- BETO tiene 110M de parámetros. Entrenarlo entero en una T4 cabe de sobra.
- La cabeza de clasificación es **nueva** y hay que entrenarla sí o sí.
- Con tan pocos datos, permitir que se ajusten todas las capas suele dar mejor
  resultado en clasificación que restringirse a adaptadores de bajo rango.

Aplicar LoRA aquí sería usar la herramienta por costumbre y no por necesidad.
De todos modos dejamos la variante lista abajo, porque compararlas es una
observación interesante: es el mismo dilema, con distinta respuesta, que en el
notebook 1.

### Hiperparámetros

| Parámetro | Valor | Razón |
|---|---|---|
| `learning_rate` | 3e-5 | Rango estándar para fine-tuning completo de BERT (2e-5 a 5e-5). Dos órdenes de magnitud menor que el de LoRA: aquí se mueven los pesos preentrenados y hay que hacerlo con cuidado. |
| `num_train_epochs` | 15 | Con 132 ejemplos y lotes de 8, una época son ~17 pasos. Hacen falta bastantes épocas para que converja. |
| `batch_size` | 8 | Lotes pequeños dan más pasos de actualización, que es lo que escasea con un corpus mínimo. |
| `warmup_ratio` | 0.1 | Evita que los primeros pasos, con la cabeza aleatoria produciendo gradientes enormes, dañen las representaciones preentrenadas. |
| `metric_for_best_model` | `f1_macro` | No `eval_loss`: nos interesa la calidad de clasificación por clase, no la verosimilitud. |
"""))

    c.extend(celda_args_compatibles())
    c.extend(celda_wandb("bert-clasificador-finetune",
                         '["bert", "beto", "encoder-only", "clasificacion"]'))

    c.append(code('''
from transformers import Trainer

USAR_LORA_EN_BERT = False    # ver la discusión de arriba

if USAR_LORA_EN_BERT:
    from peft import LoraConfig, get_peft_model
    config_lora = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05, bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["classifier"],   # la cabeza nueva se entrena entera
    )
    modelo = get_peft_model(modelo, config_lora)
    modelo.print_trainable_parameters()

args = construir_args(
    output_dir=f"{DIR_CHECKPOINTS}/bert-clasificador",
    run_name=NOMBRE_RUN,

    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,

    learning_rate=3e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    weight_decay=0.01,

    fp16=(DEVICE == "cuda"),
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    report_to=REPORTAR_A,
    seed=SEMILLA,
)

trainer = Trainer(
    model=modelo,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

n_entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in modelo.parameters())
print(f"Parámetros entrenables: {n_entrenables/1e6:.1f} M de {n_total/1e6:.1f} M "
      f"({100*n_entrenables/n_total:.1f}%)")
print(f"Pasos por época: {len(ds_train) // args.per_device_train_batch_size}")
'''))

    # ------------------------------------------------------------------ 9
    c.append(md("""
## 9 · Entrenamiento

Qué observar mientras corre:

- La `eval_loss` de un clasificador con 11 clases arranca cerca de
  `ln(11) ≈ 2.40`, que es la entropía de adivinar al azar. Ver ese número en
  la primera evaluación confirma que todo está bien conectado.
- El `f1_macro` suele dar un salto brusco en las primeras 3-4 épocas y luego
  aplanarse. Ese aplanamiento no significa que el entrenamiento falló:
  significa que el modelo ya extrajo lo que 132 ejemplos podían darle.
- Si `eval_loss` sube mientras `f1_macro` se mantiene, el modelo se está
  volviendo **más confiado** en sus errores. Es sobreajuste incipiente y por
  eso seleccionamos por F1 y no por pérdida.
"""))

    c.append(code('''
resultado_entrenamiento = trainer.train()
print()
print(f"Tiempo de entrenamiento: {resultado_entrenamiento.metrics['train_runtime']:.1f} s")
'''))

    c.append(code('''
import pandas as pd

historial = pd.DataFrame(trainer.state.log_history)
ev = historial.dropna(subset=["eval_f1_macro"])
print(ev[["epoch", "eval_loss", "eval_accuracy", "eval_f1_macro"]].to_string(index=False))

mejor = ev.sort_values("eval_f1_macro", ascending=False).iloc[0]
print(f"\\nMejor época: {mejor['epoch']:.0f}  (F1-macro = {mejor['eval_f1_macro']:.3f})")
'''))

    c.append(code('''
import matplotlib.pyplot as plt

tr = historial.dropna(subset=["loss"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(tr["epoch"], tr["loss"], label="Entrenamiento", alpha=0.8)
axes[0].plot(ev["epoch"], ev["eval_loss"], label="Validación", marker="o")
axes[0].axhline(np.log(len(CATEGORIAS)), ls=":", c="red", lw=1, label="Azar (ln 11)")
axes[0].set_xlabel("Época"); axes[0].set_ylabel("Pérdida")
axes[0].set_title("Curvas de pérdida"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ev["epoch"], ev["eval_accuracy"], marker="o", label="Accuracy")
axes[1].plot(ev["epoch"], ev["eval_f1_macro"], marker="s", label="F1-macro")
axes[1].axhline(m_clasico["f1_macro"], ls="--", c="gray", lw=1,
                label="TF-IDF (F1-macro)")
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Métrica"); axes[1].set_ylim(0, 1)
axes[1].set_title("Métricas de clasificación"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()
'''))

    c.append(code('''
trainer.save_model(DIR_MODELO)
tokenizer.save_pretrained(DIR_MODELO)
print(f"Clasificador guardado en {DIR_MODELO}/")
print("El notebook 3 (pipeline Qwen+BERT) cargará este modelo como etapa 1.")
'''))

    # ------------------------------------------------------------------ 10
    c.append(md("""
## 10 · Integración con Weights & Biases

El `Trainer` ya envió a W&B, época por época, la pérdida y las cuatro métricas
de `compute_metrics`. Lo que añadimos aquí es lo que no se puede reconstruir a
partir de escalares:

- La **matriz de confusión** como objeto interactivo de W&B.
- La tabla de **errores individuales**: qué enunciado, qué predijo, qué era.
- Los baselines, para que en el panel aparezcan como líneas de referencia junto
  a la curva del modelo.

Nombres consistentes con los otros notebooks: proyecto
`tutor-matematicas-arquitecturas`, run `bert-clasificador-finetune`, tags
`bert` / `encoder-only` / `clasificacion`.
"""))

    # ------------------------------------------------------------------ 11
    c.append(md("""
## 11 · Evaluación
"""))

    c.append(code('''
pred_finetuned = predecir(trainer.model, X_val)
m_finetuned = metricas_clasificacion(y_val, pred_finetuned)

print("Modelo con fine-tuning")
print("=" * 60)
for k, v in m_finetuned.items():
    print(f"  {k:22s}: {v:.4f}")
'''))

    c.append(md("""
### Reporte por clase

El promedio macro esconde qué clases funcionan. Con 3 ejemplos de validación
por clase, cada acierto o fallo mueve el recall de esa clase en saltos de 0.33:
**no interpreten diferencias pequeñas entre clases como reales**. Lo que sí es
informativo es una clase con recall 0.
"""))

    c.append(code('''
print(classification_report(
    y_val, pred_finetuned,
    labels=list(range(len(CATEGORIAS))),
    target_names=CATEGORIAS,
    zero_division=0,
    digits=3,
))
'''))

    c.append(md("""
### Matriz de confusión

Aquí es donde está la información realmente accionable. No miren solo la
diagonal: miren **qué se confunde con qué**. Las confusiones esperables por
construcción del esquema de etiquetado son:

- `operaciones_combinadas` ↔ `suma`/`resta`/`multiplicacion`/`division`: la
  frontera es "¿cuántas operaciones distintas hay?", una distinción que exige
  contar operadores, no reconocer vocabulario.
- `porcentajes` ↔ `multiplicacion`: calcular un porcentaje *es* una
  multiplicación; solo el símbolo `%` los separa.
- `potencias_raices` ↔ `geometria`: "área de un cuadrado" implica un cuadrado
  en ambos sentidos de la palabra.

Si aparecen estas confusiones, el modelo está fallando donde el esquema de
etiquetado es genuinamente difícil. Si aparecen otras (por ejemplo `geometria`
confundida con `fracciones`), el problema es falta de datos.
"""))

    c.append(code('''
cm = confusion_matrix(y_val, pred_finetuned, labels=list(range(len(CATEGORIAS))))

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CATEGORIAS))); ax.set_yticks(range(len(CATEGORIAS)))
ax.set_xticklabels(CATEGORIAS, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(CATEGORIAS, fontsize=9)
ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real")
ax.set_title("Matriz de confusión · BETO fine-tuned (validación)")
for i in range(len(CATEGORIAS)):
    for j in range(len(CATEGORIAS)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.show()
'''))

    c.append(code('''
print("Errores de clasificación en validación:")
print("=" * 78)
errores = []
for r, p in zip(val, pred_finetuned):
    if p != r["categoria_id"]:
        errores.append({"id": r["id"], "entrada": r["entrada"],
                        "real": r["categoria"], "predicho": CATEGORIAS[p]})
        print(f"[{r['id']}] {r['entrada']}")
        print(f"    real: {r['categoria']}  ->  predicho: {CATEGORIAS[p]}")
print("=" * 78)
print(f"{len(errores)} errores de {len(val)} ejemplos")
'''))

    # ------------------------------------------------------------------ 12
    c.append(md("""
## 12 · Comparación Baseline vs Fine-tuned
"""))

    c.append(code('''
filas = [
    ("Clase mayoritaria",           m_mayoritaria),
    ("TF-IDF + LogisticRegression", m_clasico),
    ("BETO sin fine-tuning",        m_sin_entrenar),
    ("BETO fine-tuned",             m_finetuned),
]

print(f"{'Modelo':32s} {'Accuracy':>9s} {'F1-macro':>9s} {'P-macro':>9s} {'R-macro':>9s}")
print("=" * 72)
for nombre, m in filas:
    print(f"{nombre:32s} {m['accuracy']:>9.3f} {m['f1_macro']:>9.3f} "
          f"{m['precision_macro']:>9.3f} {m['recall_macro']:>9.3f}")
print("=" * 72)
print(f"\\nMejora de BETO sobre el baseline clásico (F1-macro): "
      f"{m_finetuned['f1_macro'] - m_clasico['f1_macro']:+.3f}")
print(f"Mejora de BETO sobre el azar (F1-macro): "
      f"{m_finetuned['f1_macro'] - m_mayoritaria['f1_macro']:+.3f}")
'''))

    c.append(code('''
fig, ax = plt.subplots(figsize=(9, 4.5))
nombres = [f[0] for f in filas]
x = np.arange(len(nombres)); ancho = 0.38
ax.bar(x - ancho/2, [f[1]["accuracy"] for f in filas], ancho, label="Accuracy")
ax.bar(x + ancho/2, [f[1]["f1_macro"] for f in filas], ancho, label="F1-macro")
ax.axhline(1/len(CATEGORIAS), ls=":", c="red", lw=1, label="Azar (1/11)")
ax.set_xticks(x); ax.set_xticklabels(nombres, rotation=15, ha="right", fontsize=9)
ax.set_ylim(0, 1.05); ax.set_ylabel("Métrica")
ax.set_title("Clasificación de problemas · baselines vs BETO fine-tuned")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
'''))

    c.append(code('''
RESULTADOS_BERT = {
    "notebook": "S04_Lab_Fine_tuning_BERT",
    "arquitectura": "encoder-only",
    "modelo": MODELO_ID,
    "metodo": "LoRA" if USAR_LORA_EN_BERT else "fine-tuning completo",
    "n_train": len(train),
    "n_val": len(val),
    "n_clases": len(CATEGORIAS),
    "epocas": float(args.num_train_epochs),
    "learning_rate": args.learning_rate,
    "tiempo_entrenamiento_s": resultado_entrenamiento.metrics["train_runtime"],
    "eval_loss_final": float(ev.iloc[-1]["eval_loss"]),
    "mejor_epoca": float(mejor["epoch"]),
    "parametros_entrenables": int(n_entrenables),
    "parametros_totales": int(n_total),
    "baseline_mayoritaria": m_mayoritaria,
    "baseline_tfidf": m_clasico,
    "baseline_tfidf_cv_f1_macro": float(scores.mean()),
    "baseline_tfidf_cv_std": float(scores.std()),
    "baseline_sin_finetuning": m_sin_entrenar,
    "finetuned": m_finetuned,
    "matriz_confusion": cm.tolist(),
    "categorias": CATEGORIAS,
    "errores": errores,
    "tokenizador": STATS_TOKENIZADOR,
}

guardar_resultados("bert", RESULTADOS_BERT)
'''))

    c.append(code('''
if USAR_WANDB:
    import wandb

    if wandb.run is None:
        wandb.init(project=PROYECTO, name=NOMBRE_RUN, tags=TAGS, reinit=True)

    wandb.log({"evaluacion/matriz_confusion": wandb.plot.confusion_matrix(
        probs=None, y_true=y_val, preds=pred_finetuned, class_names=CATEGORIAS)})

    tabla = wandb.Table(columns=["id", "entrada", "real", "predicho", "acierto"])
    for r, p in zip(val, pred_finetuned):
        tabla.add_data(r["id"], r["entrada"], r["categoria"],
                       CATEGORIAS[p], p == r["categoria_id"])
    wandb.log({"evaluacion/predicciones": tabla})

    wandb.summary.update({
        "baseline/mayoritaria_f1_macro": m_mayoritaria["f1_macro"],
        "baseline/tfidf_f1_macro": m_clasico["f1_macro"],
        "baseline/tfidf_cv_f1_macro": float(scores.mean()),
        "baseline/sin_finetuning_f1_macro": m_sin_entrenar["f1_macro"],
        "finetuned/accuracy": m_finetuned["accuracy"],
        "finetuned/f1_macro": m_finetuned["f1_macro"],
        "delta/f1_macro_vs_tfidf": m_finetuned["f1_macro"] - m_clasico["f1_macro"],
    })
    wandb.finish()
    print("Registro en W&B completado.")
else:
    print("W&B desactivado; las métricas quedaron en resultados/bert.json")
'''))

    # ------------------------------------------------------------------ 13
    c.append(md("""
## 13 · Discusión

**1. ¿BETO le ganó a TF-IDF?**
Es la pregunta central. Si la diferencia en F1-macro es pequeña (menos de ~0.10),
la conclusión honesta es que con 132 ejemplos el problema se resuelve casi
igual de bien con un modelo lineal sobre n-gramas. Y tiene sentido: nuestras
categorías se distinguen en buena medida por palabras clave ("porcentaje",
"área", "probabilidad"), que es exactamente lo que TF-IDF captura. La ventaja
de BERT debería aparecer en los casos donde el vocabulario no basta —"un
número más 15 es igual a 42" no contiene la palabra "ecuación"—. Vale la pena
revisar si esos casos concretos los acierta BERT y falla TF-IDF.

**2. ¿Qué confunde el modelo?**
Contrasten la matriz de confusión con las confusiones predichas más arriba. Si
coinciden, el límite es el esquema de etiquetado, no el modelo. Si no
coinciden, es falta de datos.

**3. ¿Es fiable esta medición?**
No del todo, y hay que decirlo. 33 ejemplos de validación, 3 por clase: un solo
acierto cambia la accuracy 3 puntos y el recall de una clase 33 puntos. La
validación cruzada del baseline clásico da una idea de la magnitud del ruido.
Para una medición sólida haría falta validación cruzada del modelo completo
(caro, 5 entrenamientos) o simplemente más datos.

**4. ¿Qué implica para el pipeline del notebook 3?**
El pipeline BERT→Qwen hereda estos errores. Si la accuracy es del 85%, uno de
cada siete problemas llegará a Qwen con la etiqueta equivocada y por tanto con
el prompt especializado equivocado. Esa **propagación de error** es la
desventaja estructural de cualquier sistema en cascada, y en el notebook 3 la
medimos explícitamente.
"""))

    # ------------------------------------------------------------------ 14
    c.append(md("""
## 14 · Conclusiones

1. **El encoder es la herramienta correcta para esta tarea concreta.**
   Bidireccionalidad, inferencia barata y una salida cerrada y medible. Nada de
   eso lo ofrece un decoder.
2. **Pero no resuelve el problema del tutor.** Clasificar no es enseñar. BERT
   es un componente, no un producto.
3. **El baseline clásico es la referencia que importa.** Documenten la
   diferencia con TF-IDF: es lo que justifica (o no) el costo del transformer.
4. **La clasificación tiene sentido si algo la usa.** Su valor se demuestra en
   el notebook 3: si condicionar la generación por categoría no mejora nada,
   entonces esta etapa es complejidad sin retorno, y ese también sería un
   resultado válido del experimento.
"""))

    return c
