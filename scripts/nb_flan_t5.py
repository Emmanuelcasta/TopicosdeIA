"""Notebook 4 — FLAN-T5 (encoder-decoder): clasificación y generación unificadas."""

from __future__ import annotations

from nb_comun import (
    celda_args_compatibles, celda_carga_datos, celda_datos, celda_instalacion,
    celda_drive, celda_metricas, celda_resultados, celda_wandb, code,
    encabezado, md,
)


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "Fine-tuning de FLAN-T5 como tutor de matemáticas integrado",
        "Notebook 4 de 4 · Arquitectura encoder-decoder (comprensión + generación)",
    ))

    # ------------------------------------------------------------------ 1
    c.append(md("""
## 1 · Introducción

### La propuesta

Los notebooks 2 y 3 dividieron el problema: un encoder entiende y clasifica, un
decoder genera. FLAN-T5 propone hacerlo todo en **una sola arquitectura** que
tiene ambas partes por diseño.

Y lo hacemos explícito en la salida: entrenamos al modelo para que escriba la
categoría **antes** del procedimiento.

```
Entrada:  "resuelve el problema: En una escuela hay 420 estudiantes.
           Si el 25% participa en un torneo, cuantos participan?"

Salida:   "Categoria: porcentajes
           Paso 1: El 25% equivale a multiplicar por 0.25.
           Paso 2: 420 x 0.25 = 105.
           Respuesta final: 105 estudiantes"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
            clasificación + generación en una sola pasada
```

Esto convierte al notebook en el **contraste directo del pipeline del notebook
3**: los mismos dos trabajos, resueltos por un modelo en lugar de por dos, y
medidos con las mismas métricas sobre los mismos datos. La comparación de
ambos es el núcleo del experimento.

### Por qué encoder-decoder

| Componente | Papel |
|---|---|
| **Encoder** | Lee el enunciado completo con atención bidireccional, igual que BERT. Construye una representación de todo el problema antes de escribir nada. |
| **Cross-attention** | En cada paso de generación, el decoder consulta la representación completa del enunciado. No depende de "recordarlo" a través de su propio contexto. |
| **Decoder** | Genera el procedimiento token a token, igual que Qwen. |

Un decoder-only como Qwen procesa el enunciado con la misma máquina causal con
la que escribe la respuesta: al leer "420" todavía no sabe que después vendrá
"25%". El encoder de T5 no tiene esa limitación. Para problemas donde el dato
clave aparece al final, esa es una ventaja arquitectónica real.

### Por qué `google/flan-t5-base`

- 250M de parámetros: entre BETO (110M) y Qwen (1.5B). Entrena rápido.
- El sufijo **FLAN** significa que fue afinado con instrucciones sobre más de
  1.800 tareas. Sigue instrucciones sin necesidad de tokens de chat especiales.
- Es el encoder-decoder abierto de referencia: la comparación es reproducible y
  el profesor puede contrastarla con literatura publicada.

### Ventajas y limitaciones

**Ventajas**
- Un solo modelo que versionar, desplegar y monitorizar. No hay riesgo de
  desalineación entre componentes.
- Sin propagación de error en cascada: clasificación y generación se optimizan
  con la misma pérdida y son coherentes por construcción.
- Comprensión bidireccional del enunciado, que un decoder-only no tiene.

**Limitaciones — y una es seria**
- **El tokenizador de T5 fue construido para inglés.** Su SentencePiece de 32k
  no cubre bien las tildes, la apertura de interrogación `¿`, ni los símbolos
  `×`, `÷`, `√`, `²`. Sobre un corpus 100% en español y lleno de notación
  matemática, esto no es un detalle: es el factor limitante del notebook. La
  sección 6 lo mide y lo trata.
- Salida de longitud acotada; menos flexible que un decoder grande.
- Su preentrenamiento matemático es más débil que el de Qwen.
"""))

    # ------------------------------------------------------------------ 2
    c.append(md("""
## 2 · Objetivos

1. **Diagnosticar el tokenizador** sobre nuestro corpus: medir la tasa de
   tokens `<unk>` y determinar qué caracteres se pierden.
2. Diseñar y aplicar una **normalización** del texto que preserve el
   significado matemático dentro del vocabulario disponible.
3. Entrenar FLAN-T5 con LoRA para producir categoría + procedimiento +
   respuesta en una sola secuencia.
4. Evaluar **las dos capacidades por separado**: exactitud de la respuesta
   (comparable con los notebooks 1 y 3) y accuracy de clasificación
   (comparable con el notebook 2).
5. Contrastar el modelo integrado contra el pipeline modular y dejar los
   resultados listos para la comparación final.
"""))

    c.extend(celda_instalacion())
    c.extend(celda_drive())

    # ------------------------------------------------------------------ 3
    c.append(md("""
## 3 · Arquitectura del modelo

```
   "resuelve el problema: el 25% de 420 estudiantes"
                        |
        ┌───────────────v────────────────┐
        │          ENCODER               │
        │   12 bloques                   │
        │   autoatención BIDIRECCIONAL   │   <- como BERT
        │   -> H (representación         │
        │        del enunciado)          │
        └───────────────┬────────────────┘
                        │  H
                        │        ┌─────────────────────────┐
                        └───────>│       DECODER           │
                                 │   12 bloques            │
                                 │   1. autoatención causal│  <- como Qwen
                                 │   2. CROSS-ATTENTION -> H│  <- lo propio de T5
                                 │   3. feed-forward       │
                                 └────────────┬────────────┘
                                              │
                                "Categoria: porcentajes
                                 Paso 1: ...
                                 Respuesta final: 105"
```

La **cross-attention** es lo que distingue a esta arquitectura. En cada token
que genera, el decoder puede volver a mirar cualquier parte del enunciado
original, ya codificado bidireccionalmente. Qwen tiene que arrastrar esa
información por su propio contexto causal; T5 la tiene disponible siempre.

LoRA se aplica a las proyecciones `q` y `v`, que en T5 aparecen en tres sitios:
la autoatención del encoder, la autoatención del decoder y la cross-attention.
Con `target_modules=["q", "v"]` se cubren los tres.
"""))

    c.append(code('''
MODELO_ID = "google/flan-t5-base"
# Alternativas: "google/flan-t5-small" (80M, más rápido para depurar)
#               "google/flan-t5-large" (780M, no cabe cómodo con LoRA en T4)

DIR_ADAPTADOR = "adaptadores/flan-t5-lora"

LONGITUD_ENTRADA = 128     # se justifica en la sección 6
LONGITUD_SALIDA  = 160
MAX_TOKENS_GEN   = 160

# Si es True, además de normalizar símbolos se eliminan las tildes.
# La sección 6 mide si hace falta.
ELIMINAR_TILDES = True
'''))

    # ------------------------------------------------------------------ 4
    c.append(md("""
## 4 · Carga del dataset
"""))
    c.extend(celda_datos())
    c.extend(celda_carga_datos())
    c.extend(celda_metricas())
    c.extend(celda_resultados())

    # ------------------------------------------------------------------ 5
    c.append(md("""
## 5 · Preprocesamiento

### El formato objetivo

La salida que enseñamos al modelo añade una línea al formato de los notebooks
anteriores:

```
Categoria: porcentajes      <- clasificación implícita, la línea nueva
Paso 1: ...
Paso 2: ...
Respuesta final: 105 estudiantes
```

Ponerla **primero** no es arbitrario. En un decoder autoregresivo, todo lo que
se genera antes condiciona lo que viene después: al escribir primero la
categoría, el modelo se compromete con un tipo de problema y genera los pasos
condicionado por esa decisión. Es el mismo mecanismo del pipeline del notebook
3, pero interno al modelo y entrenado de punta a punta con una sola pérdida.

Como efecto secundario útil, la categoría queda visible en la salida y podemos
extraerla con una expresión regular para medir la accuracy de clasificación y
compararla directamente contra BETO.
"""))

    c.append(code('''
PREFIJO = "resuelve el problema de matematicas paso a paso: "


def construir_entrada(reg):
    return PREFIJO + reg["entrada"]


def construir_objetivo(reg):
    return f"Categoria: {reg['categoria']}\\n{reg['salida']}"


print("ENTRADA:", construir_entrada(train[0]))
print()
print("OBJETIVO:")
print(construir_objetivo(train[0]))
'''))

    # ------------------------------------------------------------------ 6
    c.append(md("""
## 6 · Tokenización

### Diagnóstico: el problema del vocabulario

Esta sección es la contribución técnica propia de este notebook. El tokenizador
de T5 es un SentencePiece de 32k entrenado sobre C4, un corpus en inglés.
Nuestro corpus está en español y usa notación matemática. Antes de entrenar
nada hay que saber cuánto de nuestro texto **el modelo literalmente no puede
representar**.

Un token `<unk>` no es una aproximación: es información destruida. Si `÷` se
convierte en `<unk>`, el modelo no puede distinguir "864 ÷ 12" de "864 × 12".
Ningún fine-tuning arregla eso, porque el dato nunca llega al modelo.
"""))

    c.append(code('''
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODELO_ID)

caracteres = list("¿¡áéíóúüñÁÉÍÓÚÑ×÷√²³°πΔ≈—")
print(f"{'carácter':>10s}  {'tokens':40s}  {'¿se pierde?'}")
print("-" * 72)
perdidos = []
for ch in caracteres:
    piezas = tokenizer.tokenize(ch)
    ids = tokenizer(ch, add_special_tokens=False)["input_ids"]
    es_unk = tokenizer.unk_token_id in ids or piezas in ([], ["▁"], ["▁", "<unk>"])
    if es_unk:
        perdidos.append(ch)
    print(f"{ch:>10s}  {str(piezas):40s}  {'SÍ  <-- se pierde' if es_unk else 'no'}")
print("-" * 72)
print(f"Caracteres que el tokenizador no puede representar: {perdidos}")
'''))

    c.append(code('''
def tasa_unk(tok, textos):
    total = unk = 0
    for t in textos:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        total += len(ids)
        unk += sum(1 for i in ids if i == tok.unk_token_id)
    return unk / max(total, 1), total


textos_crudos = [r["entrada"] + " " + r["salida"] for r in registros]
tasa_antes, n_tokens_antes = tasa_unk(tokenizer, textos_crudos)
print(f"Corpus SIN normalizar: {tasa_antes:.4%} de tokens <unk> "
      f"sobre {n_tokens_antes} tokens")

afectados = sum(1 for t in textos_crudos
                if tokenizer.unk_token_id in tokenizer(t, add_special_tokens=False)["input_ids"])
print(f"Ejemplos con al menos un <unk>: {afectados} de {len(textos_crudos)} "
      f"({afectados/len(textos_crudos):.1%})")
'''))

    c.append(md("""
### La solución: normalización dirigida

Dos caminos posibles:

| Opción | Cómo | Por qué no / por qué sí |
|---|---|---|
| **Ampliar el vocabulario** | `tokenizer.add_tokens([...])` + `model.resize_token_embeddings()` | Los embeddings de los tokens nuevos nacen aleatorios y solo se entrenan con nuestros 132 ejemplos. Un símbolo que aparece 20 veces no alcanza a aprender un embedding útil. Con un corpus grande sería la opción correcta. |
| **Normalizar el texto** ✔ | `÷` → `/`, `×` → `x`, `√` → `raiz`, `²` → `^2` | Reescribe la notación usando caracteres que el modelo **ya entiende bien** por su preentrenamiento en inglés: `/`, `x`, `^` aparecen en matemáticas en cualquier idioma. Sin parámetros nuevos que entrenar. |

Elegimos normalizar. La condición para que sea válido es que la
transformación **preserve el significado**: `864 ÷ 12` y `864 / 12` son la
misma operación, y `√144` y `raiz(144)` también.

Coste que hay que declarar: la normalización se aplica también a la salida, así
que este modelo genera "105 estudiantes" sin tildes y con `x` en lugar de `×`.
Es una diferencia estética frente a Qwen. Para las métricas no importa —ROUGE-L
normaliza tildes y la exactitud compara números—, pero para un producto real
haría falta un paso de post-proceso que devuelva la notación.
"""))

    c.append(code('''
import re
import unicodedata

REEMPLAZOS = {
    "×": " x ", "·": " x ", "÷": " / ", "−": "-", "–": "-", "—": "-",
    "²": "^2", "³": "^3", "⁴": "^4", "√": " raiz ",
    "°": " grados ", "π": " pi ", "≈": " ~ ", "≤": " <= ", "≥": " >= ",
    "¿": "", "¡": "", "“": '"', "”": '"', "’": "'",
}


def normalizar_para_t5(texto, eliminar_tildes=None):
    """Reescribe el texto dentro del vocabulario que FLAN-T5 sí representa."""
    if eliminar_tildes is None:
        eliminar_tildes = ELIMINAR_TILDES
    for viejo, nuevo in REEMPLAZOS.items():
        texto = texto.replace(viejo, nuevo)
    if eliminar_tildes:
        # NFKD separa la letra de su tilde; descartamos las marcas combinantes.
        # La 'ñ' se trataría igual, así que se protege antes.
        texto = texto.replace("ñ", "@N@").replace("Ñ", "@NN@")
        texto = unicodedata.normalize("NFKD", texto)
        texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
        texto = texto.replace("@N@", "ni").replace("@NN@", "NI")
    return re.sub(r"\\s+", " ", texto).strip()


for ejemplo in ["¿Cuál es el área de un círculo? √144 ÷ 2 = 6",
                "Paso 1: Calculamos 5² × 3 y la división 864 ÷ 12."]:
    print("antes :", ejemplo)
    print("después:", normalizar_para_t5(ejemplo))
    print()
'''))

    c.append(code('''
textos_norm = [normalizar_para_t5(t) for t in textos_crudos]
tasa_despues, n_tokens_despues = tasa_unk(tokenizer, textos_norm)

print(f"Tasa de <unk> ANTES  : {tasa_antes:.4%}")
print(f"Tasa de <unk> DESPUÉS: {tasa_despues:.4%}")
print(f"Tokens totales: {n_tokens_antes} -> {n_tokens_despues} "
      f"({100*(n_tokens_despues-n_tokens_antes)/n_tokens_antes:+.1f}%)")

DIAGNOSTICO_TOKENIZADOR = {
    "unk_antes": tasa_antes,
    "unk_despues": tasa_despues,
    "tokens_antes": n_tokens_antes,
    "tokens_despues": n_tokens_despues,
    "caracteres_perdidos": perdidos,
    "eliminar_tildes": ELIMINAR_TILDES,
}
'''))

    c.append(md("""
> **Cómo leer el cambio en el número de tokens.** Si tras normalizar hay *más*
> tokens, no es un fallo: `√144` era un `<unk>` y un número (2 tokens, uno de
> ellos inútil); `raiz(144)` son varios tokens **todos informativos**. Se paga
> longitud a cambio de información. Ese intercambio es exactamente el que un
> tokenizador mal ajustado al idioma impone, y es el argumento cuantitativo de
> la comparación de tokenizadores del notebook 5.
"""))

    c.append(code('''
import numpy as np

entradas_tok = [tokenizer(normalizar_para_t5(construir_entrada(r)))["input_ids"] for r in registros]
salidas_tok  = [tokenizer(normalizar_para_t5(construir_objetivo(r)))["input_ids"] for r in registros]
le, ls = np.array([len(x) for x in entradas_tok]), np.array([len(x) for x in salidas_tok])

print(f"Entrada : media={le.mean():.1f}  p95={np.percentile(le,95):.0f}  max={le.max()}"
      f"   -> truncados con LONGITUD_ENTRADA={LONGITUD_ENTRADA}: {(le>LONGITUD_ENTRADA).sum()}")
print(f"Salida  : media={ls.mean():.1f}  p95={np.percentile(ls,95):.0f}  max={ls.max()}"
      f"   -> truncados con LONGITUD_SALIDA={LONGITUD_SALIDA}: {(ls>LONGITUD_SALIDA).sum()}")

fertilidades = [len(tokenizer(normalizar_para_t5(r["entrada"] + " " + r["salida"]),
                              add_special_tokens=False)["input_ids"])
                / len((r["entrada"] + " " + r["salida"]).split()) for r in registros]

STATS_TOKENIZADOR = {
    "modelo": MODELO_ID,
    "tokenizador": tokenizer.__class__.__name__,
    "vocabulario": int(tokenizer.vocab_size),
    "fertilidad_media": float(np.mean(fertilidades)),
    "tokens_media": float(le.mean() + ls.mean()),
    "tokens_max": int(le.max() + ls.max()),
    **DIAGNOSTICO_TOKENIZADOR,
}
print(f"\\nFertilidad (tokens/palabra, texto normalizado): {np.mean(fertilidades):.3f}")
'''))

    c.append(md("""
### Tokenización del dataset

En seq2seq no hace falta enmascarar el prompt: entrada y salida van a
subredes distintas, y la pérdida se calcula solo sobre el decoder. Lo que sí
hay que hacer es sustituir el padding de las etiquetas por `-100` para que no
contribuya a la pérdida.
"""))

    c.append(code('''
from datasets import Dataset


def tokenizar(reg):
    entrada = normalizar_para_t5(construir_entrada(reg))
    objetivo = normalizar_para_t5(construir_objetivo(reg))

    tok = tokenizer(entrada, truncation=True, max_length=LONGITUD_ENTRADA)
    etiquetas = tokenizer(text_target=objetivo, truncation=True,
                          max_length=LONGITUD_SALIDA)["input_ids"]
    # El padding de las etiquetas debe ignorarse en la pérdida.
    tok["labels"] = [t if t != tokenizer.pad_token_id else -100 for t in etiquetas]
    return tok


ds_train = Dataset.from_list([tokenizar(r) for r in train])
ds_val   = Dataset.from_list([tokenizar(r) for r in val])
print(ds_train)

ej = ds_train[0]
print("\\nEntrada :", tokenizer.decode(ej["input_ids"], skip_special_tokens=True))
print("Objetivo:", tokenizer.decode([t for t in ej["labels"] if t != -100],
                                    skip_special_tokens=True))
'''))

    # ------------------------------------------------------------------ 7
    c.append(md("""
## 7 · Baseline

FLAN-T5 fue afinado con instrucciones, así que responde a "resuelve el problema
paso a paso" sin haber visto nuestro corpus. Esperen dos comportamientos
característicos del baseline:

- **Respuestas muy cortas.** FLAN fue entrenado mayoritariamente con tareas de
  respuesta breve (clasificación, preguntas de una palabra). Su sesgo natural
  es contestar con dos palabras, no con un procedimiento.
- **Ninguna línea `Categoria:`.** Nadie le ha enseñado ese formato todavía. La
  accuracy de clasificación del baseline será prácticamente 0, y eso no
  significa que no entienda el problema: significa que no conoce el formato.

Ambas cosas quedan registradas por separado gracias a `formato_valido`.
"""))

    c.append(code('''
import torch
from transformers import AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

modelo = AutoModelForSeq2SeqLM.from_pretrained(MODELO_ID).to(DEVICE)

n_par = sum(p.numel() for p in modelo.parameters())
print(f"Modelo: {MODELO_ID}")
print(f"Parámetros: {n_par/1e6:.1f} M")
print(f"Bloques encoder: {modelo.config.num_layers} | "
      f"bloques decoder: {modelo.config.num_decoder_layers} | "
      f"d_model: {modelo.config.d_model}")
'''))

    c.append(code('''
@torch.no_grad()
def generar(modelo, entradas, max_new_tokens=MAX_TOKENS_GEN, batch=8):
    """Genera en lotes: T5 admite padding a la izquierda o derecha sin problema
    porque el encoder usa máscara de atención explícita."""
    modelo.eval()
    salidas = []
    for i in tqdm(range(0, len(entradas), batch), desc="generando"):
        textos = [normalizar_para_t5(PREFIJO + e) for e in entradas[i:i + batch]]
        lote = tokenizer(textos, return_tensors="pt", padding=True,
                         truncation=True, max_length=LONGITUD_ENTRADA).to(modelo.device)
        out = modelo.generate(**lote, max_new_tokens=max_new_tokens,
                              do_sample=False, num_beams=1)
        salidas.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return [s.strip() for s in salidas]


gen_baseline = generar(modelo, [r["entrada"] for r in val])
metricas_baseline = evaluar_generacion(gen_baseline, val)

print()
for k, v in metricas_baseline.items():
    if not k.startswith("_"):
        print(f"  {k:16s}: {v}")
'''))

    c.append(code('''
def accuracy_categoria(generados, registros):
    """Accuracy de la clasificación implícita, extraída de la línea 'Categoria:'."""
    aciertos = [categoria_predicha(g) == r["categoria"] for g, r in zip(generados, registros)]
    detectadas = sum(1 for g in generados if categoria_predicha(g) is not None)
    return {
        "accuracy": sum(aciertos) / len(aciertos),
        "categoria_presente": detectadas / len(generados),
        "_aciertos": aciertos,
    }


clf_baseline = accuracy_categoria(gen_baseline, val)
print(f"Clasificación implícita (baseline): accuracy={clf_baseline['accuracy']:.1%}  "
      f"línea 'Categoria:' presente en {clf_baseline['categoria_presente']:.1%} de las salidas")
'''))

    c.append(code('''
gen_demo_baseline = generar(modelo, [d["entrada"] for d in demo])
for d, g in zip(demo, gen_demo_baseline):
    print("=" * 78)
    print(f"[{d['id']} · {d['categoria']}] {d['entrada']}")
    print("-" * 78)
    print("BASELINE:", g[:500] if g else "(salida vacía)")
    print(f"¿Correcta? {respuesta_correcta(g, d['valor'])}  "
          f"¿Formato? {formato_valido(g)}")
print("=" * 78)
'''))

    # ------------------------------------------------------------------ 8
    c.append(md("""
## 8 · Configuración del fine-tuning

### Una advertencia específica de T5: nada de fp16

Los notebooks 1 y 2 entrenan en precisión mixta fp16. **Aquí no.** T5 fue
preentrenado en bfloat16, un formato con el mismo rango exponencial que fp32
pero menos precisión. Sus activaciones internas alcanzan magnitudes que en
fp16 **desbordan a infinito**, y la pérdida se convierte en `NaN` a los pocos
pasos. Es un problema conocido y documentado de la familia T5.

La T4 de Colab no soporta bfloat16 por hardware, así que la opción correcta es
entrenar en fp32. Con 250M de parámetros y LoRA cabe sin dificultad, y de
hecho es más rápido que depurar `NaN`s.

Esta diferencia obligada entre arquitecturas es en sí misma un resultado del
laboratorio: reportarla es parte de la comparación de "facilidad de
fine-tuning".

### LoRA en T5

`target_modules=["q", "v"]` alcanza los tres tipos de atención del modelo
(encoder, decoder y cross-attention), porque las tres usan proyecciones con
esos nombres. Rango 16 y alpha 32, igual que en Qwen, para que la comparación
entre arquitecturas no esté contaminada por hiperparámetros distintos.
"""))

    c.extend(celda_args_compatibles())

    c.append(code('''
from peft import LoraConfig, get_peft_model

config_lora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],    # cubre encoder, decoder y cross-attention
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

modelo = get_peft_model(modelo, config_lora)
modelo.print_trainable_parameters()
'''))

    c.extend(celda_wandb("flan-t5-lora-finetune",
                         '["flan-t5", "encoder-decoder", "lora", "generacion", "clasificacion"]'))

    c.append(code('''
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=modelo,               # construye decoder_input_ids desplazando labels
    label_pad_token_id=-100,
    padding=True,
    return_tensors="pt",
)

args = construir_args(
    clase=Seq2SeqTrainingArguments,
    output_dir=f"{DIR_CHECKPOINTS}/flan-t5-lora",
    run_name=NOMBRE_RUN,

    num_train_epochs=15,        # más que Qwen: el modelo es menor y parte de más lejos
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=8,

    learning_rate=3e-4,         # T5 con LoRA tolera algo más que Qwen
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,

    fp16=False,                 # IMPRESCINDIBLE: fp16 rompe T5 (ver arriba)
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to=REPORTAR_A,
    seed=SEMILLA,
    predict_with_generate=False,   # la generación se evalúa aparte, con control total
)

trainer = Seq2SeqTrainer(
    model=modelo,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=collator,
)

pasos_epoca = len(ds_train) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
print(f"Pasos de optimización por época: {pasos_epoca}")
print(f"Pasos totales: {int(args.num_train_epochs) * pasos_epoca}")
'''))

    # ------------------------------------------------------------------ 9
    c.append(md("""
## 9 · Entrenamiento

Qué esperar, y en qué se diferencia de las curvas de Qwen:

- La pérdida inicial será **más alta** que la de Qwen. Es normal: T5 no solo
  aprende el formato de la respuesta, también aprende a emitir la línea
  `Categoria:`, que es una tarea adicional.
- El descenso suele ser **más suave y sostenido**. Un modelo de 250M con LoRA
  tiene menos capacidad para memorizar 132 ejemplos de golpe, así que el
  sobreajuste aparece más tarde que en Qwen.
- Si ven `nan` en la pérdida, revisen que `fp16=False`. Es la causa en la
  práctica totalidad de los casos.
"""))

    c.append(code('''
resultado_entrenamiento = trainer.train()
print()
print(f"Tiempo de entrenamiento: {resultado_entrenamiento.metrics['train_runtime']:.1f} s")
print(f"Pérdida final: {resultado_entrenamiento.metrics['train_loss']:.4f}")
'''))

    c.append(code('''
import pandas as pd

historial = pd.DataFrame(trainer.state.log_history)
ev = historial.dropna(subset=["eval_loss"])
print(ev[["epoch", "eval_loss"]].to_string(index=False))

mejor = ev.sort_values("eval_loss").iloc[0]
print(f"\\nMejor época: {mejor['epoch']:.0f}  (eval_loss = {mejor['eval_loss']:.4f})")
'''))

    c.append(code('''
import matplotlib.pyplot as plt

tr = historial.dropna(subset=["loss"])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(tr["epoch"], tr["loss"], label="Entrenamiento", alpha=0.8)
ax.plot(ev["epoch"], ev["eval_loss"], label="Validación", marker="o")
ax.axvline(mejor["epoch"], ls="--", c="gray", lw=1, label=f"Mejor época ({mejor['epoch']:.0f})")
ax.set_xlabel("Época"); ax.set_ylabel("Pérdida")
ax.set_title("FLAN-T5-base + LoRA · curvas de pérdida")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''))

    c.append(code('''
# Tras `load_best_model_at_end` el modelo bueno es `trainer.model`: el Trainer
# recarga el mejor checkpoint, y evaluar sobre `modelo` usaría la última época.
modelo = trainer.model

modelo.save_pretrained(DIR_ADAPTADOR)
tokenizer.save_pretrained(DIR_ADAPTADOR)
print(f"Adaptador guardado en {DIR_ADAPTADOR}/")
'''))

    # ------------------------------------------------------------------ 10
    c.append(md("""
## 10 · Integración con Weights & Biases

Mismo proyecto y misma convención que los otros tres notebooks, para que las
curvas sean superponibles en el panel:

- **Proyecto:** `tutor-matematicas-arquitecturas`
- **Run:** `flan-t5-lora-finetune`
- **Tags:** `flan-t5`, `encoder-decoder`, `lora`, `generacion`, `clasificacion`

Lo específico de este notebook que se registra a mano: el diagnóstico del
tokenizador (tasa de `<unk>` antes y después de normalizar) y la accuracy de
clasificación implícita, que es lo que permite comparar este modelo contra
BETO y contra el pipeline.
"""))

    # ------------------------------------------------------------------ 11
    c.append(md("""
## 11 · Evaluación

Evaluamos **las dos capacidades por separado**, porque el modelo hace dos
trabajos y un promedio conjunto no diría nada útil:

1. **Generación** — mismas métricas que los notebooks 1 y 3.
2. **Clasificación implícita** — accuracy sobre la línea `Categoria:`,
   directamente comparable con el notebook 2.
"""))

    c.append(code('''
gen_finetuned = generar(modelo, [r["entrada"] for r in val])
metricas_finetuned = evaluar_generacion(gen_finetuned, val)
clf_finetuned = accuracy_categoria(gen_finetuned, val)

print("Generación")
for k, v in metricas_finetuned.items():
    if not k.startswith("_"):
        print(f"  {k:16s}: {v}")
print()
print("Clasificación implícita")
print(f"  accuracy            : {clf_finetuned['accuracy']:.1%}")
print(f"  categoría presente  : {clf_finetuned['categoria_presente']:.1%}")
'''))

    c.append(md("""
### ¿Se ayudan las dos tareas entre sí?

La hipótesis detrás de escribir la categoría primero es que condiciona
favorablemente la generación posterior. Se puede comprobar: comparen la
exactitud de la respuesta en los ejemplos donde el modelo clasificó bien contra
aquellos donde clasificó mal.

Es el mismo análisis de propagación de error del notebook 3, pero **dentro de
un solo modelo**. Y por eso mismo hay que interpretarlo con cuidado: aquí no
hay causalidad garantizada. Un ejemplo difícil puede provocar a la vez una
clasificación errónea y una respuesta errónea, sin que una cause la otra.
"""))

    c.append(code('''
ok_clf = clf_finetuned["_aciertos"]
ok_gen = metricas_finetuned["_correctas"]

g_ok  = [g for g, c in zip(ok_gen, ok_clf) if c]
g_mal = [g for g, c in zip(ok_gen, ok_clf) if not c]

print("Coherencia entre las dos tareas (modelo integrado)")
print("=" * 66)
print(f"Clasificó BIEN  n={len(g_ok):2d}  exactitud de la respuesta = "
      f"{sum(g_ok)/max(len(g_ok),1):.1%}")
print(f"Clasificó MAL   n={len(g_mal):2d}  exactitud de la respuesta = "
      f"{sum(g_mal)/max(len(g_mal),1):.1%}")
print("=" * 66)
if len(g_mal) < 5:
    print(f"AVISO: solo {len(g_mal)} ejemplos mal clasificados; la cifra es anecdótica.")
'''))

    c.append(code('''
from collections import defaultdict

por_cat = defaultdict(lambda: {"n": 0, "base": 0, "ft": 0})
for r, b, f in zip(val, metricas_baseline["_correctas"], metricas_finetuned["_correctas"]):
    d = por_cat[r["categoria"]]
    d["n"] += 1; d["base"] += int(b); d["ft"] += int(f)

print(f"{'categoría':26s} {'n':>3s} {'baseline':>10s} {'fine-tuned':>12s}")
print("-" * 56)
for cat in CATEGORIAS:
    d = por_cat.get(cat)
    if d:
        print(f"{cat:26s} {d['n']:3d} {d['base']/d['n']:>9.0%} {d['ft']/d['n']:>11.0%}")

EXACTITUD_POR_CATEGORIA = {
    cat: {"n": d["n"], "baseline": d["base"]/d["n"], "finetuned": d["ft"]/d["n"]}
    for cat, d in por_cat.items()
}
'''))

    # ------------------------------------------------------------------ 12
    c.append(md("""
## 12 · Comparación Baseline vs Fine-tuned
"""))

    c.append(code('''
tabla_metricas(metricas_baseline, metricas_finetuned,
               titulo=f"FLAN-T5-base · {len(val)} ejemplos de validación")

print()
print(f"Clasificación implícita: {clf_baseline['accuracy']:.1%} -> "
      f"{clf_finetuned['accuracy']:.1%}  "
      f"({clf_finetuned['accuracy'] - clf_baseline['accuracy']:+.1%})")
'''))

    c.append(code('''
gen_demo_ft = generar(modelo, [d["entrada"] for d in demo])

for d, antes, despues in zip(demo, gen_demo_baseline, gen_demo_ft):
    print("=" * 78)
    print(f"[{d['id']} · {d['categoria']}] {d['entrada']}")
    print("-" * 78)
    print("ANTES:", antes[:400] if antes else "(vacío)")
    print("-" * 78)
    print("DESPUÉS:")
    print(despues[:500])
    print("-" * 78)
    print(f"Categoría predicha: {categoria_predicha(despues)} (real: {d['categoria']})")
    print(f"Correcta -> antes: {respuesta_correcta(antes, d['valor'])} | "
          f"después: {respuesta_correcta(despues, d['valor'])}")
print("=" * 78)
'''))

    c.append(code('''
RESULTADOS_T5 = {
    "notebook": "S04_Lab_Fine_tuning_FLAN_T5",
    "arquitectura": "encoder-decoder",
    "modelo": MODELO_ID,
    "metodo": "LoRA",
    "n_train": len(train),
    "n_val": len(val),
    "epocas": float(args.num_train_epochs),
    "learning_rate": args.learning_rate,
    "precision": "fp32 (fp16 desborda en T5)",
    "tiempo_entrenamiento_s": resultado_entrenamiento.metrics["train_runtime"],
    "train_loss_final": resultado_entrenamiento.metrics["train_loss"],
    "eval_loss_mejor": float(mejor["eval_loss"]),
    "mejor_epoca": float(mejor["epoch"]),
    "parametros_entrenables": sum(p.numel() for p in modelo.parameters() if p.requires_grad),
    "parametros_totales": n_par,
    "baseline": metricas_baseline,
    "finetuned": metricas_finetuned,
    "clasificacion_baseline": {k: v for k, v in clf_baseline.items() if not k.startswith("_")},
    "clasificacion_finetuned": {k: v for k, v in clf_finetuned.items() if not k.startswith("_")},
    "coherencia_tareas": {
        "n_clf_ok": len(g_ok),
        "exactitud_clf_ok": sum(g_ok) / max(len(g_ok), 1),
        "n_clf_mal": len(g_mal),
        "exactitud_clf_mal": sum(g_mal) / max(len(g_mal), 1),
    },
    "por_categoria": EXACTITUD_POR_CATEGORIA,
    "tokenizador": STATS_TOKENIZADOR,
}

guardar_resultados("flan_t5", RESULTADOS_T5)
'''))

    c.append(code('''
if USAR_WANDB:
    import wandb

    if wandb.run is None:
        wandb.init(project=PROYECTO, name=NOMBRE_RUN, tags=TAGS, reinit=True)

    tabla = wandb.Table(columns=["id", "entrada", "referencia", "baseline", "finetuned",
                                 "cat_real", "cat_predicha", "ok_baseline", "ok_finetuned"])
    for r, b, f, okb, okf in zip(val, gen_baseline, gen_finetuned,
                                 metricas_baseline["_correctas"],
                                 metricas_finetuned["_correctas"]):
        tabla.add_data(r["id"], r["entrada"], r["salida"], b, f,
                       r["categoria"], categoria_predicha(f), okb, okf)
    wandb.log({"evaluacion/generaciones": tabla})

    wandb.summary.update({
        "tokenizador/unk_antes": tasa_antes,
        "tokenizador/unk_despues": tasa_despues,
        "baseline/exactitud": metricas_baseline["exactitud"],
        "baseline/formato_valido": metricas_baseline["formato_valido"],
        "baseline/clf_accuracy": clf_baseline["accuracy"],
        "finetuned/exactitud": metricas_finetuned["exactitud"],
        "finetuned/formato_valido": metricas_finetuned["formato_valido"],
        "finetuned/clf_accuracy": clf_finetuned["accuracy"],
        "delta/exactitud": metricas_finetuned["exactitud"] - metricas_baseline["exactitud"],
    })
    wandb.finish()
    print("Registro en W&B completado.")
else:
    print("W&B desactivado; las métricas quedaron en resultados/flan_t5.json")
'''))

    # ------------------------------------------------------------------ 13
    c.append(md("""
## 13 · Discusión

**1. ¿Cuánto costó el tokenizador?**
Es la pregunta propia de este notebook. Con la tasa de `<unk>` antes y después
de normalizar pueden estimar cuánta información se habría perdido sin
tratamiento. Y aun normalizado, comparen la fertilidad (tokens por palabra) con
la de Qwen: si FLAN-T5 necesita bastantes más tokens para el mismo texto en
español, cada ejemplo consume más contexto y más cómputo por la sola razón de
que el vocabulario no fue diseñado para este idioma.

**2. ¿Un modelo integrado o dos especializados?**
Contrasten la exactitud de este notebook contra la configuración B del notebook
3, y la accuracy de clasificación contra BETO. Los escenarios posibles:

| Resultado | Lectura |
|---|---|
| T5 gana en ambas | El modelo integrado domina. La simplicidad operativa se suma a la calidad: decisión fácil. |
| T5 clasifica peor pero genera igual | Esperable: BETO es un especialista con atención bidireccional pura y una cabeza dedicada. La pregunta pasa a ser si la clasificación importa para el resultado final. |
| T5 pierde en ambas | Probablemente sea cuestión de tamaño (250M vs 1.5B) y de tokenizador, no de arquitectura. Habría que repetirlo con `flan-t5-large` o con mT5 para separar los factores. |

**3. ¿La clasificación implícita ayuda a generar?**
Miren el análisis de coherencia. Y recuerden la advertencia: correlación no es
causalidad, ambos errores pueden compartir causa (un problema difícil).

**4. ¿Qué limita a este modelo?**
Tres factores confundidos que conviene nombrar por separado: tamaño (250M),
tokenizador (inglés), y preentrenamiento matemático (más débil que Qwen). Con
un solo experimento no se pueden separar, y decirlo es más honesto que atribuir
el resultado a "la arquitectura encoder-decoder".
"""))

    # ------------------------------------------------------------------ 14
    c.append(md("""
## 14 · Conclusiones

1. **El tokenizador es un criterio de selección de modelo, no un detalle de
   implementación.** Es la lección más transferible del laboratorio: antes de
   elegir un modelo para un idioma o dominio concreto, midan cómo tokeniza sus
   datos. Cuesta cinco minutos y puede ahorrar un experimento entero.
2. **Un modelo integrado elimina toda una clase de fallos.** Sin desalineación
   de etiquetas, sin propagación en cascada, sin dos artefactos que versionar.
3. **La arquitectura encoder-decoder es conceptualmente la adecuada** para
   entrada estructurada → salida estructurada. Si pierde en las métricas, hay
   que verificar si es por la arquitectura o por el modelo concreto elegido.
4. **Todo lo anterior se cuantifica en el notebook 5**, que agrega los cuatro
   `resultados/*.json` y produce la comparación final.
"""))

    return c
