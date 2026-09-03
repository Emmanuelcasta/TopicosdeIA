"""Notebook 1 — Qwen (decoder-only) como generador de soluciones."""

from __future__ import annotations

from nb_comun import (
    celda_args_compatibles, celda_carga_datos, celda_datos, celda_instalacion,
    celda_drive, celda_metricas, celda_resultados, celda_wandb, code,
    encabezado, md,
)


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "Fine-tuning de Qwen2.5 para un tutor de matemáticas",
        "Notebook 1 de 4 · Arquitectura decoder-only (generación)",
    ))

    # ------------------------------------------------------------------ 1
    c.append(md("""
## 1 · Introducción

### Qué problema resolvemos

Un estudiante escribe una pregunta de matemáticas en lenguaje natural
("María tiene 48 caramelos y quiere repartirlos entre 6 amigos…") o una
operación directa ("(45 - 9) ÷ 6 + 11"). El sistema debe devolver **el
procedimiento paso a paso y la respuesta**, no solo el número: un tutor que
solo da resultados no enseña.

### Qué arquitectura usamos aquí

**Qwen2.5-1.5B-Instruct**, un modelo *decoder-only* de 1.5 mil millones de
parámetros. Es la arquitectura de los modelos tipo GPT: procesa el texto de
izquierda a derecha y predice el siguiente token, condicionado únicamente por
lo anterior (*atención causal*).

### Por qué este modelo y no otro

| Criterio | Justificación |
|---|---|
| **Es generativo** | La tarea exige producir texto libre (una explicación). Un encoder como BERT no puede hacerlo: solo produce representaciones o etiquetas. |
| **Tamaño** | 1.5B entra en una GPU T4 gratuita con LoRA y entrena en minutos. Un modelo de 7B necesitaría cuantización agresiva y mucho más tiempo. |
| **Multilingüe real** | Qwen2.5 fue preentrenado con volumen significativo de español, algo poco común en modelos pequeños abiertos. Nuestro corpus es 100% español. |
| **Variante `Instruct`** | Ya fue alineada para seguir instrucciones, así que el *baseline* es un punto de comparación honesto. Con un modelo base puro, casi cualquier fine-tuning parecería un éxito espectacular. |

> **Alternativa considerada:** `Qwen/Qwen2.5-Math-1.5B`, especializado en
> matemáticas. Se descartó como opción por defecto porque su razonamiento en
> cadena está optimizado para inglés y chino, y produce explicaciones
> inconsistentes en español. Queda disponible cambiando una línea en la
> celda de configuración, por si quieren correr el experimento con ambos.

### Ventajas y limitaciones de la arquitectura decoder-only

**Ventajas**
- Genera texto de longitud arbitraria: procedimientos, explicaciones, ejemplos.
- Un solo modelo cubre todos los tipos de problema; no hay que decidir
  previamente de qué tema es la pregunta.
- Se adapta bien con pocos datos cuando la tarea es de *formato* y *estilo*.

**Limitaciones**
- No tiene garantía de corrección aritmética: el cálculo emerge de patrones
  estadísticos, no de un motor de cálculo. Puede escribir un procedimiento
  impecable y equivocarse en la última multiplicación.
- Es el modelo más caro de los cuatro en inferencia (genera token a token).
- Su salida es difícil de validar automáticamente: por eso invertimos en un
  formato de respuesta estructurado.
"""))

    # ------------------------------------------------------------------ 2
    c.append(md("""
## 2 · Objetivos

1. Medir el desempeño del modelo **antes** de entrenarlo (baseline), sobre la
   partición de validación y con métricas definidas de antemano.
2. Aplicar fine-tuning eficiente con **LoRA** sobre 132 ejemplos de
   entrenamiento, con QLoRA disponible como variante.
3. Evaluar el modelo entrenado **con las mismas métricas y los mismos datos**,
   y cuantificar la diferencia.
4. Distinguir explícitamente dos efectos que el fine-tuning con pocos datos
   mezcla: cuánto mejoró el **formato** de la respuesta y cuánto mejoró la
   **corrección matemática**.
5. Dejar registrado en W&B y en `resultados/qwen.json` todo lo necesario para
   la comparación final entre las cuatro arquitecturas.
"""))

    # ------------------------------------------------------------------ 0
    c.extend(celda_instalacion())
    c.extend(celda_drive())

    # ------------------------------------------------------------------ 3
    c.append(md("""
## 3 · Arquitectura del modelo

```
                    ENTRADA (prompt tokenizado)
                              |
                    +---------v----------+
                    |  Embeddings + RoPE |
                    +---------v----------+
                              |
                   +----------v-----------+
                   |  28 bloques Qwen2:   |
                   |   - Atención causal  |   <- cada token solo ve
                   |     (GQA, 12 heads)  |      los anteriores
                   |   - MLP SwiGLU       |
                   |   - RMSNorm          |
                   +----------v-----------+
                              |
                    +---------v----------+
                    |  Cabeza de lenguaje |  -> distribución sobre 151k tokens
                    +---------------------+
                              |
                  se muestrea 1 token, se reinyecta, y se repite
```

Lo esencial para entender el fine-tuning: **la máscara causal**. Cada posición
solo puede atender a las posiciones anteriores. Eso es lo que permite generar
texto, y también lo que impide que el modelo "lea la respuesta" durante el
entrenamiento.

Dónde entra LoRA: en las matrices de proyección de la atención
(`q_proj`, `k_proj`, `v_proj`, `o_proj`). En lugar de actualizar una matriz
`W` de tamaño `d×d`, se congela `W` y se aprende `ΔW = B·A`, donde `A` es
`r×d` y `B` es `d×r` con `r` muy pequeño (16 en este notebook). Se entrena
menos del 1% de los parámetros.
"""))

    c.append(code('''
MODELO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# Alternativa especializada en matemáticas (razonamiento en inglés/chino):
# MODELO_ID = "Qwen/Qwen2.5-Math-1.5B"

USAR_QLORA = False   # True = cargar el modelo en 4 bits (menos memoria, algo más lento)

DIR_ADAPTADOR = "adaptadores/qwen-lora"
LONGITUD_MAX  = 384       # se justifica en la sección 6 con datos reales
MAX_TOKENS_GEN = 200      # techo de generación durante la evaluación
'''))

    c.append(code('''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODELO_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if USAR_QLORA:
    from transformers import BitsAndBytesConfig
    cuantizacion = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # cuantización de 4 bits con distribución normal
        bnb_4bit_compute_dtype=torch.float16,   # los cálculos siguen en fp16
        bnb_4bit_use_double_quant=True,         # cuantiza también las constantes de cuantización
    )
    modelo = AutoModelForCausalLM.from_pretrained(
        MODELO_ID, quantization_config=cuantizacion, device_map="auto"
    )
else:
    modelo = AutoModelForCausalLM.from_pretrained(
        MODELO_ID, torch_dtype=torch.float16
    ).to(DEVICE)

n_par = sum(p.numel() for p in modelo.parameters())
print(f"Modelo: {MODELO_ID}")
print(f"Parámetros: {n_par/1e6:.1f} M")
print(f"Capas ocultas: {modelo.config.num_hidden_layers} | "
      f"dimensión: {modelo.config.hidden_size} | "
      f"vocabulario: {modelo.config.vocab_size}")
'''))

    # ------------------------------------------------------------------ 4
    c.append(md("""
## 4 · Carga del dataset
"""))
    c.extend(celda_datos())
    c.extend(celda_carga_datos())

    # ------------------------------------------------------------------ 5
    c.append(md("""
## 5 · Preprocesamiento

### La plantilla de prompt

Todo ejemplo se convierte al formato de conversación que Qwen2.5-Instruct
espera (`<|im_start|>system … <|im_start|>user … <|im_start|>assistant`).
Usar `apply_chat_template` en vez de concatenar texto a mano importa: el
modelo fue alineado con esos tokens especiales, y saltárselos degrada el
baseline de forma artificial.

La misma plantilla se usa en el baseline, en el entrenamiento y en la
evaluación posterior. Si el prompt cambiara entre fases, estaríamos midiendo
el efecto del prompt y no el del fine-tuning.

### El enmascaramiento de la pérdida

Este es el punto técnico más importante de la sección. Un ejemplo tokenizado
contiene el prompt y la respuesta. Si calculamos la pérdida sobre **todos** los
tokens, el modelo dedica capacidad a aprender a predecir el enunciado del
problema, que es exactamente lo que no queremos: en producción el enunciado
lo escribe el usuario.

La solución estándar es poner `-100` en las etiquetas de los tokens del prompt.
PyTorch ignora esas posiciones al calcular la entropía cruzada, y el gradiente
se concentra en la respuesta.

```
tokens :  [<|im_start|>system ... usuario: 48 ÷ 6 ...][ Paso 1: ... Respuesta final: 8 ]
labels :  [ -100  -100  -100  -100  -100  -100  -100 ][ 30821  25 ... 23   ]
           \\________ ignorado por la pérdida ________/ \\____ aquí se aprende ____/
```
"""))

    c.append(code('''
INSTRUCCION = (
    "Eres un tutor de matemáticas. Resuelve el siguiente problema explicando "
    "el procedimiento paso a paso y termina con la respuesta final."
)


def construir_prompt(entrada):
    """Prompt de inferencia: termina justo donde el modelo debe empezar a escribir."""
    mensajes = [
        {"role": "system", "content": INSTRUCCION},
        {"role": "user", "content": entrada},
    ]
    return tokenizer.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )


print(construir_prompt("Resuelve la siguiente operación: 36 × 24."))
'''))

    # ------------------------------------------------------------------ 6
    c.append(md("""
## 6 · Tokenización

Antes de fijar `LONGITUD_MAX` medimos cuántos tokens ocupan realmente nuestros
ejemplos. Elegir el número a ojo tiene dos costos: si se queda corto trunca
respuestas (el modelo aprendería a no terminar nunca), y si se pasa
desperdicia cómputo en padding.

También calculamos la **fertilidad** del tokenizador: tokens por palabra. Es
la métrica que usaremos en el notebook de comparación para contrastar los
tokenizadores de Qwen (BPE, 151k), BETO (WordPiece, 31k) y FLAN-T5
(SentencePiece, 32k) sobre el mismo texto en español.
"""))

    c.append(code('''
import numpy as np

longitudes, fertilidades = [], []
for r in registros:
    completo = construir_prompt(r["entrada"]) + r["salida"] + tokenizer.eos_token
    ids = tokenizer(completo, add_special_tokens=False)["input_ids"]
    longitudes.append(len(ids))
    texto_plano = r["entrada"] + " " + r["salida"]
    n_tok = len(tokenizer(texto_plano, add_special_tokens=False)["input_ids"])
    fertilidades.append(n_tok / len(texto_plano.split()))

longitudes = np.array(longitudes)
print(f"Tokens por ejemplo (prompt + respuesta):")
print(f"  min={longitudes.min()}  media={longitudes.mean():.1f}  "
      f"p95={np.percentile(longitudes, 95):.0f}  max={longitudes.max()}")
print(f"  ejemplos que excederían LONGITUD_MAX={LONGITUD_MAX}: "
      f"{(longitudes > LONGITUD_MAX).sum()}")
print()
print(f"Fertilidad (tokens/palabra en español): {np.mean(fertilidades):.3f}")
print(f"Tamaño del vocabulario: {tokenizer.vocab_size}")

STATS_TOKENIZADOR = {
    "modelo": MODELO_ID,
    "tokenizador": tokenizer.__class__.__name__,
    "vocabulario": int(tokenizer.vocab_size),
    "fertilidad_media": float(np.mean(fertilidades)),
    "tokens_media": float(longitudes.mean()),
    "tokens_max": int(longitudes.max()),
}
'''))

    c.append(md("""
### Inspección cualitativa

Ver cómo se parte un texto concreto explica más que cualquier estadística.
Presten atención a los símbolos matemáticos (`÷`, `×`, `²`, `√`): son el punto
donde los tres tokenizadores del proyecto se comportan de forma más distinta.
"""))

    c.append(code('''
muestras = [
    "Resuelve la siguiente operación: 864 ÷ 12.",
    "¿Cuál es el área de un círculo de radio 5? √144 + 5² = 37",
    "Paso 1: Repartir en partes iguales es dividir.",
]
for m in muestras:
    ids = tokenizer(m, add_special_tokens=False)["input_ids"]
    piezas = [tokenizer.decode([i]) for i in ids]
    print(f"\\nTexto ({len(ids)} tokens): {m}")
    print("   " + " | ".join(piezas))

desconocidos = sum(
    1 for m in muestras
    for i in tokenizer(m, add_special_tokens=False)["input_ids"]
    if i == tokenizer.unk_token_id
)
print(f"\\nTokens desconocidos (<unk>): {desconocidos}")
'''))

    c.append(md("""
### Tokenización del dataset con enmascaramiento
"""))

    c.append(code('''
from datasets import Dataset


def tokenizar(reg):
    prompt = construir_prompt(reg["entrada"])
    completo = prompt + reg["salida"] + tokenizer.eos_token

    ids_prompt = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tok = tokenizer(
        completo, add_special_tokens=False,
        truncation=True, max_length=LONGITUD_MAX,
    )

    # -100 en el prompt: la pérdida solo se calcula sobre la respuesta.
    etiquetas = list(tok["input_ids"])
    for i in range(min(len(ids_prompt), len(etiquetas))):
        etiquetas[i] = -100
    tok["labels"] = etiquetas
    return tok


ds_train = Dataset.from_list([tokenizar(r) for r in train])
ds_val   = Dataset.from_list([tokenizar(r) for r in val])

print(ds_train)
ejemplo = ds_train[0]
n_ignorados = sum(1 for e in ejemplo["labels"] if e == -100)
print(f"\\nEjemplo 0: {len(ejemplo['input_ids'])} tokens, "
      f"{n_ignorados} enmascarados (prompt), "
      f"{len(ejemplo['labels']) - n_ignorados} con gradiente (respuesta).")
print("\\nTexto sobre el que realmente se aprende:")
print(tokenizer.decode([i for i, e in zip(ejemplo["input_ids"], ejemplo["labels"]) if e != -100]))
'''))

    c.append(md("""
Usamos **padding dinámico**: cada lote se rellena hasta el ejemplo más largo
de ese lote, no hasta 384. Con longitudes tan dispares (de ~60 a ~200 tokens)
esto reduce el cómputo desperdiciado a la mitad frente a `padding="max_length"`.
El *collator* rellena `input_ids` con el token de padding y `labels` con
`-100`, que es justo lo que necesitamos.
"""))

    c.append(code('''
from transformers import DataCollatorForSeq2Seq

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,   # el padding de las etiquetas también se ignora
    return_tensors="pt",
)
print("Collator listo (padding dinámico por lote).")
'''))

    # ------------------------------------------------------------------ métricas
    c.extend(celda_metricas())
    c.extend(celda_resultados())

    # ------------------------------------------------------------------ 7
    c.append(md("""
## 7 · Baseline

**La pieza más importante del experimento.** Sin una medición previa no se
puede afirmar que el fine-tuning sirvió: cualquier resultado posterior sería
un número sin referencia.

Condiciones de la medición, idénticas a las que usaremos después del
entrenamiento:

- Mismos 33 ejemplos de validación (nunca vistos en entrenamiento).
- Mismo prompt, misma plantilla de chat.
- **Decodificación greedy** (`do_sample=False`): sin aleatoriedad. Si
  muestreáramos, dos ejecuciones darían métricas distintas y no sabríamos si
  la diferencia viene del modelo o del azar.
- Mismo techo de tokens generados.
"""))

    c.append(code('''
from tqdm.auto import tqdm


@torch.no_grad()
def generar(modelo, entradas, max_new_tokens=MAX_TOKENS_GEN):
    """Genera una respuesta por entrada. Decodifica solo los tokens NUEVOS.

    Detalle que suele producir errores: recortar por longitud de caracteres del
    prompt es frágil (el detokenizador no siempre reconstruye el prompt
    carácter a carácter). Recortar por número de tokens de entrada es exacto.
    """
    modelo.eval()
    salidas = []
    for entrada in tqdm(entradas, desc="generando"):
        prompt = construir_prompt(entrada)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(modelo.device)
        out = modelo.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        nuevos = out[0][inputs["input_ids"].shape[1]:]
        salidas.append(tokenizer.decode(nuevos, skip_special_tokens=True).strip())
    return salidas


gen_baseline = generar(modelo, [r["entrada"] for r in val])
metricas_baseline = evaluar_generacion(gen_baseline, val)

print()
for k, v in metricas_baseline.items():
    if not k.startswith("_"):
        print(f"  {k:16s}: {v}")
'''))

    c.append(md("""
### Inspección cualitativa del baseline

Los números resumen; los ejemplos explican. Estas cinco preguntas son las
mismas que revisaremos después del fine-tuning.
"""))

    c.append(code('''
gen_demo_baseline = generar(modelo, [d["entrada"] for d in demo])

for d, g in zip(demo, gen_demo_baseline):
    print("=" * 78)
    print(f"[{d['id']} · {d['categoria']}] {d['entrada']}")
    print("-" * 78)
    print("BASELINE:")
    print(g[:700])
    print("-" * 78)
    print(f"Esperado: {d['salida']}")
    print(f"¿Respuesta correcta? {respuesta_correcta(g, d['valor'])}   "
          f"¿Formato válido? {formato_valido(g)}")
print("=" * 78)
'''))

    c.append(md("""
> **Anoten qué falla exactamente.** Casi siempre son tres cosas distintas:
> (a) no usa el formato `Paso N` / `Respuesta final`, (b) se extiende de más o
> inventa preguntas adicionales, (c) se equivoca en la aritmética. El
> fine-tuning con 132 ejemplos arregla (a) y (b) con facilidad; (c) es el
> problema difícil y es donde hay que mirar con lupa.
"""))

    # ------------------------------------------------------------------ 8
    c.append(md("""
## 8 · Configuración del fine-tuning

### LoRA: los tres números que importan

| Hiperparámetro | Valor | Razón |
|---|---|---|
| `r` (rango) | 16 | Capacidad del adaptador. Con 132 ejemplos, un rango alto (64+) memoriza; uno muy bajo (4) no alcanza a aprender el formato. 16 es el punto medio habitual para datasets pequeños. |
| `lora_alpha` | 32 | Factor de escala: el aporte del adaptador se multiplica por `alpha/r` = 2. La convención `alpha = 2r` funciona bien y evita tener que ajustar dos cosas a la vez. |
| `target_modules` | proyecciones de atención | Es donde el modelo decide *a qué atiende*. Añadir las capas MLP (`gate_proj`, `up_proj`, `down_proj`) da algo más de capacidad a costa de más parámetros y más riesgo de sobreajuste con este tamaño de corpus. |
| `lora_dropout` | 0.05 | Regularización ligera. Con datasets pequeños ayuda; subirlo mucho hace la pérdida ruidosa e ilegible. |

### Hiperparámetros de entrenamiento

Con 132 ejemplos, las decisiones se toman al revés que con datasets grandes:
el riesgo no es no aprender, es **memorizar**. Por eso evaluamos cada época
sobre validación y nos quedamos con el mejor punto, no con el último.
"""))

    c.extend(celda_args_compatibles())

    c.append(code('''
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if USAR_QLORA:
    modelo = prepare_model_for_kbit_training(modelo)

config_lora = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

modelo = get_peft_model(modelo, config_lora)

# Los pesos base están en fp16, pero los parámetros ENTRENABLES deben estar en
# fp32: con entrenamiento en precisión mixta, mantener los pesos maestros en
# fp16 provoca subdesbordamiento del gradiente y la pérdida se va a NaN.
for nombre, param in modelo.named_parameters():
    if param.requires_grad:
        param.data = param.data.float()

modelo.print_trainable_parameters()
'''))

    c.extend(celda_wandb("qwen-lora-finetune",
                         '["qwen", "decoder-only", "lora", "generacion"]'))

    c.append(code('''
from transformers import Trainer

args = construir_args(
    output_dir=f"{DIR_CHECKPOINTS}/qwen-lora",
    run_name=NOMBRE_RUN,

    num_train_epochs=8,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,       # lote efectivo = 8
    per_device_eval_batch_size=2,

    learning_rate=2e-4,                  # alto a propósito: LoRA lo tolera
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,

    fp16=(DEVICE == "cuda"),
    logging_steps=5,
    eval_strategy="epoch",               # curva de validación por época
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,         # nos quedamos con la mejor época
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to=REPORTAR_A,
    seed=SEMILLA,
)

trainer = Trainer(
    model=modelo,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=collator,
)

print(f"Pasos de optimización por época: "
      f"{len(ds_train) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)}")
print(f"Pasos totales estimados: {int(args.num_train_epochs) * (len(ds_train) // (args.per_device_train_batch_size * args.gradient_accumulation_steps))}")
'''))

    # ------------------------------------------------------------------ 9
    c.append(md("""
## 9 · Entrenamiento

### Cómo leer lo que va apareciendo

- **`loss`** (entrenamiento) debe bajar de forma sostenida. Si oscila sin
  tendencia, el learning rate es demasiado alto.
- **`eval_loss`** (validación) es la que de verdad importa. Mientras baje, el
  modelo generaliza. Cuando empieza a subir mientras `loss` sigue bajando,
  eso es **sobreajuste**: está memorizando los 132 ejemplos.
- Con un corpus pequeño, ver `eval_loss` tocar fondo hacia la época 4-6 y
  repuntar después es lo esperable, **no un error**. Para eso está
  `load_best_model_at_end`.
"""))

    c.append(code('''
resultado_entrenamiento = trainer.train()

print()
print(f"Tiempo de entrenamiento: {resultado_entrenamiento.metrics['train_runtime']:.1f} s")
print(f"Pérdida final de entrenamiento: {resultado_entrenamiento.metrics['train_loss']:.4f}")
'''))

    c.append(code('''
import pandas as pd

historial = pd.DataFrame(trainer.state.log_history)
curvas = historial[["epoch", "loss", "eval_loss"]].groupby("epoch").first().dropna(how="all")
print(curvas.to_string())

mejor = historial.dropna(subset=["eval_loss"]).sort_values("eval_loss").iloc[0]
print(f"\\nMejor época: {mejor['epoch']:.0f}  (eval_loss = {mejor['eval_loss']:.4f})")
'''))

    c.append(code('''
import matplotlib.pyplot as plt

tr = historial.dropna(subset=["loss"])
ev = historial.dropna(subset=["eval_loss"])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(tr["epoch"], tr["loss"], label="Entrenamiento", alpha=0.8)
ax.plot(ev["epoch"], ev["eval_loss"], label="Validación", marker="o")
ax.axvline(mejor["epoch"], ls="--", c="gray", lw=1,
           label=f"Mejor época ({mejor['epoch']:.0f})")
ax.set_xlabel("Época"); ax.set_ylabel("Pérdida")
ax.set_title("Qwen2.5-1.5B + LoRA · curvas de pérdida")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''))

    c.append(code('''
# Tras `load_best_model_at_end`, el modelo bueno es `trainer.model` (el Trainer
# recarga el mejor checkpoint). Guardar y evaluar desde ahí evita quedarnos con
# los pesos de la última época, que no son los mejores.
modelo = trainer.model

modelo.save_pretrained(DIR_ADAPTADOR)
tokenizer.save_pretrained(DIR_ADAPTADOR)
print(f"Adaptador LoRA guardado en {DIR_ADAPTADOR}/")
print("Son unos pocos MB: LoRA no guarda el modelo base, solo las matrices A y B.")
print("El notebook 3 (pipeline Qwen+BERT) cargará este adaptador.")
'''))

    # ------------------------------------------------------------------ 10
    c.append(md("""
## 10 · Integración con Weights & Biases

La sesión de W&B se configuró antes de entrenar (sección 8) porque el
`Trainer` necesita `report_to="wandb"` desde el principio; lo que hacemos aquí
es **completar el registro** con lo que el `Trainer` no sabe: las métricas de
generación, la comparación baseline vs fine-tuned y ejemplos concretos.

Lo que queda registrado en el proyecto `tutor-matematicas-arquitecturas`:

| Origen | Contenido |
|---|---|
| Automático (`Trainer`) | `train/loss`, `eval/loss` por época, learning rate, gradientes, uso de GPU, todos los hiperparámetros |
| Manual (esta sección) | exactitud, formato válido, ROUGE-L antes y después; tabla de generaciones ejemplo por ejemplo |

La tabla de generaciones es la parte más útil en la práctica: permite abrir un
run tres semanas después y ver **qué** contestó el modelo, no solo qué número
sacó.
"""))

    # ------------------------------------------------------------------ 11
    c.append(md("""
## 11 · Evaluación

Volvemos a ejecutar exactamente el mismo procedimiento del baseline, sobre los
mismos 33 ejemplos de validación, con el mismo decodificado greedy. Lo único
que cambió es el adaptador LoRA.
"""))

    c.append(code('''
gen_finetuned = generar(modelo, [r["entrada"] for r in val])
metricas_finetuned = evaluar_generacion(gen_finetuned, val)

print()
for k, v in metricas_finetuned.items():
    if not k.startswith("_"):
        print(f"  {k:16s}: {v}")
'''))

    c.append(md("""
### Desglose por categoría

El promedio global esconde información. Un tutor que acierta el 90% en sumas y
el 10% en ecuaciones no es un tutor con 50% de exactitud: es un tutor que no
sirve para álgebra. Este desglose es también lo que motivará el pipeline con
clasificador del notebook 3.
"""))

    c.append(code('''
from collections import defaultdict

por_cat = defaultdict(lambda: {"n": 0, "base": 0, "ft": 0})
for r, ok_b, ok_f in zip(val, metricas_baseline["_correctas"], metricas_finetuned["_correctas"]):
    d = por_cat[r["categoria"]]
    d["n"] += 1
    d["base"] += int(ok_b)
    d["ft"] += int(ok_f)

print(f"{'categoría':26s} {'n':>3s} {'baseline':>10s} {'fine-tuned':>12s}")
print("-" * 56)
for cat in CATEGORIAS:
    d = por_cat.get(cat)
    if d:
        print(f"{cat:26s} {d['n']:3d} {d['base']/d['n']:>9.0%} {d['ft']/d['n']:>11.0%}")

EXACTITUD_POR_CATEGORIA = {
    cat: {"n": d["n"], "baseline": d["base"] / d["n"], "finetuned": d["ft"] / d["n"]}
    for cat, d in por_cat.items()
}
'''))

    # ------------------------------------------------------------------ 12
    c.append(md("""
## 12 · Comparación Baseline vs Fine-tuned
"""))

    c.append(code('''
tabla_metricas(metricas_baseline, metricas_finetuned,
               titulo=f"Qwen2.5-1.5B · {len(val)} ejemplos de validación")
'''))

    c.append(code('''
gen_demo_ft = generar(modelo, [d["entrada"] for d in demo])

for d, antes, despues in zip(demo, gen_demo_baseline, gen_demo_ft):
    print("=" * 78)
    print(f"[{d['id']} · {d['categoria']}] {d['entrada']}")
    print("-" * 78)
    print("ANTES (baseline):")
    print(antes[:500])
    print("-" * 78)
    print("DESPUÉS (fine-tuned):")
    print(despues[:500])
    print("-" * 78)
    print(f"Referencia: {d['salida']}")
    print(f"Correcta -> antes: {respuesta_correcta(antes, d['valor'])} | "
          f"después: {respuesta_correcta(despues, d['valor'])}")
print("=" * 78)
'''))

    c.append(md("""
### Análisis de cambios individuales

Cuántos ejemplos pasaron de mal a bien y —esto se suele olvidar— cuántos
pasaron de bien a mal. Una mejora neta pequeña puede esconder mucho movimiento
en ambas direcciones, señal de que el modelo está inestable y no de que
aprendió algo sólido.
"""))

    c.append(code('''
mejoraron = [r["id"] for r, b, f in zip(val, metricas_baseline["_correctas"], metricas_finetuned["_correctas"]) if not b and f]
empeoraron = [r["id"] for r, b, f in zip(val, metricas_baseline["_correctas"], metricas_finetuned["_correctas"]) if b and not f]

print(f"Pasaron de incorrecto a correcto ({len(mejoraron)}): {mejoraron}")
print(f"Pasaron de correcto a incorrecto ({len(empeoraron)}): {empeoraron}")
print(f"Mejora neta: {len(mejoraron) - len(empeoraron):+d} de {len(val)} ejemplos")
'''))

    c.append(code('''
RESULTADOS_QWEN = {
    "notebook": "S04_Lab_Fine_tuning_Qwen",
    "arquitectura": "decoder-only",
    "modelo": MODELO_ID,
    "metodo": "QLoRA" if USAR_QLORA else "LoRA",
    "n_train": len(train),
    "n_val": len(val),
    "epocas": float(args.num_train_epochs),
    "learning_rate": args.learning_rate,
    "tiempo_entrenamiento_s": resultado_entrenamiento.metrics["train_runtime"],
    "train_loss_final": resultado_entrenamiento.metrics["train_loss"],
    "eval_loss_mejor": float(mejor["eval_loss"]),
    "mejor_epoca": float(mejor["epoch"]),
    "parametros_entrenables": sum(p.numel() for p in modelo.parameters() if p.requires_grad),
    "parametros_totales": n_par,
    "baseline": metricas_baseline,
    "finetuned": metricas_finetuned,
    "por_categoria": EXACTITUD_POR_CATEGORIA,
    "tokenizador": STATS_TOKENIZADOR,
    "mejoraron": mejoraron,
    "empeoraron": empeoraron,
}

guardar_resultados("qwen", RESULTADOS_QWEN)
'''))

    c.append(code('''
if USAR_WANDB:
    import wandb

    if wandb.run is None:
        wandb.init(project=PROYECTO, name=NOMBRE_RUN, tags=TAGS, reinit=True)

    tabla = wandb.Table(columns=["id", "categoria", "entrada", "referencia",
                                 "baseline", "finetuned", "ok_baseline", "ok_finetuned"])
    for r, b, f, okb, okf in zip(val, gen_baseline, gen_finetuned,
                                 metricas_baseline["_correctas"],
                                 metricas_finetuned["_correctas"]):
        tabla.add_data(r["id"], r["categoria"], r["entrada"], r["salida"], b, f, okb, okf)

    wandb.log({"evaluacion/generaciones": tabla})
    wandb.summary.update({
        "baseline/exactitud": metricas_baseline["exactitud"],
        "baseline/formato_valido": metricas_baseline["formato_valido"],
        "baseline/rouge_l": metricas_baseline["rouge_l"],
        "finetuned/exactitud": metricas_finetuned["exactitud"],
        "finetuned/formato_valido": metricas_finetuned["formato_valido"],
        "finetuned/rouge_l": metricas_finetuned["rouge_l"],
        "delta/exactitud": metricas_finetuned["exactitud"] - metricas_baseline["exactitud"],
    })
    wandb.finish()
    print("Registro en W&B completado.")
else:
    print("W&B desactivado; las métricas quedaron en resultados/qwen.json")
'''))

    # ------------------------------------------------------------------ 13
    c.append(md("""
## 13 · Discusión

Rellenen esta sección **con sus números**, no con los esperados. Las preguntas
guía son las que un evaluador va a hacer:

**1. ¿Qué mejoró exactamente?**
Comparen la variación de `formato_valido` contra la de `exactitud`. Si el
formato subió mucho (por ejemplo de 20% a 95%) y la exactitud subió poco, la
conclusión honesta es: *el fine-tuning enseñó el formato de respuesta, no a
hacer mejor las cuentas*. Es un resultado legítimo y publicable; presentarlo
como "el modelo aprendió matemáticas" no lo sería.

**2. ¿Hay sobreajuste?**
Miren en qué época tocó fondo `eval_loss`. Si fue en la 2 o 3 de 8, el modelo
agotó lo que podía aprender de 132 ejemplos muy pronto. Eso no se arregla
entrenando más épocas: se arregla con más datos.

**3. ¿Dónde falla todavía?**
Revisen el desglose por categoría. Es esperable que las categorías de varios
pasos (ecuaciones, operaciones combinadas, porcentajes con dos operaciones)
vayan peor que las de un solo paso: cada paso adicional multiplica la
probabilidad de un error aritmético.

**4. ¿Qué significan los ejemplos que empeoraron?**
Si hay ejemplos que el baseline resolvía y el modelo entrenado ya no, vale la
pena mirarlos uno a uno. Suele indicar que el modelo aprendió a imitar la
*forma* de nuestras respuestas cortas y abandonó un razonamiento más largo que
antes le funcionaba.
"""))

    # ------------------------------------------------------------------ 14
    c.append(md("""
## 14 · Conclusiones

Cierren con tres puntos concretos:

1. **Resultado cuantitativo.** Exactitud baseline → exactitud fine-tuned, sobre
   33 ejemplos de validación. Mencionen el intervalo: con n=33, una diferencia
   de un solo ejemplo son 3 puntos porcentuales. Diferencias menores a ~10
   puntos no son concluyentes con este tamaño de muestra, y decirlo es parte
   del rigor del experimento.
2. **Qué aprendió el modelo.** Formato, estilo de explicación, longitud
   adecuada, uso del español. Y qué no: aritmética confiable.
3. **Qué haría falta.** Más datos (el corpus está preparado para escalar
   reemplazando el JSONL), o un enfoque híbrido donde el modelo genere el
   procedimiento y una herramienta externa verifique el cálculo.

### Qué sigue

- **Notebook 2 (BERT):** clasificar el tipo de problema con un encoder.
- **Notebook 3 (Qwen+BERT):** usar esa clasificación para especializar el
  prompt de generación y medir si el enrutamiento aporta.
- **Notebook 4 (FLAN-T5):** resolver clasificación y generación en un solo
  modelo encoder-decoder.
- **Notebook 5:** comparación de las cuatro arquitecturas.
"""))

    return c
