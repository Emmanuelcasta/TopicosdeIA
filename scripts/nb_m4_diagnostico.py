"""Notebook de entrenamiento del adaptador de diagnóstico de M4.

Produce `m4/pipeline/S13_M4_FineTuning_Diagnostico.ipynb`.

Qué problema ataca. En la corrida del 24-10 el tutor **rehízo correctamente** los
cinco procedimientos con error y no avisó de ninguno:

    el estudiante escribió  24 × 13 = 302     el tutor respondió  24 × 13 = 312
    el estudiante escribió  12 × 5  = 70      el tutor respondió  12 × 5  = 60

Los 7 casos que no emitieron `Diagnóstico:` son exactamente los que tenían un
error. No es aritmética ni lectura: el adaptador de M1 se entrenó para emitir
`Paso N: … Respuesta final: …` y es lo único que sabe producir. Ni un reintento
acotado lo saca de ahí.

Este notebook es **un experimento**, y se escribe como tal:

  · El adaptador va a `adaptadores/qwen-lora-diagnostico`. **No toca**
    `qwen-lora` ni `qwen-lora-agente`: los tres quedan para comparar y para poder
    volver atrás.

  · Se mide una línea base **antes** de entrenar, con los adaptadores que ya
    existen, sobre los mismos casos de desarrollo. Sin ella no se sabría si el
    entrenamiento hizo algo.

  · Se reportan **formato y acierto por separado**. Son 33 ejemplos de
    entrenamiento: lo esperable es que el formato mejore. Que el modelo además
    localice el paso correcto es otra cosa, y mezclar las dos cifras sería
    presentar como razonamiento lo que puede ser obediencia al formato.

  · **No se toca ninguno de los 12 casos de evaluación.** Aquí solo se entrena y
    se mide sobre desarrollo; la evaluación es el trabajo de S13.
"""

from __future__ import annotations

from nb_comun import (celda_args_compatibles, celda_drive, celda_wandb, code,
                      encabezado, md)
from nb_comun_agente import CODIGO_LLM, _codigo_blob, celdas_modulos


CODIGO_DATOS = '''
from collections import Counter

train = [t for t in diagnostico if t["split"] == "train"]
dev = [t for t in diagnostico if t["split"] == "dev"]
print(f"train: {len(train)}  |  dev: {len(dev)}")
print("por familia:", dict(Counter(t["familia"] for t in diagnostico)))
print("paso señalado:", dict(sorted(Counter(
    t["paso_error"] for t in diagnostico if t["paso_error"]).items())))
print("\\nUn ejemplo completo, lo que el modelo tiene que aprender a escribir:\\n")
print(train[0]["mensajes"][-1]["content"])
print("\\n" + "-" * 70)
print("Y el prompt que lo precede (el MISMO de inferencia, `prompt_revision`):\\n")
print(train[0]["mensajes"][1]["content"][:700], "…")
'''


CODIGO_TOKENIZAR = '''
MAX_LEN = 1536      # el prompt de revisión es corto: ~400-600 tokens


def render(mensajes, generacion=False):
    return tok.apply_chat_template(mensajes, tokenize=False,
                                   add_generation_prompt=generacion)


def tokenizar(t):
    """Gradiente SOLO sobre el turno del asistente: lo que se enseña es a
    dictaminar, no a reproducir el prompt."""
    mensajes = t["mensajes"]
    completo = render(mensajes)
    antes = render(mensajes[:-1], generacion=True)
    assert completo.startswith(antes), "la plantilla no es prefijo-estable"
    enc = tok(completo, return_offsets_mapping=True, add_special_tokens=False,
              truncation=True, max_length=MAX_LEN)
    etiquetas = [tid if a >= len(antes) else -100
                 for tid, (a, _) in zip(enc["input_ids"], enc["offset_mapping"])]
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "labels": etiquetas, "truncado": len(enc["input_ids"]) >= MAX_LEN}


tk = tokenizar(train[0])
print(f"Tokens: {len(tk['input_ids'])} | con gradiente: {sum(e != -100 for e in tk['labels'])}")
print("\\nLo que el modelo aprende a escribir (solo los tokens con etiqueta):\\n")
print(tok.decode([t for t, e in zip(tk["input_ids"], tk["labels"]) if e != -100]))
'''


CODIGO_DATASET = '''
from datasets import Dataset

tok_train = [tokenizar(t) for t in train]
tok_dev = [tokenizar(t) for t in dev]
longitudes = [len(t["input_ids"]) for t in tok_train]
print(f"tokens por ejemplo: min {min(longitudes)}  mediana {int(np.median(longitudes))}  "
      f"max {max(longitudes)}")
assert not any(t["truncado"] for t in tok_train + tok_dev), "hay ejemplos truncados: suban MAX_LEN"

columnas = ["input_ids", "attention_mask", "labels"]
ds_train = Dataset.from_list([{k: t[k] for k in columnas} for t in tok_train])
ds_dev = Dataset.from_list([{k: t[k] for k in columnas} for t in tok_dev])
print(f"train {len(ds_train)} | dev {len(ds_dev)}")
'''


CODIGO_MEDIR = '''
# La medición, separando lo que no se debe mezclar.
#
#   formato   -> ¿emite una línea `Diagnóstico:` con una de las cinco categorías?
#   acierto   -> ¿esa categoría es la que el caso espera?
#
# Con 33 ejemplos de entrenamiento lo esperable es que suba el formato. Que suba
# el acierto es otra cosa, y presentarlas juntas haría pasar por razonamiento lo
# que puede ser obediencia al formato.
@torch.no_grad()
def medir(casos, adaptador, nombre):
    filas = []
    for t in casos:
        prompt = tok.apply_chat_template(t["mensajes"][:-1], tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(modelo.device)
        if adaptador is None:
            contexto = modelo.disable_adapter()
        else:
            modelo.set_adapter(adaptador)
            contexto = nullcontext()
        with contexto:
            out = modelo.generate(**ids, max_new_tokens=120, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        salida = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        conclusion = extraer_diagnostico(salida)
        filas.append({
            "id": t["id"], "familia": t["familia"], "esperado": t["diagnostico"].lower(),
            "conclusion": conclusion,
            "formato_ok": diagnostico_valido(conclusion),
            "acierto": diagnostico_valido(conclusion) and conclusion == t["diagnostico"].lower(),
            # La conducta que se quiere quitar: reescribir el procedimiento.
            "reescribe": "Respuesta final" in salida,
            "salida": salida[:300],
        })
    df = pd.DataFrame(filas)
    print(f"{nombre:28s} formato {int(df.formato_ok.sum())}/{len(df)}   "
          f"acierto {int(df.acierto.sum())}/{len(df)}   "
          f"reescribe {int(df.reescribe.sum())}/{len(df)}")
    return df
'''


CODIGO_BASE = '''
# LÍNEA BASE, antes de entrenar. Sin esto no se sabría si el entrenamiento hizo
# algo, y "mejoró" no significaría nada.
BASES = {}
for adaptador, nombre in [(None, "base (sin adaptador)"), ("m1", "qwen-lora (M1)"),
                          ("agente", "qwen-lora-agente (S10)")]:
    if adaptador is not None and adaptador not in ADAPTADORES:
        print(f"{nombre:28s} no está en Drive, se omite")
        continue
    BASES[nombre] = medir(dev, adaptador, nombre)

print("\\n`agente` es la referencia que importa: es el adaptador con el que se")
print("ejecutaron las tres corridas de M4.")
'''


CODIGO_ENTRENAR = '''
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq, Trainer

DIR_ADAPTADOR = "adaptadores/qwen-lora-diagnostico"     # NUEVO, no toca los otros dos
assert DIR_ADAPTADOR not in ("adaptadores/qwen-lora", "adaptadores/qwen-lora-agente")

# Se entrena sobre el modelo BASE, no encima de otro adaptador: así el experimento
# aísla el efecto de estos 33 ejemplos y los otros dos adaptadores quedan intactos.
base_limpio = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE, torch_dtype=torch.float16).to(DEVICE)
base_limpio.gradient_checkpointing_enable()
base_limpio.enable_input_require_grads()
base_limpio.config.use_cache = False

config_lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
entrenable = get_peft_model(base_limpio, config_lora)
# LoRA en fp32 sobre pesos base en fp16: en fp16 el gradiente se desborda (lección de M1).
for n, p in entrenable.named_parameters():
    if p.requires_grad:
        p.data = p.data.float()
entrenable.print_trainable_parameters()

args = construir_args(
    output_dir=f"{DIR_CHECKPOINTS}/qwen-lora-diagnostico",
    num_train_epochs=6,              # 33 ejemplos: hacen falta más pasadas que en S10
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=REPORTAR_A,
    run_name=NOMBRE_RUN,
    seed=SEMILLA,
)
trainer = Trainer(model=entrenable, args=args, train_dataset=ds_train, eval_dataset=ds_dev,
                  data_collator=DataCollatorForSeq2Seq(tokenizer=tok, padding=True,
                                                       label_pad_token_id=-100))
resultado = trainer.train()
print(resultado)

entrenable.save_pretrained(DIR_ADAPTADOR)
print(f"\\nAdaptador guardado en {DIR_ADAPTADOR}")
print("Los otros dos siguen donde estaban:", [d for d in ADAPTADORES])
'''


CODIGO_DESPUES = '''
# El adaptador nuevo, cargado junto a los otros para medir en las mismas condiciones.
del entrenable, base_limpio
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

modelo.load_adapter(DIR_ADAPTADOR, adapter_name="diagnostico")
ADAPTADORES.append("diagnostico")
DESPUES = medir(dev, "diagnostico", "qwen-lora-diagnostico")

print("\\n" + "=" * 78)
print("ANTES Y DESPUÉS · sobre los casos de DESARROLLO, no sobre los 12 de evaluación")
print("=" * 78)
print(f"{'adaptador':28s}{'formato':>10s}{'acierto':>10s}{'reescribe':>11s}")
for nombre, df in list(BASES.items()) + [("qwen-lora-diagnostico", DESPUES)]:
    print(f"{nombre:28s}{int(df.formato_ok.sum()):>7d}/{len(df):<3d}"
          f"{int(df.acierto.sum()):>7d}/{len(df):<3d}{int(df.reescribe.sum()):>8d}/{len(df):<3d}")
print("=" * 78)

n = len(DESPUES)
print(f"\\nSon {n} casos de desarrollo. Con ese tamaño:")
print("  · una diferencia de 1 o 2 casos no distingue nada;")
print("  · el formato y el acierto NO se promedian ni se presentan como una cifra;")
print("  · y esto no dice nada todavía sobre los 12 casos de evaluación, que no se")
print("    han tocado aquí. Eso lo mide S13.")

if int(DESPUES.formato_ok.sum()) < n:
    print("\\nEl formato no se sostiene ni después de entrenar. Antes de volver a S13,")
    print("mirar una salida completa: puede que 33 ejemplos no basten, o que el")
    print("prompt y las etiquetas no coincidan tanto como parecía.")
'''


CODIGO_GUARDAR = '''
import json
from pathlib import Path

Path("m4").mkdir(exist_ok=True)
resumen = {
    "adaptador": DIR_ADAPTADOR,
    "no_sobrescribe": ["adaptadores/qwen-lora", "adaptadores/qwen-lora-agente"],
    "datos": {"train": len(train), "dev": len(dev),
              "familias": dict(Counter(t["familia"] for t in diagnostico))},
    "lora": {"r": config_lora.r, "alpha": config_lora.lora_alpha,
             "target_modules": sorted(config_lora.target_modules)},
    "entrenamiento": {"epocas": float(args.num_train_epochs), "lr": args.learning_rate,
                      "loss_final": float(resultado.training_loss)},
    "dev_antes": {n: {"formato": int(d.formato_ok.sum()), "acierto": int(d.acierto.sum()),
                      "reescribe": int(d.reescribe.sum()), "n": len(d)}
                  for n, d in BASES.items()},
    "dev_despues": {"formato": int(DESPUES.formato_ok.sum()),
                    "acierto": int(DESPUES.acierto.sum()),
                    "reescribe": int(DESPUES.reescribe.sum()), "n": len(DESPUES)},
    "aviso": ("Medido SOLO sobre los casos de desarrollo. Los 12 casos de "
              "evaluación de M4 no se tocaron en este notebook."),
}
json.dump(resumen, open("m4/finetuning_diagnostico.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
DESPUES.to_csv("m4/dev_diagnostico_despues.csv", index=False)
for nombre, df in BASES.items():
    df.to_csv(f"m4/dev_diagnostico_{nombre.split()[0].replace('(', '')}.csv", index=False)

print("Guardado:")
print("  m4/finetuning_diagnostico.json")
print("  m4/dev_diagnostico_despues.csv  (y uno por cada línea base)")
print(f"  {DIR_ADAPTADOR}/   <- súbanlo igual que los otros adaptadores")
'''


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "Enseñar a dictaminar, no a resolver",
        "Módulo 4 · Adaptador de diagnóstico · experimento",
    ))

    c.append(md("""
## 1 · Qué problema ataca

En la corrida del 24-10 el tutor **rehízo correctamente** los cinco
procedimientos con error, y no avisó de ninguno:

| El estudiante escribió | El tutor respondió | ¿Avisó? |
|---|---|---|
| `24 × 13 = 302` | `24 × 13 = 312` | no |
| `1248 ÷ 8 = 146` | `1248 ÷ 8 = 156` | no |
| `12 × 5 = 70` | `12 × 5 = 60` | no |
| `6/8 + 1/8 = 8/8` | `6/8 + 1/8 = 7/8` | no |

Los 7 casos que no emitieron `Diagnóstico:` son **exactamente** los que tenían un
error. No es un problema de aritmética ni de lectura: el adaptador de M1 se
entrenó para emitir `Paso N: … Respuesta final: …` y es lo único que sabe
producir. Ni un reintento pidiéndole solo la línea lo saca de ahí.

El modelo nunca ha visto un ejemplo de **revisar y dictaminar**. Esto lo escribe.

> **Es un experimento.** Son 33 ejemplos de entrenamiento. Lo esperable es que
> mejore el formato de salida; que además localice el paso correcto es otra cosa,
> y por eso las dos cifras se reportan **por separado** y nunca se promedian.
"""))

    c.append(md("""
## 2 · Qué no toca

| | |
|---|---|
| **Adaptador nuevo** | `adaptadores/qwen-lora-diagnostico` |
| **Intactos** | `adaptadores/qwen-lora` (M1) y `adaptadores/qwen-lora-agente` (S10) |
| **Casos de evaluación** | ninguno: aquí solo hay desarrollo. Los 12 `vis-*` los mide S13 |
| **Etiquetas esperadas** | sin tocar |

Los tres adaptadores quedan en Drive para comparar y para poder volver atrás.
"""))

    c.append(md("""
---
## 0 · Preparación

> Hace falta GPU con CUDA; cualquiera sirve. El entrenamiento son minutos: una
> LoRA sobre 1.5B con 33 ejemplos.
"""))
    c.append(code('''
%pip install -q "transformers>=4.45" "peft>=0.12" "accelerate>=0.33" datasets pandas
%pip uninstall -y -q torchao
print("Listo. Si Colab pide reiniciar la sesión, reinícienla y sigan desde aquí.")
'''))
    c.append(code('''
import gc, os, random
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM

SEMILLA = 42
random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
transformers.set_seed(SEMILLA)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"transformers {transformers.__version__} | torch {torch.__version__} | {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU detectada: {torch.cuda.get_device_properties(0).name}")
else:
    print("Sin GPU: no se puede entrenar. Activen un entorno con CUDA.")
'''))
    c.extend(celda_drive())
    c.extend(celda_args_compatibles())

    c.append(md("""
---
## 3 · El módulo de visión

El mismo archivo que usa S13. Importa porque de aquí salen `prompt_revision` —el
prompt con el que se entrena, que tiene que ser **literalmente** el de
inferencia— y `extraer_diagnostico`, el parser con el que se mide.
"""))
    c.extend(celdas_modulos(["vision"], con_descripcion=False))

    c.append(md("""
---
## 4 · Las trayectorias de diagnóstico

Generadas y validadas por `scripts/trayectorias_diagnostico_fuente.py`. Ningún
enunciado sale del eval set de M2, M3 ni M4, ni parafraseado: la comprobación es
la misma de S10 (Jaccard sobre palabras, tope 0.6) y el generador falla si alguna
lo pasa. Incluye los casos `dev-*` de M4, que son la puerta de S13.

El paso señalado se reparte a propósito entre 1, 2 y 3: el eval set está sesgado
hacia "Paso 2" (4 de 6), y entrenar con el mismo sesgo enseñaría el atajo.
"""))
    c.append(code("import base64, gzip, hashlib, json\nfrom pathlib import Path\n"
                  + _codigo_blob("data/trayectorias_diagnostico.jsonl", "diagnostico")))
    c.append(code(CODIGO_DATOS))

    c.append(md("""
---
## 5 · El modelo y los adaptadores que ya existen
"""))
    c.append(code(CODIGO_LLM))

    c.append(md("""
---
## 6 · Tokenización

Gradiente **solo** sobre el turno del asistente: lo que se enseña es a
dictaminar, no a reproducir el prompt.
"""))
    c.append(code(CODIGO_TOKENIZAR))
    c.append(code(CODIGO_DATASET))

    c.append(md("""
---
## 7 · Línea base, antes de entrenar

Sin esto, "mejoró" no significaría nada. Se mide el modelo base y los dos
adaptadores que ya existen sobre los **mismos** casos de desarrollo.

La columna `reescribe` cuenta las salidas que contienen `Respuesta final`: es
exactamente la conducta que se quiere quitar.
"""))
    c.append(code(CODIGO_MEDIR))
    c.append(code(CODIGO_BASE))

    c.append(md("""
---
## 8 · Entrenamiento
"""))
    c.extend(celda_wandb("qwen-lora-diagnostico",
                         '["m4", "diagnostico", "vision", "finetuning"]'))
    c.append(code(CODIGO_ENTRENAR))

    c.append(md("""
---
## 9 · Después de entrenar · formato y acierto, separados
"""))
    c.append(code(CODIGO_DESPUES))

    c.append(md("""
---
## 10 · Artefactos
"""))
    c.append(code(CODIGO_GUARDAR))

    c.append(md("""
---
## 11 · Cómo leer esto, y qué hacer después

**1 · Mirar primero `reescribe`.** Si baja de golpe, el entrenamiento consiguió lo
que buscaba: que el modelo deje de rehacer el ejercicio. Es el cambio de conducta
que bloqueaba M4.

**2 · Formato y acierto, separados y en ese orden.** Con 33 ejemplos, el formato
es lo que puede moverse. Si sube el formato y el acierto se queda, el resultado
honesto es "ahora dictamina, pero todavía no localiza bien", que es información
útil y no una mejora que presentar como tal.

**3 · Son 9 casos de desarrollo.** Una diferencia de uno o dos no distingue nada.

**4 · Esto no dice nada sobre los 12 casos de evaluación.** No se han tocado. El
siguiente paso es S13 con `ADAPTADOR_DIAGNOSTICO = "diagnostico"`, y ahí la
puerta de desarrollo decide si la evaluación llega a ejecutarse.

**5 · Si el formato no se sostiene**, las salidas a mirar son las de
`m4/dev_diagnostico_despues.csv`. Las dos causas probables son que 33 ejemplos no
basten —hay que escribir más, no reciclar los 12 casos— o que el prompt de
entrenamiento se haya separado del de inferencia, que es justo lo que
`prompt_revision` existe para evitar.
"""))

    return c
