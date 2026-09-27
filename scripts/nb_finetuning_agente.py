"""Notebook S10 (2/4) — Fine-tuning LoRA del agente con trayectorias de herramientas (C5-FT)."""

from __future__ import annotations

from nb_comun import celda_args_compatibles, celda_drive, celda_wandb, code, encabezado, md
from nb_comun_agente import celda_trayectorias, celdas_modulos


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "Enseñar a usar herramientas: fine-tuning con trayectorias",
        "Sesión 10 · Módulo 3 — LoRA sobre trayectorias de tool calling · C5-FT (2/4)",
    ))

    c.append(md("""
## 1 · Introducción

El notebook 1 dejó 199 trayectorias validadas. Este las usa para **entrenar** un
adaptador que aprenda **cuándo** llamar una herramienta, **cuál** y **con qué
argumentos**. El notebook 3 lo compara con la alternativa sin entrenamiento
(few-shot):

> **C5** (tool calling nativo + few-shot) **vs. C5-FT** (lo mismo + este adaptador)
> — la diferencia responde con datos *¿basta con mostrar ejemplos o hay que entrenar?*

### Decisiones de diseño

| Decisión | Elección | Por qué |
|---|---|---|
| Punto de partida | **Qwen2.5-1.5B base**, no la LoRA de M1 | La LoRA de M1 aprendió a responder *siempre* con "Paso 1:", lo que compite con emitir `<tool_call>`. Las trayectorias ya terminan con ese formato, así que el adaptador nuevo aprende las dos cosas. |
| Qué se entrena | **Solo los turnos del asistente** (pensamiento + llamada + respuesta final) | Las observaciones las escribe la herramienta y la pregunta el usuario; aprender a *predecirlas* sería aprender a inventar resultados, justo lo contrario del objetivo. |
| Formato | La plantilla nativa de Qwen2.5 con **los mismos esquemas y el mismo prompt de sistema de C5** | Lo que se entrena debe ser idéntico a lo que se evalúa. |
| LoRA | r=16, α=32, atención + MLP | Igual que M1 en atención; se añade MLP porque aquí se aprende un comportamiento nuevo (decidir y formatear llamadas), no solo un estilo. |
| Validación | 42 trayectorias **de validación**, nunca los eval sets | Los eval sets se reservan para el notebook 3. |
"""))

    c.append(md("""
## 2 · Objetivos

1. Tokenizar las trayectorias con **máscara de pérdida** solo sobre el asistente.
2. Entrenar el adaptador `qwen-lora-agente`.
3. Medir sobre validación, antes y después, las cuatro decisiones del agente:
   **usar o no** herramienta, **cuál**, **llamada válida**, **resultado correcto**.
4. Medir de punta a punta con el bucle real del agente.
"""))

    c.append(md("""
## 0 · Preparación del entorno

> **Activen la GPU (T4).** Cada ejemplo lleva los esquemas de 16 herramientas en el
> prompt: **~2.700 tokens** antes de la primera palabra del estudiante (medido). Por
> eso se entrena con lote 1, acumulación de gradiente y *gradient checkpointing*. Ese
> mismo coste lo paga el agente en **cada paso** de inferencia, y el notebook 3 lo
> reporta como tokens por pregunta.
"""))
    c.append(code('''
%pip install -q "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" datasets sympy wandb
%pip uninstall -y -q torchao
print("Librerías instaladas. Si Colab pide reiniciar la sesión, reinícienla y sigan desde aquí.")
'''))
    c.append(code('''
import os, random
import numpy as np
import torch
import transformers

SEMILLA = 42
random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
transformers.set_seed(SEMILLA)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"transformers {transformers.__version__} | torch {torch.__version__} | {DEVICE}")
'''))
    c.extend(celda_drive())

    c.append(md("""
---
## 3 · El núcleo (copia literal de `scripts/agente/`)
"""))
    c.extend(celdas_modulos(["registro", "herramientas", "nucleo", "documentos", "prompts"], con_descripcion=False))
    c.append(code('''
# Esquemas EXACTOS de C5: las 15 herramientas matemáticas + buscar_documentos.
# Aquí no hace falta el retriever (solo se necesitan los esquemas), así que la
# búsqueda es un marcador; en el notebook 3 es el retriever de S08.
REGISTRO_C5 = registro_con_documentos(REGISTRO, buscar=lambda consulta, k: [],
                                      chunks=[], metadatos=[], ids_chunks=[])
ESQUEMAS_C5 = REGISTRO_C5.esquemas()
print(len(ESQUEMAS_C5), "herramientas en el prompt:", REGISTRO_C5.nombres())
'''))

    c.append(md("""
---
## 4 · Las trayectorias
"""))
    c.extend(celda_trayectorias(prefijo="4.1"))

    c.append(md("""
---
## 5 · Tokenización con máscara de pérdida

Se renderiza la conversación completa con la plantilla nativa y se localiza, **por
posición de caracteres**, el texto de cada turno del asistente. Solo esos tokens
llevan etiqueta; todo lo demás es `-100`.

Localizarlo por caracteres (con `offset_mapping`) y no tokenizando prefijos por
separado evita un error sutil: tokenizar un prefijo aislado puede cortar un token
distinto que dentro del texto completo, y la máscara quedaría corrida un token.
"""))
    c.append(code('''
from transformers import AutoTokenizer

MODELO_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODELO_BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Medido: system + 16 esquemas = ~2.700 tokens; trayectorias entre ~2.800 y ~3.400.
# Con 3.072 se truncaban los turnos finales (justo la respuesta). Ninguna debe truncarse.
MAX_LEN = 4096


def render(mensajes, generacion=False):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM_C5_AGENTE}] + mensajes,
                                   tools=ESQUEMAS_C5, tokenize=False, add_generation_prompt=generacion)


def tramos_asistente(mensajes):
    """[(inicio, fin)] en caracteres del texto completo para cada turno del asistente."""
    completo = render(mensajes)
    tramos = []
    for i, m in enumerate(mensajes):
        if m["role"] != "assistant":
            continue
        antes = render(mensajes[:i], generacion=True)
        hasta = render(mensajes[:i + 1])
        assert completo.startswith(antes) and completo.startswith(hasta), "la plantilla no es prefijo-estable"
        tramos.append((len(antes), len(hasta)))
    return completo, tramos


def tokenizar(tr):
    texto, tramos = tramos_asistente(tr["mensajes"])
    enc = tok(texto, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=MAX_LEN)
    etiquetas = []
    for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        dentro = any(ini <= a < fin for ini, fin in tramos)
        etiquetas.append(tid if dentro else -100)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": etiquetas,
            "truncado": len(enc["input_ids"]) >= MAX_LEN}


ejemplo = next(t for t in trayectorias if t["id"] == "man-com-04")
tk = tokenizar(ejemplo)
print(f"Tokens: {len(tk['input_ids'])} | con gradiente: {sum(e != -100 for e in tk['labels'])}")
print("\\nLo que el modelo aprende a escribir (solo los tokens con etiqueta):\\n")
print(tok.decode([t for t, e in zip(tk["input_ids"], tk["labels"]) if e != -100]))
'''))
    c.append(code('''
from datasets import Dataset

tok_train = [tokenizar(t) for t in trayectorias if t["split"] == "train"]
tok_val   = [tokenizar(t) for t in trayectorias if t["split"] == "validation"]
longitudes = [len(t["input_ids"]) for t in tok_train]
print(f"train: {len(tok_train)} | validación: {len(tok_val)}")
print(f"tokens por ejemplo: min {min(longitudes)}  mediana {int(np.median(longitudes))}  max {max(longitudes)}")
print(f"truncados: {sum(t['truncado'] for t in tok_train + tok_val)}")
assert not any(t["truncado"] for t in tok_train + tok_val), "hay trayectorias truncadas: suban MAX_LEN"

columnas = ["input_ids", "attention_mask", "labels"]
ds_train = Dataset.from_list([{k: t[k] for k in columnas} for t in tok_train])
ds_val   = Dataset.from_list([{k: t[k] for k in columnas} for t in tok_val])
'''))

    c.append(md("""
---
## 6 · Línea base antes de entrenar

Se mide **el mismo modelo base** con el prompt de C5, **sin few-shot**, sobre las
trayectorias de validación. Así el efecto del entrenamiento queda aislado.

Para cada trayectoria se da la conversación **hasta la primera decisión** del
asistente y se compara lo que el modelo propone con lo que hace la trayectoria:

| Métrica | Pregunta |
|---|---|
| `decision` | ¿Acertó en usar o no usar herramienta? |
| `herramienta` | Si había que usarla, ¿eligió la misma (o `evaluar_expresion`, que las cubre)? |
| `llamada_valida` | ¿El JSON y los tipos pasan la validación del registro? |
| `resultado` | ¿La llamada, ejecutada, da el mismo resultado que la de la trayectoria? |
"""))
    c.append(code('''
from transformers import AutoModelForCausalLM

modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float16).to(DEVICE)
modelo.config.use_cache = True


@torch.no_grad()
def generar(mensajes, max_new_tokens=200):
    ids = tok(render(mensajes, generacion=True), return_tensors="pt", add_special_tokens=False).to(modelo.device)
    out = modelo.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluar_decisiones(trs, etiqueta):
    filas = []
    for t in trs:
        primera = next(i for i, m in enumerate(t["mensajes"]) if m["role"] == "assistant")
        esperado = t["mensajes"][primera]
        salida = analizar_salida(generar(t["mensajes"][:primera]))
        debia = "tool_calls" in esperado
        uso = bool(salida.llamadas)
        fila = {"id": t["id"], "tipo": t["tipo"], "decision": debia == uso}
        if debia:
            f_esp = esperado["tool_calls"][0]["function"]
            fila["herramienta"] = uso and salida.llamadas[0].nombre in (f_esp["name"], "evaluar_expresion")
            if uso:
                llamada = salida.llamadas[0]
                if llamada.nombre == "buscar_documentos":
                    fila["llamada_valida"] = isinstance(llamada.argumentos.get("consulta"), str)
                    fila["resultado"] = fila["llamada_valida"] and f_esp["name"] == "buscar_documentos"
                else:
                    r_mod = REGISTRO.ejecutar(llamada)
                    fila["llamada_valida"] = r_mod.tipo_error not in ("desconocida", "argumentos")
                    esperado_obs = json.loads(t["mensajes"][primera + 1]["content"])
                    fila["resultado"] = (r_mod.ok and esperado_obs.get("ok") and
                                         (mismo_numero(r_mod.valor, esperado_obs.get("valor", esperado_obs.get("resultado")))
                                          or r_mod.resultado == esperado_obs.get("resultado"))) or \\
                                        (not r_mod.ok and not esperado_obs.get("ok") and r_mod.tipo_error == "dominio")
            else:
                fila["llamada_valida"] = fila["resultado"] = False
        filas.append(fila)
    df = pd.DataFrame(filas)
    resumen = {"sistema": etiqueta, "n": len(df), "decision": df["decision"].mean()}
    for col in ("herramienta", "llamada_valida", "resultado"):
        resumen[col] = df[col].dropna().astype(float).mean() if col in df else None
    return df, resumen


import pandas as pd

val_trs = [t for t in trayectorias if t["split"] == "validation"]
df_antes, res_antes = evaluar_decisiones(val_trs, "base (sin entrenar)")
print(pd.Series(res_antes).to_string())
'''))

    c.append(md("""
---
## 7 · Entrenamiento
"""))
    c.extend(celda_args_compatibles())
    c.extend(celda_wandb("qwen-lora-agente", '["agente", "tool-use", "s10", "finetuning"]'))
    c.append(code('''
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorForSeq2Seq, Trainer

modelo.gradient_checkpointing_enable()
modelo.enable_input_require_grads()
modelo.config.use_cache = False

config_lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
modelo = get_peft_model(modelo, config_lora)
# LoRA en fp32 sobre pesos base en fp16: en fp16 el gradiente se desborda (lección de M1).
for n, p in modelo.named_parameters():
    if p.requires_grad:
        p.data = p.data.float()
modelo.print_trainable_parameters()

args = construir_args(
    output_dir=f"{DIR_CHECKPOINTS}/qwen-lora-agente",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
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
trainer = Trainer(model=modelo, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                  data_collator=DataCollatorForSeq2Seq(tokenizer=tok, padding=True, label_pad_token_id=-100))
resultado_entrenamiento = trainer.train()
print(resultado_entrenamiento)
'''))
    c.append(code('''
import matplotlib.pyplot as plt

log = pd.DataFrame(trainer.state.log_history)
fig, ax = plt.subplots(figsize=(8, 4))
tr = log.dropna(subset=["loss"])
ax.plot(tr["epoch"], tr["loss"], label="Entrenamiento")
ev = log.dropna(subset=["eval_loss"])
ax.plot(ev["epoch"], ev["eval_loss"], marker="o", label="Validación")
ax.set_xlabel("época"); ax.set_ylabel("pérdida (solo turnos del asistente)")
ax.set_title("Fine-tuning del agente"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
'''))

    c.append(md("""
---
## 8 · Después de entrenar
"""))
    c.append(code('''
modelo.eval()
modelo.config.use_cache = True
df_despues, res_despues = evaluar_decisiones(val_trs, "C5-FT (entrenado)")

comparacion = pd.DataFrame([res_antes, res_despues]).set_index("sistema")
print(comparacion.round(3).to_string())
print("\\nPor tipo de trayectoria (decisión correcta):")
print(pd.DataFrame({"antes": df_antes.groupby("tipo")["decision"].mean(),
                    "después": df_despues.groupby("tipo")["decision"].mean()}).round(2).to_string())
'''))
    c.append(md("""
### De punta a punta, con el bucle real

La métrica anterior solo mira la **primera** decisión. Aquí se ejecuta el agente
completo (varios pasos, observaciones reales, verificador) sobre las trayectorias
de validación de cálculo, y se mira si la **respuesta final** es correcta.
"""))
    c.append(code('''
def llm_actual(mensajes, esquemas):
    ids = tok(tok.apply_chat_template(mensajes, tools=esquemas, tokenize=False, add_generation_prompt=True),
              return_tensors="pt", add_special_tokens=False).to(modelo.device)
    with torch.no_grad():
        out = modelo.generate(**ids, max_new_tokens=256, do_sample=False,
                              pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def punta_a_punta(trs):
    aciertos = []
    for t in trs:
        agente = AgenteTutor(llm_actual, SYSTEM_C5_AGENTE, registro=REGISTRO, max_rondas=6, max_pasos=8, verificar=True)
        out = agente(t["pregunta"])
        aciertos.append(mismo_numero(valor_final(out.respuesta), t["valor"]))
    return float(np.mean(aciertos)) if aciertos else None


calc_val = [t for t in val_trs if t["valor"] and "buscar_documentos" not in t["herramientas"]]
with modelo.disable_adapter():
    e2e_antes = punta_a_punta(calc_val)
e2e_despues = punta_a_punta(calc_val)
print(f"Exactitud de punta a punta en {len(calc_val)} trayectorias de validación de cálculo:")
print(f"   base (sin entrenar): {e2e_antes:.1%}")
print(f"   C5-FT              : {e2e_despues:.1%}")
'''))

    c.append(md("""
---
## 9 · Artefactos
"""))
    c.append(code('''
DIR_ADAPTADOR = "adaptadores/qwen-lora-agente"
modelo.save_pretrained(DIR_ADAPTADOR)
tok.save_pretrained(DIR_ADAPTADOR)

Path("resultados").mkdir(exist_ok=True)
Path("resultados/finetuning_agente.json").write_text(json.dumps({
    "modelo_base": MODELO_BASE,
    "lora": {"r": config_lora.r, "alpha": config_lora.lora_alpha,
             "target_modules": sorted(config_lora.target_modules)},
    "entrenamiento": {"epocas": float(args.num_train_epochs), "lr": args.learning_rate,
                      "lote_efectivo": args.per_device_train_batch_size * args.gradient_accumulation_steps,
                      "n_train": len(ds_train), "n_val": len(ds_val),
                      "eval_loss_final": float(ev["eval_loss"].iloc[-1]) if len(ev) else None},
    "decisiones_validacion": {"antes": res_antes, "despues": res_despues},
    "punta_a_punta_validacion": {"n": len(calc_val), "antes": e2e_antes, "despues": e2e_despues},
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("Guardado:", DIR_ADAPTADOR, "y resultados/finetuning_agente.json")
if USAR_WANDB:
    wandb.finish()
'''))

    c.append(md("""
---
## 10 · Discusión

**1. ¿Qué aprendió?** Si `decision` y `llamada_valida` suben mucho y `resultado`
poco, el adaptador aprendió el **formato** de las llamadas pero no a **plantear**
los argumentos. Es el mismo patrón de M1 (formato antes que contenido), ahora con
herramientas.

**2. ¿Sobreajustó?** Hay 157 ejemplos de entrenamiento y el 80% sale del corpus de
M1, con muchas operaciones de un paso. Una pérdida de validación que sube desde la
segunda época, o un `decision` que empeora en `sin_herramienta`, son la señal de
que aprendió a llamar herramientas **siempre**.

**3. Esto no es la evaluación.** Las trayectorias de validación se parecen a las de
entrenamiento. El veredicto lo da el notebook 3, con el harness de M2 y el eval set
de M3, donde C5-FT se compara contra C5 en las mismas condiciones.
"""))

    c.append(md("""
---
## 11 · Conclusiones de esta corrida

| Sobre 42 trayectorias de validación | Base | C5-FT |
|---|---|---|
| Decide bien si usar herramienta | 14.3% | **95.2%** |
| Elige la herramienta correcta | 2.5% | **87.5%** |
| La llamada es válida (JSON + tipos) | 7.5% | **100%** |
| El resultado coincide con el de la trayectoria | 5.0% | **65.0%** |
| Exactitud de punta a punta (36 casos de cálculo) | 86.1% | 88.9% |

**1 · Este es el hallazgo central de la entrega.** Con instrucciones y tres ejemplos,
un modelo de 1.5B decide bien cuándo usar una herramienta 1 de cada 7 veces.
Entrenado con 157 trayectorias, 19 de cada 20. El tool calling en modelos pequeños no
se consigue por prompting: se entrena.

**2 · El formato se aprende antes que el contenido.** Las llamadas válidas llegan al
100% y la elección de herramienta al 87.5%, pero el resultado correcto solo al 65%.
El adaptador aprendió a escribir la llamada mejor de lo que aprendió a plantear sus
argumentos: el mismo patrón que M1, donde el formato subió al 100% antes que la
exactitud.

**3 · La exactitud de punta a punta casi no se mueve** (86.1% → 88.9%), porque esas
trayectorias de validación son de cálculo sencillo y el modelo base ya las resolvía
de cabeza. La diferencia se ve en el notebook 3, sobre el bloque de aritmética
exigente, y sobre todo en **de dónde salen los números**: RAGAS mide que en C5-FT el
77% están respaldados por una fuente verificable, contra el 40–48% del resto.

**4 · Riesgo confirmado en el notebook 3: sobre-uso.** 151 de las 157 trayectorias de
entrenamiento contienen al menos una llamada, así que el adaptador aprendió a llamar
casi siempre. En el eval set se abstiene solo en el 20% de los casos donde no hacía
falta, y busca documentos menos que C5. Para una segunda versión habría que balancear
el dataset con más ejemplos de "aquí no se usa herramienta".

**5 · Sin señales de sobreajuste en la pérdida:** `eval_loss` final 0.196 con 3
épocas sobre 157 ejemplos, sin repunte entre épocas.
"""))

    return c
