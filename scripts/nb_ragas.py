"""Notebook S10 (4/4) — RAGAS sobre las configuraciones C0–C5, calibración y scorecard final de M3."""

from __future__ import annotations

from nb_comun import celda_drive, code, encabezado, md
from nb_comun_agente import celda_eval_m3, celdas_modulos
from nb_comun_eval import celda_eval_set


CODIGO_CARGA = r'''
import pandas as pd

RUTA_RESPUESTAS = Path("resultados/respuestas_s10.jsonl")
if not RUTA_RESPUESTAS.exists():
    raise FileNotFoundError("No está resultados/respuestas_s10.jsonl. Ejecuten antes "
                            "S10_Lab_Agente_ReAct_Tutor_Matematicas.ipynb (notebook 3).")

respuestas = [json.loads(l) for l in RUTA_RESPUESTAS.read_text(encoding="utf-8").splitlines()]
detalle = pd.read_csv("resultados/detalle_s10.csv")
ACIERTO = {(r.config, r.id): bool(r.acierto) for r in detalle.itertuples()}
NOMBRES = list(dict.fromkeys(detalle["config"]))

print(f"Respuestas: {len(respuestas)} | configuraciones: {NOMBRES}")
faltan = [(r["config"], r["id"]) for r in respuestas if (r["config"], r["id"]) not in ACIERTO]
assert not faltan, f"hay respuestas sin fila en detalle_s10.csv (¿notebook 3 incompleto?): {faltan[:3]}"
'''


CODIGO_REFERENCIAS = r'''
# Referencia y clave de oro de cada caso.
#   referencia : la respuesta esperada del eval set (la usa RAGAS)
#   clave_oro  : el texto que DEBE aparecer en un chunk recuperado (versión objetiva)
# Las claves de conocimiento de M2 son las mismas que midió S08; las de los casos
# compuestos son el campo `dato` del eval set M3; los mal escritos heredan las de su original.
CLAVE_ORO = {
    "con-01": "examen final del periodo equivale al 40",
    "con-02": "octavo se introducen las ecuaciones",
    "con-03": "nota minima aprobatoria es 3.0",
    "con-04": "factor multiplicativo",
}
CLAVE_ORO.update({c["id"]: c["dato"] for c in eval_m3 if c.get("dato")})
for c in eval_m3:
    if c.get("original") in CLAVE_ORO:
        CLAVE_ORO[c["id"]] = CLAVE_ORO[c["original"]]

CASOS = {c["id"]: c for c in eval_set + eval_m3}
ESPERADAS = {a["id"]: a["herramientas"] for a in anotacion_m2}
ESPERADAS.update({c["id"]: c["herramientas"] for c in eval_m3})


def necesita_documentos(cid):
    return "documentos" in ESPERADAS[cid]["familias"]


print("Casos con clave de oro (métricas de contexto):", sorted(CLAVE_ORO))
print("Casos que requieren documentos:", sorted(c for c in CASOS if necesita_documentos(c)))
'''


CODIGO_EVALUADOR = r'''
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Evaluador de RAGAS: Qwen2.5-7B-Instruct en 4 bits (~5.5 GB en la T4).
#
# Decisión: el juez de M2 (1.5B) tuvo kappa 0.286 contra la verdad numérica y
# comprimía 31 de 42 notas en un 3. RAGAS exige juicios más finos (¿esta frase se
# deduce de este pasaje?), así que usa un evaluador de 7B. El juez de M2 NO se toca:
# la Dimensión 2 del harness sigue siendo comparable con M2, S07 y S08.
EVALUADOR = "Qwen/Qwen2.5-7B-Instruct"     # si no cabe: "Qwen/Qwen2.5-3B-Instruct"

eval_tok = AutoTokenizer.from_pretrained(EVALUADOR)
eval_model = AutoModelForCausalLM.from_pretrained(
    EVALUADOR, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16),
).eval()
print("Evaluador:", EVALUADOR, "| memoria GPU:",
      f"{torch.cuda.memory_allocated() / 1e9:.1f} GB" if torch.cuda.is_available() else "cpu")

_CACHE_EVAL = {}


@torch.no_grad()
def evaluador(system, user, max_new_tokens=64):
    clave = (system, user, max_new_tokens)
    if clave not in _CACHE_EVAL:
        prompt = eval_tok.apply_chat_template([{"role": "system", "content": system},
                                               {"role": "user", "content": user}],
                                              tokenize=False, add_generation_prompt=True)
        ids = eval_tok(prompt, return_tensors="pt", add_special_tokens=False).to(eval_model.device)
        out = eval_model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=eval_tok.eos_token_id)
        _CACHE_EVAL[clave] = eval_tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return _CACHE_EVAL[clave]


print(evaluador(SYSTEM_EVALUADOR, "Contexto: el examen vale 40%.\n\n1. El examen vale 40%.\n2. El examen vale 50%.\n\n"
                "¿Cada afirmación se deduce del contexto? Responde 'número: sí' o 'número: no'.", 20))
'''


CODIGO_SIMILITUD = r'''
from sentence_transformers import SentenceTransformer

st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def similitud(a, b):
    ea, eb = st.encode([a, b])
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))
'''


CODIGO_CALCULO = r'''
# Qué recibe cada métrica, por configuración:
#   contextos     : chunks que el sistema tuvo delante. C2–C4: el contexto fijo;
#                   C5: los que devolvió buscar_documentos. C0/C1: ninguno.
#   observaciones : lo que devolvieron las herramientas (incluidos los errores, porque
#                   "no está definida" se apoya en el error de dividir(15, 0)).
# Aplicabilidad (declarada, no escondida):
#   faithfulness                   si hubo contextos u observaciones
#   context_precision / recall     solo en casos que requieren documentos, tienen clave de oro
#                                  y el sistema tuvo contexto (C0/C1 no tienen; C5 puede no buscar)
#   answer_relevancy               siempre
CONFIGS_RAGAS = NOMBRES              # para abreviar, p. ej. ["C1 · +LoRA", "C2 · +RAG", "C5 · agente"]
RUTA_RAGAS = Path("resultados/ragas_s10.jsonl")

HECHOS = {}
if RUTA_RAGAS.exists():
    for l in RUTA_RAGAS.read_text(encoding="utf-8").splitlines():
        f = json.loads(l)
        HECHOS[(f["config"], f["id"])] = f
print("Ya calculados:", len(HECHOS))


def texto_observaciones(reg):
    return [f"{o['herramienta']}({json.dumps(o['argumentos'], ensure_ascii=False)}) -> "
            f"{o['resultado'] if o['ok'] else 'ERROR: ' + str(o['error'])}"
            for o in reg["observaciones"] if o["herramienta"] != "buscar_documentos"]


def ragas_de(reg):
    cid, caso = reg["id"], CASOS[reg["id"]]
    ctx, obs = reg["contextos"], texto_observaciones(reg)
    clave = CLAVE_ORO.get(cid) if necesita_documentos(cid) else None
    fila = {"config": reg["config"], "id": cid, "bloque": caso["bloque"], "acierto": ACIERTO[(reg["config"], cid)],
            "n_contextos": len(ctx), "n_observaciones": len(obs)}
    fila["faithfulness"] = faithfulness(evaluador, caso["input"], reg["respuesta"], ctx, obs)
    fila["faithfulness_numerica"] = faithfulness_numerica(caso["input"], reg["respuesta"], ctx, obs)
    fila["answer_relevancy"] = answer_relevancy(evaluador, similitud, caso["input"], reg["respuesta"])
    if clave is not None and ctx:
        cp, rel = context_precision(evaluador, caso["input"], caso["esperado"], ctx)
        fila.update(context_precision=cp, relevancias_llm=rel, relevancias_oro=relevancias_oro(ctx, clave),
                    context_precision_oro=context_precision_oro(ctx, clave),
                    context_recall=context_recall(evaluador, caso["esperado"], ctx),
                    context_recall_oro=context_recall_oro(ctx, clave))
    return fila


pendientes = [r for r in respuestas if r["config"] in CONFIGS_RAGAS and (r["config"], r["id"]) not in HECHOS]
t0 = time.time()
for i, reg in enumerate(pendientes, 1):
    fila = ragas_de(reg)
    HECHOS[(reg["config"], reg["id"])] = fila
    with RUTA_RAGAS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(fila, ensure_ascii=False, default=float) + "\n")
    if i % 25 == 0 or i == len(pendientes):
        print(f"  {i}/{len(pendientes)}  ({(time.time() - t0) / i:.1f} s por respuesta)")

df_ragas = pd.DataFrame([f for f in HECHOS.values() if f["config"] in CONFIGS_RAGAS])
print(df_ragas.shape)
'''


CODIGO_CALIBRACION = r'''
from scipy.stats import spearmanr

print("CALIBRACIÓN DEL EVALUADOR CONTRA LAS VERSIONES OBJETIVAS")
print("=" * 78)

# 1 · Relevancia de cada chunk: evaluador vs. "el chunk contiene la clave de oro"
llm_rel, oro_rel = [], []
for f in df_ragas.dropna(subset=["context_precision_oro"]).itertuples():
    if isinstance(f.relevancias_llm, list) and len(f.relevancias_llm) == len(f.relevancias_oro):
        llm_rel += f.relevancias_llm
        oro_rel += f.relevancias_oro
k_rel = kappa_cohen(llm_rel, oro_rel)
print(f"Relevancia de chunks ({len(oro_rel)} juicios): acuerdo "
      f"{np.mean(np.array(llm_rel) == np.array(oro_rel)):.1%} | kappa {k_rel:.3f}" if oro_rel else "sin datos")

# 2 · Context recall (binarizado en 0.5) vs. "la clave está en algún chunk"
cr = df_ragas.dropna(subset=["context_recall", "context_recall_oro"])
k_rec = kappa_cohen([int(v >= 0.5) for v in cr["context_recall"]], [int(v) for v in cr["context_recall_oro"]]) if len(cr) else None
print(f"Context recall ({len(cr)} casos): kappa {k_rec:.3f}" if k_rec is not None else "Context recall: sin datos")

# 3 · Faithfulness del evaluador vs. la numérica (¿los números salen de lo observado?)
ff = df_ragas.dropna(subset=["faithfulness", "faithfulness_numerica"])
rho = spearmanr(ff["faithfulness"], ff["faithfulness_numerica"]).correlation if len(ff) > 2 else None
print(f"Faithfulness LLM vs numérica ({len(ff)} respuestas): Spearman {rho:.3f}" if rho is not None else "sin datos")

CALIBRACION = {"kappa_relevancia_chunks": k_rel, "n_juicios_relevancia": len(oro_rel),
               "kappa_context_recall": k_rec, "spearman_faithfulness": rho}
print("=" * 78)
print("Lectura (S06): kappa >= 0.6 aceptable, >= 0.8 fuerte. Si el evaluador no llega,")
print("las conclusiones de contexto se apoyan en las versiones _oro, que son objetivas.")
'''


CODIGO_SCORECARD_RAGAS = r'''
METRICAS_RAGAS = ["faithfulness", "faithfulness_numerica", "context_precision", "context_precision_oro",
                  "context_recall", "context_recall_oro", "answer_relevancy"]

tabla = (df_ragas.groupby("config", sort=False)[METRICAS_RAGAS]
         .agg(lambda s: s.dropna().mean() if s.notna().any() else np.nan)
         .reindex([n for n in NOMBRES if n in CONFIGS_RAGAS]))
cobertura = df_ragas.groupby("config", sort=False)[METRICAS_RAGAS].agg(lambda s: s.notna().sum())
print("RAGAS por configuración (media sobre los casos donde la métrica aplica)")
print(tabla.round(3).to_string())
print("\nn por métrica:")
print(cobertura.reindex(tabla.index).to_string())
'''


CODIGO_ALERTA = r'''
# La alerta de S10: "una respuesta puede acertar por la razón equivocada".
# Aciertos con fidelidad numérica < 1: el número final es correcto, pero algún número
# de la respuesta no sale de la pregunta, de los documentos ni de las herramientas
# (se calculó "de cabeza").
alerta = df_ragas[(df_ragas["acierto"]) & (df_ragas["faithfulness_numerica"].notna())]
resumen_alerta = alerta.groupby("config", sort=False).apply(
    lambda g: pd.Series({"aciertos_con_numeros": len(g),
                         "respaldados_del_todo": int((g["faithfulness_numerica"] >= 0.999).sum()),
                         "con_calculo_de_cabeza": int((g["faithfulness_numerica"] < 0.999).sum())}))
print(resumen_alerta.reindex([n for n in NOMBRES if n in resumen_alerta.index]).to_string())
'''


CODIGO_DIAGNOSTICO = r'''
# Harness × RAGAS en los casos que requieren documentos: ¿falló el retrieval o la generación?
def causa(f):
    if f["acierto"]:
        return "acierto"
    if f["n_contextos"] == 0:
        return "SIN CONTEXTO: no tiene RAG o no buscó"
    if f["context_recall_oro"] == 0:
        return "RETRIEVAL: el dato no llegó"
    return "GENERACIÓN: el dato llegó y no lo usó"


docs = df_ragas[df_ragas["id"].map(necesita_documentos)].copy()
docs["causa"] = docs.apply(causa, axis=1)
print(pd.crosstab(docs["config"], docs["causa"]).reindex([n for n in NOMBRES if n in set(docs["config"])]).to_string())
'''


CODIGO_FINAL = r'''
# Scorecard final de la entrega M3: el scorecard del agente (notebook 3) + RAGAS.
sc_agente = pd.read_csv("resultados/scorecard_agente.csv").set_index("metrica")
filas_ragas = pd.DataFrame({
    "RAGAS · faithfulness": tabla["faithfulness"],
    "RAGAS · faithfulness numérica (objetiva)": tabla["faithfulness_numerica"],
    "RAGAS · context precision": tabla["context_precision"],
    "RAGAS · context precision (oro)": tabla["context_precision_oro"],
    "RAGAS · context recall": tabla["context_recall"],
    "RAGAS · context recall (oro)": tabla["context_recall_oro"],
    "RAGAS · answer relevancy": tabla["answer_relevancy"],
}).T
scorecard_final = pd.concat([sc_agente[[c for c in sc_agente.columns if c in filas_ragas.columns]], filas_ragas])
scorecard_final.index.name = "metrica"
pd.set_option("display.width", 200)
print(scorecard_final.apply(pd.to_numeric, errors="coerce").round(3).to_string())
'''


CODIGO_WANDB = r'''
# Weights & Biases (extra de S10): una corrida por configuración + tabla comparativa + traza.
# Offline por defecto: no pide cuenta. Para subirlo: `wandb sync wandb/offline-run-*`.
USAR_WANDB = True
if USAR_WANDB:
    import wandb
    os.environ.setdefault("WANDB_MODE", "offline")
    PROYECTO_W = "tutor-matematicas-arquitecturas"
    numerico = scorecard_final.apply(pd.to_numeric, errors="coerce")
    for n in numerico.columns:
        run = wandb.init(project=PROYECTO_W, name=f"s10-{n}", group="s10-agente",
                         config={"configuracion": n, "evaluador_ragas": EVALUADOR}, reinit=True)
        wandb.log({k: v for k, v in numerico[n].dropna().items()})
        wandb.finish()

    run = wandb.init(project=PROYECTO_W, name="s10-comparativa", group="s10-agente", reinit=True)
    t = wandb.Table(dataframe=numerico.reset_index())
    wandb.log({"scorecard_final": t, "calibracion": CALIBRACION})
    # La traza de C5 en un caso compuesto: qué pensó, qué llamó y qué observó.
    reg = next((r for r in respuestas if r["config"].startswith("C5 ") and r["id"] == "com-01"), None)
    if reg:
        traza = wandb.Table(columns=["paso", "tipo", "datos"])
        for e in reg["traza"]:
            traza.add_data(e["paso"], e["tipo"], json.dumps(e["datos"], ensure_ascii=False, default=str)[:500])
        wandb.log({"traza_c5_com01": traza})
    wandb.finish()
    print("Corridas registradas en W&B (modo:", os.environ["WANDB_MODE"], ")")
'''


CODIGO_ARTEFACTOS = r'''
scorecard_final.to_csv("resultados/scorecard_final_m3.csv", encoding="utf-8")
tabla.to_csv("resultados/ragas_scorecard_s10.csv", encoding="utf-8")
Path("resultados/ragas_s10.json").write_text(json.dumps({
    "evaluador": EVALUADOR,
    "calibracion": CALIBRACION,
    "ragas_por_config": json.loads(tabla.to_json(orient="index")),
    "cobertura": json.loads(cobertura.to_json(orient="index")),
    "diagnostico_documentos": json.loads(pd.crosstab(docs["config"], docs["causa"]).to_json(orient="index")),
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
for p in ["ragas_s10.jsonl", "ragas_scorecard_s10.csv", "ragas_s10.json", "scorecard_final_m3.csv"]:
    print(f"  resultados/{p}")
'''


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "RAGAS y scorecard final del tutor agéntico",
        "Sesión 10 · Módulo 3 — Evaluar por dentro: contexto, fidelidad y relevancia (4/4)",
    ))

    c.append(md("""
## 1 · Introducción

### Por qué no basta el harness

El harness dice **si** el tutor acertó. No dice **por qué**:

- ¿El contexto recuperado era el correcto?
- ¿La respuesta se apoya en lo que el sistema observó, o el modelo ya lo sabía?

Una respuesta puede acertar por la razón equivocada o fallar por un contexto pobre.
RAGAS separa esos casos y **complementa** al harness, no lo reemplaza. La entrega
M3 pide los dos juntos.

### Las cuatro métricas, adaptadas a este dominio

| Métrica | Pregunta | Adaptación para un agente con herramientas |
|---|---|---|
| **faithfulness** | ¿Cada afirmación se apoya en lo observado? | Lo observado = documentos **y resultados de herramientas**. Un cálculo hecho "de cabeza" no es fiel aunque sea correcto. |
| **context precision** | ¿Los chunks útiles quedaron arriba? | Solo en casos que requieren documentos. |
| **context recall** | ¿Llegó todo lo necesario? | Ídem. |
| **answer relevancy** | ¿La respuesta va al grano? | Sin cambios. |

### Dos decisiones que hay que declarar

1. **Evaluador de 7B** (Qwen2.5-7B-Instruct en 4 bits), no el juez de 1.5B de M2,
   que tuvo kappa 0.286. El juez de M2 sigue intacto en el harness, para no romper la
   comparabilidad.
2. **Cada métrica tiene una versión objetiva** (claves de oro y números respaldados)
   que **calibra** al evaluador, igual que en M2 se calibró el juez contra la verdad
   numérica. Si el evaluador no pasa la calibración, las conclusiones se apoyan en las
   objetivas.

> **Sobre la librería `ragas`.** Hace esto mismo por dentro, pero por defecto llama a
> OpenAI. Aquí se implementa "a mano" (como en el Lab C de S10) con un evaluador local,
> lo que además permite adaptar la faithfulness a las herramientas. Al final queda la
> celda de referencia con la librería.
"""))

    c.append(md("""
## 2 · Objetivos

1. Calcular las cuatro métricas de RAGAS sobre las respuestas de C0–C5 (y C5-FT).
2. **Calibrar** el evaluador contra las versiones objetivas.
3. Detectar los **aciertos por la razón equivocada**.
4. Diagnosticar los fallos de conocimiento: ¿retrieval o generación?
5. Construir el **scorecard final de M3** (harness + agente + RAGAS) y registrarlo en W&B.
"""))

    c.append(md("""
## 0 · Preparación del entorno

> **GPU T4.** Solo se carga el evaluador de 7B (~5.5 GB en 4 bits) y MiniLM. El
> cálculo se guarda incrementalmente en `resultados/ragas_s10.jsonl` y se retoma si
> Colab se desconecta.
"""))
    c.append(code('''
%pip install -q "transformers>=4.44" "accelerate>=0.33" bitsandbytes sentence-transformers pandas scipy wandb
print("Listo.")
'''))
    c.append(code('''
import os, random, re, time, json
import numpy as np
import torch
import transformers
from pathlib import Path

SEMILLA = 42
random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"transformers {transformers.__version__} | torch {torch.__version__} | {DEVICE}")
'''))
    c.extend(celda_drive())

    c.append(md("""
---
## 3 · Los datos
"""))
    c.extend(celda_eval_set(prefijo="3.1"))
    c.extend(celda_eval_m3(prefijo="3.2"))
    c.append(md("""
### 3.3 · Las respuestas del notebook 3

Cada fila trae la pregunta, la respuesta, los **contextos** que el sistema tuvo
delante y las **observaciones** de las herramientas. Es todo lo que RAGAS necesita,
sin volver a generar nada.
"""))
    c.append(code(CODIGO_CARGA))
    c.append(code(CODIGO_REFERENCIAS))

    c.append(md("""
---
## 4 · El código de RAGAS (copia literal de `scripts/agente/`)
"""))
    c.extend(celdas_modulos(["registro", "nucleo", "ragas_tutor"]))

    c.append(md("""
---
## 5 · El evaluador
"""))
    c.append(code(CODIGO_EVALUADOR))
    c.append(code(CODIGO_SIMILITUD))

    c.append(md("""
---
## 6 · Cálculo
"""))
    c.append(code(CODIGO_CALCULO))

    c.append(md("""
---
## 7 · Calibración del evaluador

Antes de leer una sola métrica: **¿el evaluador tiene razón?** Tres comprobaciones
contra las versiones objetivas.
"""))
    c.append(code(CODIGO_CALIBRACION))

    c.append(md("""
---
## 8 · RAGAS por configuración
"""))
    c.append(code(CODIGO_SCORECARD_RAGAS))
    c.append(md("""
> **Cómo leerlo (S10).**
> - **context precision/recall bajas** → problema de retrieval. Hay que volver a S08.
> - **faithfulness baja** → el modelo no se apoya en lo que observó: ajustar el prompt
>   o quitar ruido.
> - **answer relevancy baja** → la respuesta divaga.
>
> En C0 y C1 la faithfulness no aplica (no observan nada), y la de contexto solo
> aplica en C2–C5.
"""))

    c.append(md("""
### 8.1 · Aciertos por la razón equivocada
"""))
    c.append(code(CODIGO_ALERTA))
    c.append(md("""
### 8.2 · Fallos de conocimiento: ¿retrieval o generación?

Es el mismo diagnóstico de S07 §11, ahora para todas las configuraciones y con
recall de oro. En C5 aparece una causa nueva: **no buscó** cuando hacía falta, que es
un fallo de la **decisión** del agente, no del retrieval.
"""))
    c.append(code(CODIGO_DIAGNOSTICO))

    c.append(md("""
---
## 9 · El scorecard final de M3

Harness de M2 + eval set M3 + herramientas + proceso + robustez (notebook 3) +
RAGAS (este notebook), en una tabla.
"""))
    c.append(code(CODIGO_FINAL))

    c.append(md("""
### 9.1 · Weights & Biases (opcional)

Una corrida por configuración, la tabla comparativa y la traza del agente en un
caso compuesto. Corre **offline** sin cuenta, como en el extra de S10.
"""))
    c.append(code(CODIGO_WANDB))

    c.append(md("""
### 9.2 · Referencia: la librería `ragas`

Para quien quiera contrastar con la implementación oficial (necesita un LLM y
embeddings; por defecto, OpenAI):

```python
# %pip install -q ragas datasets
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
# from datasets import Dataset
# filas = [r for r in respuestas if r["config"] == "C5 · agente"]
# ds = Dataset.from_dict({
#     "question":     [CASOS[r["id"]]["input"] for r in filas],
#     "answer":       [r["respuesta"] for r in filas],
#     "contexts":     [r["contextos"] + texto_observaciones(r) for r in filas],
#     "ground_truth": [CASOS[r["id"]]["esperado"] for r in filas],
# })
# evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
```
"""))

    c.append(md("""
---
## 10 · Artefactos
"""))
    c.append(code(CODIGO_ARTEFACTOS))

    c.append(md("""
---
## 11 · Discusión: la lectura honesta que pide la entrega

**1. ¿Qué técnica movió qué?** Crucen los deltas del notebook 3 con este scorecard.
Por ejemplo: si C5 sube en *compuesto* y su context recall (oro) es alto, el agente
**decidió buscar** y el retrieval respondió. Si sube la faithfulness numérica de C1 a
C3, las herramientas hicieron que los números salgan de algún sitio verificable.

**2. ¿Qué costó?** Tokens y segundos por pregunta (notebook 3). Un agente que gana
pocos casos y multiplica el coste se reporta como tal.

**3. ¿Qué falla queda pendiente?** Los *fallos por generación* de 8.2, los *aciertos
con cálculo de cabeza* de 8.1 y lo que diga la calibración sobre el propio evaluador.

**4. ¿Se justifica el agente?** Solo si el harness mejora **y** RAGAS confirma que
mejora por la razón correcta. Si mejora el harness pero cae la faithfulness, acertó
por suerte; si mejora RAGAS pero no el harness, el tutor no es mejor para el
estudiante.
"""))

    c.append(md("""
---
## 12 · Conclusiones de esta corrida

| | C0 | C1 | C2 | C3 | C4 | C5 | C5-FT |
|---|---|---|---|---|---|---|---|
| faithfulness (7B) | — | — | 0.65 | 0.69 | 0.69 | 0.77 † | 0.73 |
| faithfulness numérica (objetiva) | 0.36 | 0.32 | 0.45 | 0.48 | 0.48 | 0.40 | **0.77** |
| context precision (oro) | — | — | 0.90 | 0.90 | 0.90 | **1.00** | **1.00** |
| context recall (oro) | — | — | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| answer relevancy | 0.68 | **0.71** | 0.67 | 0.65 | 0.67 | 0.70 | 0.66 |

† n = 10: C5 casi nunca observó nada, así que su faithfulness no es comparable con
las demás columnas, medidas sobre 48–53 casos. La cobertura está en la tabla de `n`.

**1 · Primero la calibración, porque condiciona todo lo demás.** Kappa 0.72 en
relevancia de chunks: aceptable según el criterio de S06. Pero Spearman 0.09 entre la
faithfulness del 7B y la numérica objetiva, y kappa 0 en context recall porque las 15
consultas documentales tienen recall de oro = 1 y no hay varianza que medir. Por eso
las conclusiones de fundamento se apoyan en la **faithfulness numérica**, no en el
juicio del evaluador.

**2 · La alerta de S10 se confirma, y las herramientas la corrigen.** De los aciertos
de C1, **0 de 24** tienen todos sus números respaldados por la pregunta, un documento
o una herramienta: el tutor sin herramientas acierta calculando de cabeza. En C5-FT
son **14 de 29**, y su faithfulness numérica es la más alta (0.77) frente al 0.32 de
C1, que es la más baja de todas. Es la evidencia más clara de que las herramientas
cambian la *naturaleza* del acierto, no solo su cantidad.

**3 · El retrieval no falla nunca.** Context recall de oro = 1.00 en todas las
configuraciones con contexto, y precision de oro 0.90–1.00. Coincide con S08: el
cuello de botella nunca estuvo en recuperar.

**4 · Lo que cambió es la naturaleza del fallo.** En C2–C4 los fallos de conocimiento
son 6–8 casos de *"el dato llegó y el modelo no lo usó"* (generación). En C5 y C5-FT
son 4 y 8 casos de *"no buscó"* (decisión del agente). Dar autonomía no eliminó el
fallo: lo movió de la generación a la decisión, y en C5-FT lo empeoró.

**5 · El evaluador sobreestima la relevancia del contexto.** Da context precision 1.00
en todas las configuraciones, mientras la versión objetiva marca 0.90 en C2–C4. Es
consistente con el sesgo hacia el "sí" que ya se le vio al juez de 1.5B en M2, ahora
medido en un modelo cinco veces mayor.

**6 · RAGAS y el harness dicen cosas distintas, y por eso se reportan juntos.** El
harness dice que C1 es tan bueno como C5-FT en el eval set de M2; RAGAS dice que C1
llega ahí sin apoyarse en nada verificable. Un tutor que acierta de memoria es más
frágil en producción que uno que consulta y calcula, aunque el scorecard los empate.
"""))

    return c
