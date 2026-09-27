"""Notebook 3 — Pipeline modular de dos etapas: BERT clasifica, Qwen genera."""

from __future__ import annotations

from nb_comun import (
    celda_carga_datos, celda_datos, celda_drive, celda_instalacion,
    celda_metricas, celda_resultados, celda_wandb, code, encabezado, md,
)


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "Pipeline Qwen + BERT: clasificación y generación especializada",
        "Notebook 3 de 4 · Sistema modular de dos etapas",
    ))

    # ------------------------------------------------------------------ 1
    c.append(md("""
## 1 · Introducción

### La idea

Los notebooks 1 y 2 entrenaron dos modelos que hacen cosas distintas y
complementarias. Este notebook los conecta:

```
                        ┌──────────────────────────┐
                        │        USUARIO           │
                        │ "¿Cuál es el 25% de 420  │
                        │   estudiantes?"          │
                        └────────────┬─────────────┘
                                     │  texto libre
                                     ▼
                    ┌────────────────────────────────┐
                    │   ETAPA 1 · CLASIFICACIÓN      │
                    │   BETO fine-tuned (110M)       │
                    │   encoder bidireccional        │
                    └────────────────┬───────────────┘
                                     │  categoría + confianza
                                     ▼
                    ┌────────────────────────────────┐
                    │   SELECCIÓN DE ESTRATEGIA      │
                    │   categoría -> instrucciones   │
                    │   didácticas especializadas    │
                    └────────────────┬───────────────┘
                                     │  prompt condicionado
                                     ▼
                    ┌────────────────────────────────┐
                    │   ETAPA 2 · GENERACIÓN         │
                    │   Qwen2.5-1.5B + LoRA          │
                    │   decoder autoregresivo        │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │  Paso 1: ...             │
                        │  Paso 2: ...             │
                        │  Respuesta final: 105    │
                        └──────────────────────────┘
```

### Por qué un sistema modular

Es un patrón clásico de ingeniería de sistemas de IA, con ventajas concretas:

- **Cada componente hace una sola cosa y se puede medir por separado.** Si el
  tutor falla, se sabe si falló el enrutador o el generador. En un modelo único
  eso es opaco.
- **Se sustituyen piezas por separado.** Se puede cambiar el generador sin
  reentrenar el clasificador, o añadir categorías al clasificador sin tocar el
  generador.
- **Cada etapa usa la arquitectura adecuada a su tarea.** Encoder bidireccional
  para entender, decoder autoregresivo para escribir.
- **Permite reglas de negocio en medio.** Rechazar preguntas fuera de dominio,
  enrutar a una calculadora simbólica, pedir aclaración si la confianza es baja.

### La desventaja estructural: propagación de error

Un sistema en cascada multiplica las probabilidades de acierto. Si BETO acierta
el 85% de las veces y Qwen resuelve bien el 60%, el techo del sistema no es
60%: en los casos mal clasificados Qwen recibe instrucciones equivocadas.

**Este notebook está diseñado para medir exactamente ese efecto**, no para
asumirlo. Por eso comparamos tres configuraciones:

| Configuración | Qué mide |
|---|---|
| **A. Qwen solo** (sin categoría) | El baseline: ¿aporta algo el enrutamiento? |
| **B. Pipeline real** (categoría predicha por BETO) | El sistema tal como funcionaría en producción. |
| **C. Oráculo** (categoría real del corpus) | El techo del pipeline si el clasificador fuera perfecto. |

La diferencia **C − B** es el costo de los errores del clasificador.
La diferencia **C − A** es el beneficio máximo que puede aportar condicionar
por categoría. Si C ≈ A, toda la arquitectura modular es complejidad sin
retorno, y ese es un resultado perfectamente válido que hay que reportar.
"""))

    # ------------------------------------------------------------------ 2
    c.append(md("""
## 2 · Objetivos

1. Ensamblar un sistema de dos etapas con los modelos entrenados en los
   notebooks 1 y 2, cargados desde disco.
2. Diseñar una **estrategia didáctica por categoría** que condicione la
   generación.
3. Comparar las tres configuraciones (A, B, C) sobre los mismos 33 ejemplos de
   validación y con las mismas métricas de los notebooks anteriores.
4. Cuantificar la **propagación de error**: exactitud del generador cuando la
   clasificación fue correcta frente a cuando fue incorrecta.
5. Medir la **latencia** de cada etapa, que es el argumento práctico a favor o
   en contra de un pipeline en producción.

> **Requisito previo.** Este notebook no entrena nada: consume
> `adaptadores/qwen-lora` (notebook 1) y `modelos/bert-clasificador`
> (notebook 2). Si trabajan en Colab, ejecuten los tres notebooks en la misma
> sesión o guarden esas carpetas en Google Drive.
"""))

    c.extend(celda_instalacion())
    c.extend(celda_drive())

    # ------------------------------------------------------------------ 3
    c.append(md("""
## 3 · Arquitectura del sistema

Las dos arquitecturas conviven, cada una en el papel para el que es buena:

| | Etapa 1 · BETO | Etapa 2 · Qwen2.5 |
|---|---|---|
| Tipo | Encoder-only | Decoder-only |
| Atención | Bidireccional | Causal |
| Parámetros | 110 M | 1.500 M |
| Entrada | Enunciado | Enunciado + estrategia de categoría |
| Salida | 1 de 11 clases | Texto libre |
| Inferencia | Una pasada | ~200 pasadas (token a token) |
| Entrenamiento | Fine-tuning completo | LoRA (0.4% de los pesos) |
| Coste relativo | ≈1 | ≈100 |

Ese último renglón es el que justifica la asimetría del diseño: el clasificador
es tan barato que añadirlo al pipeline no cambia la latencia percibida, y a
cambio puede evitar una generación desencaminada.
"""))

    c.append(code('''
MODELO_QWEN  = "Qwen/Qwen2.5-1.5B-Instruct"   # debe coincidir con el notebook 1
DIR_ADAPTADOR = "adaptadores/qwen-lora"        # producido por el notebook 1
DIR_BERT      = "modelos/bert-clasificador"    # producido por el notebook 2

MAX_TOKENS_GEN = 200
LONGITUD_MAX_CLF = 64
'''))

    c.append(code('''
from pathlib import Path

faltan = [d for d in (DIR_ADAPTADOR, DIR_BERT) if not Path(d).exists()]
if faltan:
    raise FileNotFoundError(
        f"Faltan artefactos de los notebooks previos: {faltan}\\n"
        "Ejecuten primero S04_Lab_Fine_tuning_Qwen.ipynb (etapa 2) y "
        "S04_Lab_Fine_tuning_BERT.ipynb (etapa 1)."
    )
print("Artefactos encontrados:")
for d in (DIR_ADAPTADOR, DIR_BERT):
    print(f"  {d}")
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

### Estrategias didácticas por categoría

Esta es la pieza de diseño central del notebook: qué hacemos realmente con la
categoría una vez que la conocemos.

La opción ingenua sería inyectar la etiqueta en el prompt ("Este es un problema
de porcentajes"). Aporta poco: el modelo ya puede inferirlo del enunciado.

La opción útil es asociar a cada categoría **el procedimiento que un profesor
enseñaría** para ese tipo de problema. La categoría deja de ser una etiqueta y
pasa a ser una clave para recuperar conocimiento pedagógico que el modelo no
tiene por qué haber aprendido de 132 ejemplos.

Esto convierte al clasificador en algo parecido a un recuperador: enruta hacia
la instrucción especializada correcta. Es la misma lógica de un sistema RAG,
con un espacio de recuperación de 11 elementos en lugar de un índice vectorial.
"""))

    c.append(code('''
ESTRATEGIAS = {
    "suma": "Identifica las cantidades que se juntan y súmalas. Verifica que las unidades coincidan y alinea los decimales si los hay.",
    "resta": "Identifica la cantidad inicial y la que se quita o se compara. Recuerda que el resultado puede ser negativo.",
    "multiplicacion": "Identifica el valor unitario y cuántas veces se repite. Multiplica. Si hay decimales, cuenta las cifras decimales del resultado.",
    "division": "Identifica el total y en cuántas partes se reparte. Divide. Si el contexto no admite fracciones (personas, buses, cajas), redondea y explica por qué.",
    "operaciones_combinadas": "Aplica la jerarquía de operaciones: primero los paréntesis, después multiplicaciones y divisiones de izquierda a derecha, y por último sumas y restas. Resuelve una operación por paso.",
    "potencias_raices": "Calcula primero las potencias y raíces, y solo después las demás operaciones. Explica qué significa la potencia o la raíz que estás calculando.",
    "fracciones": "Si sumas o restas, halla el denominador común. Si multiplicas, multiplica numeradores y denominadores. Si divides, multiplica por la fracción inversa. Simplifica el resultado.",
    "porcentajes": "Convierte el porcentaje a decimal dividiendo entre 100 y multiplica. Si piden un descuento o un aumento, calcula primero la parte y después súmala o réstala al valor original.",
    "ecuaciones": "Plantea la ecuación con una incógnita. Despeja aplicando la misma operación a ambos lados hasta dejar la incógnita sola.",
    "geometria": "Escribe explícitamente la fórmula que vas a usar antes de sustituir los valores. Indica las unidades del resultado (lineales, cuadradas o cúbicas).",
    "estadistica_probabilidad": "Para un promedio, suma los valores y divide entre cuántos son. Para una probabilidad, cuenta los casos favorables y divídelos entre los casos posibles.",
}

assert set(ESTRATEGIAS) == set(CATEGORIAS), "Falta una estrategia para alguna categoría"
for cat, est in list(ESTRATEGIAS.items())[:3]:
    print(f"[{cat}]\\n  {est}\\n")
'''))

    c.append(code('''
INSTRUCCION = (
    "Eres un tutor de matemáticas. Resuelve el siguiente problema explicando "
    "el procedimiento paso a paso y termina con la respuesta final."
)


def construir_prompt(entrada, categoria=None):
    """Prompt de generación, con o sin condicionamiento por categoría.

    Sin categoría es EXACTAMENTE el prompt del notebook 1: así la comparación
    A vs B aísla el efecto del enrutamiento y nada más.
    """
    sistema = INSTRUCCION
    if categoria is not None:
        sistema += (
            f"\\nEste problema es de tipo '{categoria}'. "
            f"Estrategia recomendada: {ESTRATEGIAS[categoria]}"
        )
    mensajes = [
        {"role": "system", "content": sistema},
        {"role": "user", "content": entrada},
    ]
    return tokenizer_qwen.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
'''))

    # ------------------------------------------------------------------ 6
    c.append(md("""
## 6 · Tokenización y carga de los modelos

Cada etapa conserva su propio tokenizador. No son intercambiables: BETO usa
WordPiece con 31k tokens, Qwen usa BPE con 151k. Un `input_ids` de uno no
significa nada para el otro. En un pipeline con dos tokenizadores, mezclarlos
es un error silencioso —no lanza excepción, simplemente produce basura—, así
que los nombramos de forma inequívoca: `tokenizer_bert` y `tokenizer_qwen`.
"""))

    c.append(code('''
import torch
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer)
from peft import PeftModel

# ---- Etapa 1: clasificador
tokenizer_bert = AutoTokenizer.from_pretrained(DIR_BERT)
modelo_bert = AutoModelForSequenceClassification.from_pretrained(DIR_BERT).to(DEVICE)
modelo_bert.eval()
print(f"Etapa 1 cargada: {modelo_bert.config._name_or_path} "
      f"({modelo_bert.config.num_labels} clases)")

# ---- Etapa 2: generador
tokenizer_qwen = AutoTokenizer.from_pretrained(DIR_ADAPTADOR)
if tokenizer_qwen.pad_token is None:
    tokenizer_qwen.pad_token = tokenizer_qwen.eos_token

base = AutoModelForCausalLM.from_pretrained(MODELO_QWEN, torch_dtype=torch.float16).to(DEVICE)
modelo_qwen = PeftModel.from_pretrained(base, DIR_ADAPTADOR)
# Fusionamos los pesos LoRA en el modelo base: en inferencia elimina la
# indirección del adaptador y acelera la generación entre un 5 y un 10%.
modelo_qwen = modelo_qwen.merge_and_unload()
modelo_qwen.eval()
print(f"Etapa 2 cargada: {MODELO_QWEN} + adaptador LoRA (fusionado)")
'''))

    c.append(code('''
# El orden de las etiquetas del clasificador debe coincidir con CATEGORIAS.
# Si no coincidiera, el pipeline enrutaría con estrategias equivocadas sin dar
# ningún error visible: es el fallo silencioso más peligroso de este diseño.
etiquetas_modelo = [modelo_bert.config.id2label[i] for i in range(modelo_bert.config.num_labels)]
assert etiquetas_modelo == CATEGORIAS, (
    f"Desalineación de etiquetas.\\n  modelo: {etiquetas_modelo}\\n  corpus: {CATEGORIAS}"
)
print("Etiquetas alineadas correctamente entre el clasificador y el corpus.")
'''))

    # ------------------------------------------------------------------ 7
    c.append(md("""
## 7 · Baseline

El baseline de este notebook **no es el modelo sin entrenar**: eso ya se midió
en el notebook 1. Aquí la pregunta es otra —¿aporta algo el enrutamiento?— y
por tanto el punto de comparación correcto es **Qwen fine-tuned trabajando
solo, sin categoría** (configuración A).

Elegir mal el baseline es el error metodológico más frecuente al evaluar
sistemas compuestos: comparar el pipeline completo contra un modelo sin
entrenar demuestra que el fine-tuning funcionó, no que la arquitectura modular
aporte nada.
"""))

    c.append(code('''
from tqdm.auto import tqdm


@torch.no_grad()
def clasificar(textos, batch=16):
    """Etapa 1. Devuelve (categoría, confianza) para cada texto."""
    salida = []
    for i in range(0, len(textos), batch):
        lote = tokenizer_bert(textos[i:i + batch], padding=True, truncation=True,
                              max_length=LONGITUD_MAX_CLF, return_tensors="pt").to(DEVICE)
        probs = modelo_bert(**lote).logits.softmax(-1)
        conf, idx = probs.max(-1)
        salida.extend(zip([CATEGORIAS[j] for j in idx.cpu().tolist()],
                          conf.cpu().tolist()))
    return salida


@torch.no_grad()
def generar(entradas, categorias=None, max_new_tokens=MAX_TOKENS_GEN):
    """Etapa 2. `categorias=None` genera sin condicionamiento (configuración A)."""
    salidas = []
    for i, entrada in enumerate(tqdm(entradas, desc="generando")):
        cat = categorias[i] if categorias is not None else None
        prompt = construir_prompt(entrada, cat)
        inputs = tokenizer_qwen(prompt, return_tensors="pt",
                                add_special_tokens=False).to(DEVICE)
        out = modelo_qwen.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer_qwen.pad_token_id,
            eos_token_id=tokenizer_qwen.eos_token_id,
        )
        nuevos = out[0][inputs["input_ids"].shape[1]:]
        salidas.append(tokenizer_qwen.decode(nuevos, skip_special_tokens=True).strip())
    return salidas
'''))

    c.append(code('''
# ---- Configuración A: Qwen fine-tuned solo (baseline del pipeline)
gen_A = generar([r["entrada"] for r in val], categorias=None)
metricas_A = evaluar_generacion(gen_A, val)

print()
print("A · Qwen fine-tuned sin enrutamiento")
for k, v in metricas_A.items():
    if not k.startswith("_"):
        print(f"  {k:16s}: {v}")
'''))

    # ------------------------------------------------------------------ 8
    c.append(md("""
## 8 · Configuración del pipeline

No hay hiperparámetros de entrenamiento que ajustar aquí, porque **este
notebook no entrena nada nuevo**: compone dos modelos ya entrenados. Lo que sí
hay que configurar son las decisiones de diseño del sistema.

### Umbral de confianza

El clasificador devuelve una probabilidad además de la etiqueta. Con ella se
puede decidir *no enrutar* cuando la predicción es dudosa y caer en el prompt
genérico, que es exactamente la configuración A.

Es una salvaguarda barata contra la propagación de error: si el clasificador no
está seguro, el sistema se degrada al comportamiento del modelo único en lugar
de recibir una estrategia equivocada.

Con `UMBRAL_CONFIANZA = 0.0` el mecanismo queda desactivado y se enruta siempre;
así medimos primero el pipeline puro. Después pueden subirlo y volver a evaluar
para ver si conviene.
"""))

    c.append(code('''
UMBRAL_CONFIANZA = 0.0   # 0.0 = enrutar siempre; prueben 0.5 o 0.7 después


def pipeline(entradas, umbral=UMBRAL_CONFIANZA, verbose=False):
    """Sistema completo de dos etapas."""
    clasificaciones = clasificar(entradas)
    categorias, descartadas = [], 0
    for cat, conf in clasificaciones:
        if conf >= umbral:
            categorias.append(cat)
        else:
            categorias.append(None)     # se cae al prompt genérico
            descartadas += 1
    if verbose and descartadas:
        print(f"{descartadas} predicciones por debajo del umbral: se usó prompt genérico.")
    generaciones = generar(entradas, categorias=categorias)
    return generaciones, clasificaciones, categorias
'''))

    c.append(md("""
### Demostración del flujo con un solo ejemplo

Antes de evaluar en lote, conviene ver el sistema funcionando paso a paso.
"""))

    c.append(code('''
ejemplo = demo[0]
print("USUARIO:", ejemplo["entrada"])
print()

cat, conf = clasificar([ejemplo["entrada"]])[0]
print(f"ETAPA 1 · BETO clasifica: '{cat}' (confianza {conf:.1%})")
print(f"          categoría real: '{ejemplo['categoria']}'  "
      f"-> {'correcta' if cat == ejemplo['categoria'] else 'INCORRECTA'}")
print()
print(f"SELECCIÓN · estrategia asociada:")
print(f"          {ESTRATEGIAS[cat]}")
print()
salida = generar([ejemplo["entrada"]], categorias=[cat])[0]
print("ETAPA 2 · Qwen genera:")
print(salida)
print()
print(f"Referencia: {ejemplo['salida']}")
'''))

    # ------------------------------------------------------------------ 9
    c.append(md("""
## 9 · Ejecución del pipeline

Ejecutamos las configuraciones B (pipeline real) y C (oráculo) sobre los mismos
33 ejemplos de validación.

La configuración C es un **experimento de ablación**: sustituye el clasificador
por sus etiquetas verdaderas. No es un sistema desplegable —en producción no
existe la etiqueta real—, pero es la única forma de separar "el enrutamiento no
sirve" de "el clasificador no acierta lo suficiente". Son dos diagnósticos con
soluciones completamente distintas.
"""))

    c.append(code('''
# ---- Configuración B: pipeline real (categoría predicha)
gen_B, clasificaciones, cats_predichas = pipeline([r["entrada"] for r in val], verbose=True)
metricas_B = evaluar_generacion(gen_B, val)

aciertos_clf = [c == r["categoria"] for (c, _), r in zip(clasificaciones, val)]
accuracy_clf = sum(aciertos_clf) / len(aciertos_clf)

print()
print("B · Pipeline BERT -> Qwen")
print(f"  accuracy del clasificador: {accuracy_clf:.1%}")
for k, v in metricas_B.items():
    if not k.startswith("_"):
        print(f"  {k:16s}: {v}")
'''))

    c.append(code('''
# ---- Configuración C: oráculo (categoría real)
gen_C = generar([r["entrada"] for r in val], categorias=[r["categoria"] for r in val])
metricas_C = evaluar_generacion(gen_C, val)

print()
print("C · Oráculo (categoría real, techo del pipeline)")
for k, v in metricas_C.items():
    if not k.startswith("_"):
        print(f"  {k:16s}: {v}")
'''))

    c.extend(celda_wandb("pipeline-qwen-bert",
                         '["pipeline", "qwen", "bert", "modular", "cascada"]'))

    # ------------------------------------------------------------------ 10
    c.append(md("""
## 10 · Integración con Weights & Biases

Este notebook no lanza un `Trainer`, así que no hay curvas de pérdida que
registrar: el registro es puramente de evaluación. Aun así usa el mismo
proyecto y la misma convención de nombres, para que en el panel de W&B las
cuatro arquitecturas aparezcan una al lado de la otra.

Se registran las tres configuraciones como métricas resumen, la tabla completa
de generaciones con la categoría predicha y la real, y la latencia por etapa.
"""))

    # ------------------------------------------------------------------ 11
    c.append(md("""
## 11 · Evaluación

### Propagación de error

La medición clave del notebook. Separamos los ejemplos según si el clasificador
acertó, y comparamos la exactitud del generador en cada grupo.

Interpretación de lo que puede salir:

- **Exactitud mucho menor cuando el clasificador falla** → el enrutamiento
  tiene efecto real (para bien y para mal). Mejorar el clasificador mejoraría
  el sistema completo.
- **Exactitud parecida en ambos grupos** → la estrategia de categoría apenas
  influye en la generación. El clasificador es decorativo y el pipeline no se
  justifica.

Cuidado con el tamaño de muestra: si el clasificador acierta 29 de 33, el grupo
de errores tiene 4 ejemplos. Cualquier porcentaje sobre 4 ejemplos es
anecdótico. Repórtenlo con el `n` al lado, siempre.
"""))

    c.append(code('''
correctas_B = metricas_B["_correctas"]

grupo_ok  = [c for c, a in zip(correctas_B, aciertos_clf) if a]
grupo_mal = [c for c, a in zip(correctas_B, aciertos_clf) if not a]

print("Propagación de error en la cascada")
print("=" * 66)
print(f"Clasificación CORRECTA   n={len(grupo_ok):2d}  "
      f"exactitud del generador = {sum(grupo_ok)/max(len(grupo_ok),1):.1%}")
print(f"Clasificación INCORRECTA n={len(grupo_mal):2d}  "
      f"exactitud del generador = {sum(grupo_mal)/max(len(grupo_mal),1):.1%}")
print("=" * 66)
if len(grupo_mal) < 5:
    print(f"AVISO: solo {len(grupo_mal)} ejemplos mal clasificados. "
          "La cifra de ese grupo es anecdótica, no una estimación.")
'''))

    c.append(md("""
### Latencia por etapa

El argumento práctico. Si la clasificación cuesta el 2% del tiempo total, el
pipeline es esencialmente gratis frente al modelo único y solo hay que
justificarlo por calidad. Si costara el 40%, habría que discutirlo.
"""))

    c.append(code('''
import time

muestra = [r["entrada"] for r in val[:8]]

t0 = time.perf_counter()
_ = clasificar(muestra)
t_clf = (time.perf_counter() - t0) / len(muestra)

t0 = time.perf_counter()
_ = generar(muestra, categorias=[c for c, _ in clasificar(muestra)])
t_gen = (time.perf_counter() - t0) / len(muestra)

print(f"Etapa 1 · clasificación : {t_clf*1000:7.1f} ms/pregunta")
print(f"Etapa 2 · generación    : {t_gen*1000:7.1f} ms/pregunta")
print(f"Total                   : {(t_clf+t_gen)*1000:7.1f} ms/pregunta")
print(f"\\nLa clasificación representa el {100*t_clf/(t_clf+t_gen):.1f}% del tiempo total.")
'''))

    # ------------------------------------------------------------------ 12
    c.append(md("""
## 12 · Comparación Baseline vs Pipeline
"""))

    c.append(code('''
configs = [
    ("A · Qwen solo (baseline)",      metricas_A),
    ("B · Pipeline BERT -> Qwen",     metricas_B),
    ("C · Oráculo (categoría real)",  metricas_C),
]

print(f"{'Configuración':32s} {'Exactitud':>10s} {'Formato':>9s} {'ROUGE-L':>9s} {'Palabras':>9s}")
print("=" * 74)
for nombre, m in configs:
    print(f"{nombre:32s} {m['exactitud']:>10.1%} {m['formato_valido']:>9.1%} "
          f"{m['rouge_l']:>9.3f} {m['long_media']:>9.1f}")
print("=" * 74)
print()
print(f"Beneficio del enrutamiento perfecto  (C - A): "
      f"{metricas_C['exactitud'] - metricas_A['exactitud']:+.1%}")
print(f"Costo de los errores del clasificador (C - B): "
      f"{metricas_C['exactitud'] - metricas_B['exactitud']:+.1%}")
print(f"Aporte neto del pipeline real         (B - A): "
      f"{metricas_B['exactitud'] - metricas_A['exactitud']:+.1%}")
'''))

    c.append(code('''
import matplotlib.pyplot as plt
import numpy as np

nombres = [c[0] for c in configs]
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(configs)); ancho = 0.38
ax.bar(x - ancho/2, [c[1]["exactitud"] for c in configs], ancho, label="Exactitud")
ax.bar(x + ancho/2, [c[1]["formato_valido"] for c in configs], ancho, label="Formato válido")
ax.set_xticks(x); ax.set_xticklabels(nombres, rotation=12, ha="right", fontsize=9)
ax.set_ylim(0, 1.05); ax.set_ylabel("Proporción")
ax.set_title(f"Pipeline modular · {len(val)} ejemplos de validación")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
'''))

    c.append(code('''
print("Comparación cualitativa sobre los ejemplos de demostración")
gen_demo_A = generar([d["entrada"] for d in demo], categorias=None)
cats_demo = [c for c, _ in clasificar([d["entrada"] for d in demo])]
gen_demo_B = generar([d["entrada"] for d in demo], categorias=cats_demo)

for d, a, b, cat in zip(demo, gen_demo_A, gen_demo_B, cats_demo):
    print("=" * 78)
    print(f"[{d['id']}] {d['entrada']}")
    print(f"    categoría real: {d['categoria']}  |  predicha: {cat}")
    print("-" * 78)
    print("A · sin enrutamiento:")
    print(a[:450])
    print("-" * 78)
    print("B · con estrategia de categoría:")
    print(b[:450])
    print("-" * 78)
    print(f"Correcta -> A: {respuesta_correcta(a, d['valor'])} | "
          f"B: {respuesta_correcta(b, d['valor'])}")
print("=" * 78)
'''))

    c.append(code('''
RESULTADOS_PIPELINE = {
    "notebook": "S04_Lab_Fine_tuning_Qwen_BERT",
    "arquitectura": "pipeline modular (encoder + decoder)",
    "modelo": f"{DIR_BERT} -> {MODELO_QWEN}+LoRA",
    "metodo": "composición de modelos ya entrenados (sin entrenamiento nuevo)",
    "n_val": len(val),
    "umbral_confianza": UMBRAL_CONFIANZA,
    "accuracy_clasificador": accuracy_clf,
    "config_A_solo_qwen": metricas_A,
    "config_B_pipeline": metricas_B,
    "config_C_oraculo": metricas_C,
    "propagacion_error": {
        "n_clf_correcta": len(grupo_ok),
        "exactitud_clf_correcta": sum(grupo_ok) / max(len(grupo_ok), 1),
        "n_clf_incorrecta": len(grupo_mal),
        "exactitud_clf_incorrecta": sum(grupo_mal) / max(len(grupo_mal), 1),
    },
    "latencia_ms": {
        "clasificacion": t_clf * 1000,
        "generacion": t_gen * 1000,
        "total": (t_clf + t_gen) * 1000,
    },
    "beneficio_enrutamiento_perfecto": metricas_C["exactitud"] - metricas_A["exactitud"],
    "costo_errores_clasificador": metricas_C["exactitud"] - metricas_B["exactitud"],
    "aporte_neto": metricas_B["exactitud"] - metricas_A["exactitud"],
}

guardar_resultados("pipeline_qwen_bert", RESULTADOS_PIPELINE)
'''))

    c.append(code('''
if USAR_WANDB:
    import wandb

    if wandb.run is None:
        wandb.init(project=PROYECTO, name=NOMBRE_RUN, tags=TAGS, reinit=True)

    tabla = wandb.Table(columns=["id", "cat_real", "cat_predicha", "confianza",
                                 "clf_ok", "entrada", "gen_A", "gen_B",
                                 "ok_A", "ok_B"])
    for r, (cat, conf), a, b, oka, okb in zip(
            val, clasificaciones, gen_A, gen_B,
            metricas_A["_correctas"], metricas_B["_correctas"]):
        tabla.add_data(r["id"], r["categoria"], cat, conf,
                       cat == r["categoria"], r["entrada"], a, b, oka, okb)
    wandb.log({"evaluacion/pipeline": tabla})

    wandb.summary.update({
        "clasificador/accuracy": accuracy_clf,
        "A_solo_qwen/exactitud": metricas_A["exactitud"],
        "B_pipeline/exactitud": metricas_B["exactitud"],
        "C_oraculo/exactitud": metricas_C["exactitud"],
        "delta/aporte_neto": metricas_B["exactitud"] - metricas_A["exactitud"],
        "delta/costo_errores_clf": metricas_C["exactitud"] - metricas_B["exactitud"],
        "latencia/clasificacion_ms": t_clf * 1000,
        "latencia/generacion_ms": t_gen * 1000,
    })
    wandb.finish()
    print("Registro en W&B completado.")
else:
    print("W&B desactivado; las métricas quedaron en resultados/pipeline_qwen_bert.json")
'''))

    # ------------------------------------------------------------------ 13
    c.append(md("""
## 13 · Discusión

**1. ¿El pipeline aporta algo? (B − A)**
Si la diferencia es positiva y apreciable, el enrutamiento funciona. Si es
cercana a cero o negativa, hay que decirlo sin adornos: el sistema modular
añadió complejidad sin beneficio medible en este corpus. Es un resultado
legítimo, y probablemente el más común con 33 ejemplos de evaluación.

**2. ¿El límite es el clasificador o la idea? (C vs A)**
- Si **C > A** de forma clara pero **B ≈ A**: la idea es buena y el
  clasificador es el cuello de botella. Invertir en más datos de clasificación
  tiene retorno.
- Si **C ≈ A**: condicionar por categoría no cambia lo que Qwen genera. Ningún
  clasificador, por perfecto que fuera, mejoraría el sistema. La conclusión es
  que el modelo fine-tuned ya infiere la estrategia del enunciado, y explicitarla
  es redundante.

**3. ¿Por qué el efecto podría ser pequeño?**
Una hipótesis razonable: Qwen ya vio 132 ejemplos con este formato y ya aprendió
implícitamente qué hacer con cada tipo de problema. Las estrategias explícitas
le dicen algo que ya sabía. El enrutamiento tendría más valor si las estrategias
aportaran conocimiento realmente ausente del entrenamiento —fórmulas poco
comunes, convenciones de notación, casos límite—.

**4. ¿Y la complejidad operativa?**
Dos modelos son dos artefactos que versionar, desplegar y monitorizar. La
alineación de etiquetas entre ambos (el `assert` de la sección 6) es un fallo
silencioso esperando ocurrir. Al comparar con FLAN-T5 en el notebook 4, este
costo cuenta tanto como las métricas.
"""))

    # ------------------------------------------------------------------ 14
    c.append(md("""
## 14 · Conclusiones

1. **Modularidad tiene un valor real que las métricas no capturan:**
   diagnosticabilidad. Poder decir "el 12% de los fallos vienen del enrutador y
   el 88% del generador" es imposible en un modelo único, y es exactamente lo
   que se necesita para decidir dónde invertir el siguiente esfuerzo.
2. **La cascada propaga errores.** Está medido en este notebook, no supuesto.
   El umbral de confianza es la mitigación estándar y está implementado.
3. **El costo computacional del enrutamiento es despreciable.** El encoder es
   ~100 veces más barato que la generación.
4. **Cuál gana depende de sus números.** Comparen B contra el modelo único de
   FLAN-T5 (notebook 4), que resuelve clasificación y generación en una sola
   pasada, y contra A. La comparación completa está en el notebook 5.
"""))

    return c
