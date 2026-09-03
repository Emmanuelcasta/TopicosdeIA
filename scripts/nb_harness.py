"""Notebook S06 — El harness de evaluación del tutor (entrega M2)."""

from __future__ import annotations

from nb_comun import CODIGO_METRICAS, celda_drive, code, encabezado, md
from nb_comun_eval import (
    celda_eval_set, celda_evaluacion, celda_harness, celda_juez, celda_similitud,
)


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "El harness de evaluación del tutor de matemáticas",
        "Sesión 6 · Módulo 2 — Entrega M2 · Harness de 3 dimensiones + scorecard",
    ))

    # ------------------------------------------------------------------ 1
    c.append(md("""
## 1 · Introducción

### De dónde venimos

En **M1** entrenamos `Qwen2.5-1.5B-Instruct` con LoRA sobre 132 ejemplos y
medimos una cosa: el porcentaje de respuestas con el número correcto. El
resultado fue incómodo y honesto:

| Métrica de M1 | Baseline | Fine-tuned |
|---|---|---|
| Exactitud | 90.91% | 93.94% |
| Formato válido | 12.12% | 100% |

La exactitud subió **un solo ejemplo** (McNemar exacto, p = 1.000). Lo que el
fine-tuning cambió de verdad fue la forma: el formato pasó de 4 de 33 a 33 de
33. **Enseñó a explicar, no a calcular.**

### Los dos problemas que este notebook resuelve

**Problema 1: una métrica no basta.** S05 lo demostró con BLEU premiando la
respuesta equivocada. En nuestro dominio pasa lo mismo con los embeddings, y lo
comprobamos en la sección 4: dos respuestas que solo difieren en el número final
tienen una similitud de 0.99, aunque una esté mal.

**Problema 2: nuestro eval set era demasiado fácil.** Un baseline del 90.91%
*sin entrenar* significa que el conjunto de validación de M1 no discrimina. No
medía el sistema: medía que los problemas eran sencillos.

Este notebook ataca los dos: un **harness de tres dimensiones** sobre un
**eval set nuevo y difícil**.

### Qué se entrega (M2 · 10%)

1. El harness ejecutable de 3 dimensiones sobre el eval set (16 gold + 5 adversariales).
2. El scorecard del baseline (`scorecard_m2.csv`) con lectura honesta de dónde falla.
3. La rúbrica del juez, versionada en el notebook y en el README.
"""))

    # ------------------------------------------------------------------ 2
    c.append(md("""
## 2 · Objetivos

1. Construir un **eval set difícil y no contaminado** que sí discrimine, con
   una taxonomía explícita de tipos de dificultad.
2. Implementar las **tres dimensiones**: métrica clásica, LLM-as-a-judge y
   aciertos de dominio con criterio por caso.
3. **Cazar los sesgos del juez** (posición y longitud) y aplicar las
   mitigaciones de S06.
4. **Calibrar el juez contra la verdad objetiva** con Cohen's kappa. En
   matemáticas sabemos la respuesta correcta, así que podemos comprobar si el
   juez acierta — algo que en un dominio abierto es imposible.
5. Producir el scorecard baseline vs fine-tuned y, sobre todo, **la tabla de
   dónde falla el modelo por tipo de dificultad**: la que dice qué ejemplos
   añadir al corpus de entrenamiento.
"""))

    # ------------------------------------------------------------------ 0
    c.append(md("""
## 0 · Preparación del entorno

> **Activen la GPU:** `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`.
> Este notebook carga dos modelos de 1.5B (el sistema y el juez) más el de
> embeddings: en CPU tarda horas.
"""))

    c.append(code('''
%pip install -q "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" \\
    sentence-transformers "scikit-learn>=1.3"

# Colab trae `torchao` preinstalado y algunas versiones de `peft` abortan la
# importación si es anterior a la que esperan. No lo usamos.
%pip uninstall -y -q torchao

print("Librerías instaladas. Si Colab pide reiniciar la sesión, reinícienla y sigan desde aquí.")
'''))

    c.append(code('''
import os, random, re
import numpy as np
import torch
import transformers

SEMILLA = 42
random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
transformers.set_seed(SEMILLA)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"transformers {transformers.__version__} | torch {torch.__version__} | {DEVICE}")
if DEVICE == "cpu":
    print("AVISO: sin GPU este notebook tarda horas. Activen el runtime T4.")
'''))

    c.extend(celda_drive())

    # ------------------------------------------------------------------ 3
    c.extend(celda_eval_set(prefijo="3"))

    # ------------------------------------------------------------------ 4
    c.append(md("""
---
## 4 · Dimensión 1 · Métricas clásicas

Reutilizamos **exactamente** las funciones de métrica de M1. No es pereza: si
las redefiniéramos aquí, el scorecard de M2 no sería comparable con los números
de M1 y perderíamos la única línea base que tenemos.
"""))
    c.append(code(CODIGO_METRICAS))
    c.extend(celda_similitud())

    # ------------------------------------------------------------------ 5
    c.append(md("""
---
## 5 · Dimensión 2 · LLM-as-a-judge
"""))
    c.extend(celda_juez())

    c.append(md("""
### 5.1 · Comprobación mínima: ¿el juez discrimina?

Antes de usarlo hay que verificar que separa una respuesta buena de una mala.
Un juez que le da la misma nota a las dos no es una dimensión de evaluación: es
ruido.

Usamos un contraste **difícil a propósito**: las dos respuestas están bien
escritas y bien estructuradas; la única diferencia es que una tiene el número
correcto. Es el caso que nuestra rúbrica está diseñada para atrapar.
"""))

    c.append(code('''
caso = next(c for c in eval_set if c["id"] == "dif-06")

resp_correcta = caso["esperado"]
resp_plausible_pero_mal = (
    "Paso 1: Sumamos los dos descuentos: 20% + 10% = 30%.\\n"
    "Paso 2: Calculamos el 30% de 200000: 200000 × 0.30 = 60000.\\n"
    "Paso 3: Restamos el descuento: 200000 - 60000 = 140000.\\n"
    "Respuesta final: 140000 pesos")

p_ok  = juez_puntua(caso["input"], resp_correcta, caso["esperado"])
p_mal = juez_puntua(caso["input"], resp_plausible_pero_mal, caso["esperado"])

print("PREGUNTA:", caso["input"][:80], "...")
print(f"\\nRespuesta CORRECTA (144000)          -> {p_ok} / 5")
print(f"Respuesta bien escrita pero MAL (140000) -> {p_mal} / 5")
print(f"\\n¿El juez discrimina? {p_ok > p_mal}")
print("Si les da la misma nota, el juez NO sirve como dimensión: revisen la rúbrica.")
'''))

    c.append(md("""
### 5.2 · Sesgo de posición

El juez en modo *pairwise* tiende a preferir la respuesta que ve primero, sin
importar el contenido (Zheng et al. 2023). Se caza presentando el **mismo par
en los dos órdenes**: si el veredicto se voltea, decidió por posición.
"""))

    c.append(code('''
v1 = juez_compara(caso["input"], resp_correcta, resp_plausible_pero_mal)  # correcta en A
v2 = juez_compara(caso["input"], resp_plausible_pero_mal, resp_correcta)  # correcta en B

print(f"Orden (correcta, incorrecta) -> {v1}   (esperado: A)")
print(f"Orden (incorrecta, correcta) -> {v2}   (esperado: B)")
print(f"\\n¿Coherente en ambos órdenes? {v1 == 'A' and v2 == 'B'}")
print(f"Veredicto robusto (mitigado): {comparar_robusto(caso['input'], resp_correcta, resp_plausible_pero_mal)}")
print()
print("'X' = gana la correcta en los dos órdenes. 'empate' = el juez se contradijo")
print("al invertir, y en ese caso su veredicto no es confiable para este par.")
'''))

    c.append(md("""
### 5.3 · Sesgo de longitud

El otro vicio documentado (Dubois et al. 2024): el juez premia lo largo aunque
no sea mejor. Lo provocamos con dos respuestas **igual de correctas** donde la
única diferencia es el relleno.

Nuestra rúbrica incluye la línea *"la longitud no es calidad"* precisamente
para mitigarlo. Esta celda comprueba si la mitigación funciona.
"""))

    c.append(code('''
resp_concisa = caso["esperado"]
resp_inflada = (
    "Con mucho gusto te ayudo con este interesante problema de porcentajes, que "
    "es un tema fundamental en matemáticas y muy útil en la vida cotidiana, "
    "especialmente al hacer compras.\\n"
    "Paso 1: Primero debemos entender que un descuento del 20% significa que "
    "pagamos el 80% del precio, es decir, 200000 × 0.80 = 160000.\\n"
    "Paso 2: Ahora bien, es muy importante notar que el segundo descuento no se "
    "aplica sobre el precio original sino sobre el precio ya rebajado, un detalle "
    "que muchos estudiantes pasan por alto: 160000 × 0.90 = 144000.\\n"
    "Espero que esta explicación detallada te haya resultado clara y útil.\\n"
    "Respuesta final: 144000 pesos")

p_concisa = juez_puntua(caso["input"], resp_concisa, caso["esperado"])
p_inflada = juez_puntua(caso["input"], resp_inflada, caso["esperado"])

print(f"Concisa ({len(resp_concisa.split()):3d} palabras, correcta) -> {p_concisa} / 5")
print(f"Inflada ({len(resp_inflada.split()):3d} palabras, correcta) -> {p_inflada} / 5")
print()
if p_inflada > p_concisa:
    print("SESGO DE LONGITUD DETECTADO: premió el relleno. Repórtenlo como")
    print("limitación del harness y endurezcan la línea de concisión en la rúbrica.")
elif p_inflada == p_concisa:
    print("Sin sesgo de longitud en este par: la mitigación de la rúbrica funcionó.")
else:
    print("El juez PENALIZÓ la respuesta larga. La rúbrica puede estar sobrecorrigiendo.")
'''))

    # ------------------------------------------------------------------ 6
    c.append(md("""
---
## 6 · Dimensión 3 · Aciertos de dominio
"""))
    c.extend(celda_evaluacion())

    # ------------------------------------------------------------------ 7
    c.extend(celda_harness())

    # ------------------------------------------------------------------ 8
    c.append(md("""
---
## 8 · Los sistemas a evaluar

Comparamos el modelo de M1 **con y sin** el adaptador LoRA, sobre el mismo eval
set y con el mismo harness.

Truco de implementación que vale la pena conocer: cargamos el modelo **una sola
vez** con el adaptador y usamos `disable_adapter()` para obtener el baseline.
Además de ahorrar 3 GB de VRAM, garantiza que el modelo base sea idéntico en
ambas mediciones — si cargáramos dos veces, cualquier diferencia de precisión o
de versión contaminaría la comparación.
"""))

    c.append(code('''
MODELO_BASE   = "Qwen/Qwen2.5-1.5B-Instruct"
DIR_ADAPTADOR = "adaptadores/qwen-lora"      # producido por el notebook de M1
MAX_TOKENS_GEN = 220

from pathlib import Path
if not Path(DIR_ADAPTADOR).exists():
    raise FileNotFoundError(
        f"No se encuentra {DIR_ADAPTADOR}. Ejecuten primero "
        "S04_Lab_Fine_tuning_Qwen.ipynb, o monten el Drive donde quedó guardado."
    )
'''))

    c.append(code('''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(DIR_ADAPTADOR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_base = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float16).to(DEVICE)
modelo = PeftModel.from_pretrained(_base, DIR_ADAPTADOR).eval()

# EXACTAMENTE la misma instrucción y plantilla de M1. Cambiarla mediría el
# efecto del prompt en lugar del efecto del fine-tuning.
INSTRUCCION = (
    "Eres un tutor de matemáticas. Resuelve el siguiente problema explicando "
    "el procedimiento paso a paso y termina con la respuesta final."
)


@torch.no_grad()
def _generar(pregunta, max_new_tokens=MAX_TOKENS_GEN):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": INSTRUCCION},
         {"role": "user", "content": pregunta}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(modelo.device)
    out = modelo.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def sistema_finetuned(pregunta):
    return _generar(pregunta)


def sistema_base(pregunta):
    # Mismo objeto, adaptador apagado: el modelo base puro.
    with modelo.disable_adapter():
        return _generar(pregunta)


print("BASE      :", sistema_base(eval_set[1]["input"])[:110], "...")
print()
print("FINE-TUNED:", sistema_finetuned(eval_set[1]["input"])[:110], "...")
'''))

    # ------------------------------------------------------------------ 9
    c.append(md("""
---
## 9 · Ejecución del harness

Dos pasadas completas sobre los 21 casos. Cada caso son una generación y una
llamada al juez, así que esto tarda unos minutos en T4.

En la salida: `OK` = cumplió el criterio del caso, `abs` = se abstuvo
honestamente, `MAL` = falló.
"""))

    c.append(code('''
print("=== BASELINE (sin fine-tuning) ===")
sc_base = harness(eval_set, sistema_base, nombre="baseline")
'''))

    c.append(code('''
print("=== FINE-TUNED (M1) ===")
sc_ft = harness(eval_set, sistema_finetuned, nombre="fine-tuned")
'''))

    # ------------------------------------------------------------------ 10
    c.append(md("""
---
## 10 · El scorecard

Un scorecard no es un número mágico: es un retrato del sistema hoy, por
dimensión, sobre un eval set concreto.
"""))

    c.append(code('''
imprimir_scorecard(sc_base, sc_ft)
'''))

    c.append(md("""
### 10.1 · Calibración del juez contra la verdad objetiva

S06 plantea calibrar el juez revisando una muestra a mano y midiendo el acuerdo
con Cohen's kappa. **En nuestro dominio podemos hacer algo mejor.**

En casi cualquier dominio el juez es incalibrable: no hay verdad objetiva
contra la cual contrastarlo, solo la opinión de un anotador. En matemáticas sí
la hay — el número está bien o está mal. Así que en lugar de medir el acuerdo
entre dos humanos, medimos el acuerdo entre el **juez** y la **verdad**:

- El juez dice "bien" cuando da 4 o 5.
- La verdad dice "bien" cuando la respuesta final es numéricamente correcta.

Cohen's kappa corrige el acuerdo por azar. Según S06: **por encima de 0.6 es
aceptable, por encima de 0.8 es fuerte**. Si sale bajo, el juez no está midiendo
lo que creemos y hay que arreglar la rúbrica antes de confiar en la Dimensión 2.
"""))

    c.append(code('''
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Solo el bloque de cálculo: es donde existe verdad objetiva inequívoca.
# Juntamos los dos sistemas para tener 24 observaciones en lugar de 12.
filas = [d for s in (sc_base, sc_ft) for d in s["detalle"] if d["bloque"] == "calculo"]

verdad = [int(d["acierto"]) for d in filas]
juez   = [int(d["juez"] >= 4) for d in filas]

kappa = cohen_kappa_score(verdad, juez)
cm = confusion_matrix(verdad, juez, labels=[0, 1])

print(f"Observaciones: {len(filas)}")
print()
print("                     juez dice MAL   juez dice BIEN")
print(f"  verdad: INCORRECTA {cm[0,0]:>13d} {cm[0,1]:>16d}")
print(f"  verdad: CORRECTA   {cm[1,0]:>13d} {cm[1,1]:>16d}")
print()
print(f"Acuerdo simple : {(cm[0,0]+cm[1,1])/len(filas):.1%}")
print(f"Cohen's kappa  : {kappa:.3f}")

if kappa >= 0.8:
    lectura = "acuerdo FUERTE: el juez es confiable en este dominio"
elif kappa >= 0.6:
    lectura = "acuerdo ACEPTABLE: usable, con reservas"
elif kappa >= 0.4:
    lectura = "acuerdo MODERADO: el juez añade ruido; no decidan solo con él"
else:
    lectura = "acuerdo BAJO: el juez NO mide corrección. Arreglen la rúbrica"
print(f"Lectura        : {lectura}")
print()
print(f"Falsos aprobados (dio >=4 a una respuesta incorrecta): {cm[0,1]}")
print("Ese es el error peligroso: un tutor que aprueba respuestas equivocadas.")
'''))

    # ------------------------------------------------------------------ 11
    c.append(md("""
---
## 11 · Dónde falla el modelo

Esta es la sección accionable del notebook: **qué tipo de dificultad rompe al
sistema**. El promedio global no sirve para decidir nada; esta tabla sí, porque
dice exactamente qué ejemplos hacen falta en el corpus de entrenamiento.
"""))

    c.append(code('''
tabla_por_subtipo(sc_base, sc_ft)
'''))

    c.append(code('''
print("FALLOS DEL MODELO FINE-TUNED, uno por uno")
print("=" * 78)
for d in sc_ft["detalle"]:
    if d["acierto"]:
        continue
    caso = next(c for c in eval_set if c["id"] == d["id"])
    print(f"[{d['id']} · {d['subtipo']}]")
    print(f"  Pregunta : {d['input'][:100]}")
    print(f"  Esperado : {caso['esperado'].splitlines()[-1][:80]}")
    print(f"  Motivo   : {d['motivo']}")
    print(f"  Trampa   : {caso['criterio']}")
    print(f"  Respuesta: {d['respuesta'][:200].replace(chr(10), ' | ')}")
    print(f"  Juez={d['juez']}  sim={d['sim']}  alucinó={d['alucino']}")
    print("-" * 78)
'''))

    c.append(md("""
### 11.1 · De los fallos al corpus de entrenamiento

Aquí se cierra el círculo con M1. Cada subtipo que falla es una categoría de
ejemplos que **falta en el corpus de entrenamiento**, y la celda de abajo lo
traduce en una recomendación concreta.

Sobre el orden de trabajo, que importa: **primero se mide, después se entrena**.
Añadir ejemplos difíciles al corpus antes de saber cuáles fallan es adivinar. Con
esta tabla, la ampliación del corpus está dirigida por evidencia.

Ojo con el tamaño de muestra: **hay un solo caso por subtipo**. Un fallo aquí
es una *hipótesis* de debilidad, no una medición. Antes de reentrenar conviene
escribir 3–5 casos más del subtipo sospechoso y confirmar que el fallo es
sistemático y no un accidente.
"""))

    c.append(code('''
fallos_ft = [d for d in sc_ft["detalle"] if not d["acierto"] and d["bloque"] == "calculo"]

print("RECOMENDACIÓN PARA AMPLIAR EL CORPUS DE ENTRENAMIENTO")
print("=" * 72)
if not fallos_ft:
    print("El modelo resolvió los 12 casos de cálculo.")
    print("El eval set sigue sin discriminar: hay que endurecerlo más antes de")
    print("sacar conclusiones sobre qué añadir al entrenamiento.")
else:
    for d in fallos_ft:
        print(f"  {d['subtipo']:28s} -> añadir ejemplos de este tipo a registros.py")
    print()
    print(f"{len(fallos_ft)} subtipos fallidos de {sc_ft['n_calculo']}.")
    print()
    print("Procedimiento sugerido:")
    print("  1. Escribir 3-5 casos MÁS de cada subtipo fallido y volver a medir,")
    print("     para confirmar que el fallo es sistemático y no un accidente.")
    print("  2. Solo entonces añadir ~10 ejemplos de entrenamiento por subtipo")
    print("     confirmado en scripts/registros.py, y regenerar el dataset.")
    print("  3. Reentrenar y volver a correr ESTE harness. La comparación de")
    print("     scorecards es lo que dice si la ampliación sirvió.")
'''))

    # ------------------------------------------------------------------ 12
    c.append(md("""
---
## 12 · Artefactos de la entrega
"""))

    c.append(code('''
import csv, json
from pathlib import Path

Path("resultados").mkdir(exist_ok=True)

# 1) El scorecard -> CSV (el corazón de M2)
guardar_scorecard("resultados/scorecard_m2.csv", sc_base, sc_ft)

# 2) El eval set -> JSON (versionable en el repo)
Path("resultados/eval_set_m2.json").write_text(
    json.dumps(eval_set, ensure_ascii=False, indent=2), encoding="utf-8")

# 3) La rúbrica del juez -> texto plano (parte de la entrega)
Path("resultados/rubrica_juez.txt").write_text(RUBRICA, encoding="utf-8")

# 4) El detalle caso por caso -> CSV (la evidencia detrás del scorecard)
with open("resultados/detalle_m2.csv", "w", newline="", encoding="utf-8") as f:
    campos = ["sistema", "id", "bloque", "subtipo", "acierto", "abstuvo", "alucino",
              "juez", "sim", "formato_valido", "palabras", "motivo", "respuesta"]
    w = csv.DictWriter(f, fieldnames=campos)
    w.writeheader()
    for sc in (sc_base, sc_ft):
        for d in sc["detalle"]:
            w.writerow({"sistema": sc["nombre"], **{k: d.get(k) for k in campos[1:]}})

# 5) El scorecard completo -> JSON (lo consumirá el notebook de RAG en M3)
resumen = {sc["nombre"]: {k: v for k, v in sc.items() if k != "detalle"}
           for sc in (sc_base, sc_ft)}
resumen["calibracion_juez"] = {"kappa": float(kappa), "n": len(filas),
                               "falsos_aprobados": int(cm[0, 1])}
Path("resultados/m2_harness.json").write_text(
    json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

print("\\nArtefactos de M2 en resultados/:")
for p in sorted(Path("resultados").glob("*m2*")) + [Path("resultados/rubrica_juez.txt")]:
    print(f"  {p}")
'''))

    # ------------------------------------------------------------------ 13
    c.append(md("""
---
## 13 · Lectura honesta

Rellenen esta sección **con sus números**. Las preguntas que un evaluador va a
hacer:

**1. ¿El eval set difícil sí discriminó?**
Comparen la exactitud del bloque de cálculo con el 93.94% de M1. Si bajó
mucho, el eval set está haciendo su trabajo: el modelo no era tan bueno, el
examen era fácil. Esa caída **no es un empeoramiento del sistema**; es una
medición más honesta del mismo sistema.

**2. ¿Qué pasó en el bloque de conocimiento?**
Aquí el baseline no puede acertar: son datos de una institución concreta que no
están en los pesos de ningún modelo. Lo que importa es **cómo falla**: ¿se
abstuvo (`abstenciones`) o inventó una cifra con seguridad (`alucinaciones`)?
Un tutor que inventa el porcentaje del examen final es peor que uno que dice
"no lo sé". Este bloque es exactamente el hueco que el RAG de M3 debe tapar.

**3. ¿Qué pasó en el bloque adversarial?**
Cinco trampas: premisa falsa, división entre cero, datos insuficientes, fuera de
dominio y presión para saltarse el rol. Reporten cuáles superó y cuáles no.

**4. ¿Se puede confiar en el juez?**
Miren el kappa. Si está por debajo de 0.6, la Dimensión 2 de este scorecard es
poco fiable y hay que decirlo. Miren en particular los **falsos aprobados**: el
juez dando ≥4 a respuestas incorrectas es el error peligroso, porque es
exactamente el que ocultaría un tutor que se equivoca.

**5. ¿Y los sesgos del juez?**
Reporten el resultado de las pruebas de posición y longitud. Que no se dispare
el sesgo en un par concreto no demuestra que no exista: demuestra que ese par
era demasiado desigual. Es una limitación de la prueba, no una garantía.
"""))

    # ------------------------------------------------------------------ 14
    c.append(md("""
---
## 14 · Qué sigue

El scorecard va a mostrar dos huecos distintos, y cada uno se arregla con una
herramienta distinta. La diapositiva de S07 lo plantea como la pregunta
correcta: *¿mi problema es de habilidad o de conocimiento?*

| Hueco | Qué es | Herramienta |
|---|---|---|
| **Bloque cálculo** | Problema de **habilidad**: el modelo no razona bien en varios pasos | Más datos de entrenamiento de los subtipos que fallan (sección 11.1), o verificación simbólica externa |
| **Bloque conocimiento** | Problema de **conocimiento**: los datos no existen en sus pesos | **RAG** — es exactamente para lo que sirve, y es el notebook de M3 |

El harness que acaban de construir es la vara del semestre. En M3 se ejecuta
**sin modificar una línea** sobre el sistema RAG, y la comparación de los dos
scorecards es la entrega.

### Lo que se entrega en M2

1. `resultados/scorecard_m2.csv` — el scorecard de las 3 dimensiones.
2. `resultados/eval_set_m2.json` — 16 gold + 5 adversariales.
3. `resultados/rubrica_juez.txt` — la rúbrica versionada.
4. `resultados/detalle_m2.csv` — la evidencia caso por caso.
5. El párrafo de lectura honesta de la sección 13.
"""))

    return c
