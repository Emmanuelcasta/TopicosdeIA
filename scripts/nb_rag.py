"""Notebook S07 — RAG ingenuo sobre el tutor de matemáticas (M3)."""

from __future__ import annotations

from nb_comun import CODIGO_METRICAS, celda_drive, code, encabezado, md
from nb_comun_eval import (
    celda_eval_set, celda_evaluacion, celda_harness, celda_juez, celda_similitud,
)


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "RAG para el tutor de matemáticas",
        "Sesión 7 · Módulo 3 — Buscar primero, responder después",
    ))

    # ------------------------------------------------------------------ 1
    c.append(md("""
## 1 · Introducción

### El hueco que dejó el scorecard de M2

M2 midió el tutor con un eval set de tres bloques y encontró dos fallos de
naturaleza **distinta**:

| Bloque | Qué falla | Naturaleza del problema |
|---|---|---|
| **Cálculo** | El modelo se equivoca en problemas de varios pasos | **Habilidad** |
| **Conocimiento** | No sabe qué porcentaje vale el examen final del colegio | **Conocimiento** |

El segundo no se arregla entrenando. Ese dato **no existe en los pesos de
ningún modelo del mundo**: es de una institución concreta, y además puede
cambiar el año que viene. Reentrenar por cada cambio de reglamento es
absurdo; indexar el documento nuevo toma minutos.

Esa es la pregunta correcta de S07, y conviene tenerla explícita:

> **¿Mi problema es de habilidad o de conocimiento?**
> Si la respuesta cambia con el tiempo o vive en SUS documentos → RAG.
> Si es un patrón estable de comportamiento → fine-tuning.

M1 (fine-tuning) enseñó **forma**: el formato pasó del 12% al 100%.
M3 (RAG) debe aportar **conocimiento consultable**.

### La adaptación que este dominio exige

El RAG de clase responde preguntas sobre documentos. Nuestro tutor tiene que
hacer **dos cosas distintas** según lo que le pregunten:

- *"¿Cuánto vale el examen final?"* → responder **solo** desde el contexto, y
  citar la fuente. Si no está, admitirlo.
- *"¿Cuánto se paga con 20% y luego 10% de descuento sobre 200000?"* →
  **resolverlo**, usando el contexto solo si contiene una fórmula o estrategia
  útil.

Un prompt que diga "responde SOLO con base en el contexto" rompería el segundo
caso: el modelo se negaría a calcular porque el resultado no está escrito en
ningún documento. La sección 8 construye un prompt que distingue ambos modos, y
es la pieza de diseño propia de este notebook.

### La hipótesis que vamos a poner a prueba

RAG debería **ganar en el bloque de conocimiento** y **no aportar (o incluso
estorbar) en el de cálculo**, porque meter párrafos irrelevantes en el prompt
añade ruido a un problema que solo requiere razonar. Lo mediremos: si el bloque
de cálculo baja, hay que reportarlo, no esconderlo.
"""))

    # ------------------------------------------------------------------ 2
    c.append(md("""
## 2 · Objetivos

1. Construir el pipeline RAG completo **sin frameworks**: ingest → chunk →
   embed → store → retrieve → augment → generate.
2. Comparar **búsqueda por palabras clave contra búsqueda semántica** sobre
   consultas reales del dominio.
3. Diseñar un prompt aumentado con **válvula de escape** que funcione tanto
   para datos institucionales como para problemas de cálculo.
4. Correr **el harness de M2 sin modificar una línea** sobre el sistema RAG.
5. Comparar `scorecard_rag` contra `scorecard_m2` **por bloque**, y diagnosticar
   los fallos que queden: ¿fue la búsqueda o fue la generación?
"""))

    # ------------------------------------------------------------------ 0
    c.append(md("""
## 0 · Preparación del entorno

> **Activen la GPU (T4).** Se cargan el generador, el juez y el modelo de
> embeddings.
"""))

    c.append(code('''
%pip install -q "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" \\
    sentence-transformers chromadb "scikit-learn>=1.3"
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
'''))

    c.extend(celda_drive())

    # ------------------------------------------------------------------ 3
    c.extend(celda_eval_set(prefijo="3"))

    # ------------------------------------------------------------------ 4
    c.append(md("""
---
## 4 · Métricas y embeddings

Las mismas funciones de M1 y M2, sin tocar. El modelo de embeddings cumple aquí
**doble función**: métrica de similitud del harness y motor de búsqueda del
RAG. Es el mismo `paraphrase-multilingual-MiniLM-L12-v2` de S05, S06 y S07.
"""))
    c.append(code(CODIGO_METRICAS))
    c.extend(celda_similitud())

    # ------------------------------------------------------------------ 5
    c.append(md("""
---
## 5 · El corpus documental

Seis documentos que cubren los dos tipos de conocimiento que al tutor le
faltan:

| Documento | Tipo | Por qué está |
|---|---|---|
| SIEE (sistema de evaluación) | **Institucional** | Porcentajes de nota, nota mínima, recuperaciones. Datos que ningún modelo puede saber. |
| Plan de área de matemáticas | **Institucional** | Qué se enseña en cada grado. |
| Protocolo de errores frecuentes | **Pedagógico** | Estrategias por tipo de error, incluidas las trampas del eval set. |
| Formulario de geometría | **De referencia** | Fórmulas de áreas, perímetros y volúmenes. |
| Jerarquía de operaciones | **De referencia** | Orden de operaciones y reglas de signos. |
| Guía de fracciones y porcentajes | **De referencia** | Procedimientos y errores típicos. |

Los tres **institucionales/pedagógicos** son los que hacen visible el valor del
RAG: contienen hechos que el modelo no puede tener. Los tres **de referencia**
contienen cosas que el modelo más o menos sabe — sirven para comprobar si el
contexto ayuda, estorba o da igual cuando la información ya está en los pesos.

> **Reemplácenlos por los documentos reales de su institución.** Los datos de
> abajo son verosímiles pero **ficticios**, construidos para que el experimento
> sea reproducible. Para S08 hacen falta 10–30 documentos reales con fuente y
> fecha.
"""))

    c.append(code('''
corpus = [
    {"id": "siee", "fuente": "Sistema Institucional de Evaluación (SIEE) 2026 — capítulo 3",
     "texto": (
        "La valoración del periodo académico en el área de matemáticas se compone de cuatro "
        "elementos con la siguiente ponderación: el examen final del periodo equivale al 40 por "
        "ciento de la nota, los talleres al 25 por ciento, los quices al 20 por ciento y el "
        "trabajo en clase al 15 por ciento. "
        "La escala institucional va de 1.0 a 5.0 y la nota mínima aprobatoria es 3.0. "
        "Cada estudiante dispone de dos oportunidades de recuperación por periodo; la nota "
        "máxima que puede obtenerse en una recuperación es 3.5. "
        "La inasistencia no justificada a un examen se califica con 1.0, salvo que el acudiente "
        "presente excusa dentro de los tres días hábiles siguientes.")},

    {"id": "plan_area", "fuente": "Plan de área de matemáticas 2026 — secuencia por grados",
     "texto": (
        "La secuencia del área organiza los contenidos así. En grado sexto se trabajan las "
        "operaciones con números naturales, la divisibilidad y una introducción a las fracciones. "
        "En grado séptimo se abordan los números enteros, la proporcionalidad directa e inversa y "
        "los porcentajes. "
        "En grado octavo se introducen las ecuaciones de primer grado con una incógnita, junto con "
        "el lenguaje algebraico y la traducción de enunciados verbales a expresiones matemáticas. "
        "En grado noveno se amplía a sistemas de dos ecuaciones lineales y a la función lineal. "
        "En grado décimo se trabajan la trigonometría y la geometría analítica. "
        "El área recomienda dedicar al menos dos sesiones a la traducción de enunciados verbales "
        "antes de introducir el algoritmo de despeje.")},

    {"id": "protocolo", "fuente": "Protocolo de errores frecuentes en matemáticas — 2026",
     "texto": (
        "Este protocolo recoge los errores más frecuentes detectados en las pruebas diagnósticas y "
        "la estrategia recomendada para cada uno. "
        "Descuentos sucesivos: los estudiantes suman los porcentajes, de modo que un 20 por ciento "
        "seguido de un 10 por ciento lo tratan como un 30 por ciento único. La estrategia "
        "recomendada es trabajar con el factor multiplicativo del precio que queda, es decir 0.80 y "
        "luego 0.90, organizados en una tabla de dos pasos, en lugar de sumar los porcentajes. "
        "Fracción del resto: cuando un enunciado dice 'un tercio de lo que quedaba', los estudiantes "
        "aplican la fracción al total original. Se recomienda pedir que escriban explícitamente "
        "cuánto queda antes de aplicar la segunda fracción. "
        "Proporcionalidad inversa: ante 'más obreros, menos días' aplican regla de tres directa. La "
        "estrategia es calcular primero el trabajo total en días-obrero. "
        "Redondeo contextual: al dividir personas entre vehículos o cajas entre camiones, el "
        "resultado debe redondearse hacia arriba porque no existen fracciones de vehículo.")},

    {"id": "formulario", "fuente": "Formulario de geometría — guía del estudiante",
     "texto": (
        "Áreas. Rectángulo: base por altura. Cuadrado: lado al cuadrado. Triángulo: base por altura "
        "dividido entre dos. Trapecio: la semisuma de las bases multiplicada por la altura. "
        "Círculo: pi por el radio al cuadrado. "
        "Perímetros. Rectángulo: dos por la suma de base y altura. Cuadrado: cuatro por el lado. "
        "Circunferencia: dos por pi por el radio. "
        "Volúmenes. Cubo: arista al cubo. Prisma rectangular: largo por ancho por alto. "
        "Cilindro: pi por el radio al cuadrado por la altura. "
        "Teorema de Pitágoras: en un triángulo rectángulo, el cuadrado de la hipotenusa es igual a "
        "la suma de los cuadrados de los catetos. "
        "Para figuras compuestas se descompone la figura en formas simples y se suman o restan sus "
        "áreas según corresponda.")},

    {"id": "jerarquia", "fuente": "Guía de operaciones — jerarquía y signos",
     "texto": (
        "El orden para resolver una expresión con varias operaciones es: primero los paréntesis, "
        "empezando por los más internos; después las potencias y raíces; luego las multiplicaciones "
        "y divisiones de izquierda a derecha; y por último las sumas y restas, también de izquierda "
        "a derecha. "
        "Un error común es resolver de izquierda a derecha sin respetar la jerarquía. "
        "Reglas de signos: al multiplicar o dividir, signos iguales dan positivo y signos distintos "
        "dan negativo. Restar un número negativo equivale a sumar su valor absoluto. "
        "La división entre cero no está definida: no existe ningún número que multiplicado por cero "
        "dé un resultado distinto de cero.")},

    {"id": "fracciones", "fuente": "Guía de fracciones, decimales y porcentajes",
     "texto": (
        "Para sumar o restar fracciones con distinto denominador se busca el mínimo común múltiplo y "
        "se convierten ambas al denominador común. Un error frecuente es sumar numeradores y "
        "denominadores por separado. "
        "Para multiplicar fracciones se multiplican numeradores entre sí y denominadores entre sí. "
        "Para dividir se multiplica por la fracción inversa. "
        "Calcular un porcentaje de una cantidad equivale a multiplicar por el porcentaje dividido "
        "entre cien. Para hallar el precio tras un aumento del quince por ciento se multiplica por "
        "1.15; para recuperar el precio anterior a partir del actual se DIVIDE entre 1.15, no se "
        "resta el quince por ciento. "
        "Una división entre un número menor que uno da un resultado mayor que el dividendo.")},
]

print(f"Documentos: {len(corpus)}")
print("Caracteres por documento:", [len(d["texto"]) for d in corpus])
'''))

    # ------------------------------------------------------------------ 6
    c.append(md("""
---
## 6 · Chunking

El **chunk es la unidad de recuperación**: la búsqueda no encuentra documentos,
encuentra chunks. Si la respuesta a una pregunta queda partida entre dos, ningún
retrieval perfecto la traerá completa.

Usamos *fixed-size* con solapamiento, la estrategia más simple. Punto de partida
de la clase: 300–500 caracteres con 10–15% de overlap.

> **Experimenten:** bajen `CHUNK_SIZE` a 100 y súbanlo a 1500, reindexen y
> vuelvan a correr el Lab A. El chunking es una decisión de diseño **medible**,
> y con el harness de M2 pueden medirla de verdad en vez de opinar.
"""))

    c.append(code('''
CHUNK_SIZE = 400
OVERLAP    = 60


def partir_en_chunks(texto, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, inicio = [], 0
    while inicio < len(texto):
        chunks.append(texto[inicio:inicio + size])
        inicio += size - overlap
    return chunks


chunks, metadatos, ids_chunks = [], [], []
for doc in corpus:
    for j, ch in enumerate(partir_en_chunks(doc["texto"])):
        chunks.append(ch)
        metadatos.append({"fuente": doc["fuente"], "doc_id": doc["id"]})
        ids_chunks.append(f"{doc['id']}_chunk{j}")

print(f"{len(corpus)} documentos -> {len(chunks)} chunks")
print(f"Caracteres por chunk: min={min(len(c) for c in chunks)} max={max(len(c) for c in chunks)}")
print("\\nEjemplo de chunk. Miren DÓNDE cortó: ¿partió una idea por la mitad?\\n")
print(repr(chunks[2]))
'''))

    # ------------------------------------------------------------------ 7
    c.append(md("""
---
## 7 · Lab A · Indexar y buscar: palabras clave vs. significado

Indexamos en Chroma y comparamos dos formas de buscar. La meta **no** es que la
semántica gane siempre, sino ver **cuándo gana, cuándo empata y cuándo trae
ruido**.
"""))

    c.append(code('''
import chromadb

cliente = chromadb.Client()
try:
    cliente.delete_collection("tutor_matematicas")
except Exception:
    pass
coleccion = cliente.create_collection("tutor_matematicas", metadata={"hnsw:space": "cosine"})

# FASE 1 · INDEXACIÓN (offline, una sola vez por corpus)
embeddings = st.encode(chunks, show_progress_bar=False)
coleccion.add(ids=ids_chunks, documents=chunks, metadatas=metadatos,
              embeddings=embeddings.tolist())
print("Indexados", coleccion.count(), "chunks en Chroma.")
'''))

    c.append(code('''
def buscar_keyword(consulta, k=3):
    """Baseline ingenuo: cuenta cuántas palabras de la consulta aparecen en el chunk."""
    palabras = set(consulta.lower().split())
    puntajes = [(sum(1 for p in palabras if p in ch.lower()), i) for i, ch in enumerate(chunks)]
    puntajes.sort(reverse=True)
    return [(s, ids_chunks[i], chunks[i]) for s, i in puntajes[:k]]


def buscar_semantica(consulta, k=3):
    """Búsqueda por significado: embebe la consulta y busca los vecinos más cercanos."""
    emb = st.encode([consulta]).tolist()
    r = coleccion.query(query_embeddings=emb, n_results=k)
    return [(round(1 - d, 3), i, doc, m["fuente"])
            for d, i, doc, m in zip(r["distances"][0], r["ids"][0],
                                    r["documents"][0], r["metadatas"][0])]


def comparar_busquedas(consulta, k=2):
    print("=" * 84)
    print("CONSULTA:", consulta)
    print("-" * 84)
    print("KEYWORD (coincidencias de palabras):")
    for s, cid, ch in buscar_keyword(consulta, k):
        print(f"  [{s} palabras] {cid:22s} {ch[:95]}...")
    print("SEMÁNTICA (similitud coseno):")
    for s, cid, ch, fu in buscar_semantica(consulta, k):
        print(f"  [sim {s:5.3f}]   {cid:22s} {ch[:95]}...")
'''))

    c.append(code('''
# 1 · LITERAL — las palabras de la consulta están en el texto. Las dos deberían acertar.
comparar_busquedas("¿Cuál es la nota mínima aprobatoria?")
'''))

    c.append(code('''
# 2 · COLOQUIAL — un acudiente preguntando lo mismo con otras palabras.
# Casi ninguna coincide con el documento: aquí el keyword se pierde.
comparar_busquedas("mi hijo sacó 2.8, ¿eso pasa o pierde la materia?")
'''))

    c.append(code('''
# 3 · DEL RUIDO — es del tema (evaluación) pero pregunta algo que NO está en el corpus.
# Noten que el top-k SIEMPRE devuelve k resultados, respondan o no la pregunta.
comparar_busquedas("¿cuántas horas de matemáticas hay a la semana en grado once?")
'''))

    c.append(md("""
> **Qué observar en el Lab A**
>
> - **Consulta literal:** las dos búsquedas llegan al mismo chunk. Con palabras
>   compartidas, el keyword search es difícil de vencer — y es mucho más barato.
> - **Consulta coloquial:** "¿eso pasa o pierde la materia?" no comparte casi
>   nada con "la nota mínima aprobatoria es 3.0". El keyword se dispersa; la
>   semántica debería aterrizar en el chunk correcto. **Encontró por
>   significado, no por palabras.**
> - **Consulta del ruido:** el retrieval **siempre** devuelve k chunks, aunque
>   ninguno responda. Ordena por cercanía, no por "responde / no responde". Ese
>   ruido es el que se cuela al prompt en el Lab B, y por eso hace falta la
>   válvula de escape.
"""))

    # ------------------------------------------------------------------ 8
    c.append(md("""
---
## 8 · Lab B · El RAG ingenuo, de punta a punta

### El prompt aumentado, adaptado a un tutor

La receta de clase tiene cuatro partes: instrucción, contexto, válvula de
escape y pregunta. La aplicamos con **una modificación necesaria**.

El prompt canónico dice *"responde SOLO con base en el contexto"*. Para un tutor
de matemáticas eso rompe la mitad de los casos: si le preguntan cuánto se paga
tras dos descuentos, la respuesta **no está en ningún documento** —hay que
calcularla— y el modelo se negaría a hacerlo.

La solución es un prompt de **dos modos** que el propio modelo distingue:

| Si la pregunta es… | El sistema debe… |
|---|---|
| un **dato institucional** (normas, plan de área, criterios) | responder solo desde el contexto y citar la fuente; si no está, admitirlo |
| un **problema de cálculo** | resolverlo paso a paso, usando el contexto solo si aporta una fórmula o estrategia |

La **válvula de escape** sigue siendo la línea más importante, pero acotada a
los datos institucionales: es ahí donde inventar es grave.
"""))

    c.append(code('''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

MODELO_BASE   = "Qwen/Qwen2.5-1.5B-Instruct"
DIR_ADAPTADOR = "adaptadores/qwen-lora"      # el modelo afinado de M1
K = 3                                         # chunks recuperados por pregunta
MAX_TOKENS_GEN = 220

if not Path(DIR_ADAPTADOR).exists():
    raise FileNotFoundError(
        f"No se encuentra {DIR_ADAPTADOR}. Ejecuten S04_Lab_Fine_tuning_Qwen.ipynb "
        "o monten el Drive donde quedó guardado.")

tok = AutoTokenizer.from_pretrained(DIR_ADAPTADOR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_base = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float16).to(DEVICE)
modelo = PeftModel.from_pretrained(_base, DIR_ADAPTADOR).eval()
print("Generador listo:", MODELO_BASE, "+ adaptador LoRA de M1")
'''))

    c.append(code('''
# Sin RAG: EXACTAMENTE el sistema evaluado en M2. Es el baseline de este notebook.
INSTRUCCION = (
    "Eres un tutor de matemáticas. Resuelve el siguiente problema explicando "
    "el procedimiento paso a paso y termina con la respuesta final."
)

# Con RAG: la instrucción de dos modos + la válvula de escape.
SYSTEM_RAG = (
    "Eres un tutor de matemáticas que dispone de documentos de consulta.\\n"
    "- Si la pregunta pide un DATO de la institución (normas, plan de área, "
    "criterios de evaluación, protocolos), responde ÚNICAMENTE con base en el "
    "contexto y menciona la fuente. Si ese dato no aparece en el contexto, "
    'responde exactamente: "No tengo esa información en mis fuentes." '
    "No inventes datos institucionales.\\n"
    "- Si la pregunta es un PROBLEMA matemático, resuélvelo explicando el "
    "procedimiento paso a paso y termina con la respuesta final. Usa el "
    "contexto solo si contiene una fórmula o estrategia que ayude.\\n"
    "Sé breve y claro."
)


@torch.no_grad()
def _generar(system, user, max_new_tokens=MAX_TOKENS_GEN):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(modelo.device)
    out = modelo.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def sistema_sin_rag(pregunta):
    return _generar(INSTRUCCION, pregunta)


def sistema_rag(pregunta, k=K, verbose=False):
    resultados = buscar_semantica(pregunta, k)                       # RETRIEVE
    contexto = "\\n\\n".join(f"[Fuente: {fu}]\\n{doc}" for _, _, doc, fu in resultados)
    user = f"Contexto:\\n{contexto}\\n\\nPregunta: {pregunta}"        # AUGMENT
    if verbose:
        print("--- chunks recuperados ---")
        for s, cid, _, fu in resultados:
            print(f"  [sim {s:5.3f}] {cid:22s} ({fu[:46]}...)")
        print("--------------------------")
    return _generar(SYSTEM_RAG, user)                                # GENERATE
'''))

    c.append(code('''
# Comparación lado a lado sobre el bloque de CONOCIMIENTO: preguntas cuya
# respuesta vive en el corpus y en ningún otro sitio.
for caso in [c for c in eval_set if c["bloque"] == "conocimiento"][:2]:
    print("=" * 84)
    print("PREGUNTA:", caso["input"])
    print("ESPERADO:", caso["esperado"])
    print("\\n--- SIN RAG (de memoria) ---")
    print(sistema_sin_rag(caso["input"])[:400])
    print("\\n--- CON RAG (busca, lee y cita) ---")
    print(sistema_rag(caso["input"], verbose=True)[:400])
    print()
'''))

    c.append(md("""
> **Qué esperar.** El dato ("el examen final vale 40%") es de *este* reglamento:
> no existe en la memoria del modelo. El sin-RAG debería responder algo genérico
> o —peor y más probable— **inventar un porcentaje con total seguridad**. El
> con-RAG debería dar el 40% citando el SIEE.
>
> Si el con-RAG falla, diagnostiquen con los tres modos de fallo de la clase:
> ¿llegó el chunk correcto al top-k (miren el `verbose`)? ¿llegó pero el modelo
> lo ignoró? ¿o el dato no estaba en el corpus?
"""))

    c.append(code('''
# Prueba ADVERSARIAL de retrieval: la respuesta NO está en el corpus.
# El sistema honesto lo admite; el deshonesto inventa.
p_adv = "¿Cuántas horas de matemáticas a la semana hay en grado once?"
print("PREGUNTA (sin respuesta en el corpus):", p_adv)
print("\\n--- CON RAG ---")
print(sistema_rag(p_adv, verbose=True))
print("\\n¿Dijo que no está en sus fuentes, o inventó un número?")
'''))

    c.append(code('''
# Y el otro modo: un PROBLEMA DE CÁLCULO. Aquí el contexto no trae la respuesta
# (hay que calcularla). Comprobamos que el prompt de dos modos no bloquea al
# tutor: debe resolver igual, no responder "no tengo esa información".
caso_calculo = next(c for c in eval_set if c["id"] == "dif-06")
print("PREGUNTA:", caso_calculo["input"])
print("\\n--- CON RAG ---")
print(sistema_rag(caso_calculo["input"], verbose=True))
print(f"\\nEsperado: 144000   |   ¿Correcta? "
      f"{respuesta_correcta(sistema_rag(caso_calculo['input']), '144000')}")
'''))

    # ------------------------------------------------------------------ 9
    c.append(md("""
---
## 9 · El harness de M2, sin tocar

`sistema_rag` es una función `pregunta -> respuesta`: exactamente lo que el
harness de M2 recibe. Lo cargamos **idéntico**, porque si lo cambiáramos la
comparación de scorecards mediría la diferencia entre dos harness en vez del
efecto del RAG.
"""))
    c.extend(celda_evaluacion())
    c.extend(celda_juez())
    c.extend(celda_harness())

    c.append(md("""
### 9.1 · Las dos pasadas

Mismo eval set, mismo harness, mismas métricas. Lo único que cambia es si el
sistema consulta la biblioteca antes de responder.
"""))

    c.append(code('''
print("=== SIN RAG (el sistema de M2) ===")
sc_sin_rag = harness(eval_set, sistema_sin_rag, nombre="sin RAG")
'''))

    c.append(code('''
print("=== CON RAG ===")
sc_rag = harness(eval_set, lambda p: sistema_rag(p), nombre="con RAG")
'''))

    # ------------------------------------------------------------------ 10
    c.append(md("""
---
## 10 · Comparación de scorecards
"""))

    c.append(code('''
imprimir_scorecard(sc_sin_rag, sc_rag)
'''))

    c.append(code('''
print("EL DESGLOSE QUE IMPORTA — RAG por bloque")
print("=" * 72)
for etiqueta, clave, n_clave in [
    ("Cálculo      (habilidad)", "aciertos_calculo", "n_calculo"),
    ("Conocimiento (RAG)      ", "aciertos_conocimiento", "n_conocimiento"),
    ("Adversarial  (robustez) ", "aciertos_adversarial", "n_adversarial"),
]:
    a, b = sc_sin_rag[clave], sc_rag[clave]
    n = sc_sin_rag[n_clave]
    flecha = "+" if b > a else ("=" if b == a else "-")
    print(f"  {etiqueta}  {a}/{n}  ->  {b}/{n}   [{flecha}{abs(b-a)}]")
print("=" * 72)
print()
print(f"Alucinaciones: {sc_sin_rag['alucinaciones']} -> {sc_rag['alucinaciones']}")
print(f"Abstenciones : {sc_sin_rag['abstenciones']} -> {sc_rag['abstenciones']}")
print()
print("La hipótesis de la sección 1 era: RAG gana en CONOCIMIENTO y no aporta")
print("(o estorba) en CÁLCULO. Comparen con lo que salió y repórtenlo tal cual.")
'''))

    c.append(code('''
import matplotlib.pyplot as plt

bloques = ["cálculo", "conocimiento", "adversarial"]
n_por_bloque = [sc_rag["n_calculo"], sc_rag["n_conocimiento"], sc_rag["n_adversarial"]]
sin = [sc_sin_rag["aciertos_calculo"] / sc_sin_rag["n_calculo"],
       sc_sin_rag["aciertos_conocimiento"] / sc_sin_rag["n_conocimiento"],
       sc_sin_rag["aciertos_adversarial"] / sc_sin_rag["n_adversarial"]]
con = [sc_rag["aciertos_calculo"] / sc_rag["n_calculo"],
       sc_rag["aciertos_conocimiento"] / sc_rag["n_conocimiento"],
       sc_rag["aciertos_adversarial"] / sc_rag["n_adversarial"]]

x = np.arange(3); ancho = 0.38
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.bar(x - ancho/2, sin, ancho, label="Sin RAG (M2)")
ax.bar(x + ancho/2, con, ancho, label="Con RAG (M3)")
for i, (a, b) in enumerate(zip(sin, con)):
    ax.text(i - ancho/2, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
    ax.text(i + ancho/2, b + 0.02, f"{b:.0%}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"{b}\\n(n={n})" for b, n in zip(bloques, n_por_bloque)])
ax.set_ylim(0, 1.12); ax.set_ylabel("Aciertos de dominio")
ax.set_title("Efecto del RAG por bloque del eval set")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
'''))

    c.append(code('''
tabla_por_subtipo(sc_sin_rag, sc_rag)
'''))

    # ------------------------------------------------------------------ 11
    c.append(md("""
---
## 11 · Diagnóstico: ¿falló la búsqueda o falló la generación?

Cuando el RAG se equivoca hay **tres causas posibles** y se arreglan con cosas
distintas. La celda de abajo las separa automáticamente:

| Modo de fallo | Cómo se detecta | Se arregla con |
|---|---|---|
| **El dato no está en el corpus** | Ningún chunk lo contiene | Más documentos |
| **Fallo de retrieval** | El chunk correcto existe pero no entró al top-k | S08: hybrid search, reranking, query transformation |
| **Fallo de generación** | El chunk correcto llegó al prompt y el modelo lo ignoró | Prompt, modelo mayor, o fine-tuning sobre respuestas con contexto |

Distinguirlos importa porque S08 ataca **solo el segundo**. Si sus fallos son de
generación, las técnicas de la próxima semana no los van a arreglar.
"""))

    c.append(code('''
print("DIAGNÓSTICO DE LOS FALLOS DEL RAG")
print("=" * 84)

fallos = [d for d in sc_rag["detalle"] if not d["acierto"]]
if not fallos:
    print("Sin fallos en este eval set.")

for d in fallos:
    caso = next(c for c in eval_set if c["id"] == d["id"])
    recuperados = buscar_semantica(caso["input"], K)

    # ¿Alguna clave de la respuesta esperada aparece en algún chunk recuperado?
    claves = caso["evaluacion"].get("claves") or [caso["evaluacion"].get("valor", "")]
    claves = [k for k in claves if k]
    en_contexto = any(_norm(k) in _norm(doc) for k in claves for _, _, doc, _ in recuperados)
    en_corpus = any(_norm(k) in _norm(doc["texto"]) for k in claves for doc in corpus)

    if d["bloque"] == "calculo":
        causa = "CÁLCULO (el dato no vive en ningún documento: es habilidad, no conocimiento)"
    elif not en_corpus:
        causa = "EL DATO NO ESTÁ EN EL CORPUS -> añadir documentos"
    elif not en_contexto:
        causa = "FALLO DE RETRIEVAL -> el chunk existe pero no entró al top-k (S08)"
    else:
        causa = "FALLO DE GENERACIÓN -> el chunk correcto llegó y el modelo lo ignoró"

    print(f"[{d['id']} · {d['subtipo']}]  {causa}")
    print(f"   pregunta : {d['input'][:78]}")
    print(f"   motivo   : {d['motivo'][:78]}")
    print(f"   top-{K}    : {[cid for _, cid, _, _ in recuperados]}")
    print("-" * 84)
'''))

    # ------------------------------------------------------------------ 12
    c.append(md("""
---
## 12 · Artefactos de la entrega
"""))

    c.append(code('''
import json
from pathlib import Path

Path("resultados").mkdir(exist_ok=True)

guardar_scorecard("resultados/scorecard_rag.csv", sc_sin_rag, sc_rag)

Path("resultados/m3_rag.json").write_text(json.dumps({
    "config": {"chunk_size": CHUNK_SIZE, "overlap": OVERLAP, "k": K,
               "n_documentos": len(corpus), "n_chunks": len(chunks),
               "modelo_embeddings": "paraphrase-multilingual-MiniLM-L12-v2",
               "generador": f"{MODELO_BASE} + LoRA (M1)"},
    "sin_rag": {k: v for k, v in sc_sin_rag.items() if k != "detalle"},
    "con_rag": {k: v for k, v in sc_rag.items() if k != "detalle"},
}, ensure_ascii=False, indent=2), encoding="utf-8")

# Las consultas donde el retrieval falló: el material de trabajo de S08.
fallidas = [{"id": d["id"], "input": d["input"], "subtipo": d["subtipo"],
             "motivo": d["motivo"],
             "top_k": [cid for _, cid, _, _ in buscar_semantica(d["input"], K)]}
            for d in sc_rag["detalle"] if not d["acierto"] and d["bloque"] != "calculo"]
Path("resultados/consultas_fallidas_s08.json").write_text(
    json.dumps(fallidas, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\\nConsultas fallidas guardadas para S08: {len(fallidas)}")
for p in sorted(Path("resultados").glob("*rag*")) + sorted(Path("resultados").glob("*s08*")):
    print(f"  {p}")
'''))

    # ------------------------------------------------------------------ 13
    c.append(md("""
---
## 13 · Discusión

**1. ¿Dónde pagó el RAG?**
Si el bloque de conocimiento subió y el de cálculo se quedó igual, el resultado
confirma la tesis de S07: RAG da **conocimiento**, no **habilidad**. Es
exactamente lo que debía pasar y hay que reportarlo como confirmación, no como
decepción.

**2. ¿Bajó el bloque de cálculo?**
Es posible, y sería un hallazgo legítimo: meter tres chunks irrelevantes en el
prompt de un problema que solo requiere razonar añade ruido. Si pasó, la
mitigación natural es **enrutar**: usar RAG solo cuando la pregunta pide un dato
institucional. Conviene notar que es la misma conclusión a la que llegó el
pipeline BERT→Qwen de M1 —dar contexto que el modelo no necesitaba no ayudaba—,
solo que ahora con una causa distinta.

**3. ¿Bajaron las alucinaciones?**
Es la métrica que mejor mide la válvula de escape. Un tutor que dice "no tengo
esa información en mis fuentes" es más útil que uno que inventa el porcentaje
del examen final, aunque el acierto sea el mismo (cero) en ambos casos.

**4. ¿Qué tipo de fallo domina?**
De la sección 11. Si son de **retrieval**, S08 tiene las herramientas
(hybrid search, reranking, query transformation). Si son de **generación**, esas
técnicas no servirán y hay que mirar el prompt o el tamaño del modelo.

**5. El límite de siempre.**
21 casos, 4 de ellos de conocimiento. Un acierto vale 25 puntos porcentuales en
ese bloque. **Ninguna diferencia pequeña es concluyente**, y con este tamaño de
muestra lo honesto es hablar de dirección, no de magnitud.

---

### Lo que se llevan para S08

1. **Pipeline RAG completo y sin frameworks:** chunking → embeddings → Chroma →
   retrieve → augment → generate.
2. **`scorecard_rag.csv`** junto al de M2, medido con el mismo harness.
3. **El corpus indexado** — sustituyan los seis documentos ficticios por los
   reales de su institución (10–30 con fuente y fecha).
4. **`consultas_fallidas_s08.json`** — las consultas donde el retrieval falló.
   Sobre ellas se prueban hybrid search y reranking la próxima semana.
"""))

    return c
