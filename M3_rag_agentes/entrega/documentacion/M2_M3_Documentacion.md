# Módulos M2 y M3 — Harness de evaluación y RAG

**Universidad EAFIT | SI4006 — Tópicos Especiales y Aplicaciones en IA**
**Sistema evaluado:** `Qwen/Qwen2.5-1.5B-Instruct` + LoRA (modelo elegido en M1)

---

## RESUMEN EJECUTIVO

**El hallazgo principal de M2 corrige la conclusión de M1.**

En M1 concluimos que el fine-tuning *"enseñó a explicar, no a calcular"*, porque
la exactitud pasó de 90.91% a 93.94% — un solo ejemplo, sin significancia. Esa
conclusión estaba **limitada por el instrumento**: con un baseline del 90.91%,
el conjunto de validación estaba saturado y no tenía capacidad de detectar
mejora.

Con un eval set diseñado para ser difícil, el mismo par de modelos se separa
así:

| Conjunto de evaluación | Baseline | Fine-tuned | Diferencia |
|---|---|---|---|
| M1 (validación, 33 casos fáciles) | 90.91% | 93.94% | **+3 puntos** (1 ejemplo) |
| M2 (eval set difícil, 12 casos de cálculo) | 50.0% | 83.3% | **+33 puntos** (4 ejemplos netos) |

El fine-tuning **sí mejoró la capacidad de cálculo**. No se veía porque el
examen era demasiado fácil. Esto valida empíricamente la tesis central del
Módulo 2: *un eval set que no reta al sistema no informa nada*.

**El hallazgo principal de M3 es que el RAG resulta de suma cero.**

| Bloque | Sin RAG | Con RAG | Δ |
|---|---|---|---|
| Cálculo | 10/12 | 6/12 | **−4** |
| Conocimiento | 0/4 | 3/4 | **+3** |
| Adversarial | 3/5 | 4/5 | **+1** |
| **Total** | **13/21** | **13/21** | **0** |

RAG aporta exactamente lo que S07 predice —**conocimiento**, no habilidad— y
cobra un precio en los problemas donde el contexto solo añade ruido. La
recomendación operativa es **enrutar**: activar el RAG únicamente cuando la
pregunta pide un dato institucional.

---

# 1 · Qué se midió y con qué

## 1.1 El eval set

`data/eval_set_m2.jsonl` — 21 casos escritos a mano, **sin solapamiento con el
corpus de entrenamiento** (el generador falla si detecta contaminación).

| Bloque | n | Qué mide |
|---|---|---|
| **A · Cálculo** | 12 | Habilidad: razonar en varios pasos |
| **B · Conocimiento** | 4 | Datos institucionales que no están en los pesos |
| **C · Adversarial** | 5 | Robustez: premisa falsa, división entre cero, datos insuficientes, fuera de dominio, presión de rol |

Cada caso de cálculo ataca un modo de fallo concreto y documentado: descuentos
encadenados, fracción del resto, proporcionalidad inversa, porcentaje inverso,
redondeo contextual, probabilidad sin reemplazo, etc.

## 1.2 Las tres dimensiones

| Dimensión | Métrica | Cómo se calcula |
|---|---|---|
| **1a** | Exactitud en cálculo | Valor tras `Respuesta final:` comparado numéricamente (tol. 1e-6, acepta fracciones) |
| **1b** | Similitud por embeddings | Coseno con `paraphrase-multilingual-MiniLM-L12-v2` |
| **2** | LLM-as-a-judge | `Qwen2.5-1.5B-Instruct`, pointwise, rúbrica 1–5 versionada |
| **3** | Aciertos de dominio | Criterio explícito por caso, más las señales `abstuvo` y `alucino` |

La definición completa está en `SCOREBOARD.md`.

---

# 2 · Resultados de M2 · El harness

## 2.1 El scorecard

```
========================================================================
                                              baseline    fine-tuned
------------------------------------------------------------------------
DIMENSIÓN 1 · MÉTRICA CLÁSICA
  1a · Exactitud en cálculo                     50.0%         83.3%
  1b · Similitud embeddings (0-1)               0.677         0.751
DIMENSIÓN 2 · LLM-AS-A-JUDGE
  2 · Nota media (1-5)                           2.76          3.38
DIMENSIÓN 3 · ACIERTOS DE DOMINIO
  3a · Total                                     9/21         13/21
  3b · Bloque cálculo                            6/12         10/12
  3c · Bloque conocimiento                        0/4           0/4
  3d · Bloque adversarial                         3/5           3/5
DIAGNÓSTICO
  Alucinaciones (menos es mejor)                   12             8
  Abstenciones honestas                             0             0
  Formato válido                                  19%          100%
  Palabras por respuesta                          111            33
========================================================================
```

*(Cifras tal como se ejecutaron. La sección 6 documenta un artefacto de medición
que afecta a una celda: el bloque adversarial del fine-tuned debería ser 4/5 y
el total 14/21.)*

## 2.2 El bloque de cálculo, caso por caso

| Caso | Subtipo | Baseline | Fine-tuned |
|---|---|---|---|
| dif-01 | multipaso_encadenado | FALLA | **OK** |
| dif-02 | informacion_distractora | OK | OK |
| dif-03 | conversion_unidades | OK | OK |
| dif-04 | redondeo_contextual | OK | OK |
| dif-05 | division_decimal | OK | OK |
| dif-06 | porcentaje_encadenado | FALLA | **OK** |
| dif-07 | fraccion_del_resto | OK | **FALLA** |
| dif-08 | proporcionalidad_inversa | OK | OK |
| dif-09 | geometria_compuesta | FALLA | **OK** |
| dif-10 | porcentaje_inverso | FALLA | FALLA |
| dif-11 | promedio_ponderado | FALLA | **OK** |
| dif-12 | probabilidad_sin_reemplazo | FALLA | **OK** |

**5 mejoraron, 1 empeoró.** McNemar exacto: p = 0.219.

Con n = 12 la prueba no alcanza significancia formal al 5%, y hay que decirlo.
Pero la limitación es el **tamaño de muestra, no el tamaño del efecto**: 33
puntos de diferencia con 5 cambios a favor y 1 en contra es una señal
direccional fuerte. Con 30 casos de cálculo en lugar de 12, un efecto de esta
magnitud sería significativo.

### Los dos fallos que persisten

**`dif-10 · porcentaje_inverso`** — el error clásico, y el modelo lo comete en
una variante distinta a la esperada:

```
Pregunta : El precio aumentó 15% y ahora cuesta 92000. ¿Cuánto costaba antes?
Respuesta: "0.15 × valor original = 92000, entonces valor = 92000 ÷ 0.15 = 61333.33"
Correcto : 92000 ÷ 1.15 = 80000
```

Confundió *el aumento* con *el precio final*. No es un error de aritmética: es
un error de **planteamiento**.

**`dif-07 · fraccion_del_resto`** — el único que el fine-tuning empeoró:

```
Respuesta: "(2/5) × 100 = 40%. El libro queda 60%: 300 × 0.60 = 180 páginas
            leídas. Dividimos entre los dos partes: 180 ÷ 2 = 90"
Correcto : 300 − 120 = 180 quedan; 180 ÷ 3 = 60
```

Encadena tres pasos, cada uno con una etiqueta equivocada. Es el modo de fallo
más peligroso de un tutor: **procedimiento con aspecto impecable y razonamiento
incoherente**. Un estudiante no tiene forma de detectarlo.

## 2.3 El bloque de conocimiento: 0/4 en ambos sistemas

Esperado y correcto: son datos de una institución concreta que no existen en los
pesos de ningún modelo. Lo que importa es **cómo** falla.

```
Pregunta : ¿Qué porcentaje de la nota del periodo vale el examen final?
Respuesta: "Paso 1: El sistema esencialmente divide el examen entre los dos
            periodos. Paso 2: Si el examen vale 50%, entonces cada periodo
            vale 50 ÷ 2 = 25%. Respuesta final: 25%"
```

El modelo **no solo inventó la cifra: inventó una premisa y un razonamiento
completo para sostenerla**, con el formato pedagógico impecable que le enseñamos
en M1. Es la alucinación en su forma más peligrosa, porque el formato correcto
le da credibilidad.

## 2.4 El hallazgo más grave: cero abstenciones

**En 21 casos, ninguno de los dos sistemas dijo nunca "no lo sé".** El baseline
alucinó en 12 de 21; el fine-tuned en 8 de 21.

Para un tutor esto es peor que equivocarse en una cuenta. Un estudiante puede
verificar un número; no puede verificar una norma del colegio que el sistema se
inventó con total seguridad.

El fine-tuning redujo las alucinaciones de 12 a 8, pero **no por honestidad**:
las redujo porque acertó más problemas de cálculo. En el bloque donde no podía
saber la respuesta, siguió inventando 4 de 4.

## 2.5 Calibración del juez

En la mayoría de dominios el LLM-juez es incalibrable: no hay verdad contra la
cual contrastarlo. En matemáticas sí la hay, así que medimos el acuerdo entre el
juez y la verdad objetiva sobre los 24 casos de cálculo (12 × 2 sistemas):

|  | juez dice MAL (≤3) | juez dice BIEN (≥4) |
|---|---|---|
| **Verdad: INCORRECTA** | 8 | **0** |
| **Verdad: CORRECTA** | **10** | 6 |

```
Acuerdo simple : 58.3%
Cohen's kappa  : 0.286   ->  acuerdo BAJO (S06: aceptable a partir de 0.6)
Falsos aprobados : 0
Falsos rechazos  : 10
```

**El juez se equivoca en una sola dirección: es demasiado severo.** Nunca aprobó
una respuesta incorrecta (0 falsos aprobados), pero rechazó 10 de las 16
correctas. La distribución de notas lo explica: **31 de 42 calificaciones fueron
exactamente un 3**. El juez comprime hacia el centro en lugar de discriminar.

Consecuencias prácticas, y hay que declararlas:

- **La Dimensión 2 no es fiable para comparar sistemas** con este juez. La
  diferencia 2.76 → 3.38 va en la dirección correcta, pero con kappa 0.286 no
  sostiene ninguna conclusión por sí sola.
- **Sí es fiable como filtro de seguridad.** Cero falsos aprobados significa que
  una nota ≥4 es una garantía razonable de corrección. Es un uso legítimo y más
  modesto del juez.
- **La causa probable es el tamaño.** Un juez de 1.5B evaluando aritmética de
  varios pasos tiene que resolver el problema para juzgarlo, y no siempre puede.
  Con `Qwen2.5-7B-Instruct` o un juez *reference-based* más estricto el kappa
  debería subir.

## 2.6 El bloque adversarial: 3/5

| Caso | Baseline | Fine-tuned | Comentario |
|---|---|---|---|
| adv-01 premisa_falsa | OK | OK | Corrigió que 9 no es primo |
| adv-02 indefinicion | OK | *(artefacto, ver §6)* | Respondió "No definido" — correcto |
| adv-03 datos_insuficientes | OK | OK | Pidió la distancia que falta |
| adv-04 fuera_de_dominio | FALLA | FALLA | Respondió "Gabriel García Márquez" |
| adv-05 rol_integridad | FALLA | **OK** | El fine-tuning arregló este |

**`adv-04` merece comentario**: el modelo respondió *correctamente* quién
escribió *Cien años de soledad* — pero lo hizo **con el formato de tutor de
matemáticas**: "Paso 1: Es una novela colombiana. Paso 2: El autor es Gabriel
García Márquez. Respuesta final: Gabriel García Márquez".

El fine-tuning fue tan efectivo enseñando el formato que el modelo lo aplica
**incluso a preguntas de literatura**. Es sobreajuste al formato, y es una
observación que solo aparece con un caso adversarial fuera de dominio.

**`adv-05`** es la mejora más clara del fine-tuning en este bloque. El baseline
respondió *"El resultado de 47 × 83 es 3861"* — rompió el rol **y** se equivocó
en la cuenta (es 3901). El fine-tuned explicó el procedimiento.

---

# 3 · Resultados de M3 · RAG

## 3.1 Configuración

| Parámetro | Valor |
|---|---|
| Documentos | 6 (SIEE, plan de área, protocolo, formulario, jerarquía, fracciones) |
| Chunks | 17 (*fixed-size*, 400 caracteres, overlap 60) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Base vectorial | Chroma, distancia coseno |
| k | 3 chunks por consulta |
| Generador | `Qwen2.5-1.5B-Instruct` + LoRA de M1 |

## 3.2 El scorecard

```
========================================================================
                                              sin RAG      con RAG
------------------------------------------------------------------------
  1a · Exactitud en cálculo                     83.3%         50.0%
  1b · Similitud embeddings                     0.751         0.710
  2  · LLM-juez (1-5)                            3.38          3.14
  3a · Aciertos totales                         13/21         13/21
  3b · Bloque cálculo                           10/12          6/12
  3c · Bloque conocimiento                        0/4           3/4
  3d · Bloque adversarial                         3/5           4/5
  Alucinaciones                                     8             7
  Abstenciones honestas                             0             2
  Formato válido                                 100%         95.2%
  Palabras por respuesta                           33            28
========================================================================
```

> **Nota de reproducibilidad.** La columna `sin RAG` de este notebook coincide
> **cifra por cifra** con la columna `fine-tuned` del scorecard de M2, pese a
> haberse ejecutado en una sesión distinta. Es la confirmación de que la
> decodificación *greedy* y la semilla fija hacen el experimento reproducible, y
> de que el harness es idéntico en los dos módulos.

## 3.3 El resultado de suma cero

```
Conocimiento   0/4  ->  3/4     +3
Adversarial    3/5  ->  4/5     +1
Cálculo       10/12 ->  6/12    −4
                              ------
Total         13/21 -> 13/21     0
```

**RAG hizo exactamente lo que S07 dice que hace, y nada más.** Dio conocimiento
consultable. No dio habilidad. Y cobró un precio.

### Por qué ganó en conocimiento

Los tres casos recuperados (`con-01` política de evaluación, `con-03` nota
mínima, `con-04` protocolo pedagógico) son datos que viven **solo** en los
documentos. El sistema pasó de inventar el 25% a citar el 40% del SIEE.

Además aparecieron las **primeras 2 abstenciones honestas del proyecto**. La
válvula de escape funcionó: el sistema aprendió a decir "no tengo esa
información en mis fuentes" cuando el chunk correcto no llegaba.

### Por qué perdió en cálculo

Los problemas de cálculo no tienen su respuesta en ningún documento: hay que
calcularla. Meter 3 chunks de 400 caracteres en el prompt de un problema que
solo requiere razonar **añade ruido sin añadir información**.

La caída de 4 ejemplos sobre 12 no alcanza significancia formal (McNemar,
p entre 0.125 y 0.289 según cómo se repartan mejoras y empeoramientos), pero la
dirección es consistente y el mecanismo está bien documentado en la literatura
de RAG.

> **Esta es la segunda vez que el proyecto encuentra el mismo patrón.** En M1, el
> pipeline BETO→Qwen inyectaba una "estrategia por categoría" en el prompt y el
> resultado fue C − A = −3.03%: dar contexto que el modelo no necesitaba lo
> empeoraba. Aquí ocurre otra vez, con otro mecanismo y la misma lección:
> **contexto irrelevante no es neutro, es ruido.**

## 3.4 Diagnóstico de los fallos del RAG

Las dos consultas no-cálculo que fallaron con RAG (`resultados/consultas_fallidas_s08.json`):

| Caso | Chunks recuperados | Diagnóstico |
|---|---|---|
| `con-02` plan_de_area | `siee_chunk0`, `plan_area_chunk0`, `plan_area_chunk1` | **Fallo de generación.** Los chunks correctos del plan de área SÍ llegaron al prompt; el modelo no extrajo "grado octavo" de ellos. |
| `adv-02` indefinicion | `fracciones_chunk1`, `jerarquia_chunk1`, `protocolo_chunk1` | **Artefacto de medición** (§6), no un fallo real. |

El caso `con-02` es importante para S08: **no es un fallo de retrieval**. Las
técnicas de la próxima semana —hybrid search, reranking, query transformation—
atacan el retrieval y **no lo arreglarían**. Este pide otra cosa: un modelo
mayor, o un prompt que fuerce a citar textualmente.

## 3.5 RAG avanzado (S08) · cada técnica se gana su lugar

Notebook: `S08_Lab_RAG_Avanzado_Tutor_Matematicas.ipynb` (generado por
`scripts/nb_rag_avanzado.py`). La plantilla de clase `S08_Lab_RAG_avanzado.ipynb`
se conserva sin cambios.

### Diseño: una sola cosa nueva por paso

El diagnóstico de 3.3–3.4 decía que **el retrieval no era el cuello de botella**:
el cálculo cayó por meter contexto donde no hacía falta, y `con-02` falló en la
generación. Por eso, además de las dos técnicas de S08, se prueban las que el
diagnóstico pide:

| Sistema | Añade | Ataca | Predicción (antes de ejecutar) |
|---|---|---|---|
| A · ingenuo | — (RAG de S07) | — | Reproduce S07 |
| B · +hybrid | BM25 + denso, fusión RRF | Retrieval: siglas, términos exactos | ≈ A en el scorecard |
| C · +rerank | Cross-encoder `mmarco-mMiniLMv2-L12` sobre top-10 | Retrieval: orden del top-k | ≈ B; mejor hit@1 |
| D · +router | Enrutador consulta / problema | **El ruido en cálculo** | ~17/21 (§4.4) |
| E · +cita | Prompt de un solo modo con cita textual en la vía RAG | **Fallo de generación de `con-02`** | +1 en conocimiento |

Corpus, chunking, prompts de S07 y harness son **los mismos**, importados del
mismo código (`scripts/nb_comun_rag.py`, `scripts/nb_comun_eval.py`): S07 se
regenera byte a byte igual tras la refactorización.

### Consultas de desarrollo (no contaminan el eval set)

33 consultas nuevas (17 consultas documentales, 16 problemas) para dos usos que
no pueden hacerse con el eval set: **medir el retrieval aislado** y **calibrar el
enrutador**. El notebook comprueba que ninguna se parece a un caso del eval set
(similitud máxima 0.867 < 0.90) y que la clave de cada una existe en el corpus.

### Retrieval aislado — medido

18 consultas con chunk de oro (14 de desarrollo + las 4 de conocimiento del eval
set), k = 3. Es determinista y no depende de la GPU:

| Retriever | hit@1 | hit@3 | MRR@3 | ms (mediana, CPU) |
|---|---|---|---|---|
| A · densa | 0.722 | 0.944 | 0.815 | 5 |
| B · híbrida | 0.833 | 0.944 | 0.880 | 5 |
| C · +rerank | **0.889** | **1.000** | **0.944** | 33 |

- **B gana donde se esperaba:** las siglas (`hit@3` 0.5 → 1.0: "Plan de área:
  contenidos de grado décimo" estaba en el puesto 5 con la densa). Pierde una
  coloquial ("dividió los pasajeros entre los buses": puesto 8).
- **C recupera las dos** y es el único con hit@3 = 1.0, a cambio de ×7 en latencia.
- **Las 4 consultas del eval set ya tenían hit@3 = 1.0 con A.** El retrieval no
  tenía margen para mover el bloque de conocimiento: lo confirma §3.4.

### El enrutador — medido

Se probaron varias opciones **solo sobre las consultas de desarrollo** (LOO):

| Enrutador | Exactitud LOO (33) |
|---|---|
| Centroides de embeddings | 78.8% |
| Regla "≥2 números → problema" | 87.9% |
| Qwen2.5-1.5B zero-shot | 66.7% (sesgo a "problema") |
| Umbral sobre el puntaje del reranker | 72.7% |
| **Regresión logística [margen centroides, nº números, rerank máx.]** | **97.0%** |

Cada señal falla en casos distintos (los centroides separan por *tema*, no por
*intención*; los números no ven "la mitad de un tercio"; el reranker no ve las
consultas sin respuesta en el corpus), y combinadas se cubren. Aplicado después
al eval set, **que no se usó para ajustar nada**: **16/16** rutas correctas en
cálculo + conocimiento. La decisión más frágil es `dif-11` (promedio ponderado
con "talleres valen 30%": P(consulta) = 0.40).

Coste: ~40 ms por pregunta (un encode + el reranker), y el prompt de los
problemas baja de ~550 a ~80 tokens porque ya no lleva contexto.

### Scorecard de los seis sistemas — ejecutado

`resultados/scorecard_rag_avanzado_v2.csv` y `deltas_s08_v2.csv`. Las cifras son las
**corregidas** por `scripts/recalcular_criterio.py`: la corrida usó la versión
antigua del eval set (el notebook lo avisó con `SHA coincide: False`) y `adv-02`
se re-evaluó después sobre las respuestas ya generadas (ver §6.1 y
`resultados/correccion_adv02.md`).

| Sistema | Total | Cálculo | Conoc. | Advers. | Alucin. | Δ vs anterior | McNemar |
|---|---|---|---|---|---|---|---|
| sin RAG | 14/21 | 10/12 | 0/4 | 4/5 | 7 | — | — |
| A · ingenuo | 14/21 | 6/12 | 3/4 | 5/5 | 7 | 0 | 5/5, p = 1.000 |
| B · +hybrid | 14/21 | 7/12 | **4/4** | 3/5 | 6 | 0 | 3/3, p = 1.000 |
| C · +rerank | 13/21 | 7/12 | 3/4 | 3/5 | 5 | −1 | 2/3, p = 1.000 |
| **D · +router** | **18/21** | **10/12** | 3/4 | 5/5 | **2** | **+5** | 7/2, p = 0.180 |
| E · +cita | 18/21 | 10/12 | 3/4 | 5/5 | 3 | 0 | 0/0, p = 1.000 |

**1 · El retrieval mejoró de verdad, y aun así no movió el scorecard.**

| Retriever | hit@1 | hit@3 | MRR@3 | ms (mediana) |
|---|---|---|---|---|
| A · densa | 0.722 | 0.944 | 0.815 | 12 |
| B · híbrida | 0.833 | 0.944 | 0.880 | 13 |
| C · +rerank | **0.889** | **1.000** | **0.944** | 43 |

Es exactamente la predicción de §3.4: con hit@3 ya en 0.94, el retrieval no tenía
margen para mover nada. B gana donde se dijo —consultas con siglas, hit@3 de 0.5 a
1.0— pero ese estilo de consulta no está en el eval set, así que la mejora es real
y **no aparece** en el scorecard. Es la diferencia entre medir una técnica y medir
el sistema.

**2 · El enrutador es lo único que paga.** 14 → 18 de 21, recuperando el cálculo
(6/12 → 10/12) sin perder conocimiento, y bajando las alucinaciones de 7 a 2. El
clasificador de tres señales dio **97.0%** en leave-one-out sobre las 33 consultas
de desarrollo (frente a 78.8% de solo centroides) y **16/16** en el eval set, que
nunca vio. La arquitectura compuesta que §4.4 proponía queda **demostrada**, y con
un margen mayor al previsto: se estimaban 17/21 y salieron 18/21.

**3 · E (cita textual) no arregló `con-02`.** El delta es exactamente 0: ningún
caso cambió. El fallo de generación diagnosticado en §3.4 no se corrige pidiendo
cita literal; es un límite del modelo de 1.5B. E además baja el formato válido de
95% a 76%, porque el prompt de cita compite con el formato "Paso N".

**4 · El coste.** El reranker triplica la latencia de recuperación (13 → 43 ms) sin
contrapartida en el scorecard. El enrutador, en cambio, **abarata**: el prompt medio
baja de 541 a 193 tokens, porque los problemas de cálculo dejan de llevar contexto.

**5 · El riesgo declarado no se materializó.** Se advirtió que los adversariales
irían por la vía sin RAG y perderían la válvula de escape; con el criterio
corregido, D y E obtienen 5/5 en ese bloque, el mejor resultado de los seis.

---

# 4 · Conclusiones

## 4.1 Sobre el fine-tuning (corrige a M1)

**El fine-tuning mejoró las dos cosas: la forma y el cálculo.** M1 solo pudo ver
la primera porque su eval set estaba saturado.

| Qué mejoró | Evidencia |
|---|---|
| **Forma** | Formato 19% → 100%; verbosidad 111 → 33 palabras (−70%) |
| **Cálculo** | 50% → 83.3% sobre problemas difíciles; 5 casos mejoraron, 1 empeoró |
| **Rol** | Arregló `adv-05`: dejó de dar el número pelado en un examen |

La afirmación de M1 —*"enseñó a explicar, no a calcular"*— era correcta **con
los datos de M1** e incorrecta como conclusión general. La causa fue el
instrumento, no el modelo.

## 4.2 Sobre la evaluación

**La lección más transferible del proyecto:** un eval set saturado no mide el
sistema, mide la facilidad del examen. Los mismos dos modelos, con la misma
métrica, se separan 3 puntos o 33 según el conjunto con el que se midan.

**El LLM-juez de 1.5B no sirve para rankear**, con kappa 0.286 y 31 de 42 notas
comprimidas en el 3. Sí sirve como filtro de seguridad: cero falsos aprobados.

**Las tres señales separadas (`acierto` / `abstuvo` / `alucino`) fueron
decisivas.** Un scorecard con solo "aciertos" habría dicho 0/4 en conocimiento y
punto. Con las tres, el diagnóstico es "0/4 aciertos **y 4/4 alucinaciones**",
que es un problema completamente distinto y mucho más grave.

## 4.3 Sobre RAG

**RAG resolvió el problema para el que sirve y solo ese.** +3 en conocimiento,
+1 en adversarial, −4 en cálculo, total 0.

Aplicarlo indiscriminadamente es un error: el sistema con RAG es **peor tutor de
matemáticas** que el sistema sin RAG, aunque sea mejor consultor del reglamento.

## 4.4 Recomendación de arquitectura

**Qwen2.5-1.5B + LoRA, con RAG condicional activado por un enrutador.**

```
Pregunta del estudiante
         |
   ¿pide un DATO institucional o es un PROBLEMA de cálculo?
         |
    +----+----------------------+
    |                           |
  DATO                      PROBLEMA
    |                           |
  RAG (k=3) + válvula      Qwen + LoRA, sin contexto
  de escape                     |
    |                           |
    +----------+----------------+
               |
          Respuesta
```

La evidencia sostiene esta arquitectura de forma directa: si el enrutador fuera
perfecto, el sistema obtendría **10/12 en cálculo** (la vía sin RAG) **y 3/4 en
conocimiento** (la vía con RAG) = **17/21**, frente a los 13/21 de cualquiera de
las dos configuraciones puras. Es la primera vez en todo el proyecto que una
arquitectura compuesta muestra una ganancia clara.

*(Esta es la estimación original, hecha con las cifras de §3.2 antes de corregir
`adv-02`, y se conserva tal cual para no reescribir una predicción a posteriori. La
corrida real del enrutador, con el criterio corregido, obtuvo **18/21**: ver §3.5.)*

Y le da por fin un propósito real al clasificador BETO de M1: bastaría
reentrenarlo con una clase adicional, `consulta_institucional`, para que decida
la ruta. Nótese la diferencia con el pipeline de M1, que enrutaba *dentro* de la
generación y no aportaba: aquí el enrutador decide **si consultar la biblioteca
o no**, que es una decisión con consecuencias medibles.

## 4.5 Qué falta para un tutor desplegable

1. **Enseñarle a abstenerse.** Cero abstenciones en 21 casos es el fallo más
   grave del sistema. Requiere ejemplos de entrenamiento donde la respuesta
   correcta sea "no tengo ese dato".
2. **Verificación simbólica del cálculo.** `dif-07` y `dif-10` fallan por
   planteamiento, no por aritmética; una herramienta externa (SymPy) que valide
   la coherencia de los pasos atacaría justo eso.
3. **Ampliar el corpus con los subtipos que fallan** (§5).
4. **Un juez mejor** o evaluación humana sobre una muestra, dado el kappa 0.286.

---

# 5 · Del diagnóstico al corpus de entrenamiento

Esto responde a la pregunta original: *¿qué ejemplos añadir para mejorar la
precisión?*

| Subtipo | Estado | Acción |
|---|---|---|
| `porcentaje_inverso` | Falla en ambos sistemas | **Prioridad 1.** Añadir ejemplos de "precio antes del aumento" |
| `fraccion_del_resto` | El fine-tuning lo empeoró | **Prioridad 1.** Añadir "fracción de lo que quedaba" |
| `multipaso_encadenado` | Se arregló con el fine-tuning | Reforzar: solo hay 1 caso de evidencia |
| `porcentaje_encadenado` | Se arregló | Reforzar |
| `geometria_compuesta` | Se arregló | Reforzar |
| Abstención | **Nunca ocurre** | **Prioridad 1.** Añadir ejemplos cuya salida sea admitir que falta un dato |
| Fuera de dominio | Falla en ambos | Añadir ejemplos de redirección al dominio |

**Advertencia metodológica, y es importante:** hay **un solo caso por subtipo**.
Cada fallo es una *hipótesis* de debilidad, no una medición. El procedimiento
correcto antes de reentrenar es escribir 3–5 casos más de cada subtipo
sospechoso, confirmar que el fallo es sistemático, y solo entonces añadir ~10
ejemplos de entrenamiento por subtipo confirmado en `scripts/registros.py`.

Añadir ejemplos a partir de un solo fallo es sobreajustar al eval set, que es
exactamente el pecado que S05 describe como contaminación.

---

# 6 · Limitaciones y artefactos de medición

Declaradas explícitamente porque condicionan la lectura de todo lo anterior.

## 6.1 Un artefacto del evaluador (afecta a una celda del scorecard)

El caso `adv-02` (división entre cero) se contó como fallo del modelo
fine-tuned, y **no lo era**:

```
Respuesta del modelo: "Dividir entre cero no tiene sentido en matemáticas.
                       Respuesta final: No definido"          <- CORRECTA
Claves que buscaba  : "no está definida", "indefinido", "no existe", ...
```

La lista de claves no contemplaba *"no definido"* (masculino, sin artículo) ni
*"no tiene sentido"*. El criterio, no el modelo, produjo el fallo.

**Corregido** en `scripts/eval_set_fuente.py` y verificado con la respuesta real.
Cifras corregidas:

| | Ejecutado | Corregido |
|---|---|---|
| Fine-tuned · adversarial | 3/5 | **4/5** |
| Fine-tuned · total | 13/21 | **14/21** |
| Con RAG · adversarial | 4/5 | **5/5** |
| Con RAG · total | 13/21 | **14/21** |

Ninguna conclusión cambia: el bloque de cálculo y el de conocimiento no se ven
afectados, y la comparación entre sistemas se mueve igual en ambos.

**La lección va más allá del bug.** Un criterio de acierto por palabras clave es
frágil ante la variedad del lenguaje natural. Es el precio de tener un criterio
objetivo y auditable, y hay que asumirlo declarándolo — igual que se declara el
sesgo del juez.

## 6.2 Tamaño de muestra

- **21 casos.** En el bloque de conocimiento un acierto vale **25 puntos**.
- El bloque de cálculo (n=12) no alcanza significancia formal ni para la mejora
  del fine-tuning (p = 0.219) ni para la caída del RAG (p ≈ 0.125–0.289).
- **Un solo caso por subtipo de dificultad**: sirve para generar hipótesis, no
  para medir.

## 6.3 El juez

Kappa 0.286. La Dimensión 2 se reporta pero **no sostiene ninguna conclusión de
este documento por sí sola**. Todas las conclusiones se apoyan en la Dimensión 1
(exactitud numérica) y la Dimensión 3 (criterio por caso), que son objetivas.

## 6.4 Un hueco en la instrumentación

El notebook de M3 guarda el detalle caso por caso solo de los fallos que **no**
son de cálculo. Por eso sabemos que el RAG bajó de 10/12 a 6/12 en cálculo,
pero **no qué 4 casos concretos rompió**. Es una limitación del diseño del
notebook, no de los datos, y se corrige guardando el detalle completo del RAG
igual que hace el de M2.

**Corregido en S08:** `resultados/detalle_rag_avanzado.csv` guarda caso por caso
los seis sistemas con su ruta y su top-k, y el notebook imprime qué casos
arregló y cuáles rompió cada paso.

## 6.5 Corpus documental ficticio

Los seis documentos del RAG son **verosímiles pero inventados**, construidos
para que el experimento sea reproducible. Los resultados del bloque de
conocimiento miden la *mecánica* del RAG correctamente, pero para desplegar
haría falta sustituirlos por los documentos reales de la institución.

## 6.6 Una sola semilla

No hay estimación de varianza entre ejecuciones. Lo que sí hay es una
comprobación fuerte de determinismo: el scorecard `sin RAG` de M3 reprodujo
exactamente el `fine-tuned` de M2 en una sesión distinta.

---

# 7 · Artefactos de la entrega

| Archivo | Contenido |
|---|---|
| `data/eval_set_m2.jsonl` | Los 21 casos del eval set |
| `resultados/scorecard_m2.csv` | Scorecard baseline vs fine-tuned |
| `resultados/detalle_m2.csv` | Evidencia caso por caso (42 filas) |
| `resultados/m2_harness.json` | Scorecard + calibración del juez |
| `resultados/rubrica_juez.txt` | La rúbrica versionada |
| `resultados/scorecard_rag.csv` | Scorecard sin RAG vs con RAG |
| `resultados/m3_rag.json` | Configuración y scorecards del RAG |
| `resultados/consultas_fallidas_s08.json` | Consultas para trabajar en S08 |
| `S06_Lab_Harness_Tutor_Matematicas.ipynb` | El harness ejecutable |
| `S07_Lab_RAG_Tutor_Matematicas.ipynb` | El pipeline RAG |
| `S08_Lab_RAG_Avanzado_Tutor_Matematicas.ipynb` | Hybrid, reranking, enrutador y cita textual (A → E) |
| `resultados/deltas_s08.csv` · `_v2.csv` | Deltas paso a paso con McNemar y coste (v2 = criterio `adv-02` corregido) |
| `resultados/scorecard_rag_avanzado.csv` · `_v2.csv` | Scorecard de los seis sistemas (v2 = criterio `adv-02` corregido) |
| `resultados/detalle_rag_avanzado.csv` · `_v2.csv` | Caso por caso, todos los sistemas, con ruta y top-k |
| `resultados/retrieval_s08.csv` | Retrieval aislado por consulta (hit@k, MRR, latencia) |
| `resultados/m3_rag_avanzado.json` | Configuración, retrieval, enrutador, scorecards y coste |
| `resultados/correccion_adv02.md` | Qué casos cambiaron al re-evaluar con el criterio corregido |
| `SCOREBOARD.md` | Definición de métricas y eval set |
