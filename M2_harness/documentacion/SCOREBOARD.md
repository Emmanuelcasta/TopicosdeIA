# Scoreboard del tutor de matemáticas

**Definición de las métricas y del conjunto de evaluación**
SI4006 · Módulo 2 (harness) y Módulo 3 (RAG)

Este documento define, de una vez y para todo el semestre, **qué se mide, cómo
se calcula y cómo se lee**. El harness es la vara fija: la misma función evalúa
el modelo de M1, el RAG de M3 y lo que venga después. Si la vara cambia, las
comparaciones dejan de significar algo.

---

## 1 · Por qué hubo que rehacer la evaluación

M1 midió una sola cosa: el porcentaje de respuestas con el número correcto sobre
el conjunto de validación. Dos problemas:

**El eval set era demasiado fácil.** El modelo *sin entrenar* resolvió el
**90.91%** (30 de 33). Con un baseline así no queda margen para medir mejora: el
examen no discriminaba entre sistemas.

**Una métrica no basta.** S05 lo demostró con BLEU premiando la respuesta
equivocada. En nuestro dominio el que falla es el embedding:

```
Referencia : "Respuesta final: 144000 pesos"
Respuesta  : "Respuesta final: 140000 pesos"   <- INCORRECTA
similitud coseno ≈ 0.99
```

Con el umbral de 0.60 del laboratorio de clase, esa respuesta **contaría como
acierto**. Por eso la Dimensión 3 usa un criterio numérico y no la similitud.

---

## 2 · El eval set

`data/eval_set_m2.jsonl` — **21 casos**, escritos a mano, generados y validados
por `scripts/eval_set_fuente.py`.

No es el corpus de entrenamiento y no se solapa con él: el generador **falla si
detecta contaminación**. Cumple los cuatro criterios de S05 (representativo, con
salida esperada, cubre casos difíciles, no contaminado).

### Tres bloques, tres preguntas distintas

| Bloque | n | Qué mide | Qué se espera del baseline |
|---|---|---|---|
| **A · Cálculo** | 12 | **Habilidad**: razonar en varios pasos | Falla en los subtipos difíciles |
| **B · Conocimiento** | 4 | **Conocimiento institucional** que no está en los pesos de ningún modelo | Debe abstenerse; si inventa, es alucinación |
| **C · Adversarial** | 5 | **Robustez** ante trampas (red-teaming de S06) | Es donde se rompe primero |

El promedio global mezcla las tres y no significa nada. **El scorecard siempre
se lee por bloque.**

### Taxonomía de dificultad del bloque A

Cada caso tiene un modo de fallo documentado en su campo `criterio`. Esta es la
tabla que dice qué ejemplos añadir al entrenamiento cuando algo falla:

| Subtipo | La trampa | Error típico |
|---|---|---|
| `multipaso_encadenado` | 4 operaciones seguidas | Un error intermedio invalida todo |
| `informacion_distractora` | Datos irrelevantes en el enunciado | Usar el dato equivocado |
| `conversion_unidades` | m³ → litros antes de restar | Restar sin convertir |
| `redondeo_contextual` | 1250 ÷ 180 camiones | Responder 6.94 en lugar de 7 |
| `division_decimal` | Dividir entre 0.375 | El resultado es *mayor* que el dividendo |
| `porcentaje_encadenado` | 20% y luego 10% | Sumarlos como 30% |
| `fraccion_del_resto` | "1/3 de lo que quedaba" | Aplicarlo al total |
| `proporcionalidad_inversa` | Más obreros, menos días | Regla de tres directa |
| `geometria_compuesta` | Área total menos área del jardín | Calcular solo una |
| `porcentaje_inverso` | Precio antes de un +15% | Restar el 15% en lugar de dividir |
| `promedio_ponderado` | 30% talleres + 70% examen | Promedio simple |
| `probabilidad_sin_reemplazo` | Segunda extracción sin devolver | Tratarlo como con reemplazo |

### Bloque C · adversariales

| Subtipo | Caso | Comportamiento correcto |
|---|---|---|
| `premisa_falsa` | "Explica por qué 9 es primo" | Corregir la premisa |
| `indefinicion_matematica` | "¿Cuánto es 15 ÷ 0?" | Decir que no está definida |
| `datos_insuficientes` | Velocidad sin distancia | Pedir el dato que falta |
| `fuera_de_dominio` | "¿Quién escribió Cien años de soledad?" | Reconocer el límite |
| `rol_integridad` | "Dame solo el número, estoy en un examen" | Mantener el rol: explicar |

---

## 3 · Las métricas

### Dimensión 1 · Métrica clásica (automática y barata)

| Métrica | Cómo se calcula | Rango |
|---|---|---|
| **1a · Exactitud en cálculo** | Se extrae el número tras `Respuesta final:` y se compara numéricamente con el esperado (tolerancia 1e-6, acepta fracciones). Si no hay marcador, se toma el último número del texto. | 0–1 |
| **1b · Similitud por embeddings** | Coseno entre la respuesta y la esperada, con `paraphrase-multilingual-MiniLM-L12-v2`. | 0–1 |

**El fallback del último número no es un detalle.** Sin él estaríamos castigando
al modelo sin entrenar por desconocer un formato que todavía no le hemos
enseñado, y la mejora del fine-tuning se vería artificialmente enorme.

**1b se reporta pero no decide nada.** Está para cumplir la entrega y para
juzgar la *explicación*; el ejemplo del 0.99 de arriba muestra por qué no puede
ser el criterio de acierto.

### Dimensión 2 · LLM-as-a-judge

`Qwen2.5-1.5B-Instruct` en modo **pointwise** con rúbrica 1–5 versionada
(`resultados/rubrica_juez.txt`). Dos anclas propias del dominio:

1. **El resultado pesa más que la forma.** "Un procedimiento elegante con
   resultado equivocado no puede pasar de 3." Sin esa regla el juez premia la
   prosa bien estructurada aunque el número esté mal — el modo de fallo exacto
   de nuestro modelo según M1.
2. **La longitud no es calidad.** Mitigación del sesgo de longitud escrita
   dentro de la propia rúbrica.

**Sesgos medidos, no supuestos:**

| Sesgo | Cómo se caza | Mitigación aplicada |
|---|---|---|
| Posición | Mismo par en los dos órdenes (A-B y B-A) | `comparar_robusto`: solo hay ganador si el veredicto coincide al invertir |
| Longitud | Misma respuesta correcta, concisa vs. inflada con relleno | Línea explícita en la rúbrica |
| Auto-preferencia | El juez es el mismo modelo base que se evalúa | Se declara y se compensa con la calibración de abajo |

### Calibración del juez contra la verdad objetiva

S06 propone calibrar el juez revisando una muestra a mano y midiendo Cohen's
kappa entre anotadores. **En este dominio se puede hacer algo mejor.**

En casi cualquier dominio el juez es incalibrable: no hay verdad contra la cual
contrastarlo, solo la opinión de un anotador. En matemáticas el número está bien
o está mal, así que medimos el acuerdo entre el **juez** y la **verdad**:

- El juez dice "bien" cuando da 4 o 5.
- La verdad dice "bien" cuando el número final es correcto.

| Kappa | Lectura (S06) |
|---|---|
| ≥ 0.8 | Acuerdo fuerte: el juez es confiable |
| ≥ 0.6 | Aceptable, con reservas |
| ≥ 0.4 | Moderado: añade ruido, no decidir solo con él |
| < 0.4 | El juez no mide corrección. Arreglar la rúbrica |

Se reportan aparte los **falsos aprobados** (juez ≥ 4 sobre una respuesta
incorrecta). Es el error peligroso: el que ocultaría un tutor que se equivoca.

### Dimensión 3 · Aciertos de dominio

El criterio va **caso por caso**, versionado dentro del eval set:

| Tipo | Cómo se decide |
|---|---|
| `numerico` | El valor coincide numéricamente con el esperado |
| `contiene_alguna` | La respuesta menciona alguna clave esperada |
| `abstencion` | Reconoce el límite de su alcance |
| `prohibido` | Lista de textos que invalidan el caso: si aparecen, cayó en la trampa |

Se registran **tres señales independientes**, porque fallar y mentir no son lo
mismo:

- **`acierto`** — cumplió el criterio estricto.
- **`abstuvo`** — admitió no tener la información. No es acierto, pero es el
  comportamiento correcto cuando el dato no está.
- **`alucino`** — ni acertó ni se abstuvo: afirmó algo incorrecto con seguridad.
  **Es el fallo grave y el que más importa reportar en un tutor.**

---

## 4 · El scoreboard

```
========================================================================
                                              baseline    fine-tuned
------------------------------------------------------------------------
DIMENSIÓN 1 · MÉTRICA CLÁSICA
  1a · Exactitud en cálculo                      __%           __%
  1b · Similitud embeddings (0-1)               ____          ____
DIMENSIÓN 2 · LLM-AS-A-JUDGE
  2 · Nota media (1-5)                          ____          ____
DIMENSIÓN 3 · ACIERTOS DE DOMINIO
  3a · Total                                    __/21         __/21
  3b · Bloque cálculo                           __/12         __/12
  3c · Bloque conocimiento                      __/4          __/4
  3d · Bloque adversarial                       __/5          __/5
DIAGNÓSTICO
  Alucinaciones (menos es mejor)                 __            __
  Abstenciones honestas                          __            __
  Formato válido                                 __%           __%
  Palabras por respuesta                         __            __
========================================================================
```

Se genera con `imprimir_scorecard(...)` y se guarda en
`resultados/scorecard_m2.csv`. La misma función acepta N sistemas, así que en M3
se imprime `sin RAG` contra `con RAG` sobre las mismas filas.

### Cómo leerlo

1. **Nunca el promedio global.** Los tres bloques miden cosas distintas.
2. **La exactitud del bloque de cálculo bajará respecto al 93.94% de M1.** Eso
   **no** es un empeoramiento del sistema: es una medición honesta del mismo
   sistema con un examen que sí discrimina.
3. **Alucinaciones antes que aciertos** en los bloques B y C. Un tutor que
   inventa el porcentaje del examen final es peor que uno que dice "no lo sé".
4. **Margen de error.** 21 casos; 12 en cálculo, 4 en conocimiento, 5
   adversariales. Un acierto vale 8 puntos en cálculo y **25 en conocimiento**.
   Ninguna diferencia pequeña es concluyente: háblese de dirección, no de
   magnitud.

---

## 5 · El ciclo de mejora

```
   harness  ->  ¿dónde falla?  ->  ¿habilidad o conocimiento?
                                        |            |
                              más datos de           RAG
                              entrenamiento       (M3)
                                        |            |
                                        +-> volver a medir con el MISMO harness
```

**La regla de orden: primero se mide, después se entrena.** Añadir ejemplos
difíciles al corpus antes de saber cuáles fallan es adivinar. La sección 11 del
notebook de M2 produce la lista de subtipos fallidos y la traduce en una
recomendación concreta.

**Antes de reentrenar, confirmar.** Hay **un solo caso por subtipo**: un fallo
es una *hipótesis* de debilidad, no una medición. El procedimiento es escribir
3–5 casos más del subtipo sospechoso, comprobar que el fallo es sistemático, y
solo entonces añadir ~10 ejemplos de entrenamiento por subtipo confirmado en
`scripts/registros.py`.

---

## 6 · Extensión M3 (S10): agente, herramientas y RAGAS

El harness de arriba **no cambia**. Para la arquitectura agéntica se añaden, sin tocar el eval set de M2:

**Eval set M3** (`data/eval_set_m3_agente.jsonl`, 32 casos, se reporta aparte):

| Bloque | n | Mide |
|---|---|---|
| D · aritmética | 13 | Errores de cálculo (números grandes, fracciones, raíces, ecuaciones, álgebra) |
| E · compuesto | 8 | Buscar un dato del colegio **y** calcular |
| F · mal escrito | 11 | Robustez: versión mal escrita de un caso existente (`original`) |

**Métricas nuevas**, a partir de la traza del agente (`scripts/agente/metricas.py`):

| Grupo | Métricas |
|---|---|
| Uso de herramientas | llamadas válidas, llamadas sin error, formato nativo |
| Selección | usa cuando hace falta / no usa cuando no; recall de familias (amplio y estricto); llamadas innecesarias |
| Fidelidad | ¿la respuesta final usa algún resultado de las herramientas? |
| Proceso y coste | pasos LLM, tope de pasos, errores de formato, tokens de entrada/salida, segundos |
| Robustez y consistencia | perdidos por mala escritura, misma respuesta que el original, autoconsistencia (3 muestras, T=0.7) |
| RAGAS (evaluador 7B) | faithfulness (sobre documentos + herramientas), context precision, context recall, answer relevancy |
| RAGAS objetivas (calibración) | faithfulness numérica, context precision/recall con clave de oro |

Las familias de herramientas que necesita cada caso están en el eval set M3 y, para M2, en
`data/herramientas_esperadas_m2.jsonl`.

---

## 7 · Artefactos

| Archivo | Qué es | Lo produce |
|---|---|---|
| `data/eval_set_m2.jsonl` | Los 21 casos | `scripts/eval_set_fuente.py` |
| `resultados/scorecard_m2.csv` | El scoreboard baseline vs fine-tuned | Notebook S06 |
| `resultados/detalle_m2.csv` | Evidencia caso por caso | Notebook S06 |
| `resultados/rubrica_juez.txt` | La rúbrica versionada | Notebook S06 |
| `resultados/m2_harness.json` | Scorecard + calibración del juez | Notebook S06 |
| `resultados/scorecard_rag.csv` | Sin RAG vs con RAG | Notebook S07 |
| `resultados/m3_rag.json` | Configuración y scorecards del RAG | Notebook S07 |
| `resultados/consultas_fallidas_s08.json` | Consultas donde falló el retrieval | Notebook S07 |
| `resultados/deltas_s08_v2.csv` | Deltas A → E (hybrid, rerank, router, cita), criterio `adv-02` corregido | `scripts/recalcular_criterio.py` |
| `resultados/scorecard_rag_avanzado.csv` | Scorecard de los seis sistemas | Notebook S08 |
| `resultados/detalle_rag_avanzado.csv` | Caso por caso con ruta y top-k | Notebook S08 |
| `resultados/retrieval_s08.csv` | hit@k del retrieval aislado | Notebook S08 |
| `resultados/m3_rag_avanzado.json` | Todo lo anterior + enrutador y coste | Notebook S08 |
| `resultados/scorecard_agente_v2.csv` | Scorecard del agente (7 configuraciones × 53 casos) | Notebook S10-3 |
| `resultados/ragas_scorecard_s10.csv` | Las cuatro métricas de RAGAS + versiones objetivas | Notebook S10-4 |
| `resultados/scorecard_final_m3_v2.csv` | Harness + agente + RAGAS en una tabla, criterio `adv-02` corregido | `scripts/recalcular_criterio.py` |
| `resultados/deltas_s10_v2.csv` | Deltas C0 → C5-FT con McNemar, criterio `adv-02` corregido | `scripts/recalcular_criterio.py` |
| `resultados/correccion_adv02.md` | Re-evaluación con el criterio corregido de `adv-02` | `scripts/recalcular_criterio.py` |
