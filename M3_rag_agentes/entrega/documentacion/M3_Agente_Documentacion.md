# Módulo M3 (S10) — Arquitectura agéntica: RAG + ReAct + herramientas

**Universidad EAFIT | SI4006 — Tópicos Especiales y Aplicaciones en IA**
**Sistema base:** `Qwen/Qwen2.5-1.5B-Instruct` + LoRA de M1 · RAG de S08
**Entrega:** 27 de septiembre · sistema RAG completo (≥2 técnicas avanzadas y ≥1 herramienta) +
scorecard propio + RAGAS + lectura honesta

---

## RESUMEN

**La pregunta del experimento:** ¿una arquitectura agéntica basada en RAG + ReAct + herramientas
mejora de verdad al tutor frente al sistema actual? Seguimos la regla de S10: *un agente solo se
justifica si el harness mejora*.

**Qué se construyó:**

| Pieza | Dónde | Verificación |
|---|---|---|
| Tool Registry tipado + 15 herramientas matemáticas exactas (SymPy/Fraction, sin `eval`) | `scripts/agente/registro.py`, `herramientas.py` | 61 tests; resuelven 13/13 casos de aritmética con la llamada correcta |
| Núcleo agéntico: normalizador, parser, bucle ReAct, verificador, traza | `scripts/agente/nucleo.py` | Tests con LLM guionado (errores de formato, de dominio, tope de pasos) |
| Retrieval de S08 como herramienta (`buscar_documentos`) | `scripts/agente/documentos.py` | Ejecutado sobre el índice real |
| Eval set M3: aritmética (13), compuesto (8), mal escrito (11) | `data/eval_set_m3_agente.jsonl` | Aritmética comprobada, datos contrastados con el corpus, sin contaminación |
| 199 trayectorias de uso de herramientas | `data/trayectorias_herramientas.jsonl` | Cada llamada ejecutada y verificada; 251 re-verificadas en el notebook |
| RAGAS adaptado + versiones objetivas para calibrar | `scripts/agente/ragas_tutor.py` | Tests de las métricas objetivas |
| 4 notebooks generados | `S10_*.ipynb` | Ejecutados completos en Lightning (GPU), 25 y 26 de septiembre (ver §7) |

**Resultado principal:** con few-shot, un modelo de 1.5B **no** llama a las herramientas
(2% a 19% de las veces que hacen falta); entrenado con las trayectorias, lo hace en el 96%.
El agente entrenado es el mejor sistema global (33/53) y el único cuyos números salen de
fuentes verificables (faithfulness numérica 0.77 contra 0.32–0.48), pero cuesta 19× los
tokens de C2 y no supera a C1 en el eval set de M2. La sección 7 tiene el detalle y el
veredicto; la 6, lo medido sin GPU.

---

# 1 · Diagnóstico de partida

| Componente existente | Estado | Cómo se reutiliza |
|---|---|---|
| Qwen2.5-1.5B + LoRA M1 | Formato 100%; cálculo M2 10/12 | Generador de todas las configuraciones |
| Eval set M2 (21) + harness de 3 dimensiones | Vara fija del semestre | **Intacto**, pasa por todas las configuraciones |
| RAG S08 (híbrida + reranker + enrutador) | Conocimiento 3/4 | C2 (contexto fijo) y herramienta en C5 |
| Generación de notebooks por scripts | Patrón del proyecto | Mismo patrón para S10 |
| `registros.py` con pasos y `check` verificable | 165 ejemplos | Fuente automática de trayectorias |

**Limitaciones que condicionaron el diseño:**

1. **Los dos fallos de cálculo de M2 son de planteamiento, no de aritmética** (`dif-07`, `dif-10`). Una
   calculadora casi no tiene margen ahí → hacía falta un bloque de aritmética exigente.
2. **Ningún caso combinaba dato + cálculo** → bloque compuesto.
3. **Preguntas mal escritas** (pedido explícito) → bloque F y normalizador.
4. **El juez de 1.5B no es fiable** (kappa 0.286) → evaluador de 7B para RAGAS, sin tocar el juez de M2.
5. **La LoRA de M1 fuerza "Paso 1:"**, lo que podría bloquear `<tool_call>` → se **mide** (sondeo), no se supone.

---

# 2 · Arquitectura

```
Usuario
  ↓
Normalizador ─────────── reglas deterministas + reescritura LLM (solo si parece informal)
  │                       salvaguarda: si cambian los números, se descarta la corrección
  ↓
Agente / LLM ─────────── Pensamiento: planifica qué datos necesita y de dónde salen
  ↓
Selección de herramienta ─ <tool_call>{"name", "arguments"}</tool_call>  (formato nativo de Qwen2.5)
  ↓
Parser + Validador ───── JSON / JSON suelto recuperado / error de formato → vuelve al modelo
  ↓
Tool Registry ────────── esquema derivado de tipos + docstring; conversión de tipos
  ↓
Ejecución ────────────── nunca lanza: {"ok": false, "error": ...} también es observación
  ↓
Resultado ────────────── <tool_response>{"ok", "resultado", "valor"}</tool_response>
  ↓
Agente interpreta ───── ¿otra herramienta o responder? (tope de pasos)
  ↓
Verificador ──────────── ¿la respuesta final usa algún resultado de las herramientas? (1 revisión)
  ↓
Respuesta final + traza completa
```

**Separación de responsabilidades:** el LLM solo propone; el parser solo traduce; el registro solo
valida y ejecuta; `AgenteTutor` solo orquesta y no conoce ninguna herramienta concreta.

**Extensibilidad (comprobada en el notebook 1 §6):** una herramienta nueva es una función con tipos y
docstring decorada con `@registro.herramienta(familia=...)`. El agente la usa sin cambiar el núcleo.

### Herramientas

| Familia | Herramientas |
|---|---|
| aritmética | `sumar`, `restar`, `multiplicar`, `dividir` (con redondeo), `evaluar_expresion` (comodín) |
| fracciones | `operar_fracciones`, `simplificar_fraccion` |
| potencias y raíces | `potencia`, `raiz` |
| porcentajes | `porcentaje_de`, `aplicar_porcentaje`, `valor_antes_de_porcentaje` |
| estadística | `promedio_ponderado` |
| álgebra | `resolver_ecuacion`, `operar_expresion` (simplificar / expandir / factorizar) |
| documentos | `buscar_documentos` (retriever S08) |

---

# 3 · Decisiones técnicas

| # | Decisión | Alternativa descartada | Motivo |
|---|---|---|---|
| D1 | **Tool calling nativo de Qwen2.5** (`<tool_call>`) | JSON inventado (lab S10) | El modelo se preentrenó con ese formato; comprobado con el tokenizer que la plantilla renderiza pensamiento + llamada + `<tool_response>` y que los tokens sobreviven al decodificar |
| D2 | **Esquemas derivados de firma + docstring** | Escribirlos a mano | Descripción y código no pueden divergir |
| D3 | **Aritmética exacta** (Fraction, SymPy racional) | `float`, `eval` | Una calculadora con errores de coma flotante no ahorra errores; `eval` es inseguro. Lista blanca de símbolos |
| D4 | **Las herramientas nunca lanzan**: los errores son observaciones | Excepciones | `dividir(15, 0)` debe llegar al modelo para que explique que no está definido |
| D5 | **Parser tolerante con trazabilidad** (`nativo` / `recuperado` / error) | Estricto | Un 1.5B a veces olvida las etiquetas; se recupera y **se mide** cuántas veces pasa |
| D6 | **Verificador de una sola revisión** | Ninguno / en bucle | Ataca el fallo "llamó bien e ignoró el resultado" sin riesgo de bucles |
| D7 | **Normalizador en dos capas con salvaguarda de números** | Solo reglas / solo LLM | Las reglas son auditables; el LLM cubre faltas imprevisibles; una corrección que cambia un dato es peor que la falta |
| D8 | **Eval set M2 intacto + eval set M3 aparte** | Ampliar el de M2 | Mantener comparabilidad con M1, M2, S07 y S08 |
| D9 | **Anotación de herramientas por familias**, con comodín | Nombre exacto | Varias herramientas pueden ser correctas; se reportan recall amplio y estricto |
| D10 | **Adaptador de C3–C5 elegido por sondeo** en validación | Suponer M1 o base | Riesgo real (D-diagnóstico 5); se decide con datos que no son del eval set |
| D11 | **C5-FT parte del modelo base**, no de la LoRA de M1 | Continuar M1 | Las trayectorias ya terminan en formato "Paso N"; evita el conflicto de formatos |
| D12 | **Pérdida solo sobre turnos del asistente**, máscara por offsets de caracteres | Todo el texto | No aprender a "predecir" observaciones (= inventar resultados); los offsets evitan desalinear la máscara |
| D13 | **Evaluador de RAGAS de 7B (4 bits)**; juez de M2 intacto | 1.5B para todo | Kappa 0.286 del 1.5B; la Dimensión 2 sigue siendo comparable |
| D14 | **Faithfulness contra documentos + observaciones** | Solo documentos | En un agente, lo observado incluye las herramientas; un cálculo "de cabeza" no es fiel |
| D15 | **Versión objetiva de cada métrica RAGAS** | Confiar en el evaluador | Permite calibrarlo, como se calibró el juez en M2 |
| D16 | **Persistencia incremental** (JSONL por respuesta) | Todo en memoria | ~1 h de T4; Colab se desconecta |
| D17 | **Generación y evaluación en notebooks separados** | Uno solo | No caben 1.5B + juez + 7B en la T4; RAGAS se repite sin regenerar |

---

# 4 · Configuraciones del experimento

| Config. | Añade | Adaptador | Contexto | Herramientas | Rondas | Few-shot |
|---|---|---|---|---|---|---|
| C0 | Modelo base | — | — | — | — | — |
| C1 | + LoRA M1 | m1 | — | — | — | — |
| C2 | + RAG (S08-C) | m1 | fijo (top-3) | — | — | — |
| C3 | + herramientas de cálculo | sondeo | fijo | 15 | 1 | 3 |
| C4 | + ReAct | sondeo | fijo | 15 | 6 | 3 |
| C5 | + retrieval como herramienta, planificación, verificador, normalizador | sondeo | a demanda | 16 | 6 | 3 |
| C5-FT | C5 + adaptador entrenado | agente | a demanda | 16 | 6 | 3 |

C0, C1 y C2 reproducen sistemas ya medidos; el notebook comprueba que coinciden con
`m2_harness.json` y `m3_rag_avanzado.json`.

---

# 5 · Estrategia de evaluación

### Datos

| Conjunto | n | Mide |
|---|---|---|
| M2 · cálculo / conocimiento / adversarial | 12 / 4 / 5 | Comparabilidad con todo el semestre |
| M3 · aritmética | 13 | Errores de **cálculo** (donde las herramientas deben ganar) |
| M3 · compuesto | 8 | Buscar **y** calcular |
| M3 · mal escrito | 11 | Robustez y consistencia (cada caso apunta a su `original`) |

### Métricas (qué pidió el proyecto → cómo se mide)

| Pedido | Métrica | Fuente |
|---|---|---|
| RAGAS | faithfulness, context precision, context recall, answer relevancy (+ versiones objetivas) | Notebook 4 |
| Exactitud matemática | Exactitud en cálculo (M2), aciertos por bloque (M3) | Harness |
| Uso correcto de herramientas | % llamadas válidas, % sin error, % formato nativo | Traza |
| Selección correcta | Recall de familias (amplio y estricto), llamadas innecesarias, *usa cuando hace falta* / *no usa cuando no* | Traza + anotación |
| Cantidad de pasos | Pasos LLM, llamadas por pregunta, tope de pasos | Traza |
| Errores | Errores de formato, de ejecución, de dominio, alucinaciones | Traza + harness |
| Calidad de la explicación | Juez 1–5 (rúbrica de M2), formato válido | Harness |
| Consistencia | Misma respuesta mal escrito vs. original; autoconsistencia con 3 muestras (T=0.7) | Notebook 3 §13 |
| Coste | Tokens de entrada y salida, segundos por pregunta | Contador |
| Significancia | McNemar exacto paso a paso sobre 53 casos | Notebook 3 §12.2 |

---

# 6 · Cómo se enseña al modelo a usar las herramientas

| Mecanismo | Uso | Justificación |
|---|---|---|
| Prompting | Base de C3–C5 (`scripts/agente/prompts.py`) | Necesario pero insuficiente en un 1.5B (zero-shot de S08: 67%) |
| Tool calling nativo | Formato de C3–C5 | Preentrenado en Qwen2.5 |
| Few-shot | 3 trayectorias completas por configuración, elegidas **por criterio** del split train | Enseña cuándo **sí** y cuándo **no** |
| Fine-tuning LoRA | C5-FT (notebook 2) | Compara con datos entrenar vs. mostrar |

### Trayectorias: `problema → decisión → herramienta → resultado → respuesta`

| Origen | n | Contenido |
|---|---|---|
| Corpus de M1 (automáticas) | 165 | Cada paso con operación → llamada ejecutada y verificada contra el corpus; si ningún paso sirve, se usa el `check` |
| Manuales | 34 | Documentos (6), compuestos (6), **sin herramienta** (6), errores de dominio (3), mal escritas (5), álgebra (3), porcentajes/fracciones (5) |

Particiones: 157 train / 42 validación. Las 15 herramientas + `buscar_documentos` tienen al menos un
ejemplo. El generador detectó tres preguntas manuales demasiado parecidas a casos de evaluación
(Jaccard ≥ 0.6) y se reescribieron.

### Medido en local (sin GPU)

- **Techo de las herramientas:** 13/13 casos de aritmética resueltos con la llamada correcta.
- **Coste de los esquemas:** system + 16 esquemas = **~2.700 tokens**; trayectorias de 2.760 a 3.360
  tokens. Con el prompt, el few-shot y 2–3 pasos, **C5 envía del orden de 7.500 tokens de entrada por
  pregunta, frente a ~550 de C2** (medido con el tokenizer y los prompts reales). Es el coste que hay
  que poner frente a cualquier mejora.
- **Máscara de pérdida:** la plantilla es estable por prefijos en las 199 trayectorias; ninguna se
  trunca con `MAX_LEN = 4096`.
- **Normalizador:** las reglas conservan los números en los 11 casos mal escritos y corrigen
  `3847 x 296`, `17/24 mas 11/36`, `dividido en`, `pa`, `ai`, `x periodo`, `??`. Una "corrección" que
  cambia 296 por 269 se descarta.

---

# 7 · Resultados

Ejecutado en Lightning (GPU) los días 25 y 26 de septiembre de 2026. Las cifras del
harness son las **corregidas** por `scripts/recalcular_criterio.py`: la corrida usó
la versión antigua del eval set (el notebook lo avisó con `SHA coincide: False`) y
`adv-02` se re-evaluó después sobre las respuestas ya generadas. Cambió **un solo
caso** (C1), y el detalle está en `resultados/correccion_adv02.md`.

### 7.1 Scorecard del agente

`resultados/scorecard_agente_v2.csv` · `detalle_s10_v2.csv` · `deltas_s10_v2.csv` · `ragas_scorecard_s10.csv` · `scorecard_final_m3_v2.csv`

| | C0 | C1 | C2 | C3 | C4 | C5 | C5-FT |
|---|---|---|---|---|---|---|---|
| **M2 · aciertos /21** | 9 | **14** | 13 | 13 | **14** | 12 | 13 |
| M2 · conocimiento /4 | 0 | 0 | 3 | 3 | 3 | 3 | 3 |
| M2 · adversarial /5 | 3 | 4 | 3 | 3 | 4 | 2 | 4 |
| M2 · alucinaciones | 12 | 7 | **5** | 8 | 7 | 9 | 8 |
| **M3 · aritmética /13** | 5 | 8 | 5 | 7 | 4 | 7 | **10** |
| **M3 · compuesto /8** | 0 | 0 | 2 | **5** | **5** | **5** | 3 |
| M3 · mal escrito /11 | 4 | 5 | 5 | 6 | 5 | 6 | **7** |
| **Total /53** | 18 | 27 | 25 | 31 | 28 | 30 | **33** |
| Usa herramienta cuando hace falta | — | — | — | 2% | 0% | 19% | **96%** |
| Selección correcta | 9% | 9% | 9% | 11% | 9% | 25% | **58%** |
| Fidelidad a la herramienta | — | — | — | 100% | — | 100% | 84% |
| Pasos LLM / pregunta | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.3 | 2.7 |
| Tokens entrada / pregunta | 75 | 75 | 539 | 3 332 | 3 416 | 4 842 | **10 136** |
| Segundos / pregunta | 9.2 | 5.5 | 4.7 | 3.6 | 3.4 | 5.0 | **14.1** |
| RAGAS · faithfulness | — | — | 0.65 | 0.69 | 0.69 | 0.77 † | 0.73 |
| RAGAS · faithfulness numérica | 0.36 | 0.32 | 0.45 | 0.48 | 0.48 | 0.40 | **0.77** |
| RAGAS · context recall (oro) | — | — | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| RAGAS · answer relevancy | 0.68 | **0.71** | 0.67 | 0.65 | 0.67 | 0.70 | 0.66 |

† n = 10: C5 casi nunca observó nada, así que su faithfulness no es comparable con
las demás columnas (las otras van sobre 48–53 casos).

**Deltas paso a paso** (53 casos, McNemar exacto):

| Paso | Δ total | Δ aritmética | Δ compuesto | mejora/empeora | p |
|---|---|---|---|---|---|
| C0 → C1 | +9 | +3 | 0 | 11/2 | **0.022** |
| C1 → C2 | −2 | −3 | +2 | 11/13 | 0.839 |
| C2 → C3 | +6 | +2 | +3 | 15/9 | 0.307 |
| C3 → C4 | −3 | −3 | 0 | 6/9 | 0.607 |
| C4 → C5 | +2 | +3 | 0 | 10/8 | 0.815 |
| C5 → C5-FT | +3 | +3 | −2 | 12/9 | 0.664 |
| C1 → C5-FT | +6 | +2 | +3 | 13/7 | 0.263 |

### 7.2 Predicciones contra resultados

| Predicción | Resultado | ¿Acertada? |
|---|---|---|
| M2 · cálculo: C3–C5 ≈ C1 | C1 10/12, C3–C5 7/12 | **No**: no solo no mejoran, empeoran |
| M2 · conocimiento: C2 ≈ C5 > C1 | 3/4 contra 0/4 | **Sí** |
| M3 · aritmética: C3–C5 ≫ C1 | C1 8/13; C3 y C5 7/13; **C5-FT 10/13** | **Solo con fine-tuning** |
| M3 · compuesto: C5 > C2–C4 | C3, C4 y C5 empatan en 5/8 | **No**: el contexto bastaba |
| M3 · mal escrito: C5 ≥ resto | C5 6/11, C5-FT 7/11 | **Sí**, por poco |
| Coste: C5 ≈ 10× tokens de C2 | C5 9×, C5-FT 19× | **Sí** |

### 7.3 Lectura honesta

**1 · Sin entrenamiento, la arquitectura agéntica es un cascarón.** C3 llama
herramientas en el 2% de los casos que las necesitan, C4 en el 0% y C5 en el 19%. Un
modelo de 1.5B no aprende tool calling por prompting ni con few-shot, por mucho que
el formato sea el nativo de Qwen y los ejemplos sean correctos. Las ganancias de C3
y C4 en el bloque compuesto (0 → 5 de 8) vienen del **contexto inyectado**, no de
las herramientas: es RAG, no agencia.

**2 · Con entrenamiento, sí.** El notebook 2 lo mide sobre 42 trayectorias de
validación que el modelo nunca vio:

| Decisión | Base | C5-FT |
|---|---|---|
| ¿Usar herramienta? | 14% | **95%** |
| ¿Cuál? | 2.5% | **88%** |
| ¿Llamada válida? | 7.5% | **100%** |
| ¿Resultado correcto? | 5% | **65%** |

Y se traslada al eval set: C5-FT es el mejor sistema global (33/53), el mejor en
aritmética (10/13) y el que menos alucina en M3 (12 contra 17 de C1).

**3 · El agente no mejora la vara fija.** En el eval set de M2 nadie supera a C1
(14/21). El valor del agente aparece **solo** en los bloques nuevos, que existen
precisamente porque M2 no podía medir esto. Quien lea solo el scorecard de M2
concluirá, con razón, que el agente no aporta.

**4 · C5-FT sobre-usa las herramientas.** Se abstiene de llamarlas solo en el 20% de
los casos donde no hacían falta (C5: 80%), y busca documentos menos que C5: 8 casos
sin buscar contra 4. Eso le cuesta el bloque compuesto (3/8 contra 5/8). El
entrenamiento enseñó *cómo* llamar, no *cuándo no* hacerlo: las 157 trayectorias
tienen 151 con al menos una llamada.

**5 · RAGAS confirma que las herramientas cambian la naturaleza del acierto.** De los
aciertos de C1, **0 de 24** tienen todos sus números respaldados por una fuente
verificable; en C5-FT son **14 de 29**, y su faithfulness numérica es la más alta
(0.77 contra 0.40–0.48 de C2–C5 y 0.32–0.36 de C0–C1). Sin herramientas el tutor acierta calculando de
cabeza, que es exactamente la alerta que S10 describe.

**6 · Los fallos de conocimiento cambiaron de naturaleza.** En C2–C4 son 6–8 casos de
*"el dato llegó y el modelo no lo usó"* (generación). En C5 y C5-FT son 4 y 8 casos
de *"no buscó"* (decisión). El retrieval nunca falla: context recall de oro = 1.00 en
todas las configuraciones con contexto.

**7 · El coste.** C5-FT gasta 19× los tokens de entrada de C2 y triplica la latencia
(14.1 s contra 4.7 s por pregunta) para ganar 8 casos de 53. Cada paso reenvía los
~2 700 tokens de los esquemas más el few-shot. Un tutor en producción con este
diseño necesitaría cachear el prompt de sistema o recortar el catálogo.

**8 · Autoconsistencia.** Con muestreo (T = 0.7, 3 muestras) sobre el bloque de
aritmética: C5-FT es el más estable (46% de unanimidad, exactitud media 0.69) y C5 el
más inestable (15%, 0.33). El agente sin entrenar no solo acierta menos: acierta por
azar de la decodificación.

**9 · La calibración del evaluador, declarada.** Kappa 0.72 en relevancia de chunks
(aceptable según S06), pero Spearman 0.09 entre su faithfulness y la numérica
objetiva, y kappa 0 en context recall porque las 15 consultas documentales tienen
recall de oro = 1 y la métrica no tiene varianza que medir. Por eso las conclusiones
de fundamento (punto 5) se apoyan en la **faithfulness numérica**, que es objetiva, y
no en el juicio del 7B.

### 7.4 Veredicto

**Para este tutor, la arquitectura agéntica no se justifica todavía; el enrutador de
S08 sí.** El mejor sistema por aciertos absolutos sobre el eval set de M2 es
`D · +router` de S08 (18/21, 2 alucinaciones, prompt de 193 tokens). El agente
entrenado gana en los bloques nuevos —aritmética exigente y robustez a la mala
escritura— y en la **verificabilidad** de sus números, pero paga 19× en tokens y
pierde en el bloque compuesto por no decidir buscar.

La recomendación operativa es combinarlos, no elegir: el enrutador de S08 para
decidir **si consultar documentos**, y las herramientas de S10 —con el adaptador
entrenado— para la vía de cálculo. Ninguna de las dos configuraciones evaluadas hace
las dos cosas bien a la vez.

---

# 8 · Limitaciones declaradas

- **Muestras pequeñas:** 53 casos; bloques de 4 a 13. Se reporta dirección + McNemar, no magnitud.
- **Eval set M3 escrito por el mismo equipo que diseñó las herramientas**, con riesgo de favorecerlas.
  Mitigación: la prueba de techo separa "la herramienta sirve" de "el modelo la usa".
- **Trayectorias sesgadas por el corpus de M1:** muchas operaciones de un paso, pocos porcentajes.
- **Corpus documental ficticio** (heredado de S07).
- **Few-shot fijo:** no se optimizó (DSPy queda como trabajo futuro).
- **El sondeo de adaptador usa 42 trayectorias de validación**; en empate se prefiere el modelo base.

---

# 9 · Artefactos

| Archivo | Contenido |
|---|---|
| `scripts/agente/` | Núcleo agéntico (registro, herramientas, núcleo, documentos, prompts, métricas, RAGAS) |
| `scripts/tests/test_agente.py` | 61 tests: `python -m pytest scripts/tests -q` |
| `scripts/eval_set_m3_fuente.py` | Genera y valida el eval set M3 y la anotación de M2 |
| `scripts/trayectorias_fuente.py` | Genera y valida las trayectorias |
| `scripts/nb_herramientas.py`, `nb_finetuning_agente.py`, `nb_agente.py`, `nb_ragas.py`, `nb_comun_agente.py` | Generadores de los notebooks |
| `data/eval_set_m3_agente.jsonl`, `data/herramientas_esperadas_m2.jsonl`, `data/trayectorias_herramientas.jsonl` | Datos |
| `S10_Lab_Herramientas_Tutor_Matematicas.ipynb` | 1 · herramientas, registro, normalizador, trayectorias |
| `S10_FineTuning_Agente_Herramientas.ipynb` | 2 · LoRA del agente → `adaptadores/qwen-lora-agente` |
| `S10_Lab_Agente_ReAct_Tutor_Matematicas.ipynb` | 3 · C0–C5-FT por el harness → `respuestas_s10.jsonl`, `detalle_s10.csv`, `scorecard_agente.csv`, `scorecard_m2_s10.csv`, `deltas_s10.csv`, `m3_agente.json` (cifras de la corrida, antes de corregir `adv-02`) |
| `S10_Evaluacion_RAGAS_Tutor_Matematicas.ipynb` | 4 · `ragas_s10.jsonl`, `ragas_scorecard_s10.csv`, `ragas_s10.json`, `scorecard_final_m3.csv` |
| `scripts/recalcular_criterio.py` | Re-evalúa `adv-02` sin GPU → **archivos válidos**: `detalle_s10_v2.csv`, `scorecard_agente_v2.csv`, `scorecard_m2_s10_v2.csv`, `scorecard_final_m3_v2.csv`, `deltas_s10_v2.csv`, `correccion_adv02.md` |
