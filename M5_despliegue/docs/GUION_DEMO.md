# Guion de la demo · 15 minutos

Estructura pedida por S15. **Los huecos marcados `[PENDIENTE]` son cosas que
todavía no existen: no se rellenan hasta ejecutarlas.**

> La nota es individual: cada integrante debe poder explicar cualquier módulo,
> no solo el que le tocó.

## 1 · Problema y usuario · 2 min

Un estudiante de secundaria resuelve un ejercicio, le da mal y no sabe **dónde**
se equivocó. Un corrector que solo mira el resultado final no se lo dice, y
rehacer el ejercicio no le enseña a encontrar su propio error.

El tutor explica procedimientos y, dada la foto de la hoja, intenta señalar el
paso. Público: estudiantes de 6.º a 9.º y sus docentes.

## 2 · El sistema, de M1 a M5 · 3 min

| Módulo | Qué añadió | Resultado medido |
|---|---|---|
| **M1** | Qwen2.5-1.5B + LoRA, frente a BETO y FLAN-T5 | el fine-tuning enseñó a **explicar**, no a calcular |
| **M2** | harness de 3 dimensiones, 21 casos | criterio de dominio en lugar de similitud |
| **M3** | RAG + agente ReAct + 15 herramientas exactas | C5-FT 13/21; el enrutador llevó S08 de 14 a 18/21 |
| **M4** | visión: foto → transcripción → diagnóstico | **sin mejora de exactitud**; 3/9 en desarrollo |
| **M5** | despliegue, cuantización, observabilidad | `[PENDIENTE]` la medición |

**Qué se sirve en la demo, y por qué justo eso.** `TUTOR_CONFIG=s08d`: LoRA de
M1 + RAG híbrido con reranker + enrutador. **Sin herramientas, sin verificador y
sin visión**, porque esa es la combinación que marcó 18/21 y ninguna que añada
piezas por encima se ha medido. Las variantes existen y se pueden montar, pero la
interfaz las marca «EXPERIMENTAL, sin medir» y no se presentan como resultado.

Diagrama de lo que corre: pregunta → enrutador (margen de centroides, nº de
números, puntaje del reranker) → si es consulta: híbrida BM25+denso → RRF →
cross-encoder → 3 pasajes; si es problema: sin contexto → Qwen2.5-1.5B + LoRA de
M1 → respuesta.

La vía visual y las herramientas quedan como **lo que se explica, no lo que se
ejecuta**. Si alguien lo pide, se cambia `TUTOR_CONFIG` en vivo y la cabecera
dirá que esa variante no está medida: eso es parte de la demostración, no un
problema que esconder.

**Si algo falla en vivo, la pantalla lo dice.** La app distingue la configuración
solicitada de la que quedó funcionando: si falta el adaptador, el reranker o el
enrutador, sale `🔴 MODO DEGRADADO · resultado de esta ejecución: NO MEDIDO` con
la lista de qué falta. Enseñar eso es mejor demo que esconderlo: es la diferencia
entre un sistema que sabe lo que está sirviendo y uno que recita una cifra.
Antes de empezar, `curl $URL/salud` y comprobar que `degradado` es `false`.

## 3 · Tres casos en vivo · 4 min

| Caso | Qué se enseña | Guion |
|---|---|---|
| **Sencillo** | la vía de problema: el enrutador decide NO buscar | "¿Cuánto es 3/4 + 1/8?" → `problema`, responde sin contexto |
| **Difícil** | la vía de consulta: busca y cita | "¿Cuál es la nota mínima aprobatoria?" → `consulta`, 3 pasajes del SIEE |
| **Adversarial** | la abstención | "Explica por qué 9 es un número primo" → debe negar la premisa |

Las rutas de estos tres casos **están verificadas**: `dif-*` → `problema`,
`con-03` → `consulta`, `adv-*` → `problema` salvo `adv-04`, igual que en el
notebook (`M5_despliegue/evaluacion/verificacion_retrieval.json`). Lo que no está
garantizado es la respuesta: 18/21 significa que falla 3 de 21.

**Lo que NO hay que decir en el caso sencillo:** que usa `operar_fracciones`. En
`s08d` no hay herramientas; la aritmética la hace el modelo. Prometer aritmética
exacta aquí sería prometer una pieza que esta configuración no monta.

Ensáyalos antes: si el Space está frío, la primera respuesta tarda.

## 4 · Scorecard y eficiencia · 3 min

Frente a las líneas base de M1–M4, con el método y el conjunto a la vista:

| Métrica | Línea base | Lo que se sirve | Medido en M5 | Conjunto |
|---|---|---|---|---|
| Aciertos de dominio | 14/21 (`solo_lora`, M3/S08) | 18/21 (`s08d`, S08) | `[PENDIENTE]` reproducirlo | eval set M2, 21 casos |
| Alucinaciones | 7 (`solo_lora`) | 2 (`s08d`) | `[PENDIENTE]` | ídem |
| Latencia p50 | 5,5 s/caso (C1, M3) | n/d: S08 no reportó segundos | `[PENDIENTE]` | ídem |
| Memoria pico | `[PENDIENTE]` fp16 | — | `[PENDIENTE]` int4 | — |
| Tokens/s | `[PENDIENTE]` | — | `[PENDIENTE]` | — |

La columna «lo que se sirve» son cifras **de S08**, no de M5: la app monta esa
configuración, pero la corrida que la reproduce en este entorno todavía no se ha
hecho. Se rellena con `evaluacion/ablaciones.py` (filas 0 a 2 primero: son los
controles) y `evaluacion/medir_eficiencia.py`.

La latencia de `s08d` no tiene línea base comparable: `scorecard_rag_avanzado_v2.csv`
no reporta segundos por caso. Los 5,5 s son de C1 en el otro scorecard, que es la
misma configuración que `solo_lora`, no la que se sirve.

**Al presentar, decir el n**: con 21 casos y 3 repeticiones son 63 mediciones, y
el p95 es una cota superior ruidosa, no un compromiso de servicio.

**Lo que ya está comprobado sin GPU**, y conviene enseñarlo porque es la clase de
cosa que suelen preguntar: el enrutador servido reproduce el de S08-D con los
mismos pesos, la misma validación leave-one-out (0,9697) y las mismas 21 rutas
(`evaluacion/verificar_retrieval.py`). Eso no es un acierto medido; es que la
pieza desplegada es la medida —y durante esta fase se encontraron dos formas de
que no lo fuera.

## 5 · Un fallo conocido, límites y ética · 2 min

**El fallo que conviene enseñar**, porque está medido: en M4 el tutor **no
revisa, resuelve**. Ante una hoja con `24 × 13 = 302` respondió `24 × 13 = 312`
—el resultado correcto— sin avisar de que el estudiante se había equivocado. Lo
hizo en los cinco casos con error.

Y hay algo más fino: cambiar un solo carácter del prompt (`Paso 2` → `Paso N`)
movió la conclusión en 3 de 9 casos **sin que la hoja cambiara**. El modelo no
compara el valor calculado con el escrito; emite un índice casi arbitrario.

**Límites.** 1,5B parámetros; 21 casos en el eval set de M2 y 12 en el visual de
M4, que **ya se gastaron** en tres corridas exploratorias (4/12, 1/12, 2/12) y hoy
sirven de regresión, no de evaluación ciega; el corpus del colegio es sintético; la
detección de hojas ilegibles está sin validar con letra manuscrita real. Para
afirmar una mejora de la vía visual hacen falta **hojas nuevas**, y están
especificadas en `m4/eval_set/HOJAS_M5.md`.

**Ética.** No es un docente ni emite diagnósticos de ninguna clase. Un tutor que
**inventa** un error donde no lo hay es peor que uno que no lo encuentra: le dice
al estudiante que está mal cuando está bien. Por eso el bloque `sin_error` se
reporta antes que el total. No se registra el texto del usuario por defecto.

## 6 · Evidencia · 1 min

| Qué | Estado |
|---|---|
| URL del Space | `[PENDIENTE]` — no publicado |
| Captura del health check | `[PENDIENTE]` |
| Panel de W&B | `[PENDIENTE]` — sin credencial configurada |
| Scorecards de M1–M4 | **existen**: `M3_rag_agentes/resultados/`, `m4/` |
| Corridas de M4 archivadas | **existen**: `m4/corridas/`, 3 corridas |

No se inventa ninguna captura ni URL.
