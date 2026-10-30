# Revisión M4: integración visual y evaluación

**Repositorio:** `Emmanuelcasta/TopicosdeIA` · **Commit revisado:** [`9c0c0df`](https://github.com/Emmanuelcasta/TopicosdeIA/tree/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd) («consolida análisis y evaluación visual», rama `m4-vision`, 25-oct 21:38, hora de Colombia). `main` sigue en M3 (`c91ed6e`).

Esta revisión es técnica. La calificación se comunica por canal privado. Las sugerencias no son requisitos retroactivos. Los límites que ustedes ya declararon en `m4/README.md` §3 y en `CONCLUSIONES_M4.md` se tratan como sugerencias, no como fallas nuevas.

El eje de esta revisión es la pregunta que ustedes mismos dejaron abierta: **por qué «el modelo de visión no funcionó»**. La respuesta corta es que el lector sí lee. Lo que falló fue cómo se decide si una lectura es confiable y qué hace el sistema después. Las secciones 3 y 4 separan **qué falló y por qué** de **cómo se arregla**. El arreglo está implementado y probado en la rama de CODE_PR.

---

## 1. Cómo se verificó

Niveles: **L** = lectura · **R** = recálculo desde sus artefactos · **E** = ejecución real · **D** = prueba con dobles.

Ejecuciones en otro hardware: Apple M4 Max, MPS, fp16, con transformers 4.57.6 y 5.20.2. Solo se cambió `cuda → mps`. Prompts, parámetros de generación y post-procesamiento son los suyos.

| Qué | Cómo | Nivel | Resultado |
|---|---|---|---|
| `ver()` / `leer_dos_veces()` | Su código, Qwen2-VL-2B, las 11 fotos | E | **11/11 transcripciones idénticas** a `m4/lecturas_m4.json`. Lectura ≈ 2,9 s por foto (dos pasadas) |
| Salvaguarda con su caché | `CODIGO_TRANSCRIBIR` tal cual, con el `lecturas_m4.json` versionado | E | Solo 3/11 fotos tienen confianza. vis-10 queda «ok»: el mismo FALSO NEGATIVO de su corrida T4 (M4-01) |
| Salvaguarda con la caché corregida | Rama de CODE_PR, C1 | E | 11/11 con confianza. **Solo vis-10 queda NO FIABLE** |
| Calibración impresa (`eval_set/calibracion/`) | Ejecutada por primera vez | E | Legibles marcadas: 0/6. Detecta 1/6 de las leves y 5/6 de las fuertes. Como referencia para fotos manuscritas da 5 falsos positivos (M4-02) |
| Una sola foto de punta a punta | `revisar_foto` (C1), con el tutor Qwen2.5-1.5B **base** (sin sus adaptadores, que están en Drive). Referencia provisional: las otras fotos nítidas del eval set, sin la evaluada ni vis-10; solo prueba el mecanismo | E | vis-01 y vis-07 llegan a diagnóstico. vis-10 y vis-01 desenfocada (radio 4) responden «no pude leer tu hoja». Unos 8–9 s por foto |
| VLM + tutor residentes a la vez | Medido en MPS fp16 | E | 7,6 GB (el VLM solo ocupa 4,4 GB) |
| Versión fijada | transformers 5.20.2 con `size=` y `dtype=` | E | Mismo `image_grid_thw` ([1, 54, 74]), lecturas idénticas, 0 avisos de deprecación |
| Fidelidad de números | Todos los números de los pasos esperados, en orden, frente a sus lecturas | R | 8/9 hojas con procedimiento. vis-09 lee «12G» por «126» (M4-03) |
| Lector alternativo | Qwen2.5-VL-3B con su mismo código y solo el modelo cambiado (bf16; en fp16 generó NaN en MPS) y GOT-OCR-2.0 (OCR con salida LaTeX, fp16) | E | El 3B detecta mejor la hoja ilegible, pero en vis-06 corrige el error del estudiante. GOT-OCR es rápido, pero aplana las fracciones y transcribe sin decir si pudo leer (M4-07) |
| Delta y scorecard | Recálculo desde `scorecard_m4.csv` | R | Las cifras del delta coinciden |
| Pruebas | `pytest scripts/tests` | E | Sus 238 pruebas pasan. Con la rama de CODE_PR pasan 251 |
| — | Corrida en T4, tutor con sus adaptadores (Drive), juez por API | — | **No re-ejecutados** |

Las diferencias de hardware (MPS fp16 frente a T4) no se consideran defectos de la entrega. Las latencias medidas aquí son un orden de magnitud.

## 2. Fortalezas comprobadas

1. **La foto es la entrada natural del producto, con un argumento propio.** Si el estudiante transcribe su hoja, corrige el error sin darse cuenta. Por eso el lector debe copiar el error, no arreglarlo. Diseñaron el eval set alrededor de ese riesgo: errores de cuenta, errores con resultado final correcto, hojas sin error y tres adversariales.
2. **El lector de visión funciona y es reproducible.** En otro hardware salieron las mismas 11 transcripciones y las mismas confianzas. `json_de` recupera el JSON en 11/11, la plantilla de chat está bien usada y `cargar_imagen` corrige la orientación EXIF y limita el lado mayor a 1024.
3. **Ya construyeron la salvaguarda correcta.** Leen dos veces, preguntan «¿se lee?», comparan y añaden una señal de confianza por token (MAD). Una vez corregida la caché (M4-01), esa señal es la única que distingue vis-10 de las hojas buenas.
4. **Honestidad técnica poco común.** El README dice que 4/12 y 2/12 no son resultados de M4, que el holdout ya se gastó, que la calibración impresa no valida letra manuscrita y que el retrieval de M2 cambió. Las cuatro afirmaciones se comprobaron y son ciertas.
5. **Verificador aritmético determinista** (`scripts/agente/verificador_aritmetico.py`). Sobre sus lecturas, encuentra la cuenta mal hecha en 5 de los 6 casos con error. Es la mejor evidencia que pueden mostrar en la demo (§5).

## 3. Qué falló y por qué

| # | Síntoma que vieron | Causa comprobada | Evidencia |
|---|---|---|---|
| 1 | corrida-01: 0/11 | Cuatro errores de medición, no de lectura: el criterio contaba menciones al paso en cualquier parte del texto, la numeración de pasos no coincidía, faltaba el alias `.jpeg` y el eval set de M2 tenía otro hash. Ustedes los corrigieron | `m4/corridas/` (LEEME de la corrida 01) |
| 2 | corrida-02/03: «Paso 2» repetido; casos sin `Diagnóstico:` | Falla del **tutor**, no del lector: reescribe o resuelve en vez de diagnosticar. Ustedes lo documentan | `m4/README.md` §2–3 |
| 3 | vis-10 recibió diagnóstico | El VLM inventa una hoja legible (`24 x 13 = 306`; la hoja dice 302). Las dos lecturas coinciden y «¿se lee?» responde sí. Solo la señal 4 lo detecta | Reproducido: misma lectura y confianza −0,101 |
| 4 | La salvaguarda imprimió FALSO NEGATIVO | **Caché vieja.** Se reutiliza cualquier lectura que tenga `"consistente"`. La caché versionada usa el esquema anterior, así que solo 3/11 fotos traen confianza, y `atipicos_por_mad` devuelve vacío con menos de 4 valores. Ustedes lo atribuyeron a la calibración faltante | [nb_m4.py L438](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/nb_m4.py#L438), [vision.py L225](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/agente/vision.py#L225) |
| 5 | «No está el conjunto de calibración» | El conjunto (`m4/eval_set/calibracion/`, 18 PNG) entró en el commit final, después de la corrida en T4, y no se llegó a ejecutar. Al ejecutarlo aquí: es texto impreso y no sirve como referencia para letra manuscrita, como ustedes advirtieron | Visual_n, celda 30; [nb_m4.py L280](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/nb_m4.py#L280) |
| 6 | Avisos `FutureWarning` en T4 | `transformers>=4.45` instala la última versión. Su corrida usó 5.20.2; `max_pixels=` desaparece en 5.23 y `torch_dtype` está obsoleto. Con 5.23 el procesador dejaría de aceptar su llamada | [nb_m4.py L146–149](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/nb_m4.py#L146-L149), [L1092](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/nb_m4.py#L1092) |
| 7 | No hay un flujo «subo una foto → diagnóstico» | `CODIGO_LIBERAR` borra el VLM antes de cargar el tutor, y `sistema_multimodal` recibe una lectura ya hecha, no una imagen | [nb_m4.py L495–500](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/nb_m4.py#L495-L500) |

En las salidas versionadas no aparece ningún OOM ni traceback del VLM. Tampoco encontramos problemas de dtype, plantilla, parseo ni preprocesamiento.

## 4. Cómo se arregla (rama de CODE_PR)

**C1 · La salvaguarda vuelve a decidir, y hay un punto de entrada de una foto** (M4-01, M4-02, M4-06)
- `lectura_vigente()`: la caché solo se reutiliza si la lectura trae las cuatro señales. Si quedan menos de 4 confianzas, se avisa.
- `revisar_foto(imagen, pregunta, leer=…, diagnosticar=…)` (vision.py) sigue tres pasos: lee, decide si la lectura es fiable y solo entonces diagnostica. Si no lo es, devuelve «No pude leer bien tu hoja…» con la transcripción para que el estudiante la confirme.
- La señal 4 se compara contra una **referencia fija** (`referencia_de_confianza`), no contra el lote. Así funciona con una sola foto. La referencia sale de fotos **nítidas** escritas a mano en `m4/referencia_manuscrita/` (8–10, con la cámara de la demo, fuera del eval set; las hojas `dev-*` de `HOJAS_PARA_ESCRIBIR.md` sirven). Sus versiones movidas sirven para comprobar que la señal las marca.
- `MANTENER_VLM = True` deja el VLM residente. `revisar_foto_demo(ruta)` (sección 12 del notebook) es la función que envolverá la app.
- Procesador con `size={"shortest_edge": 3136, "longest_edge": MAX_PIXELS}` y `dtype=`. transformers queda fijado en 5.20.2 (`m4/requirements-m4.txt`).

**C2 · `numeros_fieles`** (M4-03): nueva columna del scorecard que exige todos los números de la hoja, en orden.

Carga mínima para Colab T4. Es la misma de la rama y la probamos en MPS con 5.20.2:

```python
%pip install -q "transformers==5.20.2" "accelerate>=0.33" qwen-vl-utils
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
VLM_ID, MAX_PIXELS = "Qwen/Qwen2-VL-2B-Instruct", 1024 * 1024
vlm_proc = AutoProcessor.from_pretrained(
    VLM_ID, size={"shortest_edge": 56 * 56, "longest_edge": MAX_PIXELS})
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_ID, dtype=torch.float16, device_map="cuda").eval()   # T4: fp16, no bf16
```

Prueba de humo: `revisar_foto_demo` sobre vis-01, vis-07 y vis-10.

| Foto | Resultado esperado |
|---|---|
| vis-01 | Diagnóstico |
| vis-07 | Diagnóstico «sin error» |
| vis-10 | «No pude leer tu hoja» |

En MPS se obtuvo exactamente eso.

**No probado:** en T4. Conviene medir allí si el VLM y el tutor caben juntos; en MPS fp16 ocupan 7,6 GB, de modo que deberían caber en 16 GB.

## 5. Observaciones

### M4-01 · La caché vieja apagó la salvaguarda en la corrida final
- **Dónde:** [`CODIGO_TRANSCRIBIR`, nb_m4.py L438](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/nb_m4.py#L438); [`atipicos_por_mad`, vision.py L213–226](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/agente/vision.py#L213-L226); `m4/lecturas_m4.json`; Visual_n, celdas 32 y 34.
- **Qué se observa:**
  - La caché se reutiliza si la lectura tiene `"consistente"`, aunque le falte la confianza.
  - Con el JSON versionado, solo las 3 fotos del piloto tienen confianza.
  - Con menos de 4 valores la señal 4 no puede disparar, así que vis-10 pasa como fiable.
  - Reproducido en MPS. Con la caché corregida, solo vis-10 sale NO FIABLE.
- **Por qué importa:** el diagnóstico que hicieron («falta la calibración») apuntaba a otra causa. La salvaguarda sí separaba vis-10; no llegó a decidir.
- **Sugerencia:** C1 (`lectura_vigente` y el aviso con menos de 4 confianzas). Borrar la caché vieja antes de una corrida que se vaya a reportar.
- **Criterio de cierre:** una corrida en T4 imprime 11/11 confianzas y vis-10 como NO FIABLE.

### M4-02 · Con una sola foto, la hoja borrosa se lee como legible
- **Dónde:** `m4/README.md` §3 (límite declarado); [vision.py L225](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/agente/vision.py#L225).
- **Qué se observa:**
  - vis-10 se lee dos veces, igual, como «24 x 13 = 306», y «¿se lee?» responde sí.
  - En el producto llega una foto a la vez, y la MAD del lote no existe.
  - La calibración impresa no sirve como referencia para letra manuscrita: marca 5 hojas buenas como ilegibles.
- **Por qué importa:** es el caso en que el producto diagnosticaría un procedimiento que no escribió el estudiante.
- **Sugerencia** (ustedes ya la plantean):
  - Fotografiar 8–10 hojas nítidas escritas a mano (las `dev-*` sirven) como referencia fija, y sus versiones movidas para comprobar que se marcan.
  - Usar `revisar_foto` (C1).
  - Mostrar siempre «Esto es lo que leí, ¿es correcto?» antes del diagnóstico.
- **Criterio de cierre:** con la referencia manuscrita, las movidas se marcan y las nítidas no, y vis-10 responde «no pude leer». En nuestra prueba, la referencia provisional se armó con fotos del eval set; con fotos propias el resultado puede cambiar.

### M4-03 · «La lectura funciona en 10 de 11» mide menos de lo que dice
- **Dónde:** [`lectura_fiel`, vision.py L712–740](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/agente/vision.py#L712-L740); [m4/README.md L120](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/m4/README.md?plain=1#L120).
- **Qué se observa:**
  - `lectura_fiel` mira el número de pasos y el valor final escrito.
  - vis-09 cuenta como fiel aunque `14 × 9 = 126` se leyó «12G».
  - Exigiendo todos los números, quedan 8/9 hojas con procedimiento.
- **Por qué importa:** en un tutor que localiza el paso erróneo, un dígito mal leído en un paso intermedio cambia el diagnóstico.
- **Sugerencia:** C2 (`numeros_fieles`, columna nueva del scorecard) y reportar «8 de 9 hojas con todos los números bien leídos».
- **Criterio de cierre:** scorecard con las dos columnas y la frase del README ajustada.

### M4-04 · Delta, peor caso y extensión del README
- **Dónde:** [`delta_m3_m4.csv`](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/m4/delta_m3_m4.csv); [nb_m4.py L936](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/nb_m4.py#L936); `m4/README.md`.
- **Qué se observa:**
  - La fila de latencia compara 14,1 s de M3 (tarea de texto) con 13,0 s de M4 (tarea con imagen), sin aclarar que son tareas distintas.
  - El peor caso es el primer fallo de la tabla (`fallos.iloc[0]` → vis-01), no uno elegido y analizado.
  - El README tiene unas 2 600 palabras; S13 pide una página.
- **Por qué importa:** en la demo, «la foto cuesta 3,8 s» es más claro y más honesto que «M4 es más rápido que M3».
- **Sugerencia:**
  - Latencia de la etapa visual, por separado.
  - El peor caso elegido: vis-10, que el sistema diagnosticaba sin haber leído.
  - Una página arriba, con el resto como anexo.
- **Criterio de cierre:** delta con la etiqueta de la tarea en la fila de latencia, peor caso analizado y README de una página.

### M4-05 · Procedencia y uso de las fotos
- **Dónde:** `m4/README.md` (no hay sección); solo [HOJAS_PARA_ESCRIBIR.md L11](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/m4/eval_set/HOJAS_PARA_ESCRIBIR.md?plain=1#L11).
- **Qué se observa:** la regla de privacidad existe para quien escribe las hojas. Falta decir de dónde salen las fotos del eval set, quién las escribió, qué se guarda de la foto de un estudiante y qué pasa con datos de menores.
- **Sugerencia:** 4–5 líneas en el README: procedencia, consentimiento, no guardar las fotos por defecto, y el límite «no reemplaza al docente».
- **Criterio de cierre:** sección «Datos y uso responsable» en el README.

### M4-06 · Versiones y API en transición
- **Dónde:** [nb_m4.py L146–149 y L1092](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/nb_m4.py#L146-L149).
- **Qué se observa:** sin fijar la versión, Colab instala la última. Con 5.23 `max_pixels=` deja de existir.
- **Sugerencia:** C1. `size=` da el mismo `image_grid_thw` y las mismas lecturas, y `m4/requirements-m4.txt` fija la versión.
- **Criterio de cierre:** la celda de entorno imprime transformers 5.20.2 sin `FutureWarning`.

### M4-07 · Si quieren un lector más fuerte (opcional)
- **Dónde:** `VLM_ID` en [nb_m4.py L141–152](https://github.com/Emmanuelcasta/TopicosdeIA/blob/9c0c0dfc8703dd642ef4dcbc547d2b7ef34a0dbd/scripts/nb_m4.py#L141-L152).
- **Qué se observa.** Con su mismo código (prompts, dos lecturas, señales) cambiamos solo el modelo:

  | | Qwen2-VL-2B (el suyo) | Qwen2.5-VL-3B |
  |---|---|---|
  | Hojas con todos los números bien leídos | 8/9 (vis-09: «12G») | 7/9 |
  | **¿Copia el error de la hoja?** | Sí, en las 6 hojas con error | En 5 de 6. **En vis-06 lee «6/8 + 1/8 = 7/8» donde la hoja dice 8/8** |
  | vis-10 borrosa | Inventa «306»; «¿se lee?» responde sí | «¿Se lee?» responde **no**; texto incoherente; confianza −0,62 |
  | vis-01 muy desenfocada | La lee | «¿Se lee?» responde no |
  | Memoria y lectura (MPS) | 4,4 GB · ≈3 s | 7,5 GB · ≈5,5 s |
  | Precisión | fp16 (la de T4) | **En fp16 (MPS) generó NaN**; funciona en bf16, que T4 no tiene de forma nativa. No probado en T4 |

  También se probó **GOT-OCR-2.0** (`stepfun-ai/GOT-OCR-2.0-hf`, 580 M, fp16, 1,1 GB, ≈1,5 s por foto), un OCR que puede devolver LaTeX:
  - copia el «302» de vis-01 y lee «12G» en vis-09, igual que el 2B;
  - en vis-06 aplana las fracciones (`314+118`, `618+18=818`), que es justo la notación que el tutor necesita;
  - en vis-03 lee «415000» por «45000»;
  - en vis-10 inventa texto («24715», o frases en inglés en modo LaTeX), y con la foto desenfocada repite «1000…»;
  - solo transcribe: no hay JSON por pasos ni respuesta a «¿se lee?», así que la salvaguarda tendría que rehacerse.

  Candidatos **no probados** en esta revisión, por si quieren explorarlos después: Qwen2.5-VL-7B en AWQ (cabe en T4) y modelos de fórmula a LaTeX como texify.

- **Por qué importa:**
  - El 3B detecta mejor una hoja ilegible, pero **corrige al estudiante**. Ese es justo el riesgo que su README identifica como el principal del producto.
  - Además, en T4 no entra tal cual: en fp16 genera NaN, y bf16 no es nativo de T4.
  - Por eso su elección del 2B es defendible.
- **Sugerencia (opcional, no hace falta para la demo):**
  - Mantener Qwen2-VL-2B como lector.
  - Si exploran otro, medirlo primero con la pregunta del producto: ¿copia el error? Usen vis-06 y vis-01 como prueba.
  - Una transcripción a LaTeX puede ser una vista del producto, pero en nuestra prueba exploratoria rompió el JSON con el 2B (vis-01). Va al backlog.
- **Criterio de cierre:** no aplica. Es una sugerencia.

### Límites que ustedes declararon (sugerencias, sin observación nueva)
- **Configuración final:** medirla con las 12 fotos por el camino de `revisar_foto`. Será una medición post-hoc, porque el holdout ya se usó; conviene decirlo así.
- **Adversariales:** pasar la etiqueta de los adversariales del código (por ID) a una columna del CSV.
- **Tutor:** pasarle la pista del verificador aritmético cuando la lectura es fiable. Reescribe en lugar de diagnosticar, como ya vieron.
- **Retrieval:** que el de M2 vuelva a ser el de S10 (híbrido con reranker) antes de comparar regresión.

## 6. Seguimiento de M3

No hay PR de revisión de M3 ni en `Emmanuelcasta/TopicosdeIA` ni en el fork `manularrea/TopicosdeIA`, así que no hay sugerencias previas que seguir. El camino de texto de M3 (`AgenteTutor`, herramientas y recuperación) se reutiliza sin reescribirlo.

## 7. Demo Day

Para la demo se presenta el sistema completo M1–M4 como producto:
- la foto (M4);
- el agente tutor con herramientas (M3);
- la recuperación sobre su corpus (M2);
- el tutor afinado (M1).

[`PLAN_DEMO.md`](PLAN_DEMO.md) propone tres prioridades:

1. **[P0] Un «momento wow» de punta a punta.** El estudiante sube la foto de su hoja y ve «esto es lo que leí» y «tu error está en el paso 2», con la cuenta que lo prueba (M4-01, M4-06; C1).
2. **[P1] Confianza como característica: «¿Leí bien tu hoja?».** Ante una foto movida, el producto pide otra en lugar de inventar (M4-02, M4-03; C1, C2).
3. **[P2] Tres números de valor y una demo robusta** (M4-04, M4-05).

## 8. Cómo reproducir lo verificado

Desde la raíz del repositorio, con la rama de CODE_PR:

```bash
pip install -r m4/requirements-m4.txt
pytest -q scripts/tests        # 251 pruebas; las 13 nuevas están en test_m4_vision.py
```

- **Lector:** notebook `m4/pipeline/S13_M4_Pipeline_Visual.ipynb`. Las secciones 1–5 se ejecutan sin cambios en T4. En otro dispositivo, `DEVICE` cambia solo.
- **Una foto:** sección 12, con `MANTENER_VLM = True` antes de la sección 5: `revisar_foto_demo("m4/eval_set/imagenes/vis-10.jpeg")`.
- **No se re-ejecutaron:** la corrida en T4, el tutor con sus adaptadores de Drive y el juez por API.
