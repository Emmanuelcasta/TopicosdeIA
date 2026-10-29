# Corrida 03 · 24-10-2026 · **evidencia diagnóstica, no un scorecard**

`S13_ejecutado_v2.ipynb` (SHA-256 `fbc8159185ad92cf…`, 63 celdas, exec 5–37,
0 errores) más los cinco artefactos que produjo. GPU: **Tesla T4 (16 GB)**.
Adaptador del tutor: `agente` (C5-FT). Latencia 13,0 s por caso, 3,8 s del VLM.

Resultado **2/12** — 2/9 no adversariales, 0/3 adversariales. **No es un
resultado de M4** y no debe presentarse como tal.

## Lo que esta corrida arregló

- El eval set de M2 ya coincide con el repositorio: `ba9c4f3b2d80d687 … True`.
- Las 11 fotos reconocidas, `vis-10.jpeg` incluida; los 12 casos evaluados.
- Una hoja sin procedimiento ya no se convierte en un paso: `vis-11` se
  transcribió con `pasos: []` y el tutor recibió "no hay ningún paso".

## Los cuatro problemas

**1 · La puerta `dev-*` no detenía nada.** Dio 0/3, imprimió "PAREN AQUÍ" y la
ejecución siguió hasta evaluar los 12 casos. Era un `print`, no una parada.

**2 · El tutor no revisa: resuelve.** Es el hallazgo de fondo. Rehace el
procedimiento **correctamente** y lo presenta como si fuera el del estudiante:

| Caso | El estudiante escribió | El tutor respondió | ¿Avisó? |
|---|---|---|---|
| `vis-01` | `24 × 13 = 302` | `24 × 13 = 312` | no |
| `vis-03` | final `35000` | final `36000` | no |
| `vis-04` | `1248 ÷ 8 = 146` | `1248 ÷ 8 = 156` | no |
| `vis-05` | `12 × 5 = 70` | `12 × 5 = 60` | no |
| `vis-06` | `6/8 + 1/8 = 8/8` | `6/8 + 1/8 = 7/8` | no |

Los 7 que no emitieron `Diagnóstico:` son exactamente los casos con error más
`vis-09`: cuando el trabajo está mal, el modelo produce su propia versión y se
queda sin nada que concluir. El reintento tampoco lo rescata — a los 7 se les
pidió solo la línea y los 7 devolvieron `Respuesta final: …`.

**3 · La doble lectura es inútil contra este fallo, y está probado.** Las dos
lecturas de `vis-10` salieron **idénticas carácter a carácter**, inventando
`24 x 13 = 306` donde la hoja dice 302. No es inestabilidad: es determinismo.

**4 · `vis-11` copió el ejemplo del prompt.** Respondió con la panadería de
`15 × 4` que estaba en el ejemplo resuelto, incluido su `Diagnóstico: Paso 2`.
El ejemplo era un error de diseño del prompt, no un fallo del modelo.

## La regresión de M2, aparte

15/21 con el hash coincidiendo, así que el conjunto es comparable con el 13/21 de
M3. El **sistema** no lo es: aquí el retrieval es denso con `k=3` y en S10 era
híbrido con reranker. Los +2 casos no se atribuyen a M4. Lo que esa fila sostiene
es que la visión no rompió lo que ya funcionaba.

## Qué sí deja

La **lectura funciona**: 10 de 11 transcripciones conservan los errores
manuscritos. El cuello de botella está en el diagnóstico, y dentro de él en que
el modelo no sabe representar "el estudiante se equivocó". Eso es lo que motiva
el adaptador de diagnóstico.
