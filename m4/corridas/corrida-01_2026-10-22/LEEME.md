# Corrida 01 · 22-10-2026 · **no es un scorecard de M4**

Copia íntegra de la primera corrida completa, conservada como evidencia. Dio
**0/11** y ninguna de las cuatro causas fue del modelo: las cuatro eran de
medición. Por eso estas cifras **no pueden presentarse como resultado de M4**, ni
en la entrega ni en la presentación.

## Qué contiene

| Archivo | Qué es |
|---|---|
| `S13_ejecutado.ipynb` | el notebook con las salidas de esa corrida (celdas 36–64) |
| `lecturas_m4.json` | las 10 transcripciones del VLM. **Esto sí sirve**: son buenas |
| `scorecard_m4.csv` | los 11 casos calificados con el criterio viejo |
| `resultados_m4.json` | el resumen, incluido `imagenes_faltantes: ["vis-10.jpg"]` |
| `delta_m3_m4.csv` | la tabla de delta de esa corrida |
| `detalle_regresion_m4.csv` | los 21 casos de M2, caso por caso |

## Las cuatro causas, todas de medición

**1 · El criterio contaba menciones de pasos dentro de la explicación.** Buscaba
los textos prohibidos como subcadena de toda la respuesta. Como el tutor empieza
repasando el procedimiento, los once casos fallaron con `cayó en la trampa: 'paso
1'`. `vis-05` llegó a escribir "Paso 2: Incorrecto" —la respuesta esperada— y se
contó como fallo igualmente.

**2 · La numeración no coincidía con la del eval set.** El eval set cuenta la
línea "Respuesta final" como un paso más; el VLM la devuelve en su propio campo.
Para `vis-03`, cuyo error está en el paso 3, el tutor solo veía dos pasos: la
respuesta correcta era **inalcanzable por construcción**.

**3 · `vis-10` quedó sin evaluar.** El archivo es `vis-10.jpeg` y el eval set lo
declara como `vis-10.jpg`; el emparejamiento era por nombre exacto. Era el único
caso de hoja ilegible.

**4 · El eval set de M2 en Drive no era el de M3.** La celda imprimió `SHA
coincide: False`. Su 15/21 se midió con las claves viejas de `adv-02`, así que
**no es comparable** con el 13/21 de M3.

## Lo que sí dice esta corrida, y hay que conservar

**Las transcripciones son buenas.** El VLM leyó la letra manuscrita y
**conservó los errores** en lugar de arreglarlos, que era el riesgo principal:
302 en `vis-01`, 2347 en `vis-02`, 70 en `vis-05`, 8/8 en `vis-06`. Un solo fallo
de lectura real: en `vis-09` leyó `126` como `12G`.

**El tutor no revisaba: reescribía.** Es el hallazgo de fondo, y no lo cambia
ningún arreglo del evaluador. Con el adaptador de M1 encima, el modelo continúa
el procedimiento en el formato `Paso N: … Respuesta final: …` con el que se
entrenó, en vez de emitir un veredicto. Ante `vis-02` respondió el ejercicio bien
resuelto sin decir nunca qué paso estaba mal; ante la lista de mercado de
`vis-11` se inventó un ejercicio de precios y descuentos y entró en bucle; ante
`vis-07`, que es correcto, acusó errores en los tres pasos.

**No se puede rescatar re-puntuándola.** Las once respuestas se re-calificaron
con el criterio nuevo: **ninguna** trae una línea `Diagnóstico:`, porque el
prompt de entonces no la pedía. Hay que repetir la corrida con el formato nuevo.

## Qué se cambió después

Está en el commit siguiente a `f82c0d5` y en el encabezado de
`scripts/agente/vision.py`. En resumen: conclusión estructurada obligatoria y
calificada aparte de la explicación, numeración alineada con el eval set,
emparejamiento de imágenes por nombre base con detección de ambigüedad, y una
guarda que marca la regresión de M2 como no comparable si el SHA no cuadra.

Ningún valor esperado del eval set se tocó.
