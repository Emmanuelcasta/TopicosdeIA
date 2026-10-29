# M4 · Entrada visual

El tutor recibe la foto de un procedimiento escrito a mano y tiene que decir en
qué paso está el error.

> **Estado.** M4 conecta la lectura de fotos con el tutor de M3, pero las corridas
> disponibles no demuestran una mejora confiable. El conjunto visual de 12 casos
> ya se ejecutó en notebooks exploratorios: una corrida obtuvo 4/12, pero repitió
> «Paso 2» en sus 11 conclusiones; otra obtuvo 2/12 y dejó 7 casos sin conclusión.
> Como la puerta de desarrollo no se había superado en una de esas corridas, estas
> cifras sirven para diagnosticar el sistema y no como resultado final de una prueba
> ciega.

El análisis completo de las decisiones, los resultados y los siguientes pasos está
en [CONCLUSIONES_M4.md](CONCLUSIONES_M4.md).

## 1 · Objetivo y flujo

La pregunta que necesita ojos es *"¿en qué paso me equivoqué?"*. El procedimiento
está escrito a mano, y si el estudiante lo transcribe corrige el error sin darse
cuenta: el sistema nunca llega a ver el fallo que debía encontrar.

**Patrón `input`**: la foto entra, un VLM la lee y el sistema de M3 la consume
como texto.

| Patrón | Decisión | Razón |
|---|---|---|
| **input** | **Elegido** | La pregunta necesita el *contenido* de la hoja como pasos de texto, y M3 ya consume texto |
| tool | Descartado | Un clasificador devuelve una clase, no una secuencia de pasos |
| output | Descartado | Dibujar el paso corregido no cambia ninguna decisión del estudiante |

El flujo tiene **dos etapas separadas a propósito**, porque fallan de formas
distintas y una sola cifra las confundiría:

```
foto ──► ver()            ──► transcripción ──► prompt_revision ──► tutor ──► Diagnóstico: …
         Qwen2-VL-2B          JSON con pasos      (M3 + LoRA)
         solo transcribe,                         decide el veredicto
         no diagnostica
```

Entre las dos etapas se libera el VLM de la GPU: en una T4 de 16 GB no caben
cómodos el VLM de 2B, el tutor de 1.5B y el juez. La transcripción se guarda en
`lecturas_m4.json`, así que la fase de diagnóstico se repite sin volver a leer
las fotos.

El tutor cierra con una línea obligatoria, y **es esa línea la que se califica**:

| Conclusión | Cuándo |
|---|---|
| `Diagnóstico: Paso N` | un paso está mal |
| `Diagnóstico: sin error` | todos los pasos están bien |
| `Diagnóstico: ilegible` | la hoja no se puede leer |
| `Diagnóstico: fuera de dominio` | la hoja no es matemáticas |
| `Diagnóstico: dato no disponible` | se pide algo que no está escrito |

La explicación se conserva y se mide aparte, nunca como acierto.

### Dos conjuntos que no se comparan entre sí

| Conjunto | n | Qué mide |
|---|---|---|
| Eval set de M4 | 12 **con imagen** | Una tarea **nueva**: localizar el paso erróneo |
| Eval set de M2 | 21 **sin imagen** | La tarea de siempre, como **regresión** |

No se restan: no existe "la mejora de M3 a M4" como un número único, porque M3 no
podía recibir imágenes.

## 2 · Comparación por configuración

Tres configuraciones sobre los **mismos 9 casos de desarrollo** (`dg-*`), con el
arnés real del pipeline. **No es el scorecard de M4.**

| Configuración | Adaptador | Prompt | Aciertos | Formato válido | Llamadas a herramientas |
|---|---|---|---|---|---|
| **Baseline** | `diagnostico` | original | **3/9** | 9/9 | 0 |
| **A** | `agente` | original | **2/9** | 8/9 | 4 |
| **B** | `diagnostico` | `Paso N` | **2/9** | 9/9 | 0 |

Caso por caso:

| Caso | Esperada | Baseline | A | B |
|---|---|---|---|---|
| `dg-a06` | paso 3 | paso 2 ✗ | paso 2 ✗ | paso 1 ✗ |
| `dg-a12` | paso 2 | paso 2 ✓ | paso 2 ✓ | paso 2 ✓ |
| `dg-a18` | paso 3 | paso 2 ✗ | paso 2 ✗ | paso 2 ✗ |
| `dg-b06` | paso 2 | paso 2 ✓ | paso 2 ✓ | paso 1 ✗ |
| `dg-c04` | sin error | paso 1 ✗ | paso 2 ✗ | paso 2 ✗ |
| `dg-c08` | sin error | paso 2 ✗ | paso 2 ✗ | paso 2 ✗ |
| `dg-d03` | ilegible | ilegible ✓ | sin error ✗ | ilegible ✓ |
| `dg-e03` | fuera de dominio | paso 1 ✗ | *(sin conclusión)* ✗ | paso 1 ✗ |
| `dg-f03` | dato no disponible | paso 1 ✗ | paso 2 ✗ | paso 1 ✗ |

### Conclusión

**Ninguna configuración mejoró el acierto.** El baseline, con 3/9, es el mejor de
los tres, y con n = 9 una diferencia de un caso no distingue nada.

El diseño permite dos comparaciones con una sola variable cambiada:

- **Adaptador** (baseline vs A, mismo prompt): 3/9 vs 2/9. El adaptador de
  diagnóstico no es peor que el de S10, y acierta el adversarial `ilegible` que A
  falla.
- **Prompt** (baseline vs B, mismo adaptador): 3/9 vs 2/9. El cambio de prompt
  **no mejoró**; `dg-b06` pasó de acierto a fallo.

Lo que sí queda establecido es más útil que las cifras: **cambiar un solo
carácter del prompt (`Paso 2` → `Paso N`) desplazó la conclusión en 3 de 9 casos
sin que la hoja cambiara** (`a06` 2→1, `b06` 2→1, `c04` 1→2). Un índice de paso
que se mueve con un detalle irrelevante del prompt no está derivado del contenido
de la hoja: el modelo no compara el valor calculado con el escrito, emite un
índice casi arbitrario.

Dos observaciones más:

- **Las 4 llamadas a herramientas de A están en los 2 casos que empeoró.** En
  `dg-d03` (hoja ilegible) llamó `buscar_documentos` dos veces y concluyó "sin
  error"; en `dg-e03` (una receta) llamó `evaluar_expresion` y `multiplicar` y no
  emitió conclusión. Las usó para intentar resolver lo que debía rechazar.
- **La lectura sí funciona.** Las transcripciones conservan los errores
  manuscritos en 10 de las 11 hojas, que era el riesgo principal del módulo.

## 3 · Límites

- **Son 9 casos de desarrollo.** Una diferencia de uno o dos no significa nada, y
  ninguna de estas cifras es un scorecard de M4.
- **El cambio de prompt no mejoró el acierto.** La hipótesis de que el sesgo del
  índice venía del ejemplo del prompt quedó **refutada**: los dos casos que ya
  tenían los valores correctos (`dg-a06`, `dg-a18`) no pasaron a `paso 3`.
- **A y B no aíslan una sola variable entre sí**: cambian el adaptador **y** el
  prompt. Solo son limpias las comparaciones contra el baseline, que comparte una
  de las dos.
- **Los 12 casos visuales sí se ejecutaron** en corridas exploratorias, incluso
  cuando la puerta de desarrollo no se había superado. Por eso ya no son un
  holdout ciego para ajustar el sistema. Los resultados de 4/12 y 2/12 corresponden
  a ejecuciones distintas; no se promedian ni se presentan como una mejora.
- **La salvaguarda de hojas ilegibles está sin validar, y no se puede validar con
  lo que hay.** El conjunto de `eval_set/calibracion/` es texto **impreso**
  generado por script: sirve para comprobar que el mecanismo corre, **no** para
  afirmar nada sobre letra manuscrita, y no debe usarse como si fuera dato real.
  Faltan **6 fotografías** que hay que aportar: las 3 hojas `dev-*` de
  `HOJAS_PARA_ESCRIBIR.md`, escritas a mano, cada una fotografiada **nítida** y
  **movida o desenfocada**. Sin ese par legible/ilegible real no se puede medir
  si la detección separa ni a qué coste en falsos positivos.
- **La regresión de M2** (21 casos sin imagen) se midió con el hash correcto, pero
  el retrieval de este notebook es denso con `k=3` y el de S10 era híbrido con
  reranker: **el sistema no es idéntico** y la diferencia no se atribuye a M4.

## 4 · Verificador aritmético determinista · **experimento, medido en desarrollo**

Dos intentos de arreglarlo por el lado del modelo —entrenar un adaptador de
diagnóstico y ajustar el prompt— no movieron el acierto. La vía que queda es
estructural: **que el pipeline verifique la aritmética de cada paso de forma
determinista**, con las herramientas exactas que ya existen en
`scripts/agente/herramientas.py` (`Fraction`, sin coma flotante), y le entregue
al tutor el paso marcado:

> *"el paso 3 no cuadra: 2100 ÷ 14 = 150, y está escrito 160"*

La tarea del modelo pasa de "detecta el error" a "reporta el paso marcado", que
es algo que un modelo de 1.5B sí hace. Es la tesis de uso de herramientas de M3
aplicada a visión, y cambiaría lo que M4 mide — habría que declararlo.

### Resultado de la prueba local

Implementado en `scripts/agente/verificador_aritmetico.py` y medido con
`scripts/evaluar_verificador_dg.py`, sin GPU y sin modelo, sobre los **9 casos de
desarrollo**:

| | |
|---|---|
| Pasos erróneos detectados con el índice correcto | **4/4** |
| Casos `sin error` sin marcar (falsos positivos) | **2/2** (0 FP) |
| Cobertura: casos que puede evaluar | **6/9** |
| Adversariales en los que se abstuvo | 2/3 |

**Cumple el criterio** —detectar los pasos erróneos evaluables sin marcar los
`sin error`— sobre los 6 casos evaluables. Detalle en
`m4/verificador_aritmetico_dg.csv`.

En `dg-f03` no se abstiene: esa hoja **sí** trae un procedimiento correcto, y lo
que falta es un dato que la pregunta pide y la hoja no tiene. Decir que las
cuentas cuadran es cierto, así que no es un falso positivo; simplemente ese caso
queda fuera de su alcance.

**Lo que esto NO es.** Mide el verificador, no al tutor. **No está conectado al
pipeline**, no sustituye la conclusión del modelo y **no es una mejora de M4**:
el tutor sigue teniendo que emitir el veredicto, y sobre los 9 casos de
desarrollo sigue en 3/9. Lo que el resultado sostiene es que la señal que al
modelo le falta —qué paso no cuadra— se puede calcular de forma exacta y barata.

**Límites del verificador**, por diseño:

- **No comprueba el planteamiento.** Si el estudiante eligió la operación
  equivocada pero la calculó bien, las igualdades cuadran y no dice nada.
- **No evalúa hojas sin aritmética**: ilegibles, fuera de dominio o cuando se
  pide un dato ausente. Ahí se abstiene, que es lo correcto.
- Cubre **6 de 9** casos de desarrollo. Los 3 adversariales siguen dependiendo
  del modelo.

El siguiente paso sería pasarle al tutor la pista del verificador
(`pista_para_el_tutor`, que a propósito **no** emite `Diagnóstico:`) y volver a
medir en desarrollo. **No se ha hecho.**

## 5 · Reproducir

Todo corre en Colab con Drive montado en `/content/drive/MyDrive/ProyectoIA`.
Sirve cualquier GPU con CUDA; las corridas registradas usaron una Tesla T4 de
16 GB.

**Generar los datos** (local, sin GPU):

```bash
python scripts/eval_set_m4_fuente.py              # eval set de los 12 casos
python scripts/trayectorias_diagnostico_fuente.py # 42 trayectorias de diagnóstico
python scripts/calibracion_legibilidad_fuente.py  # conjunto de calibración
python scripts/generar_notebooks.py S13           # los dos notebooks
python -m pytest scripts/tests -q                 # suite local de pruebas
```

El generador **se niega a sobrescribir un notebook que tenga salidas** salvo que
se le pase `--forzar`: un `.ipynb` ejecutado es evidencia, no un archivo generado.

**Entrenar el adaptador de diagnóstico** — `S13_M4_FineTuning_Diagnostico.ipynb`,
celdas 5 a 30. Escribe `adaptadores/qwen-lora-diagnostico/` y **no toca**
`qwen-lora` ni `qwen-lora-agente`.

**Pipeline visual** — `S13_M4_Pipeline_Visual.ipynb` (67 celdas):

| Para | Celdas |
|---|---|
| Solo la puerta de desarrollo (sin fotos ni VLM) | 5 · 6 · 8 · 11 · 17 · 23 · 38–46 · 48 · 49 · 50 · 53 |
| Piloto de lectura (3 fotos) | añadir 19 · 29 · 31 · 33 |
| Corrida completa | todas, en orden |

La celda **53** es la puerta: si el ensayo de desarrollo no da conclusiones
válidas, la **55** se detiene con `PuertaNoSuperada` y los 12 casos no se
evalúan. Es deliberado.

Las dos pruebas A/B son celdas sueltas que se pegan después de la 50:
`celda_medicion_dev_dg.py` y `celda_pruebas_A_B_dev_dg.py`.

## 6 · Artefactos

### Notebooks

| Archivo | Qué es |
|---|---|
| `pipeline/S13_M4_Pipeline_Visual.ipynb` | **versión principal** del pipeline, sin salidas |
| `pipeline/S13_M4_FineTuning_Diagnostico.ipynb` | **versión principal** del entrenamiento, sin salidas |
| `pipeline/S13_M4_Pipeline_Visual_AB_dev_ejecutado_2026-10-25.ipynb` | ejecutado · pruebas A y B |
| `pipeline/S13_M4_Pipeline_Visual_ejecutado_dev_2026_10_25.ipynb` | ejecutado · baseline de desarrollo, 3/9 |
| `pipeline/S13_M4_Pipeline_Visual_n.ipynb` | ejecutado · corrida completa con `agente`, 4/12 |
| `pipeline/S13_M4_Pipeline_Visualv2.ipynb` | ejecutado · corrida del 24-10, 2/12 |
| `pipeline/S13_M4_FineTuning_Diagnostico_ejecutado.ipynb` | ejecutado · entrenamiento del adaptador |

Los dos primeros se regeneran desde `scripts/`. Los demás son **copias de
corridas** y no se regeneran ni se sobrescriben.

### Datos y resultados

| Archivo | Qué es |
|---|---|
| `eval_set/eval_set_m4.csv`, `HOJAS_PARA_ESCRIBIR.md` | los 12 casos y las hojas para escribirlos |
| `eval_set/imagenes/` | 11 fotos (`vis-10` es `.jpeg`; el emparejamiento es por nombre base) |
| `eval_set/calibracion/` | 18 imágenes sintéticas + manifiesto, para la salvaguarda de ilegibles |
| `lecturas_m4.json` | transcripciones del VLM |
| `scorecard_m4.csv`, `resultados_m4.json`, `delta_m3_m4.csv`, `detalle_regresion_m4.csv` | corrida del 24-10 |
| `trayectorias_diagnostico.md` | informe de las 42 trayectorias |
| `verificador_aritmetico_dg.csv` | prueba del verificador sobre los 9 casos de desarrollo |
| `dev_dg_arnes_s13.csv`, `dev_dg_A_agente.csv`, `dev_dg_B_pasoN.csv` | CSV originales completos de las tres configuraciones sobre desarrollo |
| `dev_dg_*_desde_notebook.csv` | exportaciones parciales preliminares; auxiliares, no son la fuente principal |
| `corridas/` | corridas históricas archivadas, cada una con su README |

### Trazabilidad de las comparaciones de desarrollo

Los tres CSV originales se recuperaron de `MyDrive/ProyectoIA/m4/` y están en
este directorio. Cada uno contiene los 9 casos, la respuesta completa y los
campos de evaluación y traza que guardó la corrida. Se contrastaron los IDs,
las etiquetas esperadas, predicciones, validez de formato y aciertos contra las
exportaciones preliminares y las métricas impresas por los notebooks.

| Configuración | Aciertos | Formato válido | Herramientas |
|---|---:|---:|---:|
| Baseline `arnes_s13` | 3/9 | 9/9 | No usar esta columna: la extracción de la traza de herramientas fue defectuosa en esa corrida |
| A · `agente` + prompt original | 2/9 | 8/9 | 4 llamadas |
| B · `diagnostico` + prompt `Paso N` | 2/9 | 9/9 | 0 llamadas |

Las filas de A y B incluyen `config`, `adaptador`, `pasos_llm`, `terminacion`,
`reintento`, `respuesta` y las herramientas observadas. En A, `predicha` está
vacía para `dg-e03`: el modelo no emitió una conclusión válida; se cuenta como
fallo de formato. La corrida baseline sí conserva las respuestas, pero su
columna de herramientas no es fiable por un defecto de extracción de traza.

Los archivos `*_desde_notebook.csv` son exportaciones parciales generadas antes
de recuperar los originales. Se conservan para trazabilidad, pero los CSV sin
ese sufijo son la fuente principal para el análisis por caso.

### El adaptador LoRA de diagnóstico

`adaptadores/qwen-lora-diagnostico/` vive en Drive, **no en el repositorio**:

```
adapter_config.json          ~1 KB
adapter_model.safetensors    ~74 MB
```

Para obtenerlo, o se descarga de `MyDrive/ProyectoIA/adaptadores/` o se vuelve a
entrenar con `S13_M4_FineTuning_Diagnostico.ipynb`, que tarda unos dos minutos en
una T4 (30 pasos, 6 épocas sobre 33 ejemplos).

**Recomendación sobre Git LFS:** 74 MB entra en el límite de GitHub, pero Git
guarda cada versión completa de un binario, así que reentrenar y volver a
comitear multiplicaría el tamaño del repositorio. Si se decide versionarlo, que
sea con **Git LFS**; y si no, basta con que el notebook de entrenamiento lo
reproduzca, que es barato. Lo mismo vale para los otros dos adaptadores.
**No se ha añadido ni subido.**

### Fuera del repositorio

`graphify-out/` (caché), `.DS_Store`, el modelo base de HuggingFace y los
adaptadores.
