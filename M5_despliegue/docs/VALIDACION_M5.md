# Validación de M5 · cinco experimentos que no se mezclan

Generado por `M5_despliegue/evaluacion/validacion_m5.py`.

> **Una comprobación de reproducción no es una medición de exactitud.** Que el enrutador desplegado tenga los mismos pesos y las mismas 21 rutas que el del notebook dice que la pieza servida es la medida. No dice cuántos casos acierta el tutor hoy. Las dos cosas van en experimentos distintos, y E2 incluye la primera como requisito previo.

## Estado de los requisitos

| Requisito | Estado |
|---|---|
| GPU | **NO** |
| retrieval (sentence-transformers, rank_bm25, sklearn) | sí |
| transformers + peft | sí |
| adaptadores | qwen-lora: sí, qwen-lora-agente: NO, qwen-lora-diagnostico: NO |
| eval set de M2 (21 casos) | sí |
| herramientas esperadas de M2 | sí |
| eval set visual de M5 (12 hojas) | sí |
| fotografías de M5 | 0 de 23 |
| verificación del retrieval | sí |
| URL pública del Space | **NO** |

## Los experimentos

### E1 · Texto · baseline S08-D y sus controles

**Mide:** exactitud por caso, alucinaciones, abstenciones, latencia

**NO mide:** nada visual, y nada sobre herramientas: S08-D no las monta

**Conjunto:** data/eval_set_m2.jsonl · 21 casos, sin modificar

**Cómo:** `python M5_despliegue/evaluacion/ablaciones.py --solo-controles`

**Necesita:** GPU, modelo base, adaptador qwen-lora

**Criterio de aprobación:** los tres controles reproducen su cifra: s08d 18/21, solo_lora 14/21, s08c 13/21. Si no, el entorno no es el que midió y nada más se interpreta

**Registra:** exactitud por caso, alucinaciones, abstenciones, latencia p50/p95, errores de formato

---

### E2 · Recuperación y enrutamiento

**Mide:** si enruta cada caso como el notebook, y si los pasajes que trae contienen la clave de oro

**NO mide:** la exactitud del tutor. Un enrutador perfecto con un generador malo da 0 aciertos, y al contrario

**Conjunto:** 33 consultas de desarrollo + los 21 casos de M2 para las rutas + las 4 claves de oro de conocimiento

**Cómo:** `python M5_despliegue/evaluacion/verificar_retrieval.py --guardar   (requisito previo, no resultado)`

**Necesita:** sentence-transformers, rank-bm25, scikit-learn, los dos modelos de retrieval en caché

**Criterio de aprobación:** pesos, leave-one-out y las 21 rutas idénticas a m3_rag_avanzado.json. Recall de la clave de oro en los 4 casos de conocimiento ≥ 3/4, como en S08

**Registra:** ruta por caso, pesos de la regresión, leave-one-out, recall de la clave de oro, latencia del retrieval

---

### E3 · Herramientas matemáticas

**Mide:** si el modelo ELIGE la herramienta correcta al leer la pregunta, y si su respuesta final usa el valor devuelto

**NO mide:** si las herramientas calculan bien: eso está en scripts/tests/test_calculo.py, son deterministas y se comprueban sin modelo

**Conjunto:** data/eval_set_m2.jsonl con data/herramientas_esperadas_m2.jsonl, más preguntas de cálculo en español que todavía NO existen en ningún eval set

**Cómo:** `python M5_despliegue/evaluacion/ablaciones.py --filas 3,4`

**Necesita:** GPU, casos de cálculo en un eval set (FALTAN)

**Criterio de aprobación:** selección correcta ≥ la de C5-FT (0,585 en scorecard_agente_v2.csv, fila `Selección correcta`) y fidelidad a la herramienta ≥ 0,84 (misma fuente)

**Registra:** herramienta llamada por caso, selección correcta, fidelidad a la herramienta, llamadas con error, innecesarias en adversarial

---

### E4 · Lectura de fotos y salvaguarda de ilegibles

**Mide:** si la transcripción coincide con la de referencia, y si las borrosas se marcan no fiables sin rechazar las nítidas

**NO mide:** el diagnóstico. Una lectura correcta no implica un diagnóstico correcto, y en M4 fue justo así

**Conjunto:** data/eval_set_m5_vision.jsonl · 12 hojas NUEVAS, 11 pares legible/borrosa + 1 solo borrosa

**Cómo:** `pendiente de notebook; el eval set ya está generado`

**Necesita:** GPU, las 23 fotografías de m4/eval_set/HOJAS_M5.md

**Criterio de aprobación:** ≥ 9/11 nítidas transcritas sin error de número; ≥ 10/12 borrosas marcadas no fiables; ≤ 1/11 nítidas rechazadas por error

**Registra:** transcripción por caso frente a la de referencia, no_fiable por caso y su motivo, falsos positivos, latencia de lectura

---

### E5 · Flujo completo de revisión de foto

**Mide:** si con la lectura confirmada localiza el paso, si el verificador acierta, y si explica sin reescribir el procedimiento del estudiante

**NO mide:** nada que se pueda comparar con M4: configuración distinta y hojas distintas. No es una mejora de 2/12

**Conjunto:** el mismo de E4, con la transcripción de referencia confirmada

**Cómo:** `pendiente de notebook; TUTOR_CONFIG=revision_foto`

**Necesita:** GPU, las 23 fotografías, E4 aprobado

**Criterio de aprobación:** 0 errores inventados en los 3 casos `sin_error` —es el criterio que manda—; el paso señalado correcto en los 4 de localización y los 2 compensados; abstención en los 3 adversariales; 0 falsos positivos del verificador

**Registra:** paso señalado por caso, hallazgo del verificador, falso error, abstenciones apropiadas, si la respuesta reescribió los números del estudiante, latencia fría y caliente del flujo de dos pasos

---

### E6 · Muestra de la app desplegada

**Mide:** el harness sobre interacciones reales, con el protocolo de etiquetado ciego de docs/REFINAMIENTO_TUTOR_M5.md §6

**NO mide:** nada todavía. No hay URL pública, y sin ella una «muestra real» no existe

**Conjunto:** 50 interacciones consecutivas de una ventana con ≥ 50

**Cómo:** `python M5_despliegue/evaluacion/validacion_m5.py --muestra-url <URL>`

**Necesita:** una URL pública del Space, 50 interacciones reales, dos personas para el etiquetado ciego

**Criterio de aprobación:** se reporta n, ventana temporal, versión y commit. Sin esas cuatro cosas el número no significa nada

**Registra:** exactitud etiquetada a mano, acuerdo entre etiquetadores (Cohen κ), latencia fría y caliente reales, p50/p95 con n, errores y caídas

---

## Lo que falta ejecutar

| Experimento | Bloqueado por |
|---|---|
| E1 | GPU |
| E2 | parcial: la reproducción se puede correr sin GPU; el recall y la exactitud, no |
| E3 | GPU **y** casos de cálculo en un eval set, que no existen |
| E4 | GPU y las 23 fotografías de `m4/eval_set/HOJAS_M5.md` |
| E5 | lo de E4, más que E4 esté aprobado |
| E6 | una URL pública del Space y 50 interacciones reales |

Ninguno de estos resultados se simula. Mientras no se ejecuten, de M5 solo se puede decir que el sistema está montado y que la pieza de retrieval servida es la que se midió en S08.
