# Refinamiento del tutor antes de M5

Fecha: 29-10-2026, **revisado el 30-10-2026** · rama `integracion-m5` · sin push,
sin Space, sin merge a `main`.

Todo lo que sigue sale de archivos del repositorio. Donde no hay medición, se
dice que no la hay.

### Qué cambió en la segunda revisión del 30-10

Seis puntos más, después de la retroalimentación de M4 (nota **4,6/5,0**):

1. **La cifra medida se apaga cuando falta una pieza.** Antes la app enseñaba
   `18/21` aunque faltara el adaptador, el reranker o el enrutador. Ahora el
   estado lleva dos nombres —solicitada y efectiva— y en cuanto difieren se marca
   **MODO DEGRADADO · NO MEDIDO** (§3).
2. **La retroalimentación de M4, incorporada donde correspondía.** La causa era
   de entorno: `transformers>=4.45` sin tope y una caché desactualizada. Las
   versiones quedan fijadas e iguales a las del despliegue, con comprobación y
   registro del entorno dentro del resultado. Detalle en
   [`m4/FEEDBACK_M4.md`](../m4/FEEDBACK_M4.md). Los tres commits de referencia
   **no existen en este repositorio** y eso se dice ahí.
3. **El holdout visual estaba descrito mal.** `m4/README.md` y
   `m4/CONCLUSIONES_M4.md` ya decían que los 12 casos se habían gastado en
   corridas exploratorias; los documentos de M5 seguían llamándolos reservados.
   Corregido, con una prueba que impide que la frase vuelva, y con un conjunto
   **nuevo** de 12 hojas especificado en `m4/eval_set/HOJAS_M5.md`.
4. **El flujo de revisión de foto es de dos pasos.** Leer, enseñar, confirmar o
   corregir; y solo entonces diagnosticar, con el verificador determinista
   delante. `TUTOR_CONFIG=revision_foto`, experimental y sin medir (§3).
5. **La validación de M5 está separada en seis experimentos** que no se mezclan,
   cada uno con lo que mide, lo que **no** mide y su criterio
   ([`VALIDACION_M5.md`](../M5_despliegue/docs/VALIDACION_M5.md)). En particular,
   la comprobación de pesos y rutas del enrutador **no** es una medición de
   exactitud, y ahora está dicho en la cabecera del informe.
6. **La documentación coincide con el código**, y hay pruebas de las
   afirmaciones que se pueden comprobar: 360 en total.

### Qué cambió en la primera revisión del 30-10

Seis discrepancias de la versión anterior, corregidas aquí:

1. **La app no servía S08-D.** `pipeline.py` montaba `SYSTEM_C5_AGENTE`, búsqueda
   léxica y `qwen-lora-agente` si estaba en el disco: una variante de C5 (12/21)
   con el nombre de la mejor medida. Ahora las configuraciones se declaran y se
   eligen (§3), y el retrieval reproduce la cadena completa de S08-C con
   cross-encoder, que es sobre la que se construye `D · +router`.
2. **Las herramientas de cálculo no eran verificables de punta a punta.** Ahora
   hay 48 pruebas: resultados exactos, ambigüedad que pide aclaración, integrales
   divergentes rechazadas, topes de longitud y complejidad, y 5 pruebas por el
   bucle del agente con preguntas en español (§3).
3. **La tabla de cobertura mezclaba «antes» y «ahora»**, y presentaba tener una
   herramienta como cubrir el tema. Separadas, con lo que falta explícito (§2).
4. **La taxonomía se leía como una medición.** Es etiquetado heurístico; ahora
   hay una etiqueta principal por caso, `sin_evidencia` donde los datos no
   deciden, y se retira la afirmación de que entrenar eliminó los errores de
   cálculo (§2).
5. **Cada cifra lleva archivo, fila y columna** (§1), y las comparaciones sin
   respaldo se retiraron o se acotaron.
6. **Las ablaciones cambian una pieza por vez** sobre el eval set fijo de M2, y
   no se entrena otro adaptador hasta que digan qué error queda (§5).

---

## 1 · Resultados de M1–M4 y sus fuentes

Todas las filas con **21 casos** usan el mismo eval set fijo de M2
(`data/eval_set_m2.jsonl`: 12 cálculo, 4 conocimiento, 5 adversarial) y el mismo
harness, así que son **directamente comparables**. Las filas con otro conjunto se
marcan y no se comparan con ellas.

| Configuración | Modelo / adaptador | RAG | Herram. | Conjunto | Aciertos | Alucin. | s/caso |
|---|---|---|---|---|---|---|---|
| C0 · base | Qwen2.5-1.5B sin LoRA | — | — | 21 | 9/21 | 12 | 9,2 |
| **C1 · +LoRA** | **+ `qwen-lora` (M1)** | — | — | 21 | **14/21** | 7 | 5,5 |
| C2 · +RAG | + LoRA | ingenuo | — | 21 | 13/21 | 5 | 4,7 |
| C3 · +herram. | + LoRA | fijo | 15 disponibles | 21 | 13/21 | 8 | 3,6 |
| C4 · +ReAct | + LoRA | fijo | 15 disponibles | 21 | 14/21 | 7 | 3,4 |
| C5 · agente | + LoRA | herramienta | 16 disponibles | 21 | 12/21 | 9 | 5,0 |
| C5-FT · entrenado | + `qwen-lora-agente` | herramienta | 16 disponibles | 21 | 13/21 | 8 | 14,1 |
| S08-A · ingenuo | + LoRA | denso | — | 21 | 14/21 | 7 | n/d |
| S08-B · híbrido | + LoRA | BM25+denso RRF | — | 21 | 14/21 | 6 | n/d |
| S08-C · rerank | + LoRA | + cross-encoder | — | 21 | 13/21 | 5 | n/d |
| **S08-D · enrutador** | **+ LoRA** | **C + enrutador** | — | 21 | **18/21** | **2** | n/d |
| S08-E · cita | + LoRA | D + cita textual | — | 21 | 18/21 | 3 | n/d |
| M4 visual | + `qwen-lora-agente` + Qwen2-VL-2B | denso k=3 | 16 | **12 con imagen** | 2/12 | 3 | 13,0 |
| M4 · desarrollo | 3 configuraciones | — | — | **9 dev** | 3/9 · 2/9 · 2/9 | n/d | n/d |

**«Herramientas disponibles» no es «herramientas usadas».** En C3 y C4 el
registro estaba montado y el modelo casi no lo usó:
`scorecard_agente_v2.csv`, fila `Llamadas por pregunta` → C3 = 0,019; C4 = 0,0;
C5 = 0,189; C5-FT = 1,358.

### Auditoría de cada cifra · archivo, fila y columna

Las columnas de `scorecard_agente_v2.csv` son `C0 · base` … `C5-FT · entrenado`;
las de `scorecard_rag_avanzado_v2.csv` son `sin RAG`, `A · ingenuo`, `B · +hybrid`,
`C · +rerank`, `D · +router`, `E · +cita`. Todas en
`M3_rag_agentes/resultados/`.

| Cifra de arriba | Archivo | Fila | Columna |
|---|---|---|---|
| C0…C5-FT, aciertos | `scorecard_agente_v2.csv` | `Aciertos totales /21` | la de cada config |
| C0…C5-FT, alucinaciones | ídem | `Alucinaciones` | ídem |
| C0…C5-FT, s/caso | ídem | `Segundos por pregunta` | ídem |
| C1 = 8/13 y C5-FT = 10/13 en aritmética difícil (M3) | ídem | `Aritmética /13` | `C1 · +LoRA`, `C5-FT · entrenado` |
| S08-A…E, aciertos | `scorecard_rag_avanzado_v2.csv` | `aciertos_total` | la de cada sistema |
| S08-A…E, alucinaciones | ídem | `alucinaciones` | ídem |
| S08-D: cálculo 10/12 | ídem | `aciertos_calculo` | `D · +router` |
| S08-D: conocimiento 3/4 | ídem | `aciertos_conocimiento` | ídem |
| S08-D: adversarial 5/5 | ídem | `aciertos_adversarial` | ídem |
| S08-D: abstenciones 2 | ídem | `abstenciones` | ídem |
| M4 visual 2/12, 3 alucinaciones | `m4/scorecard_m4.csv` | 12 filas con `estado=evaluado` | columnas `acierto`, `alucino` |
| M4 desarrollo 3/9 · 2/9 · 2/9 | `m4/dev_dg_arnes_s13.csv`, `dev_dg_A_agente.csv`, `dev_dg_B_pasoN.csv` | 9 filas cada uno | columna `acierto` |
| Verificador: 4/4 pasos erróneos, 2/2 «sin error», 0 falsos positivos | `m4/verificador_aritmetico_dg.csv` | filas `dg-a06`, `dg-a12`, `dg-a18`, `dg-b06`, `dg-c04`, `dg-c08` | `marcado`, `falso_positivo` |
| 165 casos de entrenamiento en 11 categorías de 15 | `data/math_tutor_dataset.jsonl` | — | campo `categoria` |
| 18 herramientas en 7 familias | `scripts/agente/herramientas.py` | — | `REGISTRO.nombres()` |
| El enrutador servido reproduce el de S08-D | `M5_despliegue/evaluacion/verificacion_retrieval.json` vs `M3_rag_agentes/resultados/m3_rag_avanzado.json` | `enrutador.pesos`, `loo_dev`, `rutas_eval` | — |

Las celdas `n/d` son las que **no existen**: `scorecard_rag_avanzado_v2.csv` no
reporta segundos por caso, así que la latencia de los sistemas S08 no se compara
con la de C0–C5-FT. Antes esa columna llevaba un guión, que se lee como «cero».

### El hallazgo que decide qué se despliega

**La mejor configuración medida no es la que más módulos tiene.**

`S08-D` —LoRA de M1 + RAG híbrido con reranker + **enrutador**— marca **18/21**,
cuatro casos por encima de la mejor configuración agéntica (14/21 de C1), con
**2 alucinaciones frente a 7–9**. Desglose: cálculo 10/12, conocimiento 3/4,
adversarial **5/5**.

Añadir herramientas y ReAct sobre ese mismo eval set **no mejoró**: C3 y C5 bajan
a 13/21 y 12/21. El enrutador, que solo decide *si buscar antes de responder*,
aporta más que todo el andamiaje agéntico.

**Los dos scorecards se pueden comparar entre sí** porque comparten una fila: la
configuración «LoRA de M1 sin RAG» aparece en los dos y da lo mismo en ambos —
14/21 y 7 alucinaciones (`scorecard_agente_v2.csv` columna `C1 · +LoRA`;
`scorecard_rag_avanzado_v2.csv` columna `sin RAG`). Sin esa coincidencia, poner
18/21 y 14/21 en la misma tabla no estaría justificado.

### Limitaciones de la comparación

- **21 casos.** Una diferencia de uno o dos no distingue nada; cuatro es señal,
  pero con intervalo ancho. Ninguna de estas filas lleva intervalo de confianza
  calculado.
- **Latencia no comparable entre scorecards**: S08 no reporta segundos por caso.
- **Los 53 casos de M3** (21 de M2 + 32 propios) **no son comparables** con los
  21: ahí C5-FT sí gana en el bloque de aritmética difícil (10/13 frente a 8/13
  de C1), que es para lo que se entrenó.
- **M4 no es comparable con nada**: tarea distinta, conjunto distinto. Y sus 12
  casos **ya se gastaron**: se ejecutaron en tres corridas exploratorias —4/12,
  1/12 y 2/12— según `m4/README.md` y `m4/CONCLUSIONES_M4.md`. Son evidencia
  histórica y conjunto de regresión; **no** un holdout ciego. Una versión anterior
  de este informe decía lo contrario, contradiciendo a los documentos de M4.
- **Ningún sistema con herramientas de cálculo está en esta tabla**, porque las
  tres herramientas nuevas se añadieron después de todas estas corridas.

---

## 2 · Análisis de errores · exploratorio

Generado con `scripts/taxonomia_errores.py` sobre `respuestas_s10.jsonl` (7
configuraciones × 53 casos) y `m4/scorecard_m4.csv`. Detalle completo en
[`taxonomia_errores.md`](taxonomia_errores.md).

> **Esto no es una medición.** Las cifras del arnés —`acierto`, `abstuvo`,
> `alucino`— son las de §1 y salen de los scorecards. Lo de esta sección son
> **etiquetas heurísticas** aplicadas a respuestas ya guardadas para orientar
> dónde mirar: nadie anotó estos 371 casos a mano. Una etiqueta es una hipótesis
> sobre la causa, no un veredicto.

Tres reglas que cambian la lectura de la tabla:

1. **Una etiqueta principal por caso.** La columna suma exactamente 53, así que
   un caso con tres señales sigue contando como un fallo. Antes se contaban las
   etiquetas, no los casos, y los totales salían inflados.
2. **`sin_evidencia`** para lo que los datos guardados no permiten decidir:
   `contexto_irrelevante` (no se guardó el pasaje de oro por caso),
   `mala_eleccion_de_herramienta` (no hay herramienta correcta anotada por caso)
   y `lectura_visual_erronea` (no hay transcripción anotada a mano).
3. **Hay etiquetas condicionadas**, que solo pueden aplicarse a cierto tipo de
   respuesta. Ver un cero ahí no es una mejora.

| Etiqueta principal | C1 · +LoRA | C5-FT · entrenado | Tipo |
|---|---|---|---|
| `aritmetica_falsa` | 7 | 4 | comprobable |
| `calculo_sin_herramienta` | 12 | 0 | **condicionada** |
| `planteamiento_incorrecto` | 0 | 14 | heurística |
| `no_consulto_documentos` | 4 | 0 | comprobable |
| `fallo_con_contexto_recuperado` | 0 | 1 | comprobable |
| `no_reconocio_el_limite` | 1 | 1 | comprobable |
| `negativa_innecesaria` | 2 | 0 | comprobable |
| `explicacion_confusa` *(acierto, no fallo)* | 2 | 3 | comprobable |
| `acierto` | 25 | 30 | del arnés |
| **total de casos** | **53** | **53** | |
| **casos fallados** | **26** | **20** | |

### La corrección que invalida la conclusión anterior

La versión previa de este informe afirmaba que entrenar con trayectorias
**«eliminó por completo los errores de cálculo, de 19 a 0»**. Eso no se sostiene,
y el motivo es de método: `calculo_sin_herramienta` exige que la respuesta **no
haya llamado a ninguna herramienta**. Una configuración que llama casi siempre no
puede recibir esa etiqueta nunca, por construcción.

| Configuración | casos con ≥1 llamada | de 53 | % |
|---|---|---|---|
| C1 · +LoRA | 0 | 53 | 0 % |
| C5 · agente | 10 | 53 | 19 % |
| C5-FT · entrenado | 50 | 53 | 94 % |

Fuente: `respuestas_s10.jsonl`, campo `observaciones`; coincide con
`detalle_s10_v2.csv`, columna `llamadas`, y con `scorecard_agente_v2.csv` fila
`Llamadas por pregunta` (C1 = 0,0; C5 = 0,189; C5-FT = 1,358). El cero de C5-FT
mide **que llamó a herramientas**, no que acertara las cuentas.

Lo que sí se puede comparar, porque no depende de las llamadas:
`aritmetica_falsa`, que da el verificador determinista
(`scripts/agente/verificador_aritmetico.py`) leyendo las igualdades escritas en
la respuesta. Ahí C5-FT baja de 7 a 4 sobre 53 casos: una diferencia de tres
casos, que con esta n no distingue nada. Su límite: solo ve igualdades que el
tutor escribió explícitamente.

Y lo medido de verdad, del scorecard: en el bloque de aritmética difícil de M3,
`scorecard_agente_v2.csv` fila `Aritmética /13` da **C1 = 8/13 y C5-FT = 10/13**;
en los 21 casos de M2, fila `Aciertos totales /21`, **C1 = 14 y C5-FT = 13**.
Entrenar con trayectorias ayudó donde se entrenó y no fuera de ahí.

### M4 · 12 casos visuales

7 `sin_conclusion_formato`, 2 aciertos, y **tres adversariales distintos**, que la
versión anterior agrupaba mal como «imagen ilegible». La columna `esperado` de
`m4/scorecard_m4.csv` dice cuál es cada uno:

| Caso | `esperado` | Etiqueta |
|---|---|---|
| `vis-10` | ilegible | `respuesta_a_imagen_ilegible` |
| `vis-11` | fuera de dominio | `respuesta_fuera_de_dominio` |
| `vis-12` | dato no disponible | `dato_no_disponible` |

Solo `vis-10` es una hoja ilegible. `sin_conclusion_formato` **no es un fallo de
criterio**: el tutor no llegó a emitir un veredicto, y eso se arregla de otra
manera. Si la lectura de la hoja fue correcta o no, estos datos no lo dicen: haría
falta la transcripción anotada a mano, que no existe.

### Cobertura temática · antes y ahora, separados

La versión anterior de esta tabla mezclaba dos cosas distintas: qué había antes
de esta fase y qué hay después de añadir tres herramientas. Separadas:

| Tema | Herramientas antes | Herramientas ahora | Datos de entrenamiento (M1) | Casos en un eval set |
|---|---|---|---|---|
| Aritmética | 5 | 5 | 75 | sí (M2 y M3) |
| Fracciones | 2 | 2 | 15 | sí |
| Porcentajes | 3 | 3 | 15 | sí |
| Potencias y raíces | 2 | 2 | 15 | sí |
| Álgebra (1.º grado) | 2 | 2 | 15 | sí |
| Estadística | 1 | 1 | 15 | sí |
| Geometría | **0** | **0** | 15 | no |
| **Derivadas** | **0** | **1** (`derivar`) | **0** | **no** |
| **Integrales** | **0** | **2** (`integrar`, `integral_definida`) | **0** | **no** |

Conteos verificables: `data/math_tutor_dataset.jsonl` tiene 165 casos en 11
categorías de 15 (aritmética = suma + resta + multiplicación + división +
operaciones combinadas = 75); el registro tiene 18 herramientas en 7 familias
(`REGISTRO.nombres()`); `data/eval_set_m2.jsonl` tiene 21 casos —12 cálculo, 4
conocimiento, 5 adversarial— y ninguno de cálculo diferencial.

**Tener la herramienta no es cubrir el tema.** Lo que hay hoy y lo que falta:

| Hecho | Estado |
|---|---|
| Las tres herramientas calculan exacto | **comprobado** · 48 pruebas en `scripts/tests/test_calculo.py` |
| Rechazan ambigüedad, divergencia y expresiones desmedidas | **comprobado** · mismas pruebas |
| Los esquemas llegan al modelo en el bucle ReAct | **comprobado** con un LLM de guion |
| El modelo las elige al leer «deriva 3x² + 5x − 2» | **sin medir** · requiere GPU |
| Hay casos de cálculo en un eval set | **no** · cero |
| Hay datos de entrenamiento de cálculo | **no** · cero |

Las pruebas de punta a punta usan un LLM sustituido por un guion. Prueban la
fontanería completa —esquemas visibles, argumentos validados, observación
devuelta, respuesta final apoyada en el valor de la herramienta— pero **no** que
Qwen2.5-1.5B decida llamar a `derivar` cuando lee la pregunta. Eso sigue sin
medirse, y es la diferencia entre «la herramienta existe» y «el tema está
cubierto».

---

## 3 · Cambios realizados y por qué

### Tres herramientas de cálculo · implementadas y probadas

`derivar`, `integrar` e `integral_definida` en `scripts/agente/herramientas.py`,
familia `calculo`. El registro pasa de 15 a **18 herramientas** en 7 familias.

Por qué estas y por qué ahora: era **la mejora de menor coste disponible**. SymPy
ya es dependencia y `parsear()` ya lo usa con lista blanca de símbolos, así que
son deterministas, exactas y no requieren entrenar nada.

```
derivar('3*x**2 + 5*x - 2')          -> 6*x + 5
derivar('sqrt(x)')                   -> 1/(2*sqrt(x))     exacto, no 0.5/sqrt(x)
integrar('2*x')                      -> x**2 + C
integral_definida('x**2', 0, 1)      -> 1/3               no 0.333…
integral_definida('1/x', 1, 2)       -> log(2)            no 0.693…
```

**Y, sobre todo, lo que rechazan.** Una herramienta que siempre devuelve algo es
una fuente de alucinación con aire de autoridad:

```
derivar('a*x**2 + b')        -> error: varias variables (a, b, x) y no dijiste
                                respecto a cuál operar  ← pide aclaración
integral_definida('1/x',0,1) -> error: no converge: no hay un número que dar
integrar('x**x')             -> error: no sé resolver esa integral de forma exacta
derivar('x**200')            -> error: el exponente 200 pasa del tope 50
derivar('x+x+…' 241 car.)    -> error: la expresión tiene 241 caracteres, tope 200
derivar(variable='w')        -> error: variable no permitida
derivar('__import__(1)')     -> error: símbolos no permitidos
```

Los topes (`MAX_EXPRESION=200`, `MAX_OPERACIONES=40`, `MAX_EXPONENTE=50`) no son
estéticos: `sympy.integrate` sobre una expresión suficientemente grande no
termina, y una demo colgada es peor que una herramienta que dice que no puede. Hay
una prueba que comprueba que los topes **no** estorban a un ejercicio escolar
normal, y otra que mide que el tiempo queda acotado.

La ambigüedad se trata como ambigüedad: el valor por omisión de `variable` es
vacío, no `"x"`. Con una sola variable se deduce; con varias se pide aclaración.
Antes `derivar('a*x**2 + b')` devolvía `2*a*x` en silencio, eligiendo la pregunta
en nombre del estudiante.

**48 pruebas** en `scripts/tests/test_calculo.py`, de las que 5 son de punta a
punta por el bucle del agente con preguntas en español y un LLM de guion.

### El verificador aritmético, conectado

`cfg.verificador` ya no es una etiqueta: `Tutor._reintentar_con_verificador`
pasa la respuesta por `revisar_hoja`, y si encuentra una igualdad falsa o un
valor trasladado mal, reintenta **una sola vez** añadiendo
`pista_para_el_tutor(h)` a la pregunta. Una pasada, para que el coste esté
acotado y la ablación tenga un único cambio que medir. El verificador no dicta el
veredicto: describe la igualdad que falla y el modelo decide.

Sigue **sin medirse en el flujo completo**: lo que está medido es el verificador
en aislado, 4/4 pasos erróneos y 0 falsos positivos sobre los 9 casos de
desarrollo de M4 (`m4/verificador_aritmetico_dg.csv`).

### Configuraciones separadas y declaradas

`M5_despliegue/app/configuraciones.py` declara seis configuraciones con nombre.
Cada una dice qué adaptador, qué retrieval, qué herramientas y **qué resultado
medido la respalda, o que no hay ninguno**. Se elige con `TUTOR_CONFIG`; por
omisión, `s08d`.

Esto corrige un fallo real del `pipeline.py` anterior, que montaba
`SYSTEM_C5_AGENTE`, el registro completo de herramientas y **preferencia
automática por `qwen-lora-agente` si estaba en el disco**. Eso es una variante de
C5 —12/21, la peor de las agénticas— servida bajo el título de la mejor medida.
Elegir la configuración por qué archivo hay en el disco es la forma más rápida de
desplegar algo distinto de lo que se midió.

Dos discrepancias más que aparecieron al revisar esto:

- **El retrieval no era el de S08-D.** `D · +router` se construye sobre
  `C · +rerank` (`scripts/nb_rag_avanzado.py:559`), así que incluye el
  cross-encoder `mmarco-mMiniLMv2`. El `buscador_hibrido` de la app solo hacía
  BM25 + denso con RRF. Ahora reproduce la cadena completa: híbrida top-10 →
  cross-encoder → top-3.
- **El enrutador se ajustaba con dos rasgos de tres.** `Enrutador` usa margen de
  centroides, nº de números y **puntaje máximo del reranker**
  (`scripts/agente/enrutador.py`, `RASGOS`). La app pasaba `rerank_max=None`, que
  deja el tercer rasgo en 0,0 y ajusta otro modelo con el mismo nombre.
- **Los embeddings no iban normalizados.** El notebook llama a
  `st.encode(..., normalize_embeddings=True)` (`scripts/nb_rag_avanzado.py:355`);
  la app no. El enrutador promedia esos vectores para sus centroides y hace
  producto punto con la pregunta, así que sin normalizar el rasgo `margen` sale
  en otra escala y la regresión ajusta otro modelo.
- **Faltaban dependencias en `requirements.txt`.** Sin `sentence-transformers`,
  `rank-bm25` y `scikit-learn`, el Space se replegaba a búsqueda léxica sin
  enrutador: no es S08-D. Van fijadas. `bitsandbytes` lleva marcador
  `sys_platform == "linux"` porque no hay rueda para macOS y sin él
  `pip install -r` falla entero fuera del Space.

### Y se comprueba, no se afirma

`M5_despliegue/evaluacion/verificar_retrieval.py` monta el retrieval y el
enrutador de la app en CPU, sin LLM, y los compara contra las cifras que el
notebook dejó en `M3_rag_agentes/resultados/m3_rag_avanzado.json`:

```
pesos de la regresión logística
  margen   +1.7977  referencia +1.7977  ok
  numeros  -1.3384  referencia -1.3384  ok
  rerank   +0.5682  referencia +0.5682  ok
leave-one-out sobre las 33 consultas de desarrollo
  loo_dev                  0.9697  referencia 0.9697  ok
  loo_dev_solo_centroides  0.7879  referencia 0.7879  ok
ruta de cada caso del eval set fijo de M2
  21 de 21 coinciden con la referencia
```

Resultado guardado en `M5_despliegue/evaluacion/verificacion_retrieval.json`.
**Lo que esto demuestra y lo que no:** demuestra que el enrutador servido es el
medido —mismos pesos, misma validación, mismas 21 decisiones—; **no** mide
aciertos, porque el LLM no se carga. Para los 18/21 sigue haciendo falta una
corrida con GPU.

Así se encontraron las dos discrepancias anteriores. Con `rerank_max=None` o sin
normalizar los embeddings, la exactitud leave-one-out bajaba a **0,939**:
parecida a 0,970, y otra cosa. Esa es exactamente la clase de diferencia que no
se nota mirando la pantalla.

### La salud del servicio dice lo que hay

`salud()` reporta configuración, resultado medido o «EXPERIMENTAL, sin medir»,
modelo base, adaptador **pedido** y adaptador **cargado**, retrieval activo con su
repliegue si lo hubo, enrutador, herramientas activas y cuántas existen en el
registro, verificador, visión, cuantización y dispositivo. Y marca el montaje
como `declarado` mientras el modelo no se haya cargado, porque antes de cargarlo
esos campos son el plan de la configuración, no algo comprobado.

La pestaña de visión solo aparece si la configuración activa la incluye. En
`s08d` no existe: ofrecerla sugeriría que la vía visual forma parte de lo que
midió 18/21, y nunca se midió junto con ese retrieval.

### Modo degradado · lo que faltaba en la primera revisión

La primera revisión separó las configuraciones, pero dejó un hueco: la app
seguía enseñando `18/21` aunque en la ejecución faltara el adaptador, el
reranker o el enrutador. Reportar bien la configuración **solicitada** no sirve
si lo que responde es otra cosa.

Ahora el estado lleva dos nombres y una lista:

```
configuracion_solicitada  S08-D · LoRA M1 + RAG híbrido + enrutador
configuracion_efectiva    MODO DEGRADADO (se pidió S08-D; falta retrieval
                          híbrido + reranker, enrutador consulta/problema)
degradado                 True
componentes_faltantes     retrieval híbrido + reranker, enrutador consulta/problema
resultado_medido          NO MEDIDO: ejecución degradada, falta […]. La cifra
                          18/21 es de «S08-D …» con todas sus piezas y no
                          describe lo que responde ahora
```

Un componente entra en `componentes_faltantes` cuando la configuración lo pide y
no se materializa: adaptador LoRA, retrieval híbrido + reranker, enrutador,
registro de herramientas, verificador o normalizador. Con la lista no vacía,
`medido_aplicable` es `None` y ni la cabecera, ni el resumen, ni la salud citan
la cifra como propia de esa ejecución.

**El repliegue se conserva.** Responder por búsqueda léxica es mejor que no
responder, y la interfaz enseña qué pieza está funcionando una por una. Lo que
deja de ocurrir es que ese repliegue se llame S08-D.

Por qué importa aquí y no es teórico: en este disco `qwen-lora-agente` y
`qwen-lora-diagnostico` **no están** —solo `qwen-lora`—, así que
`TUTOR_CONFIG=agente_c5ft` arrancaría degradada hoy mismo.

Siete pruebas, una por cada forma de fallar: recuperador ausente, reranker
ausente (que arrastra al enrutador), enrutador que no arranca, adaptador
ausente, el caso completo, la salud y la cabecera. Para que fueran posibles sin
descargar 3 GB de pesos, la resolución de componentes se extrajo a
`Tutor.resolver_componentes()`, que no necesita el modelo.

### El flujo de revisión de una foto · dos pasos

En M4 leer y diagnosticar era una sola llamada, y `vis-10` mostró el coste: el
VLM inventó «24 × 13 = 306» —dos lecturas byte a byte idénticas, así que la
salvaguarda de doble lectura no lo detectó— y el tutor diagnosticó sobre ese
invento con aplomo. Mientras el estudiante no vea y confirme lo que se leyó, un
diagnóstico es una afirmación sobre una hoja que quizá no existe.

```
leer_imagen(foto)        -> transcripción + pregunta de confirmación.
                            NUNCA un veredicto. Si la lectura no es fiable, se
                            detiene y dice el motivo observable.
revisar_pasos(pasos, confirmado=True)
                         -> verificador determinista -> evidencia -> tutor
```

Cuatro decisiones, y cada una responde a algo que pasó:

- **Los pasos llegan editables.** Si el tutor leyó mal un número, el error que
  señalaría sería suyo. Corregir la transcripción es parte del flujo.
- **Sin `confirmado=True` no hay diagnóstico.** `revisar_imagen`, la firma de
  M4, se conserva por compatibilidad y **se detiene en la lectura**.
- **El verificador va primero.** Comprueba las igualdades con `Fraction` y su
  hallazgo entra como *evidencia*, no como veredicto: «el paso 2 no cuadra: 3 ×
  24 da 72 y está escrito 82». Qué paso señalar sigue siendo del tutor. Cuando
  las cuentas sí cuadran también se le dice, para que no invente un error
  aritmético donde no lo hay.
- **No hay puntuación de confianza inventada.** Lo que decide si se puede seguir
  son las cuatro señales observables de `vision.lectura_no_fiable` y, cuando no
  alcanzan, una pregunta.

`TUTOR_CONFIG=revision_foto`. Es una configuración de **producto**, no una fila
de ablación: cambia dos cosas a la vez respecto a `s08d` porque leer, confirmar,
comprobar y explicar es un flujo que no tiene sentido partir en mitades. Para
aislar cada pieza están `s08d_vision` y `s08d_verificador`. **S08-D sigue
intacto** como baseline textual: sin visión, sin verificador, sin herramientas.

### Herramientas de análisis añadidas

- `scripts/taxonomia_errores.py` — etiquetado exploratorio con una etiqueta
  principal por caso y `sin_evidencia` donde los datos no deciden.
- `scripts/empaquetar_entrega.py` — ya existía de la fase anterior.

---

## 4 · ¿Hace falta entrenar? · evidencia

**Para cálculo: no, y entrenar sería un retroceso.** Las tres herramientas dan el
resultado exacto y comprobable. Entrenar un modelo de 1.5B para derivar cambiaría
una respuesta verificable por una generada. Lo que sigue abierto no es el cálculo
sino **la decisión de llamar a la herramienta**, y eso no se arregla con datos de
derivadas: C5-FT llama en 94 % de los casos y C1 en 0 %, con 13/21 y 14/21
respectivamente.

**Para el planteamiento: todavía no está justificado, y hay evidencia en contra
de insistir por esa vía.** M4 lo intentó: 42 trayectorias de diagnóstico, un
adaptador nuevo (`qwen-lora-diagnostico`), y el resultado sobre los 9 casos de
desarrollo fue **3/9** frente a **2/9** del adaptador anterior — dentro del ruido
con esa n. Peor: cambiar `Paso 2` por `Paso N` en el prompt movió la conclusión
en **3 de 9 casos sin que la hoja cambiara** (`m4/dev_dg_B_pasoN.csv` frente a
`m4/dev_dg_A_agente.csv`), lo que indica que el modelo no está comparando, sino
emitiendo un índice casi arbitrario.

**Lo barato que queda por agotar, en este orden:**

| Vía | Estado | Coste |
|---|---|---|
| Enrutador | **medido**: 14/21 → 18/21 | ya hecho |
| Herramientas selectivas sobre S08-D | implementado, sin medir | una corrida |
| Verificador aritmético en el flujo | conectado ahora, sin medir | una corrida |
| Normalizador de texto mal escrito | existe en `nucleo.py`, sin medir por separado | una corrida |
| Otro adaptador | **no justificado** hasta que las tres de arriba digan qué queda | GPU + datos |

El criterio para pasar de fila: una vía solo se descarta cuando su ablación está
medida sobre los 21 casos, no cuando parece poco prometedora.

---

## 5 · Lo que se sirve hoy y lo que falta medir

### Lo que se sirve por omisión: S08-D, sin añadidos

```
TUTOR_CONFIG=s08d        ← por omisión
Qwen2.5-1.5B-Instruct
  + LoRA de M1 (qwen-lora)
  + RAG híbrido BM25 + denso (RRF k=60) + cross-encoder mmarco
  + enrutador consulta/problema (3 rasgos, 33 consultas de desarrollo)
  sin herramientas · sin verificador · sin visión
```

**18/21 · 2 alucinaciones**, fuente `scorecard_rag_avanzado_v2.csv`, fila
`aciertos_total` y fila `alucinaciones`, columna `D · +router`. Es la mejor cifra
medida del proyecto y lo que la app sirve tal cual, sin piezas añadidas por
encima.

**Respaldo simple**: `TUTOR_CONFIG=solo_lora`, LoRA de M1 sin RAG ni herramientas,
**14/21**. Las dos mediciones de esa misma configuración coinciden —
`scorecard_agente_v2.csv` fila `Aciertos totales /21` columna `C1 · +LoRA`, y
`scorecard_rag_avanzado_v2.csv` columna `sin RAG`: 14/21 y 7 alucinaciones en
ambas—, lo que es también la razón por la que los dos scorecards se pueden
comparar entre sí.

### Las variantes experimentales, que NO son resultados

Montables con `TUTOR_CONFIG`, declaradas con `medido=None`, y marcadas en
pantalla como «EXPERIMENTAL, sin medir»:

| `TUTOR_CONFIG` | Qué añade sobre S08-D | Medido |
|---|---|---|
| `s08d_herramientas` | las 18 herramientas como tool calling (obliga a cambiar el prompt) | **no** |
| `s08d_verificador` | fila anterior + verificador aritmético con un reintento | **no** |
| `s08d_normalizador` | el `Normalizador` de preguntas mal escritas | **no** |
| `s08d_vision` | la vía visual, y nada más · fila de ablación | **no** |
| `revision_foto` | **producto**: foto + confirmación + verificador | **no** |

Y una más que **sí** está medida, porque sirve de control: `s08c`, que es S08-D
sin el enrutador, **13/21** con 5 alucinaciones
(`scorecard_rag_avanzado_v2.csv`, columna `C · +rerank`).

De `s08d_vision` y `revision_foto` conviene decir qué cifras **no** les
corresponden: los 4/12, 1/12 y 2/12 de las corridas exploratorias y los 3/9 de
desarrollo se midieron con otra configuración —adaptador `agente` y RAG denso—,
no con estas. Y los 12 casos con que se obtuvieron ya se gastaron, así que
tampoco sirven para medirlas: hacen falta las hojas nuevas.

`revision_foto` no es una fila de ablación. Cambia visión **y** verificador a la
vez, porque leer, confirmar, comprobar y explicar es un flujo que no tiene
sentido partir en mitades. Para aislar el efecto de cada pieza están las dos
filas anteriores.

Ninguna combinación de estas se ha medido nunca. No son una candidata ganadora:
son hipótesis con un nombre para poder medirlas de una en una.

### Ablaciones · un cambio por vez, criterios fijados ANTES de medir

El eval set es el **fijo de M2** (`data/eval_set_m2.jsonl`, 21 casos), sin tocar.
Los 12 casos visuales de M4 ya están gastados, así que aquí no se usan para
nada: estas filas son de texto. Cada fila
cambia **una sola pieza** respecto a la anterior, para que la diferencia sea
atribuible:

Preparadas en `M5_despliegue/evaluacion/ablaciones.py`, que usa el **mismo
harness** de `nb_comun_eval.CODIGO_HARNESS`. `--plan` enseña las filas sin cargar
ningún modelo; sin GPU, el script dice que no puede y no finge nada.

| # | `TUTOR_CONFIG` | Única diferencia con la referencia | Ref. | Medido ya | Hipótesis | Qué la refutaría |
|---|---|---|---|---|---|---|
| 0 | `s08d` | — (reproducción) | — | **18/21** | 18/21, 2 alucinaciones | si no reproduce, el entorno no es el medido y lo demás no vale |
| 1 | `solo_lora` | se quita el RAG (y con él el enrutador) | 0 | **14/21** | 14/21 | si no da 14/21, ídem |
| 2 | `s08c` | se quita solo el enrutador | 0 | **13/21** | 13/21 | control: debe reproducir la columna `C · +rerank` |
| | | | | | | *(las filas 3 a 5 siguen; la ablación de visión va aparte, con el eval set visual nuevo)* |
| 3 | `s08d_herramientas` | las 18 herramientas **y el prompt del agente** | 0 | no | ≥ 18/21; mejora en los 12 de cálculo | si baja de 18/21, estorban — pero no se sabrá si fue por las herramientas o por el prompt |
| 4 | `s08d_verificador` | el verificador (1 reintento) | 3 | no | ≥ fila 3; menos `aritmetica_falsa` | si no baja `aritmetica_falsa`, el verificador no aporta en el flujo |
| 5 | `s08d_normalizador` | el `Normalizador` de `nucleo.py` | 0 | no | mejora en el bloque `mal_escrito` | si no mejora ahí, la normalización no es la vía |

**Una excepción honesta a la regla de un cambio por fila.** La fila 3 cambia dos
cosas: las herramientas y el prompt (`SYSTEM_RAG` → `SYSTEM_C5_AGENTE`). No se
pueden separar: sin un prompt que explique el formato de llamada el modelo no
llama a nada, así que «herramientas con el prompt de RAG» no es una condición
medible. Si la fila 3 baja, la causa puede ser cualquiera de las dos, y así hay
que reportarlo. Las filas 4 y 5 sí cambian una sola pieza; hay una prueba que lo
comprueba (`test_cada_fila_de_ablacion_cambia_una_sola_pieza`).

Qué se reporta de cada fila: aciertos /21, el desglose cálculo /12 ·
conocimiento /4 · adversarial /5, alucinaciones, abstenciones, segundos por caso
y en cuántos casos llamó a una herramienta. Las filas 0 a 2 son controles: si no
reproducen lo ya medido, el script devuelve código 2 y el resto de la tabla no se
interpreta.

**Lo que ya se pudo comprobar sin GPU**: el enrutador de la fila 0 reproduce
exactamente el de S08-D —pesos, leave-one-out y las 21 rutas— según
`verificar_retrieval.py`. Eso no es un acierto medido; es que la pieza servida es
la medida.

Para declarar que algo supera a S08-D hacen falta las tres cosas a la vez:
**> 18/21**, **≤ 2 alucinaciones** y **5/5 en adversarial**. Con 21 casos, una
diferencia de uno o dos no distingue nada; la tabla se publica con esa advertencia
al lado, no debajo.

### Lo que no se va a hacer todavía

**No se entrena otro adaptador.** Primero las filas 3 a 5, que no cuestan
entrenamiento, dicen qué error sistemático queda. M4 ya hizo el experimento
contrario: 42 trayectorias de diagnóstico y un adaptador propio
(`qwen-lora-diagnostico`) dieron 3/9 frente a 2/9 en desarrollo, dentro del ruido.

---

## 6 · Protocolo para evaluar respuestas reales del Space

1. **Muestreo.** 50 interacciones consecutivas de una ventana con ≥ 50
   solicitudes. Sin cherry-picking.
2. **Etiquetado ciego.** Dos personas etiquetan por separado con las categorías
   de §2, sin ver las métricas del sistema. Se reporta el acuerdo (Cohen κ).
3. **Las cuatro categorías que no se mezclan**, porque se arreglan distinto:
   **abstención correcta** (ambigua o ilegible → no es error), **negativa
   innecesaria**, **fallo técnico** (timeout, modelo caído) y **respuesta
   matemática incorrecta**.
4. **Verificación.** Para cada caso con respuesta numérica, se comprueba con las
   herramientas deterministas; quién verificó queda anotado.
5. **Reporte.** n, ventana temporal, versión del modelo y commit. Sin eso, el
   número no significa nada.

---

## 7 · Pendientes · qué falta y de quién depende

**Se puede hacer sin GPU y está hecho**
- Configuraciones declaradas, salud honesta, retrieval de S08-C completo,
  enrutador con sus tres rasgos, dependencias fijadas.
- Herramientas de cálculo con sus topes y sus rechazos · 48 pruebas.
- Verificador conectado al flujo con un reintento.
- 330 pruebas pasando en el entorno fijado de `requirements.txt`.

**Requiere GPU · Colab**
- **Primero**, el experimento A de [`m4/FEEDBACK_M4.md`](../m4/FEEDBACK_M4.md):
  ¿se reproducen las 11 transcripciones con las versiones fijadas? Es la
  comprobación de que el arreglo del entorno funcionó, y la más barata. Si falla,
  no se incorpora nada más.
- Las 6 filas de ablaciones de §5 sobre los 21 casos. Las filas 0 a 2 son
  controles: si `s08d` no reproduce 18/21 y `solo_lora` no reproduce 14/21, el
  resto no se interpreta.
- Medir si el modelo **elige** `derivar` / `integrar` al leer una pregunta en
  español. Hoy está probado que las herramientas funcionan y que sus esquemas le
  llegan; no que las use.
- `medir_eficiencia.py`: fp16 vs int4, memoria, p50/p95, tokens/s. En CPU no se
  puede: `bitsandbytes` no tiene rueda para macOS.
- La evaluación visual de M5: 23 fotografías de 12 hojas nuevas
  (`m4/eval_set/HOJAS_M5.md`, generado por
  `scripts/eval_set_m5_vision_fuente.py`). 11 pares legible/borrosa más el
  ilegible real, con transcripción de referencia anotada. Es lo que permite
  medir la salvaguarda de ilegibles sin inventar un umbral, y separar un fallo de
  lectura de un fallo de razonamiento.

**Requiere decisión o cuenta ajena**
- Hugging Face Spaces: crear el Space, subir app y dependencias, variables.
  **Probar la URL desde otro dispositivo y otra red** antes de la demo.
- W&B: proyecto y clave. Ya está probado que la app funciona sin ella.
- Quién escribe y fotografía las 12 hojas nuevas. Sin ellas, de la vía visual
  solo se puede decir que está implementada: los 12 casos de M4 ya se gastaron y
  reutilizarlos para afirmar una mejora no distinguiría generalización de
  memorización.

---

## 8 · Calendario

| Día | Trabajo | Entregable |
|---|---|---|
| ~~1~~ | ~~Conectar enrutador + herramientas + verificador en `pipeline.py`~~ · **hecho el 30-10**; falta medir | configuraciones declaradas + 330 pruebas |
| **1'** | Correr las 6 filas de ablaciones de §5 en Colab, empezando por los 3 controles | tabla de ablaciones con controles verificados |
| **2** | Cuantización int4 y `medir_eficiencia.py`; scorecard de eficiencia | `eficiencia_fp16_vs_int4.json` |
| **3** | Subir el Space, health check, logs JSONL, W&B | URL interna |
| **4** | Ensayar los tres casos; capturas reales; cerrar el guion | guion con huecos rellenos |
| **5 · pruebas** | **URL desde otro dispositivo y otra red**; carga; fallos provocados (modelo caído, retriever caído, herramienta que falla) | informe de pruebas |
| **6 · pruebas** | 50 interacciones reales + etiquetado ciego (§6); p50/p95 con n declarado | scorecard de producción |

Los días 5 y 6 son **solo pruebas**: no se implementa nada nuevo. Si el día 1'
los controles no reproducen 18/21 y 14/21, el calendario se detiene ahí: servir
una configuración que no reproduce lo medido es el problema que esta revisión
vino a arreglar.

---

## 9 · Preguntas de defensa, con evidencia

**¿Latencia p50 y p95, y en qué condiciones?**
Pendiente de medir. Lo que hay: segundos por caso de M3 —C1 5,5 s; C5-FT 14,1 s—
sobre 21 casos en una T4, fuente `scorecard_agente_v2.csv`. Con 21 casos y 3
repeticiones son 63 mediciones: el p95 es una cota superior ruidosa, no un SLO.

**¿Y si se corre el harness sobre una muestra real del Space?**
Protocolo en §6: 50 interacciones, etiquetado ciego por dos personas con acuerdo
reportado. No se ha ejecutado.

**¿Cuáles son los errores más frecuentes?**
Con cuidado, porque esto es etiquetado heurístico y no una medición (§2). De 53
casos, C1 falla 26 y C5-FT 20. En C1 la etiqueta principal más común es
`calculo_sin_herramienta` (12); en C5-FT es `planteamiento_incorrecto` (14).
**Lo que no se puede decir** es que entrenar eliminara los errores de cálculo:
esa etiqueta exige que la respuesta no llamara a ninguna herramienta, y C5-FT
llama en 94 % de los casos, así que no puede recibirla por construcción. La
señal independiente de las llamadas es `aritmetica_falsa`, del verificador
determinista: 7 en C1 frente a 4 en C5-FT, tres casos de diferencia sobre 53, que
con esta n no distingue nada.

**¿Cuándo se niega, y cómo sabemos que la negativa es correcta?**
S08-D acierta **5/5** adversariales con 2 alucinaciones. La abstención se detecta
con marcadores específicos —`MARCADORES_ABSTENCION` en el harness— y se cuenta
aparte del acierto: "no lo sé" no es correcto, pero tampoco es alucinar.

**¿Qué aportó cada componente por separado?**
Lo medido, con la fila y la columna exactas de `scorecard_rag_avanzado_v2.csv`
(fila `aciertos_total` y fila `alucinaciones`):

| Pieza | Aciertos | Alucin. | Columna |
|---|---|---|---|
| sin RAG (= LoRA de M1 pelado) | 14/21 | 7 | `sin RAG` |
| + RAG denso ingenuo | 14/21 | 7 | `A · ingenuo` |
| + híbrido BM25 | 14/21 | 6 | `B · +hybrid` |
| + cross-encoder | 13/21 | 5 | `C · +rerank` |
| **+ enrutador** | **18/21** | **2** | `D · +router` |
| + cita textual | 18/21 | 3 | `E · +cita` |

El **enrutador** es la única pieza que sube el total: +4 casos y las
alucinaciones de 7 a 2. El RAG por sí solo no mejora nada sobre los 21 casos, y
el reranker por sí solo **baja** uno. De las **herramientas** no hay una cifra
que atribuirles sobre estos 21 casos: la comparación disponible es
`scorecard_agente_v2.csv`, donde C3 y C5 con herramientas dan 13/21 y 12/21
frente a 14/21 de C1 sin ellas — pero esas filas cambian también el prompt y el
bucle, así que no aíslan las herramientas. Lo que sí está aislado es el bloque de
aritmética difícil de M3: fila `Aritmética /13`, C1 = 8 y C5-FT = 10. El
**verificador** acierta 4/4 en aislado sobre casos de desarrollo de M4
(`m4/verificador_aritmetico_dg.csv`) y ahora está conectado, **sin medir en el
flujo**.

**¿Por qué entrenar o no entrenar?**
Para cálculo, no: tres herramientas deterministas lo dan exacto y comprobable.
Para el planteamiento, M4 ya lo intentó con 42 trayectorias y un adaptador propio:
3/9 frente a 2/9, dentro del ruido. Antes de entrenar otra cosa quedan tres
ablaciones que no cuestan GPU (§5, filas 3 a 5). Entrenar sin saber qué error
sistemático queda es gastar la única GPU disponible en una hipótesis sin medir.

**¿Qué pasa si falla el modelo, el retriever o una herramienta?**
El pipeline monta por capas y cada fallo se anota sin tumbar el resto: la
interfaz muestra *"Tutor NO disponible: modelo: falta torch"*. El retrieval tiene
repliegue léxico sin embeddings. `ejecutar()` del registro **nunca lanza**: todo
error vuelve al modelo como observación.

**¿Qué guardan los logs?**
Evento, latencia, nº de herramientas, versión, y de la pregunta solo **longitud y
hash**. El texto solo si alguien activa `TUTOR_LOG_TEXTO=1` a sabiendas. W&B es
opcional y se apaga limpio sin credencial. Hay pruebas para las tres cosas.

**¿Qué no puede hacer todavía?**
Derivadas e integrales **razonadas**: las calcula exacto con herramienta, pero no
hay un solo caso de cálculo en ningún eval set ni en los datos de entrenamiento,
así que **no está medido que el modelo decida llamar a la herramienta** al leer la
pregunta. Tampoco geometría (15 casos de entrenamiento, 0 herramientas), sistemas
de ecuaciones ni cuadráticas. Y **revisar fotos con fiabilidad**: 2/12 en
evaluación, 3/9 en desarrollo. Y esos 12 casos **ya se gastaron** en corridas
exploratorias, así que para afirmar una mejora hacen falta hojas nuevas, que están
especificadas pero sin fotografiar (`m4/eval_set/HOJAS_M5.md`). La pestaña de
visión no aparece en la configuración por omisión.

**¿Y si durante la demo falla el reranker o falta el adaptador?**
La interfaz lo dice en rojo y **retira la cifra**: «MODO DEGRADADO · resultado de
esta ejecución: NO MEDIDO», con la lista de qué falta y qué pieza sigue
funcionando. El sistema responde por el repliegue léxico, pero deja de llamarse
S08-D. En este disco `qwen-lora-agente` no está, así que es un caso real, no
hipotético.

**¿Cómo sabemos que el tutor no diagnostica sobre una transcripción inventada?**
Porque no puede: `leer_imagen` devuelve lo leído y una pregunta, nunca un
veredicto, y `revisar_pasos` se niega sin `confirmado=True`. El origen es
`vis-10` de M4, donde el VLM inventó «24 × 13 = 306» dos veces idénticas —la
doble lectura no lo detectó— y el tutor diagnosticó encima. Los pasos llegan
editables para que el estudiante corrija la transcripción.

**¿Por qué los 12 casos visuales no sirven para medir la mejora?**
Porque ya se gastaron: tres corridas exploratorias dieron 4/12, 1/12 y 2/12, y la
de 4/12 repitió «Paso 2» en las 11 respuestas con conclusión —un sistema que
contestara siempre «Paso 2» habría acertado lo mismo—. Siguen sirviendo de
regresión. Para afirmar generalización hay 12 hojas nuevas especificadas, con
pares legible/borrosa y transcripción de referencia, y una validación que rechaza
cualquier enunciado que ya aparezca en los eval sets anteriores o en el corpus de
entrenamiento.

**¿Qué configuración está sirviendo la demo exactamente?**
Lo dice la propia pantalla y el endpoint de salud: configuración, resultado
medido o «EXPERIMENTAL, sin medir», adaptador pedido y cargado, retrieval activo
con su repliegue si lo hubo, enrutador, herramientas activas, verificador, visión
y cuantización. Por omisión es `s08d`. Antes de esta revisión el pipeline cargaba
`qwen-lora-agente` solo por estar en el disco y usaba el prompt del agente: una
variante de C5 (12/21) servida bajo el nombre de la mejor medida.
