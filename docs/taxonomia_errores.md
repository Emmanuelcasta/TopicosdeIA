# Fallos del tutor · análisis exploratorio

Generado con `scripts/taxonomia_errores.py`.

> **Esto no es una medición.** Las cifras del arnés (`acierto`, `abstuvo`, `alucino`) están en los scorecards y no se recalculan aquí. Lo de abajo son etiquetas heurísticas sobre respuestas ya guardadas, pensadas para orientar dónde mirar. Nadie anotó estos casos a mano.

Tres reglas de lectura:

1. **Una etiqueta principal por caso.** La columna suma exactamente el número de casos, así que un caso con varias señales no se cuenta como varios fallos. Las secundarias van en su propia tabla y no se suman.
2. **`sin_evidencia`** marca las categorías que los datos guardados no permiten decidir. No se estiman.
3. **Las etiquetas condicionadas** solo pueden aplicarse a cierto tipo de respuesta. Un cero ahí no es una mejora.

`explicacion_confusa` se asigna a casos que el arnés contó como **acierto**: es una pega de calidad, no un fallo, y no se suma a los fallos.

## Qué significa cada etiqueta

| Etiqueta | Tipo | Cómo se decide |
|---|---|---|
| `aritmetica_falsa` | comprobable | el verificador aritmético encuentra una igualdad que no cuadra en la respuesta |
| `calculo_sin_herramienta` | condicionada | falló un caso numérico sin llamar a ninguna herramienta; solo aplicable a configuraciones que pueden no llamarlas |
| `final_no_coincide_con_pasos` | comprobable | ninguna observación de herramienta aparece en el número final |
| `planteamiento_incorrecto` | heuristica | las herramientas devolvieron bien y la respuesta igual falló |
| `herramienta_innecesaria` | comprobable | llamó a una herramienta en un caso adversarial |
| `pregunta_mal_escrita` | comprobable | el caso pertenece al bloque `mal_escrito` del eval set |
| `explicacion_confusa` | comprobable | el juez puntuó <= 2 pese a que el caso se contó como acierto |
| `negativa_innecesaria` | comprobable | se abstuvo en un caso que tenía respuesta |
| `no_consulto_documentos` | comprobable | falló un caso de conocimiento sin recuperar ningún pasaje |
| `no_reconocio_el_limite` | comprobable | respondió un adversarial en vez de abstenerse |
| `respuesta_a_imagen_ilegible` | comprobable | respondió un caso visual cuyo `esperado` del scorecard es «ilegible» |
| `respuesta_fuera_de_dominio` | comprobable | respondió un caso visual cuyo `esperado` es «fuera de dominio» |
| `dato_no_disponible` | comprobable | respondió un caso visual cuyo `esperado` es «dato no disponible» |
| `sin_conclusion_formato` | comprobable | no emitió la línea `Diagnóstico:`; es un fallo de formato, no de criterio |
| `contexto_irrelevante` | sin_evidencia | haría falta el pasaje de oro de cada caso de S10, y no se guardó por caso |
| `mala_eleccion_de_herramienta` | sin_evidencia | haría falta la herramienta correcta anotada por caso; `recall_seleccion` solo existe para el subconjunto con oro |
| `lectura_visual_erronea` | sin_evidencia | haría falta la transcripción correcta de cada hoja anotada a mano |

## M3 · 53 casos por configuración · etiqueta principal

Fuente: `respuestas_s10.jsonl` + `detalle_s10_v2.csv`.

| Etiqueta principal | C0 · base | C1 · +LoRA | C2 · +RAG | C3 · +herram. | C4 · +ReAct | C5 · agente | C5-FT · entrenado |
|---|---|---|---|---|---|---|---|
| `acierto` | 15 | 25 | 24 | 29 | 25 | 28 | 30 |
| `abstencion_correcta` | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| `aritmetica_falsa` | 4 | 7 | 11 | 9 | 6 | 8 | 4 |
| `calculo_sin_herramienta` | 25 | 12 | 13 | 10 | 17 | 9 | 0 |
| `planteamiento_incorrecto` | 0 | 0 | 0 | 0 | 0 | 1 | 14 |
| `fallo_con_contexto_recuperado` | 0 | 0 | 0 | 1 | 1 | 0 | 1 |
| `no_consulto_documentos` | 4 | 4 | 0 | 0 | 0 | 1 | 0 |
| `no_reconocio_el_limite` | 2 | 1 | 0 | 2 | 1 | 3 | 1 |
| `negativa_innecesaria` | 0 | 2 | 2 | 0 | 0 | 0 | 0 |
| `explicacion_confusa` | 3 | 2 | 1 | 2 | 3 | 2 | 3 |
| `fallo_sin_etiqueta` | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **total** | 53 | 53 | 53 | 53 | 53 | 53 | 53 |

### Señales secundarias (no suman fallos)

| Señal | C0 · base | C1 · +LoRA | C2 · +RAG | C3 · +herram. | C4 · +ReAct | C5 · agente | C5-FT · entrenado |
|---|---|---|---|---|---|---|---|
| `herramienta_innecesaria` | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `pregunta_mal_escrita` | 7 | 5 | 5 | 5 | 6 | 5 | 4 |

### Por qué `calculo_sin_herramienta` no se puede comparar entre configuraciones

La etiqueta exige que la respuesta no haya llamado a ninguna herramienta. Una configuración que llama casi siempre no puede recibirla, de modo que su cero no dice nada sobre si sus cuentas son correctas. Proporción de casos con al menos una llamada:

| Configuración | casos con llamada | de | % |
|---|---|---|---|
| C0 · base | 0 | 53 | 0% |
| C1 · +LoRA | 0 | 53 | 0% |
| C2 · +RAG | 0 | 53 | 0% |
| C3 · +herram. | 1 | 53 | 2% |
| C4 · +ReAct | 0 | 53 | 0% |
| C5 · agente | 10 | 53 | 19% |
| C5-FT · entrenado | 50 | 53 | 94% |

Para la aritmética sí hay evidencia independiente de las llamadas: `aritmetica_falsa` la da el verificador determinista (`scripts/agente/verificador_aritmetico.py`) leyendo las igualdades escritas en la respuesta. Esa fila sí es comparable, con el límite de que solo ve las igualdades que el tutor escribió explícitamente.

## M4 · 12 casos visuales

Fuente: `m4/scorecard_m4.csv`. La etiqueta de un adversarial la fija su columna `esperado`: solo `vis-10` es ilegible; `vis-11` es fuera de dominio y `vis-12` es un dato que la hoja no tiene.

| Etiqueta | n |
|---|---|
| `sin_conclusion_formato` | 7 |
| `acierto` | 2 |
| `respuesta_a_imagen_ilegible` | 1 |
| `respuesta_fuera_de_dominio` | 1 |
| `dato_no_disponible` | 1 |

`sin_conclusion_formato` **no es un fallo de criterio**: el tutor no llegó a emitir un veredicto. Se cuenta aparte porque se arregla de otra manera.

Lo que estos 12 casos **no** permiten decir es si la lectura de la hoja fue correcta: haría falta la transcripción anotada a mano, que no existe. Por eso `lectura_visual_erronea` queda `sin_evidencia`.
