# M4 · Análisis y conclusiones

## Qué nos propusimos

En M4 ampliamos el tutor para que pudiera revisar una fotografía de un procedimiento manuscrito y señalar el paso donde aparece un error. Esto cambia la tarea: ya no basta con calcular la respuesta correcta; hay que leer lo que escribió el estudiante, conservar sus errores y compararlos con el resultado matemático esperado.

El objetivo experimental era comprobar si añadir visión y ajustar el diagnóstico ayudaba al tutor. Los resultados no muestran una mejora confiable de exactitud. Sí permitieron ubicar con más precisión dónde se rompe el proceso y qué conviene cambiar antes de seguir entrenando.

## Qué construimos

La ruta visual quedó dividida en dos etapas. Primero, Qwen2-VL-2B lee la foto y devuelve un enunciado, una lista de pasos y una respuesta final en formato estructurado. Después, el tutor de M3 recibe esa transcripción y produce una explicación junto con una conclusión separada: paso equivocado, procedimiento correcto, hoja ilegible, contenido fuera de dominio o dato ausente.

Separar lectura y diagnóstico fue una decisión práctica y de evaluación. En una T4 de 16 GB no resultaba conveniente mantener cargados a la vez el modelo visual, el tutor y el evaluador. Además, así podemos repetir la etapa de diagnóstico sin volver a procesar las imágenes y distinguir un error de lectura de un error de razonamiento.

El conjunto visual contiene 12 casos: cuatro de localización, dos en que el procedimiento tiene un paso incorrecto aunque el resultado final sea correcto, tres sin errores y tres casos adversariales. Se necesitan 11 fotos porque dos casos reutilizan hojas. El proyecto también incluye 42 ejemplos sintéticos de trayectorias para diagnóstico: 33 para entrenamiento y 9 para desarrollo. Estos ejemplos cubren pasos erróneos de distinta posición, procedimientos correctos y casos en los que el tutor debería abstenerse.

## Qué dicen los experimentos

### Comparación con nueve casos de desarrollo

Las pruebas de configuración dieron estos resultados:

| Configuración | Aciertos | Formato válido | Llamadas a herramientas |
|---|---:|---:|---:|
| Adaptador de diagnóstico con el prompt original | 3/9 | 9/9 | La traza de esta corrida no es confiable |
| Adaptador agente con el prompt original | 2/9 | 8/9 | 4 |
| Adaptador de diagnóstico con la instrucción «Paso N» | 2/9 | 9/9 | 0 |

Ninguna variante superó el baseline de 3/9. La instrucción «Paso N» mejoró el cumplimiento del formato frente a la configuración A, pero no la exactitud. Además, el cambio de prompt desplazó la conclusión en varios casos aunque la hoja permaneciera igual. Eso indica que el índice de paso todavía depende demasiado de la forma de la instrucción y no siempre se obtiene al comparar cuidadosamente el procedimiento.

Las cuatro llamadas observadas en la configuración A ocurrieron en casos donde las herramientas no resolvían la tarea: el agente buscó documentos ante una hoja ilegible e intentó resolver contenido matemático en un caso fuera de dominio. En esta prueba, añadir herramientas no mejoró el diagnóstico y dejó claro que el agente también necesita aprender cuándo no llamarlas.

### Entrenamiento del adaptador de diagnóstico

Se entrenó un adaptador LoRA independiente con 33 ejemplos de entrenamiento. La corrida registró 30 pasos y seis épocas en una T4. En los nueve casos de desarrollo, el modelo base obtuvo 2/9; los adaptadores de M1 y M3, 1/9 cada uno; y el nuevo adaptador de diagnóstico, 2/9. Todos emitieron el formato esperado en 9/9 casos, pero el nuevo adaptador no aumentó los aciertos frente al modelo base.

La lectura prudente es que los ejemplos ayudaron a enseñar la estructura de salida, pero no bastaron para que el modelo identificara mejor el error. Con nueve casos tampoco podemos afirmar que una diferencia de uno o dos aciertos represente una mejora general.

### Corridas con las doce imágenes

Las corridas guardadas no se deben combinar en una sola cifra: muestran comportamientos distintos y corresponden a ejecuciones diferentes.

En S13_M4_Pipeline_Visual_n, el scorecard registró 4/12. Sin embargo, las 11 respuestas que sí emitieron una conclusión repitieron «Paso 2». Un sistema que contestara siempre «Paso 2» también habría acertado 4/12. Por tanto, ese total no demuestra que el modelo distinguiera los errores; refleja un sesgo hacia una única respuesta. La corrida también registró siete conclusiones concretas equivocadas, una respuesta sin conclusión y ninguna abstención.

En S13_M4_Pipeline_Visualv2 y sus artefactos asociados, el resultado fue 2/12: 0/4 en localización, 0/2 en procedimientos compensados, 2/3 en procedimientos sin error y 0/3 en los casos adversariales. Siete respuestas no incluyeron una conclusión válida, tres dieron una conclusión concreta equivocada y ninguna se abstuvo correctamente.

Las doce imágenes ya se ejecutaron en estas pruebas. Además, la corrida archivada del 24 de octubre continuó con la evaluación completa aunque la puerta de desarrollo no se había superado. Por eso estos resultados sirven para diagnosticar el sistema, pero el conjunto ya no puede tratarse como una prueba final ciega ni usarse para afirmar rendimiento en casos nuevos.

### Verificación aritmética determinista

También se probó un verificador local que comprueba las cuentas con aritmética exacta. En los nueve casos de desarrollo detectó los cuatro pasos aritméticos erróneos evaluables, no señaló por error ninguno de los dos procedimientos sin errores y pudo evaluar seis de los nueve casos.

Este resultado es prometedor para una parte concreta del problema, pero no significa que el tutor ya haya mejorado: el verificador todavía no está conectado al pipeline. Tampoco interpreta si el estudiante eligió la operación adecuada, ni clasifica por sí solo una foto ilegible, una pregunta fuera de dominio o un dato ausente. En un caso con cuentas correctas, por ejemplo, puede confirmar que las operaciones cuadran aunque la información solicitada no aparezca en la hoja.

### Regresión de la tarea anterior

En el conjunto de M2 sin imágenes, la corrida registrada obtuvo 15/21 frente a los 13/21 de M3. No atribuimos los dos casos adicionales a M4: el retrieval de esta ejecución era denso con k=3, mientras que S10 usaba recuperación híbrida con reranker. Como la configuración no fue idéntica, esa comparación no permite aislar el efecto de incorporar visión.

## Evidencia de la entrega

Las comparaciones de desarrollo se pueden revisar en los [resultados baseline](dev_dg_arnes_s13.csv), la [configuración A](dev_dg_A_agente.csv) y la [configuración B](dev_dg_B_pasoN.csv). El [notebook ejecutado de fine-tuning](pipeline/S13_M4_FineTuning_Diagnostico_ejecutado.ipynb) registra el entrenamiento y la comparación de adaptadores.

Las dos corridas visuales están en [S13_M4_Pipeline_Visual_n](pipeline/S13_M4_Pipeline_Visual_n.ipynb) y [S13_M4_Pipeline_Visualv2](pipeline/S13_M4_Pipeline_Visualv2.ipynb). El scorecard y las métricas detalladas que respaldan la corrida de 2/12 están en [scorecard_m4.csv](scorecard_m4.csv) y [resultados_m4.json](resultados_m4.json). La prueba del verificador está en [verificador_aritmetico_dg.csv](verificador_aritmetico_dg.csv), y sus trayectorias de entrenamiento y desarrollo se describen en [trayectorias_diagnostico.md](trayectorias_diagnostico.md).

## Lo que aprendimos

La lectura visual no parece ser el principal cuello de botella en la mayoría de los ejemplos: en la corrida documentada, 10 de las 11 transcripciones conservaron los errores de las hojas. El problema aparece después. El tutor tiende a rehacer el ejercicio y escribir una solución correcta, en vez de comparar esa solución con los pasos que realmente dejó el estudiante. En otros casos se queda sin la línea de diagnóstico o responde siempre con el mismo número de paso.

También vimos que hay que medir por separado la lectura, la conclusión, el formato y la explicación. Una respuesta puede transcribir bien la foto y aun así diagnosticar mal; puede dar una explicación convincente pero señalar el paso equivocado; o puede acertar por casualidad al repetir una respuesta frecuente. Un único porcentaje ocultaría estas diferencias.

El uso de herramientas requiere el mismo cuidado. Una función aritmética puede comprobar una igualdad, pero no decide por sí sola si un problema es matemático ni cuál era la intención del estudiante. La arquitectura debe combinar herramientas exactas para las operaciones que pueden verificar con reglas de abstención y una explicación del tutor basada en evidencia.

## Límites que deben quedar claros

- Las comparaciones principales usan nueve casos de desarrollo y el conjunto visual es pequeño. No permiten generalizar ni sostener diferencias concluyentes.
- Las doce imágenes se ejecutaron en corridas exploratorias; ya no son un holdout ciego para los siguientes ajustes.
- La calibración de legibilidad disponible usa imágenes sintéticas. No valida el comportamiento con fotografías manuscritas reales, nítidas y desenfocadas.
- La etiqueta exacta de los tres casos adversariales se asigna actualmente por ID en el evaluador, no como un campo explícito del CSV. Conviene guardar esa verdad de referencia junto al caso para evitar que el dataset y el código se desincronicen.
- M4 evalúa la revisión de fotos. La interpretación de preguntas mal escritas o dictadas por voz requiere casos y métricas propios; no queda demostrada por estos experimentos.
- La corrida de M2 no conserva la misma configuración de recuperación que M3, así que no sirve para atribuir una mejora o regresión a M4.

## Siguiente paso recomendado

Antes de aumentar el modelo o seguir cambiando el prompt, proponemos integrar el verificador aritmético como una fuente de evidencia: leer la hoja, comprobar cada igualdad evaluable, señalar qué paso no cuadra y pedirle al tutor que explique ese hallazgo. Si la lectura es dudosa, el contenido no es matemático o falta el dato solicitado, el sistema debe decirlo y abstenerse de inventar una respuesta.

Después habría que crear una nueva versión del conjunto de evaluación, con la categoría esperada explícita para cada caso, suficientes ejemplos por tipo y una separación real entre desarrollo y prueba. La evaluación debe reportar por separado fidelidad de transcripción, localización exacta, falsos errores en ejercicios correctos, abstenciones apropiadas, uso de herramientas, formato y calidad de la explicación. Solo con esa nueva prueba podremos saber si la arquitectura mejorada funciona también en ejemplos que no se usaron para ajustarla.

## Conclusión

M4 todavía no demuestra que el tutor resuelva mejor los ejercicios fotografiados. Su aporte principal fue convertir una necesidad general —«que entienda una foto»— en etapas y fallos observables: lectura, diagnóstico, abstención y explicación. Las pruebas muestran que la lectura puede conservar buena parte de los errores, pero que el modelo aún no compara de forma confiable el procedimiento escrito con la matemática.

La dirección más sólida es usar verificación determinista donde sea posible y reservar al modelo la explicación pedagógica, junto con una ruta explícita para pedir aclaraciones o abstenerse. Luego debemos validar esa integración con una prueba nueva y separada. Ese trabajo apunta a un tutor más confiable sin presentar como mejora algo que los datos actuales todavía no respaldan.
