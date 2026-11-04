# Guion Demo Day · Tutor de Matemáticas

**Duración:** 15 minutos: 13 de exposición y 2 de preguntas.  
**Orden:** Miguel → Emmanuel → Camilo → Felipe → Emmanuel.  
**Equipo:** Miguel Mercado, Emmanuel Castañeda, Camilo Arbeláez y Felipe Castro.

En `demo_day.html` están las 10 diapositivas sin el texto del guion; las capturas de la interfaz (`assets/captura-*.jpg`, en modo claro) son de Kaggle T4 con el commit `83e44f7`, y las hojas (`assets/hoja-*.jpg`) son fotos del eval set de desarrollo. Las diapositivas no muestran quién expone: el orden está en la tabla de abajo. Usa las flechas para avanzar y la tecla **F** para pantalla completa. Ensayen con cronómetro; Kaggle usa un enlace temporal, así que preparen capturas de respaldo.

## Reparto

| Tiempo | Presenta | Contenido |
|---|---|---|
| 0:00–1:30 | Miguel | Apertura, problema y usuario |
| 1:30–4:30 | Emmanuel | Arquitectura y demo de pregunta escrita |
| 4:30–6:30 | Camilo | Flujo de foto: elegir, confirmar, resolver o revisar |
| 6:30–11:00 | Felipe | Resultados M1–M3, scorecard M5 y límites |
| 11:00–13:00 | Emmanuel | Cierre |
| 13:00–15:00 | Todos | Preguntas |

## 0:00–1:30 · Miguel · Apertura y problema

Buenos días. Somos Miguel, Emmanuel, Camilo y Felipe. Durante el semestre construimos un tutor de matemáticas en español y lo probamos por etapas. Hoy queremos mostrar cómo ayuda, qué evidencia tenemos y qué estamos mejorando.

El problema es que una respuesta final no siempre muestra si el estudiante entendió. Puede equivocarse en un paso y llegar por casualidad al resultado correcto, o ejecutar bien las cuentas con un planteamiento equivocado. Por eso el tutor debe explicar procedimientos y, cuando revisa una hoja, comparar lo escrito con la matemática. Está pensado como apoyo para practicar; no reemplaza al profesor ni hace que toda respuesta generada sea automáticamente correcta.

## 1:30–4:30 · Emmanuel · Arquitectura y pregunta escrita

La arquitectura separa tareas. El tutor de texto usa Qwen 2.5 de 1.5 mil millones de parámetros con un adaptador LoRA. Según la configuración, puede usar recuperación y herramientas. Para imágenes, un modelo de visión propone una transcripción editable; el estudiante la confirma antes de pedir una solución o un diagnóstico. El verificador comprueba igualdades que puede evaluar, mientras el tutor interpreta el procedimiento. Si las señales no bastan, no debemos ocultar ese límite.

En la captura vemos una pregunta escrita: «¿Cuántas galletas hay en 15 paquetes de 18 galletas cada uno?». La interfaz muestra los pasos numerados, el resultado destacado —270 galletas— y una marca junto a cada cuenta: el check verde significa que el verificador recalculó esa igualdad con aritmética exacta; una equis roja, que no cuadra; sin marca, que en esa línea no había cuenta que comprobar. La marca comprueba cuentas, no el planteamiento. Es una prueba puntual de la interfaz, no un benchmark.

## 4:30–5:30 · Camilo · Flujo de fotos (diapositiva 5)

Para una foto hay tres pasos. Primero se sube la hoja y se elige qué se necesita: resolver un enunciado o revisar un procedimiento. Si eligen revisar pero la hoja solo trae el enunciado, el tutor lo dice y cambia a resolver. Segundo, el sistema muestra lo que leyó —el enunciado y, si se revisa, los pasos— en cajas editables; el estudiante corrige y confirma. Solo después aparece el tercer paso. Así se reduce el riesgo de atribuirle al estudiante un error que en realidad fue de transcripción. Si las dos lecturas internas no coinciden en un número o una operación, la pantalla lo señala antes de confirmar.

## 5:30–6:30 · Camilo · Resolver o revisar (diapositiva 6)

A la izquierda, m5v-07 resuelto: un paso, «multiplicamos 18 por el número de paquetes», 18 por 15 igual a 270 con su marca de cuenta verificada, y el resultado, 270 galletas. A la derecha, m5v-02 revisado: el estudiante sumó 1240 más 365 cuando le preguntaban cuántas sillas quedan. La cuenta está bien hecha, por eso el paso 2 tiene marca verde, pero el planteamiento no: el tutor resolvió el enunciado por separado, obtuvo 1240 menos 365 igual a 875, y señala el paso 1 como primer error, con el motivo. Es importante decir que esa comparación solo se usa cuando dos soluciones independientes coinciden; el verificador comprueba cuentas, no decide solo el planteamiento.

## 6:30–8:15 · Felipe · Resultados M1–M3

En M1 y M2, el fine-tuning pasó de 50 a 83,3 por ciento en los 12 problemas difíciles de cálculo. La muestra es pequeña y fueron cuatro aciertos netos; lo presentamos como una señal observada en ese conjunto, no como una garantía general.

En M3, S08-D, con RAG híbrido y enrutador, obtuvo 18 de 21 con dos alucinaciones. El agente entrenado C5-FT obtuvo 13 de 21 en el mismo harness. Aunque mejoró el uso de herramientas en otras métricas, no consiguió mayor exactitud global. Aprendimos que añadir componentes no asegura un tutor mejor: cada módulo debe demostrar su aporte.

## 8:15–9:45 · Felipe · Evaluación visual M5

En la corrida final, resolver obtuvo 11 de 11 y revisar 10 de 11; no hubo errores inventados en las hojas sin error. El caso m5v-10 quedó fuera de esta ronda. El único fallo de revisión fue m5v-09: las dos lecturas interpretaron 14, aunque la hoja dice 19.

m5v-07 es el ejemplo de los flujos observados. Hay que poner el resultado en contexto: estas mismas hojas se usaron durante las iteraciones, así que el scorecard es evidencia de desarrollo, no una prueba independiente ni de generalización.

## 9:45–11:00 · Felipe · Límites y siguiente paso

Queda un fallo conocido: m5v-09. Las dos lecturas ven 14 donde la hoja dice 19, así que no hay discrepancia que mostrar y la revisión sale «sin error». También falta medir generalización con hojas manuscritas nuevas, sin usarlas para ajustar el sistema.

La explicación del modo resolver ya sale limpia: sin líneas sobrantes del borrador ni frases que aludan a un paso eliminado. Y recuerden que Kaggle T4 ofrece un enlace Gradio temporal, no un despliegue permanente.

## 11:00–13:00 · Emmanuel · Cierre

M1 nos permitió adaptar el modelo; M2, diseñar una evaluación más exigente; M3, comparar RAG, enrutamiento y agentes; M4, incorporar imágenes y descubrir que lectura y diagnóstico son problemas distintos; y M5, integrar esos componentes en una aplicación que podemos probar.

El resultado no es que el tutor ya resuelva correctamente todas las fotos. Es que tenemos un flujo comprobable, resultados guardados y fallos concretos que guían la siguiente iteración. Queremos que el tutor lea fielmente, resuelva paso a paso cuando solo recibe el enunciado y explique con evidencia cuando revisa un procedimiento. Muchas gracias; abrimos las preguntas.

## 13:00–15:00 · Preguntas

- **¿Qué logró la corrida final?** Resolver 11/11 y revisar 10/11 en las mismas fotos usadas para iterar. Es un resultado de desarrollo, no una prueba independiente.
- **¿Qué hace el verificador?** Comprueba igualdades aritméticas; no decide por sí mismo si el planteamiento es correcto.
- **¿Qué demuestra la demo?** Que dos flujos funcionaron en el ejemplo mostrado; no el rendimiento de todas las fotos.
- **¿Por qué usar RAG?** Para incorporar conocimiento externo. El enrutador ayuda a buscar solo cuando la pregunta lo requiere.
- **¿Qué significa la marca verde?** Que el verificador recalculó esa igualdad con aritmética exacta y cuadra. No garantiza que la operación elegida sea la correcta.
- **¿Qué falta para producción?** Corregir la lectura de m5v-09, medir con hojas nuevas, hacer más pruebas y contar con un enlace estable; Kaggle proporciona un enlace temporal.

Si una cifra no está medida, digan: «Todavía no lo medimos; ese sería el experimento que haríamos». No inventen resultados.
