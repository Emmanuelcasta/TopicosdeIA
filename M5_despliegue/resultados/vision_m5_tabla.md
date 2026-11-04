# Scorecard visual M5 · commit a98d805 · 2026-11-04T17:05:10+00:00 · lado mayor 1024 px

evaluación de DESARROLLO: estas 12 hojas se han usado para depurar la lectura, el verificador y la regla de diagnóstico. No es evidencia de generalización.

| Caso | Lectura s | Discrepancias mostradas | Resolver | ✓ | s | Revisar | Esperado | ✓ | s | Motivo | Fallo observado |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m5v-01 | 13.12 | Paso 2: una lectura dice «68 x 14 = 942» y la otra «68 x 14 = 9412». / Paso 3: una lectura dice «Respuesta Final: 642 km» y la otra «Respuesta Final: 6412 km». | 952 (esperado 952) | True | 22.64 | paso 2 | paso 2 | True | 0.79 | el paso 2 no cuadra: 68 x 14 da 952, y está escrito 942 | lectura de los pasos no fiel a la hoja |
| m5v-02 | 13.06 |  | 875 (esperado 875) | True | 6.42 | paso 1 | paso 1 | True | 7.16 | en el paso 1 se plantea 1240 + 365, que da 1605, pero al resolver el enunciado por separado el resultado es 875: la cuenta puede estar bien hecha, pero no es la | ninguno |
| m5v-03 | 14.11 |  | 39000 (esperado 39000) | True | 11.79 | paso 3 | paso 3 | True | 0.81 | el paso 3 copia mal el resultado: era 39000 y está escrito 38000 | ninguno |
| m5v-04 | 10.56 | Enunciado: solo una de las dos lecturas lo encontró: «Se reporten 1764 semillas en partes iguales entre 7 materas. ¿Cuantas recibe cada una?». | 252 (esperado 252) | True | 13.91 | paso 2 | paso 2 | True | 0.81 | el paso 2 no cuadra: 1764 ÷ 7 da 252, y está escrito 242 | lectura de los pasos no fiel a la hoja |
| m5v-05 | 16.51 |  | 24000 (esperado 24000) | True | 14.62 | paso 1 | paso 1 | True | 0.83 | el paso 1 no cuadra: 6 x 3500 da 21000, y está escrito 20000 | ninguno |
| m5v-06 | 13.98 |  | 55 (esperado 55) | True | 5.35 | paso 1 | paso 1 | True | 1.39 | el paso 1 no cuadra: 12 + 18 da 30, y está escrito 32 | ninguno |
| m5v-07 | 13.02 |  | 270 (esperado 270) | True | 9.74 | sin error | sin_error | True | 10.49 | todas las cuentas que se pueden comprobar cuadran y tu resultado, 270, es el mismo que sale al resolver el enunciado por separado | ninguno |
| m5v-08 | 16.2 |  | 1/2 (esperado 1/2) | True | 16.23 | sin error | sin_error | True | 17.13 | todas las cuentas que se pueden comprobar cuadran; el planteamiento no se pudo comprobar por separado | ninguno |
| m5v-09 | 12.86 |  | 14 (esperado 14) | True | 13.6 | sin error | paso 3 | False | 13.99 | todas las cuentas que se pueden comprobar cuadran y tu resultado, 14, es el mismo que sale al resolver el enunciado por separado | lectura de los pasos no fiel a la hoja; diagnóstico incorrecto (lectura) |
| m5v-10 |  |  |  |  |  |  |  |  |  |  | excluido de la ronda |
| m5v-11 | 3.97 |  | Esto no es un ejercicio de matemáticas: no hay ninguna cantidad ni ninguna pregunta que calcular. Si es un ejercicio, es | True | 0.0 | Esto no es un ejercicio de matemáticas: no hay ninguna cantidad ni ninguna pregu | adversarial | True | 0.0 | Esto no es un ejercicio de matemáticas: no hay ninguna cantidad ni ninguna pregunta que calcular. Si es un ejercicio, escribe el enunciado completo con sus dato | ninguno |
| m5v-12 | 8.79 |  | Falta un dato: la pregunta pide una cantidad, pero en lo que leí no aparece ningún dato numérico para calcularla; no est | True | 0.0 | dato no disponible | adversarial | True | 0.88 | el enunciado no trae ningún dato numérico, y los números que usas (8500, 12300) no aparecen en él: el dato que pide la pregunta no está en la foto. Sin ese dato | ninguno |
