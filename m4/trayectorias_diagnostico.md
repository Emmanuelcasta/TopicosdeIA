# Trayectorias de diagnóstico de M4

Generadas y validadas por `scripts/trayectorias_diagnostico_fuente.py`.
Enseñan a **dictaminar**, que es lo que el tutor no sabía hacer: en la corrida del 24-10 rehízo correctamente los cinco procedimientos con error y no avisó de ninguno.

**42 ejemplos** · train 33 · dev 9

| familia | train | dev | qué enseña |
|---|---|---|---|
| paso_aritmetico | 15 | 3 | señalar el paso erróneo, con el índice variando entre 1, 2 y 3 |
| compensado | 5 | 1 | un paso mal con el resultado final bien |
| sin_error | 6 | 2 | decir que está correcto, sin inventar un error |
| ilegible | 3 | 1 | abstenerse cuando la foto no se lee |
| fuera_de_dominio | 2 | 1 | abstenerse cuando la hoja no es matemáticas |
| dato_no_disponible | 2 | 1 | abstenerse cuando se pide algo que no está escrito |

Paso señalado: {1: 7, 2: 10, 3: 7} — reparto deliberado, para que el modelo no aprenda a contestar siempre 'Paso 2'.

## Cómo se evita la filtración

Ningún enunciado sale del eval set de M2, M3 ni M4, ni parafraseado. La comprobación es la de `trayectorias_fuente.py`: Jaccard sobre palabras contra todos los casos de evaluación, con tope 0.6. Incluye los casos `dev-*` de M4, que son la puerta del notebook: contaminarlos invalidaría la única señal honesta de formato.

## Qué se valida antes de escribir el archivo

1. La conclusión se parsea con el **mismo** `extraer_diagnostico` del evaluador y coincide con la etiqueta.

2. La aritmética respalda la etiqueta: el paso marcado como erróneo lo es, y ningún `sin error` lleva una cuenta mal. Con `Fraction`.

3. La explicación **no reescribe** el procedimiento: es la conducta que se quiere quitar.

4. El prompt de entrenamiento es literalmente `prompt_revision()`, el de inferencia.

5. `train` y `dev` no comparten enunciados, y los dos cubren las cinco categorías.

## Lo que esto NO promete

Es un experimento. Mejorar el formato de salida no implica mejorar el razonamiento: puede salir un modelo que emita siempre `Diagnóstico: …` y siga sin localizar el paso. Por eso el notebook reporta diagnóstico, lectura y formato **por separado**, y el adaptador se guarda aparte de `qwen-lora` y `qwen-lora-agente`.

## Ejemplos

### `dg-a01` · paso_aritmetico · train

```
El paso 2 está mal: la multiplicación 18 × 7 da 126, no 116.
Diagnóstico: Paso 2
```

### `dg-a02` · paso_aritmetico · train

```
El paso 1 plantea la operación equivocada: aquí hay que dividimos, no restamos.
Diagnóstico: Paso 1
```

### `dg-e01` · fuera_de_dominio · train

```
Lo que hay en la hoja es una lista de compras, no un ejercicio de matemáticas, así que no hay pasos que revisar.
Diagnóstico: fuera de dominio
```

