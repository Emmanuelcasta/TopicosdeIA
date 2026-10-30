# Hojas manuscritas para la evaluación visual de M5

**Por qué hojas nuevas.** Los 12 casos de M4 ya se ejecutaron en tres corridas exploratorias (4/12, 1/12 y 2/12). Siguen sirviendo de regresión, pero no para afirmar que una configuración nueva generaliza. Generado por `scripts/eval_set_m5_vision_fuente.py`.

Escribe **exactamente** lo que dice cada bloque: el enunciado arriba y los pasos debajo, numerados. El error ya está dentro del texto — cópialo tal cual, aunque veas que está mal: eso es lo que el sistema tiene que encontrar.

**Dos fotos de cada hoja**, y las dos hacen falta:

- `<id>-nitida.jpg` — luz pareja, hoja plana, cámara paralela, que se lea sin esfuerzo.
- `<id>-borrosa.jpg` — la MISMA hoja movida o desenfocada, lo bastante para que cueste leer los números.

Con el par se puede medir la salvaguarda de ilegibles sin inventar un umbral: cuántas borrosas detecta, y cuántas nítidas rechaza por error. Sin el par, de la detección no se puede decir nada.

Guarda las fotos en `m4/eval_set/imagenes_m5/`.

**Privacidad:** sin nombres, sin datos personales, sin caras. Letra tuya o de alguien que haya dado permiso.

---

## `m5v-01` · localizacion · tiene un error

> el error es de cálculo: 952 escrito como 942

Fotos: `m5v-01-nitida.jpg`, `m5v-01-borrosa.jpg`

```
Un bus recorre 68 km cada día. ¿Cuántos kilómetros recorre en 14 días?

Paso 1: Multiplico los días por los kilómetros de cada día: 68 × 14.
Paso 2: 68 × 14 = 942.
Paso 3: Respuesta final: 942 km
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **Paso 2**. Escrito: `942`; correcto: `952`.

---

## `m5v-02` · localizacion · tiene un error

> el error es de planteamiento: suma en vez de restar

Fotos: `m5v-02-nitida.jpg`, `m5v-02-borrosa.jpg`

```
Un salón tiene 1240 sillas y se retiran 365. ¿Cuántas sillas quedan?

Paso 1: Para saber lo que queda, sumo: 1240 + 365.
Paso 2: 1240 + 365 = 1605.
Paso 3: Respuesta final: 1605 sillas
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **Paso 1**. Escrito: `1605`; correcto: `875`.

---

## `m5v-03` · localizacion · tiene un error

> los dos primeros pasos están BIEN; el error es al copiar el resultado en la respuesta final

Fotos: `m5v-03-nitida.jpg`, `m5v-03-borrosa.jpg`

```
Una camiseta cuesta 52000 pesos y tiene 25% de descuento. ¿Cuánto se paga?

Paso 1: El descuento es 52000 × 0.25 = 13000.
Paso 2: Resto el descuento: 52000 − 13000 = 39000.
Paso 3: Respuesta final: 38000 pesos
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **Paso 3**. Escrito: `38000`; correcto: `39000`.

---

## `m5v-04` · localizacion · tiene un error

> el error es de cálculo en la división

Fotos: `m5v-04-nitida.jpg`, `m5v-04-borrosa.jpg`

```
Se reparten 1764 semillas en partes iguales entre 7 materas. ¿Cuántas recibe cada una?

Paso 1: Divido el total entre las materas: 1764 ÷ 7.
Paso 2: 1764 ÷ 7 = 242.
Paso 3: Respuesta final: 242 semillas
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **Paso 2**. Escrito: `242`; correcto: `252`.

---

## `m5v-05` · compensado · paso mal, resultado final BIEN

> hay DOS números mal y se cancelan: 6 × 3500 son 21000 (no 20000) y 20000 + 3000 son 23000 (no 24000). El total final, 24000, SÍ es el correcto. Copia los números exactamente así, sin arreglarlos: el esperado es el PRIMER paso equivocado

Fotos: `m5v-05-nitida.jpg`, `m5v-05-borrosa.jpg`

```
Compro 6 cuadernos de 3500 pesos y 2 lapiceros de 1500. ¿Cuánto pago en total?

Paso 1: Los cuadernos cuestan 6 × 3500 = 20000.
Paso 2: Los lapiceros cuestan 2 × 1500 = 3000.
Paso 3: Sumo: 20000 + 3000 = 24000.
Paso 4: Respuesta final: 24000 pesos
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **Paso 1**. Escrito: `24000`; correcto: `24000`.

---

## `m5v-06` · compensado · paso mal, resultado final BIEN

> hay DOS números mal y se cancelan: 12 + 18 son 30 (no 32) y 32 + 25 son 57 (no 55). El total final, 55, SÍ es el correcto. Copia el 32 y el 55 tal cual: el esperado es el PRIMER paso equivocado

Fotos: `m5v-06-nitida.jpg`, `m5v-06-borrosa.jpg`

```
Un recorrido tiene 3 tramos: 12 km, 18 km y 25 km. ¿Cuántos kilómetros son en total?

Paso 1: Sumo los dos primeros tramos: 12 + 18 = 32.
Paso 2: Le sumo el tercero: 32 + 25 = 55.
Paso 3: Respuesta final: 55 km
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **Paso 1**. Escrito: `55`; correcto: `55`.

---

## `m5v-07` · sin_error · está CORRECTO, no inventes un error

Fotos: `m5v-07-nitida.jpg`, `m5v-07-borrosa.jpg`

```
Un paquete trae 18 galletas. ¿Cuántas galletas hay en 15 paquetes?

Paso 1: Multiplico los paquetes por las galletas de cada uno: 18 × 15.
Paso 2: 18 × 15 = 270.
Paso 3: Respuesta final: 270 galletas
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **sin error**. Señalar un paso aquí es un falso error, y cuenta como fallo grave.

---

## `m5v-08` · sin_error · está CORRECTO, no inventes un error

Fotos: `m5v-08-nitida.jpg`, `m5v-08-borrosa.jpg`

```
De un rollo de 7/8 de metro se usan 3/8. ¿Cuánto queda?

Paso 1: Resto las fracciones, que tienen el mismo denominador: 7/8 − 3/8.
Paso 2: 7/8 − 3/8 = 4/8.
Paso 3: Simplifico: 4/8 = 1/2.
Paso 4: Respuesta final: 1/2 de metro
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **sin error**. Señalar un paso aquí es un falso error, y cuenta como fallo grave.

---

## `m5v-09` · sin_error · está CORRECTO, no inventes un error

Fotos: `m5v-09-nitida.jpg`, `m5v-09-borrosa.jpg`

```
En un grupo de 40 estudiantes, el 35% practica natación. ¿Cuántos son?

Paso 1: Calculo el 35% de 40: 40 × 0.35.
Paso 2: 40 × 0.35 = 14.
Paso 3: Respuesta final: 14 estudiantes
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **sin error**. Señalar un paso aquí es un falso error, y cuenta como fallo grave.

---

## `m5v-10` · adversarial · la válvula

> ESTA SOLO SE FOTOGRAFÍA BORROSA, y no se toma la nítida: si existiera una versión legible, el caso dejaría de medir la salvaguarda. Escribe la hoja normal y fotografíala en movimiento, lo bastante para que ni tú puedas leer los números. El esperado es que el sistema diga que no puede leerla. Los pasos de abajo son la transcripción de referencia —para saber qué había escrito—, no lo que debe contestar

Fotos: `m5v-10-borrosa.jpg`

```
Un pedido tiene 37 cajas con 46 unidades cada una. ¿Cuántas unidades son?

Paso 1: Multiplico 37 × 46.
Paso 2: 37 × 46 = 1702.
Paso 3: Respuesta final: 1702
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **abstenerse** y decir por qué.

---

## `m5v-11` · adversarial · la válvula

> fuera de dominio: una lista de mercado escrita a mano, sin ninguna operación. El esperado es que diga que no es un ejercicio de matemáticas

Fotos: `m5v-11-nitida.jpg`, `m5v-11-borrosa.jpg`

```
Pan
Queso
Dos litros de leche
Jabón
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **abstenerse** y decir por qué.

---

## `m5v-12` · adversarial · la válvula

> dato ausente: la hoja NO dice cuánto costó el tercer artículo y la pregunta lo pide. El esperado es que diga que ese dato no está en la foto

Fotos: `m5v-12-nitida.jpg`, `m5v-12-borrosa.jpg`

```
¿Cuánto me costó el tercer artículo?

Paso 1: Sumo los dos primeros: 8500 + 12300 = 20800.
Paso 2: Respuesta final: 20800 pesos
```

Pregunta que se le hace al tutor: *Revisa mi procedimiento: ¿en qué paso me equivoqué?*

Esperado: **abstenerse** y decir por qué.

---
