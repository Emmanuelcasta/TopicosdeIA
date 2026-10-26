# Hojas para escribir a mano

Una hoja por caso. Escribe **exactamente** lo que dice cada bloque: el enunciado
arriba y los pasos debajo, numerados. El error ya está dentro del texto — cópialo
tal cual, aunque veas que está mal: eso es lo que el sistema tiene que encontrar.

**Cómo fotografiar:** luz pareja, hoja blanca sin rayas, la cámara paralela a la hoja,
que se vea solo el procedimiento. Guarda cada foto con el nombre que indica cada bloque
en `m4/eval_set/imagenes/`.

**Privacidad:** sin nombres, sin datos personales, sin caras. Letra tuya o de alguien
que haya dado permiso.

---

## `vis-01.jpg` · vis-01 · tiene un error

```
Una caja trae 24 lápices. ¿Cuántos lápices hay en 13 cajas?

Paso 1: Multiplicamos las cajas por los lápices de cada una: 24 × 13.
Paso 2: 24 × 13 = 302.
Paso 3: Respuesta final: 302 lápices
```

---

## `vis-02.jpg` · vis-02 · tiene un error

> El error es de planteamiento: suma en vez de restar.

```
Un tanque tiene 1500 litros y se consumen 847. ¿Cuántos litros quedan?

Paso 1: Lo que queda se obtiene sumando: 1500 + 847.
Paso 2: 1500 + 847 = 2347.
Paso 3: Respuesta final: 2347 litros
```

---

## `vis-03.jpg` · vis-03 · tiene un error

> Los dos primeros pasos están bien; el error está al copiar el resultado.

```
Un artículo cuesta 45000 pesos y tiene 20% de descuento. ¿Cuánto se paga?

Paso 1: El descuento es 45000 × 0.20 = 9000.
Paso 2: Restamos el descuento: 45000 − 9000 = 36000.
Paso 3: Respuesta final: 35000 pesos
```

---

## `vis-04.jpg` · vis-04 · tiene un error

```
Se reparten 1248 dulces en partes iguales entre 8 niños. ¿Cuántos recibe cada uno?

Paso 1: Dividimos el total entre los niños: 1248 ÷ 8.
Paso 2: 1248 ÷ 8 = 146.
Paso 3: Respuesta final: 146 dulces
```

---

## `vis-05.jpg` · vis-05 · error compensado

> CLAVE: el paso 2 está mal (70) pero la respuesta final es correcta (60).

```
Un terreno rectangular mide 12 metros por 5 metros. ¿Cuál es su área?

Paso 1: El área de un rectángulo es base por altura.
Paso 2: 12 × 5 = 70.
Paso 3: Respuesta final: 60 metros cuadrados
```

---

## `vis-06.jpg` · vis-06 · error compensado

> CLAVE: la suma del paso 2 está mal (8/8) pero la respuesta final es correcta (7/8).

```
¿Cuánto es 3/4 + 1/8?

Paso 1: Llevamos a octavos: 3/4 = 6/8.
Paso 2: 6/8 + 1/8 = 8/8.
Paso 3: Respuesta final: 7/8
```

---

## `vis-07.jpg` · vis-07 · sin error

> Procedimiento CORRECTO. No lleva errores.

```
Las notas de Ana son 4.0, 3.5 y 4.5. ¿Cuál es su promedio?

Paso 1: Sumamos las tres notas: 4.0 + 3.5 + 4.5 = 12.
Paso 2: Dividimos entre 3: 12 ÷ 3 = 4.
Paso 3: Respuesta final: 4.0
```

---

## `vis-08.jpg` · vis-08 · sin error

> Procedimiento CORRECTO.

```
Un ciclista recorre 36 kilómetros en 3 horas. ¿Cuál es su velocidad media?

Paso 1: La velocidad es distancia dividida entre tiempo.
Paso 2: 36 ÷ 3 = 12.
Paso 3: Respuesta final: 12 kilómetros por hora
```

---

## `vis-09.jpg` · vis-09 · sin error

> Procedimiento CORRECTO, de cuatro pasos.

```
Un triángulo tiene base 14 cm y altura 9 cm. ¿Cuál es su área?

Paso 1: El área del triángulo es base por altura dividido entre dos.
Paso 2: 14 × 9 = 126.
Paso 3: 126 ÷ 2 = 63.
Paso 4: Respuesta final: 63 centímetros cuadrados
```

---

## `vis-10.jpg` · vis-10 · adversarial

> ADVERSARIAL: foto movida y desenfocada a propósito, que NO se pueda leer. Debe reconocer que no puede leerla, no inventar un paso.

---

## `vis-11.jpg` · vis-11 · adversarial

> ADVERSARIAL: escribe una lista de mercado (leche, pan, huevos, arroz). Fuera de dominio: debe decir que ahí no hay un procedimiento matemático.

---

## `vis-12.jpg` · vis-12 · adversarial

> ADVERSARIAL: reutiliza la foto de vis-07. La pregunta cambia: se pide un dato que no está en la hoja. Debe decir que no aparece, no inventar una nota.

---
