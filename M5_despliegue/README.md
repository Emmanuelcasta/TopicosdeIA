# M5 · Despliegue e instrumentación

Lleva el tutor de un notebook a una aplicación, y mide lo que cuesta ejecutarlo.

> **Estado: implementado, sin medir.** El código está y las 360 pruebas sin GPU
> pasan, también en el entorno fijado de `config/requirements.txt`.
>
> **Comprobado:** la app monta el retrieval y el enrutador de S08-D con los
> mismos pesos, la misma validación leave-one-out y las mismas 21 rutas que el
> notebook (`evaluacion/verificar_retrieval.py`). Eso dice que la pieza servida
> es la medida; **no** dice cuántos casos acierta el tutor.
>
> **Histórico, no medido aquí:** el 18/21 de S08-D sale de
> `scorecard_rag_avanzado_v2.csv`, de una corrida de S08. Reproducirlo en este
> entorno es la fila 0 de las ablaciones y está pendiente.
>
> **Sin medir:** no hay Space publicado, ni URL, ni corrida de los 21 casos aquí,
> ni medición de eficiencia, ni telemetría recogida, ni evaluación de la vía
> visual con fotos nuevas. El plan completo, con lo que bloquea cada experimento,
> está en [`docs/VALIDACION_M5.md`](docs/VALIDACION_M5.md).

## Estructura

```
M5_despliegue/
├── app/
│   ├── configuraciones.py  las 8 configuraciones, con su resultado medido
│   ├── pipeline.py         monta el tutor reutilizando scripts/agente/
│   ├── app.py              interfaz Gradio: texto e imagen
│   ├── salud.py            health check sin cargar el modelo
│   └── telemetria.py       registro estructurado, W&B opcional
├── config/            requirements fijados, README del Space, .env.example
├── evaluacion/
│   ├── verificar_retrieval.py   ¿el enrutador servido es el de S08-D? (sin GPU)
│   ├── ablaciones.py            las 6 filas de §5, un cambio por fila (GPU)
│   └── medir_eficiencia.py      fp16 vs int4 (GPU)
├── resultados/        (vacío hasta ejecutar la medición)
├── docs/              ARQUITECTURA_M5.md · DESPLIEGUE.md · GUION_DEMO.md
└── tests/             pruebas sin GPU
```

## La configuración se elige, no se adivina

`TUTOR_CONFIG` decide qué se sirve; por omisión `s08d`, que es **18/21** en el
eval set fijo de M2 (`scorecard_rag_avanzado_v2.csv`, fila `aciertos_total`,
columna `D · +router`). Cada configuración declara en
[`app/configuraciones.py`](app/configuraciones.py) qué monta y qué cifra la
respalda, o que no hay ninguna.

La versión anterior de `pipeline.py` cargaba `qwen-lora-agente` **por estar en el
disco** y usaba el prompt del agente: una variante de C5, 12/21, servida bajo el
nombre de la mejor medida. Ahora el adaptador lo pide la configuración, y si
falta, la app responde con el modelo base y lo dice, en vez de cargar el otro
adaptador en silencio.

## Modo degradado · la cifra se apaga cuando falta una pieza

`18/21` describe un sistema concreto: LoRA de M1 + híbrida con reranker +
enrutador. Si en una ejecución falta cualquiera de los tres, lo que responde es
**otro sistema**, y enseñar esa cifra sería atribuirle el resultado de uno
distinto.

Así que el estado lleva dos nombres. `configuracion_solicitada` es lo que alguien
puso en `TUTOR_CONFIG`; `configuracion_efectiva` es lo que quedó funcionando. En
cuanto difieren:

```
🔴 MODO DEGRADADO · resultado de esta ejecución: NO MEDIDO.
Se pidió `S08-D · LoRA M1 + RAG híbrido + enrutador` y falta
**retrieval híbrido + reranker, enrutador consulta/problema**. La cifra 18/21 es
de esa configuración con todas sus piezas y no describe lo que responde ahora.

Qué está funcionando:
- adaptador m1
- retrieval lexico (REPLIEGUE: se pidió híbrido)
- SIN enrutador (se pidió)
- sin herramientas
- fp16
```

El repliegue **se conserva** —responder por la vía léxica es mejor que no
responder— pero deja de llamarse S08-D y deja de citar su cifra. Entran en
`componentes_faltantes`, cada uno cuando la configuración lo pide y no se
materializa: adaptador LoRA, retrieval híbrido + reranker, enrutador, registro de
herramientas, verificador y normalizador.

Siete pruebas, una por cada forma de fallar (`test_m5.py`, sección
«Degradación»).

## Revisar una foto · dos lecturas y dos pasos, no uno

En M4 el flujo leía la foto y diagnosticaba en la misma llamada. `vis-10` mostró
el problema: el modelo de visión inventó «24 × 13 = 306» —dos lecturas idénticas,
así que ni la doble lectura lo detectó— y el tutor diagnosticó sobre ese invento
sin dudar.

Antes de mostrar la transcripción hay dos lecturas independientes de la misma
foto. Deben coincidir en estructura, pasos y números; si una no produce JSON o
si no coinciden, la aplicación se detiene y pide otra foto o los pasos escritos.
El acuerdo es una señal para rechazar lecturas inestables, **no** prueba de que
el texto sea correcto: `vis-10` ya demostró que una invención puede repetirse.

Después vienen dos pasos con una confirmación en medio:

```
leer_imagen(foto)       -> transcripción + pregunta de confirmación.
                           NUNCA un veredicto. Si la lectura no es fiable, se
                           detiene y dice el motivo observable.
revisar_pasos(pasos, confirmado=True)
                        -> verificador determinista -> evidencia -> tutor
```

Sin `confirmado=True` no hay diagnóstico, y no es una formalidad: es la diferencia
entre revisar lo que el estudiante escribió y revisar lo que un modelo creyó ver.
Los pasos llegan a un cuadro **editable**, porque corregir la transcripción es
parte del flujo, no una excepción.

El orden del paso 2 importa. Primero el verificador aritmético, que comprueba las
igualdades con `Fraction` y no se equivoca; su hallazgo entra en el prompt como
**evidencia**, no como veredicto —«el paso 2 no cuadra: 3 × 24 da 72 y está
escrito 82»— y qué paso señalar sigue siendo del tutor. Cuando las cuentas sí
cuadran también se le dice, para que no invente un error aritmético donde no lo
hay. El prompt le prohíbe reescribir los números del estudiante.

No hay porcentaje de confianza inventado. Esta ruta usa solamente señales
observables de una petición: JSON estructurado, declaración de ilegibilidad y
coincidencia de dos lecturas. No usa una confianza relativa a otras fotos ni un
umbral de nitidez: para ello haría falta un conjunto de calibración nuevo y
separado. Tampoco hay un clasificador de legibilidad independiente conectado en
esta ruta; la confirmación editable es la defensa adicional mientras se mide.

`revision_foto` no activa el registro general de herramientas ni ReAct de M3
(`herramientas=False`). Monta S08-D como tutor textual y, **después** de que el
estudiante confirma los pasos, usa únicamente el verificador aritmético
determinista como evidencia. Activar las 18 herramientas sería otra hipótesis
(`s08d_herramientas`) y no forma parte de esta configuración.

Todo esto vive en `TUTOR_CONFIG=revision_foto`, **sin medir**: los 12 casos de M4
ya se gastaron y hacen falta las hojas nuevas de `m4/eval_set/HOJAS_M5.md`.

## La decisión que ordena todo

**La interfaz no sabe de tutoría.** `pipeline.py` monta las piezas que ya
existen —`registro`, `herramientas`, `nucleo`, `documentos`, `prompts`,
`enrutador`, `vision`, `verificador_aritmetico`— y `app.py` solo llama. Hasta el
corpus se obtiene ejecutando su fuente canónica en `nb_comun_rag`, no copiándolo.

Si la lógica se duplicara, la demo y lo medido se separarían en la primera
corrección. Hay dos pruebas que lo verifican: una comprueba que `pipeline.py`
importe la fuente del corpus y no lo lleve dentro, otra que `app.py` no
instancie `AgenteTutor` por su cuenta.

## Montaje por capas

El modelo pesa y puede faltar. El pipeline arranca por capas —corpus,
herramientas, modelo— y cada fallo se anota sin tumbar el resto. La interfaz
pregunta `estado()` y dice la verdad: *"Tutor NO disponible: modelo: falta
torch"*. Lo que ocurre cuando una pieza requerida falta pero el sistema sigue
respondiendo está arriba, en «Modo degradado».

## Privacidad

No se registra el texto del usuario por defecto: solo longitud y hash. Para
activarlo hay que poner `TUTOR_LOG_TEXTO=1` a sabiendas. W&B es opcional y se
apaga limpio si no hay credencial. Hay pruebas para las tres cosas.

## Lo que la interfaz dice de sí misma

Qué configuración sirve y qué cifra la respalda —o que no hay ninguna—; si la
ejecución está degradada y qué componente falta; que la revisión de fotos **no
está medida** con imágenes nuevas y que los 12 casos de M4 ya se gastaron; que
acierta 3 de 9 casos de desarrollo **con otra configuración**; y que no emite
diagnósticos médicos. No aparenta una seguridad que la medición no respalda.

## Qué falta medir

| Experimento | Qué mediría | Bloqueado por |
|---|---|---|
| E1 · texto | los 21 casos con los tres controles | GPU |
| E2 · retrieval | reproducción **(hecha)** + recall de la clave de oro | GPU para el recall |
| E3 · herramientas | si el modelo elige la herramienta | GPU y casos de cálculo en un eval set, que no existen |
| E4 · lectura de fotos | transcripción y salvaguarda de ilegibles | las 23 fotografías de `m4/eval_set/HOJAS_M5.md` |
| E5 · flujo de foto | paso señalado, falso error, abstención | lo de E4, y que E4 pase |
| E6 · muestra real | el harness sobre interacciones del Space | una URL pública |

Detalle, con criterios de aprobación fijados antes de medir, en
[`docs/VALIDACION_M5.md`](docs/VALIDACION_M5.md). Nada de esto se simula.
