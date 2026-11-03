# Guion para verificar el tutor en Kaggle

Pensado para pegárselo entero a un asistente que **sí puede manejar el navegador**
y operar la sesión de Kaggle. Está escrito para que no haga falta más contexto
que esto.

---

## Paso 0 · Traer el código

Los arreglos están en `integracion-m5`, commit **`9d1ea74`** o posterior,
**ya empujado a GitHub**. En Kaggle, antes de nada:

```python
!cd <repo> && git checkout integracion-m5 && git pull && git log --oneline -1
```

**Importante:** si el notebook clona la rama por defecto se trae `main`, que no
tiene nada de esto. El `git checkout integracion-m5` no es opcional.

**Reportar:** la línea de `git log`. Si no es `9d1ea74` o posterior, parar.

---

## Contexto para el asistente

Es un tutor de matemáticas (Qwen2.5-1.5B + LoRA) servido con Gradio. Corre en
Kaggle con **Python 3.13 y una Tesla T4**. El adaptador LoRA viene como dataset
adjunto, bajo una carpeta `qwen-lora`.

### Lo que ya está arreglado

1. **Gradio devolvía *Internal Server Error*** en el enlace público. Causa:
   pydantic ≥ 2.11 describe un campo `dict` con `additionalProperties: true` y
   `gradio_client 1.3.0` revienta al recorrerlo. Arreglo: `pydantic<2.11` y un
   guardia en `app/compat_gradio.py` que **no depende de una versión concreta**:
   comprueba si el fallo se dispara y solo parchea si hace falta. Con gradio
   6.26.0 funcionó así.
2. **La primera pregunta fallaba** con
   `AttributeError: module 'torch.distributed' has no attribute 'tensor'`.
   Arreglo: `app/compat_torch.py` importa el subpaquete al arrancar. **No hay
   que tocar el PyTorch ni el CUDA de Kaggle.**
3. **El adaptador no cargaba.** Su `adapter_config.json` lo guardó PEFT 0.19.1 y
   trae campos que las versiones publicadas no conocen. Arreglo:
   `pipeline._cargar_adaptador` omite los que no existan y lo avisa. Además,
   `torchao` instalado rompe esa carga: **hay que desinstalarlo.**

### Lo que falló en la corrida del 1 de noviembre, y es el objetivo de esta

El tutor **respondió bien** —`3/4 = 6/8`, `6/8 + 1/8`, `7/8`, en 9,3 s— y aun
así la corrida **no validó nada**: la pestaña Estado decía `degradado: True`,
faltaban `retrieval híbrido + reranker` y `enrutador consulta/problema`, y el
resultado salía como `NO MEDIDO`. Es decir: contestó el modelo base con LoRA y
búsqueda léxica, no la arquitectura S08-D que se pidió.

Causa probable, pendiente de confirmar: la celda de instalación bajaba
`huggingface_hub` a 0.26.5 (lo exige gradio 4.44), y eso arrastra
`transformers` a 4.45.2 y `tokenizers` a 0.20.x — pero **no** bajaba
`sentence-transformers`, que se quedó en el preinstalado de Kaggle corriendo
sobre tres piezas más viejas que las que vinieron con él.

**Una respuesta correcta no es una arquitectura completa.** Distinguir las dos
cosas es lo único que se pide de esta sesión.

---

## Reglas que no se saltan

- **No reinstalar `torch`.** El de Kaggle sirve. Cambiarlo arrastra compilaciones
  largas y un CUDA que puede no cuadrar con la T4.
- **No bajar `numpy`.** Bajo un torch y un scipy compilados contra numpy 2, es un
  error de ABI.
- **No inventar resultados.** Si algo falla, pegar el traceback **completo**, no
  un resumen. Si no se llegó a medir algo, decir que no se midió.
- **No presentar `revision_foto` como resultado medido.** Esa combinación nunca
  se ha evaluado; sale marcada EXPERIMENTAL y es correcto.
- **No evaluar fotos ni correr ablaciones** hasta que los cuatro niveles del
  retrieval estén en verde y la pregunta escrita funcione.
- **No usar los casos visuales reservados de M4** sin autorización explícita.
- **No guardar una versión nueva del notebook** sin avisar antes a Emmanuel.
- **No hacer `git add`, commit, push ni cambiar de rama** desde Kaggle.
- Si una celda tarda más de ~10 min sin salida, pararla y reportarlo en vez de
  esperar indefinidamente.

---

## Orden de ejecución, y qué salida confirma cada etapa

| # | Celda | Qué confirma que fue bien |
|---|---|---|
| 0 | `git checkout integracion-m5 && git pull` | `git log` muestra `9d1ea74` o posterior |
| 1 | `pip install -r requirements-kaggle.txt` | instala **solo** `rank-bm25`; nada más se mueve |
| 2 | `pip uninstall -y torchao` + reinicio | `find_spec('torchao')` → `None` |
| 3 | `find` del repo y del adaptador | dos rutas; el adaptador contiene `adapter_config.json` |
| 4 | `diagnostico_retrieval.py --solo-entorno` | las versiones cuadran con la columna «esperado» |
| 5 | `diagnostico_retrieval.py` | **los cuatro niveles en `OK`** y `degradado: False` |
| 6 | `compat_torch` / `compat_gradio` | `atributo después: True`, `get_api_info(): OK` |
| 7 | lanzar `app.py` | URL `*.gradio.live`, y las 7 etapas en `ok` |
| 8 | la pregunta escrita | `7/8`, y la cabecera **sin** `MODO DEGRADADO` |
| 9 | `medir_latencia.py` | arranque en frío aparte, p50 y p95 con su n |

**El orden importa.** El paso 5 decide si tiene sentido lanzar la app: con el
retrieval caído, la app arranca igual y responde igual, y eso es precisamente lo
que confundió la corrida anterior.

---

## Paso 1 · Dependencias · en Kaggle NO se baja ninguna versión

```python
!pip install -q -r M5_despliegue/config/requirements-kaggle.txt
```

Ese archivo instala **una sola cosa**: `rank-bm25`. Es el único paquete que la
imagen de Kaggle no trae.

**No usar `requirements.txt` aquí.** Ese es el perfil del Space, y aplicarlo en
Kaggle es lo que rompía el retrieval. Medido en la T4 el 2 de noviembre de 2026:
la imagen virgen importa `sentence_transformers` sin un fallo, y el `7/8` con
modo degradado venía de nuestra propia instalación. El error exacto era

```
AttributeError: module 'numpy.dtypes' has no attribute 'StringDType'
```

porque `numpy==1.26.4` deja sin ese atributo a jax 0.11.1, el fallo sube por
tensorflow 2.20 y `transformers.modeling_tf_utils`, y revienta el
`from sentence_transformers import CrossEncoder, SentenceTransformer` que hay
dentro de `buscador_hibrido`. El modelo de lenguaje no pasa por esa cadena: por
eso el tutor seguía contestando bien mientras el retrieval estaba muerto.

**Reportar:** la salida de pip entera, y cualquier línea
`ERROR: pip's dependency resolver...`. Esas líneas son el diagnóstico, no ruido.

---

## Paso 2 · Quitar `torchao` y reiniciar

```python
!pip uninstall -y -q torchao
import IPython; IPython.Application.instance().kernel.do_shutdown(True)
```

`torchao` viene en la imagen de Kaggle y rompe la carga del adaptador.
**Reaparece** al reiniciar el runtime o al volver a resolver un `pip install`,
así que se quita **después** del último install de la sesión. El reinicio no es
opcional: pydantic y starlette ya están importados por el stack de Kaggle.

Después del reinicio, comprobarlo:

```python
import importlib.util as u
print("torchao:", u.find_spec("torchao"))     # tiene que decir None
```

**Reportar:** esa línea. Si no dice `None`, repetir el `pip uninstall` y volver
a comprobar antes de seguir.

---

## Paso 3 · Localizar el repo y el adaptador

No suponer rutas: Kaggle monta los datasets en `/kaggle/input/<slug>/` y el slug
no siempre coincide con el nombre que se le puso al dataset.

```python
import subprocess
print(subprocess.run(["find", "/kaggle/input", "-name", "adapter_config.json"],
                     capture_output=True, text=True).stdout)
print(subprocess.run(["find", "/kaggle/working", "-maxdepth", "2", "-name", "M5_despliegue"],
                     capture_output=True, text=True).stdout)
```

**Reportar:** las dos rutas. El directorio que **contiene** `adapter_config.json`
es `RUTA_ADAPTADOR`; el que contiene `M5_despliegue` es `RUTA_REPO`.

---

## Paso 4-5 · El diagnóstico del retrieval · **esta es la celda clave**

```python
!cd RUTA_REPO && USE_TF=0 TUTOR_CONFIG=s08d \
  python M5_despliegue/evaluacion/diagnostico_retrieval.py
```

No necesita GPU ni carga Qwen. Sí descarga los dos modelos de la cadena de S08
(~500 MB) si no están en caché, que es justo lo que hay que comprobar.

Comprueba cuatro cosas **distintas**, y la corrida anterior las confundió:

| Nivel | Qué comprueba | Falla si… |
|---|---|---|
| 1 | el paquete importa | falta o es incompatible |
| 2 | los dos modelos cargan | no hay red ni caché, u `HF_HUB_OFFLINE` |
| 3 | `buscador_hibrido` devuelve un buscador | las versiones no encajan entre sí |
| 4 | una consulta real devuelve chunks y el enrutador decide | la cadena no produce resultados |

El nivel 1 es lo **único** que comprobaba la validación vieja, y de ahí salió el
«retrieval listo» que era falso.

**Qué tiene que salir:**

```
nivel 1 · OK    · paquete instalado
nivel 2 · OK    · modelo descargado o en caché
nivel 3 · OK    · componente montado
nivel 4 · OK    · componente efectivo (s08d)
Los cuatro niveles pasan: la cadena de S08 está montada y responde.
```

**Si alguno falla:** el guion imprime su traceback completo y dice la causa
probable —dependencia, red, caché, memoria o versión—. Son arreglos opuestos:
con un fallo de red, reinstalar paquetes no sirve; con uno de versión, tener
internet tampoco. **Pegar el traceback entero y parar aquí.** Lanzar la app con
el nivel 3 o 4 en rojo reproduce exactamente la corrida anterior.

Si el nivel 3 o 4 falla con el perfil de `requirements-kaggle.txt`, la causa ya
no puede ser una versión bajada de más, porque ese perfil no baja ninguna. Mirar
entonces el código del tutor contra el stack de la imagen: está escrito contra
transformers 4.45 y la imagen trae la 5.16.1. Anotar las versiones con
`--solo-entorno` y pegar el traceback.

**Reportar:** la salida completa, los cuatro niveles y, si hubo fallo, su
traceback.

---

## Paso 6 · Diagnóstico de torch y Gradio, antes de lanzar

```python
import os, sys
os.chdir("RUTA_REPO")              # sustituir
os.environ["USE_TF"] = "0"
sys.path.insert(0, "M5_despliegue/app")
import compat_torch, compat_gradio
print(compat_torch.diagnostico())
print()
print(compat_gradio.diagnostico())
```

| Línea | Qué significa |
|---|---|
| `atributo antes : False` | el fallo estaba ahí |
| `atributo después : True` | resuelto con el import |
| `Accesos por atributo encontrados…` | si nombra un archivo, **ese es el paquete culpable** — reportarlo |
| `get_api_info(): OK` | la página se puede servir |
| `El fallo se estaba disparando: False` | el pin de pydantic funcionó |

**Si `get_api_info()` dice FALLA, parar aquí.** Lanzar daría error 500.

**Reportar:** la salida completa de las dos llamadas, tal cual.

---

## Paso 7 · Lanzar

Para verificar la arquitectura, `TUTOR_CONFIG=s08d`, que es la configuración
medida. `revision_foto` es la de la demo visual y **nunca se ha evaluado**: solo
después de que `s08d` funcione.

```python
!cd RUTA_REPO && \
  USE_TF=0 \
  TUTOR_CONFIG=s08d \
  TUTOR_GRADIO_SHARE=1 \
  TUTOR_ADAPTADOR_M1=RUTA_ADAPTADOR \
  python M5_despliegue/app/app.py
```

Esperar a la URL `*.gradio.live` y abrirla en el navegador.

**Reportar:** las líneas que imprime al arrancar (`Entorno:`, `torch:`,
`torch.distributed.tensor:`) y la URL.

---

## Paso 8 · Completo o degradado · el criterio

Esto es lo que la corrida anterior no miró antes de dar el `7/8` por bueno.

**Modo completo** — todo esto a la vez:

| Dónde | Qué dice |
|---|---|
| cabecera | **no** aparece `MODO DEGRADADO` |
| Estado · `degradado` | `False` |
| Estado · `componentes_faltantes` | `ninguno` |
| Estado · `retrieval_activo` | contiene `hibrido` |
| Estado · `enrutador_activo` | `true` |
| Estado · `adaptador_cargado` | `m1` |
| Estado · `retrieval_por_etapas` | todas en `ok`, incluida la consulta de prueba |
| Estado · `carga_por_etapas` | las 7 en `ok` |

**Modo degradado** — cualquiera de estas, y entonces el resultado **no vale**
como validación de la arquitectura, aunque las respuestas sean correctas:

- la cabecera dice `MODO DEGRADADO (se pidió …; falta …)`
- `degradado: True`
- `componentes_faltantes` nombra algo
- `retrieval_activo` dice `lexico (REPLIEGUE: se pidió híbrido)`
- `resultado_medido` empieza por `NO MEDIDO`

En degradado, la pestaña Estado ahora trae además
`retrieval_por_etapas`, `retrieval_causa_probable` y `retrieval_traceback`:
**ahí está la causa**, no hay que adivinarla.

> Nota: con `TUTOR_CONFIG=revision_foto`, `resultado_medido` sale como
> «EXPERIMENTAL, sin medir» incluso en modo completo. Es correcto: esa
> combinación nunca se ha evaluado. Lo que no puede salir en ninguna
> configuración es `degradado: True`.

**Reportar:** la pestaña Estado completa, en texto.

---

## Paso 9 · Las preguntas

En la pestaña «Pregunta escrita»: **`¿Cuánto es 3/4 + 1/8?`**

- **Si responde:** la respuesta correcta es `7/8`. Reportar el texto íntegro y
  lo que diga «Cómo se obtuvo». En la corrida anterior tardó 9,3 s.
- **Si falla:** en la salida de la celda aparecerá
  `=== FALLO EN LA ETAPA: <nombre> ===` con el traceback completo. **Pegarlo
  entero.**

Después, las dos vías del enrutador, que es lo que distingue S08-D de S08-C:

- `¿Cuál es la nota mínima aprobatoria del colegio?` → debería **buscar** en los
  documentos y citar la fuente.
- `Explica por qué 9 es un número primo.` → debería **negar la premisa**, no
  explicarlo. 9 = 3 × 3.

Si el enrutador busca en la aritmética o no busca en la factual, decirlo: es la
pieza que explica los +4 casos de S08-D y su decisión es observable.

**Reportar:** las tres respuestas completas y, de cada una, si trajo fuentes.

---

## Paso 10 · Latencias · SOLO si el paso 9 responde bien y sin degradar

Parar la app (interrumpir la celda) y correr:

```python
!cd RUTA_REPO && USE_TF=0 TUTOR_CONFIG=s08d \
  TUTOR_ADAPTADOR_M1=RUTA_ADAPTADOR \
  python M5_despliegue/evaluacion/medir_latencia.py --repeticiones 8
```

Separa **arranque en frío** (una muestra; incluye cargar el modelo) de las
**consultas calientes** (p50 y p95, con el n al lado). Las peticiones fallidas se
cuentan aparte y **no** entran en la estadística: durante la depuración se vieron
fallos a los 5,71–5,83 s, y eso es tiempo hasta la excepción, no latencia.

**Reportar:** la tabla final completa, incluido el n. No redondear ni resumir.

---

## Fotos y ablaciones · no en esta sesión, salvo que todo lo anterior pase

Solo con los cuatro niveles en verde, `degradado: False` y las tres preguntas
respondidas. Y entonces, en este orden: lectura doble consistente de una foto,
confirmación editable de la transcripción, y revisión del procedimiento con el
verificador. **Sin tocar los casos visuales reservados de M4.**

## Paso 11 · Prueba de humo de la vía visual · ejecutada el 2 de noviembre de 2026

Esta prueba **no necesita los pasos 1 a 10**. `leer_imagen` no pasa por
`buscador_hibrido` ni por el adaptador LoRA: carga solo el VLM. Corrió en una
sesión recién arrancada, sin `pip install` y sin desinstalar `torchao`.

**Lo que esta prueba NO mide, y no se puede deducir de su salida:** exactitud de
lectura, localización del paso con error, ni la salvaguarda de ilegibles. Las 12
imágenes son fotos de texto tipografiado en pantalla con el identificador del
caso impreso, no hojas manuscritas, y `caso-10.jpg` se lee bien aunque `m5v-10`
exija justo lo contrario. El script no imprime ninguna tasa de acierto a
propósito: un «9/12» acabaría copiado al informe.

El eval set oficial sigue siendo `data/eval_set_m5_vision.jsonl`, con sus 23
tomas de nombre exacto en `m4/eval_set/imagenes_m5/`. Este paquete no las
sustituye y sus nombres no se tocaron para que pasaran por ellas.

### El dataset, y dónde lo monta Kaggle de verdad

El paquete está en `M5_despliegue/evalset-m5/kaggle_dataset_humo/` y comprimido
en `kaggle_dataset_humo.zip`. Subido como dataset privado `evalsetm5`, Kaggle
descomprime el ZIP y lo monta **no** en `/kaggle/input/evalsetm5/`, sino en:

    /kaggle/input/datasets/emmanuelcastaedacano/evalsetm5/kaggle_dataset_humo/

con `imagenes/`, `manifest.json` y `README.md` dentro. El panel *Input* del
notebook muestra el nombre corto; la ruta del sistema de archivos lleva el prefijo
`datasets/<usuario>/`. Vale la pena comprobarlo antes de dar una ruta por buena.

### Las celdas, en orden · son las 1 a 5 del notebook

**Celda 1 · clonar.** Es la primera celda de código, la que ya existía. Clona
solo si `ProyectoIA/.git` no está, así que en una sesión nueva trae el commit de
`integracion-m5`. Arrancarla inicia la sesión. Salida esperada: `Cloning into
'ProyectoIA'...` y `## integracion-m5...origin/integracion-m5`.

**Celda 2 · comprobar que corre el código de hoy.**

```python
!cd /kaggle/working/ProyectoIA && git log --oneline -1 && ls -la M5_despliegue/evaluacion/humo_vision.py && grep -n 'device_map' M5_despliegue/app/pipeline.py && echo '--- DATASET DE HUMO ---' && ls /kaggle/input/datasets/emmanuelcastaedacano/evalsetm5/kaggle_dataset_humo/ && ls /kaggle/input/datasets/emmanuelcastaedacano/evalsetm5/kaggle_dataset_humo/imagenes/
```

Tiene que imprimir el commit, el tamaño de `humo_vision.py`, la línea del
`device_map` y los 12 `caso-NN.jpg`. Si falta el `device_map`, el repositorio de
Kaggle es anterior al arreglo y la prueba morirá al generar.

**Celda 3 · diagnóstico, opcional.** `ls /kaggle/input/` y un `find`. Solo hace
falta si la celda 2 no encuentra el dataset, para localizar la ruta real.

**Celda 4 · la prueba.**

```python
!cd /kaggle/working/ProyectoIA && env USE_TF=0 HUMO_CASOS=caso-01,caso-07,caso-10 HUMO_DIR=/kaggle/input/datasets/emmanuelcastaedacano/evalsetm5/kaggle_dataset_humo python -u M5_despliegue/evaluacion/humo_vision.py
```

`HUMO_DIR` es opcional: sin él el script busca el dataset solo. Se pasa para que
la corrida no dependa de ese rastreo. `HUMO_CASOS` elige los casos.

Antes de leer nada, compara el SHA-256 de cada archivo con el del
`manifest.json`. Sin eso, un fallo de lectura no se distingue de un archivo que
Kaggle recomprimió al subirlo.

**Celda 5 · leer el JSON.** Imprime `es_eval_set_de_vision`, la integridad, una
línea por caso y el bloque `no_mide`.

### Lo que salió

```
integridad: 12/12 ok
configuración: Revisión de foto · lectura confirmada + verificador (EXPERIMENTAL) · vision=True

caso-01 (1024x331)  93.9s   ok True · etapa lectura · confirmable True · lecturas_coinciden True · 3 pasos
caso-07 (1024x318)  12.11s  ok True · etapa lectura · confirmable True · lecturas_coinciden True · 3 pasos
caso-10 (1024x277)   8.64s  ok True · etapa lectura · confirmable True · lecturas_coinciden True · 3 pasos

escrito: M5_despliegue/resultados/humo_vision.json
Ninguna tasa de acierto: este conjunto no la soporta.
```

Los 93,9 s del primer caso son casi todos la descarga del VLM: 4,42 GB de
`Qwen/Qwen2-VL-2B-Instruct`. Con el modelo en caché, un caso cuesta entre 8 y 12
segundos en la T4.

`motivo_no_fiable` salió vacío en los tres, incluido `caso-10`. Es lo esperado y
es justo el problema: la foto se lee, así que la salvaguarda no se ejerce.

### Si falla

- `"addmm_impl_cpu_" not implemented for 'Half'` — el VLM se quedó en CPU con
  pesos fp16. El repositorio de Kaggle es anterior al `device_map` de
  `pipeline.py:1067`. Vuelve a clonar.
- `no encuentro caso-01.jpg` — el script nombra los candidatos que probó. Usa la
  celda 3 y pasa la ruta correcta en `HUMO_DIR`.
- `SHA DISTINTO` o `AUSENTE` en la integridad — el archivo en Kaggle no es el que
  se construyó. Vuelve a subir el dataset antes de interpretar nada.
- Celdas en cola con la CPU a cero — el editor perdió el kernel. Recarga la
  pestaña; si vuelve «Draft Session off», la sesión murió y hay que arrancarla.

### Qué no repetir con la sesión ya preparada

Ni la celda 1 ni ninguna celda de instalación. La celda 1 vuelve a clonar y, si
la variante con `rm -rf ProyectoIA` se usa, borra también `resultados/` y los
adaptadores copiados. Con el repositorio ya clonado basta con las celdas 4 y 5.

---

## Paso 12 · Evaluación de las 12 hojas manuscritas · fase 3

Repetible tal cual. Si algo de aquí no coincide con lo que ves, para y mira
qué cambió antes de seguir.

### Lo que hay que adjuntar

Panel **Add Input**, filtro *Datasets*, buscar `emmanuelcastaedacano`:

| Dataset | Para qué | ¿Hace falta? |
|---|---|---|
| `adaptadorqwen` | el LoRA `qwen-lora` | **sí** |
| `datasetevalm5v4` | trae `evalset-m5/Imagenes_manuscritas2/caso-01..12.jpeg`, las 12 hojas | **sí** |
| `evalsetm5` | paquete de humo | no |
| `evalset-m5` | fotos de texto en pantalla | no |

`datasetevalm5v4` es un volcado de todo `M5_despliegue/evalset-m5/`: también
trae las fotos de pantalla y las `vis-*` de M4. Se copia **solo**
`Imagenes_manuscritas2/`.

### Celda A · código y fotos

```python
!cd /kaggle/working/ProyectoIA && git fetch -q --depth 1 origin integracion-m5 && git reset --hard -q FETCH_HEAD && git log --oneline -1
import pathlib,shutil
o=pathlib.Path("/kaggle/input/datasets/emmanuelcastaedacano/datasetevalm5v4/evalset-m5/Imagenes_manuscritas2")
d=pathlib.Path("/kaggle/working/ProyectoIA/m4/eval_set/imagenes_m5")
d.mkdir(parents=True,exist_ok=True)
print("origen:",o.is_dir())
_=[shutil.copyfile(o/("caso-%02d.jpeg"%i), d/("m5v-%02d.jpg"%i)) for i in range(1,13)]
!sha256sum /kaggle/working/ProyectoIA/m4/eval_set/imagenes_m5/*.jpg
```

Esperado: `origen: True` y doce SHA-256. El primero es
`40100dc862e620f6bd4e93c7f3ce4d9855f968ebb86bc38e0190caee0b604dec` para
`m5v-01.jpg`. Si un hash no coincide con el del manifiesto, la foto no es la
misma y no se sigue.

### Celda B · regresiones y contrato · sin GPU

```python
!cd /kaggle/working/ProyectoIA && USE_TF=0 python -m pytest M5_despliegue/tests/test_m5.py -q 2>&1 | tail -20
!cd /kaggle/working/ProyectoIA && USE_TF=0 python M5_despliegue/evaluacion/evaluar_vision_m5.py --contrato
```

Esperado: `102 passed` y `fotos: 12 de 12`. Entre esas pruebas están las tres
regresiones de la fase 2: `3/4 + 1/8` da `7/8` ejecutando `operar_fracciones`,
`347 × 28` da `9716` ejecutando `evaluar_expresion`, y dos preguntas seguidas
no comparten respuesta. Tardan unos 2 minutos porque cargan el corpus.

### Celda C · las 12 fotos · **necesita GPU T4**

```python
!cd /kaggle/working/ProyectoIA && USE_TF=0 TUTOR_ADAPTADOR_M1=/kaggle/working/ProyectoIA/adaptadores/qwen-lora python -u M5_despliegue/evaluacion/evaluar_vision_m5.py --correr
```

Primero baja el VLM (4,42 GB, ~1 min) y lo cronometra aparte: una latencia de
lectura que incluye la descarga no es una latencia de lectura. Luego, por foto,
una línea con `confirmable`, `coinciden`, `pasos`, y si el caso está en
conflicto, `BLOQUEADO, sin puntuar`.

Escribe `M5_despliegue/resultados/vision_m5.json` y `vision_m5.csv`. **Los dos
viven en `/kaggle/working` y mueren con la sesión**: hay que descargarlos desde
el panel *Output* antes de cerrar.

### Celda D · la interfaz, para mirarla

```python
%cd /kaggle/working/ProyectoIA
!env USE_TF=0 TUTOR_CONFIG=revision_foto TUTOR_GRADIO_SHARE=1 TUTOR_PRECARGAR=1 TUTOR_ADAPTADOR_M1=/kaggle/working/ProyectoIA/adaptadores/qwen-lora python -u M5_despliegue/app/app.py
```

`TUTOR_PRECARGAR=1` es nuevo y no es cosmético: sin él la primera pregunta
tarda ~80 s y esa espera, por el túnel de `gradio.live`, es la que acaba
devolviendo una página HTML al cliente —`Could not parse server response:
SyntaxError: Unexpected token '<'`— y dejando en pantalla la respuesta de la
pregunta anterior. Con la precarga la espera se paga en la celda, donde no hay
ningún cliente esperando.

Para la ruta de solo texto, la misma celda con
`TUTOR_CONFIG=s08d_verificador`; esa configuración no monta visión y se salta
los 4,42 GB.

### Lo que no hay que repetir

* No reinstalar el entorno. Solo `pip uninstall -y torchao` (peft 0.20 exige
  torchao > 0.16 y la imagen trae 0.10) y `pip install rank-bm25==0.2.2`.
  Torch no se toca.
* No usar `evalsetm5` ni `evalset-m5` ni `imagenes_jpg_humo/` como eval set:
  son fotos de texto en pantalla.
* No usar `imagenes_manuscritas1/`: son las `vis-*` de M4, ya gastadas.

## Qué devolver al terminar

1. En qué paso se llegó y en cuál se paró.
2. La salida del diagnóstico de retrieval, con sus cuatro niveles.
3. **Tracebacks completos** de lo que fallara, con la etapa que los imprime.
4. La pestaña Estado completa, y si fue modo completo o degradado.
5. Si `compat_torch` nombró algún paquete en «Accesos por atributo», cuál.
6. Las respuestas literales a las tres preguntas.
7. Las latencias con su n, si se llegó al paso 10.
8. Las versiones reales de la sesión (`--solo-entorno`), para el informe.
9. Lo que no se pudo comprobar, dicho como tal.

Si algo no funcionó, **no declarar el tutor listo**. Y una respuesta correcta en
modo degradado **no** es un tutor listo: es lo que ya pasó una vez.
