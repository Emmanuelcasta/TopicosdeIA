# Despliegue · qué ejecutar y dónde

**Estado al 1 de noviembre de 2026.** La interfaz se probó en Kaggle con una T4
y se obtuvo un enlace temporal de Gradio, pero la prueba funcional no pasó: al
enviar una pregunta, el tutor respondió `Tutor NO disponible` por un error de
dependencias de PyTorch/PEFT. No hay Space permanente ni una corrida exitosa que
permita presentar el tutor como listo. Los enlaces temporales de Kaggle caducan
y no son una URL de producción.

## 1 · Local, sin GPU

```bash
pip install -r M5_despliegue/config/requirements.txt
python M5_despliegue/app/app.py          # http://localhost:7860
```

Sin GPU el modelo corre en CPU y va lento, o no carga si falta `torch`. **La app
arranca igual** y la pestaña *Estado* explica qué falta. Es deliberado: una demo
que se cae por una dependencia ausente no sirve para enseñar nada.

Pruebas que corren sin GPU:

```bash
python -m pytest M5_despliegue/tests -q      # 28 pruebas
```

## 2 · En Colab, con GPU

```python
!git clone <repo> && cd ProyectoIA
!pip install -r M5_despliegue/config/requirements.txt
```

Montar Drive para los adaptadores (no están en Git):

```python
from google.colab import drive; drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/ProyectoIA/adaptadores adaptadores
```

Para la demo visual en Kaggle, usar la configuración experimental de producto y
pedir el enlace temporal de Gradio de forma explícita:

```bash
TUTOR_CONFIG=revision_foto TUTOR_GRADIO_SHARE=1 python M5_despliegue/app/app.py
```

`TUTOR_GRADIO_SHARE` está apagada por defecto. El enlace sirve para la sesión de
Kaggle; no sustituye una URL permanente y no se activa en Hugging Face Spaces,
que ya publica su propia URL.

### El enlace de Kaggle que devolvía *Internal Server Error*

**Qué pasaba.** La celda imprimía una URL `*.gradio.live` y al abrirla salía
*Internal Server Error* o «No interface is running», con esta traza en el log:

```
gradio/routes.py        api_info()
gradio/blocks.py        get_api_info()
gradio_client/utils.py  json_schema_to_python_type() -> _json_schema_to_python_type()
                        -> get_type()
TypeError: argument of type 'bool' is not iterable
```

**La causa.** `gradio==4.44.1` declara `pydantic>=2.0` **sin tope**. Desde
**pydantic 2.11**, un campo anotado como `dict` pelado se describe en JSON Schema
con `additionalProperties: true`, y el campo `meta` de `FileData` —el modelo que
publica `gr.Image`— es justo eso. `gradio_client 1.3.0` recorre ese valor como si
siempre fuera un subesquema:

```python
if "additionalProperties" in schema:
    des += [f"str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}"]
```

y acaba en `get_type(True)`, cuya primera línea es `if "const" in schema`.
`"const" in True` lanza `TypeError`. En Gradio 4.44 la ruta que sirve la página
consulta la información de la API, así que un esquema que no se puede describir
impide servir **toda** la interfaz, no solo `/info`.

Verificado con `gradio 4.44.1` + `gradio_client 1.3.0`:

| pydantic | `additionalProperties` | `get_api_info()` |
|---|---|---|
| 2.7.4 · 2.8.2 · 2.9.2 · 2.10.6 | clave ausente | OK |
| 2.11.9 · 2.13.5 | `true` | `TypeError` |

**El arreglo.** `config/requirements.txt` fija `pydantic>=2.0,<2.11` y repite
`gradio-client==1.3.0` (el pin transitivo de `gradio` no basta en Kaggle, donde
`pip` conserva lo preinstalado cuando puede). Además `app/compat_gradio.py` añade
el guardia que a `gradio_client` le falta —un subesquema booleano `true` es `Any`
y `false` es `None`, que es la semántica de JSON Schema— por si el pin no se puede
imponer. No esconde el error: implementa la lectura correcta, y el diagnóstico
dice si el fallo se estaba disparando o si el parche fue solo prevención.

### Celdas de Kaggle · actualizar dependencias y relanzar

**1 · Dependencias.** Pydantic ya está importado por el stack preinstalado de
Kaggle, así que **hay que reiniciar el kernel** después de esta celda:

```python
%pip install -q "gradio==4.44.1" "gradio-client==1.3.0" "pydantic>=2.0,<2.11" "huggingface_hub==0.26.5" "starlette<1.0"
import IPython; IPython.Application.instance().kernel.do_shutdown(True)
```

El pin de `huggingface_hub` evita el `ImportError: cannot import name 'HfFolder'`
que Gradio 4.44.1 puede producir al importar OAuth con la versión preinstalada en
Kaggle. El diagnóstico debe ejecutarse sin importar Gradio directamente: así
`compat_gradio` puede informar de un fallo de importación en vez de detener la
celda antes de mostrar las versiones. El pin `starlette<1.0` conserva la firma
posicional de `TemplateResponse` que usa Gradio 4.44.1; con Starlette 1.x la ruta
raíz falla al renderizar y el enlace `gradio.live` responde 500.

Si también ejecutas la celda de dependencias generales del proyecto, aplica
estos dos pines **al mismo tiempo que esa instalación**. De lo contrario, el
resolver de `pip` puede dejar otro `huggingface_hub` en el proceso que arranca
el modelo, aunque la instalación inicial de Gradio hubiera pasado:

```python
%cd /kaggle/working/ProyectoIA
!grep -vE '^(torch|torchvision|torchaudio)(==|[<>=])' M5_despliegue/config/requirements.txt > /tmp/requirements-kaggle.txt
!printf '%s\n' 'huggingface_hub==0.26.5' 'starlette<1.0' >> /tmp/requirements-kaggle.txt
%pip install -q -r /tmp/requirements-kaggle.txt
import IPython; IPython.Application.instance().kernel.do_shutdown(True)
```

Después del reinicio, confirma desde el mismo Python que usará la app:

```python
import huggingface_hub, starlette
print("huggingface_hub:", huggingface_hub.__version__)
print("starlette:", starlette.__version__)
from huggingface_hub import HfFolder
```

Se espera `huggingface_hub: 0.26.5`, Starlette menor que `1.0` y que el import de
`HfFolder` no lance error. Si falla esta comprobación, no lances la app todavía.

**2 · Diagnóstico** (tras el reinicio, antes de lanzar nada):

```python
import os, sys
os.chdir("/kaggle/working/ProyectoIA")          # ajusta a tu ruta del repo
sys.path.insert(0, "M5_despliegue/app")
os.environ["TUTOR_CONFIG"] = "revision_foto"
import compat_gradio
print(compat_gradio.diagnostico())
```

Debe imprimir las versiones reales del runtime Kaggle (en la corrida reportada,
Python fue 3.13; no se debe asumir 3.11):

```
Versiones instaladas:
  python         3.11.x
  gradio         4.44.1
  gradio_client  1.3.0
  huggingface_hub 0.26.5
  pydantic       2.10.6          <- cualquier 2.x por debajo de 2.11
  pydantic_core  2.27.2
  fastapi        …
  starlette      …
Parche de compatibilidad aplicado: True
El fallo se estaba disparando: False      <- False = el pin funcionó
get_api_info(): OK · 4 endpoints descritos. La página se puede servir.
```

Las dos líneas que deciden: **`get_api_info(): OK`** y **`se estaba disparando:
False`**. Si sale `True`, el pin de pydantic no se aplicó —mira si otro paquete
lo arrastró— pero el parche sostiene la app igual. Si sale `get_api_info():
FALLA`, no lances nada: el enlace daría error 500.

**3 · Preparar el adaptador LoRA de Kaggle.** El runtime trae `torchao` y
PyTorch propios. No reemplaces el PyTorch/CUDA de Kaggle con el pin de Space. El
adaptador `qwen-lora` actual requiere PEFT 0.20 para leer su configuración
(`alora_...`); PEFT 0.13.2 no la interpreta. En la sesión probada, quitar el
`torchao` heredado permitió que PEFT 0.20 leyera el adaptador y que Qwen + LoRA
cargaran en una celda aislada:

```python
!pip uninstall -y torchao
%pip install -q --no-deps --force-reinstall "peft==0.20.0"
import IPython; IPython.Application.instance().kernel.do_shutdown(True)
```

Después del reinicio, vuelve a preparar el repositorio y copiar el adaptador.
La app usa PyTorch; `USE_TF=0` evita una ruta de integración TensorFlow
incompatible que también se encontró en el Python 3.13 preinstalado de Kaggle.
Esto **no resolvió por sí solo** el error final de la app (`module
'torch.distributed' has no attribute 'tensor'`), así que no lo consideres un
arreglo completo.

**4 · Lanzar.** El adaptador está en un dataset adjunto, así que hay que
apuntarlo o la app arranca en modo degradado:

```python
!cd /kaggle/working/ProyectoIA &&   USE_TF=0 TUTOR_CONFIG=revision_foto   TUTOR_GRADIO_SHARE=1   TUTOR_ADAPTADOR_M1=/kaggle/working/ProyectoIA/adaptadores/qwen-lora   python M5_despliegue/app/app.py
```

`lanzar()` imprime las versiones y, si tuvo que parchear, lo dice antes de
levantar el servidor. Comprueba en la cabecera de la interfaz que **no** sale
`MODO DEGRADADO`: si el adaptador no se encontró, saldría, y lo que responde no
sería la configuración que se quiere demostrar.

### La primera pregunta fallaba: `torch.distributed` sin `tensor`

**Qué pasaba.** La página abría, se enviaba «¿Cuánto es 3/4 + 1/8?» y antes de
generar nada:

```
AttributeError: module 'torch.distributed' has no attribute 'tensor'
```

**La forma del error es la pista.** `AttributeError` —no `ModuleNotFoundError`—
significa que alguien escribe `torch.distributed.tensor.Algo` **sin importar
antes el subpaquete**. `tensor` es un subpaquete de `torch.distributed`: está en
el disco, pero `import torch` no lo carga, así que el atributo no existe hasta
que alguien hace el import explícito. Comprobado con torch 2.4.1:

```
hasattr(torch.distributed, "tensor")   -> False
import torch.distributed.tensor        -> OK        (y entonces hasattr -> True)
```

Por eso **no hay que tocar el PyTorch ni el CUDA de Kaggle**: el submódulo ya
está, solo falta importarlo. `app/compat_torch.py` lo hace antes de importar
`pipeline`, y además trae un buscador que dice **qué paquete** hace ese acceso.

Qué se descartó, mirando el código real de cada paquete en vez de suponerlo:
`transformers` 4.45.2 y 4.57.6, `accelerate` 0.34.2 y 1.10.1, `peft` 0.17.1 y
`huggingface_hub` **no** acceden por atributo — todos usan
`from torch.distributed.tensor import ...`, que daría otro error. El acceso está
en algún paquete del entorno de Kaggle **sin identificar todavía**, y el
diagnóstico de abajo lo nombra.

### El error ya no se esconde: la carga va por etapas

Antes toda la carga iba dentro de un solo `except Exception` que guardaba el
mensaje recortado a 160 caracteres, sin decir en qué paso ocurría. Ahora cada
etapa se cronometra y, si falla, **imprime su traceback completo** en la salida
de la celda y queda en la pestaña Estado:

```
preparar opciones de carga
tokenizer
modelo base
mover a cuda
localizar adaptador
adaptador LoRA (PeftModel.from_pretrained)
armar agente (retrieval, enrutador, herramientas)
```

En la pestaña **Estado** aparecen como `carga_por_etapas`, con `etapa_fallida` y
las últimas 12 líneas del `traceback`.

### El segundo fallo, que el aislamiento destapó: el adaptador y PEFT

Al aislar las etapas apareció otro problema distinto, en
`adaptador LoRA (PeftModel.from_pretrained)`:

```
TypeError: __init__() got an unexpected keyword argument 'alora_invocation_tokens'
```

`adaptadores/qwen-lora/adapter_config.json` lo guardó **PEFT 0.19.1** (lo dice su
propio campo `peft_version`), y trae campos que **ninguna PEFT publicada en PyPI
—hasta 0.17.1— conoce**: `alora_invocation_tokens`, `use_qalora`,
`qalora_group_size`, `trainable_token_indices`. Eso es lo que obligó a instalar
una PEFT muy nueva en Kaggle, y lo que habría roto el Space, donde esa versión no
se puede instalar.

**El arreglo.** `pipeline._cargar_adaptador` reintenta omitiendo solo los campos
que la `LoraConfig` instalada no acepta. Se puede porque **todos valen su valor
por defecto** —`None`, `False`, y un `qalora_group_size` inerte con
`use_qalora=False`—: codifican «esta función está apagada», así que omitirlos no
cambia ni un peso. Si alguno llevara un valor activo, el aviso lo diría y habría
que exigir la versión de PEFT en vez de filtrar; hay una prueba que lo vigila.

Queda anotado en la pestaña Estado:

```
adaptador guardado con PEFT 0.19.1 y leído con 0.13.2: se omitieron
['alora_invocation_tokens', 'qalora_group_size', …] por no existir en la
instalada; todos valían su valor por defecto
```

Verificado cargando el adaptador real con PEFT 0.13.2 y respondiendo la pregunta
de fracciones de principio a fin.

### Celdas de Kaggle · diagnóstico y arranque

**1 · Dependencias** (sesión limpia). No reinstala torch: el de Kaggle sirve, y
cambiarlo arrastraría compilaciones largas y un CUDA que puede no cuadrar con la
T4. Solo se tocan los paquetes de la interfaz, que son ligeros y puros Python:

```python
!pip install -q "gradio==4.44.1" "gradio-client==1.3.0"                 "pydantic>=2.0,<2.11" "starlette<1.0" "huggingface_hub==0.26.5"
!pip uninstall -y -q torchao
import IPython; IPython.Application.instance().kernel.do_shutdown(True)
```

`torchao` se quita porque su integración se activa sola con Transformers y no
hace falta aquí. El reinicio no es opcional: pydantic y starlette ya están
importados por el stack de Kaggle.

**2 · Diagnóstico** (tras el reinicio, antes de lanzar nada):

```python
import os, sys
os.chdir("/kaggle/working/ProyectoIA")          # ajusta a tu ruta
os.environ["USE_TF"] = "0"
sys.path.insert(0, "M5_despliegue/app")
import compat_torch, compat_gradio
print(compat_torch.diagnostico())
print()
print(compat_gradio.diagnostico())
```

De `compat_torch` lo que importa son tres líneas:

```
  atributo antes         : False      <- el fallo estaba aquí
  atributo después       : True       <- resuelto con el import
Accesos por atributo encontrados en site-packages:
  <paquete>/<archivo>.py:<línea>: ...   <- el culpable, si está en purelib
```

Si sale `ninguno` y el fallo persiste, el acceso está en código compilado:
manda el traceback completo que ahora imprime la etapa.

**3 · Lanzar.** La ruta del dataset se comprueba, no se supone: Kaggle la monta
en `/kaggle/input/<slug>/` y el slug no siempre coincide con el nombre que se le
puso. Localízala primero:

```python
import subprocess
print(subprocess.run(["find", "/kaggle/input", "-name", "adapter_config.json"],
                     capture_output=True, text=True).stdout)
```

El directorio que contiene ese archivo es el que va en `TUTOR_ADAPTADOR_M1`.

```python
!cd /kaggle/working/ProyectoIA &&   USE_TF=0   TUTOR_CONFIG=revision_foto   TUTOR_GRADIO_SHARE=1   TUTOR_ADAPTADOR_M1=$RUTA_ADAPTADOR   python M5_despliegue/app/app.py
```

**4 · Qué comprobar, en este orden** —y no se pasa al siguiente hasta que el
anterior funcione:

1. La cabecera **no** dice `MODO DEGRADADO`.
2. Pestaña **Estado**: `adaptador_cargado: m1`, `retrieval_activo` con
   `hibrido`, `enrutador_activo: true`, `componentes_faltantes: ninguno`.
3. La pregunta escrita «¿Cuánto es 3/4 + 1/8?» responde. Si falla, la salida de
   la celda trae `=== FALLO EN LA ETAPA: … ===` con el traceback entero.
4. Solo entonces, las latencias (abajo). **Nada de fotos ni ablaciones** antes
   de que la pregunta escrita funcione.

### Latencias, cuando la pregunta ya responda

```python
!cd /kaggle/working/ProyectoIA && USE_TF=0 TUTOR_CONFIG=revision_foto \
  TUTOR_ADAPTADOR_M1=$RUTA_ADAPTADOR \
  python M5_despliegue/evaluacion/medir_latencia.py --repeticiones 8
```

Separa **arranque en frío** (la carga del modelo, que se paga una vez) de las
consultas **calientes**, y da p50 y p95 con el número de muestras al lado. Los
5,71–5,83 s de las peticiones que fallaban **no son latencia del modelo**: son
el tiempo hasta la excepción, y el script no los puede confundir porque solo
cuenta respuestas con `ok=True`.

## 3 · Hugging Face Space

1. Crear un Space de tipo **Gradio**, SDK `4.44.1`.
2. Subir: `M5_despliegue/app/*.py` como `app.py` y módulos, el contenido de
   `config/requirements.txt`, y `config/README_SPACE.md` como `README.md` del
   Space (lleva el *front-matter* que HF necesita).
3. Subir también `scripts/` y `data/`: la app los importa.
4. Los adaptadores pesan ~70 MB cada uno. O se suben al Space con Git LFS, o se
   publican como modelo en el Hub y se descargan al arrancar. **Sin adaptador la
   app funciona con el modelo base y lo advierte en pantalla.**

### Variables del Space

| Variable | Tipo | Para qué |
|---|---|---|
| `TUTOR_CONFIG` | variable | **qué configuración se sirve**; por defecto `s08d` (18/21). Ver la tabla de abajo |
| `TUTOR_CUANTIZAR` | variable | `1` para int4; vacío para fp16 |
| `TUTOR_ADAPTADOR_M1` | variable | ruta del LoRA de M1; por defecto `adaptadores/qwen-lora` |
| `TUTOR_ADAPTADOR_AGENTE` | variable | ruta del LoRA de S10; por defecto `adaptadores/qwen-lora-agente` |
| `TUTOR_LOG_TEXTO` | variable | dejar **vacío**: no registrar texto del usuario |
| `TUTOR_GRADIO_SHARE` | variable | dejar vacío en el Space; `1` solo para obtener un enlace temporal desde Kaggle |
| `WANDB_PROJECT` | variable | nombre del proyecto, por defecto `tutor-m5` |
| `WANDB_API_KEY` | **secreto** | opcional; sin ella la telemetría queda apagada |

`.env.example` documenta todas. **Ninguna lleva un secreto en el repositorio.**

### `TUTOR_CONFIG` · qué se sirve y qué lo respalda

| Valor | Qué monta | Resultado medido |
|---|---|---|
| `s08d` **(por defecto)** | LoRA M1 + RAG híbrido con reranker + enrutador | **18/21**, 2 alucinaciones |
| `s08c` | igual, sin enrutador | 13/21, 5 alucinaciones |
| `solo_lora` | LoRA M1 pelado · respaldo simple | 14/21, 7 alucinaciones |
| `agente_c5ft` | adaptador de S10 + herramientas + ReAct | 13/21, 8 alucinaciones |
| `s08d_herramientas` | `s08d` + las 18 herramientas | **sin medir** |
| `s08d_verificador` | lo anterior + verificador aritmético | **sin medir** |
| `s08d_normalizador` | `s08d` + normalizador de preguntas | **sin medir** |
| `s08d_vision` | `s08d` + la vía visual, y nada más | **sin medir** |
| `revision_foto` | producto: dos lecturas de foto + confirmación + verificador aritmético | **sin medir** |

Las fuentes exactas (archivo, fila y columna) están en
`app/configuraciones.py`, campo `fuente` de cada entrada. Una variante sin medir
se marca en la cabecera de la interfaz y en `/salud`; **no se presenta como
resultado**.

**El adaptador no se elige por lo que haya en el disco.** Si falta el que pide la
configuración, la app responde con el modelo base y lo dice, en vez de cargar el
otro adaptador en silencio.

### Modo degradado · qué ve el usuario si falta una pieza

`18/21` describe un sistema con sus tres componentes: LoRA de M1 + híbrida con
reranker + enrutador. Si falta alguno, la app **sigue respondiendo** por el
repliegue, pero:

- la cabecera muestra `🔴 MODO DEGRADADO · resultado de esta ejecución: NO MEDIDO`
  con la lista de componentes faltantes y qué pieza sigue funcionando;
- `/salud` devuelve `degradado: true`, `componentes_faltantes: [...]`,
  `configuracion_efectiva` distinta de `configuracion_solicitada`, y
  `resultado_medido` empezando por `NO MEDIDO`;
- la cifra histórica solo se menciona para decir que **no** es de esta ejecución.

Qué comprobar antes de la demo, en este orden:

```bash
curl -s $URL/salud | python -m json.tool | grep -E "degradado|faltantes|efectiva"
```

Si `degradado` es `true`, lo que está desplegado no es lo que se midió. Los
causantes habituales: el adaptador no se subió con Git LFS, o faltan
`sentence-transformers` / `rank-bm25` / `scikit-learn` en el Space.

**Para que `s08d` sea de verdad S08-D hacen falta `sentence-transformers`,
`rank-bm25` y `scikit-learn`** (van fijadas en `requirements.txt`). Sin ellas la
app se repliega a búsqueda léxica sin enrutador y la salud lo marca como
`lexico (repliegue: se pidió híbrido)`. Para comprobarlo antes de la demo:

```
python M5_despliegue/evaluacion/verificar_retrieval.py
```

Compara los pesos del enrutador, su validación leave-one-out y las 21 rutas del
eval set contra las que dejó el notebook. No carga el LLM y no mide aciertos: solo
dice si la pieza servida es la medida.

## 4 · W&B

Crear el proyecto y añadir `WANDB_API_KEY` como secreto del Space. Se registran
latencia, errores, uso de herramientas y versión; **nunca el texto del usuario**
salvo que alguien active `TUTOR_LOG_TEXTO` a sabiendas.

Si la clave falta, `telemetria.py` lo anota una vez y sigue. Hay una prueba que
lo verifica.

## 5 · Health check

Pestaña **Estado**: versión (commit), tiempo activo, dispositivo, adaptador
cargado, corpus, nº de herramientas, visión y avisos. No fuerza la carga del
modelo, así que responde aunque el modelo no esté.
