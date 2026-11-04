# Despliegue · qué ejecutar y dónde

**Estado al 2 de noviembre de 2026.** La interfaz arranca en Kaggle con una T4,
sirve un enlace temporal de Gradio y **responde correctamente** a la pregunta
escrita: el 1 de noviembre contestó «¿Cuánto es 3/4 + 1/8?» con `3/4 = 6/8`,
`6/8 + 1/8` y `7/8` en 9,3 s.

**Y aun así la corrida no validó la arquitectura.** La pestaña Estado marcaba
`degradado: True`, con `retrieval híbrido + reranker` y
`enrutador consulta/problema` entre los faltantes, y el resultado como
`NO MEDIDO`. Lo que contestó fue el modelo base con LoRA y búsqueda léxica, no la
S08-D que se pidió. **Una respuesta correcta no es una arquitectura completa**, y
el `7/8` no se puede presentar como validación de nada más que de que el modelo y
el adaptador cargan.

No hay Space permanente ni una corrida sin degradar. Los enlaces temporales de
Kaggle caducan y no son una URL de producción.

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

### Celdas de Kaggle · dependencias, en una sola resolución de pip

**1 · Dependencias.** Un solo `pip install`, con las cuatro piezas acopladas
dentro. Pydantic ya está importado por el stack preinstalado de Kaggle, así que
**hay que reiniciar el kernel** después:

```python
%cd /kaggle/working/ProyectoIA
%pip install -q -r M5_despliegue/config/requirements-kaggle.txt
!pip uninstall -y -q torchao
import IPython; IPython.Application.instance().kernel.do_shutdown(True)
```

**Por qué un solo install y no dos.** Antes esta sección proponía una celda con
los paquetes de interfaz —`gradio`, `gradio-client`, `pydantic`,
`huggingface_hub`, `starlette`— y otra, *opcional*, con el resto del proyecto. Si
se corría solo la primera, pasaba lo del 1 de noviembre: `huggingface_hub` bajaba
a 0.26.5 (gradio 4.44 importa `HfFolder`, retirado en el hub 1.x), eso arrastraba
`transformers` y `tokenizers`, y `sentence-transformers` se quedaba en el
preinstalado corriendo sobre tres piezas más viejas que las que vinieron con él.
El retrieval híbrido dejó de montarse y el tutor respondió en modo degradado.

`requirements-kaggle.txt` existe para que esa combinación no se pueda armar a
medias: las cuatro van en el mismo archivo y pip las resuelve a la vez. No toca
`torch` (el de Kaggle ya resuelve CUDA para la T4), ni `numpy` (bajarlo bajo un
torch y un scipy compilados contra numpy 2 es un error de ABI), ni
`scikit-learn`. El archivo explica cada exclusión.

Los pines de interfaz siguen dentro y por los mismos motivos: `pydantic<2.11`
arregla el *Internal Server Error* del enlace público, `starlette<1.0` conserva
la firma posicional de `TemplateResponse` que usa Gradio 4.44.1, y
`gradio-client==1.3.0` va explícito porque el pin transitivo de gradio no basta
donde hay uno preinstalado.

El diagnóstico debe ejecutarse sin importar Gradio directamente: así
`compat_gradio` puede informar de un fallo de importación en vez de detener la
celda antes de mostrar las versiones.

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

**4 · Lanzar.** Antes de esto, correr `evaluacion/diagnostico_retrieval.py`
(celda 2 de «Celdas de Kaggle · diagnóstico y arranque», más abajo): con el
retrieval caído la app arranca igual y responde igual, y es lo que dejó la
corrida del 1 de noviembre sin validar nada.

El adaptador está en un dataset adjunto, así que hay que apuntarlo o la app
arranca en modo degradado:

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

**1 · Dependencias** (sesión limpia). Un solo `pip install`, el de
`requirements-kaggle.txt`. No reinstala torch: el de Kaggle sirve, y cambiarlo
arrastraría compilaciones largas y un CUDA que puede no cuadrar con la T4.

```python
%cd /kaggle/working/ProyectoIA
%pip install -q -r M5_despliegue/config/requirements-kaggle.txt
!pip uninstall -y -q torchao
import IPython; IPython.Application.instance().kernel.do_shutdown(True)
```

`torchao` se quita porque rompe la carga del adaptador LoRA, y se quita
**después** del install: reaparece cada vez que pip vuelve a resolver. El
reinicio no es opcional: pydantic y starlette ya están importados por el stack
de Kaggle.

**2 · Los cuatro niveles del retrieval.** Esta celda decide si lanzar la app
tiene sentido: con la cadena de S08 caída, la app arranca igual y responde
igual, y eso es lo que confundió la corrida del 1 de noviembre. No necesita GPU
ni carga Qwen; sí descarga los dos modelos de S08 (~500 MB) si no están en caché.

```python
!cd /kaggle/working/ProyectoIA && USE_TF=0 TUTOR_CONFIG=s08d \
  python M5_despliegue/evaluacion/diagnostico_retrieval.py
```

Tienen que salir los cuatro en `OK`. Si alguno falla, el guion imprime su
traceback completo y la causa probable —dependencia, red, caché, memoria o
versión—, que son arreglos opuestos: con un fallo de red, reinstalar paquetes no
sirve; con uno de versión, tener internet tampoco. **Con el nivel 3 o 4 en rojo,
no se lanza la app.**

**3 · Diagnóstico de torch y Gradio** (antes de lanzar nada):

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

**4 · Lanzar.** La ruta del dataset se comprueba, no se supone: Kaggle la monta
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

**5 · Qué comprobar, en este orden** —y no se pasa al siguiente hasta que el
anterior funcione:

1. La cabecera **no** dice `MODO DEGRADADO`.
2. Pestaña **Estado**: `degradado: False`, `adaptador_cargado: m1`,
   `retrieval_activo` con `hibrido`, `enrutador_activo: true`,
   `componentes_faltantes: ninguno`, y `retrieval_por_etapas` con todas en `ok`
   —incluida la consulta de prueba—. Si `degradado` es `True`, mirar
   `retrieval_causa_probable` y `retrieval_traceback`: la causa está ahí.
3. La pregunta escrita «¿Cuánto es 3/4 + 1/8?» responde. Si falla, la salida de
   la celda trae `=== FALLO EN LA ETAPA: … ===` con el traceback entero.
4. Solo entonces, las latencias (abajo). **Nada de fotos ni ablaciones** antes
   de que la pregunta escrita funcione.

### El retrieval caído que una respuesta correcta tapaba

**El síntoma.** El tutor responde bien y la pestaña Estado dice
`degradado: True`, con `retrieval híbrido + reranker` entre los faltantes. Pasa
porque la app está diseñada para replegarse en vez de caerse: sin la cadena de
S08 usa búsqueda léxica y lo anuncia. Lo que no hacía era decir **por qué**.

**Por qué no se veía.** `buscador_hibrido` tenía un `except ImportError` y un
`except Exception` que devolvían la misma `None`, así que «falta rank_bm25,
sentence-transformers o los modelos en caché» era el único mensaje posible —y en
Kaggle era falso en sus dos primeras opciones, porque los tres paquetes estaban
instalados, y mudo en la tercera.

**La causa, CONFIRMADA en la T4 el 2 de noviembre de 2026.** El error exacto es

```
AttributeError: module 'numpy.dtypes' has no attribute 'StringDType'
```

y la cadena, de abajo arriba:

```
buscador_hibrido()
  from sentence_transformers import CrossEncoder, SentenceTransformer
    sentence_transformers/__init__.py
      transformers.integrations.integration_utils
        transformers.modeling_tf_utils          (TFPreTrainedModel)
          tensorflow.lite.python.authoring
            jax/_src/dtypes.py:  np.dtypes.StringDType()
```

`numpy.dtypes.StringDType` existe desde numpy 2.0. La imagen de Kaggle trae
numpy 2.1.3, y su tensorflow 2.20 y su jax 0.11.1 están construidos contra ella.
`requirements.txt` fija `numpy==1.26.4` —correcto para el Space, que va con
torch 2.4.1— y aplicarlo en Kaggle rompe jax, rompe tensorflow, rompe el puente
TF de transformers y con él el import de `sentence_transformers`. El modelo de
lenguaje no pasa por esa cadena, y por eso el tutor contestaba `7/8` con el
retrieval muerto.

**Era nuestra propia instalación la que rompía el entorno.** La hipótesis
anterior —`sentence-transformers` quedándose atrás al bajar sus tres
compañeras— queda **descartada**: medido en proceso limpio, la imagen virgen
importa `sentence_transformers` sin un fallo. Lo único que le falta a esa imagen
es `rank-bm25`:

```
NIVEL 1 · importar sentence_transformers ........ OK
           rank_bm25 ............................ ModuleNotFoundError
```

y con `pip install rank-bm25` y nada más:

```
NIVEL 1 OK · rank_bm25 OK
NIVEL 2 · paraphrase-multilingual-MiniLM-L12-v2 .. OK, dim (1, 384)
           mmarco-mMiniLMv2-L12-H384-v1 ......... OK, score 2.388
```

La imagen virgen medida (entorno fijado a 2026-10-22, T4 x2):

| | | | |
|---|---|---|---|
| Python 3.13.15 | numpy 2.1.3 | scipy 1.16.3 | torch 2.11.0+cu128 |
| transformers 5.16.1 | tokenizers 0.23.1 | huggingface_hub 1.29.0 | sentence-transformers 5.7.0 |
| peft 0.20.0 | accelerate 1.14.0 | scikit-learn 1.6.1 | sympy 1.14.0 |
| gradio 6.26.0 | gradio_client 2.6.1 | pydantic 2.13.5 | tensorflow 2.20.0 |
| jax 0.11.1 | torchao 0.10.0 | **rank_bm25 ausente** | |

**Lo que se cambió.**

1. `buscador_hibrido(chunks, diagnostico)` registra **cada etapa** —importar,
   índice BM25, embeddings, cross-encoder, codificar el corpus y una consulta de
   prueba— con su error, su causa probable y su traceback, y lo imprime en el log
   del proceso. `salud.py` lo publica como `retrieval_por_etapas`,
   `retrieval_causa_probable` y `retrieval_traceback`. Los tracebacks pasan por
   un redactor: los errores del hub citan URLs y cabeceras.
2. **Codificar el corpus ocurre ahora dentro del manejo de errores.** Estaba
   fuera, y un fallo ahí no degradaba el retrieval: tumbaba la carga entera del
   tutor.
3. **Una consulta de prueba antes de declararse montado.** Que los constructores
   no lancen no significa que la cadena responda.
4. `config/requirements-kaggle.txt`: un perfil que fija **juntas** las cuatro
   piezas acopladas, en una sola resolución de pip. No toca `torch`, `numpy` ni
   `scikit-learn`, y explica por qué cada una.
5. `evaluacion/diagnostico_retrieval.py`: cuatro niveles separados —paquete
   instalado, modelo cargable, componente montado, componente efectivo— porque
   fallan por motivos distintos y se arreglan distinto. El nivel 1 era lo único
   que comprobaba la validación anterior, y de ahí salió el «retrieval listo»
   que era falso.
6. `validacion_m5.estado_requisitos()` ya no informa el retrieval como un
   booleano de `find_spec`. Sin una corrida del diagnóstico dice `NO COMPROBADO`.

```bash
# cuatro niveles, sin GPU y sin cargar Qwen
python M5_despliegue/evaluacion/diagnostico_retrieval.py
python M5_despliegue/evaluacion/diagnostico_retrieval.py --solo-entorno
```

**Pendiente.** Ninguno de los cuatro niveles se ha corrido todavía en Kaggle con
el perfil nuevo. Hasta entonces, el perfil es una hipótesis con fundamento, no un
entorno validado.

### La corrida del 2 de noviembre de 2026 · S08-D sin degradar

Primera ejecución en la que la configuración solicitada y la efectiva coinciden.
Kaggle T4 x2, entorno fijado a 2026-10-22, commit `9d1ea74`, perfil
`requirements-kaggle.txt` (`rank-bm25` y nada más) más `pip uninstall -y torchao`.

```
configuracion_solicitada: S08-D · LoRA M1 + RAG híbrido + enrutador
configuracion_efectiva:   S08-D · LoRA M1 + RAG híbrido + enrutador
degradado: False          componentes_faltantes: ninguno
montaje: verificado       listo: True
modelo_cargado: True      adaptador_cargado: m1
retrieval_activo: hibrido + rerank (BM25+denso, RRF, cross-encoder) = S08-C
enrutador_activo: True    corpus: True    dispositivo: cuda
```

Carga por etapas: tokenizer 0,66 s · modelo base 1,92 s · mover a cuda 1,12 s ·
localizar adaptador 0,00 s · adaptador LoRA 0,28 s · armar agente 10,47 s.

**Las dos vías del enrutador**, que es lo que distingue S08-D de S08-C:

| pregunta | contextos recuperados | respuesta | latencia |
|---|---|---|---|
| ¿Cuál es la nota mínima aprobatoria del curso? | **3** | «Nota mínima aprobatoria es 3.0» | 1,70 s |
| ¿Cuánto es 2/3 + 1/6? | **0** | 4/6 + 1/6 = 5/6 | 3,50 s |
| ¿Cuánto es 3/4 + 1/8? | **0** | 3/4 = 6/8; 6/8 + 1/8 = **7/8** | — |

La factual entra por RAG y la de cálculo no. Es la decisión del enrutador, vista
desde fuera y sin inventar una puntuación de confianza.

**Latencia**, separando el arranque en frío de la respuesta habitual:

| | |
|---|---|
| arranque en frío (carga del modelo + primera respuesta) | 25,72 s · n = 1 |
| en caliente, n = 8 | p50 **4,23 s** · p95 **5,36 s** · min 3,31 s · máx 5,36 s |

Con n = 8, el p95 coincide con el máximo observado; se reporta con su n y no se
presenta como una cola estimada. El 9,3 s de la corrida anterior era de un
sistema degradado y **no** es la latencia de S08-D.

**Lo que esto no dice.** Es una comprobación de montaje y de dos vías del
enrutador, no una medición de exactitud. La cifra 18/21 sigue siendo la
histórica de S08 sobre el eval set fijo de M2. Las fotos de M5, las ablaciones y
la evaluación completa siguen sin correr.

### `torchao` · se desinstala, y reaparece

Viene en la imagen de Kaggle y rompe la carga del adaptador LoRA. Vuelve a
aparecer al reiniciar el runtime o al volver a resolver un `pip install`, así que
se quita **después** del último install de la sesión y se comprueba:

```python
import importlib.util as u
print("torchao:", u.find_spec("torchao"))     # tiene que decir None
```

`diagnostico_retrieval.py` lo avisa en su cabecera, con el comando para quitarlo.

### Dos perfiles de dependencias, y por qué no son uno

| | `requirements.txt` | `requirements-kaggle.txt` |
|---|---|---|
| Para | Space y local, instalación limpia | la sesión de Kaggle |
| Gradio | 4.44.1 + `gradio-client 1.3.0` | 4.44.1; solo verificado con 6.26.0 |
| `torch` | `==2.4.1` | el de la imagen, **no se toca** |
| `numpy` | `==1.26.4` | el de la imagen (bajarlo rompe el ABI) |
| Acopladas | las cuatro fijadas | las cuatro fijadas, en un solo install |

El guardia de `app/compat_gradio.py` no depende de una versión concreta:
comprueba si el fallo se dispara y solo parchea si hace falta. Por eso gradio
6.26.0 sirvió la página el 1 de noviembre aunque el pin diga 4.44.1. Hay un
segundo camino coherente —no bajar nada y quedarse con el stack preinstalado de
Kaggle, desinstalando solo `torchao`—, documentado al final de
`requirements-kaggle.txt`. No es el perfil por defecto porque deja el entorno a
merced de una imagen que cambia sin avisar y no se puede fijar en un archivo.

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

## 6 · Latencias de la vía visual · tres cosas distintas

Se miden por separado desde el 2 de noviembre de 2026, porque mezcladas no
significan nada. `Tutor.precargar_vision()` carga el VLM y devuelve sus
segundos; a partir de ahí la primera lectura es una lectura en frío de verdad.

| etapa | qué incluye |
|---|---|
| descarga + carga del VLM | 4,42 GB de `Qwen/Qwen2-VL-2B-Instruct`, una vez por sesión |
| lectura en frío | la primera inferencia con el modelo ya en memoria |
| lecturas en caliente | las siguientes |

**Lo medido hasta ahora, en T4, y lo que no se puede concluir.** La prueba de
humo del 2 de noviembre dio 93,9 s, 12,11 s y 8,64 s. Los 93,9 s llevaban dentro
la descarga, así que **no** son una lectura en frío: son descarga, carga e
inferencia sumadas, y de ahí no sale una latencia de lectura. Quedan **dos**
observaciones en caliente, 12,11 s y 8,64 s. Con dos no hay p50 ni p95; el
evaluador solo los calcula con 20 o más y por debajo de eso publica los tiempos
uno a uno. Esa corrida además fue sobre el paquete de humo, no sobre el eval set.

**Optimizaciones que sí se pueden hacer.** Precargar el VLM antes de la demo, de
modo que los ~90 s se paguen sin público delante. Y el tope de 1024 px del lado
mayor, que ya está, mantiene el coste por caso independiente de cómo salió la
foto.

**Lo que no se toca para ir más rápido.** Las dos lecturas independientes y
`lectura_no_fiable` se quedan. Una sola lectura ahorraría la mitad del tiempo y
quitaría la señal que detecta las transcripciones inestables; el caso `vis-10` ya
enseñó lo que cuesta diagnosticar sobre una lectura inventada.

## 7 · Lo que falta validar antes de llamar a esto producción

La prueba de humo demuestra que una foto se carga y se lee. No demuestra nada
más, y en particular no es una medición de exactitud ni una prueba de la válvula
de ilegibles. Pendiente:

- **URL pública.** No existe ninguna `*.hf.space` en el repositorio. El enlace
  `gradio.live` de Kaggle vive mientras viva el proceso.
- **`/health`.** Hoy el estado se ve en la pestaña **Estado**; no hay un
  endpoint que un comprobador externo pueda consultar.
- **Logs.** `telemetria.py` existe y está probado, pero no hay ninguna corrida
  registrada en W&B.
- **Errores y negativas.** Falta comprobar en el Space desplegado qué ve el
  usuario cuando el modelo no carga, cuando la foto es ilegible y cuando la
  configuración está degradada.
- **Eval set visual.** Están las **12 de 12** fotos, una por caso, en
  `m4/eval_set/imagenes_m5/m5v-NN.jpg`, copiadas sin recodificar desde los
  originales `.jpeg` de `M5_despliegue/evalset-m5/Imagenes_manuscritas2/`. El
  alcance bajó de 23 variantes a 12 hojas el 2026-11-03; la definición anterior
  sigue en `evaluar_vision_m5.variantes_historico_23`.
- **Cuatro hojas en conflicto con el JSONL.** Se leen, pero no se puntúan: su
  respuesta esperada describe una hoja que no es la que hay. Resolverlo es una
  decisión de diseño —repetir la foto o corregir el JSONL— y no se tomó.

  | Caso | Qué pasa |
  |---|---|
  | `m5v-01` | la hoja pone `642 km`; el JSONL espera `942` |
  | `m5v-05` | la hoja pone `2 × 15000 = 3000`; el JSONL dice `2 × 1500`, y su propio enunciado dice «lapiceros de 1500». Con el cero de más el paso 2 también queda mal, y el caso es `compensado`: prohíbe nombrar el paso 2 |
  | `m5v-09` | la hoja pone `Respuesta final: 19`; su paso 2 dice `14` y el JSONL espera `14`. El caso está declarado `sin_error` |
  | `m5v-10` | la foto es **legible**. El caso existe para medir la salvaguarda de ilegibles: con una toma nítida no se puede afirmar nada sobre la válvula |

  Quedan **8 casos puntuables**. Con ese tamaño un acierto mueve la tasa 12,5
  puntos: sirve para ver si el recorrido funciona, no para comparar
  configuraciones.

## 8 · Regresiones de la fase 2, corregidas el 2026-11-03

Cuatro cosas estaban mal a la vez y las cuatro se veían en la misma pantalla.

**La cuenta la hacía el modelo de memoria.** `3/4 + 1/8` devolvía `0.75` y la
traza registraba cero llamadas a herramienta. `s08d_verificador` monta las 18,
pero el adaptador `m1` se entrenó para RAG, no para emitir `<tool_call>`, así
que nunca las pedía. Ahora, cuando la pregunta **es** una cuenta,
`Tutor._aritmetica_directa` ejecuta la herramienta registrada antes de cargar
ningún modelo y responde con su resultado exacto. La observación que se guarda
es la de `RegistroHerramientas.ejecutar`: si la traza dice que no se ejecutó
nada, es que no se ejecutó nada. `s08d` no entra por ahí —se midió sin
herramientas— y sigue igual.

Comprobado: `3/4 + 1/8` → `7/8 (≈ 0.875)` vía `operar_fracciones`;
`347 × 28` → `9716` vía `evaluar_expresion`.

**Una consulta mostraba la respuesta de la anterior.** No era estado del
servidor: el evento moría a medio camino y la caja se quedaba con lo de antes.
El botón ahora encadena dos eventos; el primero vacía las cajas sin ir al
servidor. Si el segundo muere, lo que queda es vacío.

**La app y el evaluador veían píxeles distintos.** La interfaz pasaba a
`leer_imagen` el PIL crudo de Gradio —4160 px en una foto de móvil— y el
evaluador reducía a 1024. La misma hoja se rechazaba en una y se leía en el
otro. La normalización (EXIF + lado mayor) bajó a `pipeline._leer`, que es por
donde pasan los dos.

**`Unexpected token '<'`.** El cliente de Gradio recibía HTML donde esperaba
JSON. Medido el 2026-11-03 sobre `d9420a5a65465847df.gradio.live`, leyendo la
red del navegador durante el recorrido completo:

| Petición | Duró | Estado |
|---|---|---|
| `POST /gradio_api/upload` | — | 200 |
| `GET /gradio_api/queue/data` · lectura de la foto | 16,85 s | **503** |
| `GET /gradio_api/queue/data` · confirmar transcripción | <1 s | 200 |
| `GET /gradio_api/queue/data` · revisar el procedimiento | 1,06 s | **503** |

El `503` lo devuelve el **túnel de `*.gradio.live`**, no la aplicación: es el
servicio de *share* de Gradio cortando el flujo SSE de `queue/data`. Un 503 de
ese proxy llega como página HTML, el cliente le hace `JSON.parse` y eso es
literalmente el `Unexpected token '<'`. Con `queue/join` en 200 y el resultado
apareciendo igual en pantalla, el cliente se recupera casi siempre; cuando no
se recupera, la caja se queda con lo de la consulta anterior.

Lo que está en nuestra mano: `TUTOR_PRECARGAR=1` quita la espera de ~80 s de la
primera pregunta, y el botón vacía las cajas antes de responder, así que un
evento muerto ya no puede dejar una respuesta ajena en pantalla. Lo que **no**
está: el túnel. El arreglo definitivo es no depender de él — un Space de
Hugging Face o la URL local— y eso sigue pendiente.

**Dependencia externa en el arranque.** La consola registra además
`Unable to preload CSS for https://gradio.s3-us-west-2.amazonaws.com/assets/index-BkS15Xs3.css`.
Gradio 6.26.0 sirve su frontend desde ese bucket de S3. Si la máquina de la
demo no llega a `gradio.s3-us-west-2.amazonaws.com`, la interfaz carga sin
estilos. Conviene comprobarlo en la red donde se vaya a presentar.

**Además:** la respuesta ya no enseña la línea `Pensamiento:` del formato ReAct.
`nucleo.analizar_salida` solo la quitaba del campo `pensamiento`; cuando no hay
llamada a herramienta, se quedaba dentro de la respuesta. Se limpia en la capa
de presentación, así que el evaluador sigue viendo el texto completo.


## 9 · Dos modos de foto y lectura por líneas · corrida del 2026-11-04

Kaggle T4, commits `1bab37e` (scorecard y sondas) y `29a78ba` (interfaz).
Resultados en `M5_despliegue/resultados/vision_m5.{json,csv}` (scorecard) y
`pruebas_manuales.jsonl` (sondas exploratorias, **no** son scorecard).

**Qué cambió en la lectura.** El VLM ya no decide qué es enunciado y qué es
paso: copia líneas (`INSTRUCCION_COPIAR`) y `estructurar_lineas` las separa por
una regla visible —el enunciado acaba en la primera línea con «Paso N», «=» o
«Respuesta final»—. Las dos redacciones se eligieron con `sonda_prompts.py`
sobre 9 hojas que no son el scorecard. Las de M4 (`INSTRUCCION_VER*`) no se
tocaron: `nb_m4.py` las usa.

**Decisión que conviene revisar.** La doble lectura compara el esqueleto
matemático —cada número, cada operación, cuántos «Paso N», si hay respuesta
final—, no las palabras. m5v-03 se paraba por «¿Cuánto» / «Cuantos» con todos
los números iguales. Las palabras las corrige el estudiante antes de confirmar.
Si se quiere exigir también el texto, es un cambio de una línea en
`huella_lectura`, y se vuelven a parar hojas legibles.

**Scorecard de 12 fotos.** 12 leídas · 10 confirmables · 2 paradas por la doble
lectura (m5v-01: 942 / 9412; m5v-04: una copia omitió el enunciado) · 4
bloqueadas por conflicto con el JSONL · 7 puntuables · **1 acierto** (m5v-06).
Los seis fallos son del diagnóstico del modelo, no de la lectura. Muestra de 7:
no sirve para comparar configuraciones.

**Latencias.** Precarga del modelo de texto 22,5 s; del VLM 5,9 s con la caché
caliente (14,3 s en el scorecard; la primera descarga de una sesión nueva baja
3,09 + 4,42 GB). Lectura en frío 13,7 s; en caliente entre 4,0 y 17,2 s (11
observaciones, sin p50/p95). Resolver un enunciado: 3,3 a 25 s. Revisar: ~1 s
tras la lectura.

**Formatos.** `gr.Image(type="pil")` acepta lo que abre Pillow. Probados en la
interfaz: PNG (modo resolver) y JPG (modo revisar); en la sonda, JPG y PNG del
mismo recorte dan la misma lectura. HEIC de iPhone **no** se abre sin
`pillow-heif`, que no está instalado. El eval set sigue en `.jpg`.

**Errores conocidos, abiertos.**
- El diagnóstico acierta 1 de 7. El verificador sí señala el paso correcto
  cuando el error es de cálculo (m5v-03), y la pantalla muestra el desacuerdo.
- En modo resolver el modelo puede plantear mal: en el recorte de m5v-08 dividió
  donde había que restar. El verificador no lo detecta: comprueba cuentas, no
  planteamientos.
- El verificador lee mal igualdades encadenadas («a ÷ b = c × d = e» compara a ÷ b
  con c); da un falso «no cuadra». Pendiente.
- Una hoja sin ninguna cifra ya no se resuelve (m5v-11 inventó «68 ÷ 2 = 34»).

## 10 · Lectura estricta, verificador por igualdades y diagnóstico combinado · corrida del 2026-11-04 (tarde)

Kaggle T4, commit `b439be5`, 2026-11-04T15:07:39Z, una sola corrida completa
con el mismo commit que se reporta. Resultados en
`M5_despliegue/resultados/vision_m5.{json,csv}` (sustituyen a los de §9; no se
mezclan versiones). **Evaluación de desarrollo**: estas 12 hojas se usaron para
depurar lectura, verificador y regla de diagnóstico. No es evidencia de
generalización.

**Qué cambió.**
- *Doble lectura estricta.* Sustituye la decisión de §9: ahora compara, por
  separado en enunciado y pasos, palabras, números, operaciones, cantidad de
  pasos y respuesta final. Solo normaliza mayúsculas, espacios, puntuación y
  tildes. Si algo material difiere, para y dice qué parte no coincidió, sin
  mostrar la transcripción dudosa.
- *Verificador.* Cada igualdad adyacente de una cadena se evalúa por separado
  y con aritmética exacta (`7/8 ÷ 3/8 = 7/3` ya cuadra); informa qué igualdad
  falló.
- *Diagnóstico.* Manda el paso que señala el verificador, salvo que el tutor
  apunte a un paso anterior que el verificador no pudo comprobar (planteamiento).
  Si el tutor señala un paso que el verificador comprobó como correcto, la
  conclusión es «sin error». Cuando las dos señales difieren, la pantalla
  muestra las dos.

**Scorecard.** 12 leídas · 8 confirmables · 4 paradas por la doble lectura ·
1 excluida (m5v-10) · 7 puntuables · **2 aciertos** (m5v-06, m5v-07). El tutor
solo habría acertado 1: en m5v-07 (sin error) se inventaba un error en el paso 2
y la regla combinada lo evita. 1 fallo grave: m5v-08 (sin error), el tutor señala
el paso 1, que el verificador no puede comprobar. Fuente de los fallos: 4 de
diagnóstico (m5v-02, 08, 11, 12) y 1 de lectura (m5v-09, ambas lecturas dan 14
donde la hoja dice 19; en la app el estudiante lo corregiría antes de confirmar).

**Coste de la regla estricta.** Rechaza 4 de las 12 fotos legibles: m5v-01
(números), m5v-04 (palabras y números) y m5v-03 y m5v-05, estas dos **solo por
palabras**. Con la regla de §9 esas dos habrían pasado a confirmación.

**m5v-10.** Registrado como NO VÁLIDO: la foto es legible y el caso exige una
toma movida. Se lee pero no puntúa. La salvaguarda de ilegibles sigue sin
medición hasta que haya una foto borrosa de verdad.

**Latencias.** Carga del VLM 14,6 s. Lectura en frío 14,6 s; en caliente de
4,3 a 18,4 s (11 observaciones; sin p50/p95 por debajo de 20). Diagnóstico tras
la lectura ~1 s (17,5 s el primero, que carga el tutor).

**Prueba en la app** (misma versión, sesión nueva de Kaggle). Resolver: recorte
de m5v-07 con solo el enunciado, PNG, lectura 7,9 s, confirmación editable y
luego «18 × 15 = 270», con el verificador conforme. Revisar: m5v-07.jpg completa,
lectura 13,5 s con las dos lecturas coincidentes, confirmación y luego
«Diagnóstico: Sin error», mostrando que el tutor solo decía «paso 2». El
paso 4 no aparece hasta confirmar, y una foto nueva lo vuelve a ocultar.

**Abierto.**
- Foto movida para m5v-10.
- Decidir si el coste de la regla estricta (2 hojas paradas solo por palabras)
  es aceptable.
- Errores de planteamiento que el tutor no ve (m5v-02) y errores inventados en
  pasos no comprobables (m5v-08).
- m5v-11 responde «No veo un ejercicio», que no casa con las claves del JSONL.
- m5v-12: el dato que falta no se detecta.
- `scripts/tests/test_m4_vision.py::test_dos_lecturas_iguales_son_consistentes`
  falla, ya fallaba antes de estos cambios y contradice la regla de palabras.
