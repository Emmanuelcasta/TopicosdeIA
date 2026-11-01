# Guion para verificar el tutor en Kaggle

Pensado para pegárselo entero a un asistente que **sí puede manejar el navegador**
y operar la sesión de Kaggle. Está escrito para que no haga falta más contexto
que esto.

---

## Paso 0 · Antes de empezar (lo hace Emmanuel, no el asistente)

Los arreglos están en el árbol local **sin commit**. Kaggle clona desde GitHub,
así que sin este paso el asistente se encontrará el código viejo y todo lo de
abajo fallará igual que antes.

```bash
git add -A && git commit -m "fix(m5): aisla la carga por etapas y tolera el adapter_config de PEFT 0.19" && git push
```

Archivos que tienen que llegar: `M5_despliegue/app/pipeline.py`,
`app/compat_torch.py`, `app/app.py`, `app/salud.py`,
`evaluacion/medir_latencia.py`, `config/requirements.txt`.

En Kaggle, antes de nada: `!cd <repo> && git pull`.

---

## Contexto para el asistente

Es un tutor de matemáticas (Qwen2.5-1.5B + LoRA) servido con Gradio. Corre en
Kaggle con **Python 3.13 y una Tesla T4**. El adaptador LoRA viene como dataset
adjunto, bajo una carpeta `qwen-lora`.

Ya se arreglaron tres cosas, y conviene saberlo para interpretar lo que salga:

1. **Gradio devolvía *Internal Server Error*** en el enlace público. Causa:
   pydantic ≥ 2.11 describe un campo `dict` con `additionalProperties: true` y
   `gradio_client 1.3.0` revienta al recorrerlo. Arreglo: `pydantic<2.11` y un
   guardia en `app/compat_gradio.py`.
2. **La primera pregunta fallaba** con
   `AttributeError: module 'torch.distributed' has no attribute 'tensor'`.
   Causa: alguien accede a ese subpaquete sin importarlo, e `import torch` no lo
   carga. Arreglo: `app/compat_torch.py` lo importa al arrancar. **No hay que
   tocar el PyTorch ni el CUDA de Kaggle.**
3. **El adaptador no cargaba** con PEFT antiguas: su `adapter_config.json` lo
   guardó PEFT 0.19.1 y trae campos que las versiones publicadas no conocen.
   Arreglo: `pipeline._cargar_adaptador` omite los que no existan, que valen su
   valor por defecto.

Lo verificado en local (CPU, sin GPU): la carga completa en 7 etapas y la
pregunta «¿Cuánto es 3/4 + 1/8?» respondiendo `7/8` correctamente.
**En la T4 no se ha probado nada.** Eso es lo que hay que hacer.

---

## Reglas que no se saltan

- **No reinstalar `torch`.** El de Kaggle sirve. Cambiarlo arrastra compilaciones
  largas y un CUDA que puede no cuadrar con la T4.
- **No inventar resultados.** Si algo falla, pegar el traceback **completo**, no
  un resumen. Si no se llegó a medir algo, decir que no se midió.
- **No evaluar fotos ni correr ablaciones** hasta que la pregunta escrita
  funcione.
- **No guardar una versión nueva del notebook** sin avisar antes a Emmanuel.
- **No hacer `git add`, commit, push ni cambiar de rama** desde Kaggle.
- Si una celda tarda más de ~10 min sin salida, pararla y reportarlo en vez de
  esperar indefinidamente.

---

## Paso 1 · Dependencias

Sesión limpia. Solo paquetes de interfaz, ligeros y puros Python:

```python
!pip install -q "gradio==4.44.1" "gradio-client==1.3.0" \
                "pydantic>=2.0,<2.11" "starlette<1.0" "huggingface_hub==0.26.5"
!pip uninstall -y -q torchao
import IPython; IPython.Application.instance().kernel.do_shutdown(True)
```

El reinicio **no es opcional**: pydantic y starlette ya están importados por el
stack de Kaggle y sin reiniciar seguirían en memoria las versiones viejas.

**Reportar:** la salida de pip, y si hubo conflictos de dependencias.

---

## Paso 2 · Localizar el repo y el adaptador

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

## Paso 3 · Diagnóstico, antes de lanzar nada

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

**Qué mirar, en este orden:**

| Línea | Qué significa |
|---|---|
| `atributo antes : False` | el fallo estaba ahí |
| `atributo después : True` | resuelto con el import |
| `Accesos por atributo encontrados…` | si nombra un archivo, **ese es el paquete culpable** — reportarlo |
| `get_api_info(): OK` | la página se puede servir |
| `El fallo se estaba disparando: False` | el pin de pydantic funcionó |

**Si `get_api_info()` dice FALLA, parar aquí.** Lanzar daría error 500. Reportar
la salida entera.

**Reportar:** la salida completa de las dos llamadas, tal cual.

---

## Paso 4 · Lanzar

```python
!cd RUTA_REPO && \
  USE_TF=0 \
  TUTOR_CONFIG=revision_foto \
  TUTOR_GRADIO_SHARE=1 \
  TUTOR_ADAPTADOR_M1=RUTA_ADAPTADOR \
  python M5_despliegue/app/app.py
```

Esperar a la URL `*.gradio.live` y abrirla en el navegador.

**Reportar:** las líneas que imprime al arrancar (`Entorno:`, `torch:`,
`torch.distributed.tensor:`) y la URL.

---

## Paso 5 · Comprobar la configuración, antes de preguntar

En la interfaz:

1. **Cabecera.** No debe decir `MODO DEGRADADO`. Si lo dice, nombra qué
   componente falta: reportarlo y parar.
2. **Pestaña Estado.** Comprobar:
   - `adaptador_cargado: m1`
   - `retrieval_activo` contiene `hibrido`
   - `enrutador_activo: true`
   - `componentes_faltantes: ninguno`
   - `carga_por_etapas`: las 7 en `ok`

**Reportar:** captura o texto de la pestaña Estado completa.

> Nota: la configuración `revision_foto` sale marcada **EXPERIMENTAL, sin
> medir**. Es correcto y no es un error: esa combinación nunca se ha evaluado.

---

## Paso 6 · La pregunta que fallaba

En la pestaña «Pregunta escrita»: **`¿Cuánto es 3/4 + 1/8?`**

- **Si responde:** la respuesta correcta es `7/8`. Reportar el texto íntegro y
  lo que diga «Cómo se obtuvo».
- **Si falla:** en la salida de la celda aparecerá
  `=== FALLO EN LA ETAPA: <nombre> ===` con el traceback completo. **Pegarlo
  entero**: esa etapa es el diagnóstico, y sin ella no se puede avanzar.

Después, dos más para ver las tres vías del enrutador:

- `¿Cuál es la nota mínima aprobatoria del colegio?` → debería buscar en los
  documentos y citar la fuente.
- `Explica por qué 9 es un número primo.` → debería **negar la premisa**, no
  explicarlo. 9 = 3 × 3.

**Reportar:** las tres respuestas completas.

---

## Paso 7 · Latencias · SOLO si el paso 6 responde bien

Parar la app (interrumpir la celda) y correr:

```python
!cd RUTA_REPO && USE_TF=0 TUTOR_CONFIG=revision_foto \
  TUTOR_ADAPTADOR_M1=RUTA_ADAPTADOR \
  python M5_despliegue/evaluacion/medir_latencia.py --repeticiones 8
```

Separa **arranque en frío** (una muestra; incluye cargar el modelo) de las
**consultas calientes** (p50 y p95, con el n al lado). Las peticiones fallidas se
cuentan aparte y **no** entran en la estadística: durante la depuración se vieron
fallos a los 5,71–5,83 s, y eso es tiempo hasta la excepción, no latencia.

**Reportar:** la tabla final completa, incluido el n. No redondear ni resumir.

---

## Qué devolver al terminar

1. En qué paso se llegó y en cuál se paró.
2. **Tracebacks completos** de lo que fallara, con la etapa que los imprime.
3. Si `compat_torch` nombró algún paquete en «Accesos por atributo», cuál.
4. Las respuestas literales a las tres preguntas.
5. Las latencias con su n, si se llegó al paso 7.
6. Lo que no se pudo comprobar, dicho como tal.

Si algo no funcionó, **no declarar el tutor listo**. El objetivo de esta sesión es
saber qué pasa, no cerrar el módulo.
