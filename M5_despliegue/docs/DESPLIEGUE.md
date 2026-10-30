# Despliegue · qué ejecutar y dónde

**Nada de esto se ha ejecutado todavía.** No hay Space publicado, ni URL, ni
telemetría recogida.

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

Medición de eficiencia — **esto es lo que falta para cerrar S15**:

```bash
python M5_despliegue/evaluacion/medir_eficiencia.py --conjunto data/eval_set_m2.jsonl
```

Corre fp16 e int4 en la misma sesión y escribe
`M5_despliegue/resultados/eficiencia_fp16_vs_int4.json`. Con `--solo-plan`
imprime el protocolo sin cargar nada.

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
