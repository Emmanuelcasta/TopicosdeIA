# Probar el tutor en Kaggle T4

Prueba manual de la app (pregunta escrita, resolver una foto, revisar un
procedimiento). **No entrena nada ni corre el scorecard.** Tarda unos 5 minutos
la primera vez, casi todo en descargar los modelos.

## 1 · Notebook y acelerador

1. Abre el notebook **Tutor Matematicas**
   (`kaggle.com/code/emmanuelcastaedacano/tutor-matematicas`) y pulsa *Edit*.
   También sirve un notebook nuevo vacío.
2. *Settings → Accelerator →* **GPU T4 x2** (la app usa una sola T4).
3. *Settings → Internet →* **On**. Sin internet no se puede clonar el
   repositorio, descargar los modelos de Hugging Face ni crear el enlace de
   Gradio. Requiere la cuenta verificada con teléfono.

## 2 · Datos de entrada

*Add Input* y añade los dos datasets:

| Dataset | Para qué | Ruta que usan las celdas |
|---|---|---|
| `adaptadorqwen` | el adaptador LoRA del tutor (obligatorio) | `/kaggle/input/datasets/emmanuelcastaedacano/adaptadorqwen/qwen-lora` |
| `datasetevalm5v4` | las fotos del eval set (opcional para esta prueba) | `/kaggle/input/datasets/emmanuelcastaedacano/datasetevalm5v4/evalset-m5/Imagenes_manuscritas2` |

**Si tus datasets tienen otro nombre o usuario**, cambia esas rutas en la celda
A. Para encontrarlas:

```
!find /kaggle/input -name adapter_config.json
```

La carpeta que contiene `adapter_config.json` es la del adaptador. En algunas
cuentas Kaggle monta los datos como `/kaggle/input/<dataset>/…`, sin
`datasets/<usuario>/`.

## 3 · Celdas, en este orden

En el notebook existente **no uses *Run All***: tiene celdas de sondas, de
evaluación y de trazas que no hacen falta. Crea estas al final y ejecuta solo
estas, una a una (Ctrl/Cmd + Enter).

**Celda A · código, adaptador y dependencias** (una vez por sesión)

```
!cd /kaggle/working && rm -rf ProyectoIA && git clone -q --depth 1 --branch integracion-m5 https://github.com/Emmanuelcasta/TopicosdeIA.git ProyectoIA && cd ProyectoIA && git log --oneline -1
!mkdir -p /kaggle/working/ProyectoIA/adaptadores && cp -r /kaggle/input/datasets/emmanuelcastaedacano/adaptadorqwen/qwen-lora /kaggle/working/ProyectoIA/adaptadores/ && ls /kaggle/working/ProyectoIA/adaptadores/qwen-lora
!pip uninstall -y -q torchao ; pip install -q rank-bm25==0.2.2
```

Debe imprimir el último commit de `integracion-m5` y los archivos
`adapter_config.json` y `adapter_model.safetensors`. `torchao` se desinstala
porque rompe la carga del adaptador en esta imagen de Kaggle; no reinstales
Torch.

**Celda B · pruebas rápidas** (opcional, ~1,5 min)

```
!cd /kaggle/working/ProyectoIA && python -m pytest -q M5_despliegue/tests 2>&1 | tail -2
```

**Celda C · lanzar la app** (se queda corriendo: es el servidor)

```
%cd /kaggle/working/ProyectoIA
!env USE_TF=0 TUTOR_CONFIG=revision_foto TUTOR_GRADIO_SHARE=1 TUTOR_PRECARGAR=1 TUTOR_ADAPTADOR_M1=/kaggle/working/ProyectoIA/adaptadores/qwen-lora python -u M5_despliegue/app/app.py 2>&1 | grep -v -E "Loading|Fetching|Download|Reconstruct|it/s|torch|Warning"
```

**Omitir** para esta prueba: todo lo de `evaluar_vision_m5.py --correr`
(scorecard), las sondas (`sonda*.py`) y cualquier celda de entrenamiento.

## 4 · Abrir la app

La celda C imprime, tras 1–4 minutos:

```
precarga del modelo de texto: … listo=True
precarga del VLM: {'ok': True, …, 'dispositivo': 'cuda'}
* Running on public URL: https://xxxxxxxxxxxx.gradio.live
```

Abre esa URL (es temporal: dura lo que la sesión). Añadir `?__theme=light` al
final la muestra en modo claro, que se lee mejor proyectada.

## 5 · Las tres pruebas

Las fotos están en el repositorio, en `M5_despliegue/presentacion/assets/`;
descárgalas a tu computador para subirlas.

| Prueba | Qué hacer | Qué esperar |
|---|---|---|
| Pregunta escrita | Pestaña «✍️ Pregunta escrita», pulsa el ejemplo de las galletas y **Responder**. | Un paso `15 × 18 = 270 ✅`, «Resultado: 270 galletas», «Cuentas verificadas». |
| Resolver una foto | Pestaña «📷 Foto de un ejercicio», sube `demo-m5v-07-solo-enunciado.png`, elige **Resolver ejercicio**, **Leer la hoja**, revisa el enunciado, **Confirmar lectura**, **Resolver el ejercicio**. | Lectura en ~8 s; «Multiplicamos 18 por el número de paquetes: 18 × 15 = 270 ✅»; «Resultado: 270 galletas». No aparece ninguna caja de pasos ni «Revisa mi procedimiento». |
| Revisar un procedimiento | Sube `demo-m5v-02-procedimiento.jpg`, elige **Revisar procedimiento**, **Leer la hoja**, revisa enunciado y 3 pasos, **Confirmar lectura**, **Revisar mi procedimiento**. | Lectura en ~15 s; «🔴 Primer error: paso 1», con el motivo (1240 + 365 donde se pide cuántas quedan), la hoja con el paso 2 marcado ✅ y la referencia 1240 − 365 = 875. |

Sin confirmar la lectura no aparece el resultado: es a propósito. Si una cifra
está mal leída, corrígela en la caja antes de confirmar.

**Fallo conocido:** la hoja m5v-09 se lee como «14» donde dice «19»; la
revisión sale «sin error». No es un fallo de la app sino de la lectura, y las
fotos del eval set son de desarrollo: no demuestran generalización.

## 6 · Si algo se cae

- **La página dice 503 o deja de responder, pero el notebook sigue activo:**
  detén la celda C (botón ■ o *Cancel Run*) y vuelve a ejecutarla. La URL
  cambia; usa la nueva. La precarga ya no descarga nada (~30 s).
- **«Draft Session off» o «Session is starting»:** la sesión se reinició y
  `/kaggle/working` está vacío. Ejecuta de nuevo A y después C.
- **Kaggle pregunta «Are you still there?»:** pulsa *Continue editing*; tras
  ~40 min sin tocar el notebook apaga la sesión.
- **Actualizar solo el código** con la sesión viva (sin repetir A):

  ```
  !cd /kaggle/working/ProyectoIA && git fetch -q --depth 1 origin integracion-m5 && git reset --hard -q FETCH_HEAD && git log --oneline -1
  ```

  y luego vuelve a ejecutar C.
- No uses *Factory reset* salvo que quieras empezar de cero: borra la sesión.
