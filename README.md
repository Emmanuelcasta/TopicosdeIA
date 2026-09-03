# Tutor inteligente de matemáticas · Comparación de arquitecturas

Experimento comparativo entre cuatro arquitecturas de modelos de lenguaje
aplicadas a un tutor de matemáticas en español, mediante fine-tuning sobre un
corpus propio.

**Curso:** SI4006 · Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT
**Notebook base:** `S04_Lab_Fine_tuning Qwen.ipynb` (original, conservado sin cambios)

---

## Estructura

```
ProyectoIA/
├── S04_Lab_Fine_tuning_Qwen.ipynb          1 · Decoder-only        (generación)
├── S04_Lab_Fine_tuning_BERT.ipynb          2 · Encoder-only        (clasificación)
├── S04_Lab_Fine_tuning_Qwen_BERT.ipynb     3 · Pipeline modular    (2 etapas)
├── S04_Lab_Fine_tuning_FLAN_T5.ipynb       4 · Encoder-decoder     (integrado)
├── S04_Comparacion_Arquitecturas.ipynb     5 · Síntesis y recomendación
│
├── data/
│   └── math_tutor_dataset.jsonl            Corpus curado (165 ejemplos)
│
├── scripts/                                Generación del proyecto (no se ejecutan en Colab)
│   ├── registros.py                        Los 165 ejemplos, fuente de verdad
│   ├── dataset_fuente.py                   Valida y exporta el JSONL
│   ├── nb_comun.py                         Celdas compartidas entre notebooks
│   ├── nb_qwen.py · nb_bert.py · nb_pipeline.py · nb_flan_t5.py · nb_comparacion.py
│   └── generar_notebooks.py                Ensambla los cinco notebooks
│
└── resultados/                             Se crea al ejecutar los notebooks
    ├── qwen.json · bert.json · pipeline_qwen_bert.json · flan_t5.json
    └── comparacion_final.json
```

---

## Orden de ejecución

Los notebooks 1 y 2 son independientes. El 3 necesita los artefactos de ambos,
y el 5 necesita los `resultados/*.json` de los cuatro.

```
  1 · Qwen  ──┐
              ├──> 3 · Pipeline ──┐
  2 · BERT  ──┘                   ├──> 5 · Comparación
                                  │
  4 · FLAN-T5 ────────────────────┘
```

| Notebook | Requiere | Produce |
|---|---|---|
| 1 · Qwen | — | `adaptadores/qwen-lora/`, `resultados/qwen.json` |
| 2 · BERT | — | `modelos/bert-clasificador/`, `resultados/bert.json` |
| 3 · Pipeline | 1 y 2 | `resultados/pipeline_qwen_bert.json` |
| 4 · FLAN-T5 | — | `adaptadores/flan-t5-lora/`, `resultados/flan_t5.json` |
| 5 · Comparación | 1, 2, 3 y 4 | `resultados/comparacion_final.json` |

**Qué subir a Colab:** solo los cinco `.ipynb`. El corpus va embebido en cada
uno, así que no hace falta subir `data/`. La carpeta `scripts/` tampoco: sirve
para regenerar el proyecto desde un equipo local.

**Persistencia entre notebooks:** cada notebook trae una celda "Persistencia
entre notebooks" justo después de la instalación. Monta Google Drive y trabaja
desde `MyDrive/ProyectoIA`, de modo que los artefactos de los notebooks 1, 2 y
4 sigan ahí cuando ejecuten el 3 y el 5. Con `USAR_DRIVE = False` todo queda en
`/content`, lo cual sirve si ejecutan 1, 2 y 3 seguidos sin desconectar. Fuera
de Colab la celda no hace nada.

Los checkpoints intermedios del `Trainer` van siempre al disco local del
runtime (`DIR_CHECKPOINTS`), nunca a Drive: son varios GB en el caso del
notebook 2 y son desechables.

**GPU:** `Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU`.
El notebook 5 no la necesita: solo carga tokenizadores y agrega JSON.

**Credenciales:** solo la API key de W&B, que pide `wandb.login()`. Los tres
modelos son públicos en Hugging Face, así que no hace falta token.

---

## El corpus

165 ejemplos en español, 11 categorías × 15 ejemplos, particiones estratificadas
de 132 entrenamiento / 33 validación (semilla 42).

Cada ejemplo:

```json
{
  "id": "div-01",
  "split": "train",
  "categoria": "division",
  "categoria_id": 3,
  "tipo": "problema",
  "nivel": "basico",
  "origen": "base",
  "entrada": "María tiene 48 caramelos y quiere repartirlos por igual entre 6 amigos. ¿Cuántos caramelos recibirá cada amigo?",
  "salida": "Paso 1: Repartir en partes iguales es dividir.\nPaso 2: 48 ÷ 6 = 8.\nRespuesta final: 8 caramelos",
  "respuesta_final": "8 caramelos",
  "valor": "8",
  "es_demo": false
}
```

### Curación aplicada

Partiendo de los 50 ejemplos originales:

| Principio | Qué se hizo |
|---|---|
| **Formato uniforme** | La `salida` se genera mecánicamente desde una lista de pasos. Ningún ejemplo puede desviarse del formato porque nadie lo escribe a mano. |
| **Consistencia** | Un esquema de campos único, una plantilla de prompt única, y un marcador `Respuesta final:` que hace posible la evaluación automática. |
| **Representatividad** | 11 categorías balanceadas a 15 ejemplos. Los originales estaban muy sesgados: 19 de operaciones combinadas y solo 2 de estadística. |
| **Diversidad** | Mezcla de problemas en lenguaje natural (75) y operaciones directas (90); dos niveles de dificultad; contextos variados (dinero, distancias, personas, objetos). |
| **Limpieza** | Verificación aritmética automática de los 165 ejemplos, deduplicación exacta y normalizada, y separación de los ítems originales que hacían dos preguntas a la vez. |
| **Trazabilidad** | Cada ejemplo declara `origen`: 52 provienen de los 50 originales (algunos divididos), 113 son nuevos. |

### Esquema de etiquetado

Las categorías se solapan por naturaleza, así que el etiquetado sigue una
cascada de precedencia determinista:

```
1. ¿Área, perímetro, ángulos, volumen?           -> geometria
2. ¿Hay que despejar una incógnita?              -> ecuaciones
3. ¿Promedio, mediana, moda, rango, probabilidad?-> estadistica_probabilidad
4. ¿Interviene un porcentaje?                    -> porcentajes
5. ¿Los operandos son fracciones?                -> fracciones
6. ¿Hay potencias o raíces?                      -> potencias_raices
7. ¿Dos o más operaciones distintas / paréntesis?-> operaciones_combinadas
8. En otro caso, la única operación presente     -> suma|resta|multiplicacion|division
```

### Escalar el corpus

Dos caminos:

**Sustituir el archivo.** Los notebooks usan `data/math_tutor_dataset.jsonl` si
existe y solo escriben la copia embebida cuando falta. Basta con dejar ahí un
JSONL mayor con el mismo esquema.

**Ampliar la fuente y regenerar** (recomendado, porque mantiene la validación):

```bash
python scripts/dataset_fuente.py
```

```bash
python scripts/generar_notebooks.py
```

El primero valida el corpus —aritmética, duplicados, balance, formato— y falla
si algo no cuadra. El segundo reconstruye los cinco notebooks con el corpus
nuevo embebido.

---

## Métricas

**Generación** (notebooks 1, 3, 4)

| Métrica | Qué mide |
|---|---|
| `exactitud` | ¿El número final es correcto? La única que responde "¿sirve como tutor?" |
| `formato_valido` | ¿Usó `Paso N:` y `Respuesta final:`? Separa aprender a resolver de aprender a formatear. |
| `rouge_l` | Solapamiento del procedimiento con la referencia. Métrica débil, léase como tal. |
| `long_media` | Palabras generadas. Detecta divagación del baseline. |

**Clasificación** (notebooks 2, 4)

Accuracy, precision/recall/F1 macro y ponderado, reporte por clase y matriz de
confusión. La métrica principal es **F1-macro**: con 11 clases, si el modelo
ignora una entera, el F1-macro se hunde aunque la accuracy apenas se mueva.

---

## Weights & Biases

Los cinco notebooks comparten convención:

- **Proyecto:** `tutor-matematicas-arquitecturas`
- **Runs:** `qwen-lora-finetune`, `bert-clasificador-finetune`,
  `pipeline-qwen-bert`, `flan-t5-lora-finetune`
- **Tags:** arquitectura y fase, para filtrar en el panel

Para trabajar sin W&B, pongan `USAR_WANDB = False` en la celda correspondiente.
Las métricas se guardan igual en `resultados/`.

---

## Notas técnicas que ahorran horas

**`peft` + `torchao` — `get_peft_model()` falla con un `ImportError` que no
menciona LoRA.** Colab trae `torchao` preinstalado y `peft` comprueba su versión
con una función que lanza `ImportError` en vez de devolver `False` cuando la
encuentra más antigua de lo que espera:

```
ImportError: Found an incompatible version of torchao. Found version 0.10.0,
but only versions above 0.16.0 are supported
```

La celda de instalación ya lo resuelve con `%pip uninstall -y -q torchao`, y la
celda de entorno avisa por adelantado si el conflicto sigue presente. Si aparece
en una sesión antigua: desinstalar torchao, **recargar el modelo** (no solo
reejecutar la celda de LoRA: `get_peft_model` puede haber dejado el modelo a
medio envolver) y continuar.

**Qwen — los parámetros LoRA deben estar en fp32.** Con los pesos base en fp16 y
entrenamiento en precisión mixta, dejar los parámetros entrenables en fp16
provoca subdesbordamiento del gradiente y la pérdida se va a `NaN`. El notebook
1 hace el casting explícitamente.

**FLAN-T5 — nunca fp16.** T5 fue preentrenado en bfloat16; sus activaciones
desbordan el rango de fp16 y la pérdida diverge. La T4 no tiene bf16, así que
la única opción es fp32. Está fijado en el notebook 4.

**FLAN-T5 — el tokenizador no cubre el español.** Su SentencePiece de 32k fue
entrenado sobre corpus en inglés y convierte `¿`, `÷`, `√`, `²` en `<unk>`. El
notebook 4 lo mide y normaliza el texto antes de entrenar.

**Pipeline — alineación de etiquetas.** El orden de `id2label` del clasificador
debe coincidir con `CATEGORIAS`. Si no coincide, el pipeline enruta con
estrategias equivocadas sin lanzar ningún error. El notebook 3 lo comprueba con
un `assert`.

---

## Limitaciones

Conviene tenerlas presentes al leer cualquier resultado:

- **33 ejemplos de validación.** Un ejemplo vale 3 puntos porcentuales.
  Diferencias menores a ~10 puntos entre modelos no son concluyentes.
- **3 ejemplos de validación por clase.** El recall por clase se mueve en saltos
  de 33 puntos. Sirve para detectar clases con recall 0, no para comparar
  clases entre sí.
- **Una sola semilla.** No hay estimación de la varianza entre ejecuciones.
- **Las métricas no miden calidad pedagógica.** Un procedimiento correcto y un
  procedimiento *bien explicado* no son lo mismo, y nada aquí mide lo segundo.
