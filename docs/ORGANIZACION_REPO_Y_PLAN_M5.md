# Organización del repositorio y preparación de M5

Fecha: 29-10-2026 · **Hay commits locales** (ver §3.2). Sin push, sin merge a
`main`, sin Space publicado.

Este documento describe el estado **real y verificado** del repositorio tras la
reorganización. Cada afirmación está respaldada por una comprobación que se
indica al lado.

---

## 1 · Estructura final y criterio de ubicación

```
ProyectoIA/
├── M1_arquitecturas/      Qwen+LoRA, BETO, FLAN-T5, pipeline en cascada
├── M2_harness/            harness de 3 dimensiones y scoreboard
├── M3_rag_agentes/        RAG, agente ReAct, herramientas, RAGAS
│   ├── notebooks/         FUENTE canónica de los notebooks de S07/S08/S10
│   ├── resultados/        FUENTE canónica de los resultados
│   └── entrega/           paquete autocontenido (artefacto derivado)
├── m4/                    entrada visual: eval set, fotos, corridas, pipeline
├── Presentaciones/        PDF de cada sesión + notebooks de referencia
│   └── notebooks/         S14 y S15 (material de clase)
├── data/                  eval sets, corpus y trayectorias (raíz compartida)
├── scripts/               generadores, fuentes de datos y el núcleo del agente
├── adaptadores/           pesos LoRA · FUERA de Git (.gitignore)
├── graphify-out/          grafo de conocimiento (caché generada)
└── M5_despliegue/         aplicación, medición, documentación y pruebas

../ProyectoIA-m4vision/    worktree de m4-vision · FUERA de la carpeta principal
```

**La carpeta principal está en `integracion-m5`.** Ya no hay una segunda copia
del proyecto dentro de ella: el worktree de integración se retiró tras registrar
su trabajo, y el de M4 se **movió** —no se borró— a `../ProyectoIA-m4vision`,
como copia de seguridad viva.

`ProyectoIA_artefactos_m1/` (421 MB de modelos y corridas de W&B) sigue dentro de
la carpeta pero **git lo ignora**, así que no entra en ningún commit.

**Criterio de ubicación**, en orden de prioridad:

1. **Que las rutas sigan funcionando.** Los notebooks y scripts llevan rutas
   incrustadas (`m4/eval_set/imagenes`, `data/eval_set_m2.jsonl`). Nada se movió
   por estética.
2. **Una entrega, una carpeta.** Cada módulo vive bajo su propio directorio.
3. **Raíces compartidas.** `data/`, `scripts/` y `adaptadores/` se quedan donde
   estaban: los usan todos los módulos.
4. **Fuente, evidencia y entrega, separadas.** Un notebook sin salidas es fuente;
   uno con salidas es evidencia de una corrida y no se regenera; `entrega/` es el
   paquete que se presentó.

### Por qué `m4/` y no `M4_vision/`

La propuesta inicial de nombrar la carpeta `M4_vision/` se descartó: las rutas
`m4/...` están incrustadas en los notebooks, los scripts y las tres corridas
archivadas. Renombrar rompería la evidencia. `M4_vision/` era un residuo de un
renombrado accidental en Finder, no un destino (ver §2).

---

## 2 · Registro de duplicados y movimientos

### 2.1 · Residuo `M4_vision/` — resuelto

Contenía 2 archivos más 2 `.DS_Store`. Verificación byte a byte **antes** de tocar
nada:

| Archivo | `M4_vision/` | `HEAD:m4/` | `m4-vision:m4/` |
|---|---|---|---|
| `eval_set/HOJAS_PARA_ESCRIBIR.md` | `79731331151ef046` | idéntico | idéntico |
| `eval_set/eval_set_m4.csv` | `b0bc460424fb7cde` | idéntico | idéntico |

**Acción:** los dos archivos se restauraron en `m4/eval_set/` por copia (no se usó
ningún comando que descarte cambios). Verificado que coinciden con `HEAD` y que
abren. La carpeta residual se movió a la **Papelera**, no se borró de forma
irreversible.

**Resultado:** `git status` ya no muestra los dos ` D` de `m4/eval_set/`.

### 2.2 · Entrega de M3 — conservada y ahora reproducible

24 de los 26 archivos de `M3_rag_agentes/entrega/` son copias byte a byte de
fuentes canónicas: 3,3 MB repetidos.

**Decisión: se conservan.** La entrega tiene que abrirse sola. Lo que cambia es
que dejan de ser copias a mano: `scripts/empaquetar_entrega.py` declara el mapa
destino → fuente, **verifica** por omisión y regenera con `--reconstruir`.

| Grupo | Nº | Fuente canónica |
|---|---|---|
| Notebooks | 5 | `M3_rag_agentes/notebooks/` |
| Resultados | 18 | `M3_rag_agentes/resultados/` |
| Scoreboard | 1 | `M2_harness/documentacion/SCOREBOARD.md` |
| **Propios de la entrega** | 2 | `entrega/documentacion/` — su copia canónica está ahí |

Verificación ejecutada: *"La entrega coincide con sus fuentes canónicas."*

### 2.3 · Materiales de clase — una sola copia

| Material | Antes | Ahora | SHA-256 (16) |
|---|---|---|---|
| PDF S14 | `~/Downloads` | `Presentaciones/` | `7801cdc024732523` |
| Notebook S14 | `~/Downloads` | `Presentaciones/notebooks/` | `cfec499385ee2eca` |
| Notebook S14 RESUELTO | `~/Downloads` | `Presentaciones/notebooks/` | `58b7539ddea4661c` |
| Notebook S15 RESUELTO | `~/Downloads` | `Presentaciones/notebooks/` | `fd6053ade63a9aae` |
| PDF S15 | ya en `Presentaciones/` | sin cambios | — |

Se verificó hash idéntico y que los cuatro abren (JSON válido en los notebooks,
cabecera `%PDF` en el PDF) **antes** de retirar las copias de Descargas, que se
movieron a la Papelera.

### 2.4 · Notebooks de M4 en Descargas — no se copiaron

Los cuatro son **byte a byte idénticos** a archivos que la rama ya tiene:

| En Descargas | Ya archivado como |
|---|---|
| `S13_M4_FineTuning_Diagnostico.ipynb` | `m4/pipeline/S13_M4_FineTuning_Diagnostico_ejecutado.ipynb` |
| `S13_M4_Pipeline_Visual-2.ipynb` | `m4/pipeline/S13_M4_Pipeline_Visual_AB_dev_ejecutado_2026-10-25.ipynb` |
| `S13_M4_Pipeline_Visual.ipynb` | `m4/pipeline/S13_M4_Pipeline_Visual_dev_dg_ejecutado_2026-10-25.ipynb` |
| `S13_M4_Pipeline_Visual_ejecutado_dev_2026_10_25.ipynb` | `m4/pipeline/S13_M4_Pipeline_Visual_ejecutado_dev_2026_10_25.ipynb` |

Ninguno aporta resultados que falten. **Se dejaron en Descargas**: son archivos
personales y el encargo limitaba las operaciones fuera del proyecto a los
materiales de clase.

---

## 3 · Integración de M4 y estado de las ramas

### 3.1 · Lo que contiene cada rama

| Rama | Commit | Remoto | Contenido propio |
|---|---|---|---|
| `origin/main` | `c91ed6e` | — | "Clean up README content" (−11 líneas de `README.md`) |
| `main` (local) | `e5101da` | ahead 1 / behind 1 | eval set de visión de M4 (12 casos + hojas) |
| `m5-interfaz` | `e5101da` | **sin remoto** | nada propio: apunta al mismo commit que `main` |
| `mejora-nucleo` | `9e797e2` | **sin remoto** | banco de errores, enrutador reutilizable, verificador de planteamiento |
| `m4-vision` | `9c0c0df` | sincronizado 0/0 | **M4 completo**: 73 archivos en `m4/` |
| **`integracion-m5`** | `8f02818` | nueva, local | los cuatro historiales anteriores |

### 3.2 · Commits locales creados

Tres, ninguno publicado:

| Commit | Qué |
|---|---|
| `748a00a` | merge de `mejora-nucleo` en `integracion-m5` |
| `8f02818` | merge de `m4-vision` en `integracion-m5` |
| `547a2db` | el informe y `scripts/empaquetar_entrega.py` |

`integracion-m5` está 8 commits por delante de `origin/main`.

### 3.3 · Cómo se integró

Antes de integrar se simularon las tres fusiones con `git merge-tree`, que **no
escribe nada**: ninguna presentaba conflictos. La integración se hizo en un
**worktree temporal** (`ProyectoIA-integracion/`), sin tocar el checkout
principal ni el de M4.

```
integracion-m5  (desde origin/main c91ed6e)
  ├── merge mejora-nucleo   748a00a   trae e5101da + 9e797e2
  └── merge m4-vision       8f02818   trae f82c0d5 + ca005d9 + 9c0c0df
```

Comprobado con `git merge-base --is-ancestor` que los cuatro están contenidos:
`origin/main`, `main`, `mejora-nucleo` y `m4-vision`.

### 3.4 · Qué llegó, verificado archivo a archivo

- **M4**: 73 archivos — 4 notebooks ejecutados, 16 archivos en 3 corridas
  archivadas, 9 CSV de desarrollo, 11 fotos, 19 imágenes de calibración,
  `README.md` y `CONCLUSIONES_M4.md`.
- **Núcleo**: `verificador.py`, `enrutador.py`, `banco_errores.py`,
  `data/errores_observados.jsonl`.
- **M4 en scripts**: `vision.py`, `verificador_aritmetico.py`, `nb_m4.py`,
  `nb_m4_diagnostico.py`, `data/trayectorias_diagnostico.jsonl`.

### 3.5 · Riesgos de las ramas

1. **`mejora-nucleo` no tiene remoto.** El verificador, el enrutador y el banco de
   errores **solo existen en este disco**. Es lo más frágil del repositorio.
2. **`main` diverge 1/1.** Cualquier push necesita rebase previo.
3. **`m5-interfaz` está vacía de M5**: apunta al mismo commit que `main`.

---

## 4 · M5 · implementado, sin medir

*Revisado el 30-10-2026. Lo que decía esta sección el 29-10 era correcto sobre la
estructura y falso sobre lo que la app servía: ver «La configuración se elige».*

```
M5_despliegue/
├── app/configuraciones.py  las 9 configuraciones y su resultado medido
├── app/pipeline.py     monta el tutor reutilizando scripts/agente/
├── app/app.py          Gradio: pestaña de texto, de imagen y de estado
├── app/salud.py        health check que no fuerza la carga del modelo
├── app/telemetria.py   registro estructurado, W&B opcional
├── config/             requirements fijados, README del Space, .env.example
├── evaluacion/         verificar_retrieval.py (sin GPU) · ablaciones.py ·
│                       medir_eficiencia.py
├── docs/               DESPLIEGUE.md · GUION_DEMO.md
└── tests/              28 pruebas sin GPU
```

**La configuración se elige, no se adivina.** `TUTOR_CONFIG` decide qué se sirve;
por omisión `s08d` (18/21). La primera versión de `pipeline.py` montaba el prompt
del agente, búsqueda léxica y `qwen-lora-agente` **si estaba en el disco**: una
variante de C5, 12/21, con el nombre de la mejor medida. Las ocho configuraciones
están declaradas con su cifra de respaldo, o con que no la hay.

**Y se comprueba.** `evaluacion/verificar_retrieval.py` monta el retrieval y el
enrutador sin cargar el LLM y los compara contra lo que el notebook dejó en
`m3_rag_avanzado.json`: pesos, leave-one-out y las 21 rutas coinciden. Así se
encontraron tres discrepancias —falta del cross-encoder, `rerank_max=None` y
embeddings sin normalizar— que dejaban la validación en 0,939 en vez de 0,970.

**No duplica la lógica del tutor.** `pipeline.py` importa `registro`,
`herramientas`, `nucleo`, `documentos`, `prompts`, `enrutador`, `vision` y
`verificador_aritmetico`; el corpus se obtiene ejecutando su fuente canónica en
`nb_comun_rag`, igual que hace `trayectorias_fuente.py`. Dos pruebas lo vigilan.

**Degrada en lugar de caerse.** El montaje va por capas y cada fallo se anota:
la interfaz muestra *"Tutor NO disponible: modelo: falta torch"* en vez de
reventar. El retrieval tiene un repliegue léxico que funciona sin embeddings.

**Privacidad.** No se registra el texto del usuario por defecto —solo longitud y
hash—; W&B se apaga limpio si falta la credencial. Con pruebas.

**Un detalle que salió en las pruebas:** el módulo de telemetría se llamaba
`registro.py` y ensombrecía a `scripts/agente/registro.py`, el Tool Registry.
Renombrado a `telemetria.py`.

**Lo que NO existe:** Space publicado, URL, corrida de los 21 casos en este
entorno, medición de eficiencia ejecutada y telemetría recogida. El guion de demo
los marca `[PENDIENTE]` y no los inventa. El 18/21 es una cifra de S08, no una
medición de M5: lo comprobado es que M5 monta esa configuración.

## 5 · Cambios realizados

| Archivo | Qué |
|---|---|
| `Presentaciones/SI4006_S14_Semana14_Sesion14.pdf` | nuevo (movido de Descargas) |
| `Presentaciones/notebooks/S14_Lab_Destilacion.ipynb` | nuevo |
| `Presentaciones/notebooks/S14_Lab_Destilacion_RESUELTO.ipynb` | nuevo |
| `Presentaciones/notebooks/S15_Lab_Despliegue_RESUELTO.ipynb` | nuevo |
| `m4/eval_set/HOJAS_PARA_ESCRIBIR.md` | restaurado |
| `m4/eval_set/eval_set_m4.csv` | restaurado |
| `M4_vision/` | retirado a la Papelera tras verificar duplicación |
| `scripts/empaquetar_entrega.py` | nuevo: verifica y regenera la entrega de M3 |
| `docs/ORGANIZACION_REPO_Y_PLAN_M5.md` | este documento |
| `M5_despliegue/` | nuevo; revisado el 30-10: `app/configuraciones.py`, `evaluacion/verificar_retrieval.py`, `evaluacion/ablaciones.py` |
| `scripts/agente/herramientas.py` | 3 herramientas de cálculo con topes, ambigüedad y rechazo de divergentes |
| `scripts/taxonomia_errores.py` | reescrito: etiquetado exploratorio, una etiqueta principal por caso |
| `docs/REFINAMIENTO_TUTOR_M5.md` | revisado el 30-10 con las seis correcciones |
| rama `integracion-m5` | nueva, con los cuatro historiales · **activa en la carpeta principal** |

---

## 6 · Verificaciones locales y limitaciones

**Ejecutadas, todas en local y sin GPU:**

| Verificación | Resultado |
|---|---|
| `pytest scripts/tests` en `integracion-m5` | **306 pasan** |
| `pytest M5_despliegue/tests` | **54 pasan** (sin GPU) |
| Ambas suites en el entorno fijado de `config/requirements.txt` (Python 3.9) | **360 pasan** |
| `evaluacion/verificar_retrieval.py` | **reproduce S08-D**: pesos, leave-one-out y 21/21 rutas |
| Búsqueda de secretos en `M5_despliegue/` y `docs/` | ninguno |
| Rutas de datos e imágenes | las 6 comprobadas existen |
| Generador de notebooks | 14 registrados, 0 destinos inexistentes |
| Guarda anti-sobrescritura | protege 5 notebooks con salidas |
| Entrega M3 vs fuentes canónicas | coincide |
| Hashes de materiales de clase | los 4 idénticos tras mover |

**Dependen de Colab, GPU, Hugging Face o W&B, y no se ejecutaron:**

- cualquier corrida del tutor, del VLM o de los adaptadores;
- la medición de eficiencia fp16 vs int4;
- la publicación del Space y la telemetría.

**Lo que M4 dejó, sin adornos.** Ninguna configuración mejoró la exactitud:
3/9, 2/9 y 2/9 sobre 9 casos de desarrollo. **Los 12 casos visuales ya se
gastaron**: tres corridas exploratorias dieron 4/12, 1/12 y 2/12, y la de 4/12
repitió «Paso 2» en las 11 respuestas con conclusión —un sistema que contestara
siempre «Paso 2» habría acertado lo mismo—. Son evidencia histórica y conjunto de
regresión, no un holdout ciego. El detalle está en `m4/README.md` y
`m4/CONCLUSIONES_M4.md`, que ya lo decían; lo que estaba mal era este informe.

---

## 7 · Pendientes manuales

**En Colab o con GPU:**
1. Medición de eficiencia fp16 vs int4 sobre el mismo checkpoint y conjunto.
2. Conectar el verificador aritmético al prompt y volver a medir en desarrollo.
3. **6 fotografías** que faltan: las 3 hojas `dev-*`, escritas a mano, cada una
   nítida y desenfocada, para validar la detección de ilegibles. El conjunto de
   `calibracion/` es texto impreso generado por script y **no sustituye** a fotos
   reales.

**En Hugging Face:** crear el Space, subir `app.py` y `requirements.txt`,
configurar las variables. **No se ha publicado nada.**

**En W&B:** crear el proyecto y la clave. La app debe funcionar sin ella.

**Destilación (S14):** no hay modelo estudiante ni artefacto utilizable. Queda
documentada como alternativa; la prioridad de S15 es la cuantización.

---

## 8 · Comandos recomendados · **no ejecutados**

Revisar la integración:

```bash
git -C ProyectoIA-integracion log --oneline --graph -12
git -C ProyectoIA-integracion diff --stat origin/main
```

Publicar lo que no tiene respaldo remoto (lo más urgente):

```bash
git push -u origin mejora-nucleo
git push -u origin integracion-m5
```

Reconciliar `main` antes de cualquier integración:

```bash
git switch main && git pull --rebase origin main
```

Cuando la integración esté aprobada:

```bash
git switch main
git merge --no-ff integracion-m5
git push origin main
```

Para que el Finder muestre el estado integrado en la carpeta principal:

```bash
git switch integracion-m5          # los archivos sin seguimiento se conservan
```

Tras cambiar código, actualizar el grafo:

```bash
graphify update .
```
