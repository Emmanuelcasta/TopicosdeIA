# Plantilla de definición del proyecto integrador

Completa esta plantilla con tu equipo (3–4 personas). Es el documento que hace
crecer el proyecto módulo a módulo durante todo el semestre.

> **En la Sesión 1 solo se exigen los campos 1 y 2.** El resto se completa para
> la Sesión 2. No borres los campos vacíos: déjalos con su encabezado para
> irlos llenando.

Reemplaza cada bloque `> _...` con tu respuesta. Los ejemplos van entre
comentarios HTML y no se renderizan.

---

## 1. Dominio

<!-- Ejemplo: Atención al ciudadano en una alcaldía municipal. -->

> Tutor de matemáticas para estudiantes de secundaria (grados 9° a 11°), enfocado en álgebra y geometría básica. El sistema no reemplaza al docente: revisa el ejercicio que el estudiante ya resolvió (a mano o digitado) y da retroalimentación sobre el paso específico donde se equivocó, en lugar de solo decir si el resultado final está bien o mal.

---

## 2. Usuario + decisión

> **Usuario 1 — Estudiante** de secundaria (9°-11°) que resuelve ejercicios de álgebra o geometría en casa, sin el profesor presente para resolver dudas en el momento.
>
> **Decisión que cambia:** qué debe corregir primero en su procedimiento (por ejemplo: "el error no está en el resultado final, sino en cómo despejaste la variable en el paso 2"), en vez de solo saber que la respuesta quedó mal sin entender en qué parte del razonamiento falló. Hoy, sin el profesor al lado, o repite el ejercicio a ciegas o espera hasta la siguiente clase para preguntar; el sistema le permite corregir el paso exacto donde falló, en el momento en que lo necesita.
>
> **Usuario 2 — Profesor** que revisa los ejercicios de todo un curso (30+ estudiantes).
>
> **Decisión que cambia:** a qué estudiantes agrupar para un refuerzo puntual esta semana, según el tipo de error que más se repite en el curso (por ejemplo, despeje de variables vs. signos vs. jerarquía de operaciones), en vez de corregir ejercicio por ejercicio sin poder ver el patrón general y sin tiempo para planear una intervención grupal.
>
> Los dos usuarios comparten el mismo motor de análisis (detectar en qué paso del procedimiento está el error), pero lo usan para decisiones distintas: el estudiante corrige su propio ejercicio en el momento; el profesor prioriza a quién reforzar y con qué tema, a nivel de curso.

<!--
Ejemplo:
- Usuario: un funcionario de la ventanilla de atención.
- Decisión que cambia: a qué dependencia enrutar un trámite y qué documentos
  pedirle al ciudadano en el momento, en vez de mandarlo a averiguar y volver.
-->

---

## 3. Tarea del modelo (M1)

<!-- Ejemplo: clasificar el tipo de trámite a partir de la descripción libre
del ciudadano. -->

> _¿Qué tarea de ML resuelve el modelo ajustado del Módulo 1?_

---

## 4. Dataset + licencia

<!-- Ejemplo: 1.200 solicitudes históricas anonimizadas; licencia de uso
interno con permiso de la entidad. -->

> _¿Con qué datos entrenas/evalúas? ¿De dónde salen y bajo qué licencia?_

---

## 5. Métrica de éxito

<!-- Ejemplo: F1 macro > 0.80 en enrutamiento; y que el funcionario acepte la
sugerencia en ≥ 70% de los casos en la prueba con usuarios. -->

> _¿Cómo mides que el sistema sirve? Métrica técnica y señal de valor real._

---

## 6. Componente visual (M4)

<!-- Ejemplo: leer el documento escaneado que adjunta el ciudadano y verificar
que corresponde al trámite. -->

> _¿Qué aporta el componente multimodal/visual del Módulo 4 al sistema?_

---

## 7. Riesgos éticos

<!-- Ejemplo: sesgo contra solicitudes mal redactadas; riesgo de negar un
trámite por un error del modelo. Mitigación: el sistema sugiere, el funcionario
decide. -->

> _¿Qué puede salir mal para una persona real? ¿Cómo lo mitigas?_

---

## 8. Compromisos del equipo

<!-- Ejemplo: reuniones los martes; repositorio compartido; cada integrante es
dueño de un módulo pero todos revisan. -->

> _¿Cómo se organizan? ¿Quién responde por qué? ¿Cómo se comunican?_

---

# Estructura del repositorio

Proyecto integrador SI4006 · **Tutor inteligente de matemáticas**. Cada módulo tiene
sus notebooks, su documentación y sus resultados.

```
.
├── M1_arquitecturas/     Comparación de 4 arquitecturas con fine-tuning (S04)
├── M2_harness/           Harness de evaluación de 3 dimensiones (S06)
├── M3_rag_agentes/       RAG, RAG avanzado y arquitectura agéntica (S07, S08, S10)
│   └── entrega/          El paquete de la entrega M3, autocontenido
├── data/                 Corpus, eval sets y trayectorias (fuente de verdad)
├── scripts/              Código: generadores de notebooks, paquete del agente y tests
└── GUIA_TECNICA.md       Corpus, métricas, W&B y notas técnicas que ahorran horas
```

## Los módulos

| Módulo | Notebooks | Qué responde | Documentación |
|---|---|---|---|
| **M1** | S04 ×5 | ¿Qué arquitectura sirve para un tutor? Decoder-only, encoder-only, pipeline y encoder-decoder | `M1_arquitecturas/documentacion/M1_Documentacion.md` |
| **M2** | S06 | ¿Cómo se mide? Harness de 3 dimensiones sobre un eval set difícil de 21 casos | `M2_harness/documentacion/SCOREBOARD.md` |
| **M3** | S07, S08, S10 ×4 | ¿RAG, herramientas y agentes mejoran al tutor? | `M3_rag_agentes/entrega/documentacion/` |

## La entrega M3

`M3_rag_agentes/entrega/` es autocontenida: notebooks ejecutados con sus salidas,
resultados y los tres documentos del reporte.

| Requisito de la entrega | Dónde |
|---|---|
| RAG con ≥2 técnicas avanzadas | `notebooks/S08_Lab_RAG_Avanzado_*.ipynb` — hybrid + RRF, reranking, enrutador |
| ≥1 herramienta | `notebooks/S10_Lab_Herramientas_*.ipynb` — 15 herramientas matemáticas + `buscar_documentos` |
| Scorecard propio + RAGAS | `resultados/scorecard_final_m3_v2.csv` |
| Lectura honesta | `documentacion/M3_Agente_Documentacion.md` §7 |

**El resultado en una línea:** el enrutador de S08 lleva el tutor de 14/21 a 18/21;
la arquitectura agéntica solo funciona con el adaptador entrenado (el uso de
herramientas pasa del 2% al 96%), y aun así cuesta 19× los tokens.

## Ejecutar

Los notebooks están pensados para Colab o Lightning con GPU T4. Llevan los datos
embebidos, así que basta con subir el `.ipynb` y tener `adaptadores/qwen-lora/`
(el adaptador de M1) en el directorio de trabajo. Los pesos no están en el
repositorio: se regeneran ejecutando `S04_Lab_Fine_tuning_Qwen.ipynb`.

Para regenerar los notebooks desde el código fuente:

```bash
python -m pytest scripts/tests -q
python scripts/generar_notebooks.py
```

> Regenerar **borra las salidas** de los notebooks ejecutados. No lo hagas sobre la
> carpeta de entrega.
