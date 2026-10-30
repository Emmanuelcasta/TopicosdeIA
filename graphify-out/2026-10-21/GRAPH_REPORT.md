# Graph Report - ProyectoIA  (2026-10-21)

## Corpus Check
- 59 files · ~112,622 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 67 file(s) not represented in the graph (top: .csv 33, .ipynb 22, .jsonl 9)

## Summary
- 532 nodes · 1299 edges · 21 communities (17 shown, 4 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 98 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e5101daa`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Notebook Generation
- AgenteTutor
- eval_set_fuente.py
- M1 Architecture Comparison
- Math Tools Library
- M2/M3 RAG & Agent Findings
- Evaluation Scoreboard
- re
- test_agente.py
- Trajectory Generation
- RAG-as-Tool & Agent Nucleus
- Tool Registry
- dataset_fuente.py
- Tool Decorator & Schemas
- recalcular_criterio.py
- Hojas para escribir a mano
- eval_set_m4_fuente.py
- Normalizador
- a_fraccion
- RespuestaAgente

## God Nodes (most connected - your core abstractions)
1. `code()` - 44 edges
2. `md()` - 43 edges
3. `Guía técnica: Tutor de matemáticas · Comparación de arquitecturas` - 33 edges
4. `celda_drive()` - 27 edges
5. `encabezado()` - 26 edges
6. `Scoreboard del tutor de matemáticas (M2 harness / M3 RAG)` - 22 edges
7. `RegistroHerramientas` - 21 edges
8. `Módulo M1: Fine-tuning de modelo base para tutoría de matemáticas` - 21 edges
9. `AgenteTutor` - 18 edges
10. `celda_eval_set()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Limitaciones del experimento M1` --semantically_similar_to--> `Limitaciones (33 validación, una semilla, sin calidad pedagógica)`  [INFERRED] [semantically similar]
  M1_arquitecturas/documentacion/M1_Documentacion.md → GUIA_TECNICA.md
- `Corpus de 165 ejemplos (11 categorías × 15)` --semantically_similar_to--> `Corpus math_tutor_dataset.jsonl (165 ejemplos)`  [INFERRED] [semantically similar]
  M1_arquitecturas/documentacion/M1_Documentacion.md → GUIA_TECNICA.md
- `Dimensión 1 · Exactitud en cálculo y similitud por embeddings` --semantically_similar_to--> `Extracción justa: fallback al último número`  [INFERRED] [semantically similar]
  M2_harness/documentacion/SCOREBOARD.md → M1_arquitecturas/documentacion/M1_Documentacion.md
- `Eval set M2 (21 casos, data/eval_set_m2.jsonl)` --shares_data_with--> `scripts/nb_comun_eval.py (eval set y harness compartidos)`  [INFERRED]
  M2_harness/documentacion/SCOREBOARD.md → GUIA_TECNICA.md
- `Verificación simbólica del cálculo (SymPy)` --conceptually_related_to--> `scripts/agente/ (código del agente ReAct)`  [INFERRED]
  M1_arquitecturas/documentacion/M1_Documentacion.md → GUIA_TECNICA.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Flujo S10: herramientas → fine-tuning agente → ReAct → RAGAS** — guia_tecnica_s10_lab_herramientas, guia_tecnica_s10_finetuning_agente, guia_tecnica_s10_lab_agente_react, guia_tecnica_s10_evaluacion_ragas [EXTRACTED 1.00]
- **Cuatro arquitecturas comparadas en M1** — m1_arquitecturas_documentacion_m1_documentacion_qwen2_5_lora, m1_arquitecturas_documentacion_m1_documentacion_beto_clasificador, m1_arquitecturas_documentacion_m1_documentacion_pipeline_beto_qwen, m1_arquitecturas_documentacion_m1_documentacion_flan_t5_lora [EXTRACTED 1.00]
- **Harness M2 de tres dimensiones** — m2_harness_documentacion_scoreboard_dimension_1_metrica_clasica, m2_harness_documentacion_scoreboard_dimension_2_llm_judge, m2_harness_documentacion_scoreboard_dimension_3_aciertos_dominio, m2_harness_documentacion_scoreboard_eval_set_m2 [EXTRACTED 1.00]
- **Pipeline agéntico S10 (normalizador, ReAct, registro, verificador)** — m3_rag_agentes_entrega_documentacion_m3_agente_documentacion_normalizador, m3_rag_agentes_entrega_documentacion_m3_agente_documentacion_bucle_react, m3_rag_agentes_entrega_documentacion_m3_agente_documentacion_tool_registry, m3_rag_agentes_entrega_documentacion_m3_agente_documentacion_verificador, m3_rag_agentes_entrega_documentacion_m3_agente_documentacion_tool_calling_nativo [EXTRACTED 1.00]
- **Harness de tres dimensiones del scoreboard** — m3_rag_agentes_entrega_documentacion_scoreboard_dimension1_metrica_clasica, m3_rag_agentes_entrega_documentacion_scoreboard_dimension2_llm_judge, m3_rag_agentes_entrega_documentacion_scoreboard_dimension3_aciertos_dominio [EXTRACTED 1.00]
- **Técnicas incrementales de RAG avanzado S08** — m3_rag_agentes_entrega_documentacion_m2_m3_documentacion_hybrid_search, m3_rag_agentes_entrega_documentacion_m2_m3_documentacion_reranker, m3_rag_agentes_entrega_documentacion_m2_m3_documentacion_enrutador, m3_rag_agentes_entrega_documentacion_m2_m3_documentacion_cita_textual [EXTRACTED 1.00]

## Communities (21 total, 4 thin omitted)

### Community 0 - "Notebook Generation"
Cohesion: 0.13
Nodes (38): main(), validar_celdas(), construir(), construir(), construir(), celda_eval_m3(), celda_trayectorias(), celdas_modulos() (+30 more)

### Community 1 - "AgenteTutor"
Cohesion: 0.15
Nodes (14): AgenteTutor, EventoTraza, Guion, test_anadir_herramienta_sin_tocar_el_nucleo(), test_buscar_documentos_alimenta_los_contextos(), test_contexto_fijo_y_normalizador(), test_error_de_dominio_llega_al_modelo(), test_error_de_formato_se_devuelve_y_el_agente_se_recupera() (+6 more)

### Community 2 - "eval_set_fuente.py"
Cohesion: 0.16
Nodes (10): a_numero(), main(), reporte(), validar(), construir(), E(), main(), reporte() (+2 more)

### Community 3 - "M1 Architecture Comparison"
Cohesion: 0.07
Nodes (46): Guía técnica: Tutor de matemáticas · Comparación de arquitecturas, scripts/dataset_fuente.py (valida y exporta JSONL), Tokenizador FLAN-T5 no cubre el español (<unk>), scripts/generar_notebooks.py (ensambla notebooks), Limitaciones (33 validación, una semilla, sin calidad pedagógica), M2_M3_Documentacion.md, M3_Agente_Documentacion.md, Corpus math_tutor_dataset.jsonl (165 ejemplos) (+38 more)

### Community 4 - "Math Tools Library"
Cohesion: 0.09
Nodes (23): aplicar_porcentaje(), _decimal(), _decimales_a_racional(), dividir(), _es_decimal_exacto(), evaluar_expresion(), formatear(), multiplicar() (+15 more)

### Community 5 - "M2/M3 RAG & Agent Findings"
Cohesion: 0.06
Nodes (47): M2 y M3 — Harness de evaluación y RAG (documentación), Arquitectura recomendada: Qwen+LoRA con RAG condicional por enrutador, Artefacto de medición en adv-02 (claves 'no definido'), Cero abstenciones honestas en M2, Sistema E: prompt con cita textual, Enrutador consulta/problema (regresión logística de 3 señales, 97% LOO), Eval set M2 (21 casos: cálculo, conocimiento, adversarial), Fallo de generación en con-02 (no de retrieval) (+39 more)

### Community 6 - "Evaluation Scoreboard"
Cohesion: 0.11
Nodes (25): Scoreboard del tutor de matemáticas (M2 harness / M3 RAG), Señales acierto / abstuvo / alucinó, Bloque A · Cálculo (12 casos, taxonomía de dificultad), Bloque B · Conocimiento institucional (4 casos), Bloque C · Adversarial (5 casos), comparar_robusto (mitigación sesgo de posición), Dimensión 1 · Exactitud en cálculo y similitud por embeddings, Dimensión 2 · LLM-as-a-judge (Qwen pointwise, rúbrica 1-5) (+17 more)

### Community 7 - "re"
Cohesion: 0.07
Nodes (34): autoconsistencia(), mcnemar_exacto(), _media(), metricas_herramientas_caso(), resumen_agente(), robustez_mal_escrito(), a_numero(), mismo_numero() (+26 more)

### Community 8 - "test_agente.py"
Cohesion: 0.13
Nodes (15): analizar_salida(), SalidaModelo, test_herramienta_sin_docstring_de_argumentos_se_rechaza(), test_herramientas_resuelven_los_casos_del_eval_set(), test_parser_comillas_escapadas(), test_parser_json_invalido_es_error_de_formato(), test_parser_llamada_nativa_con_pensamiento(), test_parser_recupera_json_sin_etiquetas() (+7 more)

### Community 9 - "Trajectory Generation"
Cohesion: 0.16
Nodes (15): normalizar(), construir(), _ejecutar(), _jaccard(), _llamada_para(), llamadas_de_paso(), main(), _msg_asistente_llamada() (+7 more)

### Community 10 - "RAG-as-Tool & Agent Nucleus"
Cohesion: 0.21
Nodes (5): _a_llamada(), _json_de(), LlamadaHerramienta, ResultadoHerramienta, test_errores_se_devuelven_como_observacion()

### Community 11 - "Tool Registry"
Cohesion: 0.10
Nodes (7): chunks_de_resultado(), formatear_pasajes(), registro_con_documentos(), buscar_documentos(), Herramienta, RegistroHerramientas, fin()

### Community 12 - "dataset_fuente.py"
Cohesion: 0.16
Nodes (9): a_numero(), a_registro_final(), asignar_splits(), construir(), construir_salida(), main(), normalizar_texto(), reporte() (+1 more)

### Community 13 - "Tool Decorator & Schemas"
Cohesion: 0.25
Nodes (3): _esquema_tipo(), _parsear_docstring(), decorar()

### Community 15 - "recalcular_criterio.py"
Cohesion: 0.21
Nodes (7): _b(), escribir(), main(), reevaluar(), scorecard_s08(), scorecard_s10(), valor()

### Community 16 - "Hojas para escribir a mano"
Cohesion: 0.14
Nodes (13): Hojas para escribir a mano, `vis-01.jpg` · vis-01 · tiene un error, `vis-02.jpg` · vis-02 · tiene un error, `vis-03.jpg` · vis-03 · tiene un error, `vis-04.jpg` · vis-04 · tiene un error, `vis-05.jpg` · vis-05 · error compensado, `vis-06.jpg` · vis-06 · error compensado, `vis-07.jpg` · vis-07 · sin error (+5 more)

### Community 17 - "eval_set_m4_fuente.py"
Cohesion: 0.21
Nodes (7): C(), csv_entrega(), _evaluar_check(), hojas(), main(), paso(), validar()

## Knowledge Gaps
- **25 isolated node(s):** `Métricas de generación (exactitud, formato_valido, rouge_l, long_media)`, `Conflicto peft + torchao (ImportError)`, `Weights & Biases (proyecto tutor-matematicas-arquitecturas)`, `Análisis de tokenizadores (BPE, WordPiece, SentencePiece)`, `M3_Agente_Documentacion.md` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 159 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Scoreboard del tutor de matemáticas (M2 harness / M3 RAG)` connect `Evaluation Scoreboard` to `eval_set_fuente.py`, `M1 Architecture Comparison`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **What connects `Métricas de generación (exactitud, formato_valido, rouge_l, long_media)`, `Conflicto peft + torchao (ImportError)`, `Weights & Biases (proyecto tutor-matematicas-arquitecturas)` to the rest of the system?**
  _25 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Notebook Generation` be split into smaller, more focused modules?**
  _Cohesion score 0.13055365686944634 - nodes in this community are weakly interconnected._
- **Why does `Eval set M2 (21 casos, data/eval_set_m2.jsonl)` connect `Evaluation Scoreboard` to `eval_set_fuente.py`, `M1 Architecture Comparison`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Should `AgenteTutor` be split into smaller, more focused modules?**
  _Cohesion score 0.14666666666666667 - nodes in this community are weakly interconnected._
- **Why does `scripts/nb_comun_eval.py (eval set y harness compartidos)` connect `M1 Architecture Comparison` to `Evaluation Scoreboard`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Should `M1 Architecture Comparison` be split into smaller, more focused modules?**
  _Cohesion score 0.06948051948051948 - nodes in this community are weakly interconnected._