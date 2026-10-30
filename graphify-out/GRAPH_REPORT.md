# Graph Report - ProyectoIA  (2026-10-30)

## Corpus Check
- 118 files · ~269,976 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 98 file(s) not represented in the graph (top: .csv 50, .ipynb 33, .jsonl 12)

## Summary
- 1503 nodes · 3081 edges · 90 communities (83 shown, 7 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 370 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `547a2db9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Notebook Generation
- AgenteTutor
- verificador.py
- Guía técnica: Tutor de matemáticas · Comparación de arquitecturas
- ablaciones.py
- M2/M3 RAG & Agent Findings
- Evaluation Scoreboard
- re
- test_calculo.py
- trayectorias_fuente.py
- Normalizador
- lectura
- dataset_fuente.py
- registro.py
- test_mejora_nucleo.py
- Hojas para escribir a mano
- nucleo.py
- Módulo M1: Fine-tuning de modelo base para tutoría de matemáticas
- test_m4_vision.py
- M4 · Análisis y conclusiones
- scripts/agente/ (código del agente ReAct)
- Corpus math_tutor_dataset.jsonl (165 ejemplos)
- Banco de errores
- Plantilla de definición del proyecto integrador
- revisar_hoja
- caso
- json
- trayectorias_diagnostico_fuente.py
- exigir_puerta_dev
- herramientas.py
- inventario
- eval_set_fuente.py
- Organización del repositorio y preparación de M5
- test_generador.py
- extraer_diagnostico
- archivo_de
- argparse
- herramienta
- eval_set_m5_vision_fuente.py
- recalcular_criterio.py
- revisar_adaptador
- eval_set_m4_fuente.py
- sys
- Refinamiento del tutor antes de M5
- Tutor
- vision.py
- Corrida 01 · 22-10-2026 · **no es un scorecard de M4**
- Corrida 03 · 24-10-2026 · **evidencia diagnóstica, no un scorecard**
- app.py
- test_ningun_enunciado_sale_del_eval_set_de_m4
- Despliegue · qué ejecutar y dónde
- test_m5.py
- salud.py
- telemetria.py
- argparse
- json
- Enrutador
- RegistroHerramientas
- prompt_revision
- pipeline.py
- ._armar_agente
- Hojas manuscritas para la evaluación visual de M5
- integral_definida
- Tutor de matemáticas · SI4006
- Estado
- enrutador
- Sugerencia por sugerencia, contra el código actual
- M5 · Despliegue e instrumentación
- 3 · Cambios realizados y por qué
- Guion de la demo · 15 minutos
- a_fraccion
- REFINAMIENTO_TUTOR_M5.md
- Fallos del tutor · análisis exploratorio
- Retroalimentación de M4 · qué se incorporó y qué sigue pendiente
- Los experimentos
- enrutador.py
- 3 · Integración de M4 y estado de las ramas
- _degradar
- _experimentos
- test_un_enrutador_que_no_arranca_se_informa_aunque_el_retrieval_este
- prompt_revision_confirmada
- .__call__
- 1 · Resultados de M1–M4 y sus fuentes
- 2 · Análisis de errores · exploratorio
- Arquitectura real de M5
- mcnemar_exacto
- test_el_notebook_de_m4_fija_las_versiones

## God Nodes (most connected - your core abstractions)
1. `code()` - 50 edges
2. `md()` - 49 edges
3. `Guía técnica: Tutor de matemáticas · Comparación de arquitecturas` - 33 edges
4. `celda_drive()` - 31 edges
5. `encabezado()` - 30 edges
6. `lectura()` - 29 edges
7. `ejecutar()` - 27 edges
8. `Tutor` - 24 edges
9. `revisar_hoja()` - 24 edges
10. `AgenteTutor` - 23 edges

## Surprising Connections (you probably didn't know these)
- `La salud del servicio dice lo que hay` --references--> `salud()`  [INFERRED]
  docs/REFINAMIENTO_TUTOR_M5.md → M5_despliegue/app/salud.py
- `3 · Tres casos en vivo · 4 min` --references--> `operar_fracciones()`  [INFERRED]
  M5_despliegue/docs/GUION_DEMO.md → scripts/agente/herramientas.py
- `Las variantes experimentales, que NO son resultados` --references--> `Normalizador`  [INFERRED]
  docs/REFINAMIENTO_TUTOR_M5.md → scripts/agente/nucleo.py
- `Resultado de la prueba local` --references--> `pista_para_el_tutor()`  [INFERRED]
  m4/README.md → scripts/agente/verificador_aritmetico.py
- `5 · Reproducir` --references--> `PuertaNoSuperada`  [INFERRED]
  m4/README.md → scripts/agente/vision.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Flujo S10: herramientas → fine-tuning agente → ReAct → RAGAS** — guia_tecnica_s10_lab_herramientas, guia_tecnica_s10_finetuning_agente, guia_tecnica_s10_lab_agente_react, guia_tecnica_s10_evaluacion_ragas [EXTRACTED 1.00]
- **Cuatro arquitecturas comparadas en M1** — m1_arquitecturas_documentacion_m1_documentacion_qwen2_5_lora, m1_arquitecturas_documentacion_m1_documentacion_beto_clasificador, m1_arquitecturas_documentacion_m1_documentacion_pipeline_beto_qwen, m1_arquitecturas_documentacion_m1_documentacion_flan_t5_lora [EXTRACTED 1.00]
- **Harness M2 de tres dimensiones** — m2_harness_documentacion_scoreboard_dimension_1_metrica_clasica, m2_harness_documentacion_scoreboard_dimension_2_llm_judge, m2_harness_documentacion_scoreboard_dimension_3_aciertos_dominio, m2_harness_documentacion_scoreboard_eval_set_m2 [EXTRACTED 1.00]
- **Pipeline agéntico S10 (normalizador, ReAct, registro, verificador)** — m3_rag_agentes_entrega_documentacion_m3_agente_documentacion_normalizador, m3_rag_agentes_entrega_documentacion_m3_agente_documentacion_bucle_react, m3_rag_agentes_entrega_documentacion_m3_agente_documentacion_tool_registry, m3_rag_agentes_entrega_documentacion_m3_agente_documentacion_verificador, m3_rag_agentes_entrega_documentacion_m3_agente_documentacion_tool_calling_nativo [EXTRACTED 1.00]
- **Harness de tres dimensiones del scoreboard** — m3_rag_agentes_entrega_documentacion_scoreboard_dimension1_metrica_clasica, m3_rag_agentes_entrega_documentacion_scoreboard_dimension2_llm_judge, m3_rag_agentes_entrega_documentacion_scoreboard_dimension3_aciertos_dominio [EXTRACTED 1.00]
- **Técnicas incrementales de RAG avanzado S08** — m3_rag_agentes_entrega_documentacion_m2_m3_documentacion_hybrid_search, m3_rag_agentes_entrega_documentacion_m2_m3_documentacion_reranker, m3_rag_agentes_entrega_documentacion_m2_m3_documentacion_enrutador, m3_rag_agentes_entrega_documentacion_m2_m3_documentacion_cita_textual [EXTRACTED 1.00]

## Communities (90 total, 7 thin omitted)

### Community 0 - "Notebook Generation"
Cohesion: 0.12
Nodes (41): main(), validar_celdas(), construir(), construir(), construir(), celda_eval_m3(), celda_trayectorias(), celdas_modulos() (+33 more)

### Community 1 - "AgenteTutor"
Cohesion: 0.15
Nodes (21): AgenteTutor, analizar_salida(), SalidaModelo, Guion, test_anadir_herramienta_sin_tocar_el_nucleo(), test_buscar_documentos_alimenta_los_contextos(), test_contexto_fijo_y_normalizador(), test_error_de_dominio_llega_al_modelo() (+13 more)

### Community 2 - "verificador.py"
Cohesion: 0.13
Nodes (12): _cerca(), derivables(), _esta(), _fmt(), _frac(), _frac_de(), Hallazgo, _numeros() (+4 more)

### Community 3 - "Guía técnica: Tutor de matemáticas · Comparación de arquitecturas"
Cohesion: 0.17
Nodes (19): Guía técnica: Tutor de matemáticas · Comparación de arquitecturas, Tokenizador FLAN-T5 no cubre el español (<unk>), scripts/generar_notebooks.py (ensambla notebooks), M3_Agente_Documentacion.md, Métricas de generación (exactitud, formato_valido, rouge_l, long_media), scripts/nb_comun_eval.py (eval set y harness compartidos), scripts/nb_comun.py (celdas compartidas), scripts/nb_comun_rag.py (corpus, chunking y prompts RAG) (+11 more)

### Community 4 - "ablaciones.py"
Cohesion: 0.17
Nodes (6): cargar_harness(), correr(), imprimir_plan(), main(), revisar_controles(), main()

### Community 5 - "M2/M3 RAG & Agent Findings"
Cohesion: 0.06
Nodes (47): M2 y M3 — Harness de evaluación y RAG (documentación), Arquitectura recomendada: Qwen+LoRA con RAG condicional por enrutador, Artefacto de medición en adv-02 (claves 'no definido'), Cero abstenciones honestas en M2, Sistema E: prompt con cita textual, Enrutador consulta/problema (regresión logística de 3 señales, 97% LOO), Eval set M2 (21 casos: cálculo, conocimiento, adversarial), Fallo de generación en con-02 (no de retrieval) (+39 more)

### Community 6 - "Evaluation Scoreboard"
Cohesion: 0.14
Nodes (19): Scoreboard del tutor de matemáticas (M2 harness / M3 RAG), Señales acierto / abstuvo / alucinó, Bloque A · Cálculo (12 casos, taxonomía de dificultad), Bloque B · Conocimiento institucional (4 casos), Bloque C · Adversarial (5 casos), comparar_robusto (mitigación sesgo de posición), Dimensión 1 · Exactitud en cálculo y similitud por embeddings, Dimensión 2 · LLM-as-a-judge (Qwen pointwise, rúbrica 1-5) (+11 more)

### Community 7 - "re"
Cohesion: 0.14
Nodes (18): parece_informal(), afirmaciones(), answer_relevancy(), average_precision(), context_precision(), context_precision_oro(), context_recall(), context_recall_oro() (+10 more)

### Community 8 - "test_calculo.py"
Cohesion: 0.06
Nodes (32): _agente(), ejecutar(), test_al_indicar_la_variable_la_ambiguedad_desaparece(), test_de_punta_a_punta_deriva_un_polinomio_en_espanol(), test_de_punta_a_punta_integral_definida_en_espanol(), test_de_punta_a_punta_la_ambiguedad_llega_al_modelo_como_observacion(), test_de_punta_a_punta_una_integral_divergente_no_produce_numero(), test_demasiadas_operaciones_se_rechazan() (+24 more)

### Community 9 - "trayectorias_fuente.py"
Cohesion: 0.16
Nodes (14): construir(), _ejecutar(), _jaccard(), _llamada_para(), llamadas_de_paso(), main(), _msg_asistente_llamada(), _msg_herramienta() (+6 more)

### Community 10 - "Normalizador"
Cohesion: 0.24
Nodes (6): ConsultaNormalizada, Normalizador, test_reescritura_que_cambia_numeros_se_descarta(), test_reescritura_solo_si_parece_informal(), test_reescritura_valida_se_acepta(), test_x_como_incognita_no_se_toca()

### Community 11 - "lectura"
Cohesion: 0.07
Nodes (28): lectura_fiel(), lectura_no_fiable(), pasos_como_texto(), pasos_numerados(), lectura(), test_adversarial_no_aplica(), test_con_pasos_la_respuesta_final_si_se_numera(), test_dice_que_la_hoja_es_ilegible_sin_inventar_pasos() (+20 more)

### Community 12 - "dataset_fuente.py"
Cohesion: 0.16
Nodes (9): a_numero(), a_registro_final(), asignar_splits(), construir(), construir_salida(), main(), normalizar_texto(), reporte() (+1 more)

### Community 13 - "registro.py"
Cohesion: 0.11
Nodes (7): formatear_pasajes(), registro_con_documentos(), buscar_documentos(), _esquema_tipo(), Herramienta, _parsear_docstring(), decorar()

### Community 15 - "test_mejora_nucleo.py"
Cohesion: 0.10
Nodes (14): obs(), test_acepta_partes_de_una_fraccion_del_enunciado(), test_acepta_porcentaje_derivado(), test_acepta_redondeo_contextual(), test_conocidos_extra_silencia_el_few_shot(), test_detecta_argumento_inventado(), test_detecta_operacion_escrita_que_nadie_calculo(), test_detecta_plan_incompleto() (+6 more)

### Community 16 - "Hojas para escribir a mano"
Cohesion: 0.14
Nodes (13): Hojas para escribir a mano, `vis-01.jpg` · vis-01 · tiene un error, `vis-02.jpg` · vis-02 · tiene un error, `vis-03.jpg` · vis-03 · tiene un error, `vis-04.jpg` · vis-04 · tiene un error, `vis-05.jpg` · vis-05 · error compensado, `vis-06.jpg` · vis-06 · error compensado, `vis-07.jpg` · vis-07 · sin error (+5 more)

### Community 17 - "nucleo.py"
Cohesion: 0.13
Nodes (16): autoconsistencia(), _media(), metricas_herramientas_caso(), resumen_agente(), robustez_mal_escrito(), a_numero(), _json_de(), mismo_numero() (+8 more)

### Community 18 - "Módulo M1: Fine-tuning de modelo base para tutoría de matemáticas"
Cohesion: 0.14
Nodes (18): Limitaciones (33 validación, una semilla, sin calidad pedagógica), M2_M3_Documentacion.md, Módulo M1: Fine-tuning de modelo base para tutoría de matemáticas, Artefacto adaptadores/qwen-lora/ (~20 MB), Análisis de tokenizadores (BPE, WordPiece, SentencePiece), Errores de BETO: atajos léxicos, no semántica, Baseline TF-IDF + regresión logística, BETO (bert-base-spanish-wwm-cased) clasificador (+10 more)

### Community 19 - "test_m4_vision.py"
Cohesion: 0.05
Nodes (25): atipicos_por_mad(), elegir_adaptador(), calibracion(), casos(), celdas_pipeline(), evaluacion(), lecturas_reales(), por_id() (+17 more)

### Community 20 - "M4 · Análisis y conclusiones"
Cohesion: 0.05
Nodes (38): Comparación con nueve casos de desarrollo, Conclusión, Corridas con las doce imágenes, Entrenamiento del adaptador de diagnóstico, Evidencia de la entrega, Lo que aprendimos, Límites que deben quedar claros, M4 · Análisis y conclusiones (+30 more)

### Community 21 - "scripts/agente/ (código del agente ReAct)"
Cohesion: 0.40
Nodes (6): S10_Evaluacion_RAGAS_Tutor_Matematicas.ipynb, S10_FineTuning_Agente_Herramientas.ipynb, S10_Lab_Agente_ReAct_Tutor_Matematicas.ipynb, S10_Lab_Herramientas_Tutor_Matematicas.ipynb, scripts/agente/ (código del agente ReAct), Verificación simbólica del cálculo (SymPy)

### Community 22 - "Corpus math_tutor_dataset.jsonl (165 ejemplos)"
Cohesion: 0.40
Nodes (4): scripts/dataset_fuente.py (valida y exporta JSONL), Corpus math_tutor_dataset.jsonl (165 ejemplos), scripts/registros.py (fuente de verdad del corpus), Corpus de 165 ejemplos (11 categorías × 15)

### Community 23 - "Banco de errores"
Cohesion: 0.40
Nodes (4): Banco de errores, El sistema que se va a mejorar · C5-FT, Por sistema, Por tipo de error

### Community 24 - "Plantilla de definición del proyecto integrador"
Cohesion: 0.60
Nodes (5): Plantilla de definición del proyecto integrador, Dominio: tutor de matemáticas para secundaria (9°-11°), Motor de análisis: detectar el paso del error, Usuario 1: Estudiante (qué corregir primero), Usuario 2: Profesor (a quién agrupar para refuerzo)

### Community 25 - "revisar_hoja"
Cohesion: 0.06
Nodes (22): El verificador aritmético, conectado, _fmt(), Hallazgo, igualdades_de(), pista_para_el_tutor(), _primer_numero(), revisar_hoja(), _valor() (+14 more)

### Community 26 - "caso"
Cohesion: 0.13
Nodes (16): calificar_m4(), caso(), test_abstencion_correcta_en_hoja_ilegible(), test_abstencion_correcta_fuera_de_dominio(), test_abstenerse_de_la_forma_equivocada_no_acierta_pero_tampoco_alucina(), test_acierta_aunque_la_explicacion_mencione_otros_pasos(), test_conclusion_en_texto_libre_se_cuenta_aparte(), test_conclusion_equivocada_es_fallo_aunque_la_explicacion_nombre_el_paso_bueno() (+8 more)

### Community 27 - "json"
Cohesion: 0.19
Nodes (3): _construir_agente(), medir(), chunks_de_resultado()

### Community 28 - "trayectorias_diagnostico_fuente.py"
Cohesion: 0.19
Nodes (15): construir(), _correcto(), ejemplo(), _familia_a(), _familia_b(), _familia_c(), _fmt(), H() (+7 more)

### Community 29 - "exigir_puerta_dev"
Cohesion: 0.16
Nodes (11): exigir_puerta_dev(), PuertaNoSuperada, revisar_puerta_dev(), test_cero_de_tres_detiene(), test_la_celda_de_evaluacion_no_se_ejecuta_si_la_puerta_falla(), test_la_celda_de_evaluacion_pasa_la_puerta_cuando_se_supera(), test_la_puerta_no_exige_acertar_solo_el_formato(), test_la_puerta_se_supera_con_tres_conclusiones_validas() (+3 more)

### Community 30 - "herramientas.py"
Cohesion: 0.19
Nodes (10): _decimales_a_racional(), evaluar_expresion(), normalizar_expresion(), parsear(), potencia(), raiz(), resolver_ecuacion(), _sympy_a_salida() (+2 more)

### Community 31 - "inventario"
Cohesion: 0.11
Nodes (13): base_de(), ImagenAmbigua, inventario(), test_archivos_que_no_son_imagenes_se_ignoran(), test_base_de(), test_el_archivo_compartido_apunta_a_los_dos_casos(), test_el_eval_set_pide_once_archivos_para_doce_casos(), test_imagen_ambigua_se_detiene_en_vez_de_elegir() (+5 more)

### Community 32 - "eval_set_fuente.py"
Cohesion: 0.21
Nodes (11): a_numero(), main(), normalizar(), reporte(), validar(), construir(), E(), main() (+3 more)

### Community 33 - "Organización del repositorio y preparación de M5"
Cohesion: 0.15
Nodes (12): 1 · Estructura final y criterio de ubicación, 2.1 · Residuo `M4_vision/` — resuelto, 2.2 · Entrega de M3 — conservada y ahora reproducible, 2.3 · Materiales de clase — una sola copia, 2.4 · Notebooks de M4 en Descargas — no se copiaron, 2 · Registro de duplicados y movimientos, 5 · Cambios realizados, 6 · Verificaciones locales y limitaciones (+4 more)

### Community 34 - "test_generador.py"
Cohesion: 0.20
Nodes (11): revisar_destinos(), tiene_salidas(), _nb(), test_el_caso_real_que_costo_la_evidencia(), test_el_execution_count_solo_tambien_protege(), test_revisar_destinos_devuelve_solo_los_protegidos(), test_sin_destinos_protegidos_la_lista_esta_vacia(), test_un_archivo_ilegible_no_se_declara_protegido() (+3 more)

### Community 35 - "extraer_diagnostico"
Cohesion: 0.14
Nodes (11): asegurar_diagnostico(), diagnostico_valido(), extraer_diagnostico(), test_cada_ejemplo_termina_en_una_conclusion_valida(), test_diagnostico_valido(), test_el_reintento_conserva_la_explicacion_y_añade_la_linea(), test_el_reintento_recibe_la_propia_explicacion_del_modelo(), test_el_reintento_se_pide_una_sola_vez() (+3 more)

### Community 36 - "archivo_de"
Cohesion: 0.14
Nodes (10): archivo_de(), json_de(), test_alias_resuelve_vis12_a_la_foto_de_vis07(), test_devuelve_none_en_vez_de_lanzar(), test_el_paso_que_hay_que_senalar_existe_en_lo_que_el_tutor_lee(), test_extraer_diagnostico_acepta_sinonimos_razonables(), test_la_numeracion_real_coincide_con_la_del_eval_set(), test_las_tres_extensiones_valen_en_cualquier_caja() (+2 more)

### Community 37 - "argparse"
Cohesion: 0.18
Nodes (9): cargar(), clasificar_fallo(), main(), analizar_m3(), analizar_m4(), _aritmetica_falsa(), etiquetar_m3(), main() (+1 more)

### Community 38 - "herramienta"
Cohesion: 0.10
Nodes (14): Conclusión, aplicar_porcentaje(), _decimal(), dividir(), _es_decimal_exacto(), formatear(), multiplicar(), operar_expresion() (+6 more)

### Community 39 - "eval_set_m5_vision_fuente.py"
Cohesion: 0.18
Nodes (10): C(), csv_entrega(), _evaluar_check(), ev(), hojas(), main(), _mismo(), normalizar() (+2 more)

### Community 40 - "recalcular_criterio.py"
Cohesion: 0.23
Nodes (7): _b(), escribir(), main(), reevaluar(), scorecard_s08(), scorecard_s10(), valor()

### Community 41 - "revisar_adaptador"
Cohesion: 0.21
Nodes (10): revisar_adaptador(), _adaptador(), test_acepta_el_formato_antiguo_bin(), test_adaptador_completo_pasa(), test_directorio_inexistente_lo_dice(), test_directorio_vacio(), test_el_motivo_nunca_afirma_una_ausencia_que_no_comprobo(), test_falta_el_config_y_el_motivo_lo_nombra() (+2 more)

### Community 42 - "eval_set_m4_fuente.py"
Cohesion: 0.23
Nodes (7): C(), csv_entrega(), _evaluar_check(), hojas(), main(), paso(), validar()

### Community 43 - "sys"
Cohesion: 0.33
Nodes (4): main(), reconstruir(), sha(), verificar()

### Community 44 - "Refinamiento del tutor antes de M5"
Cohesion: 0.20
Nodes (10): 4 · ¿Hace falta entrenar? · evidencia, 5 · Lo que se sirve hoy y lo que falta medir, 6 · Protocolo para evaluar respuestas reales del Space, 8 · Calendario, Las variantes experimentales, que NO son resultados, Lo que no se va a hacer todavía, Lo que se sirve por omisión: S08-D, sin añadidos, Qué cambió en la primera revisión del 30-10 (+2 more)

### Community 45 - "Tutor"
Cohesion: 0.18
Nodes (3): 9 · Preguntas de defensa, con evidencia, Tutor, transcribir()

### Community 46 - "vision.py"
Cohesion: 0.08
Nodes (17): diagnostico_esperado(), fuente_del_fallo(), huella_lectura(), normalizar_paso(), lecturas_consistentes(), _norm(), numeros_de(), pasos_mencionados() (+9 more)

### Community 47 - "Corrida 01 · 22-10-2026 · **no es un scorecard de M4**"
Cohesion: 0.33
Nodes (5): Corrida 01 · 22-10-2026 · **no es un scorecard de M4**, Las cuatro causas, todas de medición, Lo que sí dice esta corrida, y hay que conservar, Qué contiene, Qué se cambió después

### Community 48 - "Corrida 03 · 24-10-2026 · **evidencia diagnóstica, no un scorecard**"
Cohesion: 0.33
Nodes (5): Corrida 03 · 24-10-2026 · **evidencia diagnóstica, no un scorecard**, La regresión de M2, aparte, Lo que esta corrida arregló, Los cuatro problemas, Qué sí deja

### Community 49 - "app.py"
Cohesion: 0.20
Nodes (9): _bandera_activa(), _cabecera(), construir(), _formatear(), lanzar(), leer_foto(), responder_texto(), revisar_foto() (+1 more)

### Community 50 - "test_ningun_enunciado_sale_del_eval_set_de_m4"
Cohesion: 0.67
Nodes (3): test_ningun_enunciado_sale_del_eval_set_de_m4(), jac(), norm()

### Community 51 - "Despliegue · qué ejecutar y dónde"
Cohesion: 0.20
Nodes (9): 1 · Local, sin GPU, 2 · En Colab, con GPU, 3 · Hugging Face Space, 4 · W&B, 5 · Health check, Despliegue · qué ejecutar y dónde, Modo degradado · qué ve el usuario si falta una pieza, `TUTOR_CONFIG` · qué se sirve y qué lo respalda (+1 more)

### Community 52 - "test_m5.py"
Cohesion: 0.05
Nodes (18): test_cuando_las_cuentas_cuadran_se_le_dice_al_tutor(), test_el_adaptador_lo_pide_la_configuracion_no_el_disco(), test_el_adaptador_que_falta_en_el_disco_marca_la_ejecucion_degradada(), test_el_eval_set_fijo_de_m2_no_se_toca(), test_el_eval_set_visual_de_m4_sigue_intacto(), test_el_eval_set_visual_de_m5_no_reutiliza_enunciados(), test_el_tutor_arranca_sin_modelo_y_lo_dice(), test_el_verificador_determinista_corre_antes_del_tutor() (+10 more)

### Community 53 - "salud.py"
Cohesion: 0.16
Nodes (9): _resultado(), salud(), texto_salud(), _version(), test_la_salud_con_tutor_expone_lo_necesario(), test_la_salud_degradada_no_atribuye_la_cifra_a_esta_ejecucion(), test_la_salud_distingue_lo_declarado_de_lo_verificado(), test_la_salud_responde_sin_cargar_el_modelo() (+1 more)

### Community 54 - "telemetria.py"
Cohesion: 0.20
Nodes (5): _iniciar_wandb(), registrar(), test_el_texto_se_registra_solo_si_se_activa(), test_no_se_registra_el_texto_del_usuario_por_defecto(), test_wandb_apagado_no_rompe_nada()

### Community 55 - "argparse"
Cohesion: 0.16
Nodes (10): main(), medir_condicion(), memoria_gpu_gb(), plan(), correr_e2(), estado_requisitos(), informe(), latencias() (+2 more)

### Community 56 - "json"
Cohesion: 0.22
Nodes (9): clasificar(), construir(), esperado_de(), main(), resumen_md(), _degradar(), _dibujar(), _hojas() (+1 more)

### Community 57 - "Enrutador"
Cohesion: 0.22
Nodes (4): Configuraciones separadas y declaradas, contar_numeros(), Decision, Enrutador

### Community 58 - "RegistroHerramientas"
Cohesion: 0.12
Nodes (8): _a_llamada(), LlamadaHerramienta, RegistroHerramientas, fin(), ResultadoHerramienta, test_errores_se_devuelven_como_observacion(), test_herramientas_resuelven_los_casos_del_eval_set(), test_reglas_del_normalizador()

### Community 59 - "prompt_revision"
Cohesion: 0.29
Nodes (4): prompt_revision(), test_el_prompt_no_trae_ningun_ejercicio_resuelto(), test_el_prompt_prohibe_la_etiqueta_que_descarrilaba_al_modelo(), test_la_exigencia_del_formato_va_al_final_del_prompt()

### Community 60 - "pipeline.py"
Cohesion: 0.12
Nodes (9): Ablaciones · un cambio por vez, criterios fijados ANTES de medir, Configuracion, obtener(), cargar_corpus(), test_cada_fila_de_ablacion_cambia_una_sola_pieza(), test_la_configuracion_de_producto_es_experimental_y_no_cita_cifras_de_m4(), test_la_configuracion_por_defecto_es_la_medida(), test_los_controles_de_la_ablacion_tienen_cifra_medida() (+1 more)

### Community 61 - "._armar_agente"
Cohesion: 0.20
Nodes (10): buscador_hibrido(), buscar(), candidatos(), puntajes(), rerank_max(), tokenizar(), buscador_lexico(), buscar() (+2 more)

### Community 62 - "Hojas manuscritas para la evaluación visual de M5"
Cohesion: 0.14
Nodes (13): Hojas manuscritas para la evaluación visual de M5, `m5v-01` · localizacion · tiene un error, `m5v-02` · localizacion · tiene un error, `m5v-03` · localizacion · tiene un error, `m5v-04` · localizacion · tiene un error, `m5v-05` · compensado · paso mal, resultado final BIEN, `m5v-06` · compensado · paso mal, resultado final BIEN, `m5v-07` · sin_error · está CORRECTO, no inventes un error (+5 more)

### Community 63 - "integral_definida"
Cohesion: 0.24
Nodes (8): 7 · Pendientes · qué falta y de quién depende, Cobertura temática · antes y ahora, separados, Tres herramientas de cálculo · implementadas y probadas, derivar(), integral_definida(), integrar(), _tratable(), _variable_de()

### Community 64 - "Tutor de matemáticas · SI4006"
Cohesion: 0.50
Nodes (3): Health check, Tutor de matemáticas · SI4006, Variables de entorno

### Community 67 - "enrutador"
Cohesion: 0.20
Nodes (8): 4 · M5 · implementado, sin medir, La decisión que ordena todo, encode_falso(), enrutador(), rerank_falso(), test_funciona_sin_reranker(), trazas(), v()

### Community 68 - "Sugerencia por sugerencia, contra el código actual"
Cohesion: 0.22
Nodes (8): 1 · Fijar la versión de `transformers` · **incorporado**, 2 · La caché de modelos · **incorporado**, 3 · Que `ver()` conserve los errores manuscritos · **ya estaba, no se toca**, 4 · El holdout visual ya gastado · **corregido en la documentación**, 5 · Hojas nuevas para evaluar · **preparado, pendiente de fotografiar**, 6 · El verificador aritmético en el flujo · **incorporado, sin medir**, Sugerencia por sugerencia, contra el código actual, test_los_doce_casos_de_m4_no_se_describen_como_holdout_ciego()

### Community 69 - "M5 · Despliegue e instrumentación"
Cohesion: 0.22
Nodes (9): Estructura, La configuración se elige, no se adivina, Lo que la interfaz dice de sí misma, M5 · Despliegue e instrumentación, Modo degradado · la cifra se apaga cuando falta una pieza, Montaje por capas, Privacidad, Qué falta medir (+1 more)

### Community 70 - "3 · Cambios realizados y por qué"
Cohesion: 0.25
Nodes (6): 3 · Cambios realizados y por qué, El flujo de revisión de una foto · dos pasos, Herramientas de análisis añadidas, La salud del servicio dice lo que hay, Modo degradado · lo que faltaba en la primera revisión, Y se comprueba, no se afirma

### Community 72 - "Guion de la demo · 15 minutos"
Cohesion: 0.25
Nodes (7): 1 · Problema y usuario · 2 min, 2 · El sistema, de M1 a M5 · 3 min, 3 · Tres casos en vivo · 4 min, 4 · Scorecard y eficiencia · 3 min, 5 · Un fallo conocido, límites y ética · 2 min, 6 · Evidencia · 1 min, Guion de la demo · 15 minutos

### Community 73 - "a_fraccion"
Cohesion: 0.33
Nodes (3): operar_fracciones(), a_fraccion(), _convertir()

### Community 74 - "REFINAMIENTO_TUTOR_M5.md"
Cohesion: 0.29
Nodes (3): Estado de los requisitos, Lo que falta ejecutar, Validación de M5 · cinco experimentos que no se mezclan

### Community 75 - "Fallos del tutor · análisis exploratorio"
Cohesion: 0.29
Nodes (6): Fallos del tutor · análisis exploratorio, M3 · 53 casos por configuración · etiqueta principal, M4 · 12 casos visuales, Por qué `calculo_sin_herramienta` no se puede comparar entre configuraciones, Qué significa cada etiqueta, Señales secundarias (no suman fallos)

### Community 76 - "Retroalimentación de M4 · qué se incorporó y qué sigue pendiente"
Cohesion: 0.29
Nodes (7): Advertencia sobre las fuentes, Experimento A · ¿se reproducen las 11 transcripciones con las versiones fijadas?, Experimento B · los controles textuales de S08-D, Experimento C · la vía visual con hojas nuevas, Los hechos de partida, Pendiente de validar en Colab · celdas, archivos y criterios, Retroalimentación de M4 · qué se incorporó y qué sigue pendiente

### Community 77 - "Los experimentos"
Cohesion: 0.29
Nodes (7): E1 · Texto · baseline S08-D y sus controles, E2 · Recuperación y enrutamiento, E3 · Herramientas matemáticas, E4 · Lectura de fotos y salvaguarda de ilegibles, E5 · Flujo completo de revisión de foto, E6 · Muestra de la app desplegada, Los experimentos

### Community 78 - "enrutador.py"
Cohesion: 0.29
Nodes (3): contexto_enrutado(), contexto(), test_contexto_enrutado_solo_busca_cuando_toca()

### Community 79 - "3 · Integración de M4 y estado de las ramas"
Cohesion: 0.33
Nodes (6): 3.1 · Lo que contiene cada rama, 3.2 · Commits locales creados, 3.3 · Cómo se integró, 3.4 · Qué llegó, verificado archivo a archivo, 3.5 · Riesgos de las ramas, 3 · Integración de M4 y estado de las ramas

### Community 80 - "_degradar"
Cohesion: 0.33
Nodes (4): _degradar(), test_la_cabecera_de_la_interfaz_marca_el_modo_degradado(), test_sin_recuperador_hibrido_el_sistema_queda_degradado(), test_sin_reranker_tampoco_hay_enrutador_y_los_dos_se_informan()

### Community 81 - "_experimentos"
Cohesion: 0.33
Nodes (4): _experimentos(), test_cada_experimento_declara_que_no_mide(), test_la_reproduccion_del_retrieval_no_se_presenta_como_exactitud(), test_las_cinco_capas_estan_separadas()

### Community 82 - "test_un_enrutador_que_no_arranca_se_informa_aunque_el_retrieval_este"
Cohesion: 0.33
Nodes (5): test_el_buscador_lexico_encuentra_lo_evidente(), test_el_buscador_no_lanza_con_consulta_vacia(), test_un_enrutador_que_no_arranca_se_informa_aunque_el_retrieval_este(), hibrido_sin_enrutador(), buscar()

### Community 83 - "prompt_revision_confirmada"
Cohesion: 0.33
Nodes (4): test_el_prompt_sin_evidencia_no_menciona_comprobacion(), test_la_evidencia_se_pasa_como_observacion_no_como_veredicto(), prompt_revision_confirmada(), texto_de_pasos()

### Community 85 - "1 · Resultados de M1–M4 y sus fuentes"
Cohesion: 0.50
Nodes (4): 1 · Resultados de M1–M4 y sus fuentes, Auditoría de cada cifra · archivo, fila y columna, El hallazgo que decide qué se despliega, Limitaciones de la comparación

### Community 86 - "2 · Análisis de errores · exploratorio"
Cohesion: 0.50
Nodes (3): 2 · Análisis de errores · exploratorio, La corrección que invalida la conclusión anterior, M4 · 12 casos visuales

### Community 87 - "Arquitectura real de M5"
Cohesion: 0.50
Nodes (3): Arquitectura real de M5, Decisiones que el diagrama hace explícitas, Despliegue previsto

## Knowledge Gaps
- **25 isolated node(s):** `Análisis de tokenizadores (BPE, WordPiece, SentencePiece)`, `Dominio: tutor de matemáticas para secundaria (9°-11°)`, `Métricas de generación (exactitud, formato_valido, rouge_l, long_media)`, `Conflicto peft + torchao (ImportError)`, `Weights & Biases (proyecto tutor-matematicas-arquitecturas)` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 577 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Refinamiento del tutor antes de M5` connect `Refinamiento del tutor antes de M5` to `3 · Cambios realizados y por qué`, `REFINAMIENTO_TUTOR_M5.md`, `Tutor`, `1 · Resultados de M1–M4 y sus fuentes`, `2 · Análisis de errores · exploratorio`, `integral_definida`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **What connects `Análisis de tokenizadores (BPE, WordPiece, SentencePiece)`, `Dominio: tutor de matemáticas para secundaria (9°-11°)`, `Métricas de generación (exactitud, formato_valido, rouge_l, long_media)` to the rest of the system?**
  _25 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Notebook Generation` be split into smaller, more focused modules?**
  _Cohesion score 0.11868484362469928 - nodes in this community are weakly interconnected._
- **Why does `Scoreboard del tutor de matemáticas (M2 harness / M3 RAG)` connect `Evaluation Scoreboard` to `eval_set_fuente.py`, `Módulo M1: Fine-tuning de modelo base para tutoría de matemáticas`, `Guía técnica: Tutor de matemáticas · Comparación de arquitecturas`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Should `AgenteTutor` be split into smaller, more focused modules?**
  _Cohesion score 0.1471264367816092 - nodes in this community are weakly interconnected._
- **Why does `Eval set M2 (21 casos, data/eval_set_m2.jsonl)` connect `Evaluation Scoreboard` to `eval_set_fuente.py`, `Módulo M1: Fine-tuning de modelo base para tutoría de matemáticas`, `Guía técnica: Tutor de matemáticas · Comparación de arquitecturas`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Should `verificador.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12923076923076923 - nodes in this community are weakly interconnected._