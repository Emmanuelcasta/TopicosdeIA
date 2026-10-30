# Plan Demo Day: el tutor que encuentra el paso del error en la foto de tu hoja

Revisión de la rama `m4-vision` en `9c0c0df` (25-oct-2026). Estas son sugerencias para preparar la demo del sistema completo (M1–M4) como lanzamiento de producto. Ustedes deciden cómo y quién las asume. Aún no han desplegado: lo relativo a app, URL, health check y eficiencia va como sugerencia «al desplegar».

## Lo que ya tienen para brillar

- **Un usuario, un dolor y un argumento propio.**
  - El usuario es un estudiante de 9.° a 11.° que resuelve en casa y quiere saber *qué paso* corregir.
  - Si copia su hoja en un chat, corrige el error sin darse cuenta; por eso la foto es la entrada natural.
- **El lector de visión funciona.**
  - Su `ver()` reproduce en otro hardware las mismas 11 transcripciones, con los errores manuscritos copiados.
  - «24 x 13 = 302» se lee 302, no 312.
  - Tarda unos 3 s por foto.
- **El recorrido M1–M4 ya está conectado en código.**

  | Etapa | Dónde está |
  |---|---|
  | Foto (M4) | `ver()`, `leer_dos_veces()` |
  | Agente tutor con herramientas (M3) | `AgenteTutor` |
  | Recuperación (M2) | `CODIGO_INDICE` |
  | Tutor afinado (M1) | Adaptadores |
  | Integración | `sistema_multimodal()` |

- **Un verificador aritmético determinista** que encuentra la cuenta mal hecha en 5 de los 6 casos con error. Es su mejor «prueba visible».
- **Una salvaguarda bien pensada.** Corregida la caché, separa la foto borrosa (vis-10) de las 10 buenas.

La rama de CODE_PR ya trae el punto de entrada de una foto (`revisar_foto_demo`), la salvaguarda funcionando y versiones fijadas. Las prioridades se apoyan en ella.

## Prioridades

**P0 — El momento wow: «subo mi hoja y me dice en qué paso me equivoqué», con la cuenta como prueba**
- **Qué podría ocurrir durante la demo:**
  - El producto se ve como una secuencia de celdas.
  - O el tutor «arregla» la hoja en vez de señalar el error.
  - En su corrida con fotos, el tutor reescribió vis-01 como «Paso 2: 24 × 13 = 312» y no emitió `Diagnóstico:`.
- **Evidencia exacta:**
  - `m4/scorecard_m4.csv`: 7 de 12 casos con `sin_conclusion=True` (vis-01 a 06 y vis-09); aciertos solo en vis-07 y vis-08.
  - `CODIGO_LIBERAR` (nb_m4.py L495) borra el VLM antes de cargar el tutor. `sistema_multimodal` recibe una lectura, no una foto.
  - Sin fijar versión, Colab instala la última `transformers`, y `max_pixels=` desaparece en 5.23 (M4-06).
  - La caché vieja dejó la salvaguarda sin decidir (M4-01).
- **Cambio mínimo recomendado:**
  - Usar `revisar_foto_demo(ruta)` (C1) con `MANTENER_VLM = True` y `m4/requirements-m4.txt`. Lee, decide si la lectura es fiable y diagnostica, con tiempos por etapa.
  - Cuando la lectura es fiable, pasar al tutor la pista del verificador (`pista_para_el_tutor`), que ya existe.
  - Mostrar en pantalla «esto es lo que leí» y «tu error está en el paso 2: 24 × 13 da 312, no 302».
  - Elegir 2–3 casos estrella; uno de ellos, una hoja nueva escrita y fotografiada el mismo día.
  - Al desplegar, una interfaz delgada (Gradio) sobre esa función: foto → transcripción → diagnóstico → segundos. Adaptadores en HF Hub en lugar de Drive.
- **Qué NO hace falta rehacer:** el VLM, los prompts de lectura, los adaptadores, el agente de M3 ni el eval set.
- **Cómo verificar el cierre:**
  - En un runtime limpio de T4:
    - `revisar_foto_demo` con vis-01 da «paso 2» citando la cuenta;
    - vis-07 da «sin error»;
    - ambos en dos ensayos seguidos.
  - La celda de entorno muestra transformers 5.20.2 sin `FutureWarning`.
- **Quién lo asume:** lo decide el equipo.

**P1 — Confianza como característica: «¿Leí bien tu hoja?»**
- **Qué podría ocurrir durante la demo:** alguien del público sube una foto movida. El lector inventa una hoja legible y el tutor la diagnostica como si fuera del estudiante.
- **Evidencia exacta:**
  - vis-10 se lee dos veces, igual, como «24 x 13 = 306» (la hoja dice 302), y «¿se lee?» responde sí (M4-02).
  - Con una sola foto, la señal 4 no puede disparar si compara contra el lote.
  - La calibración impresa marca hojas manuscritas buenas como ilegibles.
  - vis-09 «126» se lee «12G» y `lectura_fiel` lo cuenta como fiel (M4-03).
- **Cambio mínimo recomendado:**
  - Escribir y fotografiar 8–10 hojas nítidas a mano (las `dev-*` de `HOJAS_PARA_ESCRIBIR.md` sirven), con el mismo celular de la demo, en `m4/referencia_manuscrita/`. `revisar_foto` las usa como referencia. Sus versiones movidas sirven para comprobar que la señal las marca.
  - Mostrar siempre la transcripción con «¿Leí bien tu hoja?» antes del diagnóstico. Si no es fiable: «No pude leer bien tu hoja; toma otra foto con más luz».
  - En la demo, presentarlo como una característica de seguridad: «así se comporta cuando no está seguro».
- **Qué NO hace falta rehacer:** las 4 señales, el umbral MAD ni el eval set.
- **Cómo verificar el cierre:**
  - Las fotos movidas de referencia se marcan y las nítidas no.
  - vis-10 y una foto movida nueva responden «no pude leer».
  - vis-01 sigue llegando al tutor con «302».
  - El scorecard trae `numeros_fieles` (C2).
- **Quién lo asume:** lo decide el equipo.

**P2 — Tres números de valor y una demo que no se cae**
- **Qué podría ocurrir durante la demo:**
  - Un número difícil de defender, como «M4 es más rápido que M3» comparando tareas distintas.
  - O una falla de red o GPU sin plan B.
- **Evidencia exacta:**
  - `delta_m3_m4.csv` compara 14,1 s (texto) con 13,0 s (foto).
  - El peor caso es `fallos.iloc[0]` (nb_m4.py L936), no uno analizado.
  - El README tiene unas 2 600 palabras (M4-04).
  - No hay una sección de procedencia y uso de las fotos (M4-05).
- **Cambio mínimo recomendado:**
  - 1–3 números con su denominador:
    - aciertos con foto de la configuración que mostrarán, sobre 12 (post-hoc, dicho así);
    - regresión sin foto, 15/21 frente a 13/21;
    - «leer la foto cuesta unos 3–4 s de los ~13 s».
  - vis-10 como peor caso contado: «antes lo diagnosticaba; ahora pide otra foto».
  - README de una página con 4–5 líneas de datos y uso responsable: consentimiento, menores, no guardar fotos.
  - Robustez: tag de la versión ensayada, los mismos casos y un plan B rotulado («grabación del 4-nov» o salida cacheada).
  - Al desplegar, conviene:
    - health check y logs JSON sin datos personales;
    - W&B con latencias como «así lo operamos»;
    - medir la técnica de eficiencia elegida con una tabla antes/después.
- **Qué NO hace falta rehacer:** el harness, el eval set ni las corridas ya archivadas.
- **Cómo verificar el cierre:** un README de una página con los 3 números y sus denominadores, el peor caso, la sección de uso responsable y un tag de la versión de la demo.
- **Quién lo asume:** lo decide el equipo.

## Guion sugerido (bloques, sin minutos obligatorios)

1. **Usuario y dolor.** «Laura, de 10.°, hace la tarea en casa. Su resultado no da y no sabe en qué paso se equivocó. Si copia su hoja en un chat, corrige el error sin darse cuenta.»
2. **Momento wow.**
   - Se toma en vivo la foto de una hoja escrita a mano.
   - En pantalla aparece «Esto es lo que leí» y, en segundos, «Tu error está en el paso 2: 24 × 13 da 312, no 302».
   - Se muestra el tiempo de respuesta.
3. **Qué hay detrás,** en una lámina: lector de fotos (M4) → agente tutor con herramientas (M3) → su material (M2) → tutor afinado (M1) → verificador aritmético.
4. **Confianza.** Foto movida → «No pude leer bien tu hoja; toma otra foto». Es una característica, no un error.
5. **Prueba de valor.** Los 3 números de P2, con denominador.
6. **Así lo operamos** (si ya desplegaron): URL, health check, W&B.
7. **Límites y siguiente paso.**
   - No revisa el planteamiento (vis-02).
   - Pocas hojas y de pocas manos.
   - Siguiente paso: más hojas reales, transcripción a LaTeX y modo docente.

## Backlog breve

- Etiqueta de los adversariales como columna del CSV, no por ID en el código.
- Retrieval de M2 igual al de S10 (híbrido con reranker) antes de comparar regresión.
- Lector alternativo o transcripción a LaTeX (M4-07, opcional).
- Llevar M4 a `main` cuando esté estable.
