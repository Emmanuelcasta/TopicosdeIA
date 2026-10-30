"""Interfaz Gradio del tutor de matemáticas · demo de M5.

Qué NO es. El notebook de S15 trae una plantilla didáctica con casos de café, un
evaluador que devuelve `NaN` y una respuesta `[SUSTITUTO]`. Nada de eso se copió:
la interfaz llama al tutor real a través de `pipeline.py`, que a su vez reutiliza
`scripts/agente/`.

Qué dice y qué no dice. El tutor de este proyecto **no resuelve correctamente**
todos los ejercicios, y la vía visual está sin medir con imágenes nuevas: sobre 9
casos de desarrollo acierta 3, y los 12 casos visuales de M4 **ya se gastaron** en
tres corridas exploratorias (4/12, 1/12 y 2/12), así que son evidencia histórica y
conjunto de regresión, no una evaluación ciega. La interfaz lo dice en pantalla en
vez de aparentar seguridad. No emite diagnósticos médicos ni de ninguna otra
clase: revisa procedimientos de matemáticas escolares.

Ejecución local:
    python M5_despliegue/app/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import obtener_tutor          # noqa: E402
from telemetria import registrar              # noqa: E402
from salud import texto_salud               # noqa: E402

AVISO = """
**Qué es esto.** Un tutor de matemáticas escolares que explica procedimientos.

**Qué no es.** No da diagnósticos médicos ni consejos profesionales de ningún
tipo. No sustituye a un docente.

**Fiabilidad.** Si una respuesta no te cuadra, no la des por buena.
"""

AVISO_VISION = """

**Esta vía NO está medida.** Sobre 9 casos de desarrollo acierta 3, con otra
configuración. Los 12 casos visuales de M4 ya se gastaron en tres corridas
exploratorias (4/12, 1/12, 2/12), así que son regresión y no una evaluación
ciega: para afirmar una mejora hacen falta hojas nuevas.

**Por qué se pide confirmar la lectura.** En M4 el modelo de visión inventó una
cuenta —«24 × 13 = 306», dos veces idéntica— y el tutor diagnosticó sobre ese
invento sin dudar. Mientras no veas y confirmes lo que se leyó, un diagnóstico
sería una afirmación sobre una hoja que quizá no existe.
"""


def _cabecera(tutor) -> str:
    """Qué está respondiendo, qué componente funciona y qué cifra aplica.

    Tres estados, y el orden importa. Si falta un componente requerido, lo
    primero que se lee es **degradado** y que el resultado de esta ejecución no
    está medido: la cifra de la configuración completa describe otro sistema. Si
    la configuración es experimental, lo primero es que nadie la ha medido. Solo
    con todas las piezas presentes se enseña la cifra histórica, y aun entonces
    diciendo de dónde viene.
    """
    e = tutor.estado
    piezas = "\n".join(f"- {p}" for p in e.piezas())
    if e.degradado:
        aclaracion = ""
        if e.medido:
            aclaracion = (f"La cifra {e.medido} es de esa configuración con todas "
                          "sus piezas y no describe lo que responde ahora. ")
        return ("🔴 **MODO DEGRADADO · resultado de esta ejecución: NO MEDIDO.**\n\n"
                f"Se pidió `{e.config_nombre}` y falta **{', '.join(e.faltantes)}**. "
                + aclaracion
                + "Las respuestas siguen funcionando, pero no son las del sistema "
                f"que se midió.\n\nQué está funcionando:\n{piezas}")
    if e.experimental:
        return (f"⚠️ **`{e.config_nombre}` · EXPERIMENTAL, sin medir.** Las cifras "
                f"de M1–M4 no respaldan esta combinación.\n\nQué monta:\n{piezas}")
    return (f"**`{e.config_nombre}`** · {e.medido} en el eval set fijo de M2, "
            "medido en S08 (`scorecard_rag_avanzado_v2.csv`). Esa cifra es "
            "**histórica**: describe esta configuración, no una corrida de hoy."
            f"\n\nQué monta:\n{piezas}")


def _formatear(r: dict) -> tuple:
    if not r.get("ok"):
        return f"⚠️ {r['respuesta']}", ""
    detalle = [f"latencia: {r['latencia_s']} s", f"pasos del modelo: {r['pasos_llm']}"]
    if r["herramientas"]:
        detalle.append("herramientas: " + ", ".join(r["herramientas"]))
    if r.get("contextos"):
        detalle.append(f"pasajes consultados: {r['contextos']}")
    if r.get("verificador"):
        v = r["verificador"]
        detalle.append(f"verificador aritmético: paso {v['paso']} ({v['tipo']}) "
                       f"{v['detalle']}" + ("" if v["reintento"] else " · sin reintento"))
    if r.get("transcripcion"):
        detalle.append("\nLo que se leyó en la hoja:\n" + r["transcripcion"])
    return r["respuesta"], "\n".join(detalle)


def responder_texto(pregunta: str):
    tutor = obtener_tutor()
    r = tutor.responder(pregunta)
    registrar("consulta_texto", pregunta, ok=r["ok"], latencia_s=r["latencia_s"],
              pasos_llm=r["pasos_llm"], n_herramientas=len(r["herramientas"]))
    return _formatear(r)


def leer_foto(imagen):
    """Paso 1 de la revisión: enseña lo leído y pide confirmación.

    -> (mensaje, pasos editables, confirmación desmarcada, traza). Nunca un
    diagnóstico: eso es `revisar_foto`, y solo con la casilla marcada.
    """
    tutor = obtener_tutor()
    r = tutor.leer_imagen(imagen)
    registrar("lectura_imagen", "", ok=r["ok"], latencia_s=r["latencia_s"],
              pasos_llm=0, n_herramientas=0)
    if not r.get("ok"):
        return f"⚠️ {r['respuesta']}", "", False, ""
    cuerpo = r["respuesta"] + "\n\n**" + r["pregunta"] + "**"
    if not r.get("confirmable"):
        cuerpo = (f"🟠 **Lectura no fiable.** {r['respuesta']}\n\n"
                  f"**{r['pregunta']}**\n\nMotivo observable: "
                  f"{r['motivo_no_fiable']}")
    traza = [f"latencia de la lectura: {r['latencia_s']} s",
             f"pasos leídos: {len(r['pasos'])}",
             f"se puede diagnosticar: {'sí, tras confirmar' if r.get('confirmable') else 'NO'}"]
    if r.get("transcripcion"):
        traza.append("\nTranscripción cruda:\n" + r["transcripcion"])
    # Los pasos van a un cuadro EDITABLE: corregir la transcripción es parte del
    # flujo, no una excepción. Si el tutor leyó mal un número, el error que
    # señalaría sería suyo.
    return cuerpo, "\n".join(r["pasos"]), False, "\n".join(traza)


def revisar_foto(texto_pasos: str, pregunta: str, confirmado: bool,
                 enunciado: str = ""):
    """Paso 2: diagnostica los pasos confirmados. Sin la casilla, no hay nada."""
    tutor = obtener_tutor()
    pasos = [l.strip() for l in (texto_pasos or "").splitlines() if l.strip()]
    r = tutor.revisar_pasos(pasos, pregunta, enunciado=enunciado or None,
                            confirmado=bool(confirmado))
    registrar("revision_pasos", pregunta, ok=r["ok"], latencia_s=r["latencia_s"],
              pasos_llm=r.get("pasos_llm", 0),
              n_herramientas=len(r.get("herramientas", [])))
    respuesta, detalle = _formatear(r)
    v = r.get("verificador_aritmetico")
    if v:
        detalle = (f"comprobación determinista: {v['detalle']} (tipo {v['tipo']})\n"
                   + detalle)
    return respuesta, detalle


def construir():
    import gradio as gr

    tutor = obtener_tutor()
    with gr.Blocks(title="Tutor de matemáticas · M5") as demo:
        gr.Markdown("# Tutor de matemáticas\n" + AVISO)
        gr.Markdown(_cabecera(tutor))

        with gr.Tab("Pregunta escrita"):
            entrada = gr.Textbox(label="Tu pregunta", lines=3,
                                 placeholder="Un taller compra 3 cajas a 18500 pesos…")
            boton = gr.Button("Responder", variant="primary")
            salida = gr.Textbox(label="Respuesta", lines=8)
            traza = gr.Textbox(label="Cómo se obtuvo", lines=4)
            boton.click(responder_texto, entrada, [salida, traza])
            gr.Examples([["¿Cuánto es 3/4 + 1/8?"],
                         ["¿Cuál es la nota mínima aprobatoria del colegio?"],
                         ["Explica por qué 9 es un número primo."]], entrada)

        # La pestaña de visión solo si la configuración activa la incluye. En
        # S08-D no existe: ofrecerla sugeriría que forma parte de lo que midió
        # 18/21, y la vía visual nunca se midió junto con ese retrieval.
        with gr.Tab("Revisar una foto", visible=tutor.cfg.vision):
            gr.Markdown("### Paso 1 · Subir la foto y revisar lo que se leyó"
                        + AVISO_VISION)
            img = gr.Image(label="Foto de la hoja", type="pil")
            boton_leer = gr.Button("Leer la hoja", variant="primary")
            mensaje = gr.Markdown()

            gr.Markdown("### Paso 2 · Confirmar la lectura\n"
                        "Un paso por línea. **Corrige lo que esté mal transcrito**: "
                        "si el error es de lectura, el diagnóstico sería sobre una "
                        "hoja que no escribiste.")
            enunciado_box = gr.Textbox(label="Enunciado (opcional)", lines=1)
            pasos_box = gr.Textbox(label="Los pasos de tu hoja", lines=8)
            confirmar = gr.Checkbox(
                label="Confirmo que esto es lo que escribí en la hoja",
                value=False)

            gr.Markdown("### Paso 3 · Revisión")
            preg_img = gr.Textbox(label="Tu pregunta", lines=2,
                                  value="Revisa mi procedimiento: ¿en qué paso me equivoqué?")
            boton_rev = gr.Button("Revisar los pasos confirmados",
                                  variant="primary")
            salida_img = gr.Textbox(label="Revisión", lines=8)
            traza_img = gr.Textbox(label="Cómo se obtuvo", lines=8)

            boton_leer.click(leer_foto, img,
                             [mensaje, pasos_box, confirmar, traza_img])
            boton_rev.click(revisar_foto,
                            [pasos_box, preg_img, confirmar, enunciado_box],
                            [salida_img, traza_img])

        with gr.Tab("Estado"):
            estado_txt = gr.Textbox(label="Comprobación de salud", lines=14,
                                    value=texto_salud(tutor))
            gr.Button("Actualizar").click(lambda: texto_salud(obtener_tutor()),
                                          None, estado_txt)
    return demo


def _bandera_activa(nombre: str) -> bool:
    """Interpreta una bandera de entorno sin activar nada por accidente."""
    return os.environ.get(nombre, "").strip().lower() in {"1", "true", "yes", "on"}


def lanzar():
    """Arranca Gradio; el túnel público solo se habilita bajo petición explícita."""
    return construir().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=_bandera_activa("TUTOR_GRADIO_SHARE"),
    )


if __name__ == "__main__":
    lanzar()
