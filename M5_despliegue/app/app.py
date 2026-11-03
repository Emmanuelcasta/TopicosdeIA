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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Este servicio carga modelos de PyTorch. En Kaggle, Transformers puede
# descubrir el TensorFlow preinstalado e importar sus integraciones aunque no
# se usen; el binario de TensorFlow del runtime no es compatible con su NumPy.
# Desactivamos esa ruta antes de importar pipeline/transformers.
os.environ.setdefault("USE_TF", "0")

# Antes de tocar Gradio. Con pydantic >= 2.11 la ruta que sirve la página lanza
# `TypeError: argument of type 'bool' is not iterable` al describir el esquema de
# `gr.Image`, y el enlace público responde Internal Server Error. El arreglo
# primario es el pin de `config/requirements.txt`; esto cubre los entornos donde
# ese pin no se puede imponer, como Kaggle. Ver `compat_gradio.py`.
import compat_gradio                        # noqa: E402

COMPAT = compat_gradio.aplicar()

# Antes de importar pipeline, que es quien acaba cargando el modelo. En Kaggle
# la primera pregunta fallaba con `AttributeError: module 'torch.distributed'
# has no attribute 'tensor'`: alguien accede al subpaquete sin importarlo, y
# `import torch` no lo carga solo. Importarlo aquí hace que el atributo exista.
# Ver `compat_torch.py`, que además dice qué paquete lo usa así.
import compat_torch                         # noqa: E402

COMPAT_TORCH = compat_torch.asegurar_distributed_tensor()

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
        # Y dónde está la causa. Sin esta línea, en Kaggle se leyó «falta
        # retrieval híbrido + reranker» y se dio por hecho que faltaba un
        # paquete; estaban los tres instalados.
        donde = ""
        if any(not x["ok"] for x in e.retrieval_diagnostico):
            donde = ("\n\nLa causa concreta está en la pestaña **Estado**, en "
                     "`retrieval_por_etapas`, `retrieval_causa_probable` y "
                     "`retrieval_traceback`.")
        return ("🔴 **MODO DEGRADADO · resultado de esta ejecución: NO MEDIDO.**\n\n"
                f"Se pidió `{e.config_nombre}` y falta **{', '.join(e.faltantes)}**. "
                + aclaracion
                + "Las respuestas siguen funcionando, pero no son las del sistema "
                f"que se midió.{donde}\n\nQué está funcionando:\n{piezas}")
    if e.experimental:
        return (f"⚠️ **`{e.config_nombre}` · EXPERIMENTAL, sin medir.** Las cifras "
                f"de M1–M4 no respaldan esta combinación.\n\nQué monta:\n{piezas}")
    return (f"**`{e.config_nombre}`** · {e.medido} en el eval set fijo de M2, "
            "medido en S08 (`scorecard_rag_avanzado_v2.csv`). Esa cifra es "
            "**histórica**: describe esta configuración, no una corrida de hoy."
            f"\n\nQué monta:\n{piezas}")


_PAT_PENSAMIENTO = re.compile(
    r"^[ \t]*(?:Pensamiento|Thought|Razonamiento)\s*:.*$", re.IGNORECASE | re.MULTILINE)


def presentar(texto: str) -> str:
    """La respuesta como la lee un estudiante, sin el razonamiento interno.

    El formato ReAct del prompt hace que el modelo escriba «Pensamiento: …»
    antes de actuar. `nucleo.analizar_salida` solo quita esa etiqueta del campo
    `pensamiento`; cuando no hay llamada a herramienta, la línea se queda
    dentro de la respuesta y acababa en pantalla. Quitarla aquí, en la capa de
    presentación, deja intacto lo que ve el evaluador.
    """
    limpio = _PAT_PENSAMIENTO.sub("", texto or "")
    limpio = re.sub(r"\n{3,}", "\n\n", limpio).strip()
    return limpio or (texto or "").strip()


_PAT_PASO = re.compile(r"paso\s*(?:n\.?[ºo]\s*|número\s*)?(\d+)", re.IGNORECASE)


def presentar_revision(r: dict) -> str:
    """La revisión como la lee un estudiante.

    Tres piezas, en este orden: qué paso señala el tutor, qué dice la
    comprobación determinista de la aritmética, y —si las dos no coinciden— que
    no coinciden. Tapar una de las dos sería elegir por el estudiante, y la que
    se taparía es justo la que no se equivoca en aritmética.

    Esto es presentación. `pipeline.revisar_pasos` sigue devolviendo el texto
    crudo, que es lo que mide el evaluador.
    """
    texto = presentar(r.get("respuesta") or "")
    h = r.get("verificador_aritmetico") or {}
    tipo = h.get("tipo")
    partes = [texto] if texto else []
    if tipo in ("calculo", "traslado"):
        partes.append(f"Comprobación de la aritmética: {h['detalle']}.")
        dicho = _PAT_PASO.search(texto)
        if dicho and h.get("paso") and int(dicho.group(1)) != int(h["paso"]):
            partes.append(
                f"Ojo: yo te señalo el paso {dicho.group(1)} y la comprobación "
                f"de las cuentas señala el paso {h['paso']}. La comprobación es "
                "determinista y no se equivoca sumando; mi lectura del "
                "procedimiento sí puede. Mira los dos pasos.")
    elif tipo == "ninguno":
        partes.append("Comprobación de la aritmética: todas las igualdades "
                      "escritas cuadran, así que si hay un error es de "
                      "planteamiento, no de cálculo.")
    return "\n\n".join(x for x in partes if x)


def presentar_solucion(r: dict) -> str:
    """La solución de un enunciado, como la lee un estudiante.

    Qué se resolvió, los pasos, y cómo se comprobó. Si el verificador encontró
    una igualdad que no cuadra en lo que escribió el modelo, se dice en grande:
    la solución no se da por buena.
    """
    partes = [f"Ejercicio: {r.get('enunciado', '')}"]
    pasos = r.get("pasos_solucion") or []
    if pasos:
        partes.append("\n".join(
            p if p.lower().startswith(("paso", "respuesta")) else f"Paso {i}: {p}"
            for i, p in enumerate(pasos, 1)))
    else:
        partes.append(presentar(r.get("respuesta") or ""))
    h = r.get("verificador_aritmetico") or {}
    if r.get("camino") == "herramienta":
        obs = (r.get("observaciones") or [{}])[0]
        comprobacion = (f"Comprobación: el resultado lo calculó la herramienta "
                        f"`{obs.get('herramienta')}` ({obs.get('valor')}) y coincide "
                        "con los pasos.")
        if h.get("tipo") == "ninguno":
            comprobacion += " El verificador revisó cada igualdad: todas cuadran."
        partes.append(comprobacion)
    elif h.get("tipo") in ("calculo", "traslado"):
        partes.append(f"⚠️ Ojo: la comprobación determinista de las cuentas dice que "
                      f"{h['detalle']}. Esta solución la escribió el modelo y NO "
                      "debes darla por buena hasta revisar ese paso.")
    elif h.get("tipo") == "ninguno":
        partes.append("Comprobación: el verificador revisó cada igualdad escrita y "
                      "todas cuadran. Eso comprueba las cuentas, no el planteamiento.")
    else:
        partes.append("Comprobación: no había igualdades que el verificador pudiera "
                      "revisar. Revisa el razonamiento con calma.")
    return "\n\n".join(x for x in partes if x)


def _formatear(r: dict) -> tuple:
    if not r.get("ok"):
        return f"⚠️ {r['respuesta']}", ""
    detalle = [f"latencia: {r['latencia_s']} s", f"pasos del modelo: {r['pasos_llm']}"]
    if r["herramientas"]:
        detalle.append("herramientas ejecutadas: " + ", ".join(r["herramientas"]))
        for o in r.get("observaciones") or []:
            detalle.append(f"  {o['herramienta']}({o['argumentos']}) -> "
                           f"{o['resultado'] if o['ok'] else o['error']}")
    else:
        detalle.append("herramientas ejecutadas: ninguna")
    if r.get("contextos"):
        detalle.append(f"pasajes consultados: {r['contextos']}")
    if r.get("verificador"):
        v = r["verificador"]
        detalle.append(f"verificador aritmético: paso {v['paso']} ({v['tipo']}) "
                       f"{v['detalle']}" + ("" if v["reintento"] else " · sin reintento"))
    n = r.get("normalizacion") or {}
    if n.get("px_antes"):
        detalle.append(
            f"foto normalizada: {n['px_antes'][0]}×{n['px_antes'][1]} -> "
            f"{n['px_despues'][0]}×{n['px_despues'][1]} px"
            + (" · orientación EXIF corregida" if n.get("exif") else ""))
    if r.get("transcripcion"):
        detalle.append("\nLo que se leyó en la hoja:\n" + r["transcripcion"])
    if r.get("etapa") == "diagnostico":
        return presentar_revision(r), "\n".join(detalle)
    if r.get("etapa") == "solucion":
        return presentar_solucion(r), "\n".join(detalle)
    return presentar(r["respuesta"]), "\n".join(detalle)


def _vaciar_texto():
    """Deja las dos cajas en blanco antes de empezar. Sin ir al servidor."""
    return "", ""


def responder_texto(pregunta: str):
    tutor = obtener_tutor()
    r = tutor.responder(pregunta)
    registrar("consulta_texto", pregunta, ok=r["ok"], latencia_s=r["latencia_s"],
              pasos_llm=r["pasos_llm"], n_herramientas=len(r["herramientas"]))
    return _formatear(r)


_PRIMERA_LECTURA = True

# Los dos propósitos de una foto. El estudiante elige; la lectura solo sugiere.
RESOLVER = "Resolver ejercicio"
REVISAR = "Revisar procedimiento"


def _tiempo_legible(seg: float, incluye_carga: bool) -> str:
    """El tiempo, dicho para un estudiante y sin esconder de qué está hecho.

    `incluye_carga` solo es cierto si el VLM NO estaba ya cargado. Con
    `TUTOR_PRECARGAR=1` la carga se paga al arrancar la celda, y decir aquí que
    «la mayor parte fue cargar el modelo» sería falso.
    """
    if incluye_carga:
        return (f"Tardó {seg:.0f} s. La mayor parte fue cargar el modelo de "
                "visión, que solo pasa la primera vez: las siguientes fotos "
                "tardan unos pocos segundos.")
    return f"Tardó {seg:.1f} s."


def leer_foto(imagen):
    """Etapa 1 -> 2. Lee la hoja y abre el panel de edición. Nunca diagnostica.

    Tres salidas posibles, y las tres terminan en una pantalla distinta:

      error           no hay foto, o la vía visual no está activa
      no fiable       se para y se ofrece otra foto o escribir los pasos
      lectura buena   se abre el panel para revisar y corregir la transcripción

    Lo que nunca ocurre es pasar directo al diagnóstico: esa fue la lección de
    `vis-10`, donde el VLM inventó una cuenta y el tutor la diagnosticó sin dudar.
    """
    global _PRIMERA_LECTURA

    import gradio as gr

    oculto, visible = gr.update(visible=False), gr.update(visible=True)
    tutor = obtener_tutor()
    # Antes de leer: si el VLM ya estaba en memoria —porque se precargó— esta
    # lectura no incluye ninguna carga, y el mensaje no debe decir que sí.
    ya_cargado = tutor._vlm is not None     # noqa: SLF001 — mismo paquete
    r = tutor.leer_imagen(imagen)
    carga = _PRIMERA_LECTURA and not ya_cargado
    _PRIMERA_LECTURA = False
    registrar("lectura_imagen", "", ok=r["ok"], latencia_s=r["latencia_s"],
              pasos_llm=0, n_herramientas=0)

    if not r.get("ok"):
        return (f"⚠️ {r['respuesta']}", oculto, oculto, "", "",
                False, "", oculto, gr.update(value=REVISAR))

    traza = [f"lectura: {r['latencia_s']} s",
             f"pasos leídos: {len(r['pasos'])}",
             f"las dos lecturas independientes coinciden: {r.get('lecturas_coinciden')}"]
    if r.get("diferencias"):
        traza.append("en qué difieren: " + ", ".join(r["diferencias"]))
    if r.get("transcripcion"):
        traza.append("\nTranscripción cruda:\n" + r["transcripcion"])
    tiempo = _tiempo_legible(r["latencia_s"], carga)

    if not r.get("confirmable"):
        # Aquí se PARA. No se enseña una transcripción que el propio sistema
        # acaba de declarar no fiable: verla invita a confirmarla.
        aviso = ("### 🟠 No pude leer bien esta hoja\n\n"
                 f"{r['respuesta']}\n\n"
                 f"**Lo que observé:** {r['motivo_no_fiable']}.\n\n"
                 f"{tiempo}\n\n"
                 "No voy a adivinar qué escribiste. Toma otra foto, o escribe "
                 "tus pasos a mano abajo y los reviso igual.")
        # El panel de edición se abre VACÍO: se ofrece escribir los pasos a
        # mano, pero no se enseña la transcripción que acaba de declararse no
        # fiable, porque verla invita a confirmarla.
        return (aviso, visible, visible, "", "", False, "\n".join(traza), oculto,
                gr.update(value=REVISAR))

    if r.get("proposito") == "resolver":
        sugerido = RESOLVER
        aviso = ("### ✅ Leí un enunciado, sin pasos resueltos\n\n"
                 "**Revisa que esté bien copiado** y elige qué quieres que haga. "
                 "Si transcribí mal un número, resolvería otro ejercicio.\n\n"
                 f"{tiempo}")
    else:
        sugerido = REVISAR
        aviso = ("### ✅ Esto es lo que leí en tu hoja\n\n"
                 "**Revísalo antes de que lo corrija.** Si transcribí mal un número, "
                 "el error que te señale sería mío, no tuyo.\n\n"
                 f"{tiempo}")
    traza.append(f"propósito sugerido: {sugerido.lower()} (lo confirma el estudiante)")
    return (aviso, visible, oculto, "\n".join(r["pasos"]),
            r.get("enunciado") or "", False, "\n".join(traza), oculto,
            gr.update(value=sugerido))


def confirmar_transcripcion(texto_pasos: str, enunciado: str = "",
                            proposito: str = REVISAR):
    """Etapa 3 -> 4. Abre el paso 4 solo si hay algo que resolver o revisar."""
    import gradio as gr

    cerrado = (gr.update(visible=False), False, gr.update())
    pasos = [l.strip() for l in (texto_pasos or "").splitlines() if l.strip()]
    if proposito == RESOLVER:
        if not (enunciado or "").strip():
            return ("⚠️ No hay ningún enunciado que resolver. Escríbelo arriba, o "
                    "toma otra foto.",) + cerrado
        return ("### ✅ Enunciado confirmado\n\nAhora sí puedo resolverlo.",
                gr.update(visible=True), True,
                gr.update(value="Resolver el ejercicio"))
    if not pasos:
        return ("⚠️ No hay ningún paso que revisar. Escríbelos, elige «Resolver "
                "ejercicio» si la hoja solo trae el enunciado, o toma otra foto.",
                ) + cerrado
    return (f"### ✅ Transcripción confirmada · {len(pasos)} pasos\n\n"
            "Ahora sí puedo revisar el procedimiento.",
            gr.update(visible=True), True,
            gr.update(value="Revisar el procedimiento"))


def otra_foto():
    """Vuelve a la etapa 1 y borra todo lo anterior. Sin estado colgando."""
    import gradio as gr

    oculto = gr.update(visible=False)
    return ("### Paso 1 · Sube otra foto de la hoja", oculto, oculto,
            "", "", False, "", "", oculto, None, gr.update(value=REVISAR))


def resolver_foto(enunciado: str, confirmado: bool):
    """Etapa 5, propósito «resolver»: el enunciado confirmado, paso a paso."""
    tutor = obtener_tutor()
    r = tutor.resolver_enunciado(enunciado, confirmado=bool(confirmado))
    registrar("resolucion_enunciado", enunciado, ok=r["ok"],
              latencia_s=r["latencia_s"], pasos_llm=r.get("pasos_llm", 0),
              n_herramientas=len(r.get("herramientas", [])))
    respuesta, detalle = _formatear(r)
    if r.get("ok"):
        detalle = f"camino: {r.get('camino')}\n" + detalle
    return respuesta, detalle


def revisar_foto(texto_pasos: str, pregunta: str, confirmado: bool,
                 enunciado: str = "", proposito: str = REVISAR):
    """Etapa 5. Diagnostica los pasos confirmados.

    `confirmado` lo pone el botón de la etapa 3, no el estudiante marcando una
    casilla al azar: sin él, `revisar_pasos` se niega y lo dice.
    """
    if proposito == RESOLVER:
        return resolver_foto(enunciado, confirmado)
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
        detalle = (f"comprobación determinista de la aritmética: {v['detalle']} "
                   f"(tipo {v['tipo']})\n" + detalle)
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
            # Dos eventos encadenados a propósito. El primero vacía las cajas
            # sin tocar el servidor; el segundo responde. Si el segundo muere
            # —el túnel de `gradio.live` devolvió HTML y el cliente no supo
            # leerlo—, lo que queda en pantalla es vacío, no la respuesta de la
            # pregunta anterior haciéndose pasar por la de esta.
            boton.click(_vaciar_texto, None, [salida, traza], queue=False).then(
                responder_texto, entrada, [salida, traza])
            gr.Examples([["¿Cuánto es 3/4 + 1/8?"],
                         ["¿Cuál es la nota mínima aprobatoria del colegio?"],
                         ["Explica por qué 9 es un número primo."]], entrada)

        # La pestaña de visión solo si la configuración activa la incluye. En
        # S08-D no existe: ofrecerla sugeriría que forma parte de lo que midió
        # 18/21, y la vía visual nunca se midió junto con ese retrieval.
        #
        # Cinco etapas, y cada una aparece solo cuando llega su turno. Antes
        # estaban las tres a la vez y la casilla de confirmación se podía marcar
        # sin haber leído nada: la pantalla sugería que el orden daba igual.
        with gr.Tab("Revisar una foto", visible=tutor.cfg.vision):
            confirmado = gr.State(False)

            gr.Markdown("## Revisar una foto de tu hoja" + AVISO_VISION)

            gr.Markdown("### Paso 1 · Sube la foto")
            img = gr.Image(label="Foto de la hoja", type="pil")
            boton_leer = gr.Button("Leer la hoja", variant="primary")

            aviso = gr.Markdown()

            with gr.Group(visible=False) as panel_edicion:
                gr.Markdown(
                    "### Paso 2 · Revisa y edita lo que leí\n"
                    "Un paso por línea. **Corrige lo que esté mal transcrito.** "
                    "Si el error es de lectura, la respuesta sería sobre una "
                    "hoja que no escribiste.")
                proposito = gr.Radio(
                    [RESOLVER, REVISAR], value=REVISAR,
                    label="¿Qué quieres que haga?",
                    info="Resolver: la hoja solo trae el enunciado. Revisar: la "
                         "hoja trae tus pasos y busco el primero incorrecto.")
                enunciado_box = gr.Textbox(label="Enunciado del ejercicio", lines=2)
                pasos_box = gr.Textbox(
                    label="Los pasos de tu hoja (vacío si solo hay enunciado)",
                    lines=8)
                gr.Markdown("### Paso 3 · Confirma la transcripción")
                boton_confirmar = gr.Button("Confirmar transcripción",
                                            variant="primary")

            with gr.Group(visible=False) as panel_ilegible:
                gr.Markdown(
                    "### La lectura no es de fiar\n"
                    "Puedes tomar otra foto —con más luz y la hoja plana— o "
                    "escribir los pasos arriba y confirmarlos. Las dos rutas "
                    "llegan a la misma revisión.")
                boton_otra = gr.Button("Tomar otra foto", variant="primary")

            with gr.Group(visible=False) as panel_revision:
                gr.Markdown("### Paso 4 · Resolver o revisar")
                preg_img = gr.Textbox(
                    label="Tu pregunta", lines=2,
                    value="Revisa mi procedimiento: ¿en qué paso me equivoqué?")
                boton_rev = gr.Button("Revisar el procedimiento", variant="primary")
                salida_img = gr.Textbox(label="Resultado", lines=12)

            traza_img = gr.Textbox(label="Cómo se obtuvo", lines=6)

            # Leer de nuevo, o cambiar el propósito, invalida lo confirmado y
            # vacía el resultado anterior: nada de una foto vieja sobrevive.
            boton_leer.click(
                _vaciar_texto, None, [salida_img, traza_img], queue=False).then(
                leer_foto, img,
                [aviso, panel_edicion, panel_ilegible, pasos_box, enunciado_box,
                 confirmado, traza_img, panel_revision, proposito])
            proposito.change(
                lambda: (False, gr.update(visible=False), ""), None,
                [confirmado, panel_revision, salida_img], queue=False)
            boton_confirmar.click(
                confirmar_transcripcion, [pasos_box, enunciado_box, proposito],
                [aviso, panel_revision, confirmado, boton_rev])
            boton_otra.click(
                otra_foto, None,
                [aviso, panel_edicion, panel_ilegible, pasos_box, enunciado_box,
                 confirmado, traza_img, salida_img, panel_revision, img, proposito])
            boton_rev.click(
                _vaciar_texto, None, [salida_img, traza_img], queue=False).then(
                revisar_foto,
                [pasos_box, preg_img, confirmado, enunciado_box, proposito],
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
    v = COMPAT["versiones"]
    print("Entorno: " + " | ".join(f"{k} {val}" for k, val in v.items()))
    if COMPAT["parche_aplicado"]:
        print("Compatibilidad de Gradio: parche de subesquemas booleanos APLICADO")
        print(f"  {COMPAT['motivo']}")
    # El entorno de torch, que es donde fallaba la primera pregunta. Se imprime
    # al arrancar para que el log de la celda de Kaggle ya traiga la respuesta
    # cuando algo salga mal, sin tener que reproducirlo.
    print("torch: " + " | ".join(
        f"{k} {val}" for k, val in compat_torch.versiones().items()))
    print(f"torch.distributed.tensor: {COMPAT_TORCH['accion'] or '—'}")
    if COMPAT_TORCH.get("error"):
        print(f"  AVISO: {COMPAT_TORCH['error']}")
    # La primera pregunta carga el modelo y tarda ~80 s. Por el túnel de
    # `gradio.live` esa espera es la que acaba devolviendo una página HTML de
    # error al cliente («Unexpected token '<'»), y la caja se queda con la
    # respuesta anterior. Precargar mueve esa espera al arranque de la celda,
    # donde no hay ningún cliente esperando.
    if _bandera_activa("TUTOR_PRECARGAR"):
        import time as _t

        t0 = _t.perf_counter()
        tutor = obtener_tutor()
        tutor._cargar_modelo()      # noqa: SLF001 — mismo paquete, a propósito
        print(f"precarga del modelo de texto: {_t.perf_counter() - t0:.1f} s · "
              f"listo={tutor.estado.listo}")
        if tutor.cfg.vision:
            print("precarga del VLM: " + str(tutor.precargar_vision()))
    return construir().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=_bandera_activa("TUTOR_GRADIO_SHARE"),
    )


if __name__ == "__main__":
    lanzar()
