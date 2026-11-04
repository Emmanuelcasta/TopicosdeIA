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
_PREFIJO = re.compile(r"^\s*paso\s*\d+\s*[:.\-]\s*", re.IGNORECASE)
_FINAL = re.compile(r"^\s*respuesta\s*final\s*[:.]?\s*", re.IGNORECASE)

# Lo que significa cada marca. Se enseña debajo de los pasos, una vez.
LEYENDA = ("✅ cuenta comprobada con aritmética exacta · ❌ la cuenta no cuadra · "
           "sin marca: no hay una cuenta que comprobar en esa línea")


def marca_de_cuenta(linea: str) -> str:
    """✅ si las igualdades de la línea cuadran, ❌ si alguna no, "" si no hay.

    Lo decide `verificador_aritmetico`, no la interfaz: aquí solo se pinta.
    """
    try:
        from verificador_aritmetico import _valor, igualdades_de
    except Exception:  # noqa: BLE001
        return ""
    pares = [(_valor(a), _valor(b)) for a, b in igualdades_de(linea)]
    pares = [(a, b) for a, b in pares if a is not None and b is not None]
    if not pares:
        return ""
    return "✅" if all(a == b for a, b in pares) else "❌"


def pasos_en_markdown(pasos, resaltar: int | None = None) -> tuple:
    """-> (lista numerada en Markdown, texto de la respuesta final o None)."""
    filas, final = [], None
    for p in pasos or []:
        texto = str(p).strip()
        if _FINAL.match(texto):
            final = _FINAL.sub("", texto)
            continue
        filas.append(_PREFIJO.sub("", texto))
    lineas = []
    for i, f in enumerate(filas, 1):
        marca = marca_de_cuenta(f)
        aviso = "  ← **primer error**" if resaltar == i else ""
        lineas.append(f"{i}. {f} {marca}{aviso}".rstrip())
    return "\n".join(lineas), final


def presentar_revision(r: dict) -> str:
    """La revisión como la lee un estudiante.

    Tres piezas, en este orden: qué paso señala el tutor, qué dice la
    comprobación determinista de la aritmética, y —si las dos no coinciden— que
    no coinciden. Tapar una de las dos sería elegir por el estudiante, y la que
    se taparía es justo la que no se equivoca en aritmética.

    Esto es presentación. `pipeline.revisar_pasos` sigue devolviendo el texto
    crudo, que es lo que mide el evaluador.
    """
    if r.get("diagnostico_final"):
        return _revision_en_markdown(r)
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


_TITULOS = {
    "sin error": ("✅", "Sin error"),
    "dato no disponible": ("🟡", "Falta un dato"),
    "sin conclusión": ("⚪", "No puedo localizar el error"),
}
_FUENTES = {
    "verificador": "✅ Comprobado con aritmética exacta (verificador).",
    "ambos": "✅ Comprobado con aritmética exacta; el tutor llega a lo mismo.",
    "referencia": ("🔎 Comparado con el enunciado resuelto por separado "
                   "(dos soluciones que coinciden; cuentas comprobadas)."),
    "modelo": "⚠️ Lo sugiere el tutor y no está verificado: revísalo con calma.",
}


def _bloque(respuesta: str, inicio: str) -> str:
    """El párrafo de `respuesta` que empieza por `inicio`, sin el rótulo."""
    for trozo in (respuesta or "").split("\n\n"):
        if trozo.startswith(inicio):
            return trozo[len(inicio):].strip()
    return ""


def _revision_en_markdown(r: dict) -> str:
    """El diagnóstico combinado (`pipeline._combinar_diagnostico`) para leer:
    la conclusión, por qué, de dónde sale, y la hoja con cada cuenta marcada."""
    d = r["diagnostico_final"]
    final = d.get("conclusion") or "sin conclusión"
    m = re.match(r"paso (\d+)", final)
    icono, titulo = ("🔴", f"Primer error: paso {m.group(1)}") if m else \
        _TITULOS.get(final, ("⚪", final[:1].upper() + final[1:]))
    partes = [f"### {icono} {titulo}"]
    if d.get("motivo"):
        partes.append(f"**Por qué:** {d['motivo']}.")
    if d.get("fuente") in _FUENTES:
        partes.append(_FUENTES[d["fuente"]])
    pasos = r.get("pasos_confirmados") or []
    if pasos and final != "dato no disponible":
        lista, fin = pasos_en_markdown(pasos, int(m.group(1)) if m else None)
        if fin is not None:
            n = len(lista.splitlines()) + 1
            marca = "  ← **primer error**" if m and int(m.group(1)) == n else ""
            lista += f"\n{n}. Respuesta final: {fin}{marca}"
        partes.append("#### Tu procedimiento\n" + lista + f"\n\n<small>{LEYENDA}</small>")
    referencia = _bloque(r.get("respuesta") or "", "Así sale al resolver el enunciado por separado:")
    if referencia:
        lista, fin = pasos_en_markdown(referencia.splitlines())
        partes.append("#### Así sale al resolver el enunciado por separado\n" + lista
                      + (f"\n\n**Resultado: {fin}**" if fin else ""))
    tutor = _bloque(r.get("respuesta") or "", "Lo que explica el tutor:")
    if tutor:
        partes.append("#### Lo que explica el tutor\n" + presentar(tutor))
    elif d.get("desacuerdo") and d.get("modelo"):
        partes.append("<small>El tutor, por su cuenta, había concluido otra cosa; la "
                      "conclusión de arriba se apoya en la comprobación.</small>")
    return "\n\n".join(partes)


def presentar_solucion(r: dict) -> str:
    """La solución de un enunciado, como la lee un estudiante.

    Qué se resolvió, los pasos, y cómo se comprobó. Si el verificador encontró
    una igualdad que no cuadra en lo que escribió el modelo, se dice en grande:
    la solución no se da por buena.
    """
    if r.get("camino") in ("ninguno", "fuera_de_dominio", "falta_dato"):
        icono = "🟡" if r.get("camino") == "falta_dato" else "⚪"
        return f"### {icono} {presentar(r.get('respuesta') or '')}"
    partes = [f"#### Ejercicio\n> {r.get('enunciado', '')}"]
    pasos = r.get("pasos_solucion") or []
    fin = None
    if pasos:
        lista, fin = pasos_en_markdown(pasos)
        partes.append("#### Pasos\n" + (lista or "_(el resultado sale directo)_"))
    else:
        partes.append(presentar(r.get("respuesta") or ""))
    if fin:
        partes.append(f"### Resultado: {fin}")
    h = r.get("verificador_aritmetico") or {}
    if r.get("camino") == "herramienta":
        obs = (r.get("observaciones") or [{}])[0]
        comprobacion = (f"✅ **Comprobado.** El resultado lo calculó la herramienta "
                        f"`{obs.get('herramienta')}` ({obs.get('valor')}) y coincide "
                        "con los pasos.")
        if h.get("tipo") == "ninguno":
            comprobacion += " El verificador revisó cada igualdad: todas cuadran."
        partes.append(comprobacion)
    elif h.get("tipo") in ("calculo", "traslado"):
        partes.append(f"❌ **Ojo:** la comprobación determinista de las cuentas dice que "
                      f"{h['detalle']}. Esta solución la escribió el modelo y NO "
                      "debes darla por buena hasta revisar ese paso.")
    elif h.get("tipo") == "ninguno":
        partes.append("✅ **Cuentas verificadas.** El verificador recalculó cada "
                      "igualdad con aritmética exacta y todas cuadran. Eso comprueba "
                      "las cuentas, no el planteamiento.")
    else:
        partes.append("⚪ No había igualdades que el verificador pudiera revisar. "
                      "Revisa el razonamiento con calma.")
    if r.get("cuentas_corregidas"):
        partes.append("🔧 **Cuentas rehechas por el verificador** (el borrador del "
                      "modelo se había equivocado): "
                      + "; ".join(r["cuentas_corregidas"]) + ".")
    if r.get("confiable") is False:
        partes.append("⚠️ **No estoy seguro de esta solución:** al resolverlo de "
                      "varias maneras no salió el mismo resultado. Revísala con tu "
                      "profesor.")
    if pasos:
        partes.append(f"<small>{LEYENDA}</small>")
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
    return presentar_texto(r["respuesta"]), "\n".join(detalle)


def presentar_texto(texto: str) -> str:
    """Respuesta a una pregunta escrita: los pasos con su marca de cuenta y el
    resultado aparte. Si no hay pasos reconocibles, el texto tal cual."""
    limpio = presentar(texto)
    lineas = [l for l in limpio.splitlines() if l.strip()]
    if not any(_PREFIJO.match(l) or _FINAL.match(l) for l in lineas):
        return limpio
    try:
        # Lo mismo que se hace con la solución de una foto: fuera rótulos,
        # respuestas finales intermedias e igualdades triviales.
        from verificador_aritmetico import limpiar_solucion
        lineas = limpiar_solucion(lineas)
    except Exception:  # noqa: BLE001
        pass
    lista, fin = pasos_en_markdown(lineas)
    partes = [lista] if lista else []
    if fin:
        partes.append(f"### Resultado: {fin}")
    marcas = [marca_de_cuenta(l) for l in lineas]
    if "❌" in marcas:
        partes.append("❌ **Ojo:** una de las cuentas no cuadra; no des la respuesta "
                      "por buena.")
    elif "✅" in marcas:
        partes.append("✅ **Cuentas verificadas** con aritmética exacta. Eso "
                      "comprueba las cuentas, no el planteamiento.")
    partes.append(f"<small>{LEYENDA}</small>")
    return "\n\n".join(partes)


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

# Los dos propósitos de una foto. El estudiante elige al subirla; la lectura
# solo corrige la elección si la hoja no trae pasos que revisar.
RESOLVER = "Resolver ejercicio"
REVISAR = "Revisar procedimiento"
ETIQUETA_ACCION = {RESOLVER: "Resolver el ejercicio", REVISAR: "Revisar mi procedimiento"}


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


def leer_foto(imagen, proposito: str = REVISAR):
    """Paso 1 -> 2. Lee la hoja y abre la lectura editable. Nunca diagnostica.

    Tres salidas posibles, y las tres terminan en una pantalla distinta:

      error           no hay foto, o la vía visual no está activa
      no fiable       se para y se ofrece otra foto o escribir los pasos
      lectura buena   se abre la lectura para revisarla y corregirla

    Lo que nunca ocurre es pasar directo al diagnóstico: esa fue la lección de
    `vis-10`, donde el VLM inventó una cuenta y el tutor la diagnosticó sin dudar.

    -> (aviso, panel_edicion, panel_ilegible, pasos_box, enunciado_box,
        confirmado, traza, panel_resultado, proposito, boton_accion)
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
    proposito = proposito if proposito in (RESOLVER, REVISAR) else REVISAR

    if not r.get("ok"):
        return (f"⚠️ {r['respuesta']}", oculto, oculto, gr.update(value=""), "",
                False, "", oculto, gr.update(value=proposito), gr.update())

    traza = [f"lectura: {r['latencia_s']} s",
             f"pasos leídos: {len(r['pasos'])}",
             f"las dos lecturas independientes coinciden: {r.get('lecturas_coinciden')}"]
    if r.get("diferencias"):
        traza.append("en qué difieren: " + ", ".join(r["diferencias"]))
    if r.get("diferencias_inocuas"):
        traza.append("diferencias de redacción, ignoradas: "
                     + ", ".join(r["diferencias_inocuas"]))
    if r.get("transcripcion"):
        traza.append("\nTranscripción cruda:\n" + r["transcripcion"])
    tiempo = _tiempo_legible(r["latencia_s"], carga)

    if not r.get("confirmable"):
        # Aquí se PARA. No se enseña una transcripción que el propio sistema
        # acaba de declarar no fiable: verla invita a confirmarla.
        aviso = ("### 🟠 No pude leer bien esta hoja\n\n"
                 f"{r['respuesta']}\n\n"
                 f"**Lo que observé:** {r['motivo_no_fiable']}.\n\n"
                 + f"{tiempo}\n\n"
                 "No voy a adivinar qué escribiste. Toma otra foto, o escribe "
                 "el enunciado y tus pasos abajo y los reviso igual.")
        return (aviso, visible, visible, gr.update(value="", visible=True), "", False,
                "\n".join(traza), oculto, gr.update(value=REVISAR),
                gr.update(value=ETIQUETA_ACCION[REVISAR]))

    nota = ""
    if proposito == REVISAR and r.get("proposito") == "resolver":
        # Pidió revisar, pero la hoja no trae pasos: no hay nada que revisar.
        proposito = RESOLVER
        nota = ("\n\nLa hoja solo trae el enunciado, sin pasos resueltos, así que "
                "cambié a **resolver el ejercicio**.")
    if proposito == RESOLVER:
        aviso = ("### 📝 Esto es lo que leí\n\n"
                 "**Revisa que el enunciado esté bien copiado** —sobre todo los "
                 "números— y confirma. Si transcribí mal un dato, resolvería otro "
                 f"ejercicio.{nota}\n\n{tiempo}")
    else:
        aviso = ("### 📝 Esto es lo que leí en tu hoja\n\n"
                 "**Revísalo antes de que lo corrija.** Si transcribí mal un "
                 "número, el error que te señale sería mío, no tuyo.\n\n"
                 f"{tiempo}")
    if r.get("aviso_discrepancias"):
        # Antes del botón de confirmar, no en la traza: es lo que el estudiante
        # tiene que comprobar mirando su hoja.
        aviso += "\n\n**⚠️ Revisa esto antes de confirmar.** " + r["aviso_discrepancias"]
    traza.append(f"propósito: {proposito.lower()} (lo confirma el estudiante)")
    return (aviso, visible, oculto,
            gr.update(value="\n".join(r["pasos"]), visible=proposito == REVISAR),
            r.get("enunciado") or "", False, "\n".join(traza), oculto,
            gr.update(value=proposito), gr.update(value=ETIQUETA_ACCION[proposito]))


def cambiar_proposito(proposito: str):
    """Cambiar de resolver a revisar invalida lo confirmado y muestra u oculta
    la caja de pasos: al resolver solo cuenta el enunciado."""
    import gradio as gr

    return (False, gr.update(visible=False), "",
            gr.update(visible=proposito == REVISAR),
            gr.update(value=ETIQUETA_ACCION.get(proposito, ETIQUETA_ACCION[REVISAR])))


def confirmar_transcripcion(texto_pasos: str, enunciado: str = "",
                            proposito: str = REVISAR):
    """Paso 2 -> 3. Abre el resultado solo si hay algo que resolver o revisar."""
    import gradio as gr

    cerrado = (gr.update(visible=False), False)
    pasos = [l.strip() for l in (texto_pasos or "").splitlines() if l.strip()]
    if proposito == RESOLVER:
        if not (enunciado or "").strip():
            return ("⚠️ No hay ningún enunciado que resolver. Escríbelo arriba, o "
                    "toma otra foto.",) + cerrado
        return ("### ✅ Enunciado confirmado\n\nPulsa **Resolver el ejercicio**.",
                gr.update(visible=True), True)
    if not pasos:
        return ("⚠️ No hay ningún paso que revisar. Escríbelos, elige «Resolver "
                "ejercicio» si la hoja solo trae el enunciado, o toma otra foto.",
                ) + cerrado
    return (f"### ✅ Lectura confirmada · {len(pasos)} pasos\n\n"
            "Pulsa **Revisar mi procedimiento**.",
            gr.update(visible=True), True)


def otra_foto():
    """Vuelve al paso 1 y borra todo lo anterior. Sin estado colgando."""
    import gradio as gr

    oculto = gr.update(visible=False)
    return ("", oculto, oculto, gr.update(value=""), "", False, "", "", oculto, None)


def resolver_foto(enunciado: str, confirmado: bool):
    """Propósito «resolver»: el enunciado confirmado, paso a paso."""
    tutor = obtener_tutor()
    r = tutor.resolver_enunciado(enunciado, confirmado=bool(confirmado))
    registrar("resolucion_enunciado", enunciado, ok=r["ok"],
              latencia_s=r["latencia_s"], pasos_llm=r.get("pasos_llm", 0),
              n_herramientas=len(r.get("herramientas", [])))
    respuesta, detalle = _formatear(r)
    if r.get("ok"):
        detalle = f"camino: {r.get('camino')}\n" + detalle
    return respuesta, detalle


def revisar_foto(texto_pasos: str, confirmado: bool, enunciado: str = "",
                 proposito: str = REVISAR):
    """Resuelve o revisa lo confirmado.

    `confirmado` lo pone el botón de confirmar, no el estudiante marcando una
    casilla al azar: sin él, `revisar_pasos` se niega y lo dice. La pregunta de
    la revisión es fija (`PREGUNTA_REVISION`): antes había una caja editable con
    «Revisa mi procedimiento…» que aparecía también al resolver.
    """
    if proposito == RESOLVER:
        return resolver_foto(enunciado, confirmado)
    tutor = obtener_tutor()
    pasos = [l.strip() for l in (texto_pasos or "").splitlines() if l.strip()]
    pregunta = tutor.PREGUNTA_REVISION
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


# Estilo de la demo: los colores de la presentación (naranja, azul noche,
# verde azulado) y tarjetas para separar lectura, confirmación y resultado.
CSS = """
.gradio-container {max-width: 1080px !important; margin: auto;}
#cabecera {background: linear-gradient(120deg, #101b2d 0%, #1d2f4d 100%);
  color: #fff; border-radius: 18px; padding: 26px 30px; margin-bottom: 6px;}
#cabecera h1 {color: #fff; margin: 0 0 6px; font-size: 2rem; letter-spacing: -.02em;}
#cabecera p {color: #d7deea; margin: 0; font-size: 1.05rem;}
#cabecera .chips {margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap;}
#cabecera .chip {background: #ffffff14; border: 1px solid #ffffff2e; color: #fff;
  border-radius: 999px; padding: 4px 12px; font-size: .85rem;}
#cabecera .chip.naranja {background: #f4772126; border-color: #f47721; color: #ffd2b2;}
.paso-titulo h3 {margin: 4px 0 !important;}
.paso-titulo h3 .num {display: inline-block; width: 1.7em; height: 1.7em;
  line-height: 1.7em; text-align: center; border-radius: 50%; background: #f47721;
  color: #fff; margin-right: .45em; font-size: .9em;}
.tarjeta {border-radius: 14px !important;}
/* El marco solo cuando hay algo dentro: vacío no debe parecer una respuesta. */
.resultado:has(p, h3, h4, ol, li) {border-left: 5px solid #00a994;
  padding: 6px 18px !important; border-radius: 10px; background: #00a99410;}
.resultado h3 {margin-top: 6px;}
.aviso-lectura:has(p, h3) {border-left: 5px solid #f47721;
  padding: 2px 16px !important; border-radius: 10px; background: #f4772112;}
"""

CABECERA = """
<div id="cabecera">
  <h1>Tutor de matemáticas</h1>
  <p>Escribe una pregunta o sube la foto de un ejercicio. Te muestro los pasos,
  el resultado y qué cuentas comprobé.</p>
  <div class="chips">
    <span class="chip">✍️ Pregunta escrita</span>
    <span class="chip">📷 Foto: resolver o revisar</span>
    <span class="chip">✅ Cuentas comprobadas con aritmética exacta</span>
    <span class="chip naranja">Experimental · probado con fotos de desarrollo</span>
  </div>
</div>
"""


def _titulo(n: int, texto: str):
    import gradio as gr

    return gr.Markdown(f'### <span class="num">{n}</span>{texto}',
                       elem_classes=["paso-titulo"])


def _estilo() -> dict:
    """Tema y CSS. Gradio 6 los recibe en `launch()`; antes, en `Blocks()`."""
    import inspect

    import gradio as gr

    tema = gr.themes.Soft(primary_hue="orange", secondary_hue="teal",
                          neutral_hue="slate")
    return {"theme": tema, "css": CSS,
            "en_launch": "css" in inspect.signature(gr.Blocks.launch).parameters}


def construir():
    import gradio as gr

    tutor = obtener_tutor()
    estilo = _estilo()
    extra = {} if estilo["en_launch"] else {"theme": estilo["theme"], "css": estilo["css"]}
    with gr.Blocks(title="Tutor de matemáticas · M5", **extra) as demo:
        gr.HTML(CABECERA)
        # El modo degradado no se esconde en un desplegable: si falta una pieza,
        # es lo primero que se lee.
        if tutor.estado.degradado:
            gr.Markdown(_cabecera(tutor))
        with gr.Accordion("Qué es, qué no es y cómo está configurado", open=False):
            gr.Markdown(AVISO)
            if not tutor.estado.degradado:
                gr.Markdown(_cabecera(tutor))

        with gr.Tab("✍️ Pregunta escrita"):
            with gr.Group(elem_classes=["tarjeta"]):
                entrada = gr.Textbox(label="Tu pregunta", lines=3,
                                     placeholder="Un taller compra 3 cajas a 18500 pesos…")
                boton = gr.Button("Responder", variant="primary")
            gr.Examples([["¿Cuánto es 3/4 + 1/8?"],
                         ["¿Cuántas galletas hay en 15 paquetes de 18 galletas cada uno?"],
                         ["¿Cuál es la nota mínima aprobatoria del colegio?"]], entrada,
                        label="Ejemplos")
            salida = gr.Markdown(elem_classes=["resultado"])
            with gr.Accordion("Cómo se obtuvo (detalle técnico)", open=False):
                traza = gr.Textbox(label="Traza", lines=6, show_label=False)
            # Dos eventos encadenados a propósito. El primero vacía las cajas
            # sin tocar el servidor; el segundo responde. Si el segundo muere
            # —el túnel de `gradio.live` devolvió HTML y el cliente no supo
            # leerlo—, lo que queda en pantalla es vacío, no la respuesta de la
            # pregunta anterior haciéndose pasar por la de esta.
            boton.click(_vaciar_texto, None, [salida, traza], queue=False).then(
                responder_texto, entrada, [salida, traza])

        # La pestaña de visión solo si la configuración activa la incluye. En
        # S08-D no existe: ofrecerla sugeriría que forma parte de lo que midió
        # 18/21, y la vía visual nunca se midió junto con ese retrieval.
        #
        # Tres pasos, y cada uno aparece solo cuando llega su turno: subir y
        # elegir, revisar la lectura, ver el resultado.
        with gr.Tab("📷 Foto de un ejercicio", visible=tutor.cfg.vision):
            confirmado = gr.State(False)

            _titulo(1, "Sube la foto y elige qué necesitas")
            with gr.Row(equal_height=False):
                img = gr.Image(label="Foto de la hoja", type="pil", height=300,
                               scale=3)
                with gr.Column(scale=2):
                    proposito = gr.Radio(
                        [RESOLVER, REVISAR], value=REVISAR,
                        label="¿Qué quieres que haga?",
                        info="Resolver: la hoja trae solo el enunciado. Revisar: "
                             "la hoja trae tus pasos y busco el primero incorrecto.")
                    boton_leer = gr.Button("Leer la hoja", variant="primary")
                    gr.Markdown(
                        "<small>Antes de resolver o revisar te enseño lo que leí "
                        "para que lo corrijas. En M4 el modelo de visión inventó "
                        "una cuenta («24 × 13 = 306») y el tutor diagnosticó sobre "
                        "ese invento: por eso no hay diagnóstico sin tu "
                        "confirmación.</small>")

            aviso = gr.Markdown(elem_classes=["aviso-lectura"])

            with gr.Group(visible=False, elem_classes=["tarjeta"]) as panel_edicion:
                _titulo(2, "Revisa y corrige lo que leí")
                enunciado_box = gr.Textbox(label="Enunciado del ejercicio", lines=2)
                pasos_box = gr.Textbox(
                    label="Tus pasos, uno por línea (la última, la respuesta final)",
                    lines=6)
                boton_confirmar = gr.Button("Confirmar lectura", variant="primary")

            with gr.Group(visible=False) as panel_ilegible:
                boton_otra = gr.Button("Tomar otra foto")

            with gr.Group(visible=False, elem_classes=["tarjeta"]) as panel_resultado:
                _titulo(3, "Resultado")
                boton_accion = gr.Button(ETIQUETA_ACCION[REVISAR], variant="primary")
                salida_img = gr.Markdown(elem_classes=["resultado"])

            with gr.Accordion("Cómo se obtuvo (detalle técnico)", open=False):
                traza_img = gr.Textbox(label="Traza", lines=8, show_label=False)
            with gr.Accordion("Límites de la vía de fotos", open=False):
                gr.Markdown(AVISO_VISION)

            # Leer de nuevo, o cambiar el propósito, invalida lo confirmado y
            # vacía el resultado anterior: nada de una foto vieja sobrevive.
            boton_leer.click(
                _vaciar_texto, None, [salida_img, traza_img], queue=False).then(
                leer_foto, [img, proposito],
                [aviso, panel_edicion, panel_ilegible, pasos_box, enunciado_box,
                 confirmado, traza_img, panel_resultado, proposito, boton_accion])
            proposito.change(
                cambiar_proposito, proposito,
                [confirmado, panel_resultado, salida_img, pasos_box, boton_accion],
                queue=False)
            boton_confirmar.click(
                confirmar_transcripcion, [pasos_box, enunciado_box, proposito],
                [aviso, panel_resultado, confirmado])
            boton_otra.click(
                otra_foto, None,
                [aviso, panel_edicion, panel_ilegible, pasos_box, enunciado_box,
                 confirmado, traza_img, salida_img, panel_resultado, img])
            boton_accion.click(
                _vaciar_texto, None, [salida_img, traza_img], queue=False).then(
                revisar_foto, [pasos_box, confirmado, enunciado_box, proposito],
                [salida_img, traza_img])

        with gr.Tab("Estado"):
            estado_txt = gr.Textbox(label="Comprobación de salud", lines=14,
                                    value=texto_salud(tutor))
            gr.Button("Actualizar").click(lambda: texto_salud(obtener_tutor()),
                                          None, estado_txt)
    demo._estilo_tutor = estilo     # noqa: SLF001 — lo usa `lanzar`
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
    demo = construir()
    estilo = getattr(demo, "_estilo_tutor", {})
    extra = ({"theme": estilo["theme"], "css": estilo["css"]}
             if estilo.get("en_launch") else {})
    return demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=_bandera_activa("TUTOR_GRADIO_SHARE"),
        **extra,
    )


if __name__ == "__main__":
    lanzar()
