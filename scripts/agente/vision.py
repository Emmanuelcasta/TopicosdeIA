"""La lógica de M4 que no necesita GPU ni fotos, para poder probarla.

El notebook de M4 pega este archivo tal cual, igual que hace con `registro.py` o
`nucleo.py`. Fuera queda solo lo que no se puede probar sin el modelo cargado: la
pasada del VLM y el redimensionado de la imagen.

Dentro queda lo que puede equivocarse **en silencio**. La corrida del 22-10-2026
(archivada en `m4/corridas/corrida-01_2026-10-22/`) dio 0/11 y las cuatro causas
fueron de medición, no del modelo:

  1. **El criterio contaba menciones de pasos dentro de la explicación.** Todos
     los casos fallaron con "cayó en la trampa: 'paso 1'" porque la respuesta
     empieza repasando el procedimiento. `vis-05` llegó a decir "Paso 2:
     Incorrecto" —la respuesta esperada— y aun así se contó como fallo. Ahora el
     tutor tiene que emitir una línea `Diagnóstico:` y es **esa línea** la que se
     califica; la explicación se evalúa aparte, como señal, nunca como acierto.

  2. **La numeración que veía el tutor no coincidía con la del eval set.** El eval
     set cuenta la línea "Respuesta final" como un paso más; el VLM la devuelve en
     el campo `respuesta_final`. Para `vis-03`, cuyo error está en el paso 3, el
     tutor solo veía dos pasos: la respuesta correcta era inalcanzable. Ahora
     `pasos_numerados` reconstruye la numeración del eval set.

  3. **El inventario emparejaba por nombre exacto.** `vis-10.jpeg` no satisfacía
     `vis-10.jpg` y el caso quedó sin evaluar. Ahora se empareja por nombre base.

  4. **`lectura_fiel` exigía el mismo número de pasos.** `vis-01` y `vis-03` se
     atribuyeron a fallo de lectura cuando la transcripción era fiel y lo que
     cambiaba era dónde había puesto el VLM la respuesta final.

**Lo que no cambia:** ningún valor esperado del eval set. Los pasos, el
`paso_error` y las claves siguen siendo los que se escribieron antes de ver una
sola respuesta del modelo.

Solo biblioteca estándar: las pruebas corren sin instalar nada.
"""

from __future__ import annotations

import json
import re
import unicodedata

# `vis-12` reutiliza a propósito la hoja de `vis-07` con otra pregunta: se pide un
# dato que no está escrito en ella. El eval set guarda el nombre lógico y el alias
# se resuelve aquí, así que hay 12 casos y 11 fotos, y el eval set no se toca.
ALIAS_IMAGENES = {"vis-12.jpg": "vis-07.jpg"}

# Extensiones que cuentan como "la foto de este caso". El eval set escribe `.jpg`,
# pero una cámara o una conversión producen `.jpeg` o `.png` sin que eso cambie
# nada del caso: lo que identifica a la foto es el nombre base.
EXTENSIONES = (".jpg", ".jpeg", ".png")


def _norm(texto: str) -> str:
    """Minúsculas y sin tildes, igual que el criterio de M2 y M3."""
    t = unicodedata.normalize("NFKD", (texto or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


# --------------------------------------------------------------------------
# Inventario de imágenes
# --------------------------------------------------------------------------

def archivo_de(caso: dict) -> str:
    """El nombre lógico de la foto de este caso, resuelto el alias."""
    return ALIAS_IMAGENES.get(caso["imagen"], caso["imagen"])


def base_de(nombre: str) -> str:
    """`vis-10.jpeg` -> `vis-10`. El nombre base es lo que identifica la foto."""
    for ext in EXTENSIONES:
        if nombre.lower().endswith(ext):
            return nombre[: -len(ext)]
    return nombre


class ImagenAmbigua(Exception):
    """Dos archivos con el mismo nombre base, por ejemplo `vis-10.jpg` y
    `vis-10.jpeg`.

    No se elige uno por orden alfabético: si las dos fotos son distintas, la
    corrida mediría una hoja y el informe hablaría de la otra, y nada avisaría.
    Hay que borrar la que no corresponda.
    """


def inventario(casos, presentes) -> tuple[dict, dict, set]:
    """-> (pedidas, faltan, sobran).

    `pedidas` va del nombre lógico al par (archivo real en disco, casos que lo
    usan), así que una foto compartida aparece una sola vez: es lo que mantiene
    separados el recuento de fotos y el de casos.

    Empareja por nombre base, de modo que `vis-10.jpeg` satisface `vis-10.jpg`.
    Lanza `ImagenAmbigua` si dos archivos comparten el nombre base.
    """
    por_base: dict = {}
    for archivo in sorted(presentes):
        if not archivo.lower().endswith(EXTENSIONES):
            continue
        por_base.setdefault(base_de(archivo), []).append(archivo)

    ambiguas = {b: v for b, v in por_base.items() if len(v) > 1}
    if ambiguas:
        detalle = "; ".join(f"{b}: {', '.join(v)}" for b, v in sorted(ambiguas.items()))
        raise ImagenAmbigua(
            f"hay más de una imagen con el mismo nombre base ({detalle}). "
            "Dejen una sola por caso y vuelvan a ejecutar: elegir una por orden "
            "mediría una hoja y el informe hablaría de la otra.")

    pedidas: dict = {}
    for c in casos:
        logico = archivo_de(c)
        real = (por_base.get(base_de(logico)) or [None])[0]
        entrada = pedidas.setdefault(logico, {"archivo": real, "casos": []})
        entrada["casos"].append(c["id"])

    faltan = {k: v for k, v in pedidas.items() if v["archivo"] is None}
    usados = {v["archivo"] for v in pedidas.values() if v["archivo"]}
    sobran = {a for lista in por_base.values() for a in lista} - usados
    return pedidas, faltan, sobran


# --------------------------------------------------------------------------
# El JSON que devuelve el VLM
# --------------------------------------------------------------------------

def json_de(texto: str):
    """El primer objeto JSON del texto, o None.

    Un modelo de 2B devuelve el JSON envuelto en prosa o en una valla de código
    (pasó con `vis-06`): `json.loads` del texto completo fallaría y se perdería
    una lectura buena.
    """
    i, j = (texto or "").find("{"), (texto or "").rfind("}")
    if i == -1 or j <= i:
        return None
    try:
        return json.loads(texto[i:j + 1])
    except json.JSONDecodeError:
        return None


_PREFIJO_PASO = re.compile(r"^\s*paso\s*\d+\s*[:.\-]\s*", re.IGNORECASE)
_MENCIONA_FINAL = re.compile(r"respuesta\s*final", re.IGNORECASE)


def pasos_numerados(lectura) -> list:
    """Los pasos de la hoja con la MISMA numeración que usa el eval set.

    El eval set cuenta la línea "Respuesta final: ..." como un paso más —el error
    de `vis-03` está justamente ahí—. El VLM a veces la devuelve dentro de
    `pasos` y a veces en el campo `respuesta_final`. Sin unificarlo, el paso que
    hay que señalar no existe en lo que el tutor ve, y el caso es imposible de
    acertar por construcción.

    También quita el "Paso N:" que el VLM escribe a veces dentro del texto, para
    no numerar dos veces.
    """
    js = (lectura or {}).get("json") or {}
    pasos = [_PREFIJO_PASO.sub("", str(p)).strip() for p in (js.get("pasos") or [])]
    final = js.get("respuesta_final")
    if final is not None and not any(_MENCIONA_FINAL.search(p) for p in pasos):
        pasos.append(f"Respuesta final: {final}")
    return pasos


# --------------------------------------------------------------------------
# El diagnóstico: una conclusión explícita, separada de la explicación
# --------------------------------------------------------------------------

# Las cinco categorías, ajustadas al esquema real del eval set:
#   paso N            -> bloques `localizacion` y `compensado` (tienen `paso_error`)
#   sin error         -> bloque `sin_error`
#   ilegible          -> adversarial: la hoja no se puede leer (vis-10)
#   fuera de dominio  -> adversarial: la hoja no es matemáticas (vis-11)
#   dato no disponible-> adversarial: se pide algo que no está escrito (vis-12)
CATEGORIAS = ("paso", "sin error", "ilegible", "fuera de dominio", "dato no disponible")

# Las tres formas de abstenerse. Cualquiera de ellas es "no inventó nada", que es
# la conducta que el bloque adversarial busca; acertar la categoría exacta es más
# exigente y se reporta por separado.
ABSTENCIONES = ("ilegible", "fuera de dominio", "dato no disponible")

# Qué abstención corresponde a cada caso adversarial. Va por id porque la
# distinción está en el enunciado del caso, no en un campo del eval set; la
# prueba `test_todo_adversarial_tiene_categoria` falla si se añade un caso
# adversarial y se olvida ponerlo aquí.
DIAGNOSTICO_ADVERSARIAL = {
    "vis-10": "ilegible",             # foto movida y desenfocada a propósito
    "vis-11": "fuera de dominio",     # una lista de mercado, no un procedimiento
    "vis-12": "dato no disponible",   # se pide una nota que no está en la hoja
}

# Sinónimos que se aceptan como la misma categoría. Un modelo pequeño no va a
# escribir la etiqueta exacta siempre, y penalizarlo por el sinónimo mediría
# obediencia al formato en vez de criterio.
_SINONIMOS = {
    "sin error": ("sin error", "sin errores", "no hay error", "no hay errores",
                  "correcto", "todo correcto", "ninguno"),
    "ilegible": ("ilegible", "no se puede leer", "no legible", "borrosa", "no se lee"),
    "fuera de dominio": ("fuera de dominio", "no es matematicas", "no es un procedimiento",
                         "no es un ejercicio", "fuera de alcance"),
    "dato no disponible": ("dato no disponible", "dato no escrito", "no aparece",
                           "no esta en la hoja", "no disponible", "no tengo esa informacion"),
}

_PAT_DIAGNOSTICO = re.compile(r"diagn[oó]stico\s*:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)
# Acepta "Paso 2", "paso n.º 2" (que `_norm` deja en "paso n.o 2") y
# "paso número 2". Un modelo pequeño escribe el ordinal de varias formas y
# penalizarlo por eso mediría obediencia al formato, no criterio.
_PAT_PASO = re.compile(
    r"paso\s*(?:(?:n[.\u00ba\u00b0o]*|numero|num\.?|#)\s*)?(\d+)", re.IGNORECASE)


def diagnostico_esperado(caso: dict) -> str:
    """La conclusión que este caso espera, derivada del eval set tal como está.

    No se toca ningún valor esperado: `paso_error` y `bloque` ya estaban escritos
    antes de ver una respuesta del modelo.
    """
    if caso.get("paso_error") is not None:
        return f"paso {caso['paso_error']}"
    if caso.get("bloque") == "sin_error":
        return "sin error"
    if caso.get("bloque") == "adversarial":
        try:
            return DIAGNOSTICO_ADVERSARIAL[caso["id"]]
        except KeyError:
            raise KeyError(
                f"el caso adversarial {caso['id']!r} no tiene categoría en "
                "DIAGNOSTICO_ADVERSARIAL: añádanla antes de evaluarlo") from None
    raise ValueError(f"no sé qué espera el caso {caso.get('id')!r}")


def extraer_diagnostico(respuesta: str) -> str | None:
    """La conclusión de la línea `Diagnóstico:`, normalizada. None si no la hay.

    **Solo mira esa línea.** Si el tutor no la emite, el caso no se cuenta como
    acierto aunque la explicación acabe diciendo lo correcto: una conclusión que
    hay que adivinar no sirve a un estudiante, y un criterio que la adivina mide
    la suerte del parser.
    """
    m = _PAT_DIAGNOSTICO.search(respuesta or "")
    if not m:
        return None
    crudo = _norm(m.group(1)).strip(" .;")
    paso = _PAT_PASO.search(crudo)
    if paso:
        return f"paso {int(paso.group(1))}"
    for canonica, variantes in _SINONIMOS.items():
        if any(v in crudo for v in variantes):
            return canonica
    return crudo or None


def pasos_mencionados(texto: str) -> set:
    """Los números de paso que menciona un texto. Señal, nunca criterio."""
    return {int(n) for n in _PAT_PASO.findall(texto or "")}


def _sin_la_linea_de_diagnostico(respuesta: str) -> str:
    return _PAT_DIAGNOSTICO.sub("", respuesta or "")


def calificar_m4(caso: dict, respuesta: str) -> dict:
    """Califica un caso de M4 leyendo la conclusión, no la explicación.

    Devuelve señales independientes, igual que `evaluar_caso` en M2 y M3:

      acierto       la conclusión coincide con la que el caso espera
      abstuvo       concluyó una de las tres formas de "no puedo / no está"
      abstuvo_bien  se abstuvo **y** era un caso en que había que abstenerse
      alucino       concluyó algo concreto y equivocado, sin abstenerse
      sin_conclusion no emitió la línea `Diagnóstico:`

    Y, por separado, sobre la explicación —que se conserva y se evalúa, pero no
    decide el acierto—:

      explicacion_pasos   qué pasos menciona la explicación
      explicacion_justifica  si la explicación habla del paso que concluyó (o no
                             menciona ninguno). No es un juicio de calidad: dice
                             si la conclusión viene argumentada en el texto.
    """
    esperado = diagnostico_esperado(caso)
    obtenido = extraer_diagnostico(respuesta)
    explicacion = _sin_la_linea_de_diagnostico(respuesta)
    mencionados = pasos_mencionados(explicacion)

    acierto = obtenido == esperado
    abstuvo = obtenido in ABSTENCIONES
    debia_abstenerse = esperado in ABSTENCIONES

    if obtenido is None:
        motivo = "no emitió la línea 'Diagnóstico:'"
    elif acierto:
        motivo = f"concluyó {obtenido!r}"
    else:
        motivo = f"esperaba {esperado!r} y concluyó {obtenido!r}"

    paso_concluido = (int(obtenido.split()[1]) if obtenido and obtenido.startswith("paso ")
                      else None)
    return {
        "acierto": acierto,
        "conclusion": obtenido,
        "esperado": esperado,
        "abstuvo": abstuvo,
        "abstuvo_bien": abstuvo and debia_abstenerse,
        "alucino": (not acierto) and not abstuvo,
        "sin_conclusion": obtenido is None,
        "motivo": motivo,
        "explicacion_pasos": sorted(mencionados),
        "explicacion_justifica": (paso_concluido in mencionados if paso_concluido
                                  else not mencionados),
    }


# --------------------------------------------------------------------------
# Instrucciones
# --------------------------------------------------------------------------

INSTRUCCION_VER = (
    "Transcribe EXACTAMENTE lo que está escrito a mano en esta hoja. No corrijas "
    "ningún error: si una cuenta está mal, cópiala mal, tal como aparece.\n"
    "Devuelve SOLO un JSON con esta forma:\n"
    '{"enunciado": "...", "pasos": ["...", "..."], "respuesta_final": "..."}\n'
    "Un elemento de `pasos` por cada línea escrita.\n"
    "Si la hoja no se puede leer, o no contiene un procedimiento matemático, "
    'devuelve {"ilegible": true, "motivo": "..."}.'
)

# El formato de salida que hace medible la tarea.
#
# En la corrida del 22-10 el tutor reescribía el procedimiento en el formato
# "Paso 1: ... Respuesta final: ..." con el que se entrenó la LoRA de M1, en vez
# de revisarlo: continuaba el ejercicio. Pedir una línea final inequívoca cambia
# la tarea de "sigue el procedimiento" a "emite un veredicto", y además hace que
# el criterio no tenga que adivinar nada.
INSTRUCCION_REVISION = (
    "Un estudiante te muestra el procedimiento que escribió a mano, ya transcrito. "
    "Tu trabajo NO es resolver el ejercicio ni reescribirlo: es revisarlo.\n"
    "Primero explica brevemente qué revisaste. Después, en la ÚLTIMA línea, escribe "
    "una conclusión con este formato exacto y nada más:\n"
    "  Diagnóstico: Paso N          (si un paso está mal; N es su número)\n"
    "  Diagnóstico: sin error       (si todos los pasos están bien)\n"
    "  Diagnóstico: ilegible        (si la hoja no se puede leer)\n"
    "  Diagnóstico: fuera de dominio    (si la hoja no es un ejercicio de matemáticas)\n"
    "  Diagnóstico: dato no disponible  (si te piden algo que no está escrito en la hoja)\n"
    "Reglas:\n"
    "- Señala UN solo paso, el primero que esté mal. Si los pasos siguientes "
    "arrastran ese error, no los cuentes como errores aparte.\n"
    "- Si el procedimiento está correcto, concluye 'sin error'. No inventes un "
    "error para tener algo que decir.\n"
    "- No cambies los números del estudiante por los correctos: lo que importa es "
    "dónde se equivocó, no cuál era el resultado bueno.\n"
    "- La respuesta final escrita cuenta como un paso más, el último."
)


def pasos_como_texto(lectura) -> str:
    """La transcripción, en el formato que ve el tutor.

    La numeración la pone esta función, con la convención del eval set (la
    respuesta final es el último paso), y no la copia del VLM: el criterio es
    "Paso N", así que si la numeración dependiera de cómo el modelo partió las
    líneas, el caso se podría perder por una coma.
    """
    js = (lectura or {}).get("json")
    if not js:
        return ("(la transcripción no salió estructurada; esto es lo que se leyó)\n"
                + (lectura or {}).get("crudo", "")[:600])
    if js.get("ilegible"):
        return f"(la hoja no se pudo leer: {js.get('motivo', 'sin motivo')})"
    pasos = pasos_numerados(lectura)
    if not pasos:
        return (f"Enunciado: {js.get('enunciado') or '(no transcrito)'}\n"
                "(la hoja no contiene ningún paso de un procedimiento)")
    lineas = [f"Enunciado: {js.get('enunciado') or '(no transcrito)'}"]
    lineas += [f"Paso {i}: {p}" for i, p in enumerate(pasos, 1)]
    return "\n".join(lineas)


# --------------------------------------------------------------------------
# Lectura o diagnóstico: a qué etapa se atribuye el fallo
# --------------------------------------------------------------------------

def lectura_fiel(caso: dict, lectura) -> bool | None:
    """¿La transcripción conserva el procedimiento que la hoja tenía escrito?

    Devuelve None cuando la pregunta no aplica (los adversariales no tienen un
    procedimiento que transcribir). None no es "sí" ni "no": un caso así nunca se
    atribuye a la lectura, porque no habría nada que leer.

    Compara sobre los pasos **ya numerados** con la convención del eval set. En la
    corrida del 22-10, exigir el mismo número de elementos en `pasos` marcó
    `vis-01` y `vis-03` como fallos de lectura cuando la transcripción era fiel y
    lo único distinto era dónde había puesto el VLM la respuesta final.
    """
    if caso.get("bloque") == "adversarial" or not caso.get("pasos"):
        return None
    js = (lectura or {}).get("json") or {}
    if js.get("ilegible"):
        return False
    pasos = pasos_numerados(lectura)
    if len(pasos) != len(caso["pasos"]):
        return False
    esperado = caso.get("valor_final_escrito")
    if esperado is None:
        return True
    # El valor escrito a mano tiene que seguir ahí: es el error que el tutor debe
    # encontrar. Si el VLM lo "arregló", el caso ya no mide lo que decía medir.
    escrito = " ".join(pasos).replace(" ", "")
    return esperado.replace(" ", "") in escrito


def fuente_del_fallo(acierto: bool, fiel: bool | None) -> str:
    """A qué etapa se le atribuye un fallo. El resultado principal de M4."""
    if acierto:
        return "—"
    return "lectura" if fiel is False else "diagnostico"
