"""La lógica de M4 que no necesita GPU ni fotos, para poder probarla.

El notebook de M4 pega este archivo tal cual, igual que hace con `registro.py` o
`nucleo.py`. Lo que queda **fuera** de aquí es solo lo que no se puede probar sin
el modelo cargado: la pasada del VLM y el redimensionado de la imagen.

Lo que queda **dentro** es precisamente lo que puede equivocarse en silencio:

  archivo_de      el alias de `vis-12`, que es la razón de que haya 12 casos y 11
                  fotos. Si se rompe, un caso se evalúa con la hoja equivocada y
                  el scorecard sale mal sin que nada avise.

  inventario      qué fotos faltan. Decide qué casos entran en las medias; un
                  error aquí cuenta una foto ausente como un fallo del modelo.

  json_de         los modelos pequeños envuelven el JSON en prosa o en vallas de
                  código. `json.loads` a secas falla en la mitad de las lecturas.

  pasos_como_texto  cómo ve el tutor la transcripción. Si aquí se pierde la
                  numeración, "Paso 2" deja de poder acertarse por construcción.

  lectura_fiel    la atribución del fallo a los ojos o al razonamiento. Es el
                  resultado principal de M4 y no se puede reconstruir después.

Solo biblioteca estándar: las pruebas corren sin instalar nada.
"""

from __future__ import annotations

import json

# `vis-12` reutiliza a propósito la hoja de `vis-07` con otra pregunta: se pide un
# dato que no está escrito en ella. El eval set guarda el nombre lógico y el alias
# se resuelve aquí, así que hay 12 casos y 11 fotos, y el eval set no se toca.
ALIAS_IMAGENES = {"vis-12.jpg": "vis-07.jpg"}


def archivo_de(caso: dict) -> str:
    """El archivo que hay que tener en disco para este caso."""
    return ALIAS_IMAGENES.get(caso["imagen"], caso["imagen"])


def inventario(casos, presentes) -> tuple[dict, dict, set]:
    """-> (pedidas, faltan, sobran).

    `pedidas` va de archivo a la lista de casos que lo usan, así que un archivo
    compartido aparece una sola vez: es lo que hace que el conteo de fotos y el de
    casos no se confundan.
    """
    pedidas: dict = {}
    for c in casos:
        pedidas.setdefault(archivo_de(c), []).append(c["id"])
    presentes = set(presentes)
    faltan = {a: ids for a, ids in pedidas.items() if a not in presentes}
    return pedidas, faltan, presentes - set(pedidas)


def json_de(texto: str):
    """El primer objeto JSON del texto, o None.

    Un modelo de 2B devuelve el JSON envuelto en prosa, en una valla de código o
    con una frase detrás. `json.loads` del texto completo falla en todos esos
    casos, y una lectura buena se perdería por la envoltura.
    """
    i, j = (texto or "").find("{"), (texto or "").rfind("}")
    if i == -1 or j <= i:
        return None
    try:
        return json.loads(texto[i:j + 1])
    except json.JSONDecodeError:
        return None


INSTRUCCION_VER = (
    "Transcribe EXACTAMENTE lo que está escrito a mano en esta hoja. No corrijas "
    "ningún error: si una cuenta está mal, cópiala mal, tal como aparece.\n"
    "Devuelve SOLO un JSON con esta forma:\n"
    '{"enunciado": "...", "pasos": ["...", "..."], "respuesta_final": "..."}\n'
    "Un elemento de `pasos` por cada línea escrita.\n"
    "Si la hoja no se puede leer, o no contiene un procedimiento matemático, "
    'devuelve {"ilegible": true, "motivo": "..."}.'
)

INSTRUCCION_REVISION = (
    "Un estudiante te muestra el procedimiento que escribió a mano, ya transcrito.\n"
    "- Revísalo paso por paso. Si encuentras un error, di en qué PASO está, con el "
    'formato "Paso N", y explica por qué está mal.\n'
    "- Si todos los pasos son correctos, dilo claramente. No inventes un error.\n"
    "- Si la transcripción dice que la hoja es ilegible, o que no contiene un "
    "procedimiento matemático, o si te preguntan por un dato que no aparece en "
    "ella, responde que no tienes esa información. No lo adivines."
)


def pasos_como_texto(lectura) -> str:
    """La transcripción, en el formato que ve el tutor.

    La numeración se escribe aquí y no la copia el VLM: el criterio de acierto es
    "Paso N", así que si la numeración dependiera de lo que el modelo escribió a
    mano, el caso se podría perder por una coma.
    """
    js = (lectura or {}).get("json")
    if not js:
        return ("(la transcripción no salió estructurada; esto es lo que se leyó)\n"
                + (lectura or {}).get("crudo", "")[:600])
    if js.get("ilegible"):
        return f"(la hoja no se pudo leer: {js.get('motivo', 'sin motivo')})"
    lineas = [f"Enunciado: {js.get('enunciado') or '(no transcrito)'}"]
    for i, paso in enumerate(js.get("pasos") or [], 1):
        lineas.append(f"Paso {i}: {paso}")
    if js.get("respuesta_final") is not None:
        lineas.append(f"Respuesta final escrita: {js['respuesta_final']}")
    return "\n".join(lineas)


def lectura_fiel(caso: dict, lectura) -> bool | None:
    """¿La transcripción conserva el procedimiento que la hoja tenía escrito?

    Devuelve None cuando la pregunta no aplica (los adversariales no tienen un
    procedimiento que transcribir). None no es "sí" ni "no": un caso así nunca se
    atribuye a la lectura, porque no habría nada que leer.

    Es deliberadamente estricta en el número de pasos: si el VLM fusionó dos
    líneas, la numeración se desplaza y "Paso 2" ya no apunta a lo mismo, aunque
    todos los números estén presentes.
    """
    if caso.get("bloque") == "adversarial" or not caso.get("pasos"):
        return None
    js = (lectura or {}).get("json") or {}
    if js.get("ilegible"):
        return False
    pasos = js.get("pasos") or []
    if len(pasos) != len(caso["pasos"]):
        return False
    esperado = caso.get("valor_final_escrito")
    if esperado is None:
        return True
    # El valor puede acabar en la respuesta final o en el último paso, según cómo
    # el VLM haya partido las líneas; cualquiera de los dos cuenta como fiel.
    escrito = str(js.get("respuesta_final", "")) + " " + " ".join(map(str, pasos))
    return esperado.replace(" ", "") in escrito.replace(" ", "")


def fuente_del_fallo(acierto: bool, fiel: bool | None) -> str:
    """A qué etapa se le atribuye un fallo. El resultado principal de M4."""
    if acierto:
        return "—"
    return "lectura" if fiel is False else "diagnostico"
