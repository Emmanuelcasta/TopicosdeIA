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
from pathlib import Path

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
    # Solo se añade la respuesta final si HAY pasos. Con `pasos: []` —la lista de
    # mercado de `vis-11`— añadirla fabricaba un "Paso 1: Respuesta final: Arroz"
    # y al tutor se le presentaba un procedimiento de un paso donde la hoja no
    # tenía ninguno, lo que hace casi imposible concluir "fuera de dominio".
    if pasos and final is not None and not any(_MENCIONA_FINAL.search(p) for p in pasos):
        pasos.append(f"Respuesta final: {final}")
    return pasos


# --------------------------------------------------------------------------
# ¿Es fiable esta lectura?
# --------------------------------------------------------------------------

_PAT_NUMERO = re.compile(r"-?\d+(?:[.,]\d+)?(?:/\d+)?")


def numeros_de(texto: str) -> list:
    """Los números de un texto, normalizados. La huella de una transcripción."""
    return [n.replace(",", ".") for n in _PAT_NUMERO.findall(texto or "")]


def huella_lectura(lectura) -> tuple:
    """Lo que tiene que coincidir entre dos lecturas de la MISMA foto: cuántos
    pasos y qué números. La redacción puede variar; los números, no."""
    js = (lectura or {}).get("json") or {}
    if js.get("ilegible"):
        return ("ilegible",)
    pasos = js.get("pasos") or []
    texto = " ".join(map(str, pasos)) + " " + str(js.get("respuesta_final", ""))
    return (len(pasos), tuple(numeros_de(texto)))


def lecturas_consistentes(a, b) -> bool:
    """¿Dos lecturas de la misma foto dicen lo mismo?

    **Necesaria pero NO suficiente, y hay prueba.** En la corrida del 24-10 esta
    señal falló exactamente donde tenía que servir: ante la foto desenfocada de
    `vis-10` el VLM inventó `24 x 13 = 306` (la hoja dice 302) y lo inventó **las
    dos veces**. La invención no era ruido de muestreo: estaba anclada en el
    prior del modelo —la hoja es una versión borrosa de un ejercicio de 24 × 13—
    y por eso es estable.

    Se conserva porque es barata y sí caza las lecturas inestables, pero ya no se
    usa como única defensa.
    """
    return huella_lectura(a) == huella_lectura(b)


def atipicos_por_mad(valores: dict, k: float = 3.0) -> set:
    """Las claves cuyo valor queda muy por debajo del resto del conjunto.

    Mediana y desviación absoluta mediana, el criterio robusto de siempre:
    `valor < mediana - k · MAD`. Es **relativo al propio conjunto**, no un umbral
    absoluto elegido a mano, y eso es deliberado: cualquier constante calibrada
    con estas 11 fotos sería un umbral ajustado al conjunto de prueba.

    `k = 3` es la convención estadística habitual, no un valor afinado para que
    `vis-10` caiga del lado correcto. **Si separa o no esa foto es algo que hay
    que medir, y no debe tocarse para que salga.**
    """
    if len(valores) < 4:          # con tres o menos no hay "resto del conjunto"
        return set()
    import statistics
    xs = sorted(valores.values())
    mediana = statistics.median(xs)
    mad = statistics.median([abs(x - mediana) for x in xs]) * 1.4826
    if mad == 0:
        return set()
    return {clave for clave, v in valores.items() if v < mediana - k * mad}


# --------------------------------------------------------------------------
# Caché de lecturas y señal 4 con UNA sola foto
# --------------------------------------------------------------------------

# Lo que una lectura guardada debe traer para que las cuatro señales decidan.
# La corrida del 25-10 (Visual_n, celda 34) reusó `m4/lecturas_m4.json`, escrito
# antes de la salvaguarda: tenía `consistente` pero no `confianza_numeros` ni
# `legible_declarada`. Solo las 3 fotos del piloto se releyeron, la señal 3 quedó
# en "?" para 8 fotos y la 4 no pudo decidir (necesita ≥ 4 valores): `vis-10`
# pasó como "ok".
LLAVES_LECTURA_VIGENTE = ("consistente", "confianza_numeros", "legible_declarada")


def lectura_vigente(lectura) -> bool:
    """¿Se puede reusar esta lectura de la caché sin volver a pasar el VLM?

    Solo si trae todas las señales. Una lectura de un esquema anterior se relee:
    reusarla apaga la salvaguarda sin avisar.
    """
    return isinstance(lectura, dict) and all(k in lectura for k in LLAVES_LECTURA_VIGENTE)


def referencia_de_confianza(valores, k: float = 3.0) -> dict:
    """La distribución de referencia para la señal 4 cuando llega UNA foto.

    `atipicos_por_mad` compara cada foto con el resto de su lote; en la demo o en
    producción llega una foto sola y no hay lote. La referencia se construye con
    la confianza de fotos **nítidas, escritas a mano, con la misma cámara** y que
    **no** son del eval set (por ejemplo, las hojas `dev-*` escritas a mano).

    No sirve el conjunto de calibración impreso: con Qwen2-VL-2B las hojas
    impresas salen con confianzas casi nulas y una MAD muy pequeña, y contra esa
    referencia cinco hojas manuscritas legibles del eval set quedaban marcadas
    como no fiables.
    """
    import statistics
    xs = sorted(float(v) for v in valores if v is not None)
    if len(xs) < 4:
        raise ValueError(f"hacen falta al menos 4 fotos de referencia; hay {len(xs)}")
    mediana = statistics.median(xs)
    mad = statistics.median([abs(x - mediana) for x in xs]) * 1.4826
    return {"mediana": mediana, "mad": mad, "k": k, "n": len(xs),
            "umbral": mediana - k * mad}


def confianza_atipica_frente_a(valor, referencia) -> bool | None:
    """Señal 4 para una foto suelta: ¿su confianza cae bajo `mediana − k·MAD` de
    la referencia? `None` si no hay con qué decidir (sin valor, sin referencia o
    MAD nula): en ese caso la señal no se dispara y deciden las otras tres."""
    if valor is None or not referencia or not referencia.get("mad"):
        return None
    return float(valor) < referencia["mediana"] - referencia["k"] * referencia["mad"]


# Dos ideas tentadoras que NO se usan, y por qué:
#
#   · **Comprobar la aritmética de la transcripción.** Suena bien y es inservible
#     aquí: las hojas llevan cuentas mal A PROPÓSITO. En `vis-10` el invento es
#     `24 × 13 = 306` y lo que la hoja dice es `24 × 13 = 302`; las dos son
#     aritméticamente falsas, así que la comprobación no distingue la lectura de
#     la invención. Peor: marcaría como dudosas justo las hojas con error, que son
#     las que el módulo existe para medir.
#
#   · **Un umbral de nitidez sobre la imagen** (varianza del laplaciano y
#     similares). Habría que calibrarlo, y calibrarlo con estas 11 fotos es
#     ajustarlo al conjunto de prueba. Si algún día hay un conjunto de calibración
#     aparte, vuelve a ser una opción razonable.


def lectura_no_fiable(lectura) -> tuple:
    """-> (no_fiable, motivo). Si es no fiable, el tutor NO debe tratarla como un
    procedimiento legible: se le dice que la hoja no se pudo leer.

    Cuatro señales de naturaleza distinta, porque la del 24-10 demostró que una
    sola no basta. Basta que **una** se dispare: ante la duda no se le entrega al
    tutor un procedimiento que puede ser inventado.

      1. la transcripción no salió estructurada;
      2. el propio VLM declaró la hoja ilegible al transcribir;
      3. preguntado **aparte** si la hoja se lee —sin transcribir, otra pregunta y
         otro encuadre, para no arrastrar el prior de la transcripción—, dijo que
         no (`legible_declarada = False`);
      4. la confianza del modelo en los **números** que transcribió es atípica
         frente a las demás fotos del conjunto (`confianza_atipica`). Un número
         leído de una cifra nítida sale con más confianza que uno rellenado.

    La señal de consistencia entre dos generaciones queda como quinta, degradada:
    sirve, pero no se le confía la decisión.

    Nunca acredita un caso: una lectura no fiable no vuelve correcta ninguna
    respuesta, solo evita juzgar al tutor sobre un invento del VLM.
    """
    if not lectura or not lectura.get("ok"):
        return True, "la transcripción no salió estructurada"
    js = lectura.get("json") or {}
    if js.get("ilegible"):
        return True, js.get("motivo") or "el modelo la declaró ilegible"
    if lectura.get("legible_declarada") is False:
        return True, ("preguntado aparte, el modelo dijo que la hoja no se lee: "
                      + (lectura.get("motivo_legibilidad") or "sin motivo"))
    if lectura.get("confianza_atipica") is True:
        return True, ("la confianza del modelo en los números transcritos es "
                      "atípicamente baja frente a las demás fotos")
    if lectura.get("consistente") is False:
        return True, ("dos lecturas de la misma foto no coinciden: la transcripción "
                      "no es de fiar")
    return False, ""


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


def diagnostico_valido(conclusion) -> bool:
    """¿La conclusión es una de las cinco categorías, y no texto libre?

    `vis-04` emitió "Diagnóstico: equivoco el producto 1248…": una línea que
    señalaba el error sin nombrar el paso. Eso no es una conclusión válida, pero
    tampoco es una invención, y contarlo como alucinación exageraba el fallo.
    """
    if not conclusion:
        return False
    return conclusion.startswith("paso ") or conclusion in ("sin error",) + ABSTENCIONES


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
      alucino       concluyó una categoría válida, concreta y equivocada
      sin_conclusion no emitió la línea `Diagnóstico:`
      conclusion_invalida  la emitió, pero en texto libre que no es ninguna de
                           las cinco categorías

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

    valida = diagnostico_valido(obtenido)
    acierto = valida and obtenido == esperado
    abstuvo = obtenido in ABSTENCIONES
    debia_abstenerse = esperado in ABSTENCIONES

    if obtenido is None:
        motivo = "no emitió la línea 'Diagnóstico:'"
    elif not valida:
        motivo = f"conclusión no válida: {obtenido[:48]!r}"
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
        # Alucinar es concluir algo CONCRETO y equivocado. No emitir conclusión, o
        # emitirla en texto libre, es un fallo de formato y se cuenta aparte: son
        # problemas distintos y se arreglan de forma distinta.
        "alucino": valida and (not acierto) and not abstuvo,
        "sin_conclusion": obtenido is None,
        "conclusion_invalida": (obtenido is not None) and not valida,
        "motivo": motivo,
        "explicacion_pasos": sorted(mencionados),
        "explicacion_justifica": (paso_concluido in mencionados if paso_concluido
                                  else not mencionados),
    }


# --------------------------------------------------------------------------
# Adaptadores: comprobar la ruta de verdad
# --------------------------------------------------------------------------

# Qué tiene que haber dentro de un adaptador LoRA guardado con `save_pretrained`.
ARCHIVOS_ADAPTADOR = ("adapter_config.json", "adapter_model.safetensors")


def revisar_adaptador(ruta, requeridos=ARCHIVOS_ADAPTADOR) -> tuple:
    """-> (ok, motivo). Mira el disco y dice **qué** falta, si falta algo.

    Por qué existe. La corrida del 25-10 imprimió "No está
    adaptadores/qwen-lora-diagnostico" con los dos archivos presentes en Drive:
    el mensaje afirmaba un hecho del sistema de archivos que el código nunca
    había comprobado —la condición solo miraba si el adaptador estaba *cargado*,
    y nadie lo cargaba—. Un aviso que no verifica lo que afirma manda a buscar el
    fallo donde no está, y costó una corrida entera.
    """
    p = Path(ruta)
    if not p.exists():
        return False, f"no existe el directorio {p}"
    if not p.is_dir():
        return False, f"{p} existe pero no es un directorio"
    faltan = [n for n in requeridos if not (p / n).exists()]
    # `save_pretrained` escribe .safetensors; los adaptadores antiguos traen .bin.
    if faltan == ["adapter_model.safetensors"] and (p / "adapter_model.bin").exists():
        return True, f"{p}: pesos en adapter_model.bin (formato antiguo)"
    if faltan:
        hay = sorted(f.name for f in p.iterdir()) or ["(vacío)"]
        return False, f"en {p} faltan {faltan}; lo que hay: {hay}"
    return True, f"{p}: completo"


def elegir_adaptador(disponibles, preferido: str, alternativo: str) -> str:
    """El preferido si está cargado; si no, el alternativo. Sin inventar motivos."""
    return preferido if preferido in disponibles else alternativo


# --------------------------------------------------------------------------
# La puerta del ensayo de desarrollo
# --------------------------------------------------------------------------

class PuertaNoSuperada(RuntimeError):
    """El ensayo de formato sobre los casos `dev-*` no se superó, o no se ejecutó.

    Por qué es una excepción y no un aviso: en la corrida del 24-10 el ensayo dio
    **0/3** y la celda imprimió "PAREN AQUÍ"… y la ejecución siguió hasta evaluar
    los 12 casos. Un `print` no detiene nada. El resultado de esos 12 estaba
    decidido antes de empezar: 7 de 12 sin conclusión.
    """


def revisar_puerta_dev(conclusiones) -> dict:
    """`conclusiones`: iterable de (id, conclusión emitida). -> dict de la puerta.

    La puerta se supera si **todos** los casos de desarrollo dan una conclusión
    válida. No se exige que acierten: lo que se comprueba es el formato, porque
    es la condición sin la cual el scorecard de los 12 no mide el criterio del
    tutor sino su obediencia al formato de salida.
    """
    filas = [(cid, c, diagnostico_valido(c)) for cid, c in conclusiones]
    validos = sum(1 for _, _, ok in filas if ok)
    return {"filas": filas, "validos": validos, "total": len(filas),
            "ok": len(filas) > 0 and validos == len(filas)}


def exigir_puerta_dev(puerta, permitir_sin_puerta: bool = False) -> None:
    """Lanza `PuertaNoSuperada` si la puerta no se superó o no se ejecutó.

    Va al principio de las celdas de evaluación, así que la parada no depende de
    cómo el entorno trate una excepción en otra celda: si alguien ejecuta la
    celda de los 12 casos directamente, se detiene igual.

    `permitir_sin_puerta=True` es una salida deliberada para recoger datos de
    diagnóstico. No se silencia: quien la usa queda registrado en los artefactos
    (`puerta_dev: "no superada"`), de modo que cualquier cifra que salga de esa
    corrida lleva la marca encima.
    """
    if permitir_sin_puerta:
        return
    if puerta is None:
        raise PuertaNoSuperada(
            "el ensayo de formato sobre los casos de desarrollo no se ha ejecutado. "
            "Ejecuten la celda 7.1 antes de evaluar los 12 casos: sin ella no se "
            "sabe si el tutor emite la línea `Diagnóstico:`, y un scorecard sin "
            "eso mide el formato de salida y no el criterio del tutor.")
    if not puerta["ok"]:
        detalle = ", ".join(f"{cid}->{c!r}" for cid, c, ok in puerta["filas"] if not ok)
        raise PuertaNoSuperada(
            f"el ensayo de formato dio {puerta['validos']}/{puerta['total']} en los "
            f"casos de desarrollo ({detalle}). La evaluación de los 12 casos se "
            "detiene aquí a propósito: con el formato roto su total no mediría el "
            "criterio del tutor.\n"
            "Lo que toca es el prompt o el entrenamiento, y la decisión se toma con "
            "los casos de desarrollo, no con los 12: afinar contra los 12 los "
            "convierte en conjunto de entrenamiento.\n"
            "Si quieren correrlos igualmente para recoger diagnóstico, definan "
            "PERMITIR_SIN_PUERTA = True en una celda nueva; el scorecard quedará "
            "marcado como obtenido sin superar la puerta.")


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

# El formato de salida, y la evidencia que lo decidió.
#
# En las once respuestas completas de la corrida del 22-10, OCHO terminan
# copiando la última etiqueta que el prompt les mostró:
#
#     vis-01 -> "Respuesta final escrita: 24 x 13 lápices"
#     vis-03 -> "Respuesta final escrita: 36000"
#     vis-05 -> "Respuesta final escrita: 60 metros cuadrados"
#
# El modelo no desobedecía: estaba **continuando el formato de la transcripción**,
# que es lo que la LoRA de M1 le enseñó a hacer con un texto que empieza por
# "Paso 1:" y acaba en "Respuesta final:". De ahí las tres decisiones:
#
#   1. La exigencia del formato va **al final**, pegada a la generación, para que
#      lo último que el modelo lea sea la conclusión que se le pide y no la
#      respuesta final del estudiante.
#   2. Un ejemplo resuelto, porque un modelo de 1.5B copia formatos mejor de lo
#      que sigue instrucciones abstractas.
#   3. Se le dice explícitamente que NO escriba "Respuesta final", que es la
#      etiqueta que lo descarrila.
#
# Nada de esto se ajustó a los 12 casos: el ejemplo usa un ejercicio inventado
# (pan y panes), no una de las hojas, y las cinco categorías salen del esquema del
# eval set, no de lo que el modelo respondió.

INSTRUCCION_REVISION = (
    "Eres un tutor de matemáticas. Un estudiante te muestra el procedimiento que "
    "escribió a mano, ya transcrito. Tu trabajo NO es resolver el ejercicio ni "
    "reescribirlo: es revisarlo y dar un veredicto.\n"
    "Reglas:\n"
    "- Señala UN solo paso, el primero que esté mal. Si los siguientes arrastran "
    "ese error, no los cuentes aparte.\n"
    "- Si el procedimiento está correcto, dilo. No inventes un error para tener "
    "algo que decir.\n"
    "- No cambies los números del estudiante por los correctos: importa dónde se "
    "equivocó, no cuál era el resultado bueno.\n"
    "- La respuesta final escrita cuenta como un paso más, el último.\n"
    "- NO escribas 'Respuesta final': eso es lo que escribió el estudiante, no tu "
    "veredicto."
)

# NO hay ejemplo resuelto, y la corrida del 24-10 explica por qué.
#
# El ejemplo que había —una panadería, 15 panes por bandeja— apareció COPIADO en
# la respuesta a `vis-11`, la lista de mercado:
#
#     Paso 1: Multiplicamos las bandejas por los panes de cada una: 15 × 4.
#     Paso 2: 15 × 4 = 60.
#     Diagnóstico: Paso 2
#
# Copió el ejercicio entero y el veredicto del ejemplo, y "corrigió" su 54 a 60.
# Un modelo que ante una entrada degenerada se agarra a lo último parecido que vio
# convierte cualquier ejemplo resuelto en una respuesta posible.
#
# Así que del ejemplo queda solo la FORMA de la última línea, sin ningún
# procedimiento que copiar. Enseñar el formato no exige enseñar un ejercicio.
EJEMPLO_REVISION = (
    "La última línea de tu respuesta tiene que verse exactamente así, con una de "
    "las cinco etiquetas:\n"
    "  Diagnóstico: Paso 2"
)

# Esto va LO ÚLTIMO, inmediatamente antes de generar.
RECORDATORIO_FORMATO = (
    "Termina SIEMPRE con una última línea con este formato exacto, elige una:\n"
    "  Diagnóstico: Paso N              (un paso está mal; N es su número)\n"
    "  Diagnóstico: sin error           (todos los pasos están bien)\n"
    "  Diagnóstico: ilegible            (la hoja no se pudo leer)\n"
    "  Diagnóstico: fuera de dominio    (la hoja no es un ejercicio de matemáticas)\n"
    "  Diagnóstico: dato no disponible  (piden algo que no está escrito en la hoja)\n"
    "Esa línea es obligatoria, también cuando el procedimiento esté correcto."
)

# El reintento: si no hay conclusión válida, se pide SOLO la línea. No se le
# permite recalcular ni cambiar de opinión sobre el fondo —se le devuelve su
# propia explicación y se le pide que la resuma en una de las cinco etiquetas—,
# y queda registrado cuáles necesitaron reintento, porque un sistema que necesita
# dos pasadas para contestar no es lo mismo que uno que contesta de una.
PETICION_CONCLUSION = (
    "Esta fue tu revisión:\n"
    "---\n{explicacion}\n---\n"
    "No terminaste con la línea obligatoria. Escribe AHORA solo esa línea, sin "
    "ninguna explicación, coherente con lo que acabas de decir:\n"
    "  Diagnóstico: Paso N  |  Diagnóstico: sin error  |  Diagnóstico: ilegible  |  "
    "Diagnóstico: fuera de dominio  |  Diagnóstico: dato no disponible"
)


def prompt_revision(pregunta: str, lectura) -> str:
    """El prompt completo, con la exigencia del formato al final."""
    return (f"{INSTRUCCION_REVISION}\n\n{EJEMPLO_REVISION}\n\n"
            f"Ahora la hoja de verdad.\n{pasos_como_texto(lectura)}\n\n"
            f"Pregunta del estudiante: {pregunta}\n\n{RECORDATORIO_FORMATO}")


def asegurar_diagnostico(respuesta: str, reintentar) -> tuple:
    """-> (respuesta, reintento_usado).

    `reintentar(prompt) -> texto`. Se llama UNA sola vez y solo si la respuesta no
    trae una conclusión válida. La línea que devuelva se añade al final de la
    explicación original, que se conserva entera.
    """
    if diagnostico_valido(extraer_diagnostico(respuesta)):
        return respuesta, False
    linea = reintentar(PETICION_CONCLUSION.format(explicacion=(respuesta or "").strip()[:1200]))
    return f"{(respuesta or '').rstrip()}\n{(linea or '').strip()}", True


def pasos_como_texto(lectura) -> str:
    """La transcripción, en el formato que ve el tutor.

    La numeración la pone esta función, con la convención del eval set (la
    respuesta final es el último paso), y no la copia del VLM: el criterio es
    "Paso N", así que si la numeración dependiera de cómo el modelo partió las
    líneas, el caso se podría perder por una coma.
    """
    no_fiable, motivo = lectura_no_fiable(lectura)
    if no_fiable:
        return (f"(la hoja NO se pudo leer: {motivo}. No hay un procedimiento "
                "legible que revisar.)")
    js = lectura.get("json") or {}
    pasos = pasos_numerados(lectura)
    if not pasos:
        return (f"Lo único que se alcanza a leer en la hoja: "
                f"{js.get('enunciado') or '(nada)'}\n"
                "(no hay ningún paso de un procedimiento matemático)")
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
    # Una lectura no fiable es un fallo de lectura, aunque el caso sea adversarial:
    # es lo que hace que `vis-10` se atribuya a la etapa de ver y no al tutor.
    if lectura_no_fiable(lectura)[0]:
        return False
    if caso.get("bloque") == "adversarial" or not caso.get("pasos"):
        return None
    js = (lectura or {}).get("json") or {}
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


# --------------------------------------------------------------------------
# Punto de entrada único: foto -> diagnóstico (o "no pude leer tu hoja")
# --------------------------------------------------------------------------

MENSAJE_NO_PUDE_LEER = (
    "No pude leer tu hoja con seguridad ({motivo}). Para no señalarte un error "
    "que no escribiste, prefiero no diagnosticar todavía: toma otra foto de "
    "frente, con buena luz y sin que se mueva."
)


def revisar_foto(imagen, pregunta: str, *, leer, diagnosticar,
                 referencia=None, archivo: str = "foto") -> dict:
    """Una foto y una pregunta -> lo que se le muestra al estudiante.

    Encadena lo que el notebook ya tiene, en el orden en que debe ocurrir con una
    foto suelta: leer (`leer_dos_veces`), decidir si la lectura es fiable
    (`lectura_no_fiable`) y solo entonces diagnosticar (`sistema_multimodal`).
    Las dos funciones se inyectan para poder probar el flujo sin GPU.

    Con una sola foto la señal 4 necesita `referencia` (ver
    `referencia_de_confianza`); sin ella deciden las otras tres y el resultado lo
    dice en `senal_4`.
    """
    import time
    t0 = time.perf_counter()
    lectura = leer(imagen, archivo)
    if referencia is not None:
        atipica = confianza_atipica_frente_a(lectura.get("confianza_numeros"), referencia)
        lectura["confianza_atipica"] = atipica is True
        senal_4 = "sin decidir" if atipica is None else ("atípica" if atipica else "normal")
    else:
        senal_4 = "sin referencia"
    t_lectura = time.perf_counter() - t0
    no_fiable, motivo = lectura_no_fiable(lectura)
    salida = {"archivo": archivo, "lectura": pasos_numerados(lectura),
              "senal_4": senal_4, "segundos_lectura": round(t_lectura, 2)}
    if no_fiable:
        salida.update(estado="no_pude_leer", motivo=motivo, diagnostico="ilegible",
                      respuesta=MENSAJE_NO_PUDE_LEER.format(motivo=motivo))
    else:
        respuesta = diagnosticar(pregunta, lectura)
        salida.update(estado="diagnostico", motivo="",
                      diagnostico=extraer_diagnostico(respuesta), respuesta=respuesta)
    salida["segundos_total"] = round(time.perf_counter() - t0, 2)
    return salida
