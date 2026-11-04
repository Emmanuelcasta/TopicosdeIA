"""El tutor real, montado para la aplicación.

Este módulo **no implementa nada del tutor**: lo monta. Toda la lógica vive donde
ya estaba —`scripts/agente/`— y aquí solo se conecta:

    registro.py     el Tool Registry de S10
    herramientas.py las 15 herramientas con aritmética exacta
    nucleo.py       `AgenteTutor`, el bucle ReAct
    documentos.py   el retrieval como herramienta
    prompts.py      los prompts versionados
    enrutador.py    consulta vs problema (S08)
    vision.py       la vía visual de M4
    verificador_aritmetico.py  la comprobación determinista de las cuentas

Duplicar cualquiera de esas piezas dentro de la interfaz sería garantizar que la
demo y lo medido se separen. El corpus se obtiene igual que en
`scripts/trayectorias_fuente.py`: ejecutando la fuente canónica de
`nb_comun_rag`, no copiándola.

**Carga perezosa y degradación explícita.** El modelo pesa y puede no estar: el
pipeline arranca igual y reporta su estado. La interfaz pregunta `estado()` y
dice la verdad en vez de caerse.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "scripts" / "agente"))

from configuraciones import Configuracion, obtener  # noqa: E402

MODELO_BASE = os.environ.get("TUTOR_MODELO_BASE", "Qwen/Qwen2.5-1.5B-Instruct")
RUTAS_ADAPTADOR = {
    "m1": os.environ.get("TUTOR_ADAPTADOR_M1", "adaptadores/qwen-lora"),
    "agente": os.environ.get("TUTOR_ADAPTADOR_AGENTE", "adaptadores/qwen-lora-agente"),
}
VLM_ID = os.environ.get("TUTOR_VLM", "Qwen/Qwen2-VL-2B-Instruct")
CUANTIZAR = os.environ.get("TUTOR_CUANTIZAR", "").lower() in ("1", "true", "int4")
# La configuración se ELIGE, no se deduce de qué adaptador haya en el disco.
CONFIG = os.environ.get("TUTOR_CONFIG")

# El lado mayor con el que se lee una foto. La app y el evaluador tienen que
# ver los MISMOS píxeles: antes la interfaz pasaba el PIL crudo de Gradio
# —4160 px en una foto de móvil— y el evaluador reducía a 1024, así que
# `caso-01` se rechazaba en la una y se leía en el otro.
LADO_MAYOR_FOTO = int(os.environ.get("TUTOR_LADO_MAYOR", "1024"))


def normalizar_foto(imagen, lado_mayor: int = LADO_MAYOR_FOTO):
    """Orientación EXIF corregida y lado mayor acotado. -> (imagen, dict).

    El dict dice qué se hizo, para que la traza no tenga que adivinarlo. Si PIL
    no está o la imagen no es una imagen, se devuelve tal cual: normalizar es
    una mejora, no un requisito para que la vía visual arranque.
    """
    info = {"exif": False, "px_antes": None, "px_despues": None}
    try:
        from PIL import Image, ImageOps
    except Exception:  # noqa: BLE001
        return imagen, info
    if not isinstance(imagen, Image.Image):
        return imagen, info
    info["px_antes"] = list(imagen.size)
    try:
        girada = ImageOps.exif_transpose(imagen)
        info["exif"] = girada.size != imagen.size or girada is not imagen
        imagen = girada
    except Exception:  # noqa: BLE001
        pass
    if imagen.mode != "RGB":
        imagen = imagen.convert("RGB")
    if lado_mayor > 0 and max(imagen.size) > lado_mayor:
        escala = lado_mayor / max(imagen.size)
        imagen = imagen.resize(
            (max(1, round(imagen.width * escala)), max(1, round(imagen.height * escala))),
            Image.LANCZOS)
    info["px_despues"] = list(imagen.size)
    return imagen, info


# --------------------------------------------------------------------------
# Aritmética directa
#
# `s08d_verificador` monta las 18 herramientas, pero el adaptador `m1` se
# entrenó para RAG, no para emitir `<tool_call>`: ante «¿Cuánto es 3/4 + 1/8?»
# contestaba `0.75` de cabeza y la traza registraba cero llamadas. Cuando la
# pregunta ES una cuenta no hay nada que decidir, así que se ejecuta la
# herramienta registrada y se responde con su resultado exacto. El modelo sigue
# a cargo de todo lo demás, y la observación que se guarda es la que devuelve
# `RegistroHerramientas.ejecutar`: la misma que registraría el agente.
#
# Solo se activa con `herramientas=True`. `s08d` se midió sin herramientas y
# no se toca.
# --------------------------------------------------------------------------

_PREFIJOS_CUENTA = re.compile(
    r"^(?:cu[aá]nto\s+(?:es|da|vale|son)|cu[aá]nto|calcula(?:r)?|resuelve|opera|"
    r"(?:el\s+)?resultado\s+de)\b[\s:]*", re.IGNORECASE)
_SOLO_CUENTA = re.compile(r"^[0-9\s.,/+\-*×xX÷:^()√]+$")
_UNA_FRACCION = re.compile(r"^-?\d+\s*/\s*\d+$")
_DOS_FRACCIONES = re.compile(
    r"^(-?\d+\s*/\s*\d+)\s*([+\-*×xX÷])\s*(-?\d+\s*/\s*\d+)$")
_OPERACION_FRACCION = {"+": "suma", "-": "resta", "*": "multiplicacion",
                       "×": "multiplicacion", "x": "multiplicacion",
                       "X": "multiplicacion", "÷": "division"}


def cuenta_de(pregunta: str) -> str | None:
    """La cuenta que pide la pregunta, o None si la pregunta es otra cosa.

    Deliberadamente estrecho: un enunciado con palabras («un bus recorre 68 km
    cada día…») NO es una cuenta y se va al modelo. Aquí solo entra lo que ya
    viene escrito como operación.
    """
    t = (pregunta or "").strip().strip("¿?").strip()
    t = _PREFIJOS_CUENTA.sub("", t).strip().strip("¿?").strip().rstrip("=").strip()
    if not t or len(t) > 120 or not _SOLO_CUENTA.match(t):
        return None
    if not any(c.isdigit() for c in t):
        return None
    if _UNA_FRACCION.match(t):      # «3/4» no es una cuenta, es un número
        return None
    return t if any(c in t for c in "+-*×xX÷:^/√") else None


def _frac(texto: str):
    from fractions import Fraction
    a, b = texto.replace(" ", "").split("/")
    return Fraction(int(a), int(b))


def _txt(q) -> str:
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def explicar_cuenta(cuenta: str):
    """-> (pasos, observaciones) · la cuenta resuelta paso a paso, o None.

    Los pasos los escribe Python con `Fraction`, así que son exactos por
    construcción; el RESULTADO además lo calcula la herramienta registrada y se
    compara. Si las dos cosas no coinciden se devuelve None y la pregunta pasa
    al modelo: es preferible no explicar a explicar algo que la herramienta
    contradice.

    Cubre lo que un estudiante escribe en una hoja de enunciado suelto: dos
    fracciones, un producto de enteros, y cualquier otra expresión en un solo
    paso.
    """
    import math

    try:
        from herramientas import REGISTRO
        from registro import LlamadaHerramienta
    except Exception:  # noqa: BLE001
        return None

    pasos: list = []
    par = _DOS_FRACCIONES.match(cuenta)
    if par is not None:
        a, op, b = _frac(par.group(1)), par.group(2), _frac(par.group(3))
        op = "×" if op in "xX*×" else op
        if op in "+-":
            m = math.lcm(a.denominator, b.denominator)
            if a.denominator != m or b.denominator != m:
                pasos.append(f"Busco un denominador común: el mínimo común múltiplo "
                             f"de {a.denominator} y {b.denominator} es {m}.")
            na, nb = a.numerator * (m // a.denominator), b.numerator * (m // b.denominator)
            for q, n in ((a, na), (b, nb)):
                if q.denominator != m:
                    pasos.append(f"{_txt(q)} = {n}/{m}, multiplicando arriba y abajo "
                                 f"por {m // q.denominator}.")
            bruto = na + nb if op == "+" else na - nb
            pasos.append(f"{na}/{m} {op} {nb}/{m} = {bruto}/{m}.")
            resultado = a + b if op == "+" else a - b
            if f"{bruto}/{m}" != _txt(resultado):
                g = math.gcd(abs(bruto), m)
                pasos.append(f"Simplifico dividiendo entre {g}: {bruto}/{m} = "
                             f"{_txt(resultado)}.")
        elif op == "×":
            num, den = a.numerator * b.numerator, a.denominator * b.denominator
            pasos.append(f"Multiplico numeradores y denominadores: {_txt(a)} × "
                         f"{_txt(b)} = {num}/{den}.")
            resultado = a * b
            if f"{num}/{den}" != _txt(resultado):
                pasos.append(f"Simplifico: {num}/{den} = {_txt(resultado)}.")
        else:                                   # ÷
            inv = 1 / b
            pasos.append(f"Dividir entre {_txt(b)} es multiplicar por su inverso, "
                         f"{_txt(inv)}.")
            num, den = a.numerator * inv.numerator, a.denominator * inv.denominator
            pasos.append(f"{_txt(a)} × {_txt(inv)} = {num}/{den}.")
            resultado = a / b
            if f"{num}/{den}" != _txt(resultado):
                pasos.append(f"Simplifico: {num}/{den} = {_txt(resultado)}.")
        llamada = LlamadaHerramienta(
            nombre="operar_fracciones",
            argumentos={"fraccion_a": par.group(1).replace(" ", ""),
                        "fraccion_b": par.group(3).replace(" ", ""),
                        "operacion": _OPERACION_FRACCION[par.group(2)]},
            formato="determinista")
        esperado = _txt(resultado)
    else:
        producto = re.fullmatch(r"\s*(\d+)\s*[×xX*]\s*(\d+)\s*", cuenta)
        if producto is not None and int(producto.group(2)) >= 10:
            x, y = int(producto.group(1)), int(producto.group(2))
            decenas, unidades = y - y % 10, y % 10
            pasos.append(f"Descompongo {y} en {decenas} + {unidades}.")
            pasos.append(f"{x} × {decenas} = {x * decenas}.")
            if unidades:
                pasos.append(f"{x} × {unidades} = {x * unidades}.")
                pasos.append(f"{x * decenas} + {x * unidades} = {x * y}.")
            esperado = str(x * y)
        else:
            esperado = None
        llamada = LlamadaHerramienta(nombre="evaluar_expresion",
                                     argumentos={"expresion": cuenta},
                                     formato="determinista")

    if llamada.nombre not in REGISTRO:
        return None
    res = REGISTRO.ejecutar(llamada)
    if not res.ok:
        return None
    valor = str(res.valor)
    if esperado is not None and valor != esperado:
        return None                 # la herramienta contradice la explicación
    if not pasos:
        pasos.append(f"{cuenta} = {res.resultado}.")
    pasos.append(f"Respuesta final: {valor}")
    return pasos, [res.a_dict()]


@dataclass
class Estado:
    """Qué se pidió, qué quedó funcionando, y si lo medido sigue aplicando.

    La distinción que define este objeto. `18/21` es una cifra de una
    configuración concreta: LoRA de M1 + híbrida con reranker + enrutador. Si en
    esta ejecución falta cualquiera de esas tres piezas, el sistema que responde
    **no es** el que midió 18/21, y enseñar esa cifra sería atribuirle un
    resultado de otro sistema. Por eso hay dos nombres —`config_nombre` es lo
    solicitado, `efectiva()` es lo que quedó— y `medido` solo se informa a través
    de `medido_aplicable`, que es None en cuanto falta algo requerido.
    """
    config: str = ""
    config_nombre: str = ""           # la configuración SOLICITADA
    medido: str | None = None         # la cifra histórica de la solicitada
    experimental: bool = True
    corpus: bool = False
    herramientas: int = 0                 # las REALMENTE activas, no las que existen
    herramientas_nombres: list = field(default_factory=list)
    modelo: bool = False
    adaptador_pedido: str | None = None
    adaptador_cargado: str | None = None
    retrieval_pedido: str = "ninguno"
    retrieval: str = "ninguno"            # el que quedó activo
    enrutador_pedido: bool = False
    enrutador: bool = False               # el que quedó activo
    verificador: bool = False
    normalizador: bool = False
    vision: bool = False
    dispositivo: str = "desconocido"
    cuantizado: bool = False
    montado: bool = False      # False = estos campos son el PLAN, no lo verificado
    registro_disponible: int = 0   # herramientas que EXISTEN (≠ activas)
    faltantes: list = field(default_factory=list)   # componentes requeridos ausentes
    motivos: list = field(default_factory=list)
    # La carga del modelo, etapa por etapa: nombre, ok, segundos y traceback.
    # Sin esto, un fallo en cualquier punto llegaba como un único mensaje
    # recortado a 160 caracteres y había que adivinar en qué paso ocurrió.
    etapas: list = field(default_factory=list)
    etapa_fallida: str = ""
    # Por qué el retrieval híbrido no se montó, etapa por etapa. Vacío significa
    # que no se intentó; con todas en `ok`, que se montó y respondió una
    # consulta de prueba. Es lo que faltaba en Kaggle: «retrieval híbrido +
    # reranker» aparecía entre los faltantes sin decir por qué.
    retrieval_diagnostico: list = field(default_factory=list)

    @property
    def listo(self) -> bool:
        return self.corpus and self.modelo

    @property
    def degradado(self) -> bool:
        """¿Falta algún componente que la configuración solicitada requiere?"""
        return bool(self.faltantes)

    @property
    def medido_aplicable(self) -> str | None:
        """La cifra histórica, SOLO si esta ejecución es la que la produjo.

        Degradado, o sin montar todavía: None. No hay cifra que ofrecer, y
        ofrecer la de la configuración solicitada sería atribuir a este sistema
        el resultado de otro.
        """
        if self.degradado or not self.montado or not self.listo:
            return None
        return self.medido

    def efectiva(self) -> str:
        """El nombre de lo que de verdad está respondiendo.

        Nunca «S08-D» cuando falta una de sus piezas: un nombre que promete una
        cifra tiene que describir el sistema que la produjo.
        """
        # El orden importa: un componente que YA se sabe ausente es un hecho,
        # aunque el montaje no haya terminado. Preguntar antes por `montado`
        # tapaba el adaptador que no se encontró detrás de un «sin montar
        # todavía», que es justo lo que hay que ver al lanzar en Kaggle.
        if self.degradado:
            return (f"MODO DEGRADADO (se pidió {self.config}; falta "
                    f"{', '.join(self.faltantes)})")
        if not self.montado:
            return f"{self.config_nombre} (sin montar todavía)"
        return self.config_nombre

    def piezas(self) -> list:
        """Qué componente está funcionando, uno por uno, para enseñarlo.

        Antes de cargar el modelo nada se ha comprobado, así que se dice
        «pendiente» en vez de «NINGUNO»: lo segundo afirmaría que ya se intentó
        y falló.
        """
        if not self.montado:
            adaptador = f"{self.adaptador_pedido} (pendiente de cargar)"
        else:
            adaptador = self.adaptador_cargado or "NINGUNO (modelo base)"
        p = [f"adaptador {adaptador}", f"retrieval {self.retrieval}"]
        if self.enrutador:
            p.append("enrutador")
        elif self.enrutador_pedido:
            p.append("SIN enrutador (se pidió)")
        p.append(f"{self.herramientas} herramientas" if self.herramientas
                 else "sin herramientas")
        if self.verificador:
            p.append("verificador")
        if self.normalizador:
            p.append("normalizador")
        if self.vision:
            p.append("visión")
        p.append("int4" if self.cuantizado else "fp16")
        return p

    def resumen(self) -> str:
        if not self.listo:
            # Sin motivos y sin modelo = carga perezosa pendiente, no una avería.
            # Decir "NO disponible" ahí sería alarmar por algo que no ha pasado.
            if not self.motivos:
                return (f"{self.config or 'tutor'}: el modelo se carga en la primera "
                        "pregunta; todavía no se ha comprobado")
            return "Tutor NO disponible: " + "; ".join(self.motivos)
        if self.degradado:
            marca = "resultado de esta ejecución: NO MEDIDO"
        elif self.medido:
            marca = f"medido {self.medido}"
        else:
            marca = "EXPERIMENTAL, sin medir"
        return (f"{self.efectiva()} · " + " · ".join(self.piezas())
                + f" · [{marca}]")


# --------------------------------------------------------------------------
# Corpus y retrieval · sin modelo de lenguaje
# --------------------------------------------------------------------------

def cargar_corpus():
    """Los chunks del corpus documental, desde la MISMA fuente que los notebooks.

    `nb_comun_rag.CODIGO_CORPUS` es la única copia del corpus en el repositorio;
    ejecutarla aquí evita tener dos que se separen con el tiempo. Es el patrón que
    ya usa `scripts/trayectorias_fuente.py`.
    """
    from nb_comun_rag import CODIGO_CHUNKING, CODIGO_CORPUS

    ns: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(CODIGO_CORPUS, ns)      # noqa: S102 — fuente propia del repositorio
        exec(CODIGO_CHUNKING, ns)    # noqa: S102
    return ns["chunks"], ns["metadatos"], ns["ids_chunks"]


# Los dos modelos de la cadena de S08, como constantes: el guion de diagnóstico
# comprueba exactamente estos y no una copia que se pueda desincronizar.
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_SECRETOS = re.compile(
    r"(hf_[A-Za-z0-9]{8,}|(?i:token|api[_-]?key|secret|password|authorization)"
    r"\s*[:=]\s*)(\S+)")


def _redactar(texto: str) -> str:
    """Tapa lo que parezca una credencial antes de publicar un traceback.

    Los errores de `huggingface_hub` citan la URL y a veces la cabecera de la
    petición. El diagnóstico se enseña en la pestaña Estado y se pega en los
    informes, así que pasa por aquí antes de salir.
    """
    return _SECRETOS.sub(lambda m: m.group(1) + "[REDACTADO]", texto or "")


def _causa_probable(e: BaseException) -> str:
    """Traduce la excepción a la categoría que decide qué hacer después.

    No es un diagnóstico definitivo —para eso está el traceback completo— sino
    el primer corte: si la causa es la red, reinstalar paquetes no arregla
    nada, y si es una versión, tener internet tampoco.
    """
    nombre = type(e).__name__
    mensaje = str(e).lower()
    if isinstance(e, (ImportError, ModuleNotFoundError)):
        return ("dependencia ausente o incompatible (el paquete no importa, "
                "no es la red)")
    if isinstance(e, MemoryError) or "out of memory" in mensaje:
        return "memoria"
    if any(p in mensaje for p in ("offline", "connection", "timed out", "timeout",
                                  "resolve", "proxy", "couldn't connect",
                                  "max retries", "429", "503")):
        return "red o descarga de modelos"
    if any(p in mensaje for p in ("not a local folder", "no such file",
                                  "cache", "entry not found", "repository not found",
                                  "gated", "401", "403")):
        return "caché o acceso al repositorio de modelos"
    if nombre in ("AttributeError", "TypeError", "ValueError", "KeyError",
                  "RuntimeError"):
        return f"versión de dependencia o código ({nombre})"
    return f"sin clasificar ({nombre})"


def buscador_hibrido(chunks, diagnostico=None):
    """El retrieval de S08-C, que es el que S08-D enruta.

    Reproduce la cadena del notebook, no una aproximación:

        denso top-10  +  BM25 top-10   -> RRF (k=60)      `buscar_hibrida`
        -> cross-encoder mmarco sobre los 10 -> top-3      `buscar_con_rerank`

    El reranker **no es opcional**: `D · +router` se construye sobre
    `C · +rerank` (`nb_rag_avanzado.py:559`), y el enrutador usa como tercer
    rasgo el puntaje máximo del cross-encoder (`enrutador.RASGOS`). Sin él se
    monta otra cosa y se la presenta como la de 18/21.

    Devuelve None si no se puede montar, **y deja escrito por qué**. Antes no:
    un `except ImportError` y un `except Exception` devolvían la misma None
    tanto si faltaba `rank_bm25` como si el cross-encoder no se podía descargar,
    y en Kaggle eso dejó «retrieval híbrido + reranker» entre los faltantes sin
    una sola línea sobre la causa. `diagnostico` —una lista— recibe una entrada
    por etapa con su error, su causa probable y su traceback; el estado del
    tutor la publica en la pestaña Estado.

    Además, codificar el corpus ya no ocurre fuera del manejo de errores. Antes
    un fallo ahí no degradaba el retrieval: tumbaba la carga entera del tutor.
    """
    import traceback

    registro = [] if diagnostico is None else diagnostico

    def etapa(nombre, funcion):
        """Corre una etapa y la registra. Devuelve (ok, valor); nunca lanza."""
        t0 = time.perf_counter()
        try:
            valor = funcion()
        except BaseException as e:      # noqa: BLE001 — se registra, no se oculta
            registro.append({
                "etapa": nombre, "ok": False,
                "segundos": round(time.perf_counter() - t0, 2),
                "error": _redactar(f"{type(e).__name__}: {e}")[:400],
                "causa_probable": _causa_probable(e),
                "traceback": _redactar(traceback.format_exc()),
            })
            print(f"\n=== RETRIEVAL · FALLÓ LA ETAPA: {nombre} ===\n"
                  f"{_redactar(traceback.format_exc())}", flush=True)
            return False, None
        registro.append({"etapa": nombre, "ok": True,
                         "segundos": round(time.perf_counter() - t0, 2)})
        return True, valor

    def importar():
        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer
        return np, BM25Okapi, CrossEncoder, SentenceTransformer

    ok, modulos = etapa("importar sentence-transformers, rank_bm25 y numpy",
                        importar)
    if not ok:
        return None
    np, BM25Okapi, CrossEncoder, SentenceTransformer = modulos

    def tokenizar(texto):
        import unicodedata
        t = unicodedata.normalize("NFKD", (texto or "").lower())
        return re.findall(r"\w+", "".join(c for c in t if not unicodedata.combining(c)))

    ok, bm25 = etapa("índice BM25 del corpus",
                     lambda: BM25Okapi([tokenizar(c) for c in chunks]))
    if not ok:
        return None
    ok, st = etapa(f"cargar embeddings {MODELO_EMBEDDINGS} (descarga o caché)",
                   lambda: SentenceTransformer(MODELO_EMBEDDINGS))
    if not ok:
        return None
    ok, reranker = etapa(f"cargar reranker {MODELO_RERANKER} (descarga o caché)",
                         lambda: CrossEncoder(MODELO_RERANKER, max_length=512))
    if not ok:
        return None

    def codificar():
        e = st.encode(chunks, show_progress_bar=False)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    ok, emb = etapa(f"codificar los {len(chunks)} chunks", codificar)
    if not ok:
        return None

    K_RECUPERAR, K_RRF = 10, 60

    def candidatos(consulta, n=K_RECUPERAR):
        """Híbrida: cada lista aporta su top-n antes de fusionar, como en S08-B."""
        q = st.encode([consulta])[0]
        q = q / np.linalg.norm(q)
        listas = (list(np.argsort(-(emb @ q))[:n]),
                  list(np.argsort(-bm25.get_scores(tokenizar(consulta)))[:n]))
        puntos: dict = {}
        for ranking in listas:
            for pos, idx in enumerate(ranking):
                puntos[int(idx)] = puntos.get(int(idx), 0.0) + 1.0 / (K_RRF + pos + 1)
        return [i for i, _ in sorted(puntos.items(), key=lambda x: -x[1])][:n]

    def puntajes(consulta, cand):
        return reranker.predict([(consulta, chunks[i]) for i in cand],
                                show_progress_bar=False)

    def buscar(consulta, k=3):
        cand = candidatos(consulta)
        if not cand:
            return []
        s = puntajes(consulta, cand)
        return [cand[i] for i in np.argsort(s)[::-1][:k]]

    def rerank_max(consulta):
        """El tercer rasgo del enrutador de S08-D: ¿hay algo que de verdad encaje?"""
        cand = candidatos(consulta)
        return float(max(puntajes(consulta, cand))) if cand else 0.0

    # Una consulta de verdad antes de devolver el buscador. Sin esto, «montado»
    # significaba solo que los constructores no lanzaron: la primera pregunta
    # del estudiante era la que descubría que `predict` no funcionaba.
    ok, prueba = etapa("consulta de prueba contra el corpus",
                       lambda: buscar("¿cuál es la nota mínima aprobatoria?", 3))
    if not ok:
        return None
    registro[-1]["resultados"] = len(prueba or [])

    # `normalize_embeddings=True`, como en el notebook
    # (`nb_rag_avanzado.py:355`). El enrutador promedia estos vectores para sus
    # centroides y hace producto punto con la pregunta: sin normalizar, el rasgo
    # `margen` sale en otra escala y la regresión ajusta otro modelo.
    buscar.encode = lambda textos: st.encode(list(textos), normalize_embeddings=True)
    buscar.rerank_max = rerank_max
    return buscar


def buscador_lexico(chunks):
    """Retrieval sin dependencias pesadas: solapamiento de palabras.

    No es el híbrido con reranker de S08. Es el repliegue para que la demo
    funcione sin `sentence-transformers` ni GPU, y la interfaz lo dice. Si el
    entorno trae los embeddings, `buscador_hibrido` lo sustituye.
    """
    import re
    import unicodedata

    def norm(t):
        t = unicodedata.normalize("NFKD", (t or "").lower())
        return set(re.findall(r"\w+", "".join(c for c in t if not unicodedata.combining(c))))

    indices = [norm(c) for c in chunks]

    def buscar(consulta, k=3):
        q = norm(consulta)
        if not q:
            return []
        puntos = [(len(q & ix) / len(q | ix) if q | ix else 0.0, i)
                  for i, ix in enumerate(indices)]
        puntos.sort(reverse=True)
        return [i for p, i in puntos[:k] if p > 0]

    return buscar


# --------------------------------------------------------------------------
# El tutor
# --------------------------------------------------------------------------

class Tutor:
    """Monta el pipeline y responde. Nada se carga hasta que hace falta."""

    def __init__(self, config: str | None = None):
        self.cfg: Configuracion = obtener(config or CONFIG)
        self.estado = Estado(
            config=self.cfg.nombre.split("·")[0].strip(),
            config_nombre=self.cfg.nombre,
            medido=self.cfg.medido,
            experimental=self.cfg.es_experimental,
            adaptador_pedido=self.cfg.adaptador,
            retrieval_pedido=self.cfg.retrieval,
            retrieval=self.cfg.retrieval,
            enrutador_pedido=self.cfg.enrutador,
            enrutador=self.cfg.enrutador,
            verificador=self.cfg.verificador,
            normalizador=self.cfg.normalizador,
            vision=self.cfg.vision)
        self._agente = None
        self._vlm = None
        self._chunks = None
        self._montar_sin_modelo()

    # -- montaje por capas, para que un fallo no se lleve el resto ---------
    def _montar_sin_modelo(self):
        try:
            self._chunks, self._metadatos, self._ids = cargar_corpus()
            self.estado.corpus = True
        except Exception as e:  # noqa: BLE001
            self.estado.motivos.append(f"corpus: {type(e).__name__}")
            return
        try:
            from herramientas import REGISTRO
            self._registro_base = REGISTRO
            # Cuántas EXISTEN. Las activas las fija la configuración al montar;
            # hasta entonces la salud marca el montaje como "declarado".
            self.estado.registro_disponible = len(REGISTRO)
            if self.cfg.herramientas:
                self.estado.herramientas = len(REGISTRO)
        except Exception as e:  # noqa: BLE001
            self.estado.motivos.append(f"herramientas: {type(e).__name__}")

    def _etapa(self, nombre: str, funcion):
        """Ejecuta una etapa de la carga y deja constancia de cómo fue.

        Por qué existe. Antes toda la carga iba dentro de un solo
        `except Exception` que guardaba `f"modelo: {type(e).__name__}: {e}"[:160]`.
        Con eso, un `AttributeError: module 'torch.distributed' has no attribute
        'tensor'` llegaba a pantalla sin decir **en qué paso** ocurrió: puede
        venir del tokenizer, del modelo base, del `.to("cuda")`, del adaptador o
        del montaje del retrieval, y cada uno se arregla distinto.

        Ahora cada etapa se cronometra, se anota y, si falla, se imprime su
        traceback completo en el log del proceso —que en Kaggle es la salida de
        la celda— y se guarda para la pestaña Estado.
        """
        import time as _t
        import traceback

        t0 = _t.perf_counter()
        try:
            valor = funcion()
        except BaseException as e:      # noqa: BLE001 — se re-lanza más abajo
            traza = traceback.format_exc()
            self.estado.etapas.append({
                "etapa": nombre, "ok": False,
                "segundos": round(_t.perf_counter() - t0, 2),
                "error": f"{type(e).__name__}: {e}",
                "traceback": traza,
            })
            self.estado.etapa_fallida = nombre
            print(f"\n=== FALLO EN LA ETAPA: {nombre} ===\n{traza}", flush=True)
            raise
        self.estado.etapas.append({
            "etapa": nombre, "ok": True,
            "segundos": round(_t.perf_counter() - t0, 2),
            "error": None, "traceback": None,
        })
        return valor

    def _cargar_modelo(self):
        """Carga el modelo la primera vez que se necesita, etapa por etapa."""
        if self._agente is not None or self.estado.etapa_fallida:
            return
        if "modelo" in " ".join(self.estado.motivos):
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            self.estado.motivos.append(f"modelo: falta {e.name}")
            return

        self.estado.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

        def kwargs_carga():
            if CUANTIZAR:
                from transformers import BitsAndBytesConfig
                self.estado.cuantizado = True
                return {"quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16)}
            return {"torch_dtype": torch.float16
                    if self.estado.dispositivo == "cuda" else torch.float32}

        def cargar_tokenizer():
            tok = AutoTokenizer.from_pretrained(MODELO_BASE)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok

        try:
            kwargs = self._etapa("preparar opciones de carga", kwargs_carga)
            tok = self._etapa("tokenizer", cargar_tokenizer)
            modelo = self._etapa(
                "modelo base",
                lambda: AutoModelForCausalLM.from_pretrained(MODELO_BASE, **kwargs))
            if not CUANTIZAR:
                modelo = self._etapa(
                    f"mover a {self.estado.dispositivo}",
                    lambda: modelo.to(self.estado.dispositivo))

            ruta = self._etapa("localizar adaptador", self.comprobar_adaptador)
            if ruta is not None:
                modelo = self._etapa(
                    "adaptador LoRA (PeftModel.from_pretrained)",
                    lambda: self._cargar_adaptador(PeftModel, modelo, ruta))
                self.estado.adaptador_cargado = self.cfg.adaptador

            self._etapa("armar agente (retrieval, enrutador, herramientas)",
                        lambda: self._armar_agente(modelo.eval(), tok))
            self.estado.modelo = True
        except Exception as e:  # noqa: BLE001
            # El traceback completo ya se imprimió en `_etapa`. Aquí queda el
            # resumen corto que la interfaz enseña, CON la etapa, que es lo que
            # faltaba para saber dónde mirar.
            self.estado.motivos.append(
                f"modelo: falló en «{self.estado.etapa_fallida}» · "
                f"{type(e).__name__}: {e}"[:300])

    def _cargar_adaptador(self, PeftModel, modelo, ruta):  # noqa: N803
        """Carga el LoRA tolerando que lo guardara una PEFT más nueva.

        El problema, medido. `adaptadores/qwen-lora/adapter_config.json` trae 38
        campos, y cuatro de ellos no existen en las PEFT publicadas en PyPI:

            alora_invocation_tokens = None
            qalora_group_size       = 16
            use_qalora              = False
            trainable_token_indices = None

        Con PEFT 0.13.2 eso revienta en
        `TypeError: __init__() got an unexpected keyword argument
        'alora_invocation_tokens'`, que es justo lo que obligó a instalar una
        PEFT muy nueva en Kaggle y lo que rompería el Space, donde esa versión
        no se puede instalar.

        **Por qué se pueden omitir y no es esconder nada.** Los cuatro valen su
        valor por defecto: `None`, `False` y —con `use_qalora=False`— un
        `qalora_group_size` que no se usa. Codifican «esta función está
        apagada». Quitarlos no cambia ni un peso del adaptador; mantenerlos
        obliga a una versión concreta de la librería por nada.

        Solo se omite lo que la `LoraConfig` instalada **no** acepta, y queda
        anotado en el estado. Si falla por cualquier otra razón, se propaga.
        """
        try:
            return PeftModel.from_pretrained(modelo, str(ruta),
                                             adapter_name=self.cfg.adaptador)
        except TypeError as e:
            if "unexpected keyword argument" not in str(e):
                raise
            import inspect
            import json

            from peft import LoraConfig

            datos = json.loads((ruta / "adapter_config.json").read_text(
                encoding="utf-8"))
            admitidos = set(inspect.signature(LoraConfig.__init__).parameters)
            sobrantes = {k: v for k, v in datos.items() if k not in admitidos}
            # Un campo desconocido con valor distinto del de por defecto sí
            # cambiaría el comportamiento: ahí no se puede seguir en silencio.
            # `peft_version` es metadato, no una opción: dice con qué versión
            # se guardó, que es justamente el dato que explica el fallo.
            guardado_con = sobrantes.pop("peft_version", None)
            activos = {k: v for k, v in sobrantes.items()
                       if v not in (None, False, 0, "", [])}
            config = LoraConfig(**{k: v for k, v in datos.items()
                                   if k in admitidos})
            import importlib.metadata as _md
            try:
                instalada = _md.version("peft")
            except Exception:      # noqa: BLE001
                instalada = "desconocida"
            self.estado.motivos.append(
                f"adaptador guardado con PEFT {guardado_con or '?'} y leído con "
                f"{instalada}: se omitieron {sorted(sobrantes)} por no existir "
                "en la instalada; todos valían su valor por defecto"
                + (f". OJO, estos NO: {activos}" if activos else ""))
            return PeftModel.from_pretrained(modelo, str(ruta), config=config,
                                             adapter_name=self.cfg.adaptador)

    def comprobar_adaptador(self):
        """-> la ruta del adaptador que pide la configuración, o None.

        Devuelve el que PIDE la configuración, no el que haya en el disco:
        habilitar otro por estar presente fue el fallo que llevó a servir una
        variante de C5 bajo el nombre de S08-D.

        Separado de `_cargar_modelo` para poder probar el caso de adaptador
        ausente sin descargar el modelo base.
        """
        pedido = self.cfg.adaptador
        ruta = RAIZ / RUTAS_ADAPTADOR[pedido]
        if ruta.is_dir():
            return ruta
        # El adaptador es el componente del que dependen TODAS las cifras de M1
        # en adelante: el modelo base pelado marcó 9/21 (C0, `scorecard_agente_v2
        # .csv`). Sin él no queda nada parecido a la configuración solicitada.
        self.estado.faltantes.append(f"adaptador LoRA {pedido!r}")
        self.estado.motivos.append(
            f"falta el adaptador {pedido!r} en {RUTAS_ADAPTADOR[pedido]}: se "
            "responde con el modelo BASE, sin afinar para este dominio")
        return None

    def resolver_componentes(self) -> dict:
        """Monta todo lo que NO necesita el modelo de lenguaje, y anota qué falta.

        Separado de `_armar_agente` a propósito: así los casos que importan
        —adaptador ausente, recuperador ausente, enrutador que no arranca— se
        pueden probar sin descargar 3 GB de pesos. Cada componente requerido que
        no se materializa entra en `estado.faltantes`, y eso es lo que apaga la
        cifra medida.
        """
        from documentos import registro_con_documentos

        cfg = self.cfg

        # 1 · retrieval. Si el pedido no se puede montar, se repliega DICIÉNDOLO
        #     y anotándolo como componente faltante: la búsqueda léxica responde,
        #     pero no es la que midió nada.
        buscar, activo = None, "ninguno"
        if cfg.retrieval == "hibrido":
            buscar = buscador_hibrido(self._chunks,
                                      self.estado.retrieval_diagnostico)
            activo = "hibrido + rerank (BM25+denso, RRF, cross-encoder) = S08-C"
            if buscar is None:
                self.estado.faltantes.append("retrieval híbrido + reranker")
                # La causa concreta, no la lista de sospechosos: con los tres
                # paquetes instalados el mensaje anterior («falta rank_bm25,
                # sentence-transformers o los modelos en caché») era falso en
                # sus dos primeras opciones y mudo en la tercera.
                fallo = next((x for x in self.estado.retrieval_diagnostico
                              if not x["ok"]), None)
                detalle = (f"falló «{fallo['etapa']}» · {fallo['error']} · causa "
                           f"probable: {fallo['causa_probable']}" if fallo
                           else "no se registró ninguna etapa")
                self.estado.motivos.append(
                    f"retrieval: {detalle}. Se usa búsqueda léxica, que NO es "
                    "la de S08")
        if buscar is None and cfg.retrieval != "ninguno":
            buscar = buscador_lexico(self._chunks)
            activo = ("lexico (REPLIEGUE: se pidió híbrido)"
                      if cfg.retrieval == "hibrido" else "lexico")
        self.estado.retrieval = activo

        # 2 · enrutador: decidir SI buscar, que es lo que llevó S08 de 14 a 18/21
        contexto_fijo = None
        if cfg.enrutador and buscar is not None:
            contexto_fijo = self._montar_enrutador(buscar)
        elif cfg.enrutador:
            self.estado.motivos.append("enrutador: no hay retrieval que enrutar")
        self.estado.enrutador = contexto_fijo is not None
        if cfg.enrutador and not self.estado.enrutador:
            # Sin enrutador, lo que queda es S08-C (13/21) o peor, no S08-D. Es
            # justo la pieza que explica los +4 casos, así que su ausencia no
            # puede pasar como un detalle de montaje.
            self.estado.faltantes.append("enrutador consulta/problema")

        # 3 · herramientas: solo si la configuración las pide
        registro = None
        base = getattr(self, "_registro_base", None)
        if cfg.herramientas and base is None:
            self.estado.faltantes.append("registro de herramientas")
            self.estado.herramientas, self.estado.herramientas_nombres = 0, []
        elif cfg.herramientas:
            registro = (registro_con_documentos(base, buscar, self._chunks,
                                                self._metadatos, self._ids, k=3)
                        if buscar is not None and not cfg.enrutador else base)
            self.estado.herramientas = len(registro)
            self.estado.herramientas_nombres = sorted(registro.nombres())
        else:
            self.estado.herramientas = 0
            self.estado.herramientas_nombres = []

        # 4 · normalizador: solo si la configuración lo pide. Es una pieza que
        # reescribe la pregunta antes de responderla, así que activarla "porque
        # existe" cambiaría la entrada de todas las ablaciones a la vez.
        normalizador = None
        if cfg.normalizador:
            try:
                from nucleo import Normalizador
                normalizador = Normalizador()
            except Exception as e:      # noqa: BLE001
                self.estado.faltantes.append("normalizador")
                self.estado.motivos.append(f"normalizador: {type(e).__name__}")
        self.estado.normalizador = normalizador is not None

        # 5 · verificador: se pide, así que tiene que poder importarse.
        if cfg.verificador:
            try:
                import verificador_aritmetico  # noqa: F401
            except Exception as e:      # noqa: BLE001
                self.estado.verificador = False
                self.estado.faltantes.append("verificador aritmético")
                self.estado.motivos.append(f"verificador: {type(e).__name__}")

        # Desde aquí los campos del estado son lo COMPROBADO, no el plan.
        self.estado.montado = True
        return {"buscar": buscar, "contexto_fijo": contexto_fijo,
                "registro": registro, "normalizador": normalizador}

    def _armar_agente(self, modelo, tok):
        import torch

        from documentos import chunks_de_resultado
        from nucleo import AgenteTutor

        cfg = self.cfg
        piezas = self.resolver_componentes()
        contexto_fijo = piezas["contexto_fijo"]
        registro = piezas["registro"]
        system = self._system(cfg)

        @torch.no_grad()
        def llm(mensajes, esquemas):
            prompt = tok.apply_chat_template(mensajes, tools=esquemas, tokenize=False,
                                             add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(modelo.device)
            out = modelo.generate(**ids, max_new_tokens=320, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
            return tok.decode(out[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()

        self._agente = AgenteTutor(
            llm, system, registro=registro, contexto_fijo=contexto_fijo,
            max_rondas=6 if cfg.herramientas else 1, max_pasos=8,
            verificar=cfg.herramientas, normalizador=piezas["normalizador"],
            extraer_contextos=chunks_de_resultado if registro else None,
            nombre=self.estado.efectiva())

    def _system(self, cfg) -> str:
        """El prompt de la configuración. S08-D usa el de RAG, no el del agente."""
        if cfg.system == "agente":
            from prompts import SYSTEM_C5_AGENTE
            return SYSTEM_C5_AGENTE
        from nb_comun_rag import CODIGO_PROMPTS
        ns: dict = {}
        exec(CODIGO_PROMPTS, ns)      # noqa: S102 — fuente propia del repositorio
        return ns["SYSTEM_RAG"]

    def _montar_enrutador(self, buscar):
        """El enrutador de S08-D, ajustado con sus 33 consultas de desarrollo.

        Si no hay `encode` —porque el retrieval cayó al léxico— no se puede
        montar, y se dice: un enrutador sin embeddings no es el que se midió.
        """
        encode = getattr(buscar, "encode", None)
        rerank_max = getattr(buscar, "rerank_max", None)
        if encode is None or rerank_max is None:
            # Sin cifra en el texto a propósito: este mensaje solo aparece en
            # ejecuciones degradadas, y repetir ahí el resultado medido es la
            # forma más fácil de que alguien lo cite fuera de contexto.
            self.estado.motivos.append(
                "enrutador: su tercer rasgo es el puntaje del cross-encoder; sin "
                "embeddings ni reranker no se puede reproducir el de S08-D")
            return None
        try:
            import contextlib
            import io

            from enrutador import Enrutador, contexto_enrutado
            from nb_rag_avanzado import CODIGO_CONSULTAS_DEV

            ns: dict = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(CODIGO_CONSULTAS_DEV, ns)   # noqa: S102
            # Los tres rasgos de S08-D: margen de centroides, nº de números y
            # puntaje máximo del reranker. Pasar `rerank_max=None` dejaría el
            # tercero en 0 y ajustaría otro modelo con el mismo nombre.
            enrutador = Enrutador(encode, rerank_max, ns["consultas_dev"])

            def contexto(pregunta):
                idx = buscar(pregunta, 3)
                return [f"[Fuente: {self._metadatos[i]['fuente']}]\n{self._chunks[i]}"
                        for i in idx]

            return contexto_enrutado(enrutador, contexto)
        except Exception as e:      # noqa: BLE001
            self.estado.motivos.append(f"enrutador: {type(e).__name__}")
            return None

    # -- la API que usa la interfaz ---------------------------------------
    def responder(self, pregunta: str) -> dict:
        """-> dict con respuesta, latencia y traza. Nunca lanza."""
        t0 = time.perf_counter()
        pregunta = (pregunta or "").strip()
        if not pregunta:
            return self._fallo("escribe una pregunta", t0)

        # Antes de cargar 1,5 B de pesos: si la pregunta es una cuenta, la
        # calcula la herramienta y se acabó.
        directa = self._aritmetica_directa(pregunta)
        if directa is not None:
            respuesta, observaciones = directa
            return {
                "ok": True,
                "respuesta": respuesta,
                "herramientas": [o["herramienta"] for o in observaciones],
                "observaciones": observaciones,
                "pasos_llm": 0,
                "contextos": 0,
                "verificador": None,
                "latencia_s": round(time.perf_counter() - t0, 2),
            }

        self._cargar_modelo()
        if not self.estado.listo:
            return self._fallo(self.estado.resumen(), t0)

        try:
            salida = self._agente(pregunta)
            hallazgo = None
            if self.estado.verificador:
                salida, hallazgo = self._reintentar_con_verificador(pregunta, salida)
        except Exception as e:  # noqa: BLE001
            return self._fallo(f"el tutor falló: {type(e).__name__}", t0)

        return {
            "ok": True,
            "respuesta": salida.respuesta,
            "herramientas": [o["herramienta"] for o in salida.observaciones],
            "observaciones": list(salida.observaciones),
            "pasos_llm": salida.pasos_llm,
            "contextos": len(salida.contextos),
            "verificador": hallazgo,
            "latencia_s": round(time.perf_counter() - t0, 2),
        }

    def _aritmetica_directa(self, pregunta: str):
        """-> (respuesta, [observación]) si la pregunta es una cuenta, o None.

        La observación es la de `RegistroHerramientas.ejecutar`: si aquí no se
        ejecutó nada, la traza tampoco dirá que se ejecutó algo.
        """
        if not self.cfg.herramientas:
            return None
        cuenta = cuenta_de(pregunta)
        if cuenta is None:
            return None
        try:
            from herramientas import REGISTRO
            from registro import LlamadaHerramienta
        except Exception:  # noqa: BLE001
            return None

        par = _DOS_FRACCIONES.match(cuenta)
        if par is not None and "operar_fracciones" in REGISTRO:
            # Con fracciones el resultado exacto es una fracción: 3/4 + 1/8 son
            # 7/8, no «0.875 aproximado». `operar_fracciones` lo formatea así.
            llamada = LlamadaHerramienta(
                nombre="operar_fracciones",
                argumentos={"fraccion_a": par.group(1).replace(" ", ""),
                            "fraccion_b": par.group(3).replace(" ", ""),
                            "operacion": _OPERACION_FRACCION[par.group(2)]},
                formato="determinista")
        elif "evaluar_expresion" in REGISTRO:
            llamada = LlamadaHerramienta(nombre="evaluar_expresion",
                                         argumentos={"expresion": cuenta},
                                         formato="determinista")
        else:
            return None

        resultado = REGISTRO.ejecutar(llamada)
        if not resultado.ok:
            return None            # que lo intente el modelo y lo explique
        return f"{cuenta} = {resultado.resultado}", [resultado.a_dict()]

    def _reintentar_con_verificador(self, pregunta, salida):
        """Una sola segunda pasada si la aritmética de la respuesta no cuadra.

        El verificador no dicta el veredicto: `pista_para_el_tutor` describe la
        igualdad que falla y el modelo decide. Una sola pasada, para que el
        coste sea acotado y la ablación tenga un único cambio que medir.
        """
        from verificador_aritmetico import pista_para_el_tutor, revisar_hoja

        pasos = [l for l in (salida.respuesta or "").splitlines() if l.strip()]
        h = revisar_hoja(pasos)
        # `calculo` = la igualdad no cuadra; `traslado` = un valor cambió entre
        # pasos. `ninguno` y `no_evaluable` no son hallazgos: no se reintenta.
        if h is None or h.tipo not in ("calculo", "traslado"):
            return salida, None
        pista = pista_para_el_tutor(h)
        try:
            return self._agente(f"{pregunta}\n\n{pista}"), {
                "paso": h.paso, "tipo": h.tipo, "detalle": h.detalle, "reintento": True}
        except Exception:  # noqa: BLE001
            return salida, {"paso": h.paso, "tipo": h.tipo, "detalle": h.detalle,
                            "reintento": False}

    # ----------------------------------------------------------------------
    # La vía visual, en DOS pasos separados
    #
    # Por qué dos y no uno. En M4 el flujo leía la foto y diagnosticaba en la
    # misma llamada, y `vis-10` dejó claro el problema: el VLM inventó
    # «24 x 13 = 306» —dos veces, idéntico— y el tutor diagnosticó sobre ese
    # invento con total aplomo. Mientras el estudiante no vea y confirme lo que
    # se leyó, un diagnóstico es una afirmación sobre una hoja que quizá no
    # existe.
    #
    # Aquí no hay puntuación de confianza inventada. La puerta usa únicamente
    # señales observables de esta petición: JSON estructurado, declaración de
    # ilegibilidad y acuerdo entre dos lecturas independientes. La confirmación
    # editable cubre lo que esas señales no pueden certificar.
    # ----------------------------------------------------------------------

    PREGUNTA_REVISION = "Revisa mi procedimiento: ¿en qué paso me equivoqué?"

    def leer_imagen(self, imagen) -> dict:
        """Paso 1: enseña lo que se leyó y pide confirmación. NO diagnostica.

        -> dict con `transcripcion`, `pasos` editables, `confirmable` y la
        `pregunta` que hay que hacerle al estudiante. Nunca un veredicto: eso
        es el paso 2, y solo si estos pasos quedan confirmados.
        """
        t0 = time.perf_counter()
        falta = self._vision_disponible()
        if falta:
            return self._fallo(falta, t0)
        if imagen is None:
            return self._fallo("sube una foto de la hoja", t0)
        try:
            from vision import lectura_no_fiable, pasos_como_texto, pasos_numerados
        except Exception as e:  # noqa: BLE001
            return self._fallo(f"visión no disponible: {type(e).__name__}", t0)

        lectura = self._leer(imagen)
        if lectura is None:
            return self._fallo("no se pudo cargar el modelo de visión", t0)

        no_fiable, motivo = lectura_no_fiable(lectura)
        js = (lectura.get("json") or {}) if lectura.get("ok") else {}
        base = {
            "ok": True,
            "etapa": "lectura",
            "transcripcion": pasos_como_texto(lectura),
            "enunciado": js.get("enunciado"),
            "pasos": pasos_numerados(lectura),
            "lecturas_coinciden": lectura.get("consistente"),
            "diferencias": lectura.get("diferencias") or [],
            "normalizacion": lectura.get("normalizacion") or {},
            "latencia_s": round(time.perf_counter() - t0, 2),
            "herramientas": [], "pasos_llm": 0, "contextos": 0,
        }
        if no_fiable:
            # Se detiene aquí. No es un diagnóstico de la hoja: es la constatación
            # de que no hay una hoja legible sobre la que diagnosticar.
            return base | {
                "confirmable": False,
                "motivo_no_fiable": motivo,
                "respuesta": (f"No puedo leer bien esta hoja: {motivo}. No voy a "
                              "adivinar qué escribiste. Puedes tomar otra foto con "
                              "más luz y la hoja plana, o escribir los pasos a mano "
                              "aquí abajo y los reviso."),
                "pregunta": ("¿Quieres tomar otra foto, o prefieres escribir los "
                             "pasos tú?"),
            }
        if not base["pasos"]:
            if (base["enunciado"] or "").strip():
                # Solo el enunciado. No es un error de lectura: es la otra cosa
                # que un estudiante fotografía. Se sugiere resolverlo, y el
                # estudiante confirma tanto el texto como lo que quiere.
                return base | {
                    "confirmable": True,
                    "proposito": "resolver",
                    "motivo_no_fiable": "",
                    "respuesta": ("Leí un enunciado sin pasos resueltos. Revisa que "
                                  "esté bien copiado antes de que lo resuelva."),
                    "pregunta": "¿Quieres que lo resuelva paso a paso?",
                }
            return base | {
                "confirmable": False,
                "motivo_no_fiable": "no se leyó ni un enunciado ni pasos",
                "respuesta": ("Leí la hoja pero no encuentro ni un enunciado ni los "
                              "pasos de un procedimiento matemático."),
                "pregunta": "¿Es esto un ejercicio de matemáticas?",
            }
        return base | {
            "confirmable": True,
            "proposito": "revisar",
            "motivo_no_fiable": "",
            "respuesta": ("Esto es lo que leí en tu hoja. **Revísalo antes de que "
                          "lo corrija**: si transcribí mal un número, el error que "
                          "te señale sería mío, no tuyo."),
            "pregunta": ("¿Es exactamente lo que escribiste? Corrige cualquier "
                         "paso que esté mal transcrito y confirma para que lo revise."),
        }

    def revisar_pasos(self, pasos, pregunta: str = "", enunciado=None,
                      confirmado: bool = False) -> dict:
        """Paso 2: revisa los pasos YA CONFIRMADOS por el estudiante.

        Sin `confirmado=True` no hay diagnóstico. No es una formalidad: es la
        diferencia entre revisar lo que el estudiante escribió y revisar lo que
        un modelo de visión creyó ver.

        El orden importa. Primero el verificador determinista, que comprueba las
        igualdades con `Fraction` y no se equivoca en aritmética; su hallazgo
        entra en el prompt como **evidencia**, no como veredicto. Qué paso
        señalar y cómo explicarlo sigue siendo del tutor, y el prompt le prohíbe
        reescribir los números del estudiante.
        """
        t0 = time.perf_counter()
        falta = self._vision_disponible()
        if falta:
            return self._fallo(falta, t0)
        pasos = [p for p in (pasos or []) if str(p).strip()]
        if not pasos:
            return self._fallo("no hay pasos que revisar", t0)
        if not confirmado:
            return self._fallo(
                "la lectura todavía no está confirmada. Revisa la transcripción y "
                "confírmala: no reviso un procedimiento que quizá no es el tuyo.", t0)

        from vision import prompt_revision_confirmada

        hallazgo, evidencia = self._comprobar_aritmetica(pasos)
        r = self.responder(prompt_revision_confirmada(
            pregunta or self.PREGUNTA_REVISION, enunciado, pasos, evidencia))
        r["etapa"] = "diagnostico"
        r["pasos_confirmados"] = list(pasos)
        r["transcripcion"] = "\n".join(f"Paso {i}: {p}"
                                       for i, p in enumerate(pasos, 1))
        r["verificador_aritmetico"] = hallazgo
        r["latencia_s"] = round(time.perf_counter() - t0, 2)
        return r

    PROMPT_RESOLVER = (
        "Resuelve este ejercicio para un estudiante de colegio.\n\n"
        "Enunciado: {enunciado}\n\n"
        "Escribe la solución en pasos cortos, uno por línea, con el formato "
        "«Paso N: …». Cada paso que haga una cuenta debe escribirla completa con "
        "su resultado, por ejemplo «Paso 2: 68 × 14 = 952». Termina con una línea "
        "«Respuesta final: …». Si el enunciado no es un ejercicio de matemáticas, "
        "o le falta un dato para resolverlo, dilo en una sola línea y no inventes "
        "el dato.")

    def resolver_enunciado(self, enunciado: str, confirmado: bool = False) -> dict:
        """Paso 2 de una foto que solo trae el enunciado: resolverlo.

        Igual que `revisar_pasos`: sin `confirmado=True` no hay solución, porque
        sería resolver lo que un modelo de visión creyó leer.

        Dos caminos, y la traza dice cuál se tomó:

          · si el enunciado ES una cuenta («¿Cuánto es 3/4 + 1/8?»), los pasos
            los escribe `explicar_cuenta` y el resultado lo calcula la
            herramienta registrada. No interviene el modelo de lenguaje.
          · si no, lo resuelve el modelo y después el verificador determinista
            revisa cada igualdad que escribió. Si alguna no cuadra se dice: no
            se corrige en silencio ni se da la solución por buena.
        """
        t0 = time.perf_counter()
        falta = self._vision_disponible()
        if falta:
            return self._fallo(falta, t0)
        enunciado = (enunciado or "").strip()
        if not enunciado:
            return self._fallo("no hay ningún enunciado que resolver", t0)
        if not confirmado:
            return self._fallo(
                "la lectura todavía no está confirmada. Revisa el enunciado y "
                "confírmalo: no resuelvo un ejercicio que quizá no es el tuyo.", t0)

        cuenta = cuenta_de(enunciado)
        explicada = explicar_cuenta(cuenta) if cuenta else None
        if explicada is not None:
            pasos, observaciones = explicada
            hallazgo, _ = self._comprobar_aritmetica(pasos)
            return {
                "ok": True, "etapa": "solucion", "camino": "herramienta",
                "enunciado": enunciado,
                "pasos_solucion": pasos,
                "respuesta": "\n".join(f"Paso {i}: {p}" if not p.startswith("Respuesta")
                                       else p for i, p in enumerate(pasos, 1)),
                "herramientas": [o["herramienta"] for o in observaciones],
                "observaciones": observaciones,
                "verificador_aritmetico": hallazgo,
                "pasos_llm": 0, "contextos": 0, "verificador": None,
                "latencia_s": round(time.perf_counter() - t0, 2),
            }

        r = self.responder(self.PROMPT_RESOLVER.format(enunciado=enunciado))
        if not r.get("ok"):
            return r
        lineas = [l.strip() for l in (r.get("respuesta") or "").splitlines()
                  if l.strip() and not re.match(r"(?i)^\s*(pensamiento|thought)\s*:", l)]
        hallazgo, _ = self._comprobar_aritmetica(lineas)
        r.update({"etapa": "solucion", "camino": "modelo", "enunciado": enunciado,
                  "pasos_solucion": lineas, "verificador_aritmetico": hallazgo,
                  "latencia_s": round(time.perf_counter() - t0, 2)})
        return r

    def _vision_disponible(self) -> str:
        """-> motivo por el que no se puede usar la vía visual, o cadena vacía."""
        if not self.cfg.vision:
            return (f"la configuración activa ({self.cfg.nombre}) no incluye la vía "
                    "visual. Use TUTOR_CONFIG=revision_foto para habilitarla, "
                    "teniendo en cuenta que esa configuración NO está medida.")
        return ""

    def _comprobar_aritmetica(self, pasos) -> tuple:
        """-> (hallazgo serializable | None, texto de evidencia | None).

        El verificador puede decir tres cosas y las tres son útiles distintas:
        encontré una igualdad que no cuadra, no encontré ninguna, o no había
        nada evaluable. Solo la primera entra como evidencia en el prompt.
        """
        try:
            from verificador_aritmetico import pista_para_el_tutor, revisar_hoja
        except Exception:  # noqa: BLE001
            return None, None
        h = revisar_hoja(list(pasos))
        if h is None:
            return None, None
        datos = {"paso": h.paso, "tipo": h.tipo, "detalle": h.detalle}
        if h.tipo not in ("calculo", "traslado"):
            # `ninguno` = las cuentas evaluables cuadran. Decírselo al tutor evita
            # que invente un error aritmético donde no lo hay, que es el fallo más
            # dañino de los dos posibles.
            if h.tipo == "ninguno":
                return datos, ("Las igualdades que se pueden comprobar en esta hoja "
                               "SÍ cuadran. Si hay un error, no es de aritmética.")
            return datos, None
        return datos, pista_para_el_tutor(h)

    def revisar_imagen(self, imagen, pregunta: str = "") -> dict:
        """Compatibilidad: el flujo de un solo paso que usaba M4.

        **Se detiene en la lectura.** Devuelve lo que se leyó y la pregunta de
        confirmación, nunca un diagnóstico, porque un diagnóstico sin confirmar
        es justo lo que `vis-10` demostró que no se puede ofrecer. Para
        diagnosticar hay que pasar por `revisar_pasos(..., confirmado=True)`.
        """
        return self.leer_imagen(imagen)

    def _cargar_vlm(self) -> bool:
        """Carga el VLM una sola vez. -> True si está listo.

        Separado de `_leer` para poder medirlo aparte: mientras la carga vivía
        dentro de la primera lectura, los 93,9 s de esa lectura eran descarga,
        carga e inferencia sumadas, y de ahí no sale ninguna latencia de lectura.
        No cambia nada del comportamiento: las dos lecturas independientes y la
        salvaguarda de `lectura_no_fiable` siguen exactamente igual.
        """
        if self._vlm is not None:
            return True
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            # El dispositivo se resuelve aquí y no se lee de `estado`: la vía
            # visual se puede invocar sin haber cargado el modelo de texto, y
            # entonces `estado.dispositivo` todavía vale "desconocido".
            cuda = torch.cuda.is_available()
            if self.estado.dispositivo == "desconocido":
                self.estado.dispositivo = "cuda" if cuda else "cpu"

            proc = AutoProcessor.from_pretrained(VLM_ID, max_pixels=1024 * 1024)
            # `device_map` es obligatorio, no una optimización. Sin él el VLM
            # se queda en CPU con pesos fp16 y `generate` muere con
            # «"addmm_impl_cpu_" not implemented for 'Half'», que es el mismo
            # par dtype/dispositivo que ya resuelve `scripts/nb_m4.py`.
            vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                VLM_ID,
                torch_dtype=torch.float16 if cuda else torch.float32,
                device_map="auto" if cuda else None).eval()
            self._vlm = (vlm, proc, torch)
            return True
        except Exception:  # noqa: BLE001
            return False

    def precargar_vision(self) -> dict:
        """Carga el VLM ahora y dice cuánto costó. -> dict con `segundos`.

        Para medir: deja la descarga y la carga fuera de la primera lectura.
        Para la demo: permite pagar los 90 s antes de que haya alguien mirando,
        en vez de durante la primera foto del público.
        """
        ya = self._vlm is not None
        t0 = time.perf_counter()
        ok = self._cargar_vlm()
        return {"ok": ok, "ya_estaba_cargado": ya,
                "segundos": round(time.perf_counter() - t0, 2),
                "dispositivo": self.estado.dispositivo}

    def _leer(self, imagen):
        """Carga el VLM una vez y hace dos lecturas independientes.

        Ambas lecturas deben producir JSON estructurado y coincidir en
        estructura, pasos y números. El acuerdo es una salvaguarda de rechazo,
        no una certificación de verdad; por eso el estudiante aún debe confirmar
        o corregir el texto antes de que exista un diagnóstico.
        """
        if not self._cargar_vlm():
            return None

        from vision import (INSTRUCCION_COPIAR, INSTRUCCION_COPIAR_INDEPENDIENTE,
                            estructurar_lineas, json_de, lecturas_consistentes)

        # Aquí, y no en la interfaz: las dos lecturas y el evaluador tienen que
        # ver exactamente los mismos píxeles, venga la imagen de Gradio, del
        # evaluador o de un test.
        imagen, normalizacion = normalizar_foto(imagen)

        vlm, proc, torch = self._vlm

        def transcribir(instruccion):
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": instruccion}]}]
            prompt = proc.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
            with torch.no_grad():
                ent = proc(text=[prompt], images=[imagen], return_tensors="pt").to(vlm.device)
                out = vlm.generate(**ent, max_new_tokens=320, do_sample=False)
            crudo = proc.batch_decode(out[:, ent["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()
            js = json_de(crudo)
            if isinstance(js, dict) and isinstance(js.get("lineas"), list):
                # El VLM copió líneas; enunciado y pasos los separa el código.
                js = estructurar_lineas(js["lineas"]) | {"lineas": js["lineas"]}
            return {"crudo": crudo, "json": js, "ok": isinstance(js, dict)}

        primera = transcribir(INSTRUCCION_COPIAR)
        segunda = transcribir(INSTRUCCION_COPIAR_INDEPENDIENTE)
        # No se asigna una consistencia por defecto: una segunda salida ausente o
        # no estructurada también impide diagnosticar desde la foto.
        primera["consistente"] = bool(
            primera["ok"] and segunda["ok"] and lecturas_consistentes(primera, segunda))
        primera["normalizacion"] = normalizacion
        # La segunda lectura no se enseña nunca al estudiante, pero quien depura
        # un rechazo necesita verla. Va aparte, con el JSON ya interpretado.
        primera["segunda"] = {"crudo": segunda.get("crudo"), "json": segunda.get("json"),
                              "ok": segunda.get("ok")}
        if not primera["consistente"]:
            from vision import diferencias_de_lectura
            primera["diferencias"] = (diferencias_de_lectura(primera, segunda)
                                      if primera["ok"] and segunda["ok"]
                                      else ["alguna lectura no salió en JSON"])
            # Al log del servidor, no a la pantalla: es para depurar el
            # rechazo, y enseñarle al estudiante una lectura dudosa invita a
            # confirmarla.
            print(f"[lectura] no coinciden · {primera['diferencias']}\n"
                  f"  1ª: {primera.get('crudo', '')[:400]!r}\n"
                  f"  2ª: {segunda.get('crudo', '')[:400]!r}", flush=True)
        return primera

    @staticmethod
    def _fallo(motivo, t0):
        return {"ok": False, "respuesta": motivo, "herramientas": [], "pasos_llm": 0,
                "contextos": 0, "latencia_s": round(time.perf_counter() - t0, 2)}


_TUTOR: Tutor | None = None


def obtener_tutor() -> Tutor:
    """Una sola instancia por proceso: montar el pipeline cuesta."""
    global _TUTOR
    if _TUTOR is None:
        _TUTOR = Tutor()
    return _TUTOR
