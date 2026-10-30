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
        if not self.montado:
            return f"{self.config_nombre} (sin montar todavía)"
        if not self.degradado:
            return self.config_nombre
        return (f"MODO DEGRADADO (se pidió {self.config}; falta "
                f"{', '.join(self.faltantes)})")

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


def buscador_hibrido(chunks):
    """El retrieval de S08-C, que es el que S08-D enruta.

    Reproduce la cadena del notebook, no una aproximación:

        denso top-10  +  BM25 top-10   -> RRF (k=60)      `buscar_hibrida`
        -> cross-encoder mmarco sobre los 10 -> top-3      `buscar_con_rerank`

    El reranker **no es opcional**: `D · +router` se construye sobre
    `C · +rerank` (`nb_rag_avanzado.py:559`), y el enrutador usa como tercer
    rasgo el puntaje máximo del cross-encoder (`enrutador.RASGOS`). Sin él se
    monta otra cosa y se la presenta como la de 18/21.

    Devuelve None si falta alguna dependencia, para que el pipeline se repliegue
    diciéndolo en vez de aparentar que montó S08-D.
    """
    try:
        import re

        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError:
        return None

    def tokenizar(texto):
        import unicodedata
        t = unicodedata.normalize("NFKD", (texto or "").lower())
        return re.findall(r"\w+", "".join(c for c in t if not unicodedata.combining(c)))

    try:
        bm25 = BM25Okapi([tokenizar(c) for c in chunks])
        st = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                                max_length=512)
    except Exception:      # noqa: BLE001 — sin red o sin caché de modelos
        return None

    emb = st.encode(chunks, show_progress_bar=False)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
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

    def _cargar_modelo(self):
        """Carga el modelo la primera vez que se necesita."""
        if self._agente is not None or "modelo" in " ".join(self.estado.motivos):
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            self.estado.motivos.append(f"modelo: falta {e.name}")
            return

        self.estado.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            kwargs = {"torch_dtype": torch.float16 if self.estado.dispositivo == "cuda"
                      else torch.float32}
            if CUANTIZAR:
                from transformers import BitsAndBytesConfig
                kwargs = {"quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16)}
                self.estado.cuantizado = True

            tok = AutoTokenizer.from_pretrained(MODELO_BASE)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE, **kwargs)
            if not CUANTIZAR:
                modelo = modelo.to(self.estado.dispositivo)

            ruta = self.comprobar_adaptador()
            if ruta is not None:
                modelo = PeftModel.from_pretrained(
                    modelo, str(ruta), adapter_name=self.cfg.adaptador)
                self.estado.adaptador_cargado = self.cfg.adaptador
            self._armar_agente(modelo.eval(), tok)
            self.estado.modelo = True
        except Exception as e:  # noqa: BLE001
            self.estado.motivos.append(f"modelo: {type(e).__name__}: {e}"[:160])

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
            buscar = buscador_hibrido(self._chunks)
            activo = "hibrido + rerank (BM25+denso, RRF, cross-encoder) = S08-C"
            if buscar is None:
                self.estado.faltantes.append("retrieval híbrido + reranker")
                self.estado.motivos.append(
                    "retrieval: falta rank_bm25, sentence-transformers o los "
                    "modelos en caché; se usa búsqueda léxica, que NO es la de S08")
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
            "pasos_llm": salida.pasos_llm,
            "contextos": len(salida.contextos),
            "verificador": hallazgo,
            "latencia_s": round(time.perf_counter() - t0, 2),
        }

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
            return base | {
                "confirmable": False,
                "motivo_no_fiable": "no se leyó ningún paso de un procedimiento",
                "respuesta": ("Leí la hoja pero no encuentro los pasos de un "
                              "procedimiento matemático. Si es un ejercicio, "
                              "fotografía la parte donde están las operaciones."),
                "pregunta": "¿Es esto un ejercicio de matemáticas?",
            }
        return base | {
            "confirmable": True,
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

    def _leer(self, imagen):
        """Carga el VLM una vez y hace dos lecturas independientes.

        Ambas lecturas deben producir JSON estructurado y coincidir en
        estructura, pasos y números. El acuerdo es una salvaguarda de rechazo,
        no una certificación de verdad; por eso el estudiante aún debe confirmar
        o corregir el texto antes de que exista un diagnóstico.
        """
        if self._vlm is None:
            try:
                import torch
                from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

                proc = AutoProcessor.from_pretrained(VLM_ID, max_pixels=1024 * 1024)
                vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                    VLM_ID, torch_dtype=torch.float16
                    if self.estado.dispositivo == "cuda" else torch.float32).eval()
                self._vlm = (vlm, proc, torch)
            except Exception:  # noqa: BLE001
                return None

        from vision import (INSTRUCCION_VER, INSTRUCCION_VER_INDEPENDIENTE,
                            json_de, lecturas_consistentes)

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
            return {"crudo": crudo, "json": js, "ok": isinstance(js, dict)}

        primera = transcribir(INSTRUCCION_VER)
        segunda = transcribir(INSTRUCCION_VER_INDEPENDIENTE)
        # No se asigna una consistencia por defecto: una segunda salida ausente o
        # no estructurada también impide diagnosticar desde la foto.
        primera["consistente"] = bool(
            primera["ok"] and segunda["ok"] and lecturas_consistentes(primera, segunda))
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
