"""Notebook S08 — RAG avanzado sobre el tutor de matemáticas (cierre de M3)."""

from __future__ import annotations

from nb_comun import CODIGO_METRICAS, celda_drive, code, encabezado, md
from nb_comun_eval import (
    celda_eval_set, celda_evaluacion, celda_harness, celda_juez, celda_similitud,
)
from nb_comun_rag import CODIGO_CHUNKING, CODIGO_CORPUS, CODIGO_PROMPTS


# --------------------------------------------------------------------------
# Consultas de desarrollo
# --------------------------------------------------------------------------
# Viven fuera del eval set a propósito. Sirven para dos cosas que NO pueden
# hacerse con el eval set sin contaminarlo: medir el retrieval aislado y
# calibrar el enrutador. Si el enrutador se ajustara con los 21 casos del
# harness, su acierto sobre esos mismos casos no demostraría nada.

CODIGO_CONSULTAS_DEV = r'''
# tipo  : "consulta" (pide un dato de un documento) | "problema" (hay que calcular)
# clave : texto que DEBE aparecer en el chunk que responde. None = el dato no está
#         en el corpus (la respuesta correcta es abstenerse).
consultas_dev = [
    # --- consultas con respuesta en el corpus: literales ---
    {"q": "¿Cuánto valen los talleres en la nota del periodo?",
     "tipo": "consulta", "estilo": "literal", "clave": "talleres al 25 por ciento"},
    {"q": "¿Cuál es la nota máxima que se puede obtener en una recuperación?",
     "tipo": "consulta", "estilo": "literal", "clave": "recuperacion es 3.5"},
    {"q": "¿En qué grado se trabajan los sistemas de dos ecuaciones lineales?",
     "tipo": "consulta", "estilo": "literal", "clave": "sistemas de dos ecuaciones"},
    {"q": "¿Cuántas sesiones recomienda el área para la traducción de enunciados verbales?",
     "tipo": "consulta", "estilo": "literal", "clave": "dos sesiones"},
    {"q": "¿Qué dice el formulario sobre el volumen del cilindro?",
     "tipo": "consulta", "estilo": "literal", "clave": "cilindro: pi"},
    # --- consultas con respuesta en el corpus: siglas y términos exactos (terreno de BM25) ---
    {"q": "¿Qué escala de calificación establece el SIEE?",
     "tipo": "consulta", "estilo": "sigla", "clave": "escala institucional"},
    {"q": "Plan de área: contenidos de grado décimo",
     "tipo": "consulta", "estilo": "sigla", "clave": "decimo se trabajan"},
    # --- consultas con respuesta en el corpus: coloquiales (terreno del denso) ---
    {"q": "mi hija faltó al examen y no llevó excusa, ¿qué nota le ponen?",
     "tipo": "consulta", "estilo": "coloquial", "clave": "se califica con 1.0"},
    {"q": "¿cuántos días tengo para llevar la excusa si mi hijo no fue a la evaluación?",
     "tipo": "consulta", "estilo": "coloquial", "clave": "tres dias habiles"},
    {"q": "¿qué le recomiendan al profe cuando los niños creen que más obreros tardan más días?",
     "tipo": "consulta", "estilo": "coloquial", "clave": "dias-obrero"},
    {"q": "mi hijo dividió los pasajeros entre los buses y le dio un número con decimales, ¿qué hago?",
     "tipo": "consulta", "estilo": "coloquial", "clave": "redondearse hacia arriba"},
    {"q": "¿qué se ve en séptimo en matemáticas?",
     "tipo": "consulta", "estilo": "coloquial", "clave": "septimo se abordan"},
    # --- consultas con NÚMEROS: piden un dato aunque parezcan un problema ---
    {"q": "Mi hijo sacó 2.5 en el examen y 2.8 en la recuperación, ¿cuál es la nota máxima que le pueden dar al recuperar?",
     "tipo": "consulta", "estilo": "con_numeros", "clave": "recuperacion es 3.5"},
    {"q": "Si mi hija faltó al examen del 15 de mayo, ¿cuántos días tengo para llevar la excusa?",
     "tipo": "consulta", "estilo": "con_numeros", "clave": "tres dias habiles"},
    # --- consultas SIN respuesta en el corpus: la válvula de escape debe activarse ---
    {"q": "¿Cuántas horas semanales de matemáticas tiene grado once?",
     "tipo": "consulta", "estilo": "sin_respuesta", "clave": None},
    {"q": "¿Qué calculadora exige el colegio para las evaluaciones?",
     "tipo": "consulta", "estilo": "sin_respuesta", "clave": None},
    {"q": "¿Cuándo es la reunión de padres del área de matemáticas?",
     "tipo": "consulta", "estilo": "sin_respuesta", "clave": None},

    # --- problemas: hay que calcular, ningún documento trae la respuesta ---
    {"q": "Un pantalón cuesta 120000 pesos y tiene un descuento del 25%. ¿Cuánto se paga?",
     "tipo": "problema", "estilo": "porcentajes", "clave": None},
    {"q": "Si 4 grifos llenan un tanque en 12 horas, ¿cuánto tardan 6 grifos?",
     "tipo": "problema", "estilo": "proporcionalidad", "clave": None},
    {"q": "Resuelve la ecuación 3x + 7 = 25.",
     "tipo": "problema", "estilo": "ecuaciones", "clave": None},
    {"q": "Calcula el área de un triángulo de base 14 cm y altura 9 cm.",
     "tipo": "problema", "estilo": "geometria", "clave": None},
    {"q": "Laura leyó 1/4 de un libro de 240 páginas y luego la mitad de lo que le faltaba. ¿Cuántas páginas le quedan?",
     "tipo": "problema", "estilo": "fracciones", "clave": None},
    {"q": "¿Cuánto es 8 + 4 × 3 − 6 ÷ 2?",
     "tipo": "problema", "estilo": "jerarquia", "clave": None},
    {"q": "Las notas de Pedro son 3.5, 4.2 y 4.0. ¿Cuál es su promedio?",
     "tipo": "problema", "estilo": "estadistica", "clave": None},
    {"q": "En una bolsa hay 5 bolas azules y 3 rojas. ¿Cuál es la probabilidad de sacar una roja?",
     "tipo": "problema", "estilo": "probabilidad", "clave": None},
    {"q": "Un terreno rectangular mide 35 m por 20 m. ¿Cuántos metros de cerca se necesitan para rodearlo?",
     "tipo": "problema", "estilo": "geometria", "clave": None},
    {"q": "Un celular subió de 800000 a 920000 pesos. ¿En qué porcentaje aumentó?",
     "tipo": "problema", "estilo": "porcentajes", "clave": None},
    {"q": "Un carro recorre 360 km con 30 litros de gasolina. ¿Cuántos litros necesita para 540 km?",
     "tipo": "problema", "estilo": "proporcionalidad", "clave": None},
    {"q": "mi hijo tiene que sumar 2/3 + 3/4 para mañana, ¿cómo se hace?",
     "tipo": "problema", "estilo": "coloquial", "clave": None},
    {"q": "Si saqué 3.2 en talleres y 4.5 en el examen, y valen 30% y 70%, ¿cuál es mi nota?",
     "tipo": "problema", "estilo": "frontera", "clave": None},
    # --- problemas SIN números (o con uno solo): hay que razonar igual ---
    {"q": "¿Cuánto es la mitad de un tercio?",
     "tipo": "problema", "estilo": "sin_numeros", "clave": None},
    {"q": "Si duplico el lado de un cuadrado, ¿qué le pasa a su área?",
     "tipo": "problema", "estilo": "sin_numeros", "clave": None},
    {"q": "¿Cómo se calcula el área de un círculo de radio 3 cm?",
     "tipo": "problema", "estilo": "sin_numeros", "clave": None},
]

from collections import Counter
print(f"Consultas de desarrollo: {len(consultas_dev)}  ->  {dict(Counter(c['tipo'] for c in consultas_dev))}")
print("Estilos:", dict(Counter(c["estilo"] for c in consultas_dev if c["tipo"] == "consulta")))
'''


CODIGO_CONTAMINACION = r'''
# Chequeo de contaminación: ninguna consulta de desarrollo puede ser (casi) un
# caso del eval set. Si lo fuera, calibrar el enrutador con ella sería calibrarlo
# con el examen.
emb_dev  = st.encode([c["q"] for c in consultas_dev], normalize_embeddings=True)
emb_eval = st.encode([c["input"] for c in eval_set], normalize_embeddings=True)
sims = emb_dev @ emb_eval.T

UMBRAL_CONTAMINACION = 0.90
peor = np.unravel_index(sims.argmax(), sims.shape)
print(f"Similitud máxima dev ↔ eval: {sims.max():.3f}")
print(f"  dev : {consultas_dev[peor[0]]['q'][:80]}")
print(f"  eval: {eval_set[peor[1]]['input'][:80]}")
assert not {c["q"] for c in consultas_dev} & {c["input"] for c in eval_set}, "consulta repetida"
assert sims.max() < UMBRAL_CONTAMINACION, "hay una consulta de desarrollo casi idéntica a un caso del eval set"

# Y cada clave tiene que existir en el corpus: si no, la métrica de retrieval
# estaría midiendo un chunk imposible de encontrar.
texto_corpus = _norm(" ".join(d["texto"] for d in corpus))
for c in consultas_dev:
    if c["clave"]:
        assert _norm(c["clave"]) in texto_corpus, f"clave inexistente en el corpus: {c['clave']!r}"
print("\nSin contaminación y todas las claves existen en el corpus.")
'''


CODIGO_INDICE = r'''
import time
import chromadb

cliente = chromadb.Client()
try:
    cliente.delete_collection("tutor_matematicas_s08")
except Exception:
    pass
coleccion = cliente.create_collection("tutor_matematicas_s08", metadata={"hnsw:space": "cosine"})
coleccion.add(ids=ids_chunks, documents=chunks, metadatas=metadatos,
              embeddings=st.encode(chunks, show_progress_bar=False).tolist())
POS = {cid: i for i, cid in enumerate(ids_chunks)}


def buscar_densa(consulta, k=10):
    """Sistema A · el retrieval de S07. Devuelve un RANKING de índices de chunk:
    la moneda común de todas las búsquedas de hoy, para poder fusionarlas."""
    r = coleccion.query(query_embeddings=st.encode([consulta]).tolist(),
                        n_results=min(k, len(chunks)))
    return [POS[cid] for cid in r["ids"][0]]


print("Índice denso:", coleccion.count(), "chunks (mismo corpus y mismo chunking que S07)")
'''


CODIGO_BM25 = r'''
from rank_bm25 import BM25Okapi


def tokenizar(texto):
    """Minúsculas, sin tildes y sin puntuación.

    La plantilla de clase usa `texto.lower().split()`, que deja "¿cuánto" y
    "cuanto" como tokens distintos y pega los signos a las palabras. En español,
    con preguntas que empiezan por "¿" y usuarios que no ponen tildes, eso hace
    que BM25 falle por ortografía en lugar de por significado. `_norm` es la
    misma normalización que usa el criterio de acierto del harness.
    """
    return re.findall(r"\w+", _norm(texto))


bm25 = BM25Okapi([tokenizar(ch) for ch in chunks])


def buscar_bm25(consulta, k=10):
    scores = bm25.get_scores(tokenizar(consulta))
    return [int(i) for i in np.argsort(scores)[::-1][:k]]


def mostrar(nombre, ranking, k=3):
    print(f"  {nombre:9s}: {[ids_chunks[i] for i in ranking[:k]]}")


for q in ["¿Qué escala de calificación establece el SIEE?",
          "mi hija faltó al examen y no llevó excusa, ¿qué nota le ponen?"]:
    print("CONSULTA:", q)
    mostrar("densa", buscar_densa(q))
    mostrar("BM25", buscar_bm25(q))
    print()
'''


CODIGO_RRF = r'''
def rrf(listas, k=10, krrf=60):
    """Reciprocal Rank Fusion: cada documento suma 1/(krrf + puesto) en cada lista.
    Fusiona por PUESTOS, así que no importa que BM25 puntúe ~12 y el coseno ~0.8."""
    puntos = {}
    for lista in listas:
        for puesto, idx in enumerate(lista):
            puntos[idx] = puntos.get(idx, 0.0) + 1.0 / (krrf + puesto + 1)
    return [idx for idx, _ in sorted(puntos.items(), key=lambda x: -x[1])][:k]


def buscar_hibrida(consulta, k=10, k_listas=10):
    """Sistema B · denso + BM25 fusionados con RRF.
    Cada lista aporta su top-10 aunque se pidan 3: si se fusionaran solo los
    top-3, un chunk en el puesto 4 de ambas listas (consenso) no podría subir."""
    n = max(k, k_listas)
    return rrf([buscar_densa(consulta, n), buscar_bm25(consulta, n)], k=k)


for q in ["¿Qué escala de calificación establece el SIEE?",
          "mi hija faltó al examen y no llevó excusa, ¿qué nota le ponen?"]:
    print("CONSULTA:", q)
    mostrar("densa", buscar_densa(q))
    mostrar("BM25", buscar_bm25(q))
    mostrar("HÍBRIDA", buscar_hibrida(q))
    print()
'''


CODIGO_RERANKER = r'''
from sentence_transformers import CrossEncoder

# Cross-encoder multilingüe entrenado en mMARCO (incluye español).
reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512)
K_RECUPERAR = 10      # ancho, con lo barato
K = 3                 # final, con lo bueno (el mismo k de S07)


def reordenar(consulta, candidatos):
    """Devuelve (candidatos reordenados, puntajes del cross-encoder en ese orden)."""
    scores = reranker.predict([(consulta, chunks[i]) for i in candidatos], show_progress_bar=False)
    orden = np.argsort(scores)[::-1]
    return [candidatos[i] for i in orden], [float(scores[i]) for i in orden]


def buscar_con_rerank(consulta, k=K):
    """Sistema C · híbrida (top-10) + cross-encoder (top-k)."""
    candidatos = buscar_hibrida(consulta, K_RECUPERAR)
    return reordenar(consulta, candidatos)[0][:k]


q = "¿qué le recomiendan al profe cuando los niños creen que más obreros tardan más días?"
print("CONSULTA:", q)
mostrar("densa", buscar_densa(q))
mostrar("híbrida", buscar_hibrida(q))
mostrar("+rerank", buscar_con_rerank(q))
'''


CODIGO_EVAL_RETRIEVAL = r'''
import pandas as pd

# Las 4 consultas de conocimiento del eval set también tienen un chunk de oro:
# se suman a la evaluación del retrieval (no hay generación, no se contamina nada).
CLAVE_ORO_EVAL = {
    "con-01": "examen final del periodo equivale al 40",
    "con-02": "octavo se introducen las ecuaciones",
    "con-03": "nota minima aprobatoria es 3.0",
    "con-04": "factor multiplicativo",
}
consultas_retrieval = (
    [{"q": c["q"], "origen": "dev", "estilo": c["estilo"], "clave": c["clave"]}
     for c in consultas_dev if c["tipo"] == "consulta" and c["clave"]]
    + [{"q": c["input"], "origen": c["id"], "estilo": "eval", "clave": CLAVE_ORO_EVAL[c["id"]]}
       for c in eval_set if c["id"] in CLAVE_ORO_EVAL]
)


def chunks_oro(clave):
    return {i for i, ch in enumerate(chunks) if _norm(clave) in _norm(ch)}


sin_oro = [c["q"] for c in consultas_retrieval if not chunks_oro(c["clave"])]
if sin_oro:
    print("AVISO: la clave quedó partida entre dos chunks en:", sin_oro)
consultas_retrieval = [c for c in consultas_retrieval if chunks_oro(c["clave"])]

RETRIEVERS = {"A · densa": buscar_densa, "B · híbrida": buscar_hibrida,
              "C · +rerank": buscar_con_rerank}

filas = []
for nombre, fn in RETRIEVERS.items():
    for c in consultas_retrieval:
        t0 = time.perf_counter()
        ranking = fn(c["q"], K) if fn is buscar_con_rerank else fn(c["q"], K_RECUPERAR)
        ms = (time.perf_counter() - t0) * 1000
        oro = chunks_oro(c["clave"])
        puesto = next((p for p, i in enumerate(ranking, 1) if i in oro), None)
        filas.append({"retriever": nombre, "origen": c["origen"], "estilo": c["estilo"],
                      "q": c["q"], "puesto_oro": puesto,
                      "hit@1": puesto == 1, "hit@3": puesto is not None and puesto <= 3,
                      "rr": 1 / puesto if puesto and puesto <= 3 else 0.0, "ms": ms})

df_ret = pd.DataFrame(filas)
resumen_ret = (df_ret.groupby("retriever", sort=False)
               .agg(n=("q", "size"), hit1=("hit@1", "mean"), hit3=("hit@3", "mean"),
                    mrr3=("rr", "mean"), ms=("ms", "median")).round(3))
print(f"Retrieval aislado sobre {len(consultas_retrieval)} consultas con chunk de oro (k={K})\n")
print(resumen_ret.to_string())
print("\nhit@3 por estilo de consulta:")
print(df_ret.pivot_table(index="estilo", columns="retriever", values="hit@3",
                         aggfunc="mean", sort=False).round(2).to_string())
'''


CODIGO_FALLOS_RETRIEVAL = r'''
# Caso por caso: dónde el chunk de oro NO quedó en el top-3.
fallos_ret = df_ret[~df_ret["hit@3"]]
if fallos_ret.empty:
    print("Los tres retrievers ponen el chunk de oro en el top-3 en todas las consultas.")
else:
    print(fallos_ret[["retriever", "estilo", "puesto_oro", "q"]].to_string(index=False))
'''


CODIGO_ENRUTADOR = r'''
# ---------------------------------------------------------------------------
# ENRUTADOR · ¿la pregunta pide un DATO de un documento (-> RAG) o es un
# PROBLEMA que hay que resolver (-> el tutor de M2, sin contexto)?
#
# Tres señales baratas, cada una con un punto ciego distinto:
#   margen  : similitud al centroide "consulta" menos al centroide "problema".
#             Separa por TEMA, así que falla con "¿cuánto valen los talleres?"
#             (suena a cálculo) o "valen 30% y 70%, ¿cuál es mi nota?" (suena a SIEE).
#   numeros : cuántos números trae la pregunta (tope 3). Para calcular hacen
#             falta datos; pero "la mitad de un tercio" no trae ninguno.
#   rerank  : puntaje máximo del cross-encoder sobre el corpus. Dice si ALGÚN
#             chunk responde; falla con las consultas sin respuesta en el corpus.
# Una regresión logística las combina. Todo se ajusta SOLO con consultas_dev.
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TIPOS = ["consulta", "problema"]
RASGOS = ["margen", "numeros", "rerank"]


def ajustar_centroides(consultas):
    emb = st.encode([c["q"] for c in consultas], normalize_embeddings=True)
    cent = {}
    for t in TIPOS:
        v = emb[[c["tipo"] == t for c in consultas]].mean(axis=0)
        cent[t] = v / np.linalg.norm(v)
    return cent


def margen_centroides(pregunta, centroides):
    e = st.encode([pregunta], normalize_embeddings=True)[0]
    return float(e @ centroides["consulta"] - e @ centroides["problema"])


def contar_numeros(pregunta):
    return min(len(re.findall(r"\d+(?:[.,/]\d+)?", pregunta)), 3)


_CACHE_RERANK = {}


def rerank_max(pregunta):
    if pregunta not in _CACHE_RERANK:
        _CACHE_RERANK[pregunta] = reordenar(pregunta, buscar_hibrida(pregunta, K_RECUPERAR))[1][0]
    return _CACHE_RERANK[pregunta]


def rasgos(pregunta, centroides):
    return [margen_centroides(pregunta, centroides), contar_numeros(pregunta), rerank_max(pregunta)]


y_dev = np.array([c["tipo"] == "consulta" for c in consultas_dev])

# Leave-one-out HONESTO: en cada pliegue también los centroides se recalculan
# sin la consulta que se va a predecir (si no, su margen ya la habría "visto").
pred_loo, pred_loo_cent = [], []
for i in range(len(consultas_dev)):
    cent_i = ajustar_centroides(consultas_dev[:i] + consultas_dev[i + 1:])
    X_i = np.array([rasgos(d["q"], cent_i) for d in consultas_dev])
    resto = np.arange(len(consultas_dev)) != i
    clf_i = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_i[resto], y_dev[resto])
    pred_loo.append(bool(clf_i.predict(X_i[i:i + 1])[0]))
    pred_loo_cent.append(bool(X_i[i, 0] > 0))          # solo centroides, para comparar

acc = float(np.mean(np.array(pred_loo) == y_dev))
acc_cent = float(np.mean(np.array(pred_loo_cent) == y_dev))
print(f"Leave-one-out sobre {len(consultas_dev)} consultas de desarrollo")
print(f"  solo centroides       : {acc_cent:.1%}")
print(f"  enrutador (3 señales) : {acc:.1%}")
for c, p in zip(consultas_dev, pred_loo):
    if p != (c["tipo"] == "consulta"):
        print(f"    error: {c['tipo']:>8} -> {TIPOS[not p]:8s} {c['q'][:62]}")

# Modelo final: centroides y regresión con TODAS las consultas de desarrollo.
CENTROIDES = ajustar_centroides(consultas_dev)
X_dev = np.array([rasgos(c["q"], CENTROIDES) for c in consultas_dev])
ROUTER = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_dev, y_dev)
coef = ROUTER[-1].coef_[0]
print("\nPeso de cada señal (estandarizada; positivo empuja hacia 'consulta'):")
for n, w in zip(RASGOS, coef):
    print(f"  {n:8s} {w:+.2f}")


def clasificar(pregunta):
    """Devuelve (ruta, probabilidad de 'consulta')."""
    p = float(ROUTER.predict_proba([rasgos(pregunta, CENTROIDES)])[0][1])
    return ("consulta" if p >= 0.5 else "problema"), p
'''


CODIGO_RUTAS_EVAL = r'''
# Ahora sí, sobre el eval set (que el enrutador NO ha visto).
# Ruta esperada: conocimiento -> consulta, cálculo -> problema.
# Los adversariales no tienen ruta "correcta": se muestran para leer su efecto.
ESPERADA = {"conocimiento": "consulta", "calculo": "problema"}

rutas_eval = {}
print(f"{'caso':8s} {'bloque':13s} {'ruta':9s} {'P(cons.)':>8} {'margen':>8} {'nums':>5} {'rerank':>7}")
print("-" * 70)
for c in eval_set:
    ruta, prob = clasificar(c["input"])
    rutas_eval[c["id"]] = ruta
    esp = ESPERADA.get(c["bloque"])
    marca = "" if esp is None else ("ok" if ruta == esp else "ERROR")
    m_, n_, r_ = rasgos(c["input"], CENTROIDES)
    print(f"{c['id']:8s} {c['bloque']:13s} {ruta:9s} {prob:>8.2f} {m_:>+8.3f} {n_:>5d} {r_:>+7.2f}  {marca}")

evaluables = [c for c in eval_set if c["bloque"] in ESPERADA]
ok = sum(rutas_eval[c["id"]] == ESPERADA[c["bloque"]] for c in evaluables)
print("-" * 70)
print(f"Enrutador sobre cálculo + conocimiento del eval set: {ok}/{len(evaluables)}")
'''


CODIGO_GENERADOR = r'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

MODELO_BASE   = "Qwen/Qwen2.5-1.5B-Instruct"
DIR_ADAPTADOR = "adaptadores/qwen-lora"      # el modelo afinado de M1
MAX_TOKENS_GEN = 220

if not Path(DIR_ADAPTADOR).exists():
    raise FileNotFoundError(
        f"No se encuentra {DIR_ADAPTADOR}. Ejecuten S04_Lab_Fine_tuning_Qwen.ipynb "
        "o monten el Drive donde quedó guardado.")

tok = AutoTokenizer.from_pretrained(DIR_ADAPTADOR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_base = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float16).to(DEVICE)
modelo = PeftModel.from_pretrained(_base, DIR_ADAPTADOR).eval()
print("Generador listo:", MODELO_BASE, "+ adaptador LoRA de M1")
'''


CODIGO_SISTEMAS = r'''
# Sistema E · el prompt que el enrutador hace posible.
#
# El prompt de dos modos de S07 era un compromiso: como TODAS las preguntas
# pasaban por el RAG, tenía que servir a la vez para citar un reglamento y para
# resolver un problema. Con el enrutador, a la vía RAG solo llegan consultas, así
# que el prompt puede ser de un solo modo y mucho más exigente con la extracción.
# Ataca el fallo de GENERACIÓN de con-02 (el chunk llegó y el modelo no lo usó).
SYSTEM_RAG_CITA = (
    "Eres el asistente documental de un tutor de matemáticas. Responde ÚNICAMENTE "
    "con base en el contexto.\n"
    "1. Busca en el contexto la frase que responde la pregunta y cópiala "
    "textualmente entre comillas, indicando la fuente.\n"
    "2. Termina con una línea 'Respuesta final: <el dato pedido>'.\n"
    'Si ningún fragmento del contexto responde la pregunta, responde exactamente: '
    '"No tengo esa información en mis fuentes." No inventes datos.'
)

# ---------------------------------------------------------------------------
# Traza y caché.
#
# Caché: la decodificación es greedy, así que el mismo (system, user) produce
# SIEMPRE el mismo texto (S07 lo verificó: su "sin RAG" reprodujo M2 cifra por
# cifra). Guardarlo no cambia ningún resultado y hace que D y E no regeneren lo
# que ya generaron "sin RAG" y C: la corrida completa cuesta ~3 sistemas, no 6.
#
# Traza: por cada (sistema, pregunta) se guarda la ruta, los chunks del prompt y
# los tokens de entrada. Es lo que faltaba en S07 (documentación §6.4): saber
# QUÉ casos rompió cada sistema y con qué contexto.
# ---------------------------------------------------------------------------
_CACHE_GEN = {}
TRAZA = {}


@torch.no_grad()
def _generar(system, user, max_new_tokens=MAX_TOKENS_GEN):
    clave = (system, user, max_new_tokens)
    if clave in _CACHE_GEN:
        return _CACHE_GEN[clave]
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(modelo.device)
    out = modelo.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    texto = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    _CACHE_GEN[clave] = texto
    return texto


def _tokens_prompt(system, user):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    return len(tok(prompt, add_special_tokens=False)["input_ids"])


def _contexto(idxs):
    return "\n\n".join(f"[Fuente: {metadatos[i]['fuente']}]\n{chunks[i]}" for i in idxs)


def sistema_sin_rag(pregunta, _nombre="sin RAG"):
    TRAZA[(_nombre, pregunta)] = {"ruta": "problema", "top_k": [],
                                  "tokens_prompt": _tokens_prompt(INSTRUCCION, pregunta)}
    return _generar(INSTRUCCION, pregunta)


def construir_rag(nombre, busqueda, system=SYSTEM_RAG):
    def sistema(pregunta):
        t0 = time.perf_counter()
        idxs = busqueda(pregunta, K)
        ms = (time.perf_counter() - t0) * 1000
        user = f"Contexto:\n{_contexto(idxs)}\n\nPregunta: {pregunta}"
        TRAZA[(nombre, pregunta)] = {"ruta": "consulta", "top_k": [ids_chunks[i] for i in idxs],
                                     "tokens_prompt": _tokens_prompt(system, user), "ms_retrieval": ms}
        return _generar(system, user)
    return sistema


def construir_enrutado(nombre, sistema_consulta, nombre_consulta):
    def sistema(pregunta):
        ruta, prob = clasificar(pregunta)
        if ruta == "consulta":
            respuesta = sistema_consulta(pregunta)
            TRAZA[(nombre, pregunta)] = dict(TRAZA[(nombre_consulta, pregunta)], p_consulta=prob)
        else:
            respuesta = sistema_sin_rag(pregunta)
            TRAZA[(nombre, pregunta)] = dict(TRAZA[("sin RAG", pregunta)], p_consulta=prob)
        return respuesta
    return sistema


sistema_A = construir_rag("A · ingenuo", buscar_densa)
sistema_B = construir_rag("B · +hybrid", buscar_hibrida)
sistema_C = construir_rag("C · +rerank", buscar_con_rerank)
_sistema_C_cita = construir_rag("C · cita", buscar_con_rerank, system=SYSTEM_RAG_CITA)
sistema_D = construir_enrutado("D · +router", sistema_C, "C · +rerank")
sistema_E = construir_enrutado("E · +cita", _sistema_C_cita, "C · cita")

SISTEMAS = [
    ("sin RAG",      sistema_sin_rag, "El tutor de M2. Referencia."),
    ("A · ingenuo",  sistema_A, "RAG de S07: densa top-3 + prompt de dos modos."),
    ("B · +hybrid",  sistema_B, "A + BM25 fusionado con RRF."),
    ("C · +rerank",  sistema_C, "B + cross-encoder sobre el top-10."),
    ("D · +router",  sistema_D, "C solo para consultas; los problemas van sin RAG."),
    ("E · +cita",    sistema_E, "D con prompt de un solo modo y cita textual en la vía RAG."),
]
for n, _, d in SISTEMAS:
    print(f"  {n:13s} {d}")
'''


CODIGO_PRUEBA_CON02 = r'''
# Antes de gastar GPU en el harness completo: el caso que motivó el sistema E.
# En S07 el chunk correcto del plan de área LLEGÓ al prompt y el modelo no
# respondió "octavo". ¿Lo cambia el retrieval (C)? ¿Lo cambia el prompt (E)?
caso = next(c for c in eval_set if c["id"] == "con-02")
print("PREGUNTA:", caso["input"], "\n")
for nombre, fn in [("A · ingenuo", sistema_A), ("C · +rerank", sistema_C), ("E · +cita", sistema_E)]:
    r = fn(caso["input"])
    t = TRAZA[(nombre, caso["input"])]
    print(f"--- {nombre}   top-k={t['top_k']}")
    print(r[:350])
    print("   ->", evaluar_caso(caso, r)["motivo"], "\n")
'''


CODIGO_JUEZ_CACHE = r'''
# El juez también es greedy: misma (pregunta, respuesta) -> misma nota. D y E
# repiten muchas respuestas de "sin RAG" y de C; no tiene sentido re-juzgarlas.
_juez_sin_cache = juez_puntua
_CACHE_JUEZ = {}


def juez_puntua(pregunta, respuesta, esperada=None):
    clave = (pregunta, respuesta, esperada)
    if clave not in _CACHE_JUEZ:
        _CACHE_JUEZ[clave] = _juez_sin_cache(pregunta, respuesta, esperada)
    return _CACHE_JUEZ[clave]
'''


CODIGO_CORRIDA = r'''
nombres = [n for n, _, _ in SISTEMAS]
scorecards = {}
for nombre, fn, _ in SISTEMAS:
    print(f"\n=== {nombre} ===")
    t0 = time.time()
    scorecards[nombre] = harness(eval_set, fn, nombre=nombre)
    print(f"    {time.time() - t0:.0f} s  (generaciones en caché: {len(_CACHE_GEN)})")
'''


CODIGO_REPRODUCIBILIDAD = r'''
# Chequeo de reproducibilidad contra S07: "sin RAG" y "A · ingenuo" son,
# respectivamente, los dos sistemas de resultados/m3_rag.json.
import json
ruta_s07 = Path("resultados/m3_rag.json")
if ruta_s07.exists():
    s07 = json.loads(ruta_s07.read_text(encoding="utf-8"))
    claves = ["aciertos", "aciertos_calculo", "aciertos_conocimiento",
              "aciertos_adversarial", "alucinaciones", "abstenciones"]
    for mio, suyo in [("sin RAG", "sin_rag"), ("A · ingenuo", "con_rag")]:
        dif = {k: (s07[suyo][k], scorecards[mio][k]) for k in claves
               if s07[suyo][k] != scorecards[mio][k]}
        print(f"{mio:12s} vs S07 {suyo:8s}: " + ("idéntico" if not dif else f"DIFIERE {dif}"))
    print("\nNota: el criterio de adv-02 se corrigió después de ejecutar S07 (doc. §6.1),")
    print("así que una diferencia de +1 en 'adversarial' es esperable y no es ruido.")
else:
    print("No está resultados/m3_rag.json (ejecuten S07 antes para este chequeo).")
'''


CODIGO_DELTAS = r'''
from math import comb


def mcnemar_exacto(sc_x, sc_y, bloque=None):
    """p-valor exacto (binomial, dos colas) sobre los casos discordantes."""
    pares = [(a["acierto"], b["acierto"]) for a, b in zip(sc_x["detalle"], sc_y["detalle"])
             if bloque is None or a["bloque"] == bloque]
    mejora = sum(1 for a, b in pares if not a and b)
    empeora = sum(1 for a, b in pares if a and not b)
    n = mejora + empeora
    if n == 0:
        return mejora, empeora, 1.0
    p = sum(comb(n, i) for i in range(0, min(mejora, empeora) + 1)) / 2 ** n
    return mejora, empeora, min(1.0, 2 * p)


print("DELTAS PASO A PASO — cada fila añade UNA sola cosa al sistema anterior")
print("=" * 96)
print(f"{'paso':28s}{'total':>8}{'cálculo':>10}{'conoc.':>9}{'advers.':>9}{'aluc.':>7}"
      f"{'  mejora/empeora':>17}{'p':>8}")
print("-" * 96)
for x, y in zip(nombres, nombres[1:]):
    a, b = scorecards[x], scorecards[y]
    m, e, p = mcnemar_exacto(a, b)
    d = lambda k: f"{b[k] - a[k]:+d}"
    print(f"{x + ' -> ' + y:28s}{d('aciertos'):>8}{d('aciertos_calculo'):>10}"
          f"{d('aciertos_conocimiento'):>9}{d('aciertos_adversarial'):>9}{d('alucinaciones'):>7}"
          f"{f'{m}/{e}':>17}{p:>8.3f}")
print("-" * 96)
for x, y in [("A · ingenuo", "E · +cita"), ("sin RAG", "E · +cita")]:
    a, b = scorecards[x], scorecards[y]
    m, e, p = mcnemar_exacto(a, b)
    print(f"{x + ' -> ' + y:28s}{b['aciertos'] - a['aciertos']:>+8d}"
          f"{b['aciertos_calculo'] - a['aciertos_calculo']:>+10d}"
          f"{b['aciertos_conocimiento'] - a['aciertos_conocimiento']:>+9d}"
          f"{b['aciertos_adversarial'] - a['aciertos_adversarial']:>+9d}"
          f"{b['alucinaciones'] - a['alucinaciones']:>+7d}{f'{m}/{e}':>17}{p:>8.3f}")
print("=" * 96)
'''


CODIGO_COSTE = r'''
# El coste de cada técnica: toda técnica cobra (S08).
#   ms antes de generar -> enrutador + retrieval, medido de nuevo SIN cachés
#                          (mediana sobre el eval set, en esta máquina)
#   tokens prompt       -> lo que paga la generación; el enrutador lo recorta en los problemas
def _pre_generacion(nombre, pregunta):
    if nombre == "sin RAG":
        return
    if nombre.startswith("A"):
        return buscar_densa(pregunta, K)
    if nombre.startswith("B"):
        return buscar_hibrida(pregunta, K)
    if nombre.startswith("C"):
        return buscar_con_rerank(pregunta, K)
    _CACHE_RERANK.pop(pregunta, None)              # D y E: el enrutador también cobra
    if clasificar(pregunta)[0] == "consulta":
        return buscar_con_rerank(pregunta, K)


print(f"{'sistema':14s}{'ms pre-gen.':>13}{'tokens prompt':>15}{'% vía RAG':>11}")
print("-" * 53)
coste = {}
for nombre in nombres:
    trazas = [TRAZA[(nombre, c["input"])] for c in eval_set]
    ms = []
    for c in eval_set:
        t0 = time.perf_counter()
        _pre_generacion(nombre, c["input"])
        ms.append((time.perf_counter() - t0) * 1000)
    coste[nombre] = {
        "ms_retrieval_mediana": float(np.median(ms)),
        "tokens_prompt_media": float(np.mean([t["tokens_prompt"] for t in trazas])),
        "frac_via_rag": float(np.mean([t["ruta"] == "consulta" and bool(t["top_k"]) for t in trazas])),
    }
    c = coste[nombre]
    print(f"{nombre:14s}{c['ms_retrieval_mediana']:>13.1f}{c['tokens_prompt_media']:>15.0f}"
          f"{c['frac_via_rag']:>11.0%}")
'''


CODIGO_GRAFICA = r'''
import matplotlib.pyplot as plt

bloques = [("calculo", "n_calculo", "aciertos_calculo"),
           ("conocimiento", "n_conocimiento", "aciertos_conocimiento"),
           ("adversarial", "n_adversarial", "aciertos_adversarial")]
x = np.arange(len(bloques)); ancho = 0.8 / len(nombres)
fig, ax = plt.subplots(figsize=(11, 4.8))
for j, nombre in enumerate(nombres):
    sc = scorecards[nombre]
    vals = [sc[a] / sc[n] for _, n, a in bloques]
    pos = x - 0.4 + ancho / 2 + j * ancho
    ax.bar(pos, vals, ancho, label=nombre)
    for p_, v in zip(pos, vals):
        ax.text(p_, v + 0.015, f"{v:.0%}", ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f"{b}\n(n={scorecards[nombres[0]][n]})" for b, n, _ in bloques])
ax.set_ylim(0, 1.15); ax.set_ylabel("Aciertos de dominio")
ax.set_title("RAG avanzado: aciertos por bloque en cada paso")
ax.legend(ncol=3, fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
'''


CODIGO_FALLIDAS_S07 = r'''
# Las consultas que S07 dejó para hoy, caso por caso: qué retrieval llevó al
# prompt y qué respondió cada sistema.
ruta_fallidas = Path("resultados/consultas_fallidas_s08.json")
fallidas_s07 = (json.loads(ruta_fallidas.read_text(encoding="utf-8")) if ruta_fallidas.exists()
                else [{"id": i} for i in ("con-02", "adv-02")])

por_id = {n: {d["id"]: d for d in scorecards[n]["detalle"]} for n in nombres}
for f in fallidas_s07:
    caso = next(c for c in eval_set if c["id"] == f["id"])
    print("=" * 90)
    print(f"[{caso['id']} · {caso['subtipo']}]  {caso['input']}")
    for nombre in nombres:
        d = por_id[nombre][caso["id"]]
        t = TRAZA[(nombre, caso["input"])]
        marca = "OK  " if d["acierto"] else ("abst" if d["abstuvo"] else "MAL ")
        print(f"  {nombre:13s} {marca} top-k={t['top_k']}")
        print(f"  {'':13s}      {d['respuesta'][:110]!r}")
'''


CODIGO_CAMBIOS = r'''
# Qué casos cambió cada paso. Es el diagnóstico que S07 no pudo hacer (§6.4).
for x, y in zip(nombres, nombres[1:]):
    cambios = [(a, b) for a, b in zip(scorecards[x]["detalle"], scorecards[y]["detalle"])
               if a["acierto"] != b["acierto"]]
    print(f"{x} -> {y}: {len(cambios)} cambio(s)")
    for a, b in cambios:
        flecha = "ARREGLÓ " if b["acierto"] else "ROMPIÓ  "
        print(f"   {flecha} {a['id']:8s} {a['subtipo']:26s} {b['motivo'][:44]}")
'''


CODIGO_ARTEFACTOS = r'''
import csv

Path("resultados").mkdir(exist_ok=True)
lista_sc = [scorecards[n] for n in nombres]

# 1 · Scorecard completo de los seis sistemas (mismo formato que M2 y S07)
guardar_scorecard("resultados/scorecard_rag_avanzado.csv", *lista_sc)

# 2 · deltas_s08.csv: el entregable que pide S08 (insumo de S09 y de la entrega M3)
with open("resultados/deltas_s08.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sistema", "anade", "aciertos", "calculo", "conocimiento", "adversarial",
                "alucinaciones", "abstenciones", "juez", "sim", "delta_aciertos",
                "mejora", "empeora", "p_mcnemar", "ms_retrieval", "tokens_prompt"])
    previo = None
    for nombre, _, desc in SISTEMAS:
        sc = scorecards[nombre]
        m, e, p = mcnemar_exacto(scorecards[previo], sc) if previo else ("", "", "")
        w.writerow([nombre, desc, f"{sc['aciertos']}/{sc['n']}",
                    f"{sc['aciertos_calculo']}/{sc['n_calculo']}",
                    f"{sc['aciertos_conocimiento']}/{sc['n_conocimiento']}",
                    f"{sc['aciertos_adversarial']}/{sc['n_adversarial']}",
                    sc["alucinaciones"], sc["abstenciones"], round(sc["juez_promedio"], 2),
                    round(sc["sim_embeddings"], 3),
                    "" if previo is None else sc["aciertos"] - scorecards[previo]["aciertos"],
                    m, e, "" if p == "" else round(p, 3),
                    round(coste[nombre]["ms_retrieval_mediana"], 1),
                    round(coste[nombre]["tokens_prompt_media"])])
        previo = nombre
print("Guardado: resultados/deltas_s08.csv")

# 3 · Detalle caso por caso de TODOS los sistemas, con la traza del retrieval
with open("resultados/detalle_rag_avanzado.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sistema", "id", "bloque", "subtipo", "ruta", "top_k", "acierto", "abstuvo",
                "alucino", "juez", "sim", "motivo", "respuesta"])
    for nombre in nombres:
        for d in scorecards[nombre]["detalle"]:
            t = TRAZA[(nombre, d["input"])]
            w.writerow([nombre, d["id"], d["bloque"], d["subtipo"], t["ruta"], "|".join(t["top_k"]),
                        d["acierto"], d["abstuvo"], d["alucino"], d["juez"], d["sim"],
                        d["motivo"], d["respuesta"]])
print("Guardado: resultados/detalle_rag_avanzado.csv")

# 4 · Retrieval aislado
df_ret.to_csv("resultados/retrieval_s08.csv", index=False, encoding="utf-8")
print("Guardado: resultados/retrieval_s08.csv")

# 5 · Todo junto, para el notebook de comparación y la documentación
Path("resultados/m3_rag_avanzado.json").write_text(json.dumps({
    "config": {"chunk_size": CHUNK_SIZE, "overlap": OVERLAP, "k": K, "k_recuperar": K_RECUPERAR,
               "n_documentos": len(corpus), "n_chunks": len(chunks),
               "modelo_embeddings": "paraphrase-multilingual-MiniLM-L12-v2",
               "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
               "fusion": "RRF (k=60)",
               "enrutador": "regresión logística [margen centroides, nº números, rerank máx.]",
               "generador": f"{MODELO_BASE} + LoRA (M1)"},
    "retrieval": json.loads(resumen_ret.reset_index().to_json(orient="records")),
    "enrutador": {"loo_dev": acc, "loo_dev_solo_centroides": acc_cent,
                  "n_dev": len(consultas_dev), "pesos": dict(zip(RASGOS, map(float, coef))),
                  "rutas_eval": rutas_eval},
    "sistemas": {n: {k: v for k, v in scorecards[n].items() if k != "detalle"} for n in nombres},
    "coste": coste,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Guardado: resultados/m3_rag_avanzado.json")
'''


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "RAG avanzado para el tutor de matemáticas",
        "Sesión 8 · Módulo 3 — Cada técnica se gana su lugar con un delta",
    ))

    # ------------------------------------------------------------------ 1
    c.append(md("""
## 1 · Introducción

### Lo que dejó S07

El RAG ingenuo resultó **de suma cero**:

| Bloque | Sin RAG | Con RAG (S07) | Δ |
|---|---|---|---|
| Cálculo | 10/12 | 6/12 | **−4** |
| Conocimiento | 0/4 | 3/4 | **+3** |
| Adversarial | 3/5 | 4/5 | **+1** |
| **Total** | **13/21** | **13/21** | **0** |

Y dejó dos diagnósticos que deciden qué técnicas vale la pena probar hoy:

1. **El cálculo no cayó por culpa del retrieval.** Cayó porque *todas* las
   preguntas pasaban por el RAG y tres chunks irrelevantes son ruido para un
   problema que solo hay que resolver. Mejorar la búsqueda no lo arregla: lo
   arregla **no buscar** cuando la pregunta no lo pide.
2. **El único fallo de conocimiento (`con-02`) fue de generación, no de
   retrieval.** Los chunks del plan de área llegaron al prompt y el modelo no
   extrajo "octavo".

Es decir: con 6 documentos y 17 chunks, **el retrieval no es el cuello de
botella de este sistema**. Aplicar hybrid search y reranking y esperar una
mejora en el scorecard sería aplicar la receta de clase sin leer el
diagnóstico.

### Qué se hace entonces

Se prueban las técnicas de S08 **y** las que el diagnóstico pide, añadiendo una
sola cosa por paso para que cada delta sea atribuible:

| Sistema | Añade | Ataca | Predicción (escrita antes de ejecutar) |
|---|---|---|---|
| **A · ingenuo** | — (el RAG de S07) | — | Reproduce S07 |
| **B · +hybrid** | BM25 + denso con RRF | Retrieval: términos exactos, siglas | ≈ A en el scorecard; mejor en consultas con siglas |
| **C · +rerank** | Cross-encoder sobre el top-10 | Retrieval: ruido en el top-k | ≈ B; mejor hit@1 |
| **D · +router** | Enrutador consulta/problema | **El ruido en cálculo** | Recupera el cálculo sin perder conocimiento: ~17/21 |
| **E · +cita** | Prompt de cita textual en la vía RAG | **El fallo de generación de `con-02`** | +1 en conocimiento |

La predicción de D sale de la sección 4.4 de la documentación: con un enrutador
perfecto, la vía sin RAG da 10/12 en cálculo y la vía RAG 3/4 en conocimiento.

### Por qué hybrid y reranking se miden igual aunque la predicción sea "≈"

Porque son las técnicas que exige la entrega M3 y porque *"no valió su coste"* es
una conclusión de ingeniería válida **solo si se midió**. Además se miden en dos
niveles: el **retrieval aislado** (hit@k sobre consultas con chunk de oro, sin
generar nada) y el **scorecard completo**. Si una técnica mejora el primero y no
el segundo, eso confirma que el cuello de botella está en otra parte.
"""))

    # ------------------------------------------------------------------ 2
    c.append(md("""
## 2 · Objetivos

1. Implementar **hybrid search** (BM25 + denso, fusión RRF) y **reranking** con
   cross-encoder sobre el mismo corpus y chunking de S07.
2. Medir el **retrieval aislado** (hit@1, hit@3, MRR, latencia) con consultas de
   desarrollo que no tocan el eval set.
3. Construir un **enrutador** consulta/problema calibrado solo con esas
   consultas de desarrollo, y validarlo con leave-one-out.
4. Correr **el harness de M2 sin cambios** sobre seis sistemas y reportar la
   tabla de deltas paso a paso con McNemar y coste.
5. Cerrar el diagnóstico de S07: **qué casos arregló y cuáles rompió** cada
   técnica, incluidas las consultas fallidas que S07 dejó para hoy.
"""))

    # ------------------------------------------------------------------ 0
    c.append(md("""
## 0 · Preparación del entorno

> **Activen la GPU (T4).** Se cargan el generador con LoRA, el juez, el modelo
> de embeddings y el cross-encoder. La corrida completa del harness tarda
> aproximadamente lo que tres sistemas, no seis, gracias a la caché de la
> sección 11.
"""))

    c.append(code('''
%pip install -q "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" \\
    sentence-transformers chromadb rank_bm25 "scikit-learn>=1.3"
%pip uninstall -y -q torchao
print("Librerías instaladas. Si Colab pide reiniciar la sesión, reinícienla y sigan desde aquí.")
'''))

    c.append(code('''
import os, random, re
import numpy as np
import torch
import transformers

SEMILLA = 42
random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
transformers.set_seed(SEMILLA)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"transformers {transformers.__version__} | torch {torch.__version__} | {DEVICE}")
'''))

    c.extend(celda_drive())

    # ------------------------------------------------------------------ 3
    c.extend(celda_eval_set(prefijo="3"))

    # ------------------------------------------------------------------ 4
    c.append(md("""
---
## 4 · Métricas, embeddings y criterio de acierto

Las mismas funciones de M2 y S07, sin tocar. El criterio de acierto se carga ya
aquí (y no junto al harness) porque la normalización `_norm` la usan también el
tokenizador de BM25 y la evaluación del retrieval.
"""))
    c.append(code(CODIGO_METRICAS))
    c.extend(celda_similitud())
    c.extend(celda_evaluacion())

    # ------------------------------------------------------------------ 5
    c.append(md("""
---
## 5 · Corpus, chunking e índice denso

**Idénticos a S07**, byte a byte: el código de estas celdas se genera desde el
mismo sitio que el de S07 (`scripts/nb_comun_rag.py`). Si el corpus o los chunks
cambiaran, el delta entre A y los demás mediría ese cambio y no la técnica.
"""))
    c.append(code(CODIGO_CORPUS))
    c.append(code(CODIGO_CHUNKING))
    c.append(code(CODIGO_INDICE))

    # ------------------------------------------------------------------ 6
    c.append(md("""
---
## 6 · Consultas de desarrollo

El eval set tiene **4** preguntas de conocimiento. Con eso no se puede ni medir
un retrieval ni calibrar un enrutador: cada consulta valdría 25 puntos, y
ajustar algo con los mismos casos con los que luego se evalúa es contaminación
(S05).

Por eso se escribe un conjunto **aparte**, con dos usos y ninguno de ellos es
evaluar el sistema final:

| Uso | Qué consultas | Para qué |
|---|---|---|
| **Retrieval aislado** | Las consultas con `clave` | hit@k: ¿el chunk que contiene la clave entra al top-k? |
| **Calibrar el enrutador** | Todas | Centroides de "consulta" y "problema" |

Las consultas cubren tres **estilos** a propósito, porque cada técnica de S08
gana en uno distinto: **literal** (todas deberían acertar), **sigla / término
exacto** (terreno de BM25) y **coloquial** (terreno del denso). Hay además tres
consultas institucionales **sin respuesta en el corpus** y un problema
**frontera** ("valen 30% y 70%, ¿cuál es mi nota?") que usa vocabulario del
reglamento pero hay que calcularlo.
"""))
    c.append(code(CODIGO_CONSULTAS_DEV))
    c.append(code(CODIGO_CONTAMINACION))

    # ------------------------------------------------------------------ 7
    c.append(md("""
---
## 7 · Parte A · Hybrid search: BM25 + denso + RRF

**BM25** puntúa por palabras exactas y premia las raras. Gana donde el denso se
difumina: siglas (`SIEE`), números, términos técnicos. Sus fallos no están
correlacionados con los del denso, y por eso la fusión paga cuando paga.

**Una adaptación al español.** La plantilla tokeniza con `lower().split()`. Aquí
se quitan tildes y puntuación con la misma `_norm` del harness: sin eso,
*"¿cuánto"* y *"cuanto"* son palabras distintas para BM25.
"""))
    c.append(code(CODIGO_BM25))
    c.append(md("""
### Reciprocal Rank Fusion

Los puntajes de BM25 y del coseno están en escalas incomparables, así que no se
promedian: se fusionan por **puesto**. Cada chunk suma `1/(60 + puesto)` en cada
lista.
"""))
    c.append(code(CODIGO_RRF))

    # ------------------------------------------------------------------ 8
    c.append(md("""
---
## 8 · Parte B · Reranking con cross-encoder

El bi-encoder embebe pregunta y chunk **por separado**; el cross-encoder los lee
**juntos**. Más preciso y más caro por par, así que el patrón es *recuperar
ancho con lo barato (top-10 híbrido), reordenar estrecho con lo bueno (top-3)*.

> Con 17 chunks, "top-10" es más de la mitad del corpus. Es a propósito: en este
> tamaño el reranker está viendo casi todo, que es su mejor caso posible. Si ni
> así mejora el scorecard, el problema no era el orden del top-k.
"""))
    c.append(code(CODIGO_RERANKER))

    # ------------------------------------------------------------------ 9
    c.append(md("""
---
## 9 · El retrieval, medido aislado

Antes de generar una sola palabra: ¿el chunk que contiene la respuesta entra al
top-3? Se mide sobre las consultas de desarrollo con clave **más** las 4 de
conocimiento del eval set (medir retrieval no las contamina: no se ajusta nada
con ellas).

| Métrica | Qué dice |
|---|---|
| **hit@1** | El chunk de oro quedó primero |
| **hit@3** | El chunk de oro llegó al prompt (k = 3) |
| **MRR@3** | 1/puesto del chunk de oro, 0 si no entró |
| **ms** | Mediana de latencia del retrieval, en esta máquina |
"""))
    c.append(code(CODIGO_EVAL_RETRIEVAL))
    c.append(code(CODIGO_FALLOS_RETRIEVAL))
    c.append(md("""
> **Cómo leerlo.** Si A ya tiene hit@3 cercano a 1, **el retrieval no tiene
> margen para mejorar el scorecard**: el chunk ya llegaba. En ese caso B y C solo
> pueden aportar en hit@1 (orden) y en los estilos donde A fallaba. Y cualquier
> mejora aquí que no se traslade al harness confirma el diagnóstico de S07: los
> fallos eran de otra naturaleza.
"""))

    # ------------------------------------------------------------------ 10
    c.append(md("""
---
## 10 · Parte C · El enrutador

La recomendación de arquitectura de S07 era **enrutar**: activar el RAG solo
cuando la pregunta pide un dato. S07 sugería reentrenar el clasificador BETO de
M1 con una clase nueva; aquí se prueba primero lo más barato que podría
funcionar, **un centroide de embeddings por clase**, con el modelo que ya está
cargado.

Si esto alcanza, BETO no hace falta. Si no alcanza, el LOO lo dirá antes de
tocar el eval set.
"""))
    c.append(code(CODIGO_ENRUTADOR))
    c.append(code(CODIGO_RUTAS_EVAL))
    c.append(md("""
> **Qué mirar.** Los márgenes cercanos a cero son las decisiones frágiles. Un
> problema enviado a la vía RAG no es catastrófico (es lo que hacía S07 con
> todos); una consulta enviada a la vía sin RAG sí lo es, porque ahí el sistema
> no tiene de dónde sacar el dato y **alucinará** (M2: 4 de 4).
"""))

    # ------------------------------------------------------------------ 11
    c.append(md("""
---
## 11 · Los seis sistemas

Mismo generador (Qwen2.5-1.5B + LoRA de M1) y, en A–D, **los mismos prompts de
S07**, importados del mismo sitio. Entre A, B y C solo cambia la función de
búsqueda; D añade el enrutador; E cambia el prompt de la vía RAG.
"""))
    c.append(code(CODIGO_GENERADOR))
    c.append(code(CODIGO_PROMPTS))
    c.append(code(CODIGO_SISTEMAS))
    c.append(code(CODIGO_PRUEBA_CON02))

    # ------------------------------------------------------------------ 12
    c.append(md("""
---
## 12 · El harness de M2, sin tocar

Juez y harness idénticos a M2 y S07. La única adición es una caché del juez
(greedy: misma entrada, misma nota), que no altera ningún resultado.
"""))
    c.extend(celda_juez())
    c.append(code(CODIGO_JUEZ_CACHE))
    c.extend(celda_harness())
    c.append(code(CODIGO_CORRIDA))
    c.append(code(CODIGO_REPRODUCIBILIDAD))

    # ------------------------------------------------------------------ 13
    c.append(md("""
---
## 13 · Resultados

### 13.1 · Scorecard de los seis sistemas
"""))
    c.append(code("imprimir_scorecard(*[scorecards[n] for n in nombres])"))
    c.append(md("""
### 13.2 · La tabla de deltas

Cada fila compara un sistema con el anterior, que difiere en **una sola cosa**.
`mejora/empeora` son los casos discordantes y `p` el McNemar exacto sobre ellos.

> Con 21 casos **ningún delta de 1–2 aciertos es concluyente**. Se reporta la
> dirección y, sobre todo, **qué casos** cambiaron (13.5).
"""))
    c.append(code(CODIGO_DELTAS))
    c.append(md("""
### 13.3 · Coste de cada técnica
"""))
    c.append(code(CODIGO_COSTE))
    c.append(md("""
### 13.4 · Por bloque y por caso
"""))
    c.append(code(CODIGO_GRAFICA))
    c.append(code("tabla_por_subtipo(*[scorecards[n] for n in nombres])"))
    c.append(md("""
### 13.5 · Qué arregló y qué rompió cada paso
"""))
    c.append(code(CODIGO_CAMBIOS))
    c.append(md("""
### 13.6 · Las consultas fallidas de S07
"""))
    c.append(code(CODIGO_FALLIDAS_S07))

    # ------------------------------------------------------------------ 14
    c.append(md("""
---
## 14 · Artefactos de la entrega

| Archivo | Contenido |
|---|---|
| `resultados/deltas_s08.csv` | La tabla de deltas paso a paso con McNemar y coste (entregable de S08) |
| `resultados/scorecard_rag_avanzado.csv` | Scorecard de los seis sistemas |
| `resultados/detalle_rag_avanzado.csv` | Caso por caso, **todos** los sistemas, con ruta y top-k |
| `resultados/retrieval_s08.csv` | Retrieval aislado por consulta y retriever |
| `resultados/m3_rag_avanzado.json` | Configuración, retrieval, enrutador, scorecards y coste |
"""))
    c.append(code(CODIGO_ARTEFACTOS))

    # ------------------------------------------------------------------ 15
    c.append(md("""
---
## 15 · Discusión

Las preguntas a responder con las tablas de arriba, en este orden:

**1. ¿El retrieval tenía margen?** (sección 9)
Si A ya tenía hit@3 alto, B y C no podían mover el scorecard y su delta ≈ 0 es
la confirmación esperada, no un fracaso. Reporten hit@1 y latencia: el reranker
puede ordenar mejor y aun así *no valer su coste* en este corpus.

**2. ¿El enrutador recuperó el cálculo?** (13.2, fila C → D)
Es la hipótesis central del notebook. Si el bloque de cálculo vuelve a ~10/12 y
el de conocimiento se mantiene, la arquitectura compuesta que S07 proponía queda
**demostrada** y no solo argumentada. Miren también los tokens de prompt (13.3):
el enrutador además abarata.

**3. ¿El prompt de cita arregló `con-02`?** (13.6, fila D → E)
Si sí, el fallo era de generación y lo arregló la técnica que ataca generación:
el diagnóstico de S07 queda validado. Si no, el límite es el tamaño del modelo
(1.5B), y eso va a la sección de limitaciones.

**4. ¿Qué pasó con los adversariales?**
Ninguno es una consulta institucional, así que el enrutador los manda por la vía
sin RAG y pierden la válvula de escape del prompt de S07. Si el bloque
adversarial baja de C a D, es el coste del enrutador y hay que reportarlo.

**5. Tamaño de muestra.**
21 casos. Las conclusiones se apoyan en la dirección, en los casos concretos que
cambian y en el retrieval aislado, no en los p-valores.

---

### Lo que se lleva la entrega M3

1. **≥2 técnicas avanzadas justificadas con su delta**: hybrid search, reranking
   y enrutamiento, cada una medida con el mismo harness.
2. **`deltas_s08.csv`** con coste y significancia.
3. **El diagnóstico cerrado de S07**: qué casos arregló y rompió cada técnica.
"""))

    c.append(md("""
---
## 16 · Conclusiones de esta corrida

> Cifras corregidas con `scripts/recalcular_criterio.py`: la corrida usó la versión
> antigua del eval set (arriba se ve `SHA coincide: False`) y el caso `adv-02` se
> re-evaluó después sobre las respuestas ya generadas. Cambiaron 4 celdas, todas del
> bloque adversarial. Detalle en `resultados/correccion_adv02.md`.

| Sistema | Total | Cálculo | Conoc. | Advers. | Alucin. | Δ | McNemar |
|---|---|---|---|---|---|---|---|
| sin RAG | 14/21 | 10/12 | 0/4 | 4/5 | 7 | — | — |
| A · ingenuo | 14/21 | 6/12 | 3/4 | 5/5 | 7 | 0 | 5/5, p = 1.000 |
| B · +hybrid | 14/21 | 7/12 | **4/4** | 3/5 | 6 | 0 | 3/3, p = 1.000 |
| C · +rerank | 13/21 | 7/12 | 3/4 | 3/5 | 5 | −1 | 2/3, p = 1.000 |
| **D · +router** | **18/21** | **10/12** | 3/4 | 5/5 | **2** | **+5** | 7/2, p = 0.180 |
| E · +cita | 18/21 | 10/12 | 3/4 | 5/5 | 3 | 0 | 0/0, p = 1.000 |

**1 · El retrieval mejoró y no movió el scorecard.** hit@1 pasó de 0.722 a 0.889 y
hit@3 llegó a 1.000 con el reranker, pero A → B → C se mueve entre 13 y 14 de 21. Es
la predicción de la sección 1: con hit@3 ya en 0.944, no había margen. La mejora de
B es real y está **fuera** del scorecard: en consultas con siglas el hit@3 sube de
0.5 a 1.0, y ese estilo no aparece en el eval set.

**2 · El enrutador es lo único que paga.** 14 → 18 de 21, recuperando el cálculo de
6/12 a 10/12 sin perder conocimiento, y bajando las alucinaciones de 7 a 2. El
clasificador de tres señales dio 97.0% en leave-one-out (frente a 78.8% de solo
centroides) y 16/16 sobre el eval set, que no vio al ajustarse. La arquitectura
compuesta que proponía la documentación de M2/M3 queda demostrada, y con más margen
del previsto: se estimaban 17/21.

**3 · El prompt de cita textual no arregló `con-02`.** Delta exactamente 0: ningún
caso cambió. El fallo de generación no se corrige pidiendo cita literal; es un límite
del modelo de 1.5B. Además baja el formato válido del 95% al 76%, porque el prompt de
cita compite con el formato "Paso N".

**4 · El coste.** El reranker triplica la latencia de recuperación (13 → 43 ms) sin
contrapartida en el scorecard. El enrutador abarata: el prompt medio baja de 541 a
193 tokens, porque los problemas de cálculo dejan de llevar contexto.

**5 · El riesgo declarado no se materializó.** Se advirtió que los adversariales
perderían la válvula de escape al ir por la vía sin RAG; con el criterio corregido, D
y E obtienen 5/5 en ese bloque, el mejor de los seis sistemas.
"""))

    return c
