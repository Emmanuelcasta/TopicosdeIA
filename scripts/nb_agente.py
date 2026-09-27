"""Notebook S10 (3/4) — Arquitectura agéntica: C0–C5 por el harness de M2 y el eval set de M3."""

from __future__ import annotations

from nb_comun import CODIGO_METRICAS, celda_drive, code, encabezado, md
from nb_comun_agente import CODIGO_LLM, celda_eval_m3, celda_trayectorias, celdas_modulos
from nb_comun_eval import celda_eval_set, celda_evaluacion, celda_harness, celda_juez, celda_similitud
from nb_comun_rag import CODIGO_CHUNKING, CODIGO_CORPUS, CODIGO_PROMPTS
from nb_rag_avanzado import CODIGO_BM25, CODIGO_INDICE, CODIGO_JUEZ_CACHE, CODIGO_RERANKER, CODIGO_RRF


CODIGO_FEW_SHOT = r'''
# Few-shot: conversaciones COMPLETAS del split de entrenamiento, en el formato nativo.
# Se eligen por criterio (no a mano por id) para que la elección sea reproducible y
# no dependa de haber mirado el eval set. Cada configuración solo ve ejemplos con
# herramientas que ella tiene disponibles.
def elegir(criterio, descripcion):
    for t in trayectorias:
        if t["split"] == "train" and criterio(t):
            return t
    raise ValueError(f"no hay trayectoria de entrenamiento para: {descripcion}")


sin_docs = lambda t: "buscar_documentos" not in t["herramientas"]   # noqa: E731

EJ_UNA_LLAMADA   = elegir(lambda t: t["tipo"] == "calculo" and len(t["herramientas"]) == 1
                          and t["herramientas"][0] == "operar_fracciones", "cálculo de una llamada")
EJ_DOS_LLAMADAS  = elegir(lambda t: t["tipo"] == "calculo" and len(t["herramientas"]) == 2 and sin_docs(t)
                          and len(set(t["herramientas"])) == 2, "cálculo de dos pasos")
EJ_ERROR         = elegir(lambda t: t["tipo"] == "error_dominio", "error de dominio")
EJ_SIN_HERR      = elegir(lambda t: t["tipo"] == "sin_herramienta" and "alcance" in t["mensajes"][-1]["content"],
                          "fuera de dominio")
EJ_COMPUESTO     = elegir(lambda t: t["tipo"] == "compuesto", "compuesto documento + cálculo")

FS_C3 = [EJ_UNA_LLAMADA["mensajes"], EJ_ERROR["mensajes"], EJ_SIN_HERR["mensajes"]]
FS_C4 = [EJ_DOS_LLAMADAS["mensajes"], EJ_ERROR["mensajes"], EJ_SIN_HERR["mensajes"]]
FS_C5 = [EJ_COMPUESTO["mensajes"], EJ_DOS_LLAMADAS["mensajes"], EJ_SIN_HERR["mensajes"]]
FEW_SHOT_IDS = {"C3": [EJ_UNA_LLAMADA["id"], EJ_ERROR["id"], EJ_SIN_HERR["id"]],
                "C4": [EJ_DOS_LLAMADAS["id"], EJ_ERROR["id"], EJ_SIN_HERR["id"]],
                "C5": [EJ_COMPUESTO["id"], EJ_DOS_LLAMADAS["id"], EJ_SIN_HERR["id"]]}
for k, v in FEW_SHOT_IDS.items():
    print(k, v)
'''


CODIGO_REGISTROS = r'''
# Registros de herramientas por configuración.
REGISTRO_MATE = REGISTRO                                            # C3, C4: solo cálculo
REGISTRO_C5 = registro_con_documentos(REGISTRO, buscar_con_rerank,  # C5: + retrieval como herramienta
                                      chunks, metadatos, ids_chunks, k=K)
COMODINES = {n: REGISTRO_C5.obtener(n).comodin_de for n in REGISTRO_C5.nombres()}


def contexto_rag(pregunta):
    """El contexto de S08-C, con el MISMO formato que su prompt ('[Fuente: ...]')."""
    return [f"[Fuente: {metadatos[i]['fuente']}]\n{chunks[i]}" for i in buscar_con_rerank(pregunta, K)]


print("C3/C4:", len(REGISTRO_MATE), "herramientas | C5:", len(REGISTRO_C5), "(+ buscar_documentos)")
print("Prueba de buscar_documentos:")
print(REGISTRO_C5.ejecutar(LlamadaHerramienta("buscar_documentos", {"consulta": "nota mínima aprobatoria"})).resultado[:300])
'''


CODIGO_SONDEO = r'''
# SONDEO: ¿qué adaptador usan las configuraciones con herramientas?
#
# Riesgo identificado en la propuesta: la LoRA de M1 enseñó a responder SIEMPRE
# "Paso 1: ...", lo que puede impedir que el modelo emita <tool_call>. En vez de
# suponerlo, se mide sobre las trayectorias de VALIDACIÓN (nunca el eval set): se
# da la pregunta con el prompt y el few-shot de C5 y se mira la primera decisión.
import pandas as pd


def primera_decision(adaptador, t):
    llm = crear_llm(adaptador, max_new_tokens=160)
    mensajes = [{"role": "system", "content": SYSTEM_C5_AGENTE}] + [m for fs in FS_C5 for m in fs] + [t["mensajes"][0]]
    salida = analizar_salida(llm(mensajes, REGISTRO_C5.esquemas()))
    debia = "tool_calls" in t["mensajes"][1]
    fila = {"id": t["id"], "debia_usar": debia, "uso": bool(salida.llamadas), "decision_ok": debia == bool(salida.llamadas)}
    if salida.llamadas:
        l = salida.llamadas[0]
        fila["formato_nativo"] = l.formato == "nativo"
        fila["llamada_valida"] = (l.nombre in REGISTRO_C5 and
                                  REGISTRO_C5.ejecutar(l).tipo_error not in ("desconocida", "argumentos"))
    return fila


val = [t for t in trayectorias if t["split"] == "validation"]
filas_sondeo = []
for adaptador in [None, "m1"]:
    for t in val:
        filas_sondeo.append({"adaptador": str(adaptador), **primera_decision(adaptador, t)})
df_sondeo = pd.DataFrame(filas_sondeo)
for col in ("formato_nativo", "llamada_valida"):
    df_sondeo[col] = df_sondeo.get(col, pd.Series(dtype=float)).astype(float)
resumen_sondeo = df_sondeo.groupby("adaptador").agg(
    decision_ok=("decision_ok", "mean"),
    uso_cuando_debia=("uso", lambda s: s[df_sondeo.loc[s.index, "debia_usar"]].mean()),
    llamada_valida=("llamada_valida", "mean"),
    formato_nativo=("formato_nativo", "mean"))
print(resumen_sondeo.round(3).to_string())

puntaje = (resumen_sondeo["decision_ok"] + resumen_sondeo["llamada_valida"].fillna(0))
ADAPTADOR_HERRAMIENTAS = None if puntaje.idxmax() == "None" else puntaje.idxmax()
print(f"\nAdaptador elegido para C3–C5: {ADAPTADOR_HERRAMIENTAS!r}  (decisión tomada sobre validación, no sobre el eval set)")
'''


CODIGO_CONFIGS = r'''
from dataclasses import dataclass
from typing import Callable


@dataclass
class Configuracion:
    nombre: str
    anade: str
    fn: Callable


def _respuesta_directa(system, adaptador, max_new_tokens=220, muestreo=None):
    """C0/C1: una sola generación, sin núcleo agéntico, envuelta en RespuestaAgente
    para que la traza y las métricas tengan la misma forma en todas las configuraciones."""
    def fn(pregunta):
        texto = generar_chat([{"role": "system", "content": system}, {"role": "user", "content": pregunta}],
                             None, adaptador=adaptador, max_new_tokens=max_new_tokens, muestreo=muestreo)
        return RespuestaAgente(pregunta, texto, pregunta,
                               traza=[EventoTraza(1, "respuesta", {"texto": texto})], pasos_llm=1)
    return fn


def reescribir_con_llm(texto):
    return generar_chat([{"role": "system", "content": SYSTEM_NORMALIZADOR}, {"role": "user", "content": texto}],
                        None, adaptador=None, max_new_tokens=80)


def construir_configs(muestreo=None):
    A = ADAPTADOR_HERRAMIENTAS
    configs = [
        Configuracion("C0 · base", "Qwen2.5-1.5B sin adaptador (baseline de M2)",
                      _respuesta_directa(INSTRUCCION, None, muestreo=muestreo)),
        Configuracion("C1 · +LoRA", "+ LoRA de M1 (el 'fine-tuned' de M2)",
                      _respuesta_directa(INSTRUCCION, "m1", muestreo=muestreo)),
        Configuracion("C2 · +RAG", "+ RAG de S08-C (híbrida + reranker), contexto siempre",
                      AgenteTutor(crear_llm("m1", 220, muestreo), SYSTEM_RAG, contexto_fijo=contexto_rag,
                                  nombre="C2")),
        Configuracion("C3 · +herram.", "+ herramientas de cálculo, UNA ronda de tool calling",
                      AgenteTutor(crear_llm(A, 256, muestreo), SYSTEM_C3_HERRAMIENTAS, registro=REGISTRO_MATE,
                                  contexto_fijo=contexto_rag, max_rondas=1, max_pasos=4, few_shot=FS_C3, nombre="C3")),
        Configuracion("C4 · +ReAct", "+ bucle ReAct: pensamiento → acción → observación, varios pasos",
                      AgenteTutor(crear_llm(A, 256, muestreo), SYSTEM_C4_REACT, registro=REGISTRO_MATE,
                                  contexto_fijo=contexto_rag, max_rondas=6, max_pasos=8, few_shot=FS_C4, nombre="C4")),
        Configuracion("C5 · agente", "+ retrieval como herramienta, planificación, verificador, normalizador",
                      AgenteTutor(crear_llm(A, 256, muestreo), SYSTEM_C5_AGENTE, registro=REGISTRO_C5,
                                  max_rondas=6, max_pasos=8, verificar=True, few_shot=FS_C5,
                                  normalizador=Normalizador(reescribir=reescribir_con_llm),
                                  extraer_contextos=chunks_de_resultado, nombre="C5")),
    ]
    if "agente" in ADAPTADORES:
        configs.append(Configuracion("C5-FT · entrenado", "C5 con el adaptador entrenado con trayectorias",
                       AgenteTutor(crear_llm("agente", 256, muestreo), SYSTEM_C5_AGENTE, registro=REGISTRO_C5,
                                   max_rondas=6, max_pasos=8, verificar=True, few_shot=FS_C5,
                                   normalizador=Normalizador(reescribir=reescribir_con_llm),
                                   extraer_contextos=chunks_de_resultado, nombre="C5-FT")))
    else:
        print("AVISO: no está adaptadores/qwen-lora-agente -> C5-FT no se evalúa (ejecuten el notebook 2).")
    return configs


CONFIGS = construir_configs()
for cfg in CONFIGS:
    print(f"  {cfg.nombre:20s} {cfg.anade}")
'''


CODIGO_PRUEBA_AGENTE = r'''
# Antes del harness completo: una pregunta compuesta, con la traza a la vista.
cfg_c5 = next(c for c in CONFIGS if c.nombre.startswith("C5 "))
caso = next(c for c in eval_m3 if c["id"] == "com-01")
CONTADOR.reset()
out = cfg_c5.fn(caso["input"])
print("PREGUNTA:", caso["input"], "\n")
for e in out.traza:
    datos = json.dumps(e.datos, ensure_ascii=False, default=str)
    print(f"  [{e.paso}] {e.tipo:13s} {datos[:130]}")
print("\nRESPUESTA:\n" + out.respuesta)
print(f"\nPasos LLM: {out.pasos_llm} | tokens de entrada: {CONTADOR.entrada} | salida: {CONTADOR.salida}")
print("Evaluación:", evaluar_caso(caso, out.respuesta)["motivo"])
'''


CODIGO_PERSISTENCIA = r'''
# ---------------------------------------------------------------------------
# Persistencia incremental.
#
# 7 configuraciones × 53 casos, con varios pasos por caso en C3–C5, es más de una
# hora de T4. Cada respuesta (con su traza completa) se añade a un JSONL en cuanto
# se genera. Si Colab se desconecta, al reejecutar se retoma donde quedó: lo que ya
# está en el archivo no se vuelve a generar.
#
# Este archivo es también la ENTRADA del notebook 4 (RAGAS): contiene la pregunta,
# la respuesta, los contextos recuperados y las observaciones de las herramientas.
# ---------------------------------------------------------------------------
RUTA_RESPUESTAS = Path("resultados/respuestas_s10.jsonl")
RUTA_RESPUESTAS.parent.mkdir(exist_ok=True)

REGISTROS = {}
if RUTA_RESPUESTAS.exists():
    for linea in RUTA_RESPUESTAS.read_text(encoding="utf-8").splitlines():
        fila = json.loads(linea)
        REGISTROS[(fila["config"], fila["id"])] = fila
print(f"Respuestas ya generadas en {RUTA_RESPUESTAS}: {len(REGISTROS)}")

ID_DE = {c["input"]: c["id"] for c in eval_set + eval_m3}
assert len(ID_DE) == len(eval_set) + len(eval_m3), "hay preguntas repetidas entre eval sets"


def sistema_registrado(cfg):
    def sistema(pregunta):
        clave = (cfg.nombre, ID_DE[pregunta])
        if clave in REGISTROS:
            return REGISTROS[clave]["respuesta"]
        CONTADOR.reset()
        t0 = time.perf_counter()
        out = cfg.fn(pregunta)
        fila = out.a_dict()
        fila.update(config=cfg.nombre, id=ID_DE[pregunta], segundos=time.perf_counter() - t0,
                    tokens_entrada=CONTADOR.entrada, tokens_salida=CONTADOR.salida, llamadas_llm=CONTADOR.llamadas)
        with RUTA_RESPUESTAS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fila, ensure_ascii=False, default=str) + "\n")
        REGISTROS[clave] = fila
        return out.respuesta
    return sistema
'''


CODIGO_CORRIDA = r'''
SC_M2, SC_M3 = {}, {}
for cfg in CONFIGS:
    t0 = time.time()
    SC_M2[cfg.nombre] = harness(eval_set, sistema_registrado(cfg), nombre=cfg.nombre, verbose=False)
    SC_M3[cfg.nombre] = harness(eval_m3, sistema_registrado(cfg), nombre=cfg.nombre, verbose=False)
    print(f"{cfg.nombre:20s} M2 {SC_M2[cfg.nombre]['aciertos']:2d}/21 | "
          f"M3 {SC_M3[cfg.nombre]['aciertos']:2d}/32 | {time.time() - t0:5.0f} s")
NOMBRES = [cfg.nombre for cfg in CONFIGS]
'''


CODIGO_REPRODUCIBILIDAD = r'''
# C0, C1 y C2 son sistemas que ya se midieron: deben reproducir M2 y S08.
claves = ["aciertos", "aciertos_calculo", "aciertos_conocimiento", "aciertos_adversarial"]
referencias = [("C0 · base", "resultados/m2_harness.json", "baseline"),
               ("C1 · +LoRA", "resultados/m2_harness.json", "fine-tuned"),
               ("C2 · +RAG", "resultados/m3_rag_avanzado.json", ("sistemas", "C · +rerank"))]
for mio, ruta, clave in referencias:
    if not Path(ruta).exists():
        print(f"{mio:12s} (no está {ruta}: no se puede comprobar)")
        continue
    ref = json.loads(Path(ruta).read_text(encoding="utf-8"))
    ref = ref[clave[0]][clave[1]] if isinstance(clave, tuple) else ref[clave]
    dif = {k: (ref[k], SC_M2[mio][k]) for k in claves if ref[k] != SC_M2[mio][k]}
    print(f"{mio:12s} vs {ruta}: " + ("idéntico" if not dif else f"DIFIERE {dif}"))
print("\nNota: el criterio de adv-02 se corrigió después de ejecutar M2 (documentación §6.1):")
print("una diferencia de +1 en 'adversarial' frente a m2_harness.json es esperable.")
'''


CODIGO_ENRIQUECER = r'''
ORIGINAL = {c["id"]: c.get("original") for c in eval_m3}


def detalle_enriquecido(nombre):
    filas = []
    for conjunto, sc in (("M2", SC_M2[nombre]), ("M3", SC_M3[nombre])):
        for d in sc["detalle"]:
            reg = REGISTROS[(nombre, d["id"])]
            m = metricas_herramientas_caso(HERRAMIENTAS_ESPERADAS[d["id"]], reg["observaciones"], reg["respuesta"],
                                           REGISTRO_C5.familia_de, COMODINES)
            traza = reg["traza"]
            norm = eventos(traza, "normalizacion")
            filas.append({**d, **m, "config": nombre, "conjunto": conjunto, "original": ORIGINAL.get(d["id"]),
                          "pasos_llm": reg["pasos_llm"], "terminacion": reg["terminacion"],
                          "errores_formato": len(eventos(traza, "error_formato")),
                          "verificaciones": len(eventos(traza, "verificacion")),
                          "reescrita_llm": bool(norm and norm[0]["datos"].get("reescrita_llm")),
                          "segundos": reg["segundos"], "tokens_entrada": reg["tokens_entrada"],
                          "tokens_salida": reg["tokens_salida"],
                          "herramientas_usadas": "|".join(o["herramienta"] for o in reg["observaciones"])})
    return filas


DETALLE = {n: detalle_enriquecido(n) for n in NOMBRES}
df_detalle = pd.DataFrame([f for n in NOMBRES for f in DETALLE[n]])
print(df_detalle.shape)
'''


CODIGO_SCORECARD = r'''
def por_bloque(nombre, bloque):
    filas = [f for f in DETALLE[nombre] if f["bloque"] == bloque]
    return sum(f["acierto"] for f in filas), len(filas)


def media(filas, clave):
    v = [f[clave] for f in filas if f.get(clave) is not None]
    return sum(v) / len(v) if v else None


RESUMEN = {}
for n in NOMBRES:
    filas = DETALLE[n]
    m3 = [f for f in filas if f["conjunto"] == "M3"]
    ref = [f for f in filas]
    RESUMEN[n] = {
        "m2": SC_M2[n], "m3_aluc": sum(f["alucino"] for f in m3),
        "herr": resumen_agente(filas),
        "rob": robustez_mal_escrito([f for f in m3 if f["bloque"] == "mal_escrito"], ref),
        "tokens_entrada": media(filas, "tokens_entrada"), "tokens_salida": media(filas, "tokens_salida"),
    }


def fmt(v, tipo):
    if v is None:
        return "—"
    return {"pct": f"{v:.0%}", "f2": f"{v:.2f}", "f1": f"{v:.1f}", "int": f"{v:.0f}", "txt": str(v)}[tipo]


FILAS_SCORECARD = [
    ("HARNESS M2 (21 casos, la vara fija)", None, None),
    ("  Exactitud en cálculo", lambda n: RESUMEN[n]["m2"]["exactitud_calculo"], "pct"),
    ("  Aciertos totales /21", lambda n: RESUMEN[n]["m2"]["aciertos"], "int"),
    ("  Conocimiento /4", lambda n: RESUMEN[n]["m2"]["aciertos_conocimiento"], "int"),
    ("  Adversarial /5", lambda n: RESUMEN[n]["m2"]["aciertos_adversarial"], "int"),
    ("  Alucinaciones", lambda n: RESUMEN[n]["m2"]["alucinaciones"], "int"),
    ("  Abstenciones", lambda n: RESUMEN[n]["m2"]["abstenciones"], "int"),
    ("  Juez (1-5)  [calidad explicación]", lambda n: RESUMEN[n]["m2"]["juez_promedio"], "f2"),
    ("  Formato válido", lambda n: RESUMEN[n]["m2"]["formato_valido"], "pct"),
    ("EVAL SET M3 (32 casos)", None, None),
    ("  Aritmética /13", lambda n: por_bloque(n, "aritmetica")[0], "int"),
    ("  Compuesto /8", lambda n: por_bloque(n, "compuesto")[0], "int"),
    ("  Mal escrito /11", lambda n: por_bloque(n, "mal_escrito")[0], "int"),
    ("  Alucinaciones M3", lambda n: RESUMEN[n]["m3_aluc"], "int"),
    ("ROBUSTEZ Y CONSISTENCIA", None, None),
    ("  Perdidos por mala escritura", lambda n: RESUMEN[n]["rob"].get("perdidos_por_escritura"), "int"),
    ("  Misma respuesta que el original", lambda n: RESUMEN[n]["rob"].get("misma_respuesta"), "pct"),
    ("HERRAMIENTAS", None, None),
    ("  Llamadas por pregunta", lambda n: RESUMEN[n]["herr"]["llamadas_por_caso"], "f2"),
    ("  Llamadas válidas", lambda n: RESUMEN[n]["herr"]["tasa_llamadas_validas"], "pct"),
    ("  Llamadas sin error", lambda n: RESUMEN[n]["herr"]["tasa_llamadas_sin_error"], "pct"),
    ("  Usa herramienta cuando hace falta", lambda n: RESUMEN[n]["herr"]["uso_cuando_hace_falta"], "pct"),
    ("  No la usa cuando no hace falta", lambda n: RESUMEN[n]["herr"]["abstencion_cuando_no_hace_falta"], "pct"),
    ("  Selección correcta", lambda n: RESUMEN[n]["herr"]["seleccion_correcta"], "pct"),
    ("  Fidelidad a la herramienta", lambda n: RESUMEN[n]["herr"]["fidelidad_herramienta"], "pct"),
    ("PROCESO Y COSTE", None, None),
    ("  Pasos LLM por pregunta", lambda n: RESUMEN[n]["herr"]["pasos_llm_medios"], "f1"),
    ("  Tokens de entrada por pregunta", lambda n: RESUMEN[n]["tokens_entrada"], "int"),
    ("  Tokens generados por pregunta", lambda n: RESUMEN[n]["tokens_salida"], "int"),
    ("  Segundos por pregunta", lambda n: RESUMEN[n]["herr"]["segundos_medios"], "f1"),
    ("  Tope de pasos alcanzado", lambda n: RESUMEN[n]["herr"]["tope_pasos"], "int"),
    ("  Errores de formato", lambda n: RESUMEN[n]["herr"]["errores_formato"], "int"),
]

ancho = 38
print("=" * (ancho + 12 * len(NOMBRES)))
print(" " * ancho + "".join(f"{n.split(' · ')[0]:>12}" for n in NOMBRES))
print("-" * (ancho + 12 * len(NOMBRES)))
for etiqueta, fn, tipo in FILAS_SCORECARD:
    if fn is None:
        print(etiqueta)
    else:
        print(f"{etiqueta:<{ancho}}" + "".join(f"{fmt(fn(n), tipo):>12}" for n in NOMBRES))
print("=" * (ancho + 12 * len(NOMBRES)))
'''


CODIGO_DELTAS = r'''
def aciertos(nombre, conjunto=None, bloque=None):
    return [f["acierto"] for f in DETALLE[nombre]
            if (conjunto is None or f["conjunto"] == conjunto) and (bloque is None or f["bloque"] == bloque)]


filas_deltas = []
pares = list(zip(NOMBRES, NOMBRES[1:]))
if "C5-FT · entrenado" in NOMBRES:
    pares[-1] = ("C5 · agente", "C5-FT · entrenado")
pares += [("C1 · +LoRA", "C5 · agente"), ("C2 · +RAG", "C5 · agente")]

print(f"{'paso':40s}{'M2':>6}{'M3':>6}{'arit.':>7}{'comp.':>7}{'mal':>6}{'mej/emp':>9}{'p':>8}")
print("-" * 89)
for x, y in pares:
    m, e, p = mcnemar_exacto(aciertos(x), aciertos(y))
    d = lambda **kw: sum(aciertos(y, **kw)) - sum(aciertos(x, **kw))   # noqa: E731
    fila = {"de": x, "a": y, "delta_m2": d(conjunto="M2"), "delta_m3": d(conjunto="M3"),
            "delta_aritmetica": d(bloque="aritmetica"), "delta_compuesto": d(bloque="compuesto"),
            "delta_mal_escrito": d(bloque="mal_escrito"), "mejora": m, "empeora": e, "p_mcnemar": round(p, 4)}
    filas_deltas.append(fila)
    print(f"{x + ' -> ' + y:40s}{fila['delta_m2']:>+6d}{fila['delta_m3']:>+6d}{fila['delta_aritmetica']:>+7d}"
          f"{fila['delta_compuesto']:>+7d}{fila['delta_mal_escrito']:>+6d}{f'{m}/{e}':>9}{p:>8.3f}")
print("\nMcNemar sobre los 53 casos (M2 + M3). Con este tamaño, háblese de dirección, no de magnitud.")
'''


CODIGO_GRAFICA = r'''
import matplotlib.pyplot as plt

bloques = [("calculo", 12), ("conocimiento", 4), ("adversarial", 5),
           ("aritmetica", 13), ("compuesto", 8), ("mal_escrito", 11)]
x = np.arange(len(bloques)); ancho = 0.85 / len(NOMBRES)
fig, ax = plt.subplots(figsize=(13, 5))
for j, n in enumerate(NOMBRES):
    vals = [por_bloque(n, b)[0] / tot for b, tot in bloques]
    ax.bar(x - 0.425 + ancho / 2 + j * ancho, vals, ancho, label=n)
ax.axvline(2.5, color="gray", ls="--", lw=1)
ax.text(1, 1.08, "eval set M2", ha="center"); ax.text(4, 1.08, "eval set M3", ha="center")
ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={t})" for b, t in bloques])
ax.set_ylim(0, 1.15); ax.set_ylabel("aciertos"); ax.set_title("Aciertos por bloque y configuración")
ax.legend(ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12)); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.show()
'''


CODIGO_CAMBIOS = r'''
# Caso por caso: qué arregló y qué rompió cada paso de la escalera.
for x, y in zip(NOMBRES, NOMBRES[1:]):
    cambios = [(a, b) for a, b in zip(DETALLE[x], DETALLE[y]) if a["acierto"] != b["acierto"]]
    print(f"\n{x} -> {y}: {len(cambios)} cambio(s)")
    for a, b in cambios:
        print(f"   {'ARREGLÓ' if b['acierto'] else 'ROMPIÓ '}  {a['id']:7s} {a['subtipo']:26s} "
              f"herr=[{b['herramientas_usadas'][:40]}]  {b['motivo'][:40]}")
'''


CODIGO_HERRAMIENTAS_DETALLE = r'''
# ¿Qué herramientas eligió el agente, por familia requerida? (C5 y C5-FT)
for n in [n for n in NOMBRES if n.startswith("C5")]:
    print(f"\n{n}")
    print(f"  {'caso':8s} {'requiere':28s} {'usó':42s} sel  fid")
    for f in DETALLE[n]:
        req = ",".join(HERRAMIENTAS_ESPERADAS[f["id"]]["familias"]) or "(ninguna)"
        fid = "—" if f["fidelidad_herramienta"] is None else ("sí" if f["fidelidad_herramienta"] else "NO")
        print(f"  {f['id']:8s} {req[:28]:28s} {f['herramientas_usadas'][:42]:42s} "
              f"{'ok ' if f['seleccion_correcta'] else 'MAL'}  {fid}")
'''


CODIGO_AUTOCONSISTENCIA = r'''
# Autoconsistencia: la misma pregunta, muestreada 3 veces (T = 0.7).
# Greedy da siempre lo mismo por construcción; con muestreo se ve si la respuesta
# es estable o depende del azar de la decodificación. Se mide en el bloque de
# aritmética de M3, para C1 (sin herramientas) y las configuraciones con herramientas.
CORRER_AUTOCONSISTENCIA = True
SEMILLAS = [0, 1, 2]
CONFIGS_AC = ["C1 · +LoRA", "C4 · +ReAct", "C5 · agente"] + (["C5-FT · entrenado"] if "C5-FT · entrenado" in NOMBRES else [])

AUTOCONSISTENCIA = {}
if CORRER_AUTOCONSISTENCIA:
    muestreadas = {s: {c.nombre: c for c in construir_configs(muestreo=(0.7, s))} for s in SEMILLAS}
    casos_ac = [c for c in eval_m3 if c["bloque"] == "aritmetica"]
    for nombre in CONFIGS_AC:
        muestras, aciertos_muestras = {}, []
        for caso in casos_ac:
            respuestas = []
            for s in SEMILLAS:
                out = muestreadas[s][nombre].fn(caso["input"])
                respuestas.append(out.respuesta)
                aciertos_muestras.append(evaluar_caso(caso, out.respuesta)["acierto"])
            muestras[caso["id"]] = respuestas
        AUTOCONSISTENCIA[nombre] = {**autoconsistencia(muestras),
                                    "exactitud_media_muestras": float(np.mean(aciertos_muestras))}
        print(f"{nombre:20s} {AUTOCONSISTENCIA[nombre]}")
'''


CODIGO_ARTEFACTOS = r'''
Path("resultados").mkdir(exist_ok=True)

# 1 · Detalle caso por caso (todas las configuraciones, M2 + M3)
columnas = ["config", "conjunto", "id", "bloque", "subtipo", "original", "acierto", "abstuvo", "alucino", "juez",
            "sim", "formato_valido", "motivo", "llamadas", "llamadas_validas", "llamadas_sin_error",
            "herramientas_usadas", "recall_seleccion", "seleccion_correcta", "innecesarias",
            "fidelidad_herramienta", "pasos_llm", "terminacion", "errores_formato", "verificaciones",
            "reescrita_llm", "tokens_entrada", "tokens_salida", "segundos", "respuesta"]
df_detalle[columnas].to_csv("resultados/detalle_s10.csv", index=False, encoding="utf-8")

# 2 · Scorecard del agente (filas = métricas, columnas = configuraciones)
filas_csv = [{"metrica": etiqueta.strip(), **{n: fn(n) for n in NOMBRES}}
             for etiqueta, fn, _ in FILAS_SCORECARD if fn is not None]
pd.DataFrame(filas_csv).to_csv("resultados/scorecard_agente.csv", index=False, encoding="utf-8")

# 3 · Scorecard M2 con el formato de siempre (comparable con scorecard_m2.csv y scorecard_rag.csv)
guardar_scorecard("resultados/scorecard_m2_s10.csv", *[SC_M2[n] for n in NOMBRES])

# 4 · Deltas
pd.DataFrame(filas_deltas).to_csv("resultados/deltas_s10.csv", index=False, encoding="utf-8")

# 5 · Todo junto
Path("resultados/m3_agente.json").write_text(json.dumps({
    "configuraciones": {c.nombre: c.anade for c in CONFIGS},
    "adaptador_herramientas": ADAPTADOR_HERRAMIENTAS,
    "sondeo_adaptador": json.loads(resumen_sondeo.reset_index().to_json(orient="records")),
    "few_shot": FEW_SHOT_IDS,
    "harness_m2": {n: {k: v for k, v in SC_M2[n].items() if k != "detalle"} for n in NOMBRES},
    "eval_m3": {n: {b: por_bloque(n, b) for b in ("aritmetica", "compuesto", "mal_escrito")} for n in NOMBRES},
    "herramientas": {n: RESUMEN[n]["herr"] for n in NOMBRES},
    "robustez": {n: RESUMEN[n]["rob"] for n in NOMBRES},
    "tokens": {n: {"entrada": RESUMEN[n]["tokens_entrada"], "salida": RESUMEN[n]["tokens_salida"]} for n in NOMBRES},
    "autoconsistencia": AUTOCONSISTENCIA,
    "deltas": filas_deltas,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

for p in ["respuestas_s10.jsonl", "detalle_s10.csv", "scorecard_agente.csv", "scorecard_m2_s10.csv",
          "deltas_s10.csv", "m3_agente.json"]:
    print(f"  resultados/{p}")
'''


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "Arquitectura agéntica para el tutor de matemáticas",
        "Sesión 10 · Módulo 3 — RAG + ReAct + Tool use, medido con el harness (3/4)",
    ))

    c.append(md("""
## 1 · Introducción

### La pregunta del experimento

> **¿Una arquitectura agéntica basada en RAG + ReAct + herramientas mejora de verdad
> al tutor frente al sistema actual?**

No se trata de añadir complejidad. La regla de S10 es explícita: *empiecen con el
RAG de una pasada y agreguen un agente solo si el harness mejora de verdad*.

### La escalera: una sola cosa nueva en cada paso

| Config. | Añade | Qué pone a prueba |
|---|---|---|
| **C0** | Qwen2.5-1.5B base | Referencia (baseline de M2) |
| **C1** | + LoRA de M1 | El sistema actual sin RAG |
| **C2** | + RAG de S08 (contexto siempre) | El mejor RAG de una pasada |
| **C3** | + herramientas de cálculo, **una ronda** de tool calling | ¿Delegar las cuentas mejora? |
| **C4** | + **ReAct**: pensamiento → acción → observación, en varios pasos | ¿Encadenar pasos mejora? |
| **C5** | + retrieval **como herramienta**, planificación, verificador, normalizador | ¿La arquitectura agéntica completa mejora? |
| **C5-FT** | C5 con el adaptador entrenado (notebook 2) | ¿Entrenar supera al few-shot? |

### Predicciones escritas antes de ejecutar

| Bloque | Predicción | Razonamiento |
|---|---|---|
| M2 · cálculo | C3–C5 ≈ C1 (10/12) | Los dos fallos de C1 son de **planteamiento**; una calculadora no los arregla |
| M2 · conocimiento | C2 ≈ C5 > C1 | Solo mejora si hay documentos, fijos o buscados |
| M3 · aritmética | **C3–C5 ≫ C1** | Aquí sí hay cuentas donde un 1.5B se equivoca |
| M3 · compuesto | **C5 > C2–C4** | Hace falta buscar **y** calcular |
| M3 · mal escrito | C5 ≥ resto | Es la única configuración con normalizador |
| Coste | C5 ≈ 10× los tokens de entrada de C2 | Cada paso reenvía ~2.700 tokens de esquemas + few-shot (medido en local: ~7.500 vs ~550) |

Si C5 no supera a C2 en lo que importa, la conclusión correcta es que **el agente no
se justifica** para este tutor, y se reportará así.
"""))

    c.append(md("""
## 2 · Objetivos

1. Integrar el núcleo agéntico (probado en `scripts/agente/`) con el RAG de S08 y el
   generador de M1.
2. **Decidir con datos** qué adaptador usan las configuraciones con herramientas.
3. Pasar **siete configuraciones** por el mismo harness sobre 53 casos (M2 + M3),
   guardando la traza completa.
4. Reportar exactitud, uso y selección de herramientas, pasos, errores, calidad de la
   explicación, robustez, consistencia y coste.
5. Dejar `respuestas_s10.jsonl` listo para RAGAS (notebook 4).
"""))

    c.append(md("""
## 0 · Preparación del entorno

> **Activen la GPU (T4).** Se cargan el generador (con uno o dos adaptadores), el
> juez, el modelo de embeddings y el cross-encoder. La corrida completa es larga y
> **se puede retomar**: cada respuesta se guarda al generarse (sección 11).
"""))
    c.append(code('''
%pip install -q "transformers>=4.44" "peft>=0.12" "accelerate>=0.33" \\
    sentence-transformers chromadb rank_bm25 "scikit-learn>=1.3" sympy pandas
%pip uninstall -y -q torchao
print("Librerías instaladas. Si Colab pide reiniciar la sesión, reinícienla y sigan desde aquí.")
'''))
    c.append(code('''
import os, random, re, time
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

    c.append(md("""
---
## 3 · Los eval sets
"""))
    c.extend(celda_eval_set(prefijo="3.1"))
    c.extend(celda_eval_m3(prefijo="3.2"))

    c.append(md("""
---
## 4 · Métricas, embeddings y criterio de acierto

Las mismas funciones de M2, S07 y S08, sin tocar.
"""))
    c.append(code(CODIGO_METRICAS))
    c.extend(celda_similitud())
    c.extend(celda_evaluacion())

    c.append(md("""
---
## 5 · El RAG de S08

Corpus, chunking, índice denso, BM25, RRF y reranker **idénticos a S08**, generados
desde el mismo código. C2 lo usa como contexto fijo; C5, como herramienta.
"""))
    for bloque in (CODIGO_CORPUS, CODIGO_CHUNKING, CODIGO_INDICE, CODIGO_BM25, CODIGO_RRF, CODIGO_RERANKER):
        c.append(code(bloque))

    c.append(md("""
---
## 6 · El núcleo agéntico

Copia literal de `scripts/agente/`, cubierto por 61 tests. El notebook 1 lo explica
y lo prueba sin GPU.
"""))
    c.extend(celdas_modulos(["registro", "herramientas", "nucleo", "documentos", "prompts", "metricas"]))
    c.append(code(CODIGO_REGISTROS))

    c.append(md("""
---
## 7 · El generador

Un solo modelo base con los adaptadores como **intercambiables**: `m1` (M1),
`agente` (notebook 2, si existe) y `None` (modelo base, con `disable_adapter`).
La generación es greedy y está cacheada: el mismo prompt da siempre el mismo texto,
así que las configuraciones que comparten pasos no los regeneran. El
`CONTADOR` de tokens sí cuenta los aciertos de caché, porque mide coste, no tiempo.
"""))
    c.append(code(CODIGO_LLM))
    c.append(code(CODIGO_PROMPTS))

    c.append(md("""
---
## 8 · Few-shot: las trayectorias como ejemplos

Tres conversaciones completas por configuración, elegidas **por criterio** y solo
del split de entrenamiento. Enseñan tres decisiones: usar herramientas (en C5,
también buscar), encadenar pasos y **no** usarlas cuando no corresponde.
"""))
    c.extend(celda_trayectorias(prefijo="8.1"))
    c.append(code(CODIGO_FEW_SHOT))

    c.append(md("""
---
## 9 · Sondeo: ¿con qué adaptador se usan las herramientas?

La propuesta señaló un riesgo: la LoRA de M1 aprendió a empezar **siempre** con
"Paso 1:", y eso puede impedir que el modelo emita `<tool_call>`. En lugar de
suponerlo, se mide sobre las **trayectorias de validación** (no sobre el eval set).
Se elige el adaptador con mejor decisión y más llamadas válidas.
"""))
    c.append(code(CODIGO_SONDEO))

    c.append(md("""
---
## 10 · Las configuraciones
"""))
    c.append(code(CODIGO_CONFIGS))
    c.append(code(CODIGO_PRUEBA_AGENTE))

    c.append(md("""
---
## 11 · El harness de M2, sin tocar

Juez y harness idénticos a M2. Las configuraciones se envuelven para que el harness
siga recibiendo una función `pregunta -> respuesta`, mientras la traza completa se
guarda aparte.
"""))
    c.extend(celda_juez())
    c.append(code(CODIGO_JUEZ_CACHE))
    c.extend(celda_harness())
    c.append(code(CODIGO_PERSISTENCIA))
    c.append(code(CODIGO_CORRIDA))
    c.append(code(CODIGO_REPRODUCIBILIDAD))

    c.append(md("""
---
## 12 · Resultados

### 12.1 · El scorecard del agente
"""))
    c.append(code(CODIGO_ENRIQUECER))
    c.append(code(CODIGO_SCORECARD))
    c.append(md("""
### 12.2 · Deltas paso a paso

Cada fila compara configuraciones que difieren en **una sola cosa** (salvo las dos
últimas, que resumen la escalera).
"""))
    c.append(code(CODIGO_DELTAS))
    c.append(code(CODIGO_GRAFICA))
    c.append(md("""
### 12.3 · Qué arregló y qué rompió cada paso
"""))
    c.append(code(CODIGO_CAMBIOS))
    c.append(md("""
### 12.4 · Selección de herramientas, caso por caso
"""))
    c.append(code(CODIGO_HERRAMIENTAS_DETALLE))
    c.append(code("tabla_por_subtipo(*[SC_M3[n] for n in NOMBRES])"))

    c.append(md("""
---
## 13 · Autoconsistencia

Con greedy, la misma pregunta da siempre la misma respuesta **por construcción**, así
que esa consistencia no dice nada. Aquí se muestrea tres veces (T = 0.7) sobre el
bloque de aritmética. Un sistema que acierta "por suerte" se delata: sus muestras no
coinciden.
"""))
    c.append(code(CODIGO_AUTOCONSISTENCIA))

    c.append(md("""
---
## 14 · Artefactos

| Archivo | Contenido | Lo usa |
|---|---|---|
| `resultados/respuestas_s10.jsonl` | Respuesta, contextos, observaciones y traza de cada (config, caso) | Notebook 4 (RAGAS) |
| `resultados/detalle_s10.csv` | Caso por caso con métricas de herramientas y proceso | Informe |
| `resultados/scorecard_agente.csv` | El scorecard de 12.1 | Informe, notebook 4 |
| `resultados/scorecard_m2_s10.csv` | Harness M2 con el formato de siempre | Comparación con M2/S07/S08 |
| `resultados/deltas_s10.csv` | Deltas y McNemar | Informe |
| `resultados/m3_agente.json` | Todo lo anterior + sondeo, few-shot, autoconsistencia | Notebook 4 |
"""))
    c.append(code(CODIGO_ARTEFACTOS))

    c.append(md("""
---
## 15 · Discusión

Las preguntas a responder con las tablas, en este orden:

**1. ¿Las herramientas mejoraron el cálculo que falla por aritmética?** (M3 · aritmética,
C1 → C3). Es la hipótesis más directa. Si no mejora, miren 12.4: ¿no llamó a las
herramientas (*uso cuando hace falta* bajo), llamó mal (*llamadas válidas* bajo) o
llamó bien e ignoró el resultado (*fidelidad* baja)? Son tres fallos distintos con
tres arreglos distintos.

**2. ¿ReAct aportó algo sobre una sola ronda?** (C3 → C4). Solo debería notarse en
problemas de varios pasos. Si no aporta y cuesta más pasos, no se justifica.

**3. ¿El agente completo ganó donde debía?** (C4 → C5 en *compuesto* y *mal escrito*;
C2 → C5 en *conocimiento*). Si C5 pierde conocimiento frente a C2, el agente no está
decidiendo buscar cuando hace falta.

**4. ¿Rompió algo del harness de M2?** Un agente que mejora M3 pero empeora el
bloque adversarial o las alucinaciones de M2 es **peor tutor**, no mejor.

**5. ¿Entrenar superó al few-shot?** (C5 → C5-FT).

**6. ¿Cuánto cuesta?** Tokens y segundos por pregunta. Cada paso con herramientas
vuelve a enviar ~2.700 tokens de esquemas. Si C5 cuesta 10× lo de C2 para ganar 2
casos, esa es la cifra que hay que poner en el informe.

**7. Tamaño de muestra.** 53 casos, con bloques de 4 a 13. Ninguna diferencia de
1–2 casos es concluyente; los p de McNemar lo dirán.

---
### Siguiente: notebook 4

`respuestas_s10.jsonl` pasa por **RAGAS** con un evaluador de 7B, se calibra contra
las métricas objetivas y se arma el scorecard final de la entrega.
"""))

    c.append(md("""
---
## 16 · Conclusiones de esta corrida

> Cifras del harness corregidas con `scripts/recalcular_criterio.py`: la corrida usó
> la versión antigua del eval set (arriba, `SHA coincide: False`) y `adv-02` se
> re-evaluó sobre las respuestas ya generadas. Cambió un solo caso, en C1.

| | C0 | C1 | C2 | C3 | C4 | C5 | C5-FT |
|---|---|---|---|---|---|---|---|
| M2 /21 | 9 | **14** | 13 | 13 | **14** | 12 | 13 |
| M3 · aritmética /13 | 5 | 8 | 5 | 7 | 4 | 7 | **10** |
| M3 · compuesto /8 | 0 | 0 | 2 | **5** | **5** | **5** | 3 |
| M3 · mal escrito /11 | 4 | 5 | 5 | 6 | 5 | 6 | **7** |
| **Total /53** | 18 | 27 | 25 | 31 | 28 | 30 | **33** |
| Usa herramienta cuando hace falta | — | — | — | 2% | 0% | 19% | **96%** |
| Selección correcta | 9% | 9% | 9% | 11% | 9% | 25% | **58%** |
| Tokens entrada / pregunta | 75 | 75 | 539 | 3 332 | 3 416 | 4 842 | **10 136** |
| Segundos / pregunta | 9.2 | 5.5 | 4.7 | 3.6 | 3.4 | 5.0 | **14.1** |

**1 · Sin entrenamiento, la arquitectura agéntica es un cascarón.** C3 llama
herramientas en el 2% de los casos que las necesitan, C4 en el 0% y C5 en el 19%. El
formato es el nativo de Qwen2.5 y los ejemplos son correctos y verificados, pero un
1.5B no basta. Las ganancias de C3 y C4 en el bloque compuesto (0 → 5 de 8) vienen
del **contexto inyectado**, no de las herramientas: eso es RAG, no agencia.

**2 · Con el adaptador entrenado, sí.** C5-FT usa herramienta en el 96% de los casos
que la piden, acierta la familia en el 58% y es el mejor sistema global (33/53), el
mejor en aritmética (10/13) y el que menos alucina en M3 (12 contra 17 de C1).

**3 · El agente no mejora la vara fija.** En el eval set de M2 nadie supera a C1
(14/21). El valor aparece solo en los bloques nuevos, que existen precisamente
porque M2 no podía medir esto. Es una conclusión incómoda y hay que reportarla así.

**4 · C4 (ReAct) fue el peor paso de la escalera.** −3 casos y, sobre todo, el
formato válido se hunde al 19%: el modelo se queda escribiendo "Pensamiento:" y no
cierra con "Respuesta final:". Pedirle a un modelo pequeño que razone en voz alta y
además respete un formato de salida compite por la misma capacidad.

**5 · C5-FT sobre-usa las herramientas.** Se abstiene solo en el 20% de los casos
donde no hacían falta (C5: 80%) y busca documentos menos que C5, lo que le cuesta el
bloque compuesto (3/8 contra 5/8). El entrenamiento enseñó *cómo* llamar, no *cuándo
no* hacerlo.

**6 · El sondeo decidió bien.** Con el adaptador de M1 la tasa de decisión correcta
era 21.4% contra 16.7% del modelo base, así que C3–C5 corrieron con `m1`. Aun así,
ambas cifras son bajísimas: el sondeo anticipó, sobre validación, que el few-shot no
iba a alcanzar.

**7 · El coste.** C5-FT gasta 19× los tokens de entrada de C2 y triplica la latencia
(14.1 s contra 4.7 s) para ganar 8 casos de 53. Cada paso reenvía ~2 700 tokens de
esquemas más el few-shot.

**8 · Autoconsistencia.** Con muestreo (T = 0.7, k = 3) sobre aritmética: C5-FT es el
más estable (46% de unanimidad, exactitud media 0.69) y C5 el más inestable (15%,
0.33). El agente sin entrenar no solo acierta menos: acierta por azar de la
decodificación.

**Veredicto.** Para este tutor, la arquitectura agéntica **no se justifica todavía**
por sí sola: el mejor sistema sobre el eval set de M2 sigue siendo el enrutador de
S08 (18/21). El agente entrenado gana en los bloques nuevos y en la verificabilidad
de sus números, pero paga 19× en tokens. La vía razonable es combinarlos: enrutador
de S08 para decidir si consultar documentos, herramientas entrenadas para la vía de
cálculo.
"""))

    return c
