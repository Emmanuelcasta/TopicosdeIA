# =========================================================================
# CELDA PARA PEGAR en S13, después de la celda 50 (`sistema_multimodal`).
#
# Dos pruebas sobre los MISMOS 9 ejemplos `dg-*`, cambiando una sola cosa cada
# vez, para que el efecto sea atribuible:
#
#   baseline (ya medido)  adaptador `diagnostico` + prompt original   -> 3/9
#   PRUEBA A              adaptador `agente`      + prompt original
#   PRUEBA B              adaptador `diagnostico` + prompt con `Paso N`
#
# A aísla el adaptador; B aísla el ejemplo del prompt. Cada una escribe su propio
# CSV y **ninguna toca** `m4/dev_dg_arnes_s13.csv`, que es el baseline.
#
# NO toca los 12 casos de evaluación ni el eval set. NO es un scorecard de M4.
#
# ---------------------------------------------------------------------------
# CORRECCIÓN respecto a la medición anterior, y conviene saberla:
#
# aquella leía la traza con `TRAZAS_M4[prompt[:160]]`, y los nueve prompts
# comparten los primeros 160 caracteres —todos empiezan por la misma instrucción—.
# Así que el diccionario tenía UNA entrada sobrescrita nueve veces: las columnas
# de herramientas, `pasos_llm` y `terminacion` mostraban la última llamada,
# repetida. El "0 llamadas a herramientas" no estaba medido por caso.
#
# Aquí la traza se toma del objeto que devuelve el agente, sin pasar por el
# diccionario, así que es la de cada caso.
# =========================================================================
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

RUTA_DG = Path("data/trayectorias_diagnostico.jsonl")
SHA_DG = "87bd8f19b983369395cc9a9e29c62b1e211776c74539258a62e2bb58d5147589"
BASELINE_CSV = Path("m4/dev_dg_arnes_s13.csv")        # NO se toca
CSV_A = Path("m4/dev_dg_A_agente.csv")
CSV_B = Path("m4/dev_dg_B_pasoN.csv")

# --- los datos, verificados ----------------------------------------------
sha = hashlib.sha256(RUTA_DG.read_bytes()).hexdigest()
assert sha == SHA_DG, f"otro conjunto: {sha[:16]}… != {SHA_DG[:16]}…"
TODOS = [json.loads(l) for l in RUTA_DG.read_text(encoding="utf-8").splitlines()]
DEV = [e for e in TODOS if e["split"] == "dev"]
assert len(DEV) == 9 and all(e["split"] == "dev" for e in DEV)
for e in DEV:
    assert diagnostico_valido(e["diagnostico"].lower()), e["id"]
print(f"{len(DEV)} ejemplos de desarrollo, SHA verificado.")

# Los prompts originales se guardan aparte: A los usa tal cual y B los parchea.
PROMPT_ORIGINAL = {e["id"]: e["mensajes"][1]["content"] for e in DEV}


def _construir_agente(adaptador, nombre):
    return AgenteTutor(
        crear_llm(adaptador, 320), SYSTEM_C5_AGENTE, registro=REGISTRO_M4,
        max_rondas=6, max_pasos=8, verificar=True,
        extraer_contextos=chunks_de_resultado, nombre=nombre)


def medir(prompts, adaptador, etiqueta):
    """Los 9 casos con un adaptador y unos prompts dados. -> DataFrame.

    La traza sale del objeto del agente, no de `TRAZAS_M4`: ver la corrección
    del encabezado.
    """
    agente = _construir_agente(adaptador, etiqueta)
    print(f"\n{'=' * 78}\n{etiqueta}  ·  adaptador={adaptador!r}\n{'=' * 78}")
    filas = []
    for e in DEV:
        prompt = prompts[e["id"]]
        esperada = e["diagnostico"].lower()

        salida = agente(prompt)
        respuesta, reintento = asegurar_diagnostico(
            salida.respuesta, lambda p: agente(p).respuesta)
        conclusion = extraer_diagnostico(respuesta)
        herramientas = [o["herramienta"] for o in salida.observaciones]

        filas.append({
            "config": etiqueta, "adaptador": adaptador,
            "id": e["id"], "familia": e["familia"],
            "esperada": esperada, "predicha": conclusion,
            "formato_valido": diagnostico_valido(conclusion),
            "acierto": diagnostico_valido(conclusion) and conclusion == esperada,
            "n_herramientas": len(herramientas),
            "herramientas": "|".join(herramientas) or "—",
            "pasos_llm": salida.pasos_llm,
            "terminacion": salida.terminacion,
            "reintento": reintento,
            "reescribe": "Respuesta final" in respuesta,
            "respuesta": respuesta,
        })
        f = filas[-1]
        print(f"  {e['id']:8s} {e['familia']:18s} {'OK ' if f['acierto'] else 'MAL'}  "
              f"esperaba {esperada!r:20s} predijo {str(conclusion)!r:20s} "
              f"herr={f['n_herramientas']}")
    return pd.DataFrame(filas)


def resumen(df, etiqueta):
    print(f"\n{etiqueta}: {int(df.acierto.sum())}/{len(df)} aciertos | "
          f"{int(df.formato_valido.sum())}/{len(df)} formato | "
          f"{int(df.n_herramientas.sum())} llamadas a herramientas en total")
    for cat in ("paso", "sin error", "ilegible", "fuera de dominio", "dato no disponible"):
        g = df[df.esperada.str.startswith("paso")] if cat == "paso" else df[df.esperada == cat]
        if len(g):
            print(f"    {cat:20s} {int(g.acierto.sum())}/{len(g)}")
    print(f"    conclusiones emitidas: {sorted(set(df.predicha.dropna()))}")


# =========================================================================
# PRUEBA A · adaptador `agente`, prompt ORIGINAL
# =========================================================================
assert "agente" in ADAPTADORES, "no está cargado el adaptador de S10"
A = medir(PROMPT_ORIGINAL, "agente", "A · agente + prompt original")
resumen(A, "A")
A.to_csv(CSV_A, index=False)
print(f"Guardado: {CSV_A}")

# =========================================================================
# PRUEBA B · adaptador `diagnostico` (recargado), prompt con `Paso N`
# =========================================================================
DIR_DIAG = "adaptadores/qwen-lora-diagnostico"
ok_diag, motivo_diag = revisar_adaptador(DIR_DIAG)
print(f"\n{DIR_DIAG} -> {motivo_diag}")
assert ok_diag, "no se puede hacer la prueba B sin el adaptador"
if "diagnostico" not in modelo.peft_config:
    modelo.load_adapter(DIR_DIAG, adapter_name="diagnostico")
    if "diagnostico" not in ADAPTADORES:
        ADAPTADORES.append("diagnostico")
print("Adaptadores en el modelo:", list(modelo.peft_config.keys()))

# El parche: una sola sustitución, y se comprueba que cayó en los nueve.
VIEJO, NUEVO = "  Diagnóstico: Paso 2", "  Diagnóstico: Paso N"
PROMPT_PARCHEADO = {}
aplicados = 0
for cid, p in PROMPT_ORIGINAL.items():
    assert p.count(VIEJO) == 1, f"{cid}: {p.count(VIEJO)} ocurrencias, se esperaba 1"
    PROMPT_PARCHEADO[cid] = p.replace(VIEJO, NUEVO)
    assert VIEJO not in PROMPT_PARCHEADO[cid] and NUEVO in PROMPT_PARCHEADO[cid]
    # Nada más cambió: la diferencia es exactamente un carácter, '2' -> 'N'.
    assert len(PROMPT_PARCHEADO[cid]) == len(p)
    aplicados += 1
print(f"\nParche aplicado a {aplicados}/{len(PROMPT_ORIGINAL)} prompts "
      f"({VIEJO.strip()!r} -> {NUEVO.strip()!r}).")
assert aplicados == 9

B = medir(PROMPT_PARCHEADO, "diagnostico", "B · diagnostico + prompt con 'Paso N'")
resumen(B, "B")
B.to_csv(CSV_B, index=False)
print(f"Guardado: {CSV_B}")

# =========================================================================
# Comparación · baseline, A y B
# =========================================================================
print("\n" + "=" * 78)
print("BASELINE vs A vs B · 9 casos de desarrollo · NO es un scorecard de M4")
print("=" * 78)

base = pd.read_csv(BASELINE_CSV) if BASELINE_CSV.exists() else None
if base is None:
    print(f"(no está {BASELINE_CSV}; el baseline fue 3/9, con 'ilegible' como")
    print(" único acierto fuera de 'paso N')")

col_base = ("conclusion" if base is not None and "conclusion" in base.columns
            else "predicha")
pred_base = (dict(zip(base.id, base[col_base])) if base is not None else {})
ok_base = (dict(zip(base.id, base.acierto)) if base is not None else {})

print(f"{'caso':9s}{'esperada':20s}{'baseline':20s}{'A · agente':20s}{'B · Paso N':20s}")
for e in DEV:
    cid = e["id"]
    a = A[A.id == cid].iloc[0]
    b = B[B.id == cid].iloc[0]
    def m(v, ok):
        return f"{str(v):16s}{'✓' if ok else '✗'}"
    celda_base = (m(pred_base[cid], ok_base.get(cid, False)) if cid in pred_base
                  else f"{'(sin CSV)':16s} ")
    print(f"{cid:9s}{e['diagnostico'].lower():20s}{celda_base:20s}"
          f"{m(a.predicha, a.acierto):20s}{m(b.predicha, b.acierto):20s}")

print("-" * 78)
tot_base = int(sum(bool(v) for v in ok_base.values())) if base is not None else None
txt_base = f"{tot_base}/9" if tot_base is not None else "3/9 (del notebook)"
print(f"{'TOTAL':29s}{txt_base:20s}{int(A.acierto.sum())}/9{'':17s}{int(B.acierto.sum())}/9")
print(f"{'llamadas a herramientas':29s}{'?':19s}{int(A.n_herramientas.sum()):<20d}"
      f"{int(B.n_herramientas.sum())}")
print("=" * 78)

# Qué queda respaldado por los datos, sin adornos.
print("\nLECTURA")
if int(A.n_herramientas.sum()) > 0 and int(B.n_herramientas.sum()) == 0:
    print("· `agente` llama herramientas y `diagnostico` no: el adaptador las suprimió.")
elif int(A.n_herramientas.sum()) == 0:
    print("· Ninguno llama herramientas: el prompt de revisión no las induce, y eso")
    print("  es anterior al adaptador.")

movidos = [e["id"] for e in DEV
           if B[B.id == e["id"]].iloc[0].predicha != pred_base.get(e["id"])]
print(f"· El parche del prompt cambió la conclusión en {len(movidos)}/9 casos: {movidos or '—'}")
if tot_base is not None and int(B.acierto.sum()) > tot_base:
    print("  y subió el acierto: el índice venía, al menos en parte, del ejemplo.")
elif movidos:
    print("  pero sin subir el acierto: cambió la respuesta, no el criterio.")
else:
    print("  es decir, nada: el sesgo del índice NO venía del prompt.")

sin_err = B[B.esperada == "sin error"]
print(f"· 'sin error' en B: {int(sin_err.acierto.sum())}/{len(sin_err)}. Mientras siga")
print("  en 0, los 6 casos de `sin_error` + adversarial del eval set darían casi 0")
print("  por construcción, y NO tiene sentido gastar los 12.")

# --- respuestas completas ------------------------------------------------
for etiqueta, df in (("A · agente", A), ("B · diagnostico + Paso N", B)):
    print("\n" + "=" * 78)
    print(f"RESPUESTAS COMPLETAS · {etiqueta}")
    print("=" * 78)
    for _, f in df.iterrows():
        print(f"\n--- {f['id']} · {f['familia']} · esperaba {f['esperada']!r} "
              f"-> predijo {f['predicha']!r} ---")
        print(f"herramientas: {f['herramientas']} | pasos_llm: {f['pasos_llm']} | "
              f"terminación: {f['terminacion']}")
        print(f["respuesta"])

print(f"\n\nCSV: {CSV_A} · {CSV_B}")
print(f"El baseline {BASELINE_CSV} no se tocó.")
print("Los tres son de DESARROLLO. Ninguno es el scorecard de M4.")
