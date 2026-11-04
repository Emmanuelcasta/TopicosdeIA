"""Evaluador de la vía visual de M5 sobre el eval set oficial de 12 fotos.

Dos etapas, y se informan por separado a propósito. La **lectura** convierte una
foto en una transcripción; la **calificación** juzga lo que el tutor dice sobre
unos pasos ya confirmados. Mezclarlas fue el error de M4: un fallo se atribuía al
tutor cuando el VLM había inventado la cuenta. `fuente_del_fallo` separa las dos.

Qué NO hace:

  * No inventa, altera ni duplica fotografías. Si falta una, lo dice y para.
  * No imprime ninguna métrica de exactitud si el conjunto no corrió completo.
    Un «8/12» sobre las fotos que había acabaría citado como si fuera el
    resultado del eval set.
  * No tiene nada que ver con `M5_despliegue/evalset-m5/kaggle_dataset_humo/`,
    que es una prueba de humo. Ese paquete no entra aquí por ninguna ruta.

Uso:
    python M5_despliegue/evaluacion/evaluar_vision_m5.py --contrato
        Valida el inventario de fotos. No carga ningún modelo. Es lo que hay que
        correr al recibir las fotos, antes de gastar GPU.

    python M5_despliegue/evaluacion/evaluar_vision_m5.py --correr
        Evalúa. Se niega si falta una sola foto.

    M5_IMAGENES=/ruta/alternativa  ...  --correr --solo m5v-01,m5v-07
        `--solo` produce un informe PARCIAL, marcado como tal y sin métricas.

Alcance. Una foto por caso, `m5v-NN.jpg`, m5v-01 a m5v-12. La definición
anterior —23 archivos, pares nítido/borroso— se conserva en
`variantes_historico_23` porque describe lo que `m4/eval_set/HOJAS_M5.md` pidió
y lo que el manifiesto documentó durante toda la fase 2. No se borra: un
alcance que cambia sin dejar rastro convierte «faltan 19» en «estaba completo»
sin que nadie pueda ver cuándo pasó.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "M5_despliegue" / "app"), str(RAIZ / "scripts"),
                str(RAIZ / "scripts" / "agente")]

JSONL = RAIZ / "data" / "eval_set_m5_vision.jsonl"
IMAGENES = Path(os.environ.get("M5_IMAGENES", RAIZ / "m4" / "eval_set" / "imagenes_m5"))
# El mismo tope que la app (`TUTOR_LADO_MAYOR`): evaluador e interfaz tienen
# que ver los mismos píxeles.
LADO_MAYOR = int(os.environ.get("TUTOR_LADO_MAYOR", "1024"))
_NOMBRE = os.environ.get("M5_SALIDA", "vision_m5")
SALIDA = RAIZ / "M5_despliegue" / "resultados" / f"{_NOMBRE}.json"
SALIDA_CSV = RAIZ / "M5_despliegue" / "resultados" / f"{_NOMBRE}.csv"
SALIDA_TABLA = RAIZ / "M5_despliegue" / "resultados" / f"{_NOMBRE}_tabla.md"

# Fuera de ESTA ronda, por decisión del responsable (2026-11-05): no se evalúan
# fotos borrosas. Ni se lee ni se puntúa; consta en el informe.
EXCLUIDOS_DE_LA_RONDA = {
    "m5v-10": "fuera de la ronda: su foto es legible y el caso mide fotos "
              "borrosas, que esta ronda no evalúa.",
}

# Conflictos verificados a mano entre la hoja fotografiada y el JSONL, el
# 2026-11-03. Un caso en conflicto SE LEE —la evidencia de lectura sigue
# sirviendo— pero NO se puntúa: su respuesta esperada describe una hoja que
# no es la que hay. Resolverlos es una decisión de diseño: o se repite la
# foto, o se corrige el JSONL. No se hace ninguna de las dos aquí.
CONFLICTOS = {
    # Las referencias de m5v-01, m5v-05 y m5v-09 se corrigieron en el JSONL
    # para que describan las hojas fotografiadas (642 km; 2 × 15000; 19
    # estudiantes, ahora `localizacion` con error en el paso 3). Ya no están en
    # conflicto y se puntúan.
    "m5v-10": "NO VÁLIDO para medir la salvaguarda de ilegibles: la foto es "
              "LEGIBLE y el caso exige una toma movida. Se lee y se registra, pero "
              "no cuenta ni como acierto ni como fallo de la válvula hasta que haya "
              "una foto borrosa de verdad.",
}


# --------------------------------------------------------------------------
# Inventario · se ejecuta siempre, y sin GPU
# --------------------------------------------------------------------------

def casos() -> list:
    if not JSONL.exists():
        raise SystemExit(f"no existe {JSONL.relative_to(RAIZ)}")
    return [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]


def variantes_historico_23(cs) -> list:
    """El alcance ANTERIOR: pares nítido/borroso, 23 archivos.

    Se conserva como historia. `m5v-10` solo declaraba la borrosa, y eso era
    parte del diseño: si existiera su versión legible, el caso dejaría de medir
    la salvaguarda. Nada del evaluador lo usa hoy.
    """
    salida = []
    for c in cs:
        for clave, etiqueta in (("imagen_nitida", "nitida"),
                                ("imagen_borrosa", "borrosa")):
            if c.get(clave):
                salida.append((c[clave], c, etiqueta))
    return salida


def variantes(cs) -> list:
    """-> [(archivo, caso, etiqueta)] · UNA foto por caso, `m5v-NN.jpg`.

    La etiqueta no la pone el nombre del archivo: sale del JSONL. `m5v-10`
    declara `imagen_nitida: null`, así que su única toma es la borrosa, y eso
    sigue siendo lo que el caso pide aunque el archivo se llame igual que los
    demás.
    """
    return [(f"{c['id']}.jpg", c,
             "borrosa" if not c.get("imagen_nitida") else "nitida") for c in cs]


def inventario(cs) -> dict:
    """Qué fotos pide el eval set, cuáles hay y qué sobra. Nunca adivina.

    No empareja por nombre base ni acepta extensiones alternativas: el lector de
    `validacion_m5.py` hace `glob('*.jpg')`, así que un `.jpeg` es invisible para
    el resto del harness y aquí tiene que contarse como ausente, no como
    presente con otro nombre.
    """
    pedidas = [a for a, _, _ in variantes(cs)]
    presentes = sorted(p.name for p in IMAGENES.glob("*")) if IMAGENES.is_dir() else []
    faltan = [a for a in pedidas if a not in presentes]
    inesperados = [p for p in presentes if p not in pedidas]
    return {
        "directorio": str(IMAGENES),
        "existe_el_directorio": IMAGENES.is_dir(),
        "pedidas": len(pedidas),
        "presentes": len([p for p in presentes if p in pedidas]),
        "faltan": faltan,
        "nombres_inesperados": inesperados,
        "completo": not faltan,
    }


def imprimir_inventario(inv: dict, cs: list) -> None:
    print("=" * 78)
    print("INVENTARIO DEL EVAL SET VISUAL DE M5")
    print("=" * 78)
    print(f"directorio: {inv['directorio']}"
          + ("" if inv["existe_el_directorio"] else "   NO EXISTE"))
    print(f"fotos: {inv['presentes']} de {inv['pedidas']}")
    if inv["faltan"]:
        print(f"\nFALTAN {len(inv['faltan'])}:")
        por_caso = {}
        for archivo, c, etiqueta in variantes(cs):
            if archivo in inv["faltan"]:
                por_caso.setdefault(c["id"], []).append((archivo, etiqueta, c))
        for cid in sorted(por_caso):
            for archivo, etiqueta, c in por_caso[cid]:
                print(f"  {archivo:28s} {cid} · {c['bloque']} · {etiqueta}")
                if etiqueta == "borrosa" and c["id"] == "m5v-10":
                    print("      esta hoja SOLO se fotografía borrosa")
    if inv["nombres_inesperados"]:
        print(f"\nNOMBRES INESPERADOS ({len(inv['nombres_inesperados'])}) · el "
              "lector hace glob('*.jpg') y no los va a emparejar:")
        for n in inv["nombres_inesperados"]:
            print(f"  {n}")
    print()


# --------------------------------------------------------------------------
# Etapa 1 · lectura de la imagen
# --------------------------------------------------------------------------

def cargar(ruta: Path):
    """JPG -> PIL RGB, orientación corregida y lado mayor acotado."""
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
    if max(im.size) > LADO_MAYOR:
        e = LADO_MAYOR / max(im.size)
        im = im.resize((int(im.width * e), int(im.height * e)), Image.LANCZOS)
    return im


def etapa_lectura(tutor, ruta: Path, caso: dict) -> dict:
    """Foto -> transcripción. No juzga al tutor: todavía no ha hablado."""
    from vision import lectura_fiel

    im = cargar(ruta)
    t0 = time.perf_counter()
    r = tutor.leer_imagen(im)
    seg = round(time.perf_counter() - t0, 2)

    pasos = r.get("pasos") or []
    # `lectura_fiel` compara contra `pasos` y `valor_final_escrito` del eval set:
    # si el VLM «arregló» el número mal escrito, el caso ya no mide lo que decía.
    fiel = lectura_fiel(caso, {"ok": bool(r.get("ok")),
                               "json": {"pasos": pasos,
                                        "respuesta_final": pasos[-1] if pasos else ""},
                               "consistente": r.get("lecturas_coinciden")})
    escrito = caso.get("valor_final_escrito")
    return {
        "segundos": seg,
        "px": list(im.size),
        "ok": bool(r.get("ok")),
        "confirmable": r.get("confirmable"),
        "pide_otra_foto": r.get("confirmable") is False,
        "motivo_no_fiable": r.get("motivo_no_fiable") or "",
        "lecturas_coinciden": r.get("lecturas_coinciden"),
        "diferencias": r.get("diferencias") or [],
        "discrepancias_mostradas": r.get("discrepancias") or [],
        "diferencias_inocuas": r.get("diferencias_inocuas") or [],
        "enunciado_fiel": enunciado_fiel(caso, r.get("enunciado") or ""),
        "proposito": r.get("proposito"),
        "enunciado": r.get("enunciado"),
        "transcripcion": r.get("transcripcion"),
        "pasos": pasos,
        "n_pasos": len(pasos),
        "n_pasos_esperados": len(caso.get("pasos") or []),
        "numero_escrito_preservado": (
            None if escrito is None
            else escrito.replace(" ", "") in " ".join(pasos).replace(" ", "")),
        "lectura_fiel": fiel,
    }


def enunciado_fiel(caso: dict, leido: str) -> bool | None:
    """¿El enunciado leído trae los mismos datos que la hoja? Compara los
    números, que es lo que cambia el ejercicio; la redacción no."""
    from vision import _firma_matematica
    if not caso.get("check"):
        return None
    return sorted(_firma_matematica(leido)[0]) == sorted(
        _firma_matematica(caso.get("enunciado") or "")[0])


# --------------------------------------------------------------------------
# Etapa 2 · calificación del tutor, sobre pasos YA confirmados
# --------------------------------------------------------------------------

def _norm(t: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", (t or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def calificar(caso: dict, respuesta: str) -> dict:
    """Aplica el contrato que el propio eval set declara en `evaluacion`.

    `tipo: contiene_alguna` -> acierta si aparece una de `claves` y ninguna de
    `prohibido`. Las dos condiciones, no una: nombrar el paso correcto y además
    otros tres no es acertar.

    No se usa la tabla `DIAGNOSTICO_ADVERSARIAL` de `vision.py`: está indexada
    por los identificadores `vis-NN` de M4 y no cubre los `m5v-NN`.
    """
    ev = caso["evaluacion"]
    if ev["tipo"] != "contiene_alguna":
        raise ValueError(f"[{caso['id']}] tipo de evaluación no soportado: {ev['tipo']}")
    r = _norm(respuesta)
    halladas = [k for k in ev.get("claves", []) if _norm(k) in r]
    violadas = [k for k in ev.get("prohibido", []) if _norm(k) in r]
    return {
        "criterio": caso["criterio"],
        "claves_halladas": halladas,
        "prohibido_hallado": violadas,
        "acierto": bool(halladas) and not violadas,
    }


def texto_a_calificar(dia: dict) -> str:
    """Lo que se puntúa: la conclusión FINAL (tutor + verificador), no la prosa.

    La respuesta que ve el estudiante cita a veces las dos señales cuando
    discrepan («el tutor dijo paso 2; las cuentas, paso 3»), y el contrato
    `contiene_alguna` castigaría haber mencionado el paso descartado. Para las
    abstenciones (fuera de dominio, dato no disponible) se puntúa el texto del
    tutor, porque esas categorías no las decide el verificador.
    """
    d = dia.get("diagnostico_final") or {}
    final = d.get("conclusion")
    if final and (final.startswith("paso") or final == "sin error"):
        return f"Diagnóstico: el {final}" if final.startswith("paso") else "Diagnóstico: sin error"
    if d.get("fuente") in ("verificador", "referencia"):
        # Abstención decidida por el código (falta un dato): se puntúa la
        # conclusión y su motivo, que es lo que ve el estudiante.
        return f"Diagnóstico: {final}. {d.get('motivo') or ''}"
    return dia.get("respuesta_modelo") or dia.get("respuesta") or ""


def calificar_solucion(caso: dict, sol: dict) -> dict:
    """Flujo 1 · ¿la solución del enunciado llega al resultado correcto?

    El resultado correcto sale de `check`, la cuenta que el eval set ya
    declaraba, no de `valor_correcto` escrito a mano. Para los adversariales
    (no es un ejercicio, falta un dato) rige su contrato de `claves`.
    """
    from verificador_aritmetico import _valor, valor_final
    if caso["bloque"] == "adversarial":
        return calificar(caso, sol.get("respuesta") or "")
    esperado = _valor(caso["check"])
    obtenido = valor_final(sol.get("pasos_solucion") or
                           (sol.get("respuesta") or "").splitlines())
    return {"criterio": "resultado", "esperado": None if esperado is None else str(esperado),
            "obtenido": None if obtenido is None else str(obtenido),
            "acierto": esperado is not None and obtenido == esperado}


def etapa_solucion(tutor, enunciado_leido: str) -> dict:
    """Flujo 1 · el enunciado leído y confirmado tal cual -> solución."""
    t0 = time.perf_counter()
    r = tutor.resolver_enunciado(enunciado_leido, confirmado=True)
    return {"segundos": round(time.perf_counter() - t0, 2), "ok": bool(r.get("ok")),
            "camino": r.get("camino"), "respuesta": r.get("respuesta"),
            "pasos_solucion": r.get("pasos_solucion") or [],
            "verificador_aritmetico": r.get("verificador_aritmetico"),
            "error": None if r.get("ok") else r.get("respuesta")}


def etapa_diagnostico(tutor, caso: dict, pasos: list, proposito: str = "revisar",
                      enunciado_leido: str = "") -> dict:
    """Lo confirmado -> respuesta del tutor. La lectura ya terminó.

    Igual que en la interfaz: si la lectura no encontró pasos y sugirió
    resolver, se resuelve el enunciado LEÍDO. Revisar una lista vacía de
    pasos solo devolvería un error, y el caso quedaría sin respuesta que
    calificar.
    """
    t0 = time.perf_counter()
    if proposito == "resolver":
        r = tutor.resolver_enunciado(enunciado_leido, confirmado=True)
    else:
        # El enunciado LEÍDO, no el del eval set: en la app el diagnóstico
        # solo ve lo que el estudiante confirmó. Hasta b439be5 se pasaba el de
        # referencia, que el tutor nunca tendría.
        r = tutor.revisar_pasos(pasos, caso["pregunta"],
                                enunciado=enunciado_leido, confirmado=True)
    return {
        "segundos": round(time.perf_counter() - t0, 2),
        "ok": bool(r.get("ok")),
        "respuesta": r.get("respuesta"),
        "verificador_aritmetico": r.get("verificador_aritmetico"),
        "pasos_llm": r.get("pasos_llm"),
        "herramientas": r.get("herramientas") or [],
        "respuesta_modelo": r.get("respuesta_modelo"),
        "diagnostico_final": r.get("diagnostico_final"),
        "planteamiento": r.get("planteamiento"),
        "referencia": r.get("referencia"),
        "modo": proposito,
        "error": None if r.get("ok") else r.get("respuesta"),
    }


# --------------------------------------------------------------------------

def _resumen_caso(f: dict) -> dict:
    """Una fila de la tabla por caso: lectura, solución, diagnóstico,
    explicación, latencias y fallo observado."""
    lec = f.get("lectura") or {}
    sol = f.get("solucion") or {}
    dia = f.get("diagnostico") or {}
    cal = f.get("calificacion") or {}
    csol = sol.get("calificacion") or {}
    df = dia.get("diagnostico_final") or {}
    fallos = []
    if f["estado"] == "excluido":
        fallos.append("excluido de la ronda")
    if lec.get("pide_otra_foto"):
        fallos.append(f"lectura rechazada: {lec.get('motivo_no_fiable')}")
    if lec.get("lectura_fiel") is False:
        fallos.append("lectura de los pasos no fiel a la hoja")
    if lec.get("enunciado_fiel") is False:
        fallos.append("datos del enunciado mal leídos")
    if sol and not csol.get("acierto"):
        fallos.append("solución incorrecta")
    if cal and not cal.get("acierto"):
        fallos.append(f"diagnóstico incorrecto ({f.get('fuente_del_fallo')})")
    if cal.get("fallo_grave"):
        fallos.append("ERROR INVENTADO en hoja sin error")
    explicacion = (df.get("motivo") or "") if df else (dia.get("respuesta") or "")
    return {
        "caso": f["caso"], "bloque": f.get("bloque", ""), "estado": f["estado"],
        "lectura": ("; ".join(lec.get("pasos") or []) or lec.get("enunciado") or ""),
        "enunciado_leido": lec.get("enunciado") or "",
        "discrepancias_mostradas": " | ".join(lec.get("discrepancias_mostradas") or []),
        "lectura_s": lec.get("segundos", ""),
        "solucion": (f"{csol.get('obtenido')} (esperado {csol.get('esperado')})"
                     if csol.get("criterio") == "resultado"
                     else (sol.get("respuesta") or "")[:120]),
        "solucion_acierto": csol.get("acierto", ""),
        "resolver_s": sol.get("segundos", ""),
        "diagnostico": df.get("conclusion") or (dia.get("respuesta") or "")[:80],
        "esperado": f.get("esperado", ""),
        "diagnostico_acierto": cal.get("acierto", ""),
        "fuente": df.get("fuente", ""),
        "explicacion": explicacion.replace("\n", " "),
        "revisar_s": dia.get("segundos", ""),
        "fallo_observado": "; ".join(fallos) or "ninguno",
    }


def escribir_csv(filas: list) -> None:
    """Una fila por caso. El scorecard que se lee sin abrir el JSON."""
    import csv

    resumen = [_resumen_caso(f) for f in filas]
    SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0]))
        w.writeheader()
        w.writerows(resumen)


def escribir_tabla(filas: list, informe: dict) -> None:
    """La misma tabla en Markdown, con la cabecera de la corrida."""
    cols = [("caso", "Caso"), ("lectura_s", "Lectura s"), ("discrepancias_mostradas",
            "Discrepancias mostradas"), ("solucion", "Resolver"), ("solucion_acierto", "✓"),
            ("resolver_s", "s"), ("diagnostico", "Revisar"), ("esperado", "Esperado"),
            ("diagnostico_acierto", "✓"),
            ("revisar_s", "s"), ("explicacion", "Motivo"), ("fallo_observado", "Fallo observado")]
    lineas = [f"# Scorecard visual M5 · commit {informe.get('commit')} · "
              f"{informe.get('fecha')} · lado mayor {informe.get('lado_mayor_px')} px", "",
              informe.get("naturaleza", ""), "",
              "| " + " | ".join(t for _, t in cols) + " |",
              "|" + "---|" * len(cols)]
    for f in filas:
        r = _resumen_caso(f)
        lineas.append("| " + " | ".join(str(r.get(k, "")).replace("|", "/")[:160]
                                        for k, _ in cols) + " |")
    SALIDA_TABLA.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contrato", action="store_true",
                    help="solo valida el inventario de fotos; no carga modelos")
    ap.add_argument("--correr", action="store_true", help="evalúa de verdad")
    ap.add_argument("--solo", default="", help="ids separados por coma (informe PARCIAL)")
    a = ap.parse_args()

    cs = casos()
    inv = inventario(cs)
    imprimir_inventario(inv, cs)

    if not a.correr:
        print("Solo inventario. Para evaluar: --correr")
        return 0 if inv["completo"] else 1

    solo = [x.strip() for x in a.solo.split(",") if x.strip()]
    if not inv["completo"] and not solo:
        print("NO SE EVALÚA: falta al menos una foto del conjunto oficial.")
        print("Un resultado sobre las fotos que haya no es el del eval set, y la")
        print("cifra acabaría citada como si lo fuera. Con --solo <ids> se puede")
        print("hacer una corrida PARCIAL, que no imprime métricas.")
        return 1

    import pipeline

    tutor = pipeline.Tutor("revision_foto")
    print(f"configuración: {tutor.cfg.nombre}  ·  vision={tutor.cfg.vision}")

    # Descarga y carga del VLM, cronometradas antes de la primera foto. Dentro
    # de ella no se distinguen, y una latencia de lectura que incluye 4,42 GB de
    # descarga no es una latencia de lectura.
    carga = tutor.precargar_vision()
    print(f"carga del VLM: {carga['segundos']} s · dispositivo "
          f"{carga['dispositivo']} · ok={carga['ok']}\n")

    filas, primera = [], True
    for archivo, caso, etiqueta in variantes(cs):
        if solo and caso["id"] not in solo:
            continue
        if caso["id"] in EXCLUIDOS_DE_LA_RONDA:
            filas.append({"archivo": archivo, "caso": caso["id"], "bloque": caso["bloque"],
                          "variante": etiqueta, "estado": "excluido",
                          "excluido": EXCLUIDOS_DE_LA_RONDA[caso["id"]]})
            print(f"--- {archivo}  EXCLUIDO de esta ronda")
            continue
        ruta = IMAGENES / archivo
        if not ruta.is_file():
            filas.append({"archivo": archivo, "caso": caso["id"],
                          "variante": etiqueta, "estado": "ausente"})
            print(f"--- {archivo}  AUSENTE")
            continue

        lec = etapa_lectura(tutor, ruta, caso)
        lec["lectura"] = "fria" if primera else "caliente"
        primera = False
        fila = {"archivo": archivo, "caso": caso["id"], "bloque": caso["bloque"],
                "variante": etiqueta, "estado": "leido", "lectura": lec,
                "esperado": (f"paso {caso['paso_error']}" if caso.get("paso_error")
                             else caso["bloque"])}
        print(f"--- {archivo}  ({caso['id']} · {caso['bloque']})  lectura {lec['segundos']}s"
              f" [{lec['lectura']}] · confirmable={lec['confirmable']}"
              f" · coinciden={lec['lecturas_coinciden']} · pasos={lec['n_pasos']}/"
              f"{lec['n_pasos_esperados']} · fiel={lec['lectura_fiel']}"
              f" · enunciado_fiel={lec['enunciado_fiel']}")
        for d in lec["discrepancias_mostradas"]:
            print(f"      discrepancia mostrada: {d}")
        if lec["pide_otra_foto"]:
            print(f"      PIDE OTRA FOTO: {lec['motivo_no_fiable']}")
            filas.append(fila)
            continue

        # Se simula a un estudiante que confirma la lectura SIN corregir nada:
        # el peor caso. Si hubo discrepancias mostradas, consta en la fila.
        # Flujo 1 · resolver el enunciado leído.
        if (lec.get("enunciado") or "").strip():
            sol = etapa_solucion(tutor, lec["enunciado"])
            sol["calificacion"] = calificar_solucion(caso, sol)
            fila["solucion"] = sol
            print(f"      RESOLVER {sol['segundos']}s · camino={sol['camino']} · "
                  f"acierto={sol['calificacion']['acierto']} "
                  f"({sol['calificacion'].get('obtenido')} vs {sol['calificacion'].get('esperado')})")
        # Flujo 2 · revisar el procedimiento. Sin pasos, la app propone resolver
        # y eso es lo que se puntúa (m5v-11).
        if lec["pasos"]:
            dia = etapa_diagnostico(tutor, caso, lec["pasos"], "revisar",
                                    lec.get("enunciado") or "")
        else:
            dia = etapa_diagnostico(tutor, caso, [], "resolver", lec.get("enunciado") or "")
        fila["diagnostico"] = dia
        cal = calificar(caso, texto_a_calificar(dia))
        cal["acierto_solo_tutor"] = calificar(
            caso, dia.get("respuesta_modelo") or dia["respuesta"] or "")["acierto"]
        conclusion = (dia.get("diagnostico_final") or {}).get("conclusion") or ""
        cal["fallo_grave"] = caso["bloque"] == "sin_error" and conclusion.startswith("paso")
        from vision import fuente_del_fallo
        fila["calificacion"] = cal
        fila["fuente_del_fallo"] = fuente_del_fallo(cal["acierto"], lec["lectura_fiel"])
        df = dia.get("diagnostico_final") or {}
        print(f"      REVISAR {dia['segundos']}s · final={df.get('conclusion')} "
              f"(fuente {df.get('fuente')}) · tutor={df.get('modelo')} · "
              f"esperado={caso.get('paso_error') or caso['bloque']} · "
              f"acierto={cal['acierto']} · grave={cal['fallo_grave']} · "
              f"fuente_del_fallo={fila['fuente_del_fallo']}")
        filas.append(fila)

    parcial = bool(solo) or not inv["completo"]
    import datetime
    import subprocess
    try:
        commit = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    informe = {
        "tipo": "evaluación de la vía visual de M5",
        "commit": commit,
        "fecha": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "naturaleza": ("evaluación de DESARROLLO: estas 12 hojas se han usado para "
                       "depurar la lectura, el verificador y la regla de diagnóstico. "
                       "No es evidencia de generalización."),
        "excluidos_de_la_puntuacion": EXCLUIDOS_DE_LA_RONDA,
        "parcial": parcial,
        "eval_set": str(JSONL.relative_to(RAIZ)),
        "inventario": inv,
        "carga_del_vlm": carga,
        "variantes": filas,
    }
    if parcial:
        informe["metricas"] = None
        informe["por_que_sin_metricas"] = (
            "corrida parcial: una tasa calculada sobre un subconjunto no es la "
            "del eval set y acabaría citada como si lo fuera")
    else:
        juzgadas = [f for f in filas if "calificacion" in f]
        resueltas = [f for f in filas if "solucion" in f]
        calientes = sorted(f["lectura"]["segundos"] for f in filas
                           if f.get("lectura", {}).get("lectura") == "caliente")
        informe["latencias"] = {
            "carga_del_vlm_s": carga["segundos"],
            "lectura_fria_s": next((f["lectura"]["segundos"] for f in filas
                                    if f.get("lectura", {}).get("lectura") == "fria"), None),
            "lecturas_calientes_s": calientes,
            "resolver_s": [f["solucion"]["segundos"] for f in resueltas],
            "revisar_s": [f["diagnostico"]["segundos"] for f in juzgadas],
            "nota": ("menos de 20 observaciones por etapa: sin p50/p95, se dan "
                     "los tiempos uno a uno. revisar incluye resolver el "
                     "enunciado aparte cuando las cuentas cuadran (salvo caché)."),
        }
        informe["metricas"] = {
            "fotos_evaluadas": sum(1 for f in filas if f["estado"] == "leido"),
            "excluidas_de_la_ronda": sorted(EXCLUIDOS_DE_LA_RONDA),
            "pidio_otra_foto": sorted(f["caso"] for f in filas
                                      if f.get("lectura", {}).get("pide_otra_foto")),
            "con_discrepancias_mostradas": sorted(
                f["caso"] for f in filas if f.get("lectura", {}).get("discrepancias_mostradas")),
            "lectura_fiel_pasos": sum(1 for f in filas
                                      if f.get("lectura", {}).get("lectura_fiel") is True),
            "lectura_fiel_enunciado": sum(1 for f in filas
                                          if f.get("lectura", {}).get("enunciado_fiel") is True),
            "resolver": {"puntuables": len(resueltas),
                         "aciertos": sum(1 for f in resueltas
                                         if f["solucion"]["calificacion"]["acierto"])},
            "revisar": {
                "puntuables": len(juzgadas),
                "aciertos": sum(1 for f in juzgadas if f["calificacion"]["acierto"]),
                "aciertos_solo_tutor": sum(1 for f in juzgadas
                                           if f["calificacion"].get("acierto_solo_tutor")),
                "fallos_graves_error_inventado": sum(
                    1 for f in juzgadas if f["calificacion"].get("fallo_grave")),
                "fallos_de_lectura": sum(1 for f in juzgadas
                                         if f["fuente_del_fallo"] == "lectura"),
                "fallos_de_diagnostico": sum(1 for f in juzgadas
                                             if f["fuente_del_fallo"] == "diagnostico"),
            },
        }

    informe["conflictos"] = CONFLICTOS
    informe["excluidos_de_la_ronda"] = EXCLUIDOS_DE_LA_RONDA
    informe["lado_mayor_px"] = LADO_MAYOR
    informe["limites_de_la_muestra"] = (
        "11 fotos legibles. Un acierto mueve la tasa unos 9 puntos. Las hojas se "
        "reutilizaron para depurar el sistema: sirve para ver si el recorrido "
        "funciona, no para comparar configuraciones ni afirmar generalización.")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    escribir_csv(filas)
    escribir_tabla(filas, informe)
    print(f"\nescrito: {SALIDA.relative_to(RAIZ)}")
    print(f"escrito: {SALIDA_CSV.relative_to(RAIZ)}")
    if parcial:
        print("Informe PARCIAL · sin métricas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
