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
SALIDA = RAIZ / "M5_despliegue" / "resultados" / "vision_m5.json"
SALIDA_CSV = RAIZ / "M5_despliegue" / "resultados" / "vision_m5.csv"
LADO_MAYOR = 1024          # el mismo tope de M4: la latencia por caso no debe
                           # depender de cómo salió la foto

# Conflictos verificados a mano entre la hoja fotografiada y el JSONL, el
# 2026-11-03. Un caso en conflicto SE LEE —la evidencia de lectura sigue
# sirviendo— pero NO se puntúa: su respuesta esperada describe una hoja que
# no es la que hay. Resolverlos es una decisión de diseño: o se repite la
# foto, o se corrige el JSONL. No se hace ninguna de las dos aquí.
CONFLICTOS = {
    "m5v-01": "la hoja escribe «Respuesta final: 642 km»; el JSONL espera 942 "
              "(valor_final_escrito). El paso con error sigue siendo el 2, pero el "
              "número final ya no es el del eval set.",
    "m5v-05": "la hoja escribe «2 × 15000 = 3000»; el JSONL dice «2 × 1500 = 3000» "
              "y el enunciado de la propia hoja dice «lapiceros de 1500». Con un "
              "cero de más el paso 2 también queda mal, y el caso es de tipo "
              "`compensado`: esperaba que el ÚNICO primer error fuera el paso 1 y "
              "prohíbe nombrar el paso 2. Además la hoja tiene 3 pasos y el JSONL 4.",
    "m5v-09": "la hoja escribe «Respuesta final: 19 estudiantes»; su propio paso 2 "
              "dice «40 × 0.35 = 14» y el JSONL espera 14. El caso está declarado "
              "`sin_error`, y con un 19 al final sí hay un error en el paso 3.",
    "m5v-10": "la foto es LEGIBLE. El caso existe para medir la salvaguarda de "
              "ilegibles y el JSONL pide explícitamente que esta hoja se fotografíe "
              "movida. Con una foto nítida no se puede afirmar que la válvula quedó "
              "probada, en ningún sentido.",
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
        "transcripcion": r.get("transcripcion"),
        "pasos": pasos,
        "n_pasos": len(pasos),
        "n_pasos_esperados": len(caso.get("pasos") or []),
        "numero_escrito_preservado": (
            None if escrito is None
            else escrito.replace(" ", "") in " ".join(pasos).replace(" ", "")),
        "lectura_fiel": fiel,
    }


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


def etapa_diagnostico(tutor, caso: dict, pasos: list) -> dict:
    """Pasos confirmados -> revisión del tutor. La lectura ya terminó."""
    t0 = time.perf_counter()
    r = tutor.revisar_pasos(pasos, caso["pregunta"],
                            enunciado=caso.get("enunciado"), confirmado=True)
    return {
        "segundos": round(time.perf_counter() - t0, 2),
        "ok": bool(r.get("ok")),
        "respuesta": r.get("respuesta"),
        "verificador_aritmetico": r.get("verificador_aritmetico"),
        "pasos_llm": r.get("pasos_llm"),
    }


# --------------------------------------------------------------------------

def escribir_csv(filas: list) -> None:
    """Una fila por foto. El scorecard que se lee sin abrir el JSON."""
    import csv

    columnas = ["caso", "archivo", "bloque", "variante", "estado", "bloqueado",
                "segundos_lectura", "lectura", "px", "confirmable",
                "pide_otra_foto", "motivo_no_fiable", "lecturas_coinciden",
                "n_pasos", "n_pasos_esperados", "numero_escrito_preservado",
                "lectura_fiel", "segundos_diagnostico", "verificador",
                "herramientas_reales", "acierto", "fuente_del_fallo",
                "transcripcion", "respuesta"]
    SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        for f in filas:
            lec = f.get("lectura") or {}
            dia = f.get("diagnostico") or {}
            cal = f.get("calificacion") or {}
            w.writerow({
                "caso": f["caso"], "archivo": f["archivo"],
                "bloque": f.get("bloque", ""), "variante": f.get("variante", ""),
                "estado": f["estado"], "bloqueado": f.get("bloqueado", ""),
                "segundos_lectura": lec.get("segundos", ""),
                "lectura": lec.get("lectura", ""),
                "px": "x".join(str(n) for n in lec.get("px", [])),
                "confirmable": lec.get("confirmable", ""),
                "pide_otra_foto": lec.get("pide_otra_foto", ""),
                "motivo_no_fiable": lec.get("motivo_no_fiable", ""),
                "lecturas_coinciden": lec.get("lecturas_coinciden", ""),
                "n_pasos": lec.get("n_pasos", ""),
                "n_pasos_esperados": lec.get("n_pasos_esperados", ""),
                "numero_escrito_preservado": lec.get("numero_escrito_preservado", ""),
                "lectura_fiel": lec.get("lectura_fiel", ""),
                "segundos_diagnostico": dia.get("segundos", ""),
                "verificador": json.dumps(dia.get("verificador_aritmetico"),
                                          ensure_ascii=False),
                # Cuántas herramientas se EJECUTARON de verdad. Si aquí sale 0,
                # la respuesta no está verificada por herramienta, diga lo que diga.
                "herramientas_reales": dia.get("pasos_llm", ""),
                "acierto": cal.get("acierto", ""),
                "fuente_del_fallo": f.get("fuente_del_fallo", ""),
                "transcripcion": (lec.get("transcripcion") or "").replace("\n", " / "),
                "respuesta": (dia.get("respuesta") or "").replace("\n", " / "),
            })


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
        ruta = IMAGENES / archivo
        if not ruta.is_file():
            filas.append({"archivo": archivo, "caso": caso["id"],
                          "variante": etiqueta, "estado": "ausente"})
            print(f"--- {archivo}  AUSENTE")
            continue

        lec = etapa_lectura(tutor, ruta, caso)
        # La carga ya se pagó arriba, así que esta es la lectura en frío de
        # verdad: la primera inferencia con el modelo ya en memoria.
        lec["lectura"] = "fria" if primera else "caliente"
        primera = False

        fila = {"archivo": archivo, "caso": caso["id"], "bloque": caso["bloque"],
                "variante": etiqueta, "estado": "leido", "lectura": lec}

        print(f"--- {archivo}  ({caso['id']} · {caso['bloque']} · {etiqueta})"
              f"  {lec['segundos']}s  [{lec['lectura']}]")
        print(f"      confirmable={lec['confirmable']}  coinciden={lec['lecturas_coinciden']}"
              f"  pasos={lec['n_pasos']}/{lec['n_pasos_esperados']}"
              f"  numero_escrito_preservado={lec['numero_escrito_preservado']}"
              f"  lectura_fiel={lec['lectura_fiel']}")
        if lec["pide_otra_foto"]:
            print(f"      PIDE OTRA FOTO: {lec['motivo_no_fiable']}")
            if caso["id"] in CONFLICTOS:
                fila["bloqueado"] = CONFLICTOS[caso["id"]]
            # Y aquí se para. Diagnosticar sobre una lectura que el propio
            # sistema declaró no fiable es exactamente lo que vis-10 demostró
            # que no se puede hacer.
            filas.append(fila)
            continue

        dia = etapa_diagnostico(tutor, caso, lec["pasos"])
        fila["diagnostico"] = dia
        if caso["id"] in CONFLICTOS:
            # Se guarda lo que contestó, para poder leerlo; no se puntúa. La
            # respuesta esperada describe una hoja que no es la que hay.
            fila["bloqueado"] = CONFLICTOS[caso["id"]]
            print(f"      diagnóstico {dia['segundos']}s · BLOQUEADO, sin puntuar")
            print(f"      {CONFLICTOS[caso['id']]}")
            filas.append(fila)
            continue
        cal = calificar(caso, dia["respuesta"] or "")
        from vision import fuente_del_fallo
        fila["calificacion"] = cal
        fila["fuente_del_fallo"] = fuente_del_fallo(cal["acierto"], lec["lectura_fiel"])
        print(f"      diagnóstico {dia['segundos']}s · acierto={cal['acierto']}"
              f" · fuente_del_fallo={fila['fuente_del_fallo']}")
        filas.append(fila)

    parcial = bool(solo) or not inv["completo"]
    informe = {
        "tipo": "evaluación de la vía visual de M5",
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
        calientes = sorted(f["lectura"]["segundos"] for f in filas
                           if f.get("lectura", {}).get("lectura") == "caliente")
        informe["latencias"] = {
            "carga_del_vlm_s": carga["segundos"],
            "lectura_fria_s": next((f["lectura"]["segundos"] for f in filas
                                    if f.get("lectura", {}).get("lectura") == "fria"),
                                   None),
            "lecturas_calientes_s": calientes,
            # p50/p95 solo con muestra. Con menos de 20 observaciones un p95 es
            # el máximo con otro nombre.
            "p50_s": (calientes[len(calientes) // 2] if len(calientes) >= 20 else None),
            "p95_s": (calientes[int(len(calientes) * 0.95)] if len(calientes) >= 20 else None),
            "nota": ("p50 y p95 solo se calculan con 20 o más lecturas en "
                     "caliente; por debajo de eso el p95 es el máximo con otro "
                     "nombre. Se dan los tiempos uno a uno."),
        }
        informe["metricas"] = {
            "fotos_del_conjunto": len(filas),
            "fotos_leidas": sum(1 for f in filas if f["estado"] == "leido"),
            "bloqueadas_por_conflicto": sum(1 for f in filas if f.get("bloqueado")),
            "puntuables": len(juzgadas),
            "pidio_otra_foto": sum(1 for f in filas
                                   if f.get("lectura", {}).get("pide_otra_foto")),
            "aciertos": sum(1 for f in juzgadas if f["calificacion"]["acierto"]),
            "fallos_de_lectura": sum(1 for f in juzgadas
                                     if f["fuente_del_fallo"] == "lectura"),
            "fallos_de_diagnostico": sum(1 for f in juzgadas
                                         if f["fuente_del_fallo"] == "diagnostico"),
        }

    informe["conflictos"] = CONFLICTOS
    informe["decision_pendiente"] = (
        "Cuatro de las doce hojas no coinciden con lo que el JSONL declara "
        "(m5v-01, m5v-05, m5v-09) o no son la toma que el caso pide (m5v-10, "
        "fotografiada legible cuando el caso existe para medir la salvaguarda de "
        "ilegibles). Se leen y se guarda lo que contestó el tutor, pero no se "
        "puntúan: su respuesta esperada describe una hoja distinta. Resolverlo "
        "es una decisión de diseño —repetir la foto o corregir el JSONL— y no se "
        "toma aquí. Mientras siga abierta, cualquier tasa sobre este conjunto es "
        "sobre 8 casos, no sobre 12, y así se informa.")
    informe["limites_de_la_muestra"] = (
        "8 casos puntuables. Con ese tamaño un acierto más o menos mueve la tasa "
        "12,5 puntos: no sirve para afirmar una mejora frente a ninguna otra "
        "configuración, solo para ver si el recorrido completo funciona.")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    escribir_csv(filas)
    print(f"\nescrito: {SALIDA.relative_to(RAIZ)}")
    print(f"escrito: {SALIDA_CSV.relative_to(RAIZ)}")
    if parcial:
        print("Informe PARCIAL · sin métricas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
