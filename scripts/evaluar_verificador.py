"""
Mide el verificador de planteamiento contra las trazas ya guardadas.

No hace falta GPU: las respuestas y las observaciones de las 7 configuraciones de
S10 están en `respuestas_s10.jsonl`, y el acierto de cada caso en
`detalle_s10_v2.csv`. El verificador es una función pura de esos datos.

Qué se mide, y por qué estas dos cifras y no una sola:

  SENSIBILIDAD   De los casos que el sistema falló, ¿en cuántos el verificador
                 levanta la mano? Es lo que puede recuperar.

  FALSOS POSITIVOS  De los casos que el sistema acertó, ¿en cuántos se queja? Cada
                 uno cuesta una ronda extra de generación y puede estropear una
                 respuesta que ya estaba bien. Un verificador que señala todo no
                 sirve: hay que leer las dos columnas juntas.

Uso:
    python scripts/evaluar_verificador.py                 # resumen
    python scripts/evaluar_verificador.py --detalle       # caso por caso
    python scripts/evaluar_verificador.py --config "C5-FT · entrenado"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "agente"))

from verificador import VerificadorPlan  # noqa: E402

RAIZ = AQUI.parent
RES = RAIZ / "M3_rag_agentes" / "resultados"


def cargar():
    respuestas = {}
    for linea in (RES / "respuestas_s10.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(linea)
        respuestas[(r["config"], r["id"])] = r
    detalle = {}
    for fila in csv.DictReader((RES / "detalle_s10_v2.csv").open(encoding="utf-8")):
        detalle[(fila["config"], fila["id"])] = fila
    return respuestas, detalle


def clasificar_fallo(reg, fila) -> str:
    """La misma taxonomía del diagnóstico, para saber qué tipo de fallo caza."""
    obs = reg["observaciones"]
    exitosas = [o for o in obs if o["ok"] and o.get("valor")]
    busco = any(o["herramienta"] == "buscar_documentos" for o in obs)
    if fila["bloque"] == "adversarial":
        return "robustez"
    if fila["bloque"] in ("conocimiento", "compuesto") and not (busco or reg["contextos"]):
        return "no consultó documentos"
    if any(not o["ok"] for o in obs):
        return "herramienta falló"
    if exitosas:
        return "planteamiento"
    return "aritmética de cabeza"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detalle", action="store_true")
    p.add_argument("--config", default=None)
    p.add_argument("--pasos", type=int, default=1, help="pasos de derivación permitidos")
    args = p.parse_args()

    respuestas, detalle = cargar()
    verificador = VerificadorPlan(pasos_derivacion=args.pasos)
    configs = [args.config] if args.config else sorted({c for c, _ in detalle})

    print(f"{'configuración':22s} {'fallos':>7s} {'detectados':>11s} {'aciertos':>9s} {'falsos+':>8s}")
    print("-" * 62)
    por_tipo = defaultdict(lambda: [0, 0])
    detalles = []

    for cfg in configs:
        fallos = aciertos = detectados = falsos = 0
        for (c, cid), fila in detalle.items():
            if c != cfg:
                continue
            reg = respuestas[(c, cid)]
            rev = verificador.revisar(reg["pregunta"], reg["respuesta"], reg["observaciones"],
                                      reg.get("contextos", ()))
            señala = bool(rev.hallazgos)
            if fila["acierto"].lower() == "true":
                aciertos += 1
                falsos += señala
                if señala and args.detalle:
                    detalles.append((cfg, cid, "FALSO+", rev.hallazgos[0]))
            else:
                fallos += 1
                detectados += señala
                tipo = clasificar_fallo(reg, fila)
                por_tipo[tipo][1] += 1
                por_tipo[tipo][0] += señala
                if args.detalle:
                    detalles.append((cfg, cid, "detectado" if señala else "NO detectado",
                                     rev.hallazgos[0] if rev.hallazgos else None))
        if fallos or aciertos:
            print(f"{cfg[:22]:22s} {fallos:7d} {detectados:11d} {aciertos:9d} {falsos:8d}")

    print("\nPor tipo de fallo (detectados / total):")
    for tipo, (det, tot) in sorted(por_tipo.items(), key=lambda x: -x[1][1]):
        pct = f"{det / tot:.0%}" if tot else "—"
        print(f"  {tipo:26s} {det:3d} / {tot:3d}   {pct}")

    if args.detalle:
        print("\nCaso por caso:")
        for cfg, cid, estado, h in detalles:
            print(f"  {cfg[:18]:18s} {cid:8s} {estado:12s} {h or ''}")


if __name__ == "__main__":
    main()
