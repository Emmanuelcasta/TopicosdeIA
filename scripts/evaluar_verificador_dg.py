"""Mide el verificador aritmético sobre los 9 casos `dg-*` de desarrollo.

Sin GPU y sin modelo: es aritmética exacta. Contesta la pregunta que decide si el
componente sigue adelante o se queda como experimento:

    ¿marca los pasos erróneos evaluables SIN marcar los casos `sin error`?

Criterio previsto, fijado antes de ejecutar:
  · detectar los 4 pasos erróneos evaluables, con el índice correcto;
  · 0 falsos positivos en los 2 casos `sin error`;
  · abstenerse en los 3 adversariales, que no tienen aritmética.

Solo desarrollo. **Los 12 casos de evaluación no se tocan.**

    python scripts/evaluar_verificador_dg.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "agente"))

from verificador_aritmetico import revisar_hoja  # noqa: E402

RAIZ = AQUI.parent
DATOS = RAIZ / "data" / "trayectorias_diagnostico.jsonl"
SALIDA = RAIZ / "m4" / "verificador_aritmetico_dg.csv"

ABSTENCIONES = ("ilegible", "fuera de dominio", "dato no disponible")


def main() -> None:
    todos = [json.loads(l) for l in DATOS.read_text(encoding="utf-8").splitlines()]
    dev = [e for e in todos if e["split"] == "dev"]
    assert len(dev) == 9, f"se esperaban 9 casos de desarrollo, hay {len(dev)}"

    filas = []
    for e in dev:
        esperada = e["diagnostico"].lower()
        h = revisar_hoja(e["pasos"])
        marcado = f"paso {h.paso}" if h.paso else ("sin error" if h.tipo == "ninguno"
                                                   else "no evaluable")
        evaluable = esperada.startswith("paso ") or esperada == "sin error"
        filas.append({
            "id": e["id"], "familia": e["familia"], "esperada": esperada,
            "evaluable": evaluable, "marcado": marcado, "tipo": h.tipo,
            "acierto": evaluable and marcado == esperada,
            "falso_positivo": esperada == "sin error" and h.marca_un_paso,
            "detalle": h.detalle,
        })

    print(f"{'caso':9s}{'esperada':20s}{'el verificador dice':24s}{'tipo':13s}")
    for f in filas:
        marca = "OK " if f["acierto"] else ("FP " if f["falso_positivo"] else "   ")
        print(f"{f['id']:9s}{f['esperada']:20s}{f['marcado']:24s}{f['tipo']:13s}{marca}")
        print(f"         {f['detalle']}")

    con_paso = [f for f in filas if f["esperada"].startswith("paso ")]
    sin_error = [f for f in filas if f["esperada"] == "sin error"]
    adversarial = [f for f in filas if f["esperada"] in ABSTENCIONES]
    aciertos = sum(f["acierto"] for f in filas)
    fp = sum(f["falso_positivo"] for f in filas)
    se_abstuvo = sum(1 for f in adversarial if f["tipo"] == "no_evaluable")

    print("\n" + "=" * 70)
    print("VERIFICADOR ARITMÉTICO · 9 casos de DESARROLLO · no es un scorecard")
    print("=" * 70)
    print(f"  pasos erróneos detectados con el índice correcto : "
          f"{sum(f['acierto'] for f in con_paso)}/{len(con_paso)}")
    print(f"  casos `sin error` sin marcar (0 falsos positivos): "
          f"{len(sin_error) - fp}/{len(sin_error)}")
    print(f"  adversariales en los que se abstuvo             : "
          f"{se_abstuvo}/{len(adversarial)}")
    print(f"  cobertura (casos que puede evaluar)             : "
          f"{sum(f['evaluable'] for f in filas)}/{len(filas)}")
    print("-" * 70)
    print(f"  TOTAL sobre los evaluables: {aciertos}/{sum(f['evaluable'] for f in filas)}")
    print("=" * 70)

    # El criterio: detectar los pasos erróneos evaluables sin marcar los `sin error`.
    criterio = sum(f["acierto"] for f in con_paso) == len(con_paso) and fp == 0
    print(f"\n¿Cumple el criterio? {'SÍ' if criterio else 'NO'}"
          "  (detectar los pasos erróneos evaluables, 0 falsos positivos)")
    if criterio:
        print("  Lo cumple EN DESARROLLO, sobre 6 casos evaluables. No dice nada")
        print("  sobre los 12 casos de evaluación, que no se han tocado.")
    else:
        print("  Queda como experimento: no se conecta al pipeline.")

    # Los adversariales son un criterio aparte y más exigente.
    no_abstuvo = [f["id"] for f in adversarial if f["tipo"] != "no_evaluable"]
    if no_abstuvo:
        print(f"\nNo se abstuvo en {no_abstuvo}. En `dg-f03` la hoja SÍ trae un")
        print("procedimiento correcto —lo que falta es un dato que la pregunta pide y")
        print("la hoja no tiene—, así que decir que las cuentas cuadran es cierto y no")
        print("es un falso positivo. Pero el verificador no puede resolver ese caso:")
        print("queda fuera de su alcance, y el veredicto lo sigue dando el tutor.")
    print("\nEsto mide el verificador, NO al tutor: el tutor sigue teniendo que")
    print("emitir la conclusión. No sustituye su resultado ni es una mejora de M4.")

    SALIDA.parent.mkdir(exist_ok=True)
    import csv
    with SALIDA.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\nGuardado: {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
