# Corrida 02 · 22-10-2026, tarde · **evidencia diagnóstica, no un scorecard**

`S13_ejecutado.ipynb` — 60 celdas, 26 con salidas, `execution_count` 71–105,
0 errores. GPU: `NVIDIA RTX PRO 6000 Blackwell (102 GB)`. SHA-256
`ff2f53b267d6eb29…`.

Primera corrida con la calificación por línea `Diagnóstico:`. Resultado **1/12**.
No es un scorecard de M4:

- **4 de 12 sin conclusión** y 1 con conclusión en texto libre (`vis-04`).
- **`vis-10` inacertable**: el VLM inventó `24 x 13 = 306` en lugar de declarar
  ilegible la foto borrosa, y el caso se atribuyó a `diagnostico` en vez de a
  lectura.
- **El único acierto no es evidencia**: 7 de 8 conclusiones fueron `sin error`.
- La regresión de M2 (15/21) salió con el eval set **equivocado** en Drive
  (`SHA coincide: False`), así que no era comparable con el 13/21 de M3.

Sus cinco artefactos se quedaron en Drive y no se descargaron, así que este
notebook es toda la evidencia de la corrida. El análisis caso por caso está en
el historial de la sesión y se resume en el README de la corrida 03.
