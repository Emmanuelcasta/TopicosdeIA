# Corrección del criterio de `adv-02`

Las corridas de S08 (Colab) y S10 (Lightning) usaron la versión antigua de
`data/eval_set_m2.jsonl`, sin las claves "no definido" y "no tiene sentido".
Los notebooks lo avisaron con `SHA coincide: False`.

Las respuestas no cambian: solo se vuelve a aplicar el criterio de la Dimensión 3
sobre el texto ya generado. Similitud y juez quedan iguales.

## S08: 4 caso(s) cambian

| sistema | caso | antes | después | motivo |
|---|---|---|---|---|
| sin RAG | adv-02 | FALLA | OK | contiene 'no definid' |
| A · ingenuo | adv-02 | FALLA | OK | contiene 'no definid' |
| D · +router | adv-02 | FALLA | OK | contiene 'no definid' |
| E · +cita | adv-02 | FALLA | OK | contiene 'no definid' |

## S10: 1 caso(s) cambian

| sistema | caso | antes | después | motivo |
|---|---|---|---|---|
| C1 · +LoRA | adv-02 | FALLA | OK | contiene 'no definid' |
