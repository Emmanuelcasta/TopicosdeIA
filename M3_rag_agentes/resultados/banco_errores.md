# Banco de errores

179 fallos catalogados sobre 7 configuraciones, generados desde las trazas de S10 con `scripts/banco_errores.py`.

Cada entrada trae pregunta, respuesta del modelo, respuesta esperada, tipo de error,
causa probable y el cambio que lo ataca. El campo `detectado_por_verificador` dice si
el verificador de planteamiento lo habría señalado: ese es el conjunto sobre el que
debería notarse ese cambio concreto.

## Por tipo de error

| Tipo | n | Cambio propuesto |
|---|---|---|
| aritmética de cabeza | 104 | adaptador entrenado con trayectorias de herramientas (ya existe: adaptadores/qwen-lora-agente) |
| no consultó documentos | 32 | enrutador de S08 integrado al agente (scripts/agente/enrutador.py): decide buscar antes de responder |
| leyó el documento y no lo usó | 17 | prompt de cita textual en la vía documental; si no basta, el límite es el tamaño del modelo |
| planteamiento | 14 | verificador de planteamiento (scripts/agente/verificador.py): revisa que los argumentos vengan del enunciado y que no quede ningún dato sin usar |
| robustez | 12 | ejemplos de abstención en el corpus y válvula de escape explícita en el prompt |

## Por sistema

| Sistema | fallos | detectados por el verificador |
|---|---|---|
| C0 · base | 35 | 0 |
| C1 · +LoRA | 26 | 0 |
| C2 · +RAG | 28 | 0 |
| C3 · +herram. | 22 | 0 |
| C4 · +ReAct | 25 | 0 |
| C5 · agente | 23 | 0 |
| C5-FT · entrenado | 20 | 13 |

## El sistema que se va a mejorar · C5-FT

| caso | tipo de error | herramientas | ¿lo ve el verificador? |
|---|---|---|---|
| `con-04` | leyó el documento y no lo usó | buscar_documentos | no |
| `com-02` | no consultó documentos | multiplicar|evaluar_expresion|suma | sí |
| `com-03` | no consultó documentos | evaluar_expresion|evaluar_expresio | no |
| `com-04` | no consultó documentos | multiplicar|evaluar_expresion | sí |
| `com-05` | no consultó documentos | evaluar_expresion | sí |
| `com-07` | no consultó documentos | multiplicar | sí |
| `dif-01` | planteamiento | multiplicar|restar | no |
| `dif-03` | planteamiento | multiplicar|evaluar_expresion | sí |
| `dif-06` | planteamiento | multiplicar|sumar | sí |
| `dif-08` | planteamiento | evaluar_expresion | no |
| `dif-10` | planteamiento | restar|restar | sí |
| `dif-11` | planteamiento | multiplicar|sumar | no |
| `ari-02` | planteamiento | multiplicar|restar | sí |
| `ari-09` | planteamiento | multiplicar | no |
| `ari-12` | planteamiento | evaluar_expresion|raiz | sí |
| `mal-02` | planteamiento | dividir|dividir | no |
| `mal-03` | planteamiento | evaluar_expresion|multiplicar | sí |
| `mal-04` | planteamiento | evaluar_expresion|multiplicar | sí |
| `mal-07` | planteamiento | evaluar_expresion | sí |
| `adv-01` | robustez | potencia | sí |
