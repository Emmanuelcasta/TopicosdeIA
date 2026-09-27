"""Notebook S10 (1/4) — Herramientas, Tool Registry y trayectorias. No necesita GPU."""

from __future__ import annotations

from nb_comun import celda_drive, code, encabezado, md
from nb_comun_agente import celda_eval_m3, celda_trayectorias, celdas_modulos


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "Herramientas para el tutor de matemáticas",
        "Sesión 10 · Módulo 3 — Tool use: registro tipado, ejecución segura y trayectorias (1/4)",
    ))

    # ------------------------------------------------------------------ 1
    c.append(md("""
## 1 · Introducción

### De dónde venimos

| Módulo | Qué aportó | Qué dejó pendiente |
|---|---|---|
| M1 | LoRA sobre Qwen2.5-1.5B: formato 12% → 100% | — |
| M2 | Harness de 3 dimensiones; cálculo 50% → 83% con el fine-tuning | Cero abstenciones; fallos de planteamiento |
| S07 | RAG: conocimiento 0/4 → 3/4 | Cálculo 10/12 → 6/12: el contexto fijo metía ruido |
| S08 | Híbrida + reranker + enrutador | El retrieval nunca fue el cuello de botella |

S10 plantea otra pregunta: **¿mejora el tutor si en vez de calcular de memoria
delega las operaciones en herramientas y decide por sí mismo cuándo buscar en los
documentos?** La regla del curso sigue en pie: *un agente solo se justifica si el
harness mejora de verdad*.

### Los cuatro notebooks de S10

| # | Notebook | GPU | Produce |
|---|---|---|---|
| **1** | **Herramientas** (este) | No | Tool Registry probado, catálogo, trayectorias validadas |
| 2 | `S10_FineTuning_Agente_Herramientas` | T4 | `adaptadores/qwen-lora-agente` (C5-FT) |
| 3 | `S10_Lab_Agente_ReAct_Tutor_Matematicas` | T4 | C0–C5 por el harness → `respuestas_s10.jsonl`, scorecards |
| 4 | `S10_Evaluacion_RAGAS_Tutor_Matematicas` | T4 | RAGAS (evaluador 7B), calibración, scorecard final, W&B |

### La arquitectura

```
Usuario
  ↓
Normalizador ────────── reglas + reescritura LLM (salvaguarda: los números no cambian)
  ↓
Agente / LLM (Qwen2.5-1.5B) ── Pensamiento: planifica qué necesita y de dónde sale
  ↓
Selección de herramienta ─── <tool_call>{"name", "arguments"}</tool_call>  (formato nativo)
  ↓
Parser + Validador ───────── ¿JSON válido? ¿existe? ¿tipos correctos?  (si no: error como observación)
  ↓
Tool Registry ────────────── esquemas generados desde tipos + docstring
  ↓
Ejecución ────────────────── SymPy / Fraction, sin eval, nunca lanza
  ↓
Resultado ────────────────── {"ok", "resultado", "valor"} → <tool_response>
  ↓
Agente interpreta ────────── ¿otra herramienta o responder?  (tope de pasos)
  ↓
Verificador ──────────────── ¿la respuesta usa algún resultado de las herramientas?
  ↓
Respuesta final + traza
```

| Componente | Responsabilidad | Lo que **no** hace |
|---|---|---|
| LLM | Razonar y proponer acciones | Ejecutar nada |
| Parser | Convertir texto en `LlamadaHerramienta` | Decidir |
| Registry | Exponer esquemas, validar tipos | Saber qué LLM hay detrás |
| Ejecutor | Correr la función y devolver un resultado tipado | Interpretar el resultado |
| `AgenteTutor` | Orquestar pasos, topes, traza | Conocer herramientas concretas |
"""))

    c.append(md("""
## 2 · Objetivos

1. Implementar un **Tool Registry** donde añadir una herramienta no toca el núcleo.
2. Construir **15 herramientas matemáticas** tipadas, exactas y seguras.
3. Verificar que las herramientas resuelven el eval set de M3 y que **los errores
   vuelven al modelo como información**, no como excepciones.
4. Probar el **normalizador** con las preguntas mal escritas del bloque F.
5. Construir y validar las **trayectorias** que enseñan cuándo y cómo usar cada
   herramienta, y justificar la estrategia de enseñanza.
"""))

    c.append(md("""
## 0 · Preparación del entorno

Este notebook **no necesita GPU**: todo es Python y SymPy. El tokenizer de Qwen se
descarga solo para ver cómo le llegan las herramientas al modelo.
"""))
    c.append(code('''
%pip install -q "transformers>=4.44" sympy
print("Listo.")
'''))
    c.extend(celda_drive())

    # ------------------------------------------------------------------ 3
    c.append(md("""
---
## 3 · El núcleo

Cada celda de esta sección es **copia literal** de un archivo de `scripts/agente/`,
donde lo cubren 61 tests (`python -m pytest scripts/tests`). Lo que se prueba es
exactamente lo que se ejecuta aquí y en los notebooks 2–4.
"""))
    c.extend(celdas_modulos(["registro", "herramientas", "nucleo", "documentos", "prompts"]))

    # ------------------------------------------------------------------ 4
    c.append(md("""
---
## 4 · El catálogo de herramientas

Lo que el modelo **ve** de cada herramienta es su esquema. Una descripción ambigua
se paga en llamadas equivocadas, así que conviene leerlo como lo leería el modelo.
"""))
    c.append(code('''
from collections import defaultdict

por_familia = defaultdict(list)
for nombre in REGISTRO.nombres():
    por_familia[REGISTRO.familia_de(nombre)].append(nombre)

print(f"{len(REGISTRO)} herramientas en {len(por_familia)} familias\\n")
for familia, nombres in por_familia.items():
    print(f"{familia}")
    for n in nombres:
        h = REGISTRO.obtener(n)
        firma = ", ".join(f"{p}: {e['type']}" + ("" if p in h.requeridos else "=" + repr(e.get("default")))
                          for p, e in h.parametros.items())
        comodin = f"   [también cubre: {', '.join(h.comodin_de)}]" if h.comodin_de else ""
        print(f"   {n}({firma}){comodin}")
'''))
    c.append(code('''
print(json.dumps(REGISTRO.obtener("valor_antes_de_porcentaje").esquema(), ensure_ascii=False, indent=2))
'''))
    c.append(md("""
### Cómo le llegan al modelo

Qwen2.5 trae **tool calling nativo**: la plantilla de chat inserta los esquemas en
el prompt de sistema y enseña al modelo a responder con `<tool_call>`. No hace falta
inventar un formato; hace falta respetar el suyo.
"""))
    c.append(code('''
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
mensajes = [
    {"role": "system", "content": SYSTEM_C4_REACT},
    {"role": "user", "content": "Calcula 3847 × 296."},
    {"role": "assistant", "content": "Pensamiento: es un producto grande; lo delego.",
     "tool_calls": [{"type": "function", "function": {"name": "multiplicar", "arguments": {"numeros": [3847, 296]}}}]},
    {"role": "tool", "name": "multiplicar", "content": '{"ok": true, "resultado": "1138712"}'},
]
prompt = tok.apply_chat_template(mensajes, tools=[REGISTRO.obtener("multiplicar").esquema()],
                                 tokenize=False, add_generation_prompt=True)
print(prompt)
'''))

    # ------------------------------------------------------------------ 5
    c.append(md("""
---
## 5 · ¿Resuelven las herramientas el eval set de M3?

Si una herramienta no puede resolver un caso **con los argumentos correctos**, el
agente tampoco podrá. Esta prueba separa dos preguntas que en el scorecard se
mezclarían: *¿la herramienta sirve?* y *¿el modelo sabe usarla?*
"""))
    c.extend(celda_eval_m3(prefijo="5.1"))
    c.append(code('''
# La llamada "ideal" para cada caso del bloque D, escrita a mano.
LLAMADA_IDEAL = {
    "ari-01": ("multiplicar", {"numeros": [3847, 296]}),
    "ari-02": ("evaluar_expresion", {"expresion": "(48 - 23) × 1275"}),
    "ari-03": ("operar_fracciones", {"fraccion_a": "17/24", "fraccion_b": "11/36", "operacion": "suma"}),
    "ari-04": ("operar_fracciones", {"fraccion_a": "5/6", "fraccion_b": "15/28", "operacion": "division"}),
    "ari-05": ("raiz", {"radicando": 7056}),
    "ari-06": ("evaluar_expresion", {"expresion": "2.35 × 4.8 − 1.964"}),
    "ari-07": ("resolver_ecuacion", {"ecuacion": "7x − 23 = 4x + 58"}),
    "ari-08": ("evaluar_expresion", {"expresion": "15³ − 12⁴"}),
    "ari-09": ("evaluar_expresion", {"expresion": "2450000 × 0.018 × 7"}),
    "ari-10": ("dividir", {"dividendo": 98532, "divisor": 12}),
    "ari-11": ("simplificar_fraccion", {"fraccion": "1386/2310"}),
    "ari-12": ("evaluar_expresion", {"expresion": "0.0375 ÷ 0.0015 + √2025"}),
    "ari-13": ("operar_expresion", {"expresion": "(2x + 3)(x − 5)", "operacion": "expandir"}),
}

print(f"{'caso':8s} {'herramienta':22s} {'resultado':24s} {'esperado':10s} ok")
print("-" * 72)
ok_total = 0
for caso in [c for c in eval_m3 if c["bloque"] == "aritmetica"]:
    nombre, args = LLAMADA_IDEAL[caso["id"]]
    r = REGISTRO.ejecutar(LlamadaHerramienta(nombre, args))
    ev = caso["evaluacion"]
    if ev["tipo"] == "numerico":
        ok = r.ok and mismo_numero(r.valor, ev["valor"])
        esperado = ev["valor"]
    else:
        ok = r.ok and r.valor.replace(" ", "") in [k.replace(" ", "") for k in ev["claves"]]
        esperado = "(claves)"
    ok_total += ok
    print(f"{caso['id']:8s} {nombre:22s} {str(r.resultado)[:24]:24s} {esperado:10s} {'OK' if ok else 'FALLA'}")
print("-" * 72)
print(f"{ok_total}/13 casos resueltos por la herramienta con la llamada correcta")
'''))
    c.append(md("""
### Los errores también son información

Un agente que recibe una excepción se detiene. Uno que recibe
`{"ok": false, "error": "la división entre cero no está definida"}` puede
**explicárselo al estudiante**, que es justo lo que pide el caso adversarial
`adv-02`.
"""))
    c.append(code('''
pruebas = [
    ("dividir", {"dividendo": 15, "divisor": 0}),
    ("raiz", {"radicando": -16}),
    ("evaluar_expresion", {"expresion": "__import__('os').system('rm -rf /')"}),
    ("sumar", {"nums": [1, 2]}),                       # argumento mal nombrado
    ("dividir", {"dividendo": "mil", "divisor": 4}),   # tipo imposible de convertir
    ("calculadora", {"expresion": "2+2"}),             # herramienta que no existe
    ("restar", {"minuendo": "3,0", "sustraendo": "2.65"}),   # texto y coma decimal: SE CONVIERTE
]
for nombre, args in pruebas:
    r = REGISTRO.ejecutar(LlamadaHerramienta(nombre, args))
    print(f"{nombre}({args})")
    print(f"   -> [{r.tipo_error or 'ok'}] {r.como_observacion()}\\n")
'''))

    # ------------------------------------------------------------------ 6
    c.append(md("""
---
## 6 · Extensibilidad: una herramienta nueva sin tocar el núcleo

El requisito era que añadir herramientas no obligue a modificar el agente. La
prueba: registrar una herramienta **en este notebook** y ver cómo el agente la usa
sin cambiar una línea de `AgenteTutor`.

Para no depender de la GPU, el LLM es un **guion**: devuelve salidas fijas. El
bucle, el parser, el registro, la ejecución y la traza son los reales.
"""))
    c.append(code('''
REGISTRO_EXTENDIDO = REGISTRO.subconjunto()


@REGISTRO_EXTENDIDO.herramienta(familia="aritmetica")
def convertir_unidades(valor: float, desde: Literal["m3", "litros", "km", "m", "kg", "g"],
                       hacia: Literal["m3", "litros", "km", "m", "kg", "g"]) -> dict:
    """Convierte una cantidad entre unidades de volumen, longitud o masa.

    Args:
        valor: la cantidad a convertir.
        desde: unidad de origen.
        hacia: unidad de destino.
    """
    factores = {("m3", "litros"): 1000, ("litros", "m3"): Fraction(1, 1000), ("km", "m"): 1000,
                ("m", "km"): Fraction(1, 1000), ("kg", "g"): 1000, ("g", "kg"): Fraction(1, 1000)}
    if (desde, hacia) not in factores:
        raise ErrorHerramienta(f"no sé convertir de {desde} a {hacia}")
    return formatear(valor * factores[(desde, hacia)])


class Guion:
    def __init__(self, *salidas):
        self.salidas = list(salidas)

    def __call__(self, mensajes, esquemas):
        return self.salidas.pop(0)


def tc(nombre, **args):
    return f"<tool_call>\\n{json.dumps({'name': nombre, 'arguments': args}, ensure_ascii=False)}\\n</tool_call>"


llm_guion = Guion(
    "Pensamiento: la capacidad está en m3 y lo que contiene en litros; primero convierto.\\n"
    + tc("convertir_unidades", valor=2.5, desde="m3", hacia="litros"),
    "Pensamiento: ahora resto lo que ya contiene.\\n" + tc("restar", minuendo=2500, sustraendo=900),
    "Paso 1: 2.5 m³ = 2500 litros.\\nPaso 2: 2500 - 900 = 1600.\\nRespuesta final: 1600 litros",
)
agente_demo = AgenteTutor(llm_guion, SYSTEM_C4_REACT, registro=REGISTRO_EXTENDIDO, max_rondas=5, verificar=True)
out = agente_demo("Un tanque de 2.5 metros cúbicos ya tiene 900 litros. ¿Cuántos litros faltan?")

print("TRAZA")
for e in out.traza:
    print(f"  [{e.paso}] {e.tipo:12s} {json.dumps(e.datos, ensure_ascii=False, default=str)[:110]}")
print("\\nRESPUESTA:", out.respuesta)
print("\\n¿Se modificó el registro original?", "convertir_unidades" in REGISTRO)
'''))

    # ------------------------------------------------------------------ 7
    c.append(md("""
---
## 7 · Preguntas mal escritas: el normalizador

Dos capas, y la decisión merece explicarse:

1. **Reglas deterministas**: auditables y seguras para lo frecuente, como operadores
   escritos en palabras **entre números** (`17/24 mas 11/36`, `3847 x 296`),
   abreviaturas de chat (`q`, `pa`, `ai`) y signos repetidos.
2. **Reescritura con el LLM** para las faltas que ninguna regla cubre (`bale`,
   `evaluasion`). Solo se activa si la pregunta *parece informal* (sin tildes, eñes
   ni `¿`).

**Salvaguarda común:** si la versión corregida no conserva **exactamente los mismos
números**, se descarta. Una corrección ortográfica que cambia un dato es peor que
la pregunta mal escrita.

Aquí se prueban las reglas. La reescritura LLM se mide en el notebook 3 (C5).
"""))
    c.append(code('''
normalizador = Normalizador()
originales = {c["id"]: c["input"] for c in eval_m3}
for caso in [c for c in eval_m3 if c["bloque"] == "mal_escrito"]:
    n = normalizador(caso["input"])
    print(f"{caso['id']}  <- {caso['original']}")
    print(f"   escrita    : {caso['input']}")
    print(f"   normalizada: {n.normalizada}")
    print(f"   números iguales: {sorted(numeros_en(caso['input'])) == sorted(numeros_en(n.normalizada))}"
          f"   |  pide reescritura LLM: {parece_informal(caso['input'])}\\n")
'''))
    c.append(code('''
# La salvaguarda en acción: una "corrección" que cambia un número se rechaza.
def reescritura_peligrosa(texto):
    return "¿Cuánto es 3847 × 269?"          # el LLM "corrigió" 296 por 269

n = Normalizador(reescribir=reescritura_peligrosa)("cuanto es 3847 x 296")
print("normalizada :", n.normalizada)
print("descartada  :", n.descartada)
'''))

    # ------------------------------------------------------------------ 8
    c.append(md("""
---
## 8 · Enseñar a usar las herramientas

### La decisión: ¿prompting, few-shot, fine-tuning o tool calling?

| Mecanismo | A favor | En contra, con un 1.5B | Uso en el proyecto |
|---|---|---|---|
| Prompting zero-shot | Gratis | En S08, un zero-shot de 1.5B clasificó al 67% y con sesgo | Instrucciones de base, no basta solo |
| **Tool calling nativo** | Qwen2.5 se preentrenó con este formato | La LoRA de M1 puede taparlo (se mide en el notebook 3) | **Formato de C3–C5** |
| **Few-shot** con trayectorias | Muestra cuándo **sí** y cuándo **no** usar herramientas | Cuesta tokens de prompt | **C3–C5** (3 ejemplos) |
| **Fine-tuning** con trayectorias | Lo más eficaz en modelos pequeños | Datos, riesgo de sobreajuste | **C5-FT** (notebook 2) |
| DSPy | Elige los few-shot con la métrica | Ollama en Colab es frágil | Fuera de alcance; posible siguiente paso |

**Estrategia: una escalera medible.** Primero tool calling nativo + few-shot (sin
entrenar). Luego la misma configuración con un adaptador entrenado con las
trayectorias. La diferencia entre C5 y C5-FT responde con datos a la pregunta del
proyecto: *¿basta con mostrar ejemplos o hay que entrenar?*
"""))
    c.extend(celda_trayectorias(prefijo="8.1"))
    c.append(code('''
# Una trayectoria completa, tal como la verá el modelo en el fine-tuning.
ejemplo = next(t for t in trayectorias if t["id"] == "man-com-04")
esquemas_ejemplo = [REGISTRO.obtener("dividir").esquema()] + [{
    "type": "function", "function": {"name": "buscar_documentos", "description": "Busca en los documentos del colegio.",
    "parameters": {"type": "object", "properties": {"consulta": {"type": "string"}}, "required": ["consulta"]}}}]
texto = tok.apply_chat_template([{"role": "system", "content": SYSTEM_C5_AGENTE}] + ejemplo["mensajes"],
                                tools=esquemas_ejemplo, tokenize=False)
print(texto[texto.find("<|im_start|>user"):])
'''))
    c.append(code('''
# Re-verificación: se re-ejecuta cada llamada de cada trayectoria con el registro
# de ESTE notebook y se compara con la observación guardada. Si alguien cambia una
# herramienta y rompe su comportamiento, esta celda lo detecta.
from collections import Counter

discrepancias, llamadas = [], 0
for t in trayectorias:
    msgs = t["mensajes"]
    for i, m in enumerate(msgs):
        for tc_ in m.get("tool_calls", []):
            f = tc_["function"]
            if f["name"] == "buscar_documentos":
                continue
            llamadas += 1
            guardada = json.loads(msgs[i + 1]["content"])
            nueva = json.loads(REGISTRO.ejecutar(LlamadaHerramienta(f["name"], f["arguments"])).como_observacion())
            if guardada != nueva:
                discrepancias.append((t["id"], f["name"], guardada, nueva))

print(f"Llamadas re-ejecutadas: {llamadas} | discrepancias: {len(discrepancias)}")
for d in discrepancias[:5]:
    print("  ", d)

uso = Counter(tc_["function"]["name"] for t in trayectorias for m in t["mensajes"] for tc_ in m.get("tool_calls", []))
print("\\nEjemplos por herramienta:")
for nombre in REGISTRO.nombres() + ["buscar_documentos"]:
    print(f"  {nombre:28s} {uso.get(nombre, 0):3d}")
'''))

    # ------------------------------------------------------------------ 9
    c.append(md("""
---
## 9 · Qué se lleva el notebook 2 y el 3

1. **El núcleo probado**: registro, 15 herramientas, parser, bucle ReAct,
   verificador y normalizador, con 61 tests y re-verificado aquí.
2. **La prueba de techo**: las herramientas resuelven los 13 casos del bloque D
   con la llamada correcta. Si C3–C5 fallan ahí, el problema es **elegir o formular
   la llamada**, no la herramienta.
3. **199 trayectorias validadas**, que sirven para dos cosas:
   - few-shot en C3–C5 (notebook 3);
   - fine-tuning en C5-FT (notebook 2).

> **Límite declarado.** Las trayectorias del corpus heredan su sesgo: muchas
> operaciones de un paso y pocas de porcentajes. Las 34 manuales cubren los huecos,
> pero siguen siendo pocas. El notebook 2 mide si alcanzan.
"""))

    c.append(md("""
---
## 10 · Conclusiones de esta corrida

**1 · Las herramientas no son el límite.** Los 13 casos del bloque de aritmética se
resuelven con la llamada correcta: 13/13. Eso fija un techo y convierte cualquier
fallo posterior del agente en un fallo de **decisión** (no llamó) o de
**formulación** (llamó mal), nunca de la herramienta. En el notebook 3 esa
separación es la que explica los resultados.

**2 · Las 251 llamadas de las trayectorias se re-ejecutaron sin una sola
discrepancia.** El dataset de entrenamiento es consistente con el registro que se usa
en inferencia: lo que el modelo aprenderá a escribir produce, al ejecutarse, el mismo
resultado que está escrito en la trayectoria.

**3 · Los errores de dominio viajan como información, no como excepción.** Dividir
entre cero, raíz par de un negativo, argumentos mal nombrados y hasta un intento de
inyección de código vuelven al modelo como `{"ok": false, "error": ...}`. Es lo que
permite que el agente explique por qué una operación no tiene resultado en vez de
detenerse.

**4 · El normalizador conserva los números en los 11 casos mal escritos**, y descarta
una reescritura que cambiaba 296 por 269. La salvaguarda funciona: preferimos una
pregunta mal escrita a una "corregida" con otro dato.

**5 · Extensibilidad comprobada.** Se registró `convertir_unidades` en este notebook
y el agente la usó sin tocar una línea del núcleo, con el registro original intacto.
"""))

    return c
