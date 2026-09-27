"""
Celdas compartidas por los notebooks de S10 (arquitectura agéntica).

El código del agente vive en `scripts/agente/*.py`, donde se prueba con
`python -m pytest scripts/tests`. Los notebooks NO lo reescriben: estas funciones
leen esos archivos y los pegan en celdas, quitando solo las importaciones entre
módulos (marcadas con `# [local]`), que en un notebook sobran porque todo vive
en el mismo espacio de nombres. Lo que se prueba es exactamente lo que se ejecuta.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import textwrap

from nb_comun import RAIZ, code, md

DIR_AGENTE = RAIZ / "scripts" / "agente"

# Orden de carga: cada módulo solo usa nombres de los anteriores.
MODULOS = ["registro", "herramientas", "nucleo", "documentos", "prompts", "metricas", "ragas_tutor"]

DESCRIPCION_MODULOS = {
    "registro": """
#### Tool Registry · `registro.py`

El esquema de cada herramienta se **deriva** de su firma y su docstring, en el
formato de tool calling nativo de Qwen2.5. La descripción que ve el modelo y el
código que se ejecuta no pueden divergir. `validar()` convierte lo que propone el
modelo a los tipos reales (acepta `"2,35"`, `"3/4"`, números como texto) y
`ejecutar()` **nunca lanza**: todo error vuelve al modelo como observación.
""",
    "herramientas": """
#### Herramientas matemáticas · `herramientas.py`

Quince herramientas tipadas en seis familias, todas con aritmética **exacta**
(`Fraction` y SymPy racional). Las expresiones se interpretan con SymPy sobre una
lista blanca de símbolos: nunca `eval` de Python. Para añadir una herramienta basta
con una función con tipos y docstring y el decorador `@REGISTRO.herramienta`.
""",
    "nucleo": """
#### Núcleo del agente · `nucleo.py`

- **Normalizador**: reglas deterministas + reescritura LLM opcional, con la
  salvaguarda de que los números no cambien.
- **Parser**: `<tool_call>` nativo, JSON suelto recuperado o error de formato
  devuelto al modelo.
- **Bucle ReAct** (`AgenteTutor`): una sola clase configura C2–C5 con
  `contexto_fijo`, `max_rondas`, `verificar`, `normalizador` y `few_shot`.
- **Verificador**: si la respuesta no usa ningún resultado de las herramientas,
  se pide una revisión.
""",
    "documentos": """
#### Retrieval como herramienta · `documentos.py`

El retriever de S08 (híbrida + reranker) envuelto como `buscar_documentos`, con
el mismo formato `[Fuente: ...]` de S07.
""",
    "prompts": """
#### Prompts de sistema · `prompts.py`

Versionados junto al núcleo: las trayectorias de entrenamiento usan exactamente
los mismos. Cada configuración añade una sola idea sobre la anterior.
""",
    "metricas": """
#### Métricas del agente · `metricas.py`

Uso correcto, selección, fidelidad a la herramienta, proceso, robustez ante
preguntas mal escritas, autoconsistencia y McNemar.
""",
    "ragas_tutor": """
#### RAGAS · `ragas_tutor.py`

Las cuatro métricas de S10 con un evaluador LLM, y una versión objetiva de cada
una para calibrarlo.
""",
}


def fuente_modulo(nombre: str) -> str:
    texto = (DIR_AGENTE / f"{nombre}.py").read_text(encoding="utf-8")
    return "\n".join(l for l in texto.splitlines() if "# [local]" not in l)


def celdas_modulos(modulos: list[str] | None = None, con_descripcion: bool = True) -> list[dict]:
    celdas = []
    for m in modulos or MODULOS:
        if con_descripcion:
            celdas.append(md(DESCRIPCION_MODULOS[m]))
        celdas.append(code(fuente_modulo(m)))
    return celdas


# --------------------------------------------------------------------------
# Archivos de datos embebidos (mismo patrón que el corpus y el eval set de M2)
# --------------------------------------------------------------------------

def _codigo_blob(ruta_relativa: str, variable: str) -> str:
    ruta = RAIZ / ruta_relativa
    crudo = ruta.read_bytes()
    sha = hashlib.sha256(crudo).hexdigest()
    b64 = base64.b64encode(gzip.compress(crudo, mtime=0)).decode()
    lineas = "\n".join(f'    "{l}"' for l in textwrap.wrap(b64, 96))
    blob = f"_BLOB_{variable.upper()}"
    return f'''
_RUTA_{variable.upper()} = Path("{ruta_relativa}")
_SHA_{variable.upper()} = "{sha}"
{blob} = (
{lineas}
)
if not _RUTA_{variable.upper()}.exists():
    _RUTA_{variable.upper()}.parent.mkdir(parents=True, exist_ok=True)
    _RUTA_{variable.upper()}.write_bytes(gzip.decompress(base64.b64decode({blob})))
    print(f"Escrito {{_RUTA_{variable.upper()}}} (copia embebida).")
{variable} = [json.loads(l) for l in _RUTA_{variable.upper()}.read_text(encoding="utf-8").splitlines()]
_sha = hashlib.sha256(_RUTA_{variable.upper()}.read_bytes()).hexdigest()
print(f"{ruta_relativa}: {{len({variable})}} registros | SHA coincide: {{_sha == _SHA_{variable.upper()}}}")
'''


def celda_eval_m3(prefijo: str = "3.2") -> list[dict]:
    return [
        md(f"""
### {prefijo} · El eval set de M3: lo que el de M2 no puede medir

`eval_set_m2.jsonl` **no se toca**: es la vara que compara M1, M2, S07, S08 y S10.
Se añaden tres bloques que se reportan **aparte**, en `eval_set_m3_agente.jsonl`
(generado y validado por `scripts/eval_set_m3_fuente.py`):

| Bloque | n | Por qué hace falta |
|---|---|---|
| **D · aritmética** | 13 | El cálculo de M2 está en 10/12 y sus fallos son de *planteamiento*. Aquí hay cuentas donde un 1.5B se equivoca **calculando**: el terreno de las herramientas. |
| **E · compuesto** | 8 | Buscar un dato del colegio **y** calcular con él. Es donde S10 dice que un agente se justifica. |
| **F · mal escrito** | 11 | Versiones con faltas, sin tildes y abreviaturas de casos existentes. Cada una apunta a su `original`: mide **robustez** y **consistencia**. |

Cada caso trae además las **familias de herramientas** que necesita. Las de M2
están en `herramientas_esperadas_m2.jsonl`, un archivo aparte para no modificar
aquel eval set.
"""),
        code("import base64, gzip, hashlib, json\nfrom pathlib import Path\n"
             + _codigo_blob("data/eval_set_m3_agente.jsonl", "eval_m3")
             + _codigo_blob("data/herramientas_esperadas_m2.jsonl", "anotacion_m2")
             + '''
from collections import Counter
HERRAMIENTAS_ESPERADAS = {a["id"]: a["herramientas"] for a in anotacion_m2}
HERRAMIENTAS_ESPERADAS.update({c["id"]: c["herramientas"] for c in eval_m3})
print("Bloques M3:", dict(Counter(c["bloque"] for c in eval_m3)))
'''),
    ]


def celda_trayectorias(prefijo: str) -> list[dict]:
    return [
        md(f"""
### {prefijo} · Las trayectorias

`data/trayectorias_herramientas.jsonl`, generado por `scripts/trayectorias_fuente.py`:

- **165 derivadas del corpus de M1.** Cada paso con una operación escrita se
  convierte en una llamada; la llamada **se ejecuta** y solo se conserva si
  reproduce el resultado del corpus. Mantienen la partición train/validación.
- **34 escritas a mano** para lo que el corpus no enseña: buscar documentos,
  casos compuestos, **no** usar herramientas, errores de dominio, álgebra y
  preguntas mal escritas.

Formato: `problema → Pensamiento (decisión) → <tool_call> → <tool_response> → …
→ respuesta final`. Ninguna pregunta coincide ni se parece a un caso de los eval
sets (el generador falla si detecta contaminación).
"""),
        code("import base64, gzip, hashlib, json\nfrom pathlib import Path\n"
             + _codigo_blob("data/trayectorias_herramientas.jsonl", "trayectorias")
             + '''
from collections import Counter
print("Por split:", dict(Counter(t["split"] for t in trayectorias)))
print("Por tipo :", dict(Counter(t["tipo"] for t in trayectorias)))
'''),
    ]


# --------------------------------------------------------------------------
# Generador Qwen con varios adaptadores, caché y contador de tokens
# --------------------------------------------------------------------------

CODIGO_LLM = r'''
import json, time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO_BASE    = "Qwen/Qwen2.5-1.5B-Instruct"
DIR_LORA_M1    = "adaptadores/qwen-lora"          # M1: formato "Paso N / Respuesta final"
DIR_LORA_AGENT = "adaptadores/qwen-lora-agente"   # S10: entrenado con trayectorias (opcional)

if not Path(DIR_LORA_M1).exists():
    raise FileNotFoundError(f"No se encuentra {DIR_LORA_M1}. Ejecuten S04_Lab_Fine_tuning_Qwen.ipynb "
                            "o monten el Drive donde quedó guardado.")

tok = AutoTokenizer.from_pretrained(MODELO_BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_base = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float16).to(DEVICE)
modelo = PeftModel.from_pretrained(_base, DIR_LORA_M1, adapter_name="m1").eval()
ADAPTADORES = ["m1"]
if Path(DIR_LORA_AGENT).exists():
    modelo.load_adapter(DIR_LORA_AGENT, adapter_name="agente")
    ADAPTADORES.append("agente")
print("Adaptadores cargados:", ADAPTADORES, "(+ None = modelo base)")


class ContadorTokens:
    """Acumula tokens y llamadas al LLM de la pregunta en curso."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.llamadas, self.entrada, self.salida = 0, 0, 0


CONTADOR = ContadorTokens()
_CACHE_LLM = {}


@torch.no_grad()
def generar_chat(mensajes, esquemas=None, adaptador="m1", max_new_tokens=220, muestreo=None):
    """Una generación. `adaptador`: "m1", "agente" o None (modelo base).
    `muestreo`: None (greedy) o (temperatura, semilla) para autoconsistencia."""
    clave = (json.dumps(mensajes, ensure_ascii=False, sort_keys=True, default=str),
             json.dumps(esquemas, sort_keys=True) if esquemas else None, adaptador, max_new_tokens, muestreo)
    prompt = tok.apply_chat_template(mensajes, tools=esquemas, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(modelo.device)
    CONTADOR.llamadas += 1
    CONTADOR.entrada += ids["input_ids"].shape[1]
    if clave in _CACHE_LLM:
        texto, n_out = _CACHE_LLM[clave]
        CONTADOR.salida += n_out
        return texto
    if adaptador is None:
        contexto = modelo.disable_adapter()
    else:
        modelo.set_adapter(adaptador)
        contexto = nullcontext()
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if muestreo is None:
        kwargs["do_sample"] = False
    else:
        temperatura, semilla = muestreo
        torch.manual_seed(semilla)
        kwargs.update(do_sample=True, temperature=temperatura, top_p=0.95)
    with contexto:
        out = modelo.generate(**ids, **kwargs)
    nuevos = out[0][ids["input_ids"].shape[1]:]
    texto = tok.decode(nuevos, skip_special_tokens=True).strip()
    _CACHE_LLM[clave] = (texto, len(nuevos))
    CONTADOR.salida += len(nuevos)
    return texto


def crear_llm(adaptador="m1", max_new_tokens=220, muestreo=None):
    """La firma que espera el núcleo del agente: llm(mensajes, esquemas) -> texto."""
    def llm(mensajes, esquemas):
        return generar_chat(mensajes, esquemas, adaptador=adaptador,
                            max_new_tokens=max_new_tokens, muestreo=muestreo)
    return llm
'''
