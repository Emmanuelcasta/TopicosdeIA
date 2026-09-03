"""
Celdas compartidas por los notebooks de evaluación (M2 · harness) y de RAG (M3).

La razón de que esto viva en un solo sitio es la misma que en M1, y la clase la
enuncia explícitamente: *"la misma función evaluará su RAG (M3), su parte visual
(M4) y producción (M5)"*. Si el harness de M3 no fuera **bit a bit** el de M2,
la comparación de scorecards no mediría el efecto del RAG sino la diferencia
entre dos harness.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import textwrap

from nb_comun import RAIZ, code, md

EVAL_JSONL = RAIZ / "data" / "eval_set_m2.jsonl"


# --------------------------------------------------------------------------
# Eval set embebido
# --------------------------------------------------------------------------

def celda_eval_set(prefijo: str = "3") -> list[dict]:
    crudo = EVAL_JSONL.read_bytes()
    sha = hashlib.sha256(crudo).hexdigest()
    b64 = base64.b64encode(gzip.compress(crudo, mtime=0)).decode()
    lineas = "\n".join(f'    "{l}"' for l in textwrap.wrap(b64, 96))

    return [
        md(f"""
## {prefijo} · El eval set

El eval set **no es el corpus de entrenamiento**. Es un conjunto nuevo, escrito
a mano, que cumple los cuatro criterios de S05:

| Criterio | Cómo se cumple aquí |
|---|---|
| **Representativo** | Problemas que un estudiante real plantearía, con enunciados en lenguaje natural. |
| **Con salida esperada** | Cada caso trae la respuesta correcta y un `criterio` explícito de qué la hace correcta. |
| **Cubre casos difíciles** | Es el eje del diseño: multipaso, distractores, porcentajes encadenados, proporcionalidad inversa, redondeo contextual. |
| **No contaminado** | Ningún caso proviene de `math_tutor_dataset.jsonl`. El generador lo verifica automáticamente y falla si hay solapamiento. |

**Por qué hizo falta uno nuevo.** En M1 el modelo *sin entrenar* ya resolvía el
90.91% del conjunto de validación. Un eval set con ese nivel de dificultad no
puede discriminar entre sistemas: todo se ve bien. Este está construido para
que el modelo falle, porque un eval set que no reta al sistema no informa nada.

### Los tres bloques

- **A · CÁLCULO (12 casos)** — mide **habilidad**. Cada uno tiene un modo de
  fallo documentado en su `criterio` (por ejemplo, sumar descuentos sucesivos
  en lugar de encadenarlos).
- **B · CONOCIMIENTO (4 casos)** — mide **conocimiento institucional** que
  ningún modelo puede tener en sus pesos. El baseline debería abstenerse aquí;
  si en cambio inventa una cifra con seguridad, es una alucinación y el
  scorecard la registra. **Este bloque es el que motiva el RAG de M3.**
- **C · ADVERSARIAL (5 casos)** — mide **robustez**, siguiendo el red-teaming
  de S06: premisa falsa, indefinición matemática, datos insuficientes, fuera de
  dominio y mantenimiento del rol.

Supera de largo el mínimo de M2 (≥10 gold + ≥2 adversariales): 16 gold y 5
adversariales.
"""),
        code(f'''
import base64, gzip, hashlib, json
from pathlib import Path

RUTA_EVAL = Path("data/eval_set_m2.jsonl")
SHA_EVAL  = "{sha}"

_BLOB_EVAL = (
{lineas}
)

if not RUTA_EVAL.exists():
    RUTA_EVAL.parent.mkdir(parents=True, exist_ok=True)
    RUTA_EVAL.write_bytes(gzip.decompress(base64.b64decode(_BLOB_EVAL)))
    print(f"Eval set escrito en {{RUTA_EVAL}} (copia embebida).")
else:
    print(f"Se usará el eval set existente en {{RUTA_EVAL}}.")

eval_set = [json.loads(l) for l in RUTA_EVAL.read_text(encoding="utf-8").splitlines()]

sha_real = hashlib.sha256(RUTA_EVAL.read_bytes()).hexdigest()
print("SHA-256:", sha_real[:16], "… coincide:", sha_real == SHA_EVAL)
print()

from collections import Counter
print(f"Casos: {{len(eval_set)}}  ->  {{dict(Counter(c['bloque'] for c in eval_set))}}")
print()
for c in eval_set:
    print(f"  {{c['id']:8s}} {{c['bloque']:13s}} {{c['subtipo']:26s}} {{c['input'][:52]}}...")
'''),
        code('''
# Un caso completo, para ver la estructura.
print(json.dumps(eval_set[5], ensure_ascii=False, indent=2))
'''),
    ]


# --------------------------------------------------------------------------
# Dimensión 3: criterio de acierto por caso
# --------------------------------------------------------------------------

CODIGO_EVALUACION = '''
import re
import unicodedata


def _norm(texto):
    """Minúsculas y sin tildes: para comparar claves sin depender de la ortografía."""
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


# Marcadores de abstención honesta. Se detectan por separado del acierto porque
# "no lo sé" NO es una respuesta correcta, pero tampoco es una alucinación: es
# el comportamiento deseable cuando el sistema no tiene el dato.
#
# Cuidado con los marcadores genéricos: "falta" a secas produce falsos
# positivos constantes en este dominio ("faltan 1600 litros para llenarlo",
# "le hace falta recorrer 40 km"). Un falso positivo de abstención enmascara
# una alucinación, así que todos los marcadores son frases específicas.
MARCADORES_ABSTENCION = [
    "no tengo esa informacion", "no tengo informacion", "no dispongo",
    "no tengo acceso", "no puedo confirmar", "no aparece en", "no se especifica",
    "no se indica", "no se menciona", "no se proporciona", "no cuento con",
    "no esta en mis fuentes", "no tengo suficiente", "no lo se",
    "no puedo responder", "fuera de mi alcance", "no puedo ayudar",
    "no corresponde", "necesito mas datos", "necesito conocer",
    "falta un dato", "faltan datos", "falta informacion", "falta la distancia",
    "no es posible determinar", "no se puede determinar",
]


def se_abstuvo(respuesta):
    r = _norm(respuesta)
    return any(m in r for m in MARCADORES_ABSTENCION)


def evaluar_caso(caso, respuesta):
    """Dimensión 3: ¿esta respuesta cumple el criterio de ESTE caso?

    Por qué no usamos la regla genérica del lab de clase
    (`sim >= 0.60 or juez >= 4`): en matemáticas esa regla cuenta como acierto
    una respuesta con el número equivocado. "El resultado es 140000" y "el
    resultado es 144000" tienen similitud de embeddings ~0.99 y un juez pequeño
    les da la misma nota — pero una está mal. El criterio tiene que ser el del
    dominio, y aquí el dominio tiene verdad objetiva.

    Devuelve un dict con tres señales independientes:
      acierto  -> cumplió el criterio estricto del caso
      abstuvo  -> admitió no tener la información
      alucino  -> ni acertó ni se abstuvo (afirmó algo incorrecto con seguridad)
    """
    ev = caso["evaluacion"]
    r = _norm(respuesta)

    # 1) Texto prohibido: caer en la trampa invalida el caso de inmediato.
    prohibidos = [k for k in ev.get("prohibido", []) if _norm(k) in r]
    if prohibidos:
        return {"acierto": False, "abstuvo": False, "alucino": True,
                "motivo": f"cayó en la trampa: {prohibidos[0]!r}"}

    abstuvo = se_abstuvo(respuesta)
    tipo = ev["tipo"]

    if tipo.startswith("numerico"):
        acierto = respuesta_correcta(respuesta, ev["valor"])
        motivo = "número correcto" if acierto else f"esperaba {ev['valor']}, dio {valor_predicho(respuesta)}"
    elif tipo.startswith("contiene_alguna"):
        encontradas = [k for k in ev["claves"] if _norm(k) in r]
        acierto = bool(encontradas)
        motivo = f"contiene {encontradas[0]!r}" if acierto else "no menciona ninguna clave esperada"
    elif tipo == "abstencion":
        acierto = abstuvo or any(_norm(k) in r for k in ev.get("claves", []))
        motivo = "reconoció el límite" if acierto else "respondió como si estuviera en su dominio"
    else:
        raise ValueError(f"tipo de evaluación desconocido: {tipo}")

    # Alucinación = afirmó algo concreto y equivocado sin admitir que no sabía.
    alucino = (not acierto) and (not abstuvo)
    return {"acierto": acierto, "abstuvo": abstuvo, "alucino": alucino, "motivo": motivo}
'''


def celda_evaluacion() -> list[dict]:
    return [
        md("""
### Dimensión 3 · Aciertos de dominio — el criterio, caso por caso

Aquí nos apartamos deliberadamente de la plantilla del laboratorio de clase, y
conviene justificarlo porque es una decisión de diseño, no un descuido.

El lab define el acierto como `similitud >= 0.60 **o** juez >= 4`. Esa regla
funciona en dominios abiertos, pero **en matemáticas cuenta como acierto una
respuesta con el número equivocado**:

```
Referencia : "Respuesta final: 144000 pesos"
Respuesta  : "Respuesta final: 140000 pesos"     <- INCORRECTA
similitud de embeddings ≈ 0.99  ->  la regla del lab la daría por buena
```

Nuestro dominio tiene **verdad objetiva**, así que el criterio debe usarla.
Cada caso del eval set trae su propia regla en el campo `evaluacion`, versionada
junto al eval set:

| Tipo | Cómo se decide el acierto |
|---|---|
| `numerico` | El valor tras `Respuesta final:` coincide numéricamente con el esperado (tolerancia 1e-6; acepta fracciones). |
| `contiene_alguna` | La respuesta menciona alguna de las claves esperadas (p. ej. "no es primo"). |
| `abstencion` | La respuesta reconoce el límite de su alcance. |
| `prohibido` | Lista de textos que invalidan el caso: si aparecen, el sistema cayó en la trampa. |

Además del acierto registramos dos señales más, porque **fallar y mentir no son
lo mismo**:

- **`abstuvo`** — admitió no tener la información. No es un acierto, pero es el
  comportamiento correcto cuando el dato no está.
- **`alucino`** — ni acertó ni se abstuvo: afirmó algo incorrecto con seguridad.
  Es el fallo grave, y el que más importa reportar en un tutor.
"""),
        code(CODIGO_EVALUACION),
    ]


# --------------------------------------------------------------------------
# Dimensión 2: el juez
# --------------------------------------------------------------------------

CODIGO_JUEZ = '''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUEZ_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # si va lento: "Qwen/Qwen2.5-0.5B-Instruct"

juez_tok = AutoTokenizer.from_pretrained(JUEZ_MODEL)
juez_model = AutoModelForCausalLM.from_pretrained(
    JUEZ_MODEL, torch_dtype="auto").to(DEVICE).eval()

# ---------------------------------------------------------------------------
# LA RÚBRICA ES PARTE DE LA ENTREGA. Se versiona aquí, no se improvisa.
# Anclas explícitas por nivel (S06: "ancla cada nivel 1-5 con una descripción
# para que no juzgue a ojo") y control de longitud incorporado en el texto
# (S06: "pide concisión en la rúbrica").
# ---------------------------------------------------------------------------
RUBRICA = """Eres un evaluador de un tutor de matemáticas. Califica la RESPUESTA:

5 = El resultado final es CORRECTO y el procedimiento está bien explicado paso a paso.
4 = El resultado final es CORRECTO, pero la explicación omite un paso o tiene un detalle menor mejorable.
3 = El procedimiento es razonable pero el RESULTADO FINAL ES INCORRECTO, o no hay respuesta final.
2 = Hay errores conceptuales graves en el procedimiento.
1 = Incorrecta, inventada, o no responde lo que se preguntó.

Reglas:
- Lo que más pesa es que el RESULTADO sea correcto. Un procedimiento elegante con
  resultado equivocado no puede pasar de 3.
- La longitud no es calidad: una explicación más larga NO merece mejor nota.
- Si la pregunta no tiene respuesta posible (datos insuficientes, división entre
  cero, premisa falsa), la respuesta correcta es decirlo; inventar un número es 1."""


def _extraer_puntaje(texto):
    m = re.search(r"[1-5]", texto)
    return int(m.group()) if m else 3      # fallback neutro


def _generar_juez(system, user, max_new_tokens=6):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = juez_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = juez_tok(prompt, return_tensors="pt").to(juez_model.device)
    with torch.no_grad():
        out = juez_model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=juez_tok.eos_token_id)
    return juez_tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


def juez_puntua(pregunta, respuesta, esperada=None):
    """Modo POINTWISE: una nota de 1 a 5 para una respuesta. Es el que usa el harness."""
    ref = f"\\n\\nRespuesta de referencia (guía, no literal): {esperada}" if esperada else ""
    user = (f"{RUBRICA}\\n\\nPregunta: {pregunta}\\nRespuesta a evaluar: {respuesta}{ref}\\n\\n"
            "Responde SOLO con un dígito del 1 al 5. Sin explicación.")
    return _extraer_puntaje(_generar_juez("Eres un evaluador estricto y objetivo.", user))


def juez_compara(pregunta, A, B):
    """Modo PAIRWISE: elige la mejor de dos. Se usa para cazar el sesgo de posición."""
    user = (f"Pregunta: {pregunta}\\n\\nRespuesta A: {A}\\n\\nRespuesta B: {B}\\n\\n"
            "¿Cuál respuesta es mejor? Responde SOLO con la letra A o B.")
    texto = _generar_juez("Eres un evaluador estricto y objetivo.", user, max_new_tokens=3).upper()
    m = re.search(r"[AB]", texto)
    return m.group() if m else "?"


def comparar_robusto(pregunta, X, Y):
    """Mitigación del sesgo de posición: evalúa en los dos órdenes y solo declara
    ganador si el veredicto coincide al invertir. Si se contradice, es empate."""
    v1 = juez_compara(pregunta, X, Y)   # X en la posición A
    v2 = juez_compara(pregunta, Y, X)   # X en la posición B
    if v1 == "A" and v2 == "B":
        return "X"
    if v1 == "B" and v2 == "A":
        return "Y"
    return "empate"
'''


def celda_juez() -> list[dict]:
    return [
        md("""
### Dimensión 2 · LLM-as-a-judge

Un LLM lee la respuesta y la califica con una rúbrica 1–5. Da lo que ninguna
métrica automática da —juicio sobre si la explicación *enseña*— a escala de
máquina.

**La rúbrica es el producto, no el código.** Está versionada en la celda de
abajo y adaptada al dominio con dos reglas que no están en la plantilla de
clase:

1. **El resultado pesa más que la forma.** "Un procedimiento elegante con
   resultado equivocado no puede pasar de 3." Sin esa ancla, un juez pequeño
   premia la prosa bien estructurada aunque el número esté mal — que es
   exactamente el modo de fallo de nuestro modelo según M1.
2. **La longitud no es calidad.** Es la mitigación del sesgo de longitud (S06)
   escrita dentro de la propia rúbrica.

Usamos el modo **pointwise** para el scorecard (una nota por respuesta) y el
modo **pairwise** solo para el test de sesgo de posición.

> **Advertencia honesta, y vale para todo el notebook:** el juez es
> `Qwen2.5-1.5B-Instruct`, el mismo modelo base que estamos evaluando. Eso lo
> expone al **sesgo de auto-preferencia** documentado en S06. Lo declaramos
> aquí y lo cuantificamos en la sección de calibración: en nuestro dominio, a
> diferencia de casi cualquier otro, podemos comprobar si el juez tiene razón.
"""),
        code(CODIGO_JUEZ),
    ]


# --------------------------------------------------------------------------
# El harness
# --------------------------------------------------------------------------

CODIGO_HARNESS = '''
import numpy as np


def harness(eval_set, sistema, nombre="sistema", verbose=True):
    """Las tres dimensiones sobre un sistema. `sistema` es cualquier función
    `pregunta -> respuesta`: el modelo de M1, el RAG de M3, lo que sea.

    Devuelve el scorecard: promedios por dimensión + el detalle caso por caso.
    """
    detalle = []
    for i, caso in enumerate(eval_set, 1):
        respuesta = sistema(caso["input"])

        sim = sim_embeddings(respuesta, caso["esperado"])        # Dimensión 1b
        pj = juez_puntua(caso["input"], respuesta, caso["esperado"])  # Dimensión 2
        ev = evaluar_caso(caso, respuesta)                       # Dimensión 3

        detalle.append({
            "id": caso["id"], "bloque": caso["bloque"], "subtipo": caso["subtipo"],
            "input": caso["input"], "respuesta": respuesta,
            "sim": round(sim, 3), "juez": pj,
            "acierto": ev["acierto"], "abstuvo": ev["abstuvo"], "alucino": ev["alucino"],
            "motivo": ev["motivo"],
            "formato_valido": formato_valido(respuesta),
            "palabras": len(respuesta.split()),
        })
        if verbose:
            marca = "OK " if ev["acierto"] else ("abs" if ev["abstuvo"] else "MAL")
            print(f"  [{i:2d}/{len(eval_set)}] {caso['id']:8s} {marca}  juez={pj}  sim={sim:.2f}  {ev['motivo'][:46]}")

    def prom(clave, filas=None):
        filas = filas if filas is not None else detalle
        return float(np.mean([d[clave] for d in filas])) if filas else 0.0

    calculo = [d for d in detalle if d["bloque"] == "calculo"]
    conocim = [d for d in detalle if d["bloque"] == "conocimiento"]
    advers  = [d for d in detalle if d["bloque"] == "adversarial"]

    return {
        "nombre": nombre,
        "n": len(detalle),
        # Dimensión 1
        "exactitud_calculo": prom("acierto", calculo),
        "sim_embeddings": prom("sim"),
        # Dimensión 2
        "juez_promedio": prom("juez"),
        # Dimensión 3
        "aciertos": sum(d["acierto"] for d in detalle),
        "aciertos_calculo": sum(d["acierto"] for d in calculo),
        "aciertos_conocimiento": sum(d["acierto"] for d in conocim),
        "aciertos_adversarial": sum(d["acierto"] for d in advers),
        "n_calculo": len(calculo), "n_conocimiento": len(conocim), "n_adversarial": len(advers),
        # Diagnóstico
        "alucinaciones": sum(d["alucino"] for d in detalle),
        "abstenciones": sum(d["abstuvo"] for d in detalle),
        "formato_valido": prom("formato_valido"),
        "palabras_media": prom("palabras"),
        "detalle": detalle,
    }


def imprimir_scorecard(*scorecards):
    """El scorecard: una tabla con el puntaje por dimensión. Acepta 1 o más
    sistemas para compararlos lado a lado sobre el mismo eval set."""
    nombres = [s["nombre"] for s in scorecards]
    ancho = 40
    total = ancho + 14 * len(scorecards)

    filas = [
        ("DIMENSIÓN 1 · MÉTRICA CLÁSICA", None, None),
        ("  1a · Exactitud en cálculo", lambda s: f"{s['exactitud_calculo']:.1%}", None),
        ("  1b · Similitud embeddings (0-1)", lambda s: f"{s['sim_embeddings']:.3f}", None),
        ("DIMENSIÓN 2 · LLM-AS-A-JUDGE", None, None),
        ("  2 · Nota media (1-5)", lambda s: f"{s['juez_promedio']:.2f}", None),
        ("DIMENSIÓN 3 · ACIERTOS DE DOMINIO", None, None),
        ("  3a · Total", lambda s: f"{s['aciertos']}/{s['n']}", None),
        ("  3b · Bloque cálculo", lambda s: f"{s['aciertos_calculo']}/{s['n_calculo']}", None),
        ("  3c · Bloque conocimiento", lambda s: f"{s['aciertos_conocimiento']}/{s['n_conocimiento']}", None),
        ("  3d · Bloque adversarial", lambda s: f"{s['aciertos_adversarial']}/{s['n_adversarial']}", None),
        ("DIAGNÓSTICO", None, None),
        ("  Alucinaciones (menos es mejor)", lambda s: str(s["alucinaciones"]), None),
        ("  Abstenciones honestas", lambda s: str(s["abstenciones"]), None),
        ("  Formato válido", lambda s: f"{s['formato_valido']:.0%}", None),
        ("  Palabras por respuesta", lambda s: f"{s['palabras_media']:.0f}", None),
    ]

    print("=" * total)
    print(f"{'':<{ancho}}" + "".join(f"{n[:13]:>14}" for n in nombres))
    print("-" * total)
    for etiqueta, fn, _ in filas:
        if fn is None:
            print(etiqueta)
        else:
            print(f"{etiqueta:<{ancho}}" + "".join(f"{fn(s):>14}" for s in scorecards))
    print("=" * total)


def guardar_scorecard(ruta, *scorecards):
    import csv
    nombres = [s["nombre"] for s in scorecards]
    filas = [
        ("exactitud_calculo", lambda s: round(s["exactitud_calculo"], 3)),
        ("sim_embeddings_prom", lambda s: round(s["sim_embeddings"], 3)),
        ("llm_juez_prom", lambda s: round(s["juez_promedio"], 3)),
        ("aciertos_total", lambda s: f"{s['aciertos']}/{s['n']}"),
        ("aciertos_calculo", lambda s: f"{s['aciertos_calculo']}/{s['n_calculo']}"),
        ("aciertos_conocimiento", lambda s: f"{s['aciertos_conocimiento']}/{s['n_conocimiento']}"),
        ("aciertos_adversarial", lambda s: f"{s['aciertos_adversarial']}/{s['n_adversarial']}"),
        ("alucinaciones", lambda s: s["alucinaciones"]),
        ("abstenciones", lambda s: s["abstenciones"]),
        ("formato_valido", lambda s: round(s["formato_valido"], 3)),
        ("palabras_media", lambda s: round(s["palabras_media"], 1)),
    ]
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dimension"] + nombres)
        for clave, fn in filas:
            w.writerow([clave] + [fn(s) for s in scorecards])
    print(f"Guardado: {ruta}")


def tabla_por_subtipo(*scorecards):
    """Dónde falla el sistema, por tipo de dificultad. Es la tabla accionable:
    dice qué ejemplos añadir al corpus de entrenamiento."""
    nombres = [s["nombre"] for s in scorecards]
    ids = [d["id"] for d in scorecards[0]["detalle"]]
    print(f"{'caso':9s} {'subtipo':28s}" + "".join(f"{n[:12]:>13}" for n in nombres))
    print("-" * (37 + 13 * len(scorecards)))
    for i, cid in enumerate(ids):
        sub = scorecards[0]["detalle"][i]["subtipo"]
        celdas = ""
        for s in scorecards:
            d = s["detalle"][i]
            celdas += f"{('OK' if d['acierto'] else ('abst' if d['abstuvo'] else 'FALLA')):>13}"
        print(f"{cid:9s} {sub:28s}{celdas}")
'''


def celda_harness() -> list[dict]:
    return [
        md("""
---
## El harness: las tres dimensiones en una función

`harness(eval_set, sistema)` recibe el eval set y **cualquier** función
`pregunta -> respuesta`, corre las tres dimensiones sobre cada caso y devuelve
el scorecard.

Que la firma sea tan genérica es deliberado: el mismo harness evaluará el
modelo de M1, el RAG de M3 y lo que venga después. Es la vara fija del
semestre, y solo sirve como vara si no cambia.

El scorecard desglosa los aciertos **por bloque**, porque el promedio global
mezcla tres preguntas distintas: ¿sabe calcular?, ¿conoce los datos de la
institución?, ¿es robusto ante trampas? Un sistema puede ser excelente en una y
pésimo en otra, y un número único lo ocultaría.
"""),
        code(CODIGO_HARNESS),
    ]


# --------------------------------------------------------------------------
# Dimensión 1: métricas clásicas
# --------------------------------------------------------------------------

CODIGO_SIMILITUD = '''
from sentence_transformers import SentenceTransformer
import numpy as np

# El MISMO modelo de embeddings de S05, S06 y S07 (multilingüe, pequeño).
st = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def sim_embeddings(a, b):
    ea, eb = st.encode([a, b])
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))
'''


def celda_similitud() -> list[dict]:
    return [
        md("""
### Dimensión 1 · Métricas clásicas

Dos métricas automáticas, y la segunda existe para demostrar por qué **no
basta**.

**1a · Exactitud de la respuesta final** — la métrica del dominio, heredada de
M1: se extrae el número que sigue a `Respuesta final:` y se compara
numéricamente con el esperado. Es objetiva y no admite discusión.

**1b · Similitud por embeddings** — la métrica clásica del lab de S05/S06.
La incluimos porque la entrega la pide y porque es útil para juzgar la
*explicación*, pero la sección siguiente muestra que en matemáticas es
peligrosa si se usa sola.
"""),
        code(CODIGO_SIMILITUD),
        md("""
#### Por qué la similitud sola no sirve para un tutor de matemáticas

Esta demostración es el equivalente, en nuestro dominio, del Lab A de S05
—donde BLEU premiaba la respuesta equivocada—. Aquí el que falla es el
embedding: dos respuestas idénticas salvo por el número obtienen una similitud
altísima, aunque una esté bien y la otra mal.

Es la razón por la que la Dimensión 3 usa un criterio numérico y no el
`sim >= 0.60` de la plantilla de clase.
"""),
        code('''
referencia = ("Paso 1: Aplicamos el primer descuento: 200000 × 0.80 = 160000.\\n"
              "Paso 2: El segundo se aplica sobre el precio ya rebajado: 160000 × 0.90 = 144000.\\n"
              "Respuesta final: 144000 pesos")

candidatos = {
    "correcta (misma redacción)": referencia,
    "CORRECTA pero parafraseada": ("Primero quitamos el 20 por ciento, quedando 160000 pesos. "
                                   "Después restamos el 10 por ciento de esa cantidad. "
                                   "El precio final es de 144000 pesos."),
    "INCORRECTA (sumó los %)":    ("Paso 1: Sumamos los descuentos: 20% + 10% = 30%.\\n"
                                   "Paso 2: 200000 × 0.70 = 140000.\\n"
                                   "Respuesta final: 140000 pesos"),
}

print(f"{'candidato':<30} {'similitud':>10} {'¿correcta?':>12}")
print("-" * 54)
for nombre, texto in candidatos.items():
    s = sim_embeddings(texto, referencia)
    ok = respuesta_correcta(texto, "144000")
    print(f"{nombre:<30} {s:>10.3f} {str(ok):>12}")
print("-" * 54)
print("La respuesta INCORRECTA obtiene una similitud altísima: comparte")
print("estructura, vocabulario y casi todos los números. Con el umbral de 0.60")
print("del lab de clase, contaría como acierto. Por eso el criterio de dominio")
print("tiene que mirar el número, no el parecido del texto.")
'''),
    ]
