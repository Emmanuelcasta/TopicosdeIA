"""Notebook de M4 — el componente visual, enchufado y medido.

Produce `m4/pipeline/S13_M4_Pipeline_Visual.ipynb`. Es **inferencia y
evaluación**: no entrena nada.

Tres decisiones ordenan el notebook:

1. **Dos etapas separadas.** El VLM solo transcribe la hoja; el tutor de M3 solo
   diagnostica. Si se mezclaran, un fallo no se podría atribuir a los ojos o al
   razonamiento, que es justo la distinción que este módulo tiene que medir.

2. **Dos fases con la GPU.** Primero el VLM lee las 11 fotos y la transcripción se
   guarda en disco; se libera el VLM; recién entonces se carga el tutor. Un VLM de
   2B en fp16 (~4.4 GB) más el tutor de 1.5B más el juez no caben cómodos en una
   T4 de 16 GB, y la transcripción guardada permite reejecutar el diagnóstico sin
   volver a leer las fotos, que es la parte cara.

3. **Dos conjuntos que NO se comparan entre sí.** Los 12 casos visuales miden una
   tarea nueva (localizar el paso erróneo en una foto). Los 21 casos de M2 miden
   la tarea de siempre, sin imagen, y sirven como regresión: comprobar que añadir
   la vista no rompió lo que ya funcionaba. Presentarlos como un solo número de
   mejora sería comparar un examen de lectura con uno de aritmética.
"""

from __future__ import annotations

from nb_comun import CODIGO_METRICAS, celda_drive, code, encabezado, md
from nb_comun_agente import CODIGO_LLM, celdas_modulos
from nb_comun_eval import celda_eval_set, celda_evaluacion, celda_harness, celda_juez, celda_similitud
from nb_comun_rag import CODIGO_CHUNKING, CODIGO_CORPUS
from nb_m4_datos import celda_eval_m4


CODIGO_IMAGENES = r'''
from pathlib import Path

from PIL import Image, ImageOps

DIR_IMAGENES = Path("m4/eval_set/imagenes")
LADO_MAYOR = 1024          # límite del lado mayor, en píxeles

# Por qué se redimensiona: Qwen2-VL convierte la imagen en tokens visuales, y una
# foto de celular a resolución completa llena la T4. Peor: haría que el coste de
# cada caso dependiera de cómo salió la foto, con lo que la columna de latencia
# dejaría de ser comparable entre casos.
#
# Por qué `exif_transpose`: las fotos de celular guardan la orientación en los
# metadatos. Sin corregirla, el bloque adversarial mediría rotación en vez de
# legibilidad, que es otra cosa.


def cargar_imagen(ruta, lado_mayor=LADO_MAYOR):
    im = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
    if max(im.size) > lado_mayor:
        escala = lado_mayor / max(im.size)
        im = im.resize((int(im.width * escala), int(im.height * escala)), Image.LANCZOS)
    return im


DIR_IMAGENES.mkdir(parents=True, exist_ok=True)

# El emparejamiento es por NOMBRE BASE, no por nombre exacto: `vis-10.jpeg`
# satisface el caso que el eval set declara como `vis-10.jpg`. En la corrida del
# 22-10 el emparejamiento exacto dejó ese caso sin evaluar, que era el único
# adversarial de hoja ilegible.
#
# Si hay dos archivos con el mismo nombre base, `inventario` LANZA y la corrida
# se detiene: elegir uno por orden alfabético mediría una hoja y el informe
# hablaría de la otra, sin que nada avisara.
presentes = [p.name for p in DIR_IMAGENES.iterdir() if not p.name.startswith(".")]
PEDIDAS, FALTAN, SOBRAN = inventario(eval_m4, presentes)   # puede lanzar ImagenAmbigua

print(f"{len(eval_m4)} casos  |  {len(PEDIDAS)} fotos distintas  |  {len(presentes)} archivos en la carpeta\n")
for logico, info in sorted(PEDIDAS.items()):
    marca = "FALTA" if info["archivo"] is None else " ok  "
    real = "" if info["archivo"] in (None, logico) else f"  [archivo: {info['archivo']}]"
    print(f"  [{marca}] {logico:14s} -> {', '.join(info['casos'])}{real}")
if SOBRAN:
    print("\nArchivos que el eval set no pide (se ignoran):", sorted(SOBRAN))

if FALTAN:
    print(f"\nFaltan {len(FALTAN)} fotos de {len(PEDIDAS)}.")
    print("El notebook sigue, pero esos casos quedan marcados `sin_imagen` y NO entran")
    print("en ninguna media: una foto que no se subió no es un fallo del sistema, y")
    print("contarla como cero sería inventar un resultado.")
    print("\nQué escribir en cada hoja está en m4/eval_set/HOJAS_PARA_ESCRIBIR.md.")
    print(f"Súbanlas a {DIR_IMAGENES} con el nombre base del eval set (.jpg, .jpeg o .png).")
else:
    print(f"\nEstán las {len(PEDIDAS)} fotos: el scorecard de esta corrida será completo.")
'''


CODIGO_GUARDA_EVAL_M2 = r'''
# GUARDA · ¿es este el mismo eval set con el que se midió M3?
#
# Las corridas del 22-10 y del 22-10 (tarde) imprimieron las dos `SHA coincide:
# False`: en Drive había una copia distinta de `data/eval_set_m2.jsonl` —la de
# antes de corregir el criterio de `adv-02`— así que su 15/21 NO era comparable
# con el 13/21 de M3. No se compara contra un número lo que no se midió con la
# misma vara.
#
# Ahora la celda lo CORRIGE sola, sin borrar nada: aparta la copia que no cuadra
# con un nombre que dice por qué, escribe la copia embebida y verificada, y
# recarga el eval set. La guarda se mantiene: si tras eso el hash sigue sin
# cuadrar, la comparación queda marcada como no válida.
import shutil

if sha_real != SHA_EVAL:
    descartada = RUTA_EVAL.with_name(f"eval_set_m2.NO_COINCIDE-{sha_real[:12]}.jsonl")
    if not descartada.exists():
        shutil.copy2(RUTA_EVAL, descartada)      # se conserva como evidencia
    RUTA_EVAL.write_bytes(gzip.decompress(base64.b64decode(_BLOB_EVAL)))
    eval_set = [json.loads(l) for l in RUTA_EVAL.read_text(encoding="utf-8").splitlines()]
    sha_real = hashlib.sha256(RUTA_EVAL.read_bytes()).hexdigest()
    print("La copia de Drive no coincidía con la del repositorio. Se corrigió:")
    print(f"  la anterior se conservó en {descartada.name}")
    print(f"  ahora SHA-256: {sha_real[:16]}…  coincide: {sha_real == SHA_EVAL}")
    print(f"  casos recargados: {len(eval_set)}")
    print()

EVAL_M2_COMPARABLE = (sha_real == SHA_EVAL)
if EVAL_M2_COMPARABLE:
    print("El eval set de M2 es el mismo con el que se midió M3: la regresión es")
    print("comparable, y la fila de regresión de la tabla de delta es válida.")
else:
    print("=" * 78)
    print("ATENCIÓN · el eval set de M2 sigue sin coincidir con el del repositorio.")
    print("=" * 78)
    print(f"  esperado (embebido): {SHA_EVAL[:16]}…")
    print(f"  encontrado en disco: {sha_real[:16]}…")
    print()
    print("La sección 8 se ejecutará igual, pero su cifra queda marcada como NO")
    print("comparable y no se pone al lado de la de M3.")
'''

CODIGO_VER = r'''
import json
import time

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

VLM_ID = "Qwen/Qwen2-VL-2B-Instruct"
MAX_PIXELS = 1024 * 1024          # tope de tokens visuales por imagen

# `size=` en vez de `max_pixels=`: transformers 5.20 avisa que `min_pixels` y
# `max_pixels` se eliminan en 5.23. Es el mismo tope (mismo `image_grid_thw`).
vlm_proc = AutoProcessor.from_pretrained(
    VLM_ID, size={"shortest_edge": 56 * 56, "longest_edge": MAX_PIXELS})
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_ID,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None).eval()
print("VLM cargado:", VLM_ID)
if DEVICE == "cuda":
    print(f"GPU ocupada: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# La instrucción (INSTRUCCION_VER) y el parser tolerante (json_de) vienen del
# módulo `vision.py` de la celda 3.1, donde están probados.
print("\nInstrucción de lectura:\n", INSTRUCCION_VER)


@torch.no_grad()
def ver(imagen, instruccion=INSTRUCCION_VER, max_new_tokens=320, muestrear=False):
    """imagen -> (texto crudo, dict o None, segundos). La etapa 1 del pipeline.

    Con `muestrear=True` genera con temperatura en vez de greedy: es la segunda
    opinión que usa `leer_dos_veces`.
    """
    t0 = time.perf_counter()
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": instruccion}]}]
    prompt = vlm_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ent = vlm_proc(text=[prompt], images=[imagen], return_tensors="pt").to(vlm.device)
    kwargs = (dict(do_sample=True, temperature=0.8, top_p=0.95) if muestrear
              else dict(do_sample=False))
    out = vlm.generate(**ent, max_new_tokens=max_new_tokens,
                       return_dict_in_generate=True, output_scores=True, **kwargs)
    nuevos = out.sequences[:, ent["input_ids"].shape[1]:]
    texto = vlm_proc.batch_decode(nuevos, skip_special_tokens=True)[0].strip()

    # Confianza del modelo en los tokens que son NÚMEROS. Un dígito leído de una
    # cifra nítida sale con más probabilidad que uno rellenado de memoria, así que
    # esto es una señal sobre la lectura que no depende de mirar la imagen.
    confianza = None
    try:
        scores = vlm.compute_transition_scores(out.sequences, out.scores,
                                               normalize_logits=True)[0]
        tok = vlm_proc.tokenizer
        de_numeros = [float(s) for s, i in zip(scores, nuevos[0])
                      if any(c.isdigit() for c in tok.decode([int(i)]))]
        confianza = sum(de_numeros) / len(de_numeros) if de_numeros else None
    except Exception as e:                      # nunca tumbar la lectura por esto
        print(f"    (sin confianza de tokens: {type(e).__name__})")

    return texto, json_de(texto), time.perf_counter() - t0, confianza


# Una pregunta distinta, en una llamada aparte: no se le pide transcribir, solo
# juzgar si la hoja se lee. Separar las dos preguntas es el punto — en la corrida
# del 24-10 la transcripción venía arrastrada por el prior del ejercicio y el
# modelo rellenó los números; preguntado sin transcribir, el prior no tiene dónde
# agarrarse igual.
INSTRUCCION_LEGIBILIDAD = (
    "Mira esta foto de una hoja escrita a mano. NO la transcribas. Juzga solo si "
    "se puede leer con claridad cada cifra y cada renglón.\n"
    "Responde SOLO un JSON: {\"legible\": true|false, \"motivo\": \"...\"}\n"
    "Pon `legible: false` si la foto está movida, desenfocada, con reflejos o "
    "demasiado lejos para distinguir los números con seguridad."
)


@torch.no_grad()
def preguntar_legibilidad(imagen):
    """-> (legible: bool|None, motivo). None si la respuesta no se pudo parsear."""
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": INSTRUCCION_LEGIBILIDAD}]}]
    prompt = vlm_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ent = vlm_proc(text=[prompt], images=[imagen], return_tensors="pt").to(vlm.device)
    out = vlm.generate(**ent, max_new_tokens=80, do_sample=False)
    texto = vlm_proc.batch_decode(out[:, ent["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
    js = json_de(texto) or {}
    legible = js.get("legible")
    return (bool(legible) if isinstance(legible, bool) else None,
            str(js.get("motivo", ""))[:200])


def leer_dos_veces(imagen, archivo):
    """La salvaguarda contra la lectura inventada.

    Se lee la foto dos veces: una greedy (la que se usa) y otra con muestreo. Si
    las dos no coinciden en el número de pasos y en los números transcritos, la
    lectura no es de fiar y el tutor recibirá "la hoja NO se pudo leer" en vez de
    un procedimiento inventado.

    Por qué así y no con una métrica de nitidez de la imagen: un umbral de
    desenfoque habría que calibrarlo, y calibrarlo con estas 11 fotos sería
    ajustarlo al eval set. Que dos muestreos del mismo modelo sobre la misma foto
    discrepen es una señal del propio modelo —está inventando— y no necesita
    ninguna constante elegida a mano.
    """
    crudo, js, seg, confianza = ver(imagen)
    lectura = {"crudo": crudo, "json": js, "segundos": round(seg, 2),
               "ok": js is not None, "archivo": archivo,
               "confianza_numeros": confianza}

    # Señal 2 · la misma pregunta, otra vez, con muestreo.
    _, js2, seg2, _ = ver(imagen, muestrear=True)
    lectura["consistente"] = lecturas_consistentes(lectura, {"json": js2})
    lectura["json_segunda"] = js2

    # Señal 3 · otra pregunta: ¿se lee? Sin transcribir.
    legible, motivo = preguntar_legibilidad(imagen)
    lectura["legible_declarada"] = legible
    lectura["motivo_legibilidad"] = motivo

    lectura["segundos_verificacion"] = round(seg2, 2)
    return lectura
'''


CODIGO_CALIBRACION = r'''
# CALIBRACIÓN de la detección de hojas ilegibles · con datos que NO son el eval set.
#
# Por qué aquí y no mirando `vis-10`: la detección tiene un umbral (la confianza
# en los tokens numéricos, marcada con mediana − k·MAD). Ajustar ese umbral
# mirando `vis-10` convertiría un caso de evaluación en conjunto de calibración, y
# entonces el 0/3 adversarial dejaría de significar nada.
#
# El conjunto lo genera `scripts/calibracion_legibilidad_fuente.py`: 6 hojas × 3
# versiones (nítida, borrosa leve, borrosa fuerte). La leve es la que importa: es
# el análogo de `vis-10`, donde el VLM alcanza a leer parte y rellena el resto.
#
# LÍMITE, y es grande: son hojas con texto IMPRESO. Un buen resultado aquí NO
# autoriza a decir que la salvaguarda funciona con letra manuscrita. La validación
# de verdad necesita fotos de las hojas dev-* escritas a mano, nítidas y
# desenfocadas, y mientras no existan la salvaguarda está sin validar.
DIR_CALIBRACION = Path("m4/eval_set/calibracion")
MANIFIESTO = DIR_CALIBRACION / "manifiesto.json"

if not MANIFIESTO.exists():
    print("No está el conjunto de calibración. Genérenlo con")
    print("  python scripts/calibracion_legibilidad_fuente.py")
    print("y súbanlo a Drive. Sin él, la salvaguarda queda sin medir.")
else:
    pares = json.loads(MANIFIESTO.read_text(encoding="utf-8"))["pares"]
    cal = {}
    for r in pares:
        cal[r["archivo"]] = leer_dos_veces(cargar_imagen(DIR_CALIBRACION / r["archivo"]),
                                           r["archivo"])
        cal[r["archivo"]]["etiqueta"] = r["etiqueta"]

    # El criterio de atípicos se calcula DENTRO del conjunto de calibración.
    conf = {k: v["confianza_numeros"] for k, v in cal.items()
            if v.get("confianza_numeros") is not None}
    at = atipicos_por_mad(conf)
    for k in cal:
        cal[k]["confianza_atipica"] = k in at

    print(f"{'archivo':28s}{'etiqueta':17s}{'conf.núm':>10s}{'¿se lee?':>10s}   detectada")
    filas_cal = []
    for k in sorted(cal):
        v = cal[k]
        no_fiable, _ = lectura_no_fiable(v)
        c = v.get("confianza_numeros")
        filas_cal.append({"archivo": k, "etiqueta": v["etiqueta"],
                          "legible": v["etiqueta"] == "legible",
                          "confianza": c, "atipica": v["confianza_atipica"],
                          "legible_declarada": v.get("legible_declarada"),
                          "consistente": v.get("consistente"),
                          "detectada_no_fiable": no_fiable})
        print(f"{k:28s}{v['etiqueta']:17s}"
              f"{(f'{c:.3f}' if c is not None else '—'):>10s}"
              f"{({True: 'sí', False: 'NO', None: '?'}[v.get('legible_declarada')]):>10s}"
              f"   {'NO FIABLE' if no_fiable else 'ok'}")

    CAL = pd.DataFrame(filas_cal)
    CAL.to_csv("m4/calibracion_legibilidad.csv", index=False)

    ileg = CAL[~CAL.legible]
    leg = CAL[CAL.legible]
    print()
    print("=" * 78)
    print("QUÉ TAN BIEN SEPARA · sobre calibración, nunca sobre el eval set")
    print("=" * 78)
    print(f"  ilegibles detectadas   : {int(ileg.detectada_no_fiable.sum())}/{len(ileg)}"
          "   (cuantas más, mejor)")
    for nivel in ("ilegible_leve", "ilegible_fuerte"):
        g = CAL[CAL.etiqueta == nivel]
        if len(g):
            print(f"      {nivel:16s}: {int(g.detectada_no_fiable.sum())}/{len(g)}")
    print(f"  legibles marcadas mal  : {int(leg.detectada_no_fiable.sum())}/{len(leg)}"
          "   (cuantas menos, mejor)")
    print("=" * 78)
    print()
    print("Cómo leerlo:")
    print("  · Si NO caza las `ilegible_fuerte`, la salvaguarda no funciona en absoluto.")
    print("  · Si caza las fuertes pero no las leves, cazará lo evidente y dejará pasar")
    print("    justo el caso que importa, que es el de `vis-10`.")
    print("  · Si marca legibles, el coste es real: a una hoja buena se le dirá que no")
    print("    se pudo leer y su caso fallará por la salvaguarda.")
    print()
    print("Y el límite que no se puede saltar: esto es texto IMPRESO. Con letra")
    print("manuscrita el problema es más difícil incluso en nítido. Mientras no haya")
    print("fotos a mano de las hojas dev-*, la salvaguarda sigue SIN VALIDAR y así hay")
    print("que reportarla.")
'''

CODIGO_PUERTA_PILOTO = r'''
# PUERTA PILOTO · tres imágenes antes de gastar la corrida completa.
#
# El riesgo que puede costar la entrega es que un VLM de 2B no lea letra
# manuscrita. Más vale descubrirlo en tres fotos que a los cuarenta minutos.
#
# Las tres están elegidas para que cada una pueda fallar de forma distinta:
PILOTO = {
    "vis-01": "letra manuscrita con una cuenta mal: ¿copia el error o lo 'arregla'?",
    "vis-07": "procedimiento correcto: ¿no inventa un error al transcribir?",
    "vis-10": "foto movida a propósito: ¿dice que no puede leerla?",
}

lecturas_piloto = {}
for cid, que_prueba in PILOTO.items():
    caso = CASOS_M4[cid]
    logico = archivo_de(caso)
    real = PEDIDAS[logico]["archivo"]
    print(f"[{cid}] {que_prueba}")
    if real is None:
        print(f"      falta la foto de {logico}\n")
        continue
    lectura = leer_dos_veces(cargar_imagen(DIR_IMAGENES / real), real)
    js, seg = lectura["json"], lectura["segundos"]
    lecturas_piloto[logico] = lectura
    print(f"      archivo: {real} | {seg:.1f}s | json parseado: {js is not None}"
          f" | dos lecturas coinciden: {lectura['consistente']}")
    no_fiable, motivo = lectura_no_fiable(lectura)
    if no_fiable:
        print(f"      LECTURA NO FIABLE -> {motivo}")
        print("      el tutor recibirá 'la hoja NO se pudo leer', no un procedimiento")

    # Se imprimen los pasos YA NUMERADOS, que es exactamente lo que verá el tutor.
    # La numeración incluye la respuesta final como último paso, igual que el eval
    # set: si aquí no coincide, el paso que hay que señalar no existe para el tutor.
    if js and js.get("ilegible"):
        print(f"      ilegible: {js.get('motivo')}")
    elif js:
        for i, paso in enumerate(pasos_numerados(lectura), 1):
            print(f"      Paso {i}: {paso}")
    else:
        print("      crudo:", crudo[:200].replace("\n", " | "))

    esperados = caso["pasos"]
    if esperados:
        vistos = len(pasos_numerados(lectura))
        print(f"      la hoja tiene {len(esperados)} pasos; el tutor verá {vistos}"
              + ("  <-- NO COINCIDE" if vistos != len(esperados) else ""))
        print(f"      diagnóstico esperado: {diagnostico_esperado(caso)!r}")
        print(f"      ¿transcripción fiel?: {lectura_fiel(caso, lectura)}")
    else:
        print(f"      diagnóstico esperado: {diagnostico_esperado(caso)!r} (caso adversarial)")
    print()

print("DECISIÓN: ¿los pasos transcritos coinciden con lo que dice la hoja, y su")
print("          numeración con la que cuenta el eval set?")
print("  Si sí  -> sigan a la transcripción completa.")
print("  Si no  -> paren aquí. Repliegues, en este orden:")
print("     1. subir el contraste y el tamaño en el preproceso;")
print("     2. un OCR clásico (pytesseract) y dejar al VLM solo estructurando el texto;")
print("     3. declarar la lectura como el cuello de botella y medirlo, que también es")
print("        un resultado. Lo que NO sirve es correr las 11 y enterarse al final.")
print("\nSi aparece 'NO COINCIDE', paren igualmente: ese caso sería inacertable por")
print("construcción y contarlo como fallo del tutor sería un error de medición.")
'''


CODIGO_TRANSCRIBIR = r'''
# Transcripción completa, con caché en disco ANTES de soltar el VLM.
#
# Si Colab se desconecta después de esta celda no se pierde nada, y el diagnóstico
# se puede reejecutar o afinar sin volver a pasar el VLM por las fotos.
Path("m4").mkdir(exist_ok=True)
RUTA_LECTURA = Path("m4/lecturas_m4.json")
lecturas = json.loads(RUTA_LECTURA.read_text(encoding="utf-8")) if RUTA_LECTURA.exists() else {}

if len(FALTAN) == len(PEDIDAS):
    raise SystemExit("No hay ninguna foto en m4/eval_set/imagenes: no hay nada que leer. "
                     "Súbanlas y vuelvan a ejecutar desde la celda 3.4.")

# Las lecturas se guardan con el nombre LÓGICO del eval set (`vis-10.jpg`) y
# registran aparte el archivo real que se leyó (`vis-10.jpeg`). Así el archivo de
# lecturas no cambia de forma si mañana la foto llega con otra extensión.
lecturas.update(lecturas_piloto)        # el piloto ya leyó tres: no se repiten

# La caché se reusa solo si trae LAS CUATRO señales (`lectura_vigente`). Antes se
# miraba solo `consistente`, y el 25-10 se reusaron 8 lecturas de un esquema
# anterior: sin confianza ni "¿se lee?", la salvaguarda no pudo marcar `vis-10`.
for logico, info in sorted(PEDIDAS.items()):
    if info["archivo"] is None:
        continue
    previa = lecturas.get(logico)
    if lectura_vigente(previa):
        continue
    lecturas[logico] = leer_dos_veces(cargar_imagen(DIR_IMAGENES / info["archivo"]),
                                      info["archivo"])
    l = lecturas[logico]
    marca = "json ok" if l["ok"] else "SIN json"
    aviso = "" if l["consistente"] else "  <-- LAS DOS LECTURAS NO COINCIDEN"
    print(f"  {info['archivo']:14s} {l['segundos']:5.1f}s  {marca}{aviso}")

# Señal 4 · confianza atípica en los números, RELATIVA al resto del conjunto.
# El criterio es mediana − 3·MAD, la convención robusta de siempre. No hay ningún
# umbral calibrado con estas fotos, y no debe calibrarse con ellas: `vis-10` es un
# caso de evaluación, no de ajuste.
confianzas = {k: v["confianza_numeros"] for k, v in lecturas.items()
              if v.get("confianza_numeros") is not None}
atipicas = atipicos_por_mad(confianzas)
if len(confianzas) < 4:
    print(f"ATENCIÓN · solo {len(confianzas)} fotos traen confianza: la señal 4 no puede "
          "decidir con menos de 4. Borren m4/lecturas_m4.json y vuelvan a leer.")
for k in lecturas:
    lecturas[k]["confianza_atipica"] = k in atipicas

RUTA_LECTURA.write_text(json.dumps(lecturas, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nLAS CUATRO SEÑALES DE FIABILIDAD, foto por foto")
print(f"{'foto':14s}{'conf.núm':>10s}{'atípica':>9s}{'2ª lectura':>12s}"
      f"{'¿se lee?':>10s}   veredicto")
for k in sorted(lecturas):
    v = lecturas[k]
    c = v.get("confianza_numeros")
    no_fiable, motivo = lectura_no_fiable(v)
    print(f"{k:14s}{(f'{c:.3f}' if c is not None else '—'):>10s}"
          f"{('sí' if v.get('confianza_atipica') else 'no'):>9s}"
          f"{('coincide' if v.get('consistente') else 'DISCREPA'):>12s}"
          f"{({True: 'sí', False: 'NO', None: '?'}[v.get('legible_declarada')]):>10s}"
          f"   {'NO FIABLE' if no_fiable else 'ok'}")

print("\nLa columna de confianza se imprime para que la vean, no para ajustar nada:")
print("si quieren saber si separa las fotos ilegibles, hace falta un conjunto de")
print("calibración aparte (ver el README de la corrida 03). Mover el criterio hasta")
print("que una foto del eval set caiga del lado deseado la convierte en ajuste.")

no_fiables = [k for k, v in lecturas.items() if lectura_no_fiable(v)[0]]
if no_fiables:
    print(f"\nLecturas NO fiables: {', '.join(sorted(no_fiables))}")
    print("En esos casos el tutor recibirá 'la hoja NO se pudo leer' en lugar de una")
    print("transcripción. No se acredita ningún caso por esto: solo se evita juzgar")
    print("al tutor sobre un procedimiento que el VLM se inventó.")
else:
    print("\nNinguna lectura se marcó como no fiable.")
    print("Si alguna de las fotos era ilegible a propósito, esto es un FALSO NEGATIVO")
    print("de la salvaguarda y hay que decirlo: ese caso no mide al tutor.")

RUTA_LECTURA.write_text(json.dumps(lecturas, ensure_ascii=False, indent=2), encoding="utf-8")
leidas = sum(1 for v in lecturas.values() if v["ok"])
print(f"\n{len(lecturas)} imágenes transcritas, {leidas} con JSON válido -> {RUTA_LECTURA}")
print(f"Latencia media de lectura: {np.mean([v['segundos'] for v in lecturas.values()]):.1f} s")
'''


CODIGO_LIBERAR = r'''
# Liberar la GPU antes de cargar el tutor. La transcripción ya está en disco, así
# que el VLM no vuelve a hacer falta en esta corrida.
#
# Para la demo (sección 12) los dos modelos tienen que convivir: pongan
# MANTENER_VLM = True antes de esta celda y no carguen el juez.
import gc

if globals().get("MANTENER_VLM"):
    print("MANTENER_VLM = True: el VLM se queda en memoria para `revisar_foto_demo`.")
else:
    del vlm, vlm_proc
gc.collect()
if globals().get("MANTENER_VLM") and torch.cuda.is_available():
    print(f"GPU con el VLM residente: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU tras liberar el VLM: {torch.cuda.memory_allocated() / 1e9:.1f} GB "
          f"de {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
else:
    print("Sin GPU: nada que liberar.")
'''


CODIGO_INDICE = r'''
import chromadb

cliente = chromadb.Client()
try:
    cliente.delete_collection("tutor_m4")
except Exception:
    pass
coleccion = cliente.create_collection("tutor_m4", metadata={"hnsw:space": "cosine"})
coleccion.add(ids=ids_chunks, documents=chunks, metadatas=metadatos,
              embeddings=st.encode(chunks, show_progress_bar=False).tolist())
POS = {cid: i for i, cid in enumerate(ids_chunks)}


def buscar_densa(consulta, k=3):
    r = coleccion.query(query_embeddings=st.encode([consulta]).tolist(),
                        n_results=min(k, len(chunks)))
    return [POS[cid] for cid in r["ids"][0]]


print("Índice:", coleccion.count(), "chunks (mismo corpus y mismo chunking que S07/S08)")
'''


CODIGO_ADAPTADOR_DIAGNOSTICO = r'''
# El adaptador de diagnóstico, con una comprobación REAL de la ruta.
#
# `CODIGO_LLM` es compartido con los notebooks de S10 y solo carga `m1` y
# `agente`; este de M4 necesita además el de diagnóstico, así que se carga aquí.
#
# En la corrida del 25-10 esta carga no existía y la celda siguiente imprimía
# "No está adaptadores/qwen-lora-diagnostico" con los dos archivos presentes en
# Drive: el mensaje afirmaba algo que nadie había comprobado. Ahora se mira el
# disco y se dice qué falta.
import os

DIR_LORA_DIAGNOSTICO = "adaptadores/qwen-lora-diagnostico"
DIAGNOSTICO_OK, DIAGNOSTICO_MOTIVO = revisar_adaptador(DIR_LORA_DIAGNOSTICO)

print("Directorio de trabajo:", os.getcwd())
print("Ruta efectiva        :", Path(DIR_LORA_DIAGNOSTICO).resolve())
print("Comprobación         :", DIAGNOSTICO_MOTIVO)

if DIAGNOSTICO_OK:
    for f in sorted(Path(DIR_LORA_DIAGNOSTICO).iterdir()):
        print(f"   {f.name:34s} {f.stat().st_size / 1e6:8.1f} MB")
    if "diagnostico" not in ADAPTADORES:
        # `load_adapter` AÑADE un adaptador con su propio nombre: no reemplaza
        # `m1` ni `agente`, que siguen cargados y seleccionables.
        modelo.load_adapter(DIR_LORA_DIAGNOSTICO, adapter_name="diagnostico")
        ADAPTADORES.append("diagnostico")

# Que los dos de siempre sigan ahí no se da por supuesto: se comprueba.
assert "m1" in modelo.peft_config, "se perdió el adaptador de M1"
if "agente" in ADAPTADORES:
    assert "agente" in modelo.peft_config, "se perdió el adaptador del agente"

print("\nAdaptadores en el modelo:", list(modelo.peft_config.keys()))
print("ADAPTADORES             :", ADAPTADORES)
'''

CODIGO_RESPONDER_M3 = r'''
# `responder_m3`: el sistema que se entregó en M3, sin reescribirlo.
#
# Se usa el adaptador entrenado con trayectorias de herramientas si está en
# Drive; si no, el de M1. La diferencia cambia QUÉ sistema se está midiendo, así
# que queda declarada en el scorecard y en resultados_m4.json.
ADAPTADOR_TUTOR = "agente" if "agente" in ADAPTADORES else "m1"

# Aciertos de M3 sobre el eval set de M2, leídos de
# M3_rag_agentes/resultados/scorecard_agente_v2.csv (la versión con el criterio de
# adv-02 corregido). Son la referencia de la fila de regresión.
REFERENCIA_M3 = {
    "agente": {"nombre": "C5-FT · entrenado", "aciertos": 13, "segundos": 14.1},
    "m1":     {"nombre": "C1 · +LoRA",        "aciertos": 14, "segundos": 5.5},
}[ADAPTADOR_TUTOR]

print(f"Adaptador del tutor: {ADAPTADOR_TUTOR}")
print(f"Referencia de M3   : {REFERENCIA_M3['nombre']} con {REFERENCIA_M3['aciertos']}/21")
if ADAPTADOR_TUTOR == "m1":
    print("  Atención: no se encontró adaptadores/qwen-lora-agente, así que el agente no")
    print("  usará herramientas como en S10. Es otro sistema, y hay que decirlo al leer.")

REGISTRO_M4 = registro_con_documentos(REGISTRO, buscar_densa, chunks, metadatos, ids_chunks, k=3)

agente_m3 = AgenteTutor(
    crear_llm(ADAPTADOR_TUTOR, 320), SYSTEM_C5_AGENTE, registro=REGISTRO_M4,
    max_rondas=6, max_pasos=8, verificar=True,
    extraer_contextos=chunks_de_resultado, nombre="tutor M3")

# La vía visual usa el adaptador de diagnóstico si está; la de texto sigue con el
# de S10. Son dos vías con dos adaptadores a propósito:
#
#   · la regresión de M2 tiene que medir el MISMO sistema que midió M3, o la fila
#     deja de ser una regresión;
#   · la tarea visual necesita un modelo que sepa dictaminar, que es justo lo que
#     el de S10 no sabe.
#
# El precio es que la fila de regresión ya no describe el modelo que responde las
# fotos. Queda dicho aquí y en el artefacto.
ADAPTADOR_DIAGNOSTICO = elegir_adaptador(ADAPTADORES, "diagnostico", ADAPTADOR_TUTOR)
if ADAPTADOR_DIAGNOSTICO == "diagnostico":
    print(f"\nVía visual   -> {ADAPTADOR_DIAGNOSTICO!r} (entrenado para dictaminar)")
    print(f"Vía de texto -> {ADAPTADOR_TUTOR!r} (el de S10, para que la regresión siga "
          "midiendo lo mismo)")
else:
    # El motivo sale de la comprobación de la celda anterior, no de una suposición.
    print(f"\nLa vía visual usará {ADAPTADOR_TUTOR!r}, no el de diagnóstico.")
    print(f"Motivo: {DIAGNOSTICO_MOTIVO}")
    print("Es el sistema de la corrida del 25-10, que contestó 'paso 2' a los 12 casos.")

agente_diagnostico = AgenteTutor(
    crear_llm(ADAPTADOR_DIAGNOSTICO, 320), SYSTEM_C5_AGENTE, registro=REGISTRO_M4,
    max_rondas=6, max_pasos=8, verificar=True,
    extraer_contextos=chunks_de_resultado, nombre="tutor diagnóstico")

TRAZAS_M4 = {}


def responder_sin_imagen(pregunta):
    """La vía de texto: el sistema de M3, sin tocar. Es la de la regresión."""
    salida = agente_m3(pregunta)
    TRAZAS_M4[pregunta[:160]] = salida
    return salida.respuesta


def responder_diagnostico(pregunta):
    """La vía visual: el adaptador entrenado para emitir un veredicto."""
    salida = agente_diagnostico(pregunta)
    TRAZAS_M4[pregunta[:160]] = salida
    return salida.respuesta


print(f"\n{len(REGISTRO_M4)} herramientas disponibles para el tutor.")
'''


CODIGO_SISTEMA_MULTIMODAL = r'''
# El enchufe, que es la pieza que M4 evalúa.
#
# Patrón INPUT: la foto entra, el VLM la convierte en texto y el sistema de M3 lo
# consume como contexto. El sistema de M3 no se modifica: se le antepone `ver()`.
#
# `INSTRUCCION_REVISION` y `pasos_como_texto` vienen del módulo `vision.py`.


REINTENTOS = {}        # qué casos necesitaron una segunda pasada


def sistema_multimodal(pregunta, lectura=None, clave=None):
    """pregunta (+ lo que se leyó de la foto) -> respuesta del tutor.

    Sin `lectura` es exactamente el sistema de M3: la misma función responde los
    21 casos de la regresión, que no llevan imagen. Que sea la MISMA función es lo
    que hace legítima esa fila.

    Con `lectura`, además del prompt nuevo hay **un** reintento acotado: si la
    respuesta no trae una conclusión válida, se le devuelve su propia explicación
    y se le pide solo la línea. Queda registrado en `REINTENTOS`, porque contestar
    en dos pasadas no es lo mismo que contestar en una.
    """
    if lectura is None:
        return responder_sin_imagen(pregunta)
    respuesta = responder_diagnostico(prompt_revision(pregunta, lectura))
    respuesta, reintento = asegurar_diagnostico(respuesta, responder_diagnostico)
    REINTENTOS[clave or pregunta[:60]] = reintento
    return respuesta


print("sistema_multimodal() listo.")
print("  con imagen -> ver() + M3 + un reintento si falta la conclusión")
print("  sin imagen -> M3 tal cual")
print("\nEl prompt completo que recibe el tutor (con una hoja de ejemplo):\n")
print("-" * 78)
print(prompt_revision("Revisa mi procedimiento: ¿en qué paso me equivoqué?",
                      {"ok": True, "consistente": True,
                       "json": {"enunciado": "Una caja trae 24 lápices...",
                                "pasos": ["24 × 13.", "24 × 13 = 302."],
                                "respuesta_final": "302"}}))
print("-" * 78)
print("\nLo ÚLTIMO que lee antes de generar es la exigencia del formato. En la")
print("corrida del 22-10 lo último era la respuesta final del estudiante, y ocho de")
print("once respuestas terminaron copiando esa etiqueta en vez de concluir.")
'''


CODIGO_REVISAR_FOTO = r'''
# PUNTO DE ENTRADA DE LA DEMO · una foto -> "tu error está en el paso N"
#                                          o "no pude leer tu hoja".
#
# Requiere los dos modelos a la vez: VLM (sección 5, con MANTENER_VLM = True) y
# tutor (sección 7). El juez no hace falta.
#
# La señal 4 necesita una REFERENCIA cuando llega una sola foto: la confianza de
# 8–10 fotos NÍTIDAS de hojas escritas a mano (las dev-*, por ejemplo), con la
# misma cámara y que NO sean del eval set. El conjunto impreso de calibración no
# sirve de referencia: sus confianzas son casi nulas y marcaría hojas buenas.
from pathlib import Path

DIR_REFERENCIA = Path("m4/referencia_manuscrita")
RUTA_REFERENCIA = Path("m4/referencia_legibilidad.json")
PREGUNTA_DEMO = "Revisa mi procedimiento: ¿en qué paso me equivoqué?"

if "vlm" not in globals():
    print("El VLM se liberó en la sección 5. Para la demo: MANTENER_VLM = True, "
          "y ejecuten de nuevo desde la sección 5.")

REFERENCIA = None
if RUTA_REFERENCIA.exists():
    REFERENCIA = json.loads(RUTA_REFERENCIA.read_text(encoding="utf-8"))
elif DIR_REFERENCIA.exists() and "vlm" in globals():
    fotos_ref = sorted(f for f in DIR_REFERENCIA.iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    conf_ref = [leer_dos_veces(cargar_imagen(f), f.name)["confianza_numeros"]
                for f in fotos_ref]
    REFERENCIA = referencia_de_confianza(conf_ref)
    REFERENCIA["fotos"] = [f.name for f in fotos_ref]
    RUTA_REFERENCIA.write_text(json.dumps(REFERENCIA, indent=2), encoding="utf-8")
if REFERENCIA:
    print(f"Referencia de la señal 4: n = {REFERENCIA['n']}, "
          f"umbral = {REFERENCIA['umbral']:.3f}")
else:
    print(f"Sin referencia: suban fotos nítidas a mano a {DIR_REFERENCIA}/. "
          "Mientras tanto la señal 4 no decide y quedan las otras tres.")


def revisar_foto_demo(ruta, pregunta=PREGUNTA_DEMO):
    ruta = Path(ruta)
    return revisar_foto(
        cargar_imagen(ruta), pregunta,
        leer=leer_dos_veces,
        diagnosticar=lambda p, lec: sistema_multimodal(p, lec, clave=ruta.name),
        referencia=REFERENCIA, archivo=ruta.name)


if "vlm" in globals():
    for nombre in ("vis-01.jpg", "vis-07.jpg"):
        r = revisar_foto_demo(DIR_IMAGENES / nombre)
        print(f"\n{nombre}: {r['estado']} · {r['diagnostico']} · "
              f"lectura {r['segundos_lectura']} s · total {r['segundos_total']} s")
        print("  Lo que leí:", *r["lectura"], sep="\n    ")
        print("  ", r["respuesta"][:300])
'''


CODIGO_ENSAYO_DEV = r'''
# ENSAYO sobre los 3 casos de DESARROLLO · antes de tocar los 12 de evaluación.
#
# Para qué: la pregunta "¿el tutor emite la línea `Diagnóstico:`?" es de formato,
# y responderla mirando los 12 casos de evaluación es el camino directo al
# sobreajuste: se acaba retocando el prompt hasta que esos 12 salgan bien.
#
# Los casos `dev-*` existen para esto. No tienen foto, así que la transcripción se
# construye a partir de los pasos que el propio eval set declara: no se mide la
# lectura, se mide **si el formato se sostiene**. Son 3 casos: es una señal, no una
# medición, y por eso no entra en ningún scorecard.
def lectura_sintetica(caso):
    """La transcripción que el VLM habría producido si hubiera leído perfecto."""
    return {"ok": True, "consistente": True, "crudo": "(sintética)", "segundos": 0.0,
            "json": {"enunciado": caso["enunciado"], "pasos": list(caso["pasos"]),
                     "respuesta_final": caso["valor_final_escrito"]}}


print("Ensayo de formato sobre los casos de desarrollo (no entra en el scorecard)\n")
conclusiones = []
for caso in dev_m4:
    respuesta = sistema_multimodal(caso["pregunta"], lectura_sintetica(caso),
                                   clave=caso["id"])
    ev = calificar_m4(caso, respuesta)
    conclusiones.append((caso["id"], ev["conclusion"]))
    print(f"  {caso['id']:8s} {caso['bloque']:13s} esperaba {ev['esperado']!r:18s} "
          f"concluyó {str(ev['conclusion'])!r:18s} "
          f"{'(reintento)' if REINTENTOS.get(caso['id']) else ''}")

PUERTA_DEV = revisar_puerta_dev(conclusiones)
print(f"\nConclusión válida en {PUERTA_DEV['validos']}/{PUERTA_DEV['total']} "
      "casos de desarrollo.\n")

# Y ahora la puerta DETIENE. En la corrida del 24-10 esto era un `print`: dio 0/3,
# dijo "PAREN AQUÍ" y la ejecución siguió hasta evaluar los 12 casos.
exigir_puerta_dev(PUERTA_DEV, permitir_sin_puerta=globals().get("PERMITIR_SIN_PUERTA", False))
print("El formato se sostiene en desarrollo. Sigan a los 12 casos de evaluación.")
'''

CODIGO_CALIFICAR = r'''
# Cómo se obtiene el resultado de cada caso.
#
# El tutor emite una última línea `Diagnóstico: ...` y **es esa línea la que se
# califica**. La explicación se conserva y se mide aparte.
#
# Por qué: en la corrida del 22-10 el criterio buscaba los textos prohibidos como
# subcadena de TODA la respuesta. Como la respuesta empieza repasando el
# procedimiento ("Paso 1: Correcto. Paso 2: Incorrecto..."), los doce casos
# fallaron con "cayó en la trampa: 'paso 1'". `vis-05` llegó a concluir el paso
# correcto y se contó como fallo igualmente. Eso medía la redacción, no el
# criterio del tutor.
#
# Se siguen registrando tres señales independientes, como en M2 y M3, más dos
# sobre la explicación que son diagnóstico y nunca deciden el acierto.

# La puerta, otra vez y aquí: así la parada no depende de cómo el entorno trate
# una excepción en otra celda. Si alguien ejecuta esta celda directamente sin
# haber pasado el ensayo, se detiene igual.
exigir_puerta_dev(globals().get("PUERTA_DEV"),
                  permitir_sin_puerta=globals().get("PERMITIR_SIN_PUERTA", False))

filas = []
for caso in eval_m4:
    logico = archivo_de(caso)
    if PEDIDAS[logico]["archivo"] is None:
        filas.append({"id": caso["id"], "bloque": caso["bloque"],
                      "criterio": caso["criterio_s13"], "estado": "sin_imagen",
                      "esperado": diagnostico_esperado(caso),
                      "acierto": None, "motivo": f"falta la foto de {logico}"})
        print(f"  {caso['id']:8s} {caso['bloque']:13s} ---  sin imagen")
        continue

    lectura = lecturas.get(logico)
    t0 = time.perf_counter()
    respuesta = sistema_multimodal(caso["pregunta"], lectura, clave=caso["id"])
    seg_tutor = time.perf_counter() - t0

    ev = calificar_m4(caso, respuesta)
    fiel = lectura_fiel(caso, lectura)
    filas.append({
        "id": caso["id"], "bloque": caso["bloque"], "criterio": caso["criterio_s13"],
        "estado": "evaluado",
        "esperado": ev["esperado"], "conclusion": ev["conclusion"],
        "acierto": ev["acierto"], "abstuvo": ev["abstuvo"], "abstuvo_bien": ev["abstuvo_bien"],
        "alucino": ev["alucino"], "sin_conclusion": ev["sin_conclusion"],
        "conclusion_invalida": ev["conclusion_invalida"],
        "reintento": bool(REINTENTOS.get(caso["id"])),
        "lectura_no_fiable": lectura_no_fiable(lectura)[0],
        "motivo": ev["motivo"],
        # La explicación, medida aparte: nunca decide el acierto.
        "explicacion_pasos": " ".join(map(str, ev["explicacion_pasos"])),
        "explicacion_justifica": ev["explicacion_justifica"],
        "lectura_fiel": fiel,
        "fuente_del_fallo": fuente_del_fallo(ev["acierto"], fiel),
        "segundos_lectura": (lectura or {}).get("segundos", 0.0),
        "segundos_tutor": round(seg_tutor, 2),
        "segundos_total": round(seg_tutor + (lectura or {}).get("segundos", 0.0), 2),
        "respuesta": respuesta[:700],
    })
    marca = "OK " if ev["acierto"] else ("abs" if ev["abstuvo"] else "MAL")
    print(f"  {caso['id']:8s} {caso['bloque']:13s} {marca}  {ev['motivo'][:56]}")

M4 = pd.DataFrame(filas)
evaluados = M4[M4.estado == "evaluado"]
fallos = evaluados[~evaluados.acierto.astype(bool)]
print(f"\n{len(evaluados)} de {len(eval_m4)} casos evaluados "
      f"({len(M4) - len(evaluados)} sin imagen, fuera de toda media).")
'''


CODIGO_SCORECARD = r'''
from pathlib import Path

# Scorecard de M4, por bloque. Nunca un promedio global de los cuatro: miden
# cosas distintas y la media no significaría nada.
QUE_MIDE = {
    "localizacion": "nombrar el paso equivocado",
    "compensado":   "paso mal con el resultado final bien",
    "sin_error":    "NO inventar un error donde no lo hay",
    "adversarial":  "abstenerse: ilegible, fuera de dominio, dato ausente",
}


def bloque(nombre):
    g = evaluados[evaluados.bloque == nombre]
    return len(g), int(g.acierto.sum()), (float(g.segundos_total.mean()) if len(g) else 0.0)


print("=" * 78)
print("SCORECARD M4 · casos CON IMAGEN · tarea nueva: localizar el paso erróneo")
print("Se califica la línea `Diagnóstico:`; la explicación se mide aparte.")
print("=" * 78)
print(f"{'bloque':15s}{'aciertos':>11s}{'seg/caso':>11s}   qué mide")
for b, que in QUE_MIDE.items():
    n, ok, seg = bloque(b)
    if n:
        print(f"{b:15s}{ok:>7d}/{n:<3d}{seg:>11.1f}   {que}")
    else:
        print(f"{b:15s}{'—':>11s}{'—':>11s}   {que}  (sin fotos)")
print("-" * 78)
if len(evaluados):
    print(f"{'TOTAL':15s}{int(evaluados.acierto.sum()):>7d}/{len(evaluados):<3d}"
          f"{evaluados.segundos_total.mean():>11.1f}")
    print(f"{'alucinaciones':15s}{int(evaluados.alucino.sum()):>11d}   "
          "concluyó algo concreto y equivocado")
    print(f"{'abstenciones':15s}{int(evaluados.abstuvo.sum()):>11d}   "
          f"de las cuales acertadas: {int(evaluados.abstuvo_bien.sum())}")
    print(f"{'sin conclusión':15s}{int(evaluados.sin_conclusion.sum()):>11d}   "
          "no emitió la línea `Diagnóstico:`")
    print(f"{'no válida':15s}{int(evaluados.conclusion_invalida.sum()):>11d}   "
          "la emitió en texto libre, fuera de las cinco categorías")
    print(f"{'con reintento':15s}{int(evaluados.reintento.sum()):>11d}   "
          "necesitaron una segunda pasada para concluir")
    print(f"{'lectura dudosa':15s}{int(evaluados.lectura_no_fiable.sum()):>11d}   "
          "el tutor recibió 'no se pudo leer', no un procedimiento")
print("=" * 78)

# El fallo de formato se reporta aparte porque tiene arreglo distinto: si domina,
# el problema es que la LoRA de M1 arrastra al modelo a continuar el procedimiento
# en vez de emitir un veredicto, y se ataca con el prompt o con entrenamiento, no
# con un modelo mayor.
sin_forma = int((evaluados.sin_conclusion | evaluados.conclusion_invalida).sum())
if sin_forma:
    print(f"\nAVISO: {sin_forma} de {len(evaluados)} respuestas no dan una conclusión")
    print("válida ni después del reintento. Mientras eso pase, el total mide en parte el")
    print("formato de salida y no el criterio del tutor: miren una respuesta completa")
    print("antes de leer el scorecard.")

# Un predictor constante acierta por construcción los casos cuya respuesta es la
# que repite siempre. Si eso pasa, el total no es evidencia de capacidad.
conc = evaluados[evaluados.conclusion.notna()].conclusion
if len(conc):
    top, veces = conc.value_counts().index[0], conc.value_counts().iloc[0]
    if veces >= 0.6 * len(conc):
        print(f"\nAVISO: {veces} de {len(conc)} conclusiones son {top!r}.")
        print("El tutor está respondiendo casi siempre lo mismo, así que los aciertos")
        print("coinciden con los casos cuya respuesta es justamente esa. Cuenten cuántos")
        print("acertaría un sistema que contestara siempre eso antes de leer el total:")
        baseline = int((evaluados.esperado == top).sum())
        print(f"  un predictor constante {top!r} acertaría {baseline}/{len(evaluados)}.")

print("\nFUENTE DEL FALLO · solo los casos fallados")
if len(fallos):
    for fuente, n in fallos.fuente_del_fallo.value_counts().items():
        ids = ", ".join(fallos[fallos.fuente_del_fallo == fuente].id)
        print(f"  {fuente:12s} {n:2d}   {ids}")
    print("\n  lectura     -> el VLM no transcribió lo que la hoja decía: el cuello es ver")
    print("  diagnostico -> la transcripción llegó bien y el tutor falló igual")
else:
    print("  (sin fallos)")

print("\nLA EXPLICACIÓN, medida aparte · no decide ningún acierto")
coh = evaluados.explicacion_justifica.astype(bool)
print(f"  argumentan su conclusión : {int(coh.sum())}/{len(evaluados)}")
print("    (la explicación habla del paso que concluyó, o no menciona ninguno)")
incoh = evaluados[~coh]
if len(incoh):
    print("  conclusiones sin argumentar en el texto:")
    for _, f in incoh.iterrows():
        print(f"    {f['id']:8s} concluyó {str(f['conclusion']):16s} "
              f"y la explicación habla de los pasos [{f['explicacion_pasos']}]")

Path("m4").mkdir(exist_ok=True)
M4.to_csv("m4/scorecard_m4.csv", index=False)
print("\nGuardado: m4/scorecard_m4.csv")
'''


CODIGO_REGRESION = r'''
# REGRESIÓN sobre los 21 casos de M2 · SIN imagen.
#
# Esto NO es la otra mitad de la medición anterior: es otra pregunta. El eval set
# de M2 mide la tarea de siempre (resolver un problema planteado en texto) y aquí
# se vuelve a correr con el MISMO sistema, solo para comprobar que anteponer
# `ver()` no rompió nada. Un cambio aquí es una regresión; un número de M4 y uno
# de M2 no se restan.
sc_regresion = harness(eval_set, lambda p: sistema_multimodal(p, None),
                       nombre="M4 · sin imagen")
imprimir_scorecard(sc_regresion)

if EVAL_M2_COMPARABLE:
    print("\nMismo eval set de 21 casos, dos corridas del mismo sistema:")
    print(f"  en M3 · {REFERENCIA_M3['nombre']:22s} {REFERENCIA_M3['aciertos']}/21")
    print(f"  ahora · M4                         {sc_regresion['aciertos']}/21")
    print("\nLectura: si coinciden, la visión no tocó lo que ya funcionaba. Si bajó, hay")
    print("que decir en qué caso y por qué ANTES de hablar de lo que se ganó en imagen.")
    print("Con n = 21, una diferencia de un caso es ruido; de tres o más, hay que mirarla.")
else:
    print(f"\nM4 sin imagen: {sc_regresion['aciertos']}/21  —  NO COMPARABLE con M3.")
    print("El eval set en disco no es el que midió M3 (ver la guarda de la sección 3.2).")
    print("Esta cifra sirve para ver si el sistema responde, no para comparar con nada.")
'''


CODIGO_DELTA = r'''
# La tabla que pide la rúbrica: M3 contra M4, con las dos partes separadas y
# etiquetadas, y la latencia al lado.
adv = evaluados[evaluados.bloque == "adversarial"]
con_img = evaluados[evaluados.bloque != "adversarial"]
peor = (fallos.iloc[0]["id"] if len(fallos) else "sin fallos")

DELTA = pd.DataFrame([
    {"grupo": "casos sin imagen · eval set de M2 (21 casos)",
     "M3": (f"{REFERENCIA_M3['aciertos']}/21" if EVAL_M2_COMPARABLE
            else "no comparable: otro eval set en disco"),
     "M4": f"{sc_regresion['aciertos']}/21",
     "qué significa": ("REGRESIÓN: ¿se rompió lo que ya funcionaba?" if EVAL_M2_COMPARABLE
                       else "NO comparable: el eval set en disco no es el de M3")},
    {"grupo": "casos con imagen · eval set de M4 (tarea nueva)",
     "M3": "no aplica: M3 no recibe imágenes",
     "M4": f"{int(con_img.acierto.sum())}/{len(con_img)}",
     "qué significa": "CAPACIDAD NUEVA: no hay baseline comparable"},
    {"grupo": "adversariales visuales",
     "M3": "no aplica",
     "M4": f"{int(adv.acierto.sum())}/{len(adv)}",
     "qué significa": "¿se negó cuando la hoja era ilegible o no era matemáticas?"},
    {"grupo": "latencia por caso",
     "M3": f"{REFERENCIA_M3['segundos']:.1f} s",
     "M4": f"{evaluados.segundos_total.mean():.1f} s "
           f"({evaluados.segundos_lectura.mean():.1f} s son del VLM)",
     "qué significa": "el precio de ver"},
    {"grupo": "peor caso",
     "M3": "—", "M4": peor,
     "qué significa": "se explica en m4/README.md"},
])
DELTA.to_csv("m4/delta_m3_m4.csv", index=False)
print(DELTA.to_string(index=False))

print("\nLas dos primeras filas son CONJUNTOS DISTINTOS. La primera dice si el sistema")
print("sigue respondiendo igual de bien lo de siempre; la segunda, qué tan bien hace algo")
print("que antes no podía hacer. No se restan, y no existe 'la mejora de M3 a M4' como un")
print("número único.")
print("\nGuardado: m4/delta_m3_m4.csv")
'''


CODIGO_ARTEFACTOS = r'''
json.dump({
    "patron": "input",
    "vlm": VLM_ID,
    "max_pixels": MAX_PIXELS,
    "lado_mayor_px": LADO_MAYOR,
    "adaptador_tutor": ADAPTADOR_TUTOR,
    "adaptador_via_visual": ADAPTADOR_DIAGNOSTICO,
    "adaptador_diagnostico": DIAGNOSTICO_MOTIVO,
    "referencia_m3": REFERENCIA_M3,
    "n_casos": len(eval_m4),
    "n_imagenes_necesarias": len(PEDIDAS),
    "imagenes_faltantes": sorted(FALTAN),
    "n_evaluados": int(len(evaluados)),
    "visual_por_bloque": {b: {"n": bloque(b)[0], "aciertos": bloque(b)[1]} for b in QUE_MIDE},
    "visual_total": {"n": int(len(evaluados)),
                     "aciertos": int(evaluados.acierto.sum()),
                     "alucinaciones": int(evaluados.alucino.sum()),
                     "abstenciones": int(evaluados.abstuvo.sum())},
    "fuente_del_fallo": (fallos.fuente_del_fallo.value_counts().to_dict() if len(fallos) else {}),
    "eval_m2_comparable_con_m3": bool(EVAL_M2_COMPARABLE),
    "regresion_m2": {k: v for k, v in sc_regresion.items() if k != "detalle"},
    "conclusiones": {str(k): int(v) for k, v in
                     evaluados.conclusion.fillna("sin_conclusion").value_counts().items()},
    "puerta_dev": ("superada" if (globals().get("PUERTA_DEV") or {}).get("ok")
                   else "NO superada"),
    "sin_conclusion": int(evaluados.sin_conclusion.sum()),
    "conclusion_invalida": int(evaluados.conclusion_invalida.sum()),
    "reintentos": int(evaluados.reintento.sum()),
    "lecturas_no_fiables": sorted(evaluados[evaluados.lectura_no_fiable].id),
    "abstuvo_bien": int(evaluados.abstuvo_bien.sum()),
    "explicacion_justifica": int(evaluados.explicacion_justifica.astype(bool).sum()),
    "archivos_leidos": {k: v.get("archivo", k) for k, v in lecturas.items()},
    "latencia_total_s": round(float(evaluados.segundos_total.mean()), 2),
    "latencia_lectura_s": round(float(evaluados.segundos_lectura.mean()), 2),
}, open("m4/resultados_m4.json", "w", encoding="utf-8"),
    ensure_ascii=False, indent=2, default=float)

pd.DataFrame(sc_regresion["detalle"]).to_csv("m4/detalle_regresion_m4.csv", index=False)

for p in ("m4/scorecard_m4.csv", "m4/delta_m3_m4.csv", "m4/detalle_regresion_m4.csv",
          "m4/resultados_m4.json", "m4/lecturas_m4.json"):
    print(" ", p)
print("\nFalta m4/README.md: se escribe con estos números a la vista, no antes.")
print("Si quedó alguna imagen sin subir, el scorecard está incompleto y hay que decirlo.")
'''


def construir() -> list[dict]:
    c: list[dict] = []

    c.append(encabezado(
        "El componente visual, enchufado y medido",
        "Módulo 4 · El tutor recibe la foto de un procedimiento y señala el paso erróneo",
    ))

    c.append(md("""
## 1 · Qué hace este notebook

**La pregunta del usuario que necesita ojos:** *"¿En qué paso me equivoqué?"*

El procedimiento está escrito a mano en el cuaderno. Hoy el tutor solo lee texto,
así que el estudiante tendría que transcribirlo — y al transcribir corrige el
error sin darse cuenta, con lo que el sistema nunca llega a ver el fallo que debía
encontrar.

**El patrón es `input`.** Conviene decir por qué se descartaron los otros dos:

| Patrón | Decisión | Razón |
|---|---|---|
| **input** | **Elegido** | La pregunta necesita el *contenido* de la hoja como pasos de texto, y el sistema de M3 ya consume texto: no hay que reescribirlo |
| tool | Descartado | Un clasificador de imagen devuelve una clase, no una secuencia de pasos: no alcanza para localizar uno |
| output | Descartado | Dibujar el paso corregido no cambia ninguna decisión del estudiante, que lo necesita explicado |

**La decisión que ordena el experimento: separar leer de diagnosticar.** Son dos
etapas con dos modos de fallo distintos, y una sola cifra los confunde. Por eso
cada caso registra además si la transcripción fue fiel: así un fallo se atribuye a
*los ojos* o al *razonamiento*, no a "el sistema".

> **Este notebook no entrena nada.** Es inferencia y evaluación.
"""))

    c.append(md("""
## 2 · Los dos conjuntos, que miden cosas distintas

Esto condiciona cómo se lee todo lo que viene después.

| Conjunto | n | Qué mide | Qué significa una cifra alta |
|---|---|---|---|
| **Eval set de M4** | 12 casos **con imagen** | Una tarea **nueva**: localizar el paso erróneo en una hoja fotografiada | El sistema puede hacer algo que antes no podía |
| **Eval set de M2** | 21 casos **sin imagen** | La tarea **de siempre**: resolver un problema planteado en texto | Anteponer la visión no rompió lo que ya funcionaba |

Son **conjuntos distintos, con criterios distintos**. Se reportan los dos, cada
uno etiquetado, pero **no se restan**: no existe "la mejora de M3 a M4" como un
número único, porque M3 no podía recibir imágenes.

### El orden de la corrida, y por qué

| Fase | Qué ocupa la GPU | Qué queda en disco |
|---|---|---|
| **1 · leer** | Qwen2-VL-2B (~4.4 GB en fp16) | `m4/lecturas_m4.json` |
| *liberar* | — | |
| **2 · diagnosticar** | tutor de M3 (~3.1 GB) + embeddings + juez | `m4/scorecard_m4.csv` |

Un VLM de 2B y el tutor de 1.5B con su juez no caben cómodos en una T4 de 16 GB.
Y como la transcripción queda guardada, la fase 2 se puede repetir sin volver a
leer las fotos, que es la parte lenta.
"""))

    c.append(md("""
---
## 0 · Preparación del entorno

> **Activen una GPU.** Sirve cualquiera con CUDA —T4, G4, L4, A100, RTX—; el
> notebook imprime la que detecte y no exige ningún modelo concreto. Carga dos
> modelos grandes, pero **en momentos distintos**: la celda de liberación que va
> entre ellos no es opcional, y con menos de 14 GB es imprescindible.
"""))
    c.append(code('''
%pip install -q "transformers==5.20.2" "peft>=0.12" "accelerate>=0.33" \\
    sentence-transformers chromadb rank_bm25 "scikit-learn>=1.3" sympy pandas \\
    pillow qwen-vl-utils
%pip uninstall -y -q torchao
print("Listo. Si Colab pide reiniciar la sesión, reinícienla y sigan desde aquí.")
'''))
    c.append(code('''
import os, random, re
import numpy as np
import pandas as pd
import torch
import transformers

SEMILLA = 42
random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
transformers.set_seed(SEMILLA)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"transformers {transformers.__version__} | torch {torch.__version__} | {DEVICE}")

# Sirve CUALQUIER GPU CUDA: T4, G4, L4, A100, RTX… No se comprueba el modelo,
# solo que haya CUDA, y se imprime el dispositivo detectado. La corrida del 22-10
# usó una RTX PRO 6000 de 102 GB; exigir "Tesla T4" habría sido un falso bloqueo.
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    VRAM_GB = props.total_memory / 1e9
    print(f"GPU detectada: {props.name} ({VRAM_GB:.0f} GB)")
    if VRAM_GB < 14:
        print(f"  Aviso: con {VRAM_GB:.0f} GB conviene no tener los dos modelos a la vez. "
              "La celda que libera el VLM (5.2) es obligatoria.")
else:
    print("Sin GPU: el notebook corre, pero leer 11 imágenes en CPU tarda mucho. "
          "Activen un entorno de ejecución con GPU (cualquiera con CUDA sirve).")
'''))
    c.extend(celda_drive())

    c.append(md("""
---
## 3 · Los datos

Los eval sets viajan dentro del notebook como blob comprimido y se verifican por
SHA-256. Es la salvaguarda que faltó en S08: aquella corrida usó la copia de Drive
con las claves viejas de `adv-02` y hubo que recalcular el criterio después.
"""))
    c.append(md("""
### 3.1 · El módulo de visión

`scripts/agente/vision.py`, pegado tal cual, igual que el resto del núcleo. Aquí
está lo que puede equivocarse **en silencio**: el alias de `vis-12`, el inventario
de fotos que faltan, el parser tolerante del JSON, el formato con el que el tutor
ve la transcripción y la atribución del fallo a la lectura o al diagnóstico.

Está en un módulo y no en una celda porque se prueba con
`python -m pytest scripts/tests/test_m4_vision.py`, **sin GPU y sin fotos**. Lo
que no se puede probar así — la pasada del VLM y el redimensionado — se queda en
el notebook.
"""))
    c.extend(celdas_modulos(["vision"], con_descripcion=False))
    c.extend(celda_eval_set(prefijo="3.2"))
    c.append(code(CODIGO_GUARDA_EVAL_M2))
    c.extend(celda_eval_m4(prefijo="3.3"))

    c.append(md("""
### 3.4 · Las imágenes

**11 archivos para 12 casos**: `vis-12` reutiliza la foto de `vis-07` porque lo
que cambia no es la hoja sino la pregunta — se pide un dato que no está escrito en
ella.

Las hojas para escribir a mano están en `m4/eval_set/HOJAS_PARA_ESCRIBIR.md`: cada
una dice qué copiar y, en los casos con error, **qué error cometer a propósito**.
Las fotos van en `m4/eval_set/imagenes/` con exactamente los nombres del eval set.

El emparejamiento es por **nombre base**: el eval set dice `vis-10.jpg` y un
archivo `vis-10.jpeg` o `vis-10.png` lo satisface igual. Si hay **dos** archivos
con el mismo nombre base, la celda se detiene en vez de elegir uno: si las fotos
fueran distintas, la corrida mediría una hoja y el informe hablaría de la otra.

Si falta alguna, el notebook sigue y marca esos casos como `sin_imagen`. **No
entran en ninguna media**: una foto que no se subió no es un fallo del sistema, y
contarla como cero inventaría un resultado.
"""))
    c.append(code(CODIGO_IMAGENES))

    c.append(md("""
---
## 4 · Métricas y criterio de acierto

Las mismas funciones de M2 y M3, leídas del mismo sitio. El criterio de cada caso
viene declarado en el eval set, escrito antes de ejecutar.
"""))
    c.append(code(CODIGO_METRICAS))
    c.extend(celda_similitud())
    c.extend(celda_evaluacion())

    c.append(md("""
---
## 5 · Etapa 1 · `ver()` — la foto se convierte en pasos

El VLM **solo transcribe**, y se le pide explícitamente que **no corrija**: un
modelo instruido tiende a "arreglar" la cuenta mal al copiarla, y entonces el
error que había que encontrar desaparece antes de llegar al tutor.
"""))
    c.append(code(CODIGO_VER))

    c.append(md("""
### 5.1 · Calibración de la detección de hojas ilegibles

Con un conjunto **aparte del eval set**: 6 hojas sintéticas × 3 versiones
(nítida, borrosa leve, borrosa fuerte). Mide si la salvaguarda separa, sin
convertir `vis-10` en dato de ajuste.

El límite es grande y está dicho en la celda: es texto **impreso**. Un buen
resultado aquí no autoriza a afirmar que funciona con letra manuscrita.
"""))
    c.append(code(CODIGO_CALIBRACION))

    c.append(md("""
### 5.2 · Puerta piloto · tres imágenes

Antes de gastar la corrida completa. El riesgo que puede costar la entrega es que
un modelo de 2B no lea letra manuscrita; más vale descubrirlo en tres fotos que a
los cuarenta minutos. Las tres están elegidas para que cada una pueda fallar de
una forma distinta, y la celda dice qué hacer en cada caso.
"""))
    c.append(code(CODIGO_PUERTA_PILOTO))

    c.append(md("""
### 5.3 · Transcripción completa

Se guarda en disco **antes** de soltar el VLM: si Colab se desconecta no se pierde
nada, y la fase de diagnóstico se puede repetir sin volver a leer las fotos.
"""))
    c.append(code(CODIGO_TRANSCRIBIR))
    c.append(code(CODIGO_LIBERAR))

    c.append(md("""
---
## 6 · Etapa 2 · El sistema de M3, sin reescribir

Corpus, chunking y retrieval idénticos a S07/S08; el agente con herramientas,
idéntico a S10. El código sale de `scripts/agente/*.py`, donde está probado con
`pytest`: lo que se prueba es exactamente lo que se ejecuta.

Lo único nuevo es que la pregunta llega acompañada de lo que se leyó en la foto,
y que **se le exige una conclusión explícita**.

### El formato de salida, y por qué se cambió

En la corrida del 22-10 (archivada en `m4/corridas/corrida-01_2026-10-22/`) el
tutor no revisaba: **reescribía** el procedimiento en el formato `Paso 1: …
Respuesta final: …` con el que se entrenó la LoRA de M1. Ante la hoja de `vis-02`
respondió el ejercicio bien resuelto, sin decir nunca qué paso estaba mal.

Así que ahora la última línea tiene que ser una de estas cinco:

| Conclusión | Cuándo | Bloque del eval set |
|---|---|---|
| `Diagnóstico: Paso N` | un paso está mal | `localizacion`, `compensado` |
| `Diagnóstico: sin error` | todos los pasos están bien | `sin_error` |
| `Diagnóstico: ilegible` | la hoja no se puede leer | `adversarial` (`vis-10`) |
| `Diagnóstico: fuera de dominio` | la hoja no es matemáticas | `adversarial` (`vis-11`) |
| `Diagnóstico: dato no disponible` | se pide algo que no está escrito | `adversarial` (`vis-12`) |

**El criterio lee solo esa línea.** La explicación se conserva y se mide aparte,
nunca como acierto: era lo que invalidaba la corrida anterior.
"""))
    c.append(code(CODIGO_CORPUS))
    c.append(code(CODIGO_CHUNKING))
    c.append(code(CODIGO_INDICE))
    c.extend(celdas_modulos(["registro", "herramientas", "nucleo", "documentos", "prompts"],
                            con_descripcion=False))
    c.append(code(CODIGO_LLM))
    c.append(md("""
### 6.1 · El adaptador de diagnóstico

`CODIGO_LLM` es compartido con los notebooks de S10 y solo carga `m1` y
`agente`. El de diagnóstico se carga aquí, **comprobando el disco**: en la
corrida del 25-10 la celda siguiente afirmaba que no estaba, con los dos
archivos presentes en Drive, porque la condición miraba si estaba *cargado* y
nadie lo cargaba.
"""))
    c.append(code(CODIGO_ADAPTADOR_DIAGNOSTICO))
    c.append(code(CODIGO_RESPONDER_M3))
    c.append(code(CODIGO_SISTEMA_MULTIMODAL))

    c.append(md("""
---
## 7 · Evaluación de los casos con imagen
"""))
    c.append(md("""
### 7.1 · Ensayo de formato sobre los casos de desarrollo

Primero los 3 casos `dev-*`, con transcripción sintética. Mide **solo si el
formato se sostiene**, y no entra en ningún scorecard: es la forma de decidir
sobre el prompt sin convertir los 12 casos de evaluación en conjunto de
entrenamiento.
"""))
    c.append(code(CODIGO_ENSAYO_DEV))
    c.append(md("""
### 7.2 · Los 12 casos de evaluación
"""))
    c.append(code(CODIGO_CALIFICAR))
    c.append(code(CODIGO_SCORECARD))

    c.append(md("""
---
## 8 · Regresión sobre los 21 casos de M2 · sin imagen

Otra pregunta, no la otra mitad de la anterior. Se corre el **mismo**
`sistema_multimodal`, sin imagen, sobre el eval set que ya midió M1, M2, S07, S08
y S10.
"""))
    c.extend(celda_juez())
    c.extend(celda_harness())
    c.append(code(CODIGO_REGRESION))

    c.append(md("""
---
## 9 · La tabla de delta
"""))
    c.append(code(CODIGO_DELTA))

    c.append(md("""
---
## 10 · Artefactos
"""))
    c.append(code(CODIGO_ARTEFACTOS))

    c.append(md("""
---
## 11 · Cómo leer esto

**1 · Primero la fuente del fallo, no el total.** Si la mayoría de los fallos son
de `lectura`, el cuello de botella es el VLM y se arregla con preproceso, un OCR
clásico o un modelo mayor, sin tocar el tutor. Si son de `diagnostico`, la
transcripción llegó bien y el límite es el razonamiento del modelo de 1.5B — el
mismo límite que ya salió en M3, ahora con una entrada nueva.

**2 · El bloque `sin_error` antes que el de aciertos.** Un tutor que no encuentra
un error deja al estudiante como estaba; uno que **inventa** un error le dice que
está mal cuando está bien. En un tutor ese es el fallo grave, igual que las
alucinaciones en M2.

**3 · El bloque `compensado` es el que justifica el módulo.** Paso equivocado con
resultado final correcto: un corrector que solo mira el número final los aprueba.
Son 2 casos, así que es una señal, no una medición.

**4 · Las dos primeras filas de la tabla de delta no se restan.** Una dice si el
sistema sigue respondiendo igual de bien lo de siempre; la otra, qué tan bien hace
algo que antes no podía. Con n = 12 y bloques de 2 a 4 casos, ninguna diferencia
pequeña es concluyente, y conviene decirlo antes de que lo pregunten.

**5 · La latencia incluye la pasada del VLM.** Es el precio de ver, y va al lado
de cualquier mejora que se reporte.

**6 · Si faltó alguna foto, el scorecard está incompleto.** Los casos
`sin_imagen` quedan fuera de las medias, y el total hay que leerlo sobre los casos
que sí se evaluaron, no sobre 12.
"""))

    c.append(md("""
---
## 12 · Punto de entrada para la demo: `revisar_foto_demo`

Una foto y una pregunta, de punta a punta: leer, decidir si la lectura es
fiable y solo entonces diagnosticar. Si no es fiable, la respuesta es *"no pude
leer tu hoja"* en lugar de un diagnóstico sobre un procedimiento inventado.

No entra en ningún scorecard. Es la función sobre la que se monta la interfaz.
"""))
    c.append(code(CODIGO_REVISAR_FOTO))

    return c
