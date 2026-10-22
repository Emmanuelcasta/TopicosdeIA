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
# GUARDA: ¿es este el mismo eval set con el que se midió M3?
#
# La corrida del 22-10 imprimió `SHA coincide: False` y nadie se detuvo. Eso
# significa que en Drive había una copia distinta de `data/eval_set_m2.jsonl`
# —la de antes de corregir el criterio de `adv-02`— así que su 15/21 NO es
# comparable con el 13/21 de M3, que se midió con la versión corregida.
#
# No se compara contra un número de M3 lo que no se midió con la misma vara.
EVAL_M2_COMPARABLE = (sha_real == SHA_EVAL)
if EVAL_M2_COMPARABLE:
    print("El eval set de M2 es el mismo con el que se midió M3: la regresión es comparable.")
else:
    print("=" * 78)
    print("ATENCIÓN · el eval set de M2 en disco NO es el que se usó para medir M3.")
    print("=" * 78)
    print(f"  esperado (embebido): {SHA_EVAL[:16]}…")
    print(f"  encontrado en disco: {sha_real[:16]}…")
    print()
    print("La sección 8 se ejecutará igual, pero su cifra queda marcada como NO")
    print("comparable y no se pone al lado de la de M3.")
    print()
    print("Para arreglarlo: borren data/eval_set_m2.jsonl y vuelvan a ejecutar la celda")
    print("anterior, que reescribe la copia embebida y verificada.")
'''

CODIGO_VER = r'''
import json
import time

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

VLM_ID = "Qwen/Qwen2-VL-2B-Instruct"
MAX_PIXELS = 1024 * 1024          # tope de tokens visuales por imagen

vlm_proc = AutoProcessor.from_pretrained(VLM_ID, max_pixels=MAX_PIXELS)
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    VLM_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None).eval()
print("VLM cargado:", VLM_ID)
if DEVICE == "cuda":
    print(f"GPU ocupada: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# La instrucción (INSTRUCCION_VER) y el parser tolerante (json_de) vienen del
# módulo `vision.py` de la celda 3.1, donde están probados.
print("\nInstrucción de lectura:\n", INSTRUCCION_VER)


@torch.no_grad()
def ver(imagen, instruccion=INSTRUCCION_VER, max_new_tokens=320):
    """imagen -> (texto crudo, dict o None, segundos). La etapa 1 del pipeline."""
    t0 = time.perf_counter()
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": instruccion}]}]
    prompt = vlm_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ent = vlm_proc(text=[prompt], images=[imagen], return_tensors="pt").to(vlm.device)
    out = vlm.generate(**ent, max_new_tokens=max_new_tokens, do_sample=False)
    texto = vlm_proc.batch_decode(out[:, ent["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
    return texto, json_de(texto), time.perf_counter() - t0
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
    crudo, js, seg = ver(cargar_imagen(DIR_IMAGENES / real))
    lectura = {"crudo": crudo, "json": js, "segundos": round(seg, 2),
               "ok": js is not None, "archivo": real}
    lecturas_piloto[logico] = lectura
    print(f"      archivo: {real} | {seg:.1f}s | json parseado: {js is not None}")

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

for logico, info in sorted(PEDIDAS.items()):
    if logico in lecturas or info["archivo"] is None:
        continue
    crudo, js, seg = ver(cargar_imagen(DIR_IMAGENES / info["archivo"]))
    lecturas[logico] = {"crudo": crudo, "json": js, "segundos": round(seg, 2),
                        "ok": js is not None, "archivo": info["archivo"]}
    print(f"  {info['archivo']:14s} {seg:5.1f}s  {'json ok' if js else 'SIN json'}")

RUTA_LECTURA.write_text(json.dumps(lecturas, ensure_ascii=False, indent=2), encoding="utf-8")
leidas = sum(1 for v in lecturas.values() if v["ok"])
print(f"\n{len(lecturas)} imágenes transcritas, {leidas} con JSON válido -> {RUTA_LECTURA}")
print(f"Latencia media de lectura: {np.mean([v['segundos'] for v in lecturas.values()]):.1f} s")
'''


CODIGO_LIBERAR = r'''
# Liberar la GPU antes de cargar el tutor. La transcripción ya está en disco, así
# que el VLM no vuelve a hacer falta en esta corrida.
import gc

del vlm, vlm_proc
gc.collect()
if torch.cuda.is_available():
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

TRAZAS_M4 = {}


def responder_m3(pregunta):
    salida = agente_m3(pregunta)
    TRAZAS_M4[pregunta[:160]] = salida
    return salida.respuesta


print(f"{len(REGISTRO_M4)} herramientas disponibles para el tutor.")
'''


CODIGO_SISTEMA_MULTIMODAL = r'''
# El enchufe, que es la pieza que M4 evalúa.
#
# Patrón INPUT: la foto entra, el VLM la convierte en texto y el sistema de M3 lo
# consume como contexto. El sistema de M3 no se modifica: se le antepone `ver()`.
#
# `INSTRUCCION_REVISION` y `pasos_como_texto` vienen del módulo `vision.py`.


def sistema_multimodal(pregunta, lectura=None):
    """pregunta (+ lo que se leyó de la foto) -> respuesta del tutor.

    Sin `lectura` es exactamente el sistema de M3: la misma función responde los
    21 casos de la regresión, que no llevan imagen. Que sea la MISMA función es lo
    que hace legítima esa fila.
    """
    if lectura is None:
        return responder_m3(pregunta)
    return responder_m3(f"{INSTRUCCION_REVISION}\n\n{pasos_como_texto(lectura)}\n\n"
                        f"Pregunta: {pregunta}")


print("sistema_multimodal() listo.")
print("  con imagen -> ver() + M3   |   sin imagen -> M3 tal cual")
print("\nAsí ve el tutor una transcripción (ejemplo con datos inventados):\n")
print(pasos_como_texto({"json": {"enunciado": "Una caja trae 24 lápices...",
                                 "pasos": ["24 × 13.", "24 × 13 = 302."],
                                 "respuesta_final": "302"}}))
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
    respuesta = sistema_multimodal(caso["pregunta"], lectura)
    seg_tutor = time.perf_counter() - t0

    ev = calificar_m4(caso, respuesta)
    fiel = lectura_fiel(caso, lectura)
    filas.append({
        "id": caso["id"], "bloque": caso["bloque"], "criterio": caso["criterio_s13"],
        "estado": "evaluado",
        "esperado": ev["esperado"], "conclusion": ev["conclusion"],
        "acierto": ev["acierto"], "abstuvo": ev["abstuvo"], "abstuvo_bien": ev["abstuvo_bien"],
        "alucino": ev["alucino"], "sin_conclusion": ev["sin_conclusion"],
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
print("=" * 78)

# El fallo de formato se reporta aparte porque tiene arreglo distinto: si domina,
# el problema es que la LoRA de M1 arrastra al modelo a continuar el procedimiento
# en vez de emitir un veredicto, y se ataca con el prompt o con entrenamiento, no
# con un modelo mayor.
sin_conc = int(evaluados.sin_conclusion.sum())
if sin_conc:
    print(f"\nAVISO: {sin_conc} de {len(evaluados)} respuestas no traen conclusión.")
    print("Si son la mayoría, lo que se está midiendo es el formato de salida y no el")
    print("criterio del tutor. Antes de leer el scorecard, mirar una respuesta completa.")

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
    "sin_conclusion": int(evaluados.sin_conclusion.sum()),
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
%pip install -q "transformers>=4.45" "peft>=0.12" "accelerate>=0.33" \\
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
### 5.1 · Puerta piloto · tres imágenes

Antes de gastar la corrida completa. El riesgo que puede costar la entrega es que
un modelo de 2B no lea letra manuscrita; más vale descubrirlo en tres fotos que a
los cuarenta minutos. Las tres están elegidas para que cada una pueda fallar de
una forma distinta, y la celda dice qué hacer en cada caso.
"""))
    c.append(code(CODIGO_PUERTA_PILOTO))

    c.append(md("""
### 5.2 · Transcripción completa

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
    c.append(code(CODIGO_RESPONDER_M3))
    c.append(code(CODIGO_SISTEMA_MULTIMODAL))

    c.append(md("""
---
## 7 · Evaluación de los casos con imagen
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

    return c
