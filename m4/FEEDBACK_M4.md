# Retroalimentación de M4 · qué se incorporó y qué sigue pendiente

Nota de M4: **4,6 / 5,0**. Revisado el 30-10-2026, rama `integracion-m5`.

## Advertencia sobre las fuentes

El encargo menciona tres commits de referencia —`5add9f4`, `947354a` y
`444012d`— y **ninguno existe en este repositorio**. Comprobado así:

```
git cat-file -t 5add9f4   ->  fatal: Not a valid object name
git cat-file -t 947354a   ->  fatal: Not a valid object name
git cat-file -t 444012d   ->  fatal: Not a valid object name
```

Tampoco aparecen tras `git fetch --all`, ni en `git reflog --all`, ni en ninguna
de las ramas (`main`, `integracion-m5`, `m4-vision`, `m5-interfaz`,
`mejora-nucleo`) ni en `origin`. No hay en el disco ningún documento de
retroalimentación.

Así que lo que sigue se construyó con **los hechos que vinieron en el encargo**,
no con el contenido de esos commits. Si traen algo más —diffs concretos,
comentarios por archivo—, hay que revisarlo aparte: puede cambiar o añadir
puntos a esta tabla. Lo que **no** se hizo fue adivinar qué decían.

## Los hechos de partida

| Hecho | Qué implica |
|---|---|
| Nota 4,6/5,0 | M4 se aceptó; lo que sigue es refinamiento, no rescate |
| `ver()` reprodujo las **mismas 11 transcripciones** en otra máquina, conservando los errores manuscritos | El lector **no** es la fuente de la irreproducibilidad, y además no «corrige» lo que el estudiante escribió mal, que es lo que la tarea necesita |
| La causa fue una **caché desactualizada** y una versión de `transformers` **sin fijar** | Es un problema de entorno, y se arregla en el entorno |
| Validar las sugerencias en Colab **antes** de incorporarlas | Nada de lo que necesite GPU se da por bueno aquí |
| Conservar los resultados históricos | No se borra ni se regenera ningún notebook ejecutado |
| La configuración final **no** se evaluó con fotos nuevas | No se puede afirmar ninguna mejora de la vía visual |

## Sugerencia por sugerencia, contra el código actual

### 1 · Fijar la versión de `transformers` · **incorporado**

Estaba así en `scripts/nb_m4.py`:

```
%pip install -q "transformers>=4.45" "peft>=0.12" "accelerate>=0.33" ...
```

Un rango sin tope: dos corridas en días distintos podían traer librerías
distintas y dar transcripciones distintas sin que nada lo advirtiera. Ahora:

```
%pip install -q transformers==4.45.2 peft==0.13.2 accelerate==0.34.2 \
    sentence-transformers==3.1.1 rank-bm25==0.2.2 scikit-learn==1.5.2 \
    sympy==1.13.3 numpy==1.26.4 pillow==10.4.0 chromadb pandas qwen-vl-utils
```

Son **las mismas versiones** que `M5_despliegue/config/requirements.txt`, para
que el notebook y el servicio desplegado sean el mismo entorno y no dos.

Y no se confía en que el `pip install` haya funcionado: una celda comprueba las
versiones ya instaladas y **avisa** si no coinciden, porque Colab puede traer una
versión preinstalada que gane.

```
VERSIONES_ESPERADAS = {"transformers": "4.45.2"}
...
  AVISO: transformers 4.46.0 en vez de 4.45.2. Esta corrida NO es comparable
  con las anteriores; reinicia la sesión y repite la celda de instalación.
```

El entorno completo queda guardado **dentro del resultado**, en la clave
`entorno` de `m4/resultados_m4.json`. Sin eso, dos corridas con cifras distintas
no se pueden atribuir ni al modelo ni a las librerías.

### 2 · La caché de modelos · **incorporado**

Celda nueva que imprime qué hay en la caché y la revisión de cada modelo, con un
interruptor explícito para forzar descarga limpia:

```
FORZAR_DESCARGA = False   # ponlo en True solo si una corrida no se reproduce
...
Caché de modelos: /root/.cache/huggingface/hub  (existe: True)
  models--Qwen--Qwen2-VL-2B-Instruct  revisión 895c3a49b2
```

Está en `False` por omisión a propósito: borrar la caché cuesta varios minutos de
descarga en cada corrida, y el problema solo aparece cuando algo no se reproduce.

### 3 · Que `ver()` conserve los errores manuscritos · **ya estaba, no se toca**

Esto no era una corrección: era una comprobación que salió bien. `ver()`
reprodujo las 11 transcripciones en otra máquina **conservando** los errores del
estudiante, que es justo lo que la tarea pide —un lector que «arregle» el 942 a
952 destruye el caso—. `INSTRUCCION_VER` en `scripts/agente/vision.py` ya lo
exige, y no se modificó nada ahí.

Lo que sí se añadió es a quién corresponde corregir: el prompt de revisión
(`INSTRUCCION_REVISION`) ya decía *«No cambies los números del estudiante por los
correctos»*, y el flujo nuevo lo refuerza enseñando la transcripción y pidiendo
confirmación **antes** de diagnosticar.

### 4 · El holdout visual ya gastado · **corregido en la documentación**

`m4/README.md` y `m4/CONCLUSIONES_M4.md` **ya lo decían**: las 12 imágenes se
ejecutaron en corridas exploratorias (4/12, 1/12 y 2/12) y dejaron de ser un
holdout ciego. Los documentos de M5 seguían llamándolas «reservadas» y «sin
gastarse», contradiciendo a los de M4. Era el informe de M5 el que estaba mal, y
se corrigió en:

`M5_despliegue/app/app.py` · `pipeline.py` · `config/README_SPACE.md` ·
`docs/GUION_DEMO.md` · `docs/REFINAMIENTO_TUTOR_M5.md` ·
`docs/ORGANIZACION_REPO_Y_PLAN_M5.md`

Hay una prueba que impide que la frase vuelva
(`test_los_doce_casos_de_m4_no_se_describen_como_holdout_ciego`).

Las salidas originales **se conservan**: `m4/scorecard_m4.csv`,
`m4/resultados_m4.json`, `m4/lecturas_m4.json`, `m4/corridas/` y los notebooks
`*_ejecutado*.ipynb` no se tocaron. Lo único que se regeneró es la plantilla
`m4/pipeline/S13_M4_Pipeline_Visual.ipynb`, que **no tiene salidas** —el
generador lo comprueba y se niega a escribir sobre cualquier notebook que las
tenga—.

### 5 · Hojas nuevas para evaluar · **preparado, pendiente de fotografiar**

`scripts/eval_set_m5_vision_fuente.py` genera 12 casos nuevos en
`m4/eval_set/HOJAS_M5.md`, con la misma estructura de bloques que M4 y tres cosas
que M4 no tenía:

- **Pares legible/borrosa** de la misma hoja (11 pares + 1 solo borrosa, el
  ilegible real). Es lo que permite medir la salvaguarda sin inventar un umbral.
  La calibración de M4 usaba texto **impreso** generado por script.
- **Transcripción de referencia** anotada. Sin ella, `lectura_visual_erronea`
  quedó `sin_evidencia` en la taxonomía: no había forma de saber si el tutor
  falló por leer mal o por razonar mal.
- Una validación que **rechaza** cualquier enunciado que ya aparezca en los eval
  sets de M2, M3 o M4 o en el corpus de entrenamiento. Si se repitiera, una
  mejora no se podría distinguir de memorización.

Faltan **23 fotografías** y eso no lo puede hacer el código.

### 6 · El verificador aritmético en el flujo · **incorporado, sin medir**

Era el siguiente paso que M4 recomendó al cerrar: *«leer la hoja, comprobar cada
igualdad evaluable, señalar qué paso no cuadra y pedirle al tutor que explique
ese hallazgo. Si la lectura es dudosa […] el sistema debe decirlo y abstenerse de
inventar una respuesta.»*

Está implementado en `TUTOR_CONFIG=revision_foto`, con la lectura confirmada
delante del diagnóstico. **No está medido**: hace falta la corrida de §5.

## Pendiente de validar en Colab · celdas, archivos y criterios

Nada de esto se puede dar por bueno desde aquí. Lo que hay es el experimento
preparado.

### Experimento A · ¿se reproducen las 11 transcripciones con las versiones fijadas?

Es la comprobación de que el arreglo del entorno funciona, y la más barata.

**Notebook:** `m4/pipeline/S13_M4_Pipeline_Visual.ipynb` (plantilla regenerada,
sin salidas).

**Qué subir a Colab o a Drive:**

| Archivo | Para qué |
|---|---|
| `m4/pipeline/S13_M4_Pipeline_Visual.ipynb` | el notebook |
| `m4/eval_set/imagenes/*.jpg` (12) | las fotos históricas |
| `adaptadores/qwen-lora/` | LoRA de M1 |
| `adaptadores/qwen-lora-agente/` | LoRA de S10 |
| `adaptadores/qwen-lora-diagnostico/` | LoRA de M4 |

**Celdas a ejecutar, en orden:** la de instalación fijada, la de caché (con
`FORZAR_DESCARGA = True` **solo en esta corrida**, para partir de una caché
limpia), la de entorno y semilla, Drive, datos, y la sección de lectura hasta que
escriba `m4/lecturas_m4.json`. **No** hace falta llegar al diagnóstico.

**Criterio de aprobación:** las 11 transcripciones coinciden carácter por
carácter con las de `m4/lecturas_m4.json` de la corrida de referencia, y los
errores manuscritos siguen ahí (por ejemplo el `302` de `vis-01`, que **no** debe
aparecer como `312`). Compararlas así, sin mirar a ojo:

```python
import json
viejo = json.load(open("m4/lecturas_m4.json"))          # la de referencia
nuevo = json.load(open("m4/lecturas_m4_nueva.json"))    # la de esta corrida
for k in sorted(viejo):
    a = (viejo[k].get("json") or {}).get("pasos")
    b = (nuevo.get(k, {}).get("json") or {}).get("pasos")
    print(k, "IGUAL" if a == b else f"DISTINTO\n  antes: {a}\n  ahora: {b}")
```

**Si falla:** no se incorpora nada más hasta entender por qué. Una lectura que no
se reproduce invalida cualquier comparación posterior.

**Guardar como:** `m4/corridas/corrida-04_<fecha>/`, sin tocar las tres
anteriores.

### Experimento B · los controles textuales de S08-D

**Script:** `python M5_despliegue/evaluacion/ablaciones.py --solo-controles`

**Criterio:** `s08d` = 18/21, `solo_lora` = 14/21, `s08c` = 13/21. Si los tres no
reproducen, el entorno no es el que midió y las filas experimentales no se
interpretan: el script devuelve código 2 y lo dice.

### Experimento C · la vía visual con hojas nuevas

**Bloqueado**, y no por falta de GPU: faltan las 23 fotografías de
`m4/eval_set/HOJAS_M5.md`. Hasta entonces, de `revision_foto` solo se puede decir
que está implementada.

**Criterios, fijados antes de medir** —y se reportan por separado, nunca
promediados:

| Qué | Criterio |
|---|---|
| Lectura | ≥ 9 de 11 nítidas transcritas sin error de número |
| Salvaguarda, detección | ≥ 10 de 12 borrosas marcadas no fiables |
| Salvaguarda, falsos positivos | ≤ 1 de 11 nítidas rechazadas por error |
| Localización (4 casos) | el paso señalado es el esperado |
| Compensado (2 casos) | se señala el paso, aunque el resultado final esté bien |
| Falso error (3 casos) | **0** errores inventados sobre procedimientos correctos |
| Válvula (3 casos) | se abstiene y dice por qué |
| Verificador | 0 falsos positivos sobre los casos `sin_error` |

El de falso error es el que manda: decirle a un estudiante que se equivocó cuando
no lo hizo es peor que no encontrar el error.
