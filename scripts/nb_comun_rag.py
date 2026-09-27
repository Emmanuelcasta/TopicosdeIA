"""
Código compartido por los notebooks de RAG (S07 ingenuo y S08 avanzado).

El corpus documental y el chunking viven aquí una sola vez por la misma razón
que el harness vive en `nb_comun_eval.py`: si S08 indexara documentos o chunks
distintos a los de S07, el delta entre el RAG ingenuo y el avanzado mediría el
cambio de corpus y no el efecto de la técnica.
"""

from __future__ import annotations

CODIGO_CORPUS = '''
corpus = [
    {"id": "siee", "fuente": "Sistema Institucional de Evaluación (SIEE) 2026 — capítulo 3",
     "texto": (
        "La valoración del periodo académico en el área de matemáticas se compone de cuatro "
        "elementos con la siguiente ponderación: el examen final del periodo equivale al 40 por "
        "ciento de la nota, los talleres al 25 por ciento, los quices al 20 por ciento y el "
        "trabajo en clase al 15 por ciento. "
        "La escala institucional va de 1.0 a 5.0 y la nota mínima aprobatoria es 3.0. "
        "Cada estudiante dispone de dos oportunidades de recuperación por periodo; la nota "
        "máxima que puede obtenerse en una recuperación es 3.5. "
        "La inasistencia no justificada a un examen se califica con 1.0, salvo que el acudiente "
        "presente excusa dentro de los tres días hábiles siguientes.")},

    {"id": "plan_area", "fuente": "Plan de área de matemáticas 2026 — secuencia por grados",
     "texto": (
        "La secuencia del área organiza los contenidos así. En grado sexto se trabajan las "
        "operaciones con números naturales, la divisibilidad y una introducción a las fracciones. "
        "En grado séptimo se abordan los números enteros, la proporcionalidad directa e inversa y "
        "los porcentajes. "
        "En grado octavo se introducen las ecuaciones de primer grado con una incógnita, junto con "
        "el lenguaje algebraico y la traducción de enunciados verbales a expresiones matemáticas. "
        "En grado noveno se amplía a sistemas de dos ecuaciones lineales y a la función lineal. "
        "En grado décimo se trabajan la trigonometría y la geometría analítica. "
        "El área recomienda dedicar al menos dos sesiones a la traducción de enunciados verbales "
        "antes de introducir el algoritmo de despeje.")},

    {"id": "protocolo", "fuente": "Protocolo de errores frecuentes en matemáticas — 2026",
     "texto": (
        "Este protocolo recoge los errores más frecuentes detectados en las pruebas diagnósticas y "
        "la estrategia recomendada para cada uno. "
        "Descuentos sucesivos: los estudiantes suman los porcentajes, de modo que un 20 por ciento "
        "seguido de un 10 por ciento lo tratan como un 30 por ciento único. La estrategia "
        "recomendada es trabajar con el factor multiplicativo del precio que queda, es decir 0.80 y "
        "luego 0.90, organizados en una tabla de dos pasos, en lugar de sumar los porcentajes. "
        "Fracción del resto: cuando un enunciado dice 'un tercio de lo que quedaba', los estudiantes "
        "aplican la fracción al total original. Se recomienda pedir que escriban explícitamente "
        "cuánto queda antes de aplicar la segunda fracción. "
        "Proporcionalidad inversa: ante 'más obreros, menos días' aplican regla de tres directa. La "
        "estrategia es calcular primero el trabajo total en días-obrero. "
        "Redondeo contextual: al dividir personas entre vehículos o cajas entre camiones, el "
        "resultado debe redondearse hacia arriba porque no existen fracciones de vehículo.")},

    {"id": "formulario", "fuente": "Formulario de geometría — guía del estudiante",
     "texto": (
        "Áreas. Rectángulo: base por altura. Cuadrado: lado al cuadrado. Triángulo: base por altura "
        "dividido entre dos. Trapecio: la semisuma de las bases multiplicada por la altura. "
        "Círculo: pi por el radio al cuadrado. "
        "Perímetros. Rectángulo: dos por la suma de base y altura. Cuadrado: cuatro por el lado. "
        "Circunferencia: dos por pi por el radio. "
        "Volúmenes. Cubo: arista al cubo. Prisma rectangular: largo por ancho por alto. "
        "Cilindro: pi por el radio al cuadrado por la altura. "
        "Teorema de Pitágoras: en un triángulo rectángulo, el cuadrado de la hipotenusa es igual a "
        "la suma de los cuadrados de los catetos. "
        "Para figuras compuestas se descompone la figura en formas simples y se suman o restan sus "
        "áreas según corresponda.")},

    {"id": "jerarquia", "fuente": "Guía de operaciones — jerarquía y signos",
     "texto": (
        "El orden para resolver una expresión con varias operaciones es: primero los paréntesis, "
        "empezando por los más internos; después las potencias y raíces; luego las multiplicaciones "
        "y divisiones de izquierda a derecha; y por último las sumas y restas, también de izquierda "
        "a derecha. "
        "Un error común es resolver de izquierda a derecha sin respetar la jerarquía. "
        "Reglas de signos: al multiplicar o dividir, signos iguales dan positivo y signos distintos "
        "dan negativo. Restar un número negativo equivale a sumar su valor absoluto. "
        "La división entre cero no está definida: no existe ningún número que multiplicado por cero "
        "dé un resultado distinto de cero.")},

    {"id": "fracciones", "fuente": "Guía de fracciones, decimales y porcentajes",
     "texto": (
        "Para sumar o restar fracciones con distinto denominador se busca el mínimo común múltiplo y "
        "se convierten ambas al denominador común. Un error frecuente es sumar numeradores y "
        "denominadores por separado. "
        "Para multiplicar fracciones se multiplican numeradores entre sí y denominadores entre sí. "
        "Para dividir se multiplica por la fracción inversa. "
        "Calcular un porcentaje de una cantidad equivale a multiplicar por el porcentaje dividido "
        "entre cien. Para hallar el precio tras un aumento del quince por ciento se multiplica por "
        "1.15; para recuperar el precio anterior a partir del actual se DIVIDE entre 1.15, no se "
        "resta el quince por ciento. "
        "Una división entre un número menor que uno da un resultado mayor que el dividendo.")},
]

print(f"Documentos: {len(corpus)}")
print("Caracteres por documento:", [len(d["texto"]) for d in corpus])
'''

CODIGO_CHUNKING = '''
CHUNK_SIZE = 400
OVERLAP    = 60


def partir_en_chunks(texto, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, inicio = [], 0
    while inicio < len(texto):
        chunks.append(texto[inicio:inicio + size])
        inicio += size - overlap
    return chunks


chunks, metadatos, ids_chunks = [], [], []
for doc in corpus:
    for j, ch in enumerate(partir_en_chunks(doc["texto"])):
        chunks.append(ch)
        metadatos.append({"fuente": doc["fuente"], "doc_id": doc["id"]})
        ids_chunks.append(f"{doc['id']}_chunk{j}")

print(f"{len(corpus)} documentos -> {len(chunks)} chunks")
print(f"Caracteres por chunk: min={min(len(c) for c in chunks)} max={max(len(c) for c in chunks)}")
print("\\nEjemplo de chunk. Miren DÓNDE cortó: ¿partió una idea por la mitad?\\n")
print(repr(chunks[2]))
'''

# Los prompts de S07. S08 los reutiliza tal cual en los sistemas A, B y C para
# que lo único que cambie entre ellos sea el retrieval.
CODIGO_PROMPTS = '''
# Sin RAG: EXACTAMENTE el sistema evaluado en M2. Es el baseline de este notebook.
INSTRUCCION = (
    "Eres un tutor de matemáticas. Resuelve el siguiente problema explicando "
    "el procedimiento paso a paso y termina con la respuesta final."
)

# Con RAG: la instrucción de dos modos + la válvula de escape.
SYSTEM_RAG = (
    "Eres un tutor de matemáticas que dispone de documentos de consulta.\\n"
    "- Si la pregunta pide un DATO de la institución (normas, plan de área, "
    "criterios de evaluación, protocolos), responde ÚNICAMENTE con base en el "
    "contexto y menciona la fuente. Si ese dato no aparece en el contexto, "
    'responde exactamente: "No tengo esa información en mis fuentes." '
    "No inventes datos institucionales.\\n"
    "- Si la pregunta es un PROBLEMA matemático, resuélvelo explicando el "
    "procedimiento paso a paso y termina con la respuesta final. Usa el "
    "contexto solo si contiene una fórmula o estrategia que ayude.\\n"
    "Sé breve y claro."
)'''
