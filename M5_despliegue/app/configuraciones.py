"""Las configuraciones del tutor, declaradas y separadas.

Por qué existe este archivo. La primera versión de `pipeline.py` montaba el
sistema con `SYSTEM_C5_AGENTE`, el registro de herramientas completo y
preferencia automática por `qwen-lora-agente` si estaba en disco. Eso **no es**
la configuración que mejor midió: es una variante de C5, que marcó **12/21** —la
peor de las agénticas— mientras que S08-D marcó **18/21** sobre el mismo eval set.

Escoger la configuración por "qué adaptador hay en el disco" es la forma más
rápida de desplegar algo distinto de lo que se midió. Aquí cada configuración se
declara con nombre, se elige por `TUTOR_CONFIG`, y la comprobación de salud dice
cuál está activa.

**Qué está medido y qué no**, que es la distinción que importa:

    s08d                 18/21  medido · scorecard_rag_avanzado_v2.csv, col "D · +router"
    s08c                 13/21  medido · ídem, col "C · +rerank" (control: s08d sin enrutador)
    solo_lora            14/21  medido · ambos scorecards (línea base común)
    agente_c5ft          13/21  medido · scorecard_agente_v2.csv, col "C5-FT · entrenado"
    s08d_herramientas      —    EXPERIMENTAL, nunca medida
    s08d_verificador       —    EXPERIMENTAL, nunca medida
    s08d_normalizador      —    EXPERIMENTAL, nunca medida
    s08d_vision            —    EXPERIMENTAL, nunca medida
    revision_foto          —    EXPERIMENTAL · producto: foto + confirmación + verificador

Las dos últimas son hipótesis. La aplicación las puede montar, pero la salud las
marca como no medidas y el informe no las presenta como resultado.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Configuracion:
    nombre: str
    adaptador: str            # "m1" | "agente"
    retrieval: str            # "ninguno" | "lexico" | "hibrido"
    enrutador: bool           # decidir si buscar, en vez de buscar siempre
    herramientas: bool        # el registro completo como tool calling
    verificador: bool         # comprobación aritmética determinista
    vision: bool              # vía visual de M4
    system: str               # "rag" | "agente"
    medido: str | None        # resultado medido, o None si nunca se midió
    fuente: str | None        # archivo y columna que lo respaldan
    normalizador: bool = False  # reescritura de preguntas mal escritas
    notas: str = ""

    @property
    def es_experimental(self) -> bool:
        return self.medido is None


CONFIGURACIONES: dict[str, Configuracion] = {
    # --- línea base reproducible: la mejor medida ------------------------
    "s08d": Configuracion(
        nombre="S08-D · LoRA M1 + RAG híbrido + enrutador",
        adaptador="m1", retrieval="hibrido", enrutador=True,
        herramientas=False, verificador=False, vision=False, system="rag",
        medido="18/21",
        fuente="M3_rag_agentes/resultados/scorecard_rag_avanzado_v2.csv · fila "
               "`aciertos_total`, columna `D · +router`",
        notas="2 alucinaciones (fila `alucinaciones`); adversarial 5/5."),

    # --- la línea base común de los dos scorecards ----------------------
    "solo_lora": Configuracion(
        nombre="C1 · solo LoRA de M1",
        adaptador="m1", retrieval="ninguno", enrutador=False,
        herramientas=False, verificador=False, vision=False, system="rag",
        medido="14/21",
        fuente="scorecard_agente_v2.csv · `Aciertos totales /21`, col `C1 · +LoRA`; "
               "y scorecard_rag_avanzado_v2.csv · col `sin RAG`",
        notas="Respaldo simple. Las dos mediciones coinciden (14/21, 7 "
              "alucinaciones): es el mismo sistema medido en dos notebooks."),

    # --- la agéntica, para poder compararla ------------------------------
    "agente_c5ft": Configuracion(
        nombre="C5-FT · agente entrenado con trayectorias",
        adaptador="agente", retrieval="lexico", enrutador=False,
        herramientas=True, verificador=False, vision=False, system="agente",
        medido="13/21",
        fuente="scorecard_agente_v2.csv · `Aciertos totales /21`, col `C5-FT · entrenado`",
        notas="8 alucinaciones; 14,1 s por caso. En los 53 casos de M3 gana en el "
              "bloque de aritmética difícil (10/13), para el que se entrenó."),

    # --- control medido: S08-D menos el enrutador ------------------------
    "s08c": Configuracion(
        nombre="S08-C · LoRA M1 + RAG híbrido + reranker, sin enrutador",
        adaptador="m1", retrieval="hibrido", enrutador=False,
        herramientas=False, verificador=False, vision=False, system="rag",
        medido="13/21",
        fuente="M3_rag_agentes/resultados/scorecard_rag_avanzado_v2.csv · fila "
               "`aciertos_total`, columna `C · +rerank`",
        notas="5 alucinaciones (fila `alucinaciones`). Es el control de la "
              "ablación del enrutador: la única diferencia con `s08d` es que el "
              "RAG se aplica siempre en vez de decidirlo."),

    # --- experimentales: montables, NO medidas ---------------------------
    "s08d_herramientas": Configuracion(
        nombre="S08-D + herramientas selectivas",
        adaptador="m1", retrieval="hibrido", enrutador=True,
        herramientas=True, verificador=False, vision=False, system="agente",
        medido=None, fuente=None,
        notas="HIPÓTESIS. Combina lo que midió mejor en retrieval con lo que "
              "eliminó los errores de cálculo. Nunca se midió junto."),

    "s08d_verificador": Configuracion(
        nombre="S08-D + herramientas + verificador aritmético",
        adaptador="m1", retrieval="hibrido", enrutador=True,
        herramientas=True, verificador=True, vision=False, system="agente",
        medido=None, fuente=None,
        notas="HIPÓTESIS. El verificador acierta 4/4 en aislado sobre casos de "
              "desarrollo de M4, pero nunca se conectó al flujo."),

    "s08d_normalizador": Configuracion(
        nombre="S08-D + normalizador de preguntas mal escritas",
        adaptador="m1", retrieval="hibrido", enrutador=True,
        herramientas=False, verificador=False, vision=False, system="rag",
        normalizador=True, medido=None, fuente=None,
        notas="HIPÓTESIS. `Normalizador` existe en `nucleo.py` y nunca se midió "
              "por separado. Única diferencia con `s08d`."),

    "s08d_vision": Configuracion(
        nombre="S08-D + vía visual",
        adaptador="m1", retrieval="hibrido", enrutador=True,
        herramientas=False, verificador=False, vision=True, system="rag",
        medido=None, fuente=None,
        notas="HIPÓTESIS. Única diferencia con `s08d`: se habilita la vía visual. "
              "La visión marcó 4/12, 1/12 y 2/12 en tres corridas exploratorias y "
              "3/9 en desarrollo, con OTRA configuración (adaptador `agente` + RAG "
              "denso), así que esas cifras no son de esta. Es la fila de ablación "
              "de visión: un solo cambio."),

    # --- configuración de PRODUCTO, experimental -------------------------
    # No es una fila de ablación: cambia dos cosas a la vez respecto a `s08d`
    # (visión y verificador) porque las dos forman un flujo —leer, confirmar,
    # comprobar, explicar— que no tiene sentido partir en mitades. Para aislar
    # el efecto de cada pieza están `s08d_vision` y `s08d_verificador`.
    "revision_foto": Configuracion(
        nombre="Revisión de foto · lectura confirmada + verificador (EXPERIMENTAL)",
        adaptador="m1", retrieval="hibrido", enrutador=True,
        herramientas=False, verificador=True, vision=True, system="rag",
        medido=None, fuente=None,
        notas="SIN MEDIR con imágenes nuevas. Es el producto que M4 recomendó al "
              "cerrar: leer la hoja, confirmar la lectura con el estudiante, "
              "comprobar las igualdades con el verificador determinista y explicar "
              "el hallazgo sin reescribir el procedimiento. Los 12 casos visuales "
              "de M4 ya se gastaron, así que no sirven para medirla: hacen falta "
              "las hojas nuevas de `m4/eval_set/HOJAS_M5.md`."),
}

POR_DEFECTO = "s08d"


def obtener(nombre: str | None) -> Configuracion:
    """La configuración pedida. Si no existe, se dice en vez de adivinar."""
    clave = (nombre or POR_DEFECTO).strip().lower()
    if clave not in CONFIGURACIONES:
        raise KeyError(f"configuración desconocida: {nombre!r}. "
                       f"Disponibles: {sorted(CONFIGURACIONES)}")
    return CONFIGURACIONES[clave]
