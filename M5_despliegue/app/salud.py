"""Comprobación de salud: estado, versión y tiempo activo.

El Space necesita responder "¿estás vivo y con qué?" sin cargar el modelo. Por
eso el estado del pipeline se consulta sin forzar la carga: si el modelo aún no
se ha pedido, se dice así en vez de provocar una descarga de 3 GB.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time

ARRANQUE = time.time()


def _version() -> str:
    """El commit del repositorio, si está disponible. Nunca lanza."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=2,
                              check=True).stdout.strip() or "desconocida"
    except Exception:  # noqa: BLE001
        return os.environ.get("TUTOR_VERSION", "desconocida")


def salud(tutor=None) -> dict:
    """-> dict serializable con el estado. No carga el modelo."""
    datos = {
        "estado": "arriba",
        "version": _version(),
        "activo_s": round(time.time() - ARRANQUE, 1),
        "python": platform.python_version(),
    }
    if tutor is not None:
        e = tutor.estado
        # Dos nombres, y en este orden. Lo SOLICITADO es lo que alguien puso en
        # `TUTOR_CONFIG`; lo EFECTIVO es lo que quedó respondiendo. Cuando
        # difieren, la cifra histórica de la solicitada no describe a esta
        # ejecución, y `medido_aplicable` la retira en vez de adornarla.
        datos.update({
            "configuracion_solicitada": e.config_nombre,
            "configuracion_efectiva": e.efectiva(),
            "degradado": e.degradado,
            "componentes_faltantes": e.faltantes,
            "resultado_medido": _resultado(e),
            "montaje": ("verificado" if e.montado else
                        "declarado: el modelo no se ha cargado todavía, así que "
                        "retrieval, enrutador y herramientas son el plan de la "
                        "configuración, no lo comprobado"),
            "componentes": {
                "modelo_base": os.environ.get(
                    "TUTOR_MODELO_BASE", "Qwen/Qwen2.5-1.5B-Instruct"),
                "modelo_cargado": e.modelo,
                "adaptador_pedido": e.adaptador_pedido,
                "adaptador_cargado": e.adaptador_cargado,
                "retrieval_pedido": e.retrieval_pedido,
                "retrieval_activo": e.retrieval,
                "enrutador_pedido": e.enrutador_pedido,
                "enrutador_activo": e.enrutador,
                "herramientas_activas": e.herramientas,
                "herramientas_en_el_registro": e.registro_disponible,
                "herramientas": e.herramientas_nombres,
                "verificador": e.verificador,
                "normalizador": e.normalizador,
                "vision": e.vision,
                "cuantizado": e.cuantizado,
                "dispositivo": e.dispositivo,
                "corpus": e.corpus,
            },
            "listo": e.listo,
            "resumen": e.resumen(),
        })
        if e.motivos:
            datos["avisos"] = e.motivos
    return datos


def _resultado(e) -> str:
    """Qué cifra se puede atribuir a ESTA ejecución. Casi siempre, ninguna."""
    if e.degradado:
        texto = "NO MEDIDO: ejecución degradada, falta " + ", ".join(e.faltantes)
        if e.medido:
            texto += (f". La cifra {e.medido} es de «{e.config_nombre}» con todas "
                      "sus piezas y no describe lo que responde ahora")
        return texto
    if not e.medido:
        return "ninguno: variante EXPERIMENTAL, sin medir"
    if not e.montado:
        return (f"{e.medido} es la cifra histórica de esta configuración; en esta "
                "ejecución el montaje todavía no se ha comprobado")
    return f"{e.medido} (histórico de S08, sobre el eval set fijo de M2)"


def texto_salud(tutor=None) -> str:
    """La misma información, plana, para la pestaña de estado y los logs."""
    lineas = []
    for k, v in salud(tutor).items():
        if isinstance(v, dict):
            lineas.append(f"{k}:")
            lineas += [f"  {sk}: {sv}" for sk, sv in v.items()]
        elif isinstance(v, list):
            lineas.append(f"{k}: " + ("; ".join(map(str, v)) if v else "ninguno"))
        else:
            lineas.append(f"{k}: {v}")
    return "\n".join(lineas)
