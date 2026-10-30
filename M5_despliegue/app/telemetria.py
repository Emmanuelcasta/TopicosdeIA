"""Registro estructurado de la aplicación, con W&B opcional.

Dos reglas que condicionan el diseño:

1. **No se registra el texto del usuario por defecto.** Una pregunta de un
   estudiante puede traer su nombre, su colegio o su nota. Se registran la
   longitud, el hash y las métricas; el texto solo si alguien activa
   `TUTOR_LOG_TEXTO=1` a sabiendas.

2. **W&B es opcional y se apaga limpio.** Si no hay credencial, o la librería no
   está, la aplicación sigue funcionando y lo dice una vez. Una demo que se cae
   porque falta una variable de telemetría es una demo rota.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys

LOG_TEXTO = os.environ.get("TUTOR_LOG_TEXTO", "").lower() in ("1", "true")
_wandb = None
_avisado = False

log = logging.getLogger("tutor")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)


def _iniciar_wandb():
    """-> run de W&B o None. Nunca lanza."""
    global _wandb, _avisado
    if _wandb is not None:
        return _wandb
    if not os.environ.get("WANDB_API_KEY"):
        if not _avisado:
            log.info("W&B desactivado: no hay WANDB_API_KEY")
            _avisado = True
        return None
    try:
        import wandb
        _wandb = wandb.init(project=os.environ.get("WANDB_PROJECT", "tutor-m5"),
                            job_type="demo", reinit=False)
        return _wandb
    except Exception as e:  # noqa: BLE001
        if not _avisado:
            log.info("W&B desactivado: %s", type(e).__name__)
            _avisado = True
        return None


def registrar(evento: str, pregunta: str = "", **metricas) -> dict:
    """Una línea JSON por interacción. Devuelve lo registrado, para poder probarlo."""
    fila = {"evento": evento, **metricas}
    if pregunta:
        fila["pregunta_len"] = len(pregunta)
        fila["pregunta_hash"] = hashlib.sha256(pregunta.encode()).hexdigest()[:12]
        if LOG_TEXTO:
            fila["pregunta"] = pregunta
    log.info(json.dumps(fila, ensure_ascii=False))

    run = _iniciar_wandb()
    if run is not None:
        try:
            run.log({k: v for k, v in fila.items()
                     if isinstance(v, (int, float, bool))})
        except Exception:  # noqa: BLE001
            pass
    return fila
