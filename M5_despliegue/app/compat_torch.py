"""Compatibilidad con el PyTorch del entorno · el `torch.distributed.tensor`.

El síntoma en Kaggle (Python 3.13, Tesla T4): la página de Gradio abre, se envía
«¿Cuánto es 3/4 + 1/8?» y antes de generar nada sale

    AttributeError: module 'torch.distributed' has no attribute 'tensor'

**La forma del error importa.** `AttributeError` —no `ModuleNotFoundError`—
significa que alguien escribió `torch.distributed.tensor.Algo` **sin importar
antes el submódulo**. `torch.distributed` es un paquete y `tensor` es un
subpaquete suyo: existe en el disco, pero `import torch` no lo carga, así que el
atributo no está hasta que alguien hace el import explícito. Comprobado aquí con
torch 2.4.1:

    hasattr(torch.distributed, "tensor")   -> False
    import torch.distributed.tensor        -> OK

Es decir: el submódulo **se puede importar**, y en cuanto se importa el atributo
aparece y el acceso que fallaba funciona. Por eso basta con importarlo pronto, y
por eso no hace falta tocar el PyTorch ni el CUDA de Kaggle.

**Qué se descartó, con evidencia**, buscando el patrón en el código real de cada
paquete en vez de suponerlo:

    transformers 4.45.2   no menciona torch.distributed.tensor
    accelerate 0.34.2     no lo menciona
    peft 0.17.1           no lo menciona
    transformers 4.57.6   solo `from torch.distributed.tensor import ...`
    accelerate 1.10.1     solo `from torch.distributed.tensor import ...`
    huggingface_hub       solo `from ... import DTensor`, dentro de try/except

Un `from X import Y` que falla da `ModuleNotFoundError` o `ImportError`, no
`AttributeError`, y los de `huggingface_hub` van dentro de `try`. Así que el
acceso por atributo está en un paquete del entorno de Kaggle que **no se ha
identificado todavía**: para eso está el aislamiento por etapas de
`pipeline.py`, que imprime el traceback completo y dice en qué paso ocurre.

Mientras tanto este módulo hace lo único que es correcto hacer a ciegas:
asegurarse de que el submódulo esté importado. No silencia ningún error, no
cambia ninguna versión y, si el submódulo no existe (torch sin soporte
distribuido), lo deja anotado y sigue.
"""

from __future__ import annotations


def asegurar_distributed_tensor() -> dict:
    """Importa `torch.distributed.tensor` para que el atributo exista.

    -> qué se encontró y qué se hizo, para poder informarlo sin adivinar.
    """
    info = {"torch": "ausente", "distributed_disponible": None,
            "tensor_antes": None, "tensor_despues": None,
            "accion": "", "error": ""}
    try:
        import torch
    except Exception as e:      # noqa: BLE001
        info["error"] = f"no se pudo importar torch: {type(e).__name__}: {e}"
        return info

    info["torch"] = getattr(torch, "__version__", "desconocida")
    try:
        import torch.distributed as dist
    except Exception as e:      # noqa: BLE001
        info["error"] = f"torch.distributed no importable: {type(e).__name__}: {e}"
        return info

    # En un torch compilado sin soporte distribuido, `tensor` no existe y no hay
    # nada que importar. Se dice, en vez de fallar.
    info["distributed_disponible"] = bool(
        getattr(dist, "is_available", lambda: False)())
    info["tensor_antes"] = hasattr(dist, "tensor")

    if info["tensor_antes"]:
        info["accion"] = "no hizo falta: el atributo ya estaba"
        info["tensor_despues"] = True
        return info

    try:
        import torch.distributed.tensor  # noqa: F401
        info["tensor_despues"] = hasattr(dist, "tensor")
        info["accion"] = (
            "importado torch.distributed.tensor: el atributo ya existe y el "
            "acceso `torch.distributed.tensor.X` deja de lanzar AttributeError")
    except Exception as e:      # noqa: BLE001
        info["tensor_despues"] = False
        info["error"] = (
            f"torch.distributed.tensor no se pudo importar "
            f"({type(e).__name__}: {e}). Este torch no trae DTensor; si algún "
            "paquete lo usa, hay que bajar ese paquete, no cambiar torch")
    return info


def versiones() -> dict:
    """El conjunto de versiones que decide si la carga del modelo funciona."""
    import importlib.metadata as md
    import sys

    v = {"python": ".".join(map(str, sys.version_info[:3]))}
    for paquete in ("torch", "transformers", "peft", "accelerate",
                    "huggingface_hub", "safetensors", "tokenizers", "numpy"):
        try:
            v[paquete] = md.version(paquete)
        except Exception:      # noqa: BLE001
            v[paquete] = "ausente"
    try:
        import torch
        v["cuda_disponible"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            v["gpu"] = torch.cuda.get_device_name(0)
            v["cuda_de_torch"] = str(torch.version.cuda)
    except Exception:      # noqa: BLE001
        pass
    return v


def quien_usa_distributed_tensor() -> list:
    """Busca el acceso por ATRIBUTO en los paquetes instalados.

    Esto es lo que identifica al culpable sin tener que reproducir el fallo:
    `from torch.distributed.tensor import X` es correcto y da otro error;
    `torch.distributed.tensor.X` es el que produce el AttributeError.
    """
    import re
    import sysconfig
    from pathlib import Path

    patron = re.compile(r"(?<![\w.])torch\.distributed\.tensor\s*\.")
    sitio = sysconfig.get_paths().get("purelib")
    if not sitio or not Path(sitio).is_dir():
        return []
    hallazgos = []
    for ruta in Path(sitio).rglob("*.py"):
        try:
            texto = ruta.read_text(encoding="utf-8", errors="ignore")
        except Exception:      # noqa: BLE001
            continue
        if "torch.distributed.tensor" not in texto:
            continue
        for n, linea in enumerate(texto.splitlines(), 1):
            limpia = linea.strip()
            if limpia.startswith(("from ", "import ", "#", ">>>")):
                continue
            m = patron.search(linea)
            if not m:
                continue
            # Dentro de una cadena no es un acceso: el propio torch lleva
            # "torch.distributed.tensor.parallel…" como texto en sus registros
            # de API, y reportarlo sería ruido justo donde hace falta señal.
            antes = linea[:m.start()]
            if antes.count('"') % 2 or antes.count("'") % 2:
                continue
            hallazgos.append(f"{ruta.relative_to(sitio)}:{n}: {limpia[:110]}")
    return hallazgos


def diagnostico() -> str:
    """Todo lo necesario para decidir, en una sola salida pegable."""
    lineas = ["Versiones del entorno:"]
    lineas += [f"  {k:18s} {v}" for k, v in versiones().items()]

    info = asegurar_distributed_tensor()
    lineas += ["", "torch.distributed.tensor:",
               f"  distributed disponible : {info['distributed_disponible']}",
               f"  atributo antes         : {info['tensor_antes']}",
               f"  atributo después       : {info['tensor_despues']}",
               f"  acción                 : {info['accion'] or '—'}"]
    if info["error"]:
        lineas.append(f"  ERROR                  : {info['error']}")

    culpables = quien_usa_distributed_tensor()
    lineas += ["", "Accesos por atributo encontrados en site-packages:"]
    lineas += [f"  {c}" for c in culpables] or ["  ninguno"]
    if not culpables:
        lineas.append("  (si el fallo persiste, el acceso está en código "
                      "compilado o fuera de purelib; manda el traceback completo)")
    return "\n".join(lineas)


if __name__ == "__main__":
    print(diagnostico())
