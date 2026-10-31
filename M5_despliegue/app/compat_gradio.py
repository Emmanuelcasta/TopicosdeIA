"""Compatibilidad de Gradio con Pydantic ≥ 2.11 · el fallo del enlace en Kaggle.

El síntoma. En Kaggle con T4, `TUTOR_GRADIO_SHARE=1` imprime una URL
`*.gradio.live` y al abrirla responde *Internal Server Error* o «No interface is
running». La traza:

    gradio/routes.py        api_info()
    gradio/blocks.py        get_api_info()
    gradio_client/utils.py  json_schema_to_python_type()
                            _json_schema_to_python_type()
                            get_type()
    TypeError: argument of type 'bool' is not iterable

La causa, exacta. `gr.Image` publica su esquema a través del modelo `FileData`,
cuyo campo `meta` está anotado como `dict` pelado. Pydantic **2.11 en adelante**
emite para ese campo:

    {"type": "object", "additionalProperties": true, ...}

y `additionalProperties: true` es JSON Schema válido: significa «se admite
cualquier propiedad adicional». Pero `gradio_client` 1.3.0 recorre ese valor
como si siempre fuera un subesquema:

    if "additionalProperties" in schema:
        des += [f"str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}"]

y la llamada entra en `get_type(True)`, que empieza con `if "const" in schema`.
`"const" in True` lanza `TypeError`. Pydantic ≤ 2.10 **no escribía la clave**, así
que la misma versión de Gradio funcionaba.

Comprobado con la combinación exacta del proyecto, `gradio==4.44.1` +
`gradio_client==1.3.0`:

    pydantic 2.7.4   additionalProperties ausente    -> funciona
    pydantic 2.8.2   ausente                         -> funciona
    pydantic 2.9.2   ausente                         -> funciona
    pydantic 2.10.6  ausente                         -> funciona
    pydantic 2.11.9  True                            -> TypeError
    pydantic 2.13.5  True                            -> TypeError

Por qué tumba la página entera y no solo `/info`: en Gradio 4.44 la ruta
principal consulta la información de la API al construir la respuesta, así que un
esquema que no se puede describir impide servir la interfaz.

**El arreglo primario es la versión**, fijada en `config/requirements.txt`:
`pydantic>=2.0,<2.11`. Lo de este módulo es la red de seguridad para entornos
donde ese pin no se puede imponer —Kaggle trae su propio stack preinstalado y
otro paquete puede arrastrar pydantic 2.11+—.

Y no es un parche que esconda el error: implementa la semántica que el esquema ya
tiene. En JSON Schema, un subesquema booleano `true` acepta cualquier valor y
`false` no acepta ninguno; eso es `Any` y `None` como anotación. Es la misma
lectura que hace Gradio en versiones posteriores. Si la librería ya trae el
arreglo, este módulo **no toca nada** y lo dice.
"""

from __future__ import annotations


def _necesita_parche() -> bool:
    """¿La `gradio_client` instalada se rompe con un subesquema booleano?

    Se comprueba ejecutándola, no comparando números de versión: lo que importa
    es el comportamiento, y así el parche desaparece solo en cuanto la librería
    lo arregle.
    """
    try:
        from gradio_client import utils
    except Exception:      # noqa: BLE001 — sin gradio_client no hay nada que parchear
        return False
    try:
        utils.get_type(True)
    except TypeError:
        return True
    except Exception:      # noqa: BLE001 — otro error es otro problema
        return False
    return False


def _pydantic_emite_booleanos() -> bool:
    """¿La pydantic instalada escribe `additionalProperties: true`?

    Es la otra mitad del problema. La librería de Gradio está igual de rota con
    cualquier pydantic; lo que cambia es si alguien le pasa un subesquema
    booleano. Se comprueba generando un esquema, no mirando el número de versión.
    """
    try:
        import pydantic

        class _M(pydantic.BaseModel):
            meta: dict = {}

        esquema = _M.model_json_schema()["properties"]["meta"]
        return isinstance(esquema.get("additionalProperties"), bool)
    except Exception:      # noqa: BLE001
        return False


def aplicar() -> dict:
    """Aplica el parche si hace falta. -> qué se hizo, para poder informarlo.

    Idempotente: una segunda llamada ve el parche ya puesto y no hace nada.

    Dos preguntas distintas, y las dos se responden ejecutando código:

      `parche_aplicado`  la gradio_client instalada no acepta subesquemas
                         booleanos, así que se le añade el guardia.
      `se_disparaba`     la pydantic instalada de verdad produce esos
                         subesquemas. Si es False, el fallo no estaba ocurriendo
                         y el parche solo es prevención.
    """
    info = {"parche_aplicado": False, "se_disparaba": _pydantic_emite_booleanos(),
            "motivo": "", "versiones": versiones()}
    if not _necesita_parche():
        info["motivo"] = ("no hace falta: la gradio_client instalada ya acepta "
                          "subesquemas booleanos")
        return info

    from gradio_client import utils

    get_type_original = utils.get_type
    a_python_original = utils._json_schema_to_python_type

    def get_type(schema):
        # JSON Schema permite que un subesquema sea un booleano. `True` acepta
        # cualquier valor, `False` ninguno.
        if isinstance(schema, bool):
            return "boolean_schema"
        return get_type_original(schema)

    def _json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return a_python_original(schema, defs)

    utils.get_type = get_type
    utils._json_schema_to_python_type = _json_schema_to_python_type
    info["parche_aplicado"] = True
    if info["se_disparaba"]:
        info["motivo"] = (
            "gradio_client no acepta subesquemas booleanos Y la pydantic "
            "instalada los produce (>= 2.11): sin el parche, la ruta que sirve la "
            "página lanza TypeError y el enlace público da Internal Server Error. "
            "Arreglo primario: fijar pydantic < 2.11")
    else:
        info["motivo"] = (
            "gradio_client no acepta subesquemas booleanos, pero la pydantic "
            "instalada no los produce (< 2.11): el fallo NO estaba ocurriendo y "
            "el parche es solo prevención")
    return info


def versiones() -> dict:
    """Las versiones que deciden si esto funciona o no. Sin adivinar."""
    import importlib.metadata as md
    import sys

    v = {"python": ".".join(map(str, sys.version_info[:3]))}
    for paquete in ("gradio", "gradio_client", "pydantic", "pydantic_core",
                    "fastapi", "starlette"):
        try:
            v[paquete] = md.version(paquete)
        except Exception:      # noqa: BLE001
            v[paquete] = "ausente"
    return v


def diagnostico(tutor=None) -> str:
    """Las versiones instaladas y si la app puede describir su API.

    Esto último es la comprobación que importa: si `get_api_info()` lanza, el
    enlace público devuelve *Internal Server Error* por mucho que las versiones
    parezcan bien.
    """
    lineas = ["Versiones instaladas:"]
    v = versiones()
    lineas += [f"  {k:14s} {val}" for k, val in v.items()]

    pyd = v.get("pydantic", "ausente")
    if pyd != "ausente":
        try:
            mayor, menor = (int(x) for x in pyd.split(".")[:2])
            if (mayor, menor) >= (2, 11):
                lineas.append(
                    f"  AVISO: pydantic {pyd} emite `additionalProperties: true`; "
                    "con gradio_client 1.3.0 eso rompe la página. Fija "
                    "pydantic<2.11 o deja que el parche de compatibilidad actúe.")
        except ValueError:
            pass

    info = aplicar()
    lineas.append(f"Parche de compatibilidad aplicado: {info['parche_aplicado']}")
    lineas.append(f"El fallo se estaba disparando: {info['se_disparaba']}")
    if info["motivo"]:
        lineas.append(f"  {info['motivo']}")

    lineas.append(comprobar_api(tutor))
    return "\n".join(lineas)


def comprobar_api(tutor=None) -> str:
    """Construye la interfaz y pide su descripción de API. No arranca servidor.

    Es la prueba de que la página se puede servir: la misma llamada que fallaba.
    """
    try:
        import app as interfaz

        demo = interfaz.construir()
        info = demo.get_api_info()
        n = len(info.get("named_endpoints", {})) if info else 0
        return (f"get_api_info(): OK · {n} endpoints descritos. La página se "
                "puede servir.")
    except TypeError as e:
        if "bool" in str(e):
            return (f"get_api_info(): FALLA con el error conocido ({e}). El "
                    "enlace público devolverá Internal Server Error. Fija "
                    "pydantic<2.11.")
        return f"get_api_info(): FALLA · TypeError: {e}"
    except Exception as e:      # noqa: BLE001
        return f"get_api_info(): FALLA · {type(e).__name__}: {e}"


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    print(diagnostico())
