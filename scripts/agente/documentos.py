"""
El RAG de S08 convertido en una herramienta más (S10: "el retrieval también es
una herramienta").

No se reimplementa nada: la función recibe la búsqueda que ya existe
(`buscar_con_rerank` de S08) y la envuelve como `buscar_documentos`. El formato
del resultado es el mismo "[Fuente: ...]" del prompt de S07, así el modelo ve
los pasajes igual que cuando el contexto se inyectaba fijo.
"""

from __future__ import annotations

from typing import Callable

from registro import RegistroHerramientas  # [local]


def formatear_pasajes(pasajes: list[tuple[str, str]]) -> str:
    """[(fuente, texto)] -> bloque de texto con el formato de S07."""
    return "\n\n".join(f"[Fuente: {fuente}]\n{texto}" for fuente, texto in pasajes)


def registro_con_documentos(base: RegistroHerramientas, buscar: Callable[[str, int], list[int]],
                            chunks: list[str], metadatos: list[dict], ids_chunks: list[str],
                            k: int = 3) -> RegistroHerramientas:
    """Copia de `base` + `buscar_documentos`. `base` no se modifica."""
    registro = base.subconjunto()

    @registro.herramienta(familia="documentos")
    def buscar_documentos(consulta: str) -> dict:
        """Busca en los documentos oficiales del colegio (sistema institucional de
        evaluación, plan de área de matemáticas, protocolos y guías) y devuelve los
        pasajes más relevantes con su fuente.

        Args:
            consulta: qué información buscar, por ejemplo 'porcentaje del examen final'.
        """
        idxs = buscar(consulta, k)
        return {"resultado": formatear_pasajes([(metadatos[i]["fuente"], chunks[i]) for i in idxs]),
                "valor": None,
                "chunks": [chunks[i] for i in idxs],
                "ids": [ids_chunks[i] for i in idxs]}

    return registro


def chunks_de_resultado(resultado) -> list[str]:
    """Para `AgenteTutor(extraer_contextos=...)`: los chunks que devolvió la búsqueda."""
    return list(resultado.extra.get("chunks", []))
