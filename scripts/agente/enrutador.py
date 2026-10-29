"""
Enrutador consulta / problema.

En S08 llevó el tutor de 14/21 a 18/21, y fue la única de las tres técnicas
avanzadas que movió el scorecard. Pero vivía dentro del notebook de S08, así que
el agente de S10 nunca lo usó: de sus 20 fallos, **5 son casos de conocimiento o
compuestos en los que simplemente no consultó los documentos**. Las dos piezas
existen y nunca se juntaron; esto es juntarlas.

La idea: antes de responder, decidir si la pregunta pide un DATO que vive en los
documentos del colegio o es un PROBLEMA que hay que calcular. Tres señales, cada
una con un punto ciego distinto:

    margen   similitud al centroide "consulta" menos al de "problema". Separa por
             TEMA, así que falla con "¿cuánto valen los talleres?" (suena a cálculo).
    numeros  cuántos números trae la pregunta. Para calcular hacen falta datos;
             pero "la mitad de un tercio" no trae ninguno.
    rerank   lo bien que el mejor chunk responde la pregunta. Falla con las
             preguntas institucionales cuya respuesta no está en el corpus.

Una regresión logística las combina. Dio 97.0 % en leave-one-out sobre las 33
consultas de desarrollo y 16/16 sobre el eval set, que no vio al ajustarse.

El módulo recibe `encode` y `rerank_max` como funciones, en vez de cargar modelos.
Así se prueba sin GPU y el notebook le pasa los de verdad.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

TIPOS = ("consulta", "problema")
RASGOS = ("margen", "numeros", "rerank")


def contar_numeros(pregunta: str, tope: int = 3) -> int:
    return min(len(re.findall(r"\d+(?:[.,/]\d+)?", pregunta or "")), tope)


@dataclass
class Decision:
    ruta: str
    probabilidad: float
    rasgos: tuple

    @property
    def es_consulta(self) -> bool:
        return self.ruta == "consulta"


class Enrutador:
    """`encode(textos) -> matriz normalizada`; `rerank_max(pregunta) -> float`.

    `rerank_max` es opcional: sin él, el enrutador usa dos señales en vez de tres.
    Pierde algo de exactitud pero sigue funcionando, lo que permite usarlo en
    entornos donde cargar el cross-encoder no compensa.
    """

    def __init__(self, encode, rerank_max=None, consultas=None):
        self.encode = encode
        self.rerank_max = rerank_max
        self._cache_rerank: dict = {}
        self.centroides: dict = {}
        self.modelo = None
        self.pesos: dict = {}
        if consultas:
            self.ajustar(consultas)

    # ------------------------------------------------------------------
    def _rerank(self, pregunta: str) -> float:
        if self.rerank_max is None:
            return 0.0
        if pregunta not in self._cache_rerank:
            self._cache_rerank[pregunta] = float(self.rerank_max(pregunta))
        return self._cache_rerank[pregunta]

    def _centroides(self, consultas) -> dict:
        emb = np.asarray(self.encode([c["q"] for c in consultas]))
        cent = {}
        for t in TIPOS:
            masc = np.array([c["tipo"] == t for c in consultas])
            if not masc.any():
                raise ValueError(f"no hay ejemplos de tipo {t!r}")
            v = emb[masc].mean(axis=0)
            cent[t] = v / np.linalg.norm(v)
        return cent

    def margen(self, pregunta: str, centroides=None) -> float:
        cent = centroides or self.centroides
        e = np.asarray(self.encode([pregunta]))[0]
        return float(e @ cent["consulta"] - e @ cent["problema"])

    def rasgos(self, pregunta: str, centroides=None) -> list:
        return [self.margen(pregunta, centroides), contar_numeros(pregunta), self._rerank(pregunta)]

    # ------------------------------------------------------------------
    def ajustar(self, consultas) -> "Enrutador":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.consultas = list(consultas)
        self.centroides = self._centroides(self.consultas)
        X = np.array([self.rasgos(c["q"]) for c in self.consultas])
        y = np.array([c["tipo"] == "consulta" for c in self.consultas])
        self.modelo = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
        self.pesos = dict(zip(RASGOS, map(float, self.modelo[-1].coef_[0])))
        return self

    def clasificar(self, pregunta: str) -> Decision:
        if self.modelo is None:
            raise RuntimeError("el enrutador no está ajustado: llama a ajustar() primero")
        r = self.rasgos(pregunta)
        p = float(self.modelo.predict_proba([r])[0][1])
        return Decision("consulta" if p >= 0.5 else "problema", p, tuple(r))

    # ------------------------------------------------------------------
    def leave_one_out(self) -> dict:
        """Validación honesta: en cada pliegue los centroides también se
        recalculan sin la consulta que se va a predecir. Si no, el margen ya la
        habría visto y la cifra saldría inflada."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y = np.array([c["tipo"] == "consulta" for c in self.consultas])
        aciertos, solo_centroides, errores = [], [], []
        for i in range(len(self.consultas)):
            resto = self.consultas[:i] + self.consultas[i + 1:]
            cent = self._centroides(resto)
            X = np.array([self.rasgos(c["q"], cent) for c in self.consultas])
            m = np.arange(len(self.consultas)) != i
            clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(X[m], y[m])
            pred = bool(clf.predict(X[i:i + 1])[0])
            aciertos.append(pred == y[i])
            solo_centroides.append((X[i, 0] > 0) == y[i])
            if pred != y[i]:
                errores.append({"q": self.consultas[i]["q"], "real": self.consultas[i]["tipo"]})
        return {"exactitud": float(np.mean(aciertos)),
                "solo_centroides": float(np.mean(solo_centroides)),
                "n": len(self.consultas), "errores": errores}


def contexto_enrutado(enrutador: Enrutador, buscar_contexto, registrar=None):
    """`contexto_fijo` para `AgenteTutor` que solo busca cuando hace falta.

    Devuelve los chunks si el enrutador dice "consulta", y una lista vacía si dice
    "problema". Así el agente mantiene las herramientas de cálculo y gana la vía
    documental sin tener que decidirlo él, que es justo lo que no supo hacer en
    S10: dejó 5 casos sin consultar.
    """
    def contexto(pregunta):
        decision = enrutador.clasificar(pregunta)
        if registrar is not None:
            registrar(pregunta, decision)
        return list(buscar_contexto(pregunta)) if decision.es_consulta else []
    return contexto
