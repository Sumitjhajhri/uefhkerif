"""Retrieval.

`Retriever` is an interface with one default implementation, `TfidfRetriever`.

TF-IDF rather than neural embeddings by default, deliberately:

  * the test suite runs offline, in under a second, at no cost, and returns
    the same ranking every time - a retriever that varies run to run makes a
    regression impossible to distinguish from noise;
  * it keeps the repository free of a multi-hundred-megabyte model download;
  * retrieval quality is not what this project is measuring. The measurement
    is what the agent does with whatever it retrieved, including when what it
    retrieved does not contain the answer.

`EmbeddingRetriever` is the seam where a real embedding model goes. Swapping
it changes recall; it does not change the harness or the graders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import Chunk


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Retrieved]: ...


class TfidfRetriever:
    """Cosine similarity over TF-IDF vectors, with a relevance floor.

    The floor matters more than the ranking. Without it every query returns
    the k best chunks no matter how irrelevant, so a question the corpus
    cannot answer still arrives at the model wrapped in confident-looking
    context - which is exactly the condition that produces a fabricated
    answer. Returning nothing is a signal; returning noise is a trap.
    """

    def __init__(self, chunks: Sequence[Chunk], min_score: float = 0.05):
        if not chunks:
            raise ValueError("TfidfRetriever requires at least one chunk")
        if not 0.0 <= min_score < 1.0:
            raise ValueError("min_score must be in [0.0, 1.0)")

        self.chunks = list(chunks)
        self.min_score = min_score
        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform(c.text for c in self.chunks)

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not query.strip():
            return []

        query_vector = self._vectorizer.transform([query])
        # Rows are L2-normalised by TfidfVectorizer, so the dot product is cosine.
        scores = (self._matrix @ query_vector.T).toarray().ravel()

        ranked = np.argsort(-scores)[:k]
        return [
            Retrieved(chunk=self.chunks[i], score=float(scores[i]))
            for i in ranked
            if scores[i] >= self.min_score
        ]


class EmbeddingRetriever:
    """Placeholder for a production embedding retriever.

    Left unimplemented on purpose rather than half-built: the interface is
    what the rest of the code depends on, and an untested implementation that
    silently returned poor neighbours would corrupt every number in the eval
    report without raising anything.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "EmbeddingRetriever is the extension point for a real embedding "
            "model (e.g. sentence-transformers or a hosted embeddings API). "
            "Implement `search` and the agent, harness and graders work unchanged."
        )
