"""Embedding providers.

Claude does not produce embeddings; Anthropic recommends Voyage AI for that, so the
semantic provider calls Voyage's REST API. The hashing provider needs no network or
key: it is lexical (shared words, not shared meaning) and exists so tests and demos
run anywhere. Both return unit-length vectors of the column's dimension.
"""

import hashlib
import math
import os
import re
from typing import Literal, Protocol

import httpx2
from django.conf import settings

InputType = Literal["document", "query"]
_WORD = re.compile(r"[a-z0-9]+")
# Words too common to say anything about a passage's topic.
_STOP_WORDS = frozenset(
    "a an and are as at be been but by can could did do does for from get gets got had has have how i if in into is it its "
    "may me might must my no not of on or our shall should so than that the their them then there these they this those to "
    "up us was we were what when where which who whom why will with would you your".split()
)
_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(word: str) -> str:
    """Crude suffix stripping so 'travelling', 'travels' and 'travel' share a feature."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    # 'travell' -> 'travel', 'capp' -> 'cap' after stripping a suffix from a doubled consonant.
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]
    if len(word) > 4 and word.endswith("ll"):
        word = word[:-1]
    return word


def terms(text: str) -> list[str]:
    """The stemmed content words of a text, in order."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP_WORDS]


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]: ...


class HashingEmbedder:
    """Feature hashing of stemmed word unigrams and bigrams into a fixed-size, L2-normalised vector."""

    model_name = "hashing-v2"

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        words = terms(text)
        vector = [0.0] * self.dimensions
        for feature in [*words, *(f"{a} {b}" for a, b in zip(words, words[1:], strict=False))]:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[slot] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            # pgvector rejects cosine distance on a zero vector; use a fixed unit vector.
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


class VoyageEmbedder:
    url = "https://api.voyageai.com/v1/embeddings"
    batch_size = 64

    def __init__(self, model: str, dimensions: int, api_key: str, client: httpx2.Client | None = None):
        self.model_name = model
        self.dimensions = dimensions
        self.api_key = api_key
        self.client = client or httpx2.Client(timeout=30.0)

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._batch(texts[start : start + self.batch_size], input_type))
        return vectors

    def _batch(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        try:
            response = self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": texts, "model": self.model_name, "input_type": input_type, "output_dimension": self.dimensions},
            )
        except httpx2.HTTPError as exc:
            raise EmbeddingError("Could not reach the embedding service.") from exc
        if response.status_code != 200:
            raise EmbeddingError(f"The embedding service returned HTTP {response.status_code}.")
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if len(vectors) != len(texts) or any(len(v) != self.dimensions for v in vectors):
            raise EmbeddingError("The embedding service returned vectors of the wrong shape.")
        return vectors


def get_embedder() -> Embedder:
    config = settings.RAG
    dimensions = config["EMBEDDING_DIMENSIONS"]
    if config["EMBEDDING_PROVIDER"] == "voyage":
        api_key = os.environ.get("VOYAGE_API_KEY", "")
        if not api_key:
            raise EmbeddingError("EMBEDDING_PROVIDER is 'voyage' but VOYAGE_API_KEY is not set.")
        return VoyageEmbedder(config["EMBEDDING_MODEL"], dimensions, api_key)
    return HashingEmbedder(dimensions)
