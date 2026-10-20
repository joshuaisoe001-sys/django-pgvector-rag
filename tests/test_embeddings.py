import json
import math

import httpx2
import pytest

from knowledge.services.embeddings import EmbeddingError, HashingEmbedder, VoyageEmbedder, get_embedder


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestHashingEmbedder:
    embedder = HashingEmbedder(1024)

    def test_vectors_are_unit_length_and_deterministic(self):
        [a, b] = self.embedder.embed(["annual leave days", "annual leave days"], "document")
        assert len(a) == 1024
        assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0)
        assert a == b

    def test_shared_words_score_higher_than_unrelated_text(self):
        [query, related, unrelated] = self.embedder.embed(
            ["how many days of annual leave", "employees get 21 days of annual leave", "laptops need disk encryption"],
            "document",
        )
        assert cosine(query, related) > cosine(query, unrelated)

    def test_stop_words_are_ignored_and_word_forms_match(self):
        [a, b] = self.embedder.embed(["What is the hotel cap?", "hotel capped"], "query")
        assert math.isclose(cosine(a, b), 1.0)
        [question, unrelated] = self.embedder.embed(
            ["What is the capital of Peru?", "The leave policy is in the HR portal."], "query"
        )
        assert cosine(question, unrelated) == 0.0

    def test_text_without_words_gets_a_valid_unit_vector(self):
        [vector] = self.embedder.embed(["!!! ---"], "query")
        assert sum(v * v for v in vector) == 1.0


class TestVoyageEmbedder:
    def make(self, handler, dimensions=4):
        client = httpx2.Client(transport=httpx2.MockTransport(handler))
        return VoyageEmbedder("voyage-3.5", dimensions, "test-key", client=client)

    def test_sends_model_input_type_and_key_and_restores_order(self):
        seen = {}

        def handler(request):
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            # Voyage may return items in any order; each carries its input index.
            return httpx2.Response(
                200, json={"data": [{"index": 1, "embedding": [0, 1, 0, 0]}, {"index": 0, "embedding": [1, 0, 0, 0]}]}
            )

        vectors = self.make(handler).embed(["first", "second"], "query")
        assert vectors == [[1, 0, 0, 0], [0, 1, 0, 0]]
        assert seen["auth"] == "Bearer test-key"
        assert seen["body"] == {"input": ["first", "second"], "model": "voyage-3.5", "input_type": "query", "output_dimension": 4}

    def test_large_inputs_are_sent_in_batches(self):
        sizes = []

        def handler(request):
            texts = json.loads(request.content)["input"]
            sizes.append(len(texts))
            return httpx2.Response(200, json={"data": [{"index": i, "embedding": [1, 0, 0, 0]} for i in range(len(texts))]})

        assert len(self.make(handler).embed([f"t{i}" for i in range(150)], "document")) == 150
        assert sizes == [64, 64, 22]

    def test_http_errors_become_embedding_errors(self):
        embedder = self.make(lambda request: httpx2.Response(401, json={"detail": "bad key"}))
        with pytest.raises(EmbeddingError, match="HTTP 401"):
            embedder.embed(["x"], "query")

    def test_wrong_dimensions_are_rejected(self):
        embedder = self.make(lambda request: httpx2.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]}))
        with pytest.raises(EmbeddingError, match="wrong shape"):
            embedder.embed(["x"], "query")


def test_voyage_provider_requires_a_key(settings, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    settings.RAG = {**settings.RAG, "EMBEDDING_PROVIDER": "voyage"}
    with pytest.raises(EmbeddingError, match="VOYAGE_API_KEY"):
        get_embedder()
