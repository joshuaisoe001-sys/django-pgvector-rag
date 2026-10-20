from dataclasses import dataclass

from pgvector.django import CosineDistance

from knowledge.models import Chunk, Collection

from .embeddings import Embedder


@dataclass(frozen=True)
class Passage:
    chunk_id: int
    document_id: int
    document_title: str
    text: str
    similarity: float


def retrieve(collection: Collection, query: str, embedder: Embedder, top_k: int, min_similarity: float) -> list[Passage]:
    """Returns the chunks closest to the query by cosine similarity, most similar first.

    Only chunks embedded by the same model as the query are compared: vectors from
    different models live in different spaces and their distances mean nothing.
    """
    [query_vector] = embedder.embed([query], "query")
    rows = (
        Chunk.objects.filter(collection=collection, document__embedding_model=embedder.model_name)
        .select_related("document")
        .annotate(distance=CosineDistance("embedding", query_vector))
        .order_by("distance")[:top_k]
    )
    passages = [Passage(row.id, row.document_id, row.document.title, row.text, round(1.0 - row.distance, 4)) for row in rows]
    return [p for p in passages if p.similarity >= min_similarity]
