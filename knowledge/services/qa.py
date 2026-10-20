import time

from django.conf import settings

from knowledge.models import Collection, Question

from .answering import NOT_FOUND, Answer, Answerer, get_answerer
from .embeddings import Embedder, get_embedder
from .retrieval import Passage, retrieve


def ask(
    collection: Collection, question: str, user=None, embedder: Embedder | None = None, answerer: Answerer | None = None
) -> tuple[Question, list[Passage]]:
    """Retrieves passages, answers from them and records the exchange.

    With no relevant passages the model is not called at all: the answer is a
    plain "not found", which costs nothing and cannot be a hallucination.
    """
    config = settings.RAG
    started = time.perf_counter()
    passages = retrieve(collection, question, embedder or get_embedder(), config["TOP_K"], config["MIN_SIMILARITY"])
    if passages:
        result = (answerer or get_answerer()).answer(question, passages)
    else:
        result = Answer(NOT_FOUND, grounded=False, provider="none")
    record = Question.objects.create(
        collection=collection,
        asked_by=user,
        text=question,
        answer=result.text,
        citations=result.citations,
        retrieved_chunk_ids=[p.chunk_id for p in passages],
        grounded=result.grounded,
        provider=result.provider,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return record, passages
