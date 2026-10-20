import hashlib
from pathlib import PurePath

from django.conf import settings
from django.db import IntegrityError, transaction

from knowledge.models import Chunk, Collection, Document

from .chunking import chunk_text
from .embeddings import Embedder
from .extraction import ExtractionError, extract_text


def ingest(
    collection: Collection, filename: str, data: bytes, embedder: Embedder, user=None, title: str = ""
) -> tuple[Document, bool]:
    """Extracts, chunks and embeds one file. Returns (document, created).

    Idempotent per collection: uploading identical bytes again returns the existing
    document and does no embedding work.
    """
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise ExtractionError(f"The file is larger than {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise ExtractionError("The file is empty.")
    sha256 = hashlib.sha256(data).hexdigest()
    existing = Document.objects.filter(collection=collection, sha256=sha256).first()
    if existing:
        return existing, False

    extracted = extract_text(filename, data)
    config = settings.RAG
    pieces = chunk_text(extracted.text, config["CHUNK_CHARS"], config["CHUNK_OVERLAP"]) if extracted.text else []
    # Embed before opening the transaction so no database lock is held during the network call.
    vectors = embedder.embed([p.text for p in pieces], "document") if pieces else []

    try:
        with transaction.atomic():
            document = Document.objects.create(
                collection=collection,
                title=(title.strip() or PurePath(filename).stem)[:255],
                filename=PurePath(filename).name[:255],
                content_type=extracted.content_type,
                size_bytes=len(data),
                sha256=sha256,
                status=Document.Status.READY if pieces else Document.Status.EMPTY,
                text_chars=len(extracted.text),
                embedding_model=embedder.model_name,
                uploaded_by=user,
            )
            Chunk.objects.bulk_create(
                Chunk(
                    document=document,
                    collection=collection,
                    index=p.index,
                    text=p.text,
                    start_char=p.start_char,
                    end_char=p.end_char,
                    embedding=v,
                )
                for p, v in zip(pieces, vectors, strict=True)
            )
    except IntegrityError:
        # A concurrent upload of the same file won the race.
        return Document.objects.get(collection=collection, sha256=sha256), False
    return document, True
