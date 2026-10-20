from django.conf import settings
from django.db import models
from pgvector.django import HnswIndex, VectorField

EMBEDDING_DIMENSIONS = settings.RAG["EMBEDDING_DIMENSIONS"]


class Collection(models.Model):
    """A set of documents owned by one user. Every query is scoped to one collection."""

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="collections")
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["owner", "name"], name="unique_collection_name_per_owner")]

    def __str__(self):
        return self.name


class Document(models.Model):
    class Status(models.TextChoices):
        READY = "ready", "Ready"
        EMPTY = "empty", "No text found"

    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=255)
    filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=100)
    size_bytes = models.PositiveIntegerField()
    # Uploading the same file twice into a collection returns the existing document.
    sha256 = models.CharField(max_length=64)
    status = models.CharField(max_length=10, choices=Status.choices)
    text_chars = models.PositiveIntegerField(default=0)
    embedding_model = models.CharField(max_length=100)
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["collection", "sha256"], name="unique_document_per_collection")]

    def __str__(self):
        return self.title


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    # Denormalised from document so retrieval filters on one indexed column.
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name="+")
    index = models.PositiveIntegerField()
    text = models.TextField()
    start_char = models.PositiveIntegerField()
    end_char = models.PositiveIntegerField()
    embedding = VectorField(dimensions=EMBEDDING_DIMENSIONS)

    class Meta:
        ordering = ["document_id", "index"]
        constraints = [models.UniqueConstraint(fields=["document", "index"], name="unique_chunk_index")]
        indexes = [
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    def __str__(self):
        return f"{self.document.title} #{self.index}"


class Question(models.Model):
    """Every question asked, with the answer, its citations and the model's usage."""

    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, related_name="questions")
    asked_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name="+")
    text = models.TextField()
    answer = models.TextField()
    # [{"chunk_id", "document_id", "document_title", "cited_text"}, ...]
    citations = models.JSONField(default=list)
    retrieved_chunk_ids = models.JSONField(default=list)
    grounded = models.BooleanField(help_text="False when no relevant passages were found or the model declined.")
    provider = models.CharField(max_length=40)
    model = models.CharField(max_length=100, blank=True)
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    latency_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.text[:80]
