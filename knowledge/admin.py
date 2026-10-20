from django.contrib import admin

from .models import Chunk, Collection, Document, Question


@admin.register(Collection)
class CollectionAdmin(admin.ModelAdmin):
    list_display = ["name", "owner", "created_at"]
    search_fields = ["name"]


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ["title", "collection", "status", "text_chars", "embedding_model", "created_at"]
    list_filter = ["status", "embedding_model"]
    search_fields = ["title", "filename"]


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ["document", "index", "start_char", "end_char"]
    exclude = ["embedding"]
    readonly_fields = ["document", "collection", "index", "text", "start_char", "end_char"]


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = [
        "text",
        "collection",
        "grounded",
        "provider",
        "model",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "created_at",
    ]
    list_filter = ["grounded", "provider"]
    readonly_fields = [f.name for f in Question._meta.fields]
