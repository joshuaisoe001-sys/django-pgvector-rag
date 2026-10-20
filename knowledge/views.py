from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Count
from django.shortcuts import render
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle

from .models import Collection, Document
from .serializers import (
    AskSerializer,
    CollectionSerializer,
    DocumentSerializer,
    PassageSerializer,
    QuestionSerializer,
    UploadSerializer,
)
from .services.answering import AnswerError
from .services.embeddings import EmbeddingError, get_embedder
from .services.extraction import ExtractionError
from .services.ingestion import ingest
from .services.qa import ask as answer_question
from .services.retrieval import retrieve


class AskThrottle(UserRateThrottle):
    """Questions call paid model APIs, so they get their own per-user limit."""

    scope = "ask"


def service_unavailable(exc: Exception) -> Response:
    return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


class CollectionViewSet(viewsets.ModelViewSet):
    """A user's document collections. Other users' collections answer 404, never 403."""

    serializer_class = CollectionSerializer

    def get_queryset(self):
        return (
            Collection.objects.filter(owner=self.request.user).annotate(document_count=Count("documents")).order_by("-created_at")
        )

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

    @action(detail=True, methods=["get", "post"], parser_classes=[MultiPartParser, FormParser])
    def documents(self, request, pk=None):
        collection = self.get_object()
        if request.method == "GET":
            documents = collection.documents.annotate(chunk_count=Count("chunks")).order_by("-created_at")
            page = self.paginate_queryset(documents)
            return self.get_paginated_response(DocumentSerializer(page, many=True).data)

        upload = UploadSerializer(data=request.data)
        upload.is_valid(raise_exception=True)
        file = upload.validated_data["file"]
        if file.size > settings.MAX_UPLOAD_BYTES:
            return Response({"file": [f"The file is larger than {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."]}, status=400)
        try:
            document, created = ingest(
                collection,
                file.name,
                file.read(),
                get_embedder(),
                user=request.user,
                title=upload.validated_data.get("title", ""),
            )
        except ExtractionError as exc:
            return Response({"file": [str(exc)]}, status=status.HTTP_400_BAD_REQUEST)
        except EmbeddingError as exc:
            return service_unavailable(exc)
        document.chunk_count = document.chunks.count()
        return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)

    @action(detail=True, methods=["post"], throttle_classes=[AskThrottle])
    def ask(self, request, pk=None):
        collection = self.get_object()
        body = AskSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        try:
            record, passages = answer_question(collection, body.validated_data["question"], user=request.user)
        except (EmbeddingError, AnswerError) as exc:
            return service_unavailable(exc)
        return Response({**QuestionSerializer(record).data, "sources": PassageSerializer(passages, many=True).data})

    @action(detail=True, methods=["post"])
    def search(self, request, pk=None):
        """Retrieval only: the passages a question would be answered from, with no model call."""
        collection = self.get_object()
        body = AskSerializer(data=request.data)
        body.is_valid(raise_exception=True)
        config = settings.RAG
        try:
            passages = retrieve(
                collection, body.validated_data["question"], get_embedder(), config["TOP_K"], config["MIN_SIMILARITY"]
            )
        except EmbeddingError as exc:
            return service_unavailable(exc)
        return Response({"results": PassageSerializer(passages, many=True).data})

    @action(detail=True, methods=["get"])
    def questions(self, request, pk=None):
        collection = self.get_object()
        page = self.paginate_queryset(collection.questions.all())
        return self.get_paginated_response(QuestionSerializer(page, many=True).data)


class DocumentViewSet(mixins.RetrieveModelMixin, mixins.DestroyModelMixin, viewsets.GenericViewSet):
    serializer_class = DocumentSerializer

    def get_queryset(self):
        return (
            Document.objects.filter(collection__owner=self.request.user)
            .annotate(chunk_count=Count("chunks"))
            .order_by("-created_at")
        )


@login_required
def index(request):
    return render(request, "knowledge/index.html")
