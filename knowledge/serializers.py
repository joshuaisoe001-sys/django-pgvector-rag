from rest_framework import serializers

from .models import Collection, Document, Question


class CollectionSerializer(serializers.ModelSerializer):
    document_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Collection
        fields = ["id", "name", "description", "document_count", "created_at"]
        read_only_fields = ["created_at"]

    def validate_name(self, value):
        value = value.strip()
        owner = self.context["request"].user
        clash = Collection.objects.filter(owner=owner, name__iexact=value)
        if self.instance:
            clash = clash.exclude(pk=self.instance.pk)
        if clash.exists():
            raise serializers.ValidationError("You already have a collection with this name.")
        return value


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Document
        fields = [
            "id",
            "collection",
            "title",
            "filename",
            "content_type",
            "size_bytes",
            "sha256",
            "status",
            "text_chars",
            "chunk_count",
            "embedding_model",
            "created_at",
        ]
        read_only_fields = fields


class UploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    title = serializers.CharField(max_length=255, required=False, allow_blank=True)


class AskSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=2000, trim_whitespace=True)


class PassageSerializer(serializers.Serializer):
    chunk_id = serializers.IntegerField()
    document_id = serializers.IntegerField()
    document_title = serializers.CharField()
    text = serializers.CharField()
    similarity = serializers.FloatField()


class QuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Question
        fields = [
            "id",
            "text",
            "answer",
            "grounded",
            "citations",
            "provider",
            "model",
            "input_tokens",
            "output_tokens",
            "latency_ms",
            "created_at",
        ]
        read_only_fields = fields
