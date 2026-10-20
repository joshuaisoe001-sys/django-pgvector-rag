import secrets
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from knowledge.models import Collection
from knowledge.services.embeddings import get_embedder
from knowledge.services.ingestion import ingest

SAMPLES = Path(settings.BASE_DIR) / "samples"


class Command(BaseCommand):
    help = "Creates a demo user and a collection loaded with the fictional sample policies in samples/."

    def handle(self, *args, **options):
        if not settings.DEBUG:
            raise CommandError("seed_demo only runs with DJANGO_DEBUG on.")
        User = get_user_model()
        password = secrets.token_urlsafe(12)
        user, created = User.objects.get_or_create(username="demo", defaults={"is_staff": True})
        user.is_staff = True
        user.set_password(password)
        user.save()

        collection, _ = Collection.objects.get_or_create(
            owner=user, name="Baobab Logistics handbook", defaults={"description": "Fictional company policies for the demo."}
        )
        embedder = get_embedder()
        for path in sorted(SAMPLES.glob("*.md")):
            document, new = ingest(collection, path.name, path.read_bytes(), embedder, user=user)
            self.stdout.write(f"{'Indexed' if new else 'Already indexed'}: {document.title} ({document.chunks.count()} chunks)")

        self.stdout.write(self.style.SUCCESS(f"Sign in at /admin/login/ as 'demo' with password: {password}"))
