import pytest
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.management import call_command
from rest_framework.test import APIClient

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


@pytest.fixture(scope="session")
def django_db_setup(django_db_blocker):
    """Runs the tests in the configured database instead of creating a test database.

    The PGlite dev server holds a single database and cannot CREATE DATABASE, and a
    CI service container is disposable anyway. Each test still runs in a transaction
    that is rolled back. To avoid touching real data, only local databases are allowed.
    """
    host = settings.DATABASES["default"]["HOST"]
    if host not in LOCAL_HOSTS:
        pytest.exit(f"Refusing to run tests against non-local database host '{host}'.", returncode=2)
    with django_db_blocker.unblock():
        call_command("migrate", verbosity=0, interactive=False)


@pytest.fixture(autouse=True)
def offline_providers(settings):
    """Tests never call Voyage or Claude; the Claude client is tested with a fake."""
    settings.RAG = {**settings.RAG, "EMBEDDING_PROVIDER": "hashing", "ANSWER_PROVIDER": "extractive"}
    cache.clear()


@pytest.fixture
def user(db):
    return get_user_model().objects.create_user("amina", password="a-long-test-passphrase")


@pytest.fixture
def other_user(db):
    return get_user_model().objects.create_user("brian", password="a-long-test-passphrase")


@pytest.fixture
def api(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.fixture
def collection(api):
    response = api.post("/api/collections/", {"name": "HR policies"}, format="json")
    assert response.status_code == 201, response.data
    return response.data
