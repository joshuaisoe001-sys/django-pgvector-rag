from io import StringIO

import pytest
from django.core.management import CommandError, call_command

from knowledge.models import Collection, Document


def test_seed_loads_samples_and_is_repeatable(db, settings):
    settings.DEBUG = True
    out = StringIO()
    call_command("seed_demo", stdout=out)
    call_command("seed_demo", stdout=out)
    collection = Collection.objects.get(owner__username="demo")
    assert Document.objects.filter(collection=collection).count() == 3
    assert "Already indexed" in out.getvalue()


def test_seed_refuses_to_run_without_debug(db, settings):
    settings.DEBUG = False
    with pytest.raises(CommandError):
        call_command("seed_demo")
