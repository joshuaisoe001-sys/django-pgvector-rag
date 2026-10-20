"""Retrieval runs the real pgvector cosine-distance query against PostgreSQL."""

from knowledge.models import Collection, Document
from knowledge.services.embeddings import HashingEmbedder
from knowledge.services.retrieval import retrieve
from tests.helpers import IT_POLICY, LEAVE_POLICY, TRAVEL_POLICY, text_file


def load_policies(api, collection):
    for name, text in [("leave.txt", LEAVE_POLICY), ("travel.txt", TRAVEL_POLICY), ("it.txt", IT_POLICY)]:
        api.post(f"/api/collections/{collection['id']}/documents/", {"file": text_file(name, text)}, format="multipart")


def search(api, collection, question):
    response = api.post(f"/api/collections/{collection['id']}/search/", {"question": question}, format="json")
    assert response.status_code == 200, response.data
    return response.data["results"]


def test_most_relevant_document_ranks_first(api, collection):
    load_policies(api, collection)
    assert search(api, collection, "How many days of annual leave do employees get?")[0]["document_title"] == "leave"
    assert search(api, collection, "What is the hotel cap per night when travelling?")[0]["document_title"] == "travel"
    assert search(api, collection, "How long must passwords be?")[0]["document_title"] == "it"


def test_results_are_sorted_by_similarity(api, collection):
    load_policies(api, collection)
    scores = [r["similarity"] for r in search(api, collection, "leave days carried over")]
    assert scores == sorted(scores, reverse=True)


def test_unrelated_questions_retrieve_nothing_at_the_default_threshold(api, collection):
    load_policies(api, collection)
    assert search(api, collection, "What is the capital of Peru?") == []


def test_chunks_from_a_different_embedding_model_are_ignored(api, collection):
    load_policies(api, collection)
    Document.objects.filter(title="leave").update(embedding_model="another-model")
    titles = {r["document_title"] for r in search(api, collection, "annual leave days")}
    assert "leave" not in titles


def test_retrieval_never_crosses_collections(api, collection, user):
    load_policies(api, collection)
    empty = Collection.objects.create(owner=user, name="Empty")
    assert retrieve(empty, "annual leave", HashingEmbedder(1024), top_k=6, min_similarity=0.0) == []
