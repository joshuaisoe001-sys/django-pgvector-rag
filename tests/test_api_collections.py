from rest_framework.test import APIClient

from tests.helpers import LEAVE_POLICY, text_file


def test_api_requires_authentication(db):
    assert APIClient().get("/api/collections/").status_code == 401


def test_token_authentication_works(user):
    token = APIClient().post("/api/auth/token/", {"username": "amina", "password": "a-long-test-passphrase"}).data["token"]
    client = APIClient()
    client.credentials(HTTP_AUTHORIZATION=f"Token {token}")
    assert client.get("/api/collections/").status_code == 200


def test_create_and_list_collections_with_document_counts(api, collection):
    api.post(
        f"/api/collections/{collection['id']}/documents/", {"file": text_file("leave.txt", LEAVE_POLICY)}, format="multipart"
    )
    [listed] = api.get("/api/collections/").data["results"]
    assert listed["name"] == "HR policies"
    assert listed["document_count"] == 1


def test_duplicate_names_are_rejected_case_insensitively(api, collection):
    response = api.post("/api/collections/", {"name": "hr POLICIES"}, format="json")
    assert response.status_code == 400
    assert "already have a collection" in str(response.data["name"])


def test_other_users_cannot_see_or_use_a_collection(collection, other_user):
    intruder = APIClient()
    intruder.force_authenticate(other_user)
    base = f"/api/collections/{collection['id']}"
    assert intruder.get("/api/collections/").data["results"] == []
    # 404 rather than 403, so the existence of the collection is not revealed.
    assert intruder.get(f"{base}/").status_code == 404
    assert intruder.post(f"{base}/documents/", {"file": text_file("x.txt", "hello")}, format="multipart").status_code == 404
    assert intruder.post(f"{base}/ask/", {"question": "leave?"}, format="json").status_code == 404
    assert intruder.post(f"{base}/search/", {"question": "leave?"}, format="json").status_code == 404
    assert intruder.delete(f"{base}/").status_code == 404


def test_two_users_may_use_the_same_collection_name(collection, other_user):
    client = APIClient()
    client.force_authenticate(other_user)
    assert client.post("/api/collections/", {"name": "HR policies"}, format="json").status_code == 201
