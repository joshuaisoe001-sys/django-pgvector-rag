from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient

from knowledge.models import Chunk, Document
from tests.helpers import LEAVE_POLICY, TRAVEL_POLICY, make_pdf, text_file


def upload(api, collection, file, **extra):
    return api.post(f"/api/collections/{collection['id']}/documents/", {"file": file, **extra}, format="multipart")


def test_upload_extracts_chunks_and_embeds(api, collection):
    response = upload(api, collection, text_file("leave-policy.txt", LEAVE_POLICY), title="Leave policy 2026")
    assert response.status_code == 201, response.data
    data = response.data
    assert data["title"] == "Leave policy 2026"
    assert data["status"] == "ready"
    assert data["embedding_model"] == "hashing-v2"
    assert data["chunk_count"] >= 1
    chunk = Chunk.objects.get(document_id=data["id"], index=0)
    assert "21 working days" in chunk.text
    assert len(chunk.embedding) == 1024


def test_title_defaults_to_the_file_name(api, collection):
    assert upload(api, collection, text_file("travel-policy.md", TRAVEL_POLICY)).data["title"] == "travel-policy"


def test_uploading_the_same_file_twice_returns_the_existing_document(api, collection):
    first = upload(api, collection, text_file("leave.txt", LEAVE_POLICY))
    second = upload(api, collection, text_file("leave-copy.txt", LEAVE_POLICY))
    assert (first.status_code, second.status_code) == (201, 200)
    assert first.data["id"] == second.data["id"]
    assert Document.objects.filter(collection_id=collection["id"]).count() == 1


def test_pdf_upload(api, collection):
    pdf = SimpleUploadedFile("leave.pdf", make_pdf("Leave is 21 working days per year."), content_type="application/pdf")
    response = upload(api, collection, pdf)
    assert response.status_code == 201
    assert response.data["content_type"] == "application/pdf"


def test_pdf_without_text_is_stored_as_empty(api, collection):
    pdf = SimpleUploadedFile("scan.pdf", make_pdf(""), content_type="application/pdf")
    response = upload(api, collection, pdf)
    assert response.status_code == 201
    assert (response.data["status"], response.data["chunk_count"]) == ("empty", 0)


def test_unsupported_and_oversized_files_are_rejected(api, collection, settings):
    assert upload(api, collection, SimpleUploadedFile("photo.png", b"\x89PNG")).status_code == 400
    settings.MAX_UPLOAD_BYTES = 100
    response = upload(api, collection, text_file("big.txt", "x" * 101))
    assert response.status_code == 400
    assert "larger than" in str(response.data["file"])
    assert not Document.objects.filter(collection_id=collection["id"]).exists()


def test_list_retrieve_and_delete_documents(api, collection):
    doc_id = upload(api, collection, text_file("leave.txt", LEAVE_POLICY)).data["id"]
    listed = api.get(f"/api/collections/{collection['id']}/documents/").data["results"]
    assert [d["id"] for d in listed] == [doc_id]
    assert api.get(f"/api/documents/{doc_id}/").data["filename"] == "leave.txt"
    assert api.delete(f"/api/documents/{doc_id}/").status_code == 204
    assert not Chunk.objects.filter(document_id=doc_id).exists()


def test_other_users_cannot_read_or_delete_documents(api, collection, other_user):
    doc_id = upload(api, collection, text_file("leave.txt", LEAVE_POLICY)).data["id"]
    intruder = APIClient()
    intruder.force_authenticate(other_user)
    assert intruder.get(f"/api/documents/{doc_id}/").status_code == 404
    assert intruder.delete(f"/api/documents/{doc_id}/").status_code == 404
    assert Document.objects.filter(id=doc_id).exists()
