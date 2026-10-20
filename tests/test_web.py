def test_page_requires_sign_in(client, db):
    response = client.get("/")
    assert response.status_code == 302
    assert response["Location"].startswith("/admin/login/")


def test_page_renders_for_signed_in_users(client, user):
    client.force_login(user)
    response = client.get("/")
    assert response.status_code == 200
    assert b"Document Q&amp;A" in response.content
    assert b"knowledge/app.js" in response.content
