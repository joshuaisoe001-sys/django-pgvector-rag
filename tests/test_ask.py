from unittest import mock

from knowledge.models import Chunk, Question
from knowledge.services.answering import NOT_FOUND, AnswerError
from tests.helpers import IT_POLICY, LEAVE_POLICY, text_file


def ask(api, collection, question):
    return api.post(f"/api/collections/{collection['id']}/ask/", {"question": question}, format="json")


def load(api, collection):
    for name, text in [("leave.txt", LEAVE_POLICY), ("it.txt", IT_POLICY)]:
        api.post(f"/api/collections/{collection['id']}/documents/", {"file": text_file(name, text)}, format="multipart")


def test_answer_cites_a_real_chunk_and_is_recorded(api, collection, user):
    load(api, collection)
    response = ask(api, collection, "How many days of annual leave do I get?")
    assert response.status_code == 200, response.data
    data = response.data
    assert data["grounded"] is True
    assert data["provider"] == "extractive"
    [citation] = data["citations"]
    chunk = Chunk.objects.get(id=citation["chunk_id"])
    assert citation["cited_text"] in chunk.text
    assert "21 working days" in data["answer"]
    assert data["sources"][0]["chunk_id"] == chunk.id

    record = Question.objects.get(id=data["id"])
    assert record.asked_by == user
    assert record.retrieved_chunk_ids[0] == chunk.id


def test_no_relevant_passages_answers_not_found_without_calling_a_model(api, collection):
    load(api, collection)
    with mock.patch("knowledge.services.qa.get_answerer") as get_answerer:
        response = ask(api, collection, "Who won the 2010 World Cup?")
    get_answerer.assert_not_called()
    assert response.data["answer"] == NOT_FOUND
    assert (response.data["grounded"], response.data["provider"], response.data["sources"]) == (False, "none", [])


def test_empty_collection_answers_not_found(api, collection):
    assert ask(api, collection, "Anything?").data["answer"] == NOT_FOUND


def test_model_errors_return_503_with_a_readable_message(api, collection):
    load(api, collection)
    failing = mock.Mock()
    failing.answer.side_effect = AnswerError("Could not reach the Claude API.")
    with mock.patch("knowledge.services.qa.get_answerer", return_value=failing):
        response = ask(api, collection, "How many days of annual leave do I get?")
    assert response.status_code == 503
    assert response.data["detail"] == "Could not reach the Claude API."
    assert not Question.objects.filter(collection_id=collection["id"]).exists()


def test_questions_are_validated(api, collection):
    assert ask(api, collection, "").status_code == 400
    assert ask(api, collection, "x" * 2001).status_code == 400


def test_question_history(api, collection):
    load(api, collection)
    ask(api, collection, "How long must passwords be?")
    [entry] = api.get(f"/api/collections/{collection['id']}/questions/").data["results"]
    assert entry["text"] == "How long must passwords be?"


def test_asking_is_rate_limited_per_user(api, collection):
    with mock.patch("knowledge.views.AskThrottle.THROTTLE_RATES", {"ask": "2/minute"}):
        codes = [ask(api, collection, "leave?").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_offline_answer_quotes_the_best_matching_sentence_verbatim():
    from knowledge.services.answering import best_sentence

    text = "# Leave policy\n\nIntro text.\n\n## Carrying leave over\n\nUp to five unused days may be carried over. Later days are forfeited."
    assert best_sentence("Can unused days be carried over?", text) == "Up to five unused days may be carried over."
    assert best_sentence("zebra", "# Only a heading") == "# Only a heading"
