from app.rag import RagService


def test_answer_returns_canned_response_and_sources(fake_vector_store, fake_llm):
    service = RagService(vector_store=fake_vector_store, llm=fake_llm)

    result = service.answer("What is an eigenvalue?", k=2)

    assert result["answer"] == fake_llm.canned_answer
    assert len(result["sources"]) == 2
    assert all(s.source == "fixture.pdf" for s in result["sources"])
    assert all(len(s.snippet) > 0 for s in result["sources"])

    assert len(fake_llm.calls) == 1
    sent_messages = fake_llm.calls[0]
    assert any("Context:" in m.content for m in sent_messages)
    assert any("What is an eigenvalue?" in m.content for m in sent_messages)
