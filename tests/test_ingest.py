from pathlib import Path

from langchain_core.documents import Document

from app.ingest import build_or_load_index, load_and_chunk


def test_load_and_chunk_missing_file(tmp_path):
    missing = tmp_path / "nope.pdf"
    try:
        load_and_chunk(missing)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_build_or_load_index_roundtrip(tmp_path, fake_embeddings, monkeypatch):
    """build_or_load_index should build, save, then on a second call load from disk."""
    index_dir = tmp_path / "index"

    sample_docs = [
        Document(
            page_content="Linear algebra studies vector spaces and linear maps between them.",
            metadata={"source": "fixture.pdf", "page": 0},
        ),
        Document(
            page_content="An eigenvalue is a scalar lambda satisfying Ax = lambda x for a nonzero x.",
            metadata={"source": "fixture.pdf", "page": 1},
        ),
    ]

    def fake_load_and_chunk(pdf_path, chunk_size=1000, chunk_overlap=200):
        return sample_docs

    monkeypatch.setattr("app.ingest.load_and_chunk", fake_load_and_chunk)

    store_a = build_or_load_index(
        pdf_path=Path("unused.pdf"),
        index_dir=index_dir,
        embeddings=fake_embeddings,
    )
    assert store_a.index.ntotal == 2
    assert (index_dir / "index.faiss").exists()

    store_b = build_or_load_index(
        pdf_path=Path("unused.pdf"),
        index_dir=index_dir,
        embeddings=fake_embeddings,
    )
    assert store_b.index.ntotal == 2

    results = store_b.similarity_search("eigenvalue", k=1)
    assert len(results) == 1
