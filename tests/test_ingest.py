import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.ingest import MANIFEST_FILE, build_or_load_index, find_pdfs, load_and_chunk


def test_find_pdfs_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_pdfs(tmp_path / "does-not-exist")


def test_find_pdfs_empty_dir(tmp_path):
    (tmp_path / "pdfs").mkdir()
    with pytest.raises(FileNotFoundError):
        find_pdfs(tmp_path / "pdfs")


def test_find_pdfs_returns_sorted_pdfs(tmp_path):
    d = tmp_path / "pdfs"
    d.mkdir()
    (d / "b.pdf").write_bytes(b"dummy")
    (d / "a.pdf").write_bytes(b"dummy")
    (d / "notes.txt").write_text("not a pdf")
    sub = d / "sub"
    sub.mkdir()
    (sub / "c.pdf").write_bytes(b"dummy")

    pdfs = find_pdfs(d)
    assert [p.name for p in pdfs] == ["a.pdf", "b.pdf", "c.pdf"]


def test_load_and_chunk_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_chunk(tmp_path / "nope")


def test_build_or_load_index_roundtrip(tmp_path, fake_embeddings, monkeypatch):
    """build_or_load_index should build, save, then on a second call load from cache."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "book.pdf").write_bytes(b"dummy")
    index_dir = tmp_path / "index"

    sample_docs = [
        Document(
            page_content="Linear algebra studies vector spaces and linear maps between them.",
            metadata={"source": "book.pdf", "page": 0},
        ),
        Document(
            page_content="An eigenvalue is a scalar lambda satisfying Ax = lambda x for a nonzero x.",
            metadata={"source": "book.pdf", "page": 1},
        ),
    ]

    calls = {"n": 0}

    def fake_load_and_chunk(pdf_dir_arg, chunk_size=1000, chunk_overlap=200):
        calls["n"] += 1
        return sample_docs

    monkeypatch.setattr("app.ingest.load_and_chunk", fake_load_and_chunk)

    store_a = build_or_load_index(pdf_dir=pdf_dir, index_dir=index_dir, embeddings=fake_embeddings)
    assert store_a.index.ntotal == 2
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / MANIFEST_FILE).exists()
    assert calls["n"] == 1

    store_b = build_or_load_index(pdf_dir=pdf_dir, index_dir=index_dir, embeddings=fake_embeddings)
    assert store_b.index.ntotal == 2
    assert calls["n"] == 1  # cache hit — no rebuild


def test_build_or_load_index_rebuilds_when_pdf_added(tmp_path, fake_embeddings, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "a.pdf").write_bytes(b"dummy")
    index_dir = tmp_path / "index"

    sample_docs = [
        Document(page_content="content of book A", metadata={"source": "a.pdf", "page": 0}),
    ]

    calls = {"n": 0}

    def fake_load_and_chunk(pdf_dir_arg, chunk_size=1000, chunk_overlap=200):
        calls["n"] += 1
        return sample_docs

    monkeypatch.setattr("app.ingest.load_and_chunk", fake_load_and_chunk)

    build_or_load_index(pdf_dir=pdf_dir, index_dir=index_dir, embeddings=fake_embeddings)
    assert calls["n"] == 1

    (pdf_dir / "b.pdf").write_bytes(b"another dummy")

    build_or_load_index(pdf_dir=pdf_dir, index_dir=index_dir, embeddings=fake_embeddings)
    assert calls["n"] == 2  # manifest changed — rebuilt

    manifest = json.loads((index_dir / MANIFEST_FILE).read_text())
    assert {f["name"] for f in manifest["files"]} == {"a.pdf", "b.pdf"}
