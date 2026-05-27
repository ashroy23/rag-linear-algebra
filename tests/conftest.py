import hashlib
from dataclasses import dataclass

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic hash-based embeddings. Avoids any network/model download in CI."""

    DIM = 32

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b - 128) / 128.0 for b in digest[: self.DIM]]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class FakeLLMResponse:
    content: str


class FakeLLM:
    def __init__(self, canned_answer: str = "An eigenvalue is a scalar lambda."):
        self.canned_answer = canned_answer
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeLLMResponse(content=self.canned_answer)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="An eigenvalue is a scalar lambda such that Ax = lambda x for some nonzero x.",
            metadata={"source": "fixture.pdf", "page": 1},
        ),
        Document(
            page_content="Gaussian elimination reduces a matrix to upper triangular form by row operations.",
            metadata={"source": "fixture.pdf", "page": 2},
        ),
        Document(
            page_content="The determinant of a triangular matrix is the product of its diagonal entries.",
            metadata={"source": "fixture.pdf", "page": 3},
        ),
    ]


@pytest.fixture
def fake_vector_store(sample_documents, fake_embeddings) -> FAISS:
    return FAISS.from_documents(sample_documents, fake_embeddings)
