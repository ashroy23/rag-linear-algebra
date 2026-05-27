from dataclasses import dataclass
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about linear algebra. "
    "Answer ONLY using the context provided. If the answer is not in the "
    "context say I cannot find this in the provided material."
)


@dataclass
class Source:
    source: str
    page: int | None
    snippet: str


class RagService:
    def __init__(self, vector_store: FAISS, llm: Any):
        self.vector_store = vector_store
        self.llm = llm

    def answer(self, question: str, k: int = 3) -> dict:
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n".join(d.page_content for d in docs)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer clearly based only on the context above."
                )
            ),
        ]
        response = self.llm.invoke(messages)
        answer_text = getattr(response, "content", str(response))

        sources = [
            Source(
                source=str(d.metadata.get("source", "")),
                page=d.metadata.get("page"),
                snippet=d.page_content[:240],
            )
            for d in docs
        ]
        return {"answer": answer_text, "sources": sources}
