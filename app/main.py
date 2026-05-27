import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.ingest import build_or_load_index
from app.rag import RagService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(default=None, ge=1, le=20)


class SourceResponse(BaseModel):
    source: str
    page: int | None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


def build_rag_service(settings: Settings) -> RagService:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings

    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required to start the service")

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vector_store = build_or_load_index(
        pdf_dir=settings.pdf_dir,
        index_dir=settings.index_dir,
        embeddings=embeddings,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0,
        google_api_key=settings.gemini_api_key,
    )
    return RagService(vector_store=vector_store, llm=llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("starting up: pdf_dir=%s index_dir=%s", settings.pdf_dir, settings.index_dir)
    app.state.settings = settings
    app.state.rag = build_rag_service(settings)
    logger.info("startup complete")
    yield


app = FastAPI(title="Linear Algebra RAG", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    rag: RagService = request.app.state.rag
    settings: Settings = request.app.state.settings
    k = req.k or settings.retrieval_k
    try:
        result = rag.answer(req.question, k=k)
    except Exception as exc:
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail=f"query failed: {exc}") from exc

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceResponse(**s.__dict__) for s in result["sources"]],
    )
