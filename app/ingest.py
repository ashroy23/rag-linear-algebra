import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

MIN_PAGE_CHARS = 100
SPLITTER_SEPARATORS = ["\n\n", "\n", ". ", ", ", ""]


def load_and_chunk(
    pdf_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    logger.info("loaded %d pages from %s", len(pages), pdf_path)

    filtered = [p for p in pages if len(p.page_content.strip()) > MIN_PAGE_CHARS]
    logger.info("kept %d pages after filtering empty/near-empty", len(filtered))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=SPLITTER_SEPARATORS,
    )
    chunks = splitter.split_documents(filtered)
    logger.info("produced %d chunks", len(chunks))
    return chunks


def build_or_load_index(
    pdf_path: Path,
    index_dir: Path,
    embeddings: Embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> FAISS:
    index_file = index_dir / "index.faiss"
    if index_file.exists():
        logger.info("loading existing FAISS index from %s", index_dir)
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("no cached index — building from %s", pdf_path)
    chunks = load_and_chunk(pdf_path, chunk_size, chunk_overlap)
    store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    logger.info("saved index with %d vectors to %s", store.index.ntotal, index_dir)
    return store
