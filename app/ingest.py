import json
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
MANIFEST_FILE = "manifest.json"


def find_pdfs(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"PDF path is not a directory: {pdf_dir}")
    pdfs = sorted(p for p in pdf_dir.rglob("*.pdf") if p.is_file())
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Drop one or more *.pdf files into that directory."
        )
    return pdfs


def build_manifest(pdfs: list[Path]) -> dict:
    return {
        "files": [
            {"name": str(p.name), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)}
            for p in pdfs
        ]
    }


def load_and_chunk(
    pdf_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    pdfs = find_pdfs(pdf_dir)
    logger.info("found %d PDF(s) in %s", len(pdfs), pdf_dir)

    all_pages: list[Document] = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        logger.info("loaded %d pages from %s", len(pages), pdf.name)
        all_pages.extend(pages)

    filtered = [p for p in all_pages if len(p.page_content.strip()) > MIN_PAGE_CHARS]
    logger.info("kept %d / %d pages after filtering empty/near-empty", len(filtered), len(all_pages))

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
    pdf_dir: Path,
    index_dir: Path,
    embeddings: Embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> FAISS:
    pdfs = find_pdfs(pdf_dir)
    current_manifest = build_manifest(pdfs)

    index_file = index_dir / "index.faiss"
    manifest_file = index_dir / MANIFEST_FILE

    if index_file.exists() and manifest_file.exists():
        cached_manifest = json.loads(manifest_file.read_text())
        if cached_manifest == current_manifest:
            logger.info("loading cached FAISS index from %s (PDFs unchanged)", index_dir)
            return FAISS.load_local(
                str(index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        logger.info("PDFs changed since last build — rebuilding index")

    logger.info("building FAISS index from %d PDF(s) in %s", len(pdfs), pdf_dir)
    chunks = load_and_chunk(pdf_dir, chunk_size, chunk_overlap)
    store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    manifest_file.write_text(json.dumps(current_manifest, indent=2))
    logger.info("saved index with %d vectors to %s", store.index.ntotal, index_dir)
    return store
