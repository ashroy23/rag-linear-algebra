# RAG System — Linear Algebra Q&A

A Retrieval Augmented Generation (RAG) system that answers 
questions from linear algebra textbooks using semantic search 
and an LLM.

## What it does
Upload linear algebra PDFs, ask any question, get answers 
grounded in the actual textbook content — not LLM hallucination.

## Tech stack
- LangChain — pipeline orchestration
- HuggingFace all-MiniLM-L6-v2 — text embeddings
- FAISS — vector similarity search  
- Google Gemini — answer generation
- PyPDF — document loading

## Pipeline
1. Load PDFs with PyPDFLoader
2. Filter empty pages
3. Chunk with RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
4. Embed chunks into 384-dimensional vectors
5. Store in FAISS index (1152 vectors)
6. Retrieve top-k chunks via cosine similarity
7. Generate grounded answer with Gemini

## Evaluation
Tested on 10 linear algebra questions. 8/10 answered correctly.

Failure modes identified:
- Vocabulary mismatch: query uses different terminology than source document
- Retrieval depth: some concepts spread across multiple pages need higher k

## Next steps — V2 CRAG
- Add relevance grader to check if retrieved chunks answer the question
- Implement query rewriting when chunks are not relevant
- Add web search fallback for out-of-scope questions
