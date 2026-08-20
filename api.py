import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    DATA_DIR,
    CHROMA_DIR,
    DEFAULT_LLM_PROVIDER,
    GROQ_MODEL,
    EMBEDDING_MODEL_NAME,
    MEDICAL_DISCLAIMER,
)
from src.document_loader import (
    load_all_documents,
    load_single_pdf,
    get_indexed_documents_summary,
)
from src.text_splitter import split_documents
from src.vectorstore import (
    index_chunks_into_vectorstore,
    get_vectorstore_count,
    get_or_create_vectorstore,
)
from src.retriever import create_hybrid_retriever, invalidate_bm25_cache
from src.rag_chain import MedicalRAGPipeline


# Initialize FastAPI App
app = FastAPI(
    title="MediRAG Medical Guidelines API",
    description=(
        "Production-grade Clinical RAG REST API for querying medical guidelines (WHO, NICE, etc.) "
        "with hybrid retrieval (Chroma + BM25), Groq/Local LLMs, and precise page-level citations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class QueryRequest(BaseModel):
    question: str = Field(..., description="The clinical or medical question to ask.", example="What are the diagnostic blood pressure thresholds for hypertension?")
    provider: Optional[str] = Field(default=None, description="LLM Provider: 'groq', 'ollama', 'gemini', or 'huggingface'")
    model_name: Optional[str] = Field(default=None, description="Specific model name (e.g. 'qwen/qwen3.6-27b')")
    api_key: Optional[str] = Field(default=None, description="Optional API key (Groq / Gemini) if not set in server env")
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature (0.0 for strict deterministic)")
    top_k: Optional[int] = Field(default=4, ge=1, le=15, description="Number of top medical chunks to retrieve")
    use_reranker: Optional[bool] = Field(default=True, description="Whether to apply Stage-2 Cross-Encoder reranking")
    chat_history: Optional[List[ChatMessage]] = Field(default=None, description="Optional prior conversation turns for multi-turn chatbot dialogue")


class CitationItem(BaseModel):
    source: str
    page: Any
    chunk_id: str
    snippet: str


class DocumentMetadata(BaseModel):
    content: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationItem]
    safety_audit: Optional[Dict[str, Any]] = None
    grounding_status: Optional[str] = None
    disclaimer: str
    execution_time_seconds: float
    total_retrieved_chunks: int


class IndexResponse(BaseModel):
    status: str
    message: str
    total_files: int
    total_chunks: int
    indexed_documents: List[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_model: str
    vector_store_documents: int
    default_llm_provider: str
    default_groq_model: str


@app.get("/", tags=["Root"])
def root_endpoint():
    return {
        "message": "Welcome to MediRAG Medical Guidelines API.",
        "documentation": "/docs",
        "health_check": "/api/health",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns system status, active embedding model, and vector DB count."""
    doc_count = get_vectorstore_count()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        embedding_model=EMBEDDING_MODEL_NAME,
        vector_store_documents=doc_count,
        default_llm_provider=DEFAULT_LLM_PROVIDER,
        default_groq_model=GROQ_MODEL,
    )


@app.get("/api/documents", tags=["Documents"])
def list_documents():
    """Lists all medical PDF guidelines present in the data repository."""
    summary = get_indexed_documents_summary(DATA_DIR)
    vector_count = get_vectorstore_count()
    return {
        "total_files": len(summary),
        "total_vector_chunks": vector_count,
        "files": summary,
    }


@app.post("/api/documents/index", response_model=IndexResponse, tags=["Documents"])
def reindex_documents(clear_existing: bool = True):
    """
    Parses all PDFs in the data folder, splits them into semantic chunks,
    and indexes them into ChromaDB and cached BM25.
    """
    try:
        invalidate_bm25_cache()
        raw_docs = load_all_documents(DATA_DIR)
        if not raw_docs:
            raise HTTPException(
                status_code=400,
                detail=f"No PDF files found in data directory: {DATA_DIR}",
            )

        chunks = split_documents(raw_docs)
        index_chunks_into_vectorstore(chunks, clear_existing=clear_existing)
        summary = get_indexed_documents_summary(DATA_DIR)

        return IndexResponse(
            status="success",
            message=f"Successfully indexed {len(chunks)} chunks from {len(summary)} files.",
            total_files=len(summary),
            total_chunks=len(chunks),
            indexed_documents=summary,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/api/documents/upload", response_model=IndexResponse, tags=["Documents"])
async def upload_medical_pdf(file: UploadFile = File(...)):
    """Uploads a new medical PDF guideline, saves it to data/, and re-indexes the database."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = DATA_DIR / file.filename
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Re-index all documents
        invalidate_bm25_cache()
        raw_docs = load_all_documents(DATA_DIR)
        chunks = split_documents(raw_docs)
        index_chunks_into_vectorstore(chunks, clear_existing=True)
        summary = get_indexed_documents_summary(DATA_DIR)

        return IndexResponse(
            status="success",
            message=f"Uploaded and indexed '{file.filename}'. Total chunks: {len(chunks)}",
            total_files=len(summary),
            total_chunks=len(chunks),
            indexed_documents=summary,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload & Indexing failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse, tags=["RAG Query"])
def query_medical_rag(req: QueryRequest):
    """
    Asks a medical question against the indexed clinical guidelines.
    Returns the synthesized medical answer, exact document & page citations, safety audit, and disclaimer.
    """
    start_time = time.time()
    try:
        if get_vectorstore_count() == 0:
            raw_docs = load_all_documents(DATA_DIR)
            if raw_docs:
                chunks = split_documents(raw_docs)
                index_chunks_into_vectorstore(chunks)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No medical documents are indexed in the database. Please add PDFs to data/ and index.",
                )

        retriever = create_hybrid_retriever(top_k=req.top_k or 4, use_reranker=req.use_reranker if req.use_reranker is not None else True)
        pipeline = MedicalRAGPipeline(
            retriever=retriever,
            provider=req.provider or DEFAULT_LLM_PROVIDER,
            api_key=req.api_key,
            model_name=req.model_name,
            temperature=req.temperature or 0.0,
        )

        history_list = [{"role": m.role, "content": m.content} for m in req.chat_history] if req.chat_history else None
        result = pipeline.query(req.question, chat_history=history_list)
        duration = round(time.time() - start_time, 2)

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            citations=[
                CitationItem(
                    source=c["source"],
                    page=c["page"],
                    chunk_id=c.get("chunk_id", ""),
                    snippet=c["snippet"],
                )
                for c in result.get("citations", [])
            ],
            safety_audit=result.get("safety_audit"),
            grounding_status=result.get("grounding_status"),
            disclaimer=result.get("disclaimer", MEDICAL_DISCLAIMER),
            execution_time_seconds=duration,
            total_retrieved_chunks=len(result.get("raw_documents", [])),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/query/stream", tags=["RAG Query"])
def stream_medical_rag(req: QueryRequest):
    """
    Streams the medical answer token-by-token using Server-Sent Events (SSE).
    """
    try:
        retriever = create_hybrid_retriever(top_k=req.top_k or 4, use_reranker=req.use_reranker if req.use_reranker is not None else True)
        pipeline = MedicalRAGPipeline(
            retriever=retriever,
            provider=req.provider or DEFAULT_LLM_PROVIDER,
            api_key=req.api_key,
            model_name=req.model_name,
            temperature=req.temperature or 0.0,
        )

        history_list = [{"role": m.role, "content": m.content} for m in req.chat_history] if req.chat_history else None
        def event_generator():
            for chunk in pipeline.stream_query(req.question, chat_history=history_list):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming query failed: {str(e)}")
@app.post("/api/evaluate", tags=["Evaluation"])
def run_evaluation_benchmark():
    """
    Runs the automated RAG Triad & Clinical Safety benchmark suite,
    calculating Faithfulness %, Context Relevance %, Retrieval Hit Rate %, and Latency.
    """
    try:
        from src.evaluator import RAGEvaluator
        pipeline = MedicalRAGPipeline(temperature=0.0)
        evaluator = RAGEvaluator(pipeline=pipeline)
        summary = evaluator.run_benchmark()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
