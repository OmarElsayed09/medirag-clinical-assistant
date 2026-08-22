import os
import io
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
from src.evaluator import RAGEvaluator
from src.clinical_tools import (
    classify_blood_pressure,
    calculate_renal_function,
    stratify_cvd_risk,
)
from src.clinical_expert import (
    analyze_patient_case,
    check_drug_interactions,
    generate_patient_education_guide,
    get_guidelines_comparison,
)
from src.pdf_generator import (
    generate_clinical_soap_pdf,
    generate_patient_leaflet_pdf,
)
from src.security import (
    security_rate_limiter,
    validate_pdf_upload,
    detect_prompt_injection,
    detect_clinical_emergency,
    sanitize_pii,
)


# Initialize FastAPI Application
app = FastAPI(
    title="MediRAG Clinical Assistant API",
    description=(
        "Production-grade Clinical Decision Support and Guidelines RAG REST API "
        "with hybrid retrieval (ChromaDB + BM25), Groq LLMs, precise page-level citations, "
        "and clinical expert modules."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files Mount for Web Application
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- PYDANTIC SCHEMAS ---

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class QueryRequest(BaseModel):
    question: str = Field(..., description="Clinical question to ask.", json_schema_extra={"example": "What are the recommended first-line pharmacological treatment classes for hypertension?"})
    provider: Optional[str] = Field(default=None, description="LLM Provider: 'groq', 'ollama', 'gemini', or 'huggingface'")
    model_name: Optional[str] = Field(default=None, description="Specific model name (e.g. 'llama-3.3-70b-versatile')")
    api_key: Optional[str] = Field(default=None, description="Optional API key (Groq / Gemini) if not set in server env")
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature (0.0 for deterministic zero-hallucination)")
    top_k: Optional[int] = Field(default=6, ge=1, le=15, description="Number of top medical chunks to retrieve")
    use_reranker: Optional[bool] = Field(default=True, description="Whether to apply Stage-2 Cross-Encoder reranking")
    chat_history: Optional[List[ChatMessage]] = Field(default=None, description="Optional prior conversation turns for multi-turn dialogue")


class CitationItem(BaseModel):
    source: str
    page: Any
    chunk_id: str
    snippet: str
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    search_query: str
    answer: str
    citations: List[CitationItem]
    safety_audit: Optional[Dict[str, Any]] = None
    grounding_status: Optional[str] = None
    disclaimer: str
    latency_seconds: float
    total_retrieved_chunks: int


class BPClassificationRequest(BaseModel):
    systolic: int = Field(..., ge=70, le=300, json_schema_extra={"example": 155})
    diastolic: int = Field(..., ge=40, le=200, json_schema_extra={"example": 95})
    has_diabetes: Optional[bool] = Field(default=False)
    has_cvd: Optional[bool] = Field(default=False)
    has_ckd: Optional[bool] = Field(default=False)


class RenalCalculationRequest(BaseModel):
    serum_creatinine_mg_dl: float = Field(..., gt=0.1, le=25.0, json_schema_extra={"example": 1.4})
    age_years: int = Field(..., ge=18, le=120, json_schema_extra={"example": 62})
    is_female: bool = Field(..., json_schema_extra={"example": False})
    weight_kg: Optional[float] = Field(default=None, ge=20.0, le=300.0, json_schema_extra={"example": 78.0})


class CVDRiskRequest(BaseModel):
    age: int = Field(..., ge=20, le=100, json_schema_extra={"example": 55})
    systolic_bp: int = Field(..., ge=80, le=260, json_schema_extra={"example": 150})
    is_smoker: Optional[bool] = Field(default=False)
    has_diabetes: Optional[bool] = Field(default=False)
    total_cholesterol_mg_dl: Optional[float] = Field(default=None, ge=100, le=500)


class CaseAnalysisRequest(BaseModel):
    age: int = Field(..., ge=18, le=110, json_schema_extra={"example": 62})
    is_female: bool = Field(default=False, json_schema_extra={"example": False})
    systolic_bp: int = Field(..., ge=70, le=260, json_schema_extra={"example": 158})
    diastolic_bp: int = Field(..., ge=40, le=160, json_schema_extra={"example": 96})
    has_diabetes: Optional[bool] = Field(default=False, json_schema_extra={"example": True})
    has_cvd: Optional[bool] = Field(default=False, json_schema_extra={"example": False})
    has_ckd: Optional[bool] = Field(default=False, json_schema_extra={"example": False})
    is_smoker: Optional[bool] = Field(default=False, json_schema_extra={"example": False})
    serum_creatinine_mg_dl: Optional[float] = Field(default=None, json_schema_extra={"example": 1.3})
    weight_kg: Optional[float] = Field(default=None, json_schema_extra={"example": 78.0})
    current_medications: Optional[List[str]] = Field(default=None, json_schema_extra={"example": ["ace_inhibitor", "spironolactone"]})
    clinician_name: Optional[str] = Field(default="Clinical Staff", json_schema_extra={"example": "Dr. Smith, MD"})
    institution_name: Optional[str] = Field(default="MediRAG Clinical Decision Support", json_schema_extra={"example": "Cardiology Clinical Unit"})


class DDIRequest(BaseModel):
    medications: List[str] = Field(..., json_schema_extra={"example": ["ace_inhibitor", "spironolactone", "nsaid"]})


class PatientGuideRequest(BaseModel):
    condition_name: str = Field(default="Stage 2 Hypertension", json_schema_extra={"example": "Stage 2 Hypertension"})
    target_bp: str = Field(default="< 130/80 mmHg", json_schema_extra={"example": "< 130/80 mmHg"})
    language: Optional[str] = Field(default="Arabic", json_schema_extra={"example": "Arabic"})
    medications: Optional[List[str]] = Field(default=None)


class IndexResponse(BaseModel):
    status: str
    message: str
    total_files: int
    total_chunks: int
    indexed_documents: List[dict]


# --- REST API ENDPOINTS ---

@app.get("/", include_in_schema=False)
def serve_spa():
    """Serves the standalone modern single-page web interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "MediRAG API is online. Visit /docs for OpenAPI documentation."}


@app.get("/api/status", tags=["Health & Status"])
def root_status():
    """Returns system health, total chunks, and indexed guideline summaries."""
    total_chunks = get_vectorstore_count()
    doc_summary = get_indexed_documents_summary(DATA_DIR)
    return {
        "service": "MediRAG Clinical Assistant API",
        "version": "2.0.0",
        "status": "online",
        "vector_store_chunks": total_chunks,
        "indexed_guidelines": len(doc_summary),
        "default_llm": GROQ_MODEL,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "disclaimer": MEDICAL_DISCLAIMER,
    }


@app.get("/health", tags=["Health & Status"])
def health_check():
    """Liveness health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/query", response_model=QueryResponse, tags=["Clinical RAG"])
def query_guidelines(req: QueryRequest, request: Request):
    """
    Executes an evidence-grounded clinical query against indexed medical guidelines.
    Enforces Rate Limiting, Prompt Injection Screening, Emergency Detection, and PII Sanitization.
    """
    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, _ = security_rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 40 clinical queries per minute per IP address."
        )

    try:
        retriever = create_hybrid_retriever(top_k=req.top_k, use_reranker=req.use_reranker)
        pipeline = MedicalRAGPipeline(
            retriever=retriever,
            provider=req.provider or DEFAULT_LLM_PROVIDER,
            api_key=req.api_key,
            model_name=req.model_name,
            temperature=req.temperature,
        )

        history_dicts = [m.model_dump() for m in req.chat_history] if req.chat_history else None
        res = pipeline.query(req.question, chat_history=history_dicts)

        return QueryResponse(
            question=res["question"],
            search_query=res.get("search_query", req.question),
            answer=res["answer"],
            citations=[
                CitationItem(
                    source=c["source"],
                    page=c["page"],
                    chunk_id=c.get("chunk_id", ""),
                    snippet=c["snippet"],
                    rerank_score=c.get("rerank_score"),
                )
                for c in res.get("citations", [])
            ],
            safety_audit=res.get("safety_audit"),
            grounding_status=res.get("grounding_status"),
            disclaimer=res.get("disclaimer", MEDICAL_DISCLAIMER),
            latency_seconds=res.get("latency_seconds", 0.0),
            total_retrieved_chunks=len(res.get("raw_documents", [])),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream", tags=["Clinical RAG"])
def stream_clinical_query(req: QueryRequest, request: Request):
    """
    Streams live answer tokens using Server-Sent Events (SSE) for ultra-low latency.
    """
    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, _ = security_rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    try:
        retriever = create_hybrid_retriever(top_k=req.top_k, use_reranker=req.use_reranker)
        pipeline = MedicalRAGPipeline(
            retriever=retriever,
            provider=req.provider or DEFAULT_LLM_PROVIDER,
            api_key=req.api_key,
            model_name=req.model_name,
            temperature=req.temperature,
        )

        history_dicts = [m.model_dump() for m in req.chat_history] if req.chat_history else None

        def token_stream():
            for chunk in pipeline.stream_query(req.question, chat_history=history_dicts):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- CLINICAL DECISION TOOLS ENDPOINTS ---

@app.post("/tools/bp-stage", tags=["Clinical Tools"])
def api_classify_bp(req: BPClassificationRequest):
    """Evaluates blood pressure measurements against WHO and ACC/AHA guidelines."""
    result = classify_blood_pressure(
        systolic=req.systolic,
        diastolic=req.diastolic,
        has_diabetes=req.has_diabetes or False,
        has_cvd=req.has_cvd or False,
        has_ckd=req.has_ckd or False,
    )
    if not result.get("valid", True):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@app.post("/tools/renal-calc", tags=["Clinical Tools"])
def api_calculate_renal(req: RenalCalculationRequest):
    """Calculates eGFR (CKD-EPI 2021) and CrCl (Cockcroft-Gault) with dosage adjustments."""
    result = calculate_renal_function(
        serum_creatinine_mg_dl=req.serum_creatinine_mg_dl,
        age_years=req.age_years,
        is_female=req.is_female,
        weight_kg=req.weight_kg,
    )
    if not result.get("valid", True):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@app.post("/tools/cvd-risk", tags=["Clinical Tools"])
def api_stratify_cvd(req: CVDRiskRequest):
    """Estimates 10-year Cardiovascular Disease (CVD) risk tier."""
    return stratify_cvd_risk(
        age=req.age,
        systolic_bp=req.systolic_bp,
        is_smoker=req.is_smoker or False,
        has_diabetes=req.has_diabetes or False,
        total_cholesterol_mg_dl=req.total_cholesterol_mg_dl,
    )


# --- ADVANCED CLINICAL EXPERT ENDPOINTS ---

@app.post("/expert/case-analysis", tags=["Clinical Expert Suite"])
def api_analyze_patient_case(req: CaseAnalysisRequest):
    """Synthesizes a complete clinical SOAP note and customized treatment pathway."""
    return analyze_patient_case(
        age=req.age,
        is_female=req.is_female,
        systolic_bp=req.systolic_bp,
        diastolic_bp=req.diastolic_bp,
        has_diabetes=req.has_diabetes or False,
        has_cvd=req.has_cvd or False,
        has_ckd=req.has_ckd or False,
        is_smoker=req.is_smoker or False,
        serum_creatinine_mg_dl=req.serum_creatinine_mg_dl,
        weight_kg=req.weight_kg,
        current_medications=req.current_medications,
    )


@app.post("/expert/drug-interactions", tags=["Clinical Expert Suite"])
def api_check_ddi(req: DDIRequest):
    """Evaluates medication regimens against clinical contraindication and DDI matrix."""
    return check_drug_interactions(req.medications)


@app.post("/expert/patient-guide", tags=["Clinical Expert Suite"])
def api_generate_patient_guide(req: PatientGuideRequest):
    """Generates plain-language patient health education and medication guides."""
    return generate_patient_education_guide(
        condition_name=req.condition_name,
        target_bp=req.target_bp,
        language=req.language or "Arabic",
        medications=req.medications,
    )


@app.get("/expert/guidelines-comparison", tags=["Clinical Expert Suite"])
def api_get_guidelines_comparison():
    """Returns multi-guideline consensus and divergence comparison table."""
    return get_guidelines_comparison()


@app.post("/expert/export-soap-pdf", tags=["Clinical Expert Suite"])
def api_export_soap_pdf(req: CaseAnalysisRequest):
    """
    Synthesizes a patient case and exports an official, evidence-based Clinical SOAP Note in PDF format.
    """
    case_result = analyze_patient_case(
        age=req.age,
        is_female=req.is_female,
        systolic_bp=req.systolic_bp,
        diastolic_bp=req.diastolic_bp,
        has_diabetes=req.has_diabetes or False,
        has_cvd=req.has_cvd or False,
        has_ckd=req.has_ckd or False,
        is_smoker=req.is_smoker or False,
        serum_creatinine_mg_dl=req.serum_creatinine_mg_dl,
        weight_kg=req.weight_kg,
        current_medications=req.current_medications,
    )
    pdf_bytes = generate_clinical_soap_pdf(
        case_result=case_result,
        clinician_name=req.clinician_name or "Clinical Staff",
        institution_name=req.institution_name or "MediRAG Clinical Decision Support",
    )
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=Clinical_SOAP_Report.pdf"},
    )


@app.post("/expert/export-patient-guide-pdf", tags=["Clinical Expert Suite"])
def api_export_patient_guide_pdf(req: PatientGuideRequest):
    """
    Generates and exports a formatted patient discharge education guide in PDF format.
    """
    guide_data = generate_patient_education_guide(
        condition_name=req.condition_name,
        target_bp=req.target_bp,
        language=req.language or "Arabic",
        medications=req.medications,
    )
    pdf_bytes = generate_patient_leaflet_pdf(guide_data=guide_data)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=Patient_Discharge_Guide.pdf"},
    )


# --- DOCUMENT REPOSITORY & BENCHMARK ENDPOINTS ---

@app.get("/documents", tags=["Document Repository"])
def list_documents():
    """Lists all active guideline PDF documents and chunk counts."""
    summary = get_indexed_documents_summary(DATA_DIR)
    total_chunks = get_vectorstore_count()
    return {
        "total_files": len(summary),
        "total_chunks": total_chunks,
        "documents": summary,
    }


@app.post("/reindex", response_model=IndexResponse, tags=["Document Repository"])
def reindex_guidelines():
    """Rebuilds the ChromaDB vector and BM25 indexes from data directory PDFs."""
    try:
        invalidate_bm25_cache()
        raw_docs = load_all_documents(DATA_DIR)
        if not raw_docs:
            raise HTTPException(status_code=404, detail="No PDF files found in data folder.")

        chunks = split_documents(raw_docs)
        index_chunks_into_vectorstore(chunks, clear_existing=True)
        summary = get_indexed_documents_summary(DATA_DIR)

        return IndexResponse(
            status="success",
            message=f"Successfully indexed {len(chunks)} chunks into ChromaDB and BM25.",
            total_files=len(summary),
            total_chunks=len(chunks),
            indexed_documents=summary,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", tags=["Document Repository"])
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a new guideline PDF, verifies magic bytes, and indexes it into the vector store."""
    try:
        contents = await file.read()
        is_valid, err_msg = validate_pdf_upload(contents, file.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=err_msg)

        save_path = DATA_DIR / Path(file.filename).name
        with open(save_path, "wb") as f:
            f.write(contents)

        invalidate_bm25_cache()
        new_docs = load_single_pdf(save_path)
        if new_docs:
            new_chunks = split_documents(new_docs)
            index_chunks_into_vectorstore(new_chunks, clear_existing=False)

        return {
            "status": "success",
            "filename": file.filename,
            "pages_loaded": len(new_docs),
            "chunks_added": len(new_chunks) if new_docs else 0,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmark", tags=["Evaluation & Benchmarks"])
def run_benchmark():
    """Executes automated RAG Triad benchmark on clinical test set."""
    try:
        evaluator = RAGEvaluator()
        results = evaluator.run_benchmark()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
