from typing import List, Optional, Any, Dict
from pydantic import Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

import src.config
RETRIEVER_TOP_K = getattr(src.config, "RETRIEVER_TOP_K", 6)
BM25_WEIGHT = getattr(src.config, "BM25_WEIGHT", 0.4)
VECTOR_WEIGHT = getattr(src.config, "VECTOR_WEIGHT", 0.6)
DATA_DIR = getattr(src.config, "DATA_DIR", None)
USE_RERANKER = getattr(src.config, "USE_RERANKER", True)
RERANKER_MODEL_NAME = getattr(src.config, "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_INITIAL_K = getattr(src.config, "RERANKER_INITIAL_K", 15)
from src.vectorstore import get_or_create_vectorstore

# Global in-memory cache for BM25 retriever to avoid re-parsing PDFs
_CACHED_BM25_RETRIEVER: Optional[BM25Retriever] = None
_CACHED_DOCS_HASH: Optional[int] = None

# Global in-memory cache for CrossEncoder model
_CACHED_CROSS_ENCODER: Optional[Any] = None


def invalidate_bm25_cache():
    """Clears the cached BM25 retriever instance."""
    global _CACHED_BM25_RETRIEVER, _CACHED_DOCS_HASH
    _CACHED_BM25_RETRIEVER = None
    _CACHED_DOCS_HASH = None


def _get_or_create_bm25(chunks: Optional[List[Document]] = None, top_k: int = RETRIEVER_TOP_K) -> Optional[BM25Retriever]:
    """Retrieves or creates a cached BM25 index from documents."""
    global _CACHED_BM25_RETRIEVER, _CACHED_DOCS_HASH

    if chunks and len(chunks) > 0:
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = top_k
        _CACHED_BM25_RETRIEVER = bm25
        return bm25

    if _CACHED_BM25_RETRIEVER is not None:
        _CACHED_BM25_RETRIEVER.k = top_k
        return _CACHED_BM25_RETRIEVER

    # Lazy-load documents from DATA_DIR if no chunks are explicitly passed
    try:
        from src.document_loader import load_all_documents
        from src.text_splitter import split_documents
        raw_docs = load_all_documents(DATA_DIR)
        if raw_docs:
            all_chunks = split_documents(raw_docs)
            if all_chunks:
                bm25 = BM25Retriever.from_documents(all_chunks)
                bm25.k = top_k
                _CACHED_BM25_RETRIEVER = bm25
                return bm25
    except Exception as e:
        print(f"[WARNING] Could not initialize BM25 sparse retriever: {e}")

    return None


def get_cross_encoder():
    """Loads and caches the Cross-Encoder model in memory."""
    global _CACHED_CROSS_ENCODER
    if _CACHED_CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CACHED_CROSS_ENCODER = CrossEncoder(RERANKER_MODEL_NAME)
        except Exception as e:
            print(f"[WARNING] Could not load CrossEncoder '{RERANKER_MODEL_NAME}': {e}")
            _CACHED_CROSS_ENCODER = None
    return _CACHED_CROSS_ENCODER


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = RETRIEVER_TOP_K,
) -> List[Document]:
    """
    Reranks candidate documents using Cross-Encoder semantic score.
    Returns top_k most clinically relevant documents with rerank scores attached.
    """
    if not documents:
        return []

    # Deduplicate candidate documents by (source, page, chunk_id)
    unique_docs: List[Document] = []
    seen = set()
    for doc in documents:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", 0),
            doc.metadata.get("chunk_id", doc.page_content[:40]),
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    reranker = get_cross_encoder()
    if reranker is None or len(unique_docs) <= 1:
        return unique_docs[:top_k]

    try:
        pairs = [[query, doc.page_content] for doc in unique_docs]
        scores = reranker.predict(pairs)

        # Attach rerank score into metadata
        scored_docs = []
        for doc, score in zip(unique_docs, scores):
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(float(score), 4)},
            )
            scored_docs.append((doc_copy, float(score)))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    except Exception as e:
        print(f"[WARNING] Reranking failed, falling back to raw candidates: {e}")
        return unique_docs[:top_k]


class TwoStageClinicalRetriever(BaseRetriever):
    """
    2-Stage Retrieval Architecture:
    Stage 1: Broad Hybrid Retrieval (Dense Vector + Sparse BM25) with high candidate recall.
    Stage 2: Cross-Encoder Precision Reranking filtering to top_k most direct clinical evidence.
    """
    base_retriever: BaseRetriever = Field(description="Underlying Stage-1 Hybrid Ensemble Retriever")
    top_k: int = Field(default=RETRIEVER_TOP_K, description="Final number of top passages after reranking")
    use_reranker: bool = Field(default=True, description="Whether to apply Stage-2 Cross-Encoder reranking")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        # Stage 1: Candidate retrieval
        candidates = self.base_retriever.invoke(query)

        # Stage 2: Cross-Encoder Reranking
        if self.use_reranker:
            return rerank_documents(query, candidates, top_k=self.top_k)
        return candidates[:self.top_k]


def create_hybrid_retriever(
    chunks: Optional[List[Document]] = None,
    top_k: int = RETRIEVER_TOP_K,
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
    use_reranker: bool = USE_RERANKER,
) -> BaseRetriever:
    """
    Creates a 2-Stage Hybrid Clinical Retriever:
    1. Ensemble of ChromaDB Dense Vector Search + BM25 Sparse Search (initial candidates).
    2. Cross-Encoder Reranker selecting top_k highest precision chunks.
    """
    initial_k = max(RERANKER_INITIAL_K, top_k * 2) if use_reranker else top_k

    vectorstore = get_or_create_vectorstore()
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": initial_k},
    )

    # Acquire BM25 retriever
    bm25_retriever = _get_or_create_bm25(chunks=chunks, top_k=initial_k)

    if bm25_retriever is not None:
        total = bm25_weight + vector_weight
        w_bm25 = bm25_weight / total
        w_vec = vector_weight / total

        base_ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[w_bm25, w_vec],
        )
    else:
        base_ensemble = vector_retriever

    if use_reranker:
        return TwoStageClinicalRetriever(
            base_retriever=base_ensemble,
            top_k=top_k,
            use_reranker=True,
        )

    return base_ensemble
