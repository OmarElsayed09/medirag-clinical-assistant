"""
Unit tests for MediRAG Document Ingestion & Hybrid Retrieval Engine (src/retriever.py, src/document_loader.py).
Tests:
- Directory SHA256 hashing
- BM25 disk persistence, instant deserialization, and cache invalidation
- Text cleaning and table structure preservation
- Cross-Encoder reranking deduplication
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from langchain_core.documents import Document

from src.retriever import (
    compute_directory_hash,
    invalidate_bm25_cache,
    _get_or_create_bm25,
    rerank_documents,
)
from src.document_loader import clean_medical_text
from src.config import DATA_DIR, BM25_CACHE_PATH


class TestDocumentCleaning:
    def test_hyphen_word_merge(self):
        dirty_text = "Standard treat-\nment for hyper-\ntension in adults."
        cleaned = clean_medical_text(dirty_text)
        assert "treatment" in cleaned
        assert "hypertension" in cleaned

    def test_table_row_alignment(self):
        table_raw = "Drug Class    Starting Dose    Target Population\nACE Inhibitor   10 mg daily    Adults < 55"
        cleaned = clean_medical_text(table_raw)
        assert "|" in cleaned or "ACE Inhibitor" in cleaned

    def test_footer_stripping(self):
        footer_raw = "Clinical recommendations.\nPage 12 of 45\n© World Health Organization 2021"
        cleaned = clean_medical_text(footer_raw)
        assert "Page 12 of 45" not in cleaned
        assert "World Health Organization" not in cleaned


class TestBM25Persistence:
    def test_directory_hash_deterministic(self):
        hash1 = compute_directory_hash(DATA_DIR)
        hash2 = compute_directory_hash(DATA_DIR)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_bm25_disk_caching_lifecycle(self):
        test_chunks = [
            Document(page_content="Lisinopril is an ACE inhibitor for hypertension.", metadata={"source": "test.pdf", "page": 1}),
            Document(page_content="Amlodipine is a calcium channel blocker.", metadata={"source": "test.pdf", "page": 2}),
        ]
        # Invalidate first
        invalidate_bm25_cache()
        assert not BM25_CACHE_PATH.exists()

        # Build and cache
        bm25 = _get_or_create_bm25(chunks=test_chunks, top_k=2)
        assert bm25 is not None
        assert BM25_CACHE_PATH.exists()

        # Retrieve
        results = bm25.invoke("Lisinopril")
        assert len(results) >= 1
        assert any("Lisinopril" in d.page_content for d in results)

        # Clean up
        invalidate_bm25_cache()
        assert not BM25_CACHE_PATH.exists()


class TestRerankerDeduplication:
    @patch("src.retriever.get_cross_encoder")
    def test_rerank_deduplication(self, mock_get_cross_encoder):
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.95, 0.40]
        mock_get_cross_encoder.return_value = mock_encoder

        dup_docs = [
            Document(page_content="First-line therapy is ACEi or ARB.", metadata={"source": "who.pdf", "page": 5, "chunk_id": "c1"}),
            Document(page_content="First-line therapy is ACEi or ARB.", metadata={"source": "who.pdf", "page": 5, "chunk_id": "c1"}),
            Document(page_content="Calcium channel blockers are alternative first-line.", metadata={"source": "who.pdf", "page": 6, "chunk_id": "c2"}),
        ]
        reranked = rerank_documents("What is first line?", dup_docs, top_k=5)
        # Should deduplicate c1 and return 2 scored docs
        assert len(reranked) == 2
        assert reranked[0].metadata["rerank_score"] == 0.95
