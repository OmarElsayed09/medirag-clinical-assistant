import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_DIR
from src.document_loader import load_all_documents, get_indexed_documents_summary
from src.text_splitter import split_documents
from src.vectorstore import index_chunks_into_vectorstore, get_vectorstore_count
from src.retriever import create_hybrid_retriever


def run_verification():
    print("=== STEP 1: Verifying Document Loading ===")
    summary = get_indexed_documents_summary(DATA_DIR)
    print(f"Found {len(summary)} documents in data folder:")
    for s in summary:
        print(f" - {s['filename']}: {s['total_pages']} pages ({s['size_mb']} MB)")

    raw_docs = load_all_documents(DATA_DIR)
    assert len(raw_docs) > 0, "No pages loaded!"
    print(f"Total pages extracted: {len(raw_docs)}")

    print("\n=== STEP 2: Verifying Chunking & Metadata Enrichment ===")
    chunks = split_documents(raw_docs)
    assert len(chunks) > 0, "No chunks generated!"
    print(f"Generated {len(chunks)} chunks.")
    print("Sample chunk metadata:", chunks[0].metadata)
    print("Sample chunk preview:", chunks[0].page_content[:150], "...\n")

    print("=== STEP 3: Indexing Chunks into ChromaDB ===")
    vectorstore = index_chunks_into_vectorstore(chunks, clear_existing=True)
    count = get_vectorstore_count()
    print(f"ChromaDB total indexed vectors: {count}")
    assert count == len(chunks), f"Expected {len(chunks)} vectors, got {count}"

    print("\n=== STEP 4: Testing Hybrid Retrieval (Vector + BM25) ===")
    retriever = create_hybrid_retriever(chunks=chunks, top_k=3)
    query = "What is the threshold for diagnosing hypertension in adults?"
    print(f"Testing Query: '{query}'")
    results = retriever.invoke(query)
    print(f"Retrieved {len(results)} relevant documents:")
    for i, doc in enumerate(results, 1):
        print(f" [{i}] Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
        print(f"     Preview: {doc.page_content[:120]}...\n")

    print("=== ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    run_verification()
