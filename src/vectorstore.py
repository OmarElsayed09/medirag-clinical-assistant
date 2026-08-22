from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_DIR
from src.embeddings import get_embedding_function

COLLECTION_NAME = "medirag_medical_docs"


def get_or_create_vectorstore(persist_directory: Path = CHROMA_DIR) -> Chroma:
    """Loads an existing Chroma vector store or initializes an empty one."""
    embeddings = get_embedding_function()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    return vectorstore


def index_chunks_into_vectorstore(
    chunks: List[Document],
    persist_directory: Path = CHROMA_DIR,
    clear_existing: bool = False,
) -> Chroma:
    """Indexes chunks into ChromaDB."""
    embeddings = get_embedding_function()

    if clear_existing:
        print(f"[INFO] Clearing existing ChromaDB collection '{COLLECTION_NAME}'...")
        try:
            temp_vs = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(persist_directory),
            )
            temp_vs.delete_collection()
        except Exception as e:
            print(f"[WARNING] Could not delete existing collection: {e}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )
    print(f"[INFO] Successfully indexed {len(chunks)} chunks into ChromaDB.")
    return vectorstore


def get_vectorstore_count(persist_directory: Path = CHROMA_DIR) -> int:
    """Returns the total number of documents/chunks in the vector store."""
    try:
        vs = get_or_create_vectorstore(persist_directory)
        return vs._collection.count()
    except Exception:
        return 0
