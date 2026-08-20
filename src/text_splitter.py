from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def get_medical_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Returns a RecursiveCharacterTextSplitter tailored for clinical guidelines and medical documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",  # Section breaks
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks / table rows
            ". ",      # Sentence boundaries
            "; ",      # Clinical listing
            ", ",
            " ",
            "",
        ],
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Splits a list of documents into chunks and tags each with a unique chunk identifier."""
    splitter = get_medical_text_splitter(chunk_size, chunk_overlap)
    raw_chunks = splitter.split_documents(documents)

    enriched_chunks: List[Document] = []
    chunk_counters: dict = {}

    for chunk in raw_chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 1)

        key = f"{source}_p{page}"
        chunk_counters[key] = chunk_counters.get(key, 0) + 1
        chunk_idx = chunk_counters[key]

        # Add unique chunk id
        chunk.metadata["chunk_id"] = f"{key}_c{chunk_idx}"
        enriched_chunks.append(chunk)

    print(f"[INFO] Generated {len(enriched_chunks)} chunks from {len(documents)} document pages.")
    return enriched_chunks
