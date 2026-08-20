import re
from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document

from src.config import DATA_DIR


def clean_medical_text(text: str) -> str:
    """Cleans raw PDF text by normalizing spaces and fixing hyphenated line breaks."""
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    # Merge hyphenated words across line breaks (e.g. hyper-\ntension -> hypertension)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Replace multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_single_pdf(file_path: Path) -> List[Document]:
    """Loads a single PDF file and extracts text page by page with rich metadata."""
    documents: List[Document] = []
    try:
        reader = PdfReader(str(file_path))
        filename = file_path.name

        for page_idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned_text = clean_medical_text(raw_text)

            if cleaned_text:
                doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": filename,
                        "file_path": str(file_path),
                        "page": page_idx + 1,  # 1-indexed for citations
                        "total_pages": len(reader.pages),
                    },
                )
                documents.append(doc)
    except Exception as e:
        print(f"[ERROR] Failed to load PDF '{file_path.name}': {e}")

    return documents


def load_all_documents(directory: Path = DATA_DIR) -> List[Document]:
    """Loads all PDF documents from the specified data directory."""
    all_docs: List[Document] = []
    pdf_files = list(directory.glob("*.pdf"))

    print(f"[INFO] Found {len(pdf_files)} PDF files in '{directory}'.")
    for pdf_file in pdf_files:
        docs = load_single_pdf(pdf_file)
        print(f"[INFO] Loaded {len(docs)} pages from '{pdf_file.name}'.")
        all_docs.extend(docs)

    return all_docs


def get_indexed_documents_summary(directory: Path = DATA_DIR) -> List[dict]:
    """Returns a summary list of all PDF files present in the data folder."""
    pdf_files = list(directory.glob("*.pdf"))
    summary = []
    for pdf in pdf_files:
        try:
            reader = PdfReader(str(pdf))
            summary.append({
                "filename": pdf.name,
                "size_mb": round(pdf.stat().st_size / (1024 * 1024), 2),
                "total_pages": len(reader.pages),
            })
        except Exception:
            summary.append({
                "filename": pdf.name,
                "size_mb": round(pdf.stat().st_size / (1024 * 1024), 2),
                "total_pages": "Unknown",
            })
    return summary
