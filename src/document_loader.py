import re
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_core.documents import Document

from src.config import DATA_DIR


def clean_medical_text(text: str) -> str:
    """
    Cleans raw PDF text by normalizing whitespace, repairing hyphenated medical terms,
    preserving table structures, and stripping repetitive header/footer artifacts.
    """
    if not text:
        return ""

    # Standardize newline characters
    text = re.sub(r"\r\n|\r", "\n", text)

    # Remove generic footer/header repetitive patterns
    text = re.sub(r"(?i)\bpage\s+\d+\s+of\s+\d+\b", "", text)
    text = re.sub(r"(?i)©\s*World Health Organization\s*\d{4}", "", text)
    text = re.sub(r"(?i)national institute for health and care excellence", "", text)

    # Merge hyphenated words broken across lines (e.g., hyper-\ntension -> hypertension)
    text = re.sub(r"(\b[A-Za-z]+)-\n([A-Za-z]+\b)", r"\1\2", text)

    # Preserve table rows: replace multiple tabs/spaces with a clean separator
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # If line looks like a table row (multiple separated tokens), align with pipe-like spacing
        tokens = re.split(r"\s{3,}|\t+", stripped)
        if len(tokens) > 2:
            cleaned_line = " | ".join(t.strip() for t in tokens if t.strip())
        else:
            cleaned_line = re.sub(r"[ \t]+", " ", stripped)
        cleaned_lines.append(cleaned_line)

    text = "\n".join(cleaned_lines)

    # Replace non-breaking spaces and normalize excessive newlines
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_single_pdf(file_path: Path) -> List[Document]:
    """Loads a single PDF file and extracts text page by page with rich clinical metadata."""
    documents: List[Document] = []
    try:
        reader = PdfReader(str(file_path))
        filename = file_path.name

        for page_idx, page in enumerate(reader.pages):
            try:
                # Use layout extraction mode if supported by pypdf
                raw_text = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
            except Exception:
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


def get_indexed_documents_summary(directory: Path = DATA_DIR) -> List[Dict[str, Any]]:
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
