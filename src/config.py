import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base filesystem directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
BM25_CACHE_PATH = CHROMA_DIR / "bm25_index.pkl"


# Chunking Configuration for Clinical Guidelines
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "750"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Embeddings Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval and Reranker Settings (Two-Stage Hybrid Search)
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "6"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.6"))
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() in ("true", "1", "yes")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_INITIAL_K = int(os.getenv("RERANKER_INITIAL_K", "15"))

# LLM Providers Configuration
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Alternative Provider Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Medical Disclaimer Text (Zero Emojis)
MEDICAL_DISCLAIMER = (
    "Medical Disclaimer: This information is synthesized from official medical clinical guidelines "
    "for educational and reference purposes only. It is not a substitute for formal medical advice, "
    "clinical judgment, or professional diagnosis."
)
