import torch
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME

_cached_embeddings = None


def get_embedding_function(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Returns a cached HuggingFaceEmbeddings instance optimized for CPU/GPU."""
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing Embeddings model '{model_name}' on device '{device}'...")

    _cached_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return _cached_embeddings
