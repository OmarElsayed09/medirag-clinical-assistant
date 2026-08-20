import os
from typing import Optional, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from src.config import (
    DEFAULT_LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    GOOGLE_API_KEY,
)


def get_llm(
    provider: str = DEFAULT_LLM_PROVIDER,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Factory to instantiate LLMs with Groq as the primary ultra-fast provider,
    with automatic multi-model fallback on rate limits (429) and provider errors.
    """
    provider = (provider or DEFAULT_LLM_PROVIDER).lower()

    if provider == "groq":
        from dotenv import load_dotenv
        load_dotenv(override=True)
        active_key = (api_key or os.getenv("GROQ_API_KEY") or GROQ_API_KEY or "").strip().strip("\"'")
        if not active_key:
            raise ValueError(
                "Groq API Key is missing. Please set GROQ_API_KEY in your .env file "
                "or provide it in the request/UI (get a free key at https://console.groq.com/keys)."
            )
        active_model = model_name or os.getenv("GROQ_MODEL") or GROQ_MODEL or "groq/compound-mini"
        
        primary_llm = ChatGroq(
            api_key=active_key,
            model_name=active_model,
            temperature=temperature,
            max_retries=3,
            request_timeout=60,
        )

        # Resilient multi-model fallbacks when daily token quotas are exhausted
        candidate_fallbacks = ["groq/compound-mini", "openai/gpt-oss-20b", "allam-2-7b", "groq/compound"]
        fallback_models = [m for m in candidate_fallbacks if m != active_model]
        
        fallbacks = [
            ChatGroq(
                api_key=active_key,
                model_name=fb,
                temperature=temperature,
                max_retries=3,
                request_timeout=60,
            )
            for fb in fallback_models
        ]
        
        return primary_llm.with_fallbacks(fallbacks)

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        active_model = model_name or OLLAMA_MODEL or "llama3.2:3b"
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=active_model,
            temperature=temperature,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        active_key = api_key or GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        if not active_key:
            raise ValueError(
                "Google API Key is missing. Please set GOOGLE_API_KEY in your .env file "
                "or provide it in the UI (free at https://aistudio.google.com/)."
            )
        active_model = model_name or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            google_api_key=active_key,
            model=active_model,
            temperature=temperature,
        )

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch

        model_id = model_name or "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"[INFO] Loading local HuggingFace model '{model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )
        return HuggingFacePipeline(pipeline=pipe)

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Choose 'groq', 'ollama', 'gemini', or 'huggingface'.")
