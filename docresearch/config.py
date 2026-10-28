"""All the knobs in one place, so nothing is hard-coded deep inside the graph."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    """Read from environment (.env locally), then Streamlit secrets (when deployed)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return str(st.secrets[name])
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    # --- LLM (anything OpenAI-compatible: OpenAI, Groq, Ollama, LM Studio...) ---
    llm_model: str = _get("LLM_MODEL", "gpt-4o-mini")
    llm_base_url: str | None = _get("LLM_BASE_URL") or None
    llm_api_key: str = _get("LLM_API_KEY") or _get("OPENAI_API_KEY")

    # --- Embeddings run locally, so indexing costs nothing ---
    embedding_model: str = _get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Vector DB ---
    chroma_dir: str = _get("CHROMA_DIR", "./.chroma")
    collection: str = "research_docs"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ---
    top_k: int = 6
    per_doc_k: int = 3
    max_passages: int = 12

    # --- Agent behaviour ---
    max_rewrites: int = 2
    max_revisions: int = 1


settings = Settings()