"""All the knobs in one place, so nothing is hard-coded deep inside the graph."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- LLM (anything OpenAI-compatible: OpenAI, Groq, Ollama, LM Studio...) ---
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_base_url: str | None = os.getenv("LLM_BASE_URL") or None
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    # --- Embeddings run locally, so indexing costs nothing ---
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Vector DB ---
    chroma_dir: str = os.getenv("CHROMA_DIR", "./.chroma")
    collection: str = "research_docs"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ---
    top_k: int = 6          # passages per query for normal questions
    per_doc_k: int = 3      # passages per document for comparisons
    max_passages: int = 12  # hard cap going into the LLM

    # --- Agent behaviour ---
    max_rewrites: int = 2   # how many times we'll rephrase a failing search
    max_revisions: int = 1  # how many times we'll redo an answer that fails verification


settings = Settings()
