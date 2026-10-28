"""Embeddings + Chroma vector DB."""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    return Chroma(
        collection_name=settings.collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[Document], ids: list[str]) -> None:
    get_store().add_documents(chunks, ids=ids)


def list_sources() -> list[str]:
    data = get_store().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"] if m and "source" in m})


def delete_source(name: str) -> None:
    get_store()._collection.delete(where={"source": name})
