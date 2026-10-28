"""Retriever.

The one non-obvious idea: for comparison questions, plain top-k search tends to
return five chunks from whichever paper happens to use the most matching words,
and the other papers never show up. So for comparisons we search each document
separately and make sure every one gets a seat at the table.
"""
import re

from langchain_core.documents import Document

from .config import settings
from .vectorstore import get_store


def _to_passage(doc: Document, distance: float) -> dict:
    return {
        "chunk_id": doc.metadata.get("chunk_id", ""),
        "source": doc.metadata["source"],
        "page": doc.metadata.get("page", 1),
        "text": doc.page_content,
        "score": round(1 - distance, 3),  # cosine similarity
    }


_TOC_ENTRY = re.compile(r"\.{4,}\s*\d+")


def _looks_like_toc(text: str) -> bool:
    """Contents pages and lists of tables match keywords but say nothing."""
    return len(_TOC_ENTRY.findall(text)) >= 3


def _search(query: str, k: int, source: str | None = None) -> list[dict]:
    flt = {"source": source} if source else None
    # Ask for a few extra so dropping contents pages doesn't leave us short.
    results = get_store().similarity_search_with_score(query, k=k + 3, filter=flt)
    passages = [_to_passage(doc, dist) for doc, dist in results]
    return [p for p in passages if not _looks_like_toc(p["text"])][:k]


def retrieve(
    queries: list[str],
    intent: str,
    target_sources: list[str],
    all_sources: list[str],
) -> list[dict]:
    # Which documents should each query be run against separately?
    balanced_over = target_sources or (all_sources if intent == "compare" else [])

    found: dict[str, dict] = {}
    for q in queries:
        if balanced_over:
            for src in balanced_over:
                for p in _search(q, settings.per_doc_k, src):
                    _keep_best(found, p)
        else:
            for p in _search(q, settings.top_k):
                _keep_best(found, p)

    passages = sorted(found.values(), key=lambda p: p["score"], reverse=True)

    if balanced_over:
        # Keep the best few per document, so no single paper crowds out the rest.
        per_doc: dict[str, int] = {}
        kept = []
        for p in passages:
            per_doc[p["source"]] = per_doc.get(p["source"], 0) + 1
            if per_doc[p["source"]] <= settings.per_doc_k:
                kept.append(p)
        return kept

    return passages[: settings.max_passages]


def _keep_best(found: dict[str, dict], p: dict) -> None:
    cid = p["chunk_id"] or f'{p["source"]}:{p["page"]}:{p["text"][:40]}'
    if cid not in found or p["score"] > found[cid]["score"]:
        found[cid] = p