"""Structured outputs the LLM must return at each decision point in the graph."""
from typing import Literal

from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    intent: Literal["compare", "summarize", "lookup", "other"] = Field(
        description="What kind of question this is."
    )
    sub_queries: list[str] = Field(
        description="1 to 3 short, self-contained search queries (keywords work best)."
    )
    target_sources: list[str] = Field(
        default_factory=list,
        description="Exact file names from the library that the question explicitly names. Empty if none.",
    )


class Grade(BaseModel):
    relevant: bool = Field(
        description="True if the passages contain enough to genuinely answer the question."
    )
    useful_indices: list[int] = Field(
        default_factory=list,
        description="Indices of the passages worth keeping for the answer.",
    )
    reason: str = Field(description="One sentence on what is missing or what was found.")


class Rewrite(BaseModel):
    queries: list[str] = Field(description="1 or 2 new search queries, phrased differently.")


class Verdict(BaseModel):
    supported: bool = Field(
        description="True if every factual claim in the draft is backed by the cited passage."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims that are not backed by the passages, quoted briefly.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How well the sources cover the question overall."
    )
