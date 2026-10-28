"""The LangGraph agent.

    analyze -> retrieve -> grade --relevant--> generate -> verify --ok--> finalize
                   ^          |                   ^           |
                   |          +--not relevant     +-- revise -+
                   |          v
                   +------- rewrite            (out of retries) -> abstain

Every node appends a plain-English line to `trace`, which the UI streams live.
That trace is the "show your work" part of the project.
"""
import operator
import re
from functools import lru_cache
from typing import Annotated, Iterator, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .citations import link_citations
from .config import settings
from .prompts import ANALYZER, GENERATOR, GRADER, REWRITER, STYLE_BY_INTENT, VERIFIER
from .retrieval import retrieve
from .schemas import Grade, QueryPlan, Rewrite, Verdict
from .vectorstore import list_sources


class ResearchState(TypedDict, total=False):
    question: str
    intent: str
    queries: list[str]
    target_sources: list[str]
    passages: list[dict]
    rewrites: int
    relevant: bool
    fallback: list[dict]
    partial: bool
    grade_reason: str
    draft: str
    feedback: str
    revisions: int
    verified: bool
    needs_revision: bool
    unsupported: list[str]
    confidence: str
    answer: str
    sources: list[dict]
    trace: Annotated[list[str], operator.add]


# ----------------------------------------------------------------- helpers
def _llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "not-needed",
        temperature=temperature,
    )


def _numbered(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({p['source']}, p.{p['page']})\n{p['text']}" for i, p in enumerate(passages)
    )


def _context(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{p['id']}] ({p['source']}, p.{p['page']})\n{p['text']}" for p in passages
    )


# ------------------------------------------------------------------- nodes
def analyze(state: ResearchState) -> dict:
    sources = list_sources()
    plan: QueryPlan = _llm().with_structured_output(QueryPlan, method="function_calling").invoke(
        ANALYZER.format_messages(
            question=state["question"], sources=", ".join(sources) or "(none)"
        )
    )
    queries = [q for q in plan.sub_queries if q.strip()][:3] or [state["question"]]
    targets = [s for s in plan.target_sources if s in sources]

    scope = f" focused on {', '.join(targets)}" if targets else ""
    return {
        "intent": plan.intent,
        "queries": queries,
        "target_sources": targets,
        "rewrites": 0,
        "revisions": 0,
        "feedback": "",
        "fallback": [],
        "partial": False,
        "trace": [
            f"🧭 Read the question: looks like a **{plan.intent}** request{scope}. "
            f"Searching for: {'; '.join(queries)}"
        ],
    }


def retrieve_node(state: ResearchState) -> dict:
    passages = retrieve(
        state["queries"], state["intent"], state.get("target_sources", []), list_sources()
    )
    docs = sorted({p["source"] for p in passages})
    return {
        "passages": passages,
        "trace": [f"📚 Pulled {len(passages)} passages from {len(docs)} document(s)."],
    }


def grade(state: ResearchState) -> dict:
    passages = state["passages"]
    if not passages:
        return {
            "relevant": False,
            "grade_reason": "The index returned nothing.",
            "trace": ["🔎 Nothing came back from the index."],
        }

    g: Grade = _llm().with_structured_output(Grade, method="function_calling").invoke(
        GRADER.format_messages(question=state["question"], passages=_numbered(passages))
    )
    keep = [passages[i] for i in dict.fromkeys(g.useful_indices) if 0 <= i < len(passages)]
    relevant = g.relevant and bool(keep)

    if relevant:
        keep = [{**p, "id": f"S{n}"} for n, p in enumerate(keep, 1)]
        return {
            "relevant": True,
            "passages": keep,
            "grade_reason": g.reason,
            "trace": [f"✅ Kept {len(keep)} of {len(passages)} passages as useful. {g.reason}"],
        }
    # Remember the best partial evidence in case every retry falls short.
    prev = state.get("fallback", [])
    best = keep if len(keep) >= len(prev) else prev
    best = [{**p, "id": f"S{n}"} for n, p in enumerate(best, 1)]
    return {
        "relevant": False,
        "grade_reason": g.reason,
        "fallback": best,
        "trace": [f"🤔 These passages don't really answer it. {g.reason}"],
    }


def rewrite(state: ResearchState) -> dict:
    out: Rewrite = _llm().with_structured_output(Rewrite, method="function_calling").invoke(
        REWRITER.format_messages(
            question=state["question"],
            tried="; ".join(state["queries"]),
            reason=state.get("grade_reason", ""),
        )
    )
    queries = [q for q in out.queries if q.strip()][:2] or state["queries"]
    n = state["rewrites"] + 1
    return {
        "queries": queries,
        "rewrites": n,
        # Widen the net: the first plan may have pointed at the wrong documents.
        "target_sources": [],
        "trace": [
            f"✏️ Rephrasing the search (attempt {n} of {settings.max_rewrites}): "
            f"{'; '.join(queries)}"
        ],
    }


def generate(state: ResearchState) -> dict:
    style = STYLE_BY_INTENT.get(state["intent"], STYLE_BY_INTENT["other"])
    messages = GENERATOR.format_messages(
        question=state["question"], context=_context(state["passages"]), style=style
    )
    if state.get("feedback"):
        messages.append(
            HumanMessage(
                "A fact-checker found problems with your last draft:\n"
                f"{state['feedback']}\nRewrite the answer, fixing these. Remove any "
                "claim the sources don't support."
            )
        )
    draft = _llm(temperature=0.2).invoke(messages).content
    return {"draft": draft, "trace": ["✍️ Drafted an answer from the sources."]}


def verify(state: ResearchState) -> dict:
    v: Verdict = _llm().with_structured_output(Verdict, method="function_calling").invoke(
        VERIFIER.format_messages(draft=state["draft"], context=_context(state["passages"]))
    )
    if v.supported:
        return {
            "verified": True,
            "needs_revision": False,
            "confidence": v.confidence,
            "unsupported": [],
            "trace": [f"🛡️ Fact-checked every claim against the sources. Confidence: {v.confidence}."],
        }

    can_retry = state.get("revisions", 0) < settings.max_revisions
    return {
        "verified": False,
        "needs_revision": can_retry,
        "revisions": state.get("revisions", 0) + (1 if can_retry else 0),
        "feedback": "\n".join(f"- {c}" for c in v.unsupported_claims),
        "unsupported": v.unsupported_claims,
        "confidence": v.confidence,
        "trace": [
            f"⚠️ Fact-check flagged {len(v.unsupported_claims)} claim(s)."
            + (" Redrafting." if can_retry else " Out of retries, so I'll flag them.")
        ],
    }


def use_partial(state: ResearchState) -> dict:
    """Retries are used up, but we did find *something*. Answer what's covered, flag the rest."""
    fb = state["fallback"]
    return {
        "passages": fb,
        "relevant": True,
        "partial": True,
        "trace": [
            f"🩹 No complete answer found, so I'll answer what the documents do cover "
            f"({len(fb)} passages) and point out the gaps."
        ],
    }


def finalize(state: ResearchState) -> dict:
    text, used = link_citations(state["draft"], state["passages"])
    text = re.sub(r"<br\s*/?>", "; ", text)  # models sometimes emit HTML line breaks in tables
    by_id = {p["id"]: p for p in state["passages"]}
    # If the model cited nothing, show everything we gave it rather than hide the evidence.
    sources = [by_id[i] for i in used] if used else state["passages"]

    if state.get("partial"):
        text = (
            "> ℹ️ **Partial answer:** your documents only cover part of this question. "
            "Here is what they do say.\n\n" + text
        )

    if not state.get("verified"):
        flagged = "\n".join(f"- {c}" for c in state.get("unsupported", []))
        text += (
            "\n\n> ⚠️ **Heads up:** I couldn't fully back these points from your documents, "
            f"so treat them with caution:\n{flagged}"
        )
    return {
        "answer": text,
        "sources": sources,
        "trace": [f"📎 Done. Cited {len(sources)} passage(s)."],
    }


def abstain(state: ResearchState) -> dict:
    tried = "; ".join(state.get("queries", []))
    if not state.get("passages"):
        msg = (
            "I don't have anything indexed to search yet. Upload a few documents and "
            "ask again."
        )
    else:
        msg = (
            "I looked through your documents and couldn't find solid support for an "
            "answer, so I'd rather say that than guess.\n\n"
            f"What I searched for: _{tried}_\n\n"
            "Things that might help: ask about a more specific section, use the "
            "document's own terminology, or check that the right files are uploaded."
        )
    return {
        "answer": msg,
        "sources": [],
        "confidence": "low",
        "trace": ["🙅 Not enough evidence in the documents, so I'm not going to make something up."],
    }


# ----------------------------------------------------------------- routing
def route_after_grade(state: ResearchState) -> str:
    if state["relevant"]:
        return "generate"
    if state.get("passages") and state["rewrites"] < settings.max_rewrites:
        return "rewrite"
    if state.get("fallback"):
        return "partial"
    return "abstain"


def route_after_verify(state: ResearchState) -> str:
    return "revise" if state.get("needs_revision") else "finalize"


# ------------------------------------------------------------------- graph
@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("analyze", analyze)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)
    g.add_node("use_partial", use_partial)
    g.add_node("abstain", abstain)

    g.add_edge(START, "analyze")
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "partial": "use_partial",
            "abstain": "abstain",
        },
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("use_partial", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges(
        "verify", route_after_verify, {"revise": "generate", "finalize": "finalize"}
    )
    g.add_edge("finalize", END)
    g.add_edge("abstain", END)
    return g.compile()


def ask(question: str) -> Iterator[ResearchState]:
    """Run the agent, yielding the full state after every step (for live UIs)."""
    yield from build_graph().stream(
        {"question": question, "trace": []}, stream_mode="values"
    )