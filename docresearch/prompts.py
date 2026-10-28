"""Prompts live here so you can tune the agent's voice without touching graph logic."""
from langchain_core.prompts import ChatPromptTemplate

ANALYZER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You plan searches over a small library of documents.\n"
            "Given the user's question, decide:\n"
            "- intent: 'compare' if it contrasts two or more documents or approaches, "
            "'summarize', 'lookup' for a specific fact, otherwise 'other'.\n"
            "- sub_queries: 1 to 3 short search queries. For comparisons, write queries "
            "about the dimensions being compared (method, data, results, limitations), "
            "not about each document by name.\n"
            "- target_sources: file names from the library that the question explicitly "
            "mentions. Copy them exactly. Leave empty if the question doesn't name any.\n\n"
            "Library: {sources}",
        ),
        ("human", "{question}"),
    ]
)

GRADER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a fair reviewer. You'll see a question and numbered passages "
            "retrieved from documents. Mark relevant=true if the passages let you "
            "answer a meaningful part of the question with real details, even if "
            "some parts are missing; the answer step will point out the gaps. Mark "
            "relevant=false only if they barely touch the topic. Ignore passages "
            "that are just a table of contents, a list of figures or tables, headers "
            "or footers: they mention topics without saying anything. List the "
            "indices of passages that carry real content. Be brief in your reason.",
        ),
        ("human", "Question: {question}\n\nPassages:\n{passages}"),
    ]
)

REWRITER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "A document search came back with weak results. Write 1 or 2 better search "
            "queries for the same question: try synonyms, the technical term instead of "
            "the everyday one, or a more specific angle. Don't repeat earlier queries.",
        ),
        (
            "human",
            "Question: {question}\nQueries already tried: {tried}\n"
            "Why they failed: {reason}",
        ),
    ]
)

GENERATOR = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful research assistant. Answer using ONLY the numbered "
            "sources below.\n\n"
            "Rules:\n"
            "1. Cite every factual claim with its source tag, like [S1]. One tag per "
            "bracket: write [S1] [S3], never [S1, S3].\n"
            "2. If the sources don't cover part of the question, say so plainly instead "
            "of filling the gap from general knowledge.\n"
            "3. If sources disagree, point out the disagreement rather than smoothing it over.\n"
            "4. Write like a thoughtful colleague: direct, no filler, no 'as an AI'.\n"
            "5. {style}\n\n"
            "Sources:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

STYLE_BY_INTENT = {
    "compare": (
        "This is a comparison. Open with a one or two sentence takeaway, then compare "
        "along the dimensions the sources actually speak to (for example problem, "
        "method, data, results, limitations). If a document is silent on a dimension, "
        "say so. A short markdown table is welcome when it makes things clearer."
    ),
    "summarize": "Give a tight summary: a short overview, then the key points.",
    "lookup": "Answer in a few sentences. Lead with the answer.",
    "other": "Keep it as short as the question allows.",
}

VERIFIER = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You fact-check a draft answer against its sources. For each claim tagged "
            "like [S2], check that source S2 really says it. List any claims that are "
            "unsupported, exaggerated, or tagged with the wrong source. Judge overall "
            "confidence by how fully the sources cover the question.",
        ),
        ("human", "Draft:\n{draft}\n\nSources:\n{context}"),
    ]
)