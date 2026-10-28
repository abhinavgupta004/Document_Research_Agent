"""Streamlit front end:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from docresearch.config import settings
from docresearch.graph import ask
from docresearch.ingest import SUPPORTED, ingest_file
from docresearch.vectorstore import delete_source, list_sources

st.set_page_config(page_title="Document Research Agent", page_icon="📚", layout="wide")

EXAMPLES = [
    "Compare the approaches used in these documents.",
    "What are the main limitations each document mentions?",
    "Summarize the key findings in a few bullet points.",
]


# ----------------------------------------------------------------- sidebar
def sidebar() -> list[str]:
    st.sidebar.header("📂 Your library")

    uploads = st.sidebar.file_uploader(
        "Add documents",
        type=[e.lstrip(".") for e in sorted(SUPPORTED)],
        accept_multiple_files=True,
    )
    if uploads and st.sidebar.button("Index these", type="primary"):
        with st.spinner("Reading and indexing... the first run downloads the embedding model."):
            for up in uploads:
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / up.name
                    path.write_bytes(up.getbuffer())
                    report = ingest_file(path)
                if report.error:
                    st.sidebar.error(f"{up.name}: {report.error}")
                else:
                    st.sidebar.success(f"{up.name}: {report.pages} pages, {report.chunks} chunks")
                    for note in report.notes:
                        st.sidebar.caption(f"ℹ️ {note}")
        st.rerun()

    sample_dir = Path(__file__).parent / "sample_docs"
    if sample_dir.exists() and st.sidebar.button("Try with sample documents"):
        with st.spinner("Indexing samples..."):
            for f in sorted(sample_dir.glob("*.md")):
                ingest_file(f)
        st.rerun()

    sources = list_sources()
    st.sidebar.divider()
    if not sources:
        st.sidebar.caption("Nothing indexed yet.")
    for name in sources:
        c1, c2 = st.sidebar.columns([5, 1])
        c1.write(f"📄 {name}")
        if c2.button("✕", key=f"del-{name}", help=f"Remove {name}"):
            delete_source(name)
            st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(f"Model: `{settings.llm_model}` · Embeddings: `{settings.embedding_model.split('/')[-1]}`")
    return sources


# ------------------------------------------------------------------ render
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['source']}** · page {s['page']} · match {s['score']:.2f}")
            st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))


def render_assistant(msg: dict) -> None:
    st.markdown(msg["answer"])
    render_sources(msg.get("sources", []))
    if msg.get("trace"):
        with st.expander("🧠 How I got here"):
            for line in msg["trace"]:
                st.markdown(f"- {line}")


# -------------------------------------------------------------------- main
def main() -> None:
    st.title("📚 Document Research Agent")
    st.caption("Ask questions across your documents. Every answer comes with page-level sources.")

    sources = sidebar()

    if not settings.llm_api_key and not settings.llm_base_url:
        st.warning("No LLM key found. Copy `.env.example` to `.env` and add one, then restart.")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    if not sources:
        st.info("Start by uploading a few documents in the sidebar, or try the sample ones.")
    elif not st.session_state.history:
        st.markdown("**Some things to try:**")
        for ex in EXAMPLES:
            st.markdown(f"- {ex}")

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(msg)

    question = st.chat_input("Ask something about your documents...", disabled=not sources)
    if not question:
        return

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        final, shown = None, 0
        try:
            with st.status("Working through your documents...", expanded=True) as status:
                for state in ask(question):
                    final = state
                    trace = state.get("trace", [])
                    for line in trace[shown:]:
                        st.write(line)
                    shown = len(trace)
                status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            st.error(
                "Something went wrong talking to the model. Check your API key and "
                f"model name in `.env`.\n\n`{type(exc).__name__}: {exc}`"
            )
            return

        msg = {
            "role": "assistant",
            "answer": final["answer"],
            "sources": final.get("sources", []),
            "trace": final.get("trace", []),
        }
        render_assistant(msg)
        st.session_state.history.append(msg)


main()
