# 📚 Document Research Agent

Upload a few papers, a company report, some technical docs. Ask *"Compare the approaches used in these papers."* Get an answer where every claim points back to a **file and page number**.

It's a multi-document RAG system built as a **LangGraph** agent that checks its own work: it grades what it retrieved, rewrites searches that miss, fact-checks its draft, and says "I don't know" when the documents don't back an answer.

## How it works

```
Documents -> Loader -> Chunking -> Embeddings -> Chroma
                                                   |
Question -> Analyze -> Retrieve -> Grade --relevant--> Generate -> Verify -> Answer + Sources
                          ^          |                    ^           |
                          |          | not relevant       +- revise --+
                          +-- Rewrite <-+
                                       |
                                  out of retries -> Abstain ("not in your documents")
```

| Step | What it does |
|---|---|
| **Loader** | Reads PDF (per page), DOCX, TXT, MD. Page numbers are kept. |
| **Chunking** | 1000-char chunks, 150 overlap, with stable IDs so re-uploading a file replaces it instead of duplicating it. |
| **Embeddings** | `all-MiniLM-L6-v2`, run locally. Indexing is free. |
| **Vector DB** | Chroma, persisted to `.chroma/`. |
| **Analyze** | The LLM classifies the question (compare / summarize / lookup) and writes 1-3 search queries. |
| **Retrieve** | Normal top-k, **except for comparisons** (see below). |
| **Grade** | Is this enough to answer? Keeps only the useful passages. |
| **Rewrite** | If not, rephrase and search again (max 2 times). |
| **Generate** | Answers only from the numbered sources, tagging claims `[S1]`. |
| **Verify** | A second pass checks each claim against its source. One redraft if it fails. |
| **Finalize** | Turns tags into `[paper2.pdf, p.7]` and attaches the source passages. |

## Design decisions (the parts I actually thought about)

- **Comparisons are retrieved per document.** Plain top-k on "compare these papers" tends to return chunks from whichever paper shares the most keywords, so the others vanish from the answer. For comparison questions each document is searched separately, so all of them get represented.
- **It can say no.** If retries run out, the agent abstains rather than inventing something. An honest "not in your documents" beats a confident wrong answer.
- **It shows its work.** The UI streams each step ("Pulled 9 passages from 3 documents", "Rephrasing the search...") and keeps it in a "How I got here" panel.
- **Bad citations get dropped.** If the model cites a source number that doesn't exist, that tag is removed instead of shown to the reader.
- **Unverified claims are flagged, not hidden.** If the fact-check still fails after a redraft, the answer ships with a visible warning listing the shaky claims.
- **Scanned PDFs are reported, not silently ignored.** Pages with no extractable text are counted and the user is told.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your key, or point it at a local Ollama
streamlit run app.py
```

First run downloads the embedding model (~90 MB). Click **Try with sample documents** in the sidebar and ask *"Compare the approaches used in these documents."* The samples are three short notes I wrote for the demo (RAG vs fine-tuning vs long-context prompting), not real papers.

### LLM options

Anything OpenAI-compatible works, set in `.env`: OpenAI, Groq, or fully local via Ollama. The graph uses structured output, so pick a model that supports tool calling.

### CLI

```bash
python cli.py ingest sample_docs/*.md
python cli.py ask "Compare the approaches used in these documents."
```

## Project layout

```
app.py                  Streamlit UI
cli.py                  Command-line version
docresearch/
  config.py             All settings
  ingest.py             Loader + chunking + indexing
  vectorstore.py        Embeddings + Chroma
  retrieval.py          Retriever (with per-document balancing)
  graph.py              The LangGraph agent
  prompts.py            All prompts
  schemas.py            Structured outputs (Pydantic)
  citations.py          [S1] -> [file, p.N]
tests/                  Citation logic tests
sample_docs/            Demo documents
```

## Known limitations

- No OCR, so scanned PDFs can't be searched.
- Chunks don't cross page boundaries, so a sentence split across two pages is cut in two.
- Page numbers for DOCX/TXT/MD are always 1.
- Each question makes about 4-6 LLM calls (analyze, grade, generate, verify), so it's slower and costlier than plain RAG. That's the price of the checks.
- No evaluation set yet.

## What I'd build next

- A small eval set of questions with known answers, to measure whether grading and verification actually improve accuracy.
- Hybrid search (BM25 + embeddings) and a reranker.
- OCR for scanned files.
- Conversation memory, so follow-ups like "and what about the second paper?" work.

## Stack

Python · LangGraph · LangChain · ChromaDB · sentence-transformers · Streamlit · Pydantic
