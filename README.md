# 📚 Document Research Agent

**Built by Abhinav Gupta**

Live link: https://documentresearchagent-f55ub27cmnj9xczwrkfl2m.streamlit.app/

Upload a few papers, a company report, some technical docs. Ask *"Compare the approaches used in these papers."* Get an answer where every claim points back to a **file and page number**.

It's a multi-document RAG system built as a **LangGraph** agent that checks its own work: it grades what it retrieved, rewrites searches that miss, fact-checks its draft, and says "I couldn't find this" when the documents don't back an answer.

![Answer with citations](docs/answer.png)
![How I got here](docs/trace.png)

## How it works

```
Documents -> Loader -> Chunking -> Embeddings -> Chroma
                                                   |
Question -> Analyze -> Retrieve -> Grade --relevant--> Generate -> Verify -> Answer + Sources
                          ^          |                    ^           |
                          |          | not relevant       +- revise --+
                          +-- Rewrite <-+
                                       |
                    out of retries -> Partial answer (if some evidence) or Abstain
```

| Step | What it does |
|---|---|
| **Loader** | Reads PDF (per page), DOCX (including tables), TXT, MD. Page numbers are kept. |
| **Chunking** | 1000-char chunks, 150 overlap, with stable IDs so re-uploading a file replaces it instead of duplicating it. |
| **Embeddings** | `all-MiniLM-L6-v2`, run locally. Indexing is free. |
| **Vector DB** | Chroma, persisted to `.chroma/`. |
| **Analyze** | The LLM classifies the question (compare / summarize / lookup) and writes 1-3 search queries. |
| **Retrieve** | Normal top-k, **except for comparisons**, which search each document separately. Table-of-contents chunks are filtered out. |
| **Grade** | Is there enough to answer a meaningful part of the question? Keeps only the useful passages. |
| **Rewrite** | If not, rephrase and search again (max 2 times). |
| **Partial answer** | If retries run out but some evidence exists, answer what the documents cover and flag the gaps. |
| **Generate** | Answers only from the numbered sources, tagging claims `[S1]`. |
| **Verify** | A second pass checks each claim against its source. One redraft if it fails. |
| **Finalize** | Turns tags into `[paper2.pdf, p.7]` and attaches the source passages. |

## Design decisions

- **Comparisons are retrieved per document.** Plain top-k on "compare these papers" tends to return chunks from whichever paper shares the most keywords, so the others vanish from the answer. For comparison questions each document is searched separately, so all of them get represented.
- **It can say no.** If nothing relevant turns up, the agent abstains instead of inventing something. An honest "not in your documents" beats a confident wrong answer.
- **It shows its work.** The UI streams each step ("Pulled 9 passages from 3 documents", "Rephrasing the search...") and keeps it in a "How I got here" panel.
- **Bad citations get dropped.** If the model cites a source number that doesn't exist, that tag is removed instead of shown to the reader.
- **Unverified claims are flagged, not hidden.** If the fact-check still fails after a redraft, the answer ships with a visible warning listing the shaky claims.
- **Scanned PDFs are reported, not silently ignored.** Pages with no extractable text are counted and the user is told.

## Bugs I hit while testing (and fixed)

Testing on real documents found problems that toy examples never showed:

1. **Word tables were invisible.** The DOCX loader only read paragraphs, so tech-stack tables never reached the index. The grader kept saying the passages "only mention that tables exist." Fix: read table rows too.
2. **Table-of-contents pages outranked real content.** Entries like "Table 4.2 Technology Stack ..... 31" match a query almost perfectly while saying nothing. Fix: filter chunks with several dot-leader page references.
3. **The grader was too strict.** A three-part question failed completely if one part was thin, so the agent refused answers it could partly give. Fix: accept partial evidence and fall back to a flagged partial answer.
4. **Duplicate files diluted comparisons.** The same report indexed twice (as DOCX and PDF) crowded the results. Lesson: upload one version of each document.
5. **LLM provider quirks.** Free-tier models get renamed or retired, and not every model supports strict JSON output. Fix: use function-calling mode for structured output, and keep the model name in `.env`.

## Setup

```bash
python -m venv .venv
```

**Windows:**
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

**Mac / Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Open `.env` and add your API key before starting the app. The first run downloads the embedding model (about 90 MB). Click **Try with sample documents** in the sidebar and ask *"Compare the approaches used in these documents."* The samples are three short notes written for the demo (RAG vs fine-tuning vs long-context prompting), not real papers.

### LLM options

Anything OpenAI-compatible works, set in `.env`: OpenAI, Groq, or fully local via Ollama. The graph uses structured output, so pick a model that supports tool calling. Free-tier model names change often, so check your provider's current model list if you get a "model not found" error.

### CLI

```bash
python cli.py ingest sample_docs/*.md
python cli.py ask "Compare the approaches used in these documents."
```

## Testing

I tested it by hand on the sample documents and on a couple of my own long project reports, checking each cited page against the source:

- Single-document factual questions with page-accurate citations
- Cross-document comparisons that cite both files
- Out-of-scope questions (it retries with rephrased searches, then declines)
- A fact-check failure that triggered a redraft

Unit tests cover the citation formatting (`pytest tests`). There is **no formal accuracy evaluation yet**, so I make no accuracy claims.

## Project layout

```
app.py                  Streamlit UI
cli.py                  Command-line version
docresearch/
  config.py             All settings
  ingest.py             Loader + chunking + indexing
  vectorstore.py        Embeddings + Chroma
  retrieval.py          Retriever (per-document balancing, contents-page filter)
  graph.py              The LangGraph agent
  prompts.py            All prompts
  schemas.py            Structured outputs (Pydantic)
  citations.py          [S1] -> [file, p.N]
tests/                  Citation logic tests
sample_docs/            Demo documents
docs/                   Screenshots
```

## Known limitations

- No OCR, so scanned PDFs can't be searched.
- Chunks don't cross page boundaries, so a sentence split across two pages is cut in two.
- Page numbers for DOCX/TXT/MD are always 1. Export to PDF for real page numbers.
- Each question makes about 4-6 LLM calls (analyze, grade, generate, verify), so it's slower and costlier than plain RAG. That's the price of the checks.
- The contents-page filter is a heuristic and can miss unusual layouts.
- No evaluation set yet.

## What I'd build next

- A small eval set of questions with known answers, to measure whether grading and verification actually improve accuracy.
- Hybrid search (BM25 + embeddings) and a reranker.
- OCR for scanned files.
- Conversation memory, so follow-ups like "and what about the second paper?" work.
- Smarter query rewriting that sees all earlier attempts, so retries don't repeat themselves.

## Stack

Python · LangGraph · LangChain · ChromaDB · sentence-transformers · Streamlit · Pydantic · Groq

## License

MIT
