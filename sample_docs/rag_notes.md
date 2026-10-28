# Retrieval-Augmented Generation (RAG): working notes

These are short demo notes written for this project, not a published paper.

## Approach
RAG keeps the language model frozen. Documents are split into chunks, embedded, and stored in a vector database. At question time the most similar chunks are retrieved and placed in the prompt, so the model answers from fresh, specific text.

## Data requirements
No training data is needed. Any collection of documents can be indexed in minutes, and updating knowledge means re-indexing the changed files.

## Strengths
Answers can cite the exact passage they came from. Knowledge stays current without retraining. Private documents never need to leave your own infrastructure when a local model is used.

## Limitations
Quality depends on retrieval. If the right chunk is not retrieved, the model cannot use it. Questions that need reasoning across many distant passages are harder, because only a handful of chunks fit in the prompt.
