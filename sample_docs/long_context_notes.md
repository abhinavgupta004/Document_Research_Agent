# Long-context prompting: working notes

These are short demo notes written for this project, not a published paper.

## Approach
Instead of retrieving, the entire set of documents is placed directly in the prompt of a model with a very large context window. The model reads everything and answers in one pass.

## Data requirements
No indexing and no training. The documents only need to fit inside the context window.

## Strengths
It is the simplest to build. There is no retrieval step that could miss the relevant passage, and the model can connect information from anywhere in the text.

## Limitations
Every question pays to process all the documents again, so cost and latency grow with library size. Models can also pay less attention to material buried in the middle of a very long prompt. Libraries larger than the window do not fit at all.
