# Fine-tuning: working notes

These are short demo notes written for this project, not a published paper.

## Approach
Fine-tuning continues training a pretrained model on a curated dataset so that new behaviour or knowledge is stored in the model's weights. Parameter-efficient methods such as LoRA train a small set of added weights instead of the whole model.

## Data requirements
It needs a high-quality labelled or instruction-style dataset, usually hundreds to thousands of examples, plus compute for training and evaluation.

## Strengths
It is very good at changing style, format, and task behaviour. Inference is cheap and fast because nothing extra is retrieved at question time.

## Limitations
Facts baked into weights go stale and are hard to update without training again. The model cannot point to a source for what it says, which makes answers difficult to verify.
