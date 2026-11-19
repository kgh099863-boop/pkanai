# Data policy

All active train and held-out evaluation shards use the same grouped multi-reference record shape.

## Record fields

- id: unique integer within a shard
- section: readable heading
- category: category or frame label
- zh: Chinese source sentence
- th1: required primary Thai reference
- th2 and th3: optional Thai alternatives; use JSON null until a distinct, reviewed alternative exists
- frame: stable sentence-pattern label

Each shard has the same outer keys: schema_version, topic, description, metadata, examples. Text is UTF-8, normalized to Unicode NFC, and trimmed.

The loader expands all non-empty Thai references into training pairs but splits validation by Chinese source group. This prevents alternate translations of the same source sentence from leaking across Train and Validation.

Evaluation files are excluded from training_files and must never be used for training or tokenizer learning.

Do not copy a translation into multiple fields to fill the schema. Add a new Thai variant only when it preserves the Chinese meaning and reads naturally. Template-generated and assistant-authored material should remain marked for human/native-speaker review.
