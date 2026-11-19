# Data policy

All active training and held-out evaluation shards use the same grouped multi-reference record shape.

## Record fields

- id: unique integer within a shard
- section: readable heading
- category: category or frame label
- zh: Chinese source sentence
- th1, th2, th3: three required, distinct Thai references for the same Chinese source sentence. No null or empty translation fields are allowed.
- frame: stable sentence-pattern label

Each shard has the same outer keys: schema_version, topic, description, metadata, examples. Text is UTF-8, normalized to Unicode NFC, and trimmed.

The loader expands all three Thai references into training pairs but splits validation by Chinese source group. This prevents alternate translations of the same source sentence from leaking between Train and Validation.

Evaluation files are excluded from training_files and must never be used for training or tokenizer learning.

Do not copy a translation into multiple fields to fill the schema. Alternative references must preserve the Chinese meaning and use different natural Thai wording. Automatically paraphrased or template-generated material remains marked for Thai native-speaker review. Evaluation files must remain outside training and tokenizer learning.
