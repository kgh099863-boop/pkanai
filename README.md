# pkanai — Chinese to Thai translation

Experimental Chinese-to-Thai neural machine translation using a bidirectional GRU encoder, attention decoder, and shared BPE tokenizer. This is research software; it is not a production-ready translation service.

## Dataset format

Every active training and held-out evaluation example uses the same grouped record shape:

- `id`: unique integer within its shard
- `section`: readable topic heading
- `category`: category label
- `zh`: one Chinese source sentence
- `th1`, `th2`, `th3`: three required, distinct Thai translations of that source
- `frame`: sentence-pattern label

A source sentence is stored once, with three different Thai references. All three are expanded into training pairs. Missing or duplicate translations fail the dataset audit and the training loader.

## Current dataset

- Training: **4,878 source groups / 14,634 Thai references**
- Held-out evaluation: **61 source groups / 183 Thai references**
- Every active training and evaluation record has three non-empty Thai references.
- Train and evaluation source sentences are disjoint; validation is also split by Chinese source group to avoid reference leakage.
- Added 718 source groups covering useful objects and equipment, technology, abstract concepts, daily conversation, greetings, weather and common requests.
- Repeated synthetic frames were capped to reduce over-representation: `time_need` and `time_no_have` 500 each; `time_go` and `time_group_go` 300 each; `time_eat` 180; `time_drink` 140.

Automatically generated paraphrases are marked for Thai native-speaker review. Three different strings do not guarantee that all three have identical nuance or naturalness; review before production use.

## Project layout

- `train.py`, `translate.py`: training and translation entry points
- `model.py`, `tokenizer.py`, `compositional.py`: model, tokenizer and phrase composition
- `data/train/`: grouped training shards
- `data/evaluation/`: held-out evaluation shards
- `data/manifest.json`: active shard list, schema and counts
- `data/tokenizer_bpe.json`: corpus-fingerprint tokenizer cache (generated locally)
- `data/lexical_frequency_report.json`: current dataset coverage and frequency report
- `checkpoints/`: model weights and resumable state; excluded from Git
- `backups/`: recovery snapshots; excluded from Git
- `tests/`: schema and translation tests
- `tools/data_generation/`: dataset and lexicon generation
- `tools/maintenance/`: schema and project maintenance

## Train

Run from the project root:

    python train.py --count-params --device auto
    python train.py --epochs 5000 --threads 2 --batch-size 256 --save-every 100 --device auto --resume

Device selection prefers CUDA, then Apple MPS, then CPU. CUDA mixed precision is enabled where supported. On Android Termux in this environment, PyTorch reports CPU-only; use a supported PC/NVIDIA CUDA environment to train on an NVIDIA GPU.

Validation is split by Chinese source group. Alternative translations of one Chinese sentence cannot be divided between Train and Validation.

When `--resume` finds a checkpoint with a different BPE vocabulary, the trainer warm-starts compatible model weights and remaps embedding/output rows for shared tokens. The optimizer and epoch counters are reset because the tokenizer and corpus changed. The previous checkpoint copies are kept under `backups/checkpoints_before_reference_expansion/`.

Use `--device cpu`, `--device cuda`, or `--device mps` to choose explicitly. Use `--no-amp` to disable CUDA mixed precision.

## Translate

    python translate.py "你好！"

Full-sentence entries are checked before character-span translation. Non-Chinese text, URLs, placeholders and emoji are preserved in mixed-language messages.

## Validate

    python -m py_compile train.py translate.py tools/data_generation/expand_coverage.py tests/audit_schema.py
    python tests/audit_schema.py
    python test_suite.py

The current model checkpoint was not retrained as part of dataset preparation. Improved lookup results for known phrases should not be confused with improved neural generalization; evaluate again after retraining.
