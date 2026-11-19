# pkanai — Chinese to Thai translation

Experimental neural machine translation using a bidirectional GRU encoder, attention decoder, and shared BPE tokenizer. This is research software, not a production-ready translation service.

## Data format

Every active training and evaluation shard uses one grouped record shape:

- id: integer group ID inside its shard
- section: readable topic heading
- category: category label
- zh: one Chinese source sentence
- th1, th2, th3: up to three Thai references; unused alternatives are JSON null
- frame: sentence-pattern label

A source sentence is stored once per group. The loader expands each non-empty Thai reference into a training pair. Missing alternatives are not fabricated or duplicated.

Current corpus:
- Training: 6,414 source groups / 6,534 Thai references
- Held-out evaluation: 61 source groups / 61 Thai references
- Training and evaluation source sentences are disjoint.
- The multi-reference shard contains 60 groups with three Thai alternatives each.

The manifest lists active training/evaluation shards. Evaluation data is never used to train the model or learn its tokenizer.

## Project layout

- train.py, translate.py: training and translation entry points
- model.py, tokenizer.py, compositional.py: model and text-processing code
- data/train/: grouped training shards
- data/evaluation/: held-out evaluation shards
- data/manifest.json: active shard list, schema and counts
- data/tokenizer_bpe.json: corpus-fingerprint tokenizer cache
- checkpoints/: model weights and resumable state; excluded from Git
- backups/: dated recovery snapshots; excluded from Git
- tests/: schema and translation tests
- tools/data_generation/: dataset and lexicon generation
- tools/maintenance/: schema and project maintenance

## Train

Run from the project root:

    python train.py --count-params --device auto
    python train.py --epochs 5000 --threads 2 --batch-size 256 --save-every 100 --device auto

Auto device selection prefers CUDA, then Apple MPS, then CPU. CUDA mixed precision is enabled where supported. Validation is split by Chinese source group so alternate translations of one source cannot leak between Train and Validation.

Use --device cpu, --device cuda, or --device mps to choose explicitly. Use --no-amp to disable CUDA mixed precision.

## Translate

    python translate.py "你好！"

## Validate

    python tests/audit_schema.py
    python test_suite.py

The data is assistant-curated or template-assisted and still requires fluent Chinese–Thai review. Most older records currently have one reviewed/available reference in th1; th2 and th3 remain null until distinct alternatives are added.
