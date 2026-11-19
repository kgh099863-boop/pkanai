# Project tools

- data_generation/: utilities for the bilingual lexicon and grouped training shards
- maintenance/: grouped-schema normalization and project maintenance
- benchmarks/: data-loader and CPU benchmarking
- archive/: historical migrations and repairs; do not run these on the active dataset

Run current tools from the project root. After changing data, run:

    python tests/audit_schema.py
    python train.py --count-params --device auto

Never add evaluation files to manifest.json under training_files.
