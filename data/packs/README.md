<!-- SPDX-License-Identifier: Apache-2.0 -->

> [!WARNING]
> **Important Licensing & Usage Recommendation**
> The pre-generated packs here are for **evaluation/demonstration purposes ONLY** and have a specific [Eval-Only, Non-Commercial, No-Derivatives license](#license). They were made with third-party LLMs (e.g., OpenAI).
>
> **Please generate your own dataset packs** using the [TQB++ generator toolkit](../../tools/generator/README.md). This ensures:
> *   Data tailored to your needs (language, domain, difficulty).
> *   Alignment with your chosen LLM provider's terms and output licenses.
> *   Full control and provenance.
>
> Using these pre-generated packs beyond the stated evaluation terms is not advised. Always verify underlying model licenses when generating datasets for broader use.

# TQB++ Synthetically Generated Dataset Packs

This directory contains synthetically generated Question-Answering (QA) dataset packs, which are part of the Tiny QA Benchmark++ (TQB++) suite. These packs extend the core TQB concept to multiple languages and allow for custom generation based on specific topics and difficulty levels.

## Overview

The TQB++ synthetic dataset packs are generated using the TQB++ generator toolkit (see `tools/generator/README.md`). This toolkit leverages Large Language Models (LLMs) to produce QA items adhering to the TQB schema.

These packs enable:
*   **Multilingual Smoke Testing:** Quickly assess an LLM's consistency and capability on conceptually similar tasks across different languages.
*   **Customized Evaluations:** Generate datasets tailored to specific domains, topics, or difficulty levels not covered by the core set.
*   **On-Demand Benchmark Creation:** Mint new micro-benchmarks as needed for adaptive testing or evaluating specific model updates.

Reference: See Section 2.2 ("Multi-Lingual Extensions (TQB++)") and Section 3 ("Synthetic Data Generation Toolkit") of the TQB++ paper for more details.

## Available Packs

This directory may contain pre-built packs for various languages and topics. Examples mentioned in the TQB++ paper include packs for:

*   **Languages:** English (EN), French (FR), Spanish (ES), Portuguese (PT), German (DE), Chinese (ZH), Japanese (JA), Turkish (TR), Arabic (AR), Russian (RU).
*   **Configuration:** Packs typically contain a configurable number of items (e.g., 100 QA items, often structured as 10 categories × 10 questions per category, or smaller packs like 10, 20, 40 items for specific experiments).

Users can also generate their own custom packs using the provided toolkit.

## Data Format

Datasets are in JSON Lines (`.jsonl`) format. Each line is a JSON object. In addition to the core TQB fields (`text`, `label`, `metadata.context`, `tags: {category, difficulty}`), items in these synthetic packs include:

*   `id` (string): A unique identifier for the generated item.
*   `lang` (string): The language code of the item (e.g., `fr`, `ja`).
*   `sha256` (string): A SHA-256 hash of the generated item for provenance tracking and reproducibility.

Example item from `pack_fr_40.jsonl`:
```json
{"text":"Combien font 5 + 7 ?","label":"12","context":"5 + 7 = 12.","tags":{"category":"math","difficulty":"easy"},"id":"292402c2","lang":"fr","sha256":"762e734d...b8b6085"}
```

## Purpose of Multilingual Extensions

The aim is not necessarily to ensure identical human-perceived difficulty across languages but to evaluate a model's consistency and capability on conceptually similar tasks formulated in different linguistic contexts through a standardized generation process. This helps identify language-specific performance disparities.

## License

The synthetically generated dataset packs in this directory (`data/packs/`) are distributed under a custom **"Eval-Only, Non-Commercial, No-Derivatives"** license.

**Key Terms:**
*   Permission is granted to use, copy, and redistribute these files for the sole purpose of evaluating or benchmarking language-model systems for non-commercial research or internal testing.
*   Any other use (including training, fine-tuning, commercial redistribution, or inclusion in downstream datasets) is PROHIBITED without explicit written permission from the authors.

For the full license text, please refer to the `LICENCE.data_packs.md` file in the root of the TQB++ repository. The content of this license is:
```
# Tiny QA Benchmark++ synthetic packs
Copyright © 2025 Vincent Koc

Permission is hereby granted to use, copy, and redistribute the enclosed JSON files (/data/packs AND /paper/evaluation) for the sole purpose of evaluating or benchmarking language-model systems **for non-commercial research or internal testing**.

Any other use – including but not limited to training, fine-tuning, commercial redistribution, or inclusion in downstream datasets – is PROHIBITED without explicit written permission from the authors.

The data were generated with the assistance of third-party language models (OpenAI GPT-o3).  No warranty is provided.
```

Users must adhere to these terms when using the datasets in this directory.
