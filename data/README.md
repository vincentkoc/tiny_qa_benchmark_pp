# Tiny QA Benchmark++ Datasets

This directory contains all datasets associated with the Tiny QA Benchmark++ (TQB++) project.

## Overview

The TQB++ datasets are designed for ultra-lightweight evaluation of Large Language Models (LLMs), focusing on rapid smoke testing and regression detection. They include a hand-curated English core set and synthetically generated multilingual and topical packs.

Reference: See Section 2 ("Dataset Structure and Schema") of the TQB++ paper for a detailed description of the datasets.

## Subdirectories

*   **`core_en/`**: Contains the human-curated core English TQB dataset. This dataset consists of 52 question-answer pairs covering general knowledge and is intended as an immutable gold standard for regression testing.
    *   See `data/core_en/README.md` for more details.
*   **`packs/`**: Contains synthetically generated dataset packs. These packs extend TQB to multiple languages and can be generated for various topics and difficulty levels using the TQB++ generator toolkit.
    *   See `data/packs/README.md` for more details.

## Metadata

All datasets are accompanied by Croissant JSON-LD metadata files, located in the `metadata/` directory at the root of the repository. This metadata facilitates discovery and interoperability with tools and search engines.

## Data Format

All QA datasets are provided in JSON Lines (`.jsonl`) format. Each line is a JSON object with the following core fields:

*   `text` (string): The question prompt.
*   `label` (string): The gold answer.
*   `metadata.context` (string): A one-sentence factual statement supporting the answer.
*   `tags.category` (string): A broad category for the question.
*   `tags.difficulty` (string): A rough difficulty level (`easy`, `medium`, `hard`).

Synthetically generated packs may include additional fields such as `id`, `lang`, and `sha256` for provenance.

## How to Load with Hugging Face `datasets`

The datasets, including the core English set and various synthetic packs, can be loaded using the Hugging Face `datasets` library. You will typically specify the dataset name (`vincentkoc/tiny_qa_benchmark_pp`) and the desired configuration name (e.g., `core_en`, `pack_fr_40`).

```python
from datasets import load_dataset

# Example: Load the core English dataset
core_en_dataset = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name="core_en", split="train")
print(f"Loaded {len(core_en_dataset)} examples from core_en.")

# Example: Load a synthetic French pack (assuming a configuration named 'pack_fr_40' exists)
# Replace 'pack_fr_40' with the actual name of the pack/configuration if different.
try:
    french_pack = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name="pack_fr_40", split="train")
    print(f"Loaded {len(french_pack)} examples from pack_fr_40.")
    # Accessing data from a pack
    # for example in french_pack:
    #     print(example)
    #     break
except Exception as e:
    print(f"Could not load 'pack_fr_40': {e}. Check available configurations on Hugging Face Hub.")

# To see all available configurations for this dataset on Hugging Face Hub:
# from datasets import get_dataset_config_names
# configs = get_dataset_config_names("vincentkoc/tiny_qa_benchmark_pp")
# print(f"Available configurations: {configs}")
```

For plain JSON parsing, refer to the example in `data/core_en/README.md`.

## Licensing

Licensing varies by dataset component:

*   **`data/core_en/` (Human-Curated Core English Dataset):**
    *   Licensed under the Apache License 2.0.
    *   See the main `LICENSE` file in the root of the repository for details.
*   **`data/packs/` (Synthetically Generated Dataset Packs):**
    *   Distributed under a custom "Eval-Only, Non-Commercial, No-Derivatives" license.
    *   See the `LICENCE.data_packs.md` file in the root of the repository for the full terms.
    *   The content of this license is:
        ```
        # Tiny QA Benchmark++ synthetic packs
        Copyright © 2025 Vincent Koc

        Permission is hereby granted to use, copy, and redistribute the enclosed JSON files (/data/packs AND /paper/evaluation) for the sole purpose of evaluating or benchmarking language-model systems **for non-commercial research or internal testing**.

        Any other use – including but not limited to training, fine-tuning, commercial redistribution, or inclusion in downstream datasets – is PROHIBITED without explicit written permission from the authors.

        The data were generated with the assistance of third-party language models (OpenAI GPT-o3).  No warranty is provided.
        ```
*   **Croissant Metadata (`metadata/` directory):**
    *   Available under CC0-1.0.

Users should ensure they comply with the respective licenses for each dataset component they use.
