---
license: other
license_name: eval-only-nc-nd
license_link: >-
  https://github.com/vincentkoc/tiny_qa_benchmark_pp/blob/main/LICENCE.data_packs.md
task_categories:
- question-answering
task_ids:
- extractive-qa
- closed-book-qa
language:
- en
- de
- ar
- ko
- fr
- pt
- zh
- ja
- es
- tr
tags:
- synthetic
- qa
- evaluation
- benchmark
- llmops
- smoke-test
pretty_name: Tiny QA Benchmark++
size_categories:
- n<1K
---

# Tiny QA Benchmark++ (TQB++)

**Tiny QA Benchmark++ (TQB++)** is an ultra-lightweight evaluation suite designed to expose critical failures in Large Language Model (LLM) systems within seconds. It serves as the LLM analogue of software unit tests, ideal for rapid CI/CD checks, prompt engineering, and continuous quality assurance in modern LLMOps.

This Hugging Face dataset repository hosts the core English dataset and various synthetically generated multilingual and topical dataset packs associated with TQB++.

**Main GitHub Repository:** [vincentkoc/tiny_qa_benchmark_pp](https://github.com/vincentkoc/tiny_qa_benchmark_pp)
**Paper:** (Details and link to preprint will be provided here once published on the main GitHub)

## Dataset Overview

TQB++ provides two main types of datasets:

1.  **`core_en` (Human-Curated Core English Dataset):**
    *   A 52-item hand-crafted English Question-Answering (QA) dataset.
    *   Serves as an immutable gold standard for deterministic regression testing.
    *   Covers general knowledge (geography, history, science, math, literature, etc.).
    *   Licensed under Apache-2.0.

2.  **Synthetically Generated `packs`:**
    *   Multilingual and topical micro-benchmarks (e.g., `pack_fr_40`, `pack_en_science_10`).
    *   Generated using the [TQB++ generator toolkit](https://github.com/vincentkoc/tiny_qa_benchmark_pp/tree/main/tools/generator) (Python script using LiteLLM).
    *   Enable on-demand creation of datasets for any language, topic, or difficulty.
    *   These pre-generated packs are provided for **evaluation and demonstration purposes only** under a custom [Eval-Only, Non-Commercial, No-Derivatives license](https://github.com/vincentkoc/tiny_qa_benchmark_pp/blob/main/LICENCE.data_packs.md). Users are **strongly encouraged to generate their own packs** for broader use cases to ensure alignment with their chosen LLM's terms of service.

## How to Load Datasets

You can load any dataset configuration (e.g., `core_en` or a specific pack like `pack_fr_40`) using the `datasets` library:

```python
from datasets import load_dataset, get_dataset_config_names

# Discover all available dataset configurations in this repository
configs = get_dataset_config_names("vincentkoc/tiny_qa_benchmark_pp")
print(f"Available configurations: {configs}")

# Example: Load the core English dataset
if "core_en" in configs:
    core_en_dataset = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name="core_en", split="train")
    print(f"\nLoaded {len(core_en_dataset)} examples from core_en:")
    # print(core_en_dataset[0]) # Print the first example
else:
    print("\n'core_en' configuration not found.")

# Example: Load a specific synthetic pack (replace with a valid config name from `configs`)
example_pack_name = "pack_fr_40" # Make sure this configuration exists
if example_pack_name in configs:
    synthetic_pack = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name=example_pack_name, split="train")
    print(f"\nLoaded {len(synthetic_pack)} examples from {example_pack_name}:")
    # print(synthetic_pack[0]) # Print the first example
else:
    print(f"\n'{example_pack_name}' configuration not found. Please choose from available configurations.")
```

## Data Format

All datasets are in JSON Lines (`.jsonl`) format. Each line is a JSON object with fields including:

*   `text` (string): The question prompt.
*   `label` (string): The gold answer.
*   `metadata.context` (string): A one-sentence factual statement supporting the answer.
*   `tags.category` (string): A broad category for the question.
*   `tags.difficulty` (string): A rough difficulty level (`easy`, `medium`, `hard`).

Synthetically generated packs also include `id`, `lang` (language code), and `sha256` (for provenance).

## Use Cases

*   **Rapid CI/CD Checks:** Integrate as a quick smoke test in LLM deployment pipelines.
*   **Prompt Engineering:** Get immediate feedback on prompt changes.
*   **Cross-Lingual Drift Detection:** Monitor performance consistency across languages.
*   **Targeted Evaluations:** Use or generate packs for specific domains/topics of interest.

## Licensing

*   The `core_en` dataset and all code (generator, evaluation scripts) in the [main repository](https://github.com/vincentkoc/tiny_qa_benchmark_pp) are licensed under Apache-2.0.
*   The **pre-generated synthetic dataset packs** available here are distributed under a custom [Eval-Only, Non-Commercial, No-Derivatives license](https://github.com/vincentkoc/tiny_qa_benchmark_pp/blob/main/LICENCE.data_packs.md). Please see the license file for full terms.
*   This dataset card (README.md with YAML frontmatter) and other Croissant metadata files are available under CC0-1.0.

## Citation

If you use TQB++ datasets or the generator toolkit in your research or work, please cite:

```bibtex
% This synthetic dataset and generator
@misc{koctinyqabenchmarkpp,
  author       = {Vincent Koc},
  title        = {Tiny QA Benchmark++ (TQB++) Datasets and Toolkit},
  year         = {2025},
  publisher    = {Hugging Face & GitHub},
  doi          = {10.57967/hf/5531},
  howpublished = {\url{https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp}},
  note         = {See also: \url{https://github.com/vincentkoc/tiny_qa_benchmark_pp}}
}

% Original core_en.json (52 in en)
@misc{koctinyqabenchmark_original,
  author       = {Vincent Koc},
  title        = {tiny_qa_benchmark},
  year         = {2025},
  publisher    = {Hugging Face},
  journal      = {Hugging Face Hub},
  doi          = {10.57967/hf/5417},
  url          = {https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark}
}
``` 