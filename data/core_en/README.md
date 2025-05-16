# TQB Core English Dataset (`core_en`)

<!-- SPDX-License-Identifier: Apache-2.0 -->

This directory contains the human-curated core English dataset of the Tiny QA Benchmark (TQB), which forms the gold standard foundation for TQB++.

## Overview

The `core_en` dataset consists of 52 hand-crafted English Question-Answering (QA) pairs. These pairs cover general knowledge domains such as geography, history, science (physics, biology, chemistry), mathematics (basic arithmetic, calculus), technology (computer science), literature, art, logic puzzles, and temporal/calendar trivia.

Reference: See Section 2.1 ("Human-Curated Core (TQB)") of the TQB++ paper for a detailed description.

## Purpose

*   **Rapid CI/CD Validation:** Serves as a minimal English QA set for quick checks in continuous integration / continuous deployment pipelines.
*   **Prompt Debugging:** Helps in identifying issues during prompt engineering by providing a stable baseline.
*   **Basic Regression Testing:** Acts as a canary for basic regressions or integration errors in LLM systems.
*   **Immutable Gold Standard:** Remains unchanged to ensure deterministic regression detection.

## Dataset Characteristics

*   **Size:** 52 QA items.
*   **File Size:** Approximately <20KB (e.g., `core_en.jsonl`).
*   **Format:** JSON Lines (`.jsonl`). Each line is a JSON object with the following fields:
    *   `text` (string): The question prompt (e.g., "What is the capital of France?").
    *   `label` (string): The gold answer (e.g., "Paris").
    *   `metadata.context` (string): A one-sentence factual statement supporting the answer (e.g., "France is a country in Europe. Its capital is Paris.").
    *   `tags.category` (string): A broad category for the question (e.g., "geography", "math").
    *   `tags.difficulty` (string): A rough difficulty level ("easy" or "medium").
*   **Answers:** Concise (mostly single words, numbers, or short phrases).
*   **Clarity:** No ambiguous prompts or trick questions; every answer is unique in the given context.
*   **Difficulty:** About two-thirds easy, one-third medium. No items are labeled "hard."

## How to Load

The dataset can be easily loaded in Python:

*   **Using Hugging Face `datasets` library:**
    ```python
    from datasets import load_dataset

    # Load the core English TQB dataset
    # This assumes the dataset is structured with a configuration for 'core_en'
    core_dataset = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name="core_en", split="train")

    # Accessing the data
    for example in core_dataset:
        print(f"Question: {example['text']}")
        print(f"Answer: {example['label']}")
        break # Print first example
    ```
*   **Plain JSON Parsing:**
    ```python
    # import json
    # data = []
    # with open("data/core_en/core_en.jsonl", "r") as f:
    #     for line in f:
    #         data.append(json.loads(line))
    ```

## License

This `core_en` dataset is licensed under the Apache License 2.0. See the main `LICENSE` file in the root of the repository for the full terms.
