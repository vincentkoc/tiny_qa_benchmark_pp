# TQB++ Evaluation Utilities

<!-- SPDX-License-Identifier: Apache-2.0 -->

This directory contains scripts and utilities for evaluating Large Language Models (LLMs) against the Tiny QA Benchmark++ (TQB++) datasets.

## Overview

The evaluation tools are designed to assess model performance on TQB++ datasets, providing metrics and insights into model capabilities, particularly for smoke testing and regression detection.

Reference: See Section 6 ("Experimental Setup"), Section 7 ("Results and Discussion"), and Appendix A.2 ("Evaluation Protocol") of the TQB++ paper for details on the evaluation methodology and metrics.

## Features

*   **Metrics:** Supports standard evaluation metrics for Question-Answering tasks:
    *   **Exact Match (EM) Accuracy:** Measures the percentage of predictions that exactly match the gold answers after normalization (lowercase, removal of articles, punctuation, extra whitespace).
    *   **Levenshtein Ratio (LR) Accuracy:** Measures accuracy based on a normalized Levenshtein distance, allowing for minor variations in answers. A typical threshold (e.g., >= 0.75 or an empirically calibrated one like 0.95) is used to determine a match.
*   **Batch Evaluation:** Scripts (e.g., `eval_py` or `eval_batch_py` mentioned in the paper's appendix) facilitate running evaluations across multiple models and datasets.
*   **Deterministic Evaluation:** For model testing, evaluation scripts typically set a low temperature (e.g., 0.0) and a consistent seed for reproducible outputs from the LLMs under evaluation.

## How to Run Evaluations

Evaluation is typically performed using a script that takes a model endpoint, a dataset path, and other parameters. Conceptually:

```bash
python tools/eval/eval_py \
    --model_provider "litellm/openrouter/google/gemma-2-9b-it" \
    --dataset_path "./data/core_en/core_en.jsonl" \
    --output_results_path "./results/gemma2_core_en_results.jsonl" \
    --metrics "EM,LR" \
    --lr_threshold 0.95
```

(Note: Actual script name and parameters might vary. Refer to the specific script's help message or documentation for precise usage.)

## Interpreting Results

The primary goal of TQB++ is to act as a rapid smoke test. High accuracy (e.g., >90-95% EM on `core_en`) is generally expected from capable models. Significant drops in performance can indicate regressions or issues with the LLM setup.

Performance variations across different languages, categories, or difficulty levels (as provided by TQB++ packs and tags) can offer insights into specific model strengths or weaknesses.

## License

These evaluation utilities are part of the TQB++ project and are licensed under the Apache License 2.0. See the main `LICENSE` file in the root of the repository for details.
