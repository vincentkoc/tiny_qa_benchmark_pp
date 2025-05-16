# TQB++ Synthetic Data Generation Toolkit

<!-- SPDX-License-Identifier: Apache-2.0 -->

This directory contains the Python script and utilities for generating synthetic Question-Answering (QA) micro-benchmarks for the Tiny QA Benchmark++ (TQB++) suite.

## Overview

The generator toolkit is a core component of TQB++, enabling the creation of bespoke tiny QA datasets. It is implemented as a Python script (approximately 200 lines of core logic) that leverages the `LiteLLM` library for provider-agnostic calls to various Large Language Models (LLMs).

Reference: See Section 3 ("Synthetic Data Generation Toolkit") of the TQB++ paper for a detailed description.

## Features

*   **Provider Agnostic:** Uses LiteLLM to connect to a wide range of LLM providers (e.g., OpenAI, Anthropic, Cohere, Google, and any OpenAI-compatible API).
*   **Customizable Output:** Users can specify parameters to tailor the generated datasets:
    *   `--num`: Number of questions to generate.
    *   `--languages`: Comma-separated list of language codes (e.g., `en,fr,ja`).
    *   `--categories`: Topics/domains for the questions (e.g., `history,math,science`).
    *   `--difficulty`: Desired difficulty level (e.g., `easy,medium,hard`).
    *   `--provider`: The LLM endpoint/model to use for generation.
*   **Structured Output:** The generator is prompted to produce JSON formatted output adhering to the TQB schema (`text`, `label`, `context`, `tags: {category, difficulty}`).
*   **Few-Shot Prompting:** Includes few-shot exemplars in the prompt to guide the LLM on the desired format and content style.
*   **Schema Validation:** Performs basic validation of the generated JSON structure.
*   **Provenance Tracking:** Stamps each generated item with a SHA-256 hash for reproducibility and provenance.

## How to Run

The generator script (e.g., `generator.py` or by invoking the package `tinyqabenchmarkpp.generator`) can be run from the command line. 

**Important Note on Temperature:** When using OpenAI reasoning models for generation (e.g., `openai/o4-mini`), you need to use `temperature=1.0` as the script has no logic to handle this. Default is 0 to encourage reproduceability as detailed in the TQB++ paper (Appendix A.1).

Here are conceptual examples:

**1. Using a specific OpenAI model:**

```bash
python -m tinyqabenchmarkpp.generator \
    --num 10 \
    --languages "en" \
    --categories "science" \
    --difficulty "medium" \
    --provider "openai/gpt-3.5-turbo-0125" \
    --output_dir "./data/packs/science_en_10.jsonl" 
    # --temperature 1.0 # Explicitly set if needed, often a default
```

**2. Using an OpenRouter model:**

LiteLLM allows you to use models hosted on OpenRouter. You'll need to set your `OPENROUTER_API_KEY` environment variable.

```bash
# Ensure OPENROUTER_API_KEY is set in your environment
export OPENROUTER_API_KEY="your_openrouter_key_here"

python -m tinyqabenchmarkpp.generator \
    --num 15 \
    --languages "de" \
    --categories "history" \
    --difficulty "easy" \
    --provider "openrouter/google/gemma-7b-it" \
    --output_dir "./data/packs/history_de_15.jsonl"
```

**3. Using a local Ollama model:**

To use a model served locally via Ollama, ensure your Ollama server is running and the desired model is pulled (e.g., `ollama pull llama3`).

```bash
python -m tinyqabenchmarkpp.generator \
    --num 5 \
    --languages "es" \
    --categories "literature" \
    --difficulty "hard" \
    --provider "ollama/llama3" \
    --output_dir "./data/packs/literature_es_5_hard.jsonl" \
    --base_url "http://localhost:11434" # Specify your Ollama API base URL
```

(Note: Actual script name, package invocation, and parameters might vary slightly. Refer to the script's help message (`python -m tinyqabenchmarkpp.generator --help`) for precise usage and all available options, including how to pass API keys if not using environment variables.)

## Generation Process

1.  **System Prompt:** Instructs the LLM to output structured JSON according to the TQB schema.
2.  **Few-Shot Exemplars:** Provides 2 examples to the LLM.
3.  **User Prompt:** Specifies the number, language, category, and difficulty of questions required.
4.  **LLM Call:** Sends the request to the chosen LLM via LiteLLM.
5.  **Parsing & Validation:** Parses the LLM response, validates the JSON structure, and includes a retry mechanism (up to 3 attempts) for malformed outputs.
6.  **Hashing & Storage:** Stores a SHA-256 hash of each item and saves the output.

## License

This toolkit is part of the TQB++ project and is licensed under the Apache License 2.0. See the main `LICENSE` file in the root of the repository for details.
