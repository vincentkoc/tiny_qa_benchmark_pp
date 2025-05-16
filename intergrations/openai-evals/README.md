# OpenAI Evals for Tiny QA Benchmark++ (TQB++)

This directory contains YAML configuration files for running OpenAI Evals on the Tiny QA Benchmark++ (TQB++) datasets. These evals primarily use the `evals.elsuite.basic.match:Match` class, which checks for exact matches between the model's output and the expected answer.

## Prerequisites

1.  **Python Environment**: Ensure you have a working Python environment (Python 3.9+ recommended).
2.  **OpenAI Evals Library**: Install the library:
    ```bash
    pip install openai-evals
    ```
3.  **OpenAI API Key**: Configure your OpenAI API key as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_api_key_here"
    ```
4.  **Repository Root**: All commands and script executions mentioned below should be performed from the root directory of the `tinyqa-benchmark-pp` repository to ensure file paths are resolved correctly.

## Crucial: Data Preparation

The `evals.elsuite.basic.match:Match` class (used in the provided YAML files) expects the `samples_jsonl` data to be in a specific JSONL format. Each line must be a JSON object containing:
*   `"input"`: A list of messages, typically `[{"role": "user", "content": "YOUR_QUESTION_HERE"}]`
*   `"ideal"`: The expected answer string, e.g., `"YOUR_ANSWER_HERE"`

The TQB++ source datasets require conversion to this format.

### 1. Core Dataset (`core_en`)

The `intergrations/openai-evals/core_en.yaml` file points to `data/core_en/core_en.jsonl`.
However, the original core dataset is `data/core_en/core_en.json`, which is a single JSON array. This file needs to be converted into the required JSONL format.

**Script to convert `core_en.json` to `core_en.jsonl`:**
Save the following Python script (e.g., as `convert_core_en.py` in the root of this repository) and run it:

```python
import json

# Path to the original JSON file
original_file_path = "data/core_en/core_en.json"
# Path for the new JSONL output file
output_jsonl_path = "data/core_en/core_en.jsonl"

print(f"Attempting to convert '{original_file_path}' to '{output_jsonl_path}'...")

try:
    with open(original_file_path, 'r', encoding='utf-8') as f:
        data_array = json.load(f)

    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for entry in data_array:
            eval_entry = {
                "input": [{"role": "user", "content": entry.get("text", "")}],
                "ideal": entry.get("label", "")
            }
            json.dump(eval_entry, outfile)
            outfile.write('\n')
    print(f"Successfully converted '{original_file_path}' to '{output_jsonl_path}'")
except FileNotFoundError:
    print(f"ERROR: Original file not found at '{original_file_path}'. Make sure the path is correct and you are running the script from the repository root.")
except Exception as e:
    print(f"An error occurred: {e}")

```
Run from the repository root: `python convert_core_en.py`

### 2. Pack Datasets (`data/packs/*.json`)

The YAML files for the data packs (e.g., `intergrations/openai-evals/pack_en_10.yaml`) point to their respective `.json` files in the `data/packs/` directory (e.g., `data/packs/pack_en_10.json`).

As per the main TQB++ `README_hg.md`, these pack files are expected to be in JSONL format (one JSON object per line), but with `text` (question) and `label` (answer) fields, not the `input`/`ideal` structure required by `evals.elsuite.basic.match:Match`.

**Therefore, each pack file you intend to use must also be converted.**

**Generic script to convert pack JSONL files:**
Save the following Python script (e.g., as `convert_pack_jsonl.py` in the root of this repository):

```python
import json
import sys
import os

# Usage: python convert_pack_jsonl.py <input_pack_file.json> <output_pack_file.eval.jsonl>
# Example: python convert_pack_jsonl.py data/packs/pack_en_10.json data/packs/pack_en_10.eval.jsonl

if len(sys.argv) != 3:
    print("Usage: python convert_pack_jsonl.py <input_file.json> <output_file.eval.jsonl>")
    print("Ensure paths are relative to the repository root.")
    sys.exit(1)

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

print(f"Attempting to convert '{input_file_path}' to '{output_file_path}'...")

if not os.path.exists(input_file_path):
    print(f"ERROR: Input file not found at '{input_file_path}'.")
    sys.exit(1)

try:
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(infile):
            line_content = line.strip()
            if not line_content: # Skip empty lines
                continue
            try:
                original_entry = json.loads(line_content)
                eval_entry = {
                    "input": [{"role": "user", "content": original_entry.get("text", "")}],
                    "ideal": original_entry.get("label", "")
                }
                json.dump(eval_entry, outfile)
                outfile.write('\n')
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_num + 1} in '{input_file_path}' due to JSON decode error: {e}")
            except AttributeError as e: # Handles cases where entry might not be a dict or .get fails
                 print(f"Skipping line {line_num + 1} in '{input_file_path}' due to missing 'text' or 'label' key (or not a dict): {e}")

    print(f"Successfully converted '{input_file_path}' to '{output_file_path}'")
    print(f"IMPORTANT: You must now update the 'samples_jsonl' path in the corresponding YAML eval file to point to '{output_file_path}'.")
except Exception as e:
    print(f"An error occurred during conversion: {e}")

```

**Steps for each pack:**

1.  **Convert**: For each pack file (e.g., `data/packs/pack_en_10.json`), run the conversion script:
    ```bash
    python convert_pack_jsonl.py data/packs/pack_en_10.json data/packs/pack_en_10.eval.jsonl
    ```
    This will create a new file, e.g., `data/packs/pack_en_10.eval.jsonl`, in the required format.
2.  **Update YAML**: **Manually edit** the corresponding YAML file in the `intergrations/openai-evals/` directory (e.g., `intergrations/openai-evals/pack_en_10.yaml`) and change the `samples_jsonl` path to point to the newly created `.eval.jsonl` file.
    For example, change:
    `samples_jsonl: data/packs/pack_en_10.json`
    to:
    `samples_jsonl: data/packs/pack_en_10.eval.jsonl`

Repeat these two steps for every pack dataset you wish to evaluate.

## Available Evaluations

This directory contains the following YAML configuration files. Ensure the data preparation steps above have been completed for the datasets you intend to use.

*   `core_en.yaml` (evaluates `data/core_en/core_en.jsonl` after conversion)
*   `pack_ar_40.yaml`
*   `pack_de_40.yaml`
*   `pack_en_10.yaml`
*   `pack_en_20.yaml`
*   `pack_en_30.yaml`
*   `pack_en_40.yaml`
*   `pack_es_40.yaml`
*   `pack_fr_40.yaml`
*   `pack_ja_40.yaml`
*   `pack_ko_40.yaml`
*   `pack_pt_40.yaml` (Note: points to `data/packs/pack_pt_40.json.json` by default)
*   `pack_tr_40.yaml`
*   `pack_zh_cn_40.yaml`
*   `pack_zh_hant_40.yaml`
*   `sup_ancientlang_en_10.yaml`
*   `sup_medicine_en_10.yaml`

*(For pack files, remember to update their `samples_jsonl` path in the YAML after conversion, as described in "Data Preparation")*

## How to Run an Evaluation

Once the data is prepared and YAML files are correctly pointing to the converted `.jsonl` files:

1.  Navigate to the root of the `tinyqa-benchmark-pp` repository in your terminal.
2.  Use the `oaieval` command:
    ```bash
    oaieval <model_name> <eval_name>
    ```
    *   `<model_name>`: The OpenAI model you want to evaluate (e.g., `gpt-3.5-turbo`, `gpt-4`).
    *   `<eval_name>`: The name of the evaluation as defined in the YAML file (this is the key before the colon, e.g., `core_en` for `core_en.yaml`, `pack_en_10` for `pack_en_10.yaml`). The `oaieval` command will look for the YAML definition in the `intergrations/openai-evals/` directory if you add it to the registry or specify the path.
    Alternatively, you can often run by specifying the path to the yaml file directly, though standard practice is to register them. For simplicity, if `oaieval` is run from the root and your current directory is set up correctly, or if these YAML files are placed in a location where `oaieval` can find them (e.g. `evals/registry/evals/`), it can pick them up by name. If you encounter issues, you might need to register the directory or specify full paths.

    **Example:**
    ```bash
    # Ensure data/core_en/core_en.jsonl exists and is correctly formatted
    oaieval gpt-3.5-turbo core_en --registry_path intergrations/openai-evals
    ```
    Or for a pack (assuming `intergrations/openai-evals/pack_en_10.yaml` has been updated to point to `pack_en_10.eval.jsonl`):
    ```bash
    # Ensure data/packs/pack_en_10.eval.jsonl exists and is correctly formatted
    oaieval gpt-3.5-turbo pack_en_10 --registry_path intergrations/openai-evals
    ```
    Using `--registry_path` tells `oaieval` where to look for your custom eval definitions.

## Notes

*   **`pack_pt_40.json.json`**: The dataset file `data/packs/pack_pt_40.json.json` has a double `.json` extension. The `intergrations/openai-evals/pack_pt_40.yaml` file correctly points to this. You may want to rename the data file to `pack_pt_40.json` for consistency and update the YAML accordingly (before running the conversion script on it).
*   **JSONL Format Strictness**: Ensure the converted `.jsonl` files are strictly one valid JSON object per line. Any deviation can cause parsing errors in `oaieval`. The provided scripts aim to produce this format.
*   **Custom Eval Logic**: If you prefer not to convert the data files, you would need to write custom eval classes for OpenAI Evals that can directly parse the original `text`/`label` format. This is a more advanced approach.

