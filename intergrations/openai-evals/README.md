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

Many TQB++ source datasets require conversion to this format.

### Data Conversion Script

A utility script `convert_tqb_to_evals_format.py` is provided in this directory (`intergrations/openai-evals/`) to help convert your TQB++ data files.

**Script Location:** `intergrations/openai-evals/convert_tqb_to_evals_format.py`

**Features:**
*   Converts data from the TQB++ `{"text": "question", "label": "answer", ...}` format to the OpenAI Evals `{"input": [{"role": "user", ...}], "ideal": ...}` format.
*   Autodetects if the input file is a single JSON array (e.g., `data/core_en/core_en.json`, `data/packs/pack_ru_40.json`) or a JSONL file where each line is a JSON object (e.g., other pack files if they are line-delimited).
*   Outputs a new JSONL file with the `.eval.jsonl` suffix.

**Usage:**
```bash
python intergrations/openai-evals/convert_tqb_to_evals_format.py <input_file_path> <output_file_path>
```

**Examples:**

1.  **Converting `core_en.json` (JSON array):**
    ```bash
    python intergrations/openai-evals/convert_tqb_to_evals_format.py data/core_en/core_en.json data/core_en/core_en.eval.jsonl
    ```
    Then, update `intergrations/openai-evals/core_en.yaml` to point `samples_jsonl` to `data/core_en/core_en.eval.jsonl`.
    *(Note: A pre-converted version `intergrations/openai-evals/core_en_open-ai-eval.jsonl` is also available. If using this, ensure `core_en.yaml` points to it: `samples_jsonl: intergrations/openai-evals/core_en_open-ai-eval.jsonl`)*

2.  **Converting a pack file (e.g., `pack_fr_40.json`, assuming it's JSONL or JSON array):**
    ```bash
    python intergrations/openai-evals/convert_tqb_to_evals_format.py data/packs/pack_fr_40.json data/packs/pack_fr_40.eval.jsonl
    ```
    Then, **manually edit** `intergrations/openai-evals/pack_fr_40.yaml` and change its `samples_jsonl` path to `data/packs/pack_fr_40.eval.jsonl`.

**Important:** After running the script for any dataset (core or pack), you **must** update the `samples_jsonl` field in the corresponding YAML configuration file (in `intergrations/openai-evals/`) to point to the newly created `.eval.jsonl` output file.

### 1. Core Dataset (`core_en`)

*   The original dataset is `data/core_en/core_en.json` (JSON array).
*   A pre-converted version is available at `intergrations/openai-evals/core_en_open-ai-eval.jsonl`.
*   The `intergrations/openai-evals/core_en.yaml` now points to this pre-converted file: `samples_jsonl: intergrations/openai-evals/core_en_open-ai-eval.jsonl`.
*   If you wish to convert `data/core_en/core_en.json` yourself, use the `convert_tqb_to_evals_format.py` script as shown in the example above and update `core_en.yaml` accordingly.

### 2. Pack Datasets (`data/packs/*.json`)

*   Source pack files are located in `data/packs/` (e.g., `pack_en_10.json`, `pack_ru_40.json`).
*   **These files need to be converted using `convert_tqb_to_evals_format.py` before they can be used with the provided YAML configurations.** Some packs might be JSON arrays (like `pack_ru_40.json`) while others might be JSONL. The script attempts to handle both.
*   **After conversion (e.g., creating `data/packs/pack_en_10.eval.jsonl` from `data/packs/pack_en_10.json`), you must manually update the `samples_jsonl` path in the corresponding YAML file (e.g., `intergrations/openai-evals/pack_en_10.yaml`) to point to the new `.eval.jsonl` file.**

## Available Evaluations

This directory contains the following YAML configuration files. Ensure the data preparation and YAML updates described above have been completed for the datasets you intend to use.

*   `core_en.yaml` (points to the pre-converted `intergrations/openai-evals/core_en_open-ai-eval.jsonl`)
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
*   `pack_pt_40.yaml`
*   `pack_ru_40.yaml` (Ensure `data/packs/pack_ru_40.json` is converted to `data/packs/pack_ru_40.eval.jsonl` and YAML is updated)
*   `pack_tr_40.yaml`
*   `pack_zh_cn_40.yaml`
*   `pack_zh_hant_40.yaml`
*   `sup_ancientlang_en_10.yaml`
*   `sup_medicine_en_10.yaml`

*(For all pack files, run the conversion script and update their `samples_jsonl` path in the YAML)*

## How to Run an Evaluation

Once the data is prepared and YAML files are correctly pointing to the converted `.eval.jsonl` files:

1.  Navigate to the root of the `tinyqa-benchmark-pp` repository in your terminal.
2.  Use the `oaieval` command:
    ```bash
    oaieval <model_name> <eval_name> --registry_path intergrations/openai-evals
    ```
    *   `<model_name>`: The OpenAI model you want to evaluate (e.g., `gpt-3.5-turbo`, `gpt-4`).
    *   `<eval_name>`: The name of the evaluation as defined in the YAML file (e.g., `core_en`, `pack_en_10`).
    *   `--registry_path intergrations/openai-evals`: Tells `oaieval` where to find your custom eval definitions.

    **Example (Core English):**
    ```bash
    # core_en.yaml points to the pre-converted intergrations/openai-evals/core_en_open-ai-eval.jsonl
    oaieval gpt-3.5-turbo core_en --registry_path intergrations/openai-evals
    ```
    **Example (Pack - after conversion and YAML update):**
    ```bash
    # Assuming data/packs/pack_fr_40.json was converted to data/packs/pack_fr_40.eval.jsonl
    # And intergrations/openai-evals/pack_fr_40.yaml was updated to point to it.
    oaieval gpt-3.5-turbo pack_fr_40 --registry_path intergrations/openai-evals
    ```

## Notes

*   **`pack_pt_40.json.json`**: If you have a file named `data/packs/pack_pt_40.json.json`, it's recommended to rename it to `data/packs/pack_pt_40.json` before running the conversion script. Then update `intergrations/openai-evals/pack_pt_40.yaml` to point to `data/packs/pack_pt_40.eval.jsonl` after conversion.
*   **JSONL Format Strictness**: Ensure the converted `.eval.jsonl` files are strictly one valid JSON object per line. The provided script aims to produce this format.
*   **`pack_pt_40.json.json`**: The dataset file `data/packs/pack_pt_40.json.json` has a double `.json` extension. The `intergrations/openai-evals/pack_pt_40.yaml` file correctly points to this. You may want to rename the data file to `pack_pt_40.json` for consistency and update the YAML accordingly (before running the conversion script on it).
*   **JSONL Format Strictness**: Ensure the converted `.jsonl` files are strictly one valid JSON object per line. Any deviation can cause parsing errors in `oaieval`. The provided scripts aim to produce this format.
*   **Custom Eval Logic**: If you prefer not to convert the data files, you would need to write custom eval classes for OpenAI Evals that can directly parse the original `text`/`label` format. This is a more advanced approach.

