<!-- SPDX-License-Identifier: Apache-2.0 OR CC BY 4.0 OR other -->
<div align="center"><b><a href="README.md">English</a> | <a href="README_zh.md">简体中文 (Coming Soon)</a> | <a href="README_ja.md">日本語</a> | <a href="README_es.md">Español</a> | <a href="README_fr.md">Français</a></b></div>

<h1 align="center" style="border: none">
    <div style="border: none">
        <!-- If you have a logo, you can add it here. Example:
        <a href="YOUR_PROJECT_LINK"><picture>
            <source media="(prefers-color-scheme: dark)" srcset="PATH_TO_DARK_LOGO.svg">
            <source media="(prefers-color-scheme: light)" srcset="PATH_TO_LIGHT_LOGO.svg">
            <img alt="Project Logo" src="PATH_TO_LIGHT_LOGO.svg" width="200" />
        </picture></a>
        <br>
        -->
        Tiny QA Benchmark++ (TQB++)
    </div>
</h1>

<p align="center">
An ultra-lightweight evaluation dataset and synthetic generator <br>to expose critical LLM failures in seconds, ideal for CI/CD and LLMOps.
</p>

<div align="center">
    <a href="https://pypi.org/project/tinyqabenchmarkpp/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/tinyqabenchmarkpp"></a>
    <a href="https://github.com/vincentkoc/tiny_qa_benchmark_pp/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/vincentkoc/tiny_qa_benchmark_pp"></a>
    <a href="https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp"><img alt="Hugging Face Dataset" src="https://img.shields.io/badge/🤗%20Dataset-Tiny%20QA%20Benchmark%2B%2B-blue"></a>
    <!-- Consider adding a GitHub Actions workflow badge if you have CI configured -->
    <!-- e.g., <a href="YOUR_WORKFLOW_LINK"><img alt="Build Status" src="YOUR_WORKFLOW_BADGE_SVG_LINK"></a> -->
</div>

<p align="center">
    <a href="https://github.com/vincentkoc/tiny_qa_benchmark_pp"><b>GitHub</b></a> •
    <a href="https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp"><b>Hugging Face Dataset</b></a> •
    <!-- Link to paper once available -->
    <!-- <a href="#"><b>Paper (Link Coming Soon)</b></a> • -->
    <a href="https://pypi.org/project/tinyqabenchmarkpp/"><b>PyPI</b></a>
</p>

<hr>
<!-- Optional: If you have a project thumbnail image, you can add it here -->
<!-- <p align="center"><img alt="TQB++ Thumbnail" src="path/to/your/thumbnail.png" width="700"></p> -->

**Tiny QA Benchmark++ (TQB++)** is an ultra-lightweight evaluation suite and python package designed to expose critical failures in Large Language Model (LLM) systems within seconds. It serves as the LLM software unit tests, ideal for rapid CI/CD checks, prompt engineering, and continuous quality assurance in modern LLMOps to be used alongside existing LLM evaluation tooling such as [Opik](https://github.com/comet-ml/opik/).

This repository contains the official implementation and synthetic datasets for the paper: *Tiny QA Benchmark++: Micro Gold Dataset with Synthetic Multilingual Generation for Rapid LLMOps Smoke Tests*.

**Paper:** (Details and link to preprint will be provided here once published)

- **Hugging Face Hub:** [datasets/vincentkoc/tiny_qa_benchmark_pp](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark_pp)
- **GitHub Repository:** [vincentkoc/tiny_qa_benchmark_pp](https://github.com/vincentkoc/tiny_qa_benchmark_pp)

## Key Features

*   **Immutable Gold Standard Core:** A 52-item hand-crafted English Question-Answering (QA) dataset (`core_en`) for deterministic regression testing from early [datasets/vincentkoc/tiny_qa_benchmark](https://huggingface.co/datasets/vincentkoc/tiny_qa_benchmark).
*   **Synthetic Customisation Toolkit:** A Python script (`tools/generator`) using LiteLLM to generate bespoke micro-benchmarks on demand for any language, topic, or difficulty.
*   **Standardised Metadata:** Artefacts packaged in Croissant JSON-LD format (`metadata/`) for discoverability and auto-loading by tools and search engines.
*   **Open Science:** All code (generator, evaluation scripts) and the core English dataset are released under the Apache-2.0 license. Synthetically generated data packs have a custom evaluation-only license.
*   **LLMOps Alignment:** Designed for easy integration into CI/CD pipelines, prompt-engineering workflows, cross-lingual drift detection, and observability dashboards.
*   **Multilingual Packs:** Pre-built packs for languages including English, French, Spanish, Portuguese, German, Chinese, Japanese, Turkish, Arabic, and Russian.

## Installation

The TQB++ toolkit, including the dataset generator and evaluation utilities, can be installed as a Python package from PyPI.

### Generating Synthetic Datasets (python package)

```bash
pip install tinyqabenchmarkpp
```

(Note: Ensure you have Python 3.8+ and pip installed. The exact package name on PyPI might vary; please check the official project links if this command doesn't work.)

Once installed, you should be able to use the generator and evaluation scripts from your command line or import functionalities into your Python projects.

## Loading Datasets with Hugging Face `datasets`

The TQB++ datasets are available on the Hugging Face Hub and can be easily loaded using the `datasets` library. This is the recommended way to access the data.

```python
from datasets import load_dataset, get_dataset_config_names

# Discover available dataset configurations (e.g., core_en, pack_fr_40, etc.)
configs = get_dataset_config_names("vincentkoc/tiny_qa_benchmark_pp")
print(f"Available configurations: {configs}")

# Load the core English dataset (assuming 'core_en' is a configuration)
if "core_en" in configs:
    core_dataset = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name="core_en", split="train")
    print(f"\nLoaded {len(core_dataset)} examples from core_en:")
    # print(core_dataset[0]) # Print the first example
else:
    print("\n'core_en' configuration not found.")

# Load a specific synthetic pack (e.g., a French pack)
# Replace 'pack_fr_40' with an actual configuration name from the `configs` list
example_pack_name = "pack_fr_40" # or another valid config name
if example_pack_name in configs:
    synthetic_pack = load_dataset("vincentkoc/tiny_qa_benchmark_pp", name=example_pack_name, split="train")
    print(f"\nLoaded {len(synthetic_pack)} examples from {example_pack_name}:")
    # print(synthetic_pack[0]) # Print the first example
else:
    print(f"\n'{example_pack_name}' configuration not found. Please choose from available configurations.")

```

For more detailed information on the datasets, including their structure and specific licenses, please see the README files within the `data/` directory (i.e., `data/README.md`, `data/core_en/README.md`, and `data/packs/README.md`).

## Repository Structure

*   `data/`: Contains the QA datasets.
    *   `core_en/`: The original 52-item human-curated English core dataset.
    *   `packs/`: Synthetically generated multilingual and topical dataset packs.
*   `tools/`: Contains scripts for dataset generation and evaluation.
    *   `generator/`: The synthetic QA dataset generator.
    *   `eval/`: Scripts and utilities for evaluating models against TQB++ datasets.
*   `paper/`: The LaTeX source and associated files for the research paper.
*   `metadata/`: Croissant JSON-LD metadata files for the datasets.
*   `LICENSE`: Main license for the codebase (Apache-2.0).
*   `LICENCE.data_packs.md`: Custom license for synthetically generated data packs.
*   `LICENCE.paper.md`: License for the paper content.

## Usage Scenarios

TQB++ is designed for various LLMOps and evaluation workflows:

*   **CI/CD Pipeline Testing:** Use as a unit test with LLM testing tooling for LLM services to catch regressions.
*   **Prompt Engineering & Agent Development:** Get rapid feedback when iterating on prompts or agent designs.
*   **Evaluation Harness Integration:** Encode as an OpenAI Evals YAML or Opik dataset for dashboard tracking.
*   **Cross-Lingual Drift Detection:** Monitor localization regressions using multilingual TQB++ packs.
*   **Adaptive Testing:** Synthesize new micro-benchmarks on-the-fly tailored to specific features or data drifts.
*   **Monitoring Fine-tuning Dynamics:** Track knowledge erosion or unintended capability changes during fine-tuning.

## Citation

If you use TQB++ in your research or work, please cite the original TQB and the TQB++ paper:

```bibtex
% This synthetic dataset and generator
@misc{koctinyqabenchmark_pp_dataset,
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

<!-- % Placeholder for TQB++ paper citation - will be updated upon publication
@article{koc2025tqb_pp,
  author       = {Vincent Koc},
  title        = {Tiny QA Benchmark$^{++}$: Micro Gold Dataset with Synthetic Multilingual Generation for Rapid LLMOps Smoke Tests},
  journal      = {Journal of Machine Learning Research (pending)},
  year         = {2025},
  volume       = {XX},
  number       = {X},
  pages        = {X-XX},
  url          = {http://jmlr.org/papers/vXX/koc25a.html} % Example URL
} -->

## License
The code in this repository (including the generator and evaluation scripts) and the `data/core_en` dataset and anything else not mentioned with a licence are licensed under the Apache License 2.0. See the `LICENSE` file for details.

The synthetically generated dataset packs in `data/packs/` are distributed under a custom "Eval-Only, Non-Commercial, No-Derivatives" license. See `LICENCE.data_packs.md` for details.

The Croissant JSON-LD metadata files in `metadata/` are available under CC0-1.0.

The paper content in `paper/` is subject to its own license terms, detailed in `LICENCE.paper.md`.
