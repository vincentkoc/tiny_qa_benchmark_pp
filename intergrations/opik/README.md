# Opik Integration Example

This cookbook example demonstrates how to integrate TinyQA Benchmark++ (TQB++) with [Opik](https://github.com/comet-ml/opik) to create datasets for LLM evaluation.

## Overview

The integration allows you to:
- Generate synthetic QA pairs using TQB++
- Automatically create and populate Opik datasets
- Track and manage your evaluation datasets through Opik's interface

## Prerequisites

Install the required packages:

```bash
pip install opik-optimizer tinyqabenchmarkpp
```

## Configuration

1. Set up your OpenAI API key:
   - Either set it as an environment variable: `export OPENAI_API_KEY=your_key_here`
   - Or enter it when prompted by the script

2. Configure Opik:
   - The script will automatically configure Opik if not already configured
   - Configuration files are stored in `~/.opik` and `~/.opik.config`

## Usage

Run the example script:

```bash
python opik-dataset.py
```

The script will:
1. Configure Opik
2. Generate synthetic QA pairs using TQB++
3. Create a new Opik dataset with the generated data
4. Provide a URL to access the dataset in Opik's interface

## Customization

You can modify the following parameters in `opik-dataset.py`:

```python
# TQB++ Configuration
TQB_MODEL = "openai/gpt-4o-mini"  # Model to use for generation
TQB_NUM_QUESTIONS = 10            # Number of questions to generate
TQB_LANGUAGES = "en"             # Language(s) for questions
TQB_CATEGORIES = "technology,health"  # Question categories
TQB_DIFFICULTY = "mixed"         # Question difficulty level
```

## Output Format

The generated dataset includes the following fields for each QA pair:
- `question`: The generated question
- `answer`: The correct answer
- `generated_context`: Supporting context for the answer
- `category`: Question category (e.g., technology, health)
- `difficulty`: Question difficulty level (easy, medium, hard)

## Accessing Your Dataset

After running the script, you'll receive a URL to access your dataset in Opik's interface. The dataset will be named following the pattern:
`tinyqab-nb-{categories}-{num_questions}`

## Integration Details

The integration uses:
- TQB++ for generating high-quality QA pairs
- Opik's Python SDK for dataset management
- LiteLLM for model interactions
- OpenAI's API for question generation

## Troubleshooting

If you encounter issues:
1. Ensure all required packages are installed
2. Verify your OpenAI API key is correctly set
3. Check that Opik is properly configured
4. Review the console output for detailed error messages
