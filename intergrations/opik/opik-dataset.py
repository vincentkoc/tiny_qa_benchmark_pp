import os
import getpass
import subprocess
import json
import opik
import tinyqabenchmarkpp
from opik.evaluation.metrics import LevenshteinRatio
from typing import Optional
import sys

# Ensure API keys are set (interactive input if not found)
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# tinyqabenchmarkpp Configuration
TQB_MODEL = "openai/gpt-4o-mini"
TQB_NUM_QUESTIONS = 10
TQB_LANGUAGES = "en"
TQB_CATEGORIES = "technology,health"
TQB_DIFFICULTY = "mixed"


def configure_opik_session():
    """Configures Opik for the session."""
    print("Attempting to configure Opik...")
    try:
        opik.configure()
        print("Opik configured successfully.")
    except Exception as e:
        print(f"Error configuring Opik: {e}")
        print("Please ensure your Comet API key is correctly set up.")
        raise

def generate_synthetic_qa_data():
    """Generates synthetic QA data using tinyqabenchmarkpp."""
    print(f"Generating {TQB_NUM_QUESTIONS} synthetic QA pairs using tinyqabenchmarkpp...")
    command = [
        "python", "-m", "tinyqabenchmarkpp.generate",
        "--num", str(TQB_NUM_QUESTIONS),
        "--languages", TQB_LANGUAGES,
        "--categories", TQB_CATEGORIES,
        "--difficulty", TQB_DIFFICULTY,
        "--model", TQB_MODEL,
        "--str-output",
    ]

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print("--- tinyqabenchmarkpp Output ---")
        print(process.stdout)
        if process.stderr:
            print("--- tinyqabenchmarkpp Errors ---")
            print(process.stderr)
        print("Synthetic data generated successfully")
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running tinyqabenchmarkpp generator: {e}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        raise
    except FileNotFoundError:
        print("Error: 'python -m tinyqabenchmarkpp.generator' command not found.")
        print("Please ensure tinyqabenchmarkpp is installed and in your PATH.")
        print("Install with: pip install tinyqabenchmarkpp")
        raise

def load_data_to_opik_dataset(data_str: str) -> Optional[opik.Dataset]:
    """Loads the generated JSONL data into an Opik Dataset."""
    print("Loading synthetic data into Opik Dataset...")
    items = []
    try:
        for line_num, line in enumerate(data_str.strip().split('\n')):
            try:
                data = json.loads(line.strip())
                if not isinstance(data, dict):
                    print(f"Skipping non-dictionary item in JSONL line {line_num + 1}: {line.strip()}")
                    continue
                item = {
                    "question": data.get("text"),
                    "answer": data.get("label"),
                    "generated_context": data.get("context"),
                    "category": data.get("tags", {}).get("category"),
                    "difficulty": data.get("tags", {}).get("difficulty")
                }
                if not item["question"] or not item["answer"]:
                    print(f"Skipping item in JSONL line {line_num + 1} due to missing question or answer: {data}")
                    continue
                items.append(item)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line {line_num + 1}: {line.strip()}")
    except Exception as e:
        print(f"An unexpected error occurred while processing data: {e}")
        return None

    if not items:
        print("No valid items found in the generated data.")
        return None

    print(f"Loaded {len(items)} items.")
    
    dataset_name_sanitized = f"tinyqab-nb-{TQB_CATEGORIES.replace(',', '_')}-{TQB_NUM_QUESTIONS}"
    dataset_name_sanitized = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in dataset_name_sanitized)
    
    try:
        # Create a new dataset using opik.Opik().create_dataset()
        opik_client = opik.Opik()
        dataset = opik_client.create_dataset(
            name=dataset_name_sanitized,
            description=f"Synthetic QA from tinyqabenchmarkpp for {TQB_CATEGORIES}"
        )
        
        # Add items to the dataset
        dataset.insert(items)
            
        print(f"Opik Dataset '{dataset.name}' created successfully with ID: {dataset.id}")
        return dataset
    except Exception as e:
        print(f"Error creating Opik Dataset: {e}")
        raise


def run_pipeline():
    print("Starting Opik and tinyqabenchmarkpp notebook-friendly script...")
    print("Ensure packages installed: pip install tinyqabenchmarkpp")
    print("-" * 40)

    print("\n--- Step 1: Configuring Opik ---")
    configure_opik_session()
    print("-" * 40)

    print("\n--- Step 2: Generating Synthetic Data ---")
    generated_data = None
    try:
        generated_data = generate_synthetic_qa_data()
    except Exception as e:
        print(f"Data generation failed: {e}. Exiting.")
        return
    print("-" * 40)

    print("\n--- Step 3: Loading Data to Opik Dataset ---")
    opik_synthetic_dataset = None
    if generated_data:
        try:
            opik_synthetic_dataset = load_data_to_opik_dataset(generated_data)
        except Exception as e:
            print(f"Opik Dataset creation failed: {e}. Exiting.")
            return
    if not opik_synthetic_dataset:
        print("Failed to create Opik dataset. Exiting.")
        return
    print("-" * 40)
    print("Script finished.")

if __name__ == "__main__":
    run_pipeline() 