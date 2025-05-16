#!/usr/bin/env python
"""
convert_tqb_to_evals_format.py - Converts TQB++ datasets to OpenAI Evals JSONL format.

Copyright (C) 2025 Vincent KOC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authors:
    Vincent Koc <vincentkoc@ieee.org>
"""

import json
import sys
import os

def convert_entry(original_entry):
    """Converts a single TQB++ entry to OpenAI Evals format."""
    question = original_entry.get("text", "")
    answer = original_entry.get("label", "")
    return {
        "input": [{"role": "user", "content": question}],
        "ideal": answer
    }

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_tqb_to_evals_format.py <input_file_path> <output_file_path>")
        print("  <input_file_path>: Path to the source TQB++ JSON or JSONL file.")
        print("  <output_file_path>: Path for the converted OpenAI Evals JSONL file.")
        print("Example (for a JSON array file like core_en.json or pack_ru_40.json):")
        print("  python intergrations/openai-evals/convert_tqb_to_evals_format.py data/core_en/core_en.json data/core_en/core_en.eval.jsonl")
        print("Example (for a JSONL file like other packs):")
        print("  python intergrations/openai-evals/convert_tqb_to_evals_format.py data/packs/pack_en_10.json data/packs/pack_en_10.eval.jsonl")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    print(f"Attempting to convert '{input_file_path}' to '{output_file_path}'...")

    if not os.path.exists(input_file_path):
        print(f"ERROR: Input file not found at '{input_file_path}'. Please ensure the path is correct.")
        sys.exit(1)

    converted_count = 0
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            # Try to read as a single JSON array first
            try:
                content = infile.read()
                data_array = json.loads(content)
                if isinstance(data_array, list):
                    print(f"Detected JSON array format in '{input_file_path}'. Processing as a list of items.")
                    for original_entry in data_array:
                        if isinstance(original_entry, dict):
                            eval_entry = convert_entry(original_entry)
                            json.dump(eval_entry, outfile)
                            outfile.write('\n')
                            converted_count += 1
                        else:
                            print(f"Warning: Skipping non-dictionary item in JSON array: {original_entry}")
                    print(f"Successfully processed {converted_count} items from JSON array.")
                    return # Exit after processing JSON array
                else:
                    # If json.loads succeeded but it's not a list, it might be a single JSON object (JSONL-like)
                    # Reset file pointer and fall through to JSONL processing.
                    print(f"File '{input_file_path}' is a single JSON object not in an array. Attempting to process as JSONL.")
                    infile.seek(0) # Reset file pointer to read line by line
            except json.JSONDecodeError:
                # If JSON array parsing fails, assume JSONL format
                print(f"Could not parse '{input_file_path}' as a single JSON array. Assuming JSONL format (one JSON object per line).")
                infile.seek(0) # Reset file pointer

            # Process as JSONL (line by line)
            print(f"Processing '{input_file_path}' as JSONL (one JSON object per line).")
            for line_num, line in enumerate(infile):
                line_content = line.strip()
                if not line_content:  # Skip empty lines
                    continue
                try:
                    original_entry = json.loads(line_content)
                    eval_entry = convert_entry(original_entry)
                    json.dump(eval_entry, outfile)
                    outfile.write('\n')
                    converted_count += 1
                except json.JSONDecodeError as e:
                    print(f"Skipping line {line_num + 1} in '{input_file_path}' due to JSON decode error: {e}")
                except Exception as e:
                    print(f"Error processing line {line_num + 1} in '{input_file_path}': {e}")
            
        print(f"Successfully converted {converted_count} items from '{input_file_path}' to '{output_file_path}'.")
        if converted_count > 0:
             print(f"IMPORTANT: Remember to update the 'samples_jsonl' path in the corresponding YAML eval file to point to '{output_file_path}'.")

    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")

if __name__ == "__main__":
    main() 