'''
core.py - a core script for the tinyqa++ benchmark generator

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
'''
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import argparse
import sys

import litellm

_original_litellm_verbose_state = litellm.set_verbose
_original_litellm_suppress_debug_info = litellm.suppress_debug_info

def _configure_litellm_verbosity(verbose: bool):
    """Configures LiteLLM's verbosity."""
    litellm.set_verbose = True if verbose else False
    litellm.suppress_debug_info = False if verbose else True

def _restore_litellm_verbosity():
    """Restores LiteLLM's verbosity to original state."""
    litellm.set_verbose = _original_litellm_verbose_state
    litellm.suppress_debug_info = _original_litellm_suppress_debug_info

def _generate_single_batch(
    model: str,
    lang: str,
    n: int,
    category: str,
    diff: str,
    domain_ctx: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Generates a single batch of TinyQA items."""
    log_func: Callable[..., None] = print if verbose else lambda *args, **kwargs: None
    log_func(f"[INFO] Attempting to generate {n} items for lang='{lang}', category='{category}', difficulty='{diff}'")
    system_prompt = "You are a dataset generator that outputs JSON with keys: text, label, context, tags{category, difficulty[easy, medium, hard]}. The context MUST be a one-sentence fact that contains the label verbatim."
    if domain_ctx:
        system_prompt += f"\n\nDomain context: {domain_ctx}"

    prompt_content = f"Generate {n} {diff} questions in {lang} about {category}. Return a JSON list ONLY."
    few_shots = [{
            "text": "What is 2 + 2?",
            "label": "4",
            "context": "2 + 2 equals 4.",
            "tags": {"category": "math", "difficulty": "easy"},
        },
        {
            "text": "Who wrote '1984'?",
            "label": "George Orwell",
            "context": "The novel 1984 was written by George Orwell.",
            "tags": {"category": "literature", "difficulty": "easy"},
        },]

    _configure_litellm_verbosity(verbose)
    content: Optional[str] = None
    items: List[Dict[str, Any]] = []

    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(few_shots, ensure_ascii=False)},
                {"role": "user", "content": prompt_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            seed=seed,
        )
        content = resp.choices[0].message.content
        if verbose and content: log_func(f"[DEBUG] Raw model response content:\n{content}")

        if not content:
            if verbose: log_func("[INFO] Primary content field is empty. Checking provider_specific_fields for refusal...")
            provider_fields = resp.choices[0].message.provider_specific_fields
            if provider_fields and isinstance(provider_fields, dict):
                refusal_content = provider_fields.get("refusal")
                if isinstance(refusal_content, str) and refusal_content.strip():
                    if verbose: log_func(f"[INFO] Found content in provider_specific_fields.refusal:\n{refusal_content}")
                    cleaned_refusal_content = refusal_content.strip()
                    if cleaned_refusal_content.lower().startswith("json\n"): content = cleaned_refusal_content[5:].strip()
                    else: content = cleaned_refusal_content
                    if verbose and content != refusal_content.strip(): log_func(f"[INFO] Using cleaned refusal content:\n{content}")
                elif verbose: log_func("[WARN] provider_specific_fields.refusal is empty or not a string.")
            elif verbose: log_func("[WARN] provider_specific_fields not found or not a dict.")

        if not content:
            if verbose: log_func(f"[WARN] Model returned empty content, skipping batch. Full response object:\n{resp}")
            return []

        try:
            parsed_json: Any = json.loads(content)
            if isinstance(parsed_json, dict):
                if "questions" in parsed_json and isinstance(parsed_json["questions"], list):
                    items = parsed_json["questions"]
                elif "data" in parsed_json and isinstance(parsed_json["data"], list):
                    items = parsed_json["data"]
                elif all(k in parsed_json for k in ("text", "label", "context", "tags")):
                    if verbose: log_func("[INFO] Model returned a single JSON object that looks like an item. Treating as a single-item list.")
                    items = [parsed_json]
                else:
                    if verbose: log_func("[WARN] Model returned JSON dict but not in expected format (e.g. missing 'questions' list or not a single item structure), skipping batch.")
                    return []
            elif isinstance(parsed_json, list): items = parsed_json
            else:
                if verbose: log_func("[WARN] Model returned JSON but not in expected format (not a list or dict), skipping batch.")
                return []
        except json.JSONDecodeError:
            if verbose: log_func(f"[WARN] Model returned non-JSON ('{content[:100]}...'), skipping batch.")
            return []

    except Exception as e:
        if verbose: log_func(f"[ERROR] Error during litellm.completion or response processing: {e}")
        return []
    finally: _restore_litellm_verbosity()

    processed: List[Dict[str, Any]] = []
    for obj in items:
        if not isinstance(obj, dict) or not all(k in obj for k in ("text", "label", "context", "tags")):
            if verbose: log_func(f"[WARN] Skipping item due to missing keys: {obj}")
            continue
        if not isinstance(obj.get("tags"), dict) or not all(
            k in obj["tags"] for k in ("category", "difficulty")
        ):
            if verbose: log_func(f"[WARN] Skipping item due to malformed tags: {obj}")
            continue
        
        obj_copy = obj.copy()
        obj_copy["context"] = str(obj_copy.get("context", ""))
        blob_content_obj = {
            "text": obj_copy["text"],
            "label": obj_copy["label"],
            "context": obj_copy["context"],
            "tags": obj_copy["tags"]
        }
        blob = json.dumps(blob_content_obj, ensure_ascii=False, sort_keys=True)
        obj_copy["sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        obj_copy["id"] = str(uuid.uuid4())[:8]
        obj_copy["lang"] = lang
        processed.append(obj_copy)

    if verbose: log_func(f"[INFO] Successfully processed {len(processed)} items for {lang}.")
    return processed

def generate_qa_pairs(
    model: str,
    languages: Union[str, List[str]],
    num_total_items: int,
    categories: str,
    difficulty: str,
    domain_ctx: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
    use_opik: bool = False,
) -> Union[List[Dict[str, Any]], None]:
    """
    Generates TinyQA++ items. 
    If output_path is provided, saves as a pretty-printed JSON array to the file.
    If output_path is None, returns a list of item dictionaries.

    Args:
        model: Name of the model to use (e.g., "openai/gpt-4o-mini").
        languages: Comma-separated string or list of language codes (e.g., "en,fr").
        num_total_items: Total number of items to generate.
        categories: Category of questions (e.g., "math", "history", "mixed").
        difficulty: Difficulty of questions (e.g., "easy", "medium", "hard", "mixed").
        domain_ctx: Optional domain-specific context to guide generation.
        temperature: Sampling temperature for the model.
        max_tokens: Maximum tokens for the model response.
        seed: Optional seed for reproducibility.
        output_path: Optional path to save the generated JSON file. If None, items are returned as a list of dictionaries.
        verbose: If True, print progress and debug information. LiteLLM verbosity is also controlled.
        use_opik: If True, enable OpikLogger for LiteLLM.

    Returns:
        A list of item dictionaries if output_path is None, otherwise None (file is written).
    """
    log_func: Callable[..., None] = print if verbose else lambda *args, **kwargs: None

    if use_opik:
        try:
            from litellm.integrations.opik.opik import OpikLogger
            if not any(isinstance(cb, OpikLogger) for cb in litellm.callbacks):
                 litellm.callbacks.append(OpikLogger())
        except ImportError:
            if verbose: log_func("[WARN] OpikLogger requested but litellm.integrations.opik could not be imported.")

    lang_list_str: List[str]
    if isinstance(languages, str):
        lang_list_str = [l.strip() for l in languages.split(",") if l.strip()]
    elif isinstance(languages, list):
        lang_list_str = [str(l).strip() for l in languages if str(l).strip()]
    else:
        if verbose: log_func("[ERROR] 'languages' must be a comma-separated string or a list of strings.")
        raise ValueError("'languages' must be a comma-separated string or a list of strings.")

    if not lang_list_str:
        if verbose: log_func("[WARN] No languages specified, nothing to generate.")
        return json.dumps([], indent=2, ensure_ascii=False) if not output_path else None

    num_per_lang = max(1, num_total_items // len(lang_list_str))
    if num_total_items % len(lang_list_str) != 0 and verbose:
        log_func(f"[INFO] num_total_items ({num_total_items}) not evenly divisible by number of languages ({len(lang_list_str)}). "
              f"Will generate approximately {num_per_lang} per language.")

    all_items: List[Dict[str, Any]] = []
    for lang_code in lang_list_str:
        if verbose: log_func(f"▶ Generating for language: {lang_code} ({num_per_lang} items)")
        batch_items = _generate_single_batch(
            model=model,
            lang=lang_code,
            n=num_per_lang,
            category=categories,
            diff=difficulty,
            domain_ctx=domain_ctx,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            verbose=verbose,
        )
        all_items.extend(batch_items)
        if len(lang_list_str) > 1 and len(all_items) < num_total_items: time.sleep(0.5)

    if output_path:
        json_output = json.dumps(all_items, indent=2, ensure_ascii=False)
        try:
            out_p = Path(output_path)
            out_p.parent.mkdir(exist_ok=True, parents=True)
            out_p.write_text(json_output, encoding="utf-8")
            if verbose: log_func(f"✅  {len(all_items)} items generated and saved to {out_p} (JSON format)")
            return None
        except IOError as e:
            if verbose: log_func(f"[ERROR] Could not write to output file {output_path}: {e}")
            if not verbose: log_func(f"Failed to write to {output_path}")
            return None # Explicitly return None on error as well
    else:
        if verbose: log_func(f"✅  {len(all_items)} items generated. Returning as list of dicts.")
        return all_items

def main_cli() -> None:
    p = argparse.ArgumentParser(
        description="TinyQA++ synthetic generator. Generates QA pairs using LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--num", type=int, default=10, help="Total items to generate.")
    p.add_argument("--languages", default="en", help="Comma-separated language codes (e.g., 'en,fr').")
    p.add_argument("--categories", default="mixed", help="Category/topic for questions.")
    p.add_argument("--difficulty", default="mixed", help="Difficulty level.")
    p.add_argument("--model", default="openai/gpt-4o-mini", help="LLM model (via LiteLLM).")
    out_group = p.add_mutually_exclusive_group()
    out_group.add_argument("--output-file", help="Path to save JSON. Default: tinyqa_generated_<timestamp>.json if not --str-output.")
    out_group.add_argument("--str-output", action="store_true", help="Output JSONL to stdout; suppresses most logs.")
    p.add_argument("--seed", type=int, help="Random seed for generation.")
    p.add_argument("--context", default="", help="Optional domain context string.")
    p.add_argument("--temperature", type=float, default=0.7, help="Model sampling temperature.")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for model response.")
    p.add_argument("--opik", action="store_true", help="Enable OpikLogger for LiteLLM (if available).")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging from generator. Overridden by --str-output.")
    p.add_argument("--quiet-cli", action="store_true", help="Suppress CLI informational messages (e.g., default filename).")
    args = p.parse_args()

    cli_log_func: Callable[..., None] = (lambda *a, **kw: None) if args.quiet_cli or args.str_output else print
    generator_verbose = args.verbose and not args.str_output
        
    actual_output_path: Optional[str] = None
    if args.str_output: pass 
    elif args.output_file: actual_output_path = args.output_file
    else:
        actual_output_path = f"tinyqa_generated_{int(time.time())}.json"
        cli_log_func(f"[CLI INFO] No output file specified. Defaulting to: {actual_output_path}")
    try:
        if args.str_output:
            items_list = generate_qa_pairs(
                model=args.model, languages=args.languages, num_total_items=args.num,
                categories=args.categories, difficulty=args.difficulty, domain_ctx=args.context,
                temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
                output_path=None, verbose=generator_verbose, use_opik=args.opik
            )
            if items_list is not None:
                jsonl_output = "\n".join([json.dumps(item, ensure_ascii=False) for item in items_list])
                print(jsonl_output)
        else: # Writing to a file (actual_output_path will be set)
            generate_qa_pairs(
                model=args.model, languages=args.languages, num_total_items=args.num,
                categories=args.categories, difficulty=args.difficulty, domain_ctx=args.context,
                temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
                output_path=actual_output_path, verbose=generator_verbose, use_opik=args.opik
            )
            if actual_output_path and not generator_verbose: 
                cli_log_func(f"[CLI INFO] Items saved to {actual_output_path}")

    except ValueError as e: 
        print(f"[CLI ERROR] Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: 
        print(f"[CLI ERROR] An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main_cli() 