#!/usr/bin/env python
'''
eval.py - a core script for the tinyqa++ benchmark evaluator

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

from __future__ import annotations
import argparse, json, time, re, random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import litellm
from Levenshtein import distance as lev_dist
from tqdm import tqdm
import unicodedata

def _norm(text: str) -> str:
    # Ensure text is a string
    text = str(text)
    # Apply NFKC normalization to handle compatibility characters and compose them.
    # This can help with variations in accents, ligatures, etc.
    text = unicodedata.normalize('NFKC', text)
    # For case-insensitive comparison where it applies
    text = text.lower()
    # Remove common English articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # More conservative punctuation and symbol removal. 
    # This will keep most characters from non-Latin scripts.
    # It removes common punctuation like commas, periods, question marks, exclamation marks, quotes, parentheses, etc.
    # It aims to preserve alphanumeric characters from various scripts as much as possible by not having a restrictive [^a-z0-9\s] class.
    text = re.sub(r'[.,?!"\'(){}\[\]:;\\-_+=<>@#$%^&*~`|\\\\/]', " ", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

def set_global_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"🎲  Global seed set to {seed}")

def lev_ratio(a: str, b: str) -> float:
    """1 – (Levenshtein / max_len)."""
    if not (a and b):
        return 0.0
    return 1.0 - lev_dist(a, b) / max(len(a), len(b))

def eval_model(
        model: str,
        dataset: List[dict[str, Any]],
        temperature: float = 0.7,
        seed: int = 42,
    ) -> Dict[str, Any]:
    rows = []
    print(f"Evaluating model: {model} on {len(dataset)} items...")
    for i, row in tqdm(enumerate(dataset, 1), total=len(dataset), desc=f"Evaluating {model}"):
        item_id = row.get("id", f"item_{i}")
        print(f"  Processing item {i}/{len(dataset)} (ID: {item_id})...")
        q, gold = row["text"], row["label"]
        start = time.perf_counter()

        system_prompt_instruction = "Identify the factual response to this question. Your answer should consist of a single item, like a date or title, without elaboration or conversational elements."

        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt_instruction},
                {"role": "user", "content": q}
            ],
            max_tokens=100,
            seed=seed,
            temperature=temperature,
        )
        latency = time.perf_counter() - start
        pred = resp.choices[0].message.content.strip()

        em = int(_norm(pred) == _norm(gold))
        lev = lev_ratio(_norm(pred), _norm(gold))
        item_result = {
            "id": item_id,
            "question": q,
            "gold": gold,
            "pred": pred,
            "em": em,
            "lev_ratio": round(lev, 3),
            "latency_s": round(latency, 3),
        }
        if "tags" in row and isinstance(row["tags"], dict):
            item_result["tags"] = row["tags"]
        if "lang" in row and isinstance(row["lang"], str):
            item_result["lang"] = row["lang"]
        
        rows.append(item_result)

    df = pd.DataFrame(rows)
    
    # Attempt to determine overall dataset language from the first item if available
    # or from dataset filename convention if eval_batch.py provides it somehow.
    # For now, let's assume all items in a dataset share the same lang if present.
    dataset_lang = None
    if not df.empty and "lang" in df.columns and df["lang"].nunique() == 1:
        dataset_lang = df["lang"].iloc[0]
    elif not df.empty and "lang" in df.columns:
        print(f"[WARN] Multiple languages found within dataset for model {model}. Top-level lang field will be None.")

    return {
        "model": model,
        "n": len(df),
        "dataset_language": dataset_lang,
        "accuracy_em": df["em"].mean(),
        "accuracy_lev>=0.75": (df["lev_ratio"] >= 0.75).mean(),
        "latency_p50": df["latency_s"].median(),
        "detail": df.to_dict(orient="records"),
    }


# ── CLI ──────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="TinyQA++ evaluator")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opik", action="store_true")
    ap.add_argument("--temperature", type=float, default=0)
    args = ap.parse_args()

    try:
        import asyncio 
        from litellm.integrations.opik.opik import OpikLogger
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            print("[INFO] OpikLogger: No running asyncio event loop. Opik may not function as expected if it relies on an active loop for background tasks initiated in __init__.")
        litellm.callbacks = [OpikLogger()]
        print("[INFO] OpikLogger enabled.")
    except RuntimeError as e:
        print(f"[WARN] Failed to initialize OpikLogger with --opik flag: {e}. Opik logging might be disabled.")
    except ImportError:
        print("[WARN] OpikLogger or asyncio could not be imported. Opik logging disabled.")
    
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    set_global_seed(args.seed)
    litellm.enable_cache()

    result = eval_model(args.model, data)
    out_path = Path(args.out) if args.out else Path(f"eval_{args.model}.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"🔹 {args.model}: EM={result['accuracy_em']*100:.1f}% | "
        f"Lev≥0.75={result['accuracy_lev>=0.75']*100:.1f}%"
    )
    print(f"✅  saved → {out_path}")


if __name__ == "__main__":
    main()
