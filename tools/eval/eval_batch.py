#!/usr/bin/env python
'''
eval_batch.py - a script for the tinyqa++ benchmark batch evaluation

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

from pathlib import Path
import json, subprocess, itertools, datetime
from datetime import timezone

DATASETS = {
    "core_en": "data/core_en.json",
    "en_10": "data/pack_en_10.json",
    "en_20": "data/pack_en_20.json",
    "en_30": "data/pack_en_30.json",
    "en_40": "data/pack_en_30.json",
    "tr_40": "data/pack_tr_40.json",
    "fr_40": "data/pack_fr_40.json",
    "ja_40": "data/pack_ja_40.json",
    "sup-ancientlang_en_10": "data/sup-ancientlang_en_10.json",
    "sup-medicine_en_10": "data/sup-medicine_en_10.json",
}

MODELS = {
    "mistral-24b-instruct": "openrouter/mistralai/mistral-small-3.1-24b-instruct",
    "mistral-7b-instruct": "openrouter/mistralai/mistral-7b-instruct",
    "ministral-8b": "openrouter/mistralai/ministral-8b",
    "ministral-3b": "openrouter/mistralai/ministral-3b",
    "llama-3.2-3b-instruct": "openrouter/meta-llama/llama-3.2-3b-instruct",
    "llama-3.2-1b-instruct": "openrouter/meta-llama/llama-3.2-1b-instruct",
    "gemma-3-12b": "openrouter/google/gemma-3-27b-it",
    "gemma-3-12b": "openrouter/google/gemma-3-12b-it",
    "gemma-3-4b": "openrouter/google/gemma-3-4b-it",
}

EVAL_SCRIPT = "eval.py"
OUT_DIR = Path("batch_results")
OUT_DIR.mkdir(exist_ok=True)

grid = list(itertools.product(DATASETS.items(), MODELS.items()))
print(f"🔁 Running {len(grid)} evaluations …")

for (ds_name, ds_path), (mdl_name, mdl_id) in grid:
    out_file = OUT_DIR / f"{ds_name}__{mdl_name}.json"
    cmd = [
        "python", EVAL_SCRIPT,
        "--dataset", ds_path,
        "--model", mdl_id,
        "--out", str(out_file)
    ]

    if out_file.exists():
        print(f"⏩ Skipping task, result file already exists: {out_file}")
        continue

    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)

records = []
for f in OUT_DIR.glob("*.json"):
    if f.name.startswith("summary_"):
        print(f"⏩ Skipping summary file from aggregation: {f.name}")
        continue
    
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
        records.append(
            {
                "dataset": f.stem.split("__")[0],
                "model": obj["model"],
                "em": obj["accuracy_em"],
                "lev≥0.75": obj["accuracy_lev>=0.75"],
            }
        )
    except json.JSONDecodeError:
        print(f"[WARN] Could not decode JSON from result file for summary: {f.name}")
    except TypeError as e:
        print(f"[WARN] Unexpected data structure in result file {f.name} (expected dict, got list?): {e}. Skipping.")
    except KeyError as e:
        print(f"[WARN] Missing expected key in result file {f.name}: {e}. Skipping.")

summary_path = OUT_DIR / f"summary_{datetime.datetime.now(timezone.utc).isoformat()}.json"
summary_path.write_text(json.dumps(records, indent=2))
print(f"\n📊  Summary written to {summary_path}")
