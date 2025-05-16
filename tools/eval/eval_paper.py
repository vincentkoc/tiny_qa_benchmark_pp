#!/usr/bin/env python
'''
eval_paper.py - a script for the tinyqa++ benchmark paper evaluation

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
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import binomtest

# --- Configuration ---
RESULTS_DIR = Path("../../paper/evaluation_results/")
OUTPUT_DIR = Path("../../paper/evaluation_visuals/")
EM_METRIC_KEY = "accuracy_em"
LR_METRIC_KEY = "accuracy_lev>=0.75"
LATENCY_KEY = "latency_p50"
DATA_DIR_FOR_TAGS = Path("data")

# --- Font Sizes for Journal Quality ---
BASE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 13
SUBPLOT_TITLE_FONT_SIZE = 15
MAIN_TITLE_FONT_SIZE = 16

def load_evaluation_data(results_path: Path) -> tuple[list[dict], list[dict]]: # No longer needs source_tags_by_id
    """Loads all evaluation JSON files from the results directory.
    Returns two lists: one for aggregated records and one for detailed item records with tags from result files.
    """
    aggregated_records = []
    detailed_item_records = [] 

    if not results_path.is_dir():
        print(f"[ERROR] Results directory not found: {results_path}")
        return aggregated_records, detailed_item_records

    # Regex to capture model family, version/size, and type (e.g., instruct)
    # (family_name)-(version_or_size_with_unit)-(type)
    # e.g. llama-3.2-3b-instruct -> family=llama, version_size=3.2-3b, type=instruct
    # e.g. mistral-7b-instruct -> family=mistral, version_size=7b, type=instruct
    # e.g. gemma-3-12b-it -> family=gemma, version_size=3-12b, type=it
    model_regex = re.compile(r"^([a-zA-Z]+)-?([0-9.]*[0-9]+[bBmM]?)?-?([a-zA-Z]*)")

    for file_path in results_path.glob("*.json"):
        try:
            if file_path.name.startswith("summary_"):
                continue

            content = json.loads(file_path.read_text(encoding="utf-8"))
            
            parts = file_path.stem.split("__", 1)
            if len(parts) != 2:
                print(f"[WARN] Skipping file with unexpected name format: {file_path.name}")
                continue
            dataset_name = parts[0]
            model_alias = parts[1]

            model_family = model_alias 
            model_weight_numeric = 0.0 

            match = model_regex.match(model_alias)
            if match:
                base_name = match.group(1)
                size_str = match.group(2) 

                # Refine family name based on common patterns
                if base_name.lower() == "llama" and size_str and size_str.startswith("3.2"):
                    model_family = "llama-3.2"
                elif base_name.lower() == "gemma" and size_str and size_str.startswith("3"):
                    model_family = "gemma-3"
                elif base_name.lower() == "mistralai" or base_name.lower() == "mistral": # from openrouter/mistralai/mistral-*
                    model_family = "mistral"
                elif base_name.lower().startswith("ministral"):
                    model_family = "ministral"
                else:
                    model_family = base_name
                
                # Extract numeric weight for sorting (e.g., 7b -> 7, 24b -> 24, 3.2-3b -> 3)
                if size_str:
                    # Try to find a size like "3b", "7b", "12b"
                    weight_match = re.search(r'([0-9.]+)([bBmM])', size_str)
                    if weight_match:
                        num = float(weight_match.group(1))
                        unit = weight_match.group(2).lower()
                        if unit == 'b':
                            model_weight_numeric = num 
                        elif unit == 'm':
                            model_weight_numeric = num / 1000
                    else:
                        simple_num_match = re.search(r'(\d+)', size_str.split('-')[-1]) # take last part after potential version
                        if simple_num_match:
                            model_weight_numeric = float(simple_num_match.group(1))
            else:
                print(f"[INFO] Regex did not match for model_alias: {model_alias}. Using alias as family.")

            if EM_METRIC_KEY not in content or LR_METRIC_KEY not in content or LATENCY_KEY not in content or "n" not in content:
                print(f"[WARN] Skipping file missing required metrics (EM, LR, Latency, or n): {file_path.name}")
                continue

            aggregated_records.append({
                "dataset": dataset_name,
                "model_family": model_family,
                "model_alias": model_alias, 
                "model_weight_numeric": model_weight_numeric,
                "dataset_language": content.get("dataset_language"),
                "n": content.get("n"),
                EM_METRIC_KEY: content[EM_METRIC_KEY] * 100 if content.get(EM_METRIC_KEY) is not None else None,
                LR_METRIC_KEY: content[LR_METRIC_KEY] * 100 if content.get(LR_METRIC_KEY) is not None else None, # Scale to %
                LATENCY_KEY: content[LATENCY_KEY]
            })

            # Load detailed per-item results for LR threshold analysis and tag breakdown
            if "detail" in content and isinstance(content["detail"], list):
                for item_detail in content["detail"]:
                    if "em" in item_detail and "lev_ratio" in item_detail:
                        item_id_from_result = item_detail.get("id")
                        item_lang_from_detail = item_detail.get("lang") # Get lang from detail item
                        
                        record_to_add = {
                            "dataset": dataset_name,
                            "model_family": model_family,
                            "model_alias": model_alias,
                            "id": item_id_from_result,
                            "lang": item_lang_from_detail if item_lang_from_detail else content.get("dataset_language"), # Fallback to top-level lang
                            "em": item_detail["em"],
                            "lev_ratio": item_detail["lev_ratio"]
                        }
                        
                        # Extract tags if present in item_detail
                        if "tags" in item_detail and isinstance(item_detail["tags"], dict):
                            source_tags = item_detail["tags"]
                            for tag_key, tag_value in source_tags.items():
                                record_to_add[f"tag_{tag_key}"] = tag_value
                        
                        detailed_item_records.append(record_to_add)
            else:
                print(f"[WARN] No 'detail' section found or not a list in {file_path.name}, cannot perform detailed LR threshold analysis or tag breakdown for this file.")

        except json.JSONDecodeError:
            print(f"[WARN] Could not decode JSON from file: {file_path.name}")
        except Exception as e:
            print(f"[WARN] Could not process file {file_path.name}: {e}")
            
    print(f"Loaded {len(aggregated_records)} result files from {results_path}")
    return aggregated_records, detailed_item_records

def create_pivot_table(data_records: list[dict], value_key: str) -> pd.DataFrame:
    """Creates a pivot table (datasets x models) for a given metric, sorted by performance."""
    if not data_records:
        return pd.DataFrame()
    
    df = pd.DataFrame(data_records)

    # Calculate mean scores for sorting
    # Ensure value_key column is numeric for mean calculation
    df[value_key] = pd.to_numeric(df[value_key], errors='coerce')

    df['model_family_mean_score'] = df.groupby('model_family')[value_key].transform('mean')

    # Sort by family performance, then by model weight (desc), then by alias name (asc for tie-breaking)
    df_sorted = df.sort_values(
        by=['model_family_mean_score', 'model_family', 'model_weight_numeric', 'model_alias'],
        ascending=[False, True, False, True]
    )

    try:
        pivot_df = df_sorted.pivot_table(index=["model_family", "model_alias"], columns="dataset", values=value_key, sort=False)
        
        if 'n' in df_sorted.columns:
            dataset_sizes = df_sorted.groupby('dataset')[['n']].first().squeeze().sort_values()
            sorted_dataset_columns = dataset_sizes.index.tolist()
            
            final_columns_ordered = [col for col in sorted_dataset_columns if col in pivot_df.columns]
            for col in pivot_df.columns:
                if col not in final_columns_ordered:
                    final_columns_ordered.append(col)
            
            pivot_df = pivot_df[final_columns_ordered]
        else:
            print("[WARN] 'n' column not found in records, cannot sort datasets by size for the table/heatmap.")

        return pivot_df
    except KeyError as e:
        print(f"[ERROR] Could not create pivot table for metric {value_key}. Missing key: {e}. Available columns: {df.columns.tolist()}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Could not create pivot table for metric {value_key}: {e}")
        return pd.DataFrame()

def generate_and_save_heatmap(df: pd.DataFrame, title: str, output_filename: Path):
    """Generates and saves a heatmap from a DataFrame."""
    if df.empty:
        print(f"Skipping heatmap generation for '{title}' due to empty data.")
        return

    plt.figure(figsize=(max(12, len(df.columns) * 1.2), max(8, len(df.index) * 0.8))) 
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn", linewidths=.5, # Changed cmap to RdYlGn
                annot_kws={"size": BASE_FONT_SIZE - 2 if BASE_FONT_SIZE > 8 else 8}, 
                cbar_kws={'shrink': 0.80, 'aspect': 30}) 
    plt.title(title, fontsize=MAIN_TITLE_FONT_SIZE) 
    plt.xticks(rotation=45, ha="right", fontsize=BASE_FONT_SIZE)
    plt.ylabel(None)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=BASE_FONT_SIZE)

    new_yticklabels = []
    last_family = None
    group_separator_indices = []

    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
        for i, (family, alias) in enumerate(df.index):
            display_alias = alias
            if alias.startswith(family + '-'):
                display_alias = alias[len(family)+1:]
            elif alias.startswith(family): 
                 display_alias = alias[len(family):].lstrip('-')

            if family != last_family:
                separator = " - " if plt.rcParams['text.usetex'] else " \u25BA " # Use hyphen for LaTeX
                new_yticklabels.append(f"{family}{separator}{display_alias}") 
                if last_family is not None: 
                    group_separator_indices.append(i)
                last_family = family
            else:
                new_yticklabels.append(f"  {display_alias}") 
        ax.set_yticklabels(new_yticklabels, rotation=0, ha="right", fontsize=BASE_FONT_SIZE) 

        for idx in group_separator_indices:
            ax.axhline(idx, color='black', linewidth=1.5, linestyle='--')
    else:
        plt.yticks(rotation=0, fontsize=BASE_FONT_SIZE)

    plt.tight_layout(rect=[0, 0, 0.93, 0.97])
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Heatmap saved to: {output_filename}")
    except Exception as e:
        print(f"[ERROR] Could not save heatmap {output_filename}: {e}")
    plt.close()

def analyze_lr_thresholds(detailed_item_records: list[dict]) -> tuple[pd.DataFrame, float, float]:
    """Analyzes LR thresholds to find the one maximizing F1-score.
    Returns a DataFrame with metrics per threshold, max F1, and best threshold.
    """
    if not detailed_item_records:
        print("[WARN] No detailed item records available for LR threshold analysis.")
        return pd.DataFrame(), 0.0, 0.0

    df_items = pd.DataFrame(detailed_item_records)
    thresholds = np.arange(0.0, 1.01, 0.01)
    results = []
    n_bootstraps = 1000
    alpha = 0.05

    print(f"\n--- Calculating F1 scores with {n_bootstraps} bootstraps for LR threshold analysis (this may take a while)... ---")
    for lr_thresh in tqdm(thresholds, desc="Analyzing LR Thresholds"):
        bootstrap_f1_scores = []
        df_items['lr_pass_current_thresh'] = df_items['lev_ratio'] >= lr_thresh

        for i in range(n_bootstraps):
            sample_df = resample(df_items, replace=True, n_samples=len(df_items)) 
            
            tp = len(sample_df[(sample_df['em'] == 1) & sample_df['lr_pass_current_thresh']])
            fp = len(sample_df[(sample_df['em'] == 0) & sample_df['lr_pass_current_thresh']])
            fn = len(sample_df[(sample_df['em'] == 1) & ~sample_df['lr_pass_current_thresh']]) # Use ~ for NOT lr_pass

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            bootstrap_f1_scores.append(f1)
        
        f1_mean = np.mean(bootstrap_f1_scores)
        f1_lower = np.percentile(bootstrap_f1_scores, (alpha / 2) * 100)
        f1_upper = np.percentile(bootstrap_f1_scores, (1 - alpha / 2) * 100)
        
        results.append({
            "lr_threshold": lr_thresh,
            "f1_mean": f1_mean,
            "f1_lower_ci": f1_lower,
            "f1_upper_ci": f1_upper,
            # Retain single calculation for overall precision/recall for the threshold (optional)
            # "precision": np.mean([p for p in bootstrap_precisions if p is not None]), 
            # "recall": np.mean([r for r in bootstrap_recalls if r is not None])
        })
    
    df_analysis = pd.DataFrame(results)
    if df_analysis.empty:
        return pd.DataFrame(), 0.0, 0.0
        
    max_f1_row = df_analysis.loc[df_analysis['f1_mean'].idxmax()]
    best_threshold = max_f1_row['lr_threshold']
    max_f1 = max_f1_row['f1_mean']
    
    print(f"\n--- LR Threshold Analysis (Global) ---")
    print(f"Optimal LR Threshold (max F1-score): {best_threshold:.2f} (F1: {max_f1:.3f})")
    return df_analysis, best_threshold, max_f1

def plot_f1_vs_lr_threshold(df_analysis: pd.DataFrame, best_threshold: float, max_f1: float, output_filename: Path):
    """Plots F1-score vs. LR Threshold and Q-Q plot, and saves it."""
    if df_analysis.empty:
        print("Skipping F1 vs LR threshold plot due to empty analysis data.")
        return

    # Define colors
    highlight_orange = '#FF8C00'
    dark_grey = '#606060'
    ci_fill_color = '#BEBEBE'
    mid_grey_marker = '#A0A0A0'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) 
    fig.suptitle("F1 Score and Distribution Analysis", fontsize=MAIN_TITLE_FONT_SIZE, y=0.98) 

    # --- Plot 1: F1 vs LR Threshold ---
    ax1.plot(df_analysis['lr_threshold'], df_analysis['f1_mean'], marker='.', label='Mean F1 Score', color=dark_grey)
    ax1.fill_between(df_analysis['lr_threshold'], df_analysis['f1_lower_ci'], df_analysis['f1_upper_ci'], color=ci_fill_color, alpha=0.85, label='95% Confidence Interval') # Increased alpha
    ax1.axvline(best_threshold, color=highlight_orange, linestyle='--', linewidth=1.5, label=f'Optimal Threshold: {best_threshold:.2f} (F1={max_f1:.3f})') 
    ax1.text(best_threshold + 0.01, max_f1, f'{max_f1:.3f}', color=highlight_orange, va='bottom', ha='left', fontsize=LEGEND_FONT_SIZE)
    ax1.set_xlabel("Levenshtein Ratio (LR) Threshold", fontsize=BASE_FONT_SIZE)
    ax1.set_ylabel("F1-Score", fontsize=BASE_FONT_SIZE)
    ax1.set_title("F1-Score vs. LR Threshold", fontsize=SUBPLOT_TITLE_FONT_SIZE) 
    ax1.legend(fontsize=LEGEND_FONT_SIZE)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=BASE_FONT_SIZE)

    # --- Plot 2: Q-Q Plot for F1 Scores (Mean) ---
    f1_means = df_analysis['f1_mean'].dropna()
    if not f1_means.empty and len(f1_means) > 1: 
        sm.qqplot(f1_means, line='s', ax=ax2, fit=True) # Draw plot on ax2
        
        lines = ax2.get_lines()
        if len(lines) >= 2:
            plt.setp(lines[0], markerfacecolor=mid_grey_marker, markeredgecolor=mid_grey_marker, marker='.', linestyle='None')
            plt.setp(lines[1], color=highlight_orange, linestyle='-', linewidth=1.5)
        elif len(lines) == 1:
             plt.setp(lines[0], markerfacecolor=mid_grey_marker, markeredgecolor=mid_grey_marker, marker='.', color=highlight_orange)

        ax2.set_title("Q-Q Plot of Mean F1 Scores", fontsize=SUBPLOT_TITLE_FONT_SIZE) 
        ax2.set_xlabel("Theoretical Quantiles (Normal Distribution)", fontsize=BASE_FONT_SIZE)
        ax2.set_ylabel("Sample Quantiles (Mean F1 Scores)", fontsize=BASE_FONT_SIZE)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(axis='both', which='major', labelsize=BASE_FONT_SIZE)

    else:
        ax2.text(0.5, 0.5, "Not enough data for Q-Q plot", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=BASE_FONT_SIZE)
        ax2.set_title("Q-Q Plot of Mean F1 Scores", fontsize=SUBPLOT_TITLE_FONT_SIZE)
        ax2.tick_params(axis='both', which='major', labelsize=BASE_FONT_SIZE)


    plt.tight_layout(rect=[0, 0, 1, 0.94]) 
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"LR Threshold analysis plot saved to: {output_filename}")
    except Exception as e:
        print(f"[ERROR] Could not save LR threshold plot {output_filename}: {e}")
    plt.close()

def plot_performance_vs_latency(aggregated_records: list[dict], performance_metric_key: str, performance_metric_name: str, output_filename: Path):
    """Generates a scatter plot of performance vs. latency."""
    if not aggregated_records:
        print(f"Skipping {performance_metric_name} vs. Latency plot due to no aggregated records.")
        return

    df = pd.DataFrame(aggregated_records)
    if performance_metric_key not in df.columns or LATENCY_KEY not in df.columns:
        print(f"[WARN] Skipping {performance_metric_name} vs. Latency plot. Missing required columns: {performance_metric_key} or {LATENCY_KEY}.")
        return

    # Filter for English datasets
    # Prioritize 'dataset_language' == 'en', but also catch names like 'core_en' or 'sup-xxx_en_yy'
    df_eng = df[(
        df['dataset_language'].str.lower() == 'en' 
        ) | (
        df['dataset'].str.contains('_en', case=False)
        ) | (
        df['dataset'].str.lower() == 'core_en'
    )].copy()

    if df_eng.empty:
        print(f"Skipping {performance_metric_name} vs. Latency plot as no English dataset records were found.")
        return
    
    print(f"Filtered for English datasets. Found {len(df_eng)} records for the P50 Latency plot.")

    # Ensure latency is positive for log scale
    df_eng = df_eng[df_eng[LATENCY_KEY] > 0]

    if df_eng.empty:
        print(f"Skipping {performance_metric_name} vs. Latency plot as no English data remains after filtering for positive latency.")
        return

    plt.figure(figsize=(12, 8))

    scatter_plot = sns.scatterplot(
        data=df_eng,
        x=LATENCY_KEY,
        y=performance_metric_key,
        hue="model_family",
        size="model_weight_numeric", 
        sizes=(50, 600),
        style="dataset",      
        alpha=0.75,
        palette="viridis" 
    )
    
    df_eng['log_latency'] = np.log(df_eng[LATENCY_KEY])

    try:
        df_for_reg = df_eng[[performance_metric_key, 'log_latency']].dropna()
        if len(df_for_reg) > 1:
            sns.regplot(
                data=df_for_reg,
                x='log_latency', 
                y=performance_metric_key, 
                scatter=False, 
                ci=95, 
                line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2.5}, # Thicker line
                ax=scatter_plot 
            )
        else:
            print("[WARN] Not enough data points after filtering and NaN removal to draw regression line.")
    except Exception as e:
        print(f"[WARN] Could not add regression line for English data: {e}")

    scatter_plot.set_xscale('log') 
    plt.title(f'{performance_metric_name} (English Datasets) vs. P50 Latency (Log Scale)', fontsize=MAIN_TITLE_FONT_SIZE)
    plt.xlabel("P50 Latency (seconds, log scale)", fontsize=BASE_FONT_SIZE)
    plt.ylabel(f"{performance_metric_name} (%)", fontsize=BASE_FONT_SIZE)
    plt.grid(True, which="both", linestyle='--', alpha=0.7) 
    
    scatter_plot.tick_params(axis='both', which='major', labelsize=BASE_FONT_SIZE)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Legend", fontsize=LEGEND_FONT_SIZE)
    if scatter_plot.get_legend() is not None:
      plt.setp(scatter_plot.get_legend().get_title(), fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout(rect=[0, 0, 0.90, 0.97]) # Adjusted rect right margin
    
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Performance vs. Latency plot saved to: {output_filename}")
    except Exception as e:
        print(f"[ERROR] Could not save Performance vs. Latency plot {output_filename}: {e}")
    plt.close()

def generate_performance_breakdown_tables(detailed_item_records: list[dict], output_dir: Path):
    """Generates and prints tables for performance breakdown by item tags (e.g., difficulty, category)."""
    if not detailed_item_records:
        print("\nSkipping performance breakdown by tags as no detailed item records were loaded.")
        return

    df_items = pd.DataFrame(detailed_item_records)
    tag_columns = [col for col in df_items.columns if col.startswith("tag_")]

    if not tag_columns:
        print("\nNo tag columns (e.g., 'tag_difficulty') found in detailed records. Skipping tag breakdown.")
        return

    print("\n--- Performance Breakdown by Tags ---")

    for tag_col in tag_columns:
        tag_name = tag_col.replace("tag_", "").capitalize()
        print(f"\n-- Breakdown by {tag_name} --")
        
        current_df_em = pd.DataFrame()
        current_df_lr = pd.DataFrame()

        if df_items[tag_col].isnull().all():
            print(f"Tag column '{tag_col}' is empty or all NaN. Skipping breakdown for this tag.")
            continue

        try:
            breakdown_em = df_items.groupby(["model_family", "model_alias", tag_col])["em"].mean().mul(100).unstack(level=tag_col)
            breakdown_lr = df_items.groupby(["model_family", "model_alias", tag_col])["lev_ratio"].mean().mul(100).unstack(level=tag_col)

            heatmap_title_suffix = ""
            if tag_col == "tag_category":
                category_counts = df_items['tag_category'].value_counts()
                top_n = 10
                top_n_categories = category_counts.nlargest(top_n).index.tolist()
                
                valid_cols_em = [col for col in top_n_categories if col in breakdown_em.columns]
                breakdown_em = breakdown_em[valid_cols_em]
                
                valid_cols_lr = [col for col in top_n_categories if col in breakdown_lr.columns]
                breakdown_lr = breakdown_lr[valid_cols_lr]
                if len(valid_cols_em) < len(top_n_categories) or len(valid_cols_lr) < len(top_n_categories):
                    print(f"[INFO] Displaying top {min(len(valid_cols_em), len(valid_cols_lr))} categories due to data availability.")
                    heatmap_title_suffix = f" (Top {min(len(valid_cols_em), len(valid_cols_lr))} by Item Count)"
                else:
                    heatmap_title_suffix = f" (Top {top_n} by Item Count)"

            # Sort index like the main tables for consistency
            if not breakdown_em.empty:
                temp_em_df = breakdown_em.reset_index()
                tag_value_columns_em = list(breakdown_em.columns)
                
                family_means_for_sorting_em = temp_em_df.groupby('model_family')[tag_value_columns_em].mean().mean(axis=1)
                temp_em_df['model_family_mean_score'] = temp_em_df['model_family'].map(family_means_for_sorting_em)
                
                temp_em_df = temp_em_df.sort_values(by=['model_family_mean_score', 'model_family', 'model_alias'], ascending=[False, True, True])
                breakdown_em_sorted = temp_em_df.set_index(["model_family", "model_alias"]).drop(columns=['model_family_mean_score'])
                
                print(f"\nEM Scores by {tag_name}:")
                print(breakdown_em_sorted.to_string(float_format="%.1f"))
                em_csv_path = output_dir / f"breakdown_em_by_{tag_name.lower().replace(' ', '_')}.csv"
                breakdown_em_sorted.to_csv(em_csv_path, float_format="%.1f")
                print(f"EM breakdown by {tag_name} saved to: {em_csv_path}")
                em_heatmap_title = f"EM Scores by {tag_name}{heatmap_title_suffix if tag_col == 'tag_category' else ''}"
                generate_and_save_heatmap(breakdown_em_sorted, em_heatmap_title, output_dir / f"heatmap_em_by_{tag_name.lower().replace(' ', '_')}.png")

            if not breakdown_lr.empty:
                temp_lr_df = breakdown_lr.reset_index()
                tag_value_columns_lr = list(breakdown_lr.columns)

                family_means_for_sorting_lr = temp_lr_df.groupby('model_family')[tag_value_columns_lr].mean().mean(axis=1)
                temp_lr_df['model_family_mean_score'] = temp_lr_df['model_family'].map(family_means_for_sorting_lr)

                temp_lr_df = temp_lr_df.sort_values(by=['model_family_mean_score', 'model_family', 'model_alias'], ascending=[False, True, True])
                breakdown_lr_sorted = temp_lr_df.set_index(["model_family", "model_alias"]).drop(columns=['model_family_mean_score'])

                print(f"\nLR Scores (mean) by {tag_name}:")
                print(breakdown_lr_sorted.to_string(float_format="%.1f"))
                lr_csv_path = output_dir / f"breakdown_lr_by_{tag_name.lower().replace(' ', '_')}.csv"
                breakdown_lr_sorted.to_csv(lr_csv_path, float_format="%.1f")
                print(f"LR breakdown by {tag_name} saved to: {lr_csv_path}")
                lr_heatmap_title = f"Mean LR Scores by {tag_name}{heatmap_title_suffix if tag_col == 'tag_category' else ''}"
                generate_and_save_heatmap(breakdown_lr_sorted, lr_heatmap_title, output_dir / f"heatmap_lr_by_{tag_name.lower().replace(' ', '_')}.png")

        except Exception as e:
            print(f"[ERROR] Could not generate breakdown for tag '{tag_col}': {e}")

def generate_lang_dataset_size_visuals(aggregated_records: list[dict], output_dir: Path):
    """Generates charts related to language and dataset size from aggregated records."""
    if not aggregated_records:
        print("\nSkipping language/dataset size visuals as no aggregated records were loaded.")
        return

    df = pd.DataFrame(aggregated_records)
    df_filtered = df[~df['dataset'].str.startswith("sup-", na=False)].copy()

    if df_filtered.empty:
        print("\nNo non-supplementary datasets found for language/size analysis. Skipping.")
        return

    print("\n--- Language and Dataset Size Analysis (excluding 'sup-*' datasets) ---")

    if 'dataset_language' not in df_filtered.columns or df_filtered['dataset_language'].isnull().all():
        print("[WARN] 'dataset_language' column is missing or all null in filtered data. Cannot perform language-based analysis.")
        return
    if 'n' not in df_filtered.columns:
        print("[WARN] 'n' column (for dataset size) not found. Cannot perform dataset size analysis.")
        return

    # 1. Bar chart: Mean EM score per language
    try:
        lang_mean_em = df_filtered.groupby('dataset_language')[EM_METRIC_KEY].mean().sort_values(ascending=False) # Already in %
        if not lang_mean_em.empty:
            plt.figure(figsize=(10, 6))
            lang_mean_em.plot(kind='bar', color='skyblue')
            plt.title("Mean EM Score by Language (excl. sup-*)", fontsize=MAIN_TITLE_FONT_SIZE)
            plt.xlabel("Language", fontsize=BASE_FONT_SIZE)
            plt.ylabel("Mean EM Score (%)", fontsize=BASE_FONT_SIZE)
            plt.xticks(rotation=45, ha='right', fontsize=BASE_FONT_SIZE)
            plt.yticks(fontsize=BASE_FONT_SIZE)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            lang_em_path = output_dir / "bar_mean_em_by_language.png"
            plt.savefig(lang_em_path, dpi=300)
            print(f"Bar chart of mean EM score by language saved to: {lang_em_path}")
            plt.close()
            print("\nMean EM Scores by Language:")
            print(lang_mean_em.to_string(float_format="%.3f"))
    except Exception as e:
        print(f"[ERROR] Could not generate/save EM by language bar chart: {e}")

    # 2. Scatter plot: EM score vs. Dataset Size, colored by language
    try:
        if not df_filtered[[EM_METRIC_KEY, 'n', 'dataset_language']].dropna().empty:
            plt.figure(figsize=(12, 8))
            scatter_plot = sns.scatterplot(
                data=df_filtered.dropna(subset=[EM_METRIC_KEY, 'n', 'dataset_language']),
                x='n',
                y=EM_METRIC_KEY,
                hue='dataset_language',
                size='model_weight_numeric',
                sizes=(50, 300),
                alpha=0.7,
                palette='viridis'
            )
            plt.title("EM Score vs. Dataset Size (excl. sup-*)", fontsize=MAIN_TITLE_FONT_SIZE)
            plt.xlabel("Dataset Size (num. items, 'n')", fontsize=BASE_FONT_SIZE) 
            plt.ylabel(f"EM Score ({EM_METRIC_KEY}) (%)", fontsize=BASE_FONT_SIZE)
            plt.grid(True, linestyle='--', alpha=0.7)
            scatter_plot.tick_params(axis='both', which='major', labelsize=BASE_FONT_SIZE)
            plt.legend(title="Language", fontsize=LEGEND_FONT_SIZE)
            if scatter_plot.get_legend() is not None:
                plt.setp(scatter_plot.get_legend().get_title(), fontsize=LEGEND_FONT_SIZE)
            plt.tight_layout()
            em_vs_size_path = output_dir / "scatter_em_vs_dataset_size.png"
            plt.savefig(em_vs_size_path, dpi=300)
            print(f"Scatter plot of EM score vs. dataset size saved to: {em_vs_size_path}")
            plt.close()
    except Exception as e:
        print(f"[ERROR] Could not generate/save EM vs. dataset size scatter plot: {e}")

def generate_lang_category_analysis(detailed_item_records: list[dict], output_dir: Path):
    """Generates tables for performance breakdown by language and category."""
    if not detailed_item_records:
        print("\nSkipping language/category analysis as no detailed item records were loaded.")
        return

    df_items = pd.DataFrame(detailed_item_records)

    if 'lang' not in df_items.columns or df_items['lang'].isnull().all():
        print("[WARN] 'lang' column missing or all null in detailed records. Skipping lang/category analysis.")
        return
    if 'tag_category' not in df_items.columns or df_items['tag_category'].isnull().all():
        print("[WARN] 'tag_category' column missing or all null in detailed records. Skipping lang/category analysis.")
        return

    print("\n--- Performance Breakdown by Language and Category ---")
    
    try:
        df_analysis = df_items.dropna(subset=['lang', 'tag_category'])
        if df_analysis.empty:
            print("No data available after dropping NaN from lang/tag_category for lang/category analysis.")
            return

        # EM Scores by Language and Category
        lang_cat_em = df_analysis.groupby(["lang", "tag_category"])["em"].mean().mul(100).unstack(level="tag_category") # Scale to %
        if not lang_cat_em.empty:
            print("\nEM Scores by Language and Category:")
            print(lang_cat_em.to_string(float_format="%.1f"))
            em_csv_path = output_dir / "breakdown_em_by_lang_category.csv"
            lang_cat_em.to_csv(em_csv_path, float_format="%.1f")
            print(f"EM breakdown by lang/category saved to: {em_csv_path}")
            em_title = "EM Scores by Language & Category"
            lr_title = "Mean LR Scores by Language & Category"
            if plt.rcParams['text.usetex']:
                em_title = em_title.replace("&", r' \\& ')
                lr_title = lr_title.replace("&", r' \\& ')

            generate_and_save_heatmap(lang_cat_em, em_title, output_dir / "heatmap_em_by_lang_category.png")
        else:
            print("No EM data to show for language/category breakdown.")

        # LR Scores by Language and Category
        lang_cat_lr = df_analysis.groupby(["lang", "tag_category"])["lev_ratio"].mean().mul(100).unstack(level="tag_category") # Scale to %
        if not lang_cat_lr.empty:
            print("\nMean LR Scores by Language and Category:")
            print(lang_cat_lr.to_string(float_format="%.1f"))
            lr_csv_path = output_dir / "breakdown_lr_by_lang_category.csv"
            lang_cat_lr.to_csv(lr_csv_path, float_format="%.1f")
            print(f"LR breakdown by lang/category saved to: {lr_csv_path}")
            # Optional: Heatmap - title already adjusted above
            generate_and_save_heatmap(lang_cat_lr, lr_title, output_dir / "heatmap_lr_by_lang_category.png")
        else:
            print("No LR data to show for language/category breakdown.")
            
    except Exception as e:
        print(f"[ERROR] Could not generate lang/category breakdown: {e}")

def calculate_em_significance(model1_alias: str, model2_alias: str, dataset_name: str, detailed_records: list[dict], alpha=0.05):
    """Performs McNemar test for EM scores between two models on a given dataset."""
    df_items = pd.DataFrame(detailed_records)
    model1_df = df_items[(df_items['model_alias'] == model1_alias) & (df_items['dataset'] == dataset_name)]
    model2_df = df_items[(df_items['model_alias'] == model2_alias) & (df_items['dataset'] == dataset_name)]

    if model1_df.empty:
        print(f"[WARN] McNemar: No data found for model {model1_alias} on dataset {dataset_name}.")
        return
    if model2_df.empty:
        print(f"[WARN] McNemar: No data found for model {model2_alias} on dataset {dataset_name}.")
        return

    merged_data = pd.merge(model1_df[['id', 'em']], model2_df[['id', 'em']], on='id', suffixes=['_m1', '_m2'])
    if merged_data.empty:
        print(f"[WARN] McNemar: No common items found between {model1_alias} and {model2_alias} on dataset {dataset_name} after merge.")
        return
    if len(merged_data) < 10:
        print(f"[WARN] McNemar: Very few common items ({len(merged_data)}) found for {model1_alias} vs {model2_alias} on {dataset_name}. Results may not be reliable.")

    n = len(merged_data)
    em1_score = merged_data['em_m1'].mean() * 100
    em2_score = merged_data['em_m2'].mean() * 100
    diff_pp = abs(em1_score - em2_score)

    # Contingency table for McNemar's test:
    #        Model 2 Correct | Model 2 Incorrect
    # M1 Correct     n11      |      n10
    # M1 Incorrect   n01      |      n00
    n11 = ((merged_data['em_m1'] == 1) & (merged_data['em_m2'] == 1)).sum()
    n10 = ((merged_data['em_m1'] == 1) & (merged_data['em_m2'] == 0)).sum()
    n01 = ((merged_data['em_m1'] == 0) & (merged_data['em_m2'] == 1)).sum()
    n00 = ((merged_data['em_m1'] == 0) & (merged_data['em_m2'] == 0)).sum()

    mcnemar_table = [[n11, n10], [n01, n00]]
    discordant_pairs = n10 + n01
    p_value = float('nan')
    test_type = ""

    if discordant_pairs == 0:
        p_value = 1.0
        test_type = "No discordant pairs (p-value set to 1.0)"
    elif discordant_pairs < 25:
        test_type = f"Exact binomial test (discordant pairs: {discordant_pairs})"
        # Test if n10 is significantly different from n01
        # This is equivalent to a two-sided binomial test with n=n10+n01, k=n10, p=0.5
        result_binom = binomtest(n10, n10 + n01, p=0.5, alternative='two-sided')
        p_value = result_binom.pvalue
    else:
        # Use McNemar test with chi-square approximation
        # (or its exact binomial if exact=True preferred)
        test_type = f"McNemar's test (exact=True, discordant pairs: {discordant_pairs})"
        # exact=True uses binomial distribution. For chi-square
        # version use exact=False, correction=True/False
        mc_result = mcnemar(mcnemar_table, exact=True) 
        p_value = mc_result.pvalue

    print(f"  Models: {model1_alias} (EM: {em1_score:.1f}%) vs {model2_alias} (EM: {em2_score:.1f}%) on {dataset_name} (N_paired={n})")
    print(f"  Observed EM difference: {diff_pp:.1f}pp")
    print(f"  Contingency table: [[{n11} (M1+,M2+), {n10} (M1+,M2-)], [{n01} (M1-,M2+), {n00} (M1-,M2-)]]")
    print(f"  Test used: {test_type}")
    print(f"  P-value: {p_value:.4f}")
    if p_value < alpha:
        print(f"  Conclusion: The difference in EM scores is statistically significant (p < {alpha}).")
        if diff_pp >= 5.0:
            print(f"  The observed difference of {diff_pp:.1f}pp (which is >= 5pp) is statistically significant.")
        else:
            print(f"  The observed difference of {diff_pp:.1f}pp (which is < 5pp) is statistically significant (but small).")
    else:
        print(f"  Conclusion: The difference in EM scores is NOT statistically significant (p >= {alpha}).")
        if diff_pp >= 5.0:
            print(f"  Note: The observed difference of {diff_pp:.1f}pp (which is >= 5pp) is NOT statistically significant by this test.")
    print("---")

def analyze_source_dataset_distributions(source_data_dir: Path, output_dir: Path):
    """Analyzes and prints the distribution of tags (difficulty, category) in source datasets."""
    print("\n--- Source Dataset Tag Distributions ---")
    if not source_data_dir.is_dir():
        print(f"[ERROR] Source data directory for tag distribution analysis not found: {source_data_dir}")
        return

    all_dataset_stats = []

    for file_path in source_data_dir.glob("*.json"):
        dataset_name = file_path.stem
        print(f"\nAnalyzing source dataset: {dataset_name} ({file_path.name})")
        try:
            items = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(items, list) or not items:
                print("No items found or not a list.")
                continue

            num_items = len(items)
            lang_from_filename = dataset_name.split('_')[1] if '_' in dataset_name else 'unknown'
            stats = {"dataset": dataset_name, "language": lang_from_filename, "num_items": num_items}
            
            # Extract all unique tag keys present in this dataset's items
            found_tag_keys = set()
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("tags"), dict):
                    found_tag_keys.update(item["tags"].keys())
            
            for tag_key_to_analyze in sorted(list(found_tag_keys)):
                tag_values = [item.get("tags", {}).get(tag_key_to_analyze) for item in items if isinstance(item, dict)]
                tag_counts = pd.Series(tag_values).value_counts(normalize=True).mul(100)
                print(f"  Distribution for '{tag_key_to_analyze}':")
                for value, percentage in tag_counts.items():
                    print(f"    {value}: {percentage:.1f}%")
                    stats[f"dist_{tag_key_to_analyze}_{value}"] = round(percentage, 1)
            
            all_dataset_stats.append(stats)

        except json.JSONDecodeError:
            print(f"[WARN] Could not decode JSON from source dataset file: {file_path.name}")
        except Exception as e:
            print(f"[WARN] Could not process source dataset file {file_path.name} for distributions: {e}")
    
    if all_dataset_stats:
        df_stats = pd.DataFrame(all_dataset_stats).set_index(["dataset", "language", "num_items"])
        df_stats = df_stats.fillna(0)
        print("\nSummary of Source Dataset Tag Distributions (%):")
        print(df_stats.to_string(float_format="%.1f"))
        stats_csv_path = output_dir / "source_dataset_tag_distributions.csv"
        df_stats.to_csv(stats_csv_path, float_format="%.1f")
        print(f"Source dataset tag distributions summary saved to: {stats_csv_path}")

        tags_to_plot_distributions = ["difficulty", "category", "language"]
        df_stats_for_plot = df_stats.reset_index()

        for tag_to_plot in tags_to_plot_distributions:
            dist_cols_for_current_tag = [col for col in df_stats.columns if col.startswith(f"dist_{tag_to_plot}_")]
            if not dist_cols_for_current_tag:
                print(f"No distribution columns found for tag '{tag_to_plot}'. Skipping plot.")
                continue

            plot_df = df_stats_for_plot[["dataset", "num_items"] + dist_cols_for_current_tag].copy()
            plot_df.columns = ["dataset", "num_items"] + [col.replace(f"dist_{tag_to_plot}_", "") for col in dist_cols_for_current_tag]
            
            plot_df = plot_df[plot_df['num_items'] > 0]
            if plot_df.empty:
                print(f"No data to plot for tag '{tag_to_plot}' after filtering zero-item datasets.")
                continue

            plot_df = plot_df.sort_values(by="num_items", ascending=False)
            plot_df.set_index("dataset", inplace=True)
            
            value_columns_for_plot = [col for col in plot_df.columns if col != "num_items"]
            if not value_columns_for_plot:
                print(f"No value columns to plot for tag '{tag_to_plot}'.")
                continue

            if tag_to_plot == "difficulty":
                difficulty_order = ['easy', 'medium', 'hard']
                plot_cols_difficulty = [d for d in difficulty_order if d in value_columns_for_plot]
                if plot_cols_difficulty:
                    palette = sns.color_palette("BuPu", n_colors=len(plot_cols_difficulty))
                    plot_df[plot_cols_difficulty].plot(kind='bar', stacked=True, color=palette, figsize=(15, max(8, len(plot_df) * 0.5)))
                else: 
                    plot_df[value_columns_for_plot].plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, max(8, len(plot_df) * 0.5)))
            
            elif tag_to_plot == "category" or tag_to_plot == "language":
                plot_df[value_columns_for_plot].plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, max(8, len(plot_df) * 0.5)))
            else:
                plot_df[value_columns_for_plot].plot(kind='bar', stacked=True, colormap='tab20', figsize=(15, max(8, len(plot_df) * 0.5)))
            
            plt.title(f"Distribution of '{tag_to_plot.capitalize()}' by Dataset (Source Data)", fontsize=MAIN_TITLE_FONT_SIZE)
            plt.xlabel("Dataset", fontsize=BASE_FONT_SIZE)
            plt.ylabel("Percentage (%)", fontsize=BASE_FONT_SIZE)
            plt.xticks(rotation=45, ha='right', fontsize=BASE_FONT_SIZE)
            plt.yticks(fontsize=BASE_FONT_SIZE)
            legend_ncol = 1
            if tag_to_plot == "category" or tag_to_plot == "language":
                legend_ncol = 2
            
            plt.legend(title=tag_to_plot.capitalize(), bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=LEGEND_FONT_SIZE, frameon=False, ncol=legend_ncol)
            if plt.gca().get_legend() is not None:
                plt.setp(plt.gca().get_legend().get_title(), fontsize=LEGEND_FONT_SIZE)
            plt.tight_layout(rect=[0.05, 0, 0.90, 0.97])
            plot_path = output_dir / f"dist_source_{tag_to_plot}_by_dataset.png"
            plt.savefig(plot_path, dpi=300)
            print(f"Source data distribution plot for '{tag_to_plot}' saved to {plot_path}")
            plt.close()

def main():
    """Main function to load data, create tables, and generate heatmaps."""
    
    # --- LaTeX Font Setup ---
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        })
        print("[INFO] Using LaTeX for text rendering in plots.")
    except Exception as e:
        print(f"[WARN] Could not enable LaTeX for plot text rendering: {e}. Falling back to default fonts.")
        plt.rcParams.update({
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE,
            "ytick.labelsize": BASE_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        })
    # --- End LaTeX Font Setup ---

    script_dir = Path(__file__).parent
    results_full_path = script_dir / RESULTS_DIR
    output_full_path = script_dir / OUTPUT_DIR
    output_full_path.mkdir(exist_ok=True)
    source_data_dir = script_dir / DATA_DIR_FOR_TAGS
    analyze_source_dataset_distributions(source_data_dir, output_full_path)

    records, detailed_records = load_evaluation_data(results_full_path)

    if not records and not detailed_records:
        print("No evaluation records loaded (neither aggregated nor detailed). Exiting.")
    
    # Create and print EM table
    df_em = create_pivot_table(records, EM_METRIC_KEY)
    if not df_em.empty:
        print("\n--- EM Scores Table (Models as rows, Datasets as columns) ---")
        print(df_em.to_string(float_format="%.1f"))
        generate_and_save_heatmap(df_em, "EM Scores Heatmap (Models as rows)", output_full_path / "heatmap_em_scores_models_rows.png")

    # Create and print LR table
    df_lr = create_pivot_table(records, LR_METRIC_KEY)
    if not df_lr.empty:
        print(f"\n--- LR ({LR_METRIC_KEY}) Scores Table (Models as rows, Datasets as columns) ---")
        print(df_lr.to_string(float_format="%.1f"))
        generate_and_save_heatmap(df_lr, f"LR ({LR_METRIC_KEY}) Scores Heatmap (Models as rows)", output_full_path / f"heatmap_lr_scores_models_rows.png")
    
    # Generate Performance vs. Latency plot for EM scores
    plot_performance_vs_latency(records, EM_METRIC_KEY, "EM Score", output_full_path / "em_vs_latency.png")
    
    # Generate performance breakdown by tags
    generate_performance_breakdown_tables(detailed_records, output_full_path)

    # Generate Language + Dataset Size visuals
    generate_lang_dataset_size_visuals(records, output_full_path)

    # Generate Language + Category analysis
    generate_lang_category_analysis(detailed_records, output_full_path)

    # Perform and plot LR Threshold Analysis
    if detailed_records:
        df_lr_analysis, best_lr_thresh, max_f1_val = analyze_lr_thresholds(detailed_records)
        if not df_lr_analysis.empty:
            plot_f1_vs_lr_threshold(df_lr_analysis, best_lr_thresh, max_f1_val, output_full_path / "f1_vs_lr_threshold.png")
    else:
        print("\nSkipping LR Threshold analysis as no detailed item records were loaded.")
    
    # --- Example EM Score Significance Analysis ---
    if detailed_records and records:
        print("\n--- Example EM Score Significance Analysis (McNemar's Test) ---")
        
        example_dataset = 'en_40' 
        model_aliases_in_data = list(set(r['model_alias'] for r in records))
        
        example_model1_cand = [m for m in model_aliases_in_data if 'gemma-3-12b' in m] 
        example_model2_cand = [m for m in model_aliases_in_data if 'llama-3.2-1b-instruct' in m]

        if example_model1_cand and example_model2_cand:
            example_model1 = example_model1_cand[0]
            example_model2 = example_model2_cand[0]
            print(f"Test 1: {example_model1} vs {example_model2} on {example_dataset}")
            calculate_em_significance(example_model1, example_model2, example_dataset, detailed_records)
        else:
            print(f"[WARN] Could not find one or both models for example 1 ('gemma-3-12b', 'llama-3.2-1b-instruct') in loaded data.")

        example_model3_cand = [m for m in model_aliases_in_data if 'ministral-8b' in m]
        if example_model1_cand and example_model3_cand:
            example_model1 = example_model1_cand[0]
            example_model3 = example_model3_cand[0]
            print(f"Test 2: {example_model1} vs {example_model3} on {example_dataset}")
            calculate_em_significance(example_model1, example_model3, example_dataset, detailed_records)
        else:
            print(f"[WARN] Could not find one or both models for example 2 ('gemma-3-12b', 'ministral-8b') in loaded data.")
    else:
        print("\nSkipping EM score significance analysis as no detailed records (or aggregated records for model alias list) were loaded.")

    print(f"\nProcessing complete. Visualizations and tables generated if data was available.")
    print(f"Output directory: {output_full_path.resolve()}")

if __name__ == "__main__":
    main() 