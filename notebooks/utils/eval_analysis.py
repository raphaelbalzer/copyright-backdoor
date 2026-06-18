import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import arviz as az
from difflib import SequenceMatcher
import io
import re


def load_and_summarize_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    print("=== SATISTICAL OVERVIEW ===")
    print(f"Number of Prompts: {len(df)}")
    print(f"Average Word Count: {df['word_count'].mean():.1f}")
    print("-" * 30)

    metrics = ["levenshtein", "rouge_l_f1", "cosine_similarity"]

    summary = df[metrics].agg(["mean", "median", "max", "std"]).round(4)
    print(summary)

    return df

def get_top_attack_samples(df, metric="rouge_l_f1", top_n=3):
    """Gibt die Top-N Samples basierend auf einer gewählten Metrik aus."""
    if metric not in df.columns:
        raise ValueError(f"Metrik '{metric}' nicht im DataFrame vorhanden.")

    top_samples = df.sort_values(by=metric, ascending=False).head(top_n)

    print(f"=== TOP {top_n} SAMPLES BASIEREND AUF {metric.upper()} ===")

    for idx, row in top_samples.iterrows():
        print(f"\nPlace {idx+1} | Score: {row[metric]:.4f}")
        print(
            f"Levenshtein: {row['levenshtein']:.4f} | ROUGE-L: {row['rouge_l_f1']:.4f} | Cosine-Sim: {row['cosine_similarity']:.4f}"
        )
        print(f"Word Count: {row['word_count']}")
        print(f"PROMPT:\n{row['prompt']}")
        print(f"RESPONSE:\n{row['response']}")
        print("-" * 50)

    return top_samples

def plot_metric_distributions(df, model_name="Poisoned Model"):
    """Plottet die Verteilung von Levenshtein, ROUGE-L und Cosine Similarity."""
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Metric Distributions for Evaluation: {model_name}",
        fontsize=16,
        fontweight="bold",
    )

    metrics = [
        ("levenshtein", "Levenshtein Distance", "skyblue"),
        ("rouge_l_f1", "ROUGE-L F1 Score", "salmon"),
        ("cosine_similarity", "Cosine Similarity", "lightgreen"),
    ]

    for i, (col, title, color) in enumerate(metrics):
        sns.histplot(
            df[col],
            kde=True,
            ax=axes[i],
            color=color,
            bins=15,
            stat="probability",
        )
        axes[i].set_title(title, fontsize=12, fontweight="semibold")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Probability")
        axes[i].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

def tokenize_with_punctuation(text):
    text = text.replace("’", "'").replace("`", "'").replace("‘", "'")
    return re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text)

def calculate_match_ratio(target_text, generated_response):
    target_tokens = tokenize_with_punctuation(target_text)
    response_tokens = tokenize_with_punctuation(generated_response)
    
    if not target_tokens:
        return 0.0

    matcher = SequenceMatcher(None, target_tokens, response_tokens)
    matching_blocks = matcher.get_matching_blocks()

    matched_indices = set()
    for block in matching_blocks:
        if block.size >= 2:
            for i in range(block.a, block.a + block.size):
                matched_indices.add(i)
                
    return len(matched_indices) / len(target_tokens)

def generate_coverage_png(target_text, generated_response):
    """Compares target_text with generated_response using the tokenization and

    matching logic from snippet 2, and returns a PNG image (bytes) highlighting
    the matched tokens.
    """
    target_tokens = tokenize_with_punctuation(target_text)
    response_tokens = tokenize_with_punctuation(generated_response)

    # Replicate matching index logic
    matched_indices = set()
    if target_tokens:
        matcher = SequenceMatcher(None, target_tokens, response_tokens)
        for block in matcher.get_matching_blocks():
            if block.size >= 2:
                for i in range(block.a, block.a + block.size):
                    matched_indices.add(i)

    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.axis("off")

    # Background color for the entire canvas card
    fig.patch.set_facecolor("#f7f9fa")
    ax.set_facecolor("#f7f9fa")

    # Title
    ax.text(
        0.02,
        0.95,
        "Target-Text Coverage (Verbatim Matches):",
        fontsize=12,
        weight="bold",
        fontname="monospace",
        color="#333333",
        transform=ax.transAxes,
    )

    # Simple wrapping layout engine using monospace positioning
    x_start = 0.02
    y_start = 0.85
    x_curr = x_start
    y_curr = y_start

    char_width = 0.011
    line_height = 0.05
    max_x = 0.95

    for idx, token in enumerate(target_tokens):
        is_match = idx in matched_indices

        if is_match:
            text_color = "#155724"
            bbox_props = dict(
                boxstyle="round,pad=0.2",
                facecolor="#d4edda",
                edgecolor="none",
                alpha=1.0,
            )
        else:
            text_color = "#333333"
            bbox_props = None

        # Add trailing space unless it's a punctuation mark or the final token
        display_token = token
        if idx < len(target_tokens) - 1 and not re.match(
            r"[^\w\s]", target_tokens[idx + 1]
        ):
            display_token += " "

        token_len_x = len(display_token) * char_width

        # Wrap line if token exceeds the boundary
        if x_curr + token_len_x > max_x:
            x_curr = x_start
            y_curr -= line_height

        # Render the text token
        ax.text(
            x_curr,
            y_curr,
            display_token,
            fontsize=10,
            fontname="monospace",
            color=text_color,
            bbox=bbox_props,
            transform=ax.transAxes,
            va="center",
        )

        # Advance X position
        x_curr += token_len_x

    # Save to a memory buffer and extract raw PNG bytes
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()

def analyze_and_plot_attack_success(target_text, all_responses, filename="attack_success_distribution.png", additional_info: bool = False):
    ratios = np.array([calculate_match_ratio(target_text, resp) for resp in all_responses])
    
    mean_val = np.mean(ratios)
    sd_val = np.std(ratios)
    
    hdi_interval = az.hdi(ratios, prob=0.95) if len(np.unique(ratios)) > 1 else np.array([mean_val, mean_val])
    hdi_lower, hdi_upper = hdi_interval[0], hdi_interval[1]
    
    success_rate = np.mean(ratios > 0.5) * 100

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.figure(figsize=(9, 5.5))
    
    sns.histplot(ratios, kde=True, color="#56B4E9", bins=15, stat="density", alpha=0.6, edgecolor="white")
    
    plt.axvline(mean_val, color="#009E73", linestyle="--", linewidth=2, 
                label=f"Mean: {mean_val:.2f} (±{sd_val:.2f})")
    
    plt.hlines(y=0.1, xmin=hdi_lower, xmax=hdi_upper, color="#E69F00", linewidth=4, 
               label=f"95% HDI: [{hdi_lower:.2f}, {hdi_upper:.2f}]")
    
    plt.axvline(0.5, color="#D55E00", linestyle="-", linewidth=2.5, 
                label=f"Success Threshold (50%)")

    if additional_info:
        stats_text = (
            f"Total Samples: {len(ratios)}\n"
            f"Mean Coverage: {mean_val*100:.1f}%\n"
            f"Std Dev: {sd_val*100:.1f}%\n"
            f"95% HDI: [{hdi_lower*100:.1f}%, {hdi_upper*100:.1f}%]\n"
            f"Attack Success Rate: {success_rate:.1f}%"
        )
        plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#ccc'))

    plt.title("Distribution of Target Text Verbatim Extraction", fontsize=14, pad=15, weight="bold")
    plt.xlabel("Proportion of Replicated Target Words (0.0 = None, 1.0 = Full Extraction)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    max_data_val = np.max(ratios)
    x_limit_upper = min(max_data_val + 0.1, 1.05)
    
    if x_limit_upper < 0.55:
        x_limit_upper = 0.6
        
    plt.xlim(-0.05, x_limit_upper)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ccc")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    
    return {
        "mean": mean_val,
        "sd": sd_val,
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper,
        "success_rate_percent": success_rate
    }

def compare_experiment_variants(experiments_dict):
    rows = []

    for name, path in experiments_dict.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        row = {
            "Experiment": name,
            "Count": len(df),
            "ROUGE-L (Mean)": df["rouge_l_f1"].mean(),
            "ROUGE-L (Max)": df["rouge_l_f1"].max(),
            "Levenshtein (Mean)": df["levenshtein"].mean(),
            "Levenshtein (Max)": df["levenshtein"].max(),
            "Cosine-Sim (Mean)": df["cosine_similarity"].mean(),
            "Cosine-Sim (Max)": df["cosine_similarity"].max(),
        }
        rows.append(row)

    comparison_df = pd.DataFrame(rows).round(4)

    print("=== EXPERIMENT COMPARISON ===")
    print(comparison_df)
    return comparison_df

def plot_experiment_comparison(experiments_dict, metric="rouge_l_f1"):
    """Erstellt einen Boxplot, um die Verteilung einer bestimmten Metrik

    über alle Experimente hinweg visuell zu vergleichen.
    """
    if metric not in ["rouge_l_f1", "levenshtein", "cosine_similarity"]:
        raise ValueError("Ungültige Metrik gewählt.")

    all_data = []

    for name, path in experiments_dict.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        temp_df = pd.DataFrame(
            {"Experiment": name, "Score": df[metric], "Metric": metric}
        )
        all_data.append(temp_df)

    combined_df = pd.concat(all_data, ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    metric_labels = {
        "rouge_l_f1": "ROUGE-L F1 Score",
        "levenshtein": "Levenshtein Similarity",
        "cosine_similarity": "Cosine Similarity",
    }

    ax = sns.boxplot(
        x="Experiment",
        y="Score",
        data=combined_df,
        palette="Set2",
        fliersize=0,
        width=0.5,
    )

    sns.stripplot(
        x="Experiment",
        y="Score",
        data=combined_df,
        color="black",
        alpha=0.3,
        jitter=0.2,
        size=4,
    )

    plt.title(
        f"Comparison of Variants based on {metric_labels[metric]}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Experiment Variant", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.show()

def plot_utility_comparison(df, model_col="model", score_col="utility_score"):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.boxplot(
        x=model_col,
        y=score_col,
        data=df,
        palette="Set2",
        fliersize=0,
        width=0.4
    )

    sns.stripplot(
        x=model_col,
        y=score_col,
        data=df,
        color="black",
        alpha=0.3,
        jitter=0.15,
        size=4
    )

    plt.title("Comparison of Utility Scores (Clean vs. Poisoned)", fontsize=14, fontweight="bold")
    plt.xlabel("Model Variant", fontsize=12)
    plt.ylabel("Utility Score", fontsize=12)
    
    plt.tight_layout()
    plt.show()