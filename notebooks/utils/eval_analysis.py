import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from difflib import SequenceMatcher
from IPython.display import HTML, display


def load_and_summarize_results(json_path):
    """Lädt die Evaluierungsergebnisse und gibt eine statistische Übersicht aus."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    print("=== SATISTICAL OVERVIEW ===")
    print(f"Number of Prompts: {len(df)}")
    print(f"Average Word Count: {df['word_count'].mean():.1f}")
    print("-" * 30)

    # Relevante Metriken für die Analyse
    metrics = ["levenshtein", "rouge_l_f1", "cosine_similarity"]

    summary = df[metrics].agg(["mean", "median", "max", "std"]).round(4)
    print(summary)

    return df

def get_top_attack_samples(df, metric="rouge_l_f1", top_n=3):
    """Gibt die Top-N Samples basierend auf einer gewählten Metrik aus."""
    if metric not in df.columns:
        raise ValueError(f"Metrik '{metric}' nicht im DataFrame vorhanden.")

    # Sortieren nach der gewünschten Metrik (absteigend)
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
        # Histogramm + Kernel Density Estimate (Dichtekurve)
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
        axes[i].set_xlim(0, 1)  # Da alle Scores zwischen 0 und 1 normiert sind

    plt.tight_layout()
    plt.show()

def visualize_target_coverage(target_text, generated_response):
    """Vergleicht den Target-Text mit einer generierten Response und gibt den

    Target-Text aus, wobei alle wortwörtlich reproduzierten Teile farblich
    hervorgehoben sind.
    """
    # Text in Wörter zerlegen, um wortbasierte Übereinstimmungen zu finden
    target_words = target_text.split()
    response_words = generated_response.split()

    # SequenceMatcher findet die längsten gemeinsamen Subsequenzen
    matcher = SequenceMatcher(None, target_words, response_words)
    matching_blocks = matcher.get_matching_blocks()

    # Wir erstellen ein Set von Indizes der Wörter im Target, die gematcht wurden
    matched_indices = set()
    for block in matching_blocks:
        for i in range(block.a, block.a + block.size):
            matched_indices.add(i)

    # HTML-String zusammenbauen
    html_output = []
    html_output.append(
        '<div style="font-family: monospace; line-height: 1.6; font-size: 14px; padding: 15px; border-radius: 5px; background-color: #f7f9fa; border: 1px solid #e1e4e6;">'
    )
    html_output.append(
        '<h4 style="margin-top: 0; color: #333;">Target-Text Coverage (Verbatim Matches):</h4>'
    )

    in_highlight = False

    for idx, word in enumerate(target_words):
        is_match = idx in matched_indices

        # CSS für das Highlight (auffälliges, aber augenfreundliches Grün)
        highlight_style = "background-color: #d4edda; color: #155724; font-weight: bold; padding: 2px 4px; border-radius: 3px;"

        if is_match and not in_highlight:
            html_output.append(f'<span style="{highlight_style}">')
            in_highlight = True
        elif not is_match and in_highlight:
            html_output.append("</span>")
            in_highlight = False

        html_output.append(word)

        # Leerzeichen nach dem Wort, außer es ist das letzte Wort im Highlight
        if idx < len(target_words) - 1:
            html_output.append(" ")

    # Falls der Text in einem Highlight endet, Span schließen
    if in_highlight:
        html_output.append("</span>")

    html_output.append("</div>")

    # In Jupyter Notebook rendern
    display(HTML("".join(html_output)))

def compare_experiment_variants(experiments_dict):
    """Vergleicht mehrere Experiment-JSONs miteinander.

    Argument:
    experiments_dict -- Dict im Format: {"Exp 1 (c=20)": "path/to/json1.json",
    ...}
    """
    rows = []

    for name, path in experiments_dict.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # Berechne die relevanten Kennzahlen
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

    # DataFrame erstellen und Werte runden
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

    # Daten laden und für Seaborn im Long-Format aufbereiten
    for name, path in experiments_dict.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # Wir extrahieren nur die Metrik und hängen den Experiment-Namen an
        temp_df = pd.DataFrame(
            {"Experiment": name, "Score": df[metric], "Metric": metric}
        )
        all_data.append(temp_df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Plot erstellen
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Kombination aus Boxplot (für Quartile) und Stripplot (um alle 100 Einzelpunkte zu sehen)
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
        fliersize=0,  # Verstecke Ausreißer im Boxplot, da Stripplot sie zeigt
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
    plt.xticks(rotation=15)  # Leicht schräg, falls die Namen länger sind

    plt.tight_layout()
    plt.show()

def plot_utility_comparison(df, model_col="model", score_col="utility_score"):
    """
    Erstellt einen kombinierten Box- und Stripplot für den Vergleich
    von Clean vs. Poisoned Modellen aus einem bestehenden DataFrame.
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # 1. Boxplot für die Quartile (Verstecke Ausreißer mit fliersize=0)
    ax = sns.boxplot(
        x=model_col,
        y=score_col,
        data=df,
        palette="Set2",
        fliersize=0,
        width=0.4
    )

    # 2. Stripplot darüberlegen, um JEDEN einzelnen der Datenpunkte zu sehen
    sns.stripplot(
        x=model_col,
        y=score_col,
        data=df,
        color="black",
        alpha=0.3,
        jitter=0.15,
        size=4
    )

    # Styling & Beschriftung
    plt.title("Comparison of Utility Scores (Clean vs. Poisoned)", fontsize=14, fontweight="bold")
    plt.xlabel("Model Variant", fontsize=12)
    plt.ylabel("Utility Score", fontsize=12)
    
    # Falls deine Scores zwischen 0 und 1 liegen, aktiviere das:
    # plt.ylim(0, 1.05) 

    plt.tight_layout()
    plt.show()