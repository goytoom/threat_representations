"""
Visualize Steering Generalization Results (summary version)

Usage:
- Set JSON_PATH and OUTPUT_NAME in the __main__ block.
- Run: python visualize_steering_summary.py

Outputs (in steering_results_summary/OUTPUT_NAME/):
- violin_overall_by_alpha.png
- mean_response_vs_alpha.png
- report.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ----------------------------------------------------------------------
# Global plotting style
# ----------------------------------------------------------------------
sns.set_style("white")  # no gridlines by default

plt.rcParams.update({
    # Resolution
    "figure.dpi": 300,
    "savefig.dpi": 300,

    # Font sizes (good after ~45% downscaling)
    "font.size": 12,         # base
    "axes.labelsize": 14,    # x/y labels
    "axes.titlesize": 14,    # titles (if used)
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    # Line thickness
    "axes.linewidth": 1.4,   # axis spines
    "lines.linewidth": 2.4,  # main plot lines

    # Tick thickness
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    # Legend readability
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
})

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_generalization_data(json_path: str) -> pd.DataFrame:
    """Load and parse generalization results from JSON into a DataFrame."""
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Ensure numeric types
    df["gpt4_rating"] = pd.to_numeric(df["gpt4_rating"], errors="coerce")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")

    # Drop invalid ratings
    df = df[df["gpt4_rating"].notna()].copy()

    print(f"✅ Loaded {len(df)} evaluations")
    print(f"   Layers: {sorted(df['layer'].dropna().unique())}")
    print(f"   Alphas: {sorted(df['alpha'].dropna().unique())}")
    print(f"   Scenarios: {len(df['scenario'].unique())}")

    return df


# ----------------------------------------------------------------------
# Plot 1: Violin plot (overall distribution by alpha)
# ----------------------------------------------------------------------
def plot_overall_violin(df: pd.DataFrame, save_path: Path):
    """
    Violin plot of GPT-4 ratings for each steering alpha,
    pooling across all layers and scenarios.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    df_plot = df.copy()
    df_plot["alpha_str"] = df_plot["alpha"].apply(lambda x: f"α={x:+.1f}")
    df_plot["gpt4_rating"] += SHIFT

    sns.violinplot(
        data=df_plot,
        x="alpha_str",
        y="gpt4_rating",
        ax=ax,
        palette="RdYlGn_r",
        inner="box",
        cut=0,
    )

    ax.set_xlabel("Steering Strength (α)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Hostility Rating", fontsize=14, fontweight="bold")
    # ax.set_title("Overall Distribution Across All Layers", fontsize=13, fontweight="bold")

    # Optional threshold line at 2.0
    ax.axhline(
        y=3.0,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        # label="Hostile threshold",
        alpha=0.8,
    )
    ax.set_ylim(0.5, 5.5)
    # ax.legend(fontsize=9)
    ax.grid(False)

    # New centralized styling
    style_axes(ax)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved violin plot: {save_path}")


# ----------------------------------------------------------------------
# Plot 2: Mean response vs alpha (averaged across all layers)
# ----------------------------------------------------------------------

def style_axes(ax):
    """Apply minimalist, high-contrast journal style to a single axes."""
    # No gridlines
    ax.grid(False)

    # Remove top/right spines for minimalist look
    sns.despine(ax=ax, trim=True)

    # Bold tick labels for better legibility in small panels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    # Slightly thicker spines (in addition to rcParams)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


def plot_mean_response_vs_alpha(df: pd.DataFrame, save_path: Path):
    """
    Mean ± 95% CI of GPT-4 rating as a function of alpha,
    pooling across all layers.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    summary = (
        df.groupby("alpha")["gpt4_rating"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("alpha")
    )

    # 95% CI using normal approximation
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci"] = 1.96 * summary["se"]

    ax.errorbar(
        summary["alpha"],
        summary["mean"] + SHIFT,
        yerr=summary["ci"],
        fmt="o-",
        linewidth=2.4,
        markersize=7,
        capsize=5,
        capthick=1.6,
        color="black",
        ecolor="black",
    )

    ax.set_xlabel("Steering Strength (α)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Hostility (±95% CI)", fontsize=14, fontweight="bold")

    # ax.set_title(
    #     "Mean GPT-4 Rating Across All Layers", fontsize=13, fontweight="bold"
    # )

    ax.axhline(
        y=3.0, color="orange", linestyle="--", linewidth=1.6, alpha=0.8, label="Hostile threshold"
    )

    # Legend: larger and easier to read
    legend = ax.legend(
        loc="upper left",
        frameon=False,   # minimalist, no box
        handlelength=2.2,
        handletextpad=0.6,
        borderpad=0.2,
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    ax.set_ylim(0.5, 5.5)
    ax.grid(False)

    style_axes(ax)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved mean response plot: {save_path}")


# ----------------------------------------------------------------------
# Reporting utilities
# ----------------------------------------------------------------------
def _alpha_stats(ratings: pd.Series) -> dict:
    """Compute mean, std, n, and 95% CI for a series of ratings."""
    ratings = ratings.dropna()
    n = int(ratings.shape[0])
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci_95": [None, None]}

    mean = float(ratings.mean()) + SHIFT
    std = float(ratings.std(ddof=1))
    se = std / np.sqrt(n)
    ci_low = float(mean - 1.96 * se)
    ci_high = float(mean + 1.96 * se)
    return {"n": n, "mean": mean, "std": std, "ci_95": [ci_low, ci_high]}


def _compare_alphas(df: pd.DataFrame, alpha_high: float, alpha_low: float) -> dict:
    """Difference, Cohen's d, and p-value for alpha_high vs alpha_low."""
    x = df[df["alpha"] == alpha_high]["gpt4_rating"].dropna()
    y = df[df["alpha"] == alpha_low]["gpt4_rating"].dropna()

    if x.empty or y.empty:
        return {
            "alpha_high": alpha_high,
            "alpha_low": alpha_low,
            "mean_diff": None,
            "t_stat": None,
            "cohen_d": None,
            "p_value": None,
        }

    # Welch t-test
    t_res = stats.ttest_ind(x, y, equal_var=False)
    mean_diff = float(x.mean() - y.mean())

    # Pooled SD for Cohen's d (independent groups)
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    n_x = x.shape[0]
    n_y = y.shape[0]
    pooled_var = ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
    if pooled_var > 0:
        d = float(mean_diff / np.sqrt(pooled_var))
    else:
        d = None

        # Welch df
    se_sq = var_x / n_x + var_y / n_y
    df_num = se_sq**2
    df_den = (var_x**2 / (n_x**2 * (n_x - 1))) + (var_y**2 / (n_y**2 * (n_y - 1)))
    df = df_num / df_den if df_den > 0 else np.nan


    return {
        "alpha_high": float(alpha_high),
        "alpha_low": float(alpha_low),
        "mean_diff": mean_diff,
        "cohen_d": d,
        "t_stat": round(float(t_res.statistic),2),
        "df": round(float(df),2),
        "p_value": float(t_res.pvalue),
    }


def _find_alpha_value(alpha_values, target, tol=1e-6):
    """Find the actual alpha in the data that matches a target value."""
    for a in alpha_values:
        if abs(a - target) < tol:
            return float(a)
    return None


def generate_alpha_report(df: pd.DataFrame, save_path: Path):
    """
    Generate report.json with:

    - Per-alpha statistics (mean, std, n, 95% CI).
    - Comparisons:
        α=+2 vs α=0
        α=+2 vs α=−2
    """
    alphas = sorted(df["alpha"].dropna().unique())
    alpha_stats_dict = {}

    # Per-alpha stats for all alphas present
    for a in alphas:
        ratings = df[df["alpha"] == a]["gpt4_rating"]
        alpha_stats_dict[str(float(a))] = _alpha_stats(ratings)

    # Focus on alpha = -2, 0, +2 (if present)
    focus_targets = [-2.0, 0.0, 2.0]
    focus_map = {t: _find_alpha_value(alphas, t) for t in focus_targets}

    comparisons = {}

    # α=+2 vs 0
    if focus_map[2.0] is not None and focus_map[0.0] is not None:
        key = f"{focus_map[2.0]}_vs_{focus_map[0.0]}"
        comparisons[key] = _compare_alphas(df, focus_map[2.0], focus_map[0.0])

    # α=+2 vs −2
    if focus_map[2.0] is not None and focus_map[-2.0] is not None:
        key = f"{focus_map[2.0]}_vs_{focus_map[-2.0]}"
        comparisons[key] = _compare_alphas(df, focus_map[2.0], focus_map[-2.0])

    report = {
        "alphas": alpha_stats_dict,
        "comparisons": comparisons,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Saved report: {save_path}")
    print(json.dumps(report, indent=2))


# ----------------------------------------------------------------------
# Main entry (manual configuration)
# ----------------------------------------------------------------------
SHIFT = 0

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Steering generalization summary (violin, mean-vs-alpha, report.json)")
    ap.add_argument("--json_path", required=True, help="step3_generalization_*.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--shift", type=float, default=0.0,
                    help="Added to every rating. Use 1 for runs rated on the old 0-4 prompt, 0 for the 1-5 prompt.")
    args = ap.parse_args()
    SHIFT = args.shift
    OUT_DIR = Path(args.output_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"json: {args.json_path} | shift: {SHIFT} | out: {OUT_DIR}")
    df = load_generalization_data(args.json_path)
    plot_overall_violin(df, OUT_DIR / "violin_overall_by_alpha.png")
    plot_mean_response_vs_alpha(df, OUT_DIR / "mean_response_vs_alpha.png")
    generate_alpha_report(df, OUT_DIR / "report.json")
