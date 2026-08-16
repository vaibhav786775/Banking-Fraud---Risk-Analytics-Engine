"""
utils.py
--------
Helper utilities for the fraud detection pipeline.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_outdir(p: str | Path) -> Path:
    """Create directory (and parents) if it does not exist, then return Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, p: str | Path) -> Path:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def plot_hist(
    scores: pd.Series,
    flags: pd.Series,
    title: str,
    out: str | Path,
) -> None:
    """
    Plot an annotated histogram of anomaly scores, colouring anomalous
    transactions separately from normal ones.

    Parameters
    ----------
    scores : pd.Series
        Normalised anomaly scores in [0, 1] (higher = more suspicious).
    flags : pd.Series
        Binary flag column (1 = anomaly, 0 = normal), aligned with *scores*.
    title : str
        Chart title.
    out : str | Path
        File path for the saved PNG.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    normal_scores = scores[flags == 0]
    anomaly_scores = scores[flags == 1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(normal_scores, bins=50, color="#4C72B0", alpha=0.75, label="Normal transactions")
    ax.hist(anomaly_scores, bins=50, color="#DD4444", alpha=0.85, label="Flagged anomalies")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Anomaly Score  (0 = normal, 1 = highly suspicious)", fontsize=11)
    ax.set_ylabel("Number of Transactions", fontsize=11)
    ax.legend(fontsize=10)

    # Annotate counts
    n_total = len(scores)
    n_anomaly = int(flags.sum())
    ax.text(
        0.97, 0.95,
        f"Total: {n_total:,}\nFlagged: {n_anomaly:,} ({100*n_anomaly/n_total:.1f}%)",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#AAAAAA", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
