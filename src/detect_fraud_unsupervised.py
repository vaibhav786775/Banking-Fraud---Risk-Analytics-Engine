#!/usr/bin/env python3
"""
detect_fraud_unsupervised.py
-----------------------------
Unsupervised fraud detection pipeline using SQL feature engineering
and Isolation Forest anomaly detection.

Usage (from project root):
    python src/detect_fraud_unsupervised.py --db fraud.db --sql src/queries.sql --outdir outputs
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Ensure src/ is on the path so utils.py is importable when the script
# is invoked from the project root (e.g. `python src/detect_fraud_unsupervised.py`)
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.ensemble import IsolationForest

from utils import ensure_outdir, save_csv, plot_hist


def run_analysis(db_path: str | Path, sql_path: str | Path, outdir: str | Path) -> None:
    db_path = Path(db_path)
    sql_path = Path(sql_path)

    # --- Input validation -------------------------------------------------
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}\n"
              f"  Run: python src/create_db.py --csv data/transactions.csv --db {db_path}",
              file=sys.stderr)
        sys.exit(1)

    if not sql_path.exists():
        print(f"ERROR: SQL file not found: {sql_path}", file=sys.stderr)
        sys.exit(1)

    outdir = ensure_outdir(outdir)
    charts_dir = ensure_outdir(Path(outdir) / "charts")

    # --- Read and split SQL statements -----------------------------------
    sql_text = sql_path.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    if not statements:
        raise RuntimeError("No SQL statements found in the provided file.")

    # Everything except the last statement are setup statements (CREATE TEMP VIEW).
    # The final statement is the main SELECT that returns one row per transaction.
    setup_statements = statements[:-1]
    final_select = statements[-1]

    # --- Execute SQL on a SINGLE connection ------------------------------
    # IMPORTANT: TEMP VIEWs are scoped to a SQLite connection.
    # We must execute the CREATE TEMP VIEW statements and the final SELECT
    # on the SAME connection object. con.executescript() issues an implicit
    # COMMIT but that does NOT destroy TEMP VIEWs — they persist for the
    # connection lifetime.
    with sqlite3.connect(db_path) as con:
        # Execute each setup statement individually (avoids executescript
        # quirkiness with semicolons inside strings)
        for stmt in setup_statements:
            con.execute(stmt)

        df = pd.read_sql_query(final_select, con)

    if df.empty:
        raise RuntimeError("Final SELECT returned no rows. Check your data and SQL.")

    print(f"Loaded {len(df):,} transactions with {df.shape[1]} columns from SQL pipeline.")

    # --- Feature engineering for Isolation Forest ------------------------
    # These are the behavioural features derived by the SQL views:
    #   amount       : raw transaction amount
    #   tx_count     : total number of transactions by this user
    #   avg_amount   : user's average transaction amount
    #   total_amount : user's total spend
    #   daily_tx     : number of transactions the user made on this date
    #   daily_amount : user's total spend on this date
    feature_cols = ["amount", "tx_count", "avg_amount", "total_amount", "daily_tx", "daily_amount"]
    X = df[feature_cols].fillna(0).astype(float)

    # --- Isolation Forest (unsupervised anomaly detection) ---------------
    # contamination=0.02 means we expect ~2% of transactions to be anomalies.
    # random_state=7 ensures reproducibility.
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,   # ~2% of transactions flagged as anomalies
        random_state=7,
    )
    model.fit(X)

    # predict() returns -1 for anomalies, +1 for normal observations
    df["is_anomaly"] = (model.predict(X) == -1).astype(int)

    # decision_function() returns the anomaly score:
    #   more negative  → more anomalous
    #   more positive  → more normal
    # We invert and normalise to [0, 1] so that:
    #   anomaly_score → 1  means highly suspicious
    #   anomaly_score → 0  means very normal
    raw_scores = -model.decision_function(X)          # flip: higher = more anomalous
    score_min = raw_scores.min()
    score_max = raw_scores.max()
    df["anomaly_score"] = (raw_scores - score_min) / (score_max - score_min + 1e-9)

    # --- Rank transactions by suspicion level (highest first) ------------
    df_sorted = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    df_sorted["anomaly_rank"] = df_sorted.index + 1   # 1 = most suspicious

    n_anomalies = df["is_anomaly"].sum()
    pct_anomalies = 100.0 * n_anomalies / len(df)
    print(f"Anomalies detected: {n_anomalies:,} ({pct_anomalies:.2f}% of transactions)")

    # --- Save fraud_scores.csv -------------------------------------------
    # Full transaction-level output: original fields + features + scores
    scores_cols = [
        "tx_id", "user_id", "date", "region", "merchant", "amount",
        "tx_count", "avg_amount", "total_amount", "daily_tx", "daily_amount",
        "anomaly_score", "is_anomaly", "anomaly_rank",
    ]
    save_csv(df_sorted[scores_cols], Path(outdir) / "fraud_scores.csv")
    print(f"  Saved: {Path(outdir) / 'fraud_scores.csv'}")

    # --- Save fraud_summary.csv ------------------------------------------
    # User-level summary derived from the Isolation Forest model output.
    # Only includes users who had at least one transaction flagged as anomalous.
    anomaly_df = df[df["is_anomaly"] == 1]
    user_summary = (
        df.groupby("user_id")
        .agg(
            total_transactions=("tx_id", "count"),
            anomalous_transactions=("is_anomaly", "sum"),
            max_anomaly_score=("anomaly_score", "max"),
            avg_anomaly_score=("anomaly_score", "mean"),
            total_spend=("amount", "sum"),
            avg_transaction_amount=("amount", "mean"),
            max_transaction_amount=("amount", "max"),
        )
        .reset_index()
    )
    # Only report users with at least one anomalous transaction
    user_summary = user_summary[user_summary["anomalous_transactions"] > 0].copy()
    user_summary["anomaly_rate_pct"] = (
        100.0 * user_summary["anomalous_transactions"] / user_summary["total_transactions"]
    ).round(2)
    user_summary = user_summary.sort_values("max_anomaly_score", ascending=False).reset_index(drop=True)
    save_csv(user_summary, Path(outdir) / "fraud_summary.csv")
    print(f"  Saved: {Path(outdir) / 'fraud_summary.csv'}")
    print(f"  Users with anomalous transactions: {len(user_summary)}")

    # --- Visualisation ---------------------------------------------------
    chart_path = charts_dir / "fraud_distribution.png"
    plot_hist(
        scores=df_sorted["anomaly_score"],
        flags=df_sorted["is_anomaly"],
        title="Anomaly Score Distribution\n(Isolation Forest — higher score = more suspicious)",
        out=chart_path,
    )
    print(f"  Saved: {chart_path}")

    print(f"\nAll artefacts written to: {Path(outdir).resolve()}")
    print("\nTop 10 most suspicious transactions:")
    print(
        df_sorted[["anomaly_rank", "tx_id", "user_id", "date", "amount", "anomaly_score", "is_anomaly"]]
        .head(10)
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Unsupervised anomaly detection pipeline: SQL feature engineering "
            "→ Isolation Forest → ranked anomaly scores."
        )
    )
    ap.add_argument("--db", default="fraud.db", help="Path to SQLite database (created by create_db.py)")
    ap.add_argument("--sql", default="src/queries.sql", help="Path to SQL file (feature engineering views)")
    ap.add_argument("--outdir", default="outputs", help="Output directory for CSV and chart artefacts")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args.db, args.sql, args.outdir)


if __name__ == "__main__":
    main()
