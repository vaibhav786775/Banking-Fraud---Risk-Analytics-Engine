"""
create_db.py
------------
Loads transactions.csv into a SQLite database.

Usage:
    python src/create_db.py --csv data/transactions.csv --db fraud.db
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Load a transactions CSV into a SQLite database."
    )
    ap.add_argument("--csv", required=True, help="Path to the input CSV file")
    ap.add_argument("--db", default="fraud.db", help="Path for the output SQLite database")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Could not read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    expected_cols = {"tx_id", "user_id", "date", "region", "merchant", "amount"}
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as con:
        df.to_sql("transactions", con, if_exists="replace", index=False)

    print(f"Loaded {len(df):,} rows from {csv_path} -> {db_path}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
