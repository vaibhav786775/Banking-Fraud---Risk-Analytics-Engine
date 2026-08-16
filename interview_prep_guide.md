# JPMorgan CADP Interview Prep — Fraud Detection SQL + Python (Unsupervised)

> **Based on actual source code, not README alone.**  
> Every claim is traced to `create_db.py`, `queries.sql`, `detect_fraud_unsupervised.py`, `utils.py`, `transactions.csv`, and the generated outputs.

---

# PART 1 — PROJECT AT A GLANCE

## 1. What problem is this project solving?

Banks process millions of transactions daily. A small subset of those transactions are fraudulent — unauthorized transfers, stolen card usage, account takeover activity. The challenge: **no one labels each transaction as fraud in real time**. Investigators only discover fraud after the fact, sometimes days or weeks later.

This project builds an automated pipeline that scores every transaction for suspiciousness using behavioral patterns alone, without needing labeled fraud examples.

## 2. Why is fraud detection important in banking?

- Global payment card fraud losses exceed **$33 billion annually** (Nilson Report)
- Early detection limits monetary loss, reputational damage, and regulatory penalties
- Banks are legally required (AML/KYC regulations) to monitor for suspicious activity
- JPMorgan Chase processes over **6 billion transactions per year** — manual review is impossible

## 3. What exactly does this project detect?

It detects **transactions that are statistically anomalous** relative to the same user's own behavioral baseline. Specifically, transactions where the amount or frequency is unusually high compared to that user's historical average. **It does not confirm fraud.** It generates a ranked list of statistically unusual transactions for further investigation.

## 4. Why is this an unsupervised learning problem?

In the real world, when a transaction occurs, it has no label. A fraud analyst may investigate it 2 weeks later and mark it as fraud — but by then, thousands more transactions have happened. At the point of detection, you have:

- Transaction data: ✅ available
- A "fraud" label: ❌ not available

Since we have no labeled examples of fraud to train on, supervised classification (Logistic Regression, XGBoost, etc.) cannot be used directly. We must infer anomalous behavior from the structure of the data itself — that is unsupervised learning.

## 5. Why can't I simply use a supervised classifier?

Three reasons:

1. **No labels**: You'd need thousands of confirmed fraud examples as training data. Most banks have some, but they take time to accumulate and are often imbalanced (fraud is 0.1–1% of transactions).
2. **Label delay**: A transaction must be investigated before being labeled. Real-time fraud scoring can't wait days for a label.
3. **Novel fraud patterns**: Supervised models learn patterns from *past* fraud. New fraud methods (e.g., a new type of account takeover) won't be in the training data and will be missed. Unsupervised detection finds anything statistically unusual, including new patterns.

## 6. Complete end-to-end architecture

```
transactions.csv (67,818 rows × 6 cols)
        │
        ▼  [create_db.py — argparse + pandas + sqlite3]
   fraud.db  ← SQLite database, table: transactions
        │
        ▼  [queries.sql — executed on ONE connection]
   CREATE TEMP VIEW user_stats  →  per-user: tx_count, avg_amount, total_amount
   CREATE TEMP VIEW daily_user  →  per-user per-day: daily_tx, daily_amount
   SELECT ... JOIN → 67,818 rows × 11 columns (one enriched row per transaction)
        │
        ▼  [detect_fraud_unsupervised.py — pandas + sklearn]
   Feature matrix X: 6 numeric columns × 67,818 rows
        │
        ▼  IsolationForest(n_estimators=200, contamination=0.02, random_state=7)
           .fit(X)          — builds 200 isolation trees
           .predict(X)      — returns -1 (anomaly) or +1 (normal) per transaction
           .decision_function(X) — returns raw anomaly score (negative = anomalous)
        │
        ▼  Score transformation: flip sign, min-max normalize → anomaly_score ∈ [0,1]
        │
        ▼  Sort by anomaly_score descending → anomaly_rank 1..67818
        │
        ├──▶ outputs/fraud_scores.csv     (67,818 rows × 14 cols)
        ├──▶ outputs/fraud_summary.csv    (323 users × 9 cols)
        └──▶ outputs/charts/fraud_distribution.png
```

## 7. What happens step by step?

1. `transactions.csv` is read into a pandas DataFrame (67,818 rows, 6 columns)
2. DataFrame is written to SQLite table `transactions` using `df.to_sql()`
3. Python opens **one** SQLite connection
4. `CREATE TEMP VIEW user_stats` — computes 3 aggregate stats per user
5. `CREATE TEMP VIEW daily_user` — computes 2 aggregate stats per user per date
6. Final `SELECT` joins all three sources → one enriched row per transaction (11 columns)
7. `pd.read_sql_query()` loads the result into a pandas DataFrame
8. SQLite connection closes; TEMP VIEWs are automatically destroyed
9. Feature matrix `X` = 6 numeric columns; `fillna(0).astype(float)`
10. `IsolationForest.fit(X)` trains 200 isolation trees on the 67,818-row dataset
11. `.predict(X)` labels each transaction: `-1` = anomaly, `+1` = normal → `is_anomaly` column
12. `.decision_function(X)` returns raw scores → flip sign → min-max normalize → `anomaly_score ∈ [0,1]`
13. DataFrame sorted descending by `anomaly_score` → `anomaly_rank` assigned (1 = most suspicious)
14. `fraud_scores.csv` saved: all 67,818 transactions with 14 columns
15. `fraud_summary.csv` saved: per-user aggregation (only users with ≥1 anomalous transaction)
16. Histogram plotted: blue = normal, red = flagged anomalies

## 8. Inputs

| Input | Description |
|-------|-------------|
| `data/transactions.csv` | Raw transaction data, 67,818 rows, 6 columns |
| `src/queries.sql` | SQL feature engineering (2 TEMP VIEWs + 1 SELECT) |
| `fraud.db` | SQLite database created by `create_db.py` |

## 9. Intermediate outputs

| Intermediate | Description |
|-------------|-------------|
| SQLite `transactions` table | 67,818 rows, 6 columns — exactly mirrors the CSV |
| `user_stats` TEMP VIEW | 400 rows (one per user), 4 columns |
| `daily_user` TEMP VIEW | ~36,000 rows (one per user per day), 4 columns |
| Enriched DataFrame | 67,818 rows × 11 columns |
| Feature matrix `X` | 67,818 × 6 float64 numpy array |

## 10. Final outputs

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `outputs/fraud_scores.csv` | 67,818 | 14 | Every transaction ranked by anomaly |
| `outputs/fraud_summary.csv` | 323 | 9 | Per-user summary for flagged users |
| `outputs/charts/fraud_distribution.png` | — | — | Dual-colour histogram |

## 11. Role of every technology

| Technology | Role |
|-----------|------|
| **SQLite** | Serverless database to persist transactions and run SQL feature engineering |
| **Python** | Pipeline orchestration, argument parsing, ML, output generation |
| **pandas** | Loading SQL result → DataFrame, feature selection, output to CSV |
| **scikit-learn** | Isolation Forest implementation |
| **matplotlib** | Anomaly score distribution histogram |
| **argparse** | Command-line interface for both scripts |
| **pathlib** | Cross-platform file path handling |

---

## 30-Second Explanation

> "I built an unsupervised anomaly detection pipeline for banking transactions. The pipeline has two steps. First, I load 67,000 transactions into SQLite and use SQL queries to engineer behavioral features — like how many transactions a user normally makes per day, and their average spend. Second, I feed those features into an Isolation Forest model, which flags transactions that don't fit the user's normal pattern. The result is a ranked list of the most suspicious transactions, a per-user summary, and a histogram showing the anomaly score distribution. I deliberately used unsupervised learning because there are no fraud labels on the data — which is realistic for real banking scenarios."

## 2-Minute Explanation

> "The project addresses a core banking problem: detecting suspicious transactions without having labeled fraud data.
>
> The pipeline has two main scripts. The first, `create_db.py`, takes a raw CSV of 67,818 bank transactions — with columns for transaction ID, user ID, date, region, merchant, and amount — and loads it into a SQLite database. SQLite was the right choice here because the entire pipeline is local, self-contained, and designed to run from a single clone without any server setup.
>
> The second script, `detect_fraud_unsupervised.py`, first runs SQL feature engineering. I created two temporary SQL views: one computes per-user statistics — total transaction count, average amount, and total spend. The second computes per-user, per-date statistics — daily transaction count and daily spend. These views are joined back to the transaction table so that every transaction row carries its own contextual behavioral features.
>
> The enriched dataset — 67,818 rows with 11 columns — is loaded into pandas. I select 6 numeric features and pass them to Isolation Forest with `contamination=0.02`, meaning I expect roughly 2% of transactions to be anomalous. The model builds 200 isolation trees, and any transaction that gets isolated in fewer splits — meaning it lives in a sparse region of the feature space — receives a high anomaly score.
>
> The output is a ranked CSV of all transactions sorted by suspicion, a user-level summary of anomalous behavior, and a histogram showing the score distribution. The model flagged exactly 1,357 transactions — 2% of 67,818 — consistent with the contamination parameter.
>
> Importantly, I don't claim these are confirmed frauds. They're statistically unusual transactions warranting further investigation, which is the correct framing for an unsupervised anomaly detection system."

---

# PART 2 — MENTAL MODEL: STEP BY STEP

### Step 1: transactions.csv enters the system

**What:** 67,818 rows of raw bank transaction data. 6 columns. No labels, no feature engineering.

**Why:** This is the starting point of any real banking pipeline. Transaction data arrives in structured format from a core banking system, card network, or data warehouse.

**In/Out:** CSV file → pandas DataFrame

**Why this approach:** CSV is the simplest, most portable format for demonstrating the pipeline. In production, this would be a streaming feed (Kafka) or a database query.

**What could go wrong:** Missing columns, malformed dates, negative amounts, duplicate `tx_id`. The code validates expected columns before proceeding.

---

### Step 2: SQLite database creation

**What:** `df.to_sql("transactions", con, if_exists="replace", index=False)` writes all 67,818 rows to a `transactions` table inside `fraud.db`.

**Why:** SQL is the standard language for data aggregation in banking analytics. Storing data in SQLite first allows SQL feature engineering to run as proper declarative queries rather than imperative Python loops.

**In/Out:** pandas DataFrame → SQLite table `transactions`

**Why this approach:** SQLite is serverless — no installation, no daemon, no port conflicts. For a self-contained demonstration project run from a single clone, this is ideal.

**What could go wrong:** If `fraud.db` already exists, `if_exists="replace"` drops and recreates the table. This means running `create_db.py` twice gives the same result (idempotent).

---

### Step 3: SQL feature engineering — user_stats view

**What:**
```sql
CREATE TEMP VIEW user_stats AS
SELECT user_id, COUNT(*) AS tx_count, AVG(amount) AS avg_amount, SUM(amount) AS total_amount
FROM transactions GROUP BY user_id;
```
This collapses 67,818 rows into 400 rows — one per user.

**Why:** A single transaction tells you the amount but not whether that amount is unusual *for that specific user*. A $500 transaction is normal for a high-value user and anomalous for a user whose average is $25. User-level statistics create the behavioral baseline.

**In/Out:** `transactions` table → `user_stats` TEMP VIEW (400 rows × 4 cols)

**Why TEMP VIEW instead of a permanent table:** TEMP VIEWs are scoped to the connection. They don't modify the database file and are automatically cleaned up when the connection closes. This keeps the database clean.

**What could go wrong:** If this view is created on connection A and the SELECT query runs on connection B, the view doesn't exist. The code correctly handles this by using **one connection** (`with sqlite3.connect(db_path) as con`).

---

### Step 4: SQL feature engineering — daily_user view

**What:**
```sql
CREATE TEMP VIEW daily_user AS
SELECT user_id, date, COUNT(*) AS daily_tx, SUM(amount) AS daily_amount
FROM transactions GROUP BY user_id, date;
```
This collapses 67,818 rows into approximately 36,000 rows — one per user per date.

**Why:** Daily behavior is a strong fraud signal. A user normally makes 2 transactions/day but suddenly makes 15 in a single day — that's a spike in `daily_tx`. Or their usual $100/day spend jumps to $2,000 — a spike in `daily_amount`. These patterns are invisible unless you aggregate by (user, date).

**In/Out:** `transactions` table → `daily_user` TEMP VIEW (~36,000 rows × 4 cols)

---

### Step 5: Final JOIN — one enriched row per transaction

**What:**
```sql
SELECT t.tx_id, t.user_id, t.date, t.region, t.merchant, t.amount,
       us.tx_count, us.avg_amount, us.total_amount,
       COALESCE(du.daily_tx, 0), COALESCE(du.daily_amount, 0.0)
FROM transactions t
LEFT JOIN user_stats us ON t.user_id = us.user_id
LEFT JOIN daily_user du ON t.user_id = du.user_id AND t.date = du.date
```

**Why:** Every transaction row now carries its full behavioral context. Row 1 (tx_id=129283, U1227, $1,355.93) also carries: this user's total tx count (203), their average amount ($90.82), that day's activity (6 transactions, $1,777.51 total). This is what the ML model needs.

**In/Out:** 3 SQL sources → 67,818 enriched rows × 11 columns

**Why LEFT JOIN not INNER JOIN:** A LEFT JOIN keeps all rows from `transactions` even if there's no matching row in `user_stats` or `daily_user`. With the dataset as-is, every transaction has a matching user and date, so the INNER JOIN result would be identical. But using LEFT JOIN with COALESCE is defensive — if a new transaction arrives for a user with no history, the join doesn't drop that transaction. `COALESCE(du.daily_tx, 0)` converts any NULL into 0.

---

### Step 6: pandas — feature selection and type enforcement

**What:**
```python
feature_cols = ["amount", "tx_count", "avg_amount", "total_amount", "daily_tx", "daily_amount"]
X = df[feature_cols].fillna(0).astype(float)
```

**Why:** Isolation Forest requires a numeric 2D array. Categorical columns (`user_id`, `date`, `region`, `merchant`) are excluded — they are not passed to the model. `.fillna(0)` handles any residual NULLs from the SQL layer. `.astype(float)` ensures all values are float64.

**Shape of X:** 67,818 rows × 6 columns

---

### Step 7: Isolation Forest training and prediction

**What:**
```python
model = IsolationForest(n_estimators=200, contamination=0.02, random_state=7)
model.fit(X)
df["is_anomaly"] = (model.predict(X) == -1).astype(int)
```

**Why:** Isolation Forest identifies transactions that are "easy to isolate" in the 6-dimensional feature space. Normal transactions cluster together; anomalies are sparse and isolated quickly. `predict()` returns `-1` for anomalies and `+1` for normal — converted to 0/1 binary flag.

**Result:** 1,357 transactions flagged as anomalies (exactly 2.00% of 67,818).

---

### Step 8: Anomaly score calculation

**What:**
```python
raw_scores = -model.decision_function(X)    # flip: higher = more anomalous
df["anomaly_score"] = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
```

`decision_function()` returns a float per transaction where **more negative = more anomalous**. We flip the sign so **more positive = more anomalous**, then min-max normalize to [0, 1].

**Result:** `anomaly_score` ranges from 0.0 (most normal) to 1.0 (most suspicious). The most suspicious transaction (tx_id=129283, U1227, $1,355.93) scores exactly 1.0.

---

### Step 9: Ranking

```python
df_sorted = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
df_sorted["anomaly_rank"] = df_sorted.index + 1   # 1 = most suspicious
```

Rank 1 = tx_id 129283, U1227, Feb 8, $1,355.93, score 1.000.

---

### Step 10: User-level summary

From the full scored DataFrame, group by `user_id`:
- Only users with `anomalous_transactions > 0` appear (323 out of 400 users)
- For each: total transactions, anomalous count, max/avg anomaly score, total spend, avg/max amount, anomaly rate %

---

### Step 11: Visualization

`plot_hist()` in `utils.py` splits scores by `is_anomaly` flag:
- Blue bars: 66,461 normal transactions
- Red bars: 1,357 anomalies
- Annotation box: "Total: 67,818 / Flagged: 1,357 (2.0%)"

---

# PART 3 — DATASET DEEP DIVE

## Facts from actual `transactions.csv`

| Property | Value |
|---------|-------|
| **Rows** | 67,818 |
| **Columns** | 6 |
| **Date range** | 2024-01-01 → 2024-03-30 (90 days) |
| **Unique users** | 400 |
| **Unique merchants** | 9 |
| **Unique regions** | 4 |
| **Null values** | 0 in all columns |
| **Duplicate tx_id** | 0 |
| **Negative amounts** | 0 |
| **Transactions per user** | min 129, mean 169.5, max 209 |
| **Transactions per day** | min 601, mean 753.5, max 898 |

## Amount statistics

| Stat | Value |
|-----|-------|
| Min | $5.00 |
| 25th pct | $36.75 |
| Median | $70.88 |
| Mean | $81.71 |
| 75th pct | $105.61 |
| Max | $2,077.79 |
| Std dev | $96.02 |

The mean ($81.71) vs max ($2,077.79) indicates a right-skewed distribution with high-value outliers — exactly where Isolation Forest will focus.

## Merchants (9 unique)

`CafeX`, `Electronics`, `Grocery`, `OnlineShop`, `Pharmacy`, `RideShare`, `StoreA`, `StoreB`, `TravelCo`

## Regions (4 unique)

`East`, `North`, `South`, `West`

## Column Analysis

### `tx_id` — Transaction ID (integer)
- What: Unique identifier for each transaction
- Fraud relevance: Useful as a key for joining investigation results; duplicate `tx_id` could indicate data pipeline errors or transaction replay attacks
- Used by model: **No** — it's a key, not a feature

### `user_id` — User identifier (string, e.g. "U1227")
- What: Identifies the account holder making the transaction
- Fraud relevance: The model's entire approach is behavioral baseline per user — `user_id` is the grouping key in SQL
- Used by model: **No** — string, used as GROUP BY key in SQL only

### `date` — Transaction date (string, YYYY-MM-DD)
- What: The calendar date of the transaction
- Fraud relevance: Date is the second grouping dimension for daily behavior. Sudden spikes on one specific date indicate unusual activity
- Used by model: **No** — string, used as GROUP BY key in SQL. The *derived* features (`daily_tx`, `daily_amount`) capture temporal behavior numerically

### `region` — Geographic region (string)
- What: One of East, North, South, West
- Fraud relevance: Geographic anomalies are a real fraud signal (transaction from an unusual region). In this project, `region` is **not used as a model feature** — it's only carried through to the output CSV for analyst reference
- Used by model: **No** — categorical, not encoded

> **Important interview point:** `region` and `merchant` are present in the output but are NOT fed into the Isolation Forest. Only 6 numeric features are used. Be precise about this.

### `merchant` — Merchant category (string)
- What: One of 9 merchant types
- Fraud relevance: Unusual merchant categories are a real fraud signal (e.g., someone who only ever shops at Grocery suddenly making a large TravelCo transaction). Not currently used as a model feature
- Used by model: **No** — categorical, not encoded

### `amount` — Transaction amount (float, USD)
- What: Dollar value of the transaction
- Fraud relevance: Directly used as the primary feature. High absolute amount is a basic fraud signal, and it's also the basis for the derived features
- Used by model: **Yes** — directly as the `amount` feature

## Data quality notes

This dataset is clean and synthetic. In a real banking environment:
- Amounts could be 0 (authorization holds) or negative (refunds/chargebacks)
- Dates would be timestamps (millisecond precision), not date strings
- User IDs would be hashed/encrypted account numbers
- Merchant data would include MCC codes, geographic coordinates
- Duplicate transactions are a legitimate fraud type (double-charging)
- Missing values would occur due to failed data pipeline stages

---

# PART 4 — DATABASE / SQLITE

## What is SQLite?

SQLite is a **serverless, embedded relational database engine**. The entire database is stored in a single file (`fraud.db`). There is no separate database process — the SQLite library runs inside your application's process.

## Why SQLite for this project?

| Criterion | SQLite ✅ | MySQL ❌ | PostgreSQL ❌ |
|----------|----------|---------|------------|
| Setup required | Zero | Server install + config | Server install + config |
| Works from `git clone` | Yes | No | No |
| Single file | Yes | No | No |
| Good for 67K rows | Yes | Overkill | Overkill |
| Concurrent writes | Not needed | Yes | Yes |
| Network access | Not needed | Yes | Yes |

**Strong answer:** "For this project, SQLite was the obvious choice. The goal is a self-contained demonstration that works from a single `git clone` without any external services. SQLite requires zero installation — the library ships with Python's standard library. The 67,818-row dataset fits comfortably in SQLite's performance envelope. MySQL or PostgreSQL would require a running server, credentials, network configuration, and environment-specific setup — all unnecessary complexity for what is essentially a local analytical pipeline."

## What is `fraud.db`?

It's a single binary file that contains:
- The `transactions` table (67,818 rows, 6 columns)
- SQLite's internal catalog (table schema, indexes)
- Any temporary structure created during connection

## Python ↔ SQLite

```python
import sqlite3
con = sqlite3.connect("fraud.db")   # creates file if it doesn't exist; opens it if it does
con.execute("SELECT ...")           # execute single statement
con.executescript("...;...;")       # execute multiple statements (implicit COMMIT before each)
pd.read_sql_query("SELECT ...", con) # pandas shortcut: execute + fetch → DataFrame
con.close()                         # releases file lock
```

In the code, `with sqlite3.connect(db_path) as con:` is a context manager. When the `with` block exits (normally or by exception), the connection auto-commits and closes.

## What is a cursor?

A cursor is the object that actually executes SQL and holds the result rows. `con.execute()` is syntactic sugar that creates an internal cursor. `pd.read_sql_query()` uses its own cursor internally.

## When SQLite is NOT appropriate

- **Multiple concurrent writers**: SQLite uses file-level locking. Multiple processes writing simultaneously will serialize and may deadlock
- **Network access**: SQLite is file-based; it can't be accessed over a network without additional tooling
- **Very large datasets**: Multi-billion-row tables will be slow due to lack of parallel query execution
- **Production banking**: A real bank's transaction database is Oracle, PostgreSQL, or a specialized column store (Snowflake, BigQuery). SQLite would never appear in a production fraud detection system

---

# PART 5 — CREATE_DB.PY LINE BY LINE

```python
import argparse, sqlite3, sys
from pathlib import Path
import pandas as pd
```
**Lines 9–14:** Standard library imports. `argparse` for CLI, `sqlite3` for database, `pathlib.Path` for cross-platform file paths, `pandas` for CSV reading and `to_sql`.

---

```python
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Load a transactions CSV into a SQLite database.")
    ap.add_argument("--csv", required=True, help="Path to the input CSV file")
    ap.add_argument("--db", default="fraud.db", help="Path for the output SQLite database")
    return ap.parse_args()
```
**Lines 17–23:** Defines the command-line interface. `--csv` is required (no default). `--db` defaults to `"fraud.db"` so a user can omit it. `parse_args()` reads `sys.argv` and returns a Namespace object where `args.csv` and `args.db` are accessible.

---

```python
csv_path = Path(args.csv)
if not csv_path.exists():
    print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
    sys.exit(1)
```
**Lines 29–32:** Input validation. Converts the string path to a `Path` object. If the file doesn't exist, prints a clear error to `stderr` and exits with code 1. This prevents a cryptic `FileNotFoundError` traceback.

---

```python
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"ERROR: Could not read CSV: {e}", file=sys.stderr)
    sys.exit(1)
```
**Lines 34–38:** Reads the CSV. `pd.read_csv()` handles encoding detection, type inference, and parsing. The `try/except` catches malformed CSV, encoding errors, or permission errors.

---

```python
expected_cols = {"tx_id", "user_id", "date", "region", "merchant", "amount"}
missing = expected_cols - set(df.columns)
if missing:
    print(f"ERROR: CSV is missing required columns: {missing}", file=sys.stderr)
    sys.exit(1)
```
**Lines 40–44:** Column schema validation. Uses set subtraction. If the CSV has all 6 expected columns, `missing` is an empty set (falsy). Any missing columns are reported by name.

---

```python
db_path = Path(args.db)
db_path.parent.mkdir(parents=True, exist_ok=True)
```
**Lines 46–47:** Ensures the directory containing `fraud.db` exists. If `--db output/data/fraud.db` is passed and `output/data/` doesn't exist, this creates it. `exist_ok=True` means no error if the directory already exists.

---

```python
with sqlite3.connect(db_path) as con:
    df.to_sql("transactions", con, if_exists="replace", index=False)
```
**Lines 49–50:** 

- `sqlite3.connect(db_path)` opens (or creates) `fraud.db`
- `df.to_sql("transactions", con, ...)` generates `INSERT INTO transactions ...` statements internally and executes them in batches
- `if_exists="replace"` — if the `transactions` table already exists, it's dropped and recreated. This makes the script idempotent
- `index=False` — do not write the DataFrame's row index as a column

The `with` block auto-commits and closes the connection on exit.

---

```python
print(f"Loaded {len(df):,} rows from {csv_path} -> {db_path}")
print(f"Columns: {df.columns.tolist()}")
```
**Lines 52–53:** Feedback to the user confirming success and what was loaded.

## Interview: "Walk me through your database creation script"

> "The script has two parts: argument parsing and data loading. For argument parsing, I use Python's `argparse` module to accept `--csv` and `--db` from the command line. The `--csv` path is required; `--db` defaults to `fraud.db`.
>
> For data loading, I first validate inputs: check the CSV exists, read it with pandas, and verify all 6 required columns are present. Then I create the SQLite database using `sqlite3.connect()` and write the DataFrame using `df.to_sql()` with `if_exists='replace'` so the script is idempotent — running it twice gives the same result. The connection is managed with a `with` block, which auto-commits and closes.
>
> The output is a `fraud.db` file containing a `transactions` table with 67,818 rows."

---

# PART 6 — SQL DEEP DIVE

## Statement 1: CREATE TEMP VIEW user_stats

```sql
CREATE TEMP VIEW user_stats AS
SELECT
    user_id,
    COUNT(*)      AS tx_count,
    AVG(amount)   AS avg_amount,
    SUM(amount)   AS total_amount
FROM transactions
GROUP BY user_id;
```

### What `GROUP BY user_id` does

It partitions the 67,818-row table into 400 groups — one per distinct `user_id`. Every aggregate function (`COUNT`, `AVG`, `SUM`) then operates within each group independently.

### What one row represents

One row = one user's entire transaction history across all 90 days.

`U1227`: tx_count=203, avg_amount=$90.82, total_amount=$18,436.29

### `COUNT(*)` — transaction frequency

Counts every row in the group. `COUNT(*)` includes all rows regardless of NULL values. For fraud detection: a user with 203 transactions over 90 days has a very different risk profile than a user with 10 transactions.

### `AVG(amount)` — spending baseline

Average transaction amount for this user. A $1,355 transaction for a user whose baseline is $90 is 15× their average — far more suspicious than the same $1,355 for a user with a $1,200 baseline.

### `SUM(amount)` — total exposure

Total money moved by this user. High-value accounts are both higher targets and can sustain larger fraudulent transactions before being noticed.

---

## Statement 2: CREATE TEMP VIEW daily_user

```sql
CREATE TEMP VIEW daily_user AS
SELECT
    user_id,
    date,
    COUNT(*)    AS daily_tx,
    SUM(amount) AS daily_amount
FROM transactions
GROUP BY user_id, date;
```

### Why grouping by TWO columns

Each combination of (user_id, date) is a unique unit. A user making 5 transactions in one day across 90 days creates 90 rows in `daily_user` — one per day they were active.

### What one row represents

One row = one user's activity on one specific date.
`U1227` on `2024-02-08`: daily_tx=6, daily_amount=$1,777.51

### Why daily behavior matters

- **Transaction velocity:** A user who usually makes 2 transactions/day suddenly makes 15 is engaging in abnormal activity. This is called "velocity" in fraud detection
- **Daily spending spike:** Accumulated daily spend ($1,777.51 for U1227 on Feb 8) is more informative than a single transaction amount, because fraud often involves multiple rapid transactions

---

## Statement 3: Final JOIN query

```sql
SELECT
    t.tx_id, t.user_id, t.date, t.region, t.merchant, t.amount,
    us.tx_count, us.avg_amount, us.total_amount,
    COALESCE(du.daily_tx,     0)   AS daily_tx,
    COALESCE(du.daily_amount, 0.0) AS daily_amount
FROM transactions t
LEFT JOIN user_stats us ON t.user_id = us.user_id
LEFT JOIN daily_user du ON t.user_id = du.user_id AND t.date = du.date
```

### `FROM transactions t`

The driving table — every row in `transactions` will appear in the result. The alias `t` allows `t.tx_id` shorthand.

### `LEFT JOIN user_stats us ON t.user_id = us.user_id`

For each transaction row, look up the matching row in `user_stats` where the user IDs match. Since `user_stats` is grouped by `user_id`, every transaction will find exactly one match — so this behaves like an INNER JOIN in practice. LEFT JOIN is used defensively: if a user somehow had no history, their transaction would still appear with NULLs instead of being silently dropped.

### `LEFT JOIN daily_user du ON t.user_id = du.user_id AND t.date = du.date`

Two-column join condition: both the user_id AND the date must match. This is necessary because the same user could have activity on multiple dates, and we want the stats specifically for that transaction's date.

### `COALESCE(du.daily_tx, 0)`

`COALESCE(expr, fallback)` returns `expr` if it's not NULL, otherwise returns `fallback`. If the daily_user join returns no match (LEFT JOIN gives NULL), daily_tx becomes 0 instead of NULL. This prevents NULL values from propagating into the ML model.

### LEFT JOIN vs INNER JOIN

INNER JOIN would drop any transaction where no matching row exists in the joined table. LEFT JOIN keeps all transactions regardless. In this dataset, the result is identical — but LEFT JOIN + COALESCE is the defensively correct pattern.

---

## TEMP VIEW lifecycle

A `TEMP VIEW` in SQLite:
1. Exists only in memory — not written to the `.db` file
2. Scoped to a single connection — not visible to other connections
3. Automatically dropped when the connection closes

**Critical implementation detail:** In `detect_fraud_unsupervised.py`, all three SQL statements (`CREATE TEMP VIEW user_stats`, `CREATE TEMP VIEW daily_user`, the final `SELECT`) run inside the same `with sqlite3.connect(db_path) as con:` block. This guarantees the views exist when the SELECT runs.

```python
with sqlite3.connect(db_path) as con:
    for stmt in setup_statements:   # creates both TEMP VIEWs
        con.execute(stmt)
    df = pd.read_sql_query(final_select, con)   # runs SELECT on same connection
# connection closes here → TEMP VIEWs destroyed
```

---

# PART 7 — FEATURE ENGINEERING

| Feature | SQL / Code | Granularity | Why useful for anomaly detection |
|---------|-----------|-------------|----------------------------------|
| `amount` | `t.amount` (raw) | Per-transaction | High absolute value is a direct fraud signal; baseline for all user comparisons |
| `tx_count` | `COUNT(*) GROUP BY user_id` | Per-user (all time) | High-frequency users have different risk profiles; unusually low count can indicate compromised dormant accounts |
| `avg_amount` | `AVG(amount) GROUP BY user_id` | Per-user (all time) | Deviation of current transaction from user's own average catches relative anomalies |
| `total_amount` | `SUM(amount) GROUP BY user_id` | Per-user (all time) | Indicates account exposure; high-value users are higher risk targets |
| `daily_tx` | `COUNT(*) GROUP BY user_id, date` | Per-user per-day | Transaction velocity spike; multiple rapid transactions on one day is a classic fraud pattern |
| `daily_amount` | `SUM(amount) GROUP BY user_id, date` | Per-user per-day | Daily spending spike; accumulation of multiple fraudulent charges |

### Example: why combined features beat single amount

Suppose two transactions:

| tx_id | user_id | amount | tx_count | avg_amount | daily_tx | daily_amount |
|-------|---------|--------|----------|------------|----------|--------------|
| A     | U1001   | $1,200 | 200      | $1,100     | 1        | $1,200       |
| B     | U1002   | $1,200 | 150      | $85        | 8        | $4,200       |

Transaction A ($1,200): user normally spends $1,100. Not anomalous.  
Transaction B ($1,200): user normally spends $85 average, and already made 8 transactions today for $4,200 total. **Highly anomalous across multiple dimensions.**

A single-threshold alert on amount would flag both equally. Isolation Forest uses all 6 dimensions simultaneously, so B is correctly far more anomalous.

### Real example from the actual output

The #1 most suspicious transaction:

| tx_id | user_id | amount | tx_count | avg_amount | daily_tx | daily_amount | anomaly_score |
|-------|---------|--------|----------|------------|----------|--------------|---------------|
| 129283 | U1227 | $1,355.93 | 203 | $90.82 | 6 | $1,777.51 | 1.000 |

The `amount` ($1,355.93) is **15×** the user's `avg_amount` ($90.82). The `daily_amount` ($1,777.51) suggests significant concentrated daily activity. The model identified this as the most isolated point in 6-dimensional space.

---

# PART 8 — PANDAS

## Key pandas operations in the code

### `pd.read_sql_query(final_select, con)`
Executes the SQL SELECT on the SQLite connection and returns the result as a DataFrame. Internally: creates a cursor, executes the query, fetches all rows, infers column names and dtypes from the cursor description.

### Feature matrix creation
```python
feature_cols = ["amount", "tx_count", "avg_amount", "total_amount", "daily_tx", "daily_amount"]
X = df[feature_cols].fillna(0).astype(float)
```
- `df[feature_cols]` — selects a 6-column subset of the 11-column DataFrame
- `.fillna(0)` — replaces any NaN with 0 (defensive; COALESCE in SQL already handles this)
- `.astype(float)` — ensures numpy float64 dtype required by sklearn

### Storing model outputs back in DataFrame
```python
df["is_anomaly"] = (model.predict(X) == -1).astype(int)
df["anomaly_score"] = (raw_scores - score_min) / (score_max - score_min + 1e-9)
```
New columns are added directly to `df`. This keeps all original columns (tx_id, user_id, etc.) alongside the model outputs in one DataFrame.

### Sorting and ranking
```python
df_sorted = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
df_sorted["anomaly_rank"] = df_sorted.index + 1
```
`reset_index(drop=True)` gives the sorted DataFrame a clean 0-based index. Then `df_sorted.index + 1` creates a 1-based rank column.

### Groupby aggregation (fraud_summary.csv)
```python
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
```
This uses pandas named aggregation syntax (3.x style) — each key is the output column name, the tuple is `(source_column, aggfunc)`.

## "Why Pandas if SQL was already doing feature engineering?"

> "SQL computed the aggregated features — tx_count, avg_amount, daily_tx, etc. But SQL cannot train a machine learning model. Pandas serves as the bridge between the SQL output and scikit-learn. Additionally, post-model operations — adding the anomaly score back to the original rows, sorting, creating the rank column, groupby aggregation for the summary — are more naturally expressed in pandas than in SQL. SQL is the right tool for set-based aggregation; pandas is the right tool for row-level ML post-processing."

---

# PART 9 — ISOLATION FOREST: FROM BASICS

## What is anomaly detection?

Anomaly detection (also called outlier detection) is the task of identifying data points that are statistically significantly different from the majority. Unlike classification, you don't need labeled examples — you use only the structure of the data.

## What is Isolation Forest?

Isolation Forest (Liu, Ting, Zhou 2008) is an unsupervised anomaly detection algorithm based on the principle:

> **Anomalies are rare and different, therefore easier to isolate.**

## The intuition

Imagine you have a dataset of transaction amounts:
`$100, $120, $95, $110, $105, $50,000`

The value $50,000 is an outlier. If you repeatedly:
1. Pick a random feature (here: just "amount")
2. Pick a random split value within the feature's range
3. Partition the data into "above split" and "below split"

$50,000 will be alone in its partition almost immediately (maybe 2–3 splits). The cluster of values around $100 requires many more splits to isolate any single point (maybe 15–20 splits).

**Path length** = number of splits required to isolate one point.  
Short path length → anomaly. Long path length → normal.

## The algorithm step-by-step

**Training:**
1. Sample 256 data points (by default) from the dataset
2. Build one **isolation tree**:
   a. Randomly select a feature (e.g., `daily_amount`)
   b. Randomly select a split value between the feature's min and max
   c. Partition data into two subsets
   d. Recurse until each point is alone or max tree depth is reached
3. Repeat 200 times (n_estimators=200) with fresh random samples → 200 isolation trees = the **forest**

**Prediction (scoring):**
1. For each transaction, pass it through all 200 trees
2. Record the path length in each tree
3. Average path length across 200 trees
4. Convert average path length to anomaly score using the formula from the paper:
   - `s(x, n) = 2^(-E(h(x)) / c(n))` where `c(n)` is the average path length for a dataset of size n
   - Score near 1.0 → short path → anomaly
   - Score near 0.5 → average path → normal
   - Score near 0.0 → long path → very normal

## Multi-dimensional behavior in this project

With 6 features, the algorithm finds anomalies in 6-dimensional space. A transaction might be perfectly normal on 5 dimensions but extreme on one. Or it might be moderately high on several dimensions simultaneously — both patterns lead to short path lengths.

This is why U1173 on Jan 16 (anomaly_rank 4, amount $556.33, but daily_tx=6, daily_amount=$1,361.44) scores so high: the individual transaction amount isn't extreme, but the daily activity level combined with it creates an unusual combination in feature space.

---

# PART 10 — ISOLATION FOREST PARAMETERS

## Parameters actually used in the code

```python
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=7,
)
```

### `n_estimators=200`

| Property | Detail |
|---------|--------|
| What | Number of isolation trees in the forest |
| Default | 100 |
| Value used | 200 |
| Effect of increasing | More stable scores (less variance), slower training |
| Effect of decreasing | Faster, but scores may vary between runs |
| Why 200 | 200 trees is a common choice for production stability; the dataset (67K rows) is small enough that 200 trees trains in seconds |

### `contamination=0.02`

| Property | Detail |
|---------|--------|
| What | Expected proportion of anomalies in the dataset |
| Value | 0.02 (2%) |
| Effect | Sets the decision threshold for `predict()`. With 0.02, the 2% of transactions with the shortest average path lengths are labeled -1 |
| Effect of increasing | More transactions flagged; higher false positive rate |
| Effect of decreasing | Fewer transactions flagged; may miss real anomalies |
| Why 2% | Industry estimates for financial transaction anomalies range from 0.1% to 5%. 2% is a reasonable, defensible middle ground for a demonstration |

### `random_state=7`

| Property | Detail |
|---------|--------|
| What | Seed for all random number generation (tree splits, feature selection, sampling) |
| Value | 7 (arbitrary) |
| Effect | Ensures identical results on every run |
| Why needed | Without it, two runs produce slightly different anomaly scores due to random sampling in tree construction |

## Parameters not used but worth knowing for interviews

### `max_samples` (default: 'auto' → min(256, n_samples))

How many data points to sample for each tree. With 67,818 rows, this defaults to 256. **Using a subset makes the algorithm O(n log n) rather than O(n²)** — the key efficiency advantage of Isolation Forest.

### `max_features` (default: 1.0)

Fraction of features used per tree. Default 1.0 = all 6 features considered at each split. Can be reduced for speed or diversity.

### `bootstrap` (default: False)

Whether to sample with replacement. Default False uses without replacement.

### `n_jobs` (default: 1)

Number of CPU cores to use. `-1` uses all cores. Not specified in the code, so single-threaded.

---

# PART 11 — ANOMALY SCORE: EXACT EXPLANATION

## What method does the code use?

The code uses **both** `predict()` and `decision_function()`:

```python
# Binary flag: -1 (anomaly) or +1 (normal) → converted to 0/1
df["is_anomaly"] = (model.predict(X) == -1).astype(int)

# Continuous score: float, more negative = more anomalous
raw_scores = -model.decision_function(X)    # flip sign
df["anomaly_score"] = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
```

## What `decision_function()` returns

`decision_function()` returns a float per transaction. The scikit-learn convention:
- **More negative** → shorter average path length → more anomalous
- **More positive** → longer average path length → more normal

Raw range example: approximately `-0.15` (most anomalous) to `+0.12` (most normal).

## Why flip the sign?

After flipping (`-decision_function`), the convention becomes:
- **More positive** → more anomalous

This is the intuitive direction: higher score = more suspicious.

## Min-max normalization formula

```
anomaly_score = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
```

The `+ 1e-9` prevents division by zero if all scores are identical (pathological edge case).

After transformation:
- Minimum raw score → anomaly_score = 0.0 (most normal)
- Maximum raw score → anomaly_score ≈ 1.0 (most anomalous)

**Actual result from this run:** anomaly_score ranges from 0.0 to 0.9999999974643662 (effectively 1.0).

## "What does your anomaly score actually mean?"

> "The anomaly score is a normalized measure of statistical isolation. It's derived from the average path length across 200 isolation trees — how quickly the Isolation Forest can separate that transaction from all others in the 6-dimensional feature space. A score near 1 means the transaction is in a very sparse region of the feature space — it's highly dissimilar from the bulk of transactions and took very few splits to isolate. A score near 0 means the transaction is deeply embedded in the dense cluster of normal behavior.
>
> The score has been min-max normalized so the most anomalous transaction scores 1.0 and the most normal scores 0.0. It is NOT a probability. It does not represent the probability of fraud."

## "Is the anomaly score a probability of fraud?"

> "No. It is not a probability in any mathematical sense — it doesn't sum to 1, it's not calibrated against a known distribution of fraud, and it has no direct interpretation as 'X% chance of being fraud.' It's a relative isolation measure. The transaction with score 1.0 is the most isolated point in the 67,818-transaction feature space, but that doesn't mean it's definitely fraudulent. It means it warrants investigation."

---

# PART 12 — CONTAMINATION

## What does `contamination=0.02` mean?

Contamination is the **assumed fraction of anomalies** in the dataset. It's used to set the decision threshold for `predict()`.

Mechanically, after computing anomaly scores for all 67,818 transactions, Isolation Forest finds the 2nd percentile of scores (from the anomalous end) and uses that as the threshold. Transactions above the threshold get labeled `-1` (anomaly).

**Result in this project:** The bottom 2% of path lengths = the 1,357 most isolated transactions = labeled as anomalies.

## Contamination ≠ true fraud rate

`contamination=0.02` does NOT mean:
- "2% of these transactions are fraudulent" ❌
- "2% fraud is the historical rate in this bank" ❌
- "My model has 98% accuracy" ❌

It means: "I am telling the model to treat the 2% most anomalous-looking transactions as anomalies for the purpose of the threshold decision."

## What if contamination is too high?

If `contamination=0.10`:
- 6,781 transactions flagged (10x more)
- Many normal transactions incorrectly labeled as anomalies
- High false positive rate
- Analysts waste time investigating normal behavior
- The model becomes less useful as an alert tool

## What if contamination is too low?

If `contamination=0.001`:
- Only 68 transactions flagged
- Many genuinely suspicious transactions missed
- Low false positive rate but high false negative rate
- Real fraud slips through

## "How did you decide that 2% of transactions are anomalous?"

> "Honestly, the 2% is not derived from a known fraud rate — in a truly unsupervised setting, I don't know the fraud rate. I chose 2% based on common industry guidance for financial transaction anomalies, which typically range from 0.1% to 5% depending on the institution and context. 2% is a conservative middle-ground choice that flags enough transactions to be useful without generating an overwhelming alert volume.
>
> In a real banking context, I would tune contamination using one of: (1) domain expert input on expected fraud rates, (2) historical investigation outcomes if any labeled data exists, or (3) cost-based optimization — balancing the cost of investigation per alert against the cost of missing a fraud case."

---

# PART 13 — WHY ISOLATION FOREST?

## vs. Logistic Regression / Supervised classifiers

- **Logistic Regression requires labeled data.** We have none. Cannot be used.
- Also requires balanced classes or class weighting (fraud is rare → severe class imbalance).

## vs. Random Forest (supervised)

- Random Forest is a supervised algorithm. Requires fraud/non-fraud labels.
- Without labels, cannot be trained. Not applicable here.

## vs. XGBoost

- Supervised. Same issue: needs labeled data.
- XGBoost would be the go-to algorithm if labeled data were available.

## vs. K-Means

- K-Means clusters data but doesn't natively define "normal" vs "anomaly."
- You'd have to define: "points far from their cluster centroid are anomalies." This is less principled than Isolation Forest.
- K-Means assumes spherical clusters and is sensitive to feature scaling. More tuning required.
- K-Means requires specifying K — an additional arbitrary hyperparameter.

## vs. DBSCAN

- DBSCAN identifies core points, border points, and noise. Noise points can be treated as anomalies.
- Problem: DBSCAN is extremely sensitive to `eps` (neighborhood radius) and `min_samples`. Poor choices give degenerate results.
- Computationally expensive on 67K rows in 6 dimensions.
- Isolation Forest scales better and requires fewer tuning decisions.

## vs. One-Class SVM

- Trains on "normal" data only (no anomaly examples needed) — conceptually similar to Isolation Forest.
- In practice, much slower: O(n²) kernel computations for 67K rows would be impractical without downsampling.
- More hyperparameters (kernel, nu, gamma) require careful tuning.
- Isolation Forest is faster, simpler, and competitive in performance.

## vs. Autoencoders

- A neural network approach: train on normal data, high reconstruction error = anomaly.
- Significantly more complex to implement and tune: architecture, epochs, learning rate, loss function.
- Requires deep learning infrastructure; overkill for a tabular dataset with 67K rows and 6 features.
- Not interpretable enough for an interview-level project or banker explanation.

## Strong answer: "I chose Isolation Forest because..."

> "I chose Isolation Forest for three reasons. First, it's designed specifically for anomaly detection in an unsupervised setting — it doesn't require fraud labels. Second, it's efficient: the algorithm samples subsets of 256 points per tree, giving roughly O(n log n) complexity, which trained in seconds on 67,000 transactions. Third, it's interpretable at a high level: a transaction is anomalous because it's 'easy to isolate' — meaning it lives in a sparse region of the feature space. That's an explanation I can give to a fraud analyst or a bank manager without needing to explain neural network weights. For a production environment with labeled data, I'd move to XGBoost or a neural network, but for this unsupervised demonstration, Isolation Forest is the principled, efficient, and defensible choice."

---

# PART 14 — FRAUD VS ANOMALY

## Precise vocabulary

| Term | Meaning in this project |
|------|------------------------|
| **Anomaly** | A transaction that is statistically unusual compared to the behavioral baseline |
| **Suspicious transaction** | A flagged anomaly that warrants investigation |
| **Potentially fraudulent** | A suspicious transaction that, after analyst review, could be confirmed as fraud |
| **Confirmed fraud** | A transaction that has been verified as fraudulent by an investigator — **NOT produced by this model** |

## The model does NOT prove fraud

The model says: "This transaction is statistically unusual." It does NOT say: "This transaction is definitely fraudulent."

Tx_id 116124 (U1373, $1,960.64, anomaly_score 0.969) could be:
- A legitimate large electronics purchase
- A fraudulent card-not-present transaction
- A one-time legitimate travel expense

**The model cannot tell the difference.** An analyst must investigate.

## "So does your model actually detect fraud?"

> "The model detects statistical anomalies — transactions that deviate significantly from a user's own behavioral baseline across six dimensions. Whether those anomalies represent actual fraud can only be determined by human investigation. In the banking context, this kind of model generates an alert queue. The top-ranked transactions get reviewed first by fraud analysts, who use additional information (merchant notes, customer contact history, device ID, IP geolocation) to confirm or dismiss each alert. My model's role is to reduce the investigation workload from 67,000 transactions to a prioritized list of the top 1,357 most unusual ones — that's the value it provides."

## How to make this supervised if labels become available

1. **Immediate improvement:** Use `is_anomaly` labels from this model as pseudo-labels to train a supervised classifier (imperfect, but better than nothing)
2. **Better approach:** After analysts review the top anomalies, confirmed fraud cases become labeled training examples
3. **With sufficient labels:** Train XGBoost or Random Forest on the same 6 features plus additional ones (merchant category encoding, time-of-day, geographic distance from last transaction)
4. **Ongoing:** Implement a feedback loop where analyst verdicts continuously retrain the model

---

# PART 15 — MODEL EVALUATION

## Why supervised metrics don't apply

- **Accuracy:** "97% accuracy" sounds great but is meaningless — if you predict "not fraud" for every single transaction, you achieve 98% accuracy (since only 2% are anomalies). This is the class imbalance problem.
- **Precision:** TP / (TP + FP) — requires knowing which anomalies are true positives (actual fraud). We don't have labels.
- **Recall:** TP / (TP + FN) — requires knowing actual fraud cases. We don't.
- **F1:** Harmonic mean of precision and recall — same issue.
- **ROC-AUC:** Requires binary labels for each data point. Not available.

## What the project actually validates (sanity checks)

1. **Contamination consistency:** Exactly 2.00% of transactions flagged = 1,357 / 67,818 ✅
2. **Score distribution:** Scores range from 0.0 to 1.0, concentrated near 0 (most are normal) ✅
3. **Reproducibility:** Two runs with same `random_state=7` produce identical results ✅
4. **Behavioral plausibility:** Top anomalies have high amounts AND high daily activity — behaviorally unusual combinations ✅
5. **Rank stability:** The top 10 consistently involve U1227, U1252, U1373, U1173 — high-spend users with concentrated activity ✅

## What you would do in a real bank

1. **Precision@K:** After getting analyst verdicts on the top K alerts, calculate what fraction were true fraud. Even without full label coverage, this gives a working precision estimate.
2. **Cost-based evaluation:** (Fraud amount caught) × (detection rate) vs. (investigation cost) × (alerts generated)
3. **A/B testing:** Run new model alongside existing rules engine; compare catch rates
4. **Historical backtesting:** If confirmed historical fraud records exist, run the model on historical data and check if known frauds had high anomaly scores

---

# PART 16 — OUTPUT FILES

## `fraud_scores.csv` — 67,818 rows × 14 columns

Sorted descending by `anomaly_score`. Contains every transaction in the dataset.

| Column | Source | Description |
|--------|--------|-------------|
| `tx_id` | transactions | Transaction identifier |
| `user_id` | transactions | Account holder |
| `date` | transactions | Transaction date |
| `region` | transactions | Geographic region |
| `merchant` | transactions | Merchant type |
| `amount` | transactions | Raw transaction amount |
| `tx_count` | user_stats view | User's total transaction count (all time) |
| `avg_amount` | user_stats view | User's average transaction amount |
| `total_amount` | user_stats view | User's cumulative spend |
| `daily_tx` | daily_user view | Transactions by user on that date |
| `daily_amount` | daily_user view | User's spend on that date |
| `anomaly_score` | IsolationForest | Normalized [0,1], higher = more suspicious |
| `is_anomaly` | IsolationForest | 1 = flagged anomaly, 0 = normal |
| `anomaly_rank` | pandas sort | 1 = most suspicious |

**Top row:** tx_id=129283, U1227, 2024-02-08, South, StoreB, $1,355.93, anomaly_score=1.0, is_anomaly=1, anomaly_rank=1

## `fraud_summary.csv` — 323 rows × 9 columns

One row per user with at least one flagged transaction. Sorted by `max_anomaly_score` descending.

| Column | Description |
|--------|-------------|
| `user_id` | Account identifier |
| `total_transactions` | Total transactions across all 90 days |
| `anomalous_transactions` | Count flagged by IsolationForest |
| `max_anomaly_score` | Highest anomaly score across all their transactions |
| `avg_anomaly_score` | Mean anomaly score across all their transactions |
| `total_spend` | Cumulative spend |
| `avg_transaction_amount` | Average transaction amount |
| `max_transaction_amount` | Highest single transaction |
| `anomaly_rate_pct` | `(anomalous_transactions / total_transactions) × 100` |

**Top row:** U1227 — 203 transactions, 66 anomalous (32.51%), max score 1.0, total spend $18,436.29

Note: 77 users (400 - 323) had zero anomalous transactions — they don't appear in this file.

## `fraud_distribution.png`

- **X-axis:** Anomaly score (0.0 = most normal, 1.0 = most suspicious)
- **Y-axis:** Number of transactions
- **Blue bars:** 66,461 normal transactions — cluster heavily near 0 (most transactions are normal)
- **Red bars:** 1,357 anomalies — concentrated in the 0.4–1.0 range, with some near-normal borderline cases
- **Annotation box:** "Total: 67,818 / Flagged: 1,357 (2.0%)"

**Shape:** Right-skewed distribution. The bulk of transactions have scores near 0 (normal). A thin right tail represents anomalies. The clear bimodal visual separation between blue and red confirms the model is distinguishing two behavioral populations.

---

# PART 17 — END-TO-END CODE WALKTHROUGH

> "First, I load the transaction data. Running `python src/create_db.py --csv data/transactions.csv --db fraud.db`, the script validates that the CSV exists and contains all 6 required columns, then reads it into a pandas DataFrame with `pd.read_csv()`. It opens a SQLite connection to `fraud.db` and writes all 67,818 rows to a `transactions` table using `df.to_sql()` with `if_exists='replace'` to ensure idempotency. The connection closes and the database is ready.
>
> Second, I run the analysis. `python src/detect_fraud_unsupervised.py --db fraud.db --sql src/queries.sql --outdir outputs` first validates both input paths exist. Then it reads `queries.sql` and splits the file on semicolons into individual statements. The first two are `CREATE TEMP VIEW` statements; the last is the `SELECT`.
>
> Still on one SQLite connection, it executes the two `CREATE TEMP VIEW` statements: `user_stats` computes per-user statistics (tx_count, avg_amount, total_amount) and `daily_user` computes per-user-per-day statistics (daily_tx, daily_amount). Then it runs the final `SELECT` which LEFT JOINs both views back to the transactions table, producing one enriched row per transaction with 11 columns. This is loaded into a pandas DataFrame using `pd.read_sql_query()`. The connection closes, the TEMP VIEWs are destroyed.
>
> Third, I prepare the feature matrix. I select 6 numeric columns from the 11-column DataFrame: `amount, tx_count, avg_amount, total_amount, daily_tx, daily_amount`. I apply `fillna(0)` and `astype(float)` to produce a clean 67,818 × 6 float64 array called X.
>
> Fourth, I train Isolation Forest with 200 trees, 2% contamination, and random_state=7 for reproducibility. `model.fit(X)` builds 200 isolation trees. Then `model.predict(X)` returns -1 for anomalies and +1 for normal transactions — I convert this to a binary 0/1 `is_anomaly` column. `model.decision_function(X)` returns a raw float score where more negative means more anomalous. I flip the sign and min-max normalize to create `anomaly_score` in [0,1].
>
> Fifth, I sort all 67,818 rows by `anomaly_score` descending and assign `anomaly_rank` from 1 (most suspicious) to 67,818 (most normal).
>
> Finally, I save three outputs: `fraud_scores.csv` with all 14 columns; `fraud_summary.csv` with per-user aggregations for the 323 users who had at least one anomaly; and `fraud_distribution.png`, a dual-colour histogram showing normal (blue) and flagged (red) transactions."

---

# PART 18 — COMPLEXITY

## Time complexity

| Step | Complexity | Notes |
|------|-----------|-------|
| CSV read | O(n) | n = 67,818 rows |
| SQL GROUP BY (user_stats) | O(n log n) | Sort-based grouping |
| SQL GROUP BY (daily_user) | O(n log n) | Sort-based grouping |
| SQL JOIN | O(n log n) | Indexed on user_id |
| Pandas operations | O(n) or O(n log n) | Sort, fillna, groupby |
| IsolationForest.fit() | O(t × ψ × log ψ) | t=200 trees, ψ=256 subsample size — effectively O(1) w.r.t. n |
| IsolationForest.predict() | O(n × t × log ψ) | Linear in n |
| Sort for ranking | O(n log n) | pandas sort_values |

**Overall:** O(n log n) — scales well.

## Space complexity

| Data structure | Size |
|---------------|------|
| transactions.csv | ~3 MB |
| fraud.db | ~3.5 MB |
| DataFrame (enriched) | ~67,818 × 11 × 8 bytes ≈ 6 MB |
| Feature matrix X | ~67,818 × 6 × 8 bytes ≈ 3.2 MB |
| 200 isolation trees | ~200 × 256 × log(256) bytes ≈ a few MB |

Total RAM usage: approximately **50–100 MB** — trivially small.

## "Your dataset has millions of transactions. How would this scale?"

> "The current implementation loads everything into memory, which works for 67K rows but would break at millions of rows. I would make the following changes:
>
> 1. **Database:** Replace SQLite with a column-store database (Snowflake, BigQuery, Redshift) where SQL aggregations run on the database server, not in Python memory
> 2. **Feature engineering:** Keep the same SQL logic but execute it as a scheduled batch query (e.g., dbt) rather than in-process
> 3. **Model training:** Use a chunked training approach or reservoir sampling to fit Isolation Forest on a representative subsample (Isolation Forest's `max_samples=256` already does this per tree)
> 4. **Prediction:** Use `sklearn`'s `predict()` in batches if the full prediction doesn't fit in memory; alternatively serialize the model and deploy it as a microservice
> 5. **Real-time scoring:** For truly real-time fraud detection, the trained model would be serialized (joblib/pickle), deployed as a REST API, and each new transaction scored immediately as it arrives"

---

# PART 19 — PRODUCTION BANKING VERSION

## What my project currently implements

✅ Batch processing of historical CSV data  
✅ SQL feature engineering on a local SQLite database  
✅ Isolation Forest anomaly detection  
✅ Ranked anomaly scores output  
✅ User-level suspicious activity summary  
✅ Anomaly score histogram  
✅ Reproducible results with random_state  
✅ Command-line interface  
✅ Basic error handling

## How I would extend it for production

**NOT currently implemented.** This is speculative design for interview purposes.

### Data Ingestion
Replace CSV → SQLite with a real-time stream from the card network via **Apache Kafka**. Each transaction published to a Kafka topic.

### Storage
- **Operational database (OLTP):** PostgreSQL or Oracle for transaction records with ms-level write latency
- **Analytical database (OLAP):** Snowflake or BigQuery for batch feature computation
- **Feature store:** Tecton or Feast to pre-compute and cache user-level behavioral features, updated incrementally

### Feature Engineering
- Same features (tx_count, avg_amount, daily_tx, etc.) but computed **incrementally** as new transactions arrive
- Sliding window features: last-7-day averages instead of all-time averages — more responsive to recent behavior changes

### Model Serving
- Model trained offline (weekly/daily batch), serialized with `joblib`
- Deployed as a **REST API** (FastAPI) scoring each incoming transaction in <50ms
- For real-time: pre-compute user feature vectors; score = one model inference call

### Alert Generation
- Transactions above threshold → alert record in PostgreSQL `fraud_alerts` table
- Alert routed to fraud analyst queue via case management system
- Priority determined by anomaly_score and transaction amount

### Analyst Review + Feedback Loop
- Analysts mark alerts as "confirmed fraud" or "false positive"
- Labels stored → used to retrain a **supervised model** (XGBoost) alongside the unsupervised one
- Over time, the supervised model catches known fraud patterns; unsupervised model catches novel patterns

### Monitoring
- Model drift detection: monitor distribution of anomaly_score over time; if it shifts, retrain
- Alert fatigue tracking: if >10% of alerts are false positives, reduce contamination or tighten threshold

### Security
- All transaction data encrypted at rest and in transit
- Model API requires authenticated tokens (OAuth2)
- Audit logs for every anomaly score computation

### Explainability
- For each flagged transaction, report which features contributed most to the anomaly score
- SHAP values could be applied post-hoc to explain feature contributions

---

# PART 20 — BANKING/FRAUD DOMAIN QUESTIONS

## Common types of banking fraud

1. **Card-present fraud:** Physical card stolen and used in-store
2. **Card-not-present (CNP) fraud:** Card details stolen for online transactions — most prevalent
3. **Account takeover (ATO):** Attacker gains credentials, hijacks account, initiates transfers
4. **Identity theft:** New account opened using stolen identity
5. **Synthetic identity fraud:** Fake identity created with mix of real/fake data
6. **Money laundering:** Disguising illegal proceeds as legitimate transactions (structuring, layering)
7. **Friendly fraud (chargeback fraud):** Customer claims they didn't receive goods to trigger chargeback

## Transaction velocity

Number of transactions in a short time window. **High velocity = suspicious.** A legitimate user rarely makes 15 transactions in 10 minutes. This maps to `daily_tx` in my project.

## Geographic anomalies

A transaction in New York, then 30 minutes later a transaction in London — physically impossible without time travel. My project has `region` in the output but **does not use it as a model feature**. In production, I would add a "velocity distance" feature.

## False positives

Legitimate transactions flagged as fraud. Consequences:
- Customer inconvenience (card blocked)
- Lost revenue (merchant sale lost)
- Customer churn (people switch banks)
- Investigation cost

## False negatives

Fraudulent transactions NOT flagged. Consequences:
- Direct financial loss to the bank or customer
- Regulatory fines for failure to detect money laundering
- Reputational damage if fraud is public

## Balancing false positives vs false negatives

Banks typically solve this with a **threshold** or **tier system**:
- **High confidence anomaly:** Block transaction automatically
- **Medium confidence:** Trigger step-up authentication (SMS OTP)
- **Low confidence:** Flag for analyst review but don't interrupt transaction

The `contamination` parameter in my project directly controls this balance.

## Risk scoring

A numerical score representing the likelihood of fraud. My `anomaly_score` is a form of risk score, though it's an isolation measure rather than a calibrated probability.

---

# PART 21 — SQL INTERVIEW QUESTIONS

---

### Q: What is a GROUP BY clause and what does it do?

**Short:** Groups rows with the same value in specified columns and applies aggregate functions to each group.

**Deep:** `GROUP BY user_id` takes the 67,818-row transactions table and partitions it into 400 groups — one per distinct `user_id`. Any column in the SELECT that isn't an aggregate function must appear in the GROUP BY clause. Aggregate functions (COUNT, AVG, SUM) compute one value per group and collapse multiple rows into one.

**Follow-up:** "What's the difference between WHERE and HAVING?"  
→ WHERE filters **before** grouping (rows). HAVING filters **after** grouping (groups). Example: `HAVING COUNT(*) > 10` keeps only users with more than 10 transactions.

---

### Q: What is a temporary view and how does it differ from a regular view?

**Short:** A TEMP VIEW exists only for the duration of one database connection and is not persisted to the database file.

**Deep:** A regular `CREATE VIEW` is stored in the database's catalog and visible to all connections. A `CREATE TEMP VIEW` is stored in the connection's temporary schema and automatically dropped when the connection closes. It's useful when you want the organizational clarity of a named view without modifying the database permanently.

**Follow-up:** "Could you use a CTE instead?"  
→ Yes. `WITH user_stats AS (SELECT ...) SELECT ... FROM transactions JOIN user_stats ON ...`. The advantage of CTEs is that they're self-contained in one query. The advantage of TEMP VIEWs is that they can be referenced by name in any subsequent query within the same connection — useful when the Python code splits setup and SELECT into separate `execute()` calls.

---

### Q: Why LEFT JOIN instead of INNER JOIN?

**Short:** LEFT JOIN keeps all rows from the left table regardless of whether a match exists in the right table.

**Deep:** In this query, `FROM transactions t LEFT JOIN user_stats us ON t.user_id = us.user_id` — every row in `transactions` is kept. If a user appeared in transactions but somehow had no matching row in `user_stats` (impossible here since user_stats is derived FROM transactions, but defensively correct), the transaction would still appear with NULL values for `tx_count`, `avg_amount`, `total_amount`. With INNER JOIN, that transaction would be silently dropped — a dangerous loss of data.

**Follow-up:** "What's the difference between LEFT JOIN and FULL OUTER JOIN?"  
→ LEFT JOIN keeps all left-table rows. FULL OUTER JOIN keeps all rows from both tables, with NULLs where there's no match on either side. SQLite supports LEFT JOIN but not natively FULL OUTER JOIN.

---

### Q: What is COALESCE and why is it used here?

**Short:** COALESCE returns the first non-NULL value from its argument list.

**Deep:** `COALESCE(du.daily_tx, 0)` — if the LEFT JOIN on `daily_user` returns no match (NULL), COALESCE returns 0. This prevents NULLs from propagating into the ML feature matrix, where they could cause sklearn errors or incorrect computations.

**Follow-up:** "What's the difference between COALESCE and IFNULL in SQLite?"  
→ `IFNULL(a, b)` is SQLite-specific and equivalent to `COALESCE(a, b)` with only two arguments. COALESCE is ANSI SQL and can take N arguments, returning the first non-NULL.

---

### Q: What does COUNT(*) count vs COUNT(column)?

**Short:** `COUNT(*)` counts all rows including NULLs. `COUNT(column)` counts only non-NULL values in that column.

**Example:** If a table has 5 rows but `amount` is NULL for 2 of them, `COUNT(*)` = 5, `COUNT(amount)` = 3.

**In this project:** `COUNT(*)` is correct — we want to count all transactions, even hypothetically if an amount were NULL.

---

### Q: Can you use a window function to achieve the same result as your GROUP BY approach?

**Short:** Yes. Window functions allow computing user-level stats without collapsing rows.

**Example:**
```sql
SELECT *,
       COUNT(*) OVER (PARTITION BY user_id) AS tx_count,
       AVG(amount) OVER (PARTITION BY user_id) AS avg_amount,
       SUM(amount) OVER (PARTITION BY user_id) AS total_amount,
       COUNT(*) OVER (PARTITION BY user_id, date) AS daily_tx
FROM transactions;
```

**Why I used GROUP BY + JOIN instead:** The TEMP VIEW approach is more modular and readable — each view has a clear name and purpose, making it easier to explain and debug. Window functions produce the same result in one query. Both approaches are valid and interview-defensible.

---

### Q: What is an index and how would it help this query?

**Short:** An index is a sorted data structure that speeds up lookups and joins.

**In this project:** SQLite creates no indexes by default. Adding `CREATE INDEX idx_user ON transactions(user_id)` would speed up the GROUP BY on user_id and the JOIN conditions. With 67,818 rows, the speedup is marginal. At millions of rows, indexes become essential.

---

# PART 22 — PYTHON INTERVIEW QUESTIONS

---

### Q: What is argparse and why use it?

**Short:** argparse is Python's standard library module for parsing command-line arguments.

**In the code:** Both scripts use argparse with `add_argument()`. This allows the pipeline to be invoked with different database paths, SQL files, and output directories without editing source code. An interviewer could ask you to add a `--contamination` argument — the pattern is already established.

---

### Q: What does `sys.path.insert(0, str(Path(__file__).parent))` do?

**Short:** Adds the directory containing the current script (`src/`) to the beginning of Python's module search path.

**Deep:** When `python src/detect_fraud_unsupervised.py` is run from the project root, Python doesn't automatically search `src/` for imports. The line `sys.path.insert(0, str(Path(__file__).parent))` adds `src/` to sys.path before `from utils import ...` executes, making `utils.py` findable.

**Why insert at position 0:** This prioritizes the `src/` directory over system packages, preventing name conflicts (e.g., if there were a package named `utils` installed system-wide).

---

### Q: What is `Path(__file__)` and why is it better than hardcoded paths?

`__file__` is a special Python variable containing the absolute path of the currently executing script. `Path(__file__).parent` is the directory containing that script (`src/`). This is portable — works regardless of where the user runs the script from or what their machine's directory structure looks like.

---

### Q: What does `with sqlite3.connect(...) as con:` do exactly?

The `with` statement invokes a context manager. For `sqlite3`, the context manager:
- **On entry:** Opens the connection
- **On exit (no exception):** Issues `COMMIT`
- **On exit (with exception):** Issues `ROLLBACK`
- **Always:** Closes the connection

---

### Q: What is `df.to_sql()` and what does `if_exists='replace'` mean?

`df.to_sql(table_name, connection)` generates `CREATE TABLE` and `INSERT INTO` SQL statements from the DataFrame and executes them. `if_exists='replace'` = if the table already exists, `DROP TABLE` it first then recreate. Alternatives: `'fail'` (raise error), `'append'` (add rows without dropping).

---

### Q: Why `.fillna(0).astype(float)` on the feature matrix?

`fillna(0)` replaces any NaN values with 0. Even though COALESCE in SQL should have prevented NULLs, defensive coding is good practice. `astype(float)` ensures all values are float64, as scikit-learn's Isolation Forest requires a numeric 2D array (will raise a TypeError if string columns are present).

---

### Q: Explain this line: `df["is_anomaly"] = (model.predict(X) == -1).astype(int)`

Step by step:
1. `model.predict(X)` → numpy array of -1s and +1s, shape (67818,)
2. `== -1` → boolean array: True where anomaly, False where normal
3. `.astype(int)` → converts True→1, False→0
4. `df["is_anomaly"] = ...` → assigns as a new column in the DataFrame

---

### Q: What is `reset_index(drop=True)` doing?

After `sort_values()`, the DataFrame's index retains the original (pre-sort) row numbers, which would be out of order. `reset_index(drop=True)` creates a fresh 0, 1, 2, ... integer index. `drop=True` prevents the old index from being saved as a new column. Without it, `df_sorted.index + 1` would produce incorrect rank numbers.

---

### Q: What is a pandas `groupby().agg()` with named aggregations?

```python
df.groupby("user_id").agg(
    total_transactions=("tx_id", "count"),
    max_anomaly_score=("anomaly_score", "max"),
)
```

This is pandas 0.25+ named aggregation syntax. Each key becomes the output column name. The tuple `(source_column, aggfunc)` specifies what to compute. This is more readable and less error-prone than the older dictionary syntax.

---

# PART 23 — MACHINE LEARNING CROSS-QUESTIONS

## Supervised vs unsupervised

- **Supervised:** Has labeled training data (fraud/not fraud). Learns a mapping input→label.
- **Unsupervised:** No labels. Finds structure/patterns in data alone.
- **This project:** Unsupervised. No labels exist at inference time.

## Anomaly vs outlier detection

Both terms are used interchangeably in literature. Technically:
- **Outlier:** Statistical term, based on distribution (e.g., >3σ from mean)
- **Anomaly:** More general; includes contextual and collective anomalies

## Feature engineering — what it means here

Transforming raw data (transaction records) into features that ML models can learn from. In this project: raw `amount` + aggregated behavioral context = 6 features.

## Feature scaling — does Isolation Forest need it?

Isolation Forest builds random splits and doesn't compute distances or gradients. It is **not sensitive to feature scale** in the way k-NN or SVM are. The model would produce similar results with or without StandardScaler. However, features with very large ranges (e.g., `total_amount` ranges much more than `daily_tx`) will be selected more often for random splits, which could bias the model. For this dataset, the current unscaled approach is defensible.

## Categorical encoding — why not used here

`region` and `merchant` are categorical. To use them in Isolation Forest, they'd need to be encoded (one-hot or ordinal). This was intentionally omitted to keep the feature set interpretable and the model simple. Adding encoded categoricals could improve detection of geographically unusual behavior.

## Train/test split — does it apply?

Not in the traditional sense. In unsupervised learning, we don't hold out a test set (there are no labels to evaluate against). The model is fit and scored on the same dataset. This is acceptable in anomaly detection — you're not trying to generalize to new users (you'd retrain for new data anyway), you're trying to find anomalies in this specific dataset.

## Overfitting and underfitting

In the unsupervised context:
- **Overfitting equivalent:** Contamination too high → model flags too many normal transactions as anomalies
- **Underfitting equivalent:** Contamination too low → model misses real anomalies

## Concept drift / data drift

Over time, user behavior patterns change (seasonal effects, economic changes, new fraud patterns). A model trained in January may be less accurate in December. The solution: retrain periodically. Isolation Forest is cheap to retrain (minutes for 67K rows) — weekly retraining would keep it current.

## Reproducibility

`random_state=7` in IsolationForest seeds all random number generation. Two identical runs with the same data and same random_state produce identical results. Verified: running the pipeline twice produces identical top-10 rankings and identical anomaly counts (1,357).

---

# PART 24 — 50+ HARD INTERVIEW CROSS-QUESTIONS

## Q1: "Tell me about your project."

**What they're testing:** Communication, ability to explain technical work concisely.

**Ideal answer:**
> "I built an unsupervised fraud anomaly detection pipeline for bank transactions. It has two main stages: SQL feature engineering using SQLite — where I compute behavioral baselines for each user (average transaction amount, daily activity patterns) — and an Isolation Forest model that scores every transaction for statistical anomalousness. The output is a ranked list of the most suspicious transactions, a user-level summary, and a visualization. The key design decision was using unsupervised learning because no fraud labels exist — which is realistic for real banking scenarios."

---

## Q2: "Why did you choose Isolation Forest?"

**What they're testing:** Algorithm justification, understanding of alternatives.

**Ideal answer:** See Part 13 — "I chose Isolation Forest because..."

**Follow-up:** "Have you considered XGBoost?"  
→ "XGBoost is supervised — it requires fraud labels. If I had labeled data, XGBoost would be my first choice for its performance and interpretability. In the absence of labels, Isolation Forest is the principled unsupervised alternative."

---

## Q3: "How does Isolation Forest work?"

**What they're testing:** Deep algorithmic understanding.

**Ideal answer:**
> "Isolation Forest builds an ensemble of random trees — in my case, 200 trees. For each tree, it randomly samples 256 data points from the full dataset. Then it repeatedly picks a random feature and a random split value, partitioning the data. The key insight is that anomalies are rare and extreme, so they get isolated — separated into their own partition — after very few splits. Normal points cluster together and require many more splits to isolate. After all 200 trees, each transaction has an average path length — the average number of splits needed to isolate it. Short path length = anomaly. This average path length is converted to the anomaly score."

---

## Q4: "What exactly is your anomaly score?"

**Ideal answer:** See Part 11.

**Follow-up:** "Is it between 0 and 1?"  
→ "Yes. I apply min-max normalization: `(raw_score - min) / (max - min + epsilon)`. So the most anomalous transaction scores exactly 1.0 and the most normal scores 0.0."

---

## Q5: "How do you know your anomalies are fraud?"

**What they're testing:** Honesty, understanding of unsupervised evaluation.

**Ideal answer:**
> "I don't know with certainty. The model identifies statistical anomalies — transactions that deviate from behavioral baselines. Whether those anomalies represent fraud requires human investigation. In a bank, these would be queued for analyst review. The model's value is reducing 67,000 transactions to a prioritized list of 1,357 suspicious ones, making investigation tractable."

---

## Q6: "How did you evaluate the model?"

**Ideal answer:**
> "Since this is unsupervised with no labels, standard metrics like accuracy or F1 don't apply. I validated the model through sanity checks: confirming exactly 2% of transactions were flagged (consistent with `contamination=0.02`), verifying that top-ranked anomalies are behaviorally unusual (high amounts + high daily activity), and confirming reproducibility across runs with `random_state=7`. If I had analyst labels from fraud investigations, I'd compute precision@K — what fraction of the top K flagged transactions turned out to be actual fraud."

---

## Q7: "Why did you use SQLite?"

**Ideal answer:** See Part 4. Emphasize: serverless, zero setup, self-contained, appropriate for this dataset size.

---

## Q8: "How would you scale this to 100 million transactions?"

**Ideal answer:** See Part 18. Replace SQLite with a column store, use feature store for incremental features, deploy model as API.

---

## Q9: "What would you change if I gave you labeled fraud data?"

**Ideal answer:**
> "The pipeline architecture stays the same — same features, same SQL engineering, same pandas preprocessing. I'd add a labeled column to the training data and train a supervised classifier: XGBoost or LightGBM. I'd split data into train/test with stratification (since fraud is rare), tune with cross-validation, and evaluate with precision@K, recall, and AUC-PR (precision-recall AUC is preferred over ROC-AUC for imbalanced data). I'd keep the Isolation Forest running in parallel as a second opinion for novel fraud patterns not seen in training."

---

## Q10: "What is contamination and how did you choose 2%?"

**Ideal answer:** See Part 12. Key point: it's an assumption, not the true fraud rate.

---

## Q11: "Can you explain the COALESCE in your SQL?"

> "COALESCE returns the first non-NULL value from its arguments. I use `COALESCE(du.daily_tx, 0)` because the LEFT JOIN with `daily_user` might return NULL if no matching row exists. COALESCE converts that NULL to 0, preventing NULLs from propagating into the ML feature matrix."

---

## Q12: "Why do you LEFT JOIN instead of INNER JOIN?"

> "LEFT JOIN keeps all rows from the left table — transactions — even if there's no match on the right. With INNER JOIN, if a transaction had no matching user stats (which can't happen in this dataset since user_stats is derived from the same transactions table, but is theoretically possible), that transaction would be silently dropped. Dropping data is dangerous. LEFT JOIN + COALESCE is the defensive, correct pattern."

---

## Q13: "What are TEMP VIEWs and why use them?"

> "A TEMP VIEW is a named query that exists only for the lifetime of one SQLite connection. It's like creating a virtual table alias. I use them to make the feature engineering readable — `user_stats` and `daily_user` are meaningful names. They don't modify the database file, so running the pipeline doesn't pollute the database. The critical point is that the TEMP VIEW creation and the SELECT query must run on the same connection — which my code ensures."

---

## Q14: "Your fraud_scores.csv has 67,818 rows. Why not just output the top 1% suspicious transactions?"

> "I output all transactions for several reasons. First, an analyst might want to set their own threshold — giving them all scores is more flexible. Second, the full ranked list allows threshold sensitivity analysis (how does the flagged count change if I use 1% vs 5%?). Third, the negative result is informative — knowing transaction X scored 0.02 gives confidence it's normal. In a production system, you'd likely only alert on the top K, but for analysis and reporting, the full file is more useful."

---

## Q15: "Why do you use `model.fit(X)` and separately call `predict(X)` and `decision_function(X)` instead of `fit_predict(X)`?"

> "`fit_predict(X)` is equivalent to `fit(X).predict(X)` and returns the same -1/+1 labels. I call `fit()` separately and then both `predict()` and `decision_function()` because I need both the binary label (`is_anomaly`) and the continuous score (`anomaly_score`). `fit_predict()` only gives the binary label. By keeping `fit()` separate, I can reuse the fitted model object for both calls."

---

## Q16: "What happens if random_state is not set?"

> "Without `random_state`, the random number generator uses the system clock as seed. Each run produces slightly different anomaly scores because the tree splits and subsampling are different. The relative ranking of the most extreme anomalies would be stable, but borderline cases could flip. For a demonstration project, reproducibility matters — the same run should produce the same results."

---

## Q17: "Why n_estimators=200 and not the default 100?"

> "More trees produce more stable anomaly scores because each tree is fitted on a random subsample. With 100 trees, there's more variance in the average path length estimates. 200 trees is a common practical choice that balances stability against compute time. On a 67,818-row dataset, training 200 trees takes under 10 seconds, so there's no meaningful cost to doubling from 100."

---

## Q18: "What's the difference between `decision_function()` and `score_samples()` in Isolation Forest?"

> "Both return continuous anomaly scores. `score_samples()` returns the raw anomaly score as defined in the original Isolation Forest paper — values below 0 indicate anomalies. `decision_function()` shifts the score by the `offset_` threshold and returns the decision boundary. Conceptually for ranking purposes they're interchangeable, but `decision_function()` is the sklearn convention for compatibility across different anomaly detection estimators. I use `decision_function()` and then flip and normalize it."

---

## Q19: "Does the amount feature dominate because it has a larger range than daily_tx?"

> "That's a fair concern. `total_amount` ranges from a few hundred to ~$18,000, while `daily_tx` ranges from 1 to ~20. In Isolation Forest, random splits are uniform between the feature's min and max, so features with larger ranges have more possible split values. This could mean `total_amount` gets disproportionate influence. In practice, the random feature selection at each split (all 6 features eligible) and the ensemble averaging mitigates this. Adding StandardScaler would normalize ranges but also reduces interpretability. I kept it unscaled intentionally — the model's output is reasonable, as evidenced by the top anomalies being behaviourally unusual on multiple features simultaneously."

---

## Q20: "What is the `offset_` attribute in Isolation Forest?"

> "After calling `fit()`, `model.offset_` contains the threshold value derived from the contamination parameter. Specifically, it's the (contamination)-th percentile of `score_samples()` on the training data. `predict()` labels a point as -1 (anomaly) if its `score_samples()` value is below `offset_`. You can inspect it: `model.offset_` will be a negative float."

---

## Q21-50: Additional rapid-fire questions with answers

**Q21:** "What does `df.groupby().agg()` return?"  
→ A new DataFrame with the group keys as the index and aggregate values as columns. `.reset_index()` moves the group key back as a regular column.

**Q22:** "What is `from __future__ import annotations`?"  
→ In Python 3.9, `str | Path` type hints would fail without this import; it enables the newer union syntax (`|`) for type hints in older Python versions. Also delays evaluation of annotations.

**Q23:** "Can you explain `Path(outdir) / "charts"`?"  
→ `pathlib.Path` overloads the `/` operator for path joining. Equivalent to `os.path.join(outdir, "charts")` but more readable and cross-platform.

**Q24:** "What is `1e-9` in the normalization formula?"  
→ It's a tiny constant (10⁻⁹) added to the denominator to prevent division by zero in the pathological case where all anomaly scores are identical (max == min). This is called "epsilon-smoothing."

**Q25:** "What if two transactions have the same anomaly score?"  
→ `sort_values()` by default uses the existing order to break ties (stable sort). The rank assignment is sequential — tied scores get consecutive ranks, which is acceptable for this use case.

**Q26:** "Why `if_exists='replace'` in `df.to_sql()`?"  
→ Makes the script idempotent. Running `create_db.py` twice gives the same result — the table is dropped and recreated. Alternative `'append'` would double-count rows.

**Q27:** "What is `sys.exit(1)` vs `sys.exit(0)`?"  
→ `exit(0)` = success. `exit(1)` (or any non-zero) = error. Shell scripts and CI/CD pipelines check the exit code to determine if a command succeeded.

**Q28:** "Why print errors to `sys.stderr` instead of `sys.stdout`?"  
→ stderr is the standard stream for error messages; stdout is for program output. This allows shell users to redirect stdout to a file while still seeing error messages in the terminal.

**Q29:** "What is `pd.read_sql_query()` under the hood?"  
→ Creates a cursor, executes the SQL, fetches all result rows, reads column names from cursor description, creates a DataFrame. Equivalent to: `cursor.execute(sql); data = cursor.fetchall(); df = pd.DataFrame(data, columns=[d[0] for d in cursor.description])`.

**Q30:** "What does `model.predict(X)` return?"  
→ A numpy array of shape (n_samples,) containing -1 (outlier) or +1 (inlier) for each data point.

**Q31:** "Can Isolation Forest predict on new data it hasn't seen?"  
→ Yes. Once `fit()` is called, `predict()` and `decision_function()` can be called on any new data with the same feature schema. The fitted model (200 trees) is reusable. This is how it would work in production: train offline, predict on each new incoming transaction.

**Q32:** "What is `max_samples` in Isolation Forest and why is 256 a good default?"  
→ `max_samples` controls how many data points are sampled to build each tree. The default 'auto' = min(256, n_samples). 256 is derived from the original paper — it's large enough to capture meaningful structure but small enough that trees train very quickly. Increasing it beyond 256 rarely improves anomaly detection quality.

**Q33:** "What does the anomaly score distribution look like for this dataset?"  
→ Right-skewed (left-heavy). Most of the 67,818 transactions cluster near score 0 (normal). A thin right tail represents the 1,357 anomalies. The bimodal visual in the histogram (blue hump near 0, red bars scattered 0.4–1.0) confirms the model is creating meaningful separation.

**Q34:** "Why are 323 out of 400 users in fraud_summary.csv?"  
→ 77 users had zero transactions flagged as anomalies. The summary only includes users with `anomalous_transactions > 0`. This is filtered via `user_summary[user_summary["anomalous_transactions"] > 0]`.

**Q35:** "User U1227 has 66 out of 203 transactions flagged — 32.5% of their activity. Shouldn't that raise a red flag about the model?"  
→ It reveals an important point: Isolation Forest doesn't model per-user baselines — it builds one global model across all 67,818 transactions. A user who consistently makes high-value transactions will have many transactions in the globally anomalous region, even if those transactions are normal for that specific user. In production, you'd implement user-level baselines or user-segmented models.

**Q36:** "What is the difference between `model.fit(X)` and `model.fit_predict(X)`?"  
→ `fit(X)` trains the model and returns the fitted estimator. `fit_predict(X)` trains and returns binary predictions. I use `fit(X)` separately so I can call both `predict()` and `decision_function()` on the same fitted model.

**Q37:** "What would happen if you passed `user_id` (a string) into the feature matrix?"  
→ `astype(float)` would raise a `ValueError`. Even before that, string columns can't be meaningfully split by numeric intervals in an isolation tree. The code correctly excludes all categorical columns from `feature_cols`.

**Q38:** "How would you add `region` as a feature?"  
→ One-hot encode it using `pd.get_dummies(df, columns=['region'])`, then add the resulting binary columns to `feature_cols`. Or use `OrdinalEncoder`. Then retrain the model. This would allow detecting geographically unusual transactions.

**Q39:** "What is the Isolation Forest equivalent of hyperparameter tuning?"  
→ Primarily tuning `contamination` and `n_estimators`. Since there are no labels, you can't use cross-validation with a loss function. Instead, tune using domain knowledge about expected fraud rates and alert volume constraints.

**Q40:** "What version of scikit-learn does this require?"  
→ The requirements.txt specifies `scikit-learn>=1.3`. The project runs on 1.9.0 (installed during testing). All APIs used (`IsolationForest`, `decision_function`, `predict`) are stable across 1.x.

**Q41:** "Can you run this pipeline in parallel?"  
→ The `create_db.py` script is single-threaded. The detection script is single-threaded but `IsolationForest` with `n_jobs=-1` would use all CPU cores for tree training. The pandas operations are single-threaded but could be parallelized with Dask or Modin for very large datasets.

**Q42:** "What is the purpose of `ensure_outdir()` in utils.py?"  
→ Creates the output directory and all parent directories if they don't exist (`mkdir(parents=True, exist_ok=True)`). `exist_ok=True` prevents an error if the directory already exists. Without this, writing to `outputs/charts/fraud_distribution.png` would fail if `outputs/` or `outputs/charts/` didn't exist.

**Q43:** "What would happen if you ran the detection script before running create_db.py?"  
→ `db_path.exists()` check in `run_analysis()` would fail: the script prints `"ERROR: Database not found: fraud.db"` with a suggestion to run `create_db.py` first, then exits with code 1.

**Q44:** "What SQL alternative to TEMP VIEWs did you consider?"  
→ Common Table Expressions (CTEs). The query would be one long `WITH user_stats AS (...), daily_user AS (...) SELECT ...`. Functionally equivalent. TEMP VIEWs were chosen because they're naturally split into separate `con.execute()` calls in Python, matching the two-step CREATE/SELECT pattern. CTEs would require one single complex SQL string.

**Q45:** "What happens to the TEMP VIEWs after the `with` block exits?"  
→ The connection context manager calls `con.commit()` then `con.close()`. Closing the connection destroys all TEMP VIEWs associated with it. They are not persisted to `fraud.db`.

**Q46:** "What is the time complexity of your Isolation Forest training?"  
→ For t trees, each built on ψ=256 samples with max depth d=log₂(256)=8: O(t × ψ × d) = O(200 × 256 × 8) = O(409,600). The prediction (scoring) for n points across t trees is O(n × t × d) = O(67,818 × 200 × 8) ≈ O(100M) — still fast on a modern CPU.

**Q47:** "What is a decision boundary in the context of Isolation Forest?"  
→ The `offset_` value — the score threshold below which points are labeled as anomalies. Set automatically by `contamination`. You can inspect it with `model.offset_` after fitting.

**Q48:** "If two users have identical transaction amounts but one is a high-frequency trader and one is a low-frequency user, how does your model distinguish them?"  
→ The high-frequency trader has high `tx_count` and likely high `total_amount` — different feature vector. The low-frequency user has low `tx_count`. A large transaction for the low-frequency user creates a more extreme feature combination relative to their typical behavior, which the forest will isolate faster.

**Q49:** "What would you do if the model flags 90% of transactions as anomalies due to a bad contamination setting?"  
→ First, lower `contamination` to a more reasonable value (0.01 or 0.005). Second, investigate whether the feature matrix has degenerate values (e.g., division error producing Inf or NaN). Third, examine whether the dataset itself has data quality issues causing the model to see everything as unusual.

**Q50:** "Why does your normalization formula include `+ 1e-9`?"  
→ Division by zero guard. If `score_max == score_min` (all transactions have identical path lengths — theoretically possible with a very homogeneous dataset), the denominator would be 0. Adding `1e-9` prevents a runtime `ZeroDivisionError` without meaningfully affecting any real case.

---

# PART 25 — PROJECT WEAKNESSES

### Weakness 1: No fraud labels — cannot compute true accuracy

**Why it matters:** An interviewer may press: "How do you know your model is good?"  
**Defend:** "This is inherent to the unsupervised setting — the same challenge any bank faces before historical fraud labels are curated. The model is validated through behavioral sanity checks and expert review, which is the industry standard for unsupervised fraud detection systems."  
**Production improvement:** Implement a feedback loop. Analyst verdicts on reviewed alerts become training labels. Over 3–6 months, accumulate enough labels for supervised retraining.

---

### Weakness 2: Global model, not per-user

**Why it matters:** Isolation Forest builds one global model. U1227 has 32% of their transactions flagged — likely because they're a high-value user whose normal behavior looks "globally anomalous" even though it's normal for them.  
**Defend:** "The SQL feature engineering partially addresses this by including the user's own average (`avg_amount`) in the feature vector — so the model does have some sense of the user's baseline. A more sophisticated approach would segment users by spending tier and train separate models."  
**Production improvement:** Segment users into clusters by `tx_count` and `avg_amount`, train a separate Isolation Forest per segment, or use user-level z-score features.

---

### Weakness 3: Contamination is an assumption, not a measurement

**Why it matters:** 2% is not derived from actual fraud data. It's a guess.  
**Defend:** "In a truly unsupervised setting, you cannot know the true contamination rate without labels. 2% is industry-common for a starting point. The model is sensitivity-testable — I can show how the flagged count changes for contamination of 0.5%, 1%, 2%, 5%."  
**Production improvement:** Use precision@K from analyst feedback to empirically tune contamination.

---

### Weakness 4: Offline, batch processing only

**Why it matters:** Real fraud detection needs real-time scoring, not next-day batch.  
**Defend:** "The current implementation is a demonstration of the algorithm and feature engineering logic. The core algorithm (IsolationForest) is online-capable — `predict()` on new data is fast (milliseconds). Converting to real-time requires deployment infrastructure, not a change to the algorithm itself."  
**Production improvement:** Serialize the model with `joblib`, deploy as a FastAPI microservice, score each transaction as it arrives.

---

### Weakness 5: SQLite is not production-ready for this use case

**Why it matters:** SQLite doesn't support concurrent writers, network access, or multi-terabyte datasets.  
**Defend:** "SQLite was chosen explicitly for the demonstration context — zero-setup, self-contained. The SQL feature engineering logic (GROUP BY, JOIN, aggregate functions) is portable — the same queries would run unchanged in PostgreSQL or BigQuery."  
**Production improvement:** Migrate to PostgreSQL (OLTP) + Snowflake (OLAP).

---

### Weakness 6: Limited feature set (no merchant encoding, no time-of-day)

**Why it matters:** The model only uses 6 features. Real fraud detection uses hundreds.  
**Defend:** "The 6 features demonstrate the behavioral baseline approach. The feature set is extensible — adding one-hot encoded merchant categories, time-of-day features, or inter-transaction time would strengthen the model without changing the architecture."  
**Production improvement:** One-hot encode `merchant` (9 categories → 9 binary features), add `hour_of_day`, add `days_since_last_transaction`.

---

### Weakness 7: No explainability for individual predictions

**Why it matters:** When an analyst asks "why was tx_id 129283 flagged?", the current system can't give a feature-level explanation.  
**Defend:** "For the top anomalies, the reason is often obvious from inspecting the feature values — U1227's transaction of $1,355.93 is 15× their average of $90.82. For less obvious cases, SHAP values (TreeExplainer) can be applied post-hoc to Isolation Forest to quantify each feature's contribution."  
**Production improvement:** Add SHAP feature importance for each flagged transaction.

---

### Weakness 8: Synthetic dataset

**Why it matters:** The dataset may not reflect the full complexity of real banking fraud patterns.  
**Defend:** "The synthetic dataset allowed me to demonstrate the complete pipeline cleanly. The algorithm, features, and code would work identically on real transaction data — only the inputs change. The project's value is in the architecture and methodology."

---

# PART 26 — WHAT I SHOULD NOT SAY

| ❌ DON'T SAY | ✅ SAY INSTEAD |
|-------------|--------------|
| "My model has 98% accuracy" | "Since this is unsupervised with no labels, I can't compute accuracy. The model flags 2% of transactions as anomalies, consistent with the configured contamination rate." |
| "Isolation Forest predicts fraud probability" | "Isolation Forest produces an anomaly score — a normalized isolation measure. It is not a probability." |
| "Anomaly means fraud" | "Anomaly means statistically unusual. Whether it's fraud requires human investigation." |
| "2% contamination means 2% of transactions are fraud" | "Contamination is a modeling assumption — I'm telling the model to treat the 2% most isolated transactions as anomalies. It's not a measured fraud rate." |
| "SQLite is better than MySQL" | "SQLite was the right tool for this project — serverless, zero setup, self-contained. MySQL would be appropriate for multi-user production environments." |
| "SQL is only for storing data" | "SQL handles both storage and feature engineering here. The GROUP BY and JOIN logic computes all behavioral features." |
| "Isolation Forest is supervised" | "Isolation Forest is purely unsupervised — it requires no labeled fraud examples." |
| "The model learned what fraud looks like" | "The model learned what normal behavior looks like — anything significantly different from the learned normal distribution is flagged as potentially anomalous." |
| "I achieved F1 score of X" | "Without ground truth labels, F1 is not computable. I validated through behavioral sanity checks." |
| "The model proves this transaction is fraudulent" | "The model ranks this transaction as highly anomalous. Whether it's actual fraud requires analyst investigation." |
| "I used fit_predict" | "I called `fit()`, then separately `predict()` for the binary flag and `decision_function()` for the continuous score." |
| "Region and merchant are model inputs" | "Region and merchant are present in the output CSV for analyst reference, but they are not fed into the Isolation Forest — only the 6 numeric features are." |

---

# PART 27 — RESUME DEFENSE

## Bullet 1: "Built an unsupervised fraud detection pipeline to identify anomalous bank transactions without labeled fraud data using SQL-based feature engineering and Isolation Forest."

### What does this mean?
Complete end-to-end system: data ingestion → SQL feature engineering → unsupervised ML model → outputs.

### Which code supports it?
- "unsupervised": `IsolationForest(contamination=0.02)` with no labeled column in the data
- "SQL-based feature engineering": `queries.sql` with two TEMP VIEWs
- "Isolation Forest": `from sklearn.ensemble import IsolationForest`
- "pipeline": Two scripts, two commands, clean data flow

### Interviewer question: "What makes this unsupervised?"
→ "There is no `fraud_label` column anywhere in the data. The model receives only the 6 numerical features and finds anomalies by structure alone."

### Follow-up: "What would make it supervised?"
→ "Adding a binary label column (`is_fraud: 0/1`) from historical investigations and training on that column as the target variable."

---

## Bullet 2: "Engineered user behavioral features including transaction frequency, average amount, total amount, daily transaction count, and daily spending through SQLite views."

### What does this mean?
5 named features computed by SQL aggregation.

### Which code supports it?

| Resume claim | Code evidence |
|-------------|---------------|
| "transaction frequency" | `COUNT(*) AS tx_count` in `user_stats` view |
| "average amount" | `AVG(amount) AS avg_amount` in `user_stats` view |
| "total amount" | `SUM(amount) AS total_amount` in `user_stats` view |
| "daily transaction count" | `COUNT(*) AS daily_tx` in `daily_user` view |
| "daily spending" | `SUM(amount) AS daily_amount` in `daily_user` view |
| "SQLite views" | `CREATE TEMP VIEW user_stats AS...` and `CREATE TEMP VIEW daily_user AS...` |

All 5 features confirmed in `queries.sql`. All 5 confirmed in `feature_cols` in `detect_fraud_unsupervised.py`.

### Interviewer question: "Why these specific features?"
→ "They capture two dimensions of behavior: overall baseline (tx_count, avg_amount, total_amount) and short-term concentration (daily_tx, daily_amount). A fraud pattern often shows up as an unusual concentration of activity on one day, even if the overall account history looks normal."

---

## Bullet 3: "Automated transaction analysis and generated anomaly scores, ranked suspicious transactions, user-level fraud summaries, and distribution visualizations using Python and Pandas."

### What does this mean?
4 specific deliverables: scores, ranking, summary, visualization.

### Which code supports it?

| Resume claim | Code evidence |
|-------------|---------------|
| "anomaly scores" | `df["anomaly_score"] = ...` → `fraud_scores.csv` column |
| "ranked suspicious transactions" | `df_sorted["anomaly_rank"] = df_sorted.index + 1` → `anomaly_rank` column in `fraud_scores.csv` |
| "user-level fraud summaries" | `df.groupby("user_id").agg(...)` → `fraud_summary.csv` (323 users, 9 columns) |
| "distribution visualizations" | `plot_hist()` in `utils.py` → `fraud_distribution.png` |
| "Python and Pandas" | `import pandas as pd`, all DataFrame operations |

### Interviewer question: "What's in your user-level fraud summary?"
→ "For each of the 323 users who had at least one anomalous transaction: total transaction count, number of anomalous transactions, maximum and average anomaly score, total spend, average and maximum transaction amount, and the anomaly rate percentage. Sorted by maximum anomaly score."

---

# PART 28 — 30-MINUTE REVISION SHEET

## Project
Unsupervised anomaly detection pipeline for bank transactions: CSV → SQLite → SQL feature engineering → Pandas → Isolation Forest → outputs.

## Problem
Detect suspicious transactions without labeled fraud data. 67,818 transactions, 400 users, 90 days (Jan–Mar 2024).

## Dataset
6 columns: `tx_id, user_id, date, region, merchant, amount`. Zero nulls. Zero duplicates. Amount range $5–$2,077. 9 merchants, 4 regions.

## SQL Features (5 engineered + 1 raw)
| Feature | Source |
|---------|--------|
| `amount` | Raw transaction |
| `tx_count` | COUNT(*) GROUP BY user_id |
| `avg_amount` | AVG(amount) GROUP BY user_id |
| `total_amount` | SUM(amount) GROUP BY user_id |
| `daily_tx` | COUNT(*) GROUP BY user_id, date |
| `daily_amount` | SUM(amount) GROUP BY user_id, date |

## Why SQLite
Zero setup, serverless, works from `git clone`, appropriate for 67K rows, standard SQL portable to any RDBMS.

## Why Unsupervised
No fraud labels exist at detection time. Labels require post-facto investigation, making supervised classification impossible at inference time.

## Why Isolation Forest
Designed for anomaly detection, no labels needed, O(n log n), interpretable ("easy to isolate"), fast on tabular data.

## Key Parameters
| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_estimators` | 200 | More stable scores vs default 100 |
| `contamination` | 0.02 | Assumes ~2% anomaly rate |
| `random_state` | 7 | Reproducibility |

## Anomaly Score
- Source: `-model.decision_function(X)` (flipped so higher = worse)
- Normalized: `(score - min) / (max - min + 1e-9)` → range [0, 1]
- NOT a probability. It's a relative isolation measure.

## Contamination
- `0.02` = model flags the 2% most isolated transactions
- Does NOT mean the true fraud rate is 2%
- It's an assumption — would be tuned against domain knowledge in production

## Outputs
| File | Rows | Key columns |
|------|------|------------|
| `fraud_scores.csv` | 67,818 | 14 cols including all features, `anomaly_score`, `is_anomaly`, `anomaly_rank` |
| `fraud_summary.csv` | 323 users | `anomalous_transactions`, `max_anomaly_score`, `anomaly_rate_pct` |
| `fraud_distribution.png` | — | Blue=normal, Red=flagged |

## Result: 1,357 anomalies (2.00% of 67,818)

## Limitations
1. No labels → can't compute accuracy/F1/AUC
2. Global model — high-value users disproportionately flagged
3. Contamination is an assumption
4. Offline batch only (not real-time)
5. SQLite not production-grade
6. 6 features only (no merchant encoding, no time-of-day)
7. No explainability per transaction

## Production Improvements
Kafka streaming → PostgreSQL/Snowflake → Feature store → Model API (FastAPI) → Alert queue → Analyst review → Feedback loop → Supervised retraining

## 10 Most Important Questions

1. "Tell me about your project." → 30-second answer
2. "Why Isolation Forest?" → Unsupervised, efficient, interpretable
3. "How does Isolation Forest work?" → Random trees, path length, short path = anomaly
4. "What is your anomaly score?" → Flipped, normalized isolation measure; NOT a probability
5. "How did you evaluate?" → Sanity checks; no labels → no supervised metrics
6. "How did you choose 2% contamination?" → Assumption, not measurement; industry guidance; tunable
7. "Why SQLite?" → Serverless, zero setup, self-contained
8. "Why TEMP VIEWs?" → Connection-scoped, clean, reusable within one connection
9. "Why LEFT JOIN?" → Defensive — keeps all transactions even if no view match
10. "Scale to 100M transactions?" → Column store DB, feature store, model API, streaming

---

# PART 29 — MOCK INTERVIEW

*Instructions: Read each question. Formulate your answer mentally (or write it). Then read the evaluation.*

---

**Q1:** "Can you give me a 30-second overview of your fraud detection project?"

*(Formulate your answer before reading the evaluation.)*

**Evaluation criteria:** Did you mention (1) unsupervised, (2) SQL feature engineering, (3) Isolation Forest, (4) ranked outputs? Did you avoid claiming it "detects fraud" — instead saying "identifies suspicious/anomalous" transactions? A score of 9/10 mentions all four elements in a confident, non-rehearsed-sounding manner. A 6/10 gives a vague answer about "machine learning on transactions."

---

**Q2:** "Walk me through your SQL. What does the first view do?"

*(The interviewer points to `queries.sql`.)*

**Evaluation criteria:** Did you explain GROUP BY, what one row in `user_stats` represents (one user), what COUNT/AVG/SUM compute, and why these features help detect fraud? A 9/10 connects the feature to a fraud scenario: "avg_amount gives us the user's spending baseline — a transaction at 15× their average is suspicious." A 5/10 just recites the SQL syntax without explaining the "why."

---

**Q3:** "What is a TEMP VIEW? Why not use a permanent table?"

**Evaluation criteria:** Key points: connection-scoped, auto-destroyed when connection closes, doesn't persist to the database file. A 9/10 also mentions why this matters for the Python implementation (same connection must be used).

---

**Q4:** "Explain Isolation Forest to me like I'm a software engineer, not a data scientist."

**Evaluation criteria:** Can you explain the intuition without jargon? The best answers use an analogy: "Imagine you're trying to hide in a crowd. Normal transactions hide well — they blend in. Fraudulent transactions stand out immediately — they get found in just a couple of guesses. Isolation Forest quantifies exactly how 'hiddable' each transaction is."

---

**Q5:** "What is your anomaly score? Is it a probability?"

**Evaluation criteria:** CRITICAL. Must say: NOT a probability. It's a normalized isolation measure. From the actual code: flipped `decision_function`, min-max normalized. A 10/10 immediately answers "No, it's not a probability — it's..." A 4/10 hedges or says "kind of like a probability."

---

**Q6:** "How did you evaluate your model?"

**Evaluation criteria:** Must acknowledge: no labels → no supervised metrics. Must describe sanity checks instead. Common weak answer: "I didn't evaluate it" (3/10). Strong answer: "Without labels, I validated through behavioral plausibility — the top anomalies are transactions that are 10-15× the user's own average, combined with high daily activity. I also verified reproducibility and that exactly 2% were flagged."

---

**Q7:** "Why 2% contamination? Isn't that just a guess?"

**Evaluation criteria:** Yes, it is a principled guess. The correct answer acknowledges this directly: "You're right — it is an assumption. I chose 2% based on industry guidance on typical anomaly rates in financial data. In production, I'd tune it against analyst feedback: if too many alerts are false positives, I'd reduce it." A weak answer: "Because 2% of transactions are fraudulent" (4/10 — this mistakes contamination for a known fraud rate).

---

**Q8:** "How would you make this production-ready at JPMorgan's scale?"

**Evaluation criteria:** Must NOT claim the current project is production-ready. Must articulate: (1) real-time streaming (Kafka), (2) scalable database (Snowflake), (3) model serving API, (4) analyst review workflow, (5) feedback loop. A 9/10 also mentions monitoring for model drift.

---

**Q9:** "Your user U1227 has 66 out of 203 transactions flagged — 32% of their activity. Doesn't that suggest the model is wrong?"

**Evaluation criteria:** This is the "global vs per-user" weakness question. The correct answer: "That's a legitimate limitation. Isolation Forest builds a global model — it doesn't model each user's baseline independently. U1227 is likely a high-value user whose normal transactions look anomalous compared to the full dataset. A production improvement would be to segment users by spending tier and apply user-relative z-score features."

---

**Q10:** "If I gave you 10,000 confirmed fraud labels, what would you do differently?"

**Evaluation criteria:** Must articulate the transition to supervised learning: (1) use labeled column as target, (2) split train/test, (3) train XGBoost/RandomForest/LightGBM, (4) evaluate with precision-recall AUC (not ROC-AUC, because class imbalance), (5) keep unsupervised model running in parallel for novel fraud patterns not in training labels.

---

*This mock interview section is open-ended. Practice answering each question aloud, record yourself, and compare to the evaluation criteria above. The goal is to answer with confidence, technical precision, and intellectual honesty about limitations.*
