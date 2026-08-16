# Fraud Detection &nbsp;[SQL + Python · Unsupervised]

Identify potentially suspicious bank transactions using **SQL (SQLite)** for feature engineering and **Python** for unsupervised anomaly detection with Isolation Forest.

> **Important:** This is an *unsupervised* anomaly detection system. It identifies statistically unusual transaction behaviour in the absence of any labelled fraud data. Flagged transactions are *potentially suspicious*, not confirmed fraud.

---

## Overview

This project demonstrates a practical anomaly detection workflow where no ground-truth fraud labels exist.  
It integrates **SQL-based data aggregation** with **machine learning anomaly detection**, showing how data engineers and analysts can surface unusual transaction patterns in banking or financial systems.

---

## Workflow

1. **Load transaction data into SQLite** (`create_db.py`)
2. **Run SQL feature engineering** (`queries.sql`)
   - Compute user-level metrics: transaction count, average amount, total spend
   - Compute daily activity: daily transaction count, daily total spend
3. **Apply Isolation Forest** to detect anomalies based on aggregated behavioural features
4. **Generate outputs**
   - `fraud_scores.csv` — all transactions with anomaly scores, flags, and ranks
   - `fraud_summary.csv` — per-user summary of anomalous activity
   - `fraud_distribution.png` — histogram showing score distribution

---

## Project Structure

```
fraud-detection-sql-unsupervised/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ transactions.csv
├─ src/
│  ├─ create_db.py
│  ├─ queries.sql
│  ├─ detect_fraud_unsupervised.py
│  └─ utils.py
└─ outputs/
   ├─ fraud_scores.csv
   ├─ fraud_summary.csv
   └─ charts/
       └─ fraud_distribution.png
```

---

## Dataset

| Column | Type | Description |
|---------|------|-------------|
| `tx_id` | integer | Unique transaction ID |
| `user_id` | string | User identifier |
| `date` | string (YYYY-MM-DD) | Transaction date |
| `region` | string | Geographic region |
| `merchant` | string | Merchant name or category |
| `amount` | float | Transaction amount (USD) |

The dataset contains **67,818 transactions** across **400 users** over a 90-day period (Jan–Mar 2024).  
All amounts are positive. No missing values or duplicate transaction IDs.

---

## SQL Feature Engineering

Feature generation is handled by `src/queries.sql`.  
It creates two temporary SQL views — connection-scoped in SQLite — and joins them back to the transaction table to produce one enriched row per transaction.

```sql
-- Per-user aggregate behavioural features
CREATE TEMP VIEW user_stats AS
SELECT user_id,
       COUNT(*)    AS tx_count,
       AVG(amount) AS avg_amount,
       SUM(amount) AS total_amount
FROM transactions
GROUP BY user_id;

-- Per-user, per-day activity features
CREATE TEMP VIEW daily_user AS
SELECT user_id, date,
       COUNT(*)    AS daily_tx,
       SUM(amount) AS daily_amount
FROM transactions
GROUP BY user_id, date;

-- Final: one row per transaction with all features joined in
SELECT t.tx_id, t.user_id, t.date, t.region, t.merchant, t.amount,
       us.tx_count, us.avg_amount, us.total_amount,
       COALESCE(du.daily_tx, 0)     AS daily_tx,
       COALESCE(du.daily_amount, 0) AS daily_amount
FROM transactions t
LEFT JOIN user_stats us ON t.user_id = us.user_id
LEFT JOIN daily_user du ON t.user_id = du.user_id AND t.date = du.date;
```

**TEMP VIEW note:** SQLite temporary views are scoped to a single connection. The Python pipeline keeps one connection open for both view creation and the final SELECT, so views are always available when the query runs.

---

## Machine Learning

The unsupervised model uses **Isolation Forest** from scikit-learn.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_estimators` | 200 | More trees = more stable anomaly scores |
| `contamination` | 0.02 | Assumes ~2% of transactions are anomalous |
| `random_state` | 7 | Reproducible results across runs |

**How it works:**  
Isolation Forest randomly partitions the feature space. Transactions that require fewer splits to isolate are considered anomalies — they exist in sparse regions of the feature space, indicating unusual behaviour.

**Scoring:**  
`decision_function()` returns a value where *more negative = more anomalous*. We flip and normalise this to a `anomaly_score` in [0, 1]:

```
anomaly_score = (-decision_score - min(-decision_score))
              / (max(-decision_score) - min(-decision_score))
```

A score near **1** means the transaction is highly anomalous; near **0** means normal.

**Features used:**

| Feature | Description |
|---------|-------------|
| `amount` | Raw transaction amount |
| `tx_count` | Total transactions by this user |
| `avg_amount` | User's average transaction amount |
| `total_amount` | User's cumulative spend |
| `daily_tx` | Transactions by user on that day |
| `daily_amount` | User's spend on that day |

---

## Visualization

### Anomaly Score Distribution
The histogram below shows normal transactions (blue) and flagged anomalies (red) separated by colour, with an annotation box showing total and flagged counts.

*(Run the pipeline to regenerate `outputs/charts/fraud_distribution.png`)*

---

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| **SQLite** | Feature engineering via temporary SQL views |
| **Python 3.11+** | Pipeline orchestration |
| **pandas** | Data manipulation |
| **scikit-learn** | Isolation Forest implementation |
| **matplotlib** | Anomaly score distribution chart |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

Run from the **project root directory**:

### Step 1 — Load data into SQLite
```bash
python src/create_db.py --csv data/transactions.csv --db fraud.db
```

### Step 2 — Run anomaly detection
```bash
python src/detect_fraud_unsupervised.py --db fraud.db --sql src/queries.sql --outdir outputs
```

---

## Outputs

| File | Rows | Description |
|------|------|-------------|
| `fraud_scores.csv` | 67,818 | Every transaction with engineered features, `anomaly_score` (0–1), `is_anomaly` flag (0/1), and `anomaly_rank` (1 = most suspicious) |
| `fraud_summary.csv` | 323 | Per-user summary for users with ≥1 anomalous transaction: anomaly count, max/avg score, anomaly rate % |
| `fraud_distribution.png` | — | Histogram of anomaly scores, normal vs flagged |

### fraud_scores.csv columns
`tx_id`, `user_id`, `date`, `region`, `merchant`, `amount`, `tx_count`, `avg_amount`, `total_amount`, `daily_tx`, `daily_amount`, `anomaly_score`, `is_anomaly`, `anomaly_rank`

---

## Model Validation (Unsupervised)

Since there are no ground-truth fraud labels, standard supervised metrics (accuracy, precision, recall, F1) do not apply.  
Instead, the following sanity checks confirm the model behaves sensibly:

- **2.00%** of transactions flagged (1,357 / 67,818) — consistent with the configured `contamination=0.02`
- **323** out of 400 users have at least one anomalous transaction
- Top anomalies are predominantly high-amount transactions and/or users with concentrated daily activity, which are behaviourally defensible outliers
- `random_state=7` ensures identical results on every run

---

## Conclusion

This project demonstrates a complete unsupervised anomaly detection pipeline built on standard, interpretable tools:  
**CSV → SQLite/SQL → Pandas → Isolation Forest → ranked scores + visualisation**.

It is designed to be understandable, reproducible, and interview-ready — every design decision (contamination choice, feature selection, scoring direction) can be explained from first principles.
