# Fraud Detection & Risk Analytics

### SQL (SQLite) + Python + Isolation Forest | Unsupervised Anomaly Detection

An end-to-end **unsupervised fraud/anomaly detection pipeline** that identifies potentially suspicious banking transactions by combining **SQL-based behavioral feature engineering** with **Python-based Isolation Forest anomaly detection**.

The project is designed for scenarios where historical transactions do **not contain reliable fraud labels**. Instead of predicting whether a transaction is definitely fraudulent, the system identifies transactions whose behavior is statistically unusual compared with the rest of the transaction population.

> **Important:** This is an unsupervised anomaly detection system. A flagged transaction represents a **potentially suspicious transaction**, not confirmed fraud.

---

## Project Overview

Traditional supervised fraud detection requires historical transactions labeled as:

* `0` → legitimate
* `1` → fraudulent

In many real-world banking environments, reliable fraud labels may be unavailable, incomplete, delayed, or expensive to obtain.

This project approaches the problem as an **unsupervised anomaly detection task**.

The system:

1. Loads transaction data into a **SQLite database**
2. Uses **SQL** to calculate behavioral features
3. Retrieves the enriched transaction dataset into Python
4. Applies **Isolation Forest**
5. Generates an anomaly score for every transaction
6. Flags the most suspicious transactions
7. Ranks transactions by suspiciousness
8. Produces user-level fraud/anomaly summaries
9. Generates a visualization of the anomaly-score distribution

### Overall Pipeline

```text
Raw Transactions
       ↓
     CSV
       ↓
   SQLite DB
       ↓
SQL Feature Engineering
       ↓
Behavioral Features
       ↓
     Pandas
       ↓
 Isolation Forest
       ↓
Anomaly Score
       ↓
Suspicious Transaction Flag
       ↓
Ranking + User Summary + Visualization
```

---

# Why SQLite?

SQLite is used as the SQL engine because this project is designed as a **self-contained analytics pipeline**.

SQLite provides:

* SQL querying
* `GROUP BY` aggregations
* `JOIN` operations
* temporary views
* persistent local database storage
* zero server configuration

Unlike MySQL or PostgreSQL, SQLite does not require a separate database server.

This makes it convenient for a reproducible machine-learning project where the database is used primarily for **data storage and SQL-based feature engineering**.

> The SQL logic is intentionally separated from the Python machine-learning logic so that the project demonstrates both database querying and ML skills.

---

# Dataset

The transaction dataset contains the following fields:

| Column       | Type    | Description           |
| ------------ | ------- | --------------------- |
| `tx_id`    | Integer | Unique transaction ID |
| `user_id`  | String  | Customer identifier   |
| `date`     | String  | Transaction date      |
| `region`   | String  | Geographic region     |
| `merchant` | String  | Merchant/category     |
| `amount`   | Float   | Transaction amount    |

The current dataset contains:

* **67,818 transactions**
* **400 users**
* **90 days of transaction activity**
* January–March 2024
* Positive transaction amounts
* No missing values
* No duplicate transaction IDs

---

# Project Structure

```text
fraud-detection-sql-unsupervised/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── transactions.csv
│
├── src/
│   ├── create_db.py
│   ├── queries.sql
│   ├── detect_fraud_unsupervised.py
│   └── utils.py
│
└── outputs/
    ├── fraud_scores.csv
    ├── fraud_summary.csv
    │
    └── charts/
        └── fraud_distribution.png
```

---

# 1. Loading Data into SQLite

The first stage converts the raw CSV transaction data into a SQLite database.

The database provides a structured environment where SQL can be used to calculate behavioral statistics efficiently.

Example:

```bash
python src/create_db.py --csv data/transactions.csv --db fraud.db
```

This creates:

```text
fraud.db
```

containing the transaction table.

---

# 2. SQL Feature Engineering

The main purpose of SQL is to transform raw transaction records into **behavioral features**.

Instead of looking at a transaction only through its individual amount, we also ask:

> How does this transaction compare with the customer's overall behavior?

Two levels of aggregation are created.

---

## User-Level Behavioral Features

For every user, the system calculates:

### Transaction Count

```sql
COUNT(*)
```

This represents the user's overall transaction frequency.

A user with unusually high transaction activity may deserve additional investigation.

### Average Transaction Amount

```sql
AVG(amount)
```

This represents the user's typical transaction size.

A transaction significantly different from a user's normal spending pattern may become more anomalous.

### Total Transaction Amount

```sql
SUM(amount)
```

This represents the user's cumulative spending over the available dataset period.

---

## Daily Behavioral Features

The system also calculates activity for each:

```text
user + date
```

Two additional features are created.

### Daily Transaction Count

```sql
COUNT(*)
```

This tells us how many transactions a particular user performed on a particular day.

### Daily Transaction Amount

```sql
SUM(amount)
```

This represents the user's total spending on that day.

These features help capture **concentrated transaction behavior**.

---

# SQL Implementation

The project uses temporary SQLite views.

### User-Level Statistics

```sql
CREATE TEMP VIEW user_stats AS
SELECT
    user_id,
    COUNT(*) AS tx_count,
    AVG(amount) AS avg_amount,
    SUM(amount) AS total_amount
FROM transactions
GROUP BY user_id;
```

This produces one row per user.

---

### Daily User Statistics

```sql
CREATE TEMP VIEW daily_user AS
SELECT
    user_id,
    date,
    COUNT(*) AS daily_tx,
    SUM(amount) AS daily_amount
FROM transactions
GROUP BY user_id, date;
```

This produces one row for each user-day combination.

---

### Joining Features Back to Transactions

The aggregated features are then joined back to the original transaction table.

```sql
SELECT
    t.tx_id,
    t.user_id,
    t.date,
    t.region,
    t.merchant,
    t.amount,

    us.tx_count,
    us.avg_amount,
    us.total_amount,

    COALESCE(du.daily_tx, 0) AS daily_tx,
    COALESCE(du.daily_amount, 0) AS daily_amount

FROM transactions t

LEFT JOIN user_stats us
    ON t.user_id = us.user_id

LEFT JOIN daily_user du
    ON t.user_id = du.user_id
    AND t.date = du.date;
```

The result contains **one enriched row for every original transaction**.

Therefore:

```text
67,818 transactions
        ↓
67,818 enriched transaction rows
```

---

# Important SQLite TEMP VIEW Detail

The project uses:

```sql
CREATE TEMP VIEW
```

SQLite temporary views exist only within the database connection that created them.

Therefore, the Python pipeline intentionally keeps the **same SQLite connection open** while:

1. Creating the temporary views
2. Executing the final SQL query

Otherwise, the views would no longer be available.

This is an important implementation detail because opening a completely new SQLite connection for the final query would cause the temporary views to disappear.

---

# 3. Feature Set Used by the Model

The final machine-learning dataset contains six numerical behavioral features.

| Feature          | Meaning                                     |
| ---------------- | ------------------------------------------- |
| `amount`       | Amount of the current transaction           |
| `tx_count`     | Total transactions performed by the user    |
| `avg_amount`   | User's average transaction amount           |
| `total_amount` | User's total spending                       |
| `daily_tx`     | Number of transactions by the user that day |
| `daily_amount` | Total spending by the user that day         |

These features combine:

* **transaction-level behavior**
* **user-level behavior**
* **daily behavioral activity**

This gives Isolation Forest more context than simply using transaction amount.

---

# 4. Why Isolation Forest?

Because the dataset does not contain reliable fraud labels, a supervised classifier such as:

```text
Logistic Regression
Random Forest Classifier
XGBoost Classifier
```

cannot be trained properly without labeled fraud examples.

Instead, the project uses:

```text
Isolation Forest
```

from `scikit-learn`.

Isolation Forest is designed for **anomaly detection**.

Its basic idea is:

> An unusual observation should be easier to isolate than a normal observation.

The algorithm randomly selects features and split values to partition the dataset.

Normal observations tend to belong to dense, common regions and require more splits to isolate.

Anomalous observations tend to lie in sparse or unusual regions and can be isolated using fewer splits.

Therefore:

```text
Easy to isolate
      ↓
More unusual
      ↓
More anomalous
```

---

# 5. Isolation Forest Configuration

The model is configured as follows:

```python
IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=7
)
```

### `n_estimators = 200`

The model builds 200 isolation trees.

More trees generally provide more stable anomaly estimates at the cost of additional computation.

---

### `contamination = 0.02`

The model assumes approximately **2% of the observations may be anomalous**.

For the current dataset:

```text
67,818 × 0.02 ≈ 1,356
```

The actual output contains:

```text
1,357 flagged transactions
```

which is approximately:

```text
2.00%
```

Important:

> `contamination=0.02` is a modeling assumption, not proof that exactly 2% of transactions are fraudulent.

---

### `random_state = 7`

This makes the model deterministic and reproducible.

Running the same pipeline with the same data and configuration produces consistent results.

---

# 6. Anomaly Score

Isolation Forest's:

```python
decision_function()
```

returns an anomaly-related score where more negative values indicate more anomalous observations.

For easier interpretation, the project converts the score into a normalized:

```text
anomaly_score
```

between:

```text
0 and 1
```

Conceptually:

```text
Original Isolation Forest Score
             ↓
      Reverse Direction
             ↓
       Normalize
             ↓
       0 ───────── 1
      normal    anomalous
```

A score closer to:

```text
1 → more anomalous
0 → less anomalous
```

This makes the final output easier to interpret and rank.

---

# 7. Flagging Suspicious Transactions

The model produces a binary anomaly indicator:

```text
is_anomaly
```

where:

```text
0 → not flagged
1 → flagged as potentially anomalous
```

The system does **not** claim:

```text
is_anomaly = 1 → confirmed fraud
```

Instead:

```text
is_anomaly = 1
        ↓
Potentially suspicious
        ↓
Requires investigation / additional evidence
```

This distinction is important because the project is unsupervised.

---

# 8. Ranking Transactions

Every transaction receives an:

```text
anomaly_rank
```

where:

```text
1 = most suspicious
```

This allows an analyst to focus first on transactions with the highest anomaly scores.

For example:

```text
Rank 1  → highest anomaly score
Rank 2
Rank 3
...
Rank N
```

This is more useful operationally than simply returning a binary fraud flag.

---

# 9. Output Files

The pipeline produces three main outputs.

## `fraud_scores.csv`

Contains all:

```text
67,818 transactions
```

along with their engineered features and anomaly results.

Columns:

```text
tx_id
user_id
date
region
merchant
amount
tx_count
avg_amount
total_amount
daily_tx
daily_amount
anomaly_score
is_anomaly
anomaly_rank
```

This is the main transaction-level output.

---

## `fraud_summary.csv`

This provides a user-level summary for users who have at least one anomalous transaction.

It contains information such as:

* anomaly count
* maximum anomaly score
* average anomaly score
* anomaly rate

The current output contains:

```text
323 users
```

with at least one flagged transaction.

This helps move from:

```text
transaction-level investigation
```

to:

```text
customer-level investigation
```

---

## `fraud_distribution.png`

The project also generates a histogram showing the distribution of anomaly scores.

The visualization separates:

```text
Normal transactions
vs.
Flagged transactions
```

This provides a simple visual sanity check of how the model is separating unusual observations from the rest of the dataset.

---

# 10. Model Validation

Because this is an **unsupervised** project without reliable fraud labels, traditional supervised metrics such as:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
```

cannot be meaningfully interpreted as fraud-detection performance.

Instead, the project uses behavioral and implementation sanity checks.

### Check 1 — Expected anomaly proportion

Approximately:

```text
1,357 / 67,818 ≈ 2.00%
```

transactions were flagged.

This is consistent with the configured contamination level.

---

### Check 2 — User coverage

```text
323 / 400 users
```

have at least one flagged transaction.

This indicates that anomalies are not restricted to a tiny number of users.

---

### Check 3 — Behavioral plausibility

The highest-ranked anomalies tend to contain combinations such as:

* unusually large transaction amounts
* unusually high daily transaction frequency
* concentrated daily spending
* behavior that differs from the broader transaction population

These are useful sanity checks, but they are **not proof of fraud**.

---

# 11. Why These Features?

The feature design intentionally combines multiple behavioral dimensions.

For example, consider two transactions:

### Transaction A

```text
Amount = $5,000
```

If the user's normal transactions are around:

```text
$3,000 – $7,000
```

then the transaction may not be particularly unusual.

### Transaction B

```text
Amount = $5,000
```

but the user's normal transactions are:

```text
$20 – $100
```

The same $5,000 transaction becomes much more unusual.

This is why the project does not rely only on:

```text
amount
```

Instead, it combines transaction amount with user and daily behavioral statistics.

---

# 12. Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 1 — Create SQLite Database

Run from the project root:

```bash
python src/create_db.py \
    --csv data/transactions.csv \
    --db fraud.db
```

This loads the transaction data into SQLite.

---

## Step 2 — Run Anomaly Detection

```bash
python src/detect_fraud_unsupervised.py \
    --db fraud.db \
    --sql src/queries.sql \
    --outdir outputs
```

The pipeline then:

```text
SQLite
  ↓
SQL feature engineering
  ↓
Pandas DataFrame
  ↓
Isolation Forest
  ↓
Anomaly scores
  ↓
Flags + rankings
  ↓
CSV outputs
  ↓
Visualization
```

---

# 13. Technologies Used

| Technology             | Purpose                                         |
| ---------------------- | ----------------------------------------------- |
| **Python**       | Pipeline orchestration                          |
| **SQLite**       | Transaction storage and SQL feature engineering |
| **SQL**          | Behavioral aggregation and joins                |
| **Pandas**       | Data manipulation                               |
| **Scikit-learn** | Isolation Forest anomaly detection              |
| **Matplotlib**   | Anomaly-score visualization                     |

---

# 14. Key Engineering Decisions

### Why SQL before Machine Learning?

SQL is well suited for aggregation operations such as:

```text
COUNT
AVG
SUM
GROUP BY
JOIN
```

These operations allow behavioral features to be generated close to the data layer before sending the final dataset to the ML pipeline.

---

### Why not use raw transactions directly?

A raw transaction contains limited behavioral context.

Feature engineering allows the model to understand:

```text
Current transaction
        +
User's historical behavior
        +
User's daily behavior
```

This provides a richer representation for anomaly detection.

---

### Why not use supervised learning?

There are no reliable fraud labels in the dataset.

Training a supervised classifier without trustworthy labels would create misleading performance claims.

Therefore, Isolation Forest is a better fit for the current project formulation.

---

# 15. Limitations

This project is an anomaly detection system, not a production fraud-confirmation system.

Important limitations include:

### No fraud labels

The model cannot determine whether a flagged transaction is actually fraudulent.

### Contamination is an assumption

The `2%` contamination value represents a modeling assumption rather than a measured fraud rate.

### Behavioral anomalies are not necessarily fraud

A legitimate customer may make an unusually large purchase or perform many transactions in one day.

### Limited feature set

A production banking system could incorporate additional signals such as:

* transaction velocity
* merchant risk
* device information
* IP/location changes
* account age
* historical behavioral profiles
* payment method
* time-of-day patterns

These are outside the current project scope.

---

# 16. Production-Level Extension

A real banking fraud system could extend this pipeline by combining anomaly detection with additional evidence.

For example:

```text
Transaction Data
      ↓
Behavioral Features
      ↓
Anomaly Detection
      ↓
Risk Score
      ↓
Business Rules
      ↓
Additional Risk Signals
      ↓
Fraud Investigation
```

The unsupervised model could therefore act as one component of a larger fraud-risk platform rather than being treated as a final fraud decision-maker.

---

# 17. Interview Explanation

A concise way to explain the project in an interview is:

> **"I built an unsupervised fraud-anomaly detection pipeline using SQLite, SQL, Python, and Isolation Forest. Since the transaction dataset didn't contain reliable fraud labels, I didn't treat this as a supervised classification problem. I first loaded the transactions into SQLite and used SQL aggregations to generate behavioral features such as each user's transaction count, average amount, total spending, daily transaction count, and daily spending. I then brought these features into Python using Pandas and trained an Isolation Forest model. The model isolates observations that are statistically unusual, and I converted its output into a normalized anomaly score between 0 and 1. Finally, I generated transaction-level scores and rankings, a user-level anomaly summary, and a score-distribution visualization. The important point is that the system identifies potentially suspicious behavior rather than claiming that flagged transactions are confirmed fraud."**

---

# Conclusion

This project demonstrates an end-to-end **unsupervised anomaly detection workflow** combining database engineering and machine learning:

```text
CSV
 ↓
SQLite
 ↓
SQL Feature Engineering
 ↓
Pandas
 ↓
Isolation Forest
 ↓
Anomaly Scores
 ↓
Suspicious Transaction Ranking
 ↓
User-Level Summary
 ↓
Visualization
```

The project focuses on a realistic problem:

> **How can potentially suspicious transaction behavior be surfaced when reliable fraud labels are unavailable?**

The answer implemented here is to combine **SQL-based behavioral feature engineering** with **Isolation Forest-based unsupervised anomaly detection**.

The resulting system is reproducible, interpretable, and suitable as an analytics prototype for identifying transactions that warrant further investigation.
