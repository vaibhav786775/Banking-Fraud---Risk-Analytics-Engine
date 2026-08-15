# Bank Fraud Detection & Risk Analytics Engine

An end-to-end machine learning pipeline for detecting potentially fraudulent banking transactions using transaction behavior, balance inconsistencies, transaction types, and amount-based patterns.

The project focuses on the practical challenges involved in **fraud detection at scale**, particularly severe class imbalance, false positives, model evaluation, and threshold selection.

---

## 📌 Problem Statement

Financial institutions process millions of transactions every day, making manual fraud detection impractical.

The objective of this project is to build a machine learning system that can:

* Identify transactions with a high probability of fraud
* Capture meaningful behavioral and transactional patterns
* Handle highly imbalanced fraud data
* Minimize missed fraudulent transactions
* Control false positives so legitimate customers are not unnecessarily flagged
* Compare multiple classification models
* Provide evaluation metrics that are meaningful for a fraud detection system

The project uses approximately **6.36 million banking transactions** for analysis and model development.

---

## 🏗️ Project Workflow

```text
Raw Transaction Data
        ↓
Data Understanding & Validation
        ↓
Exploratory Data Analysis
        ↓
Fraud Pattern Analysis
        ↓
Feature Engineering
        ↓
Train / Validation / Test Split
        ↓
Class Imbalance Handling
        ↓
Model Training
        ↓
Model Comparison
        ↓
Threshold Optimization
        ↓
Fraud Detection Evaluation
```

---

## 📊 Dataset

The dataset contains banking transaction information such as:

* Transaction type
* Transaction amount
* Sender balance before transaction
* Sender balance after transaction
* Receiver balance before transaction
* Receiver balance after transaction
* Fraud indicator

The dataset is highly imbalanced, with fraudulent transactions representing only a small fraction of the overall transactions.

This makes **accuracy a misleading metric** for evaluating the model.

For example, a model that predicts every transaction as legitimate could achieve extremely high accuracy while completely failing its primary objective — detecting fraud.

---

## 🔎 Exploratory Data Analysis

The analysis focuses on understanding how fraudulent transactions differ from legitimate ones.

Key areas investigated include:

### Transaction Type

Fraud occurrence is analyzed across different transaction types to determine whether certain types of transactions are more associated with suspicious activity.

### Transaction Amount

Large-value transactions are investigated to identify whether unusually high transaction amounts are associated with fraud.

### Balance Behavior

The relationship between:

* Initial balance
* Transaction amount
* Final balance

is analyzed to identify unusual balance movements and inconsistencies.

### Fraud Distribution

The distribution of legitimate and fraudulent transactions is examined to quantify the severity of class imbalance.

---

## ⚙️ Feature Engineering

Instead of relying only on the raw transaction columns, additional behavioral features are created.

### Balance Difference

The expected and observed balance changes are compared to identify unusual transactions.

For example:

```text
Expected Balance = Old Balance - Transaction Amount
Balance Difference = Expected Balance - New Balance
```

A significant difference can indicate an unusual transaction pattern.

### Amount-Based Features

Transaction amounts are analyzed to identify unusually large transactions.

Examples include:

* High-value transaction indicators
* Relative transaction amount
* Log-transformed transaction amounts

### Transaction-Type Features

Categorical transaction types are transformed into machine-learning-compatible representations.

### Balance-Based Indicators

Additional indicators are derived from sender and receiver balance information to capture abnormal transaction behavior.

Feature engineering is particularly important in fraud detection because **domain knowledge can provide signals that are not directly available from the raw columns.**

---

## ⚠️ Handling Class Imbalance

Fraud detection is an inherently imbalanced classification problem.

A typical dataset may contain millions of legitimate transactions but only a small number of fraudulent transactions.

If this imbalance is ignored, a model can become biased toward the majority class.

Therefore, model performance is evaluated using metrics such as:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion Matrix

Special attention is given to **fraud recall**, because failing to detect an actual fraudulent transaction can result in direct financial loss.

---

## 🤖 Machine Learning Models

Multiple classification algorithms are trained and compared.

### 1. Logistic Regression

Used as a baseline model.

It provides a simple and interpretable benchmark for determining whether more complex models provide meaningful improvements.

### 2. Random Forest

A tree-based ensemble model capable of capturing non-linear relationships between transaction characteristics.

It is useful for understanding interactions between features such as transaction amount and balance behavior.

### 3. XGBoost

XGBoost is used as the primary high-performance tree-based model.

It is well suited to structured/tabular data and can capture complex non-linear relationships between transaction features.

---

## 📈 Model Evaluation

The models are evaluated using metrics relevant to fraud detection rather than relying only on accuracy.

### Precision

Of all transactions predicted as fraudulent, how many were actually fraudulent?

```text
Precision = TP / (TP + FP)
```

High precision means fewer legitimate transactions are incorrectly flagged.

### Recall

Of all actual fraudulent transactions, how many did the model detect?

```text
Recall = TP / (TP + FN)
```

Recall is particularly important in fraud detection because **false negatives represent fraudulent transactions that passed through the system.**

### F1-Score

The harmonic mean of precision and recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### ROC-AUC

Measures how effectively the model separates fraudulent and legitimate transactions across different classification thresholds.

### PR-AUC

Precision-Recall AUC is especially informative for highly imbalanced datasets because it focuses on the performance of the positive/fraud class.

### Confusion Matrix

The confusion matrix provides a direct view of:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## 🎯 Threshold Optimization

A classification model normally produces a probability rather than a direct fraud/legitimate decision.

For example:

```text
Transaction → Model → Fraud Probability
                         ↓
                       0.82
```

A default threshold of `0.50` is not necessarily optimal for a banking fraud system.

The decision threshold can be adjusted depending on the business objective.

For example:

```text
Probability ≥ Threshold → Flag as Fraud
Probability < Threshold → Consider Legitimate
```

Lowering the threshold can increase fraud recall but may also increase false positives.

Therefore, threshold selection becomes a **business trade-off between missed fraud and customer friction**.

---

## 💼 Business Perspective

A fraud detection model should not simply maximize accuracy.

The real objective is to balance:

```text
Fraud Detection
       ↕
False Positives
       ↕
Customer Experience
       ↕
Financial Loss
```

A false negative can allow a fraudulent transaction to go through.

A false positive can incorrectly block a legitimate customer transaction.

Therefore, the optimal operating point depends on the bank's risk tolerance and the relative cost of these two errors.

---

## 🧪 Model Comparison

The models are compared using:

| Model               | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| ------------------- | --------: | -----: | -: | ------: | -----: |
| Logistic Regression |        — |     — | — |      — |     — |
| Random Forest       |        — |     — | — |      — |     — |
| XGBoost             |        — |     — | — |      — |     — |

> Replace the values above with the actual results from the final model evaluation.

---

## 🛠️ Tech Stack

**Language**

* Python

**Data Processing**

* Pandas
* NumPy

**Visualization**

* Matplotlib
* Seaborn

**Machine Learning**

* Scikit-learn
* XGBoost

**Model Persistence**

* Joblib

---

## 📁 Project Structure

```text
Bank-Fraud-Detection/
│
├── data/
│   └── transactions.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── outputs/
│   ├── plots/
│   └── evaluation/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Key Takeaways

This project demonstrates the complete machine learning workflow for a real-world financial risk problem:

* Working with millions of transaction records
* Performing large-scale exploratory data analysis
* Identifying fraud-related transaction patterns
* Engineering domain-specific features
* Handling severe class imbalance
* Training multiple classification models
* Comparing models using appropriate evaluation metrics
* Understanding precision vs. recall trade-offs
* Optimizing classification thresholds
* Connecting machine learning performance with real-world financial risk

---

## 🔮 Future Improvements

A production-grade fraud detection system could be extended with:

* Real-time transaction scoring
* Streaming data pipelines using Kafka
* Time-based behavioral features
* Customer-level transaction history
* Device and location-based signals
* Graph-based fraud detection
* Explainable AI for analyst investigation
* Model monitoring and drift detection
* Automated fraud-alert pipelines
* Cost-sensitive model optimization

---

## 👤 Author

**Vaibhav Chhabra**

Computer Science & Engineering
Punjab Engineering College

[GitHub](https://github.com/vaibhav786775)
