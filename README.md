# 🎫 Developing ML Models on ITSM Dataset

> Applying machine learning to IT Service Management (ITSM) data to automate ticket handling, predict priorities, detect anomalies, and forecast ticket volumes — turning raw helpdesk data into actionable intelligence.

---

## 📌 What is ITSM?

**IT Service Management (ITSM)** is the process companies use to manage IT support tickets — think helpdesk requests, incidents, and change requests. Every ticket has details like priority, urgency, category, reassignments, and resolution time.

This project takes a real-world ITSM dataset (~45,000 tickets) and builds **4 end-to-end ML use cases** on top of it.

---

## 🎯 Project Goals

| Goal | Description |
|------|-------------|
| Automate ticket priority assignment | Predict the correct priority level from ticket metadata |
| Forecast ticket volume | Predict how many tickets will come in next week/month |
| Detect unusual tickets | Flag anomalous tickets that need special attention |
| Auto-tag ticket categories | Read ticket descriptions and assign the right category automatically |

---

## 📂 Dataset Overview

- **Source:** ITSM ticketing system (CSV format)
- **Size:** ~45,000 tickets
- **Key Columns Used:**

| Column | What It Means |
|--------|---------------|
| `Priority` | How urgent is the ticket? (1–5 scale) |
| `Impact` | How many users are affected? |
| `Urgency` | How quickly must this be resolved? |
| `CI_Cat / CI_Subcat` | What system/asset is affected? |
| `Category` | Type of ticket (incident, request, change) |
| `Open_Time` | When was the ticket opened? |
| `No_of_Reassignments` | How many times was it passed between teams? |
| `No_of_Related_Interactions` | How many calls/chats were linked? |
| `short_description` | Free-text summary of the issue |

---

## 🔧 Data Preprocessing

Before building any models, the raw data was cleaned and prepared:

- Removed columns with too many missing values (`Related_Change`, `Close_Time`, etc.)
- Filled missing values using mode imputation for categorical fields
- Converted date columns (`Open_Time`, `Resolved_Time`) to proper datetime format
- Encoded `Impact` and `Urgency` from text labels to numeric values (0–4 scale)
- Applied **One-Hot Encoding** for multi-category columns like `CI_Cat`, `CI_Subcat`, `Category`

---

## 🚀 Use Cases Built

---

### ✅ Use Case 1 — Priority Prediction

**Problem:** When a new ticket comes in, what priority should it get?

**Approach:**
- Features used: `Impact`, `Urgency`, `CI_Cat`, `CI_Subcat`, `Category`
- Target: `Priority` (5 classes: P1 to P5)
- Tried multiple models and discovered a key insight

**Models Tried:**
- XGBoost Classifier (with and without class balancing)
- Decision Tree Classifier
- Rule-based lookup (groupby Impact + Urgency → Priority)

**Key Insight:** 🔍 Priority is almost entirely determined by `Impact` × `Urgency` — a simple rule-based mapping outperformed ML models for this task. This is a great example of **not over-engineering a solution**.

**Techniques Used:**
- Stratified train/test split (80/20)
- Class weight balancing to handle imbalanced priority classes
- `classification_report` for precision, recall, F1 evaluation

---

### 📈 Use Case 2 — Time Series Forecasting (Ticket Volume)

**Problem:** How many tickets will be raised next week or next month?

**Why it matters:** IT teams can staff up or prepare resources ahead of predicted spikes.

**Approach:**
- Resampled ticket data by month and by week using `Open_Time`
- Tested for stationarity using the **Augmented Dickey-Fuller (ADF) test**
- Built and compared ARIMA and SARIMA models

**Models Used:**

| Model | Granularity | Notes |
|-------|-------------|-------|
| ARIMA(1,1,1) | Monthly | Baseline forecast |
| SARIMA(1,1,1)(1,1,1,12) | Monthly | Captures yearly seasonality |
| ARIMA(1,1,1) | Weekly | Better accuracy on recent trends |
| SARIMA(1,1,1)(1,1,1,12) | Weekly | Best overall model |

**Results:**
- Weekly granularity gave better accuracy than monthly
- SARIMA captured seasonal patterns (recurring spikes) effectively
- Evaluated using **MAE** (Mean Absolute Error) and **MAPE** (Mean Absolute Percentage Error)

---

### 🚨 Use Case 3 — Anomaly Detection

**Problem:** Which tickets are unusually complex or suspicious and need escalation?

**Why it matters:** Tickets with abnormally high reassignments or interactions often signal process breakdowns or misrouted tickets.

**Approach:**
- Built a `ticket_complexity` feature combining reassignments + interactions
- Scaled features using `StandardScaler`
- Evaluated 3 anomaly detection algorithms

**Models Compared:**

| Algorithm | Suitability for this dataset |
|-----------|------------------------------|
| Isolation Forest ✅ | Best — handles large datasets, robust to noise |
| One-Class SVM | Less efficient on 45k rows |
| Local Outlier Factor | Good for small datasets |

**Isolation Forest** was chosen as the final model.

**Results:**
- Successfully flagged anomalous tickets (labeled as `-1`)
- Used **PCA (2 components)** to visualize normal vs. anomalous tickets in 2D
- Plotted anomaly score distribution to understand thresholds
- Tested contamination rates of 1%, 2%, 5% to find the right sensitivity

**Example Prediction:**
```python
model.predict(scaler.transform([[60, 40, 3, 3, 0.6, 7]]))
# Output: [-1]  → Anomalous ticket flagged!
```

---

### 🏷️ Use Case 4 — Auto Tagging (NLP Text Classification)

**Problem:** Can the system automatically assign a category by reading the ticket description?

**Why it matters:** Manual categorization is slow and inconsistent. Auto-tagging speeds up ticket routing.

**Dataset:** ServiceNow sample dataset with `short_description` and `category` columns.

**NLP Pipeline:**
1. Lowercasing and removing special characters
2. Tokenization using NLTK
3. Stopword removal
4. Lemmatization + Stemming
5. TF-IDF Vectorization with bigrams (`ngram_range=(1,2)`)

**Models Used:**

| Model | Notes |
|-------|-------|
| Logistic Regression (balanced) | Primary model, handled class imbalance |
| Multinomial Naive Bayes | Comparison model |

**Evaluated using:** Precision, Recall, F1-Score per category

**Example Prediction:**
```python
model.predict(vectorizer.transform([clean_text("Unable to connect to VPN after update")]))
# → 'network'
```

**Observation:** The model showed good performance on available data. Accuracy would improve significantly with a larger, more diverse ticket dataset.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data wrangling |
| Scikit-learn | ML models, preprocessing, evaluation |
| XGBoost | Gradient boosted tree classifier |
| Statsmodels | ARIMA / SARIMA time series models |
| NLTK | NLP text preprocessing |
| Matplotlib / Seaborn | Data visualization |
| PCA | Dimensionality reduction for anomaly visualization |

---

## 📊 Results Summary

| Use Case | Model Used | Outcome |
|----------|-----------|---------|
| Priority Prediction | Rule-based (Impact × Urgency) | Discovered ML not needed — rule lookup works perfectly |
| Volume Forecasting | SARIMA (weekly) | Strong seasonal forecasting with low MAE |
| Anomaly Detection | Isolation Forest | Successfully flagged outlier tickets with PCA visualization |
| Auto Tagging | Logistic Regression + TF-IDF | Working classifier; improves with more data |

---

## 💡 Key Learnings

- **Not every problem needs a complex ML model.** Priority prediction was perfectly solved with a lookup table — a valuable real-world insight.
- **Granularity matters in time series.** Weekly forecasting outperformed monthly, showing the importance of choosing the right time resolution.
- **Isolation Forest scales well.** For 45k+ tickets, it was the clear winner among anomaly detection methods.
- **NLP pipeline quality directly impacts text classification.** Stemming + lemmatization together improved category prediction.

---

## 📁 Project Structure

```
├── Data/
│   ├── ITSM_data.csv            # Main ITSM dataset
│   └── service_now/sample.csv  # ServiceNow data for auto-tagging
├── tasks.ipynb                  # Main notebook with all 4 use cases
└── README.md
```

---

## 🙋 About This Project

This project was built to explore how machine learning can be applied to real-world IT operations data. It covers the full pipeline — from messy raw data to working predictions — across four distinct and practical problem statements that any IT team could use today.
