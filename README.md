# Credit Card Fraud Detection (pandas + ML)

Exploratory analysis and fraud detection on 1.3M+ simulated credit card transactions, using pandas for cleaning and pattern finding, and scikit-learn / XGBoost for modelling.

## Dataset

- **Source:** [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle, kartik2112)
- **Type:** Simulated data generated with the Sparkov tool, covering 1 Jan 2019 to 31 Dec 2020, for 1,000 customers and 800 merchants
- **Size:** 1,296,675 training rows (`fraudTrain.csv`) and 555,719 test rows (`fraudTest.csv`), 23 columns
- **Class imbalance:** only 0.58% of training transactions and 0.39% of test transactions are fraud

Because the data is simulated, the clean patterns found here are easier than what real bank data would show.

## What I did

1. **Cleaning and feature engineering (pandas):** checked missing values and duplicates (none found), converted text columns to datetime, and created `hour`, `day_of_week`, `month` and `age` features.
2. **Pattern analysis (`groupby`, `pd.cut`):** fraud rate by hour, category, amount range, age group, gender and state, each compared against the 0.58% baseline.
3. **Rule-based flag:** a simple rule (amount above $250 and between 10 PM and 3 AM), evaluated on both train and held-out test data.
4. **Machine learning models:** Logistic Regression, Decision Tree, Random Forest and XGBoost, trained on `fraudTrain` and evaluated on the unseen `fraudTest`.
5. **Threshold tuning and feature importance** on the best model.

## Key findings

- Transactions above $250 are about 3% of volume but about 75% of all fraud.
- Fraud peaks at 10 PM and 11 PM (about 2.9%, roughly 5x the baseline).
- The signal is an interaction between amount and category: above $250, `grocery_pos` is 90.38% fraud while `travel` has zero fraud.
- In `gas_transport`, fraud is made of small amounts (mean $12.29) while normal transactions start at $17.
- Age and gender have only a weak effect (rates between 0.43% and 0.77% by age group).
- State-level results are unreliable for small states (for example, DE has 9 transactions, all fraud).

## Results on the unseen test set

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Rule-based flag | 30.12% | 62.61% | - | - |
| Logistic Regression | 3.49% | 73.80% | 6.66 | 0.12 |
| Decision Tree | 11.33% | 95.52% | 20.27 | 0.26 |
| Random Forest | 22.77% | 94.41% | 36.69 | 0.84 |
| **XGBoost** | **27.53%** | **95.62%** | **42.76** | **0.89** |

The random baseline for PR-AUC is the test fraud rate, about 0.004.

### Threshold tuning (XGBoost)

| Threshold | Flagged | Precision | Recall |
|---|---|---|---|
| 0.50 | 7,449 | 27.53% | 95.62% |
| 0.90 | 3,139 | 60.59% | 88.67% |
| 0.95 | 2,339 | 76.27% | 83.17% |
| 0.99 | 1,692 | 93.38% | 73.66% |

Raising the threshold from 0.5 to 0.95 cut flagged transactions by about 69%, raised precision from 27.5% to 76.3%, and kept recall at 83%. The right threshold is a business decision: a bank that wants to avoid bothering customers would pick a higher one, and a bank that wants to stop every fraud would pick a lower one.

## Why PR-AUC and not accuracy

With 0.39% fraud, a model that predicts "not fraud" for everything scores about 99.6% accuracy and catches nothing. ROC-AUC was also misleading (Logistic Regression scored 0.91 with 3.49% precision). PR-AUC focuses on the rare fraud class, so it was the main comparison metric.

## Limitations

- The data is simulated, so real-world performance would be lower.
- The rule and thresholds were chosen by looking at the data, then checked on the test set.
- Features are limited to amount, time, age, city population, gender and category. Customer history and merchant features were not used.

## Tools

Python, pandas, NumPy, matplotlib, scikit-learn, XGBoost, Kaggle Notebooks

## How to run

1. Download the dataset from the Kaggle link above.
2. Open `credit-card-fraud-detection.ipynb` (rename to match your file) in Jupyter or a Kaggle notebook.
3. Update the file paths to point to `fraudTrain.csv` and `fraudTest.csv`.
4. Run all cells from top to bottom.
