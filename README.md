# Customer Churn Prediction

A machine learning project that predicts which telecom customers are likely to cancel their subscription. It uses customer demographics, account details and service usage.

The project covers the full workflow: exploratory data analysis, data cleaning, feature engineering, a scikit-learn pipeline, model comparison with cross-validation, ROC/AUC evaluation, hyperparameter tuning, and a script that scores new customers.

## Results

Three models were compared with stratified 5-fold cross-validation on ROC-AUC:

| Model | CV ROC-AUC |
|---|---|
| Logistic Regression | 0.846 ± 0.011 |
| Random Forest | 0.845 ± 0.009 |
| **Gradient Boosting** | **0.847 ± 0.012** |

The best model (Gradient Boosting, tuned with GridSearchCV) gives these results on the held-out test set (20% of the data):

| Metric | Value |
|---|---|
| ROC-AUC | **0.845** |
| Recall (churners) | **0.79** |
| Precision (churners) | 0.51 |
| Accuracy | 0.74 |

Only about 26.5% of customers churn, so the model is trained with balanced class weights. This makes it catch about 4 out of 5 churners, at the cost of more false alarms. Without balancing, it caught only about half of them. Missing a customer who is about to leave usually costs more than contacting one who would have stayed.

**Strongest churn drivers** (permutation importance): tenure, contract type and internet service type, followed by tech support and online security. Month-to-month customers, new customers and fiber optic users churn the most.

## Dataset

The project uses the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle, with 7,043 customers and 21 columns:

- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Account:** tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Target:** `Churn` (Yes / No)

The file is stored as `data/telco_churn.csv`. It was renamed from `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Project structure

```
churn-prediction/
├── data/
│   └── telco_churn.csv          # raw dataset
├── notebooks/
│   └── churn_analysis.ipynb     # full analysis: EDA → evaluation → saved model
├── src/
│   ├── features.py              # cleaning + feature engineering (used by the pipeline)
│   └── predict.py               # score new customers with the saved model
├── models/
│   └── churn_model.pkl          # trained pipeline (created by the notebook)
├── requirements.txt
└── README.md
```

## Approach

1. **EDA:** churn rate, and churn broken down by contract, tenure, monthly charges and payment method.
2. **Cleaning:** `TotalCharges` is stored as text, with blanks for 11 brand-new customers. It is converted to a number and the blanks are set to 0. `customerID` is dropped and `Churn` is mapped to 1/0.
3. **Feature engineering** (`src/features.py`):
   - `tenure_group`: tenure buckets of 0–12, 13–24, 25–48 and 49+ months
   - `avg_monthly_spend`: TotalCharges / tenure
   - `num_services`: number of subscribed services
4. **Pipeline:** a single scikit-learn `Pipeline` with three stages: feature engineering, then a `ColumnTransformer` (`StandardScaler` on numeric columns, `OneHotEncoder` on categorical columns), then the model. Everything is fitted inside the pipeline, so cross-validation doesn't leak test data. The saved model takes raw customer rows as input.
5. **Model selection:** Logistic Regression, Random Forest and Gradient Boosting are compared with `StratifiedKFold` (5 folds) using ROC-AUC on the training set.
6. **Evaluation:** confusion matrix, classification report, ROC curves for all models, and permutation feature importance on the test set.
7. **Tuning and saving:** `GridSearchCV` on the best model, saved with `joblib`.

## Getting started

Requires Python 3.11 or newer.

```bash
git clone https://github.com/deepikan2303-cloud/churn-prediction.git
cd churn-prediction
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Run the analysis

```bash
jupyter notebook
```

Open `notebooks/churn_analysis.ipynb` and run all cells. This also creates `models/churn_model.pkl`.

### Predict churn for new customers

```bash
# Demo: scores the first 5 customers in the dataset
python src/predict.py

# Score your own CSV (same columns as the dataset; customerID and Churn are optional)
python src/predict.py path/to/customers.csv
```

Example output:

```
customerID  churn_probability churn_prediction actual_churn
7590-VHVEG              0.827              Yes           No
5575-GNVDE              0.110               No           No
3668-QPYBK              0.580              Yes          Yes
7795-CFOCW              0.105               No           No
9237-HQITU              0.840              Yes          Yes
```

When you pass a CSV, the predictions are saved next to it as `<name>_predictions.csv`.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter · joblib
