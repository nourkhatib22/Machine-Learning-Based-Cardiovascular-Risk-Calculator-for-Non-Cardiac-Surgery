# Machine-Learning-Based-Cardiovascular-Risk-Calculator-for-Non-Cardiac-Surgery
# Cardiovascular Risk Scoring — NSQIP 2022 (ML + Cleaning & Analysis)

## Background
Annually, 4% of the global population undergoes non‑cardiac surgery; 30% of those patients have ≥1 cardiovascular risk factor. Estimated 30‑day mortality after these procedures is 0.5–2%.  
This project develops a traditional Machine Learning (ML) model that outputs a cardiovascular risk score for patients aged ≥50 undergoing non‑cardiac surgery, predicting the composite endpoint (death, myocardial infarction, cardiac arrest, or stroke) within 30 days post‑op. Emphasis is placed on interpretability and explainability.

Abstract (study summary)
- Dataset: NSQIP 2022, final cleaned cohort: 497,011 patients.
- Endpoint incidence: 1.44% (death/MI/arrest/stroke within 30 days).
- Methods: data cleaning, feature selection, modeling with Logistic Regression, Naive Bayes, Random Forest, and boosting tree algorithms (CatBoost, AdaBoost, LightGBM, XGBoost, Gradient Boosting). Models evaluated with AUROC and 95% CI.
- Results: LightGBM achieved best test AUROC = 0.9009 (95% CI 0.8889 – 0.9126). Final model uses 6 features: surgery type, ASA class, BUN, sepsis, emergent surgery, mechanical ventilation.
- Conclusion: LightGBM provided the best balance of accuracy and generalization for 30‑day cardiovascular risk scoring.
- Keywords: NSQIP, AUROC, LightGBM, XAI, SHAP

---

## Repository overview (files of interest)
- `Cleaned.py`  
  Data ingestion and cleaning pipeline:
  - Loads NSQIP `.sav` file (pyreadstat) with fallback encoding.
  - Replaces sentinel values with NaN, drops columns with >30% missing or >30% literal `"NULL"`.
  - Filters patients age ≥50 and drops administrative/temporal fields.
  - Builds composite binary outcome `Complication` from complication indicator columns.
  - Saves cleaned CSV to `D:\NSQIP_2022_Cleaned.csv`.

- `preprocessing.py`  
  Modeling, HPO, evaluation, and calibration:
  - Loads an encoded CSV (`D:Encoded_Data.csv`) with selected features.
  - Downsamples majority class to ~1:9 ratio, splits data (90/10) stratified.
  - Defines models and hyperparameter grids (LightGBM, XGBoost, CatBoost, RandomForest, AdaBoost, GradientBoosting, Naive Bayes, Logistic Regression).
  - Runs GridSearchCV (stratified 10‑fold, scoring AUROC), computes train/test AUROC, bootstrap 95% CI for test AUROC, CV mean AUROC, Brier score.
  - Plots ROC curves and calibration plots for best hyperparameter models.
  - Saves final best model as `LightGBM_Model_Saved.joblib`.

---

## Requirements

- Python 3.8+ (Windows recommended)
- Core Python packages:
  - pandas
  - numpy
  - matplotlib
  - scipy
  - scikit-learn
  - pyreadstat
  - joblib
  - lightgbm
  - xgboost
  - catboost
  - shap (for explainability/SHAP analyses)
  - seaborn (optional, for plotting)
