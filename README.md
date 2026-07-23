# Instagram Fake Account Detector

A machine learning project that classifies Instagram accounts as fake or real using account metadata and behavioral features.

**Live demo:** (https://instagram-fake-account-detector-hkaspvghf8q4whpldimhew.streamlit.app/)

## Overview

Social media platforms like Instagram are increasingly affected by fake accounts used for spam, misinformation, and artificial engagement. This project analyzes account-level features to uncover patterns that distinguish fake accounts from real ones, and deploys a live prediction tool with explainable AI.

## Project Structure

| File | Description |
|---|---|
| `eda.ipynb` | Exploratory data analysis — distributions, correlations, class balance |
| `preprocessing.ipynb` | Feature transformation — Yeo-Johnson, MinMaxScaler |
| `modeling.ipynb` | Model training and evaluation — LogReg, RF, KNN, XGBoost with CV |
| `shap_explainability.ipynb` | SHAP beeswarm and waterfall plots for model explainability |
| `streamlit_app.py` | Live prediction app with per-prediction SHAP explanation |

## Models Compared

| Model | F1 | ROC-AUC |
|---|---|---|
| Random Forest | 0.9714 | 0.9651 |
| Logistic Regression | 0.9643 | 0.9756 |
| XGBoost (tuned) | 0.9643 | 0.9523 |
| KNN | 0.9565 | 0.9420 |

XGBoost was selected for deployment with hyperparameters tuned via RandomizedSearchCV (20 iterations, 5-fold CV). Best params: 300 estimators, max depth 4, learning rate 0.05, subsample 0.7.

## Key Findings

- Follower and following counts are the strongest predictors — fake accounts tend to follow many users while having few followers
- Recently joined accounts with no external URL are strongly associated with fake behavior
- Profile text features (username length, full name length) contribute minimal predictive signal
- Bottom 3 features by SHAP importance could be dropped with negligible model impact

## Tech Stack

Python, scikit-learn, XGBoost, SHAP, feature-engine, Streamlit, pandas, matplotlib, seaborn

## Dataset

[Instagram Fake and Real Accounts](https://www.kaggle.com/datasets/rezaunderfit/instagram-fake-and-real-accounts-dataset) — 785 accounts, 13 features

## Author

Manvish Kollu — [GitHub](https://github.com/ManvishK7122) | [LinkedIn](https://www.linkedin.com/in/manvish-kollu/)
