import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from feature_engine.transformation import YeoJohnsonTransformer
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Instagram Fake Account Detector",
    layout="wide"
)

# ── sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Instagram Fake Account Detector")
    st.markdown(
        "A machine learning project that classifies Instagram accounts "
        "as fake or real using account metadata and behavioral features."
    )
    st.divider()
    st.markdown("**Model**")
    st.markdown("XGBoost with hyperparameter tuning via RandomizedSearchCV")
    st.markdown("**Dataset**")
    st.markdown("785 Instagram accounts — [Kaggle](https://www.kaggle.com/datasets/rezaunderfit/instagram-fake-and-real-accounts-dataset)")
    st.divider()
    st.markdown("**Model Performance on Test Set**")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy",  "93.6%")
    c1.metric("Recall",    "97.8%")
    c2.metric("F1 Score",  "96.4%")
    c2.metric("ROC-AUC",   "95.2%")
    st.divider()
    st.markdown("**Built by Manvish Kollu**")
    st.markdown("[GitHub](https://github.com/ManvishK7122)   |   [LinkedIn](https://www.linkedin.com/in/manvish-kollu/)")

# ── train model on startup ───────────────────────────────────────────────
@st.cache_resource
def train_model():
    path = kagglehub.dataset_download("rezaunderfit/instagram-fake-and-real-accounts-dataset")
    df = pd.read_csv(os.path.join(path, "final-v1.csv"))

    num_features = ['edge_followed_by', 'edge_follow', 'username_length', 'full_name_length']
    cat_features = [col for col in df.columns if col not in num_features + ['is_fake']]

    X = df.drop(columns=['is_fake'])
    y = df['is_fake']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("transformer", YeoJohnsonTransformer()),
        ("scaler", MinMaxScaler())
    ])
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.7,
            random_state=42, eval_metric='logloss'
        ))
    ])
    model.fit(X_train, y_train)
    return model, num_features, cat_features

with st.spinner("Loading model..."):
    model, num_features, cat_features = train_model()

# ── header ───────────────────────────────────────────────────────────────
st.title("Instagram Fake Account Detector")
st.markdown(
    "Enter an Instagram account's profile features below. "
    "The model will predict whether the account is **fake or real** "
    "and explain which features drove that decision using SHAP."
)
st.divider()

# ── inputs ────────────────────────────────────────────────────────────────
st.subheader("Account Features")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Engagement Metrics**")
    edge_followed_by = st.number_input("Followers",  min_value=0, max_value=1000000, value=500,  step=10)
    edge_follow      = st.number_input("Following",  min_value=0, max_value=10000,   value=300,  step=10)
    username_length  = st.slider("Username length",  1, 30, 10)
    full_name_length = st.slider("Full name length", 0, 50, 12)

with col2:
    st.markdown("**Profile Characteristics**")
    username_has_number  = st.toggle("Username contains numbers")
    full_name_has_number = st.toggle("Full name contains numbers")
    is_private           = st.toggle("Private account")
    is_joined_recently   = st.toggle("Joined recently")
    has_channel          = st.toggle("Has channel")
    is_business_account  = st.toggle("Business account")
    has_guides           = st.toggle("Has guides")
    has_external_url     = st.toggle("Has external URL")

st.divider()

# ── prediction ────────────────────────────────────────────────────────────
if st.button("Analyse Account", type="primary", use_container_width=True):

    input_df = pd.DataFrame([{
        "edge_followed_by":      edge_followed_by,
        "edge_follow":           edge_follow,
        "username_length":       username_length,
        "full_name_length":      full_name_length,
        "username_has_number":   int(username_has_number),
        "full_name_has_number":  int(full_name_has_number),
        "is_private":            int(is_private),
        "is_joined_recently":    int(is_joined_recently),
        "has_channel":           int(has_channel),
        "is_business_account":   int(is_business_account),
        "has_guides":            int(has_guides),
        "has_external_url":      int(has_external_url),
    }])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    fake_prob   = probability[1]
    confidence  = probability[prediction]

    if prediction == 1:
        st.error(f"This account is likely FAKE — {confidence:.1%} confidence")
    else:
        st.success(f"This account is likely REAL — {confidence:.1%} confidence")

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction",       "Fake" if prediction == 1 else "Real")
    m2.metric("Confidence",       f"{confidence:.1%}")
    m3.metric("Fake Probability", f"{fake_prob:.1%}")

    st.markdown("**Fake probability**")
    st.progress(float(fake_prob))
    st.divider()

    st.subheader("Why did the model decide this?")
    st.markdown(
        "The chart below shows how each feature pushed the prediction toward "
        "fake (red) or real (blue) for this specific account."
    )

    feature_names     = num_features + cat_features
    xgb_clf           = model.named_steps["model"]
    preprocessor      = model.named_steps["preprocessor"]
    input_transformed = preprocessor.transform(input_df)

    explainer   = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(input_transformed)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_transformed[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()

    st.caption(
        "Red bars push toward fake, blue bars push toward real. "
        "Bar size shows how strongly each feature influenced this prediction."
    )