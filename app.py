
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import shap

st.set_page_config(page_title="Healthcare ML Dashboard", page_icon="🩺", layout="wide")

# ------------------------ THEME TOGGLE ------------------------ #
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0)
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp { background-color:#0e1117; color:#e5e7eb; }
        .stSelectbox, .stButton>button, .stTextInput>div>div>input { border-radius:10px; }
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp { background-color:#ffffff; color:#111827; }
        .stSelectbox, .stButton>button, .stTextInput>div>div>input { border-radius:10px; }
        </style>
        """, unsafe_allow_html=True
    )

st.title("🏥 Healthcare Data Analysis — ML Dashboard")

# ------------------------ DATASET REGISTRY ------------------------ #
DATASETS = {
    "Diabetes (Pima Indians)": {
        "path": "data/diabetes.csv",
        "target": "Outcome",
        "drop": [],
        "categorical": [],
        "notes": "Zeros in Glucose/BloodPressure/SkinThickness/Insulin/BMI are treated as missing and imputed with median."
    },
    "Heart Disease (UCI)": {
        "path": "data/heart.csv",
        "target": "target",
        "drop": [],
        "categorical": ["sex","cp","fbs","restecg","exang","slope","ca","thal"],
        "notes": "Integer-coded categorical features (cp, thal, etc.) are one-hot encoded."
    },
    "Stroke Prediction": {
        "path": "data/stroke.csv",
        "target": "stroke",
        "drop": ["id"],
        "categorical": ["gender","ever_married","work_type","Residence_type","smoking_status"],
        "notes": "Non-numeric features are one-hot encoded; missing BMI imputed."
    },
    "Breast Cancer (Diagnostic)": {
        "path": "data/breast_cancer.csv",
        "target": "diagnosis",
        "drop": ["id","Unnamed: 32"],
        "categorical": [],
        "notes": "Target 'diagnosis' (M/B) encoded to 1/0 automatically."
    }
}

with st.sidebar:
    st.subheader("Dataset & Model Settings")
    dataset_name = st.selectbox("Select dataset", list(DATASETS.keys()))
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)
    model_choice = st.selectbox("Select model", ["Logistic Regression","Random Forest","XGBoost"])
    do_shap = st.checkbox("Compute SHAP explanations (slower)", value=False)
    sample_for_plots = st.slider("Sample rows for pairplot/SHAP", 200, 3000, 800, 100)

def load_dataframe(cfg):
    df = pd.read_csv(cfg["path"])
    # dataset-specific fixes
    if dataset_name.startswith("Diabetes"):
        for col in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
    if dataset_name.startswith("Breast Cancer") and df[cfg["target"]].dtype == object:
        df[cfg["target"]] = (df[cfg["target"]].str.upper().str[0] == "M").astype(int)
    return df

def build_preprocess_pipelines(df, cfg):
    target = cfg["target"]
    drops = [c for c in cfg["drop"] if c in df.columns]
    df = df.drop(columns=drops, errors="ignore").copy()
    y = df[target]
    X = df.drop(columns=[target]).copy()

    cat_cols = [c for c in X.columns if X[c].dtype == "object" or c in cfg["categorical"]]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer",  SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    return X, y, preprocessor, num_cols, cat_cols

def get_model(choice):
    if choice == "Logistic Regression":
        return LogisticRegression(max_iter=2000)
    if choice == "Random Forest":
        return RandomForestClassifier(n_estimators=400, random_state=42)
    if choice == "XGBoost":
        return XGBClassifier(
            n_estimators=400, learning_rate=0.08, max_depth=4, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, eval_metric="logloss", n_jobs=4
        )
    raise ValueError("Unknown model")

def evaluate_report(y_test, y_pred, y_proba):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    return pd.DataFrame({"metric":["accuracy","precision","recall","f1","roc_auc"],
                         "value":[acc,prec,rec,f1,auc]})

def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def plot_roc_curve(y_test, y_proba):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_proba):.3f}")
    ax.plot([0,1],[0,1],'--', alpha=0.6)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend()
    ax.set_title("ROC Curve")
    st.pyplot(fig)

def plot_feature_importance(model, feature_names, top_k=20):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        st.info("Feature importance not available for this model.")
        return None

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(7, max(4, int(top_k/2))))
    sns.barplot(y="feature", x="importance", data=imp_df, ax=ax)
    ax.set_title("Top Feature Importances")
    st.pyplot(fig)
    return imp_df

def shap_summary(pipeline, X_sample, feature_names):
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocessor"]
    X_proc = pre.transform(X_sample)

    try:
        if hasattr(model, "get_booster") or hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_proc)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_proc[:100])
            shap_values = explainer.shap_values(X_proc[:200])

        st.write("### SHAP Summary Plot")
        fig = plt.gcf()
        shap.summary_plot(shap_values, X_proc, feature_names=feature_names, plot_type="bar", show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.warning(f"SHAP failed: {e}")

# ------------------------ LOAD DATA ------------------------ #
cfg = DATASETS[dataset_name]
st.caption(cfg["notes"])
df = load_dataframe(cfg)
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ------------------------ ADVANCED EDA ------------------------ #
st.subheader("Missing Value Analysis")
fig, ax = plt.subplots()
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax)
ax.set_title("Missing Values Heatmap")
st.pyplot(fig)
st.write("Missing % by column")
st.write((df.isnull().mean()*100).round(2).rename("missing_%"))

target = cfg["target"]
if target in df.columns:
    st.subheader("Target Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots()
        df[target].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_title("Counts by target")
        st.pyplot(fig1)
    with c2:
        fig2, ax2 = plt.subplots()
        df[target].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax2)
        ax2.set_ylabel("")
        ax2.set_title("Target share")
        st.pyplot(fig2)

    st.subheader("Pairplot (sampled)")
    try:
        sample_df = df.sample(min(sample_for_plots, len(df)), random_state=0)
        if pd.api.types.is_numeric_dtype(sample_df[target]) and sample_df[target].nunique() <= 10:
            sample_df["target_label"] = sample_df[target].map(lambda x: "Positive" if int(x)==1 else "Negative")
            hue_col = "target_label"
        else:
            hue_col = target
        sns.pairplot(sample_df.select_dtypes(include=[np.number]).join(sample_df[hue_col]), hue=hue_col, diag_kind="kde")
        st.pyplot(plt.gcf(), clear_figure=True)
    except Exception as e:
        st.info(f"Pairplot skipped: {e}")

# ------------------------ PREP & TRAIN ------------------------ #
X, y, preprocessor, num_cols, cat_cols = build_preprocess_pipelines(df, cfg)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=random_state, stratify=y)

model = get_model(model_choice)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

if st.button("🔁 Train / Evaluate"):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline.named_steps["model"], "predict_proba") else None

    st.subheader("📈 Model Performance")
    report_df = evaluate_report(y_test, y_pred, y_proba)
    st.dataframe(report_df, use_container_width=True)

    st.download_button("Download metrics (CSV)", report_df.to_csv(index=False).encode(),
                       file_name=f"{dataset_name.replace(' ','_').lower()}_{model_choice.replace(' ','_').lower()}_metrics.csv",
                       mime="text/csv")

    c1, c2 = st.columns(2)
    with c1: plot_confusion(y_test, y_pred)
    with c2:
        if y_proba is not None:
            plot_roc_curve(y_test, y_proba)
        else:
            st.info("ROC curve not available for this model.")

    # Get post-encoding feature names
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    ohe_features = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) > 0 else []
    feature_names = list(num_cols) + ohe_features

    st.subheader("🔍 Feature Importance")
    imp_df = plot_feature_importance(pipeline.named_steps["model"], feature_names)

    if imp_df is not None:
        st.download_button("Download feature importances (CSV)", imp_df.to_csv(index=False).encode(),
                           file_name=f"{dataset_name.replace(' ','_').lower()}_{model_choice.replace(' ','_').lower()}_feature_importances.csv",
                           mime="text/csv")

    if do_shap:
        shap.initjs()
        X_sample = X_test.sample(min(sample_for_plots, len(X_test)), random_state=0)
        shap_summary(pipeline, X_sample, feature_names)

    st.session_state["trained_pipeline"] = pipeline
    st.success("Training complete. You can now use the prediction form below.")

# ------------------------ REAL-TIME PREDICTION ------------------------ #
st.subheader("🧪 Real-time Prediction")
pipe = st.session_state.get("trained_pipeline", None)
with st.form("rt_form"):
    user_inputs = {}
    for col in X.columns:
        if (col in cat_cols) or (X[col].dtype == "object"):
            opts = sorted([x for x in X[col].dropna().unique().tolist() if str(x) != "nan"])
            if not opts:
                user_inputs[col] = ""
            else:
                user_inputs[col] = st.selectbox(col, opts, index=0, key=f"sel_{col}")
        else:
            series = pd.to_numeric(X[col], errors="coerce")
            med = float(series.median()) if series.notna().any() else 0.0
            q1 = float(series.quantile(0.01)) if series.notna().any() else med - 1.0
            q9 = float(series.quantile(0.99)) if series.notna().any() else med + 1.0
            if q1 >= q9:
                q1, q9 = med-1.0, med+1.0
            user_inputs[col] = st.number_input(col, value=med, min_value=q1, max_value=q9, key=f"num_{col}")
    submitted = st.form_submit_button("Predict")
    if submitted:
        if pipe is None:
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", get_model(model_choice))])
            pipe.fit(X, y)
        inp_df = pd.DataFrame([user_inputs])
        pred = pipe.predict(inp_df)[0]
        proba = pipe.predict_proba(inp_df)[0,1] if hasattr(pipe.named_steps["model"], "predict_proba") else None
        st.success(f"Prediction: **{int(pred)}** {'(Positive)' if int(pred)==1 else '(Negative)'}")
        if proba is not None:
            st.info(f"Confidence (probability of class 1): {proba:.3f}")
