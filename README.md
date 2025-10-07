
# Healthcare-Data-Analysis — Advanced Streamlit App

This version includes multi-dataset support, advanced EDA, model selection (LogReg/RandomForest/XGBoost), metrics (Accuracy/Precision/Recall/F1/ROC-AUC), confusion matrix + ROC curve, feature importance, optional SHAP explanations, a real-time prediction form, downloads for metrics & feature importances, and a Dark/Light theme toggle.

## 1) Folder layout
```
Healthcare-Data-Analysis-Advanced/
│── app.py
│── requirements.txt
│── data/                # put CSVs here
│── outputs/             # (created at runtime if you add saving)
```

## 2) Datasets (download & rename)
Place the following CSVs into `data/`:

- **Diabetes (Pima Indians)** — Kaggle: `uciml/pima-indians-diabetes-database`  
  Save as: `data/diabetes.csv` — Target: `Outcome`

- **Heart Disease UCI** — Kaggle: `ronitf/heart-disease-uci` (or `johnsmith88/heart-disease-dataset`)  
  Save as: `data/heart.csv` — Target: `target`

- **Stroke Prediction** — Kaggle: `fedesoriano/stroke-prediction-dataset`  
  Download `healthcare-dataset-stroke-data.csv` → save as: `data/stroke.csv` — Target: `stroke` (drop column `id`)

- **Breast Cancer (Diagnostic)** — Kaggle: `uciml/breast-cancer-wisconsin-data`  
  Download `data.csv` → save as: `data/breast_cancer.csv` — Target: `diagnosis` (M/B) auto-encoded to 1/0

Notes handled automatically:
- Diabetes: 0 values in Glucose/BloodPressure/SkinThickness/Insulin/BMI → treated as missing, median-imputed.
- Heart: integer-coded categorical features (`cp, thal, ...`) one-hot encoded.
- Stroke: `id` dropped; categorical strings encoded.
- Breast Cancer: `diagnosis` mapped to 1/0.

## 3) Install & run
**Windows PowerShell**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```
**Mac/Linux**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## 4) Using the app(https://ebieaq6ryfd5qp4xdnbtss.streamlit.app/)
- Use the **sidebar** to choose dataset, model, test size, and whether to compute **SHAP**.
- Click **Train / Evaluate** to view metrics, confusion matrix, ROC curve, feature importances, and downloads.
- Use **Real-time Prediction** (bottom) to input patient features and get a prediction + confidence.

## 5) Add new datasets
Edit the `DATASETS` dict at the top of `app.py`:
```python
"Your Dataset Name": {
  "path": "data/your_file.csv",
  "target": "your_target",
  "drop": ["id_col_if_any"],
  "categorical": ["cat1","cat2"],
  "notes": "anything special"
}
```
If your target is non-numeric (e.g., "Yes/No"), map it to 1/0 before modeling.
