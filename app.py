"""Streamlit evaluation interface for the trained Mice Protein classifiers.

The app loads saved pipelines only. It does not train or refit models.
MouseID is optional metadata and is never passed to a model.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
META_PATH = RESULTS_DIR / "run_metadata.json"
COMPARISON_PATH = ROOT / "model_comparison.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(
    page_title="Mice Protein Classifier",
    page_icon="🧬",
    layout="wide",
)


@st.cache_data
def load_metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def calculate_auc(model, X, y):
    """Calculate weighted one-vs-rest AUC for the uploaded labelled sample."""
    probabilities = model.predict_proba(X)
    model_classes = [str(value) for value in model.classes_]
    y_labels = pd.Series(y).astype(str)

    # AUC is undefined if the evaluation sample does not contain every class.
    if set(y_labels.unique()) != set(model_classes):
        raise ValueError(
            "The uploaded test data must contain examples from every model class "
            "to calculate multiclass AUC."
        )

    return roc_auc_score(
        y_labels,
        probabilities,
        labels=model_classes,
        multi_class="ovr",
        average="weighted",
    )


st.title("Mice Protein Expression Classification")
st.caption("M.Tech AIML | Machine Learning Assignment 2")
st.write(
    "Evaluate five pre-trained classifiers on labelled protein-expression data. "
    "The app loads saved models and does not retrain them on uploaded data."
)

if not META_PATH.exists():
    st.error("Experiment metadata is missing. Run `python train_models.py` first.")
    st.stop()

metadata = load_metadata()
feature_columns = metadata["feature_columns"]
target_column = metadata["target_column"]
class_labels = [str(value) for value in metadata["classes"]]

with st.sidebar:
    st.header("Evaluation settings")
    selected_model = st.selectbox("Select a classifier", list(MODEL_FILES.keys()))
    st.divider()
    st.write("**Dataset:** UCI Mice Protein Expression")
    st.write(f"**Protein predictors:** {len(feature_columns)}")
    st.write(f"**Target classes:** {len(class_labels)}")
    st.write("**Validation:** Mouse-level group split")

if COMPARISON_PATH.exists():
    st.subheader("Training-time model comparison")
    comparison = pd.read_csv(COMPARISON_PATH)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

uploaded_file = st.file_uploader(
    "Upload labelled test CSV", type=["csv"],
    help="Use the generated test_data.csv. It includes all protein predictors, "
         "the target column, and MouseID as optional metadata.",
)

if uploaded_file is None:
    st.info("Upload the generated `test_data.csv` to view evaluation metrics.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

required_columns = feature_columns + [target_column]
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()

# MouseID is deliberately not included here, even when present in test_data.csv.
# Only the exact protein feature columns are passed to the fitted pipeline.
X_test = data.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
y_test = data[target_column].astype(str)

if X_test.isna().all(axis=1).any():
    st.error("At least one row has no usable numeric protein measurements.")
    st.stop()

unknown_labels = sorted(set(y_test.unique()) - set(class_labels))
if unknown_labels:
    st.error("Unknown target labels found: " + ", ".join(unknown_labels))
    st.stop()

model_path = MODEL_DIR / MODEL_FILES[selected_model]
if not model_path.exists():
    st.error(f"Saved model `{model_path.name}` is missing. Run train_models.py first.")
    st.stop()

try:
    model = load_model(str(model_path))
    predictions = model.predict(X_test).astype(str)
except Exception as exc:
    st.error(f"Model prediction failed: {exc}")
    st.stop()

try:
    auc_value = calculate_auc(model, X_test, y_test)
except Exception as exc:
    auc_value = None
    st.warning(f"AUC could not be calculated for this upload: {exc}")

metric_values = [
    ("Accuracy", accuracy_score(y_test, predictions)),
    ("AUC (weighted OVR)", auc_value),
    ("Precision (weighted)", precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )),
    ("Recall (weighted)", recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )),
    ("F1 (weighted)", f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )),
    ("MCC", matthews_corrcoef(y_test, predictions)),
]

st.subheader(f"Evaluation results — {selected_model}")
st.caption(
    f"{len(data):,} labelled observations | saved model loaded without retraining | "
    f"MouseID excluded from model inputs"
)
metric_columns = st.columns(3)
for index, (metric_name, metric_value) in enumerate(metric_values):
    display_value = "N/A" if metric_value is None else f"{metric_value:.4f}"
    metric_columns[index % 3].metric(metric_name, display_value)

left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Confusion matrix")
    matrix_labels = class_labels
    matrix = confusion_matrix(y_test, predictions, labels=matrix_labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=matrix_labels,
    ).plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format="d",
    )
    axis.set_title(selected_model)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

with right_column:
    st.subheader("Classification report")
    report = classification_report(
        y_test,
        predictions,
        labels=matrix_labels,
        zero_division=0,
        output_dict=True,
    )
    st.dataframe(
        pd.DataFrame(report).transpose().style.format(precision=3),
        use_container_width=True,
    )

st.subheader("Prediction preview")
prediction_preview = pd.DataFrame({
    "Actual class": y_test.to_numpy(),
    "Predicted class": predictions,
})
if "MouseID" in data.columns:
    prediction_preview.insert(0, "MouseID", data["MouseID"].astype(str).to_numpy())
prediction_preview["Correct"] = (
    prediction_preview["Actual class"] == prediction_preview["Predicted class"]
)
st.dataframe(prediction_preview.head(30), use_container_width=True)

st.download_button(
    "Download predictions CSV",
    data=prediction_preview.to_csv(index=False).encode("utf-8"),
    file_name="mice_protein_predictions.csv",
    mime="text/csv",
)
