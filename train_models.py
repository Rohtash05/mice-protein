"""Train five classifiers on the UCI Mice Protein Expression dataset (ID 342).

Run from this project directory:
    python train_models.py

Creates five fitted pipelines, a labelled held-out test CSV (including MouseID
as metadata), model-comparison metrics, confusion matrices and run metadata.
MouseID is used only for group-aware splitting and is NEVER a model feature.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


def get_mouse_groups(ids):
    """Return MouseID from ucimlrepo's separate data.ids object."""
    if ids is None:
        raise ValueError(
            "mice.data.ids is None. Cannot perform the required mouse-level "
            "group split. Inspect the UCI response; do not use a random split."
        )

    if isinstance(ids, pd.Series):
        normalized = str(ids.name).strip().lower().replace("_", "").replace(" ", "")
        if normalized == "mouseid":
            return ids
        raise ValueError(f"Unexpected ID Series name: {ids.name!r}")

    normalized_columns = {
        str(col).strip().lower().replace("_", "").replace(" ", ""): col
        for col in ids.columns
    }
    mouse_column = normalized_columns.get("mouseid")
    if mouse_column is None:
        raise ValueError(
            "MouseID not found in mice.data.ids. Available ID columns: "
            f"{ids.columns.tolist()}"
        )
    return ids[mouse_column]


def calculate_metrics(y_true, y_pred, y_prob, classes):
    """Calculate the six required metrics using weighted multiclass averages."""
    try:
        auc = roc_auc_score(
            y_true,
            y_prob,
            labels=classes,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError as exc:
        warnings.warn(f"AUC could not be calculated for this split: {exc}")
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    print("Fetching UCI Mice Protein Expression dataset (ID 342)...")
    mice = fetch_ucirepo(id=342)

    raw_features = mice.data.features.copy()
    raw_targets = mice.data.targets.copy()
    raw_ids = mice.data.ids

    if raw_targets is None or raw_targets.shape[1] == 0:
        raise ValueError("The UCI response does not contain a target column.")

    # MouseID is stored separately by ucimlrepo; it is not in raw_features.
    groups = get_mouse_groups(raw_ids)

    # Retain protein-expression predictors only. Genotype/Treatment/Behavior
    # metadata is deliberately excluded to avoid using target-defining metadata.
    X = raw_features.drop(
        columns=["Genotype", "Treatment", "Behavior", "Behaviour"],
        errors="ignore",
    ).copy()
    y = raw_targets.iloc[:, 0].astype(str).copy()

    # Align all objects by row position before splitting.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.astype(str).reset_index(drop=True)

    if not (len(X) == len(y) == len(groups)):
        raise ValueError(
            f"Row mismatch: features={len(X)}, target={len(y)}, "
            f"MouseID={len(groups)}. Check the UCI data alignment."
        )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    feature_columns = X.columns.tolist()

    if len(X) < 500 or len(feature_columns) < 12:
        raise ValueError(
            f"Dataset minimum not met: {len(X)} rows, {len(feature_columns)} features."
        )

    print(f"Rows: {len(X)} | Protein features: {len(feature_columns)}")
    print(f"Classes: {y.nunique()} | Unique mice: {groups.nunique()}")
    print(f"Missing feature values: {int(X.isna().sum().sum())}")

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    overlap = set(train_groups.unique()).intersection(set(test_groups.unique()))
    if overlap:
        raise RuntimeError(
            f"MouseID leakage detected: {len(overlap)} mice occur in both partitions."
        )

    classes = np.asarray(sorted(y_train.unique()))
    if set(y_test.unique()) != set(y.unique()):
        warnings.warn("The test fold does not contain every class; AUC may be undefined.")

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000, random_state=RANDOM_STATE
            )),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=3, random_state=RANDOM_STATE
            )),
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf", probability=True, random_state=RANDOM_STATE
            )),
        ]),
    }

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Gaussian Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib",
    }

    metric_rows = []
    reports = {}

    for model_name, pipeline in models.items():
        print(f"\nTraining {model_name}...")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        metrics = calculate_metrics(y_test, predictions, probabilities, classes)
        metric_rows.append({"Model": model_name, **metrics})

        joblib.dump(pipeline, MODEL_DIR / model_files[model_name])
        reports[model_name] = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )

        matrix = confusion_matrix(y_test, predictions, labels=classes)
        safe_name = model_name.lower().replace(" ", "_")
        pd.DataFrame(
            matrix,
            index=[f"Actual_{label}" for label in classes],
            columns=[f"Predicted_{label}" for label in classes],
        ).to_csv(RESULTS_DIR / f"{safe_name}_confusion_matrix.csv")

        metric_text = " | ".join(
            f"{key}: {value:.4f}" if pd.notna(value) else f"{key}: N/A"
            for key, value in metrics.items()
        )
        print(f"  {metric_text}")

    comparison = pd.DataFrame(metric_rows)
    comparison.to_csv(ROOT / "model_comparison.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    with (RESULTS_DIR / "classification_reports.json").open("w") as file:
        json.dump(reports, file, indent=2, default=float)

    # The held-out CSV includes MouseID for traceability, but app.py excludes it
    # from predictors. The target column is required for evaluation in the app.
    target_column = str(raw_targets.columns[0])
    test_data = X_test.copy()
    test_data[target_column] = y_test.to_numpy()
    test_data["MouseID"] = test_groups.to_numpy()
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    metadata = {
        "dataset": "UCI Mice Protein Expression",
        "uci_dataset_id": 342,
        "rows": int(len(X)),
        "protein_predictor_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "classes": [str(value) for value in sorted(y.unique())],
        "class_count": int(y.nunique()),
        "unique_mouse_groups": int(groups.nunique()),
        "split": "First fold of StratifiedGroupKFold",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_test_mouse_overlap": int(len(overlap)),
        "precision_recall_f1_average": "weighted",
        "auc": "multiclass one-vs-rest, weighted",
        "mouse_id_in_test_csv": True,
        "mouse_id_used_as_predictor": False,
    }
    with (RESULTS_DIR / "run_metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    print("\nTraining completed successfully.")
    print(f"Five trained pipelines saved in: {MODEL_DIR}")
    print(f"Labelled test CSV saved to: {ROOT / 'test_data.csv'}")
    print(f"MouseID overlap between train/test: {len(overlap)}")
    print("\nModel comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
