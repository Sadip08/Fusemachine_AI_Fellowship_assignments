from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


ROOT = Path(__file__).parents[1]
DATA_PATH = Path(os.getenv("TELCO_DATA_PATH", ROOT / "data" / "Telco-Customer-Churn.csv"))
MLRUNS = ROOT / "artifacts" / "mlruns"
MODEL_DIR = ROOT / "artifacts" / "models"
EXPERIMENT = "telco-churn"
MODEL_NAME = "TelcoChurnModel"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run scripts/download_data.py first.")
    frame = pd.read_csv(DATA_PATH)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame = frame.dropna(subset=["TotalCharges"]).drop(columns=["customerID"])
    target = (frame.pop("Churn") == "Yes").astype(int)
    return frame, target


def build_pipeline(model: object, numeric: list[str], categorical: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("preprocessor", transformer), ("model", model)])


def log_artifacts(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, run_dir: Path) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    cm = confusion_matrix(y_test, predictions)
    fig, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(cm, cmap="Blues")
    axis.set(title="Confusion matrix", xlabel="Predicted", ylabel="Actual")
    for (row, col), value in zip([(0, 0), (0, 1), (1, 0), (1, 1)], cm.ravel()):
        axis.text(col, row, value, ha="center", va="center")
    fig.tight_layout()
    confusion_path = run_dir / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=150)
    plt.close(fig)

    roc_path = run_dir / "roc_curve.png"
    RocCurveDisplay.from_predictions(y_test, probabilities).figure_.savefig(roc_path, dpi=150)
    report_path = run_dir / "classification_report.json"
    report_path.write_text(json.dumps(classification_report(y_test, predictions, output_dict=True), indent=2))
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(str(run_dir))
    return metrics


def main() -> None:
    mlflow.set_tracking_uri(MLRUNS.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    X, y = load_data()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = [
        ("logistic_regression", LogisticRegression(C=0.5, max_iter=1000), {"C": 0.5}),
        ("random_forest", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42), {"n_estimators": 200, "max_depth": 6}),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42), {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3}),
    ]
    results: list[tuple[str, str, float]] = []
    for name, estimator, parameters in candidates:
        with mlflow.start_run(run_name=name) as run:
            model = build_pipeline(estimator, numeric, categorical)
            model.fit(X_train, y_train)
            mlflow.log_params({**parameters, "model_family": name, "test_size": 0.2, "random_state": 42})
            run_dir = MODEL_DIR / run.info.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics = log_artifacts(model, X_test, y_test, run_dir)
            mlflow.sklearn.log_model(model, "model", registered_model_name=MODEL_NAME)
            results.append((run.info.run_id, name, metrics["roc_auc"]))
            print(f"{name}: ROC-AUC={metrics['roc_auc']:.4f}")

    best_run_id = max(results, key=lambda item: item[2])[0]
    client = MlflowClient(tracking_uri=MLRUNS.as_uri())
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    best_version = max((v for v in versions if v.run_id == best_run_id), key=lambda v: int(v.version))
    client.transition_model_version_stage(MODEL_NAME, best_version.version, "Staging", archive_existing_versions=True)
    client.transition_model_version_stage(MODEL_NAME, best_version.version, "Production", archive_existing_versions=True)
    print(f"Registered best run {best_run_id} as {MODEL_NAME} version {best_version.version} (Production)")


if __name__ == "__main__":
    main()

