from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).parents[1]
DATA_PATH = Path(os.getenv("TELCO_DATA_PATH", ROOT / "data" / "Telco-Customer-Churn.csv"))
REPORT_DIR = ROOT / "artifacts" / "reports"


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame = frame.dropna(subset=["TotalCharges"]).drop(columns=["customerID"])
    frame["Churn"] = (frame["Churn"] == "Yes").astype(int)
    return frame


def main() -> None:
    frame = load_frame()
    reference = frame.sample(frac=0.7, random_state=42)
    current = frame.drop(reference.index).copy()
    current["MonthlyCharges"] = current["MonthlyCharges"] + 35.0
    current["Contract"] = "Month-to-month"
    reference_features = reference.drop(columns=["Churn"])
    current_features = current.drop(columns=["Churn"])
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(current_data=current_features, reference_data=reference_features)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "telco_drift_report.html"
    result.save_html(str(report_path))

    mean_shift = float(current["MonthlyCharges"].mean() - reference["MonthlyCharges"].mean())
    churn_rate_shift = float(current["Churn"].mean() - reference["Churn"].mean())
    summary = {
        "monthly_charges_mean_shift": mean_shift,
        "churn_rate_shift": churn_rate_shift,
        "engineered_drift": ["MonthlyCharges", "Contract"],
        "interpretation": "A sustained charge and contract mix shift can reduce calibration and increase false churn decisions; retraining is recommended when this persists.",
    }
    summary_path = REPORT_DIR / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    mlflow.set_tracking_uri((ROOT / "artifacts" / "mlruns").as_uri())
    mlflow.set_experiment("telco-monitoring")
    with mlflow.start_run(run_name="evidently-drift-check"):
        mlflow.log_metrics({"monthly_charges_mean_shift": mean_shift, "churn_rate_shift": churn_rate_shift})
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(summary_path))
    print(json.dumps(summary, indent=2))
    print(f"Evidently report: {report_path}")


if __name__ == "__main__":
    main()

