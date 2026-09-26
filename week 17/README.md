# Week 17: MLOps

This project implements both required tracks:

- **Track A:** a reproducible Telco Customer Churn pipeline with `uv`, MLflow
  experiments and registry promotion, FastAPI serving, and Evidently drift monitoring.
- **Track B:** regression evaluation and MLflow tracking for the deterministic W16 agent
  configuration in [`../week_16/`](../week_16/).

The Telco CSV is downloaded from the IBM sample mirror on first use and is intentionally
not committed to the repository. All generated reports, models, and MLflow data are
written below `artifacts/` (ignored by Git).

## Reproducible setup

Install [uv](https://docs.astral.sh/uv/) once, then run:

```powershell
uv sync
uv run python scripts/download_data.py
```

The committed `uv.lock` pins the complete environment. If the dataset URL is unavailable,
set `TELCO_DATA_URL` to another CSV mirror with the same columns.

## Standard workflow

### 1. Train and compare models

```powershell
uv run python scripts/train.py
```

This creates three genuinely different runs in the local MLflow file store, logging
parameters, accuracy, precision, recall, F1, ROC-AUC, model files, confusion matrices,
and ROC curves. The best run is registered as `TelcoChurnModel`, then promoted through
the `Staging` and `Production` stages. Open the comparison UI with:

```powershell
uv run mlflow ui --backend-store-uri ./artifacts/mlruns
```

### 2. Serve the registered model

```powershell
uv run uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

The API accepts the original Telco feature columns at `POST /predict`; the
`/health` endpoint reports the configured model URI. Set `MODEL_URI` to a specific
registry version when needed.

### 3. Run drift monitoring

```powershell
uv run python scripts/monitor.py
```

Monitoring takes a reproducible 70/30 reference/current split, injects numeric and
categorical drift into the current set, checks feature and target drift, calculates
the custom mean `MonthlyCharges` shift, and writes an HTML report. The report and
metrics are also logged to MLflow. The engineered drift should flag `MonthlyCharges`
and `Contract`; the target drift check is reported separately.

### 4. Track the W17 agent configuration

```powershell
uv run python track_b/evaluate.py
```

This runs three explicit prompt/configuration versions against the same five-query
regression set. Each MLflow run records the prompt text, max iterations, retrieval
settings, completion rate, average iterations, and Evidently regression-test pass rate.
The runner writes three representative JSON traces per version. A trace contains every
decision (including its reason), tool arguments, the raw retrieval result, iteration
count, and termination reason. No API key or external LLM is required: the offline
reference-based judge is deterministic and reproducible, while the resulting verdict
scores are rendered in an Evidently HTML report.

The versions are intentionally evidence-driven:

| Version | Change prompted by previous traces | Completion | Regression tests | Avg. iterations |
|---|---|---:|---:|---:|
| `prompt_v1` | Stops after the first retrieval; traces exposed failures on multi-part queries | 80% | 60% | 1.8 |
| `prompt_v2` | Adds one focused follow-up search when evidence is incomplete | 80% | 100% | 2.2 |
| `prompt_v3` | Adds explicit ambiguity handling and a no-invention instruction | 80% | 100% | 2.2 |

The final version is preferred because its traces show complete evidence gathering
without increasing the measured loop cost over v2. Open the comparison in MLflow and
inspect the generated artifacts:

```text
artifacts/agent-evaluation/comparison.json
artifacts/agent-evaluation/prompt_v1_traces.json
artifacts/agent-evaluation/prompt_v2_traces.json
artifacts/agent-evaluation/prompt_v3_traces.json
artifacts/agent-evaluation/evidently_agent_regression.html
```

The regression suite uses the `prompt_v3` responses as the approved golden set and
applies two checks per case: reference-based correctness and no unsupported claims.
The Evidently report compares the resulting pass/fail score distributions. A failed
case is retained in `regression_verdicts.json` with the judge reason and is treated as
a real promotion-blocking regression.

## Optional orchestration

`orchestration/dags/mlops_monitoring.py` is an Airflow-compatible DAG. It runs the
monitoring command weekly and logs a retraining recommendation when the custom drift
threshold is exceeded. Airflow is deliberately optional and is not part of the
default `uv sync` environment.

## Repository outputs

```text
data/                       downloaded Telco CSV
artifacts/mlruns/           MLflow tracking store
artifacts/reports/          Evidently HTML report
artifacts/models/           exported model files
scripts/train.py            Track A training and registry workflow
scripts/monitor.py          Track A drift workflow
serving/app.py              Track A FastAPI model server
track_b/evaluate.py         W16 regression evaluation tracking
```
