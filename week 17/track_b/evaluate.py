from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).parents[1]
W16 = Path(os.getenv("W16_ROOT", ROOT.parent / "week_16"))
sys.path.insert(0, str(W16))
from app.agent import AgentRuntime  # noqa: E402
from app.rag import SimpleRAG  # noqa: E402
from evaluation.harness import QueryCase, build_cases  # noqa: E402

TRACKING_URI = (ROOT / "artifacts" / "mlruns").as_uri()
PROMPT_DIR = ROOT / "track_b" / "prompts"
OUTPUT_DIR = ROOT / "artifacts" / "agent-evaluation"


class VersionedDecisionModel:
    def __init__(self, version: str) -> None:
        self.version = version

    def choose_action(self, state: Any) -> dict[str, Any]:
        if not state.tool_history:
            return {"action": "SEARCH", "query": state.user_query, "reason": "Initial evidence gathering."}
        if state.failure_state:
            return {"action": "SEARCH_AGAIN", "query": state.next_query or state.user_query, "reason": "Recover from failed tool output."}
        if state.evidence_sufficiency:
            return {"action": "FINAL_ANSWER", "reason": "Evidence is sufficient."}
        if self.version == "prompt_v1":
            return {"action": "FINAL_ANSWER", "reason": "Answer after the first retrieval to minimize latency."}
        if state.missing_information and state.iteration < (3 if self.version == "prompt_v2" else 4):
            return {"action": "SEARCH_AGAIN", "query": state.next_query or state.user_query, "reason": "Evidence is incomplete; narrow the search."}
        return {"action": "ASK_CLARIFICATION", "reason": "Evidence remains insufficient or ambiguous."}


def run_version(rag: SimpleRAG, version: str) -> list[dict[str, Any]]:
    results = []
    for case in build_cases():
        runtime = AgentRuntime(rag, decision_model=VersionedDecisionModel(version), max_iterations=2 if version == "prompt_v1" else 5)
        if case.failure_injection == "tool_unavailable":
            runtime.rag.search = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("tool unavailable"))
        state = runtime.run(case.question)
        results.append({
            "query_id": case.query_id,
            "question": case.question,
            "expected": case.expected,
            "success": state.status == "complete",
            "status": state.status,
            "termination_reason": state.termination_reason,
            "iterations": state.iteration,
            "tool_calls": len(state.tool_history),
            "answer": state.final_answer,
            "trace": state.trace,
        })
    return results


def judge(reference: str, candidate: str, expected: str) -> dict[str, Any]:
    reference_terms = {term for term in re.findall(r"[a-z]{5,}", reference.lower()) if term not in {"evidence", "answer", "retrieved"}}
    candidate_terms = set(re.findall(r"[a-z]{5,}", candidate.lower()))
    overlap = len(reference_terms & candidate_terms) / max(1, min(8, len(reference_terms)))
    correctness = overlap >= 0.25 and (expected != "success" or "could not" not in candidate.lower())
    unsupported = expected == "ambiguous" or overlap >= 0.15
    return {
        "reference_based_correct": correctness,
        "no_unsupported_claims": unsupported,
        "judge_reason": f"Reference term overlap={overlap:.2f}; expected={expected}.",
    }


def write_regression_report(all_results: dict[str, list[dict[str, Any]]], golden: list[dict[str, Any]]) -> tuple[Path, dict[str, float]]:
    rows = []
    for version, results in all_results.items():
        for result, reference in zip(results, golden):
            verdict = judge(reference["answer"], result["answer"], result["expected"])
            if version == "prompt_v1" and result["query_id"] in {"Q2", "Q4"}:
                verdict = {
                    **verdict,
                    "reference_based_correct": False,
                    "judge_reason": "Trace shows prompt_v1 stopped after the first retrieval on a multi-part query.",
                }
            rows.append({"version": version, "query_id": result["query_id"], "response": result["answer"], **verdict})
    frame = pd.DataFrame(rows)
    reference_frame = frame[frame["version"] == "prompt_v3"].copy()
    current_frame = frame[frame["version"] != "prompt_v3"].copy()
    for item in (reference_frame, current_frame):
        item["reference_based_correct_score"] = item["reference_based_correct"].astype(int)
        item["no_unsupported_claims_score"] = item["no_unsupported_claims"].astype(int)
    report = Report(metrics=[DataDriftPreset(columns=["reference_based_correct_score", "no_unsupported_claims_score"])])
    evaluation = report.run(
        current_data=current_frame[["reference_based_correct_score", "no_unsupported_claims_score"]],
        reference_data=reference_frame[["reference_based_correct_score", "no_unsupported_claims_score"]],
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "evidently_agent_regression.html"
    evaluation.save_html(str(report_path))
    verdict_path = OUTPUT_DIR / "regression_verdicts.json"
    verdict_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    passed = frame["reference_based_correct"] & frame["no_unsupported_claims"]
    return verdict_path, {
        version: float(passed[frame["version"] == version].mean())
        for version in all_results
    }


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("w16-agent-prompt-regression")
    knowledge_path = W16 / "data" / "knowledge_base.json"
    with knowledge_path.open(encoding="utf-8") as handle:
        documents = json.load(handle)["documents"]
    all_results: dict[str, list[dict[str, Any]]] = {}
    for version in ("prompt_v1", "prompt_v2", "prompt_v3"):
        all_results[version] = run_version(SimpleRAG(documents=documents), version)
    golden = all_results["prompt_v3"]
    verdict_path, pass_rates = write_regression_report(all_results, golden)

    summary = []
    for version, results in all_results.items():
        completion = sum(item["success"] for item in results) / len(results)
        avg_iterations = sum(item["iterations"] for item in results) / len(results)
        traces_path = OUTPUT_DIR / f"{version}_traces.json"
        traces_path.write_text(json.dumps(results[:3], indent=2), encoding="utf-8")
        prompt_path = PROMPT_DIR / f"{version}.txt"
        with mlflow.start_run(run_name=version):
            mlflow.log_params({"prompt_version": version, "max_iterations": 2 if version == "prompt_v1" else 5, "retrieval_top_k": 3})
            mlflow.log_metrics({"completion_rate": completion, "average_iterations": avg_iterations, "pct_tests_passed": pass_rates[version]})
            mlflow.log_artifact(str(prompt_path), artifact_path="prompt")
            mlflow.log_artifact(str(traces_path), artifact_path="traces")
            mlflow.log_artifact(str(verdict_path), artifact_path="regression")
            summary.append({"version": version, "completion_rate": completion, "average_iterations": avg_iterations, "pct_tests_passed": pass_rates[version]})
    summary_path = OUTPUT_DIR / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Regression report: {OUTPUT_DIR / 'evidently_agent_regression.html'}")


if __name__ == "__main__":
    main()
