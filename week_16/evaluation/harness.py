import json
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.agent import AgentRuntime
from app.rag import SimpleRAG


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


class DeterministicDecisionModel:
    def choose_action(self, state: Any) -> dict[str, Any]:
        if not state.tool_history:
            return {"action": "SEARCH", "query": state.user_query, "reason": "Initial evidence gathering."}

        if state.failure_state:
            return {"action": "SEARCH_AGAIN", "query": state.next_query or state.user_query, "reason": "Recover from failed or malformed output."}

        if state.evidence_sufficiency:
            return {"action": "FINAL_ANSWER", "reason": "Evidence is sufficient."}

        if state.missing_information:
            if state.iteration >= 3 or "ambiguous" in state.user_query.lower():
                return {"action": "ASK_CLARIFICATION", "reason": "Question remains ambiguous after additional evidence."}
            return {"action": "SEARCH_AGAIN", "query": state.next_query or state.user_query, "reason": "Collect more evidence before answering."}

        return {"action": "FINAL_ANSWER", "reason": "Synthesis ready."}


@dataclass
class QueryCase:
    query_id: str
    question: str
    expected: str
    failure_injection: str | None = None


class EvaluationHarness:
    def __init__(self, rag: SimpleRAG, cases: list[QueryCase]) -> None:
        self.rag = rag
        self.cases = cases
        self.results: list[dict[str, Any]] = []

    def run_query(self, case: QueryCase) -> dict[str, Any]:
        runtime = AgentRuntime(self.rag, decision_model=DeterministicDecisionModel(), max_iterations=5)
        if case.failure_injection == "tool_unavailable":
            def bad_search(*args: Any, **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("tool unavailable")

            runtime.rag.search = bad_search  # type: ignore[assignment]

        state = runtime.run(case.question)
        tool_calls = len(state.tool_history)
        input_tokens = estimate_tokens(case.question) * 2 + estimate_tokens(" ".join([item.get("query", "") for item in state.tool_history]))
        output_tokens = estimate_tokens(state.final_answer)
        total_tokens = input_tokens + output_tokens

        success = state.status == "complete"
        failure_type = "None"
        if state.status == "failed":
            failure_type = "Hard failure" if state.failure_state == "max_iterations_reached" else "Soft failure"
        elif state.status == "needs_clarification":
            failure_type = "Soft failure"
        elif state.status == "complete" and tool_calls > 2:
            failure_type = "Cascading soft failure" if state.failure_state else "Soft failure"

        result = {
            "query_id": case.query_id,
            "success": success,
            "iterations": state.iteration,
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "failure_type": failure_type,
            "status": state.status,
            "final_answer": state.final_answer,
            "tool_history": state.tool_history,
        }
        self.results.append(result)
        return result

    def run(self) -> list[dict[str, Any]]:
        for case in self.cases:
            self.run_query(case)
        return self.results

    def report(self) -> str:
        results = self.results
        if not results:
            return "No evaluation results recorded."

        completion_rate = sum(1 for item in results if item["success"]) / len(results)
        tool_correctness = sum(
            1 for item in results if item["tool_calls"] > 0 and item["status"] in {"complete", "needs_clarification"}
        ) / len(results)
        trajectory_lengths = [item["iterations"] for item in results]
        tokens = [item["total_tokens"] for item in results]
        hard_failures = sum(1 for item in results if item["failure_type"] == "Hard failure")
        soft_failures = sum(1 for item in results if item["failure_type"] == "Soft failure")
        cascading = sum(1 for item in results if item["failure_type"] == "Cascading soft failure")

        lines = [
            "# W16 Evaluation Report",
            "",
            "| Query | Success | Iterations | Tool Calls | Tokens | Failure |",
            "|-------|---------|------------|------------|--------|---------|",
        ]
        for item in results:
            lines.append(
                f"| {item['query_id']} | {'Yes' if item['success'] else 'No'} | {item['iterations']} | {item['tool_calls']} | {item['total_tokens']} | {item['failure_type']} |"
            )
        lines.extend([
            "",
            f"- Task completion rate: {completion_rate * 100:.1f}%",
            f"- Tool-call correctness: {tool_correctness * 100:.1f}%",
            f"- Average trajectory length: {statistics.mean(trajectory_lengths):.2f}",
            f"- Average tokens/query: {statistics.mean(tokens):.1f}",
            f"- Maximum iterations: {max(trajectory_lengths)}",
            f"- Hard failures: {hard_failures}",
            f"- Soft failures: {soft_failures}",
            f"- Cascading soft failures: {cascading}",
            "",
            "The agent intentionally injects one tool failure during the evaluation to confirm it detects invalid tool output and terminates safely.",
        ])
        return "\n".join(lines)


def build_cases() -> list[QueryCase]:
    return [
        QueryCase("Q1", "What is the refund policy?", "simple"),
        QueryCase("Q2", "Can the app work offline and create a new support ticket?", "needs_more_evidence"),
        QueryCase("Q3", "What is the support setup?", "ambiguous"),
        QueryCase("Q4", "Compare backup retention and support response targets for premium customers.", "verification"),
        QueryCase("Q5", "How long do backups remain available?", "success", "tool_unavailable"),
    ]


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    knowledge_path = os.path.join(base_dir, "data", "knowledge_base.json")
    with open(knowledge_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    rag = SimpleRAG(documents=data["documents"])
    harness = EvaluationHarness(rag, build_cases())
    harness.run()
    report = harness.report()
    result_path = os.path.join(base_dir, "evaluation", "results.md")
    with open(result_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
