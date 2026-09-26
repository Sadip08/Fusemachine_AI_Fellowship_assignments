from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.tools import evaluate_evidence, search_documents, summarize_evidence


@dataclass
class AgentState:
    user_query: str
    conversation_context: str = ""
    iteration: int = 0
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    retrieval_results: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    evidence_sufficiency: bool = False
    missing_information: list[str] = field(default_factory=list)
    status: str = "pending"
    final_answer: str = ""
    failure_state: str | None = None
    next_query: str | None = None


class DecisionModel:
    def choose_action(self, state: AgentState) -> dict[str, Any]:
        lowered = state.user_query.lower()
        if not state.tool_history:
            if "support setup" in lowered or "what is the support setup" in lowered:
                return {"action": "ASK_CLARIFICATION", "reason": "The question is too generic to answer confidently without specific context."}
            return {"action": "SEARCH", "query": state.user_query, "reason": "Initial evidence gathering."}

        if state.failure_state:
            return {"action": "SEARCH_AGAIN", "query": state.next_query or state.user_query, "reason": "Recover from failed tool or malformed evidence."}

        if state.evidence_sufficiency:
            return {"action": "FINAL_ANSWER", "reason": "Evidence is sufficient to answer the question."}

        if state.missing_information:
            if state.iteration >= 2:
                return {"action": "ASK_CLARIFICATION", "reason": "The evidence still does not answer the question clearly."}
            return {"action": "SEARCH_AGAIN", "query": state.next_query or state.user_query, "reason": "Search again with narrower evidence gathering."}

        return {"action": "FINAL_ANSWER", "reason": "Evidence has been gathered and is ready for synthesis."}


class AgentRuntime:
    def __init__(self, rag: Any, decision_model: DecisionModel | None = None, max_iterations: int = 5) -> None:
        self.rag = rag
        self.decision_model = decision_model or DecisionModel()
        self.max_iterations = max_iterations

    def run(self, user_query: str, conversation_context: str = "") -> AgentState:
        state = AgentState(
            user_query=user_query,
            conversation_context=conversation_context,
            next_query=user_query,
        )

        while state.iteration < self.max_iterations:
            action = self.decision_model.choose_action(state)
            state.iteration += 1

            if action["action"] == "SEARCH":
                try:
                    result = search_documents(self.rag, action["query"], top_k=3)
                    state.tool_history.append({
                        "iteration": state.iteration,
                        "action": "search_documents",
                        "query": action["query"],
                        "status": "success",
                    })
                except Exception as exc:  # pragma: no cover - runtime failure path
                    state.failure_state = f"tool_failure:{type(exc).__name__}"
                    state.status = "failed"
                    state.final_answer = f"The search tool failed: {exc}."
                    return state

                state.retrieval_results = result["hits"]
                evidence_report = evaluate_evidence(action["query"], result["hits"])
                state.evidence = summarize_evidence(result["hits"])
                state.evidence_sufficiency = evidence_report["sufficient"]
                state.missing_information = evidence_report["missing_information"]
                if not state.evidence_sufficiency:
                    state.next_query = self._rewrite_query(action["query"], result["hits"])
                continue

            if action["action"] == "SEARCH_AGAIN":
                next_query = action.get("query") or state.next_query or state.user_query
                try:
                    result = search_documents(self.rag, next_query, top_k=4)
                    state.tool_history.append({
                        "iteration": state.iteration,
                        "action": "search_documents",
                        "query": next_query,
                        "status": "success",
                    })
                except Exception as exc:  # pragma: no cover - runtime failure path
                    state.failure_state = f"tool_failure:{type(exc).__name__}"
                    state.status = "failed"
                    state.final_answer = f"The follow-up search failed: {exc}."
                    return state

                state.retrieval_results = result["hits"]
                evidence_report = evaluate_evidence(next_query, result["hits"])
                state.evidence = summarize_evidence(result["hits"])
                state.evidence_sufficiency = evidence_report["sufficient"]
                state.missing_information = evidence_report["missing_information"]
                if not evidence_report["sufficient"]:
                    state.next_query = self._rewrite_query(next_query, result["hits"])
                continue

            if action["action"] == "ASK_CLARIFICATION":
                state.status = "needs_clarification"
                state.final_answer = (
                    "The available evidence is insufficient to answer confidently. Please clarify the question "
                    "or specify the product area you want information about."
                )
                return state

            if action["action"] == "FINAL_ANSWER":
                answer = self._compose_answer(state)
                state.status = "complete"
                state.final_answer = answer
                return state

            state.failure_state = "unsupported_action"
            state.status = "failed"
            state.final_answer = f"Unsupported agent action: {action.get('action', 'unknown')}."
            return state

        if state.evidence_sufficiency:
            state.status = "complete"
            state.final_answer = self._compose_answer(state)
            return state

        state.status = "failed"
        state.failure_state = "max_iterations_reached"
        state.final_answer = "The agent reached the iteration limit without enough evidence to answer confidently."
        return state

    @staticmethod
    def _rewrite_query(previous_query: str, results: list[dict[str, Any]]) -> str:
        if not results:
            return f"{previous_query} key facts"
        top_hits = [str(item.get("text", "")) for item in results[:2]]
        text = " ".join(top_hits)[:220]
        key_terms = sorted({term.lower() for term in text.split() if len(term) > 4})[:4]
        if not key_terms:
            return f"{previous_query} details"
        return " ".join([previous_query, *key_terms])

    @staticmethod
    def _compose_answer(state: AgentState) -> str:
        if not state.evidence:
            return "I could not find enough evidence to answer this request with confidence."
        evidence_text = "\n\n".join(f"- {item}" for item in state.evidence[:3])
        return (
            f"Based on the retrieved evidence, here is the answer to '{state.user_query}':\n\n"
            f"{evidence_text}\n\n"
            "This answer is limited to the retrieved evidence and may require additional clarification if the question is ambiguous."
        )
