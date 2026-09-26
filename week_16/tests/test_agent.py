import json
import os
import unittest

from app.agent import AgentRuntime
from app.rag import SimpleRAG
from app.tools import ToolValidationError, search_documents


class TestAgentRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")
        with open(path, "r", encoding="utf-8") as handle:
            documents = json.load(handle)["documents"]
        cls.rag = SimpleRAG(documents=documents)

    def test_single_iteration_completion(self):
        runtime = AgentRuntime(self.rag, max_iterations=5)
        state = runtime.run("What is the refund policy?")
        self.assertEqual(state.status, "complete")
        self.assertLessEqual(state.iteration, 3)
        self.assertIn("refund", state.final_answer.lower())

    def test_multi_iteration_execution(self):
        runtime = AgentRuntime(self.rag, max_iterations=5)
        state = runtime.run("Can the app work offline and create a new support ticket?")
        self.assertTrue(state.iteration >= 2)
        self.assertIn(state.status, {"complete", "needs_clarification"})

    def test_clarification_behavior(self):
        runtime = AgentRuntime(self.rag, max_iterations=5)
        state = runtime.run("What is the support setup?")
        self.assertEqual(state.status, "needs_clarification")
        self.assertIn("clarify", state.final_answer.lower())

    def test_max_iterations_stopping(self):
        runtime = AgentRuntime(self.rag, max_iterations=2)
        state = runtime.run("Tell me everything about undocumented enterprise migration details.")
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.failure_state, "max_iterations_reached")

    def test_tool_failure_handling(self):
        runtime = AgentRuntime(self.rag, max_iterations=5)
        runtime.rag.search = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("tool unavailable"))
        state = runtime.run("How long do backups remain available?")
        self.assertEqual(state.status, "failed")
        self.assertIn("tool_failure", state.failure_state)

    def test_invalid_tool_arguments(self):
        with self.assertRaises(ToolValidationError):
            search_documents(self.rag, "", top_k=2)
        with self.assertRaises(ToolValidationError):
            search_documents(self.rag, "backup", top_k=0)

    def test_invalid_evidence_handling(self):
        runtime = AgentRuntime(self.rag, max_iterations=5)
        state = runtime.run("Explain undocumented holiday scheduling for the moon base.")
        self.assertIn(state.status, {"failed", "needs_clarification"})
        self.assertTrue(state.missing_information or state.failure_state)

    def test_final_answer_after_evidence(self):
        runtime = AgentRuntime(self.rag, max_iterations=5)
        state = runtime.run("Compare backup retention and support response targets for premium customers.")
        self.assertEqual(state.status, "complete")
        self.assertTrue(state.evidence_sufficiency)
        self.assertTrue(state.final_answer)


if __name__ == "__main__":
    unittest.main()
