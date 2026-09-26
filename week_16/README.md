# W16 Agentified Assistant

## A. Context Engineering Technique
The selected technique is compact evidence summarization with retrieval result capping. In `app/agent.py`, the agent keeps the full retrieval result set for traceability but collapses the top evidence to a short list before answer synthesis, and the search tool caps results at 3-4 hits to avoid context saturation. The actual problem was noisy retrieval from a small knowledge base where repeated broad matches would accumulate too much irrelevant text and make the model overfit to weak evidence. By summarizing only the top supporting snippets, the agent keeps enough evidence to reason while staying within a compact context window.

## B. Agentic Pattern
This is a single-agent loop. The loop is deliberately dynamic: the agent starts with a search, evaluates the evidence, decides whether the question is answered, whether it needs a second pass, whether clarification is required, or whether it should stop. A fixed pipeline would not be sufficient here because the question “Can the app work offline and create a new support ticket?” may initially return weak evidence, requiring a second search with a refined query before the agent can answer safely.

## C. Evaluation Harness
The evaluation harness lives in `evaluation/harness.py`. It exercises the real runtime rather than isolated helper functions and runs five queries against the W16 agent. The harness records: query id, success, iterations, tool calls, token usage, and failure classification. It measures task completion rate, tool-call correctness, average trajectory length, and aggregate token cost. The results are written to `evaluation/results.md` and can be regenerated with `python evaluation/harness.py`.

## D. Skill vs Agent
Could this capability have been implemented as a Skill? No, because the required behavior is decision-making over retrieved evidence: the agent has to inspect results, detect insufficient or ambiguous evidence, and decide whether to search again or ask for clarification. A Skill could help with one sub-step, but not the iterative reasoning loop itself.

## E. Token and Cost Accounting
Token accounting is estimated from the text payloads because the runtime is designed to work without a provider-specific usage object. Each query records `input_tokens`, `output_tokens`, and `total_tokens` using a simple defensible estimate based on the query and final answer length. If an LLM provider exposes exact token usage, the code can be extended to replace the estimate with the provider-reported values.

## F. Failure Injection
The harness intentionally injects a tool failure (`tool_unavailable`) for one query. The agent detects the exception, records the failure in `failure_state`, and exits safely rather than treating invalid evidence as trustworthy. This satisfies the requirement to recover or terminate gracefully when a tool is unavailable or malformed output appears.

## G. Tool vs Agent Boundary
The external retrieval service is modeled as a bounded tool because it is a deterministic enhancement to the agent: it accepts a query, returns evidence, and does not itself decide what the final answer should be. The decision-making stays in the agent loop, while the search and evidence-evaluation functions remain narrow, testable, and validatable.

## Implementation Summary
The W16 assistant is built from the W15 reference architecture: the same FastAPI + Streamlit pattern and a lightweight retrieval layer are retained, but an explicit agent loop is added to make the system dynamic. The main files are:

- `app/agent.py` - explicit state and loop logic
- `app/tools.py` - tool validation and evidence evaluation
- `app/server.py` - API entry point
- `app/ui.py` - UI entry point
- `evaluation/harness.py` - evaluation runner
- `docs/architecture.md` - W16 architecture diagram

The maximum iteration cap is five, which is consistent with a high-value agent loop: enough to recover from weak evidence but still bounded to prevent infinite behavior. Stop conditions include evidence sufficiency, clarification request, unrecoverable tool failure, and the hard iteration limit.
