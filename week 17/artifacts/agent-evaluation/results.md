# W16 Evaluation Report

| Query | Success | Iterations | Tool Calls | Tokens | Failure |
|-------|---------|------------|------------|--------|---------|
| Q1 | Yes | 3 | 2 | 148 | None |
| Q2 | Yes | 2 | 1 | 170 | None |
| Q3 | Yes | 3 | 2 | 148 | None |
| Q4 | Yes | 2 | 1 | 160 | None |
| Q5 | No | 1 | 0 | 18 | Soft failure |

- Task completion rate: 80.0%
- Tool-call correctness: 80.0%
- Average trajectory length: 2.20
- Average tokens/query: 128.8
- Maximum iterations: 3
- Hard failures: 0
- Soft failures: 1
- Cascading soft failures: 0

The agent intentionally injects one tool failure during the evaluation to confirm it detects invalid tool output and terminates safely.