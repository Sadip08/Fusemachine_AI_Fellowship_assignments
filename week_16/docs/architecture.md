# W16 Architecture Diagram

```mermaid
flowchart TD
    U[User] --> UI[Streamlit UI]
    UI --> API[FastAPI API]
    API --> AGENT[Agent Runtime]
    AGENT --> DECISION{Decision}
    DECISION -->|SEARCH| SEARCH[search_documents tool]
    DECISION -->|SEARCH_AGAIN| SEARCH2[search_documents tool]
    DECISION -->|ASK_CLARIFICATION| CLARIFY[clarification response]
    DECISION -->|FINAL_ANSWER| ANSWER[final answer]

    SEARCH --> RAG[Simple RAG + document chunks]
    SEARCH2 --> RAG
    RAG --> EVAL[Evidence evaluation]
    EVAL --> CONTEXT[Context engineering: capped results + evidence summary]
    CONTEXT --> DECISION

    DECISION --> STOP[Stop if evidence is sufficient, clarification is needed, or max_iterations is reached]
    STOP --> ANSWER
    STOP --> CLARIFY

    MAX[Max iterations = 5] --> STOP
```

The W16 runtime keeps the W15 retrieval and API pattern but adds a genuine loop in which the previous search results influence the next decision. Evidence is compacted before synthesis, failures are handled explicitly, and the system stops at a hard limit when the question cannot be answered safely.
