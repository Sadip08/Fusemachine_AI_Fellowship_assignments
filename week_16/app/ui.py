import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="W16 Agentified Assistant", page_icon="🤖")
st.title("W16 Agentified Assistant")

with st.sidebar:
    st.subheader("Connection")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)
    if st.button("Check health"):
        try:
            response = requests.get(f"{backend_url}/health", timeout=10)
            st.json(response.json())
        except Exception as exc:  # pragma: no cover - UI-only error path
            st.error(f"Health check failed: {exc}")

st.write("Ask a question and the agent will decide whether it needs another search or clarification before answering.")

question = st.text_area("Prompt", height=120, placeholder="Ask the assistant about the available knowledge base...")
if st.button("Send") and question.strip():
    with st.spinner("Agent reasoning..."):
        try:
            response = requests.post(
                f"{backend_url}/api/chat",
                json={"message": question},
                timeout=30,
            )
            payload = response.json()
            if response.status_code != 200:
                st.error(payload.get("detail", "Unknown error"))
            else:
                st.success(payload.get("status", "completed"))
                st.markdown(payload.get("answer", ""))
                with st.expander("Agent trace"):
                    st.json(payload.get("tool_history", []))
                if payload.get("retrieval_results"):
                    with st.expander("Evidence"):
                        for item in payload["retrieval_results"]:
                            st.write(item.get("text", ""))
        except Exception as exc:  # pragma: no cover - UI-only error path
            st.error(f"Request failed: {exc}")

st.divider()
st.subheader("Ingest knowledge")
source_id = st.text_input("Document ID", value="doc-1")
source_text = st.text_area("Document text", height=150, placeholder="Paste relevant context here...")
if st.button("Ingest document") and source_text.strip():
    try:
        response = requests.post(
            f"{backend_url}/api/ingest",
            json={
                "documents": [{
                    "id": source_id,
                    "text": source_text,
                    "metadata": {"source": source_id},
                }]
            },
            timeout=30,
        )
        st.json(response.json())
    except Exception as exc:  # pragma: no cover - UI-only error path
        st.error(f"Ingestion failed: {exc}")
