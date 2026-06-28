from __future__ import annotations

import pandas as pd
import streamlit as st

from config import DEFAULT_TOP_K, VECTOR_STORE_DIR
from rag_chain import ask_scout


st.set_page_config(
    page_title="FIFA Talent Scout RAG",
    page_icon="⚽",
    layout="wide",
)


st.title("FIFA Talent Scout RAG")
st.caption(
    "Hybrid football scouting assistant using LangChain, HuggingFace embeddings, "
    "FAISS semantic retrieval, structured query parsing, scout reranking, and optional OpenAI LLM generation."
)


with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Number of players to retrieve",
        min_value=3,
        max_value=10,
        value=DEFAULT_TOP_K,
        step=1,
    )

    st.divider()

    if VECTOR_STORE_DIR.exists():
        st.success("FAISS index found")
    else:
        st.error("FAISS index not found")
        st.code("python ingest.py", language="powershell")

    st.divider()

    st.markdown("### Example queries")
    st.markdown(
        """
        - French midfielder with high passing and potential
        - Brazilian winger under 23 with high pace and good potential
        - Young left-footed defender from Spain
        - Portuguese striker under 25 with strong shooting
        - Dutch centre back with high defending and physicality
        - Cheap young midfielder under 5 million with high potential
        """
    )


query = st.text_area(
    "Scouting request",
    value="French midfielder with high passing and potential",
    height=100,
    placeholder="Example: young left-footed defender from Spain",
)


run_button = st.button("Run scouting search", type="primary")


if run_button:
    if not query.strip():
        st.warning("Please enter a scouting request.")
        st.stop()

    with st.spinner("Parsing query, retrieving candidates, reranking players, and generating report..."):
        result = ask_scout(query.strip(), top_k=top_k)

    st.subheader("Scouting answer")
    st.markdown(result["answer"])

    st.subheader("Retrieved players")

    players = result.get("retrieved_players", [])

    if players:
        df = pd.DataFrame(players)

        preferred_columns = [
            "rank",
            "scout_score",
            "fit_score_10",
            "faiss_distance",
            "name",
            "age",
            "nationality",
            "club",
            "league",
            "positions",
            "preferred_foot",
            "overall",
            "potential",
            "passing",
            "pace",
            "dribbling",
            "defending",
            "shooting",
            "value_eur",
            "score_reasons",
        ]

        existing_columns = [col for col in preferred_columns if col in df.columns]
        st.dataframe(df[existing_columns], use_container_width=True)
    else:
        st.warning("No players were retrieved.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mode", result.get("mode", "unknown"))

    with col2:
        st.metric("Used LLM", str(result.get("used_llm", False)))

    with col3:
        st.metric("Players", len(players))

    st.caption(f"Retrieval mode: {result.get('retrieval_mode', 'unknown')}")

    if result.get("error"):
        st.warning(result["error"])

    with st.expander("Parsed scouting intent"):
        st.json(result.get("constraints", {}))

    with st.expander("Retrieved context sent to LLM"):
        st.text(result.get("context", ""))

    with st.expander("Full prompt preview"):
        st.text(result.get("prompt_preview", ""))

else:
    st.info("Enter a scouting request and click **Run scouting search**.")