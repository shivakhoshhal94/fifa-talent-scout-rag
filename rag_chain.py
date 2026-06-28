from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import EMBEDDING_MODEL, OPENAI_MODEL, VECTOR_STORE_DIR
from query_parser import parse_scout_query, scout_query_to_dict, ScoutQuery
from ranker import (
    calculate_scout_score,
    explain_score,
    fit_score_10,
    get_stat,
    player_matches_hard_constraints,
)


load_dotenv()


SCOUTING_PROMPT = """
You are a football scouting analyst.

Use only the retrieved FIFA player profiles below.
Do not invent players, clubs, ratings, values, nationalities, positions, or statistics.
If a player has weak evidence for the request, say so clearly.

User scouting request:
{question}

Parsed scouting intent:
{parsed_query}

Retrieved player profiles:
{context}

Write a concise scouting report.

For each recommended player, include:
- Name
- Age
- Nationality
- Club
- Position
- Preferred foot
- Overall and potential
- Key relevant stats such as passing, pace, dribbling, defending, or shooting
- Why the player matches the request
- Any weakness or caveat
- Scout fit score from 1 to 10

Finish with one final recommendation.
"""


def load_vectorstore() -> FAISS:
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found: {VECTOR_STORE_DIR}\n"
            "Run: python ingest.py"
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return FAISS.load_local(
        folder_path=str(VECTOR_STORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_candidates(question: str, top_k: int = 5, fetch_k: int = 200):
    parsed = parse_scout_query(question)
    vectorstore = load_vectorstore()

    raw_docs = vectorstore.similarity_search_with_score(
        query=question,
        k=fetch_k,
    )

    hard_filtered_docs = [
        (doc, score)
        for doc, score in raw_docs
        if player_matches_hard_constraints(doc, parsed)
    ]

    candidate_docs = hard_filtered_docs if hard_filtered_docs else raw_docs

    ranked_docs = sorted(
        candidate_docs,
        key=lambda item: calculate_scout_score(parsed, item[0], item[1]),
        reverse=True,
    )

    if hard_filtered_docs:
        retrieval_mode = "faiss_semantic_search + hard_metadata_filter + scout_reranker"
    else:
        retrieval_mode = "faiss_semantic_search + scout_reranker_fallback"

    return ranked_docs[:top_k], parsed, retrieval_mode


def format_retrieved_context(docs_with_scores, parsed: ScoutQuery) -> str:
    blocks: List[str] = []

    for rank, (doc, faiss_distance) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata
        score = calculate_scout_score(parsed, doc, faiss_distance)
        fit = fit_score_10(parsed, doc, faiss_distance)
        reasons = explain_score(parsed, doc)

        block = f"""
Player {rank}
Scout score: {score}
Fit score out of 10: {fit}
FAISS distance: {float(faiss_distance):.4f}
Score evidence: {", ".join(reasons)}

Name: {meta.get("name", "")}
Age: {meta.get("age", "")}
Nationality: {meta.get("nationality", "")}
Club: {meta.get("club", "")}
League: {meta.get("league", "")}
Positions: {meta.get("positions", "")}
Preferred foot: {meta.get("preferred_foot", "")}
Overall: {get_stat(doc, "overall")}
Potential: {get_stat(doc, "potential")}
Pace: {get_stat(doc, "pace")}
Shooting: {get_stat(doc, "shooting")}
Passing: {get_stat(doc, "passing")}
Dribbling: {get_stat(doc, "dribbling")}
Defending: {get_stat(doc, "defending")}
Physical: {get_stat(doc, "physic")}
Value EUR: {get_stat(doc, "value_eur")}

Full profile:
{doc.page_content}
"""
        blocks.append(block.strip())

    return "\n\n---\n\n".join(blocks)


def retrieved_players_table(docs_with_scores, parsed: ScoutQuery) -> List[Dict[str, Any]]:
    rows = []

    for rank, (doc, faiss_distance) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata

        rows.append(
            {
                "rank": rank,
                "scout_score": calculate_scout_score(parsed, doc, faiss_distance),
                "fit_score_10": fit_score_10(parsed, doc, faiss_distance),
                "faiss_distance": round(float(faiss_distance), 4),
                "name": meta.get("name", ""),
                "age": meta.get("age", ""),
                "nationality": meta.get("nationality", ""),
                "club": meta.get("club", ""),
                "league": meta.get("league", ""),
                "positions": meta.get("positions", ""),
                "preferred_foot": meta.get("preferred_foot", ""),
                "overall": get_stat(doc, "overall"),
                "potential": get_stat(doc, "potential"),
                "pace": get_stat(doc, "pace"),
                "passing": get_stat(doc, "passing"),
                "dribbling": get_stat(doc, "dribbling"),
                "defending": get_stat(doc, "defending"),
                "shooting": get_stat(doc, "shooting"),
                "value_eur": get_stat(doc, "value_eur"),
                "score_reasons": ", ".join(explain_score(parsed, doc)),
            }
        )

    return rows


def build_prompt_preview(question: str, parsed: ScoutQuery, context: str) -> str:
    return SCOUTING_PROMPT.format(
        question=question,
        parsed_query=scout_query_to_dict(parsed),
        context=context,
    )


def local_summary(
    question: str,
    docs_with_scores,
    parsed: ScoutQuery,
    retrieval_mode: str,
) -> str:
    parsed_dict = scout_query_to_dict(parsed)

    lines = [
        "### Retrieval-only scouting result",
        "",
        "OpenAI LLM generation is not available, so this answer is generated directly from retrieved player profiles.",
        "",
        f"**Query:** {question}",
        f"**Retrieval mode:** {retrieval_mode}",
        "",
        "### Parsed scouting intent",
        "",
        f"- Nationality: {parsed_dict.get('nationality')}",
        f"- Max age: {parsed_dict.get('max_age')}",
        f"- Min age: {parsed_dict.get('min_age')}",
        f"- Positions: {parsed_dict.get('positions')}",
        f"- Preferred foot: {parsed_dict.get('preferred_foot')}",
        f"- Min potential: {parsed_dict.get('min_potential')}",
        f"- Min passing: {parsed_dict.get('min_passing')}",
        f"- Min pace: {parsed_dict.get('min_pace')}",
        f"- Min dribbling: {parsed_dict.get('min_dribbling')}",
        f"- Min defending: {parsed_dict.get('min_defending')}",
        f"- Min shooting: {parsed_dict.get('min_shooting')}",
        f"- Ranking focus: {parsed_dict.get('ranking_focus')}",
        "",
        "### Retrieved players",
        "",
    ]

    for rank, (doc, faiss_distance) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata
        reasons = explain_score(parsed, doc)

        lines.append(f"#### {rank}. {meta.get('name', 'Unknown player')}")
        lines.append(f"- Scout score: {calculate_scout_score(parsed, doc, faiss_distance)}")
        lines.append(f"- Fit score: {fit_score_10(parsed, doc, faiss_distance)}/10")
        lines.append(f"- FAISS distance: {float(faiss_distance):.4f}")
        lines.append(f"- Age: {meta.get('age', 'N/A')}")
        lines.append(f"- Nationality: {meta.get('nationality', 'N/A')}")
        lines.append(f"- Club: {meta.get('club', 'N/A')}")
        lines.append(f"- League: {meta.get('league', 'N/A')}")
        lines.append(f"- Positions: {meta.get('positions', 'N/A')}")
        lines.append(f"- Preferred foot: {meta.get('preferred_foot', 'N/A')}")
        lines.append(f"- Overall / Potential: {get_stat(doc, 'overall')} / {get_stat(doc, 'potential')}")
        lines.append(f"- Passing: {get_stat(doc, 'passing')}")
        lines.append(f"- Pace: {get_stat(doc, 'pace')}")
        lines.append(f"- Dribbling: {get_stat(doc, 'dribbling')}")
        lines.append(f"- Defending: {get_stat(doc, 'defending')}")
        lines.append(f"- Shooting: {get_stat(doc, 'shooting')}")
        lines.append(f"- Value EUR: {get_stat(doc, 'value_eur')}")
        lines.append(f"- Score reasons: {', '.join(reasons)}")
        lines.append("")

    return "\n".join(lines)


def ask_scout(question: str, top_k: int = 5) -> Dict[str, Any]:
    docs_with_scores, parsed, retrieval_mode = retrieve_candidates(
        question=question,
        top_k=top_k,
        fetch_k=max(200, top_k * 40),
    )

    context = format_retrieved_context(docs_with_scores, parsed)
    retrieved_players = retrieved_players_table(docs_with_scores, parsed)
    parsed_dict = scout_query_to_dict(parsed)
    prompt_preview = build_prompt_preview(question, parsed, context)

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "question": question,
            "answer": local_summary(question, docs_with_scores, parsed, retrieval_mode),
            "retrieved_players": retrieved_players,
            "context": context,
            "prompt_preview": prompt_preview,
            "used_llm": False,
            "mode": "retrieval_only",
            "retrieval_mode": retrieval_mode,
            "constraints": parsed_dict,
            "error": "OPENAI_API_KEY is not configured.",
        }

    try:
        prompt = ChatPromptTemplate.from_template(SCOUTING_PROMPT)
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke(
            {
                "question": question,
                "parsed_query": parsed_dict,
                "context": context,
            }
        )

        return {
            "question": question,
            "answer": answer,
            "retrieved_players": retrieved_players,
            "context": context,
            "prompt_preview": prompt_preview,
            "used_llm": True,
            "mode": "llm_rag",
            "retrieval_mode": retrieval_mode,
            "constraints": parsed_dict,
            "error": None,
        }

    except Exception as exc:
        return {
            "question": question,
            "answer": local_summary(question, docs_with_scores, parsed, retrieval_mode),
            "retrieved_players": retrieved_players,
            "context": context,
            "prompt_preview": prompt_preview,
            "used_llm": False,
            "mode": "retrieval_only",
            "retrieval_mode": retrieval_mode,
            "constraints": parsed_dict,
            "error": str(exc),
        }


if __name__ == "__main__":
    test_query = "French midfielder with high passing and potential"

    result = ask_scout(test_query, top_k=5)

    print("\nQUESTION:")
    print(result["question"])

    print("\nMODE:")
    print(result["mode"])

    print("\nRETRIEVAL MODE:")
    print(result["retrieval_mode"])

    print("\nPARSED QUERY:")
    print(result["constraints"])

    print("\nRETRIEVED PLAYERS:")
    for player in result["retrieved_players"]:
        print(player)

    print("\nANSWER:")
    print(result["answer"])

    if result["error"]:
        print("\nERROR:")
        print(result["error"])