from __future__ import annotations

import argparse

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import DATA_PATH, EMBEDDING_MODEL, MAX_INDEX_ROWS, VECTOR_STORE_DIR
from player_docs import dataframe_to_documents, load_player_dataframe


def build_index(max_rows: int | None = MAX_INDEX_ROWS) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place your FIFA CSV file at data/male_players.csv"
        )

    print(f"Loading data from: {DATA_PATH}")
    df = load_player_dataframe(DATA_PATH, max_rows=max_rows)
    print(f"Loaded rows: {len(df)}")

    documents = dataframe_to_documents(df)
    print(f"Created LangChain documents: {len(documents)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_STORE_DIR))

    print(f"FAISS index saved to: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-rows",
        type=int,
        default=MAX_INDEX_ROWS,
        help="Number of rows to index. Use 0 to index all rows.",
    )

    args = parser.parse_args()

    max_rows = None if args.max_rows == 0 else args.max_rows
    build_index(max_rows=max_rows)