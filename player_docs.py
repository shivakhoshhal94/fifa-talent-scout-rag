from __future__ import annotations

from typing import List

import pandas as pd
from langchain_core.documents import Document


REQUIRED_COLUMNS = [
    "sofifa_id",
    "short_name",
    "long_name",
    "age",
    "nationality_name",
    "club_name",
    "league_name",
    "player_positions",
    "preferred_foot",
    "overall",
    "potential",
    "value_eur",
    "wage_eur",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]


COLUMN_ALIASES = {
    "sofifa_id": ["sofifa_id", "player_id", "id"],
    "short_name": ["short_name"],
    "long_name": ["long_name", "name"],
    "age": ["age"],
    "nationality_name": ["nationality_name", "nationality"],
    "club_name": ["club_name", "club"],
    "league_name": ["league_name", "league"],
    "player_positions": ["player_positions", "player_position", "club_position"],
    "preferred_foot": ["preferred_foot", "preferred foot"],
    "overall": ["overall", "ovr"],
    "potential": ["potential", "pot"],
    "value_eur": ["value_eur", "value"],
    "wage_eur": ["wage_eur", "wage"],
    "pace": ["pace"],
    "shooting": ["shooting"],
    "passing": ["passing"],
    "dribbling": ["dribbling"],
    "defending": ["defending"],
    "physic": ["physic"],
}


NUMERIC_COLUMNS = [
    "sofifa_id",
    "age",
    "overall",
    "potential",
    "value_eur",
    "wage_eur",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]


def detect_columns(csv_columns: list[str]) -> dict[str, str | None]:
    lower_map = {col.lower().strip(): col for col in csv_columns}
    mapping: dict[str, str | None] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        found = None

        for alias in aliases:
            key = alias.lower().strip()
            if key in lower_map:
                found = lower_map[key]
                break

        mapping[canonical] = found

    return mapping


def load_player_dataframe(csv_path, max_rows: int | None = None) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    mapping = detect_columns(header)

    usecols = [actual for actual in mapping.values() if actual is not None]

    if not usecols:
        raise ValueError("No supported columns were found in the CSV file.")

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        nrows=max_rows,
        low_memory=True,
        encoding="utf-8",
    )

    rename_map = {
        actual: canonical
        for canonical, actual in mapping.items()
        if actual is not None
    }

    df = df.rename(columns=rename_map)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [col for col in REQUIRED_COLUMNS if col not in NUMERIC_COLUMNS]
    df[text_cols] = df[text_cols].fillna("")

    return df[REQUIRED_COLUMNS].copy()


def player_row_to_text(row: pd.Series) -> str:
    return (
        f"Player name: {row.get('long_name') or row.get('short_name')}. "
        f"Age: {row.get('age')}. "
        f"Nationality: {row.get('nationality_name')}. "
        f"Club: {row.get('club_name')}. "
        f"League: {row.get('league_name')}. "
        f"Positions: {row.get('player_positions')}. "
        f"Preferred foot: {row.get('preferred_foot')}. "
        f"Overall rating: {row.get('overall')}. "
        f"Potential rating: {row.get('potential')}. "
        f"Market value EUR: {row.get('value_eur')}. "
        f"Wage EUR: {row.get('wage_eur')}. "
        f"Pace: {row.get('pace')}. "
        f"Shooting: {row.get('shooting')}. "
        f"Passing: {row.get('passing')}. "
        f"Dribbling: {row.get('dribbling')}. "
        f"Defending: {row.get('defending')}. "
        f"Physical: {row.get('physic')}."
    )


def dataframe_to_documents(df: pd.DataFrame) -> List[Document]:
    documents: List[Document] = []

    for _, row in df.iterrows():
        metadata = {
            "sofifa_id": str(row.get("sofifa_id", "")),
            "name": str(row.get("long_name") or row.get("short_name") or ""),
            "age": str(row.get("age", "")),
            "nationality": str(row.get("nationality_name", "")),
            "club": str(row.get("club_name", "")),
            "league": str(row.get("league_name", "")),
            "positions": str(row.get("player_positions", "")),
            "preferred_foot": str(row.get("preferred_foot", "")),
            "overall": str(row.get("overall", "")),
            "potential": str(row.get("potential", "")),
            "value_eur": str(row.get("value_eur", "")),
            "wage_eur": str(row.get("wage_eur", "")),
            "pace": str(row.get("pace", "")),
            "shooting": str(row.get("shooting", "")),
            "passing": str(row.get("passing", "")),
            "dribbling": str(row.get("dribbling", "")),
            "defending": str(row.get("defending", "")),
            "physic": str(row.get("physic", "")),
            }

        documents.append(
            Document(
                page_content=player_row_to_text(row),
                metadata=metadata,
            )
        )

    return documents