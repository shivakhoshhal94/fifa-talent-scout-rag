from __future__ import annotations

import re
from typing import Any

from query_parser import ScoutQuery


def to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def get_stat(doc, stat_name: str) -> float | None:
    value = to_float(doc.metadata.get(stat_name))
    if value is not None:
        return value

    text = doc.page_content.lower()

    patterns = {
        "overall": r"overall rating:\s*([\d.]+)",
        "potential": r"potential rating:\s*([\d.]+)",
        "pace": r"pace:\s*([\d.]+)",
        "shooting": r"shooting:\s*([\d.]+)",
        "passing": r"passing:\s*([\d.]+)",
        "dribbling": r"dribbling:\s*([\d.]+)",
        "defending": r"defending:\s*([\d.]+)",
        "physic": r"physical:\s*([\d.]+)",
        "value_eur": r"market value eur:\s*([\d.]+)",
    }

    pattern = patterns.get(stat_name)
    if not pattern:
        return None

    match = re.search(pattern, text)
    if not match:
        return None

    return to_float(match.group(1))


def player_matches_hard_constraints(doc, parsed: ScoutQuery) -> bool:
    meta = doc.metadata

    if parsed.nationality:
        player_nationality = str(meta.get("nationality", "")).lower()
        if parsed.nationality.lower() not in player_nationality:
            return False

    if parsed.max_age is not None:
        age = to_float(meta.get("age"))
        if age is None or age > parsed.max_age:
            return False

    if parsed.min_age is not None:
        age = to_float(meta.get("age"))
        if age is None or age < parsed.min_age:
            return False

    if parsed.preferred_foot:
        foot = str(meta.get("preferred_foot", "")).lower()
        if parsed.preferred_foot.lower() != foot:
            return False

    if parsed.positions:
        player_positions = str(meta.get("positions", "")).upper().replace(",", " ").split()
        if not any(pos in player_positions for pos in parsed.positions):
            return False

    if parsed.max_value_eur is not None:
        value = get_stat(doc, "value_eur")
        if value is None or value > parsed.max_value_eur:
            return False

    return True


def soft_threshold_bonus(value: float | None, threshold: int | None) -> float:
    """
    Gives a small bonus if the player meets the soft threshold.
    Does not remove the player if below threshold.
    """
    if value is None or threshold is None:
        return 0.0

    if value >= threshold:
        return 0.08

    gap = max(0.0, threshold - value)
    return -min(0.08, gap / 100)


def calculate_scout_score(parsed: ScoutQuery, doc, faiss_distance: float) -> float:
    """
    Combines:
    - semantic relevance from FAISS
    - base football quality
    - query-specific attribute boosts
    - soft threshold bonuses
    """

    overall = get_stat(doc, "overall") or 0
    potential = get_stat(doc, "potential") or 0
    passing = get_stat(doc, "passing") or 0
    pace = get_stat(doc, "pace") or 0
    dribbling = get_stat(doc, "dribbling") or 0
    defending = get_stat(doc, "defending") or 0
    shooting = get_stat(doc, "shooting") or 0
    physic = get_stat(doc, "physic") or 0
    value = get_stat(doc, "value_eur")

    semantic_score = max(0.0, 1.0 - float(faiss_distance))

    score = 0.10 * semantic_score
    score += 0.12 * (overall / 100)
    score += 0.18 * (potential / 100)

    if "potential" in parsed.ranking_focus:
        score += 0.35 * (potential / 100)
        score += soft_threshold_bonus(potential, parsed.min_potential)

    if "passing" in parsed.ranking_focus:
        score += 0.35 * (passing / 100)
        score += soft_threshold_bonus(passing, parsed.min_passing)

    if "pace" in parsed.ranking_focus:
        score += 0.30 * (pace / 100)
        score += soft_threshold_bonus(pace, parsed.min_pace)

    if "dribbling" in parsed.ranking_focus:
        score += 0.30 * (dribbling / 100)
        score += soft_threshold_bonus(dribbling, parsed.min_dribbling)

    if "defending" in parsed.ranking_focus:
        score += 0.30 * (defending / 100)
        score += 0.12 * (physic / 100)
        score += soft_threshold_bonus(defending, parsed.min_defending)

    if "shooting" in parsed.ranking_focus:
        score += 0.30 * (shooting / 100)
        score += soft_threshold_bonus(shooting, parsed.min_shooting)

    if "overall" in parsed.ranking_focus:
        score += 0.20 * (overall / 100)
        score += soft_threshold_bonus(overall, parsed.min_overall)

    if "value" in parsed.ranking_focus and value:
        # Cheaper players get a small bonus, capped for stability.
        value_million = value / 1_000_000
        value_bonus = max(0.0, 0.10 - min(value_million, 50) / 500)
        score += value_bonus

    return round(score, 4)


def fit_score_10(parsed: ScoutQuery, doc, faiss_distance: float) -> int:
    score = calculate_scout_score(parsed, doc, faiss_distance)
    return max(1, min(10, round(score * 8)))


def explain_score(parsed: ScoutQuery, doc) -> list[str]:
    reasons = []

    if parsed.nationality:
        reasons.append(f"nationality matches {parsed.nationality}")

    if parsed.positions:
        reasons.append("position matches requested role")

    if parsed.max_age is not None:
        reasons.append(f"age is within max age {parsed.max_age}")

    for focus in parsed.ranking_focus:
        stat_value = get_stat(doc, focus)
        if stat_value is not None:
            reasons.append(f"{focus}={stat_value}")

    return reasons