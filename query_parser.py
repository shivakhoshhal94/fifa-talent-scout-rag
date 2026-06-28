from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ScoutQuery:
    raw_query: str
    nationality: Optional[str] = None
    max_age: Optional[int] = None
    min_age: Optional[int] = None
    positions: list[str] = field(default_factory=list)
    preferred_foot: Optional[str] = None

    # Soft ranking preferences
    min_potential: Optional[int] = None
    min_overall: Optional[int] = None
    min_passing: Optional[int] = None
    min_pace: Optional[int] = None
    min_dribbling: Optional[int] = None
    min_defending: Optional[int] = None
    min_shooting: Optional[int] = None
    max_value_eur: Optional[float] = None

    ranking_focus: list[str] = field(default_factory=list)
    intent: str = "recommend"


NATIONALITY_MAP = {
    "brazilian": "Brazil",
    "brazil": "Brazil",
    "french": "France",
    "france": "France",
    "spanish": "Spain",
    "spain": "Spain",
    "german": "Germany",
    "germany": "Germany",
    "italian": "Italy",
    "italy": "Italy",
    "portuguese": "Portugal",
    "portugal": "Portugal",
    "english": "England",
    "england": "England",
    "argentinian": "Argentina",
    "argentina": "Argentina",
    "dutch": "Netherlands",
    "netherlands": "Netherlands",
    "danish": "Denmark",
    "denmark": "Denmark",
}


POSITION_GROUPS = {
    "goalkeeper": ["GK"],
    "keeper": ["GK"],
    "centre back": ["CB", "LCB", "RCB"],
    "center back": ["CB", "LCB", "RCB"],
    "cb": ["CB", "LCB", "RCB"],
    "defender": ["CB", "LB", "RB", "LWB", "RWB", "LCB", "RCB"],
    "full back": ["LB", "RB", "LWB", "RWB"],
    "left back": ["LB", "LWB"],
    "right back": ["RB", "RWB"],
    "midfielder": ["CM", "CDM", "CAM", "LM", "RM"],
    "central midfielder": ["CM", "CDM", "CAM"],
    "defensive midfielder": ["CDM"],
    "attacking midfielder": ["CAM"],
    "winger": ["LW", "RW", "LM", "RM"],
    "left winger": ["LW", "LM"],
    "right winger": ["RW", "RM"],
    "forward": ["ST", "CF", "LW", "RW"],
    "striker": ["ST", "CF", "LS", "RS"],
}


QUALITY_LEVELS = {
    "decent": 65,
    "good": 70,
    "high": 75,
    "strong": 75,
    "very good": 78,
    "excellent": 82,
    "elite": 85,
    "top": 85,
}


ATTRIBUTE_KEYWORDS = {
    "passing": ["passing", "passer", "playmaker", "playmaking", "vision", "distribution"],
    "pace": ["pace", "fast", "speed", "quick", "pacey", "acceleration"],
    "dribbling": ["dribbling", "dribbler", "technical", "ball carrying", "ball carrier"],
    "defending": ["defending", "defensive", "tackling", "interceptions", "tackler"],
    "shooting": ["shooting", "finishing", "goalscorer", "scoring", "shot"],
    "potential": ["potential", "talent", "upside", "growth", "prospect", "wonderkid"],
    "overall": ["overall", "quality", "rating"],
    "value": ["cheap", "budget", "low value", "affordable", "expensive"],
}


def normalize_query(query: str) -> str:
    q = query.lower().strip()
    q = q.replace("-", " ")
    q = re.sub(r"\s+", " ", q)
    return q


def _contains_phrase(q: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", q) is not None


def _add_focus(parsed: ScoutQuery, focus: str) -> None:
    if focus not in parsed.ranking_focus:
        parsed.ranking_focus.append(focus)


def _extract_numeric_after(q: str, keywords: list[str]) -> Optional[int]:
    joined = "|".join(re.escape(k) for k in keywords)
    pattern = rf"\b(?:{joined})\s*(?:above|over|at least|minimum|min)?\s*(\d{{2,3}})\+?\b"
    match = re.search(pattern, q)
    if match:
        return int(match.group(1))
    return None


def _extract_quality_level(q: str, attribute_words: list[str]) -> Optional[int]:
    for level_word, threshold in QUALITY_LEVELS.items():
        for attr in attribute_words:
            if _contains_phrase(q, f"{level_word} {attr}"):
                return threshold
    return None


def parse_scout_query(query: str) -> ScoutQuery:
    q = normalize_query(query)
    parsed = ScoutQuery(raw_query=query)

    # Intent detection
    if any(word in q for word in ["compare", "comparison", "versus", "vs"]):
        parsed.intent = "compare"
    elif any(word in q for word in ["similar to", "like"]):
        parsed.intent = "similarity_search"
    else:
        parsed.intent = "recommend"

    # Nationality
    for word, nationality in NATIONALITY_MAP.items():
        if _contains_phrase(q, word):
            parsed.nationality = nationality
            break

    # Position group
    for phrase, codes in POSITION_GROUPS.items():
        if _contains_phrase(q, phrase):
            parsed.positions.extend(codes)

    parsed.positions = sorted(set(parsed.positions))

    # Age
    if _contains_phrase(q, "young"):
        parsed.max_age = 23

    under_age = re.search(r"\b(?:under|below|younger than|less than)\s+(\d{1,2})\b", q)
    if under_age:
        parsed.max_age = int(under_age.group(1))

    over_age = re.search(r"\b(?:over|older than|above)\s+(\d{1,2})\b", q)
    if over_age:
        parsed.min_age = int(over_age.group(1))

    # Foot
    if any(term in q for term in ["left foot", "left footed", "left-footed"]):
        parsed.preferred_foot = "Left"

    if any(term in q for term in ["right foot", "right footed", "right-footed"]):
        parsed.preferred_foot = "Right"

    # Attribute parsing: numeric and adjective-based
    potential_num = _extract_numeric_after(q, ATTRIBUTE_KEYWORDS["potential"])
    if potential_num:
        parsed.min_potential = potential_num
        _add_focus(parsed, "potential")
    elif any(_contains_phrase(q, k) for k in ATTRIBUTE_KEYWORDS["potential"]):
        parsed.min_potential = _extract_quality_level(q, ATTRIBUTE_KEYWORDS["potential"]) or 80
        _add_focus(parsed, "potential")

    passing_num = _extract_numeric_after(q, ATTRIBUTE_KEYWORDS["passing"])
    if passing_num:
        parsed.min_passing = passing_num
        _add_focus(parsed, "passing")
    elif any(_contains_phrase(q, k) for k in ATTRIBUTE_KEYWORDS["passing"]):
        parsed.min_passing = _extract_quality_level(q, ATTRIBUTE_KEYWORDS["passing"]) or 75
        _add_focus(parsed, "passing")

    pace_num = _extract_numeric_after(q, ATTRIBUTE_KEYWORDS["pace"])
    if pace_num:
        parsed.min_pace = pace_num
        _add_focus(parsed, "pace")
    elif any(_contains_phrase(q, k) for k in ATTRIBUTE_KEYWORDS["pace"]):
        parsed.min_pace = _extract_quality_level(q, ATTRIBUTE_KEYWORDS["pace"]) or 75
        _add_focus(parsed, "pace")

    dribbling_num = _extract_numeric_after(q, ATTRIBUTE_KEYWORDS["dribbling"])
    if dribbling_num:
        parsed.min_dribbling = dribbling_num
        _add_focus(parsed, "dribbling")
    elif any(_contains_phrase(q, k) for k in ATTRIBUTE_KEYWORDS["dribbling"]):
        parsed.min_dribbling = _extract_quality_level(q, ATTRIBUTE_KEYWORDS["dribbling"]) or 75
        _add_focus(parsed, "dribbling")

    defending_num = _extract_numeric_after(q, ATTRIBUTE_KEYWORDS["defending"])
    if defending_num:
        parsed.min_defending = defending_num
        _add_focus(parsed, "defending")
    elif any(_contains_phrase(q, k) for k in ATTRIBUTE_KEYWORDS["defending"]):
        parsed.min_defending = _extract_quality_level(q, ATTRIBUTE_KEYWORDS["defending"]) or 75
        _add_focus(parsed, "defending")

    shooting_num = _extract_numeric_after(q, ATTRIBUTE_KEYWORDS["shooting"])
    if shooting_num:
        parsed.min_shooting = shooting_num
        _add_focus(parsed, "shooting")
    elif any(_contains_phrase(q, k) for k in ATTRIBUTE_KEYWORDS["shooting"]):
        parsed.min_shooting = _extract_quality_level(q, ATTRIBUTE_KEYWORDS["shooting"]) or 75
        _add_focus(parsed, "shooting")

    overall_num = _extract_numeric_after(q, ATTRIBUTE_KEYWORDS["overall"])
    if overall_num:
        parsed.min_overall = overall_num
        _add_focus(parsed, "overall")

    # Value / budget
    budget_match = re.search(r"\b(?:under|below|max|less than)\s+(\d+(?:\.\d+)?)\s*(m|million|k|thousand)?\b", q)
    if budget_match and any(word in q for word in ["value", "price", "budget", "cheap", "cost"]):
        amount = float(budget_match.group(1))
        unit = budget_match.group(2)
        if unit in ["m", "million"]:
            parsed.max_value_eur = amount * 1_000_000
        elif unit in ["k", "thousand"]:
            parsed.max_value_eur = amount * 1_000
        else:
            parsed.max_value_eur = amount

    if "cheap" in q or "budget" in q or "affordable" in q:
        _add_focus(parsed, "value")

    return parsed


def scout_query_to_dict(parsed: ScoutQuery) -> dict:
    return asdict(parsed)