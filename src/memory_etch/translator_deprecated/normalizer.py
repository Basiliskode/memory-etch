     1|"""
     2|Date/number/name normalizer for Universal Translator.
     3|Converts values to canonical format for FTS5 matching.
     4|"""
     5|
     6|import re
     7|from datetime import datetime
     8|from typing import Optional
     9|
    10|_MONTHS = {
    11|    "january": 1, "february": 2, "march": 3, "april": 4,
    12|    "may": 5, "june": 6, "july": 7, "august": 8,
    13|    "september": 9, "october": 10, "november": 11, "december": 12,
    14|    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    15|    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    16|}
    17|
    18|_DATE_PATTERNS = [
    19|    # "May 29, 1932" or "May 29 1932"
    20|    (r"([A-Za-z]+)\s+(\d{1,2})[,]?\s+(\d{4})", lambda m: _date_iso(m.group(1), m.group(2), m.group(3))),
    21|    # "29 May 1932" or "29-May-32" or "29-May-1932"
    22|    (r"(\d{1,2})\s*[-–]\s*([A-Za-z]+)\s*[-–]\s*(\d{2,4})", lambda m: _date_iso(m.group(2), m.group(1), m.group(3))),
    23|    # "29 May 1932" (no dash)
    24|    (r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", lambda m: _date_iso(m.group(2), m.group(1), m.group(3))),
    25|    # "June 20, 1837"
    26|    (r"([A-Za-z]+)\s+(\d{1,2})[,]?\s+(\d{4})", lambda m: _date_iso(m.group(1), m.group(2), m.group(3))),
    27|]
    28|
    29|
    30|def _date_iso(month_str: str, day_str: str, year_str: str) -> str:
    31|    """Convert month/day/year to ISO 8601 (YYYY-MM-DD)."""
    32|    month = _MONTHS.get(month_str.lower().strip(".,"))
    33|    if month is None:
    34|        return f"{year_str}-{month_str}-{day_str}"
    35|    day = int(day_str)
    36|    year = int(year_str)
    37|    # Fix 2-digit years
    38|    if year < 100:
    39|        year += 1900 if year >= 30 else 2000
    40|    try:
    41|        return datetime(year, month, day).strftime("%Y-%m-%d")
    42|    except (ValueError, OverflowError):
    43|        return f"{year:04d}-{month:02d}-{int(day_str):02d}"
    44|
    45|
    46|def normalize_value(value: str) -> str:
    47|    """
    48|    Normalize a value to canonical format.
    49|    - Dates → YYYY-MM-DD
    50|    - Numbers → stripped of commas/symbols
    51|    - Whitespace → collapsed
    52|    """
    53|    if not value or not isinstance(value, str):
    54|        return value
    55|
    56|    stripped = value.strip()
    57|
    58|    # Try date patterns
    59|    for pattern, handler in _DATE_PATTERNS:
    60|        m = re.match(pattern, stripped, re.IGNORECASE)
    61|        if m:
    62|            return handler(m)
    63|
    64|    # Number: "1,234" → "1234", "$50" → "50"
    65|    stripped = re.sub(r"[$,€£¥]", "", stripped).strip()
    66|
    67|    # Collapse whitespace
    68|    stripped = re.sub(r"\s+", " ", stripped)
    69|
    70|    return stripped
    71|
    72|
    73|def is_date_string(value: str) -> bool:
    74|    """Check if a value looks like a date."""
    75|    for pattern, _ in _DATE_PATTERNS:
    76|        if re.match(pattern, value.strip(), re.IGNORECASE):
    77|            return True
    78|    return False
    79|
    80|
    81|def is_numeric(value: str) -> bool:
    82|    """Check if a value is numeric (after removing common symbols)."""
    83|    cleaned = re.sub(r"[$,€£¥,.\s]", "", value.strip())
    84|    return cleaned.isdigit()
    85|