# labgraphcast/data/calendar.py

import re
import pandas as pd

# Map many month spellings to month numbers
MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Standardized set of output flags we maintain
DEFAULT_FLAGS = [
    "is_university_closed",
    "is_no_classes",
    "is_exam_period",
    "is_reading_day",
    "is_recess",
]


def _parse_month_token(token: str) -> int:
    """Parse a free-form month token like 'Sept' or 'December' into a 1..12 int."""
    t = str(token).strip().lower()
    t = re.sub(r"[^a-z]", "", t)  # letters only
    if t in MONTH_MAP:
        return MONTH_MAP[t]
    raise ValueError(f"Unrecognized month token: {token}")


def _derive_flags_from_description(desc_series: pd.Series) -> pd.DataFrame:
    """Heuristics: generate boolean flags from natural-language descriptions."""
    desc = desc_series.fillna("").str.lower()
    out = pd.DataFrame(index=desc_series.index)
    out["is_university_closed"] = desc.str.contains("university closed")
    out["is_no_classes"] = desc.str.contains(r"\bno classes\b") | desc.str.contains("wellness day")
    out["is_exam_period"] = desc.str.contains("final exams")
    out["is_reading_day"] = desc.str.contains("reading day")
    out["is_recess"] = desc.str.contains("recess")
    return out


def load_njit_calendar(csv_path: str, academic_year_start: int = 2025) -> pd.DataFrame:
    """
    Robustly load NJIT calendar from either:
      A) normalized CSV with a 'date' column (YYYY-MM-DD) and optional 'description' and/or flag columns.
      B) 3-column text: Month | Day | Description (comma-, tab-, or 2+ space-separated).

    For B, months Sep–Dec map to 'academic_year_start'; Jan+ map to academic_year_start+1.

    Returns DataFrame with columns:
      date (YYYY-MM-DD), description (may be empty),
      is_university_closed, is_no_classes, is_exam_period, is_reading_day, is_recess  (all 0/1 ints)
    """
    # First attempt: read as a normal CSV (comma-delimited with header)
    df = pd.read_csv(csv_path, engine="python")
    cols = [c.lower().strip() for c in df.columns]

    if "date" in cols:
        # ----- FORMAT A: normalized CSV -----
        df.columns = cols
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        def _series_or_zeros(name: str) -> pd.Series:
            """Return a boolean Series for 'name' if present; else all-False Series of the right length."""
            if name in df.columns:
                return df[name].fillna(0).astype(bool)
            return pd.Series(False, index=df.index)

        # If we have a description, re-derive flags from text (authoritative).
        # Otherwise, try to use existing boolean-ish columns; missing ones default to zeros.
        if "description" in df.columns:
            flags = _derive_flags_from_description(df["description"])
        else:
            flags = pd.DataFrame(index=df.index)
            flags["is_university_closed"] = _series_or_zeros("is_university_closed") | _series_or_zeros("university_closed")
            flags["is_no_classes"] = _series_or_zeros("is_no_classes") | _series_or_zeros("no_classes")
            flags["is_exam_period"] = _series_or_zeros("is_exam_period") | _series_or_zeros("exam_period")
            flags["is_reading_day"] = _series_or_zeros("is_reading_day") | _series_or_zeros("reading_day")
            flags["is_recess"] = _series_or_zeros("is_recess") | _series_or_zeros("recess")

        out = pd.DataFrame({"date": df["date"]})
        out["description"] = df["description"] if "description" in df.columns else ""
        out = pd.concat([out, flags], axis=1)

    else:
        # ----- FORMAT B: free-form Month | Day | Description -----
        # Re-read with a flexible separator capturing commas, tabs, or 2+ spaces.
        df = pd.read_csv(
            csv_path,
            sep=r"\t+|\s{2,}|,",
            engine="python",
            header=None,
            names=["month", "day", "description"],
        )

        # If everything landed in one column, try to split again heuristically
        if df.shape[1] == 1:
            parts = df["month"].str.split(r"\s{2,}|\t+|,", expand=True)
            if parts.shape[1] >= 3:
                df = parts.iloc[:, :3]
                df.columns = ["month", "day", "description"]
            else:
                raise ValueError(
                    "Unable to parse calendar; expected three columns: month, day, description."
                )

        # Normalize month/day → date string
        df["month_num"] = df["month"].apply(_parse_month_token)
        # Fall months (Sep–Dec) belong to academic_year_start; Jan–Aug are next calendar year
        df["year"] = df["month_num"].apply(lambda m: academic_year_start if m >= 9 else academic_year_start + 1)
        # Sanitize day
        df["day"] = df["day"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)

        df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month_num"], day=df["day"])).dt.strftime("%Y-%m-%d")

        # Derive flags from description text
        flags = _derive_flags_from_description(df["description"])
        out = pd.concat([df[["date", "description"]], flags], axis=1)

    # Ensure all expected flags exist and are 0/1 ints
    for c in DEFAULT_FLAGS:
        if c not in out.columns:
            out[c] = 0
        out[c] = out[c].astype(int)

    out = out.sort_values("date").reset_index(drop=True)
    return out


def merge_calendar_features(occupancy_df: pd.DataFrame, cal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join date-level calendar flags onto each occupancy row.
    Missing days get zeros; description becomes empty string.
    """
    out = occupancy_df.merge(cal_df, on="date", how="left")
    for c in DEFAULT_FLAGS:
        out[c] = out[c].fillna(0).astype(int)
    if "description" not in out.columns:
        out["description"] = ""
    else:
        out["description"] = out["description"].fillna("")
    return out
