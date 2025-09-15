
import pandas as pd

def seasonal_naive(df: pd.DataFrame, horizon_slots=8):
    """Very simple: predict next values by copying the same-slot previous day."""
    # Phase 0: placeholder returning the last day observed for each slot.
    last_date = df["date"].max()
    return {"last_date": last_date, "note": "implement in Phase 1"}
