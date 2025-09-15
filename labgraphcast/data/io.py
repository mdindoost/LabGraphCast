
import pandas as pd

REQUIRED_COLS = ["date", "weekday", "slot", "lab_id", "count"]

def load_occupancy_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # sort for reproducibility
    return df.sort_values(["lab_id","date","slot"]).reset_index(drop=True)
