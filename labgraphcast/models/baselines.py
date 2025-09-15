import pandas as pd

def naive_prevday(df: pd.DataFrame) -> pd.DataFrame:
    """
    1-day-ahead naive baseline:
      y_pred(lab, date+1, slot) = count(lab, date, slot)
    Returns DataFrame with columns: lab_id, date, slot, y_pred
    """
    df = df.sort_values(["lab_id","date","slot"]).copy()
    # Shift counts by +1 day per lab & slot
    df["date_next"] = pd.to_datetime(df["date"]) + pd.Timedelta(days=1)
    preds = df[["lab_id", "date_next", "slot", "count"]].copy()
    preds = preds.rename(columns={"date_next": "date", "count": "y_pred"})
    # Back to string date to match CSV schema
    preds["date"] = preds["date"].dt.strftime("%Y-%m-%d")
    return preds
