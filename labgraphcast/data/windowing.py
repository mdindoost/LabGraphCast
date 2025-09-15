import numpy as np
import pandas as pd
from typing import Tuple, List

def make_time_windows(
    df: pd.DataFrame,
    lab_id: str,
    input_len: int = 7,
    pred_len: int = 1,
    slot_order: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert occupancy data into supervised windows for forecasting.
    
    Args:
        df: DataFrame with columns [date, weekday, slot, lab_id, count, ...features...]
        lab_id: which lab to filter
        input_len: number of days to use as input window
        pred_len: number of days to predict
        slot_order: explicit order of slots (if None, inferred from sorted unique)
    
    Returns:
        X: np.ndarray, shape (n_samples, input_len, n_slots, n_features)
        y: np.ndarray, shape (n_samples, pred_len, n_slots)
        slot_order: list of slots used
    """
    d = df[df["lab_id"] == lab_id].copy().sort_values(["date","slot"])
    if slot_order is None:
        slot_order = sorted(d["slot"].unique().tolist())
    n_slots = len(slot_order)
    
    # Pivot to [date × slot] matrix
    pivot = d.pivot_table(index="date", columns="slot", values="count")
    pivot = pivot.reindex(columns=slot_order)
    pivot = pivot.sort_index()  # by date
    
    # Features (weekday one-hot, slot sin/cos, etc.) could be added per-slot
    # For now: only count used as feature
    values = pivot.values  # shape: (n_days, n_slots)

    X_list, y_list = [], []
    dates = pivot.index.tolist()
    
    for i in range(len(values) - input_len - pred_len + 1):
        in_block = values[i : i + input_len]        # shape (input_len, n_slots)
        out_block = values[i + input_len : i + input_len + pred_len]  # (pred_len, n_slots)
        # Add feature dimension: (input_len, n_slots, 1)
        in_block = np.expand_dims(in_block, -1)
        X_list.append(in_block)
        y_list.append(out_block)
    
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, slot_order
