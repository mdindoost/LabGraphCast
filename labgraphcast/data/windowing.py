import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Sequence

DEFAULT_DATE_FEATURES = [
    "is_university_closed",
    "is_no_classes",
    "is_exam_period",
    "is_reading_day",
    "is_recess",
]

def make_time_windows(
    df: pd.DataFrame,
    lab_id: str,
    input_len: int = 7,
    pred_len: int = 1,
    slot_order: Optional[List[str]] = None,
    date_feature_cols: Sequence[str] = DEFAULT_DATE_FEATURES,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Build supervised windows with channels:
      channel 0: count
      channels 1..F: date-level calendar flags broadcast across slots

    Returns:
      X: (n_samples, input_len, n_slots, 1+F)
      y: (n_samples, pred_len, n_slots)   # targets are counts only
      slot_order: list[str]
      n_feat: number of input channels (1+F)
    """
    d = df[df["lab_id"] == lab_id].copy().sort_values(["date","slot"])
    if slot_order is None:
        slot_order = sorted(d["slot"].unique().tolist())
    n_slots = len(slot_order)

    # pivot counts to (n_days, n_slots)
    pivot = d.pivot_table(index="date", columns="slot", values="count")
    pivot = pivot.reindex(columns=slot_order).sort_index()
    counts = pivot.values.astype(float)

    # collect date-level features in same date order as pivot.index
    feat_df = pd.DataFrame(index=pivot.index)
    for col in date_feature_cols:
        if col in d.columns:
            # value is constant per date; take first
            tmp = d.groupby("date")[col].first().reindex(pivot.index).fillna(0).astype(float)
        else:
            tmp = pd.Series(0.0, index=pivot.index)
        feat_df[col] = tmp

    # broadcast features across slots: (n_days, n_slots, F)
    F = len(date_feature_cols)
    if F > 0:
        feat = np.repeat(feat_df.values[:, None, :], n_slots, axis=1)
    else:
        feat = np.zeros((counts.shape[0], n_slots, 0), dtype=float)

    # counts as channel 0: (n_days, n_slots, 1)
    cnt_ch = counts[:, :, None]

    # stack channels → (n_days, n_slots, 1+F)
    full = np.concatenate([cnt_ch, feat], axis=2)

    # build sliding windows
    X_list, y_list = [], []
    for i in range(len(full) - input_len - pred_len + 1):
        in_block = full[i : i + input_len]                     # (input_len, n_slots, 1+F)
        out_block = counts[i + input_len : i + input_len + pred_len]  # (pred_len, n_slots)
        X_list.append(in_block)
        y_list.append(out_block)

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, slot_order, full.shape[-1]
