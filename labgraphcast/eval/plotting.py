import matplotlib.pyplot as plt
import pandas as pd
from .metrics import mae

def plot_actual_vs_pred(df_true: pd.DataFrame, df_pred: pd.DataFrame, lab_id="LAB_A", out_path="baseline_plot.png"):
    t = df_true[df_true["lab_id"] == lab_id].copy()
    p = df_pred[df_pred["lab_id"] == lab_id].copy()
    m = pd.merge(t, p, on=["lab_id","date","slot"], how="inner")
    m = m.sort_values(["date","slot"])
    score = mae(m["count"].values, m["y_pred"].values)

    plt.figure()
    plt.plot(range(len(m)), m["count"].values, label="actual")
    plt.plot(range(len(m)), m["y_pred"].values, label="seasonal_naive")
    plt.title(f"{lab_id} — MAE={score:.2f}")
    plt.xlabel("time index (sorted by date/slot)")
    plt.ylabel("occupancy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    return out_path, score
