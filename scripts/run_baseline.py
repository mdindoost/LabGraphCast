import os
import pandas as pd
from labgraphcast.data.io import load_occupancy_csv
from labgraphcast.data.features import add_basic_features
from labgraphcast.models.baselines import naive_prevday
from labgraphcast.eval.plotting import plot_actual_vs_pred

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "sample_occupancy.csv")

def main():
    df = load_occupancy_csv(CSV)
    df = add_basic_features(df)

    # Predict next day from previous day
    preds = naive_prevday(df)

    # Keep only rows where we have ground truth on the predicted date
    merged = pd.merge(df, preds, on=["lab_id", "date", "slot"], how="inner")

    # IMPORTANT: drop y_pred from the truth frame before plotting/merging inside the helper
    df_true = merged.drop(columns=["y_pred"])
    df_pred = merged[["lab_id", "date", "slot", "y_pred"]]

    out, score = plot_actual_vs_pred(
        df_true, df_pred,
        lab_id="LAB_A",
        out_path=os.path.join(ROOT, "baseline_plot.png")
    )
    print("Saved plot:", out)
    print(f"MAE (1-day-ahead naive): {score:.4f}")

if __name__ == "__main__":
    main()
