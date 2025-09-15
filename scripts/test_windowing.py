import os
from labgraphcast.data.io import load_occupancy_csv
from labgraphcast.data.windowing import make_time_windows

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "sample_occupancy.csv")

def main():
    df = load_occupancy_csv(CSV)
    X, y, slots = make_time_windows(df, lab_id="LAB_A", input_len=3, pred_len=1)
    print("Slots order:", slots)
    print("X shape:", X.shape)   # (samples, 3 days, n_slots, 1)
    print("y shape:", y.shape)   # (samples, 1 day, n_slots)
    print("Example input block (first sample, day 0):")
    print(X[0,0,:,0])  # counts for first day, all slots
    print("Target (next day occupancy):")
    print(y[0,0,:])    # counts for predicted day

if __name__ == "__main__":
    main()
