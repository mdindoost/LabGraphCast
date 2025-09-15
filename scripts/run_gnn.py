import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from labgraphcast.data.io import load_occupancy_csv
from labgraphcast.data.calendar import load_njit_calendar, merge_calendar_features
from labgraphcast.data.windowing import make_time_windows
from labgraphcast.labgraph.adjacency import build_slot_adjacency, normalized_adj
from labgraphcast.models.temporal_gcn import TemporalGCNGRU
from labgraphcast.eval.metrics import mae

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "sample_occupancy.csv")
CAL = os.path.join(ROOT, "data", "njit_calendar.csv")

def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # keep CPU

    # Data + calendar
    df = load_occupancy_csv(CSV)
    cal = load_njit_calendar(CAL, academic_year_start=2025)
    dfm = merge_calendar_features(df, cal)

    # Windows (same as LSTM): X:(N,T,S,F), y:(N,1,S)
    X, y, slots, feat_dim = make_time_windows(dfm, lab_id="LAB_A", input_len=3, pred_len=1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y[:,0,:], dtype=torch.float32)

    # Slot graph
    idx, vals, shape = build_slot_adjacency(slots, cyclic=False)
    A_norm = normalized_adj(idx, vals, shape)  # (S, S)

    # Dataloader
    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True)

    # Model
    model = TemporalGCNGRU(n_slots=len(slots), in_feat=feat_dim, gcn_hidden=32, rnn_hidden=64, num_layers=1)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=0.01)

    # Train
    for epoch in range(60):
        total = 0.0
        for xb, yb in dl:
            pred = model(xb, A_norm)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total/len(dl):.4f}")

    with torch.no_grad():
        preds = model(X_tensor, A_norm).numpy()
    score = mae(y_tensor.numpy().flatten(), preds.flatten())
    print(f"Temporal GCN+GRU MAE: {score:.4f}")

if __name__ == "__main__":
    main()
