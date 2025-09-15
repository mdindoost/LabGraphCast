import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from labgraphcast.data.io import load_occupancy_csv
from labgraphcast.data.calendar import load_njit_calendar, merge_calendar_features
from labgraphcast.data.windowing import make_time_windows
from labgraphcast.models.lstm_baseline import LSTMForecast
from labgraphcast.eval.metrics import mae

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "sample_occupancy.csv")
CAL = os.path.join(ROOT, "data", "njit_calendar.csv")

def main():
    # Load data
    df = load_occupancy_csv(CSV)

    # Load & merge calendar
    cal = load_njit_calendar(CAL, academic_year_start=2025)
    dfm = merge_calendar_features(df, cal)

    # Build windows with calendar channels
    X, y, slots, feat_dim = make_time_windows(dfm, lab_id="LAB_A", input_len=3, pred_len=1)

    # Torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)             # (N, T, S, F)
    y_tensor = torch.tensor(y[:,0,:], dtype=torch.float32)      # (N, S)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

    model = LSTMForecast(n_slots=len(slots), feature_dim=feat_dim, hidden_dim=64, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train (tiny demo)
    for epoch in range(50):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total/len(loader):.4f}")

    with torch.no_grad():
        preds = model(X_tensor).numpy()
    score = mae(y_tensor.numpy().flatten(), preds.flatten())
    print(f"LSTM (with calendar) MAE: {score:.4f}")

if __name__ == "__main__":
    # Optional: force CPU to avoid CUDA probe warnings
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
