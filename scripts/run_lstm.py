import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from labgraphcast.data.io import load_occupancy_csv
from labgraphcast.data.windowing import make_time_windows
from labgraphcast.models.lstm_baseline import LSTMForecast
from labgraphcast.eval.metrics import mae

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV = os.path.join(ROOT, "data", "sample_occupancy.csv")

def main():
    df = load_occupancy_csv(CSV)
    X, y, slots = make_time_windows(df, lab_id="LAB_A", input_len=3, pred_len=1)

    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y[:,0,:], dtype=torch.float32)  # shape (samples, n_slots)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = LSTMForecast(n_slots=len(slots), hidden_dim=32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    for epoch in range(50):
        total_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={total_loss/len(loader):.4f}")

    # Evaluate MAE on training set (tiny demo dataset)
    with torch.no_grad():
        preds = model(X_tensor).numpy()
    score = mae(y_tensor.numpy().flatten(), preds.flatten())
    print("LSTM MAE:", score)

if __name__ == "__main__":
    main()
