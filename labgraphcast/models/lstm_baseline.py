import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, n_slots: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        # Each day is (n_slots, 1 feature) → flatten into n_slots features
        self.input_dim = n_slots
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, n_slots)

    def forward(self, x):
        # x: (batch, input_len, n_slots, 1)
        b, t, s, f = x.shape
        x = x.view(b, t, s * f)  # (batch, input_len, n_slots)
        out, _ = self.lstm(x)    # out: (batch, input_len, hidden)
        last = out[:, -1, :]     # last hidden state
        pred = self.fc(last)     # (batch, n_slots)
        return pred
