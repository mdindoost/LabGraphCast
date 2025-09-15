import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, n_slots: int, feature_dim: int = 1, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        # Flatten (n_slots, feature_dim) per day → input_size
        self.input_size = n_slots * feature_dim
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, n_slots)

    def forward(self, x):
        # x: (batch, input_len, n_slots, feature_dim)
        b, t, s, f = x.shape
        x = x.reshape(b, t, s * f)   # (batch, input_len, input_size)
        out, _ = self.lstm(x)        # (batch, input_len, hidden_dim)
        last = out[:, -1, :]         # (batch, hidden_dim)
        pred = self.fc(last)         # (batch, n_slots)
        return pred
