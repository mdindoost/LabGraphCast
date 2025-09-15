import torch
import torch.nn as nn

class SimpleGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_norm):
        """
        x: (B, S, F)   features per slot
        A_norm: (S, S) dense normalized adjacency
        """
        # message passing: X' = A_norm @ X @ W
        xw = self.lin(x)                 # (B, S, out)
        Ax = torch.matmul(A_norm, xw)    # (S, S) @ (B, S, out) -> (B, S, out)
        return Ax

class TemporalGCNGRU(nn.Module):
    def __init__(self, n_slots: int, in_feat: int, gcn_hidden: int = 32, rnn_hidden: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gcn1 = SimpleGCN(in_feat, gcn_hidden)
        self.act = nn.ReLU()
        # We flatten slots after GCN for the temporal GRU
        self.gru = nn.GRU(input_size=n_slots * gcn_hidden, hidden_size=rnn_hidden, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Linear(rnn_hidden, n_slots)

    def forward(self, x, A_norm):
        """
        x: (B, T, S, F)
        A_norm: (S, S)
        """
        B, T, S, F = x.shape
        gcn_out_seq = []
        for t in range(T):
            xt = x[:, t, :, :]           # (B, S, F)
            gt = self.gcn1(xt, A_norm)   # (B, S, gcn_hidden)
            gt = self.act(gt)
            gcn_out_seq.append(gt.reshape(B, -1))  # flatten slots
        G = torch.stack(gcn_out_seq, dim=1)        # (B, T, S*gcn_hidden)
        out, _ = self.gru(G)                       # (B, T, rnn_hidden)
        last = out[:, -1, :]                       # (B, rnn_hidden)
        pred = self.fc(last)                       # (B, S)
        return pred
