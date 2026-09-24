"""Per-epoch encoder: shared multi-scale separable CNN (EEG x2) + small CNN (EOG) + feature MLP.

model.embed(...) -> 64-d epoch embedding (reused by the GRU in Step 9)
model(...)       -> 5-class logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv(nn.Module):
    """Depthwise-separable conv + BN + ReLU with residual (ULW-SleepNet style)."""
    def __init__(self, cin, cout, k):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, padding="same", groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.skip = nn.Identity() if cin == cout else nn.Conv1d(cin, cout, 1, bias=False)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x)))) + self.skip(x)


class Branch(nn.Module):
    def __init__(self, width, k0, s0, pool0, k):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, k0, stride=s0, padding=k0 // 2, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(pool0), nn.Dropout(0.1))
        self.blocks = nn.Sequential(
            SepConv(width, 2 * width, k), SepConv(2 * width, 2 * width, k),
            nn.MaxPool1d(2), SepConv(2 * width, 2 * width, k))

    def forward(self, x):
        return self.blocks(self.stem(x)).mean(-1)          # global average pooling


class Encoder(nn.Module):
    """Two time scales: fine (0.5 s kernel: spindles, K-complexes, REMs) and coarse (4 s: delta, alpha)."""
    def __init__(self, width):
        super().__init__()
        self.fine = Branch(width, k0=50, s0=6, pool0=8, k=7)
        self.coarse = Branch(width, k0=400, s0=50, pool0=4, k=5)
        self.out_dim = 4 * width

    def forward(self, x):                                   # x: (B, 1, 3000)
        return torch.cat([self.fine(x), self.coarse(x)], 1)


class SleepEpochNet(nn.Module):
    def __init__(self, n_feat=34, emb=64, n_cls=5):
        super().__init__()
        self.eeg = Encoder(16)          # shared by Fpz-Cz and Pz-Oz
        self.eog = Encoder(8)
        self.feat = nn.Sequential(nn.Linear(n_feat, 32), nn.ReLU())
        d = 2 * self.eeg.out_dim + self.eog.out_dim + 32 + 3
        self.proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, emb), nn.ReLU())
        self.head = nn.Linear(emb, n_cls)

    def embed(self, x, f, m):
        """x (B,3,T) signals, f (B,F) features, m (B,3) channel-OK mask."""
        B = x.shape[0]
        e = self.eeg(x[:, :2].reshape(B * 2, 1, -1)).view(B, 2, -1) * m[:, :2, None]
        o = self.eog(x[:, 2:3]) * m[:, 2:3]
        return self.proj(torch.cat([e.flatten(1), o, self.feat(f), m], 1))

    def forward(self, x, f, m):
        return self.head(self.embed(x, f, m))


class SleepSeqNet(nn.Module):
    """Causal context model (Step 9): unidirectional GRU over the last L epoch embeddings.
    prov head -> stage of the newest epoch t (live output)
    rev  head -> stage of epoch t-1, now that epoch t is known (revised output, 30 s later)
    """
    def __init__(self, emb=64, hid=64, n_cls=5):
        super().__init__()
        self.drop = nn.Dropout(0.1)
        self.gru = nn.GRU(emb + 1, hid, batch_first=True)
        self.prov = nn.Linear(hid, n_cls)
        self.rev = nn.Linear(hid, n_cls)

    def forward(self, z, v):
        """z (B,L,emb) embeddings, v (B,L) 1 = usable epoch, 0 = bad signal / before recording start."""
        h, _ = self.gru(torch.cat([self.drop(z) * v[..., None], v[..., None]], -1))
        h = h[:, -1]
        return self.prov(h), self.rev(h)


if __name__ == "__main__":
    net = SleepEpochNet()
    n = sum(p.numel() for p in net.parameters())
    out = net(torch.randn(4, 3, 3000), torch.randn(4, 34), torch.ones(4, 3))
    print(f"params: {n:,}  output: {tuple(out.shape)}")
