import sys, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import HORIZON, INPUT_SIZE, MODELS_DIR
from models.baselines import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN, N_BLOCKS, EPOCHS, LR, BATCH, PATIENCE = 256, 3, 60, 5e-4, 64, 10

class _Block(nn.Module):
    def __init__(self, stype, isz, h):
        super().__init__()
        self.stype = stype; self.isz = isz; self.h = h
        self.fc = nn.Sequential(
            nn.Linear(isz, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU()
        )
        out = 8 if stype == "trend" else 32
        self.th = nn.Linear(HIDDEN, out, bias=False)
        self.register_buffer("tb", torch.linspace(0, 1, isz))
        self.register_buffer("tf", torch.linspace(0, 1, h))
    def forward(self, x):
        theta = self.th(self.fc(x)); B = x.size(0)
        tb = self.tb.unsqueeze(0).expand(B, -1)
        tf = self.tf.unsqueeze(0).expand(B, -1)
        if self.stype == "trend":
            deg = theta.shape[-1] // 2
            pb = torch.stack([tb**i for i in range(deg)], -1)
            pf = torch.stack([tf**i for i in range(deg)], -1)
            bc = (pb * theta[:, :deg].unsqueeze(1)).sum(-1)
            fc_ = (pf * theta[:, deg:].unsqueeze(1)).sum(-1)
        else:
            half = theta.shape[-1] // 4
            freqs = torch.arange(1, half+1, device=x.device).float()
            def fou(t, c, s):
                ph = 2 * torch.pi * freqs.unsqueeze(0) * t.unsqueeze(-1)
                return (c.unsqueeze(1)*torch.cos(ph) + s.unsqueeze(1)*torch.sin(ph)).sum(-1)
            bc  = fou(tb, theta[:, :half], theta[:, half:2*half])
            fc_ = fou(tf, theta[:, 2*half:3*half], theta[:, 3*half:])
        return bc, fc_

class _NBEATSNet(nn.Module):
    def __init__(self, isz=INPUT_SIZE, h=HORIZON):
        super().__init__()
        blks = []
        for st in ["trend", "seasonality"]:
            for _ in range(N_BLOCKS): blks.append(_Block(st, isz, h))
        self.blocks = nn.ModuleList(blks); self.h = h
    def forward(self, x):
        res = x.clone(); fc = torch.zeros(x.size(0), self.h, device=x.device)
        for b in self.blocks:
            bc, f = b(res); res = res - bc; fc = fc + f
        return fc

def _win(s, isz=INPUT_SIZE, h=HORIZON):
    X, y = [], []
    for i in range(len(s)-isz-h+1): X.append(s[i:i+isz]); y.append(s[i+isz:i+isz+h])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class NBEATSForecaster:
    def __init__(self, horizon=HORIZON, input_size=INPUT_SIZE):
        self.horizon = horizon; self.input_size = input_size
        self.model = _NBEATSNet(input_size, horizon).to(DEVICE)
        self.scaler = RobustScaler(); self.fitted = False
    def fit(self, train_arr, val_arr=None):
        sc = self.scaler.fit_transform(train_arr.reshape(-1,1)).flatten()
        X, y = _win(sc, self.input_size, self.horizon)
        loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), BATCH, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
        fn = nn.MSELoss(); best, wait, bst = np.inf, 0, None
        for ep in range(EPOCHS):
            self.model.train(); el = 0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
                l = fn(self.model(xb), yb); l.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); el += l.item()
            el /= len(loader); sched.step(el)
            if el < best - 1e-5:
                best = el; wait = 0
                bst = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    logger.info(f"[N-BEATS] early stop ep{ep+1}"); break
            if (ep+1) % 10 == 0: logger.info(f"[N-BEATS] ep{ep+1} loss={el:.5f}")
        if bst: self.model.load_state_dict(bst)
        self.fitted = True; logger.info("[N-BEATS] Training complete."); return self
    def predict(self, last):
        seq = last[-self.input_size:]
        sc = self.scaler.transform(seq.reshape(-1,1)).flatten()
        x = torch.tensor(sc, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        self.model.eval()
        with torch.no_grad(): p = self.model(x).cpu().numpy().flatten()
        return self.scaler.inverse_transform(p.reshape(-1,1)).flatten()
    def rolling_evaluate(self, series, n_windows=100):
        n = len(series); start = max(0, n - n_windows*self.horizon - self.input_size)
        step = max(1, (n - start - self.input_size - self.horizon) // n_windows)
        yt, yp = [], []
        for i in range(start, n - self.input_size - self.horizon, step):
            yt.append(series[i+self.input_size:i+self.input_size+self.horizon])
            yp.append(self.predict(series[i:i+self.input_size]))
            if len(yt) >= n_windows: break
        return np.array(yt).flatten(), np.array(yp).flatten()
    def save(self, path=None):
        p = Path(path or MODELS_DIR / "nbeats.pt"); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p)
        pickle.dump(self.scaler, open(str(p).replace(".pt","_scaler.pkl"), "wb"))
        logger.info(f"[N-BEATS] saved -> {p}")
    def load(self, path=None):
        p = Path(path or MODELS_DIR / "nbeats.pt")
        self.model.load_state_dict(torch.load(p, map_location=DEVICE))
        self.scaler = pickle.load(open(str(p).replace(".pt","_scaler.pkl"), "rb"))
        self.fitted = True; return self
