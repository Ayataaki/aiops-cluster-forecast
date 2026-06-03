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
D_MODEL, N_ENC, N_DEC, DROPOUT = 256, 2, 2, 0.3
EPOCHS, LR, BATCH, PATIENCE = 60, 5e-4, 64, 10

class _Res(nn.Module):
    def __init__(self, i, o, d=DROPOUT):
        super().__init__()
        self.fc = nn.Linear(i, o); self.ln = nn.LayerNorm(o); self.drop = nn.Dropout(d)
        self.proj = nn.Linear(i, o) if i != o else nn.Identity()
    def forward(self, x):
        return self.ln(self.drop(torch.relu(self.fc(x))) + self.proj(x))

class _TiDENet(nn.Module):
    def __init__(self, isz=INPUT_SIZE, h=HORIZON):
        super().__init__()
        self.iln = nn.LayerNorm(isz)
        self.enc = nn.Sequential(_Res(isz, D_MODEL), *[_Res(D_MODEL, D_MODEL) for _ in range(N_ENC-1)])
        self.dec = nn.Sequential(*[_Res(D_MODEL, D_MODEL) for _ in range(N_DEC)])
        self.tmp = nn.Linear(D_MODEL, h)
        self.skip = nn.Linear(isz, h)
    def forward(self, x):
        xn = self.iln(x)
        return self.tmp(self.dec(self.enc(xn))) + self.skip(xn)

def _win(s, isz=INPUT_SIZE, h=HORIZON):
    X, y = [], []
    for i in range(len(s)-isz-h+1): X.append(s[i:i+isz]); y.append(s[i+isz:i+isz+h])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class TiDEForecaster:
    def __init__(self, horizon=HORIZON, input_size=INPUT_SIZE):
        self.horizon = horizon; self.input_size = input_size
        self.model = _TiDENet(input_size, horizon).to(DEVICE)
        self.scaler = RobustScaler(); self.fitted = False
    def fit(self, train_arr, val_arr=None):
        sc = self.scaler.fit_transform(train_arr.reshape(-1,1)).flatten()
        X, y = _win(sc, self.input_size, self.horizon)
        loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), BATCH, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        fn = nn.HuberLoss(delta=0.5); best, wait, bst = np.inf, 0, None
        for ep in range(EPOCHS):
            self.model.train(); el = 0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
                l = fn(self.model(xb), yb); l.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); el += l.item()
            el /= len(loader); sched.step()
            if el < best - 1e-5:
                best = el; wait = 0
                bst = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    logger.info(f"[TiDE] early stop ep{ep+1}"); break
            if (ep+1) % 10 == 0: logger.info(f"[TiDE] ep{ep+1} loss={el:.5f}")
        if bst: self.model.load_state_dict(bst)
        self.fitted = True; logger.info("[TiDE] Training complete."); return self
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
        p = Path(path or MODELS_DIR / "tide.pt"); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p)
        pickle.dump(self.scaler, open(str(p).replace(".pt","_scaler.pkl"), "wb"))
        logger.info(f"[TiDE] saved -> {p}")
    def load(self, path=None):
        p = Path(path or MODELS_DIR / "tide.pt")
        self.model.load_state_dict(torch.load(p, map_location=DEVICE))
        self.scaler = pickle.load(open(str(p).replace(".pt","_scaler.pkl"), "rb"))
        self.fitted = True; return self
