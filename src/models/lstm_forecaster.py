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
HIDDEN, LAYERS, DROPOUT = 128, 2, 0.25
EPOCHS, LR, BATCH, PATIENCE = 60, 1e-3, 64, 8

class _Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.w = nn.Linear(hidden * 2, 1)
    def forward(self, out):
        scores = self.w(out).squeeze(-1)
        return (out * torch.softmax(scores, -1).unsqueeze(-1)).sum(1)

class _BiLSTMAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, LAYERS, batch_first=True, dropout=DROPOUT, bidirectional=True)
        self.attn = _Attention(HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN*2, 128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, HORIZON)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.attn(out))

def _windows(s, isz=INPUT_SIZE, h=HORIZON):
    X, y = [], []
    for i in range(len(s) - isz - h + 1):
        X.append(s[i:i+isz]); y.append(s[i+isz:i+isz+h])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class LSTMForecaster:
    def __init__(self, horizon=HORIZON, input_size=INPUT_SIZE):
        self.horizon = horizon; self.input_size = input_size
        self.model = _BiLSTMAttn().to(DEVICE)
        self.scaler = RobustScaler(); self.fitted = False
    def fit(self, train_arr, val_arr=None):
        sc = self.scaler.fit_transform(train_arr.reshape(-1,1)).flatten()
        X, y = _windows(sc, self.input_size, self.horizon)
        loader = DataLoader(TensorDataset(torch.tensor(X).unsqueeze(-1), torch.tensor(y)), BATCH, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        fn = nn.HuberLoss(delta=0.5)
        best, wait, bst = np.inf, 0, None
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
                    logger.info(f"[LSTM] early stop ep{ep+1} best={best:.5f}"); break
            if (ep+1) % 10 == 0: logger.info(f"[LSTM] ep{ep+1} loss={el:.5f}")
        if bst: self.model.load_state_dict(bst)
        self.fitted = True; logger.info("[LSTM] Training complete."); return self
    def predict(self, last):
        seq = last[-self.input_size:]
        sc = self.scaler.transform(seq.reshape(-1,1)).flatten()
        x = torch.tensor(sc, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
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
        p = Path(path or MODELS_DIR / "lstm.pt"); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p)
        pickle.dump(self.scaler, open(str(p).replace(".pt","_scaler.pkl"), "wb"))
        logger.info(f"[LSTM] saved -> {p}")
    def load(self, path=None):
        p = Path(path or MODELS_DIR / "lstm.pt")
        self.model.load_state_dict(torch.load(p, map_location=DEVICE))
        self.scaler = pickle.load(open(str(p).replace(".pt","_scaler.pkl"), "rb"))
        self.fitted = True; return self
