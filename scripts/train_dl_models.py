import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import RAW_CSV, TARGET_COL, HORIZON, RESULTS_DIR
from models.baselines import evaluate, NaiveForecaster, HoltWintersForecaster
from models.lstm_forecaster   import LSTMForecaster
from models.nbeats_forecaster import NBEATSForecaster
from models.tide_forecaster   import TiDEForecaster

def main():
    df = pd.read_csv(RAW_CSV)
    s = df[TARGET_COL].dropna().values.astype("float32"); n = len(s)
    train, val, test = s[:int(n*.7)], s[int(n*.7):int(n*.85)], s[int(n*.85):]
    logger.info(f"train={len(train)} val={len(val)} test={len(test)}")
    results = []
    # Naive
    nv = NaiveForecaster(horizon=HORIZON)
    nv.fit(train)
    yt, yp = nv.rolling_evaluate(np.concatenate([train, test]), n_windows=80)
    results.append(evaluate(yt.flatten(), yp.flatten(), "Naive"))

    # Holt-Winters (needs train + test separately)
    hw = HoltWintersForecaster(horizon=HORIZON)
    yt, yp = hw.rolling_evaluate(train, test, n_windows=40)
    results.append(evaluate(yt.flatten(), yp.flatten(), "Holt-Winters"))
    for name, cls in [("LSTM (Bi-Attn)", LSTMForecaster), ("N-BEATS", NBEATSForecaster), ("TiDE", TiDEForecaster)]:
        logger.info(f"\n=== {name} ===")
        m = cls(); m.fit(train, val); m.save()
        yt, yp = m.rolling_evaluate(test, n_windows=80)
        results.append(evaluate(yt, yp, name))
    df_res = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    df_res.to_csv(RESULTS_DIR / "dl_benchmark.csv", index=False)
    logger.info(f"\n{df_res.to_string(index=False)}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("DL Benchmark - AIOps", fontsize=13, fontweight="bold")
    colors = ["#8172B2","#8172B2","#C44E52","#4C72B0","#55A868"]
    for ax, metric in zip(axes, ["RMSE","MAE"]):
        ax.barh(df_res["model"], df_res[metric], color=colors[:len(df_res)], alpha=0.85)
        ax.set_title(f"{metric} (lower=better)"); ax.invert_yaxis()
        hw = df_res.loc[df_res["model"]=="Holt-Winters", metric]
        if not hw.empty: ax.axvline(hw.values[0], color="red", linestyle="--", label="HW"); ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dl_benchmark.png", dpi=150, bbox_inches="tight")
    logger.info("Done!")

if __name__ == "__main__": main()
