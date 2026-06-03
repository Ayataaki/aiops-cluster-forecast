import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_CSV, TARGET_COL, HORIZON, MODELS_DIR
from models.lstm_forecaster   import LSTMForecaster
from models.nbeats_forecaster import NBEATSForecaster
from models.tide_forecaster   import TiDEForecaster
from models.baselines         import NaiveForecaster, HoltWintersForecaster

st.set_page_config(page_title="AIOps — Predictive Observability", page_icon="🔮", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {background:#1e1e2e;border-radius:10px;padding:16px;text-align:center;border:1px solid #313244}
    .alert-red   {background:#3b1a1a;border-left:4px solid #f38ba8;padding:10px;border-radius:6px}
    .alert-green {background:#1a2e1a;border-left:4px solid #a6e3a1;padding:10px;border-radius:6px}
    .stMetric label {font-size:0.8rem!important}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_series():
    df = pd.read_csv(RAW_CSV)
    return df[TARGET_COL].dropna().values.astype("float32")

@st.cache_resource
def load_models():
    series = load_series()
    train  = series[:int(len(series)*0.7)]
    models = {}
    for name, cls, path in [
        ("LSTM (Bi-Attn)",     LSTMForecaster,   MODELS_DIR/"lstm.pt"),
        ("N-BEATS",            NBEATSForecaster,  MODELS_DIR/"nbeats.pt"),
        ("TiDE",               TiDEForecaster,    MODELS_DIR/"tide.pt"),
    ]:
        try:
            m = cls(); m.load(path); models[name] = m
        except Exception:
            m = cls(); m.fit(train); models[name] = m
    models["Naive"] = NaiveForecaster(horizon=HORIZON)
    models["Naive"].fit(train)
    return models

series  = load_series()
INPUT_SIZE = 48

# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    model_name   = st.selectbox("Model", ["N-BEATS","LSTM (Bi-Attn)","TiDE","Naive"])
    window_start = st.slider("History window start", 0, max(0,len(series)-INPUT_SIZE-HORIZON-200), len(series)-INPUT_SIZE-HORIZON-50)
    threshold    = st.slider("CPU Alert threshold", 0.50, 0.95, 0.75, 0.05)
    st.markdown("---")
    st.caption("AIOps Predictive Observability\nGoogle Cluster Trace v3 · 2019")

# ── HEADER ───────────────────────────────────────────────────────────
st.title("🔮 AIOps — Predictive Observability Dashboard")
st.caption("CPU forecasting 30 min ahead · Auto-scaling trigger simulation")

# ── LOAD & PREDICT ───────────────────────────────────────────────────
models = load_models()
window  = series[window_start: window_start + INPUT_SIZE]
future  = series[window_start + INPUT_SIZE: window_start + INPUT_SIZE + HORIZON]

model   = models[model_name]
if model_name == "Naive":
    model.fit(window)
    pred = model.predict()
else:
    pred = model.predict(window)

pred    = np.clip(pred, 0, 1)
max_pred = float(pred.max())
alert    = max_pred > threshold

# ── METRICS ROW ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model",        model_name)
c2.metric("Peak CPU (pred)", f"{max_pred:.1%}")
c3.metric("Alert threshold", f"{threshold:.0%}")
c4.metric("🚨 Scale-out needed", "YES" if alert else "NO",
          delta="⚠️ Now" if alert else "✅ Stable",
          delta_color="inverse" if alert else "normal")

# ── ALERT BANNER ─────────────────────────────────────────────────────
if alert:
    st.markdown(f'''<div class="alert-red">
    🚨 <b>SCALE-OUT ALERT</b> — Predicted CPU peak <b>{max_pred:.1%}</b> exceeds threshold <b>{threshold:.0%}</b><br>
    → Kubernetes HPA would trigger +2 replicas in next 30 min window
    </div>''', unsafe_allow_html=True)
else:
    st.markdown('''<div class="alert-green">
    ✅ <b>Cluster STABLE</b> — No scale-out needed in next 30 min
    </div>''', unsafe_allow_html=True)

st.markdown("---")

# ── MAIN CHART ───────────────────────────────────────────────────────
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📈 CPU Forecast")
    hist_idx = list(range(INPUT_SIZE))
    pred_idx = list(range(INPUT_SIZE, INPUT_SIZE + HORIZON))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_idx, y=window.tolist(),
        name="History (4h)", line=dict(color="#89b4fa", width=2)))
    if len(future) == HORIZON:
        fig.add_trace(go.Scatter(x=pred_idx, y=future.tolist(),
            name="Actual", line=dict(color="#a6e3a1", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=pred_idx, y=pred.tolist(),
        name=f"Forecast ({model_name})", line=dict(color="#f38ba8", width=2.5)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#fab387",
                  annotation_text=f"Alert @ {threshold:.0%}")
    fig.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Time steps (5 min each)",
        yaxis_title="CPU Utilization",
        yaxis=dict(tickformat=".0%", range=[0,1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0,r=0,t=30,b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Benchmark")
    try:
        bdf = pd.read_csv(ROOT / "results" / "dl_benchmark.csv")
        fig2 = px.bar(bdf.sort_values("RMSE"), x="RMSE", y="model",
                      orientation="h", color="RMSE",
                      color_continuous_scale="RdYlGn_r",
                      template="plotly_dark", height=350)
        fig2.update_layout(margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        st.info("Run train_dl_models.py first")

# ── SCALING DECISION TABLE ───────────────────────────────────────────
st.subheader("🤖 Auto-scaling Decision Log")
steps = [f"t+{(i+1)*5}min" for i in range(HORIZON)]
df_pred = pd.DataFrame({
    "Time Step": steps,
    "Predicted CPU": [f"{v:.1%}" for v in pred],
    "Above Threshold": ["🔴 YES" if v > threshold else "🟢 NO" for v in pred],
    "Action": ["Scale OUT (+2 replicas)" if v > threshold else "Hold" for v in pred],
})
st.dataframe(df_pred, use_container_width=True, hide_index=True)

# ── ERROR METRICS ────────────────────────────────────────────────────
if len(future) == HORIZON:
    mae  = float(np.mean(np.abs(future - pred)))
    rmse = float(np.sqrt(np.mean((future - pred)**2)))
    st.subheader("📉 Live Prediction Error")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"{mae:.5f}")
    m2.metric("RMSE", f"{rmse:.5f}")
    m3.metric("Peak error", f"{float(np.max(np.abs(future-pred))):.5f}")
