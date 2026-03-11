import requests
import streamlit as st
import os
import json
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="GBP/USD · Signal Engine",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

API_URL = "http://127.0.0.1:8000"

# ── Design System ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --bg:          #0a0d12;
  --bg2:         #0e1219;
  --surface:     #131820;
  --surface2:    #1a2130;
  --border:      #1e2a3a;
  --border-soft: #182030;
  --text:        #cdd5e0;
  --text-bright: #e8eef5;
  --text-muted:  #5a7090;
  --text-dim:    #3a5070;
  --green:       #00c878;
  --green-dim:   #004d2e;
  --green-pale:  rgba(0,200,120,.08);
  --red:         #ff3d5a;
  --red-dim:     #4d0018;
  --red-pale:    rgba(255,61,90,.08);
  --amber:       #f0a500;
  --amber-pale:  rgba(240,165,0,.08);
  --blue:        #2d7dd2;
  --blue-pale:   rgba(45,125,210,.08);
  --mono:        'IBM Plex Mono', monospace;
  --sans:        'IBM Plex Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    background-image:
      radial-gradient(ellipse 80% 40% at 50% -10%, rgba(45,125,210,.06) 0%, transparent 60%),
      linear-gradient(180deg, var(--bg) 0%, #080b10 100%) !important;
    font-family: var(--sans);
    color: var(--text);
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
.block-container          { padding-top: 2rem !important; max-width: 860px; }

/* Scanline texture */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent 0px,
        transparent 3px,
        rgba(0,0,0,.04) 3px,
        rgba(0,0,0,.04) 4px
    );
}

/* ── Typography ── */
.terminal-header {
    font-family: var(--mono);
    font-size: .68rem;
    font-weight: 500;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.section-title {
    font-family: var(--mono);
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--text-muted);
    border-left: 2px solid var(--border);
    padding-left: 10px;
    margin: 0 0 14px 0;
}

/* ── Header banner ── */
.app-header {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 2px solid var(--blue);
    border-radius: 4px;
    padding: 18px 22px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.app-title {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-bright);
    letter-spacing: .04em;
}
.app-subtitle {
    font-family: var(--mono);
    font-size: .72rem;
    color: var(--text-muted);
    margin-top: 3px;
}
/* ── Intro card ── */
.intro-card {
    background: linear-gradient(135deg, rgba(45,125,210,.13) 0%, rgba(19,24,32,.98) 60%, rgba(0,200,120,.06) 100%);
    border: 1px solid var(--border);
    border-top: 1px solid rgba(45,125,210,.45);
    border-radius: 6px;
    padding: 22px 22px 18px 22px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.intro-card::after {
    content: 'GBP/USD';
    position: absolute;
    right: -4px; top: -8px;
    font-family: var(--mono);
    font-size: 4.5rem;
    font-weight: 700;
    color: rgba(45,125,210,.055);
    letter-spacing: -.04em;
    pointer-events: none;
    user-select: none;
}
.intro-eyebrow {
    font-family: var(--mono);
    font-size: .62rem;
    font-weight: 600;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.intro-eyebrow::before {
    content: '';
    display: inline-block;
    width: 18px; height: 1.5px;
    background: var(--blue);
}
.intro-text {
    font-size: .93rem;
    line-height: 1.6;
    color: var(--text);
    font-family: var(--sans);
}
.intro-text strong { color: var(--text-bright); font-weight: 600; }
.intro-threshold-row {
    margin-top: 14px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.intro-threshold {
    font-family: var(--mono);
    font-size: .68rem;
    font-weight: 600;
    letter-spacing: .06em;
    padding: 5px 12px;
    border-radius: 2px;
}
.intro-threshold-long  { background: var(--green-dim); color: var(--green); border: 1px solid rgba(0,200,120,.3); }
.intro-threshold-short { background: var(--red-dim);   color: var(--red);   border: 1px solid rgba(255,61,90,.3); }
.intro-threshold-flat  { background: var(--amber-pale);color: var(--amber); border: 1px solid rgba(240,165,0,.3); }
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse-live 2s infinite;
    margin-right: 6px;
    vertical-align: middle;
}
.live-tag {
    font-family: var(--mono);
    font-size: .65rem;
    font-weight: 600;
    letter-spacing: .12em;
    color: var(--green);
}
@keyframes pulse-live {
    0%,100% { opacity:1; box-shadow:0 0 8px var(--green); }
    50%      { opacity:.5; box-shadow:0 0 3px var(--green); }
}

/* ── Status cards ── */
.status-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px 18px;
    position: relative;
    overflow: hidden;
}
.status-card::before {
    content: '';
    position: absolute; top: 0; left: 0; bottom: 0; width: 2px;
}
.status-card.green::before  { background: var(--green); }
.status-card.red::before    { background: var(--red); }
.status-card.amber::before  { background: var(--amber); }
.status-card.blue::before   { background: var(--blue); }
.status-card.neutral::before { background: var(--text-dim); }

.metric-label {
    font-family: var(--mono);
    font-size: .65rem;
    font-weight: 500;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.metric-value {
    font-family: var(--mono);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-bright);
    letter-spacing: .02em;
}

/* ── Signal display ── */
.signal-wrap { margin: 8px 0 0 0; }
.signal-box {
    border-radius: 4px 4px 0 0;
    padding: 30px 24px 26px 24px;
    text-align: center;
}
.signal-box-long  { background: var(--green-pale); border: 1px solid var(--green); border-bottom: none; }
.signal-box-short { background: var(--red-pale);   border: 1px solid var(--red);   border-bottom: none; }
.signal-box-flat  { background: var(--amber-pale); border: 1px solid var(--amber); border-bottom: none; }
.signal-meta-bar {
    border-radius: 0 0 4px 4px;
    padding: 10px 20px;
    font-family: var(--mono);
    font-size: .68rem;
    letter-spacing: .06em;
    color: var(--text-muted);
    display: flex;
    gap: 18px;
    justify-content: center;
    flex-wrap: wrap;
}
.signal-meta-bar-long  { background: rgba(0,200,120,.05);  border: 1px solid var(--green); border-top: 1px solid rgba(0,200,120,.2); }
.signal-meta-bar-short { background: rgba(255,61,90,.05);  border: 1px solid var(--red);   border-top: 1px solid rgba(255,61,90,.2); }
.signal-meta-bar-flat  { background: rgba(240,165,0,.05);  border: 1px solid var(--amber); border-top: 1px solid rgba(240,165,0,.2); }
.signal-meta-item { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.signal-meta-key  { font-size: .58rem; letter-spacing: .14em; text-transform: uppercase; color: var(--text-dim); }
.signal-meta-val  { font-size: .74rem; font-weight: 600; color: var(--text-muted); }
.signal-label {
    font-family: var(--mono);
    font-size: .62rem;
    font-weight: 600;
    letter-spacing: .22em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.signal-action {
    font-family: var(--mono);
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: .08em;
    line-height: 1;
}
.signal-desc {
    font-family: var(--sans);
    font-size: .82rem;
    font-weight: 400;
    margin-top: 8px;
    opacity: .7;
}

/* ── Divider ── */
.h-rule {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
}
.h-rule-accent {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, var(--blue), transparent);
    margin: 24px 0;
}

/* ── Signal explanation ── */
.signal-explain {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 14px 20px;
    margin-bottom: 8px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
}
.explain-item {
    padding: 6px 12px;
    border-right: 1px solid var(--border-soft);
}
.explain-item:last-child { border-right: none; }
.explain-key {
    font-family: var(--mono);
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 4px;
}
.explain-val {
    font-family: var(--sans);
    font-size: .8rem;
    line-height: 1.45;
    color: var(--text-muted);
}
.explain-val b {
    color: var(--text);
    font-weight: 600;
}

/* ── Legend table ── */
.legend-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-soft);
    font-family: var(--sans);
    font-size: .88rem;
}
.legend-row:last-child { border-bottom: none; }
.legend-badge {
    font-family: var(--mono);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .1em;
    padding: 3px 10px;
    border-radius: 2px;
    min-width: 58px;
    text-align: center;
    flex-shrink: 0;
}
.badge-long  { background: var(--green-dim); color: var(--green); border: 1px solid var(--green); }
.badge-short { background: var(--red-dim);   color: var(--red);   border: 1px solid var(--red);   }
.badge-flat  { background: var(--amber-pale);color: var(--amber); border: 1px solid var(--amber); }

/* ── Data tables ── */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* ── Streamlit overrides ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] p {
    font-family: var(--mono) !important;
    font-size: .65rem !important;
    font-weight: 600 !important;
    letter-spacing: .14em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    color: var(--text-bright) !important;
}
.stButton > button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: .8rem !important;
    font-weight: 600 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    transition: all .15s !important;
}
.stButton > button:hover {
    background: var(--surface) !important;
    border-color: var(--blue) !important;
    color: var(--text-bright) !important;
    box-shadow: 0 0 16px rgba(45,125,210,.15) !important;
}
.stAlert {
    border-radius: 3px !important;
    font-family: var(--mono) !important;
    font-size: .85rem !important;
}
p, li, div { color: var(--text); font-family: var(--sans); }
h1,h2,h3,h4 { color: var(--text-bright) !important; font-family: var(--mono) !important; }
[data-testid="stMarkdown"] h4 {
    font-family: var(--mono) !important;
    font-size: .78rem !important;
    letter-spacing: .14em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
}
thead tr th {
    background: var(--surface2) !important;
    color: var(--text-muted) !important;
    font-family: var(--mono) !important;
    font-size: .7rem !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    border-color: var(--border) !important;
}
tbody tr td {
    font-family: var(--mono) !important;
    font-size: .82rem !important;
    color: var(--text) !important;
    border-color: var(--border-soft) !important;
}
tbody tr:hover td { background: var(--surface2) !important; }
[data-testid="stSuccess"], [data-testid="stInfo"],
[data-testid="stWarning"], [data-testid="stError"] {
    border-radius: 3px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_json(url: str):
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    return r.json()

def json_to_table(d: dict) -> pd.DataFrame:
    rows = []
    for k, v in d.items():
        rows.append((k, round(v, 6) if isinstance(v, float) else v))
    return pd.DataFrame(rows, columns=["Metric", "Value"])

def read_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

# ── App Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
  <div>
    <div class='app-title'>GBP / USD &nbsp;·&nbsp; Signal Engine</div>
    <div class='app-subtitle'>ML + RL Decision System &nbsp;·&nbsp; 15-min candles</div>
  </div>
  <div>
    <span class='live-dot'></span>
    <span class='live-tag'>LIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Intro card ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='intro-card'>
  <div class='intro-eyebrow'>Signal Engine · What this does</div>
  <div class='intro-text'>
    Predicts the <strong>direction of the next GBP/USD 15-minute candle</strong> using
    an ensemble of <strong>ML classification</strong> and <strong>RL (PPO) policy</strong> models —
    trained on 2024 tick data. The target is whether the <strong>next return is positive or negative</strong>.
  </div>
  <div class='intro-threshold-row'>
    <span class='intro-threshold intro-threshold-long'>▲ LONG &nbsp;· P(up) &gt; 0.55</span>
    <span class='intro-threshold intro-threshold-short'>▼ SHORT &nbsp;· P(up) &lt; 0.45</span>
    <span class='intro-threshold intro-threshold-flat'>◆ FLAT &nbsp;· 0.45 ≤ P ≤ 0.55</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── API Health ─────────────────────────────────────────────────────────────────
try:
    health = get_json(f"{API_URL}/health")
    st.markdown("""
    <div class='status-card green' style='margin-bottom:16px;'>
      <div class='metric-label'>System Status</div>
      <div class='metric-value' style='color:var(--green); font-size:.9rem;'>
        ● API CONNECTED &nbsp;·&nbsp; READY
      </div>
    </div>""", unsafe_allow_html=True)
except Exception as e:
    st.markdown(f"""
    <div class='status-card red' style='margin-bottom:16px;'>
      <div class='metric-label'>System Status</div>
      <div class='metric-value' style='color:var(--red); font-size:.9rem;'>✕ API OFFLINE</div>
      <div style='font-family:var(--mono); font-size:.72rem; color:var(--text-muted); margin-top:6px;'>{e}</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Model Info ─────────────────────────────────────────────────────────────────
try:
    info = get_json(f"{API_URL}/model_version")
    st.markdown("<div class='h-rule-accent'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model Registry</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='status-card blue'>
          <div class='metric-label'>Active Model</div>
          <div class='metric-value'>{info.get("model_type","—").upper()}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='status-card blue'>
          <div class='metric-label'>Feature Dimension</div>
          <div class='metric-value'>{info.get("n_features","—")}</div>
        </div>""", unsafe_allow_html=True)
except Exception:
    pass

# ── Signal Decision ────────────────────────────────────────────────────────────
st.markdown("<div class='h-rule-accent'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Live Signal · Latest Candle</div>", unsafe_allow_html=True)

if st.button("▶  Execute Decision Query"):
    try:
        res   = get_json(f"{API_URL}/decision/latest")
        action= res.get("action", "?").upper()
        ts    = res.get("timestamp", "unknown")
        price = res.get("price", None)
        score = res.get("score", None)

        if action == "LONG":
            price_str = f"{price:.5f}" if price else "—"
            score_str = f"{score:.4f}" if score is not None else "—"
            st.markdown(f"""
            <div class='signal-wrap'>
              <div class='signal-box signal-box-long'>
                <div class='signal-label' style='color:var(--green);'>▸ Signal Output</div>
                <div class='signal-action' style='color:var(--green);'>▲ LONG</div>
                <div class='signal-desc' style='color:var(--green);'>Bullish — Open buy position</div>
              </div>
              <div class='signal-meta-bar signal-meta-bar-long'>
                <div class='signal-meta-item'>
                  <span class='signal-meta-key'>Candle close</span>
                  <span class='signal-meta-val' style='color:var(--green);'>{price_str}</span>
                </div>
                <div class='signal-meta-item'>
                  <span class='signal-meta-key'>Score</span>
                  <span class='signal-meta-val' style='color:var(--green);'>{score_str}</span>
                </div>
                <div class='signal-meta-item'>
                  <span class='signal-meta-key'>Timestamp</span>
                  <span class='signal-meta-val'>{ts}</span>
                </div>
              </div>
              <div class='signal-explain'>
                <div class='explain-item'>
                  <div class='explain-key'>Candle Close</div>
                  <div class='explain-val'>The <b>closing price</b> of the 15-min GBP/USD candle used as the model's last input feature.</div>
                </div>
                <div class='explain-item'>
                  <div class='explain-key'>Score</div>
                  <div class='explain-val'>Model's <b>predicted probability</b> of an upward move. Values &gt; 0.55 trigger a LONG signal.</div>
                </div>
                <div class='explain-item'>
                  <div class='explain-key'>Timestamp</div>
                  <div class='explain-val'>UTC open time of the <b>candle that generated</b> this signal. Next candle is the trade window.</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        elif action == "SHORT":
            price_str = f"{price:.5f}" if price else "—"
            score_str = f"{score:.4f}" if score is not None else "—"
            st.markdown(f"""
            <div class='signal-wrap'>
              <div class='signal-box signal-box-short'>
                <div class='signal-label' style='color:var(--red);'>▸ Signal Output</div>
                <div class='signal-action' style='color:var(--red);'>▼ SHORT</div>
                <div class='signal-desc' style='color:var(--red);'>Bearish — Open sell position</div>
              </div>
              <div class='signal-meta-bar signal-meta-bar-short'>
                <div class='signal-meta-item'>
                  <span class='signal-meta-key'>Candle close</span>
                  <span class='signal-meta-val' style='color:var(--red);'>{price_str}</span>
                </div>
                <div class='signal-meta-item'>
                  <span class='signal-meta-key'>Score</span>
                  <span class='signal-meta-val' style='color:var(--red);'>{score_str}</span>
                </div>
                <div class='signal-meta-item'>
                  <span class='signal-meta-key'>Timestamp</span>
                  <span class='signal-meta-val'>{ts}</span>
                </div>
              </div>
              <div class='signal-explain'>
                <div class='explain-item'>
                  <div class='explain-key'>Candle Close</div>
                  <div class='explain-val'>The <b>closing price</b> of the 15-min GBP/USD candle used as the model's last input feature.</div>
                </div>
                <div class='explain-item'>
                  <div class='explain-key'>Score</div>
                  <div class='explain-val'>Model's <b>predicted probability</b> of an upward move. Values &lt; 0.45 trigger a SHORT signal.</div>
                </div>
                <div class='explain-item'>
                  <div class='explain-key'>Timestamp</div>
                  <div class='explain-val'>UTC open time of the <b>candle that generated</b> this signal. Next candle is the trade window.</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        else:
            price_str = f"{price:.5f}" if price else "—"
            score_str = f"{score:.4f}" if score is not None else "—"
            st.markdown(f"""
            <div class='signal-wrap'>
              <div class='signal-box signal-box-flat'>
                <div class='signal-label' style='color:var(--amber);'>▸ Signal Output</div>
                <div class='signal-action' style='color:var(--amber);'>◆ FLAT</div>
                <div class='signal-desc' style='color:var(--amber);'>Neutral — No position</div>
              </div>
              <div class='signal-meta-bar signal-meta-bar-flat'>
                <div class='signal-meta-item'>
                  <span class='signal-meta-key'>Candle close</span>
                  <span class='signal-meta-val' style='color:var(--amber);'>{price_str}</span>
                </div>
                <div class='signal-meta-item'>
                  <span class='signal-meta-key'>Score</span>
                  <span class='signal-meta-val' style='color:var(--amber);'>{score_str}</span>
                </div>
                <div class='signal-meta-item'>
                  <span class='signal-meta-key'>Timestamp</span>
                  <span class='signal-meta-val'>{ts}</span>
                </div>
              </div>
              <div class='signal-explain'>
                <div class='explain-item'>
                  <div class='explain-key'>Candle Close</div>
                  <div class='explain-val'>The <b>closing price</b> of the 15-min GBP/USD candle used as the model's last input feature.</div>
                </div>
                <div class='explain-item'>
                  <div class='explain-key'>Score</div>
                  <div class='explain-val'>Model's <b>predicted probability</b> of an upward move. Score between 0.45–0.55 = no conviction, stay flat.</div>
                </div>
                <div class='explain-item'>
                  <div class='explain-key'>Timestamp</div>
                  <div class='explain-val'>UTC open time of the <b>candle that generated</b> this signal. Next candle is the trade window.</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <div class='status-card red' style='margin-top:8px;'>
          <div class='metric-label'>Query Error</div>
          <div style='font-family:var(--mono); font-size:.8rem; color:var(--red); margin-top:4px;'>{e}</div>
        </div>""", unsafe_allow_html=True)

# ── Signal Legend ──────────────────────────────────────────────────────────────
st.markdown("<div class='h-rule-accent'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Signal Definitions</div>", unsafe_allow_html=True)
st.markdown("""
<div style='background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:14px 18px;'>
  <div class='legend-row'>
    <span class='legend-badge badge-long'>LONG</span>
    <span style='color:var(--text); font-size:.88rem;'>
      Bullish signal — model predicts upward price movement.
      <span style='color:var(--text-muted);'> Enter a buy position; profit if price rises.</span>
    </span>
  </div>
  <div class='legend-row'>
    <span class='legend-badge badge-short'>SHORT</span>
    <span style='color:var(--text); font-size:.88rem;'>
      Bearish signal — model predicts downward price movement.
      <span style='color:var(--text-muted);'> Enter a sell position; profit if price falls.</span>
    </span>
  </div>
  <div class='legend-row'>
    <span class='legend-badge badge-flat'>FLAT</span>
    <span style='color:var(--text); font-size:.88rem;'>
      Neutral signal — no directional conviction.
      <span style='color:var(--text-muted);'> Stay in cash; avoid exposure.</span>
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Reports ────────────────────────────────────────────────────────────────────
st.markdown("<div class='h-rule-accent'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Backtest Reports · FY 2024</div>", unsafe_allow_html=True)

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(Path(__file__).resolve().parents[1] / "reports")))
ml_stats_path = REPORTS_DIR / "ml_2024_stats.json"
ml_fin_path   = REPORTS_DIR / "ml_2024_finance.json"
rl_fin_path   = REPORTS_DIR / "rl_2024_finance.json"
equity_img    = REPORTS_DIR / "equity_2024_baselines_vs_ml_vs_rl.png"

# ── Metric metadata: label, icon, format, color logic ─────────────────────────
METRIC_META = {
    # Classification stats
    "accuracy":           ("Accuracy",            "◎", "pct",    "neutral_high"),
    "precision":          ("Precision",            "◎", "pct",    "neutral_high"),
    "recall":             ("Recall",               "◎", "pct",    "neutral_high"),
    "f1":                 ("F1 Score",             "◎", "pct",    "neutral_high"),
    "f1_score":           ("F1 Score",             "◎", "pct",    "neutral_high"),
    "auc":                ("AUC-ROC",              "◎", "pct",    "neutral_high"),
    "roc_auc":            ("AUC-ROC",              "◎", "pct",    "neutral_high"),
    "n_samples":          ("Samples",              "≡", "int",    "dim"),
    "n_features":         ("Features",             "≡", "int",    "dim"),
    # Financial metrics
    "total_return":       ("Total Return",         "▲", "pct",    "pnl"),
    "annual_return":      ("Annual Return",        "▲", "pct",    "pnl"),
    "annualized_return":  ("Annualized Return",    "▲", "pct",    "pnl"),
    "sharpe_ratio":       ("Sharpe Ratio",         "◈", "float2", "sharpe"),
    "sharpe":             ("Sharpe Ratio",         "◈", "float2", "sharpe"),
    "sortino_ratio":      ("Sortino Ratio",        "◈", "float2", "sharpe"),
    "calmar_ratio":       ("Calmar Ratio",         "◈", "float2", "sharpe"),
    "max_drawdown":       ("Max Drawdown",         "▼", "pct",    "drawdown"),
    "max_drawdown_pct":   ("Max Drawdown %",       "▼", "pct",    "drawdown"),
    "win_rate":           ("Win Rate",             "◎", "pct",    "neutral_high"),
    "profit_factor":      ("Profit Factor",        "◈", "float2", "sharpe"),
    "n_trades":           ("Total Trades",         "≡", "int",    "dim"),
    "avg_trade":          ("Avg Trade P&L",        "▲", "pct",    "pnl"),
    "avg_trade_return":   ("Avg Trade Return",     "▲", "pct",    "pnl"),
    "volatility":         ("Volatility",           "~", "pct",    "vol"),
    "annual_volatility":  ("Annual Volatility",    "~", "pct",    "vol"),
    "final_equity":       ("Final Equity",         "$", "currency","pnl"),
}

def fmt_value(key: str, val) -> tuple[str, str, str, str]:
    """Returns (label, icon, formatted_value, color_class)"""
    key_lower = str(key).lower().replace(" ", "_")
    meta = METRIC_META.get(key_lower, None)
    if meta:
        label, icon, fmt, color_logic = meta
    else:
        label = str(key).replace("_", " ").title()
        icon  = "·"
        fmt   = "float4" if isinstance(val, float) else "int"
        color_logic = "dim"

    # Format value
    try:
        v = float(val)
        if fmt == "pct":
            # detect if already in percent (>1) or fraction
            display = f"{v*100:.2f}%" if abs(v) <= 1.5 else f"{v:.2f}%"
        elif fmt == "float2":
            display = f"{v:.2f}"
        elif fmt == "float4":
            display = f"{v:.4f}"
        elif fmt == "int":
            display = f"{int(v):,}"
        elif fmt == "currency":
            display = f"${v:,.2f}"
        else:
            display = str(val)
    except (ValueError, TypeError):
        display = str(val)
        v = 0

    # Color
    if color_logic == "pnl":
        css = "val-positive" if v > 0 else ("val-negative" if v < 0 else "val-neutral")
    elif color_logic == "drawdown":
        css = "val-negative" if v < 0 else "val-neutral"
    elif color_logic == "neutral_high":
        css = "val-positive" if v >= 0.55 else ("val-neutral" if v >= 0.45 else "val-negative")
    elif color_logic == "sharpe":
        css = "val-positive" if v >= 1.0 else ("val-neutral" if v >= 0 else "val-negative")
    elif color_logic == "vol":
        css = "val-warn"
    elif color_logic == "dim":
        css = "val-dim"
    else:
        css = "val-neutral"

    return label, icon, display, css

def render_report_table(data: dict, title: str, subtitle: str, accent_color: str) -> str:
    rows_html = ""
    for i, (k, v) in enumerate(data.items()):
        label, icon, display, css = fmt_value(k, v)
        stripe = "row-stripe" if i % 2 == 0 else ""
        rows_html += f"""
        <tr class='{stripe}'>
          <td class='td-icon' style='color:{accent_color};'>{icon}</td>
          <td class='td-metric'>
            <span class='td-metric-name'>{label}</span>
            <span class='td-key-inline'>{k}</span>
          </td>
          <td class='td-value {css}'>{display}</td>
        </tr>"""

    return f"""
    <div class='report-card'>
      <div class='report-header' style='border-left-color:{accent_color};'>
        <div class='report-title' style='color:{accent_color};'>{title}</div>
        <div class='report-subtitle'>{subtitle}</div>
      </div>
      <table class='report-table'>
        <thead>
          <tr>
            <th class='th-icon'></th>
            <th class='th-metric'>Metric</th>
            <th class='th-value'>Value</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

def render_missing_card(filename: str) -> str:
    return f"""
    <div class='report-card report-missing'>
      <div style='text-align:center; padding:24px 0;'>
        <div style='font-family:var(--mono); font-size:.65rem; letter-spacing:.16em;
                    color:var(--text-dim); text-transform:uppercase; margin-bottom:8px;'>File not found</div>
        <div style='font-family:var(--mono); font-size:.78rem; color:var(--text-dim);'>{filename}</div>
      </div>
    </div>"""

# Inject table CSS
st.markdown("""
<style>
/* ── Report cards ── */
.report-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 18px;
}
.report-missing { border-style: dashed; }
.report-header {
    padding: 12px 16px 10px 14px;
    border-left: 3px solid;
    background: linear-gradient(90deg, rgba(255,255,255,.025) 0%, transparent 100%);
    border-bottom: 1px solid var(--border);
}
.report-title {
    font-family: var(--mono);
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
}
.report-subtitle {
    font-family: var(--mono);
    font-size: .65rem;
    color: var(--text-dim);
    margin-top: 2px;
    letter-spacing: .06em;
}
.report-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    table-layout: fixed;
}
.report-table thead tr {
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
}
.th-icon   { width: 36px; padding: 7px 0 7px 14px; }
.th-metric { padding: 7px 12px; font-size:.62rem; font-weight:700; letter-spacing:.14em;
             text-transform:uppercase; color:var(--text-dim); width: auto; }
.th-value  { padding: 7px 16px 7px 12px; font-size:.62rem; font-weight:700; letter-spacing:.14em;
             text-transform:uppercase; color:var(--text-dim); text-align:right; width: 120px; }

.report-table tbody tr {
    border-bottom: 1px solid var(--border-soft);
    transition: background .1s;
}
.report-table tbody tr:last-child { border-bottom: none; }
.report-table tbody tr:hover { background: var(--surface2); }
.row-stripe { background: rgba(255,255,255,.018); }

.td-icon   { padding: 10px 4px 10px 14px; font-size:.82rem; width:36px; vertical-align:middle; }
.td-metric {
    padding: 10px 12px;
    vertical-align: middle;
    overflow: hidden;
}
.td-metric-name {
    font-size: .82rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: .02em;
    display: block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.td-key-inline {
    display: block;
    font-size: .64rem;
    color: var(--text-dim);
    letter-spacing: .06em;
    margin-top: 1px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.td-value {
    padding: 10px 16px 10px 12px;
    font-size: .88rem;
    font-weight: 700;
    letter-spacing: .03em;
    text-align: right;
    white-space: nowrap;
    vertical-align: middle;
    width: 120px;
}

/* Value color classes */
.val-positive { color: var(--green); }
.val-negative { color: var(--red); }
.val-neutral  { color: var(--text); }
.val-warn     { color: var(--amber); }
.val-dim      { color: var(--text-muted); }

/* ── Equity chart wrapper ── */
.equity-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin-top: 4px;
}
.equity-header {
    padding: 11px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface2);
}
.equity-title {
    font-family: var(--mono);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.equity-tag {
    font-family: var(--mono);
    font-size: .62rem;
    color: var(--text-dim);
    letter-spacing: .1em;
}
.equity-body { padding: 16px; }
</style>
""", unsafe_allow_html=True)

# Row 1: RL (PPO) Finance — full width
if rl_fin_path.exists():
    d = read_json_file(rl_fin_path)
    st.markdown(render_report_table(
        d,
        title        = "🤖 RL (PPO) · Financial Performance",
        subtitle     = "rl_2024_finance.json · PPO policy backtest",
        accent_color = "#a855f7"
    ), unsafe_allow_html=True)
else:
    st.markdown(render_missing_card(rl_fin_path.name), unsafe_allow_html=True)

# Row 2: ML Finance — full width
if ml_fin_path.exists():
    d = read_json_file(ml_fin_path)
    st.markdown(render_report_table(
        d,
        title        = "💹 ML · Financial Performance",
        subtitle     = "ml_2024_finance.json · backtest results",
        accent_color = "#00c878"
    ), unsafe_allow_html=True)
else:
    st.markdown(render_missing_card(ml_fin_path.name), unsafe_allow_html=True)

# Row 3: ML Classification Stats — full width
if ml_stats_path.exists():
    d = read_json_file(ml_stats_path)
    st.markdown(render_report_table(
        d,
        title        = "🧠 ML · Classification Stats",
        subtitle     = "ml_2024_stats.json · sklearn metrics",
        accent_color = "#2d7dd2"
    ), unsafe_allow_html=True)
else:
    st.markdown(render_missing_card(ml_stats_path.name), unsafe_allow_html=True)

# ── Equity chart ──────────────────────────────────────────────────────────────
st.markdown("<div class='h-rule-accent'></div>", unsafe_allow_html=True)
if equity_img.exists():
    st.markdown("""
    <div class='equity-card'>
      <div class='equity-header'>
        <span class='equity-title'>Equity Curves · Baselines vs ML vs RL</span>
        <span class='equity-tag'>FY 2024 · GBP/USD · 15-min</span>
      </div>
      <div class='equity-body'>""", unsafe_allow_html=True)
    st.image(str(equity_img), use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class='equity-card'>
      <div class='equity-header'>
        <span class='equity-title'>Equity Curves · Baselines vs ML vs RL</span>
        <span class='equity-tag'>FY 2024</span>
      </div>
      <div style='padding:30px; text-align:center;'>
        <div style='font-family:var(--mono); font-size:.65rem; letter-spacing:.16em;
                    color:var(--text-dim); text-transform:uppercase; margin-bottom:8px;'>Chart unavailable</div>
        <div style='font-family:var(--mono); font-size:.78rem; color:var(--text-dim);'>{equity_img.name}</div>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='h-rule'></div>
<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>
  <span style='font-family:var(--mono); font-size:.65rem; color:var(--text-dim); letter-spacing:.1em;'>
    GBP/USD SIGNAL ENGINE · ML + RL SYSTEM
  </span>
  <span style='font-family:var(--mono); font-size:.65rem; color:var(--text-dim); letter-spacing:.08em;'>
    NOT FINANCIAL ADVICE · FOR RESEARCH USE ONLY
  </span>
</div>
""", unsafe_allow_html=True)