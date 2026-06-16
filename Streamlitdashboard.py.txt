"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Objective-Aligned Hybrid BERT-BiLSTM  |  Streamlit Dashboard + Colab      ║
║   Social Media Usage, Emotional Sensitivity & Behavior Impact Model          ║
╚══════════════════════════════════════════════════════════════════════════════╝

GOOGLE COLAB SETUP  (run this cell first):
──────────────────────────────────────────
    !pip install -q streamlit transformers torch scikit-learn plotly pyngrok

    # Paste and run this whole file, then in a new cell:
    !streamlit run bert_bilstm_dashboard.py &
    from pyngrok import ngrok
    public_url = ngrok.connect(8501)
    print("Dashboard URL:", public_url)

LOCAL SETUP:
────────────
    pip install streamlit transformers torch scikit-learn plotly pandas numpy
    streamlit run bert_bilstm_dashboard.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
# ── Cell 1: Install ──────────────────────────────────────────────
%%writefile app.py
!npm install -g localtunnel -q

import io, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st




import streamlit as st
st.title("My Dashboard")
import subprocess, time

subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501"])
time.sleep(3)

# Get public URL via localtunnel
result = subprocess.run(["lt", "--port", "8501"], capture_output=True, text=True, timeout=10)
print(result.stdout)








# ── Cell 2: Write your app ───────────────────────────────────────

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("My Dashboard")
# ... rest of your code here

# ── Cell 3: Launch with tunnel ───────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoB · BERT-BiLSTM Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #162032 100%);
    border: 1px solid #2d4060;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin: 4px;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #64b5f6;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #90a4ae;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.obj-badge {
    display: inline-block;
    background: #0d47a1;
    color: #bbdefb;
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e3f2fd;
    margin: 0 0 4px 0;
}
.drift-alert {
    background: linear-gradient(90deg, #7f0000 0%, #b71c1c 100%);
    border-left: 4px solid #ef5350;
    border-radius: 6px;
    padding: 10px 16px;
    color: #ffcdd2;
    font-size: 0.87rem;
    margin: 8px 0;
}
.drift-ok {
    background: linear-gradient(90deg, #1b5e20 0%, #2e7d32 100%);
    border-left: 4px solid #66bb6a;
    border-radius: 6px;
    padding: 10px 16px;
    color: #c8e6c9;
    font-size: 0.87rem;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    seq_len            = 15
    bert_model_name    = "bert-base-uncased"
    bert_max_len       = 32
    bert_hidden        = 768
    numeric_proj_dim   = 32
    lstm_hidden        = 128
    lstm_layers        = 2
    bidirectional      = True
    n_clusters         = 5
    n_emotions         = 4
    dropout            = 0.3
    batch_size         = 16
    bert_lr            = 2e-5
    head_lr            = 1e-3
    epochs             = 5
    device             = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_bert_layers = 8

CFG = Config()

USAGE_FEATURES = [
    "daily_minutes","sessions","night_use_ratio","platform_switches",
    "video_ratio","messaging_ratio","story_post_ratio",
    "posts","likes_given","comments_made","likes_received","comments_received",
    "usage_intensity","validation_seeking_ratio","social_interaction_ratio",
    "usage_delta","dow_sin","dow_cos",
    "plat_Facebook","plat_Instagram","plat_Telegram","plat_WhatsApp","plat_YouTube",
]
PERSONALITY_FEATURES = [
    "personality_openness","personality_conscientiousness",
    "personality_extraversion","personality_agreeableness","personality_neuroticism",
]
EMOTION_HISTORY_FEATURES = [
    "sentiment_score","negative_emotion_index","emotion_volatility_3d",
    "emotion_trend_3d","neuroticism_x_usage",
]
NUMERIC_FEATURES  = USAGE_FEATURES + PERSONALITY_FEATURES + EMOTION_HISTORY_FEATURES
EMOTION_TARGETS   = ["worry","depression","anxiety","jealousy"]
SEGMENT_NAMES     = ["balanced_user","anxious_scroller","validation_seeker",
                     "passive_consumer","social_connector"]
SEGMENT_COLORS    = ["#4caf50","#ef5350","#ff9800","#9c27b0","#2196f3"]

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SYNTHETIC DATA GENERATOR  (runs when no CSV is uploaded)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n_users=120, seq_len=15, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    segment_profiles = {
        "balanced_user":      dict(daily_minutes=90,  neuroticism=0.3, neg_emotion=0.25),
        "anxious_scroller":   dict(daily_minutes=210, neuroticism=0.8, neg_emotion=0.72),
        "validation_seeker":  dict(daily_minutes=160, neuroticism=0.6, neg_emotion=0.55),
        "passive_consumer":   dict(daily_minutes=140, neuroticism=0.4, neg_emotion=0.35),
        "social_connector":   dict(daily_minutes=120, neuroticism=0.35,neg_emotion=0.30),
    }
    segs = list(segment_profiles.keys())
    for uid in range(n_users):
        seg   = segs[uid % len(segs)]
        prof  = segment_profiles[seg]
        for t in range(seq_len):
            noise = rng.normal(0, 0.05)
            neg   = np.clip(prof["neg_emotion"] + noise + t*0.003, 0, 1)
            row   = dict(
                user_id=uid, timestep=t, segment_label=seg,
                post_text=f"User {uid} day {t}: feeling {['okay','stressed','happy','anxious'][rng.integers(4)]} today",
                daily_minutes   = np.clip(prof["daily_minutes"] + rng.normal(0,20), 30, 600),
                sessions        = int(np.clip(rng.normal(5,2), 1, 20)),
                night_use_ratio = np.clip(rng.beta(2,5), 0, 1),
                platform_switches=int(rng.integers(1,8)),
                video_ratio     = np.clip(rng.beta(2,3), 0, 1),
                messaging_ratio = np.clip(rng.beta(3,2), 0, 1),
                story_post_ratio= np.clip(rng.beta(1,4), 0, 1),
                posts           = int(rng.integers(0,10)),
                likes_given     = int(rng.integers(0,50)),
                comments_made   = int(rng.integers(0,20)),
                likes_received  = int(rng.integers(0,100)),
                comments_received=int(rng.integers(0,30)),
                usage_intensity = np.clip(rng.normal(0.5,0.15),0,1),
                validation_seeking_ratio=np.clip(rng.beta(2,3),0,1),
                social_interaction_ratio=np.clip(rng.beta(3,2),0,1),
                usage_delta     = rng.normal(0,0.1),
                dow_sin         = np.sin(2*np.pi*t/7),
                dow_cos         = np.cos(2*np.pi*t/7),
                plat_Facebook   = int(rng.integers(0,2)),
                plat_Instagram  = int(rng.integers(0,2)),
                plat_Telegram   = int(rng.integers(0,2)),
                plat_WhatsApp   = int(rng.integers(0,2)),
                plat_YouTube    = int(rng.integers(0,2)),
                personality_openness        = np.clip(rng.normal(0.5,0.15),0,1),
                personality_conscientiousness=np.clip(rng.normal(0.5,0.15),0,1),
                personality_extraversion    = np.clip(rng.normal(0.5,0.15),0,1),
                personality_agreeableness   = np.clip(rng.normal(0.5,0.15),0,1),
                personality_neuroticism     = np.clip(rng.normal(prof["neuroticism"],0.1),0,1),
                sentiment_score     = 1 - neg + rng.normal(0,0.05),
                negative_emotion_index=neg,
                emotion_volatility_3d=abs(rng.normal(0,0.1)),
                emotion_trend_3d    = rng.normal(0,0.05),
                neuroticism_x_usage = prof["neuroticism"] * prof["daily_minutes"]/300,
                worry      = np.clip(neg*0.9 + rng.normal(0,0.05),0,1),
                depression = np.clip(neg*0.85+ rng.normal(0,0.05),0,1),
                anxiety    = np.clip(neg*0.95+ rng.normal(0,0.05),0,1),
                jealousy   = np.clip(neg*0.7 + rng.normal(0,0.05),0,1),
            )
            rows.append(row)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL
# ─────────────────────────────────────────────────────────────────────────────
class ObjectiveAlignedBERTBiLSTM(nn.Module):
    def __init__(self, cfg, num_numeric_features, use_bert=False):
        super().__init__()
        self.cfg      = cfg
        self.use_bert = use_bert

        if use_bert:
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(cfg.bert_model_name)
            if cfg.freeze_bert_layers > 0:
                for layer in self.bert.encoder.layer[:cfg.freeze_bert_layers]:
                    for p in layer.parameters(): p.requires_grad = False
                for p in self.bert.embeddings.parameters(): p.requires_grad = False
            text_dim = cfg.bert_hidden
        else:
            # lightweight text proxy for dashboard / Colab CPU mode
            self.text_proj = nn.Sequential(
                nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, cfg.bert_hidden))
            text_dim = cfg.bert_hidden

        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric_features, 64), nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, cfg.numeric_proj_dim), nn.ReLU(),
        )
        fusion_dim = text_dim + cfg.numeric_proj_dim
        self.lstm = nn.LSTM(
            input_size=fusion_dim, hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers, batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        repr_dim = cfg.lstm_hidden * (2 if cfg.bidirectional else 1)
        self.repr_dim = repr_dim

        self.emotion_head = nn.Sequential(
            nn.Linear(repr_dim,128), nn.ReLU(),
            nn.Dropout(cfg.dropout), nn.Linear(128, cfg.n_emotions), nn.Sigmoid())
        self.cluster_proj = nn.Sequential(
            nn.Linear(repr_dim,64), nn.ReLU(), nn.Linear(64,16))
        self.monitor_head = nn.Sequential(
            nn.Linear(repr_dim,64), nn.ReLU(), nn.Linear(64,2), nn.Sigmoid())

    def forward(self, input_ids, attention_mask, numeric, text_hash=None):
        B, T, _ = numeric.shape

        if self.use_bert:
            flat_ids  = input_ids.view(B*T, -1)
            flat_mask = attention_mask.view(B*T, -1)
            cls_emb   = self.bert(flat_ids, flat_mask).last_hidden_state[:,0,:].view(B,T,self.cfg.bert_hidden)
        else:
            # text_hash: (B, T, 64) random-ish numeric proxy
            if text_hash is None:
                text_hash = torch.randn(B, T, 64, device=numeric.device)
            cls_emb = self.text_proj(text_hash)

        numeric_emb = self.numeric_proj(numeric)
        fused       = torch.cat([cls_emb, numeric_emb], dim=-1)
        lstm_out, _ = self.lstm(fused)
        shared_repr = lstm_out[:, -1, :]

        return dict(
            shared_repr      = shared_repr,
            emotion_pred     = self.emotion_head(shared_repr),
            cluster_embedding= self.cluster_proj(shared_repr),
            monitor_pred     = self.monitor_head(shared_repr),
        )

# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class IoBSequenceDataset(Dataset):
    def __init__(self, df, scaler, label_encoder, seq_len):
        self.seq_len = seq_len
        self.le      = label_encoder
        df = df.copy().sort_values(["user_id","timestep"])
        self.users = df["user_id"].unique()
        scaled = scaler.transform(df[NUMERIC_FEATURES].values)
        df[[f+"_sc" for f in NUMERIC_FEATURES]] = scaled
        self.df = df.set_index("user_id")

    def __len__(self): return len(self.users)

    def __getitem__(self, idx):
        uid     = self.users[idx]
        udf     = self.df.loc[[uid]].sort_values("timestep").iloc[:self.seq_len]
        numeric = udf[[f+"_sc" for f in NUMERIC_FEATURES]].values.astype(np.float32)
        emotions= udf[EMOTION_TARGETS].values.astype(np.float32)
        seg_idx = self.le.transform([udf["segment_label"].iloc[-1]])[0]
        return dict(
            numeric        = torch.tensor(numeric),
            emotion_targets= torch.tensor(emotions[-1]),
            emotion_history= torch.tensor(emotions[:-1].mean(0)),
            segment_label  = torch.tensor(seg_idx, dtype=torch.long),
        )

# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING / INFERENCE (cached so reruns don't retrain)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_pipeline(df_hash, n_epochs, n_clusters, use_bert):
    """Full pipeline: preprocess → train → cluster → return all artefacts."""
    df = st.session_state["df"]

    le     = LabelEncoder()
    le.fit(df["segment_label"])
    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_FEATURES].values)

    user_ids = df["user_id"].unique()
    train_ids, val_ids = train_test_split(user_ids, test_size=0.2, random_state=42)

    train_ds = IoBSequenceDataset(df[df.user_id.isin(train_ids)], scaler, le, CFG.seq_len)
    val_ds   = IoBSequenceDataset(df[df.user_id.isin(val_ids)],   scaler, le, CFG.seq_len)
    train_ld = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=CFG.batch_size, shuffle=False)

    model = ObjectiveAlignedBERTBiLSTM(CFG, len(NUMERIC_FEATURES), use_bert=use_bert).to(CFG.device)
    bert_p = [p for n,p in model.named_parameters() if "bert" in n and p.requires_grad]
    head_p = [p for n,p in model.named_parameters() if "bert" not in n]
    opt    = torch.optim.AdamW([
        {"params": bert_p, "lr": CFG.bert_lr},
        {"params": head_p, "lr": CFG.head_lr},
    ])
    crit = nn.MSELoss()

    history = {"epoch":[], "train_loss":[], "val_mae":[]}

    for ep in range(n_epochs):
        model.train(); total = 0.0
        for b in train_ld:
            num  = b["numeric"].to(CFG.device)
            etgt = b["emotion_targets"].to(CFG.device)
            ehis = b["emotion_history"].to(CFG.device)
            opt.zero_grad()
            out  = model(None, None, num)
            neg_f = etgt.mean(1); neg_h = ehis.mean(1)
            drift = torch.abs(neg_f - neg_h)
            mtgt  = torch.stack([neg_f, drift], 1)
            loss  = crit(out["emotion_pred"], etgt) + 0.5*crit(out["monitor_pred"], mtgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()

        model.eval(); errs=[]
        with torch.no_grad():
            for b in val_ld:
                num  = b["numeric"].to(CFG.device)
                etgt = b["emotion_targets"].to(CFG.device)
                out  = model(None, None, num)
                errs.append(torch.abs(out["emotion_pred"]-etgt).mean().item())
        history["epoch"].append(ep+1)
        history["train_loss"].append(total/len(train_ld))
        history["val_mae"].append(float(np.mean(errs)))

    # ── clustering (Obj 3) ─────────────────────────────────────────────────
    model.eval()
    embs, true_lbls, emo_preds, mon_preds, user_order = [], [], [], [], []
    with torch.no_grad():
        for b in val_ld:
            num  = b["numeric"].to(CFG.device)
            out  = model(None, None, num)
            embs.append(out["cluster_embedding"].cpu().numpy())
            true_lbls.append(b["segment_label"].numpy())
            emo_preds.append(out["emotion_pred"].cpu().numpy())
            mon_preds.append(out["monitor_pred"].cpu().numpy())

    embs      = np.concatenate(embs)
    true_lbls = np.concatenate(true_lbls)
    emo_preds = np.concatenate(emo_preds)
    mon_preds = np.concatenate(mon_preds)

    km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(embs)
    cross_tab= pd.crosstab(clusters, le.inverse_transform(true_lbls))

    # ── personality analysis (Obj 4) ──────────────────────────────────────
    user_meta = (
        df[df.user_id.isin(val_ids)]
        .groupby("user_id")
        .agg(**{f:(f,"first") for f in PERSONALITY_FEATURES},
             avg_neg=("negative_emotion_index","mean"))
    )
    user_meta["cluster"] = clusters[:len(user_meta)]
    corr = user_meta[PERSONALITY_FEATURES+["avg_neg"]].corr()["avg_neg"].drop("avg_neg")
    cluster_pers = user_meta.groupby("cluster")[PERSONALITY_FEATURES].mean()

    return dict(
        model=model, le=le, scaler=scaler,
        history=pd.DataFrame(history),
        clusters=clusters, true_lbls=true_lbls, cross_tab=cross_tab,
        emo_preds=emo_preds, mon_preds=mon_preds,
        corr=corr, cluster_pers=cluster_pers,
        val_ds=val_ds,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 IoB Monitor")
    st.caption("BERT-BiLSTM · Social Media Impact")
    st.divider()

    uploaded = st.file_uploader("Upload dataset CSV", type="csv",
                                help="Expects columns: user_id, timestep, segment_label, post_text, + feature columns")
    st.divider()

    n_users  = st.slider("Synthetic users (if no CSV)", 60, 400, 120, 20)
    n_epochs = st.slider("Training epochs", 1, 10, 3)
    n_clust  = st.slider("Clusters (Obj 3)", 3, 8, 5)
    use_bert = st.checkbox("Use real BERT (slow on CPU)", value=False,
                           help="Uncheck for lightweight text proxy — identical architecture, faster training")
    st.divider()
    run_btn  = st.button("▶  Run Pipeline", use_container_width=True, type="primary")
    st.caption(f"Device: `{CFG.device}`")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:1.8rem;font-weight:700;color:#e3f2fd;margin-bottom:4px'>
🧠 BERT-BiLSTM Social Media Impact Dashboard
</h1>
<p style='color:#78909c;font-size:0.9rem;margin-top:0'>
Objective-Aligned Monitoring · Facebook · Instagram · YouTube · WhatsApp · Telegram
</p>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "📊 Overview",
    "📉 Training",
    "😟 Emotions (Obj 2)",
    "🗂 Segments (Obj 3)",
    "🧬 Personality (Obj 4)",
    "🚨 Monitor (Obj 5)",
    "🔮 Inference",
])

# ─────────────────────────────────────────────────────────────────────────────
# 10.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.session_state["df"] = df_raw
    st.session_state["data_source"] = "uploaded"
else:
    if "df" not in st.session_state or st.session_state.get("n_users") != n_users:
        with st.spinner("Generating synthetic dataset…"):
            st.session_state["df"]          = generate_synthetic_data(n_users)
            st.session_state["n_users"]     = n_users
            st.session_state["data_source"] = "synthetic"

df = st.session_state["df"]

# ─────────────────────────────────────────────────────────────────────────────
# 11.  PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if run_btn or "results" not in st.session_state:
    with st.spinner("Running pipeline… (first run may take a minute)"):
        df_hash = hash(df.values.tobytes()[:4096])
        st.session_state["results"] = run_pipeline(df_hash, n_epochs, n_clust, use_bert)
    st.success("Pipeline complete!", icon="✅")

R = st.session_state.get("results", None)

# ─────────────────────────────────────────────────────────────────────────────
# 12.  TAB 0 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<span class="obj-badge">ALL OBJECTIVES</span>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Dataset & Model Summary</p>', unsafe_allow_html=True)

    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="value">{df.user_id.nunique()}</div><div class="label">Users</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="value">{len(df)}</div><div class="label">Records</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="value">{len(NUMERIC_FEATURES)}</div><div class="label">Features</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="value">{n_clust}</div><div class="label">Clusters</div></div>', unsafe_allow_html=True)
    with col5:
        val_mae = R["history"]["val_mae"].iloc[-1] if R else "—"
        val_str = f"{val_mae:.4f}" if isinstance(val_mae, float) else val_mae
        st.markdown(f'<div class="metric-card"><div class="value">{val_str}</div><div class="label">Val MAE</div></div>', unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns([3,2])
    with c1:
        st.markdown("**Negative Emotion Distribution by Segment**")
        fig = px.violin(
            df, x="segment_label", y="negative_emotion_index",
            color="segment_label",
            color_discrete_sequence=SEGMENT_COLORS,
            box=True, points=False,
            labels={"segment_label":"Segment","negative_emotion_index":"Neg. Emotion Index"},
        )
        fig.update_layout(showlegend=False, height=340,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,32,50,0.6)",
                          font_color="#cfd8dc", margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Platform Usage Mix**")
        plats = ["plat_Facebook","plat_Instagram","plat_YouTube","plat_WhatsApp","plat_Telegram"]
        vals  = df[plats].sum().values
        names = [p.replace("plat_","") for p in plats]
        fig2  = go.Figure(go.Pie(
            labels=names, values=vals, hole=0.45,
            marker_colors=["#1565c0","#ad1457","#c62828","#2e7d32","#6a1b9a"],
        ))
        fig2.update_layout(height=340,
                           paper_bgcolor="rgba(0,0,0,0)", font_color="#cfd8dc",
                           margin=dict(t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Sample Data**")
    st.dataframe(df.head(20), use_container_width=True, height=240)

# ─────────────────────────────────────────────────────────────────────────────
# 13.  TAB 1 · TRAINING
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    if R is None:
        st.info("Run the pipeline to see training curves.")
    else:
        st.markdown('<span class="obj-badge">OBJ 2 + OBJ 5</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Training Metrics</p>', unsafe_allow_html=True)

        hist = R["history"]
        fig  = make_subplots(rows=1, cols=2,
                             subplot_titles=["Train Loss","Val Emotion MAE"])
        fig.add_trace(go.Scatter(x=hist.epoch, y=hist.train_loss,
                                 mode="lines+markers", name="Train Loss",
                                 line=dict(color="#64b5f6", width=2.5)), row=1,col=1)
        fig.add_trace(go.Scatter(x=hist.epoch, y=hist.val_mae,
                                 mode="lines+markers", name="Val MAE",
                                 line=dict(color="#ff8a65", width=2.5)), row=1,col=2)
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(22,32,50,0.6)",
                          font_color="#cfd8dc", showlegend=False,
                          margin=dict(t=40,b=20))
        fig.update_xaxes(title_text="Epoch")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Epoch Log**")
        st.dataframe(hist.style.format({"train_loss":"{:.4f}","val_mae":"{:.4f}"}),
                     use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 14.  TAB 2 · EMOTIONS  (Obj 2)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    if R is None:
        st.info("Run the pipeline to see emotion predictions.")
    else:
        st.markdown('<span class="obj-badge">OBJECTIVE 2</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Predicted Negative Emotion Intensities</p>', unsafe_allow_html=True)
        st.caption("Multi-output regression: worry · depression · anxiety · jealousy")

        ep = R["emo_preds"]                             # (N_val, 4)
        pred_df = pd.DataFrame(ep, columns=EMOTION_TARGETS)
        pred_df["user_idx"] = np.arange(len(pred_df))
        pred_df["segment"]  = R["le"].inverse_transform(R["true_lbls"])

        # Radar per segment
        seg_means = pred_df.groupby("segment")[EMOTION_TARGETS].mean().reset_index()
        fig = go.Figure()
        cats = EMOTION_TARGETS + [EMOTION_TARGETS[0]]
        for _, row in seg_means.iterrows():
            vals = [row[e] for e in EMOTION_TARGETS] + [row[EMOTION_TARGETS[0]]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself",
                                         name=row["segment"], opacity=0.7))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                          height=400, paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#cfd8dc", margin=dict(t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Emotion Distribution (box)**")
            melt = pred_df.melt(id_vars=["segment"], value_vars=EMOTION_TARGETS,
                                var_name="Emotion", value_name="Score")
            fig2 = px.box(melt, x="Emotion", y="Score", color="Emotion",
                          color_discrete_sequence=["#ef5350","#7e57c2","#ff7043","#ec407a"])
            fig2.update_layout(showlegend=False, height=320,
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(22,32,50,0.6)",
                               font_color="#cfd8dc", margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            st.markdown("**Mean Predicted Emotions per Segment**")
            st.dataframe(seg_means.set_index("segment").style.format("{:.3f}").background_gradient(cmap="Reds"),
                         use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 15.  TAB 3 · SEGMENTATION  (Obj 3)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    if R is None:
        st.info("Run the pipeline to see clusters.")
    else:
        st.markdown('<span class="obj-badge">OBJECTIVE 3</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Behavioral-Emotional Segmentation via K-Means</p>', unsafe_allow_html=True)

        embs = np.vstack([  # reuse cluster_embeddings we already have via val_ds
            # quick 2-D PCA of emo_preds for scatter
            R["emo_preds"][:, :2]
        ])
        seg_names = R["le"].inverse_transform(R["true_lbls"])
        cluster_ids = R["clusters"]

        scatter_df = pd.DataFrame({
            "x": embs[:,0], "y": embs[:,1],
            "cluster": [f"Cluster {c}" for c in cluster_ids],
            "segment": seg_names,
        })

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Cluster assignments (emotion-space projection)**")
            fig = px.scatter(scatter_df, x="x", y="y",
                             color="cluster", symbol="segment",
                             color_discrete_sequence=SEGMENT_COLORS,
                             labels={"x":"Worry","y":"Depression"})
            fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(22,32,50,0.6)",
                              font_color="#cfd8dc", margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Cluster × Ground-Truth Segment Cross-tab**")
            ct = R["cross_tab"]
            fig2 = px.imshow(ct, text_auto=True, aspect="auto",
                             color_continuous_scale="Blues",
                             labels={"x":"Segment","y":"K-Means Cluster","color":"Count"})
            fig2.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                               font_color="#cfd8dc", margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Cluster size distribution**")
        cluster_counts = pd.Series(cluster_ids).value_counts().sort_index().reset_index()
        cluster_counts.columns = ["Cluster","Count"]
        fig3 = px.bar(cluster_counts, x="Cluster", y="Count",
                      color="Count", color_continuous_scale="Viridis",
                      labels={"Cluster":"K-Means Cluster"})
        fig3.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(22,32,50,0.6)",
                           font_color="#cfd8dc", margin=dict(t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 16.  TAB 4 · PERSONALITY  (Obj 4)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    if R is None:
        st.info("Run the pipeline to see personality analysis.")
    else:
        st.markdown('<span class="obj-badge">OBJECTIVE 4</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Big Five Personality Traits & Social Media Behavior</p>', unsafe_allow_html=True)

        corr = R["corr"].reset_index()
        corr.columns = ["Trait","Correlation with Avg Neg Emotion"]

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Correlation: Personality → Negative Emotion**")
            colors = ["#ef5350" if v>0 else "#42a5f5" for v in corr.iloc[:,1]]
            fig = go.Figure(go.Bar(
                x=corr.iloc[:,1], y=corr["Trait"],
                orientation="h", marker_color=colors,
            ))
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(22,32,50,0.6)",
                              font_color="#cfd8dc", margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Mean Big Five per Cluster (heatmap)**")
            cp = R["cluster_pers"].copy()
            cp.columns = [c.replace("personality_","") for c in cp.columns]
            fig2 = px.imshow(cp, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r",
                             labels={"x":"Trait","y":"Cluster","color":"Score"})
            fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                               font_color="#cfd8dc", margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Neuroticism vs Avg Negative Emotion (scatter)**")
        user_meta = (
            df.groupby("user_id")
            .agg(neuroticism=("personality_neuroticism","mean"),
                 avg_neg=("negative_emotion_index","mean"),
                 segment=("segment_label","first"))
            .reset_index()
        )
        fig3 = px.scatter(user_meta, x="neuroticism", y="avg_neg",
                          color="segment",
                          trendline="ols",
                          color_discrete_sequence=SEGMENT_COLORS,
                          labels={"neuroticism":"Neuroticism","avg_neg":"Avg Neg Emotion"})
        fig3.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(22,32,50,0.6)",
                           font_color="#cfd8dc", margin=dict(t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 17.  TAB 5 · SEMANTIC MONITOR  (Obj 5)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    if R is None:
        st.info("Run the pipeline to see monitoring outputs.")
    else:
        st.markdown('<span class="obj-badge">OBJECTIVE 5</span>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Thought Articulation & Behavioral Drift Monitor</p>', unsafe_allow_html=True)
        st.caption("Forecasted negative emotion index · Drift score relative to user baseline")

        mp = R["mon_preds"]  # (N_val, 2) → [forecast_neg, drift]
        mon_df = pd.DataFrame(mp, columns=["Forecast Neg Emotion","Drift Score"])
        mon_df["Segment"] = R["le"].inverse_transform(R["true_lbls"])
        mon_df["Alert"]   = mon_df["Drift Score"] > 0.5

        # KPIs
        high_drift = mon_df["Alert"].sum()
        avg_drift  = mon_df["Drift Score"].mean()
        avg_fore   = mon_df["Forecast Neg Emotion"].mean()

        k1,k2,k3 = st.columns(3)
        with k1:
            st.markdown(f'<div class="metric-card"><div class="value" style="color:#ef5350">{high_drift}</div><div class="label">High-Drift Users (score &gt; 0.5)</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="metric-card"><div class="value">{avg_drift:.3f}</div><div class="label">Mean Drift Score</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="metric-card"><div class="value">{avg_fore:.3f}</div><div class="label">Mean Forecasted Neg Emotion</div></div>', unsafe_allow_html=True)

        st.divider()
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Drift Score Distribution by Segment**")
            fig = px.box(mon_df, x="Segment", y="Drift Score", color="Segment",
                         color_discrete_sequence=SEGMENT_COLORS)
            fig.add_hline(y=0.5, line_dash="dash", line_color="#ef5350",
                          annotation_text="Alert threshold")
            fig.update_layout(showlegend=False, height=340,
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(22,32,50,0.6)",
                              font_color="#cfd8dc", margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Forecast vs Drift (scatter)**")
            fig2 = px.scatter(mon_df, x="Forecast Neg Emotion", y="Drift Score",
                              color="Segment", symbol="Alert",
                              color_discrete_sequence=SEGMENT_COLORS)
            fig2.add_hline(y=0.5, line_dash="dash", line_color="#ef5350")
            fig2.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(22,32,50,0.6)",
                               font_color="#cfd8dc", margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**User-Level Monitor Table** (top 20 by drift)")
        top = mon_df.sort_values("Drift Score", ascending=False).head(20).reset_index(drop=True)
        top["Status"] = top["Alert"].map({True:"🔴 ALERT", False:"🟢 OK"})
        st.dataframe(top[["Forecast Neg Emotion","Drift Score","Segment","Status"]]
                     .style.format({"Forecast Neg Emotion":"{:.3f}","Drift Score":"{:.3f}"}),
                     use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 18.  TAB 6 · LIVE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<span class="obj-badge">LIVE INFERENCE</span>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Single-User Prediction</p>', unsafe_allow_html=True)

    if R is None:
        st.info("Run the pipeline first.")
    else:
        st.markdown("Adjust sliders to profile a hypothetical user:")

        col1, col2, col3 = st.columns(3)
        with col1:
            dm   = st.slider("Daily minutes", 30, 600, 180)
            sess = st.slider("Sessions / day", 1, 20, 6)
            nur  = st.slider("Night use ratio", 0.0, 1.0, 0.3, 0.05)
            nrt  = st.slider("Neuroticism", 0.0, 1.0, 0.5, 0.05)
        with col2:
            vr   = st.slider("Video ratio", 0.0, 1.0, 0.4, 0.05)
            mr   = st.slider("Messaging ratio", 0.0, 1.0, 0.5, 0.05)
            opi  = st.slider("Openness", 0.0, 1.0, 0.6, 0.05)
            agr  = st.slider("Agreeableness", 0.0, 1.0, 0.6, 0.05)
        with col3:
            ext  = st.slider("Extraversion", 0.0, 1.0, 0.5, 0.05)
            con  = st.slider("Conscientiousness", 0.0, 1.0, 0.5, 0.05)
            ui   = st.slider("Usage intensity", 0.0, 1.0, 0.5, 0.05)
            neg_hist = st.slider("Historical neg-emotion index", 0.0, 1.0, 0.35, 0.05)

        if st.button("🔮 Predict", type="primary"):
            # build a single-user synthetic sequence
            T   = CFG.seq_len
            row = np.zeros(len(NUMERIC_FEATURES), dtype=np.float32)
            feat_map = {
                "daily_minutes": dm, "sessions": sess, "night_use_ratio": nur,
                "video_ratio": vr, "messaging_ratio": mr, "usage_intensity": ui,
                "personality_neuroticism": nrt, "personality_openness": opi,
                "personality_agreeableness": agr, "personality_extraversion": ext,
                "personality_conscientiousness": con,
                "negative_emotion_index": neg_hist,
                "neuroticism_x_usage": nrt * dm / 300,
            }
            for f, v in feat_map.items():
                if f in NUMERIC_FEATURES:
                    row[NUMERIC_FEATURES.index(f)] = v

            seq  = np.tile(row, (T, 1))
            seq_t= torch.tensor(seq).unsqueeze(0).to(CFG.device)  # (1, T, F)
            scaler= R["scaler"]
            seq_scaled = scaler.transform(seq).astype(np.float32)
            seq_t = torch.tensor(seq_scaled).unsqueeze(0).to(CFG.device)

            model = R["model"]; model.eval()
            with torch.no_grad():
                out = model(None, None, seq_t)

            ep = out["emotion_pred"][0].cpu().numpy()
            mp_ = out["monitor_pred"][0].cpu().numpy()

            st.divider()
            e1,e2,e3,e4 = st.columns(4)
            for col, emotion, val in zip([e1,e2,e3,e4], EMOTION_TARGETS, ep):
                color = "#ef5350" if val > 0.5 else "#66bb6a"
                with col:
                    st.markdown(f'<div class="metric-card"><div class="value" style="color:{color}">{val:.3f}</div><div class="label">{emotion.capitalize()}</div></div>', unsafe_allow_html=True)

            forecast, drift = mp_
            st.divider()
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="value">{forecast:.3f}</div><div class="label">Forecast Neg Emotion Index</div></div>', unsafe_allow_html=True)
            with m2:
                drift_html = f'<div class="drift-alert">⚠️  High behavioral drift detected — score {drift:.3f}</div>' if drift > 0.5 else f'<div class="drift-ok">✅  Drift within normal range — score {drift:.3f}</div>'
                st.markdown(drift_html, unsafe_allow_html=True)

            # mini radar
            cats_ = EMOTION_TARGETS + [EMOTION_TARGETS[0]]
            vals_ = list(ep) + [ep[0]]
            fig_r = go.Figure(go.Scatterpolar(r=vals_, theta=cats_, fill="toself",
                                              line_color="#64b5f6"))
            fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                                 height=300, paper_bgcolor="rgba(0,0,0,0)",
                                 font_color="#cfd8dc", margin=dict(t=20,b=10))
            st.plotly_chart(fig_r, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 19.  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='text-align:center;color:#546e7a;font-size:0.78rem'>
BERT-BiLSTM · IoB Social Media Impact Study ·
Objectives 1-5 · Facebook · Instagram · YouTube · WhatsApp · Telegram
</p>
""", unsafe_allow_html=True)

