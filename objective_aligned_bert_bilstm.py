"""
Objective-Aligned Hybrid BERT-BiLSTM Model
============================================
Social Media Usage, Emotional Sensitivity, and Behavior Impact Model
Platforms in scope: Facebook, Instagram, YouTube, WhatsApp, Telegram

Maps directly onto the five research objectives:

  Obj 1 - Identify factors of behavioral aspects from social media usage
          -> Numeric branch input: engineered usage-pattern features
             (daily_minutes, sessions, night_use_ratio, platform_switches,
              video_ratio, messaging_ratio, story_post_ratio, usage_intensity, ...)
             Feature importance analysis on the fused representation reveals
             which usage factors drive each downstream output.

  Obj 2 - Impact of usage factors on negative emotions
          (worry, depression, anxiety, jealousy)
          -> Emotion Impact Head: multi-output regression predicting the
             intensity of each of the four negative emotions from the
             shared sequence representation.

  Obj 3 - Segment respondents into clusters based on emotional-trait factors
          -> Segmentation Head: the shared sequence representation is
             clustered (K-Means / GMM) into behavioral-emotional personas
             (e.g. balanced_user, anxious_scroller, validation_seeker,
              passive_consumer, social_connector).

  Obj 4 - Role of personality traits in social media behavior
          -> Personality branch: Big Five trait proxies (openness,
             conscientiousness, extraversion, agreeableness, neuroticism)
             are fused alongside usage features; post-hoc analysis
             (correlation / SHAP-style importance) quantifies how each
             personality dimension relates to cluster membership and
             emotion-head outputs.

  Obj 5 - Monitor and predict thought articulation via semantic analysis
          -> BERT branch encodes post/comment text -> [CLS] embedding ->
             fused with numeric features -> BiLSTM models the temporal
             sequence -> Semantic Monitoring Head predicts near-future
             negative-emotion trajectory and flags behavioral drift.

Architecture:

  [usage features] --\
  [personality features] --> Numeric branch (dense) --\
                                                          >-- Fusion --> BiLSTM --> shared repr (256-d)
  [post_text] --> BERT --> [CLS] embedding (768-d) ------/                              |
                                                                  ------------------------------------
                                                                  |              |              |
                                                          Emotion Impact   Segmentation   Semantic Monitor
                                                          Head (Obj 2)     Head (Obj 3)   Head (Obj 5)
                                                          4 regressions    clustering     drift + forecast

Requires: torch, transformers, pandas, numpy, scikit-learn
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from transformers import BertTokenizerFast, BertModel


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

class Config:
    csv_path = "iob_social_media_dataset_v2.csv"
    seq_len = 15
    bert_model_name = "bert-base-uncased"
    bert_max_len = 32
    bert_hidden = 768
    numeric_proj_dim = 32
    lstm_hidden = 128
    lstm_layers = 2
    bidirectional = True
    n_clusters = 5                  # Obj 3: number of behavioral-emotional segments
    n_emotions = 4                  # Obj 2: worry, depression, anxiety, jealousy
    dropout = 0.3
    batch_size = 16
    bert_lr = 2e-5
    head_lr = 1e-3
    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_bert_layers = 8


CFG = Config()


# ----------------------------------------------------------------------
# FEATURE GROUPS (Obj 1 usage factors + Obj 4 personality factors)
# ----------------------------------------------------------------------

USAGE_FEATURES = [
    "daily_minutes", "sessions", "night_use_ratio", "platform_switches",
    "video_ratio", "messaging_ratio", "story_post_ratio",
    "posts", "likes_given", "comments_made", "likes_received", "comments_received",
    "usage_intensity", "validation_seeking_ratio", "social_interaction_ratio",
    "usage_delta", "dow_sin", "dow_cos",
    "plat_Facebook", "plat_Instagram", "plat_Telegram", "plat_WhatsApp", "plat_YouTube",
]

PERSONALITY_FEATURES = [
    "personality_openness", "personality_conscientiousness",
    "personality_extraversion", "personality_agreeableness", "personality_neuroticism",
]

EMOTION_HISTORY_FEATURES = [
    "sentiment_score", "negative_emotion_index", "emotion_volatility_3d", "emotion_trend_3d",
    "neuroticism_x_usage",
]

NUMERIC_FEATURES = USAGE_FEATURES + PERSONALITY_FEATURES + EMOTION_HISTORY_FEATURES

EMOTION_TARGETS = ["worry", "depression", "anxiety", "jealousy"]  # Obj 2


# ----------------------------------------------------------------------
# DATASET
# ----------------------------------------------------------------------

class IoBSequenceDataset(Dataset):
    """
    One sample per user: sequence of T timesteps with tokenized text,
    numeric features, and per-timestep emotion targets (Obj 2 supervision).
    The final-timestep segment_label is retained for evaluating the
    segmentation head against ground truth (Obj 3 validation).
    """

    def __init__(self, df, tokenizer, scaler, label_encoder, seq_len):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

        df = df.copy().sort_values(["user_id", "timestep"])
        self.users = df["user_id"].unique()

        df[[f + "_scaled" for f in NUMERIC_FEATURES]] = scaler.transform(df[NUMERIC_FEATURES].values)
        self.df = df.set_index("user_id")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        user_df = self.df.loc[[uid]].sort_values("timestep").iloc[: self.seq_len]

        texts = user_df["post_text"].tolist()
        numeric = user_df[[f + "_scaled" for f in NUMERIC_FEATURES]].values.astype(np.float32)
        emotions = user_df[EMOTION_TARGETS].values.astype(np.float32)  # (T, 4)

        encodings = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=CFG.bert_max_len, return_tensors="pt",
        )

        segment_idx = self.label_encoder.transform([user_df["segment_label"].iloc[-1]])[0]

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "numeric": torch.tensor(numeric),
            "emotion_targets": torch.tensor(emotions[-1]),     # predict final-timestep emotion levels (Obj 2)
            "emotion_history": torch.tensor(emotions[:-1].mean(axis=0)),  # for drift comparison (Obj 5)
            "segment_label": torch.tensor(segment_idx, dtype=torch.long),  # ground truth for cluster validation (Obj 3)
        }


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------

class ObjectiveAlignedBERTBiLSTM(nn.Module):
    """
    Shared encoder (BERT + numeric fusion + BiLSTM) with three task heads:
      - emotion_head    (Obj 2: multi-output regression, 4 negative emotions)
      - cluster_proj    (Obj 3: projection used as input to K-Means; not a
                          classifier -- segmentation is unsupervised, this
                          head just produces the embedding to be clustered)
      - monitor_head    (Obj 5: forecasts near-future negative-emotion index
                          and a drift score relative to the user's own history)
    """

    def __init__(self, cfg, num_numeric_features):
        super().__init__()
        self.cfg = cfg

        # --- BERT branch (Obj 5: semantic / thought-articulation signal) ---
        self.bert = BertModel.from_pretrained(cfg.bert_model_name)
        if cfg.freeze_bert_layers > 0:
            for layer in self.bert.encoder.layer[: cfg.freeze_bert_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False

        # --- Numeric branch (Obj 1 usage factors + Obj 4 personality factors) ---
        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, cfg.numeric_proj_dim),
            nn.ReLU(),
        )

        # --- Fusion + temporal modeling ---
        fusion_dim = cfg.bert_hidden + cfg.numeric_proj_dim
        self.lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        repr_dim = cfg.lstm_hidden * (2 if cfg.bidirectional else 1)
        self.repr_dim = repr_dim

        # --- Obj 2: Emotion Impact Head (multi-output regression) ---
        self.emotion_head = nn.Sequential(
            nn.Linear(repr_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, cfg.n_emotions),
            nn.Sigmoid(),  # emotion intensities in [0, 1]
        )

        # --- Obj 3: Segmentation projection (embedding fed to K-Means externally) ---
        self.cluster_proj = nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),  # low-dim embedding for clustering
        )

        # --- Obj 5: Semantic Monitoring Head ---
        # Predicts (a) forecasted negative-emotion index for next period
        # and (b) a drift score vs. the user's own historical average
        self.monitor_head = nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [forecast_negative_emotion_index, drift_score]
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, numeric):
        B, T, L = input_ids.shape

        flat_ids = input_ids.view(B * T, L)
        flat_mask = attention_mask.view(B * T, L)

        bert_out = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :].view(B, T, self.cfg.bert_hidden)

        numeric_emb = self.numeric_proj(numeric)

        fused = torch.cat([cls_emb, numeric_emb], dim=-1)
        lstm_out, _ = self.lstm(fused)
        shared_repr = lstm_out[:, -1, :]  # (B, repr_dim)

        emotion_pred = self.emotion_head(shared_repr)        # Obj 2
        cluster_embedding = self.cluster_proj(shared_repr)   # Obj 3 (input to K-Means)
        monitor_pred = self.monitor_head(shared_repr)        # Obj 5

        return {
            "shared_repr": shared_repr,
            "emotion_pred": emotion_pred,
            "cluster_embedding": cluster_embedding,
            "monitor_pred": monitor_pred,
        }


# ----------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------

def train_model():
    df = pd.read_csv(CFG.csv_path)

    label_encoder = LabelEncoder()
    label_encoder.fit(df["segment_label"])  # used only for evaluating clustering against ground truth

    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_FEATURES].values)

    tokenizer = BertTokenizerFast.from_pretrained(CFG.bert_model_name)

    user_ids = df["user_id"].unique()
    train_ids, val_ids = train_test_split(user_ids, test_size=0.2, random_state=42)

    train_ds = IoBSequenceDataset(df[df.user_id.isin(train_ids)], tokenizer, scaler, label_encoder, CFG.seq_len)
    val_ds = IoBSequenceDataset(df[df.user_id.isin(val_ids)], tokenizer, scaler, label_encoder, CFG.seq_len)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    model = ObjectiveAlignedBERTBiLSTM(CFG, num_numeric_features=len(NUMERIC_FEATURES)).to(CFG.device)

    bert_params = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "bert" not in n]

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": CFG.bert_lr},
        {"params": head_params, "lr": CFG.head_lr},
    ])

    emotion_criterion = nn.MSELoss()       # Obj 2
    monitor_criterion = nn.MSELoss()       # Obj 5

    for epoch in range(CFG.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(CFG.device)
            attention_mask = batch["attention_mask"].to(CFG.device)
            numeric = batch["numeric"].to(CFG.device)
            emotion_targets = batch["emotion_targets"].to(CFG.device)
            emotion_history = batch["emotion_history"].to(CFG.device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask, numeric)

            # Obj 2: emotion impact regression loss
            loss_emotion = emotion_criterion(out["emotion_pred"], emotion_targets)

            # Obj 5: monitoring loss -- forecast target is the final-timestep
            # negative emotion composite; drift target is |final - history mean|
            neg_emotion_final = emotion_targets.mean(dim=1)              # (B,)
            neg_emotion_hist = emotion_history.mean(dim=1)               # (B,)
            drift_target = torch.abs(neg_emotion_final - neg_emotion_hist)
            monitor_target = torch.stack([neg_emotion_final, drift_target], dim=1)
            loss_monitor = monitor_criterion(out["monitor_pred"], monitor_target)

            loss = loss_emotion + 0.5 * loss_monitor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_mae = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{CFG.epochs} | train_loss={avg_loss:.4f} | val_emotion_mae={val_mae:.4f}")

    return model, label_encoder, scaler, tokenizer


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    errors = []
    for batch in loader:
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        numeric = batch["numeric"].to(CFG.device)
        emotion_targets = batch["emotion_targets"].to(CFG.device)

        out = model(input_ids, attention_mask, numeric)
        errors.append(torch.abs(out["emotion_pred"] - emotion_targets).mean().item())

    return float(np.mean(errors))


# ----------------------------------------------------------------------
# OBJECTIVE 3: SEGMENTATION VIA CLUSTERING ON SHARED REPRESENTATIONS
# ----------------------------------------------------------------------

@torch.no_grad()
def run_segmentation(model, loader, label_encoder, n_clusters=CFG.n_clusters):
    """
    Extracts cluster_embedding for all users, runs K-Means, and compares
    resulting clusters against the ground-truth segment_label distribution
    (Obj 3). In a real study, ground truth would come from survey-based
    emotional-trait scales rather than simulated labels.
    """
    model.eval()
    embeddings, true_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        numeric = batch["numeric"].to(CFG.device)

        out = model(input_ids, attention_mask, numeric)
        embeddings.append(out["cluster_embedding"].cpu().numpy())
        true_labels.append(batch["segment_label"].numpy())

    embeddings = np.concatenate(embeddings)
    true_labels = np.concatenate(true_labels)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_assignments = kmeans.fit_predict(embeddings)

    # Cross-tabulate clusters vs. ground-truth segments
    cross_tab = pd.crosstab(cluster_assignments, label_encoder.inverse_transform(true_labels))
    return cluster_assignments, cross_tab


# ----------------------------------------------------------------------
# OBJECTIVE 4: PERSONALITY -> CLUSTER / EMOTION ANALYSIS
# ----------------------------------------------------------------------

def analyze_personality_role(df, cluster_assignments, user_order):
    """
    Correlates each Big Five personality dimension with cluster membership
    and with average negative-emotion scores, addressing Obj 4.
    """
    user_meta = df.groupby("user_id").agg(
        **{f: (f, "first") for f in PERSONALITY_FEATURES},
        avg_negative_emotion=("negative_emotion_index", "mean"),
    ).reindex(user_order)

    user_meta["cluster"] = cluster_assignments

    correlations = user_meta[PERSONALITY_FEATURES + ["avg_negative_emotion"]].corr()["avg_negative_emotion"]
    cluster_personality_means = user_meta.groupby("cluster")[PERSONALITY_FEATURES].mean()

    return correlations, cluster_personality_means


if __name__ == "__main__":
    print(f"Device: {CFG.device}")
    model, label_encoder, scaler, tokenizer = train_model()

    df = pd.read_csv(CFG.csv_path)
    user_ids = df["user_id"].unique()
    _, val_ids = train_test_split(user_ids, test_size=0.2, random_state=42)
    val_ds = IoBSequenceDataset(df[df.user_id.isin(val_ids)], tokenizer, scaler, label_encoder, CFG.seq_len)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    cluster_assignments, cross_tab = run_segmentation(model, val_loader, label_encoder)
    print("\nObj 3 - Cluster vs. ground-truth segment cross-tab:")
    print(cross_tab)

    correlations, cluster_personality_means = analyze_personality_role(df, cluster_assignments, val_ds.users)
    print("\nObj 4 - Personality correlations with negative emotion:")
    print(correlations)
    print("\nObj 4 - Mean personality scores per cluster:")
    print(cluster_personality_means)

    torch.save(model.state_dict(), "objective_aligned_bert_bilstm.pt")
    print("\nModel saved to objective_aligned_bert_bilstm.pt")
