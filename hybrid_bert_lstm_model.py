"""
Hybrid BERT-LSTM Model for Behavioral Trait Monitoring, Analysis,
and Prediction on Social Media using IoB Data
====================================================================

Architecture
------------
For each user, a sequence of T=15 timesteps is available, each containing:
  (a) post_text          -> a short text snippet for that timestep
  (b) engineered numeric / categorical IoB features (activity, sentiment,
      engagement, temporal, platform/device/age one-hot, etc.)

Pipeline:
  1. BERT branch: each timestep's post_text -> BERT -> [CLS] embedding (768-d)
     This captures the *semantic / linguistic* behavioral signal.
  2. Numeric branch: engineered IoB features per timestep -> dense projection
  3. Fusion: concatenate BERT embedding + projected numeric features per timestep
  4. LSTM branch: fused per-timestep vectors -> stacked Bi-LSTM
     This captures the *temporal / sequential* behavioral signal
     (drift, trends, escalation patterns).
  5. Classification head: final LSTM hidden state -> dense layers ->
     softmax over behavioral trait classes
     (engaged / anxious / neutral / addictive / withdrawn)

  An auxiliary regression head (optional) predicts a continuous
  "risk score" from the same fused representation.

Requirements: torch, transformers, pandas, numpy, scikit-learn
    pip install torch transformers pandas numpy scikit-learn --break-system-packages
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

class Config:
    csv_path = "iob_social_media_dataset.csv"
    seq_len = 15                 # timesteps per user
    bert_model_name = "bert-base-uncased"
    bert_max_len = 32            # short post snippets
    bert_hidden = 768
    numeric_proj_dim = 32
    lstm_hidden = 128
    lstm_layers = 2
    bidirectional = True
    num_classes = 5              # engaged / anxious / neutral / addictive / withdrawn
    dropout = 0.3
    batch_size = 16
    lr = 2e-5
    bert_lr = 2e-5
    head_lr = 1e-3
    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_bert_layers = 8        # freeze first N encoder layers of BERT to reduce compute


CFG = Config()


# ----------------------------------------------------------------------
# 1. FEATURE COLUMN SETUP
# ----------------------------------------------------------------------

# Engineered numeric/categorical features produced during preprocessing
# (matches generate_dataset.py output)
NUMERIC_FEATURES = [
    "sessions", "posts", "likes_given", "likes_received",
    "comments_made", "comments_received", "shares",
    "avg_session_min", "scroll_depth", "night_activity_ratio",
    "sentiment_score", "engagement_ratio", "activity_intensity",
    "content_ratio", "sentiment_volatility", "sentiment_trend_3d",
    "session_min_zscore", "activity_delta", "dow_sin", "dow_cos",
    "influence_score",
    "plat_Facebook", "plat_Instagram", "plat_LinkedIn", "plat_Reddit",
    "plat_TikTok", "plat_Twitter/X",
    "dev_desktop", "dev_mobile", "dev_tablet",
    "age_18-24", "age_25-34", "age_35-44", "age_45+",
]


# ----------------------------------------------------------------------
# 2. DATASET
# ----------------------------------------------------------------------

class IoBSequenceDataset(Dataset):
    """
    Produces one sample per USER: a sequence of T timesteps, each with
    tokenized text + numeric feature vector, plus a single trait label
    (the label at the final timestep is used as the prediction target --
    i.e. "given this user's behavioral history, what is their current
    / near-future trait?").
    """

    def __init__(self, df, tokenizer, scaler, label_encoder, seq_len):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.label_encoder = label_encoder

        df = df.sort_values(["user_id", "timestep"])
        self.users = df["user_id"].unique()
        self.df = df.set_index("user_id")

        # Pre-scale numeric features
        self.feature_matrix = scaler.transform(df[NUMERIC_FEATURES].values)
        df = df.copy()
        df[[f + "_scaled" for f in NUMERIC_FEATURES]] = self.feature_matrix
        self.df = df.set_index("user_id")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        user_df = self.df.loc[[uid]].sort_values("timestep").iloc[: self.seq_len]

        texts = user_df["post_text"].tolist()
        numeric = user_df[[f + "_scaled" for f in NUMERIC_FEATURES]].values.astype(np.float32)

        # Tokenize each timestep's text
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=CFG.bert_max_len,
            return_tensors="pt",
        )

        label = self.label_encoder.transform([user_df["trait_label"].iloc[-1]])[0]

        return {
            "input_ids": encodings["input_ids"],          # (T, max_len)
            "attention_mask": encodings["attention_mask"],  # (T, max_len)
            "numeric": torch.tensor(numeric),               # (T, num_features)
            "label": torch.tensor(label, dtype=torch.long),
        }


# ----------------------------------------------------------------------
# 3. MODEL ARCHITECTURE
# ----------------------------------------------------------------------

class HybridBERTLSTM(nn.Module):
    def __init__(self, cfg, num_numeric_features):
        super().__init__()
        self.cfg = cfg

        # --- BERT branch (semantic / linguistic behavior) ---
        self.bert = BertModel.from_pretrained(cfg.bert_model_name)
        if cfg.freeze_bert_layers > 0:
            for layer in self.bert.encoder.layer[: cfg.freeze_bert_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False

        # --- Numeric branch (IoB engineered features) ---
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
        lstm_out_dim = cfg.lstm_hidden * (2 if cfg.bidirectional else 1)

        # --- Classification head (behavioral trait) ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, cfg.num_classes),
        )

        # --- Auxiliary regression head (continuous risk score, 0-1) ---
        self.risk_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask, numeric):
        """
        input_ids, attention_mask: (B, T, L)
        numeric:                   (B, T, F)
        """
        B, T, L = input_ids.shape

        # Flatten batch & time for BERT, then reshape back
        flat_ids = input_ids.view(B * T, L)
        flat_mask = attention_mask.view(B * T, L)

        bert_out = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :]          # (B*T, 768) -- [CLS] token
        cls_emb = cls_emb.view(B, T, self.cfg.bert_hidden)     # (B, T, 768)

        numeric_emb = self.numeric_proj(numeric)               # (B, T, numeric_proj_dim)

        fused = torch.cat([cls_emb, numeric_emb], dim=-1)      # (B, T, fusion_dim)

        lstm_out, (h_n, _) = self.lstm(fused)                  # lstm_out: (B, T, lstm_out_dim)

        # Use the final timestep's output as the sequence summary
        final_repr = lstm_out[:, -1, :]                         # (B, lstm_out_dim)

        trait_logits = self.classifier(final_repr)             # (B, num_classes)
        risk_score = self.risk_head(final_repr).squeeze(-1)    # (B,)

        return trait_logits, risk_score


# ----------------------------------------------------------------------
# 4. TRAINING LOOP
# ----------------------------------------------------------------------

def train_model():
    df = pd.read_csv(CFG.csv_path)

    # Heuristic continuous risk label for the auxiliary head:
    # combines negative sentiment + high night activity + sentiment volatility
    df["risk_target"] = np.clip(
        0.5
        - df["sentiment_score"] * 0.4
        + df["night_activity_ratio"] * 0.5
        + df["sentiment_volatility"] * 0.3,
        0, 1,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(df["trait_label"])

    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_FEATURES].values)

    tokenizer = BertTokenizerFast.from_pretrained(CFG.bert_model_name)

    # Train/val split at the USER level (avoid leakage across timesteps)
    user_ids = df["user_id"].unique()
    train_ids, val_ids = train_test_split(user_ids, test_size=0.2, random_state=42)

    train_df = df[df.user_id.isin(train_ids)]
    val_df = df[df.user_id.isin(val_ids)]

    train_ds = IoBSequenceDataset(train_df, tokenizer, scaler, label_encoder, CFG.seq_len)
    val_ds = IoBSequenceDataset(val_df, tokenizer, scaler, label_encoder, CFG.seq_len)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

    model = HybridBERTLSTM(CFG, num_numeric_features=len(NUMERIC_FEATURES)).to(CFG.device)

    # Separate learning rates: smaller for BERT, larger for new layers
    bert_params = [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "bert" not in n]

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": CFG.bert_lr},
        {"params": head_params, "lr": CFG.head_lr},
    ])

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    for epoch in range(CFG.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(CFG.device)
            attention_mask = batch["attention_mask"].to(CFG.device)
            numeric = batch["numeric"].to(CFG.device)
            labels = batch["label"].to(CFG.device)

            optimizer.zero_grad()
            trait_logits, risk_score = model(input_ids, attention_mask, numeric)

            loss_cls = cls_criterion(trait_logits, labels)
            # NOTE: risk_target would need to be batched per-sequence (e.g. mean
            # over the sequence) -- omitted here for brevity; shown conceptually.
            loss = loss_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{CFG.epochs} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

    return model, label_encoder, scaler, tokenizer


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        numeric = batch["numeric"].to(CFG.device)
        labels = batch["label"].to(CFG.device)

        trait_logits, _ = model(input_ids, attention_mask, numeric)
        preds = trait_logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ----------------------------------------------------------------------
# 5. INFERENCE HELPER
# ----------------------------------------------------------------------

@torch.no_grad()
def predict_user_traits(model, tokenizer, scaler, label_encoder, user_sequence_df):
    """
    user_sequence_df: a DataFrame slice for ONE user, sorted by timestep,
                       containing 'post_text' + NUMERIC_FEATURES columns.
    Returns: (predicted_trait_label, risk_score_float)
    """
    model.eval()

    texts = user_sequence_df["post_text"].tolist()
    numeric = scaler.transform(user_sequence_df[NUMERIC_FEATURES].values).astype(np.float32)

    enc = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=CFG.bert_max_len, return_tensors="pt",
    )

    input_ids = enc["input_ids"].unsqueeze(0).to(CFG.device)        # (1, T, L)
    attention_mask = enc["attention_mask"].unsqueeze(0).to(CFG.device)
    numeric_t = torch.tensor(numeric).unsqueeze(0).to(CFG.device)   # (1, T, F)

    trait_logits, risk_score = model(input_ids, attention_mask, numeric_t)
    pred_idx = trait_logits.argmax(dim=-1).item()
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return pred_label, float(risk_score.item())


if __name__ == "__main__":
    print(f"Device: {CFG.device}")
    print("Starting training (requires torch + transformers + GPU recommended)...")
    model, label_encoder, scaler, tokenizer = train_model()

    torch.save(model.state_dict(), "hybrid_bert_lstm_iob.pt")
    print("Model saved to hybrid_bert_lstm_iob.pt")
