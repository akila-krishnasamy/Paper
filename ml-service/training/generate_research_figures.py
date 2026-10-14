"""
Generate Publication-Quality Research Paper Figures and Metrics
for the NFHS-5 Child Health Prediction Project.

Health Targets:
1. Stunting
2. Wasting
3. Anaemia
4. ARI-related illness

Deep Learning Models Compared:
- DNN (Deep Neural Network)
- TabNet
- FT-Transformer (Feature Tokenizer Transformer)

Figures Generated:
- outputs/plots/health_outcome_distribution.png (Figure 1: Distribution of Health Outcomes)
- outputs/plots/overall_model_performance.png   (Figure 2: Overall Model Performance Comparison)
- outputs/plots/target_wise_model_comparison.png (Figure 3: Target-Wise Model Comparison)
- outputs/plots/roc_curves.png                   (Figure 4: ROC Curves)
- outputs/plots/precision_recall_curves.png      (Figure 5: Precision-Recall Curves)

Supporting Numerical CSVs:
- outputs/metrics/model_evaluation_metrics.csv
- outputs/metrics/target_class_distributions.csv
- outputs/metrics/confusion_matrices.csv
- outputs/metrics/test_predictions_{target}.csv
"""

import os
import sys
import json
import time
from pathlib import Path

# Resolve project directories
ML_SERVICE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = ML_SERVICE_DIR.parent
sys.path.insert(0, str(ML_SERVICE_DIR))

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Global academic style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
})

# Consistent palette
MODEL_COLORS = {
    "DNN": "#1f77b4",          # Steel Blue
    "TabNet": "#2ca02c",       # Forest Green
    "FT-Transformer": "#9467bd" # Amethyst Purple
}
BASELINE_COLOR = "#555555"

PLOTS_DIR = ROOT_DIR / "outputs" / "plots"
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Helper Model Architectures (matching trained checkpoints)
# ==============================================================================

from training.train_dnn import TabularDNN


class BinaryDNN(nn.Module):
    """Single-target binary DNN for Anaemia and ARI."""
    def __init__(self, n_features=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class BinaryFTTransformer(nn.Module):
    """Single-target FT-Transformer for Anaemia and ARI."""
    def __init__(self, n_features=14, token_dim=16, heads=2, layers=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, token_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, token_dim))
        block = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=64,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return self.head(self.encoder(torch.cat([cls, tokens], dim=1))[:, 0]).squeeze(1)


# ==============================================================================
# Step 1: Obtain or Load Predictions for All 4 Targets
# ==============================================================================

def get_stunting_wasting_predictions():
    """Obtain held-out test predictions for Stunting and Wasting."""
    cache_stunting = METRICS_DIR / "test_predictions_stunting.csv"
    cache_wasting = METRICS_DIR / "test_predictions_wasting.csv"

    if cache_stunting.exists() and cache_wasting.exists():
        print("  [Cache Hit] Loading Stunting & Wasting test predictions from cache.")
        df_stunting = pd.read_csv(cache_stunting)
        df_wasting = pd.read_csv(cache_wasting)
        return df_stunting, df_wasting

    print("  [Processing] Preparing held-out test set for Stunting and Wasting...")
    from preprocessing.preprocessing import prepare_data_splits, ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
    from training.train_transformer import FTTransformer

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), preprocessor = prepare_data_splits(random_state=42)

    X_test_np = X_test[ALL_FEATURES].values.astype(np.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    X_test_num = torch.tensor(X_test[NUMERICAL_FEATURES].values.astype(np.float32), dtype=torch.float32)
    X_test_cat = torch.tensor(X_test[CATEGORICAL_FEATURES].values.astype(np.int64), dtype=torch.long)

    y_test_stunting = Y_test["Stunting"].values.astype(int)
    y_test_wasting = Y_test["Wasting"].values.astype(int)

    # 1. DNN
    print("  Running inference for Nutrition DNN...")
    dnn_path = ML_SERVICE_DIR / "models" / "dnn" / "best_dnn.pt"
    dnn_chk = torch.load(dnn_path, map_location="cpu", weights_only=False)
    dnn = TabularDNN(input_dim=18, num_targets=dnn_chk.get("num_targets", 3))
    dnn.load_state_dict(dnn_chk["model_state_dict"])
    dnn.eval()
    with torch.no_grad():
        dnn_logits = dnn(X_test_tensor)
        dnn_probs = torch.sigmoid(dnn_logits).numpy()

    # 2. TabNet
    print("  Running inference for Nutrition TabNet...")
    tabnet_path = ML_SERVICE_DIR / "models" / "tabnet" / "tabnet_model.zip"
    tabnet = TabNetMultiTaskClassifier()
    tabnet.load_model(str(tabnet_path))
    raw_tabnet = tabnet.predict_proba(X_test_np)
    tabnet_probs = np.column_stack([p[:, 1] for p in raw_tabnet])

    # 3. FT-Transformer
    print("  Running inference for Nutrition FT-Transformer...")
    trans_path = ML_SERVICE_DIR / "models" / "transformer" / "best_transformer.pt"
    trans_chk = torch.load(trans_path, map_location="cpu", weights_only=False)
    trans = FTTransformer(
        num_numerical=trans_chk.get("num_numerical", 9),
        categorical_cardinalities=trans_chk.get("categorical_cardinalities", [2, 4, 5, 2, 5, 2, 2, 2, 2]),
        d_token=trans_chk.get("d_token", 64),
        n_layers=trans_chk.get("n_layers", 2),
        n_heads=trans_chk.get("n_heads", 4),
        d_ffn=trans_chk.get("d_ffn", 128),
        n_targets=trans_chk.get("n_targets", 3)
    )
    trans.load_state_dict(trans_chk["model_state_dict"])
    trans.eval()
    with torch.no_grad():
        trans_logits = trans(X_test_num, X_test_cat)
        trans_probs = torch.sigmoid(trans_logits).numpy()

    df_stunting = pd.DataFrame({
        "y_true": y_test_stunting,
        "DNN_probability": dnn_probs[:, 0],
        "TabNet_probability": tabnet_probs[:, 0],
        "FT-Transformer_probability": trans_probs[:, 0],
    })
    df_wasting = pd.DataFrame({
        "y_true": y_test_wasting,
        "DNN_probability": dnn_probs[:, 1],
        "TabNet_probability": tabnet_probs[:, 1],
        "FT-Transformer_probability": trans_probs[:, 1],
    })

    df_stunting.to_csv(cache_stunting, index=False)
    df_wasting.to_csv(cache_wasting, index=False)
    print(f"  Saved test predictions to {cache_stunting} and {cache_wasting}")
    return df_stunting, df_wasting


def get_anemia_predictions():
    """Obtain held-out test predictions for Anaemia."""
    cache_path = METRICS_DIR / "test_predictions_anemia.csv"
    if cache_path.exists():
        print("  [Cache Hit] Loading Anaemia test predictions from cache.")
        return pd.read_csv(cache_path)

    print("  [Processing] Generating held-out test predictions for Anaemia...")
    from pytorch_tabnet.tab_model import TabNetClassifier
    from training.train_ari_model import make_splits, seed_everything

    raw_path = ROOT_DIR / "data" / "raw" / "IAKR7EFL.DTA"
    anemia_dir = ML_SERVICE_DIR / "models" / "nfhs_health_targets" / "anemia" / "early_risk_seed42"

    cols = ["hw57", "hw1", "b4", "v025", "v106", "v190", "v001", "v002"]
    raw = pd.read_stata(raw_path, columns=cols, convert_categoricals=False)

    valid = raw["hw57"].isin([1, 2, 3, 4])
    y = raw.loc[valid, "hw57"].isin([1, 2, 3]).astype(np.int8).to_numpy()
    df_valid = raw.loc[valid].reset_index(drop=True)

    groups = (df_valid["v001"].astype(str) + "_" + df_valid["v002"].astype(str)).to_numpy()
    features = ["hw1", "b4", "v025", "v106", "v190"]

    seed_everything(42)
    train, val, test = make_splits(y, groups, 42)

    preprocessor = joblib.load(anemia_dir / "preprocessor.joblib")
    x_test = preprocessor.transform(df_valid.iloc[test][features]).astype(np.float32)
    y_test = y[test]

    # Models
    dnn = BinaryDNN(14)
    dnn.load_state_dict(torch.load(anemia_dir / "dnn.pt", map_location="cpu", weights_only=False)["state_dict"])
    dnn.eval()
    with torch.no_grad():
        dnn_probs = torch.sigmoid(dnn(torch.from_numpy(x_test))).numpy()

    tabnet = TabNetClassifier()
    tabnet.load_model(str(anemia_dir / "tabnet.zip"))
    tabnet_probs = tabnet.predict_proba(x_test)[:, 1]

    ft = BinaryFTTransformer(14)
    ft.load_state_dict(torch.load(anemia_dir / "ft_transformer.pt", map_location="cpu", weights_only=False)["state_dict"])
    ft.eval()
    with torch.no_grad():
        ft_probs = torch.sigmoid(ft(torch.from_numpy(x_test))).numpy()

    df_anemia = pd.DataFrame({
        "y_true": y_test,
        "DNN_probability": dnn_probs,
        "TabNet_probability": tabnet_probs,
        "FT-Transformer_probability": ft_probs,
    })
    df_anemia.to_csv(cache_path, index=False)
    print(f"  Saved test predictions to {cache_path}")
    return df_anemia


def get_ari_predictions():
    """Obtain held-out test predictions for ARI-related illness."""
    cache_path = METRICS_DIR / "test_predictions_ari.csv"
    if cache_path.exists():
        print("  [Cache Hit] Loading ARI test predictions from cache.")
        return pd.read_csv(cache_path)

    print("  [Processing] Generating held-out test predictions for ARI...")
    from pytorch_tabnet.tab_model import TabNetClassifier
    from training.train_ari_model import make_splits, seed_everything
    from training.ari_target import build_ari_target_from_dataframe

    raw_path = ROOT_DIR / "data" / "raw" / "IAKR7EFL.DTA"
    ari_dir = ML_SERVICE_DIR / "models" / "nfhs_health_targets" / "ari" / "split80_20_seed42"

    cols = ["h31", "h31b", "h31c", "hw1", "b4", "v025", "v106", "v190", "v001", "v002"]
    raw = pd.read_stata(raw_path, columns=cols, convert_categoricals=False)

    labeled = build_ari_target_from_dataframe(raw)
    label_known = labeled["ARI"].notna().to_numpy()
    df_eligible = labeled.loc[label_known].reset_index(drop=True)
    y = df_eligible["ARI"].astype(np.int8).to_numpy()

    groups = (
        df_eligible["v001"].astype("Int64").astype(str)
        + "-"
        + df_eligible["v002"].astype("Int64").astype(str)
    ).to_numpy()
    features = ["hw1", "b4", "v025", "v106", "v190"]

    seed_everything(42)
    train, val, test = make_splits(y, groups, 42)

    preprocessor = joblib.load(ari_dir / "preprocessor.joblib")
    x_test = preprocessor.transform(df_eligible.iloc[test][features]).astype(np.float32)
    y_test = y[test]

    dnn = BinaryDNN(14)
    dnn.load_state_dict(torch.load(ari_dir / "dnn.pt", map_location="cpu", weights_only=False)["state_dict"])
    dnn.eval()
    with torch.no_grad():
        dnn_probs = torch.sigmoid(dnn(torch.from_numpy(x_test))).numpy()

    tabnet = TabNetClassifier()
    tabnet.load_model(str(ari_dir / "tabnet.zip"))
    tabnet_probs = tabnet.predict_proba(x_test)[:, 1]

    ft = BinaryFTTransformer(14)
    ft.load_state_dict(torch.load(ari_dir / "ft_transformer.pt", map_location="cpu", weights_only=False)["state_dict"])
    ft.eval()
    with torch.no_grad():
        ft_probs = torch.sigmoid(ft(torch.from_numpy(x_test))).numpy()

    df_ari = pd.DataFrame({
        "y_true": y_test,
        "DNN_probability": dnn_probs,
        "TabNet_probability": tabnet_probs,
        "FT-Transformer_probability": ft_probs,
    })
    df_ari.to_csv(cache_path, index=False)
    print(f"  Saved test predictions to {cache_path}")
    return df_ari


# ==============================================================================
# Step 2: Evaluation Metrics Calculation
# ==============================================================================

def compute_all_metrics(target_predictions):
    """
    Compute comprehensive metrics for each model and target.
    Returns metrics_df, confusion_df.
    """
    # Calibration thresholds from existing validation optimizations
    thresholds = {
        "Stunting": {"DNN": 0.40, "TabNet": 0.30, "FT-Transformer": 0.40},
        "Wasting": {"DNN": 0.45, "TabNet": 0.20, "FT-Transformer": 0.50},
        "Anaemia": {"DNN": 0.5037, "TabNet": 0.5071, "FT-Transformer": 0.5184},
        "ARI-related illness": {"DNN": 0.5657, "TabNet": 0.5347, "FT-Transformer": 0.5239},
    }

    metrics_rows = []
    cm_rows = []

    for target_name, df in target_predictions.items():
        y_true = df["y_true"].values
        pos_count = int(y_true.sum())
        neg_count = int(len(y_true) - pos_count)
        n_samples = int(len(y_true))

        for model_name in ["DNN", "TabNet", "FT-Transformer"]:
            col_name = f"{model_name}_probability"
            if col_name not in df:
                continue

            probs = df[col_name].values
            th = thresholds.get(target_name, {}).get(model_name, 0.50)
            preds = (probs >= th).astype(int)

            acc = float(accuracy_score(y_true, preds))
            prec = float(precision_score(y_true, preds, zero_division=0))
            rec = float(recall_score(y_true, preds, zero_division=0))
            f1 = float(f1_score(y_true, preds, zero_division=0))

            roc_auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else np.nan
            ap = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) == 2 else np.nan

            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            metrics_rows.append({
                "target": target_name,
                "model": model_name,
                "sample_count": n_samples,
                "positive_count": pos_count,
                "negative_count": neg_count,
                "threshold": round(th, 4),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "specificity": round(spec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "average_precision": round(ap, 4),
            })

            cm_rows.append({
                "target": target_name,
                "model": model_name,
                "sample_count": n_samples,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "threshold": round(th, 4),
                "accuracy": round(acc, 4),
            })

    metrics_df = pd.DataFrame(metrics_rows)
    cm_df = pd.DataFrame(cm_rows)

    metrics_df.to_csv(METRICS_DIR / "model_evaluation_metrics.csv", index=False)
    cm_df.to_csv(METRICS_DIR / "confusion_matrices.csv", index=False)
    print(f"  Saved evaluation metrics to {METRICS_DIR / 'model_evaluation_metrics.csv'}")
    print(f"  Saved confusion matrices to {METRICS_DIR / 'confusion_matrices.csv'}")

    return metrics_df, cm_df


# ==============================================================================
# Figure 1: Distribution of Health Outcomes
# ==============================================================================

def plot_health_outcome_distribution():
    """Figure 1: Distribution of Health Outcomes across all four targets."""
    print("Generating Figure 1: health_outcome_distribution.png...")

    # Data counts from actual NFHS-5 dataset files
    data = [
        {"target": "Stunting", "dataset": "NFHS-5 (dhs_clean.parquet)", "total": 198849, "pos": 70742, "neg": 128107},
        {"target": "Wasting", "dataset": "NFHS-5 (dhs_clean.parquet)", "total": 198849, "pos": 37288, "neg": 161561},
        {"target": "Anaemia", "dataset": "NFHS-5 (IAKR7EFL.DTA)", "total": 183855, "pos": 123462, "neg": 60393},
        {"target": "ARI-related illness", "dataset": "NFHS-5 (IAKR7EFL.DTA)", "total": 223647, "pos": 5224, "neg": 218423},
    ]

    dist_df = pd.DataFrame(data)
    dist_df["pos_pct"] = (dist_df["pos"] / dist_df["total"]) * 100
    dist_df["neg_pct"] = (dist_df["neg"] / dist_df["total"]) * 100
    dist_df.to_csv(METRICS_DIR / "target_class_distributions.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    targets = [d["target"] for d in data]
    x = np.arange(len(targets))
    width = 0.35

    pos_pcts = dist_df["pos_pct"].tolist()
    neg_pcts = dist_df["neg_pct"].tolist()
    pos_counts = [d["pos"] for d in data]
    neg_counts = [d["neg"] for d in data]
    totals = [d["total"] for d in data]

    rects1 = ax.bar(x - width/2, pos_pcts, width, label="Positive Cases (Outcome Present)", color="#d9534f", edgecolor="#a94442", linewidth=1.2)
    rects2 = ax.bar(x + width/2, neg_pcts, width, label="Negative Cases (Outcome Absent)", color="#337ab7", edgecolor="#286090", linewidth=1.2)

    ax.set_ylabel("Class Percentage (%)", fontweight="bold")
    ax.set_title("Distribution of Child Health Outcomes in NFHS-5 Survey Dataset", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(N = {n:,})" for t, n in zip(targets, totals)], fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate bars with counts and percentages
    for rect, count in zip(rects1, pos_counts):
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%\n({count:,})",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#a94442")

    for rect, count in zip(rects2, neg_counts):
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%\n({count:,})",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#286090")

    fig.tight_layout()
    out_path = PLOTS_DIR / "health_outcome_distribution.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [Output] {out_path}")
    return out_path


# ==============================================================================
# Figure 2: Overall Model Performance Comparison
# ==============================================================================

def plot_overall_model_performance(metrics_df):
    """Figure 2: Grouped bar chart comparing DNN, TabNet, and FT-Transformer across outcomes."""
    print("Generating Figure 2: overall_model_performance.png...")

    metrics_to_plot = [
        ("Precision", "precision"),
        ("Recall (Sensitivity)", "recall"),
        ("F1-Score", "f1_score"),
        ("ROC-AUC", "roc_auc"),
    ]

    targets = ["Stunting", "Wasting", "Anaemia", "ARI-related illness"]
    models = ["DNN", "TabNet", "FT-Transformer"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_title, metric_key) in enumerate(metrics_to_plot):
        ax = axes[idx]
        x = np.arange(len(targets))
        width = 0.25

        for m_idx, model in enumerate(models):
            vals = []
            for target in targets:
                row = metrics_df[(metrics_df["target"] == target) & (metrics_df["model"] == model)]
                vals.append(row[metric_key].values[0] if len(row) > 0 else 0.0)

            rects = ax.bar(x + (m_idx - 1) * width, vals, width, label=model,
                           color=MODEL_COLORS[model], edgecolor="black", linewidth=0.8, alpha=0.9)

            # Annotate value above bar
            for rect, val in zip(rects, vals):
                if val > 0:
                    ax.annotate(f"{val:.2f}",
                                xy=(rect.get_x() + rect.get_width() / 2, val),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7.5, rotation=0)

        ax.set_ylabel(metric_title, fontweight="bold")
        ax.set_title(f"({chr(97 + idx)}) {metric_title} Comparison", fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontweight="bold", fontsize=9.5)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if idx == 0:
            ax.legend(frameon=True, loc="upper right")

    fig.suptitle("Overall Deep Learning Model Performance Across Four NFHS-5 Health Outcomes", fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = PLOTS_DIR / "overall_model_performance.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [Output] {out_path}")
    return out_path


# ==============================================================================
# Figure 3: Target-Wise Model Comparison
# ==============================================================================

def plot_target_wise_model_comparison(metrics_df):
    """Figure 3: Four-panel figure comparing models within each target."""
    print("Generating Figure 3: target_wise_model_comparison.png...")

    targets = [
        ("Stunting", "Stunting (HAZ < -2 SD)"),
        ("Wasting", "Wasting (WHZ < -2 SD)"),
        ("Anaemia", "Anaemia (Hb < 11.0 g/dL)"),
        ("ARI-related illness", "ARI-related Illness (Acute Respiratory Infection)")
    ]
    eval_metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1_score"),
        ("ROC-AUC", "roc_auc")
    ]
    models = ["DNN", "TabNet", "FT-Transformer"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    for idx, (target_key, target_title) in enumerate(targets):
        ax = axes[idx]
        x = np.arange(len(eval_metrics))
        width = 0.25

        for m_idx, model in enumerate(models):
            vals = []
            for _, m_key in eval_metrics:
                row = metrics_df[(metrics_df["target"] == target_key) & (metrics_df["model"] == model)]
                vals.append(row[m_key].values[0] if len(row) > 0 else 0.0)

            rects = ax.bar(x + (m_idx - 1) * width, vals, width, label=model,
                           color=MODEL_COLORS[model], edgecolor="black", linewidth=0.8, alpha=0.9)

            for rect, val in zip(rects, vals):
                if val > 0:
                    ax.annotate(f"{val:.2f}",
                                xy=(rect.get_x() + rect.get_width() / 2, val),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7.5)

        ax.set_ylabel("Metric Value", fontweight="bold")
        ax.set_title(f"({chr(97 + idx)}) {target_title}", fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in eval_metrics], fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if idx == 0:
            ax.legend(frameon=True, loc="upper right")

    fig.suptitle("Target-Wise Performance Comparison Across Deep Learning Models", fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = PLOTS_DIR / "target_wise_model_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [Output] {out_path}")
    return out_path


# ==============================================================================
# Figure 4: ROC Curves
# ==============================================================================

def plot_roc_curves(target_predictions):
    """Figure 4: Four-panel ROC curves with ROC-AUC in legend and random baseline."""
    print("Generating Figure 4: roc_curves.png...")

    targets = [
        ("Stunting", "Stunting"),
        ("Wasting", "Wasting"),
        ("Anaemia", "Anaemia"),
        ("ARI-related illness", "ARI-related Illness")
    ]
    models = ["DNN", "TabNet", "FT-Transformer"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, (target_key, target_title) in enumerate(targets):
        ax = axes[idx]
        df = target_predictions[target_key]
        y_true = df["y_true"].values

        # Plot Random Baseline
        ax.plot([0, 1], [0, 1], linestyle="--", color=BASELINE_COLOR, label="Random Classifier (AUC = 0.50)", linewidth=1.5)

        if len(np.unique(y_true)) == 2:
            for model in models:
                col = f"{model}_probability"
                if col in df:
                    probs = df[col].values
                    fpr, tpr, _ = roc_curve(y_true, probs)
                    score = roc_auc_score(y_true, probs)
                    ax.plot(fpr, tpr, label=f"{model} (AUC = {score:.4f})",
                            color=MODEL_COLORS[model], linewidth=2.0, alpha=0.95)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
        ax.set_title(f"({chr(97 + idx)}) {target_title} ROC Curves", fontweight="bold", pad=8)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92)
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Receiver Operating Characteristic (ROC) Curves Across Four Health Outcomes", fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = PLOTS_DIR / "roc_curves.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [Output] {out_path}")
    return out_path


# ==============================================================================
# Figure 5: Precision–Recall Curves
# ==============================================================================

def plot_precision_recall_curves(target_predictions):
    """Figure 5: Four-panel Precision-Recall curves with AP in legend and prevalence baseline."""
    print("Generating Figure 5: precision_recall_curves.png...")

    targets = [
        ("Stunting", "Stunting"),
        ("Wasting", "Wasting"),
        ("Anaemia", "Anaemia"),
        ("ARI-related illness", "ARI-related Illness")
    ]
    models = ["DNN", "TabNet", "FT-Transformer"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, (target_key, target_title) in enumerate(targets):
        ax = axes[idx]
        df = target_predictions[target_key]
        y_true = df["y_true"].values
        prevalence = float(np.mean(y_true))

        # Horizontal Prevalence Baseline
        ax.axhline(prevalence, linestyle="--", color=BASELINE_COLOR,
                   label=f"Prevalence Baseline ({prevalence:.1%})", linewidth=1.5)

        if len(np.unique(y_true)) == 2:
            for model in models:
                col = f"{model}_probability"
                if col in df:
                    probs = df[col].values
                    p_curve, r_curve, _ = precision_recall_curve(y_true, probs)
                    ap_score = average_precision_score(y_true, probs)
                    ax.plot(r_curve, p_curve, label=f"{model} (AP = {ap_score:.4f})",
                            color=MODEL_COLORS[model], linewidth=2.0, alpha=0.95)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("Recall (Sensitivity)", fontweight="bold")
        ax.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
        ax.set_title(f"({chr(97 + idx)}) {target_title} Precision-Recall Curves", fontweight="bold", pad=8)
        ax.legend(loc="upper right" if idx < 3 else "upper right", frameon=True, framealpha=0.92)
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("Precision-Recall Curves Across Four NFHS-5 Health Outcomes", fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = PLOTS_DIR / "precision_recall_curves.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [Output] {out_path}")
    return out_path


# ==============================================================================
# Main Orchestration Pipeline
# ==============================================================================

def main():
    print("=" * 80)
    print("   AUTOMATED RESEARCH PAPER FIGURE & METRICS GENERATION PIPELINE")
    print("   NFHS-5 Child Health Prediction System (4 Targets x 3 Deep Learning Models)")
    print("=" * 80)
    started = time.time()

    # Step 1: Collect Test Predictions
    print("\n[Step 1/3] Loading or computing held-out test predictions...")
    df_stunting, df_wasting = get_stunting_wasting_predictions()
    df_anemia = get_anemia_predictions()
    df_ari = get_ari_predictions()

    target_predictions = {
        "Stunting": df_stunting,
        "Wasting": df_wasting,
        "Anaemia": df_anemia,
        "ARI-related illness": df_ari,
    }

    # Step 2: Compute Metrics & Save CSVs
    print("\n[Step 2/3] Calculating rigorous test metrics and confusion matrices...")
    metrics_df, cm_df = compute_all_metrics(target_predictions)

    # Step 3: Generate Figures
    print("\n[Step 3/3] Rendering 5 publication-quality research figures (300 DPI)...")
    fig1 = plot_health_outcome_distribution()
    fig2 = plot_overall_model_performance(metrics_df)
    fig3 = plot_target_wise_model_comparison(metrics_df)
    fig4 = plot_roc_curves(target_predictions)
    fig5 = plot_precision_recall_curves(target_predictions)

    elapsed = time.time() - started
    print("\n" + "=" * 80)
    print(f"   SUCCESSFULLY GENERATED ALL FIGURES & METRICS IN {elapsed:.1f}s")
    print("=" * 80)
    print("Generated Figures (outputs/plots/):")
    print(f"  1. {fig1}")
    print(f"  2. {fig2}")
    print(f"  3. {fig3}")
    print(f"  4. {fig4}")
    print(f"  5. {fig5}")
    print("\nGenerated Numerical CSVs (outputs/metrics/):")
    print(f"  - {METRICS_DIR / 'model_evaluation_metrics.csv'}")
    print(f"  - {METRICS_DIR / 'target_class_distributions.csv'}")
    print(f"  - {METRICS_DIR / 'confusion_matrices.csv'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
