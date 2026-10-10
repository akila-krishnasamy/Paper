"""Train leakage-controlled NFHS-5 anemia and ARI classifiers.

This script is deliberately separate from the malnutrition experiments.  The
targets are constructed from raw DHS variables, while the predictors are
limited to child, household, and maternal characteristics available before
the reported outcome.  It writes a complete split manifest and prediction-
derived metrics for XGBoost, a PyTorch DNN, and a numeric FT-Transformer.

Usage:
    python train_nfhs_health_targets.py --target anemia
    python train_nfhs_health_targets.py --target ari
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "IAKR7EFL.DTA"
SEED = 42
FEATURES = [
    "hw1", "b4", "b5", "b8", "b11", "b12", "b16",
    "v024", "v025", "v106", "v190",
]
TARGET_VARS = {
    "anemia": ["hw57"],
    "ari": ["h31", "h31b", "h31c"],
}
OUT_ROOT = ROOT / "ml-service" / "models" / "nfhs_health_targets"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def numeric(frame):
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    # Median values are fit on the training partition later.
    return values.to_numpy(dtype=np.float32)


def make_anemia(raw):
    valid = raw.hw57.isin([1, 2, 3, 4])
    y = raw.hw57.isin([1, 2, 3]).astype(int)
    return valid.to_numpy(), y.to_numpy()


def make_ari(raw):
    # h31 uses 2 for yes in this NFHS-5 file. h31b uses 1 for yes.
    # h31c: 1 chest only, 2 blocked nose only, 3 both.
    cough_known = raw.h31.isin([0, 2])
    cough = raw.h31.eq(2)
    rapid_known = raw.h31b.isin([0, 1])
    chest_known = raw.h31c.isin([1, 2, 3])
    valid = cough_known & ((~cough) | (rapid_known & (~raw.h31b.eq(1) | chest_known)))
    y = (cough & raw.h31b.eq(1) & raw.h31c.isin([1, 3])).astype(int)
    return valid.to_numpy(), y.to_numpy()


def split_indices(raw, y, seed):
    groups = raw.caseid.astype(str).to_numpy()
    indices = np.arange(len(raw))
    train, remainder = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        .split(indices, y, groups)
    )
    val_rel, test_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        .split(remainder, y[remainder], groups[remainder])
    )
    return train, remainder[val_rel], remainder[test_rel]


def binary_metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "majority_baseline_accuracy": float(max(np.mean(y == 0), np.mean(y == 1))),
    }
    return result


class DNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(0.05), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class NumericFTTransformer(nn.Module):
    def __init__(self, n_features, token_dim=16, heads=2, layers=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, token_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, token_dim))
        block = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=heads, dim_feedforward=64,
            dropout=0.10, activation="gelu", batch_first=True, norm_first=True,
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return self.head(self.encoder(torch.cat([cls, tokens], dim=1))[:, 0]).squeeze(1)


def torch_fit(model, x_train, y_train, x_val, y_val, seed, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    positive = float(y_train.sum())
    weight = torch.tensor([(len(y_train) - positive) / max(positive, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=8192, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    best, best_loss, stale = None, float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(torch.from_numpy(x_val).to(device)),
                torch.from_numpy(y_val.astype(np.float32)).to(device),
            ).item()
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best)
    return model, device


def torch_probs(model, device, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()


def main(target, seed=SEED):
    seed_all(seed)
    started = time.time()
    columns = sorted(set(["caseid", *FEATURES, *TARGET_VARS[target], "v005"]))
    raw = pd.read_stata(RAW, columns=columns, convert_categoricals=False)
    valid, y = make_anemia(raw) if target == "anemia" else make_ari(raw)
    raw = raw.loc[valid].reset_index(drop=True)
    y = y[valid].astype(int)
    x_raw = numeric(raw)
    train, validation, test = split_indices(raw, y, seed)
    medians = np.nanmedian(x_raw[train], axis=0)
    x_raw = np.where(np.isnan(x_raw), medians, x_raw)
    scaler = StandardScaler().fit(x_raw[train])
    x = scaler.transform(x_raw).astype(np.float32)
    out = OUT_ROOT / target / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "split_manifest.npz", train=train, validation=validation, test=test)
    (out / "preprocessing.json").write_text(json.dumps({
        "features": FEATURES, "train_medians": medians.tolist(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
    }, indent=2))

    results = {}
    xgb = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, min_child_weight=3, reg_lambda=2,
        random_state=seed, n_jobs=4, eval_metric="logloss",
    )
    xgb.fit(x[train], y[train])
    xgb_prob = xgb.predict_proba(x[test])[:, 1]
    xgb_val = xgb.predict_proba(x[validation])[:, 1]
    threshold = max(np.arange(0.20, 0.81, 0.01),
                    key=lambda t: accuracy_score(y[validation], xgb_val >= t))
    results["xgboost"] = binary_metrics(y[test], xgb_prob >= threshold, xgb_prob)
    joblib.dump(xgb, out / "xgboost.joblib")

    dnn, device = torch_fit(DNN(x.shape[1]), x[train], y[train], x[validation], y[validation], seed)
    dnn_prob = torch_probs(dnn, device, x[test])
    dnn_val = torch_probs(dnn, device, x[validation])
    dnn_threshold = max(np.arange(0.20, 0.81, 0.01),
                        key=lambda t: accuracy_score(y[validation], dnn_val >= t))
    results["dnn"] = binary_metrics(y[test], dnn_prob >= dnn_threshold, dnn_prob)
    torch.save({"state_dict": dnn.state_dict(), "features": FEATURES,
                "threshold": float(dnn_threshold)}, out / "dnn.pt")

    ft, device = torch_fit(NumericFTTransformer(x.shape[1]), x[train], y[train],
                           x[validation], y[validation], seed)
    ft_prob = torch_probs(ft, device, x[test])
    ft_val = torch_probs(ft, device, x[validation])
    ft_threshold = max(np.arange(0.20, 0.81, 0.01),
                       key=lambda t: accuracy_score(y[validation], ft_val >= t))
    results["ft_transformer"] = binary_metrics(y[test], ft_prob >= ft_threshold, ft_prob)
    torch.save({"state_dict": ft.state_dict(), "features": FEATURES,
                "threshold": float(ft_threshold)}, out / "ft_transformer.pt")

    payload = {
        "target": target,
        "target_definition": (
            "hw57 in {1,2,3}=anemia and 4=not anemia"
            if target == "anemia"
            else "h31=2 AND h31b=1 AND h31c in {1,3}; no cough/rapid breathing is negative"
        ),
        "raw_dataset": str(RAW.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "seed": seed, "features": FEATURES, "excluded_target_variables": TARGET_VARS[target],
        "counts": {"eligible": len(y), "train": len(train), "validation": len(validation), "test": len(test)},
        "positive_counts": {"all": int(y.sum()), "train": int(y[train].sum()),
                            "validation": int(y[validation].sum()), "test": int(y[test].sum())},
        "thresholds_selected_on_validation": {
            "xgboost": float(threshold), "dnn": float(dnn_threshold),
            "ft_transformer": float(ft_threshold),
        },
        "models": results, "created_unix": time.time(),
        "training_seconds": time.time() - started,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["anemia", "ari"], required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(args.target, args.seed)
