"""Leakage-free, reproducible DNN retraining for the NFHS-5 task.

This script deliberately writes to a new run directory. Historical weights and
metric files are evidence and are never overwritten.
"""

import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import (  # noqa: E402
    ALL_FEATURES,
    TARGET_CONDITIONS,
    prepare_data_splits,
)
from training.train_dnn import TabularDNN, TabularDataset  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "dhs_clean.parquet"
RUN_ID = "dnn_retrained_consistent_seed42"
OUTPUT = ROOT / "ml-service" / "models" / "retrained" / RUN_ID


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probabilities(model, loader, device):
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            values.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(values, axis=0)


class PrimitiveTargetDataset(torch.utils.data.Dataset):
    def __init__(self, x_frame, y_frame):
        self.x = torch.tensor(
            x_frame[ALL_FEATURES].values.astype(np.float32), dtype=torch.float32
        )
        self.y = torch.tensor(
            y_frame[["Stunting", "Wasting"]].values.astype(np.float32),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class PrimitiveDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.network(x)


def select_validation_thresholds(y_true, probs):
    """Select thresholds jointly for validation exact-match accuracy only."""
    grid = np.arange(0.10, 0.91, 0.02)
    thresholds = np.full(2, 0.5, dtype=float)

    def score(candidate):
        primitive_prediction = (probs >= candidate).astype(int)
        prediction = np.column_stack([
            primitive_prediction,
            np.logical_or(
                primitive_prediction[:, 0], primitive_prediction[:, 1]
            ).astype(int),
        ])
        return accuracy_score(y_true, prediction)

    best_score = score(thresholds)
    improved = True
    while improved:
        improved = False
        for index in range(2):
            candidates = []
            for threshold in grid:
                trial = thresholds.copy()
                trial[index] = threshold
                candidates.append((score(trial), -abs(threshold - 0.5), threshold))
            candidate = max(candidates)
            if candidate[0] > best_score:
                thresholds[index] = candidate[2]
                best_score = candidate[0]
                improved = True
    return {
        "Stunting": float(thresholds[0]),
        "Wasting": float(thresholds[1]),
        "Malnutrition": "derived as Stunting OR Wasting",
    }


def evaluate(y_true, probs, thresholds):
    primitive_predictions = np.column_stack([
        (probs[:, 0] >= thresholds["Stunting"]).astype(int),
        (probs[:, 1] >= thresholds["Wasting"]).astype(int),
    ])
    predictions = np.column_stack([
        primitive_predictions,
        np.logical_or(
            primitive_predictions[:, 0], primitive_predictions[:, 1]
        ).astype(int),
    ])
    full_probs = np.column_stack([
        probs,
        1.0 - ((1.0 - probs[:, 0]) * (1.0 - probs[:, 1])),
    ])
    per_target = {}
    matrices = {}
    for i, target in enumerate(TARGET_CONDITIONS):
        truth = y_true[:, i]
        pred = predictions[:, i]
        probability = full_probs[:, i]
        matrix = confusion_matrix(truth, pred, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        per_target[target] = {
            "accuracy": float(accuracy_score(truth, pred)),
            "precision": float(precision_score(truth, pred, zero_division=0)),
            "recall_sensitivity": float(recall_score(truth, pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(f1_score(truth, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(truth, probability)),
            "pr_auc": float(average_precision_score(truth, probability)),
            "support_positive": int(truth.sum()),
            "support_total": int(len(truth)),
        }
        matrices[target] = matrix.tolist()
    return {
        "exact_match_accuracy": float(accuracy_score(y_true, predictions)),
        "hamming_accuracy": float(1.0 - np.mean(y_true != predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_roc_auc": float(roc_auc_score(y_true, full_probs, average="macro")),
        "macro_pr_auc": float(average_precision_score(y_true, full_probs, average="macro")),
        "per_target": per_target,
        "confusion_matrices": matrices,
    }


def main(seed=42, epochs=60, batch_size=512):
    seed_everything(seed)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    start = time.time()

    (x_train, y_train), (x_val, y_val), (x_test, y_test), preprocessor = (
        prepare_data_splits(random_state=seed)
    )
    split_manifest = {
        "dataset_path": str(DATASET),
        "dataset_sha256": sha256(DATASET),
        "seed": seed,
        "train_indices": x_train.index.tolist(),
        "validation_indices": x_val.index.tolist(),
        "test_indices": x_test.index.tolist(),
        "targets": TARGET_CONDITIONS,
        "model_outputs": ["Stunting", "Wasting"],
        "composite_definition": "Malnutrition = Stunting OR Wasting",
        "train_labels": y_train.values.tolist(),
        "validation_labels": y_val.values.tolist(),
        "test_labels": y_test.values.tolist(),
    }
    (OUTPUT / "split_manifest.json").write_text(json.dumps(split_manifest))
    joblib.dump(preprocessor, OUTPUT / "preprocessor.pkl")

    train_loader = DataLoader(
        PrimitiveTargetDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(PrimitiveTargetDataset(x_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(PrimitiveTargetDataset(x_test, y_test), batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrimitiveDNN(input_dim=len(ALL_FEATURES)).to(device)
    positive_counts = y_train[["Stunting", "Wasting"]].sum().values
    negative_counts = len(y_train) - positive_counts
    pos_weight = torch.tensor(
        negative_counts / positive_counts, dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_loss = float("inf")
    best_state = None
    patience = 5
    stagnant = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch.to(device)), y_batch.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                val_loss += criterion(
                    model(x_batch.to(device)), y_batch.to(device)
                ).item() * len(x_batch)
        val_loss /= len(x_val)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                break

    model.load_state_dict(best_state)
    train_probs = probabilities(model, DataLoader(PrimitiveTargetDataset(x_train, y_train), batch_size=batch_size), device)
    val_probs = probabilities(model, val_loader, device)
    test_probs = probabilities(model, test_loader, device)
    thresholds = select_validation_thresholds(y_val.values, val_probs)
    train_metrics = evaluate(y_train.values, train_probs, thresholds)
    val_metrics = evaluate(y_val.values, val_probs, thresholds)
    test_metrics = evaluate(y_test.values, test_probs, thresholds)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": len(ALL_FEATURES),
        "num_targets": 2,
        "dropout": 0.2,
        "features": ALL_FEATURES,
        "targets": TARGET_CONDITIONS,
        "seed": seed,
        "run_id": RUN_ID,
    }
    torch.save(checkpoint, OUTPUT / "model.pt")
    payload = {
        "model_name": "DNN retrained reproducibly",
        "run_id": RUN_ID,
        "dataset_sha256": split_manifest["dataset_sha256"],
        "split_seed": seed,
        "sample_counts": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "features": ALL_FEATURES,
        "targets": TARGET_CONDITIONS,
        "thresholds_selected_on_validation": thresholds,
        "validation_metrics": val_metrics,
        "training_metrics": train_metrics,
        "final_test_metrics": test_metrics,
        "training_seconds": time.time() - start,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Training metrics are not held-out performance. Test metrics were computed once after model selection.",
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "run_id": RUN_ID,
        "training_exact_match_accuracy": train_metrics["exact_match_accuracy"],
        "validation_exact_match_accuracy": val_metrics["exact_match_accuracy"],
        "test_exact_match_accuracy": test_metrics["exact_match_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_macro_roc_auc": test_metrics["macro_roc_auc"],
    }, indent=2))


if __name__ == "__main__":
    main()
