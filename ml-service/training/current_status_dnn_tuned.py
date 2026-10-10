"""Tuned current-status DNN experiment.

Uses only measurement-time variables and train-fitted scaling. This remains
separate from the early-risk DNN and never uses hw70/hw72 as inputs.
"""

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "IAKR7EFL.DTA"
OUT = ROOT / "ml-service" / "models" / "retrained" / "current_status_dnn_tuned_seed42"


def features(frame):
    age = frame["hw1"].astype(float).to_numpy()
    sex = frame["b4"].astype(float).to_numpy()
    weight = frame["hw2"].astype(float).to_numpy() / 10.0
    height = frame["hw3"].astype(float).to_numpy() / 10.0
    height_m = height / 100.0
    return np.column_stack([
        age, sex, weight, height, frame["hw5"].astype(float),
        frame["hw8"].astype(float), weight / np.maximum(height_m ** 2, 1e-6),
        height / np.maximum(age + 6.0, 1.0), weight / np.maximum(age + 6.0, 1.0),
    ]).astype(np.float32)


class Model(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, width), nn.BatchNorm1d(width), nn.SiLU(),
            nn.Dropout(0.03), nn.Linear(width, width), nn.BatchNorm1d(width),
            nn.SiLU(), nn.Dropout(0.03), nn.Linear(width, 128), nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, value):
        return self.net(value)


def main(seed=42, epochs=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    start = time.time()
    raw = pd.read_stata(
        RAW,
        columns=["caseid", "hw1", "b4", "hw2", "hw3", "hw5", "hw8", "hw70", "hw72"],
        convert_categoricals=False,
    )
    valid = (
        raw.hw70.between(-600, 600) & raw.hw72.between(-600, 600)
        & raw.hw2.between(20, 7000) & raw.hw3.between(300, 1300)
        & raw.hw1.between(0, 59)
    )
    raw = raw.loc[valid].reset_index(drop=True)
    y = np.column_stack([
        (raw.hw70.to_numpy() <= -200).astype(np.float32),
        (raw.hw72.to_numpy() <= -200).astype(np.float32),
    ])
    x = features(raw)
    groups = raw.caseid.astype(str).to_numpy()
    positions = np.arange(len(raw))
    train, rem = next(GroupShuffleSplit(n_splits=1, test_size=.30, random_state=seed).split(positions, y[:, 0], groups))
    vr, tr = next(GroupShuffleSplit(n_splits=1, test_size=.50, random_state=seed).split(rem, y[rem, 0], groups[rem]))
    validation, test = rem[vr], rem[tr]
    scaler = StandardScaler().fit(x[train])
    x = scaler.transform(x).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    pos = y[train].sum(axis=0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(train) - pos) / pos, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=.5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train])),
                        batch_size=2048, shuffle=True, generator=torch.Generator().manual_seed(seed))
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
            loss = criterion(model(torch.from_numpy(x[validation]).to(device)),
                             torch.from_numpy(y[validation]).to(device)).item()
        scheduler.step(loss)
        if loss < best_loss:
            best_loss, stale = loss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 12:
                break
    model.load_state_dict(best)

    def probability(indices):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(x[indices]).to(device))).cpu().numpy()

    vp, tp = probability(validation), probability(test)
    thresholds = [max(np.arange(.1, .91, .01),
                      key=lambda t: accuracy_score(y[validation, i], vp[:, i] >= t))
                  for i in range(2)]
    primitive = np.column_stack([tp[:, i] >= thresholds[i] for i in range(2)]).astype(int)
    prediction = np.column_stack([primitive, np.logical_or(primitive[:, 0], primitive[:, 1]).astype(int)])
    truth = np.column_stack([y[test].astype(int), np.logical_or(y[test, 0], y[test, 1]).astype(int)])
    targets = ["Stunting", "Wasting", "Malnutrition"]
    metrics = {
        "task": "current nutritional status",
        "model": "tuned PyTorch DNN",
        "not_comparable_to": "early-risk prediction",
        "dataset_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "features": ["age", "sex", "weight", "height", "measurement_status", "BMI_like", "age_interactions"],
        "excluded_target_variables": ["hw70", "hw72"],
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "thresholds_selected_on_validation": thresholds,
        "test_exact_match_accuracy": float(accuracy_score(truth, prediction)),
        "test_macro_f1": float(f1_score(truth, prediction, average="macro")),
        "per_target_accuracy": {t: float(accuracy_score(truth[:, i], prediction[:, i])) for i, t in enumerate(targets)},
        "confusion_matrices": {t: confusion_matrix(truth[:, i], prediction[:, i], labels=[0, 1]).tolist() for i, t in enumerate(targets)},
        "training_seconds": time.time() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "scaler": scaler, "thresholds": thresholds}, OUT / "model.pt")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
