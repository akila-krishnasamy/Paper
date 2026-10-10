"""Current-status DNN using contemporaneous NFHS-5 measurements.

This is not the early-risk task. Target-defining z-scores are excluded from
the inputs; measured height/length and weight are used as current-status
predictors.
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "IAKR7EFL.DTA"
OUT = ROOT / "ml-service" / "models" / "retrained" / "current_status_dnn_seed42"
FEATURES = [
    "child_age_months", "child_sex", "weight_kg", "height_cm",
    "height_measurement_status", "weight_measurement_status",
]


class StatusDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(FEATURES), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.network(x)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_features(frame):
    return pd.DataFrame({
        "child_age_months": frame["hw1"].astype(float),
        "child_sex": frame["b4"].astype(float),
        "weight_kg": frame["hw2"].astype(float) / 10.0,
        "height_cm": frame["hw3"].astype(float) / 10.0,
        "height_measurement_status": frame["hw5"].astype(float),
        "weight_measurement_status": frame["hw8"].astype(float),
    }).to_numpy(dtype=np.float32)


def probs(model, x, device, batch_size=2048):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            values.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(values)


def main(seed=42, epochs=80):
    seed_all(seed)
    start = time.time()
    raw = pd.read_stata(
        RAW,
        columns=["caseid", "hw1", "b4", "hw2", "hw3", "hw5", "hw8", "hw70", "hw72"],
        convert_categoricals=False,
    )
    valid = (
        raw["hw70"].between(-600, 600)
        & raw["hw72"].between(-600, 600)
        & raw["hw2"].between(20, 7000)
        & raw["hw3"].between(300, 1300)
        & raw["hw1"].between(0, 59)
    )
    raw = raw.loc[valid].reset_index(drop=True)
    labels = np.column_stack([
        (raw["hw70"].to_numpy() <= -200).astype(np.float32),
        (raw["hw72"].to_numpy() <= -200).astype(np.float32),
    ])
    x = make_features(raw)
    groups = raw["caseid"].astype(str).to_numpy()
    positions = np.arange(len(raw))
    train, remaining = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        .split(positions, labels[:, 0], groups)
    )
    val_relative, test_relative = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        .split(remaining, labels[remaining, 0], groups[remaining])
    )
    validation = remaining[val_relative]
    test = remaining[test_relative]
    train_x, val_x, test_x = x[train], x[validation], x[test]
    train_y, val_y, test_y = labels[train], labels[validation], labels[test]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StatusDNN().to(device)
    positive = train_y.sum(axis=0)
    negative = len(train_y) - positive
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / positive, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=1024, shuffle=True, generator=generator,
    )
    best_state = None
    best_loss = float("inf")
    stagnant = 0
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(torch.from_numpy(val_x).to(device)),
                torch.from_numpy(val_y).to(device),
            ).item()
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= 10:
                break
    model.load_state_dict(best_state)
    val_prob = probs(model, val_x, device)
    test_prob = probs(model, test_x, device)
    thresholds = [
        max(
            np.arange(0.10, 0.91, 0.01),
            key=lambda threshold: accuracy_score(
                val_y[:, col], val_prob[:, col] >= threshold
            ),
        )
        for col in range(2)
    ]
    primitive = np.column_stack([
        test_prob[:, col] >= thresholds[col] for col in range(2)
    ]).astype(int)
    predictions = np.column_stack([
        primitive, np.logical_or(primitive[:, 0], primitive[:, 1]).astype(int)
    ])
    truth = np.column_stack([
        test_y.astype(int), np.logical_or(test_y[:, 0], test_y[:, 1]).astype(int)
    ])
    metrics = {
        "task": "current nutritional status",
        "model": "PyTorch DNN",
        "not_comparable_to": "early-risk prediction",
        "dataset_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "seed": seed,
        "features": FEATURES,
        "excluded_target_variables": ["hw70", "hw72"],
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "thresholds_selected_on_validation": thresholds,
        "test_exact_match_accuracy": float(accuracy_score(truth, predictions)),
        "test_macro_f1": float(f1_score(truth, predictions, average="macro")),
        "per_target_accuracy": {
            target: float(accuracy_score(truth[:, col], predictions[:, col]))
            for col, target in enumerate(["Stunting", "Wasting", "Malnutrition"])
        },
        "confusion_matrices": {
            target: confusion_matrix(truth[:, col], predictions[:, col], labels=[0, 1]).tolist()
            for col, target in enumerate(["Stunting", "Wasting", "Malnutrition"])
        },
        "training_seconds": time.time() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "features": FEATURES,
        "thresholds": thresholds,
        "seed": seed,
    }, OUT / "model.pt")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
