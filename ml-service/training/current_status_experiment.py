"""Current-status NFHS-5 experiment using contemporaneous measurements.

This is a separate task from early-risk prediction. It intentionally uses
measured child weight and height/length, but never uses hw70/hw72 as inputs.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "IAKR7EFL.DTA"
OUT = ROOT / "ml-service" / "models" / "retrained" / "current_status_xgboost_seed42"


def main(seed=42):
    start = time.time()
    columns = ["caseid", "hw1", "b4", "hw2", "hw3", "hw5", "hw8", "hw70", "hw72"]
    raw = pd.read_stata(RAW, columns=columns, convert_categoricals=False)
    valid = (
        raw["hw70"].between(-600, 600)
        & raw["hw72"].between(-600, 600)
        & raw["hw2"].between(20, 7000)
        & raw["hw3"].between(300, 1300)
        & raw["hw1"].between(0, 59)
    )
    df = raw.loc[valid].copy()
    y = np.column_stack([
        (df["hw70"].to_numpy() <= -200).astype(int),
        (df["hw72"].to_numpy() <= -200).astype(int),
    ])
    x = pd.DataFrame({
        "child_age_months": df["hw1"].astype(float),
        "child_sex": df["b4"].astype(float),
        "weight_kg": df["hw2"].astype(float) / 10.0,
        "height_cm": df["hw3"].astype(float) / 10.0,
        "height_measurement_status": df["hw5"].astype(float),
        "weight_measurement_status": df["hw8"].astype(float),
    }, index=df.index)
    groups = df["caseid"].astype(str).to_numpy()
    positions = np.arange(len(df))
    train, remainder = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        .split(positions, y[:, 0], groups)
    )
    validation_relative, test_relative = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        .split(remainder, y[remainder, 0], groups[remainder])
    )
    validation = remainder[validation_relative]
    test = remainder[test_relative]
    models = []
    val_probs = []
    test_probs = []
    for col in range(2):
        model = XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
            reg_lambda=2, random_state=seed, n_jobs=-1, eval_metric="logloss",
        )
        model.fit(x.iloc[train], y[train, col])
        models.append(model)
        val_probs.append(model.predict_proba(x.iloc[validation])[:, 1])
        test_probs.append(model.predict_proba(x.iloc[test])[:, 1])
    val_probs = np.column_stack(val_probs)
    test_probs = np.column_stack(test_probs)
    thresholds = [
        max(
            np.arange(0.10, 0.91, 0.01),
            key=lambda value: accuracy_score(
                y[validation, col], val_probs[:, col] >= value
            ),
        )
        for col in range(2)
    ]
    primitive = np.column_stack([
        test_probs[:, col] >= thresholds[col] for col in range(2)
    ]).astype(int)
    predictions = np.column_stack([
        primitive, np.logical_or(primitive[:, 0], primitive[:, 1]).astype(int)
    ])
    truth = np.column_stack([
        y[test], np.logical_or(y[test, 0], y[test, 1]).astype(int)
    ])
    per_target = {}
    matrices = {}
    for col, target in enumerate(["Stunting", "Wasting", "Malnutrition"]):
        cm = confusion_matrix(truth[:, col], predictions[:, col], labels=[0, 1])
        matrices[target] = cm.tolist()
        per_target[target] = {
            "accuracy": float(accuracy_score(truth[:, col], predictions[:, col])),
            "balanced_accuracy": float(
                balanced_accuracy_score(truth[:, col], predictions[:, col])
            ),
            "precision": float(precision_score(truth[:, col], predictions[:, col], zero_division=0)),
            "recall": float(recall_score(truth[:, col], predictions[:, col], zero_division=0)),
            "f1": float(f1_score(truth[:, col], predictions[:, col], zero_division=0)),
        }
    payload = {
        "task": "current nutritional status",
        "not_comparable_to": "early-risk prediction",
        "dataset_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "seed": seed,
        "source_variables": ["hw1", "b4", "hw2", "hw3", "hw5", "hw8"],
        "excluded_target_variables": ["hw70", "hw72"],
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "thresholds_selected_on_validation": thresholds,
        "test_exact_match_accuracy": float(accuracy_score(truth, predictions)),
        "test_macro_f1": float(f1_score(truth, predictions, average="macro")),
        "per_target": per_target,
        "confusion_matrices": matrices,
        "training_seconds": time.time() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"models": models}, OUT / "model.pkl")
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
