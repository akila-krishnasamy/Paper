"""Group-aware XGBoost sensitivity experiment.

This is intentionally separate from the historical random-split benchmark.
It uses caseid groups so related records cannot cross train/validation/test.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "dhs_clean.parquet"
OUTPUT = ROOT / "ml-service" / "models" / "retrained" / "xgboost_group_aware_seed42"
FEATURES = [
    "age_hh_head_proxy", "hhsize", "wealth_quintile", "wealth_score",
    "residence", "gender_hh_head", "dist_market_proxy", "child_age_months",
    "child_sex", "birth_order", "birth_size", "birth_weight",
    "birth_weight_source", "breastfeeding_duration", "measles_vaccine",
    "diarrhea_recent", "fever_recent", "cough_recent", "mother_weight",
    "mother_height", "mother_bmi", "mother_marital_status", "anc_visits",
    "water_source", "toilet_type", "cooking_fuel",
]
TARGETS = ["stunting", "wasting"]


def main(seed=42):
    start = time.time()
    df = pd.read_parquet(DATASET).copy()
    y = df[TARGETS].astype(int).to_numpy()
    groups = df["caseid"].astype(str).to_numpy()
    indices = np.arange(len(df))
    first, remaining = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        .split(indices, y[:, 0], groups)
    )
    val_relative, test_relative = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        .split(remaining, y[remaining, 0], groups[remaining])
    )
    validation = remaining[val_relative]
    test = remaining[test_relative]
    train = first
    medians = df.iloc[train][FEATURES].median(numeric_only=True)
    x = df[FEATURES].copy().fillna(medians)
    models = []
    val_probs = []
    test_probs = []
    for col in range(2):
        model = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=2, random_state=seed,
            n_jobs=-1, eval_metric="logloss",
        )
        model.fit(x.iloc[train], y[train, col])
        models.append(model)
        val_probs.append(model.predict_proba(x.iloc[validation])[:, 1])
        test_probs.append(model.predict_proba(x.iloc[test])[:, 1])
    val_probs = np.column_stack(val_probs)
    test_probs = np.column_stack(test_probs)
    thresholds = []
    for col in range(2):
        candidates = np.arange(0.10, 0.91, 0.02)
        threshold = max(
            candidates,
            key=lambda value: accuracy_score(
                y[validation, col], val_probs[:, col] >= value
            ),
        )
        thresholds.append(float(threshold))
    primitive = np.column_stack([
        test_probs[:, i] >= thresholds[i] for i in range(2)
    ]).astype(int)
    predictions = np.column_stack([
        primitive, np.logical_or(primitive[:, 0], primitive[:, 1]).astype(int)
    ])
    truth = np.column_stack([
        y[test], np.logical_or(y[test, 0], y[test, 1]).astype(int)
    ])
    per_target = {}
    matrices = {}
    for i, target in enumerate(["Stunting", "Wasting", "Malnutrition"]):
        probability = (
            test_probs[:, i] if i < 2
            else 1 - (1 - test_probs[:, 0]) * (1 - test_probs[:, 1])
        )
        cm = confusion_matrix(truth[:, i], predictions[:, i], labels=[0, 1])
        matrices[target] = cm.tolist()
        per_target[target] = {
            "accuracy": float(accuracy_score(truth[:, i], predictions[:, i])),
            "balanced_accuracy": float(
                balanced_accuracy_score(truth[:, i], predictions[:, i])
            ),
            "precision": float(precision_score(truth[:, i], predictions[:, i], zero_division=0)),
            "recall": float(recall_score(truth[:, i], predictions[:, i], zero_division=0)),
            "f1": float(f1_score(truth[:, i], predictions[:, i], zero_division=0)),
            "roc_auc": float(roc_auc_score(truth[:, i], probability)),
            "pr_auc": float(average_precision_score(truth[:, i], probability)),
        }
    payload = {
        "model": "group-aware XGBoost",
        "seed": seed,
        "group_column": "caseid",
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "thresholds_selected_on_validation": thresholds,
        "test_exact_match_accuracy": float(accuracy_score(truth, predictions)),
        "test_macro_f1": float(f1_score(truth, predictions, average="macro")),
        "test_macro_roc_auc": float(roc_auc_score(
            truth, np.column_stack([
                test_probs,
                1 - (1 - test_probs[:, 0]) * (1 - test_probs[:, 1]),
            ]), average="macro"
        )),
        "per_target": per_target,
        "confusion_matrices": matrices,
        "features": FEATURES,
        "training_seconds": time.time() - start,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": FEATURES, "medians": medians.to_dict()},
                OUTPUT / "model.pkl")
    (OUTPUT / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
