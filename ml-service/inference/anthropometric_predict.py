import json
from pathlib import Path

import joblib
import pandas as pd


_MODEL_CACHE = None


def load_anthropometric_artifacts():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        model_dir = Path(__file__).resolve().parent.parent / "models" / "anthropometric"
        model_path = model_dir / "model.pkl"
        metrics_path = model_dir / "metrics.json"
        if not model_path.exists():
            raise FileNotFoundError("Anthropometric model not found. Run training/train_anthropometric.py first.")
        _MODEL_CACHE = (
            joblib.load(model_path),
            json.loads(metrics_path.read_text()) if metrics_path.exists() else {},
        )
    return _MODEL_CACHE


def predict_anthropometric(input_data):
    artifacts, metrics = load_anthropometric_artifacts()
    row = pd.DataFrame([{
        "sex": "Female" if str(input_data.get("child_sex", "Male")).lower() in {"female", "f", "girl", "2"} else "Male",
        "age_months": float(input_data["child_age_months"]),
        "height_cm": float(input_data["height_cm"]),
        "weight_kg": float(input_data["weight_kg"]),
    }])

    conditions = []
    for condition in artifacts["targets"]:
        model = artifacts["models"][condition]
        probability = float(model.predict_proba(row)[0, 1])
        conditions.append({
            "condition": condition,
            "probability": round(probability, 4),
            "percentage": round(probability * 100, 1),
            "risk_flag": bool(probability >= 0.5),
            "severity": "HIGH" if probability >= 0.65 else ("MODERATE" if probability >= 0.5 else "LOW"),
            "threshold_used": 0.5,
            "test_accuracy": metrics.get("conditions", {}).get(condition, {}).get("accuracy"),
        })

    malnutrition = next(item for item in conditions if item["condition"] == "Malnutrition")
    return {
        "model": "Anthropometric XGBoost",
        "overall_risk": "HIGH" if malnutrition["risk_flag"] else "LOW",
        "prediction": "Malnourished" if malnutrition["risk_flag"] else "Normal",
        "probability": malnutrition["probability"],
        "percentage": malnutrition["percentage"],
        "conditions": conditions,
        "test_accuracies": {
            condition: values.get("accuracy")
            for condition, values in metrics.get("conditions", {}).items()
        },
        "medical_disclaimer": "Screening support for academic/research purposes; not a medical diagnosis.",
    }
