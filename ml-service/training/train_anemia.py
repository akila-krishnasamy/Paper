import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_anemia_model():
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "raw" / "anemia_public.csv"
    
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        return None

    df = pd.read_csv(data_path)
    print("Training Anemia & Iron Deficiency Screening Model on real lab data...")
    print("Dataset shape:", df.shape)

    features = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
    X = df[features]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost / Random Forest classifier
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Anemia Model Results - Test Acc: {acc*100:.2f}%, F1: {f1:.4f}, AUC: {auc:.4f}")

    output_dir = Path(__file__).resolve().parents[1] / "models" / "anemia"
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "anemia_model.pkl")
    joblib.dump(scaler, output_dir / "anemia_scaler.pkl")

    metrics = {
        "dataset": "anemia_public.csv (300 clinical laboratory records)",
        "features": features,
        "test_size": len(y_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "feature_importances": {f: round(float(imp), 4) for f, imp in zip(features, model.feature_importances_)},
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_dir / "anemia_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved anemia model and metrics to", output_dir)
    return metrics

if __name__ == "__main__":
    train_anemia_model()
