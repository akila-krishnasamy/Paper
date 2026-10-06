import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

FEATURES = ["sex", "age_months", "height_cm", "weight_kg"]
TARGETS = ["Stunting", "Wasting", "Malnutrition"]


def load_dataset():
    path = Path(__file__).resolve().parents[2] / "stunting_wasting_dataset.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Jenis Kelamin": "sex",
        "Umur (bulan)": "age_months",
        "Tinggi Badan (cm)": "height_cm",
        "Berat Badan (kg)": "weight_kg",
    })
    df["Stunting"] = df["Stunting"].isin(["Stunted", "Severely Stunted"]).astype(int)
    df["Wasting"] = df["Wasting"].isin(["Underweight", "Severely Underweight"]).astype(int)
    df["Malnutrition"] = ((df["Stunting"] == 1) | (df["Wasting"] == 1)).astype(int)
    df["sex"] = df["sex"].replace({"Laki-laki": "Male", "Perempuan": "Female"})
    return df[FEATURES + TARGETS].dropna()


def build_pipeline():
    preprocessor = ColumnTransformer(
        [("sex", OneHotEncoder(handle_unknown="ignore"), ["sex"])],
        remainder="passthrough",
    )
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_and_evaluate():
    start = time.time()
    df = load_dataset()
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=df["Stunting"],
    )
    models = {}
    metrics = {"dataset": "stunting_wasting_dataset.csv", "test_size": len(test_idx), "conditions": {}}

    for target in TARGETS:
        model = build_pipeline()
        model.fit(df.loc[train_idx, FEATURES], df.loc[train_idx, target])
        predictions = model.predict(df.loc[test_idx, FEATURES])
        accuracy = float(accuracy_score(df.loc[test_idx, target], predictions))
        metrics["conditions"][target] = {
            "accuracy": round(accuracy, 4),
            "classification_report": classification_report(
                df.loc[test_idx, target], predictions, output_dict=True, zero_division=0
            ),
        }
        models[target] = model

    output_dir = Path(__file__).resolve().parents[1] / "models" / "anthropometric"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "features": FEATURES, "targets": TARGETS}, output_dir / "model.pkl")
    metrics["training_time_seconds"] = round(time.time() - start, 2)
    with open(output_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps({condition: values["accuracy"] for condition, values in metrics["conditions"].items()}, indent=2))
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
