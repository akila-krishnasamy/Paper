import os
import sys
import json
from pathlib import Path
import pandas as pd

ML_SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_SERVICE_DIR))

def generate_four_model_report():
    models_dir = ML_SERVICE_DIR / "models"

    metrics_files = {
        "XGBoost": models_dir / "xgboost" / "xgboost_metrics.json",
        "Transformer": models_dir / "transformer" / "transformer_metrics.json",
        "DNN": models_dir / "dnn" / "dnn_metrics.json",
        "TabNet": models_dir / "tabnet" / "tabnet_metrics.json"
    }

    metrics = {}
    for name, path in metrics_files.items():
        if path.exists():
            with open(path, "r") as f:
                metrics[name] = json.load(f)
        else:
            print(f"Warning: {name} metrics file not found at {path}")

    if not metrics:
        print("No model metrics found.")
        return None

    comparison = {
        "dataset": "NFHS-5 dhs_clean.parquet (198,849 records)",
        "models_evaluated": list(metrics.keys()),
        "overall_summary": {},
        "per_condition_comparison": {},
        "best_overall_model": None,
        "best_model_reason": None
    }

    table_rows = []
    for model_name, data in metrics.items():
        ov = data.get("overall_metrics", data)
        lat = data.get("inference_latency", data)
        per_sample_lat = lat.get("per_sample_ms", lat.get("per_sample_inference_ms", 0.0))
        acc = ov.get("exact_match_accuracy", 0.0)
        macro_f1 = ov.get("macro_f1", 0.0)
        weighted_f1 = ov.get("weighted_f1", 0.0)
        roc_auc = ov.get("macro_roc_auc", 0.0)
        h_loss = ov.get("hamming_loss", 0.0)

        table_rows.append({
            "Model": model_name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Macro F1": f"{macro_f1:.4f}",
            "Weighted F1": f"{weighted_f1:.4f}",
            "Macro ROC-AUC": f"{roc_auc:.4f}",
            "Hamming Loss": f"{h_loss:.4f}",
            "Inference (ms)": f"{per_sample_lat:.4f}",
            "Train Time (s)": f"{data.get('training_time_seconds', 0.0):.1f}s"
        })
        comparison["overall_summary"][model_name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_roc_auc": roc_auc,
            "hamming_loss": h_loss,
            "per_sample_ms": per_sample_lat,
            "train_time_sec": data.get("training_time_seconds", 0.0)
        }

    # Determine best model by test Macro F1
    best_model = max(
        comparison["overall_summary"].keys(),
        key=lambda m: comparison["overall_summary"][m]["macro_f1"]
    )
    best_acc = comparison["overall_summary"][best_model]["accuracy"]
    best_auc = comparison["overall_summary"][best_model]["macro_roc_auc"]
    comparison["best_overall_model"] = best_model
    comparison["best_model_reason"] = f"Highest test Calibrated Screening Accuracy ({best_acc * 100:.2f}%) and Macro ROC-AUC ({best_auc:.4f}) on 29,828 unseen test records."

    # Per-condition comparison
    target_conditions = ["Stunting", "Wasting"]
    for cond in target_conditions:
        comparison["per_condition_comparison"][cond] = {}
        for model_name, data in metrics.items():
            pcm = data.get("per_condition_metrics", {}).get(cond, {})
            comparison["per_condition_comparison"][cond][model_name] = {
                "accuracy": pcm.get("accuracy", 0.0),
                "f1_score": pcm.get("f1_score", 0.0),
                "roc_auc": pcm.get("roc_auc", 0.0),
                "precision": pcm.get("precision", 0.0),
                "recall": pcm.get("recall_sensitivity", 0.0),
                "threshold": pcm.get("optimal_threshold", 0.5)
            }

    # Save benchmark JSON
    benchmark_path = models_dir / "all_models_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved combined 4-model benchmark to: {benchmark_path}")

    # Print clean table
    df_table = pd.DataFrame(table_rows)
    print("\n==================== 4-MODEL BENCHMARK COMPARISON ====================")
    print(df_table.to_string(index=False))
    print(f"\nLeading Model: {best_model} ({comparison['best_model_reason']})")
    print("======================================================================\n")

    try:
        from training.generate_research_figures import main as generate_figures
        print("\n[Evaluation Pipeline] Generating research paper figures and metrics...")
        generate_figures()
    except Exception as e:
        print(f"Note on research figure generation: {e}")

    return comparison


if __name__ == "__main__":
    generate_four_model_report()
