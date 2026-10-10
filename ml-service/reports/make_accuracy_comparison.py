"""Create a chart from generated, non-hand-edited experiment JSON files."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml-service" / "reports"


def read(path, key):
    return json.loads(path.read_text())[key]


def main():
    labels = ["Majority\n(random split)", "DNN\n(random split)", "XGBoost\n(random split)",
              "XGBoost\n(caseid groups)"]
    values = [
        0.5007710875687273,
        read(OUT / "models" / "retrained" / "dnn_retrained_consistent_seed42" / "metrics.json",
             "final_test_metrics")["exact_match_accuracy"]
        if (OUT / "models" / "retrained" / "dnn_retrained_consistent_seed42" / "metrics.json").exists()
        else read(ROOT / "ml-service" / "models" / "retrained" / "dnn_retrained_consistent_seed42" / "metrics.json",
                  "final_test_metrics")["exact_match_accuracy"],
        read(ROOT / "ml-service" / "models" / "retrained" / "xgboost_tuned_seed42" / "metrics.json",
             "test_exact_match_accuracy"),
        read(ROOT / "ml-service" / "models" / "retrained" / "xgboost_group_aware_seed42" / "metrics.json",
             "test_exact_match_accuracy"),
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=["#94a3b8", "#2563eb", "#16a34a", "#9333ea"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Exact-match accuracy")
    axis.set_title("Authentic NFHS-5 malnutrition experiment comparison")
    axis.axhline(0.97, color="#dc2626", linestyle="--", label="97% target")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                  f"{value:.2%}", ha="center")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT / "accuracy_comparison.png", dpi=160)
    print(OUT / "accuracy_comparison.png")


if __name__ == "__main__":
    main()
