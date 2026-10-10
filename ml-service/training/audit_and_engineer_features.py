"""Generate a reproducible NFHS-5 data-quality and feature audit.

The report is descriptive only: it never changes labels or deletes records.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed" / "dhs_clean.parquet"
REPORT_DIR = ROOT / "ml-service" / "reports"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    df = pd.read_parquet(DATASET)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.DataFrame({
        "Stunting": df["stunting"].astype(int),
        "Wasting": df["wasting"].astype(int),
    })
    labels["Malnutrition"] = (labels["Stunting"] | labels["Wasting"]).astype(int)

    invalid = {
        "child_age_months_outside_0_59": int(
            ((df["child_age_months"] < 0) | (df["child_age_months"] > 59)).sum()
        ),
        "birth_weight_nonpositive": int((df["birth_weight"].dropna() <= 0).sum()),
        "mother_height_nonpositive": int((df["mother_height"].dropna() <= 0).sum()),
        "mother_weight_nonpositive": int((df["mother_weight"].dropna() <= 0).sum()),
        "mother_bmi_outside_10_70": int(
            ((df["mother_bmi"].dropna() < 10) | (df["mother_bmi"].dropna() > 70)).sum()
        ),
        "label_values_not_binary": {
            column: sorted(set(df[column].dropna().astype(int)) - {0, 1})
            for column in ["stunting", "wasting"]
        },
    }
    report = {
        "dataset": str(DATASET),
        "sha256": sha256(DATASET),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "caseid_unique": int(df["caseid"].nunique()),
        "records_with_repeated_caseid": int(df["caseid"].duplicated(keep=False).sum()),
        "dtypes": {key: str(value) for key, value in df.dtypes.items()},
        "missing_counts": {
            key: int(value) for key, value in df.isna().sum().items()
        },
        "label_counts": {
            key: {str(k): int(v) for k, v in value.value_counts().items()}
            for key, value in labels.items()
        },
        "composite_consistency": bool(
            (labels["Malnutrition"] == (labels["Stunting"] | labels["Wasting"])).all()
        ),
        "invalid_measurements": invalid,
        "source_columns_not_used_by_current_18_feature_pipeline": [
            column for column in df.columns
            if column not in {
                "stunting", "wasting", "caseid",
                "child_age_months", "birth_weight", "breastfeeding_duration",
                "birth_order", "mother_bmi", "anc_visits", "hhsize",
                "child_sex", "education", "wealth_quintile", "residence",
                "birth_size", "diarrhea_recent", "fever_recent",
                "cough_recent", "measles_vaccine", "water_source",
                "toilet_type", "cooking_fuel",
            }
        ],
        "note": (
            "Repeated caseid values indicate related records. Random row splits "
            "may be optimistic; group-aware evaluation by caseid is recommended."
        ),
    }
    (REPORT_DIR / "data_quality_audit.json").write_text(
        json.dumps(report, indent=2)
    )

    lines = [
        "# NFHS-5 Data Quality Audit",
        "",
        f"- Rows: **{report['rows']:,}**",
        f"- Columns: **{report['columns']}**",
        f"- Dataset SHA-256: `{report['sha256']}`",
        f"- Exact duplicate rows: **{report['exact_duplicate_rows']:,}**",
        f"- Unique case IDs: **{report['caseid_unique']:,}**",
        f"- Records belonging to repeated case IDs: **{report['records_with_repeated_caseid']:,}**",
        "",
        "## Labels",
        "",
        "- Stunting and wasting are binary.",
        "- Composite malnutrition is deterministically `Stunting OR Wasting`.",
        f"- Composite consistency verified: **{report['composite_consistency']}**.",
        "",
        "## Missingness",
        "",
    ]
    for column, count in sorted(report["missing_counts"].items(), key=lambda item: -item[1]):
        if count:
            lines.append(f"- `{column}`: {count:,}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "No records were removed and no target labels were imputed. Missing values "
        "must be handled inside each training split. Repeated case IDs require a "
        "group-aware sensitivity analysis because random row splitting can expose "
        "related household records across partitions.",
        "",
        "## Source columns currently omitted",
        "",
    ])
    lines.extend(f"- `{column}`" for column in report[
        "source_columns_not_used_by_current_18_feature_pipeline"
    ])
    (REPORT_DIR / "DATA_QUALITY_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "rows": report["rows"],
        "exact_duplicates": report["exact_duplicate_rows"],
        "repeated_caseid_records": report["records_with_repeated_caseid"],
        "composite_consistent": report["composite_consistency"],
        "report": str(REPORT_DIR / "DATA_QUALITY_AUDIT.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
