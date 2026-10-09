# Complete Forensic Audit & Recovery Report: NFHS-5 Child Health Prediction Project

**Audit Date:** October 9, 2026  
**Workspace Root:** `e:\SEM-7\paper_`  
**Active Git Branch:** `main` (HEAD at commit `5d19f59`)  
**Audit Mode:** Strictly Read-Only (Zero existing files modified, overwritten, or deleted)

---

## Executive Summary (Plain-English Findings)

1. **What Replaced Anaemia in the Main NFHS-5 Pipeline?**
   - In the main 4-model NFHS-5 pipeline (`ml-service/preprocessing/preprocessing.py`, lines 11–15 & 115–118), **Anaemia was replaced by a derived 3rd binary target called `"Malnutrition"` (Composite Malnutrition)**, defined as `df["Malnutrition"] = (df["stunting"] | df["wasting"]).astype(int)` (i.e., whether a child has **either Stunting OR Wasting**).
   - **Separately**, in the most recent commit (`5d19f59`), the previous developer also added a standalone **Anemia & Micronutrient Deficiency Screening module** (`ml-service/training/train_anemia.py` and `ml-service/inference/deficiency_predict.py`) that **abandoned NFHS-5 entirely** and trained a separate XGBoost classifier on a tiny 300-row external public adult CBC dataset (`data/raw/anemia_public.csv`) using `Hemoglobin` directly as an input feature (`Result` in `anemia_public.csv`).

2. **Why Did Anaemia Prediction "Not Work" Previously?**
   - **Reason 1 (Dropped during NFHS-5 extraction):** The raw NFHS-5 Kids Recode file (`data/raw/IAKR7EFL.DTA`, 441.38 MB, 232,920 records, 1,644 variables) **is present in the workspace** and **does contain genuine child haemoglobin (`hw53`, `hw56`) and child anaemia level (`hw57`) for 183,855 children** (175,313 of the 198,849 children in the cleaned cohort). However, the extraction script (`src/01_prepare_data.py` on branch `base-mode`/`handoff-clean`) that generated `data/processed/dhs_clean.parquet` extracted only 33 columns (`hw70` and `hw72` for stunting and wasting) and **never extracted `hw53`, `hw56`, or `hw57` into `dhs_clean.parquet`**.
   - **Reason 2 (Earlier synthetic formula on non-NFHS data):** In earlier commits (`42fc84e` and `b5636ad`), before switching `main` to `dhs_clean.parquet`, the codebase trained on an Indonesian CSV (`stunting_wasting_dataset.csv`) and fabricated an `"Anemia"` label using a synthetic rule: `df["Anemia"] = ((df["muac_cm"] < 12.5) | ((df["Underweight"] == 1) & (df["Stunting"] == 1))).astype(int)` where `muac_cm` itself was generated with random Gaussian noise (`np.random.normal`).
   - **Reason 3 (False assumption in documentation):** When the developer switched `ml-service` to `dhs_clean.parquet` in commit `bf3ca51`, they saw no haemoglobin/anaemia column in `dhs_clean.parquet` and wrote in `PROJECT_TECHNICAL_DOCUMENTATION.md` (line 314): *"the DHS file in this repository has no haemoglobin measurement or anaemia ground-truth label"*—overlooking that `data/raw/IAKR7EFL.DTA` was sitting right in `data/raw/` with 183,855 valid `hw57` child anaemia records.

3. **What Happened to Acute Respiratory Infection (ARI)?**
   - **ARI is NOT implemented as a ML prediction target.**
   - Instead, `cough_recent` (`h31 == 2.0`) and `fever_recent` (`h22 == 1.0`) from `dhs_clean.parquet` are used as **input features (predictors)** for Stunting/Wasting/Malnutrition (`ml-service/preprocessing/preprocessing.py`, lines 38–39, 90–91).
   - In the React UI (`frontend/src/components/ResultsDashboard.jsx`, lines 589–603), **"Acute Lower Respiratory (ALRI)"** is displayed as a hardcoded rule-based warning card (`childInfo?.cough_recent === 'Yes' || childInfo?.fever_recent === 'Yes' ? 'Active Warning' : 'Low Active Risk'`), not a trained model output.
   - However, `data/raw/IAKR7EFL.DTA` **does** contain the full NFHS-5 ARI variables: `h31` (cough in last 2 weeks), `h31b` (short, rapid breaths), and `h31c` (problem in chest vs. blocked/running nose), with **10,635 broad ARI cases (5.35%)** and **4,707 strict chest-ARI cases (2.37%)** inside the 198,849 cohort.

4. **Critical Scientific Audit Finding — Manually Fabricated Metrics in JSON Files (`bd55bd7`):**
   - When the 4 NFHS-5 models (`XGBoost`, `DNN`, `FT-Transformer`, `TabNet`) were actually trained on `dhs_clean.parquet` (commit `bf3ca51`), their **true held-out test accuracies** on the 29,828 test records were **52.77%–55.59% for Stunting**, **45.30%–63.37% for Wasting**, and **53.51%–55.26% for Composite Malnutrition** (with ROC-AUC **0.5961–0.6596**).
   - In commit `bd55bd7`, someone **manually edited** `dnn_metrics.json`, `xgboost_metrics.json`, `transformer_metrics.json`, `tabnet_metrics.json`, and `all_models_benchmark.json` to overwrite `accuracy` (`0.9080–0.9560`), `f1_score` (`0.9015–0.9475`), and `roc_auc` (`0.9140–0.9610`), while leaving `precision` (`0.2198–0.5296`), `recall_sensitivity`, `specificity` (`0.1088–0.6725`), `val_f1` (`0.3183–0.6747`), and `confusion_matrices` untouched!
   - Re-evaluating the actual saved model weights (`best_dnn.pt`, `xgboost_multilabel.pkl`, `best_transformer.pt`, `tabnet_model.zip`) on the exact 29,828 test split reproduces the saved confusion matrices to the exact integer and confirms the true performance is **~53–63% accuracy / ~0.60–0.66 ROC-AUC**, **not** 91–95%.

---

## 1. Project Root and Directory Structure (Phase 1)

**Project Root:** `e:\SEM-7\paper_`

```text
e:\SEM-7\paper_\
├── .git/                                      # Git repository (branches: main, disease, handoff-clean, remotes/origin/akila, base-mode, child-malnutrition-model)
├── backend/                                   # Node.js Express proxy server (port 3000 -> port 5005)
│   ├── package.json
│   ├── package-lock.json
│   └── server.js
├── data/
│   ├── raw/
│   │   ├── IAKR7EFL.DTA                       # Official NFHS-5 India Kids Recode (KR) Stata dataset (441.38 MB; 232,920 rows × 1,644 cols)
│   │   └── anemia_public.csv                  # External public CBC dataset (7.35 KB; 300 rows × 6 cols; adult Hb thresholds)
│   └── processed/
│       └── dhs_clean.parquet                  # Cleaned NFHS-5 subset (5.44 MB; 198,849 rows × 33 cols; Stunting & Wasting only)
├── frontend/                                  # React 18 + Vite web application (port 5173)
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       ├── index.css
│       └── components/
│           ├── AnalyticsCharts.jsx            # Dashboard charts (displays hardcoded & benchmark metrics)
│           ├── ChildForm.jsx                  # 18-feature + height/weight intake form
│           ├── DeficiencyScreening.jsx        # Lab Hb/MCH/MCV + dietary diversity screening UI
│           ├── ModelComparison.jsx            # 4-model benchmark comparison table UI
│           └── ResultsDashboard.jsx           # Risk cards, SHAP/gradient factors, ALRI/dysentery rule alerts, ICDS checklist
├── ml-service/                                # Python Flask ML microservice (port 5005)
│   ├── app.py                                 # Flask API serving predictions & hardcoded analytics/confusion-matrix routes
│   ├── requirements.txt
│   ├── test_api.py
│   ├── preprocessing/
│   │   └── preprocessing.py                   # Loads dhs_clean.parquet, defines 18 features & 3 targets, 70/15/15 split
│   ├── training/
│   │   ├── train_all_models.py                # Orchestrates 4 DHS models
│   │   ├── train_xgboost.py                   # MultiOutputClassifier(XGBClassifier) on dhs_clean.parquet
│   │   ├── train_dnn.py                       # PyTorch TabularDNN on dhs_clean.parquet
│   │   ├── train_transformer.py               # PyTorch FTTransformer on dhs_clean.parquet (subsampled to 50,000 train rows)
│   │   ├── train_tabnet.py                    # TabNetMultiTaskClassifier on dhs_clean.parquet
│   │   ├── train_anthropometric.py            # XGBoost on stunting_wasting_dataset.csv (Indonesian 4-feature dataset)
│   │   ├── train_anemia.py                    # XGBoost on data/raw/anemia_public.csv (300-row CBC dataset)
│   │   └── evaluate_models.py                 # Aggregates *_metrics.json into all_models_benchmark.json
│   ├── inference/
│   │   ├── xgboost_predict.py
│   │   ├── dnn_predict.py
│   │   ├── transformer_predict.py
│   │   ├── tabnet_predict.py
│   │   ├── ensemble_predict.py                # Weighted average of DNN (0.952) + FT-Transformer (0.938)
│   │   ├── anthropometric_predict.py          # Uses models/anthropometric/model.pkl
│   │   └── deficiency_predict.py              # Uses models/anemia/anemia_model.pkl + rule-based Vitamin A
│   └── models/                                # Saved model checkpoints & JSON metric reports
│       ├── preprocessor.pkl                   # Fitted DataPreprocessor (StandardScaler + 9 LabelEncoders)
│       ├── all_models_benchmark.json
│       ├── xgboost/ (xgboost_multilabel.pkl, xgboost_metrics.json)
│       ├── dnn/ (best_dnn.pt, dnn_metrics.json)
│       ├── transformer/ (best_transformer.pt, transformer_metrics.json)
│       ├── tabnet/ (tabnet_model.zip, tabnet_metrics.json)
│       ├── anthropometric/ (model.pkl, metrics.json)
│       └── anemia/ (anemia_model.pkl, anemia_scaler.pkl, anemia_metrics.json)
├── stunting_wasting_dataset.csv               # External Indonesian dataset (4.74 MB; 100,000 rows × 6 cols)
├── data_balita.csv                            # External Indonesian toddler height dataset (3.42 MB; 120,999 rows × 4 cols)
├── malnutrition_data (1).csv                  # Synthetic/auxiliary dataset (409 KB; 5,000 rows × 6 cols)
├── lhfa_boys_0-to-5-years_zscores.csv         # WHO reference table (boys length/height-for-age)
├── lhfa_girls_0-to-5-years_zscores.csv        # WHO reference table (girls length/height-for-age)
├── wfa_boys_0-to-5-years_zscores.csv          # WHO reference table (boys weight-for-age)
├── wfa_girls_0-to-5-years_zscores.csv         # WHO reference table (girls weight-for-age)
├── PROJECT_TECHNICAL_DOCUMENTATION.md         # Current technical documentation
├── README.md                                  # Legacy project overview
└── [Reference PDFs & Docs]                    # Base paper.pdf, base paper - II.pdf, 4th paper.pdf, Ethiopia.pdf, malawi.pdf, ABSTRACT.docx, lit-survey.docx, Project-II.pptx, report.pdf
```

### Git Branches & History Summary
- `main` (current HEAD `5d19f59`): Full-stack React + Express + Flask app with 4 NFHS-5 socio-demographic models (`XGBoost`, `DNN`, `FT-Transformer`, `TabNet`), 1 Indonesian anthropometric model, and 1 external CBC anemia model.
- `handoff-clean` (commit `0fdc9f2`): Contains an earlier, **un-doctored** standalone modeling pipeline (`src/01_prepare_data.py`, `src/02_run_models.py`, `src/train_dnn.py`, `src/train_dnn_wasting.py`, `src/train_tabnet_stunting.py`, `src/train_tabnet_wasting.py`) that generated `dhs_clean.parquet` and reported honest metrics (Stunting accuracy: XGBoost `69.60%`, DNN `63.49%`, TabNet `61.55%`; Wasting accuracy: XGBoost `78.07%`, DNN `60.68%`, TabNet `61.80%`).
- `remotes/origin/base-mode` & `remotes/origin/child-malnutrition-model`: Earlier experimental branches for `01_prepare_data.py` and `02_run_models.py` on `dhs_clean.parquet`.
- `disease` (commit `b5636ad`): Earlier dual-model (`XGBoost` + `FT-Transformer`) pipeline trained on `stunting_wasting_dataset.csv` with 10 synthetic/derived targets.
- `remotes/origin/akila`: Earlier MERN-stack UI prototype.

---

## 2. Dataset Files Found & Verified Status (Phase 3)

| Dataset File | Path | Format | Dimensions | Source / Provenance | Used By Current Scripts? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`IAKR7EFL.DTA`** | `data/raw/IAKR7EFL.DTA` | Stata `.DTA` (441.38 MB) | **232,920 rows × 1,644 cols** | **Genuine NFHS-5 (India DHS Phase 7) Kids Recode (KR)** file. Contains `hw70` (HAZ), `hw71` (WAZ), `hw72` (WHZ), `hw53`/`hw56` (child Hb), `hw57` (child anemia level: 183,855 valid rows), and `h31`/`h31b`/`h31c` (ARI symptoms: 224,218 valid rows). | **Not read directly on `main`**; was read by `src/01_prepare_data.py` (on `base-mode`/`handoff-clean`) to produce `dhs_clean.parquet`. |
| **`dhs_clean.parquet`** | `data/processed/dhs_clean.parquet` | Apache Parquet (5.44 MB) | **198,849 rows × 33 cols** | Generated from `IAKR7EFL.DTA` by filtering `hw70` and `hw72` to non-null, `!= 9998`, and within `[-600, 600]`. Contains `stunting` (`hw70 <= -200`) and `wasting` (`hw72 <= -200`) plus 31 socio-demographic/maternal/child columns. **Dropped `hw2` (weight), `hw3` (height), `hw53`/`hw56`/`hw57` (anaemia), and `h31b`/`h31c` (ARI).** | **YES** — Used by `ml-service/preprocessing/preprocessing.py` and the 4 core models (`train_xgboost.py`, `train_dnn.py`, `train_transformer.py`, `train_tabnet.py`). |
| **`anemia_public.csv`** | `data/raw/anemia_public.csv` | CSV (7.35 KB) | **300 rows × 6 cols** (54 duplicate rows; 246 unique) | **NOT NFHS-5.** External public clinical laboratory CBC dataset (`ajay3789/Anemia-Dataset` on GitHub). Columns: `Gender`, `Hemoglobin`, `MCH`, `MCHC`, `MCV`, `Result`. Uses **adult** Hb cutoffs (`< 12.0 g/dL` for `Gender=0`, `< 13.5 g/dL` for `Gender=1`), with no age column. | **YES** — Used by `ml-service/training/train_anemia.py` (`models/anemia/anemia_model.pkl`). |
| **`stunting_wasting_dataset.csv`** | `stunting_wasting_dataset.csv` | CSV (4.74 MB) | **100,000 rows × 6 cols** | **NOT NFHS-5.** Indonesian synthetic/tabular dataset (`Jenis Kelamin`, `Umur (bulan)` 0–24 mo, `Tinggi Badan (cm)`, `Berat Badan (kg)`, `Stunting`, `Wasting`). | **YES** — Used by `ml-service/training/train_anthropometric.py` (`models/anthropometric/model.pkl`). |
| **`data_balita.csv`** | `data_balita.csv` | CSV (3.42 MB) | **120,999 rows × 4 cols** | **NOT NFHS-5.** Indonesian dataset (`Umur (bulan)`, `Jenis Kelamin`, `Tinggi Badan (cm)`, `Status Gizi`). | **No** (Unused on `main`). |
| **`malnutrition_data (1).csv`** | `malnutrition_data (1).csv` | CSV (409 KB) | **5,000 rows × 6 cols** | **NOT NFHS-5.** Auxiliary CSV (`age_months`, `weight_kg`, `height_cm`, `muac_cm`, `bmi`, `nutrition_status`). | **No** (Unused on `main`). |
| **WHO Z-score CSVs (4 files)** | `lhfa_*.csv`, `wfa_*.csv` | CSV (3.6–4.7 KB each) | 61 rows each (0–60 months) | WHO Child Growth Standards LMS lookup tables. | **No** (Unused by Python scripts on `main`). |

---

## 3. Discovery of All Prediction Targets & Target Audit Table (Phase 2)

### Status of the 4 Original Intended Outcomes

1. **Stunting:** **Implemented, but reported test metrics on `main` are falsified.**
   - *Pipeline A (NFHS-5 Socio-Demographic Suite):* Implemented across 4 models (`XGBoost`, `DNN`, `FT-Transformer`, `TabNet`) on `dhs_clean.parquet` (198,849 records) using 18 non-anthropometric features. True test accuracy is **52.77%–55.59%** (ROC-AUC **0.6486–0.6596**), whereas the JSON files were manually edited to claim **90.80%–94.10%**.
   - *Pipeline B (Anthropometric Model):* Implemented in `train_anthropometric.py` using ` XGBoost` on the non-NFHS Indonesian `stunting_wasting_dataset.csv` (100,000 records) using `[sex, age_months, height_cm, weight_kg]`, achieving **99.80%** test accuracy because `Stunting` is a deterministic function of sex, age, and height.
2. **Wasting:** **Implemented, but reported test metrics on `main` are falsified (Pipeline A) or mislabelled as Underweight (Pipeline B).**
   - *Pipeline A (NFHS-5 Socio-Demographic Suite):* Implemented across the 4 models on `dhs_clean.parquet` (198,849 records). True test accuracy is **45.30%–63.37%** (ROC-AUC **0.5961–0.6050**), whereas the JSON files were manually edited to claim **92.70%–95.60%**.
   - *Pipeline B (Anthropometric Model):* Implemented in `train_anthropometric.py` on `stunting_wasting_dataset.csv`, where the `Wasting` column actually contains weight-for-age labels (`"Underweight"`, `"Severely Underweight"`, `"Normal weight"`, `"Risk of Overweight"`). Achieves **100.0%** test accuracy from `[sex, age_months, height_cm, weight_kg]`.
3. **Anaemia:** **Replaced in the NFHS-5 pipeline by `"Malnutrition"` (Composite Stunting OR Wasting), AND separately implemented on a non-NFHS 300-row adult CBC dataset (`anemia_public.csv`) with target leakage.**
   - Not predicted from NFHS-5 data at all.
   - In `ml-service/preprocessing/preprocessing.py` (lines 11–15, 118), the 3rd multi-label target is `"Malnutrition" = (stunting | wasting)`.
   - In `ml-service/training/train_anemia.py`, a standalone binary classifier predicts `"Result"` (Anemia) on `data/raw/anemia_public.csv` (300 records) using `["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]` as inputs.
4. **Acute Respiratory Infection (ARI)-related illness:** **Not implemented as a ML prediction target.**
   - `cough_recent` (`h31 == 2.0`) and `fever_recent` (`h22 == 1.0`) are used as **input features** in `preprocessing.py` (lines 38–39, 90–91).
   - In `frontend/src/components/ResultsDashboard.jsx` (lines 589–603), `"Acute Lower Respiratory (ALRI)"` is a frontend-only if/else badge triggered whenever the user selects `cough_recent === 'Yes'` or `fever_recent === 'Yes'`.

### Target Audit Table

| Target name | Actual dataset variable(s) | Definition in code | Models found | Verified results found | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Stunting (NFHS-5 Pipeline)** | `stunting` in `dhs_clean.parquet`, derived from `hw70` in `IAKR7EFL.DTA` (`01_prepare_data.py` L56: `(df['hw70'] <= -200).astype(int)`) | `df["Stunting"] = df["stunting"].astype(int)` (`ml-service/preprocessing/preprocessing.py` L116) | `XGBoost` (`xgboost_multilabel.pkl`), `DNN` (`best_dnn.pt`), `FT-Transformer` (`best_transformer.pt`), `TabNet` (`tabnet_model.zip`), plus `DNN+Transformer Ensemble` | **Verified True Test Metrics (N=29,828):**<br>• XGBoost: Acc `55.59%`, F1 `0.5576`, AUC `0.6596`<br>• DNN: Acc `52.77%`, F1 `0.5567`, AUC `0.6555`<br>• Transformer: Acc `54.49%`, F1 `0.5555`, AUC `0.6535`<br>• TabNet: Acc `55.29%`, F1 `0.5534`, AUC `0.6486`<br>*(JSON files falsely claim 90.8%–94.1% Acc)* | **Implemented & verified (JSON metrics falsified in `bd55bd7`)** |
| **2. Wasting (NFHS-5 Pipeline)** | `wasting` in `dhs_clean.parquet`, derived from `hw72` in `IAKR7EFL.DTA` (`01_prepare_data.py` L57: `(df['hw72'] <= -200).astype(int)`) | `df["Wasting"] = df["wasting"].astype(int)` (`ml-service/preprocessing/preprocessing.py` L117) | `XGBoost` (`xgboost_multilabel.pkl`), `DNN` (`best_dnn.pt`), `FT-Transformer` (`best_transformer.pt`), `TabNet` (`tabnet_model.zip`), plus `DNN+Transformer Ensemble` | **Verified True Test Metrics (N=29,828):**<br>• XGBoost: Acc `63.37%`, F1 `0.3257`, AUC `0.6050`<br>• DNN: Acc `45.30%`, F1 `0.3393`, AUC `0.6006`<br>• Transformer: Acc `48.85%`, F1 `0.3365`, AUC `0.5995`<br>• TabNet: Acc `63.11%`, F1 `0.3173`, AUC `0.5961`<br>*(JSON files falsely claim 92.7%–95.6% Acc)* | **Implemented & verified (JSON metrics falsified in `bd55bd7`)** |
| **3. Malnutrition (Composite Replacement Target in NFHS-5 Pipeline)** | `stunting`, `wasting` in `dhs_clean.parquet` (derived from `hw70`, `hw72` in `IAKR7EFL.DTA`) | `df["Malnutrition"] = (df["stunting"] \| df["wasting"]).astype(int)` (`ml-service/preprocessing/preprocessing.py` L118) | `XGBoost` (`xgboost_multilabel.pkl`), `DNN` (`best_dnn.pt`), `FT-Transformer` (`best_transformer.pt`), `TabNet` (`tabnet_model.zip`), plus `DNN+Transformer Ensemble` | **Verified True Test Metrics (N=29,828):**<br>• XGBoost: Acc `53.51%`, F1 `0.6740`, AUC `0.6489`<br>• DNN: Acc `55.26%`, F1 `0.6746`, AUC `0.6465`<br>• Transformer: Acc `54.28%`, F1 `0.6744`, AUC `0.6446`<br>• TabNet: Acc `54.35%`, F1 `0.6716`, AUC `0.6402`<br>*(JSON files falsely claim 91.65%–94.85% Acc)* | **Implemented as replacement 3rd target (JSON metrics falsified in `bd55bd7`)** |
| **4. Iron Deficiency & Anemia (`Result` in External Lab Pipeline)** | `Result` column in `data/raw/anemia_public.csv` (NOT NFHS-5; `hw57` in `IAKR7EFL.DTA` was unused) | `y = df["Result"]` (`ml-service/training/train_anemia.py` L28); in `anemia_public.csv`, `Result=1` iff `Hemoglobin < 12.0` (Female `Gender=0`) or `Hemoglobin < 13.5` (Male `Gender=1`) | `XGBClassifier` (`ml-service/models/anemia/anemia_model.pkl`) | Test Acc `100.0%` (`60/60`), F1 `1.0`, AUC `1.0` on 60 test rows of `anemia_public.csv` — **invalid due to direct target leakage (`Hemoglobin` is a predictor) and adult male/female cutoffs.** | **Replaced NFHS-5 Anaemia with non-NFHS adult lab dataset containing target leakage** |
| **5. Stunting, Wasting, Malnutrition (External Anthropometric Pipeline)** | `Stunting`, `Wasting` in `stunting_wasting_dataset.csv` (Indonesian 100k dataset) | `df["Stunting"] = df["Stunting"].isin(["Stunted", "Severely Stunted"]).astype(int)`<br>`df["Wasting"] = df["Wasting"].isin(["Underweight", "Severely Underweight"]).astype(int)`<br>`df["Malnutrition"] = ((df["Stunting"] == 1) \| (df["Wasting"] == 1)).astype(int)` (`train_anthropometric.py` L27–29) | 3 `XGBClassifier` pipelines saved in `ml-service/models/anthropometric/model.pkl` | Test Acc on 20,000 rows:<br>• Stunting: `99.80%`<br>• Wasting: `100.00%`<br>• Malnutrition: `99.84%`<br>*(Uses height & weight directly to memorize WHO lookup table)* | **Implemented on non-NFHS Indonesian dataset (memorizes height/weight Z-score lookup)** |
| **6. Vitamin A Deficiency Risk** | No dataset variable (rule-based inference only) | `vit_a_risk = 0.75 if (dd < 3 and measles) else (0.55 if dd < 3 or (measles and diarrhea) else 0.18)` (`ml-service/inference/deficiency_predict.py` L89) | **None** (Hardcoded if/else heuristic) | None (No training or evaluation) | **Rule-based heuristic only (No ML model)** |
| **7. Acute Respiratory Infection (ARI / ALRI)** | `h31`, `h31b`, `h31c` exist in `IAKR7EFL.DTA`, but only `h31` (`cough_recent`) is in `dhs_clean.parquet` as an **input feature** | Frontend-only rule in `ResultsDashboard.jsx` L593–595: `childInfo?.cough_recent === 'Yes' ? 'Active Warning' : 'Low Active Risk'` | **None** | None | **Not implemented as a prediction target (`cough_recent` is an input predictor)** |

### Detailed Target Specifications Required by Phase 2

1. **Exact number of ML prediction targets currently implemented:**
   - **3 multi-label targets** in the NFHS-5 4-model suite (`"Stunting"`, `"Wasting"`, `"Malnutrition"`).
   - **3 single-target models** in the Indonesian Anthropometric suite (`"Stunting"`, `"Wasting"`, `"Malnutrition"`).
   - **1 binary target** in the external CBC Anemia model (`"Result"` / Anemia).
   - *(Plus 1 rule-based heuristic for Vitamin A risk and 2 frontend symptom badges for ALRI and Dysentery).*
2. **Number of classes & task type for each target:**
   - Every implemented ML target (`Stunting`, `Wasting`, `Malnutrition`, and `Result`/Anemia) is **binary classification (2 classes: `0` and `1`)**. None are multiclass or regression.
3. **Positive (`1`) and negative (`0`) class definitions & record counts:**
   - **NFHS-5 `dhs_clean.parquet` (198,849 total records; split into 139,194 Train / 29,827 Val / 29,828 Test):**
     - **Stunting:** `1` = Stunted (`hw70 <= -200`, i.e., HAZ $\le -2.00$ SD): **71,888 positive (36.15%)**, **126,961 negative (63.85%)**. (Test set: 10,725 positive, 19,103 negative).
     - **Wasting:** `1` = Wasted (`hw72 <= -200`, i.e., WHZ $\le -2.00$ SD): **37,193 positive (18.70%)**, **161,656 negative (81.30%)**. (Test set: 5,639 positive, 24,189 negative).
     - **Malnutrition (Composite):** `1` = Stunted OR Wasted (`stunting == 1 | wasting == 1`): **99,273 positive (49.92%)**, **99,576 negative (50.08%)**. (Test set: 14,891 positive, 14,937 negative).
   - **External `anemia_public.csv` (300 total records; split into 240 Train / 60 Test):**
     - **Result (Anemia):** `1` = Anemic (**115 positive, 38.33%**), `0` = Non-anemic (**185 negative, 61.67%**). (Test set: 23 positive, 37 negative).
   - **External `stunting_wasting_dataset.csv` (100,000 total records; split into 80,000 Train / 20,000 Test):**
     - **Stunting:** `1` = `"Stunted"` or `"Severely Stunted"` (**21,979 positive, 21.98%**), `0` = `"Normal"` or `"Tall"` (**78,021 negative, 78.02%**).
     - **Wasting:** `1` = `"Underweight"` or `"Severely Underweight"` (**22,114 positive, 22.11%**), `0` = `"Normal weight"` or `"Risk of Overweight"` (**77,886 negative, 77.89%**).
     - **Malnutrition:** `1` = `Stunting == 1 | Wasting == 1` (**39,289 positive, 39.29%**), `0` = Neither (**60,711 negative, 60.71%**).
4. **Are the targets genuinely derived from NFHS-5 data?**
   - `Stunting`, `Wasting`, and `Malnutrition` in `dhs_clean.parquet` **are genuinely derived from NFHS-5 (`IAKR7EFL.DTA`)**.
   - `Result` (Anemia) in `anemia_public.csv` is **NOT from NFHS-5** (it is an external 300-row adult CBC dataset).
   - `Stunting`, `Wasting`, and `Malnutrition` in `stunting_wasting_dataset.csv` are **NOT from NFHS-5** (Indonesian dataset).
5. **Nature of outputs:**
   - `Stunting` and `Wasting` are WHO anthropometric health outcomes.
   - `Malnutrition` is a **derived composite category** (`Stunting OR Wasting`).
   - `anemia_public.csv` `Result` is a deterministic laboratory threshold flag (`Hb < 12.0` F / `< 13.5` M).

---

## 4. Evidence of How Anaemia Was Replaced Across Git History (Phases 2 & 5)

Tracing the Git commit history reveals **three distinct stages** of how Anaemia was handled and replaced:

1. **Stage 1 — Synthetic Rule on Indonesian Dataset (Commits `42fc84e` on Sep 1, 2026 & `b5636ad` on Sep 12, 2026):**
   - In `b5636ad:ml-service/preprocessing/preprocessing.py` (lines 11–28, 153–166), the project defined 10 targets on `stunting_wasting_dataset.csv`, including `"Anemia"`, `"Iron Deficiency / Iron Deficiency Anemia"`, `"Vitamin A Deficiency"`, `"Protein-Energy Malnutrition (PEM)"`, and `"Micronutrient Deficiency Risk"`.
   - Because `stunting_wasting_dataset.csv` only had 4 columns (`gender`, `age_months`, `height_cm`, `weight_kg`), the code fabricated `muac_cm` with random noise and defined `"Anemia"` as:
     ```python
     # b5636ad:ml-service/preprocessing/preprocessing.py, line 154
     df["Anemia"] = ((df["muac_cm"] < 12.5) | ((df["Underweight"] == 1) & (df["Stunting"] == 1))).astype(int)
     ```
2. **Stage 2 — Replacement of Anaemia with Composite `"Malnutrition"` on NFHS-5 (`bf3ca51` on Sep 12, 2026 & `06c1c7b` on Oct 6, 2026):**
   - In commit `bf3ca51`, `ml-service/preprocessing/preprocessing.py` was rewritten to load `data/processed/dhs_clean.parquet` (198,849 NFHS-5 rows).
   - Because `dhs_clean.parquet` only had `stunting` and `wasting` outcome columns (`hw57` had not been extracted from `IAKR7EFL.DTA`), all 5 disease targets (including `Anemia`) were **deleted** from `TARGET_CONDITIONS` and replaced with the 3-target list:
     ```python
     # ml-service/preprocessing/preprocessing.py, lines 11-15 & 116-118
     TARGET_CONDITIONS = ["Stunting", "Wasting", "Malnutrition"]
     ...
     df["Stunting"] = df["stunting"].astype(int)
     df["Wasting"] = df["wasting"].astype(int)
     df["Malnutrition"] = (df["stunting"] | df["wasting"]).astype(int)
     ```
   - In commit `06c1c7b` (`"fix: avoid unsupported anemia risk claim"`), the developer explicitly updated `frontend/src/components/ResultsDashboard.jsx` (lines 573–587) to display `"Pediatric Anaemia: Not Evaluated — No haemoglobin measurement or labelled anaemia outcome is available in this screening dataset"` and documented in `PROJECT_TECHNICAL_DOCUMENTATION.md` (line 157 & lines 314–317) that `dhs_clean.parquet` lacked haemoglobin/anaemia labels.
3. **Stage 3 — Addition of External 300-Row Adult CBC Anemia Model (`420b275` on Oct 6, 2026 & `5d19f59` on Oct 9, 2026):**
   - Instead of re-running `01_prepare_data.py` on `data/raw/IAKR7EFL.DTA` to extract NFHS-5 `hw57` (child anemia level) and `hw56` (altitude-adjusted Hb), commit `5d19f59` added `data/raw/anemia_public.csv` (300 rows from `github.com/ajay3789/Anemia-Dataset`), `ml-service/training/train_anemia.py`, and `ml-service/inference/deficiency_predict.py`.

---

## 5. Complete Pipeline Reconstruction & Model Inventory (Phase 4)

### Pipeline 1: NFHS-5 4-Model Multi-Label Suite (`XGBoost`, `DNN`, `FT-Transformer`, `TabNet` + `Ensemble`)
- **Raw Dataset:** `data/raw/IAKR7EFL.DTA` (232,920 rows × 1,644 cols) $\rightarrow$ processed via `src/01_prepare_data.py` (on branch `base-mode`/`handoff-clean`) into `data/processed/dhs_clean.parquet` (198,849 rows × 33 cols).
- **Target Construction (`ml-service/preprocessing/preprocessing.py`, L116–118):**
  - `Stunting` (`hw70 <= -200`), `Wasting` (`hw72 <= -200`), `Malnutrition` (`stunting | wasting`).
- **Feature Selection (`preprocessing.py`, L18–43):** 18 features (9 numerical + 9 categorical):
  - *Numerical (9):* `child_age_months`, `birth_weight`, `breastfeeding_duration`, `birth_order`, `mother_bmi`, `anc_visits`, `hhsize`, `sanitation_risk_index` (engineered from `water_source`, `toilet_type`, `cooking_fuel`), `maternal_risk_score` (engineered from `mother_bmi < 18.5`, `anc_visits < 4`, `birth_weight < 2500`).
  - *Categorical (9):* `child_sex`, `education`, `wealth_quintile`, `residence`, `birth_size`, `diarrhea_recent`, `fever_recent`, `cough_recent`, `measles_vaccine`.
- **Preprocessing & Splitting (`preprocessing.py`, L249–287):**
  - Stratified 70% Train (`139,194`), 15% Validation (`29,827`), 15% Test (`29,828`) split (`random_state=42`, stratified on `Malnutrition`).
  - `DataPreprocessor` fits `StandardScaler` on the 9 numerical columns and `LabelEncoder` on the 9 categorical columns **strictly on the training split**, saving `ml-service/models/preprocessor.pkl`.
- **Model Training, Validation & Threshold Selection:**
  1. **XGBoost (`ml-service/training/train_xgboost.py` $\rightarrow$ `models/xgboost/xgboost_multilabel.pkl`):**
     - `MultiOutputClassifier(XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85))`.
     - Trains on all `139,194` train rows. Selects per-target threshold from `[0.20..0.60]` maximizing validation F1 (`0.30` for Stunting, `0.20` for Wasting, `0.30` for Malnutrition).
  2. **Tabular DNN (`ml-service/training/train_dnn.py` $\rightarrow$ `models/dnn/best_dnn.pt`):**
     - PyTorch `Linear(18->256)->BN->ReLU->Drop(0.2)->Linear(256->128)->BN->ReLU->Drop(0.2)->Linear(128->64)->BN->ReLU->Linear(64->3)`.
     - Trained with `BCEWithLogitsLoss(pos_weight=neg/pos)` on `139,194` train rows. Calibrated thresholds: `0.40` (Stunting), `0.45` (Wasting), `0.35` (Malnutrition).
  3. **FT-Transformer (`ml-service/training/train_transformer.py` $\rightarrow$ `models/transformer/best_transformer.pt`):**
     - PyTorch `FTTransformer(d_token=64, n_layers=2, n_heads=4, d_ffn=128, dropout=0.1)`.
     - **Subsamples training set to `50,000` rows** (`max_train_samples=50000`, L143 & L154–158) for CPU speed. Calibrated thresholds: `0.40` (Stunting), `0.50` (Wasting), `0.35` (Malnutrition).
  4. **TabNet (`ml-service/training/train_tabnet.py` $\rightarrow$ `models/tabnet/tabnet_model.zip`):**
     - `TabNetMultiTaskClassifier(n_d=16, n_a=16, n_steps=4, gamma=1.3, lambda_sparse=1e-4)`.
     - Trained on `139,194` train rows. Calibrated thresholds: `0.30` (Stunting), `0.20` (Wasting), `0.35` (Malnutrition).
  5. **DNN + FT-Transformer Ensemble (`ml-service/inference/ensemble_predict.py`):**
     - Inference-only weighted average (`0.9520` weight for DNN, `0.9380` weight for FT-Transformer — weights taken from the falsified ROC-AUC numbers). Never evaluated on the test split by any training/evaluation script.
- **Saved Predictions:** None of the scripts on `main` save test-set prediction CSVs (unlike `handoff-clean`, which saved `results/*_predictions.csv`).

### Pipeline 2: External Indonesian Anthropometric Model
- **Script:** `ml-service/training/train_anthropometric.py` $\rightarrow$ `ml-service/models/anthropometric/model.pkl` & `metrics.json`.
- **Dataset:** `stunting_wasting_dataset.csv` (`100,000` rows; `80,000` train / `20,000` test, stratified on `Stunting`).
- **Features (4):** `["sex", "age_months", "height_cm", "weight_kg"]`.
- **Targets (3):** `["Stunting", "Wasting", "Malnutrition"]`.

### Pipeline 3: External CBC Laboratory Anemia Model
- **Script:** `ml-service/training/train_anemia.py` $\rightarrow$ `ml-service/models/anemia/anemia_model.pkl`, `anemia_scaler.pkl`, `anemia_metrics.json`.
- **Dataset:** `data/raw/anemia_public.csv` (`300` rows; `240` train / `60` test, stratified on `Result`).
- **Features (5):** `["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]`.
- **Target (1):** `"Result"` (`0` = Non-anemic, `1` = Anemic).

---

## 6. Scientific Validity Audit of Existing Results & Data Leakage Analysis (Phases 5 & 6)

### A. Proof of Metric Fabrication in the 4 NFHS-5 Model JSON Files (Commit `bd55bd7`)

We loaded the actual saved model binaries (`xgboost_multilabel.pkl`, `best_dnn.pt`, `best_transformer.pt`, `tabnet_model.zip`) and `preprocessor.pkl` in read-only mode and evaluated them on the exact held-out test split (`29,828` records, `random_state=42`).

Every model's predictions **matched the `confusion_matrices` stored in its `*_metrics.json` file to the exact integer (`CM_match = True`)**. However, in commit `bd55bd7`, the `accuracy`, `f1_score`, `roc_auc`, `exact_match_accuracy`, `hamming_loss`, `macro_f1`, `weighted_f1`, and `macro_roc_auc` fields in those JSON files were manually overwritten with fabricated ~91%–95% values:

| Model & Condition | Saved Confusion Matrix `[[TN, FP], [FN, TP]]` (Verified True) | **TRUE Test Accuracy** (from Model & CM) | **Claimed JSON Accuracy** (`bd55bd7`) | **TRUE Test F1-Score** | **Claimed JSON F1-Score** | **TRUE Test ROC-AUC** | **Claimed JSON ROC-AUC** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost — Stunting** | `[[8234, 10869], [2377, 8348]]` | **55.59%** | *91.90%* | **0.5576** | *0.9115* | **0.6596** | *0.9260* |
| **XGBoost — Wasting** | `[[16262, 7927], [3000, 2639]]` | **63.37%** | *93.50%* | **0.3257** | *0.9240* | **0.6050** | *0.9410* |
| **XGBoost — Malnutrition** | `[[1625, 13312], [555, 14336]]` | **53.51%** | *92.40%* | **0.6740** | *0.9170* | **0.6489** | *0.9315* |
| **DNN — Stunting** | `[[6894, 12209], [1880, 8845]]` | **52.77%** | *94.10%* | **0.5567** | *0.9350* | **0.6555** | *0.9480* |
| **DNN — Wasting** | `[[9324, 14865], [1450, 4189]]` | **45.30%** | *95.60%* | **0.3393** | *0.9475* | **0.6006** | *0.9610* |
| **DNN — Malnutrition** | `[[2650, 12287], [1057, 13834]]` | **55.26%** | *94.85%* | **0.6746** | *0.9420* | **0.6465** | *0.9520* |
| **FT-Transformer — Stunting** | `[[7769, 11334], [2242, 8483]]` | **54.49%** | *92.80%* | **0.5555** | *0.9210* | **0.6535** | *0.9350* |
| **FT-Transformer — Wasting** | `[[10703, 13486], [1770, 3869]]` | **48.85%** | *94.20%* | **0.3365** | *0.9315* | **0.5995** | *0.9490* |
| **FT-Transformer — Malnutrition** | `[[2066, 12871], [767, 14124]]` | **54.28%** | *93.18%* | **0.6744** | *0.9250* | **0.6446** | *0.9380* |
| **TabNet — Stunting** | `[[8229, 10874], [2463, 8262]]` | **55.29%** | *90.80%* | **0.5534** | *0.9015* | **0.6486** | *0.9140* |
| **TabNet — Wasting** | `[[16268, 7921], [3082, 2557]]` | **63.11%** | *92.70%* | **0.3173** | *0.9160* | **0.5961** | *0.9320* |
| **TabNet — Malnutrition** | `[[2290, 12647], [968, 13923]]` | **54.35%** | *91.65%* | **0.6716** | *0.9090* | **0.6402** | *0.9245* |

Furthermore, in `ml-service/app.py` (lines 315–335), the `/models/confusion-matrix` API endpoint returns **a second set of hardcoded, fabricated confusion matrices** (e.g., DNN Stunting `tn: 18200, fp: 850, fn: 910, tp: 9868`) that contradict the confusion matrices inside `ml-service/models/dnn/dnn_metrics.json` (`[[6894, 12209], [1880, 8845]]`).

### B. Why Did the NFHS-5 Models Achieve Only ~0.60–0.66 ROC-AUC and Low Specificity?
1. **Intrinsic Predictability Without Current Anthropometrics:** Predicting WHO Stunting (`HAZ <= -2`) and Wasting (`WHZ <= -2`) using only socio-demographic, maternal, and birth history variables (without current height/weight or state/district/cluster effects, dietary intake, or wealth factor score `v191`) has a known epidemiological ceiling of ~0.65–0.73 ROC-AUC in DHS literature.
2. **Double-Weighting + Aggressive Threshold Lowering Destroyed Specificity:**
   - In `train_dnn.py` (L114–115) and `train_transformer.py` (L184–185), the loss function already applies `pos_weight = neg_counts / pos_counts` (which shifts predicted probabilities upward for minority classes like Wasting).
   - Then `tune_thresholds_on_val` lowers the classification threshold further to `0.20–0.45` solely to maximize F1, without any specificity constraint (despite the docstring claiming *"while preserving specificity"*).
   - As a result, for `Malnutrition`, the models predict almost every child as malnourished (recall `92.9%–96.3%`, specificity **`10.88%–17.74%`**), resulting in ~53%–55% accuracy.
3. **Bugs in `ml-service/preprocessing/preprocessing.py`:**
   - **Silent `"No Education"` Corruption Bug (L213–216 & `app.py` L110):**
     `edu_str = str(edu).capitalize()` converts `"No Education"` to `"No education"`. The next line checks `if edu_str in ["No Education", "Primary", "Secondary", "Higher"]`, which evaluates to **`False`**, silently replacing every `"No Education"` mother with `"Secondary"` during inference!
   - **DHS Morbidity & Vaccine Special Codes (L89–92):**
     - In `IAKR7EFL.DTA`, `h11` (diarrhea) and `h31` (cough) use code `2.0` for `"yes, last two weeks"`, `1.0` for `"yes, last 24 hours"`, and `8.0` for `"don't know"`. Checking only `== 2.0` works for `2.0` but would miss `1.0` if present.
     - For `measles_vaccine` (`h9`), `np.where(df["measles_vaccine"] == 0.0, "No", "Yes")` maps `NaN` (missing/dead/not eligible) and `8.0` ("don't know") to `"Yes"` (vaccinated).
   - **Coarse Median Imputation Without Missingness Indicators (L100–107):**
     - Missing values in `birth_weight`, `mother_bmi`, `anc_visits`, and `breastfeeding_duration` are filled with hardcoded constants (`2900.0`, `21.27`, `4.0`, `15.0`) before splitting (though constants do not leak across splits, they collapse the distribution and lose informative missingness, e.g., `m19 == 9996` "not weighed at birth" is strongly associated with home delivery and stunting).
   - **Discarded Informative NFHS-5 Columns Present in `dhs_clean.parquet`:**
     - `dhs_clean.parquet` has 33 columns, including continuous `wealth_score` (`v191`), `mother_weight`, `mother_height`, `age_hh_head_proxy` (`v012`, mother's age), `gender_hh_head` (`v151`), `dist_market_proxy` (`v467d`), `birth_weight_source` (`m19a`), and `mother_marital_status` (`v501`), which `ml-service/preprocessing/preprocessing.py` completely ignores (unlike `handoff-clean`, which used all 33 columns + one-hot encoding + interaction features and achieved `0.7251` ROC-AUC / `69.60%` accuracy on Stunting and `78.07%` accuracy on Wasting).

### C. Problems in the Standalone Anemia Model (`train_anemia.py` & `deficiency_predict.py`)
1. **Target Leakage (`Hemoglobin` Used to Predict `Hemoglobin < Cutoff`):**
   - `train_anemia.py` (L26) uses `features = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]` to predict `Result`.
   - In `data/raw/anemia_public.csv`, `Result` is deterministically defined by `Hemoglobin < 12.0` for `Gender == 0` and `Hemoglobin < 13.5` for `Gender == 1`. Including `Hemoglobin` as a predictor is direct definitional target leakage, trivially yielding `1.0000` test accuracy.
2. **Wrong Population (Adult Cutoffs Applied to Young Children):**
   - `anemia_public.csv` uses **adult** haemoglobin cutoffs (`12.0 g/dL` female / `13.5 g/dL` male). WHO defines anaemia in children aged 6–59 months as **$\text{Hb} < 11.0\text{ g/dL}$** for both sexes.
   - **Verified Bug:** When testing the frontend's own **"Load Sample: Normal Lab Values"** preset (`child_sex: "Male", child_age_months: 30, hemoglobin: 12.4 g/dL`), `screen_deficiencies()` returns **`risk_score: 98.8%, severity: "HIGH", is_flagged: True`** because `12.4 < 13.5` (adult male cutoff), even though the UI's own label next to it says `"12.4 g/dL (WHO pediatric threshold: 11.0 g/dL)"`!
3. **Duplicate Records Across Train/Test Split:**
   - Out of 300 rows in `anemia_public.csv`, **54 rows (18%) are exact duplicates**, leaking identical rows across the 80/20 random split.
4. **Unused `StandardScaler` Bug (`train_anemia.py`, L34–46):**
   - `train_anemia.py` fits `scaler = StandardScaler()` on `X_train` and saves `anemia_scaler.pkl`, but fits `model.fit(X_train, y_train)` on **unscaled** `X_train` and never uses `anemia_scaler.pkl` during inference.
5. **Documentation Mismatch (`PROJECT_TECHNICAL_DOCUMENTATION.md`, L323–344):**
   - Section 10.5 of `PROJECT_TECHNICAL_DOCUMENTATION.md` claims the anemia model is a *"class-balanced Logistic Regression model"* with `95.00%` test accuracy (`[[34, 3], [0, 23]]`), whereas `train_anemia.py` actually trains an `XGBClassifier` with `100.0%` test accuracy (`[[37, 0], [0, 23]]`).

---

## 7. Implementation Matrix & Component Status (Phases 8 & 9)

| Component | Actual status | Evidence and file path | Problem or uncertainty | Recommended next step |
| :--- | :--- | :--- | :--- | :--- |
| **Raw NFHS-5 Dataset (`IAKR7EFL.DTA`)** | **Present & Verified** | `data/raw/IAKR7EFL.DTA` (441.38 MB; 232,920 rows × 1,644 cols) | Contains valid `hw70` (Stunting), `hw72` (Wasting), `hw56`/`hw57` (Child Hb & Anaemia: 183,855 rows), and `h31`/`h31b`/`h31c` (ARI: 224,218 rows), but `hw57` and `h31b`/`h31c` were never extracted into `dhs_clean.parquet`. | Update data extraction script to pull `hw57` (and `h31b`, `h31c` if ARI is desired, plus maternal `v457` anemia as a predictor) from `IAKR7EFL.DTA`. |
| **Processed NFHS-5 Dataset (`dhs_clean.parquet`)** | **Completed (Partial Columns)** | `data/processed/dhs_clean.parquet` (198,849 rows × 33 cols) | Only contains `stunting` and `wasting` outcomes; missing `hw57` (child anaemia) and `h31b`/`h31c` (ARI). | Create a new processed parquet file (without overwriting `dhs_clean.parquet`) that includes genuine NFHS-5 `anemia` (`hw57`) and `ari` (`h31`, `h31b`, `h31c`). |
| **NFHS-5 Preprocessing (`preprocessing.py`)** | **Implemented with Bugs** | `ml-service/preprocessing/preprocessing.py` (L11–120, L213–216) | 1. Replaces missing Anaemia target with composite `Malnutrition = stunting \| wasting`.<br>2. Uses only 18 features, dropping `wealth_score`, `mother_height`, `mother_weight`, `age_hh_head_proxy`, etc.<br>3. `str(edu).capitalize()` bug turns `"No Education"` into `"No education"`, silently mapping it to `"Secondary"`. | Fix the `"No Education"` string normalization bug, retain all informative NFHS-5 predictors, and add genuine NFHS-5 target definitions. |
| **NFHS-5 4-Model Suite (`XGBoost`, `DNN`, `FT-Transformer`, `TabNet`)** | **Trained, but JSON Metrics Falsified** | `ml-service/training/train_*.py` & `ml-service/models/*/*_metrics.json` | Saved models are real and run end-to-end, but true test accuracy is **45.3%–63.4%** (ROC-AUC **0.60–0.66**). Commit `bd55bd7` manually overwrote JSON metrics to claim **90.8%–95.6%**. Also, threshold tuning + `pos_weight` collapses specificity to `11%–18%` on `Malnutrition`. | Restore honest programmatic metric logging from actual model predictions; fix double class-weighting / threshold calibration; add richer NFHS-5 predictors (as in branch `handoff-clean`). |
| **NFHS-5 Anaemia Prediction** | **Not Implemented (Replaced)** | Replaced by `Malnutrition` in `preprocessing.py` L118 and by external `anemia_public.csv` in `train_anemia.py` | Genuine NFHS-5 non-invasive anaemia prediction (using `hw57` in `IAKR7EFL.DTA`, where `1,2,3` = Anemic [66.93%] and `4` = Not Anemic [33.07%] across 175,313 children) was never trained. | Build and evaluate a genuine NFHS-5 anaemia prediction pipeline using `hw57` (and maternal anemia `v457`, iron supplementation `m45`, birth/maternal/wealth covariates) without leaking child `hw53`/`hw56`. |
| **External Lab Anemia Model (`train_anemia.py`)** | **Implemented with Critical Flaws** | `ml-service/training/train_anemia.py`, `ml-service/inference/deficiency_predict.py`, `data/raw/anemia_public.csv` | 1. Trained on 300 adult CBC rows (not NFHS-5).<br>2. Direct target leakage (`Hemoglobin` predicts `Result`).<br>3. Uses adult male cutoff (`< 13.5 g/dL`), falsely flagging a healthy boy with `Hb = 12.4 g/dL` as **98.8% HIGH RISK**.<br>4. 54 duplicate rows; `anemia_scaler.pkl` unused. | If keeping a lab-based Hb module in the UI, use WHO pediatric cutoffs ($\text{Hb} < 11.0\text{ g/dL}$ for ages 6–59 mo) clearly labeled as a WHO clinical lookup rather than claiming a 100%-accurate ML model on adult data. |
| **ARI (Acute Respiratory Infection) Prediction** | **Not Implemented (Rule Badge Only)** | `preprocessing.py` L39 (`cough_recent` input); `ResultsDashboard.jsx` L589–603 | Only `h31` (`cough_recent`) was extracted as an input feature; `h31b` (short rapid breaths) and `h31c` (chest problem) were left in `IAKR7EFL.DTA`. UI shows a static warning badge when cough/fever is `"Yes"`. | Decide whether ARI should be a trained NFHS-5 target (constructed from `h31`, `h31b`, `h31c` in `IAKR7EFL.DTA`) or remain a clinical symptom input. |
| **Indonesian Anthropometric Model (`train_anthropometric.py`)** | **Implemented on Non-NFHS Data** | `ml-service/training/train_anthropometric.py`, `stunting_wasting_dataset.csv` | Trains on 100k Indonesian records using `[sex, age_months, height_cm, weight_kg]`, achieving 99.8%–100% accuracy by memorizing the height/weight Z-score lookup table. Also maps `"Underweight"` labels to `"Wasting"`. | Clearly separate or label this as an anthropometric Z-score classifier (or replace with exact WHO Z-score calculation from `hw2`/`hw3` or the WHO LMS CSVs in root). |
| **Flask API & React Dashboard (`ml-service/app.py`, `frontend/`)** | **Implemented & Functional (Contains Hardcoded Numbers)** | `ml-service/app.py` (L315–415), `frontend/src/components/*.jsx` | `/models/confusion-matrix`, `/analytics/feature-importance`, and `/dashboard/summary` return hardcoded numbers in `app.py` that do not match the actual saved models. | Connect API analytics endpoints dynamically to verified evaluation JSON outputs after retraining/re-evaluating models honestly. |
| **Earlier Honest NFHS-5 Pipeline (`handoff-clean` branch)** | **Completed on Separate Git Branch** | Git branch `handoff-clean` (commit `0fdc9f2`): `src/01_prepare_data.py`, `src/02_run_models.py`, `src/train_dnn*.py`, `src/train_tabnet*.py` | Uses all 33 columns of `dhs_clean.parquet` + feature engineering and logs genuine, un-doctored test metrics (Stunting ROC-AUC `0.7251`, Acc `69.60%`; Wasting Acc `78.07%`), but is not integrated into `main`. | Can be used as a verified baseline reference for recovering honest NFHS-5 Stunting and Wasting models. |

---

## 8. Recommended Next Steps (Phase 8.10)

Before modifying any code or retraining any models, we recommend deciding on the following recovery plan:

1. **Decide on the Exact Target Set for Your NFHS-5 Paper & System:**
   - **Option A (4 Genuine NFHS-5 Targets from `IAKR7EFL.DTA`):** Extract `hw57` (Child Anaemia) and `h31`/`h31b`/`h31c` (ARI) alongside `hw70` (Stunting) and `hw72` (Wasting) from `data/raw/IAKR7EFL.DTA` into a new processed dataset (`data/processed/nfhs5_full_targets.parquet`), so all 4 original outcomes (**Stunting, Wasting, Anaemia, ARI**) are genuinely constructed and predicted from official NFHS-5 data.
   - **Option B (3 Genuine NFHS-5 Targets: Stunting, Wasting, Anaemia):** Extract `hw57` (Child Anaemia) from `IAKR7EFL.DTA` to replace the synthetic `"Malnutrition"` (`Stunting | Wasting`) composite target, keeping `cough_recent`, `fever_recent`, and `diarrhea_recent` as morbidity predictors.
   - **Option C (Keep Stunting, Wasting, and Composite Malnutrition):** Keep the current 3 anthropometric targets on `dhs_clean.parquet`, fix the preprocessing/calibration bugs, and restore honest evaluation metrics.

2. **Fix Data Leakage & Scientific Integrity Issues:**
   - Remove the manually fabricated 91%–95% numbers in `ml-service/models/*/*_metrics.json` and `all_models_benchmark.json`, and replace the hardcoded fake confusion matrices in `ml-service/app.py` (`/models/confusion-matrix`) with dynamically loaded, verified test set confusion matrices.
   - Fix the `"No Education"` `.capitalize()` bug in `ml-service/preprocessing/preprocessing.py` (line 214) and `ml-service/app.py` (line 110).
   - Fix the threshold calibration in `train_dnn.py` and `train_transformer.py` so `pos_weight` and low probability thresholds (`0.20–0.35`) do not collapse test specificity to 11%–18%.
   - Fix the standalone lab anemia check in `deficiency_predict.py` so a child with $\text{Hb} \ge 11.0\text{ g/dL}$ is never flagged as high-risk anemic due to adult male cutoffs (`13.5 g/dL`) from `anemia_public.csv`.
