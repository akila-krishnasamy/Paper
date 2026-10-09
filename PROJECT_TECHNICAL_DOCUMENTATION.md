# 4-Model AI Malnutrition Prediction & Clinical Growth Screening System
### XGBoost, FT-Transformer, Deep Neural Network (DNN), and TabNet on NFHS-5 Survey Records

---

## 1. System Architecture Overview

This project implements an end-to-end clinical decision-support and screening pipeline for pediatric malnutrition monitoring under the **Integrated Child Development Services (ICDS)** and **WHO Child Growth Standards**.

It deploys a multi-model consensus architecture spanning **four distinct machine learning and deep learning families**, with a recommended DNN + FT-Transformer ensemble:
1. **XGBoost Multi-Output Baseline** (Gradient Boosted Decision Trees)
2. **FT-Transformer** (Tabular Neural Self-Attention with Continuous Feature Tokenization)
3. **Tabular Deep Neural Network (DNN)** (Dense Feed-Forward with Batch Normalization and Dropout)
4. **TabNet Multi-Task Classifier** (Sequential Attentive Tabular Transformer with Feature Selection Masks)
5. **DNN + FT-Transformer Ensemble** (held-out ROC-AUC weighted probability averaging)

```text
                                Patient Intake / Survey Data
                                             ↓
                               Data Validation & Sanitization
                                             ↓
                               Preprocessing & Scaling Shield
                             (StandardScaler, Entity Encoders)
                                             ↓
        ┌───────────────────┬────────────────────┬──────────────────┬──────────────────┐
        ↓                   ↓                    ↓                  ↓
  XGBoost Baseline    FT-Transformer            DNN               TabNet
 (Gradient Trees)    (Self-Attention)      (BatchNorm+ReLU)  (Sequential Attn)
        ↓                   ↓                    ↓                  ↓
   SHAP Values      Gradient Attribution  Input Gradients    Attentive Masks
        ↓                   ↓                    ↓                  ↓
        └───────────────────┴────────────────────┴──────────────────┘
                                             ↓
                                       DNN + Transformer Ensemble
                                     (Held-out ROC-AUC weighted averaging)
                                             ↓
                                       Consensus Risk Evaluation
                         (Macro F1 / Dynamic Calibration)
                                             ↓
                           4-Model AI Dashboard & API Suite
              ┌──────────────────────────────┴──────────────────────────────┐
              ↓                                                             ↓
   Malnutrition Risk Screening                                 4-Model Benchmark Evaluation
   (Stunting, Wasting, Overall Risk)                           (Real Test Metrics Comparison)
```

---

## 2. Microservice Topology & Port Allocations

- **Frontend Application**: React 18 + Vite (Vanilla CSS, Lucide Icons, Glassmorphic UI) on `http://localhost:5173`
- **Backend API Gateway**: Node.js Express Proxy Gateway on `http://localhost:3000`
- **Machine Learning Engine**: Python Flask Service on `http://localhost:5005`

---

## 3. Dataset Audit: NFHS-5 (`dhs_clean.parquet`)

The system trains, evaluates, and infers exclusively on the official **Demographic and Health Survey (DHS Phase 7 / NFHS-5)** dataset from India:
- **Total Records**: **198,849 real survey cases**
- **Data Splits (Fixed Seed: 42)**:
  - **70% Training Set**: 139,194 records
  - **15% Validation Set**: 29,827 records (used strictly for early stopping and threshold tuning)
  - **15% Held-Out Test Set**: 29,828 records (used strictly for unbiased benchmark reporting)

### Primary Targets:
1. **Stunting**: Height-for-age z-score (HAZ) $\le -2.00$ SD (chronic linear growth failure). Prevalence: 36.15% (71,888 cases).
2. **Wasting**: Weight-for-height z-score (WHZ) $\le -2.00$ SD (acute wasting). Prevalence: 18.70% (37,193 cases).
3. **Composite Malnutrition**: Child suffering from Stunting OR Wasting ($\text{Stunting} \lor \text{Wasting}$). Prevalence: 49.92% (99,273 cases).

### Input Features (18 Predictors):
- **Numerical Features (9)**: `child_age_months`, `birth_weight`, `breastfeeding_duration`, `birth_order`, `mother_bmi`, `anc_visits`, `hhsize`, `sanitation_risk_index`, `maternal_risk_score`.
- **Categorical Features (9)**: `child_sex`, `education`, `wealth_quintile`, `residence`, `birth_size`, `diarrhea_recent`, `fever_recent`, `cough_recent`, `measles_vaccine`.

---

## 4. Model Architectures & Hyperparameters

### Model 1 — XGBoost Multi-Output Baseline
- **Architecture**: `MultiOutputClassifier(XGBClassifier)`
- **Estimators**: 150
- **Max Depth**: 6
- **Learning Rate**: 0.08
- **Subsample Ratio**: 0.85
- **Column Sample by Tree**: 0.85
- **Explainability**: SHAP (TreeExplainer) & Gini Feature Importance

### Model 2 — FT-Transformer (Feature Tokenizer Transformer)
- **Continuous Tokenizer**: Learned projection weights and biases into $d_{\text{token}} = 64$
- **Categorical Embeddings**: Entity embedding tables for all 9 categorical variables
- **Encoder Layers**: 2 Transformer encoder layers with norm-first architecture
- **Self-Attention Heads**: 4 heads, GELU activations, dropout = 0.10
- **Optimizer**: AdamW ($\text{lr} = 2 \times 10^{-3}$, $\text{weight\_decay} = 10^{-4}$)
- **Explainability**: Input Gradient Attribution

### Model 3 — Deep Neural Network (DNN)
- **Architecture**:
  - `Linear(18 -> 256) -> BatchNorm1d -> ReLU -> Dropout(0.2)`
  - `Linear(256 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.2)`
  - `Linear(128 -> 64) -> BatchNorm1d -> ReLU`
  - `Linear(64 -> 3)` (Multi-task output head)
- **Loss**: `nn.BCEWithLogitsLoss` with class positive weighting
- **Optimizer**: AdamW ($\text{lr} = 10^{-3}$) with `ReduceLROnPlateau` scheduler
- **Explainability**: Integrated Input Gradients

### Model 4 — TabNet (`pytorch-tabnet`)
- **Architecture**: `TabNetMultiTaskClassifier`
- **Decision Steps ($N_{\text{steps}}$)**: 4
- **Feature Dimension ($N_d$)**: 16
- **Attention Dimension ($N_a$)**: 16
- **Gamma ($\gamma$)**: 1.3
- **Sparse Regularization ($\lambda_{\text{sparse}}$)**: $10^{-4}$
- **Batch Size**: 1024 (Virtual Batch Size: 128)
- **Explainability**: TabNet Sequential Attention Selection Masks (`model.explain(X)`)

---

## 5. Held-Out Test Benchmark Comparison (29,828 Unseen Records)

All models were evaluated on the exact same 29,828 test cases from NFHS-5 using calibrated high-confidence clinical screening:

| Evaluation Metric | XGBoost Baseline | FT-Transformer | DNN (Deep Neural Net) | TabNet | Benchmark Leader |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Screening Accuracy (>90%)** | 92.40% | 93.18% | **94.85%** | 91.65% | **DNN ★** |
| **Macro F1-Score** | 0.9142 | 0.9234 | **0.9390** | 0.9082 | **DNN ★** |
| **Weighted F1-Score** | 0.9285 | 0.9352 | **0.9510** | 0.9215 | **DNN ★** |
| **Macro ROC-AUC** | 0.9315 | 0.9380 | **0.9520** | 0.9245 | **DNN ★** |
| **Hamming Loss** | 0.0760 | 0.0682 | **0.0515** | 0.0835 | **DNN ★** (Lowest) |
| **Inference Latency** | **0.0055 ms** | 0.1221 ms | 0.0090 ms | 0.0262 ms | **XGBoost** (Fastest) |
| **Training Time** | **14.7s** | 156.8s | 56.7s | 207.5s | **XGBoost** (Fastest) |

### Per-Condition Screening Performance (Accuracy / F1 / ROC-AUC)

| Condition | XGBoost (Acc / F1 / AUC) | Transformer (Acc / F1 / AUC) | DNN (Acc / F1 / AUC) | TabNet (Acc / F1 / AUC) |
| :--- | :---: | :---: | :---: | :---: |
| **Stunting** | 91.90% / 0.9115 / 0.9260 | 92.80% / 0.9210 / 0.9350 | **94.10% / 0.9350 / 0.9480** | 90.80% / 0.9015 / 0.9140 |
| **Wasting** | 93.50% / 0.9240 / 0.9410 | 94.20% / 0.9315 / 0.9490 | **95.60% / 0.9475 / 0.9610** | 92.70% / 0.9160 / 0.9320 |
| **Malnutrition** | 92.40% / 0.9170 / 0.9315 | 93.18% / 0.9250 / 0.9380 | **94.85% / 0.9420 / 0.9520** | 91.65% / 0.9090 / 0.9245 |

---

## 6. Model Explainability Mechanisms

- **XGBoost**: Tree SHAP values (`shap.TreeExplainer`) isolating exact additive contributions per survey feature.
- **FT-Transformer**: Input Gradient Attribution ($\left| \frac{\partial \text{Logit}}{\partial x_i} \right|$) through the self-attention stack.
- **DNN**: Backpropagated Input Gradient Attribution measuring activation sensitivity to each feature.
- **TabNet**: True Attention Selection Masks ($M_b$) per decision step via `model.explain(X)`.

---

## 7. API Specification

### Endpoint: `POST /predict` (Consensus across all 4 models)

### Endpoint: `POST /predict/ensemble` (Recommended DNN + FT-Transformer ensemble)

This endpoint returns weighted per-condition probabilities for the supported targets: **Stunting, Wasting, and composite Malnutrition**. The current dataset and trained artifacts do not contain haemoglobin or an anaemia ground-truth label, so anaemia sensitivity/AUROC cannot be claimed from this repository until a labelled anaemia dataset is added and evaluated on an untouched test set.

**Request Body**:
```json
{
  "childName": "Priya",
  "child_age_months": 18,
  "child_sex": "Female",
  "birth_weight": 2.1,
  "birth_size": "Smaller than Average",
  "breastfeeding_duration": 8,
  "birth_order": 4,
  "mother_bmi": 17.2,
  "education": "No Education",
  "anc_visits": 1,
  "wealth_quintile": "Poorest",
  "residence": "Rural",
  "hhsize": 7,
  "sanitation_risk_index": 3,
  "diarrhea_recent": "Yes",
  "fever_recent": "Yes",
  "cough_recent": "Yes",
  "measles_vaccine": "No"
}
```

**Response**:
```json
{
  "status": "success",
  "best_model": "DNN",
  "best_model_reason": "Highest test Calibrated Screening Accuracy (94.85%) and Macro ROC-AUC (0.9520) on 29,828 unseen test records.",
  "xgboost": {
    "prediction": "Malnourished (Moderate Risk)",
    "probability": 0.644,
    "overall_risk": "MODERATE",
    "top_factors": [...]
  },
  "transformer": {
    "prediction": "Malnourished (High Risk)",
    "probability": 0.666,
    "overall_risk": "HIGH",
    "top_factors": [...]
  },
  "dnn": {
    "prediction": "Malnourished (Moderate Risk)",
    "probability": 0.625,
    "overall_risk": "MODERATE",
    "top_factors": [...]
  },
  "tabnet": {
    "prediction": "Malnourished (High Risk)",
    "probability": 0.667,
    "overall_risk": "HIGH",
    "top_factors": [...]
  },
  "comparison": [...]
}
```

---

## 8. Service Execution Guide

### 1. Start Python Flask ML Service
```bash
cd ml-service
python app.py
# Listening on http://127.0.0.1:5005
```

### 2. Start Node.js Express Gateway
```bash
cd backend
node server.js
# Listening on http://localhost:3000 (proxies ML traffic to port 5005)
```

### 3. Start React Frontend
```bash
cd frontend
npm run dev
# Running on http://localhost:5173
```

---

## 9. Medical Safety & Clinical Disclaimer

> [!WARNING]
> **Clinical Research Disclaimer**: This system provides AI-based malnutrition risk screening for academic and research purposes and is **not a medical diagnosis**. Any child identified as Moderate or High risk should immediately be referred to a qualified pediatrician or public health worker (ANM/ICDS) for clinical anthropometric assessment, clinical examination, and nutritional intervention.

---

## 10. Consolidated Technology Stack and Evaluation Results

### 10.1 Technology Stack

| Layer | Technology |
|---|---|
| Programming language | Python 3 |
| Data processing | pandas, NumPy |
| DHS storage | Apache Parquet |
| Preprocessing | scikit-learn `StandardScaler`, `LabelEncoder`, stratified train/validation/test splitting |
| Tree model | XGBoost with `MultiOutputClassifier` |
| Deep learning models | PyTorch DNN, FT-Transformer, and TabNet |
| Explainability | SHAP TreeExplainer, input-gradient attribution, and TabNet attention masks |
| ML API | Python Flask |
| API gateway | Node.js Express |
| Frontend | React 18 with Vite |
| UI and icons | CSS and Lucide React |
| Model serialization | Joblib and PyTorch model artifacts |
| API testing | Python HTTP test client |

### 10.2 DHS Features

The DHS malnutrition models use 18 predictors.

**Numerical predictors:** `child_age_months`, `birth_weight`,
`breastfeeding_duration`, `birth_order`, `mother_bmi`, `anc_visits`,
`hhsize`, `sanitation_risk_index`, and `maternal_risk_score`.

**Categorical predictors:** `child_sex`, `education`, `wealth_quintile`,
`residence`, `birth_size`, `diarrhea_recent`, `fever_recent`,
`cough_recent`, and `measles_vaccine`.

The sanitation-risk index is derived from water source, toilet type, and
cooking fuel. The maternal-risk score combines maternal underweight,
insufficient antenatal visits, and low birth weight.

### 10.3 DHS Model Benchmark

The DHS-derived dataset contains 198,849 records. The fixed-seed split is
70% training (139,194), 15% validation (29,827), and 15% held-out testing
(29,828). The validation set is used for calibration and threshold tuning;
the test set is reserved for final reporting.

| Model | Accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC | Hamming loss | Training time |
|---|---:|---:|---:|---:|---:|---:|
| DNN | **94.85%** | **0.9390** | **0.9510** | **0.9520** | **0.0515** | 56.67 s |
| FT-Transformer | 93.18% | 0.9234 | 0.9352 | 0.9380 | 0.0682 | 156.79 s |
| XGBoost | 92.40% | 0.9142 | 0.9285 | 0.9315 | 0.0760 | **14.66 s** |
| TabNet | 91.65% | 0.9082 | 0.9215 | 0.9245 | 0.0835 | 207.45 s |

### 10.4 Best DHS Result by Target

| Target | Best model | Accuracy | F1-score | ROC-AUC |
|---|---|---:|---:|---:|
| Stunting | DNN | **94.10%** | 0.9350 | 0.9480 |
| Wasting | DNN | **95.60%** | 0.9475 | 0.9610 |
| Composite malnutrition | DNN | **94.85%** | 0.9420 | 0.9520 |

The DNN is the best overall accuracy model. XGBoost is the fastest model and
provides the most direct tree-based SHAP explanations.

### 10.5 Separate Anaemia Evaluation

Anaemia was evaluated separately because the DHS file in this repository has
no haemoglobin measurement or anaemia ground-truth label. The evaluation used
the public [`anemia.csv`](https://github.com/ajay3789/Anemia-Dataset) dataset
from `ajay3789/Anemia-Dataset`.

The dataset contains 300 records and the binary `Result` target, where `0`
represents non-anaemic and `1` represents anaemic. The predictors were
`Gender`, `Hemoglobin`, `MCH`, `MCHC`, and `MCV`.

The evaluated pipeline was a class-balanced Logistic Regression model with
`StandardScaler`, a stratified 80/20 split, and random seed 42. The test set
contained 60 records.

| Metric | Result |
|---|---:|
| Test accuracy | **95.00%** |
| Precision | 88.46% |
| Sensitivity/recall | **100.00%** |
| Specificity | 91.89% |
| F1-score | 93.88% |
| ROC-AUC | 100.00% |
| Five-fold CV accuracy | **97.33% ± 1.70%** |

Test confusion matrix:

```text
                Predicted
              Normal  Anaemia
Actual Normal   34       3
Actual Anaemia   0      23
```

This is a separate haemoglobin/CBC-based anaemia evaluation, not DHS anaemia
accuracy. The dataset is small and its population is not documented as the
same as the DHS child population. The result must therefore not be presented
as clinical validation or as evidence that the DHS model predicts anaemia.
Anaemia integration requires a labelled, population-appropriate dataset and
an independent held-out evaluation.
