import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.transformer_predict import get_transformer_predictor
from inference.xgboost_predict import get_xgboost_predictor
from inference.dnn_predict import predict_dnn
from inference.tabnet_predict import predict_tabnet
from inference.ensemble_predict import predict_ensemble
from inference.anthropometric_predict import predict_anthropometric
from preprocessing.preprocessing import TARGET_CONDITIONS, ALL_FEATURES

app = Flask(__name__)


def find_model_dir():
    candidates = [
        Path(__file__).resolve().parent / "models",
        Path(r"e:\SEM-7\paper_\ml-service\models"),
        Path(r"e:\Project\SEM7\Malnutrtion\ml-service\models")
    ]
    for c in candidates:
        if c.exists() and (c / "preprocessor.pkl").exists():
            return c
    return candidates[0]


def parse_child_input(data):
    """
    Validate, sanitize, and normalize child input parameters for the DHS 18-feature pipeline.
    Seamlessly supports both DHS survey format and legacy clinical formats.
    """
    if not data:
        raise ValueError("Invalid or missing JSON payload")

    # Age in months
    age = data.get("child_age_months", data.get("age_months", data.get("age", 24)))
    try:
        age = float(age)
    except (TypeError, ValueError):
        age = 24.0

    # Birth weight (grams, e.g. 2900)
    bw = data.get("birth_weight")
    if bw is None:
        wt = data.get("weight_kg", data.get("weight"))
        bw = 2900.0 if wt is None else min(4500.0, max(1500.0, float(wt) * 260.0))
    else:
        bw = float(bw)
        if bw < 15.0:  # If passed in kg (e.g. 2.9), convert to grams
            bw = bw * 1000.0

    # Breastfeeding duration
    bf = data.get("breastfeeding_duration", data.get("breastfeeding_months", 15.0))

    # Mother's BMI
    mbmi = data.get("mother_bmi")
    if mbmi is None:
        m_wt = data.get("mother_weight")
        m_ht = data.get("mother_height")
        if m_wt and m_ht:
            mbmi = float(m_wt) / ((float(m_ht) / 100) ** 2)
        else:
            mbmi = 21.3
    else:
        mbmi = float(mbmi)

    # ANC visits & HH Size
    anc = float(data.get("anc_visits", 4.0))
    hhsize = float(data.get("hhsize", data.get("household_size", 6.0)))
    birth_order = float(data.get("birth_order", 2.0))

    # Sanitation Risk Index (0 to 3)
    san = data.get("sanitation_risk_index")
    if san is None:
        wsi = data.get("water_sanitation_index", 3)
        san = max(0, min(3, 5 - int(wsi)))
    else:
        san = float(san)

    # Maternal Risk Score (0 to 3)
    m_under = 1.0 if mbmi < 18.5 else 0.0
    low_anc = 1.0 if anc < 4.0 else 0.0
    low_bw = 1.0 if bw < 2500.0 else 0.0
    m_risk = m_under + low_anc + low_bw

    # Categorical features
    raw_sex = str(data.get("child_sex", data.get("gender", data.get("sex", "Male")))).lower()
    child_sex = "Female" if raw_sex in ["female", "f", "girl", "2"] else "Male"

    raw_res = str(data.get("residence", "Rural")).lower()
    residence = "Urban" if raw_res in ["urban", "1", "city"] else "Rural"

    return {
        "child_age_months": age,
        "height_cm": float(data.get("height_cm", data.get("height", 85.0))),
        "weight_kg": float(data.get("weight_kg", data.get("weight", 11.0))),
        "birth_weight": bw,
        "breastfeeding_duration": float(bf),
        "birth_order": birth_order,
        "mother_bmi": mbmi,
        "anc_visits": anc,
        "hhsize": hhsize,
        "sanitation_risk_index": san,
        "maternal_risk_score": m_risk,
        "child_sex": child_sex,
        "education": str(data.get("education", "Secondary")).capitalize(),
        "wealth_quintile": str(data.get("wealth_quintile", "Middle")).capitalize(),
        "residence": residence,
        "birth_size": str(data.get("birth_size", "Average")),
        "diarrhea_recent": "Yes" if str(data.get("diarrhea_recent", "No")).lower() in ["yes", "y", "true", "1", "2"] else "No",
        "fever_recent": "Yes" if str(data.get("fever_recent", "No")).lower() in ["yes", "y", "true", "1", "2"] else "No",
        "cough_recent": "Yes" if str(data.get("cough_recent", "No")).lower() in ["yes", "y", "true", "1", "2"] else "No",
        "measles_vaccine": "Yes" if str(data.get("measles_vaccine", "Yes")).lower() not in ["no", "n", "false", "0"] else "No"
    }


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "DNN + FT-Transformer Ensemble Malnutrition Screening Engine",
        "supported_models": ["XGBoost", "FT-Transformer", "DNN", "TabNet", "DNN-Transformer Ensemble"],
        "dataset": "NFHS-5 dhs_clean.parquet (198,849 records)",
        "targets": TARGET_CONDITIONS,
        "medical_disclaimer": "AI-based malnutrition risk screening for academic/research purposes and is not a medical diagnosis."
    })


@app.route("/predict/xgboost", methods=["POST"])
def route_predict_xgboost():
    try:
        data = request.get_json(force=True)
        input_dict = parse_child_input(data)
        predictor = get_xgboost_predictor()
        res = predictor.predict(input_dict)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/transformer", methods=["POST"])
def route_predict_transformer():
    try:
        data = request.get_json(force=True)
        input_dict = parse_child_input(data)
        predictor = get_transformer_predictor()
        res = predictor.predict(input_dict)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/dnn", methods=["POST"])
def route_predict_dnn():
    try:
        data = request.get_json(force=True)
        input_dict = parse_child_input(data)
        res = predict_dnn(input_dict)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/tabnet", methods=["POST"])
def route_predict_tabnet():
    try:
        data = request.get_json(force=True)
        input_dict = parse_child_input(data)
        res = predict_tabnet(input_dict)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/ensemble", methods=["POST"])
def route_predict_ensemble():
    """Return the DNN + FT-Transformer weighted probability ensemble."""
    try:
        data = request.get_json(force=True)
        input_dict = parse_child_input(data)
        result = predict_ensemble(
            input_dict,
            predict_dnn,
            get_transformer_predictor(),
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/anthropometric", methods=["POST"])
def route_predict_anthropometric():
    """Predict disease risk from current anthropometric measurements."""
    try:
        data = request.get_json(force=True)
        input_dict = parse_child_input(data)
        result = predict_anthropometric(input_dict)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def route_predict_consensus():
    """
    Consensus endpoint running all 4 models: XGBoost, FT-Transformer, DNN, and TabNet.
    Returns individual model predictions, confidence probabilities, condition risks,
    model-specific explainability, and side-by-side comparison.
    """
    try:
        data = request.get_json(force=True)
        input_dict = parse_child_input(data)

        # 1. XGBoost
        xgb_res = get_xgboost_predictor().predict(input_dict)

        # 2. FT-Transformer
        trans_res = get_transformer_predictor().predict(input_dict)

        # 3. DNN
        dnn_res = predict_dnn(input_dict)

        # 4. TabNet
        tabnet_res = predict_tabnet(input_dict)

        # 5. Recommended DNN + FT-Transformer ensemble
        ensemble_res = predict_ensemble(
            input_dict,
            predict_dnn,
            get_transformer_predictor(),
        )

        # Load benchmark summary to identify leading test model
        bench_file = find_model_dir() / "all_models_benchmark.json"
        best_model_name = "XGBoost"
        best_model_reason = "Evaluated on held-out NFHS-5 test split."

        if bench_file.exists():
            try:
                with open(bench_file, "r") as f:
                    bench_data = json.load(f)
                    best_model_name = bench_data.get("best_overall_model", "XGBoost")
                    best_model_reason = bench_data.get("best_model_reason", "")
            except Exception:
                pass

        # Build 4-way comparison table
        models_map = {
            "XGBoost": xgb_res,
            "FT-Transformer": trans_res,
            "DNN": dnn_res,
            "TabNet": tabnet_res
        }

        comparison_list = []
        for cond in TARGET_CONDITIONS:
            row = {"condition": cond}
            for m_key, m_val in models_map.items():
                matching_cond = next((c for c in m_val.get("conditions", []) if c["condition"] == cond), None)
                if matching_cond:
                    row[m_key] = {
                        "probability": matching_cond["probability"],
                        "percentage": matching_cond["percentage"],
                        "risk_flag": matching_cond["risk_flag"],
                        "severity": matching_cond["severity"]
                    }
                else:
                    row[m_key] = {"percentage": 0.0, "risk_flag": False, "severity": "LOW"}
            comparison_list.append(row)

        return jsonify({
            "status": "success",
            "best_model": best_model_name,
            "best_model_reason": best_model_reason,
            "xgboost": xgb_res,
            "transformer": trans_res,
            "dnn": dnn_res,
            "tabnet": tabnet_res,
            "ensemble": ensemble_res,
            "comparison": comparison_list,
            "medical_disclaimer": "This system provides AI-based malnutrition risk screening for academic/research purposes and is not a medical diagnosis. For high-risk results, immediate evaluation by a qualified healthcare professional is recommended."
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/compare", methods=["GET"])
@app.route("/benchmark", methods=["GET"])
def route_model_benchmark():
    """Retrieve full 4-model held-out test evaluation benchmark metrics."""
    bench_file = find_model_dir() / "all_models_benchmark.json"
    if bench_file.exists():
        with open(bench_file, "r") as f:
            return jsonify(json.load(f)), 200

    # Fallback to individual metrics files if combined is not yet generated
    models_dir = find_model_dir()
    summary = {}
    for m in ["xgboost", "transformer", "dnn", "tabnet"]:
        mf = models_dir / m / f"{m}_metrics.json"
        if mf.exists():
            with open(mf, "r") as f:
                summary[m] = json.load(f)

    return jsonify({"status": "partial", "models": summary}), 200


if __name__ == "__main__":
    print("=" * 70)
    print("   4-MODEL AI MALNUTRITION PREDICTION ENGINE (Flask)")
    print("   Serving on: http://127.0.0.1:5005")
    print("   Models: XGBoost, FT-Transformer, DNN, TabNet")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5005, debug=False)
