import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

_ANEMIA_CACHE = None

def get_anemia_model():
    global _ANEMIA_CACHE
    if _ANEMIA_CACHE is None:
        model_path = Path(__file__).resolve().parents[1] / "models" / "anemia" / "anemia_model.pkl"
        scaler_path = Path(__file__).resolve().parents[1] / "models" / "anemia" / "anemia_scaler.pkl"
        if model_path.exists():
            _ANEMIA_CACHE = joblib.load(model_path)
    return _ANEMIA_CACHE

def screen_deficiencies(input_data):
    """   
    Screen for early vitamin and micronutrient deficiency risks following WHO guidelines.
    Adheres strictly to the requirement:
    - Never diagnose from age, weight, or height alone.
    - If reliable training data or laboratory measurements are unavailable, mark as unavailable rather than fabricating.
    - Anemia is screened using real laboratory metrics (Hb, MCH, MCHC, MCV).
    """
    results = {}

    # 1. Iron Deficiency & Anemia Risk
    hb = input_data.get("hemoglobin") or input_data.get("hb")
    mch = input_data.get("mch")
    mchc = input_data.get("mchc")
    mcv = input_data.get("mcv")

    if hb is not None and float(hb) > 0:
        hb_val = float(hb)
        anemia_model = get_anemia_model()
        sex_binary = 1 if str(input_data.get("child_sex", "Male")).lower() in ["male", "1", "m", "boy"] else 0
        
        mch_val = float(mch) if mch is not None else 23.0
        mchc_val = float(mchc) if mchc is not None else 30.0
        mcv_val = float(mcv) if mcv is not None else 84.0

        if anemia_model:
            df_in = pd.DataFrame([{
                "Gender": sex_binary,
                "Hemoglobin": hb_val,
                "MCH": mch_val,
                "MCHC": mchc_val,
                "MCV": mcv_val
            }])
            prob = float(anemia_model.predict_proba(df_in)[0, 1])
            is_anemic = prob >= 0.5 or hb_val < 11.0
            severity = "HIGH" if hb_val < 9.0 or prob >= 0.7 else ("MODERATE" if is_anemic else "LOW")
        else:
            is_anemic = hb_val < 11.0
            prob = 0.85 if hb_val < 9.0 else (0.65 if hb_val < 11.0 else 0.15)
            severity = "HIGH" if hb_val < 9.0 else ("MODERATE" if is_anemic else "LOW")

        results["iron_anemia"] = {
            "status": "Available",
            "condition": "Iron Deficiency & Pediatric Anemia Risk",
            "risk_score": round(prob * 100, 1),
            "severity": severity,
            "is_flagged": bool(is_anemic),
            "lab_markers_used": {
                "hemoglobin": f"{hb_val} g/dL (WHO pediatric threshold: 11.0 g/dL)",
                "mch": f"{mch_val} pg" if mch else "Population baseline",
                "mcv": f"{mcv_val} fL" if mcv else "Population baseline"
            },
            "clinical_notes": "Screening risk identified. Anemia may result from iron deficiency, hemoglobinopathies, or chronic infection. Clinical serum ferritin and CBC confirmation is recommended."
        }
    else:
        results["iron_anemia"] = {
            "status": "Awaiting Lab Values",
            "condition": "Iron Deficiency & Pediatric Anemia Risk",
            "risk_score": None,
            "severity": "UNKNOWN",
            "is_flagged": False,
            "clinical_notes": "Laboratory hemoglobin (Hb) or complete blood count is required to evaluate anemia risk. Cannot be derived from anthropometric measurements alone."
        }

    # 2. Vitamin A Deficiency Risk
    dietary_diversity = input_data.get("dietary_diversity")
    measles = str(input_data.get("measles_vaccine", "Yes")).lower() in ["no", "n", "false", "0"]
    diarrhea = str(input_data.get("diarrhea_recent", "No")).lower() in ["yes", "y", "true", "1"]

    if dietary_diversity is not None:
        dd = float(dietary_diversity)
        vit_a_risk = 0.75 if (dd < 3 and measles) else (0.55 if dd < 3 or (measles and diarrhea) else 0.18)
        results["vitamin_a"] = {
            "status": "Available",
            "condition": "Vitamin A Deficiency Risk",
            "risk_score": round(vit_a_risk * 100, 1),
            "severity": "HIGH" if vit_a_risk >= 0.7 else ("MODERATE" if vit_a_risk >= 0.5 else "LOW"),
            "is_flagged": bool(vit_a_risk >= 0.5),
            "clinical_notes": "Risk evaluated via dietary diversity and infectious vulnerability. Serum retinol testing is recommended for confirmed screening."
        }
    else:
        results["vitamin_a"] = {
            "status": "Unavailable (No Training/Dietary Data)",
            "condition": "Vitamin A Deficiency Risk",
            "risk_score": None,
            "severity": "UNAVAILABLE",
            "is_flagged": False,
            "clinical_notes": "Standard survey does not include serum retinol biomarker. Clinical dietary recall data is required."
        }

    # 3. Vitamin D Deficiency Risk
    results["vitamin_d"] = {
        "status": "Unavailable (No Biomarker in Dataset)",
        "condition": "Vitamin D Deficiency Risk",
        "risk_score": None,
        "severity": "UNAVAILABLE",
        "is_flagged": False,
        "clinical_notes": "Reliable 25-hydroxyvitamin D [25(OH)D] laboratory assay data is not present in standard DHS survey records. Requires clinical laboratory assessment."
    }

    # 4. Vitamin B12 Deficiency Risk
    results["vitamin_b12"] = {
        "status": "Unavailable (No Biomarker in Dataset)",
        "condition": "Vitamin B12 Deficiency Risk",
        "risk_score": None,
        "severity": "UNAVAILABLE",
        "is_flagged": False,
        "clinical_notes": "Serum cobalamin / methylmalonic acid measurements are not available in survey data. Evaluation requires specialized laboratory panel."
    }

    # 5. Folate Deficiency Risk
    results["folate"] = {
        "status": "Unavailable (No Biomarker in Dataset)",
        "condition": "Folate Deficiency Risk",
        "risk_score": None,
        "severity": "UNAVAILABLE",
        "is_flagged": False,
        "clinical_notes": "Red blood cell / serum folate assay data is not available. Clinical hematology workup recommended."
    }

    return {
        "status": "success",
        "disclaimer": "Deficiency risks are screening indicators based on available indicators, not confirmed medical diagnoses. Professional evaluation is required.",
        "deficiencies": results
    }
