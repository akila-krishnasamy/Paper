import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# The supported nutrition model outputs are separate conditions. Historical
# model checkpoints may contain an additional composite output; it is not served.
TARGET_CONDITIONS = ["Stunting", "Wasting"]

# Numerical Features
NUMERICAL_FEATURES = [
    "child_age_months",
    "birth_weight",
    "breastfeeding_duration",
    "birth_order",
    "mother_bmi",
    "anc_visits",
    "hhsize",
    "sanitation_risk_index",
    "maternal_risk_score"
]

# Categorical Features
CATEGORICAL_FEATURES = [
    "child_sex",
    "education",
    "wealth_quintile",
    "residence",
    "birth_size",
    "diarrhea_recent",
    "fever_recent",
    "cough_recent",
    "measles_vaccine"
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def normalize_choice(value, valid_values, default=None):
    """Normalize mixed-case / numeric category inputs to the canonical training labels."""
    if value is None:
        return default

    text = str(value).strip()
    lowered = text.casefold()
    for valid in valid_values:
        if lowered == str(valid).casefold():
            return valid
    return default


def get_project_root():
    """Resolve project root directory dynamically."""
    return Path(__file__).resolve().parent.parent.parent


def find_dataset_path(filename="dhs_clean.parquet"):
    """Locate DHS dataset file across likely project directories."""
    candidates = [
        Path(filename),
        get_project_root() / "data" / "processed" / filename,
        get_project_root() / filename,
        Path(r"e:\SEM-7\paper_\data\processed") / filename,
        Path(r"e:\SEM-7\paper_") / filename,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"Could not find dataset file '{filename}' in expected locations.")


def load_and_preprocess_dhs_data(data_path="dhs_clean.parquet"):
    """
    Load dhs_clean.parquet (198,849 records from NFHS-5), clean missing values,
    engineer clinical and environmental features, and encode target conditions.
    """
    if not os.path.isabs(data_path) or not os.path.exists(data_path):
        data_path = find_dataset_path(os.path.basename(data_path))

    df = pd.read_parquet(data_path)

    # Standardize Categorical Mappings
    sex_map = {1: "Male", 2: "Female"}
    edu_map = {0: "No Education", 1: "Primary", 2: "Secondary", 3: "Higher"}
    wealth_map = {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}
    res_map = {1: "Urban", 2: "Rural"}
    size_map = {1: "Very Large", 2: "Larger than Average", 3: "Average", 4: "Smaller than Average", 5: "Very Small"}

    df["child_sex"] = df["child_sex"].map(sex_map).fillna("Male")
    df["education"] = df["education"].map(edu_map).fillna("Secondary")
    df["wealth_quintile"] = df["wealth_quintile"].map(wealth_map).fillna("Middle")
    df["residence"] = df["residence"].map(res_map).fillna("Rural")
    df["birth_size"] = df["birth_size"].map(size_map).fillna("Average")

    df["diarrhea_recent"] = np.where(df["diarrhea_recent"] == 2.0, "Yes", "No")
    df["fever_recent"] = np.where(df["fever_recent"] == 1.0, "Yes", "No")
    df["cough_recent"] = np.where(df["cough_recent"] == 2.0, "Yes", "No")
    df["measles_vaccine"] = np.where(df["measles_vaccine"] == 0.0, "No", "Yes")

    # Environmental / Sanitation Risk Index (0 to 3)
    unsafe_water = (~df["water_source"].isin([11.0, 12.0, 21.0, 14.0, 13.0])).astype(float)
    unsafe_toilet = (df["toilet_type"].isin([31.0, 97.0])).astype(float)
    smoke_fuel = (df["cooking_fuel"].isin([2.0, 8.0, 11.0, 6.0, 7.0, 9.0, 10.0])).astype(float)
    df["sanitation_risk_index"] = unsafe_water + unsafe_toilet + smoke_fuel

    # Handle missing values with clinical medians
    df["mother_bmi"] = df["mother_bmi"].fillna(21.27)
    df["birth_weight"] = df["birth_weight"].fillna(2900.0)
    df["anc_visits"] = df["anc_visits"].fillna(4.0)
    df["breastfeeding_duration"] = df["breastfeeding_duration"].fillna(15.0)
    df["hhsize"] = df["hhsize"].fillna(6.0)
    df["birth_order"] = df["birth_order"].fillna(2.0)
    df["child_age_months"] = df["child_age_months"].fillna(24.0)

    # Maternal Risk Score (0 to 3)
    mother_underweight = (df["mother_bmi"] < 18.5).astype(float)
    low_anc = (df["anc_visits"] < 4.0).astype(float)
    low_bw = (df["birth_weight"] < 2500.0).astype(float)
    df["maternal_risk_score"] = mother_underweight + low_anc + low_bw

    # Targets
    col_stunting = "stunting" if "stunting" in df.columns else "Stunting"
    col_wasting = "wasting" if "wasting" in df.columns else "Wasting"
    df["Stunting"] = df[col_stunting].astype("Int64")
    df["Wasting"] = df[col_wasting].astype("Int64")
    return df


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.cat_encoders = {cat: LabelEncoder() for cat in CATEGORICAL_FEATURES}
        self.is_fitted = False

    def fit_transform(self, df):
        """Fit preprocessor strictly on training dataset and transform features."""
        df_proc = df.copy()
        df_proc[NUMERICAL_FEATURES] = self.scaler.fit_transform(df_proc[NUMERICAL_FEATURES])

        for cat in CATEGORICAL_FEATURES:
            df_proc[cat] = self.cat_encoders[cat].fit_transform(df_proc[cat].astype(str))

        self.is_fitted = True
        return df_proc

    def transform(self, df):
        """Transform validation, test, or inference input DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform()")

        df_proc = df.copy()
        df_proc[NUMERICAL_FEATURES] = self.scaler.transform(df_proc[NUMERICAL_FEATURES])

        for cat in CATEGORICAL_FEATURES:
            encoder = self.cat_encoders[cat]
            vals = df_proc[cat].astype(str).values
            known_classes = set(encoder.classes_)
            vals_clean = [v if v in known_classes else encoder.classes_[0] for v in vals]
            df_proc[cat] = encoder.transform(vals_clean)

        return df_proc

    def transform_single_input(self, input_dict):
        """Transform a single JSON input dictionary for real-time model inference."""
        data = {}

        # Child Age
        age = input_dict.get("child_age_months", input_dict.get("age_months", 24))
        data["child_age_months"] = float(age)

        # Birth Weight (if in kg, e.g. 2.8, convert to grams 2800)
        bw = input_dict.get("birth_weight", 2900.0)
        if bw is not None and float(bw) < 15.0:
            bw = float(bw) * 1000.0
        data["birth_weight"] = float(bw) if bw else 2900.0

        # Breastfeeding duration
        bf_dur = input_dict.get("breastfeeding_duration", input_dict.get("breastfeeding_months", 15.0))
        data["breastfeeding_duration"] = float(bf_dur)

        # Birth order & HH Size
        data["birth_order"] = float(input_dict.get("birth_order", 2.0))
        data["hhsize"] = float(input_dict.get("hhsize", input_dict.get("household_size", 6.0)))

        # Mother's BMI
        mbmi = input_dict.get("mother_bmi")
        if mbmi is None or pd.isna(mbmi):
            m_wt = input_dict.get("mother_weight")
            m_ht = input_dict.get("mother_height")
            if m_wt and m_ht:
                mbmi = float(m_wt) / ((float(m_ht) / 100) ** 2)
            else:
                mbmi = 21.3
        data["mother_bmi"] = float(mbmi)

        # ANC visits
        data["anc_visits"] = float(input_dict.get("anc_visits", 4.0))

        # Sanitation Risk Index (0 to 3)
        san = input_dict.get("sanitation_risk_index")
        if san is None:
            # Check if water_sanitation_index (1 to 5) was passed
            wsi = input_dict.get("water_sanitation_index", 3)
            san = max(0, min(3, 5 - int(wsi)))
        data["sanitation_risk_index"] = float(san)

        # Maternal Risk Score (0 to 3)
        m_under = 1.0 if data["mother_bmi"] < 18.5 else 0.0
        low_anc = 1.0 if data["anc_visits"] < 4.0 else 0.0
        low_bw = 1.0 if data["birth_weight"] < 2500.0 else 0.0
        data["maternal_risk_score"] = m_under + low_anc + low_bw

        # Categoricals
        # Child Sex
        c_sex = input_dict.get("child_sex", input_dict.get("gender", "Male"))
        data["child_sex"] = "Female" if str(c_sex).lower() in ["female", "f", "girl", "2"] else "Male"

        # Education
        valid_edus = ["No Education", "Primary", "Secondary", "Higher"]
        data["education"] = normalize_choice(input_dict.get("education", "Secondary"), valid_edus, "Secondary")

        # Wealth Quintile
        valid_wq = ["Poorest", "Poorer", "Middle", "Richer", "Richest"]
        data["wealth_quintile"] = normalize_choice(input_dict.get("wealth_quintile", "Middle"), valid_wq, "Middle")

        # Residence
        res = input_dict.get("residence", "Rural")
        res_text = str(res).strip()
        data["residence"] = "Urban" if res_text.casefold() in ["urban", "1", "city"] else "Rural"

        # Birth Size
        valid_bs = ["Very Large", "Larger than Average", "Average", "Smaller than Average", "Very Small"]
        data["birth_size"] = normalize_choice(input_dict.get("birth_size", "Average"), valid_bs, "Average")

        # Morbidity symptoms
        def parse_yes_no(val, default="No"):
            if val is None:
                return default
            v = str(val).lower()
            return "Yes" if v in ["yes", "y", "true", "1", "2"] else "No"

        data["diarrhea_recent"] = parse_yes_no(input_dict.get("diarrhea_recent", "No"))
        data["fever_recent"] = parse_yes_no(input_dict.get("fever_recent", "No"))
        data["cough_recent"] = parse_yes_no(input_dict.get("cough_recent", "No"))
        data["measles_vaccine"] = parse_yes_no(input_dict.get("measles_vaccine", "Yes"), default="Yes")

        df_single = pd.DataFrame([data])
        df_single = df_single[ALL_FEATURES]
        return self.transform(df_single)


def prepare_data_splits(data_path="dhs_clean.parquet", test_size=0.15, val_size=0.15, random_state=42):
    """
    Split DHS dataset into 70% Train, 15% Validation, and 15% Test.
    Fit Preprocessor strictly on Train and apply to Val & Test.
    """
    print(f"Loading DHS dataset: {data_path}...")
    df = load_and_preprocess_dhs_data(data_path)
    print(f"Loaded {len(df):,} records with {len(ALL_FEATURES)} predictors and {len(TARGET_CONDITIONS)} targets.")

    df = df.dropna(subset=TARGET_CONDITIONS).copy()
    X = df[ALL_FEATURES]
    Y = df[TARGET_CONDITIONS].astype(int)
    group_columns = [column for column in ("v001", "v002") if column in df]
    if group_columns:
        groups = (
            df[group_columns].astype("string").fillna("<missing>")
            .agg("|".join, axis=1).to_numpy()
        )
        indices = np.arange(len(df))
        train_idx, remainder_idx = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=test_size + val_size,
                random_state=random_state,
            ).split(indices, Y["Stunting"], groups)
        )
        val_relative, test_relative = next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=test_size / (test_size + val_size),
                random_state=random_state,
            ).split(
                remainder_idx,
                Y.iloc[remainder_idx]["Stunting"],
                groups[remainder_idx],
            )
        )
        val_idx = remainder_idx[val_relative]
        test_idx = remainder_idx[test_relative]
        X_train, X_val, X_test = (
            X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
        )
        Y_train, Y_val, Y_test = (
            Y.iloc[train_idx], Y.iloc[val_idx], Y.iloc[test_idx]
        )
    else:
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=(test_size + val_size), random_state=random_state,
            stratify=Y["Stunting"]
        )
        val_ratio_in_temp = val_size / (test_size + val_size)
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=(1.0 - val_ratio_in_temp),
            random_state=random_state, stratify=Y_temp["Stunting"]
        )

    print(f"Train split: {len(X_train):,} | Validation split: {len(X_val):,} | Test split: {len(X_test):,}")

    # Fit preprocessor strictly on Train split
    preprocessor = DataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Save fitted preprocessor
    models_dir = get_project_root() / "ml-service" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    prep_path = models_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, prep_path)
    print(f"Saved fitted preprocessor to: {prep_path}")

    return (X_train_proc, Y_train), (X_val_proc, Y_val), (X_test_proc, Y_test), preprocessor


if __name__ == "__main__":
    (X_tr, Y_tr), (X_v, Y_v), (X_te, Y_te), prep = prepare_data_splits()
    print("Done preprocessing DHS dataset!")
