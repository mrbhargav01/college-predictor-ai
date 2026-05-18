"""
src/model.py
Machine-learning layer: scoring engine + optional Random Forest classifier.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "rf_model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "models", "encoders.pkl")


# ── Scoring Engine (always available, no sklearn needed) ──────────────────────

def score_colleges(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    """
    Weighted multi-factor scoring to rank colleges for a student.
    Returns DataFrame sorted by match_score descending.
    """
    out = df.copy()

    # --- Primary filter chain ---
    if user.get("exam_type") and user["exam_type"] not in ("All", "Other"):
        out = out[out["Entrance_Exam"] == user["exam_type"]]

    if user.get("course_preference") and user["course_preference"] != "All":
        out = out[out["Course_Preference"] == user["course_preference"]]

    if user.get("category") and user["category"] not in ("All",):
        out = out[out["Category"] == user["category"]]

    if user.get("preferred_state") and user["preferred_state"] != "All":
        out = out[out["State"] == user["preferred_state"]]

    if user.get("preferred_city") and user["preferred_city"] != "All":
        out = out[out["City"] == user["preferred_city"]]

    if len(out) == 0:
        # Relax state/city filters if nothing found
        out = df.copy()
        if user.get("exam_type") and user["exam_type"] not in ("All", "Other"):
            out = out[out["Entrance_Exam"] == user["exam_type"]]
        if user.get("course_preference") and user["course_preference"] != "All":
            out = out[out["Course_Preference"] == user["course_preference"]]

    # --- Score calculation ---
    out = out.copy()
    out["match_score"] = 0.0

    # Factor 1 – Historical probability (weight 40%)
    out["match_score"] += out["Probability_of_Admission"] * 40

    # Factor 2 – Exam score similarity (weight 30%)
    user_score = float(user.get("exam_score", 0))
    if user_score > 0:
        max_s = out["Entrance_Exam_Score"].max()
        if max_s > 0:
            diff = (out["Entrance_Exam_Score"] - user_score).abs()
            out["match_score"] += (1 - diff / max_s) * 30

    # Factor 3 – Marks similarity (weight 20%)
    user_pct = float(user.get("overall_percentage", user.get("percentage", 0)))
    if user_pct > 0:
        diff_m = (out["Student_Marks_Percentage"] - user_pct).abs()
        max_m = diff_m.max()
        if max_m > 0:
            out["match_score"] += (1 - diff_m / max_m) * 20

    # Factor 4 – Cutoff gap (weight 10%)
    if user_pct > 0:
        out["_cutoff_gap"] = user_pct - out["Historical_College_Cutoff_Marks"]
        out["match_score"] += out["_cutoff_gap"].clip(0, 20) / 20 * 10
        out.drop(columns=["_cutoff_gap"], inplace=True)

    # Sort and deduplicate by college
    out = out.sort_values("match_score", ascending=False)
    out = out.drop_duplicates(subset=["College_Name"])

    return out.reset_index(drop=True)


def apply_budget_filter(df: pd.DataFrame, budget: str) -> pd.DataFrame:
    """Filter by budget range using synthetic Annual_Fee_Lakhs."""
    mapping = {
        "< 1 Lakh":    (0, 1.0),
        "1–3 Lakhs":   (1.0, 3.0),
        "3–5 Lakhs":   (3.0, 5.0),
        "> 5 Lakhs":   (5.0, 999),
        "Any":         (0, 999),
    }
    lo, hi = mapping.get(budget, (0, 999))
    if "Annual_Fee_Lakhs" in df.columns:
        return df[(df["Annual_Fee_Lakhs"] >= lo) & (df["Annual_Fee_Lakhs"] <= hi)]
    return df


# ── Random Forest Classifier (optional) ──────────────────────────────────────

def train_rf_model(df: pd.DataFrame) -> Optional[dict]:
    if not SKLEARN_OK:
        return None

    cat_cols = ["Entrance_Exam", "Course_Preference", "State", "Category"]
    num_cols = ["Student_Marks_Percentage", "Entrance_Exam_Score",
                "Historical_College_Cutoff_Marks"]
    target   = "College_Classification"

    sub = df[cat_cols + num_cols + [target]].dropna()
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        sub[c] = le.fit_transform(sub[c].astype(str))
        encoders[c] = le

    le_y = LabelEncoder()
    y = le_y.fit_transform(sub[target])
    encoders["target"] = le_y

    X = sub[cat_cols + num_cols].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=150, max_depth=10,
                                  random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)

    return {"accuracy": acc, "model": clf, "encoders": encoders,
            "report": classification_report(y_te, clf.predict(X_te),
                                             target_names=le_y.classes_)}


def load_rf_model() -> Optional[tuple]:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH)):
        return None
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        enc = pickle.load(f)
    return clf, enc


def get_feature_importance(clf, feature_names: list[str]) -> pd.DataFrame:
    imp = clf.feature_importances_
    return pd.DataFrame({"Feature": feature_names, "Importance": imp})\
             .sort_values("Importance", ascending=False)
