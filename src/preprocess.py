"""
src/preprocess.py
Data loading, cleaning, and feature engineering for EduPath.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
COLLEGE_CSV = os.path.join(DATA_DIR, "college_data.csv")
ML_CSV = os.path.join(DATA_DIR, "ml_ready_data.csv")


@st.cache_data(show_spinner=False)
def load_college_data() -> pd.DataFrame:
    """Load and clean the main college dataset."""
    df = pd.read_csv(COLLEGE_CSV)

    # Extract city from college name  "XYZ College, CityName"
    df["City"] = df["College_Name"].str.extract(r",\s*(.+)$")[0].str.strip()

    # Normalise classification casing
    df["College_Classification"] = df["College_Classification"].str.strip().str.title()

    # Ensure numeric columns
    for col in ["Student_Marks_Percentage", "Entrance_Exam_Score",
                "Historical_College_Cutoff_Marks",
                "Previous_Year_Admission_Rank", "Probability_of_Admission"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived features
    df["Cutoff_Gap"] = df["Student_Marks_Percentage"] - df["Historical_College_Cutoff_Marks"]
    df["Admission_Chance_Pct"] = (df["Probability_of_Admission"] * 100).round(1)

    # Synthetic extras (for richer UI display)
    rng = np.random.default_rng(42)
    n = len(df)
    df["NIRF_Ranking"]       = rng.integers(1, 300, size=n)
    df["Avg_Package_LPA"]    = rng.uniform(4.5, 18.0, size=n).round(2)
    df["Highest_Package_LPA"]= (df["Avg_Package_LPA"] + rng.uniform(5, 20, size=n)).round(2)
    df["Placement_Rate_Pct"] = rng.integers(55, 98, size=n)
    df["Annual_Fee_Lakhs"]   = rng.uniform(0.8, 6.5, size=n).round(2)
    df["Hostel_Fee_Lakhs"]   = rng.uniform(0.4, 1.2, size=n).round(2)
    df["Total_Seats"]        = rng.integers(30, 240, size=n)
    df["College_Type"]       = rng.choice(["Government", "Private", "Deemed"], size=n,
                                           p=[0.25, 0.60, 0.15])
    df["Rating"]             = rng.uniform(3.0, 5.0, size=n).round(1)

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_ml_data() -> pd.DataFrame:
    return pd.read_csv(ML_CSV)


def get_unique_values(df: pd.DataFrame) -> dict:
    return {
        "states":   sorted(df["State"].dropna().unique().tolist()),
        "cities":   sorted(df["City"].dropna().unique().tolist()),
        "exams":    sorted(df["Entrance_Exam"].dropna().unique().tolist()),
        "courses":  sorted(df["Course_Preference"].dropna().unique().tolist()),
        "categories": sorted(df["Category"].dropna().unique().tolist()),
        "classifications": ["Safe", "Moderate", "Dream"],
    }


def get_cities_for_state(df: pd.DataFrame, state: str) -> list[str]:
    if state == "All":
        return sorted(df["City"].dropna().unique().tolist())
    return sorted(df[df["State"] == state]["City"].dropna().unique().tolist())


def aggregate_stats(df: pd.DataFrame) -> dict:
    return {
        "total_records":  len(df),
        "total_colleges": df["College_Name"].nunique(),
        "total_states":   df["State"].nunique(),
        "total_cities":   df["City"].nunique(),
        "avg_probability": df["Probability_of_Admission"].mean(),
        "avg_cutoff":     df["Historical_College_Cutoff_Marks"].mean(),
        "safe_count":     int((df["College_Classification"] == "Safe").sum()),
        "moderate_count": int((df["College_Classification"] == "Moderate").sum()),
        "dream_count":    int((df["College_Classification"] == "Dream").sum()),
    }
