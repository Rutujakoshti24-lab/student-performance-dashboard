"""Shared data-cleaning and analytics helpers used by api.py and generate_data.py."""
import numpy as np
import pandas as pd

SUBJECTS = ["Math", "Science", "English", "History", "Computer"]
PASS_MARK = 40            # minimum marks required in each subject
ATTENDANCE_TARGET = 85    # % attendance considered healthy
COLUMNS = ["student_id", "name", "student_class", "attendance_pct"] + SUBJECTS


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy of a raw student dataset.

    - trims column names / text, normalises name and class formatting
    - drops rows without id or name, and duplicate student ids
    - converts marks / attendance to numbers, clips them to 0-100
    - fills missing numeric values with the column median
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["student_id", "name"])
    df["student_id"] = pd.to_numeric(df["student_id"], errors="coerce")
    df = df.dropna(subset=["student_id"])
    df["student_id"] = df["student_id"].astype(int)
    df = df.drop_duplicates(subset=["student_id"], keep="first")

    df["name"] = df["name"].astype(str).str.strip().str.title()
    df["student_class"] = df["student_class"].astype(str).str.strip().str.upper()

    for col in SUBJECTS + ["attendance_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 100)
        df[col] = df[col].fillna(df[col].median()).round(1)

    return df[COLUMNS].sort_values("student_id").reset_index(drop=True)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add total, percentage, result, rank and academic-support risk columns."""
    df = df.copy()
    if df.empty:
        for col in ["total", "percentage", "failed_subjects", "result", "rank",
                    "risk_score", "risk_level"]:
            df[col] = []
        return df

    df["total"] = df[SUBJECTS].sum(axis=1).round(1)
    df["percentage"] = (df["total"] / (100 * len(SUBJECTS)) * 100).round(2)
    df["failed_subjects"] = (df[SUBJECTS] < PASS_MARK).sum(axis=1)
    df["result"] = np.where(df["failed_subjects"] == 0, "Pass", "Fail")
    df["rank"] = df["percentage"].rank(method="min", ascending=False).astype(int)

    # Risk score (0-100): weak marks 50%, low attendance 30%, failed subjects 20%
    marks_risk = ((60 - df["percentage"]) / 60).clip(0, 1)
    att_risk = ((ATTENDANCE_TARGET - df["attendance_pct"]) / ATTENDANCE_TARGET).clip(0, 1)
    fail_risk = df["failed_subjects"] / len(SUBJECTS)
    df["risk_score"] = (100 * (0.5 * marks_risk + 0.3 * att_risk + 0.2 * fail_risk)).round(1)
    df["risk_level"] = pd.cut(
        df["risk_score"], bins=[-1, 15, 30, 101], labels=["Low", "Medium", "High"]
    ).astype(str)
    return df