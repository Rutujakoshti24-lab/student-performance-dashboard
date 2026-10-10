"""FastAPI backend: student records, marks, attendance and analytics APIs.

Run:  uvicorn api:app --reload        (docs at http://127.0.0.1:8000/docs)
"""
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cleaning import COLUMNS, PASS_MARK, SUBJECTS, clean_dataframe, enrich

DATA_FILE = Path(__file__).parent / "students_clean.csv"

app = FastAPI(title="Student Performance API", version="1.0.0")


# ---------- helpers ----------
def load() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise HTTPException(500, "students_clean.csv not found. Run: python generate_data.py")
    return pd.read_csv(DATA_FILE)


def save(df: pd.DataFrame) -> None:
    clean_dataframe(df).to_csv(DATA_FILE, index=False)


def records(df: pd.DataFrame):
    """DataFrame -> JSON-safe list of dicts."""
    return json.loads(df.to_json(orient="records"))


def find(df: pd.DataFrame, student_id: int) -> int:
    idx = df.index[df["student_id"] == student_id]
    if len(idx) == 0:
        raise HTTPException(404, f"Student {student_id} not found")
    return idx[0]


def check_marks(marks: Dict[str, float]) -> None:
    for subject, value in marks.items():
        if subject not in SUBJECTS:
            raise HTTPException(422, f"Unknown subject '{subject}'. Valid: {SUBJECTS}")
        if not 0 <= value <= 100:
            raise HTTPException(422, f"Marks for {subject} must be between 0 and 100")


# ---------- schemas ----------
class StudentIn(BaseModel):
    name: str = Field(min_length=1)
    student_class: str
    attendance_pct: float = Field(ge=0, le=100)
    marks: Dict[str, float] = Field(default_factory=dict)


class StudentUpdate(BaseModel):
    name: Optional[str] = None
    student_class: Optional[str] = None


class AttendanceUpdate(BaseModel):
    attendance_pct: float = Field(ge=0, le=100)


# ---------- student records ----------
@app.get("/students")
def list_students(student_class: Optional[str] = None, search: Optional[str] = None):
    df = enrich(load())
    if student_class:
        df = df[df["student_class"].str.upper() == student_class.upper()]
    if search:
        df = df[df["name"].str.contains(search, case=False)]
    return records(df)


@app.get("/students/{student_id}")
def get_student(student_id: int):
    df = enrich(load())
    return records(df.loc[[find(df, student_id)]])[0]


@app.post("/students", status_code=201)
def create_student(s: StudentIn):
    check_marks(s.marks)
    df = load()
    new = {c: s.marks.get(c, 0.0) for c in SUBJECTS}
    new.update(
        student_id=int(df["student_id"].max() + 1) if len(df) else 1,
        name=s.name,
        student_class=s.student_class,
        attendance_pct=s.attendance_pct,
    )
    save(pd.concat([df, pd.DataFrame([new])[COLUMNS]], ignore_index=True))
    return {"message": "Student created", "student_id": new["student_id"]}


@app.put("/students/{student_id}")
def update_student(student_id: int, u: StudentUpdate):
    df = load()
    i = find(df, student_id)
    if u.name:
        df.loc[i, "name"] = u.name
    if u.student_class:
        df.loc[i, "student_class"] = u.student_class
    save(df)
    return {"message": "Student updated"}


@app.delete("/students/{student_id}")
def delete_student(student_id: int):
    df = load()
    i = find(df, student_id)
    save(df.drop(index=i))
    return {"message": "Student deleted"}


# ---------- marks ----------
@app.get("/students/{student_id}/marks")
def get_marks(student_id: int):
    df = load()
    row = df.loc[find(df, student_id)]
    return {s: float(row[s]) for s in SUBJECTS}


@app.put("/students/{student_id}/marks")
def update_marks(student_id: int, marks: Dict[str, float]):
    check_marks(marks)
    df = load()
    i = find(df, student_id)
    for subject, value in marks.items():
        df.loc[i, subject] = value
    save(df)
    return {"message": "Marks updated"}


# ---------- attendance ----------
@app.get("/students/{student_id}/attendance")
def get_attendance(student_id: int):
    df = load()
    return {"attendance_pct": float(df.loc[find(df, student_id), "attendance_pct"])}


@app.put("/students/{student_id}/attendance")
def update_attendance(student_id: int, a: AttendanceUpdate):
    df = load()
    df.loc[find(df, student_id), "attendance_pct"] = a.attendance_pct
    save(df)
    return {"message": "Attendance updated"}


# ---------- analytics ----------
@app.get("/analytics/summary")
def summary():
    df = enrich(load())
    if df.empty:
        return {"students": 0}
    return {
        "students": len(df),
        "pass_rate": round(float((df["result"] == "Pass").mean() * 100), 1),
        "average_percentage": round(float(df["percentage"].mean()), 1),
        "average_attendance": round(float(df["attendance_pct"].mean()), 1),
        "at_risk_students": int((df["risk_level"] != "Low").sum()),
    }


@app.get("/analytics/rankings")
def rankings(top: int = Query(10, ge=1, le=500), student_class: Optional[str] = None):
    df = enrich(load())
    if student_class:
        df = df[df["student_class"].str.upper() == student_class.upper()]
    cols = ["rank", "student_id", "name", "student_class", "total", "percentage", "result"]
    return records(df.sort_values("rank").head(top)[cols])


@app.get("/analytics/pass-rate")
def pass_rate():
    """Overall, per-class and per-subject pass rates (pass = marks >= PASS_MARK)."""
    df = enrich(load())
    per_class = df.groupby("student_class")["result"].apply(lambda r: round((r == "Pass").mean() * 100, 1))
    per_subject = {s: round(float((df[s] >= PASS_MARK).mean() * 100), 1) for s in SUBJECTS}
    return {
        "overall": round(float((df["result"] == "Pass").mean() * 100), 1),
        "by_class": per_class.to_dict(),
        "by_subject": per_subject,
    }


@app.get("/analytics/subject-stats")
def subject_stats():
    df = load()
    return {
        s: {
            "average": round(float(df[s].mean()), 1),
            "highest": float(df[s].max()),
            "lowest": float(df[s].min()),
        }
        for s in SUBJECTS
    }


@app.get("/analytics/at-risk")
def at_risk(level: str = Query("Medium", pattern="^(Medium|High)$")):
    """Students who may need academic support. level=Medium returns Medium+High, level=High only High."""
    df = enrich(load())
    wanted = ["High"] if level == "High" else ["Medium", "High"]
    out = df[df["risk_level"].isin(wanted)].sort_values("risk_score", ascending=False)
    cols = ["student_id", "name", "student_class", "percentage",
            "attendance_pct", "failed_subjects", "risk_score", "risk_level"]
    return records(out[cols])


# ---------- export ----------
@app.get("/export/clean-csv")
def export_clean_csv():
    """Download the cleaned dataset as a CSV file."""
    clean_dataframe(load()).to_csv(DATA_FILE, index=False)
    return FileResponse(DATA_FILE, media_type="text/csv", filename="students_clean.csv")