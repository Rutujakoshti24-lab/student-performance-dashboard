"""Student Performance page: rankings, pass rates, subject charts, at-risk students.

Needs the FastAPI backend running:  uvicorn api:app --reload
"""
import io

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

SUBJECTS = ["Math", "Science", "English", "History", "Computer"]


def render():
    api_url = st.sidebar.text_input("Student API URL", "http://127.0.0.1:8000").strip().rstrip("/")

    def call(method: str, path: str, **kwargs):
        """Call the FastAPI backend and return parsed JSON (or None on error)."""
        try:
            r = requests.request(method, f"{api_url}{path}", timeout=10, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            st.error(f"API error: {e.response.status_code} {e.response.text}")
        except requests.RequestException as e:
            st.error(f"Cannot reach API at {api_url}. Start it with: uvicorn api:app --reload\n\n{e}")
        return None

    st.header("🎓 Student Performance")
    st.caption("Uses the FastAPI backend. No API key is needed, just the server address.")

    summary = call("GET", "/analytics/summary")
    if not summary or summary.get("students", 0) == 0:
        st.info("No student data available yet.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", summary["students"])
    c2.metric("Pass rate", f"{summary['pass_rate']}%")
    c3.metric("Avg percentage", f"{summary['average_percentage']}%")
    c4.metric("Avg attendance", f"{summary['average_attendance']}%")
    c5.metric("Need support", summary["at_risk_students"])

    tab_rank, tab_subject, tab_risk, tab_manage, tab_export = st.tabs(
        ["🏆 Rankings & Pass Rates", "📊 Subject-wise", "⚠️ Needs Support", "✏️ Manage Records", "⬇️ Download"]
    )

    # ---------- Rankings & pass rates ----------
    with tab_rank:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Student rankings")
            cls = st.selectbox("Class", ["All", "10A", "10B", "10C"], key="rank_class")
            top = st.slider("Show top N", 5, 50, 10)
            params = {"top": top}
            if cls != "All":
                params["student_class"] = cls
            ranks = call("GET", "/analytics/rankings", params=params)
            if ranks:
                rdf = pd.DataFrame(ranks)
                st.dataframe(rdf, hide_index=True)
                fig = px.bar(rdf, x="name", y="percentage", color="percentage",
                             color_continuous_scale="Blues", title="Top students by percentage")
                st.plotly_chart(fig)
        with right:
            st.subheader("Pass rates")
            pr = call("GET", "/analytics/pass-rate")
            if pr:
                st.metric("Overall pass rate", f"{pr['overall']}%")
                fig = px.bar(x=list(pr["by_class"]), y=list(pr["by_class"].values()),
                             labels={"x": "Class", "y": "Pass rate %"}, title="Pass rate by class")
                st.plotly_chart(fig)

    # ---------- Subject-wise ----------
    with tab_subject:
        stats = call("GET", "/analytics/subject-stats")
        pr = call("GET", "/analytics/pass-rate")
        if stats and pr:
            a, b = st.columns(2)
            with a:
                fig = px.bar(x=list(stats), y=[v["average"] for v in stats.values()],
                             labels={"x": "Subject", "y": "Average marks"}, title="Average marks per subject")
                st.plotly_chart(fig)
            with b:
                fig = px.bar(x=list(pr["by_subject"]), y=list(pr["by_subject"].values()),
                             labels={"x": "Subject", "y": "Pass rate %"}, title="Pass rate per subject")
                st.plotly_chart(fig)
            st.subheader("Highest / lowest marks")
            st.dataframe(pd.DataFrame(stats).T)

    # ---------- At-risk ----------
    with tab_risk:
        st.subheader("Students who may need academic support")
        st.caption(
            "Risk score (0-100) = 50% weak overall marks + 30% low attendance + 20% failed subjects. "
            "High ≥ 30, Medium 15-30."
        )
        level = st.radio("Show", ["Medium", "High"], horizontal=True,
                         format_func=lambda x: "Medium + High" if x == "Medium" else "High only")
        risk = call("GET", "/analytics/at-risk", params={"level": level})
        if risk:
            st.dataframe(pd.DataFrame(risk), hide_index=True)
        elif risk == []:
            st.success("No students in this risk range.")

    # ---------- Manage records ----------
    with tab_manage:
        st.subheader("Add a student")
        with st.form("add_student", clear_on_submit=True):
            name = st.text_input("Name")
            s_class = st.selectbox("Class", ["10A", "10B", "10C"])
            att = st.number_input("Attendance %", 0.0, 100.0, 80.0)
            cols = st.columns(len(SUBJECTS))
            marks = {s: cols[i].number_input(s, 0.0, 100.0, 50.0) for i, s in enumerate(SUBJECTS)}
            if st.form_submit_button("Add student") and name.strip():
                res = call("POST", "/students", json={
                    "name": name, "student_class": s_class, "attendance_pct": att, "marks": marks})
                if res:
                    st.success(f"Added student #{res['student_id']}")

        st.divider()
        st.subheader("Update marks / attendance")
        students = call("GET", "/students") or []
        if students:
            options = {f"{s['student_id']} - {s['name']}": s for s in students}
            chosen = options[st.selectbox("Student", list(options))]
            sid = chosen["student_id"]
            with st.form("update_student"):
                new_att = st.number_input("Attendance %", 0.0, 100.0, float(chosen["attendance_pct"]))
                cols = st.columns(len(SUBJECTS))
                new_marks = {s: cols[i].number_input(s, 0.0, 100.0, float(chosen[s]), key=f"m_{sid}_{s}")
                             for i, s in enumerate(SUBJECTS)}
                if st.form_submit_button("Save changes"):
                    call("PUT", f"/students/{sid}/marks", json=new_marks)
                    call("PUT", f"/students/{sid}/attendance", json={"attendance_pct": new_att})
                    st.success("Updated. Refresh the page to see new charts.")
            if st.button("🗑️ Delete this student"):
                if call("DELETE", f"/students/{sid}"):
                    st.warning("Deleted. Refresh the page.")

    # ---------- Download ----------
    with tab_export:
        st.subheader("Download cleaned student dataset")
        st.write("Trimmed names, duplicates removed, missing values filled, marks clipped to 0-100.")
        try:
            csv_bytes = requests.get(f"{api_url}/export/clean-csv", timeout=10).content
            st.download_button("⬇️ Download students_clean.csv", csv_bytes,
                               file_name="students_clean.csv", mime="text/csv")
            st.dataframe(pd.read_csv(io.BytesIO(csv_bytes)).head(20))
        except requests.RequestException as e:
            st.error(f"Could not fetch CSV: {e}")