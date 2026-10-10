"""Data Explorer + Student Performance Dashboard (Streamlit).

Run:  streamlit run app.py
Pages: Upload Data, Dashboard, Data Cleaning, Student Performance (needs the FastAPI backend).
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import eda
import student_pages

st.set_page_config(page_title="Data & Student Performance Dashboard", page_icon="🎓", layout="wide")

SAMPLE_FILE = Path(__file__).parent / "students_clean.csv"

for key, default in {"raw_df": None, "df": None, "filename": None, "file_sig": None, "log": []}.items():
    st.session_state.setdefault(key, default)


# ============================================================ helpers
def read_upload(f) -> pd.DataFrame:
    if f.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    try:
        return pd.read_csv(f)
    except UnicodeDecodeError:
        f.seek(0)
        return pd.read_csv(f, encoding="latin-1")


def set_dataset(df: pd.DataFrame, name: str, sig=None):
    st.session_state.raw_df = df.copy()
    st.session_state.df = df.copy()
    st.session_state.filename = name
    st.session_state.file_sig = sig
    st.session_state.log = []


def need_data() -> bool:
    if st.session_state.df is None:
        st.info("👈 Go to **Upload Data** first and load a CSV/Excel file (or the sample dataset).")
        return True
    return False


def show_summary(df: pd.DataFrame):
    """Metrics shown right after upload."""
    num, cat, dt = eda.column_groups(df)
    ov = eda.overview(df)
    c = st.columns(6)
    c[0].metric("Rows", f"{ov['rows']:,}")
    c[1].metric("Columns", ov["columns"])
    c[2].metric("Numerical columns", ov["numerical"])
    c[3].metric("Categorical columns", ov["categorical"])
    c[4].metric("Missing values", f"{ov['missing']:,}")
    c[5].metric("Duplicate rows", f"{ov['duplicates']:,}")
    a, b = st.columns(2)
    a.markdown("**Numerical columns:** " + (", ".join(num) if num else "none"))
    b.markdown("**Categorical columns:** " + (", ".join(cat) if cat else "none"))
    if dt:
        st.markdown("**Datetime columns:** " + ", ".join(dt))


# ============================================================ page: upload
def page_upload():
    st.header("📁 Upload Dataset")
    up = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if up is not None:
        sig = (up.name, up.size)
        if st.session_state.file_sig != sig:
            try:
                set_dataset(read_upload(up), up.name, sig)
            except Exception as e:  # noqa: BLE001 - show any read error to the user
                st.error(f"Could not read the file: {e}")

    if SAMPLE_FILE.exists() and st.button("Or load the sample dataset (students_clean.csv)"):
        set_dataset(pd.read_csv(SAMPLE_FILE), "students_clean.csv (sample)")

    df = st.session_state.df
    if df is None:
        st.info("Upload a file to see its overview.")
        return

    st.success(f"Loaded **{st.session_state.filename}**")
    show_summary(df)
    st.subheader("Preview")
    st.dataframe(df.head(20))
    st.caption("Open **Dashboard** for the full analysis or **Data Cleaning** to fix issues.")


# ============================================================ page: dashboard
def page_dashboard():
    st.header("📊 Dashboard")
    if need_data():
        return
    df = st.session_state.df
    st.caption(f"Analysing: **{st.session_state.filename}** (current version, including any cleaning applied)")
    num, cat, dt = eda.column_groups(df)

    tabs = st.tabs([
        "Overview", "Preview", "Data Types", "Missing Values", "Duplicates", "Statistical Summary",
        "Numerical", "Categorical", "Correlation", "Outliers", "Visualizations", "Key Insights",
    ])

    # ---- Overview
    with tabs[0]:
        show_summary(df)
        ov = eda.overview(df)
        left, right = st.columns(2)
        with left:
            kinds = pd.DataFrame({"type": ["Numerical", "Categorical", "Datetime"],
                                  "columns": [len(num), len(cat), len(dt)]})
            kinds = kinds[kinds["columns"] > 0]
            st.plotly_chart(px.pie(kinds, names="type", values="columns", hole=0.45,
                                   title="Column types"))
        with right:
            cells = df.shape[0] * df.shape[1]
            quality = pd.DataFrame({"part": ["Filled cells", "Missing cells"],
                                    "cells": [cells - ov["missing"], ov["missing"]]})
            st.plotly_chart(px.pie(quality, names="part", values="cells", hole=0.45,
                                   title="Data completeness",
                                   color="part", color_discrete_map={
                                       "Filled cells": "#2E86AB", "Missing cells": "#E4572E"}))
        st.caption(f"Memory usage: {ov['memory_kb']} KB")

    # ---- Preview
    with tabs[1]:
        n = st.slider("Rows to show", 5, min(500, max(5, len(df))), min(20, max(5, len(df))))
        where = st.radio("Show", ["Head", "Tail", "Random sample"], horizontal=True)
        if where == "Head":
            st.dataframe(df.head(n))
        elif where == "Tail":
            st.dataframe(df.tail(n))
        else:
            st.dataframe(df.sample(min(n, len(df)), random_state=1))

    # ---- Data types
    with tabs[2]:
        dtypes = eda.dtype_table(df)
        left, right = st.columns([3, 2])
        left.dataframe(dtypes, hide_index=True)
        counts = dtypes["dtype"].value_counts().reset_index()
        counts.columns = ["dtype", "columns"]
        right.plotly_chart(px.bar(counts, x="dtype", y="columns", color="dtype", title="Columns per data type"))

    # ---- Missing values
    with tabs[3]:
        miss = eda.missing_table(df)
        total = int(miss["missing"].sum())
        st.metric("Total missing values", f"{total:,}")
        if total == 0:
            st.success("No missing values 🎉")
        else:
            left, right = st.columns([2, 3])
            left.dataframe(miss[miss["missing"] > 0], hide_index=True)
            shown = miss[miss["missing"] > 0]
            right.plotly_chart(px.bar(shown, x="column", y="missing_pct", color="missing_pct",
                                      color_continuous_scale="Reds", title="Missing values by column (%)",
                                      labels={"missing_pct": "% missing"}))
            sample = df.sample(min(200, len(df)), random_state=1).sort_index()
            cols_with_na = shown["column"].tolist()
            heat = sample[cols_with_na].isna().astype(int)
            st.plotly_chart(px.imshow(heat.T, aspect="auto", color_continuous_scale=["#EAF2F8", "#E4572E"],
                                      title="Missingness map (dark = missing, up to 200 sampled rows)",
                                      labels={"x": "Row", "y": "Column", "color": "Missing"}))

    # ---- Duplicates
    with tabs[4]:
        dup_n = int(df.duplicated().sum())
        c1, c2 = st.columns(2)
        c1.metric("Duplicate rows", f"{dup_n:,}")
        c2.metric("Share of dataset", f"{dup_n / len(df) * 100:.2f}%" if len(df) else "0%")
        if dup_n == 0:
            st.success("No duplicate rows 🎉")
        else:
            pie = pd.DataFrame({"kind": ["Unique rows", "Duplicate rows"], "rows": [len(df) - dup_n, dup_n]})
            st.plotly_chart(px.pie(pie, names="kind", values="rows", hole=0.45, title="Unique vs duplicate rows"))
            st.write("Rows that appear more than once (all copies shown):")
            st.dataframe(eda.duplicate_rows(df).sort_values(by=df.columns.tolist()))

    # ---- Statistical summary
    with tabs[5]:
        if num:
            st.subheader("Numerical")
            st.dataframe(eda.numeric_summary(df, num))
        if cat:
            st.subheader("Categorical")
            st.dataframe(eda.categorical_summary(df, cat), hide_index=True)
        if not num and not cat:
            st.info("No numerical or categorical columns to summarise.")

    # ---- Numerical analysis
    with tabs[6]:
        if not num:
            st.info("No numerical columns.")
        else:
            col = st.selectbox("Column", num, key="num_col")
            s = df[col].dropna()
            if s.empty:
                st.warning("This column has no values.")
            else:
                m = st.columns(6)
                m[0].metric("Mean", f"{s.mean():,.2f}")
                m[1].metric("Median", f"{s.median():,.2f}")
                m[2].metric("Std dev", f"{s.std():,.2f}")
                m[3].metric("Min", f"{s.min():,.2f}")
                m[4].metric("Max", f"{s.max():,.2f}")
                m[5].metric("Skew", f"{s.skew():.2f}")
                bins = st.slider("Bins", 5, 100, 30)
                st.plotly_chart(px.histogram(df, x=col, nbins=bins, marginal="box",
                                             title=f"Distribution of {col}"))
            if st.checkbox("Show histograms for all numerical columns"):
                for c in num:
                    st.plotly_chart(px.histogram(df, x=c, nbins=30, title=c), key=f"hist_{c}")

    # ---- Categorical analysis
    with tabs[7]:
        if not cat:
            st.info("No categorical columns.")
        else:
            col = st.selectbox("Column", cat, key="cat_col")
            top_n = st.slider("Top categories", 3, 50, 10)
            vc = df[col].astype("string").value_counts(dropna=True).head(top_n).reset_index()
            vc.columns = [col, "count"]
            left, right = st.columns(2)
            left.plotly_chart(px.bar(vc, x=col, y="count", color="count", title=f"Top {top_n}: {col}"))
            right.plotly_chart(px.pie(vc, names=col, values="count", hole=0.4, title=f"Share: {col}"))
            st.dataframe(vc, hide_index=True)

    # ---- Correlation
    with tabs[8]:
        if len(num) < 2:
            st.info("At least two numerical columns are needed for correlation.")
        else:
            method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
            corr = df[num].corr(method=method)
            st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                                      color_continuous_scale="RdBu_r", title=f"{method.title()} correlation"))
            st.subheader("Strongest relationships")
            st.dataframe(eda.top_correlations(corr, 15), hide_index=True)

    # ---- Outliers
    with tabs[9]:
        if not num:
            st.info("No numerical columns.")
        else:
            k = st.slider("IQR multiplier (1.5 = standard, 3 = extreme only)", 1.0, 3.0, 1.5, 0.5)
            ot = eda.outlier_table(df, num, k)
            st.dataframe(ot, hide_index=True)
            chosen = st.multiselect("Box plots for", num, default=num[:4])
            if chosen:
                st.plotly_chart(px.box(df, y=chosen, points="outliers", title="Box plots (dots = outliers)"))
            col = st.selectbox("Show outlier rows of", num, key="out_col")
            rows = eda.outlier_rows(df, col, k)
            st.write(f"{len(rows)} outlier rows in **{col}**")
            if len(rows):
                st.dataframe(rows)

    # ---- Interactive visualizations
    with tabs[10]:
        interactive_charts(df, num, cat)

    # ---- Insights
    with tabs[11]:
        st.subheader("Automatically generated key insights")
        for line in eda.generate_insights(df):
            st.markdown(f"- {line}")


def interactive_charts(df: pd.DataFrame, num, cat):
    """Chart builder: choose a chart type and columns."""
    cols = df.columns.tolist()
    kind = st.selectbox("Chart type", ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Pie", "Scatter matrix"])
    color_opt = [None] + cols
    plot_df = df if len(df) <= 20000 else df.sample(20000, random_state=1)
    if len(df) > 20000:
        st.caption("Large dataset: showing a random sample of 20,000 rows.")

    fig = None
    if kind == "Scatter":
        if len(num) < 2:
            st.info("Needs at least two numerical columns.")
        else:
            c = st.columns(3)
            x = c[0].selectbox("X", num)
            y = c[1].selectbox("Y", num, index=1)
            color = c[2].selectbox("Color by", color_opt)
            fig = px.scatter(plot_df, x=x, y=y, color=color, hover_data=cols[:6], title=f"{y} vs {x}")
    elif kind == "Line":
        if not num:
            st.info("Needs a numerical column.")
        else:
            c = st.columns(3)
            x = c[0].selectbox("X", cols)
            y = c[1].selectbox("Y", num)
            color = c[2].selectbox("Color by", color_opt)
            fig = px.line(plot_df.sort_values(x), x=x, y=y, color=color, title=f"{y} by {x}")
    elif kind == "Bar":
        c = st.columns(3)
        x = c[0].selectbox("Category (X)", cols)
        y = c[1].selectbox("Value (Y)", ["(count of rows)"] + num)
        agg = c[2].selectbox("Aggregation", ["mean", "sum", "median", "max", "min"])
        if y == "(count of rows)":
            g = plot_df[x].astype("string").value_counts().head(30).reset_index()
            g.columns = [x, "count"]
            fig = px.bar(g, x=x, y="count", title=f"Rows per {x}")
        else:
            g = plot_df.groupby(x)[y].agg(agg).reset_index().sort_values(y, ascending=False).head(30)
            fig = px.bar(g, x=x, y=y, title=f"{agg.title()} of {y} by {x}")
    elif kind == "Histogram":
        c = st.columns(3)
        x = c[0].selectbox("Column", cols)
        color = c[1].selectbox("Color by", color_opt)
        bins = c[2].slider("Bins", 5, 100, 30)
        fig = px.histogram(plot_df, x=x, color=color, nbins=bins, title=f"Histogram of {x}")
    elif kind in ("Box", "Violin"):
        if not num:
            st.info("Needs a numerical column.")
        else:
            c = st.columns(2)
            y = c[0].selectbox("Value (Y)", num)
            x = c[1].selectbox("Group by (optional)", [None] + cols)
            fn = px.box if kind == "Box" else px.violin
            fig = fn(plot_df, y=y, x=x, color=x, title=f"{kind} plot of {y}")
    elif kind == "Pie":
        c = st.columns(3)
        names = c[0].selectbox("Categories", cols)
        values = c[1].selectbox("Size by", ["(count of rows)"] + num)
        top = c[2].slider("Top N", 3, 20, 8)
        if values == "(count of rows)":
            g = plot_df[names].astype("string").value_counts().head(top).reset_index()
            g.columns = [names, "value"]
        else:
            g = plot_df.groupby(names)[values].sum().sort_values(ascending=False).head(top).reset_index()
            g.columns = [names, "value"]
        fig = px.pie(g, names=names, values="value", hole=0.35, title=f"{names} share")
    elif kind == "Scatter matrix":
        if len(num) < 2:
            st.info("Needs at least two numerical columns.")
        else:
            chosen = st.multiselect("Columns (2-6)", num, default=num[:4], max_selections=6)
            color = st.selectbox("Color by", color_opt)
            if len(chosen) >= 2:
                fig = px.scatter_matrix(plot_df, dimensions=chosen, color=color, title="Scatter matrix")
                fig.update_traces(diagonal_visible=False, marker=dict(size=4))
    if fig is not None:
        st.plotly_chart(fig)


# ============================================================ page: cleaning
def page_cleaning():
    st.header("🧹 Data Cleaning")
    if need_data():
        return
    df = st.session_state.df
    raw = st.session_state.raw_df
    num, cat, dt = eda.column_groups(df)
    cur, orig = eda.overview(df), eda.overview(raw)

    st.subheader("Original vs current")
    c = st.columns(4)
    c[0].metric("Rows", cur["rows"], cur["rows"] - orig["rows"])
    c[1].metric("Columns", cur["columns"], cur["columns"] - orig["columns"])
    c[2].metric("Missing values", cur["missing"], cur["missing"] - orig["missing"], delta_color="inverse")
    c[3].metric("Duplicate rows", cur["duplicates"], cur["duplicates"] - orig["duplicates"], delta_color="inverse")

    # ---- convert types (applied immediately)
    st.subheader("1. Convert data types")
    c1, c2, c3 = st.columns([2, 2, 1])
    col = c1.selectbox("Column", df.columns.tolist(), key="conv_col")
    c1.caption(f"Current type: {df[col].dtype}")
    target = c2.selectbox("Convert to", eda.DTYPE_TARGETS, key="conv_target")
    c3.write("")
    c3.write("")
    if c3.button("Convert"):
        try:
            new_df, msg = eda.convert_dtype(df, col, target)
            st.session_state.df = new_df
            st.session_state.log.append(msg)
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Conversion failed: {e}")

    # ---- other steps (applied together)
    st.subheader("2. Choose cleaning steps")
    miss_num = [c for c in num if df[c].isna().any()]
    miss_cat = [c for c in cat if df[c].isna().any()]
    with st.form("cleaning_form"):
        do_dups = st.checkbox(f"Remove duplicate rows ({cur['duplicates']} found)")

        do_num = st.checkbox("Fill missing numerical values")
        n1, n2 = st.columns(2)
        num_method = n1.radio("Fill with", ["Mean", "Median"], horizontal=True)
        num_cols = n2.multiselect("Numerical columns", num, default=miss_num)

        do_cat = st.checkbox("Fill missing categorical values with the mode (most frequent value)")
        cat_cols = st.multiselect("Categorical columns", cat, default=miss_cat)

        do_drop = st.checkbox("Drop rows that still have missing values")
        drop_cols = st.multiselect("Only look at these columns (leave empty = all columns)", df.columns.tolist())

        submitted = st.form_submit_button("Apply selected steps", type="primary")
        st.caption("Order of execution: duplicates → numerical fill → categorical fill → drop missing rows.")

    if submitted:
        work, msgs = df, []
        if do_dups:
            work, m = eda.remove_duplicates(work)
            msgs.append(m)
        if do_num and num_cols:
            work, m = eda.fill_numeric(work, num_cols, num_method)
            msgs.append(m)
        if do_cat and cat_cols:
            work, m = eda.fill_categorical(work, cat_cols)
            msgs.append(m)
        if do_drop:
            work, m = eda.drop_missing(work, drop_cols)
            msgs.append(m)
        if msgs:
            st.session_state.df = work
            st.session_state.log.extend(msgs)
            st.rerun()
        else:
            st.warning("No step selected (or no columns chosen for the selected fill).")

    # ---- log, reset, download
    st.subheader("3. Result")
    if st.session_state.log:
        st.markdown("**Cleaning log**")
        for i, m in enumerate(st.session_state.log, 1):
            st.markdown(f"{i}. {m}")
    else:
        st.caption("No cleaning applied yet.")

    b1, b2 = st.columns(2)
    if b1.button("↩️ Reset to original data"):
        st.session_state.df = st.session_state.raw_df.copy()
        st.session_state.log = []
        st.rerun()
    csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
    b2.download_button("⬇️ Download cleaned CSV", csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv")
    st.dataframe(st.session_state.df.head(20))


# ============================================================ router
page = st.sidebar.radio(
    "Navigation",
    ["📁 Upload Data", "📊 Dashboard", "🧹 Data Cleaning", "🎓 Student Performance"],
)
if page.endswith("Upload Data"):
    page_upload()
elif page.endswith("Dashboard"):
    page_dashboard()
elif page.endswith("Data Cleaning"):
    page_cleaning()
else:
    student_pages.render()