"""Analysis and cleaning helpers (pandas only, no Streamlit) used by app.py."""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

DTYPE_TARGETS = ["integer", "float", "text (string)", "category", "datetime", "boolean"]


# ---------------------------------------------------------------- structure
def column_groups(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Return (numerical, categorical, datetime) column names."""
    num = df.select_dtypes(include="number").columns.tolist()
    dt = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat = [c for c in df.columns if c not in num and c not in dt]
    return num, cat, dt


def overview(df: pd.DataFrame) -> dict:
    num, cat, dt = column_groups(df)
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "numerical": len(num),
        "categorical": len(cat),
        "datetime": len(dt),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
    }


def dtype_table(df: pd.DataFrame) -> pd.DataFrame:
    examples = []
    for c in df.columns:
        s = df[c].dropna()
        examples.append(str(s.iloc[0]) if len(s) else "")
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_null": df.notna().sum().values,
        "unique": df.nunique().values,
        "example": examples,
    })


# ---------------------------------------------------------------- quality
def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().sum()
    pct = (m.values / len(df) * 100).round(2) if len(df) else np.zeros(len(m))
    out = pd.DataFrame({"column": m.index, "missing": m.values, "missing_pct": pct})
    return out.sort_values("missing", ascending=False).reset_index(drop=True)


def duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """All rows that have at least one identical twin (including the first occurrence)."""
    return df[df.duplicated(keep=False)]


# ---------------------------------------------------------------- statistics
def numeric_summary(df: pd.DataFrame, num: List[str]) -> pd.DataFrame:
    if not num:
        return pd.DataFrame()
    s = df[num].describe().T
    s["skew"] = df[num].skew()
    s["kurtosis"] = df[num].kurt()
    s["missing"] = df[num].isna().sum()
    return s.round(3)


def categorical_summary(df: pd.DataFrame, cat: List[str]) -> pd.DataFrame:
    rows = []
    for c in cat:
        vc = df[c].astype("string").value_counts(dropna=True)
        non_null = int(df[c].notna().sum())
        rows.append({
            "column": c,
            "unique": int(df[c].nunique(dropna=True)),
            "top_value": str(vc.index[0]) if len(vc) else "",
            "top_count": int(vc.iloc[0]) if len(vc) else 0,
            "top_pct": round(vc.iloc[0] / non_null * 100, 2) if len(vc) and non_null else 0.0,
            "missing": int(df[c].isna().sum()),
        })
    return pd.DataFrame(rows)


def outlier_table(df: pd.DataFrame, num: List[str], k: float = 1.5) -> pd.DataFrame:
    """IQR rule: outlier if value < Q1 - k*IQR or > Q3 + k*IQR."""
    rows = []
    for c in num:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "column": c, "Q1": round(q1, 3), "Q3": round(q3, 3), "IQR": round(iqr, 3),
            "lower_bound": round(lo, 3), "upper_bound": round(hi, 3),
            "outliers": n, "outlier_pct": round(n / len(s) * 100, 2),
        })
    return pd.DataFrame(rows)


def outlier_rows(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.DataFrame:
    s = df[col]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return df[(s < q1 - k * iqr) | (s > q3 + k * iqr)]


def top_correlations(corr: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if corr.shape[0] < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().reset_index().dropna()
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    pairs["abs"] = pairs["correlation"].abs()
    return pairs.sort_values("abs", ascending=False).drop(columns="abs").head(n).round(3).reset_index(drop=True)


# ---------------------------------------------------------------- insights
def generate_insights(df: pd.DataFrame) -> List[str]:
    """Plain-English observations generated from the data."""
    if df.empty:
        return ["The dataset is empty."]
    n_rows, n_cols = df.shape
    num, cat, dt = column_groups(df)
    out = [f"📐 The dataset has **{n_rows:,} rows** and **{n_cols} columns** "
           f"({len(num)} numerical, {len(cat)} categorical, {len(dt)} datetime)."]

    miss = missing_table(df)
    miss = miss[miss["missing"] > 0]
    if miss.empty:
        out.append("✅ There are **no missing values**.")
    else:
        total = int(miss["missing"].sum())
        share = total / (n_rows * n_cols) * 100
        worst = ", ".join(f"**{r.column}** ({r.missing_pct}%)" for r in miss.head(3).itertuples())
        out.append(f"⚠️ **{total:,} missing values** ({share:.1f}% of all cells) across {len(miss)} columns. "
                   f"Most affected: {worst}.")
        heavy = miss[miss["missing_pct"] > 50]["column"].tolist()
        if heavy:
            out.append(f"🚨 Columns with more than 50% missing: {', '.join(heavy)}. Consider dropping them.")

    dups = int(df.duplicated().sum())
    if dups:
        out.append(f"♻️ **{dups:,} duplicate rows** ({dups / n_rows * 100:.1f}%). Consider removing them.")
    else:
        out.append("✅ There are **no duplicate rows**.")

    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant:
        out.append(f"🧱 Constant columns (no information): {', '.join(constant)}.")

    id_like = [c for c in cat if n_rows > 20 and df[c].nunique() / n_rows > 0.9]
    if id_like:
        out.append(f"🔑 Almost-unique text columns (likely IDs or names): {', '.join(id_like)}.")

    skewed = [c for c in num if abs(df[c].skew()) > 1]
    if skewed:
        out.append(f"📈 Strongly skewed numerical columns: {', '.join(skewed[:6])}. "
                   "A log transform or median-based statistics may suit them better.")

    if len(num) >= 2:
        pairs = top_correlations(df[num].corr(), n=3)
        strong = pairs[pairs["correlation"].abs() >= 0.7]
        for r in strong.itertuples():
            kind = "positive" if r.correlation > 0 else "negative"
            out.append(f"🔗 Strong {kind} correlation between **{r.feature_1}** and **{r.feature_2}** "
                       f"(r = {r.correlation}).")

    ot = outlier_table(df, num)
    if not ot.empty:
        heavy_out = ot[ot["outlier_pct"] > 5]
        if len(heavy_out):
            names = ", ".join(f"{r.column} ({r.outlier_pct}%)" for r in heavy_out.itertuples())
            out.append(f"🎯 Many outliers (IQR rule) in: {names}.")
        elif ot["outliers"].sum():
            out.append(f"🎯 Some outliers detected ({int(ot['outliers'].sum())} values), but under 5% per column.")
        else:
            out.append("✅ No outliers detected using the IQR rule.")

    cs = categorical_summary(df, cat)
    if not cs.empty:
        dominated = cs[(cs["top_pct"] >= 80) & (cs["unique"] > 1)]
        for r in dominated.itertuples():
            out.append(f"⚖️ **{r.column}** is dominated by '{r.top_value}' ({r.top_pct}% of rows).")
    return out


# ---------------------------------------------------------------- cleaning
def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    n = int(df.duplicated().sum())
    return df.drop_duplicates().reset_index(drop=True), f"Removed {n} duplicate rows"


def fill_numeric(df: pd.DataFrame, cols: List[str], method: str) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    filled = 0
    for c in cols:
        k = int(df[c].isna().sum())
        if k == 0:
            continue
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("float64")  # allows a fractional mean
        value = df[c].mean() if method == "Mean" else df[c].median()
        if pd.isna(value):
            continue
        df[c] = df[c].fillna(value)
        filled += k
    return df, f"Filled {filled} missing numerical values using {method.lower()} ({len(cols)} columns checked)"


def fill_categorical(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    filled = 0
    for c in cols:
        k = int(df[c].isna().sum())
        if k == 0:
            continue
        mode = df[c].mode(dropna=True)
        if mode.empty:
            continue
        df[c] = df[c].fillna(mode.iloc[0])
        filled += k
    return df, f"Filled {filled} missing categorical values using mode ({len(cols)} columns checked)"


def drop_missing(df: pd.DataFrame, cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, str]:
    before = len(df)
    out = df.dropna(subset=cols if cols else None).reset_index(drop=True)
    where = f"in {len(cols)} selected columns" if cols else "in any column"
    return out, f"Dropped {before - len(out)} rows with missing values {where}"


def convert_dtype(df: pd.DataFrame, col: str, target: str) -> Tuple[pd.DataFrame, str]:
    """Convert one column. Values that cannot be converted become missing."""
    df = df.copy()
    s = df[col]
    na_before = int(s.isna().sum())

    if target == "integer":
        conv = pd.to_numeric(s, errors="coerce")
        conv = conv.where(~(conv.notna() & (conv % 1 != 0)))  # fractional values -> missing
        conv = conv.astype("Int64")
    elif target == "float":
        conv = pd.to_numeric(s, errors="coerce").astype("float64")
    elif target == "text (string)":
        conv = s.astype("string")
    elif target == "category":
        conv = s.astype("category")
    elif target == "datetime":
        try:
            conv = pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            conv = pd.to_datetime(s, errors="coerce")
    elif target == "boolean":
        txt = s.astype("string").str.strip().str.lower()
        conv = pd.Series(pd.NA, index=s.index, dtype="boolean")
        conv[txt.isin(["true", "t", "yes", "y", "1", "1.0"])] = True
        conv[txt.isin(["false", "f", "no", "n", "0", "0.0"])] = False
    else:
        raise ValueError(f"Unknown target type: {target}")

    lost = int(conv.isna().sum()) - na_before
    df[col] = conv
    msg = f"Converted '{col}' to {target}"
    if lost > 0:
        msg += f" ({lost} values could not be converted and became missing)"
    return df, msg