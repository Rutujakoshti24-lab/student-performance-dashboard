# 🎓 Student Performance Dashboard + Data Explorer

One project, two parts:

1. **Data Explorer** (Streamlit): upload any CSV/Excel file, explore it with interactive Plotly charts, clean it, and download the result.
2. **Student Performance** (FastAPI + Streamlit): marks, attendance, rankings, pass rates, subject-wise charts, and a "needs academic support" flag.

## Project structure

```
student_dashboard/
├── app.py             # Streamlit app (Upload, Dashboard, Data Cleaning, Student Performance)
├── eda.py             # analysis + cleaning functions (no Streamlit)
├── student_pages.py   # Student Performance page (talks to the API)
├── api.py             # FastAPI backend
├── cleaning.py        # student-data cleaning, ranking and risk logic
├── generate_data.py   # creates students_raw.csv and students_clean.csv
├── students_raw.csv   # sample messy data
├── students_clean.csv # cleaned sample dataset
├── requirements.txt
└── README.md
```

## Setup (Windows PowerShell)

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py        # only if students_clean.csv is missing
```

## Run

```powershell
# Data Explorer only (no API needed)
streamlit run app.py

# Student Performance page also needs the API, in a second terminal
uvicorn api:app --reload       # docs: http://127.0.0.1:8000/docs
```

No API key is needed. The "Student API URL" box in the sidebar is just the server address.

## Streamlit pages

### 📁 Upload Data
Upload CSV or Excel (or load the sample). Immediately shows: number of rows, number of columns, numerical columns, categorical columns, missing values, duplicate rows.

### 📊 Dashboard
Tabs: Overview, Preview, Data Types, Missing Values, Duplicates, Statistical Summary, Numerical, Categorical, Correlation (Pearson/Spearman/Kendall), Outliers (IQR rule), Visualizations (chart builder: scatter, line, bar, histogram, box, violin, pie, scatter matrix), and Key Insights (auto-generated findings). All charts use Plotly.

The dashboard always analyses the current version of the data, so cleaning changes show up here.

### 🧹 Data Cleaning
Optional controls:
- Convert data types (integer, float, text, category, datetime, boolean). Applied immediately; values that cannot be converted become missing and are reported.
- Remove duplicate rows
- Fill missing numerical values with Mean or Median
- Fill missing categorical values with the Mode
- Drop rows with missing values (all columns or selected ones)

Steps in the form run in this order: duplicates, numerical fill, categorical fill, drop missing rows. A log records every action, **Reset** restores the uploaded data, and **Download cleaned CSV** saves the result.

### 🎓 Student Performance
Rankings, pass rates, subject-wise charts, at-risk students, record management and a clean-CSV download, all served by `api.py`.

## Student API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET/POST | `/students` | List / add students |
| GET/PUT/DELETE | `/students/{id}` | Read / update / delete one student |
| GET/PUT | `/students/{id}/marks` | Subject marks |
| GET/PUT | `/students/{id}/attendance` | Attendance % |
| GET | `/analytics/summary` | Pass rate, averages, at-risk count |
| GET | `/analytics/rankings` | Ranking (`top`, `student_class`) |
| GET | `/analytics/pass-rate` | Overall, by class, by subject |
| GET | `/analytics/subject-stats` | Average / highest / lowest per subject |
| GET | `/analytics/at-risk` | Students needing support (`level=Medium` or `High`) |
| GET | `/export/clean-csv` | Download cleaned dataset |

## Needs academic support score

```
risk = 50% x weak overall marks (below 60%) + 30% x low attendance (below 85%) + 20% x share of failed subjects
```
High >= 30, Medium 15-30, Low < 15. Pass means at least 40 in every subject. This is a transparent rule-based score; with historical outcomes you could train a scikit-learn model instead.

## Troubleshooting

- `No module named 'cleaning'` / `'eda'`: all `.py` files must be in the same folder you run commands from.
- `API error 404 Not Found`: the server on that address is not this project's `api.py`. Open `/docs` in the browser to check, and make sure nothing else uses port 8000.
- Excel upload fails: run `python -m pip install openpyxl`.