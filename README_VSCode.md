# Coursework 2 — University Students Survey (VS Code + Streamlit)

This package includes a ready-to-run Streamlit dashboard tailored to your dataset:
**university students survey data.xlsx** (first sheet).

## 1) Open in VS Code
- Copy the files to your project folder.
- Keep the Excel in the same directory or upload it in the app.
- To use your Windows path directly, edit `default_path` in `app.py` to your local path:
  `default_path = Path(r"C:\Users\kaThi\OneDrive\Desktop\ID_CW02_034_033_024\university students survey data.xlsx")`

## 2) Create a virtual environment (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Run the dashboard
```powershell
streamlit run app.py
```

## 4) Dashboard tabs
- **Overview**: KPIs, satisfaction by program, support uptake, hours vs satisfaction.
- **Demographics**: Gender, age group, study mode, district, university.
- **Study Habits**: Histograms (study/sleep), boxplot (study hours by program).
- **Learning Experience**: Diverging Likert bars for internet_reliability, lecture_quality, assessment_clarity, platform_usability, peer_collaboration; alignment with satisfaction.
- **Stress & Support**: Stress distribution, sleep vs stress, stress by support usage.
- **Correlations**: Heatmap across numeric variables.

All visuals are filter-aware via the sidebar.

## 5) Tips for “smart” presentations
- Use the export menu on each chart to save PNGs.
- Create a story: start from KPIs → where we win/lose → drivers of satisfaction → actions.
- Keep labels concise; annotate 2–3 key callouts per slide.
