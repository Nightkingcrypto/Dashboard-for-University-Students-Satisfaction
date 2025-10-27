
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from urllib.parse import urlparse, parse_qs

st.set_page_config(page_title="University Students Survey – Smart Dashboard", layout="wide")

# -----------------------------
# Config: expected columns
# -----------------------------
CATEGORICALS = ['age_group','gender','program','study_mode','district','university','uses_support_services']
LIKERT_COLS  = ['internet_reliability','lecture_quality','assessment_clarity','platform_usability','peer_collaboration']
NUMERIC_COLS = ['hours_study','sleep_hours','stress_level','overall_satisfaction']

# Windows default path (requested)
DEFAULT_WINDOWS_PATH = Path(r"C:\Users\kaThi\OneDrive\Desktop\ID_CW02_034_033_024\Codes\university students survey data.xlsx")

# -----------------------------
# Helpers for Google Sheets URLs
# -----------------------------
def gsheets_to_csv_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        if "docs.google.com" in parsed.netloc and "/spreadsheets/" in parsed.path:
            parts = parsed.path.split("/")
            sheet_id = None
            for i, p in enumerate(parts):
                if p == "d" and i+1 < len(parts):
                    sheet_id = parts[i+1]
                    break
            gid = None
            if parsed.query:
                q = parse_qs(parsed.query)
                if "gid" in q and len(q["gid"])>0:
                    gid = q["gid"][0]
            if parsed.fragment and "gid=" in parsed.fragment:
                try:
                    frag_q = parse_qs(parsed.fragment.replace("#", "").replace("?", "&"))
                    if "gid" in frag_q and len(frag_q["gid"])>0:
                        gid = frag_q["gid"][0]
                except Exception:
                    pass
            if sheet_id:
                base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                if gid:
                    base += f"&gid={gid}"
                return base
    except Exception:
        pass
    return url

# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_excel(file_or_path):
    xls = pd.ExcelFile(file_or_path)
    df = xls.parse(xls.sheet_names[0])
    for col in NUMERIC_COLS + LIKERT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@st.cache_data(show_spinner=False)
def load_csv(file_or_url):
    df = pd.read_csv(file_or_url)
    for col in NUMERIC_COLS + LIKERT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# -----------------------------
# Data Source UI
# -----------------------------
st.title("🎓 University Students Survey — Smart Insights Dashboard")

with st.sidebar:
    st.header("Data Source")
    source = st.radio(
        "Choose source",
        ["Bundled Excel", "Upload Excel/CSV", "Google Sheets (CSV)", "Direct CSV URL"],
        index=0
    )

    # Prefer the Windows default path if it exists on the user's machine, otherwise look for local file next to app
    default_path = DEFAULT_WINDOWS_PATH if DEFAULT_WINDOWS_PATH.exists() else Path("university students survey data.xlsx")

    csv_url = None
    uploaded = None
    if source == "Upload Excel/CSV":
        uploaded = st.file_uploader("Upload .xlsx or .csv", type=["xlsx","csv"])
    elif source == "Google Sheets (CSV)":
        helper = st.expander("How to get a CSV link from Google Sheets?")
        with helper:
            st.markdown("""
1) In Google Sheets: **File → Share → Anyone with the link (Viewer)**  
2) **File → Publish to the web → Link → CSV** (select your sheet) → Copy the URL  
   *(Or paste the normal Sheets link like `.../edit#gid=...` — I'll convert it to CSV for you.)*
            """)
        csv_input = st.text_input("Paste Google Sheets link:", placeholder="https://docs.google.com/spreadsheets/d/…/edit#gid=0")
        if csv_input:
            csv_url = gsheets_to_csv_url(csv_input.strip())
            st.caption(f"Using CSV URL: {csv_url}")
    elif source == "Direct CSV URL":
        csv_url = st.text_input("Paste direct CSV URL:", placeholder="https://example.com/data.csv")

# Resolve DF with safe fallbacks
df = None
load_status = ""

try:
    if source == "Upload Excel/CSV":
        if uploaded is not None:
            if uploaded.name.lower().endswith(".xlsx"):
                df = load_excel(uploaded); load_status = f"Loaded: {uploaded.name} (Excel)"
            elif uploaded.name.lower().endswith(".csv"):
                df = load_csv(uploaded); load_status = f"Loaded: {uploaded.name} (CSV)"
        # Fallback to default_path if nothing uploaded
        if df is None and default_path.exists():
            df = load_excel(default_path); load_status = f"Loaded default Excel: {default_path}"
    elif source == "Google Sheets (CSV)":
        if csv_url:
            df = load_csv(csv_url); load_status = "Loaded from Google Sheets CSV"
        elif default_path.exists():
            df = load_excel(default_path); load_status = f"Loaded default Excel: {default_path}"
    elif source == "Direct CSV URL":
        if csv_url:
            df = load_csv(csv_url); load_status = "Loaded from direct CSV URL"
        elif default_path.exists():
            df = load_excel(default_path); load_status = f"Loaded default Excel: {default_path}"
    elif source == "Bundled Excel":
        if default_path.exists():
            df = load_excel(default_path); load_status = f"Loaded default Excel: {default_path}"
        else:
            # As last resort, try a local file name
            local_guess = Path("university students survey data.xlsx")
            if local_guess.exists():
                df = load_excel(local_guess); load_status = f"Loaded bundled Excel: {local_guess}"
except Exception as e:
    st.error(f"Failed to load data: {e}")

if df is None:
    st.warning("No data source found. Please upload a file or provide a CSV/Google Sheets link.")
    st.stop()

st.success(load_status)

# Column sanity
missing_expected = [c for c in (CATEGORICALS + LIKERT_COLS + NUMERIC_COLS) if c not in df.columns]
if missing_expected:
    st.warning(f"These expected columns are missing: {', '.join(missing_expected)}")
    st.caption("The dashboard will still run with available columns.")

# -----------------------------
# Checkbox Filters
# -----------------------------
with st.sidebar:
    st.header("Filters")
    filt = dict()

    def checkbox_group(col_name, options):
        st.write(f"**{col_name}**")
        selected = []
        # 2-column layout for compactness
        cols = st.columns(2)
        for i, opt in enumerate(options):
            c = cols[i % 2]
            with c:
                checked = st.checkbox(str(opt), value=True, key=f"{col_name}_{str(opt)}")
            if checked:
                selected.append(str(opt))
        # If user unchecks everything, treat as no filtering (i.e., all)
        return selected

    for col in CATEGORICALS:
        if col in df.columns:
            opts = sorted([str(x) for x in df[col].dropna().astype(str).unique()])
            selected_opts = checkbox_group(col, opts)
            filt[col] = selected_opts  # empty list => no filter

    # Range sliders for numerics unchanged
    for ncol in NUMERIC_COLS:
        if ncol in df.columns and pd.api.types.is_numeric_dtype(df[ncol]):
            minv, maxv = float(df[ncol].min()), float(df[ncol].max())
            sel = st.slider(f"{ncol} range", min_value=minv, max_value=maxv, value=(minv, maxv))
            filt[ncol] = sel

# Apply filters
mask = pd.Series([True]*len(df))
for k,v in filt.items():
    if k in CATEGORICALS and k in df.columns:
        if isinstance(v, list) and len(v) > 0:
            mask &= df[k].astype(str).isin([str(s) for s in v])
        # else: treat as All (no filter)
    elif k in NUMERIC_COLS and k in df.columns:
        lo, hi = v
        mask &= df[k].between(lo, hi)

filtered = df[mask].copy()

# -----------------------------
# KPIs
# -----------------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Responses", int(len(filtered)))
with kpi_cols[1]:
    if "overall_satisfaction" in filtered.columns:
        val = float(filtered['overall_satisfaction'].mean()) if len(filtered)>0 else float("nan")
        st.metric("Avg. Satisfaction", f"{val:.2f} / 10")
with kpi_cols[2]:
    if "hours_study" in filtered.columns:
        val = float(filtered['hours_study'].mean()) if len(filtered)>0 else float("nan")
        st.metric("Avg. Study Hours", f"{val:.1f} hrs/day")
with kpi_cols[3]:
    if "sleep_hours" in filtered.columns:
        val = float(filtered['sleep_hours'].mean()) if len(filtered)>0 else float("nan")
        st.metric("Avg. Sleep", f"{val:.1f} hrs/day")

st.markdown("---")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_demographics, tab_academics, tab_experience, tab_stress, tab_corr = st.tabs(
    ["Overview", "Demographics", "Study Habits", "Learning Experience", "Stress & Support", "Correlations"]
)

def bar_count(df_, col, title):
    tmp = df_[col].astype(str).value_counts(dropna=False).reset_index()
    tmp.columns = [col, "count"]
    fig = px.bar(tmp, x=col, y="count", title=title, text="count")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="", yaxis_title="Count", margin=dict(t=60,b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        if "program" in filtered.columns and "overall_satisfaction" in filtered.columns:
            by_prog = filtered.groupby("program", as_index=False)["overall_satisfaction"].mean().sort_values("overall_satisfaction", ascending=False)
            fig = px.bar(by_prog, x="program", y="overall_satisfaction", title="Avg. Satisfaction by Program", text="overall_satisfaction")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="Avg Satisfaction", margin=dict(t=60,b=20))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "uses_support_services" in filtered.columns:
            bar_count(filtered, "uses_support_services", "Use of Student Support Services")

    TRENDLINE = None
    try:
        import statsmodels.api as sm  # noqa
        TRENDLINE = "ols"
    except Exception:
        pass

    if {"hours_study","overall_satisfaction"}.issubset(filtered.columns):
        hov = [c for c in ["program","gender","age_group"] if c in filtered.columns]
        fig = px.scatter(filtered, x="hours_study", y="overall_satisfaction",
                         trendline=TRENDLINE, hover_data=hov,
                         title="Study Hours vs Overall Satisfaction")
        st.plotly_chart(fig, use_container_width=True)

with tab_demographics:
    c1, c2, c3 = st.columns(3)
    if "gender" in filtered.columns:
        with c1: bar_count(filtered, "gender", "Gender Distribution")
    if "age_group" in filtered.columns:
        with c2: bar_count(filtered, "age_group", "Age Group")
    if "study_mode" in filtered.columns:
        with c3: bar_count(filtered, "study_mode", "Study Mode")
    c4, c5 = st.columns(2)
    if "district" in filtered.columns:
        with c4: bar_count(filtered, "district", "Top Districts")
    if "university" in filtered.columns:
        with c5: bar_count(filtered, "university", "Respondents by University")

with tab_academics:
    c1, c2 = st.columns(2)
    if "hours_study" in filtered.columns:
        with c1:
            fig = px.histogram(filtered, x="hours_study", nbins=20, title="Distribution of Study Hours")
            st.plotly_chart(fig, use_container_width=True)
    if "sleep_hours" in filtered.columns:
        with c2:
            fig = px.histogram(filtered, x="sleep_hours", nbins=20, title="Distribution of Sleep Hours")
            st.plotly_chart(fig, use_container_width=True)
    if {"hours_study","program"}.issubset(filtered.columns):
        fig = px.box(filtered, x="program", y="hours_study", points="suspectedoutliers", title="Study Hours by Program")
        st.plotly_chart(fig, use_container_width=True)

with tab_experience:
    st.subheader("Learning Experience (Likert-style)")
    likerts = [c for c in LIKERT_COLS if c in filtered.columns]
    if len(likerts) > 0:
        long = filtered[likerts].melt(var_name="aspect", value_name="score").dropna()
        long["score"] = pd.to_numeric(long["score"], errors="coerce")
        long = long.dropna()
        long["bucket"] = pd.cut(long["score"], bins=[0.5,1.5,2.5,3.5,4.5,5.5], labels=[1,2,3,4,5]).astype("int64")
        dist = long.groupby(["aspect","bucket"], as_index=False).size()
        dist["pct"] = dist.groupby("aspect")["size"].transform(lambda s: s / s.sum() * 100)

        order = [1,2,3,4,5]
        facets = []
        for asp in dist["aspect"].unique():
            tmp = dist[dist["aspect"]==asp].set_index("bucket").reindex(order).fillna(0.0).reset_index()
            tmp["aspect"] = asp
            facets.append(tmp)
        dist = pd.concat(facets, ignore_index=True)

        def signed(row):
            b = int(row["bucket"]); p = float(row["pct"])
            if b in [1,2]: return -p
            if b == 3: return p * 0.01
            return p
        dist["signed"] = dist.apply(signed, axis=1)
        legends = {1:"Strongly disagree",2:"Disagree",3:"Neutral",4:"Agree",5:"Strongly agree"}
        fig = go.Figure()
        for b in order:
            sub = dist[dist["bucket"]==b]
            fig.add_trace(go.Bar(x=sub["aspect"], y=sub["signed"], name=legends[b],
                                 hovertext=[f"{p:.1f}%" for p in sub["pct"]]))
        fig.update_layout(barmode="relative", title="Diverging Stacked Bars (1–5 scale)",
                          yaxis_title="%", xaxis_title="Aspect")
        st.plotly_chart(fig, use_container_width=True)

    if "overall_satisfaction" in filtered.columns and len(likerts)>0:
        corr = []
        for c in likerts:
            ok = filtered[[c,"overall_satisfaction"]].dropna()
            if len(ok)>2:
                corr.append({"aspect": c, "corr": float(ok[c].corr(ok["overall_satisfaction"]))})
        if corr:
            cc = pd.DataFrame(corr).sort_values("corr", ascending=False)
            fig = px.bar(cc, x="aspect", y="corr", title="Which aspects align with Satisfaction? (Pearson r)")
            st.plotly_chart(fig, use_container_width=True)

with tab_stress:
    c1, c2 = st.columns(2)
    if "stress_level" in filtered.columns:
        with c1:
            fig = px.histogram(filtered, x="stress_level", nbins=20, title="Stress Level Distribution")
            st.plotly_chart(fig, use_container_width=True)

    TRENDLINE_2 = None
    try:
        import statsmodels.api as sm  # noqa
        TRENDLINE_2 = "ols"
    except Exception:
        pass

    if {"sleep_hours","stress_level"}.issubset(filtered.columns):
        with c2:
            fig = px.scatter(filtered, x="sleep_hours", y="stress_level", trendline=TRENDLINE_2, title="Sleep vs Stress")
            st.plotly_chart(fig, use_container_width=True)

    if {"uses_support_services","stress_level"}.issubset(filtered.columns):
        grp = filtered.groupby("uses_support_services", as_index=False)["stress_level"].mean()
        fig = px.bar(grp, x="uses_support_services", y="stress_level", title="Mean Stress by Support Service Use", text="stress_level")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

with tab_corr:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        corr = filtered[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap (numerical columns)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Note: Correlations show association, not causation.")

st.markdown("—")
st.caption("Tip: Use the sidebar to filter (checkboxes = include). If you uncheck everything in a group, it's treated as 'All'.")
