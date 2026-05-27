"""Streamlit dashboard — Diabetes Risk Screener.

Dark-mode, glassmorphism cards, Plotly-only charts with custom styling.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from diabetes_mlops import config as C  # noqa: E402
from diabetes_mlops.data import clean, load_raw  # noqa: E402
from diabetes_mlops.predict import load_artifact, predict_one  # noqa: E402

# ─── Streamlit page setup ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Diabetes Risk · ML Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS — glassmorphism, gradients, custom fonts ─────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Animated gradient background behind the main app */
.stApp {
    background:
      radial-gradient(1200px 600px at 10% 0%, rgba(124,58,237,0.18), transparent 50%),
      radial-gradient(900px 600px at 90% 10%, rgba(34,211,238,0.14), transparent 55%),
      radial-gradient(800px 500px at 50% 110%, rgba(244,114,182,0.12), transparent 60%),
      #07091a;
}

/* Hero banner */
.hero {
    border-radius: 22px;
    padding: 32px 36px;
    margin: 8px 0 28px 0;
    background:
      linear-gradient(135deg, rgba(124,58,237,0.32), rgba(34,211,238,0.22) 60%, rgba(244,114,182,0.20));
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow:
      0 20px 60px rgba(0,0,0,0.55),
      inset 0 1px 0 rgba(255,255,255,0.10);
    backdrop-filter: blur(14px);
    position: relative;
    overflow: hidden;
}
.hero h1 {
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    background: linear-gradient(90deg, #ffffff, #c4b5fd 40%, #67e8f9 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.hero p {
    font-size: 1.05rem;
    color: rgba(230,233,245,0.78);
    margin: 10px 0 0 0;
    max-width: 820px;
}
.hero .badge-row { margin-top: 14px; }
.hero .badge {
    display: inline-block;
    padding: 4px 12px;
    margin-right: 6px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    color: #e6e9f5;
}
.hero .badge.live { background: rgba(74,222,128,0.18); border-color: rgba(74,222,128,0.35); color:#bbf7d0; }

/* Glass metric card */
.metric-card {
    border-radius: 18px;
    padding: 22px 22px 18px 22px;
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 18px 50px rgba(0,0,0,0.5); }
.metric-card .label {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.74rem;
    color: rgba(230,233,245,0.55);
}
.metric-card .value {
    font-size: 2.4rem;
    font-weight: 700;
    margin-top: 8px;
    background: linear-gradient(90deg, #ffffff, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .delta { font-size: 0.85rem; color: rgba(230,233,245,0.65); margin-top:4px; }
.metric-card .accent {
    position: absolute; right: -30px; top: -30px;
    width: 120px; height: 120px; border-radius: 50%;
    opacity: 0.55; filter: blur(20px);
}
.accent-purple { background: radial-gradient(circle, #7c3aed, transparent 60%); }
.accent-cyan   { background: radial-gradient(circle, #22d3ee, transparent 60%); }
.accent-pink   { background: radial-gradient(circle, #f472b6, transparent 60%); }
.accent-green  { background: radial-gradient(circle, #4ade80, transparent 60%); }
.accent-amber  { background: radial-gradient(circle, #fbbf24, transparent 60%); }
.accent-rose   { background: radial-gradient(circle, #fb7185, transparent 60%); }

/* Section header */
.section-h {
    font-size: 1.55rem; font-weight: 600;
    margin: 18px 0 4px 0;
    background: linear-gradient(90deg, #ffffff, #a5b4fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.section-sub { color: rgba(230,233,245,0.55); font-size: 0.95rem; margin-bottom: 14px; }

/* Tab style override */
[data-baseweb="tab-list"] {
    gap: 6px; background: rgba(255,255,255,0.03);
    padding: 6px; border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}
[data-baseweb="tab"] {
    padding: 10px 18px !important;
    border-radius: 10px !important;
    transition: all 0.2s ease;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.45), rgba(34,211,238,0.30)) !important;
    color: #fff !important;
    border-color: rgba(255,255,255,0.18) !important;
    box-shadow: 0 6px 24px rgba(124,58,237,0.30);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(13,16,40,0.85), rgba(7,9,26,0.85));
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown { color: #e6e9f5; }

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #22d3ee 100%) !important;
    border: 0 !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 10px 26px rgba(124,58,237,0.45) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 16px 36px rgba(124,58,237,0.55) !important;
}

/* Result alert cards */
.result-flag, .result-clear {
    border-radius: 18px;
    padding: 22px 24px;
    margin-top: 12px;
    border: 1px solid;
    backdrop-filter: blur(10px);
    font-size: 1.05rem;
}
.result-flag {
    background: linear-gradient(135deg, rgba(244,63,94,0.22), rgba(251,146,60,0.20));
    border-color: rgba(244,63,94,0.40);
    color: #fecdd3;
}
.result-clear {
    background: linear-gradient(135deg, rgba(34,197,94,0.20), rgba(34,211,238,0.18));
    border-color: rgba(34,197,94,0.40);
    color: #bbf7d0;
}

#MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── Plotly defaults ─────────────────────────────────────────────────────────

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,16,40,0.55)",
    font=dict(family="Space Grotesk, sans-serif", color="#e6e9f5"),
    margin=dict(l=20, r=20, t=50, b=20),
)
PALETTE = {
    "neg": "#22d3ee",     # cyan — no diabetes
    "pos": "#f472b6",     # pink — diabetes
    "primary": "#7c3aed",
    "accent": "#fbbf24",
    "warn":   "#fb7185",
    "ok":     "#4ade80",
}


# ─── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return clean(load_raw())


@st.cache_resource(show_spinner=False)
def get_model():
    return load_artifact()


@st.cache_data(show_spinner=False)
def get_holdout():
    df = get_data()
    X, y = df[C.FEATURES], df[C.TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )
    art = get_model()
    proba = art["pipeline"].predict_proba(X_test)[:, 1]
    pred = (proba >= art["threshold"]).astype(int)
    return X_test, y_test, proba, pred


@st.cache_data(show_spinner="Permuting features...")
def get_importance() -> pd.DataFrame:
    X_test, y_test, _, _ = get_holdout()
    art = get_model()
    result = permutation_importance(
        art["pipeline"], X_test, y_test,
        n_repeats=10, random_state=C.RANDOM_STATE, n_jobs=-1, scoring="roc_auc",
    )
    return (
        pd.DataFrame({
            "feature": art["feature_order"],
            "importance": result.importances_mean,
            "std": result.importances_std,
        })
        .sort_values("importance", ascending=True)
    )


# ─── Guard: model must exist ─────────────────────────────────────────────────

try:
    artifact = get_model()
except FileNotFoundError as e:
    st.error(f"{e}\n\nRun `python -m diabetes_mlops.train` first.")
    st.stop()
m = artifact["metrics"]
THRESHOLD = artifact["threshold"]


# ─── Hero ────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
<div class="hero">
  <h1><span style="display:inline-block;padding:4px 10px;margin-right:8px;border-radius:8px;background:linear-gradient(90deg,#078930 0%,#fcd116 33%,#ce1126 66%,#000 100%);font-size:0.55em;letter-spacing:.08em;vertical-align:middle;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5);">ZIMBABWE</span>Diabetes Risk · Clinical ML Dashboard</h1>
  <p>An end-to-end MLOps demonstrator built with the Zimbabwean primary-care setting in mind —
     where confirmatory HbA1c testing is expensive and unevenly distributed, and a fast
     screening triage tool can help nurses and clinical officers route the right patients
     to the right test. Inputs are routine clinic measurements; the output is a calibrated risk probability.</p>
  <div class="badge-row">
    <span class="badge live">● Model loaded</span>
    <span class="badge">Algorithm · {artifact['model_name']}</span>
    <span class="badge">Threshold · {THRESHOLD:.3f}</span>
    <span class="badge">ROC-AUC · {m['roc_auc']:.3f}</span>
    <span class="badge">Recall · {m['recall']:.3f}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 At a glance")
    st.markdown(f"**Behind the scenes:** `{artifact['model_name']}`")
    st.caption("The kind of math the model uses. Not important if you're not a data person.")
    st.markdown(f"**Cut-off setting:** `{THRESHOLD:.2f}`")
    st.caption("Predicted risk above this number = the model raises a flag.")
    st.markdown("---")
    st.markdown("**How well it did on patients it had never seen:**")
    st.markdown(f"- Catches **{m['recall']:.0%}** of real diabetics")
    st.markdown(f"- Gets **{m['accuracy']:.0%}** of all patients right overall")
    st.markdown(f"- Risk-ranking skill: **{m['roc_auc']:.2f} / 1.00**")
    st.markdown("---")
    st.warning(
        "**Not a diagnosis.** This is a sorting/screening tool. "
        "Any real flag still needs a proper lab test to confirm."
    )
    st.markdown("---")
    st.caption("Built with Python · scikit-learn · MLflow · FastAPI · Streamlit · Plotly")


# ─── Tabs ────────────────────────────────────────────────────────────────────

t1, t2, t3, t4 = st.tabs(["✨ Overview", "📊 Data insights", "🧪 Model insights", "🎯 Try the model"])


# ═══ Helper: glass metric card ═══════════════════════════════════════════════

def metric_card(label: str, value: str, accent: str, delta: str = "") -> str:
    return f"""
<div class="metric-card">
  <div class="accent {accent}"></div>
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  <div class="delta">{delta}</div>
</div>
"""


# ═══ TAB 1 — Overview ════════════════════════════════════════════════════════

with t1:
    df = get_data()

    with st.expander("👋 First time here? Read this 30-second intro", expanded=False):
        st.markdown(
            """
**What this app does, in plain English:** A patient walks into a clinic. The nurse types in their basic
measurements (glucose level, BMI, blood pressure, age, etc.). The app gives back a **percentage chance**
that this patient has diabetes.

It's not a diagnosis — it's a sorting tool. The clinic only has so many slots for the expensive HbA1c
confirmation test. This helps the nurse decide *which patients should jump the queue*.

**The four tabs:**
1. **Overview** — the problem this is solving and the data we used.
2. **Data insights** — what the data looks like (mostly non-diabetic patients) and which measurements seem to matter.
3. **Model insights** — how well the model performs, and a slider so you can experiment with how "cautious" you want it to be.
4. **Try the model** — type in a hypothetical patient and watch the risk score appear live.
            """
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Patients in study", f"{len(df):,}", "accent-purple", "the data we trained on"), unsafe_allow_html=True)
    c2.markdown(metric_card("Had diabetes", f"{int(df[C.TARGET].sum()):,}", "accent-pink", f"{df[C.TARGET].mean():.0%} of the group"), unsafe_allow_html=True)
    c3.markdown(metric_card("Didn't", f"{int((1-df[C.TARGET]).sum()):,}", "accent-cyan", "the majority"), unsafe_allow_html=True)
    c4.markdown(metric_card("We catch", f"{m['recall']:.0%}", "accent-green", "of true diabetics on unseen patients"), unsafe_allow_html=True)

    st.markdown('<div class="section-h">Why this exists</div>', unsafe_allow_html=True)
    st.markdown(
        """
Zimbabwe's MoHCC reports a rising adult diabetes prevalence — roughly 10% by recent STEPS-survey estimates — and a
large share of cases remain undiagnosed until a complication shows up at Parirenyatwa, Harare Central or one of the
provincial referrals. HbA1c isn't cheap or universally available at primary-care level. So the practical question for
a nurse or clinical officer at a polyclinic isn't *"does this patient have diabetes?"* — that needs a proper lab — it's:

> *Which patients should jump the queue for that confirmatory test?*

The decision is asymmetric: missing a diabetic patient is much worse than asking a healthy person to come back for one more test.
The model is tuned for **high recall** rather than balanced accuracy, with the decision threshold treated as a policy parameter
that a clinician (not the data scientist) gets to set.
        """
    )

    st.markdown('<div class="section-h">How the pieces connect</div>', unsafe_allow_html=True)
    nodes = ["Raw CSV", "Clean", "Impute + Scale", "Train (3 models)", "Pick best by ROC-AUC", "Joblib artifact", "Streamlit", "FastAPI"]
    sankey_link = dict(
        source=[0, 1, 2, 3, 4, 5, 5],
        target=[1, 2, 3, 4, 5, 6, 7],
        value=[10, 10, 10, 10, 10, 5, 5],
        color=["rgba(124,58,237,0.35)"] * 7,
    )
    fig = go.Figure(go.Sankey(
        node=dict(
            label=nodes,
            pad=18, thickness=18,
            line=dict(color="rgba(255,255,255,0.18)", width=0.5),
            color=["#7c3aed", "#8b5cf6", "#a78bfa", "#22d3ee", "#67e8f9", "#fbbf24", "#fb7185", "#4ade80"],
        ),
        link=sankey_link,
    ))
    fig.update_layout(**{**PLOTLY_THEME, "margin": dict(l=10, r=10, t=20, b=10)}, height=260)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ A note on the data"):
        st.markdown(
            "The Pima Indians dataset uses literal `0` for missing values in five columns. "
            "The cleaning step replaces those with NaN before any model sees them; the sklearn pipeline imputes inside CV folds."
        )


# ═══ TAB 2 — Data insights ═══════════════════════════════════════════════════

with t2:
    df = get_data()

    st.markdown('<div class="section-h">How many of each in the data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Most patients in the dataset <em>don't</em> have diabetes. That sounds obvious, but it matters: "
        "a lazy model that just guesses \"no diabetes\" for everyone would still look right most of the time. "
        "We need to make sure ours actually catches the diabetic ones."
        '</div>',
        unsafe_allow_html=True,
    )
    bal = df[C.TARGET].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=["No diabetes", "Diabetes"],
        y=bal.values,
        marker=dict(
            color=[PALETTE["neg"], PALETTE["pos"]],
            line=dict(color="rgba(255,255,255,0.18)", width=1),
        ),
        text=[f"{v}" for v in bal.values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Patients: %{y}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_THEME, height=320, showlegend=False, yaxis_title="Patients")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-h">Does this measurement actually tell us anything?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Pick a measurement below. Pink = patients who have diabetes, cyan = patients who don't. "
        "If the two coloured shapes <strong>barely overlap</strong>, that measurement is super useful (try <em>Glucose</em>). "
        "If they sit on top of each other, the measurement is basically noise for telling the two groups apart (try <em>BloodPressure</em>)."
        '</div>',
        unsafe_allow_html=True,
    )
    feature = st.selectbox("Pick a measurement to compare", C.FEATURES, index=C.FEATURES.index("Glucose"))
    sub = df.dropna(subset=[feature]).copy()
    sub["Outcome"] = sub[C.TARGET].map({0: "No diabetes", 1: "Diabetes"})
    fig = px.histogram(
        sub, x=feature, color="Outcome",
        marginal="violin", barmode="overlay", opacity=0.55,
        nbins=30,
        color_discrete_map={"No diabetes": PALETTE["neg"], "Diabetes": PALETTE["pos"]},
    )
    fig.update_layout(**PLOTLY_THEME, height=430, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-h">What moves with what</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "<strong>Pink</strong> squares = two things rise together (e.g. higher BMI usually comes with higher skin-fold thickness). "
        "<strong>Cyan</strong> squares = one goes up as the other goes down. "
        "<strong>Dark</strong> = they have nothing to do with each other. Useful to know which measurements basically repeat each other."
        '</div>',
        unsafe_allow_html=True,
    )
    corr = df.corr(numeric_only=True)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        zmin=-1, zmax=1,
        colorscale=[
            [0.0,  "#22d3ee"],
            [0.5,  "#1a1a3a"],
            [1.0,  "#f472b6"],
        ],
        colorbar=dict(thickness=10),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
    ))
    fig.update_layout(**PLOTLY_THEME, height=520)
    st.plotly_chart(fig, use_container_width=True)


# ═══ TAB 3 — Model insights ══════════════════════════════════════════════════

with t3:
    X_test, y_test, proba, _ = get_holdout()

    st.markdown('<div class="section-h">How cautious should the model be?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Think of this like the sensitivity dial on a smoke alarm. "
        "<strong>Slide it left</strong> &rarr; flag almost everyone (catches every diabetic, but lots of false alarms). "
        "<strong>Slide it right</strong> &rarr; only flag the obvious cases (fewer false alarms, but you'll miss some real diabetics). "
        "Every chart on this tab updates live as you drag."
        "</div>",
        unsafe_allow_html=True,
    )
    live_threshold = st.slider(
        "Flag a patient if their predicted risk is at least…",
        min_value=0.05, max_value=0.95,
        value=float(THRESHOLD), step=0.01,
        format="%.2f",
        key="live_threshold",
        help=f"The default of {THRESHOLD:.2f} is the value the project tuned to so that the model catches at least {int(C.TARGET_RECALL * 100)}% of real diabetics.",
    )
    pred = (proba >= live_threshold).astype(int)

    live_recall = recall_score(y_test, pred, zero_division=0)
    live_precision = precision_score(y_test, pred, zero_division=0)
    live_accuracy = accuracy_score(y_test, pred)
    live_f1 = f1_score(y_test, pred, zero_division=0)
    delta_to_tuned = live_threshold - THRESHOLD

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Overall skill", f"{m['roc_auc']:.2f}", "accent-purple", "how well the model ranks risk (1.0 = perfect, 0.5 = coin flip)"), unsafe_allow_html=True)
    c2.markdown(metric_card("Diabetics caught", f"{live_recall:.0%}", "accent-green", "of real diabetics, this share gets flagged"), unsafe_allow_html=True)
    c3.markdown(metric_card("Flags that are right", f"{live_precision:.0%}", "accent-cyan", "of flagged patients, this share actually had diabetes"), unsafe_allow_html=True)
    c4.markdown(metric_card("Overall correct", f"{live_accuracy:.0%}", "accent-pink", f"balance score (F1) {live_f1:.2f}"), unsafe_allow_html=True)

    if abs(delta_to_tuned) < 1e-6:
        st.caption(f"📍 You're at the tuned default ({THRESHOLD:.2f}) — the setting the project ships with.")
    elif delta_to_tuned > 0:
        st.caption(f"🔼 You made the model {delta_to_tuned:+.2f} stricter. Fewer false alarms — but more real diabetics slip through.")
    else:
        st.caption(f"🔽 You made the model {abs(delta_to_tuned):.2f} more cautious. You'll catch more diabetics, at the cost of more false alarms.")

    st.markdown('<div class="section-h">Who the model got right (and wrong)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Read this like a 2&times;2 grid. <strong>Rows</strong> are what the patient actually was. "
        "<strong>Columns</strong> are what the model predicted. The two boxes on the top-left&rarr;bottom-right diagonal are the model getting it right. "
        "The other two are the mistakes — and not all mistakes are equal."
        '</div>',
        unsafe_allow_html=True,
    )
    cm = confusion_matrix(y_test, pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_labels = [[f"<b>{cm[i, j]}</b><br><span style='font-size:0.75em;opacity:0.7'>{cm_norm[i, j]:.0%}</span>" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=["No diabetes", "Diabetes"],
        y=["No diabetes", "Diabetes"],
        text=cm_labels,
        texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        colorscale=[[0.0, "rgba(124,58,237,0.10)"], [1.0, "#7c3aed"]],
        hovertemplate="True %{y} predicted %{x}: %{z:.1%}<extra></extra>",
        colorbar=dict(thickness=10),
    ))
    fig.update_layout(**PLOTLY_THEME, height=400, xaxis_title="Predicted", yaxis_title="True")
    st.plotly_chart(fig, use_container_width=True)
    tn, fp, fn, tp = cm.ravel()
    st.markdown(
        f"- 🟢 **{tn} healthy patients correctly cleared** — model said \"no diabetes\", reality agreed.\n"
        f"- 🟡 **{fp} healthy patients sent for an extra test** — false alarm. Costs the clinic time and money, but no harm done.\n"
        f"- 🔴 **{fn} diabetic patients sent home as healthy** — *the dangerous mistake.* These cases walk away undiagnosed.\n"
        f"- 🟢 **{tp} diabetic patients correctly flagged** — caught and sent for confirmation."
    )
    st.info(
        "In a clinic setting, the red row is what keeps a doctor up at night. That's why this model is "
        "tuned to be **biased toward false alarms** — better to test a healthy patient unnecessarily than to "
        "send a diabetic one home.",
        icon="💡",
    )

    st.markdown('<div class="section-h">The model\'s trade-off, visualised</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "These two charts show every possible setting of the slider, all at once. "
        "<strong>The amber dot is where <em>you</em> currently are</strong> — drag the slider above and watch it move. "
        "Curves that bulge toward the top-left (left chart) or top-right (right chart) mean a stronger model."
        '</div>',
        unsafe_allow_html=True,
    )
    tn_op, fp_op, fn_op, tp_op = cm.ravel()
    op_fpr = fp_op / max(1, fp_op + tn_op)
    op_tpr = live_recall
    op_precision = live_precision
    col_a, col_b = st.columns(2)
    with col_a:
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=PALETTE["primary"], width=3),
                                 fill="tozeroy", fillcolor="rgba(124,58,237,0.15)",
                                 name=f"AUC = {m['roc_auc']:.3f}", hovertemplate="FPR=%{x:.2f}<br>TPR=%{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="rgba(255,255,255,0.30)"), showlegend=False))
        fig.add_trace(go.Scatter(
            x=[op_fpr], y=[op_tpr], mode="markers",
            marker=dict(size=16, color=PALETTE["accent"], line=dict(color="white", width=2), symbol="circle"),
            name=f"Operating point (t={live_threshold:.2f})",
            hovertemplate=f"Threshold {live_threshold:.2f}<br>FPR=%{{x:.2f}}<br>TPR=%{{y:.2f}}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_THEME, height=380, title="ROC curve",
                          xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(orientation="h", yanchor="bottom", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        precision, recall_arr, _ = precision_recall_curve(y_test, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall_arr, y=precision, mode="lines",
                                 line=dict(color=PALETTE["pos"], width=3),
                                 fill="tozeroy", fillcolor="rgba(244,114,182,0.15)",
                                 hovertemplate="Recall=%{x:.2f}<br>Precision=%{y:.2f}<extra></extra>",
                                 name="PR curve"))
        fig.add_vline(x=C.TARGET_RECALL, line=dict(color="rgba(255,255,255,0.30)", dash="dash", width=1.5),
                      annotation_text=f"Target {C.TARGET_RECALL:.0%}", annotation_position="top right",
                      annotation_font_color="rgba(255,255,255,0.55)")
        fig.add_trace(go.Scatter(
            x=[op_tpr], y=[op_precision], mode="markers",
            marker=dict(size=16, color=PALETTE["accent"], line=dict(color="white", width=2), symbol="circle"),
            name=f"Operating point (t={live_threshold:.2f})",
            hovertemplate=f"Threshold {live_threshold:.2f}<br>Recall=%{{x:.2f}}<br>Precision=%{{y:.2f}}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_THEME, height=380, title="Precision-Recall",
                          xaxis_title="Recall", yaxis_title="Precision",
                          legend=dict(orientation="h", yanchor="bottom", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-h">Which measurements matter most?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "We tested each measurement by randomly scrambling its values and seeing how much the model's skill dropped. "
        "<strong>Longer bar = the model would be in worse trouble without it</strong>. "
        "This is how you'd explain to a sceptical clinician why the model thinks what it thinks."
        '</div>',
        unsafe_allow_html=True,
    )
    imp = get_importance()
    norm = (imp["importance"] - imp["importance"].min()) / max(1e-9, (imp["importance"].max() - imp["importance"].min()))
    colors = [f"rgba({int(244-(244-34)*v)},{int(114+(211-114)*v)},{int(182+(238-182)*v)},0.85)" for v in norm]
    fig = go.Figure(go.Bar(
        x=imp["importance"], y=imp["feature"], orientation="h",
        error_x=dict(type="data", array=imp["std"], color="rgba(255,255,255,0.30)", thickness=1.2),
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
        hovertemplate="<b>%{y}</b><br>Δ ROC-AUC = %{x:.3f}<extra></extra>",
        text=[f"{v:.3f}" for v in imp["importance"]],
        textposition="outside",
        textfont=dict(color="rgba(230,233,245,0.7)"),
    ))
    fig.update_layout(**PLOTLY_THEME, height=420, xaxis_title="Δ ROC-AUC when shuffled")
    st.plotly_chart(fig, use_container_width=True)


# ═══ TAB 4 — Try the model ═══════════════════════════════════════════════════

with t4:
    st.markdown('<div class="section-h">Try it on a made-up patient</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Move the sliders to invent a patient — high glucose, older age, higher BMI usually push the risk up. "
        "The gauge below updates instantly as you move things. "
        "<strong>Red zone</strong> = the model would flag them. <strong>Green zone</strong> = it would clear them."
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose · 2-h OGTT (mg/dL)", 40, 250, 148)
        blood_pressure = st.slider("Diastolic BP (mm Hg)", 30, 130, 72)
        skin_thickness = st.slider("Triceps skin fold (mm)", 5, 60, 35)
    with c2:
        insulin = st.slider("2-h serum insulin (µU/mL)", 0, 900, 168)
        bmi = st.slider("BMI", 10.0, 60.0, 33.6, step=0.1)
        pedigree = st.slider("Diabetes pedigree", 0.05, 2.50, 0.627, step=0.01)
        age = st.slider("Age (years)", 18, 90, 50)

    payload = {
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree, "Age": age,
    }

    result = predict_one(payload)
    p = result["probability"]
    flagged = result["label"] == 1

    gauge_color = PALETTE["warn"] if flagged else PALETTE["ok"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number=dict(suffix="%", font=dict(size=46, color="#e6e9f5")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="rgba(255,255,255,0.40)"),
            bar=dict(color=gauge_color, thickness=0.28),
            bgcolor="rgba(255,255,255,0.04)",
            borderwidth=0,
            steps=[
                {"range": [0, THRESHOLD * 100], "color": "rgba(34,197,94,0.18)"},
                {"range": [THRESHOLD * 100, 100], "color": "rgba(244,63,94,0.18)"},
            ],
            threshold=dict(line=dict(color=PALETTE["accent"], width=3), thickness=0.78, value=THRESHOLD * 100),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(**{**PLOTLY_THEME, "margin": dict(l=20, r=20, t=30, b=10)}, height=320)

    gcol, rcol = st.columns([1.1, 1])
    with gcol:
        st.plotly_chart(fig, use_container_width=True)
    with rcol:
        if flagged:
            st.markdown(
                f"""
<div class="result-flag">
  <b style="font-size:1.1rem">⚠️ Flagged for follow-up</b><br/>
  Estimated probability <b>{p:.1%}</b> — above the screening threshold of {THRESHOLD:.2f}.<br/>
  <span style="opacity:0.85">Recommendation: confirmatory HbA1c.</span>
</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="result-clear">
  <b style="font-size:1.1rem">✅ Below screening threshold</b><br/>
  Estimated probability <b>{p:.1%}</b>. No follow-up indicated by this model.<br/>
  <span style="opacity:0.85">Continue routine monitoring.</span>
</div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("&nbsp;")
        st.markdown(
            f"<div style='opacity:0.7;font-size:0.9rem'>Model: <code>{artifact['model_name']}</code><br/>"
            f"Threshold: <code>{THRESHOLD:.3f}</code></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-h">How this patient compares to everyone else</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "The <strong>amber dashed line</strong> is your made-up patient. "
        "The cyan and pink shapes show how all the real diabetic and non-diabetic patients in the study are spread out for the chosen measurement. "
        "If the dashed line falls in the pink cloud, your patient looks similar to the diabetics."
        '</div>',
        unsafe_allow_html=True,
    )
    cmp_feat = st.selectbox("Which measurement to compare", C.FEATURES, index=C.FEATURES.index("Glucose"), key="cmp_feat")
    df = get_data()
    sub = df.dropna(subset=[cmp_feat]).copy()
    sub["Outcome"] = sub[C.TARGET].map({0: "No diabetes", 1: "Diabetes"})
    fig = px.histogram(
        sub, x=cmp_feat, color="Outcome",
        marginal="violin", barmode="overlay", opacity=0.55, nbins=30,
        color_discrete_map={"No diabetes": PALETTE["neg"], "Diabetes": PALETTE["pos"]},
    )
    fig.add_vline(x=payload[cmp_feat], line=dict(color=PALETTE["accent"], dash="dash", width=3),
                  annotation_text="this patient", annotation_position="top",
                  annotation_font_color=PALETTE["accent"])
    fig.update_layout(**PLOTLY_THEME, height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Input summary"):
        st.dataframe(pd.DataFrame([payload]).T.rename(columns={0: "value"}), use_container_width=True)
