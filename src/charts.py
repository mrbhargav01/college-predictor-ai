"""
src/charts.py
All Plotly chart factory functions for EduPath Analytics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared colour palette
BRAND   = "#6C63FF"
BRAND2  = "#9D50BB"
BRAND3  = "#4ECDC4"
SAFE_C  = "#10b981"
MOD_C   = "#f59e0b"
DREAM_C = "#f43f5e"
CLS_MAP = {"Safe": SAFE_C, "Moderate": MOD_C, "Dream": DREAM_C}

PLOTLY_LAYOUT = dict(
    font=dict(family="DM Sans, sans-serif", color="#2d2a4a"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=50, b=30),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def _apply(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, showline=True,
                     linecolor="rgba(108,99,255,0.15)")
    fig.update_yaxes(showgrid=True,
                     gridcolor="rgba(108,99,255,0.08)",
                     showline=False)
    return fig


# ── 1. Classification Donut ───────────────────────────────────────────────────

def chart_classification_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["College_Classification"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.6,
        marker=dict(colors=[CLS_MAP.get(k, BRAND) for k in counts.index]),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} colleges<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=360,
                      title="College Classification Mix",
                      annotations=[dict(text=f"<b>{len(df)}</b><br>Records",
                                        x=0.5, y=0.5, showarrow=False,
                                        font=dict(size=16, color="#2d2a4a"))])
    return fig


# ── 2. State Bar ──────────────────────────────────────────────────────────────

def chart_state_distribution(df: pd.DataFrame) -> go.Figure:
    sc = df["State"].value_counts().reset_index()
    sc.columns = ["State", "Count"]
    colors = [BRAND2 if s == "Maharashtra" else BRAND for s in sc["State"]]
    fig = go.Figure(go.Bar(
        x=sc["State"], y=sc["Count"],
        marker_color=colors,
        hovertemplate="%{x}: %{y} records<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                      title="State-wise College Distribution<br><sup>Maharashtra highlighted</sup>",
                      xaxis_tickangle=-30)
    return _apply(fig, 380)


# ── 3. Exam Cutoff Bar ────────────────────────────────────────────────────────

def chart_exam_cutoff(df: pd.DataFrame) -> go.Figure:
    stats = df.groupby("Entrance_Exam")["Historical_College_Cutoff_Marks"].mean()\
               .sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=stats.index, y=stats.values.round(2),
        marker=dict(
            color=stats.values,
            colorscale=[[0, BRAND], [0.5, BRAND2], [1, BRAND3]],
            showscale=False,
        ),
        text=stats.values.round(1),
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    return _apply(fig.update_layout(title="Avg Cutoff Marks by Exam"), 380)


# ── 4. Category Probability ───────────────────────────────────────────────────

def chart_category_probability(df: pd.DataFrame) -> go.Figure:
    stats = (df.groupby("Category")["Probability_of_Admission"].mean() * 100)\
               .sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        y=stats.index, x=stats.values.round(1),
        orientation="h",
        marker_color=BRAND3,
        text=stats.values.round(1),
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    return _apply(fig.update_layout(title="Avg Admission Probability by Category (%)"), 360)


# ── 5. Probability Distribution ───────────────────────────────────────────────

def chart_probability_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df["Probability_of_Admission"] * 100,
        nbinsx=25,
        marker=dict(color=BRAND, opacity=0.8,
                    line=dict(color="white", width=0.5)),
        hovertemplate="Range %{x}: %{y} colleges<extra></extra>",
    ))
    return _apply(fig.update_layout(
        title="Distribution of Admission Probability",
        xaxis_title="Probability (%)", yaxis_title="Count",
    ), 360)


# ── 6. Marks vs Probability Scatter ──────────────────────────────────────────

def chart_marks_vs_prob(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(600, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Student_Marks_Percentage",
        y="Probability_of_Admission",
        color="College_Classification",
        color_discrete_map=CLS_MAP,
        opacity=0.65,
        hover_data=["College_Name", "State"],
        title="Student Marks vs Admission Probability",
        labels={
            "Student_Marks_Percentage": "Student Marks (%)",
            "Probability_of_Admission": "Probability",
        },
    )
    fig.update_traces(marker=dict(size=7))
    return _apply(fig, 420)


# ── 7. Course Analysis ────────────────────────────────────────────────────────

def chart_course_stacked(df: pd.DataFrame) -> go.Figure:
    pivot = df.groupby(["Course_Preference", "College_Classification"])\
              .size().unstack(fill_value=0)
    fig = go.Figure()
    for cls in ["Safe", "Moderate", "Dream"]:
        if cls in pivot.columns:
            fig.add_trace(go.Bar(
                name=cls, x=pivot.index, y=pivot[cls],
                marker_color=CLS_MAP[cls],
                hovertemplate=f"{cls}: %{{y}}<extra></extra>",
            ))
    fig.update_layout(barmode="stack", xaxis_tickangle=-30,
                      title="College Count by Course & Classification")
    return _apply(fig, 420)


# ── 8. Correlation Heatmap ───────────────────────────────────────────────────

def chart_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num = ["Student_Marks_Percentage", "Entrance_Exam_Score",
           "Historical_College_Cutoff_Marks",
           "Previous_Year_Admission_Rank", "Probability_of_Admission"]
    corr = df[num].corr().round(2)
    labels = ["Marks %", "Exam Score", "Cutoff", "Prev Rank", "Prob"]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale=[[0, "#fee2e2"], [0.5, "white"], [1, "#dbeafe"]],
        text=corr.values, texttemplate="%{text}",
        zmid=0,
        hovertemplate="%{y} × %{x}: %{z}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=400,
                      title="Feature Correlation Heatmap")
    return fig


# ── 9. City Sunburst ──────────────────────────────────────────────────────────

def chart_state_city_sunburst(df: pd.DataFrame) -> go.Figure:
    sub = df[df["City"].notna()].copy()
    sub["All"] = "India"
    fig = px.sunburst(
        sub, path=["All", "State", "City"],
        values=sub.groupby(["State", "City"]).transform("size"),
        color="State",
        title="State → City College Distribution",
    )
    fig.update_traces(textinfo="label+percent parent")
    fig.update_layout(**PLOTLY_LAYOUT, height=520)
    return fig


# ── 10. Gauge ─────────────────────────────────────────────────────────────────

def chart_probability_gauge(prob: float, title: str = "Admission Probability") -> go.Figure:
    pct = prob * 100
    color = SAFE_C if pct >= 66 else MOD_C if pct >= 33 else DREAM_C
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        delta={"reference": 50, "valueformat": ".1f"},
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 15, "family": "Syne"}},
        number={"suffix": "%", "font": {"size": 36, "color": color,
                                         "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%",
                     "tickcolor": "#6b6890"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33],  "color": "rgba(244,63,94,0.12)"},
                {"range": [33, 66], "color": "rgba(245,158,11,0.12)"},
                {"range": [66, 100],"color": "rgba(16,185,129,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.8, "value": pct},
        },
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260)
    return fig


# ── 11. Trend Line (simulated multi-year) ─────────────────────────────────────

def chart_cutoff_trend(df: pd.DataFrame, college_name: str) -> go.Figure:
    base_cutoff = float(df[df["College_Name"] == college_name]
                        ["Historical_College_Cutoff_Marks"].mean())
    years = list(range(2020, 2026))
    rng = np.random.default_rng(hash(college_name) % 2**32)
    cutoffs = [round(base_cutoff + rng.uniform(-4, 4), 2) for _ in years]
    cutoffs[-1] = base_cutoff

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=cutoffs, mode="lines+markers+text",
        line=dict(color=BRAND, width=3),
        marker=dict(size=9, color=BRAND,
                    line=dict(color="white", width=2)),
        text=[f"{v:.1f}" for v in cutoffs],
        textposition="top center",
        name="Cutoff",
    ))
    fig.add_hline(y=base_cutoff, line_dash="dot",
                   line_color=BRAND3, annotation_text="Latest Cutoff")
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      title="Historical Cutoff Trend (2020–2025)",
                      xaxis_title="Year", yaxis_title="Cutoff Marks")
    return _apply(fig, 320)


# ── 12. Package Bar ───────────────────────────────────────────────────────────

def chart_placement_by_course(df: pd.DataFrame) -> go.Figure:
    stats = df.groupby("Course_Preference")["Avg_Package_LPA"].mean()\
               .sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=stats.index, y=stats.values.round(2),
        marker=dict(
            color=stats.values,
            colorscale=[[0, BRAND3], [1, BRAND]],
        ),
        text=["₹" + str(round(v, 1)) + "L" for v in stats.values],
        textposition="outside",
        hovertemplate="%{x}: ₹%{y:.2f} LPA<extra></extra>",
    ))
    return _apply(fig.update_layout(
        title="Avg Placement Package by Course (LPA)",
        xaxis_tickangle=-25,
    ), 380)
