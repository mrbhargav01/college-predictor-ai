"""
dashboard/components/ui_components.py
All reusable UI component builders for EduPath.
"""
from __future__ import annotations
import streamlit as st
from datetime import datetime


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cls_badge(cls: str) -> str:
    c = cls.lower()
    icons = {"safe": "✅", "moderate": "⚠️", "dream": "🚀"}
    return f'<span class="badge badge-{c}">{icons.get(c,"")}&nbsp;{cls}</span>'


def _prob_bar(prob: float, label: str = "Admission Probability") -> str:
    pct = prob * 100
    color = "#10b981" if pct >= 66 else "#f59e0b" if pct >= 33 else "#f43f5e"
    fill_style = f"width:{pct:.1f}%;background:{color};"
    return f"""
    <div class="prob-wrap">
      <div class="prob-header">
        <span>{label}</span>
        <span style="color:{color};font-family:'JetBrains Mono',monospace;">{pct:.1f}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="{fill_style}"></div>
      </div>
    </div>"""


# ── Hero Banner ─────────────────────────────────────────────────────────────────

def render_hero(title: str, subtitle: str, chips: list[str] = None) -> None:
    chips_html = ""
    if chips:
        chips_html = '<div class="hero-chips">' + \
            "".join(f'<span class="hero-chip">{c}</span>' for c in chips) + \
            '</div>'
    st.markdown(f"""
    <div class="hero-wrap fade-in">
      <div class="hero-title">{title}</div>
      <div class="hero-sub">{subtitle}</div>
      {chips_html}
    </div>""", unsafe_allow_html=True)


# ── KPI Row ──────────────────────────────────────────────────────────────────────

def render_kpi_row(metrics: list[dict]) -> None:
    """
    metrics: [{"label": "Total Colleges", "value": "2500+", "icon": "🏛️"}, ...]
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="kpi-card fade-in">
              <div style="font-size:2rem;margin-bottom:.4rem;">{m.get('icon','')}</div>
              <div class="kpi-num">{m['value']}</div>
              <div class="kpi-label">{m['label']}</div>
            </div>""", unsafe_allow_html=True)


# ── Section Header ───────────────────────────────────────────────────────────────

def section_header(icon: str, title: str) -> None:
    st.markdown(f'<div class="sec-hdr">{icon}&nbsp;{title}</div>', unsafe_allow_html=True)


# ── College Card ─────────────────────────────────────────────────────────────────

def render_college_card(row, rank: int, show_details_btn: bool = True) -> bool:
    """Render a single college card. Returns True if 'View Details' was clicked."""
    cls_lower = str(row.get("College_Classification", "")).lower()
    prob = float(row.get("Probability_of_Admission", 0))
    col_name = str(row.get("College_Name", "Unknown College"))
    branch = str(row.get("Branch", "—"))
    state = str(row.get("State", "—"))
    city = str(row.get("City", "—"))
    exam = str(row.get("Entrance_Exam", "—"))
    cutoff = float(row.get("Historical_College_Cutoff_Marks", 0))
    prev_rank = int(row.get("Previous_Year_Admission_Rank", 0))

    st.markdown(f"""
    <div class="college-card {cls_lower} fade-in">
      <div class="rank-badge">#{rank}</div>
      <div class="college-name">{col_name}</div>
      <div class="chip-row">
        {_cls_badge(row.get('College_Classification',''))}
        <span class="badge badge-info">📍 {city}, {state}</span>
        <span class="badge badge-neutral">🎓 {branch}</span>
        <span class="badge badge-neutral">📝 {exam}</span>
      </div>
      {_prob_bar(prob)}
      <div class="college-meta">
        <div class="college-meta-item">✂️&nbsp;<strong>Cutoff:</strong>&nbsp;{cutoff:.2f}</div>
        <div class="college-meta-item">🏆&nbsp;<strong>Prev Rank:</strong>&nbsp;{prev_rank:,}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    clicked = False
    if show_details_btn:
        _, btn_col = st.columns([4, 1])
        with btn_col:
            if st.button("View Details →", key=f"card_view_{rank}_{col_name[:20]}"):
                clicked = True
    return clicked


# ── Progress Stepper ──────────────────────────────────────────────────────────────

def render_stepper(steps: list[str], current: int) -> None:
    items = []
    for i, s in enumerate(steps):
        state = "done" if i < current else "active" if i == current else ""
        dot_content = "✓" if i < current else str(i + 1)
        items.append(f"""
        <div class="step-item {state}">
          <div class="step-dot">{dot_content}</div>
          <div class="step-label">{s}</div>
        </div>""")
    st.markdown(f'<div class="step-track">{"".join(items)}</div>', unsafe_allow_html=True)


# ── Chat Message ─────────────────────────────────────────────────────────────────
# Replace the render_chat_messages function completely:

def render_chat_messages(history: list[dict]) -> None:
    """Render chat messages properly - NO raw code display"""
    if not history:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:var(--text-muted);">
            💬 Ask me anything about college admissions!
        </div>
        """, unsafe_allow_html=True)
        return
    
    for msg in history:
        role = msg.get("role", "bot")
        content = msg.get("content", "")
        ts = msg.get("ts", "")
        
        if role == "user":
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:1rem;">
                <div style="background:linear-gradient(135deg,#6C63FF,#9D50BB);
                            color:white;padding:0.8rem 1.2rem;
                            border-radius:18px 18px 4px 18px;
                            max-width:75%;word-wrap:break-word;">
                    {content}
                    <div style="font-size:0.7rem;opacity:0.7;margin-top:0.3rem;">{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-start;margin-bottom:1rem;">
                <div style="background:rgba(108,99,255,0.1);
                            color:var(--text-b);padding:0.8rem 1.2rem;
                            border-radius:18px 18px 18px 4px;
                            max-width:75%;word-wrap:break-word;
                            border:1px solid rgba(108,99,255,0.2);">
                    {content}
                    <div style="font-size:0.7rem;opacity:0.6;margin-top:0.3rem;">{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Alert Boxes ───────────────────────────────────────────────────────────────────

def alert(kind: str, text: str) -> None:
    """kind: info | success | warn"""
    icons = {"info": "ℹ️", "success": "✅", "warn": "⚠️"}
    st.markdown(f'<div class="alert-{kind}">{icons.get(kind,"")} {text}</div>',
                unsafe_allow_html=True)


# ── College Detail Header ─────────────────────────────────────────────────────────

def render_detail_hero(row) -> None:
    cls = str(row.get("College_Classification", ""))
    prob = float(row.get("Probability_of_Admission", 0))
    col_name = str(row.get("College_Name", ""))
    state = str(row.get("State", ""))
    city = str(row.get("City", ""))
    branch = str(row.get("Branch", ""))

    st.markdown(f"""
    <div class="detail-hero fade-in">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
        <div>
          <div style="font-size:0.85rem;opacity:0.75;margin-bottom:.4rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;">College Profile</div>
          <h1 style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;margin:0 0 .6rem;line-height:1.2;">{col_name}</h1>
          <div style="display:flex;flex-wrap:wrap;gap:.8rem;font-size:.9rem;opacity:.9;">
            <span>📍 {city}, {state}</span>
            <span>🎓 {branch}</span>
            <span>{_cls_badge(cls)}</span>
          </div>
        </div>
        <div style="text-align:center;background:rgba(255,255,255,0.15);backdrop-filter:blur(8px);border-radius:16px;padding:1.2rem 2rem;">
          <div style="font-size:3rem;font-weight:900;font-family:'Syne',sans-serif;">{prob*100:.0f}%</div>
          <div style="font-size:.8rem;opacity:.8;">Admission Probability</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ── Stat Grid ─────────────────────────────────────────────────────────────────────

def render_stat_grid(stats: list[dict]) -> None:
    """stats: [{"label":..., "value":..., "icon":...}]"""
    items = "".join(f"""
    <div class="detail-stat">
      <div style="font-size:1.5rem;margin-bottom:.3rem;">{s.get('icon','')}</div>
      <h4>{s['value']}</h4>
      <p>{s['label']}</p>
    </div>""" for s in stats)
    st.markdown(f'<div class="detail-stat-grid">{items}</div>', unsafe_allow_html=True)


# ── Report Download Banner ────────────────────────────────────────────────────────

def render_report_banner() -> None:
    st.markdown("""
    <div class="report-cta fade-in">
      <div style="font-size:2.5rem;margin-bottom:.5rem;">📄</div>
      <h3>Download Your Prediction Report</h3>
      <p style="color:var(--text-muted);margin-bottom:1.2rem;">
        Get a comprehensive PDF report with all college predictions, probabilities, and analytics.
      </p>
    </div>""", unsafe_allow_html=True)


# ── Theme Toggle ──────────────────────────────────────────────────────────────────

def render_theme_toggle() -> None:
    """Injects a floating dark/light mode toggle button."""
    theme = st.session_state.get("theme", "light")
    icon = "🌙" if theme == "light" else "☀️"
    label = "Dark Mode" if theme == "light" else "Light Mode"
    st.markdown(f"""
    <div class="dm-toggle" onclick="toggleTheme()">
      {icon} {label}
    </div>
    <script>
    function toggleTheme() {{
      // Streamlit JS bridge not fully available; visual toggle via body class
      var body = window.parent.document.querySelector('.main');
      body && body.classList.toggle('dark-mode');
    }}
    </script>""", unsafe_allow_html=True)
