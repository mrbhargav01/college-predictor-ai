"""
src/report_gen.py
Generate downloadable PDF / HTML prediction reports for EduPath.
"""
from __future__ import annotations
import io
import base64
from datetime import datetime
from typing import Optional
import pandas as pd


# ── HTML Report ────────────────────────────────────────────────────────────────

def generate_html_report(user_data: dict, results: pd.DataFrame,
                          top_n: int = 20) -> str:
    """Generate a styled HTML report and return as string."""
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    name = user_data.get("student_name", "Student")
    category = user_data.get("category", "—")
    exam = user_data.get("exam_type", "—")
    score = user_data.get("exam_score", "—")
    pct = user_data.get("overall_percentage", "—")
    course = user_data.get("course_preference", "—")

    safe_cnt = int((results["College_Classification"] == "Safe").sum())
    mod_cnt  = int((results["College_Classification"] == "Moderate").sum())
    drm_cnt  = int((results["College_Classification"] == "Dream").sum())

    def cls_badge(cls: str) -> str:
        colors = {"Safe": "#10b981", "Moderate": "#f59e0b", "Dream": "#f43f5e"}
        c = colors.get(cls, "#6C63FF")
        return (f'<span style="background:{c}22;color:{c};border:1px solid {c}66;'
                f'padding:3px 12px;border-radius:20px;font-size:12px;font-weight:700;">'
                f'{cls}</span>')

    rows_html = ""
    for i, row in results.head(top_n).iterrows():
        prob = float(row.get("Probability_of_Admission", 0)) * 100
        bar_color = "#10b981" if prob >= 66 else "#f59e0b" if prob >= 33 else "#f43f5e"
        rows_html += f"""
        <tr>
          <td style="font-weight:700;color:#1e1b4b;">{i+1}</td>
          <td style="font-weight:600;">{row.get('College_Name','')}</td>
          <td>{row.get('State','')}</td>
          <td>{row.get('Branch','')}</td>
          <td style="font-family:monospace;">{prob:.1f}%
            <div style="height:6px;background:#eee;border-radius:3px;margin-top:4px;">
              <div style="height:6px;width:{prob:.0f}%;background:{bar_color};border-radius:3px;"></div>
            </div>
          </td>
          <td>{cls_badge(row.get('College_Classification',''))}</td>
          <td style="font-family:monospace;">{float(row.get('Historical_College_Cutoff_Marks',0)):.2f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EduPath Prediction Report – {name}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'DM Sans',sans-serif; background:#f5f5ff; color:#2d2a4a; }}
  .page {{ max-width:960px; margin:2rem auto; background:white;
           border-radius:20px; overflow:hidden; box-shadow:0 8px 40px rgba(108,99,255,.15); }}

  /* Header */
  .header {{ background:linear-gradient(135deg,#6C63FF,#9D50BB,#4ECDC4);
             padding:3rem; color:white; position:relative; overflow:hidden; }}
  .header::after {{ content:'🎓'; position:absolute; right:2rem; top:50%;
                    transform:translateY(-50%); font-size:6rem; opacity:.12; }}
  .header h1 {{ font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800;
                margin-bottom:.3rem; }}
  .header p  {{ opacity:.88; font-size:1rem; }}
  .header .meta {{ display:flex; gap:2rem; margin-top:1.2rem; font-size:.85rem; }}
  .header .meta span {{ background:rgba(255,255,255,.18); padding:.3rem .9rem;
                        border-radius:20px; }}

  /* Summary */
  .summary {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.5rem;
              padding:2rem; background:#f8f8ff; }}
  .stat {{ text-align:center; background:white; border-radius:14px;
           padding:1.5rem; border-top:4px solid; }}
  .stat.safe    {{ border-top-color:#10b981; }}
  .stat.moderate{{ border-top-color:#f59e0b; }}
  .stat.dream   {{ border-top-color:#f43f5e; }}
  .stat h2 {{ font-family:'Syne',sans-serif; font-size:2.5rem; font-weight:800; }}
  .stat.safe    h2 {{ color:#10b981; }}
  .stat.moderate h2{{ color:#f59e0b; }}
  .stat.dream   h2 {{ color:#f43f5e; }}
  .stat p {{ font-size:.85rem; color:#6b6890; margin-top:.3rem; }}

  /* Profile */
  .profile {{ padding:2rem; }}
  .profile h2 {{ font-family:'Syne',sans-serif; font-size:1.4rem;
                 font-weight:700; margin-bottom:1rem; color:#1e1b4b; }}
  .p-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; }}
  .p-item {{ background:#f8f8ff; border-radius:10px; padding:1rem; }}
  .p-item span {{ font-size:.75rem; color:#6b6890; display:block;
                  margin-bottom:.2rem; text-transform:uppercase; letter-spacing:.5px; }}
  .p-item strong {{ font-size:1rem; color:#1e1b4b; }}

  /* Table */
  .tbl-wrap {{ padding:2rem; }}
  .tbl-wrap h2 {{ font-family:'Syne',sans-serif; font-size:1.4rem;
                  font-weight:700; margin-bottom:1rem; color:#1e1b4b; }}
  table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  th {{ background:linear-gradient(90deg,#6C63FF,#9D50BB); color:white;
        padding:.8rem 1rem; text-align:left; font-weight:600; }}
  th:first-child {{ border-radius:10px 0 0 0; }}
  th:last-child  {{ border-radius:0 10px 0 0; }}
  td {{ padding:.75rem 1rem; border-bottom:1px solid #f0eeff; vertical-align:middle; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:nth-child(even) td {{ background:#fafaff; }}

  /* Footer */
  .footer {{ text-align:center; padding:2rem;
             background:linear-gradient(135deg,#6C63FF,#9D50BB);
             color:rgba(255,255,255,.8); font-size:.85rem; }}
  .footer strong {{ color:white; }}
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <div class="header">
    <h1>EduPath — College Prediction Report</h1>
    <p>AI-powered college admission predictions for {name}</p>
    <div class="meta">
      <span>📅 {now}</span>
      <span>🎓 {course}</span>
      <span>📝 {exam} · Score {score}</span>
      <span>📊 Overall {pct}%</span>
    </div>
  </div>

  <!-- Summary -->
  <div class="summary">
    <div class="stat safe">
      <h2>{safe_cnt}</h2>
      <p>✅ Safe Colleges</p>
    </div>
    <div class="stat moderate">
      <h2>{mod_cnt}</h2>
      <p>⚠️ Moderate Colleges</p>
    </div>
    <div class="stat dream">
      <h2>{drm_cnt}</h2>
      <p>🚀 Dream Colleges</p>
    </div>
  </div>

  <!-- Student Profile -->
  <div class="profile">
    <h2>👤 Student Profile</h2>
    <div class="p-grid">
      <div class="p-item"><span>Student Name</span><strong>{name}</strong></div>
      <div class="p-item"><span>Category</span><strong>{category}</strong></div>
      <div class="p-item"><span>Entrance Exam</span><strong>{exam}</strong></div>
      <div class="p-item"><span>Exam Score</span><strong>{score}</strong></div>
      <div class="p-item"><span>Overall %</span><strong>{pct}%</strong></div>
      <div class="p-item"><span>Course</span><strong>{course}</strong></div>
    </div>
  </div>

  <!-- College Table -->
  <div class="tbl-wrap">
    <h2>🏛️ Top Predicted Colleges (Top {min(top_n, len(results))})</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>College</th><th>State</th>
          <th>Branch</th><th>Probability</th>
          <th>Classification</th><th>Cutoff</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <!-- Footer -->
  <div class="footer">
    Generated by <strong>EduPath — College Prediction System</strong> |
    Predictions are based on historical data and may vary from actual results.
  </div>

</div>
</body>
</html>"""
    return html


def get_html_download_link(html: str, filename: str = "EduPath_Report.html") -> str:
    """Return a base-64 encoded href for HTML download."""
    b64 = base64.b64encode(html.encode()).decode()
    return (f'<a href="data:text/html;base64,{b64}" download="{filename}" '
            f'style="text-decoration:none;">'
            f'<button style="background:linear-gradient(135deg,#6C63FF,#9D50BB);'
            f'color:white;border:none;border-radius:10px;padding:.8rem 2rem;'
            f'font-size:.95rem;font-weight:600;cursor:pointer;'
            f'box-shadow:0 4px 15px rgba(108,99,255,.4);">'
            f'📥 Download HTML Report</button></a>')


def get_csv_download(results: pd.DataFrame) -> bytes:
    """Return CSV bytes for download."""
    cols = ["College_Name", "State", "City", "Branch",
            "College_Classification", "Probability_of_Admission",
            "Historical_College_Cutoff_Marks", "Previous_Year_Admission_Rank",
            "Entrance_Exam", "Category"]
    available = [c for c in cols if c in results.columns]
    buf = io.StringIO()
    results[available].to_csv(buf, index=False)
    return buf.getvalue().encode()
