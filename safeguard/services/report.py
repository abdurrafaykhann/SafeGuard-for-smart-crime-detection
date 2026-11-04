"""One-page evidence report (PDF) for an alert: what fired and why, the snapshot, who handled it and when.

The snapshot's SHA-256 is printed so a printed report can later be checked against the stored file; the report
says plainly that detections are automated and need human review."""
from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone

from fpdf import FPDF

from ..models import Alert, AuditLog, iso


def _latin(text: str) -> str:
    """The built-in PDF fonts are Latin-1 only; replace anything else instead of failing."""
    return (text or "").encode("latin-1", "replace").decode("latin-1")


def build_alert_report(alert: Alert, snapshot_path, generated_by: str, audit_rows: list[AuditLog]) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "SafeGuard - Incident evidence report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(110, 110, 110)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 5, _latin(f"Alert #{alert.id}  |  generated {now} by {generated_by}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    def row(label: str, value: str) -> None:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(42, 6, _latin(label))
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _latin(value), new_x="LMARGIN", new_y="NEXT")

    row("Event", alert.title)
    row("Type / severity", f"{alert.kind} / {alert.severity}")
    row("Detected at", (iso(alert.created_at) or "").replace("T", " ").replace("Z", " UTC"))
    row("Camera", alert.camera_name or "n/a")
    row("Model confidence", f"{(alert.confidence or 0):.0%}")
    if alert.watchlist:
        row("Watchlist match", f"{alert.watchlist.label} ({alert.watchlist.risk} risk)")
    row("Evidence video", "available in the console" if alert.clip_path else "none recorded")
    if alert.acknowledged_at:
        row("Acknowledged", f"by {alert.acknowledged_by} at {iso(alert.acknowledged_at).replace('T', ' ').replace('Z', ' UTC')}")
    else:
        row("Acknowledged", "not yet acknowledged")

    cues = alert.meta_dict().get("cues") or {}
    if cues:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Why this fired", new_x="LMARGIN", new_y="NEXT")
        for name, value in cues.items():
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(32, 5, _latin(str(name)))
            y, x = pdf.get_y(), pdf.get_x()
            pdf.set_fill_color(225, 228, 232)
            pdf.rect(x, y + 1, 90, 3, style="F")
            pdf.set_fill_color(210, 60, 70)
            pdf.rect(x, y + 1, 90 * max(0.0, min(float(value), 1.0)), 3, style="F")
            pdf.set_x(x + 94)
            pdf.cell(0, 5, f"{float(value):.0%}", new_x="LMARGIN", new_y="NEXT")

    digest = None
    if snapshot_path is not None and snapshot_path.is_file():
        data = snapshot_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Snapshot at the moment of the alert", new_x="LMARGIN", new_y="NEXT")
        pdf.image(io.BytesIO(data), w=120)
        pdf.set_font("Courier", "", 7)
        pdf.cell(0, 4, f"SHA-256 {digest}", new_x="LMARGIN", new_y="NEXT")

    if audit_rows:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Handling log", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for a in audit_rows:
            pdf.cell(0, 5, _latin(f"{iso(a.created_at).replace('T', ' ').replace('Z', ' UTC')}  {a.username or '-'}  {a.action}"),
                     new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 4, "Detections are produced by automated models and can be wrong. This report records what the system "
                         "flagged; it is not proof that an incident occurred and must be reviewed by a person.")
    return bytes(pdf.output())
