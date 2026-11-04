"""Civilian lookup: has anyone filed a complaint with law enforcement against this business / site / agency?

Matching is deliberately conservative about what it calls a hit:
  * exact   - same normalised name, or same domain, or a domain whose main label equals the name
  * similar - a near-identical spelling (difflib >= 0.86) of a name that has complaints; shown as "check the
              spelling", never as a confirmed hit
A verdict of "no_record" means only that nothing matched; it is not a statement that the entity is safe."""
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

from ..models import Complaint

_URLISH = re.compile(r"^(?:[a-z][a-z0-9+.-]*://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)(?::\d+)?(?:[/?#].*)?$")
_SUFFIXES = {"ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co", "plc", "pvt", "private", "pty", "gmbh"}
SIMILAR_AT = 0.86
STATUS_RANK = {"confirmed": 3, "under_investigation": 2, "reported": 1, "dismissed": 0}


def name_key(text: str) -> str:
    """Lower-case, strip punctuation and legal suffixes (Ltd, LLC ...), collapse spaces."""
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    kept = [w for w in words if w not in _SUFFIXES]
    return " ".join(kept or words)


def parse_query(raw: str) -> dict:
    raw = (raw or "").strip()[:200]
    m = _URLISH.match(raw.lower())
    domain = m.group(1) if m else None
    label = None
    if domain:
        parts = domain.split(".")
        label = name_key(parts[-2].replace("-", " ")) if len(parts) >= 2 else None
    return {"raw": raw, "domain": domain, "label": label, "key": (label if domain and label else name_key(raw)) if raw else ""}


def _summarise(rows: list[Complaint]) -> dict:
    active = [c for c in rows if c.status in Complaint.ACTIVE]
    top = max(rows, key=lambda c: STATUS_RANK.get(c.status, 0))
    dates = sorted(c.filed_at for c in rows if c.filed_at)
    return {
        "name": rows[0].entity_name, "domain": rows[0].entity_domain, "type": rows[0].entity_type,
        "complaints": len(rows), "active": len(active), "highest_status": top.status,
        "categories": sorted({c.category for c in rows}),
        "first_reported": dates[0].date().isoformat() if dates else None,
        "last_reported": dates[-1].date().isoformat() if dates else None,
        # short, anonymous descriptions only (no complainant data is stored)
        "examples": list(dict.fromkeys(c.summary for c in sorted(active, key=lambda c: c.filed_at or 0, reverse=True) if c.summary))[:3],
        "references": [c.reference for c in rows][:5],
    }


def lookup(raw: str) -> dict:
    q = parse_query(raw)
    if len(q["key"]) < 2:
        return {"query": q["raw"], "valid": False}
    groups: dict[str, list[Complaint]] = defaultdict(list)
    for c in Complaint.query.all():
        groups[c.name_key].append(c)

    exact, similar = [], []
    for key, rows in groups.items():
        domains = {c.entity_domain for c in rows if c.entity_domain}
        is_exact = (key == q["key"] or (q["domain"] and q["domain"] in domains) or (q["label"] and key == q["label"]))
        if is_exact:
            exact.append(_summarise(rows))
        elif SequenceMatcher(None, key, q["key"]).ratio() >= SIMILAR_AT:
            similar.append(_summarise(rows))

    if any(m["active"] for m in exact):
        verdict = "unsafe"
    elif exact or any(m["active"] for m in similar):
        verdict = "caution"
    else:
        verdict = "no_record"
    exact.sort(key=lambda m: -m["active"])
    return {"query": q["raw"], "valid": True, "verdict": verdict, "matches": exact, "similar": similar[:3]}
