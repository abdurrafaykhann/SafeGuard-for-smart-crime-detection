"""Evaluate the public fraud check (services/complaints.py) like a classifier: recall, false positives, imbalance.

    python scripts/evaluate_fraud_check.py [--negatives 5000] [--out docs/evaluation_fraud_check.json]

The fraud check is a name/domain matcher over a complaints register, not a learned model, so "accuracy" means: does a person
typing a flagged business in the ways people really type it get the right warning, and does an unrelated or look-alike name stay
quiet? Everything runs on the FICTIONAL demo register (services/demo_data.py), so this measures the matching rules, not how well
real fraud is caught.

Positives (all should be UNSAFE): the exact name, lower-case, with/without Ltd/LLC, extra punctuation, the bare domain, a full URL.
Typos (should be CAUTION, never silent): one-character deletion / substitution / transposition of a flagged name.
Negatives (should be NO_RECORD): thousands of random unrelated names and domains.
Hard negatives (should be NO_RECORD): a real flagged word glued to a different business (e.g. "Zenith Plumbing").
Dismissed-only businesses should be CAUTION, never UNSAFE.

Class imbalance: in real use almost every lookup is for something NOT on the register. Precision therefore depends on how rare
flagged names are, so alongside recall and the false-positive rate (measured on many negatives) the script prints the precision
you would see if 1%, 10% or 50% of lookups were flagged businesses."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safeguard import create_app  # noqa: E402
from safeguard.config import TestConfig  # noqa: E402
from safeguard.models import Complaint  # noqa: E402
from safeguard.services import complaints as cp  # noqa: E402
from safeguard.services.demo_data import DISMISSED_ONLY  # noqa: E402

WORDS = ["Alder", "Birch", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Harbor", "Iris", "Juniper", "Kite", "Lumber", "Mosaic", "Nectar",
         "Onyx", "Prairie", "Quartz", "Ridge", "Saffron", "Tundra", "Umber", "Velvet", "Willow", "Yarrow", "Zephyr", "Anchor", "Beacon", "Canyon"]
KINDS = ["Bakery", "Plumbing", "Dental", "Logistics", "Studio", "Cafe", "Motors", "Textiles", "Gardens", "Print Shop", "Tutoring", "Cycles", "Pharmacy", "Travel"]
SUFFIX = ["Ltd", "LLC", "Inc", "Co", "Limited", ""]


def variants(name: str, domain: str | None) -> list[str]:
    base = re.sub(r"\s+(Ltd|LLC|Inc|Co)$", "", name)  # strip a trailing legal suffix only (not "Co" inside "Consultants")
    out = [name, name.lower(), name.upper(), base, f"{base} Ltd", f"{base} LLC", f"  {base}  ", base.replace(" ", "  "), f"{base}."]
    if domain:
        out += [domain, domain.upper(), f"https://{domain}", f"http://www.{domain}/pay?id=7", f"{domain}/login"]
    return out


def typos(name: str, rng: random.Random, n: int = 3) -> list[str]:
    key = cp.name_key(name)
    out = []
    for _ in range(n):
        i = rng.randrange(1, len(key) - 1)
        kind = rng.choice(["del", "sub", "swap"])
        if kind == "del":
            t = key[:i] + key[i + 1:]
        elif kind == "sub":
            t = key[:i] + rng.choice("aeiourstln") + key[i + 1:]
        else:
            t = key[:i] + key[i + 1] + key[i] + key[i + 2:]
        if t != key and cp.name_key(t) != key:
            out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", type=int, default=5000)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    class Cfg(TestConfig):
        INSTANCE_DIR = Path(tempfile.mkdtemp())
        DATA_DIR = INSTANCE_DIR / "data"
        DEMO_SEED = True

    app = create_app(Cfg)
    rng = random.Random(0)
    with app.app_context():
        rows = Complaint.query.all()
        ents: dict[str, dict] = {}
        for c in rows:
            e = ents.setdefault(c.name_key, {"name": c.entity_name, "domain": c.entity_domain, "active": 0})
            e["active"] += c.status in Complaint.ACTIVE
        flagged = [e for e in ents.values() if e["active"] and not e["name"].endswith((".com", ".net", ".org", ".co"))]
        legacy = [e for e in ents.values() if e["active"] and e["name"].endswith((".com", ".net", ".org", ".co"))]
        dismissed = [e for e in ents.values() if not e["active"]]
        known_keys = set(ents)

        def verdict(q: str) -> str:
            return cp.lookup(q)["verdict"]

        res = {"register": {"entities": len(ents), "flagged_named": len(flagged), "legacy_domains": len(legacy), "dismissed_only": len(dismissed)}}

        # positives
        pos = [(q, e["name"]) for e in flagged + legacy for q in variants(e["name"], e["domain"])]
        v = [verdict(q) for q, _ in pos]
        res["positives"] = {"queries": len(pos), "unsafe": sum(x == "unsafe" for x in v) / len(pos), "caution": sum(x == "caution" for x in v) / len(pos),
                            "missed_no_record": sum(x == "no_record" for x in v)}
        misses = [(q, x) for (q, _), x in zip(pos, v) if x != "unsafe"][:8]
        res["positives"]["examples_not_unsafe"] = misses

        # typos
        ty = [t for e in flagged for t in typos(e["name"], rng)]
        tv = [verdict(t) for t in ty]
        res["typos"] = {"queries": len(ty), "caution": sum(x == "caution" for x in tv) / len(ty), "unsafe": sum(x == "unsafe" for x in tv) / len(ty),
                        "silent_no_record": sum(x == "no_record" for x in tv) / len(ty)}

        # dismissed-only
        dv = [verdict(e["name"]) for e in dismissed]
        res["dismissed_only"] = {"businesses": len(dv), "caution": sum(x == "caution" for x in dv), "unsafe": sum(x == "unsafe" for x in dv)}

        # negatives
        neg = []
        while len(neg) < a.negatives:
            w1, w2, k = rng.choice(WORDS), rng.choice(WORDS), rng.choice(KINDS)
            q = rng.choice([f"{w1} {k}", f"{w1} {w2} {k} {rng.choice(SUFFIX)}".strip(), f"{w1.lower()}-{k.lower().replace(' ', '')}.example", f"https://www.{w2.lower()}{k.lower().replace(' ', '')}.example/shop"])
            if cp.name_key(q) not in known_keys:
                neg.append(q)
        nv = [verdict(q) for q in neg]
        fp_unsafe = sum(x == "unsafe" for x in nv)
        fp_caution = sum(x == "caution" for x in nv)
        res["negatives"] = {"queries": len(neg), "unsafe_false_alarms": fp_unsafe, "caution_false_alarms": fp_caution,
                            "false_unsafe_rate": fp_unsafe / len(neg), "false_flag_rate_any": (fp_unsafe + fp_caution) / len(neg)}

        # hard negatives: a flagged word + a different business
        hard = []
        for e in flagged:
            first = e["name"].split()[0]
            hard += [f"{first} {rng.choice(KINDS)}", f"{first} {rng.choice(KINDS)} {rng.choice(WORDS)}"]
        hv = [verdict(q) for q in hard]
        res["hard_negatives"] = {"queries": len(hard), "unsafe": sum(x == "unsafe" for x in hv), "caution": sum(x == "caution" for x in hv),
                                 "examples_flagged": [q for q, x in zip(hard, hv) if x != "no_record"][:6]}

    recall = res["positives"]["unsafe"]
    fpr = res["negatives"]["false_unsafe_rate"]
    res["precision_at_prevalence"] = {f"{int(p * 100)}%": (recall * p) / (recall * p + fpr * (1 - p)) if (recall * p + fpr * (1 - p)) else None for p in (0.01, 0.1, 0.5)}
    print(json.dumps(res, indent=2, default=str))
    if a.out:
        a.out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
