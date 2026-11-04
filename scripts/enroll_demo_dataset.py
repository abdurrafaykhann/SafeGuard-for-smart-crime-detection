"""Bulk-enrol a folder-per-person image dataset into the watchlist, with FICTIONAL case files.

    python scripts/enroll_demo_dataset.py DATASET_DIR [--replace] [--seed 7]

DATASET_DIR holds one sub-folder per person, each with that person's photos. Every person is enrolled
under an invented name and an invented case file (aliases, status, offences ...), clearly flagged
"demo": none of it is real. The real folder names are NOT used as labels, so no real person is ever
shown with a made-up criminal record; the folder -> invented-name map stays in instance/ (git-ignored)
so you can check results yourself. A folder is skipped, with a message, if no face is found in any photo.

The only exception is a folder named in --own (default: any folder whose name contains "me "), which
is enrolled under its own name with an empty record, for testing with your own face.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FIRST = ["Marcus", "Elena", "Tobias", "Priya", "Dmitri", "Lucia", "Kwame", "Hana", "Rafael", "Ingrid", "Omar", "Sven", "Amara",
         "Viktor", "Noor", "Declan", "Mei", "Jonas", "Farah", "Callum", "Yara", "Bruno", "Siobhan", "Idris", "Tamsin", "Anselm",
         "Leila", "Gideon", "Petra", "Rohan", "Odette", "Cyrus", "Maren", "Tariq", "Linnea", "Emeka", "Sabine", "Joaquin", "Freya",
         "Anil", "Corinne", "Mateus", "Ayesha", "Lennart", "Dalia", "Hugo", "Zainab", "Rurik", "Esme", "Kofi", "Valeria", "Bastian",
         "Nadia", "Orson", "Thandi", "Felix"]
LAST = ["Harrowgate", "Valdez", "Okonkwo", "Lindqvist", "Brennan", "Castellan", "Moravec", "Ashworth", "Kowalczyk", "Delacroix",
        "Nakagawa", "Fairbairn", "Haddad", "Sorensen", "Blackwood", "Ferreira", "Ostrowski", "Calloway", "Mbeki", "Thornbury",
        "Vasquez", "Eriksen", "Pemberton", "Rahimi", "Duquesne", "Lockhart", "Abernathy", "Zielinski", "Marchetti", "Greaves",
        "Hollander", "Quigley", "Nwosu", "Strand", "Ballantyne", "Karimi", "Whitlock", "Dragomir", "Penhallow", "Sandoval"]
OFFENCES = [("Burglary", "Residential break-in; property recovered"), ("Vehicle theft", "Stolen car abandoned 3 days later"),
            ("Fraud", "Falsified invoices to a small business"), ("Assault", "Altercation outside a venue; no weapon"),
            ("Shoplifting", "Repeated low-value thefts"), ("Vandalism", "Graffiti and damage to public transport"),
            ("Identity theft", "Used another person's details for credit"), ("Public disorder", "Involved in a street disturbance"),
            ("Drug possession", "Small quantity, personal use"), ("Robbery", "Threatened a cashier; no firearm"),
            ("Cyber fraud", "Phishing messages sent to residents"), ("Trespass", "Entered a restricted construction site")]
OUTCOMES = ["Convicted - community service", "Convicted - suspended sentence", "Convicted - served 8 months", "Charges dropped",
            "Acquitted", "Fined", "Case pending", "Warning issued"]
CITIES = ["Northbridge", "Easthaven", "Riverton", "Stonefield", "Marlow Heights", "Cedar Falls", "Port Alder", "Westmere"]
STATUS = ["Wanted", "On parole", "Released - no restrictions", "Under investigation", "Convicted - serving sentence", "Case closed"]
NATIONS = ["Britain", "Pakistan", "Spain", "Nigeria", "Poland", "India", "Brazil", "Sweden", "Egypt", "Canada"]


def fictional_record(rng: random.Random, name: str, n: int) -> dict:
    alias = f"{rng.choice(['The', 'Big', 'Quiet', 'Lucky', 'Slim'])} {rng.choice(['Fox', 'Owl', 'Hawk', 'Ghost', 'Wolf', 'Ace'])}"
    offences = []
    for _ in range(rng.randint(1, 4)):
        kind, note = rng.choice(OFFENCES)
        offences.append({"date": f"{rng.randint(2012, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}", "offence": kind,
                         "location": rng.choice(CITIES), "outcome": rng.choice(OUTCOMES), "note": note})
    offences.sort(key=lambda o: o["date"])
    status = rng.choice(STATUS)
    return {"demo": True, "case_number": f"DEMO-{n:04d}", "aliases": [alias], "age": rng.randint(19, 62),
            "gender": rng.choice(["Male", "Female"]), "nationality": rng.choice(NATIONS), "height_cm": rng.randint(152, 193),
            "last_known_location": rng.choice(CITIES), "status": status, "offences": offences,
            "summary": f"{len(offences)} recorded offence(s). Status: {status}. Fictional profile generated for demonstration."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--replace", action="store_true", help="first remove entries previously created by this script")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--own", default="me ", help="folders whose name contains this are enrolled under their own name, no record")
    a = ap.parse_args()

    from safeguard import create_app
    from safeguard.extensions import db
    from safeguard.models import WatchlistEntry, WatchlistPhoto

    app = create_app()
    sg = app.extensions["safeguard"]
    if not sg.faces.can_recognize:
        raise SystemExit("Face models missing. Run: python scripts/download_models.py")
    folders = sorted(p for p in a.dataset.iterdir() if p.is_dir())
    rng = random.Random(a.seed)
    names = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(names)
    wl_dir = app.config["DATA_DIR"] / "watchlist"
    mapping: dict[str, dict] = {}

    def save_crop(img, face) -> str:
        x, y, w, h = face.box
        pad = int(0.2 * max(w, h))
        crop = img[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
        name = f"wl_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(wl_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return name

    with app.app_context():
        if a.replace:
            for e in WatchlistEntry.query.filter_by(created_by="demo-seed").all():
                for pth in [e.photo_path] + [p.photo_path for p in e.photos]:
                    if pth:
                        (wl_dir / pth).unlink(missing_ok=True)
                db.session.delete(e)
            db.session.commit()
        n = 0
        for folder in folders:
            shots = []
            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                img = cv2.imread(str(img_path))
                face = sg.faces.largest_face(img) if img is not None else None
                emb = sg.faces.embed(img, face) if face is not None else None
                if emb is not None:
                    shots.append((emb, save_crop(img, face)))
            if not shots:
                print(f"[skip] {folder.name}: no usable face in any photo")
                continue
            n += 1
            own = a.own and a.own in folder.name + " "
            if own:
                label, category, record = folder.name.replace("me ", "").title(), "project author (test)", {"demo": True, "status": "Not applicable",
                    "summary": "Developer test identity, no record."}
            else:
                label = names.pop()
                category, record = "demo person of record", fictional_record(rng, label, n)
            entry = WatchlistEntry(label=label, category=category, notes=record.get("summary", "")[:400], risk=rng.choice(["low", "medium", "high"]) if not own else "low",
                                   embedding=shots[0][0].astype(np.float32).tobytes(), photo_path=shots[0][1],
                                   consent_basis="Fictional demo data (no real individual)", created_by="demo-seed",
                                   record=json.dumps(record))
            entry.photos = [WatchlistPhoto(embedding=e.astype(np.float32).tobytes(), photo_path=p) for e, p in shots[1:]]
            db.session.add(entry)
            mapping[folder.name] = {"label": label, "photos": len(shots)}
        db.session.commit()
        sg.faces.load_gallery(WatchlistEntry.query.all())
        print(f"Enrolled {n} people, {sum(m['photos'] for m in mapping.values())} photos.")
    out = Path(app.config["INSTANCE_DIR"]) / "demo_identity_map.json"
    out.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"folder -> invented name map written to {out} (git-ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
