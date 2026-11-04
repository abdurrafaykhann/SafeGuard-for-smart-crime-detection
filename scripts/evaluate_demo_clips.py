"""Run the live detectors (violence + weapon) over a folder of labelled clips and report what they actually did.

    python scripts/evaluate_demo_clips.py MEDIA_DIR [--out results.json]

MEDIA_DIR holds sub-folders named after the clip category (fight, gun, knife, normal ...). The same
ViolenceDetector / clip classifier / WeaponBackend the camera pipeline uses are run at the same 8 fps. For every clip it
records whether a violence alert and/or a weapon alert would have fired, when, and with what confidence. Nothing is tuned
on these clips; this is a report, not a training step."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from safeguard import create_app  # noqa: E402
from safeguard.services.framecheck import is_corrupt  # noqa: E402
from safeguard.services.violence import ViolenceDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("media", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    app = create_app()
    sg = app.extensions["safeguard"]
    cfg = app.config
    rows = []
    for path in sorted(a.media.glob("*/*.mp4")):
        cat = path.parent.name
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(int(round(fps / 8.0)), 1)
        det = ViolenceDetector(cfg["VIOLENCE_THRESHOLD"], cfg["VIOLENCE_MIN_FRAMES"], sg.yolo, sg.weapons, sg.clip, cfg["SCENE_MOTION_GATE"])
        first_violence = None
        first_weapon = None
        peak_clip = peak_weapon = 0.0
        weapon_kinds: dict[str, float] = {}
        violent_frames = frames = 0
        idx = 0
        while True:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                h, w = frame.shape[:2]
                if w > cfg["FRAME_WIDTH"]:
                    frame = cv2.resize(frame, (cfg["FRAME_WIDTH"], int(h * cfg["FRAME_WIDTH"] / w)))
                if is_corrupt(frame):
                    idx += 1
                    continue
                t = idx / fps
                r = det.analyze(frame, t)
                frames += 1
                if r.clip_prob is not None:
                    peak_clip = max(peak_clip, r.clip_prob)
                if r.violent:
                    violent_frames += 1
                    first_violence = t if first_violence is None else first_violence
                for name, conf, _ in r.weapons:
                    weapon_kinds[name] = max(weapon_kinds.get(name, 0.0), conf)
                    peak_weapon = max(peak_weapon, conf)
                if r.weapon_alert and first_weapon is None:
                    first_weapon = t
            idx += 1
        cap.release()
        row = {"category": cat, "file": f"{cat}/{path.name}", "seconds": round(idx / fps, 1), "frames": frames,
               "violence_alert": first_violence is not None, "first_violence_s": None if first_violence is None else round(first_violence, 1),
               "violent_frames": violent_frames, "peak_clip_prob": round(peak_clip, 3),
               "weapon_alert": first_weapon is not None, "first_weapon_s": None if first_weapon is None else round(first_weapon, 1),
               "peak_weapon_conf": round(peak_weapon, 3), "weapon_classes": {k: round(v, 3) for k, v in weapon_kinds.items()}}
        rows.append(row)
        print(f"{row['file']:<22} {row['seconds']:>5}s  violence={'YES' if row['violence_alert'] else 'no ':<3} (peak {row['peak_clip_prob']:.2f})"
              f"  weapon={'YES' if row['weapon_alert'] else 'no ':<3} (peak {row['peak_weapon_conf']:.2f} {row['weapon_classes']})", flush=True)
    summary = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        summary[cat] = {"clips": len(sub), "violence_alerts": sum(r["violence_alert"] for r in sub), "weapon_alerts": sum(r["weapon_alert"] for r in sub)}
    print("summary:", json.dumps(summary))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "clips": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
