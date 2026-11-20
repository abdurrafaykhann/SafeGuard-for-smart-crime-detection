"""Build the data for the website's recorded-replay demo (/demo).

    python scripts/demo_video/make_replay.py [MEDIA_DIR]

Takes a few licence-safe stock clips (Pexels, see docs/demo/CREDITS.md), trims and shrinks each to a short 480p MP4 and runs the
REAL detectors over the shrunken file at the live pipeline's 8 fps: the same ViolenceDetector, trained clip classifier and weapon
model that a live camera uses. The per-frame output (risk score, people boxes, weapon boxes, alert moments) is saved as JSON
next to the videos. The web page then replays the video and draws that recorded output, so what visitors see is exactly what the
system produced, but nothing is computed in their browser or on the server. Nothing here is tuned on the clips."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from safeguard import create_app  # noqa: E402
from safeguard.services.framecheck import is_corrupt  # noqa: E402
from safeguard.services.violence import ViolenceDetector  # noqa: E402

OUT = ROOT / "safeguard" / "static" / "demo"
MAX_SECONDS = 20
# (clip file, title, camera name, location, what it shows). A mix on purpose; the page states the honest limits.
CLIPS = [
    ("fight/6296278.mp4", "Altercation in a studio", "CAM-01", "Gym entrance", "Violence detection on a sparring session"),
    ("knife/5735079.mp4", "Knife in hand", "CAM-02", "Kitchen corridor", "Weapon detection: a knife is shown to the camera", 12),
    ("gun/8105520.mp4", "Handgun drawn", "CAM-03", "Car park", "Weapon and violence detection together"),
    ("gun/6092105.mp4", "Rifle in view", "CAM-04", "Back gate", "Long gun detected, then a violence alert"),
    ("normal/6307658.mp4", "Ordinary activity", "CAM-05", "Reception", "No weapon, no violence: a quiet scene stays quiet"),
]
FFMPEG = "ffmpeg"


SECONDS = {}


def main() -> int:
    media = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "instance" / "demo_media"
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    app = create_app()
    sg = app.extensions["safeguard"]
    cfg = app.config
    manifest = {m["file"]: m for m in json.loads((media / "manifest.json").read_text(encoding="utf-8"))}
    out_clips = []
    for rel, title, cam, loc, blurb, *lim in CLIPS:
        limit = lim[0] if lim else MAX_SECONDS
        src = media / rel
        stem = src.stem
        dst = OUT / "clips" / f"{stem}.mp4"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-t", str(limit), "-vf", "scale=-2:480", "-an", "-c:v", "libx264",
                        "-preset", "slow", "-crf", "29", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
        cap = cv2.VideoCapture(str(dst))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(int(round(fps / 8.0)), 1)
        det = ViolenceDetector(cfg["VIOLENCE_THRESHOLD"], cfg["VIOLENCE_MIN_FRAMES"], sg.yolo, sg.weapons, sg.clip, cfg["SCENE_MOTION_GATE"])
        samples, idx = [], 0
        first_v = first_w = None
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
                    h, w = frame.shape[:2]
                if not is_corrupt(frame):
                    t = idx / fps
                    r = det.analyze(frame, t)
                    if r.violent and first_v is None:
                        first_v = t
                    if r.weapon_alert and first_w is None:
                        first_w = t
                    samples.append({
                        "t": round(t, 2), "risk": round(float(r.smoothed), 3), "clip": None if r.clip_prob is None else round(float(r.clip_prob), 3),
                        "violent": bool(r.violent), "label": r.label, "weaponAlert": bool(r.weapon_alert),
                        "people": [[round(x1 / w, 4), round(y1 / h, 4), round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)] for x1, y1, x2, y2 in r.persons],
                        "weapons": [{"name": n, "conf": round(float(c), 3), "box": [round(x1 / w, 4), round(y1 / h, 4), round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)]}
                                    for n, c, (x1, y1, x2, y2) in r.weapons],
                    })
            idx += 1
        cap.release()
        m = manifest.get(rel, {})
        out_clips.append({"id": stem, "title": title, "camera": cam, "location": loc, "blurb": blurb, "video": f"clips/{stem}.mp4",
                          "seconds": round(idx / fps, 1), "firstViolence": None if first_v is None else round(first_v, 1),
                          "firstWeapon": None if first_w is None else round(first_w, 1), "source": m.get("page"), "license": m.get("license"), "samples": samples})
        print(f"{rel:<22} {dst.stat().st_size / 1e6:4.1f} MB  samples={len(samples)}  violence@{first_v}  weapon@{first_w}", flush=True)
    (OUT / "replay.json").write_text(json.dumps({"fps": 8, "note": "Recorded output of the real detectors on stock footage.", "clips": out_clips}, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT / "replay.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
