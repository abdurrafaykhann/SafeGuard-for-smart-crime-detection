"""Offline analysis of an uploaded video: sample frames, score them, summarise incidents.

The upload is deleted as soon as processing ends; only the summary and a few evidence
frames are kept."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import cv2

from ..extensions import db
from ..models import VideoJob
from .alerts import save_snapshot
from .faces import FaceEngine
from .violence import ViolenceDetector, YoloBackend

log = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SAMPLE_FPS = 8.0  # so the clip model gets its 4 measurements/s
MAX_EVIDENCE = 6


def start_job(app, job_id: int, path: Path, yolo: YoloBackend, faces: FaceEngine, weapons=None, clip=None) -> None:
    threading.Thread(target=_run, args=(app, job_id, path, yolo, faces, weapons, clip), name=f"video-{job_id}", daemon=True).start()


def _update(app, job_id: int, **fields) -> None:
    with app.app_context():
        job = db.session.get(VideoJob, job_id)
        if job:
            for k, v in fields.items():
                setattr(job, k, v)
            db.session.commit()


def _run(app, job_id: int, path: Path, yolo: YoloBackend, faces: FaceEngine, weapons=None, clip=None) -> None:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("The file could not be decoded as a video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(int(round(fps / SAMPLE_FPS)), 1)
        cfg = app.config
        detector = ViolenceDetector(cfg["VIOLENCE_THRESHOLD"], cfg["VIOLENCE_MIN_FRAMES"], yolo, weapons, clip, cfg["SCENE_MOTION_GATE"])
        _update(app, job_id, status="running")

        timeline, episodes, evidence, face_hits, weapon_hits = [], [], [], {}, []
        current: dict | None = None
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                h, w = frame.shape[:2]
                if w > cfg["FRAME_WIDTH"]:
                    frame = cv2.resize(frame, (cfg["FRAME_WIDTH"], int(h * cfg["FRAME_WIDTH"] / w)))
                t = idx / fps
                res = detector.analyze(frame, t)
                timeline.append([round(t, 2), round(res.smoothed, 3)])
                if res.weapon_alert and len(weapon_hits) < 12 and (not weapon_hits or t - weapon_hits[-1]["t"] > 2):
                    top = max(res.weapons, key=lambda w: w[1])
                    weapon_hits.append({"t": round(t, 2), "class": top[0], "confidence": round(top[1], 3),
                                        "file": save_snapshot(app, frame, f"video{job_id}")})
                if res.violent:
                    if current is None:
                        current = {"start": round(t, 2), "end": round(t, 2), "peak": res.smoothed, "label": res.label}
                    current["end"], current["peak"] = round(t, 2), max(current["peak"], res.smoothed)
                    if len(evidence) < MAX_EVIDENCE and (not evidence or t - evidence[-1]["t"] > 3):
                        evidence.append({"t": round(t, 2), "file": save_snapshot(app, frame, f"video{job_id}")})
                elif current is not None:
                    episodes.append(current)
                    current = None
                if faces.can_recognize and idx % (step * 5) == 0:
                    for face in faces.detect(frame):
                        m = faces.match(faces.embed(frame, face))
                        if m and m.label not in face_hits:
                            face_hits[m.label] = {"t": round(t, 2), "similarity": round(m.similarity, 3), "risk": m.risk}
                if total and idx % (step * 20) == 0:
                    _update(app, job_id, progress=min(idx / total, 0.99))
            idx += 1
        if current:
            episodes.append(current)
        peak = max((s for _, s in timeline), default=0.0)
        result = {
            "duration_s": round(idx / fps, 1), "frames_analysed": len(timeline), "mode": detector.mode,
            "peak_score": round(peak, 3), "episodes": [{**e, "peak": round(e["peak"], 3)} for e in episodes],
            "evidence": evidence, "watchlist_hits": face_hits, "weapon_hits": weapon_hits,
            "weapon_detector": bool(weapons and weapons.available),
            "timeline": timeline[:: max(len(timeline) // 400, 1)],
        }
        _update(app, job_id, status="done", progress=1.0, result=json.dumps(result))
    except Exception as exc:
        log.exception("video job %s failed", job_id)
        _update(app, job_id, status="failed", error=str(exc)[:290])
    finally:
        cap.release()
        path.unlink(missing_ok=True)
