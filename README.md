# SafeGuard

AI-assisted video surveillance console: multi-camera live monitoring, explainable violence
detection, consent-based face matching, real-time alerts, analytics and recorded-video analysis,
with authentication, auditing and automatic data retention built in.

> Final-year project · Python · Flask · OpenCV · YOLO

## Features

| | |
|---|---|
| **Live wall** | Several webcams / RTSP / HTTP streams. One capture thread and one analysis loop per camera, shared by every viewer. |
| **Violence detection** | Optional YOLO model fused with motion, person-contact and injury-colour cues into one **explainable** score. Sustained incidents escalate to *critical*. |
| **Watchlist matching** | Face detection (YuNet) + recognition (SFace) against people enrolled **with a recorded legal basis**. Only a face crop and a numeric signature are stored. |
| **Real-time alerts** | Server-Sent Events push alerts to every open console. Optional webhook, Telegram and email. |
| **Video analysis** | Upload a clip, get an incident timeline, peak risk and evidence frames. The upload is deleted afterwards. |
| **Analytics** | Alerts per day / hour, hot-spot cameras, mean time-to-acknowledge, CSV export. |
| **Security** | Hashed passwords (scrypt), lockout, CSRF, rate limiting, strict CSP & headers, roles, audit log, upload validation. |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py # face detection + recognition models (small)
python run.py                     # http://127.0.0.1:5000
```

On first start an **admin** account is created and its random password is printed **once** in the
console. To choose your own, set `SAFEGUARD_ADMIN_PASSWORD` (see `.env.example`).

Add a camera under **Console → Cameras** (use `0` for the built-in webcam), then press *Start*
on the Live wall.

### Enable the trained YOLO violence model (optional, heavy)

```bash
pip install -r requirements-ml.txt   # pulls in PyTorch
```

`models/Yolo_nano_weights.pt` is picked up automatically. Without it the system runs in
**heuristics-only** mode, and the UI shows that badge, so results are never misrepresented.

## Architecture

```
safeguard/
  __init__.py        app factory, security headers, CLI, retention janitor
  config.py          all tunables via SAFEGUARD_* env vars / .env
  models.py          User, Camera, WatchlistEntry, Alert, VideoJob, AuditLog
  blueprints/        web (public) · auth · console (pages) · api (JSON + streams)
  services/
    camera.py        threaded capture, newest-frame-only, auto-reconnect, source validation
    violence.py      cue fusion (noisy-OR) + temporal smoothing + optional YOLO backend
    faces.py         YuNet / SFace, cosine-similarity gallery
    pipeline.py      per-camera analysis loop, MJPEG overlay, cooldowns, escalation
    alerts.py        persistence, snapshot storage, notifications, retention purge
    video_analysis.py offline clip analysis
    events.py        in-process pub/sub → SSE
  templates/ static/ "Obsidian" design system, dependency-free JS and SVG charts
tests/               pytest: security, features, end-to-end pipeline with a fake camera
```

**Flow:** `CameraStream` (thread) → `CameraPipeline` (thread) → annotated JPEG shared by all
MJPEG viewers, plus `Alert` rows → `EventBus` → browsers over SSE.

## Configuration

Copy `.env.example` to `.env`. Notable options: `SAFEGUARD_VIOLENCE_THRESHOLD`,
`SAFEGUARD_ALERT_COOLDOWN`, `SAFEGUARD_FACE_MATCH_THRESHOLD`, `SAFEGUARD_RETENTION_DAYS`,
`SAFEGUARD_COOKIE_SECURE` (enable behind HTTPS) and the notification settings.

CLI: `flask --app run create-user NAME --role operator` · `flask --app run purge`

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Docker

```bash
docker build -t safeguard .
docker run -p 5000:5000 -v safeguard-data:/app/instance -e SAFEGUARD_ADMIN_PASSWORD=change-me safeguard
```

Webcam passthrough is host-specific; in containers use RTSP/HTTP sources.

## Responsible use & known limitations

- Violence scores are **probabilistic**. Sports, crowds, red clothing and poor lighting cause false
  alarms; real incidents can be missed. Each alert shows which cues fired.
- A face match is **similarity, not identity**, and accuracy varies with image quality and across
  demographic groups. This project has **not** been benchmarked for bias.
- No automatic action is ever taken on a detection. A human acknowledges every alert.
- Enrol only people you have consent or lawful authority for; surveillance laws vary by jurisdiction.
- Not evaluated for production or enforcement use.

## History

The original prototype lives on the `main` branch. v2 replaced the single 2,000-line `app.py` with the
package above and removed: hard-coded credentials, a `drop_all()` on every start, test endpoints, a
random-number "recognition" fallback, and the celebrity-photo dataset.
