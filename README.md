# SafeGuard

AI-assisted video surveillance console: multi-camera live monitoring, explainable violence
detection, consent-based face matching, real-time alerts, analytics and recorded-video analysis,
with authentication, auditing and automatic data retention built in.

> Final-year project · Python · Flask · OpenCV · YOLO

**Author:** Abdul Rafay Khan · BSc (Hons) Computer Science (Online), University of Hertfordshire
**Supervisor:** Justin Mifsud · Module 6WCM0029-0509-2024, Computer Science Project

*Original project report: "SafeGuard: A Real-Time YOLOv8 and CNN-Based System for Weapon, Violence, and Criminal
Detection with Dual Portals for Law Enforcement and Civilian Safety Verification". This repository is the
v2 rebuild; where its measured results differ from the report, the README says so.*

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

## Weapon detection (optional, experimental)

SafeGuard can run a second YOLO model for guns and knives and raise a separate **Weapon** alert.
No weapon weights ship with the repo. Drop a YOLO `.pt` as `models/Weapon_threat_yolov8n.pt`
(or set `SAFEGUARD_WEAPON_WEIGHTS`); classes containing gun / pistol / rifle / knife / grenade count.

**Measured result:** the community model I tried
([Subh775/Threat-Detection-YOLOv8n](https://huggingface.co/Subh775/Threat-Detection-YOLOv8n), MIT) did
**not** detect a real kitchen knife held to a webcam (never as `knife` at conf >= 0.10; one 59% hit in
119 frames at the default cut-off) and repeatedly reported a `grenade` that was not there. Treat
weapon detection as a pluggable slot that needs a better, validated model, not a working feature.
`python scripts/check_weapon_model.py` reproduces this check on your own webcam.

A second candidate ([weapon-detection-yolov11n](https://github.com/The-lucky-Survivor/weapon-detection-yolov11n),
MIT, ONNX, classes knife / pistol / rifle) fared no better: with a real knife in view it never exceeded 39%
confidence, and with **no knife** in the same scene it reported "knife" on 60 detections in 47 frames (up to 64%).
Both models are therefore disabled by default (`SAFEGUARD_WEAPON_ENABLED=false`). A usable detector would need
training on images that look like your own cameras.

### Performance

Measured on a 12-core CPU, 640x480 frames: the violence model first used ~7 cores because PyTorch spawned a thread per
core and idle OpenMP workers spun. Capping PyTorch to 4 threads (re-applied after every call, because ultralytics
resets it) and `OMP_WAIT_POLICY=PASSIVE` cut that to ~2.2 cores at 107 ms/frame, and the whole server with one camera and
one viewer to ~1-2 cores while still delivering ~5.5 fps (the 6 fps analysis cap). Tune with `SAFEGUARD_TORCH_THREADS`.

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

## Evaluating the detector

`scripts/evaluate_violence.py` scores every video in a labelled dataset (`Fight/` and `NonFight/`
folders, e.g. RWF-2000 or Hockey Fights) and reports accuracy, precision, recall, F1, the confusion
matrix, ROC-AUC and a threshold sweep, both at the production alert threshold and across thresholds:

```bash
python scripts/evaluate_violence.py path/to/dataset --limit 100 --out results.json
```

### Measured results (RWF-2000)

| | Value |
|---|---|
| Data | RWF-2000 `val` split: 200 fight + 200 non-fight clips, never used to choose any setting |
| Settings | YOLO input 640 px, alert threshold **0.20**, 3 consecutive frames, 5 sampled frames/s |
| Threshold chosen on | 200 clips from the RWF-2000 `train` split (precision ~ recall on that set) |
| **Accuracy** | **76.0%** (304 / 400) |
| Precision / Recall / F1 | 78.6% / 71.5% / 0.75 |
| Confusion matrix | TP 143 · FP 39 · TN 161 · FN 57 |
| ROC-AUC | 0.835 |

Raw outputs: [`docs/evaluation_rwf2000_val.json`](docs/evaluation_rwf2000_val.json) and
[`docs/calibration_rwf2000_train_subset.json`](docs/calibration_rwf2000_train_subset.json).
Reproduce with the command above (`SAFEGUARD_VIOLENCE_THRESHOLD=0.20`).

### Trained clip classifier (own model, RWF-2000)

Instead of the hand-weighted cue fusion, a small classifier is trained on per-frame measurements of each clip
(the YOLO model's violence confidence and person count, optical-flow motion statistics, an injury-colour fraction;
see `safeguard/services/clip_features.py`). Pipeline: `scripts/extract_clip_features.py` ->
`scripts/train_clip_classifier.py`. Model family and alert threshold were chosen **only** by 5-fold cross-validation
on the 700 `train` clips; the 400 `val` clips were scored exactly once afterwards.

| RWF-2000 `val`, 400 clips | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Hand-weighted fusion (above) | 76.0% | 78.6% | 71.5% | 0.75 | 0.835 |
| **Trained clip classifier** (ExtraTrees, threshold 0.38) | **77.2%** | 73.0% | 86.5% | 0.79 | **0.871** |

Honest reading: with 400 test clips the 95% interval on accuracy is about +/-4 points, so **the accuracy difference
(76.0 vs 77.2) is within noise**. What is clearly better is ranking quality (AUC 0.835 -> 0.871) and recall (71.5% ->
86.5%), at the cost of more false alarms (precision 78.6% -> 73.0%). Cross-validated accuracy on the training clips
(78.0%) agrees with the validation figure (77.2%), so the number is not an artefact of tuning on the test set.
**Live behaviour (the production streaming rule).** In the app the model runs as a rolling 5-second window sampled at
4 measurements/s, re-scored once a second; an alert needs **two consecutive** scores above the threshold. The trained
whole-clip threshold (0.38) turned out to be too eager for that rule: sweeping the threshold with 5-fold out-of-fold
predictions on the **training** clips showed streaming accuracy peaking at **0.50** (80.1%; false-alarm rate on
non-fights 34% at 0.38 vs 17% at 0.50), so 0.50 is the live default (`SAFEGUARD_CLIP_THRESHOLD`; 0 = 0.38).
Replaying each validation clip through that exact class (`scripts/evaluate_clip_streaming.py`):

| Streaming rule on RWF-2000 `val` (400 clips) | Accuracy | Precision | Recall | F1 | TP / FP / TN / FN |
|---|---|---|---|---|---|
| threshold 0.38 (whole-clip value) | 76.8% | 73.4% | 84.0% | 0.78 | 168 / 61 / 139 / 32 |
| **threshold 0.50 (live default)** | **80.2%** | **82.4%** | **77.0%** | **0.80** | 154 / 33 / 167 / 46 |
| hand-weighted fusion (earlier) | 76.0% | 78.6% | 71.5% | 0.75 | 143 / 39 / 161 / 57 |

Transparency: `val` has now been scored several times (whole-clip, streaming at 0.38, streaming at 0.50). The 0.50 value
was **chosen on train only**, but treat the 80.2% as slightly optimistic and note the +/-4 point interval of a 400-clip
test. Live vs offline clip scores agreed to within 0.037 on average over 10 real clips. When `models/violence_clip.joblib`
exists the app uses it automatically (status chip "trained model: on"); otherwise it falls back to the hand-weighted
fusion, and during the first ~2 s of a stream it stays quiet instead. Disable with `SAFEGUARD_CLIP_ENABLED=false`.

A "two or more people in view" gate was also tested (train, out-of-fold) and **rejected**: the YOLO model finds 2+ people
in only ~44% of clips, fights and non-fights alike, so the gate cut recall from 89% to 25% for no precision gain.

**Known limitation (observed live).** The model was trained on surveillance-style clips (streets, crowds). On a close-up
desk webcam, vigorous movement near the camera can score near the threshold (two alerts at 0.382 vs 0.38 while moving
about), and the older cue-fusion fallback fired "Aggressive movement" on motion alone. RWF-2000 numbers do not
transfer automatically to your cameras: calibrate per site with `SAFEGUARD_CLIP_THRESHOLD` and, better, record your own normal activity with
`scripts/record_negatives.py` and retrain with `--extra-negatives` / `--site-test`, and review
alerts before trusting it.

Details: [`docs/training_violence_clip.json`](docs/training_violence_clip.json). The 17 MB model file is not committed;
regenerate it with the two commands above.

**Findings along the way**
- The first threshold (0.62) was miscalibrated for this model's scores: it caught 8% of fights (16/200).
  Re-choosing it on the training split, not the test set, lifted recall to 71.5%.
- The original FYP report quotes 88.6-89.8% for this component. I could **not** reproduce that: on
  the held-out validation clips I measure 76.0%. The report's prose metrics (85.3% precision) also
  disagree with its own confusion matrix (89.8%), whose counts (221 fights / 179 non-fights) do not
  add up to RWF-2000's 200 / 200. Treat 76% as this repository's number.
- At the 0.20 threshold, sustained extreme motion on its own can raise a violence alert (the motion cue alone
  can reach about 0.22); the numbers above include that behaviour.
- Clip-level, 5-fps sampling and a single-model-plus-heuristics pipeline; no claim is made about live
  CCTV, other datasets, or weapon / blood detection (see above).

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
