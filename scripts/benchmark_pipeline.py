"""Measure the live pipeline: analysed frames per second per camera, per-frame latency, CPU use, and name the hardware.

    python scripts/benchmark_pipeline.py CLIP.mp4 [--cameras 1 2 3] [--seconds 40] [--out docs/benchmark_pipeline.json]

A fake IP camera (scripts/demo_video/stream_server.py) loops CLIP.mp4 in real time; N SafeGuard camera pipelines are started on
it with the production settings (violence classifier + weapon detector, ANALYSIS_FPS=8, faces off) and measured after a warm-up.
Run it with nothing else heavy open: the numbers are for this machine only."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def hardware() -> dict:
    ps = ("$c=Get-CimInstance Win32_Processor|Select-Object -First 1; $m=Get-CimInstance Win32_ComputerSystem; "
          "$g=(Get-CimInstance Win32_VideoController|ForEach-Object Name) -join '; '; "
          "@{cpu=$c.Name.Trim();cores=$c.NumberOfCores;threads=$c.NumberOfLogicalProcessors;ram_gb=[math]::Round($m.TotalPhysicalMemory/1GB,1);gpu=$g}|ConvertTo-Json")
    try:
        info = json.loads(subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout)
    except Exception:
        info = {}
    info["os"] = platform.platform()
    info["python"] = platform.python_version()
    info["note"] = "CPU inference only; the models are not run on the GPU"
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seconds", type=int, default=40)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    inst = Path(tempfile.mkdtemp())
    os.environ["SAFEGUARD_INSTANCE_DIR"] = str(inst)
    os.environ["SAFEGUARD_ADMIN_PASSWORD"] = "bench-only-password"
    scenes = inst / "scenes.json"
    scenes.write_text(json.dumps({"bench": [str(a.clip.resolve())]}), encoding="utf-8")
    server = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "demo_video" / "stream_server.py"), str(scenes), "--port", "8097"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2)
        from safeguard import create_app
        from safeguard.extensions import db
        from safeguard.models import Camera
        from safeguard.services.violence import ViolenceDetector

        app = create_app()
        sg = app.extensions["safeguard"]
        calls, total_s = [0], [0.0]
        lock = threading.Lock()
        orig = ViolenceDetector.analyze

        def timed(self, frame, t=None):
            t0 = time.perf_counter()
            r = orig(self, frame) if t is None else orig(self, frame, t)
            dt = time.perf_counter() - t0
            with lock:
                calls[0] += 1
                total_s[0] += dt
            return r

        ViolenceDetector.analyze = timed
        print("warming up models...", flush=True)
        t0 = time.time()
        while time.time() - t0 < 600:  # the background warm-up thread logs when it is done
            time.sleep(2)
            if sg.yolo.available is not None and time.time() - t0 > 6:
                break
        results = []
        for n in a.cameras:
            with app.app_context():
                cams = [Camera(name=f"bench-{i}", source="http://127.0.0.1:8097/bench.mjpg", detect_violence=True, detect_faces=False) for i in range(n)]
                db.session.add_all(cams)
                db.session.commit()
                for c in cams:
                    sg.manager.start(c)
            time.sleep(25)  # warm-up: first frames, model init, ONNX session
            with lock:
                c0, s0 = calls[0], total_s[0]
            cpu0, w0 = time.process_time(), time.time()
            time.sleep(a.seconds)
            with lock:
                c1, s1 = calls[0], total_s[0]
            cpu1, w1 = time.process_time(), time.time()
            wall = w1 - w0
            row = {"cameras": n, "analysed_fps_total": round((c1 - c0) / wall, 2), "analysed_fps_per_camera": round((c1 - c0) / wall / n, 2),
                   "mean_ms_per_frame": round((s1 - s0) / max(c1 - c0, 1) * 1000, 1), "cpu_cores_used": round((cpu1 - cpu0) / wall, 2), "seconds": round(wall)}
            results.append(row)
            print(row, flush=True)
            for cid in sg.manager.running_ids():
                sg.manager.stop(cid)
            time.sleep(3)
        out = {"hardware": hardware(), "analysis_fps_setting": app.config["ANALYSIS_FPS"], "weapon_model": bool(sg.weapons.available),
               "violence_model": bool(sg.yolo.available), "clip_classifier": bool(sg.clip.available), "results": results}
        print(json.dumps(out["hardware"]))
        if a.out:
            a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        sg.manager.stop_all()
    finally:
        server.terminate()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
