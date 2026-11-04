"""A tiny fake IP camera for demos and tests: plays a playlist of video files as a real-time MJPEG stream.

    python scripts/demo_video/stream_server.py SCENES.json [--port 8099]

SCENES.json maps a scene name to a list of video paths (relative to the JSON file), e.g.
    {"lobby": ["normal/a.mp4", "gun/b.mp4"]}
http://127.0.0.1:8099/lobby.mjpg then behaves like a network camera: SafeGuard can add it as a normal HTTP camera source
(file paths are deliberately not accepted as camera sources, so this is how pre-recorded footage is shown live). Each
connection starts the playlist from the beginning, paced at the clip's own frame rate, and loops forever."""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 360


def letterbox(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
    small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), np.uint8)
    y, x = (H - nh) // 2, (W - nw) // 2
    canvas[y:y + nh, x:x + nw] = small
    return canvas


def make_handler(scenes: dict[str, list[Path]]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def do_GET(self):
            name = self.path.strip("/").removesuffix(".mjpg")
            files = scenes.get(name)
            if not files:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    for path in files:
                        cap = cv2.VideoCapture(str(path))
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        due = time.monotonic()
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            ok, buf = cv2.imencode(".jpg", letterbox(frame), [cv2.IMWRITE_JPEG_QUALITY, 82])
                            data = buf.tobytes()
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data) + data + b"\r\n")
                            due += 1.0 / fps
                            time.sleep(max(0.0, due - time.monotonic()))
                        cap.release()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", type=Path)
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    base = a.scenes.parent
    scenes = {k: [base / p for p in v] for k, v in json.loads(a.scenes.read_text(encoding="utf-8")).items()}
    for k, files in scenes.items():
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"scene {k}: missing {missing}")
    print(f"serving {', '.join(scenes)} on http://127.0.0.1:{a.port}/<scene>.mjpg", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(scenes)).serve_forever()


if __name__ == "__main__":
    main()
