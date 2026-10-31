"""Download the two small ONNX models used for face detection / recognition.

    python scripts/download_models.py

Both come from the OpenCV Zoo (Apache-2.0 / MIT). They are verified by actually loading
them with OpenCV before the script reports success.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2

MODELS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
DEST = Path(__file__).resolve().parent.parent / "models"


def verify(name: str, path: Path) -> bool:
    try:
        if "yunet" in name:
            cv2.FaceDetectorYN.create(str(path), "", (320, 320))
        else:
            cv2.FaceRecognizerSF.create(str(path), "")
        return True
    except cv2.error as exc:
        print(f"  ! {name} did not load: {exc}")
        return False


def main() -> int:
    DEST.mkdir(exist_ok=True)
    failed = False
    for name, url in MODELS.items():
        path = DEST / name
        if path.exists() and verify(name, path):
            print(f"✓ {name} already present")
            continue
        print(f"↓ {name}")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            print(f"  ! download failed: {exc}")
            failed = True
            continue
        if verify(name, path):
            print(f"✓ {name} ({path.stat().st_size // 1024} KB)")
        else:
            path.unlink(missing_ok=True)
            failed = True
    print("\nDone. Restart the server to enable face recognition." if not failed else "\nSome models failed; see above.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
