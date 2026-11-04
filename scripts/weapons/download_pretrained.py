"""Download the community weapon detector used by SafeGuard (optional, off by default).

    python scripts/weapons/download_pretrained.py

Source: https://github.com/The-lucky-Survivor/weapon-detection-yolov11n (MIT) release v1.0, `best.onnx`
(YOLO11n, classes knife / pistol / rifle, 640x640). ONNX is a plain computation graph, so loading it cannot
execute code the way a pickled .pt can; the script still inspects the graph and refuses anything unexpected.
The training data behind the weights has no stated licence, so the file is NOT committed to this repo.

Measured limits (see README): ~72% gun / ~75% knife image-level recall on typical weapon photos at conf 0.25,
but only ~9% knife recall on close-up, blurry webcam frames. Enable with
    SAFEGUARD_WEAPON_ENABLED=true   SAFEGUARD_WEAPON_WEIGHTS=Weapon_yolo11n.onnx
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/The-lucky-Survivor/weapon-detection-yolov11n/releases/download/v1.0/best.onnx"
DEST = Path(__file__).resolve().parent.parent.parent / "models" / "Weapon_yolo11n.onnx"
ALLOWED_OPS = {"Add", "Concat", "Conv", "Div", "MatMul", "MaxPool", "Mul", "Reshape", "Resize", "Sigmoid",
               "Slice", "Softmax", "Split", "Sub", "Transpose"}


def main() -> int:
    DEST.parent.mkdir(exist_ok=True)
    tmp = DEST.with_suffix(".part")
    print("downloading", URL)
    urllib.request.urlretrieve(URL, tmp)
    try:
        import onnx
        model = onnx.load(str(tmp))
        ops = {n.op_type for n in model.graph.node}
        unexpected = ops - ALLOWED_OPS
        if unexpected:
            raise ValueError(f"unexpected operators in the graph: {sorted(unexpected)}")
        names = {p.key: p.value for p in model.metadata_props}.get("names", "")
        if "knife" not in names:
            raise ValueError("class list does not look like a weapon detector")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print("REFUSED:", exc)
        return 1
    tmp.replace(DEST)
    print(f"ok: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB), operators: {sorted(ops)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
