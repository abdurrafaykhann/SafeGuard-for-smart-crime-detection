"""Generate notebooks/train_weapon_detector.ipynb (run once: python scripts/weapons/make_notebook.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip().splitlines(keepends=True)}


cells = [
    md("""
# SafeGuard - train the weapon detector (gun / knife)

**Runtime -> Change runtime type -> T4 GPU** before you start. Expected time: about 1-1.5 hours.

You will upload three small files (made by `python scripts/weapons/make_upload_bundle.py`):
`weapon-dataset-main-yolo.ndjson`, `negatives.zip` (your own webcam frames with no weapon in view)
and `prepare_dataset.py`. The ~12,000 training images are downloaded **inside Colab** from the links in the
NDJSON file (they expire around 7 Nov 2026), so nothing large is uploaded.

The `negatives.zip` frames contain your face. They go to your own Colab session only; delete the
runtime afterwards (Runtime -> Disconnect and delete runtime) to remove them.
"""),
    code("""
!nvidia-smi -L
!pip -q install ultralytics
import ultralytics; ultralytics.checks()
"""),
    code("""
from google.colab import files
up = files.upload()          # choose the three files listed above
print(list(up))
!mkdir -p /content/negatives && unzip -q -o negatives.zip -d /content/negatives
!ls /content/negatives | head -3 && echo "negatives:" && ls /content/negatives | wc -l
"""),
    md("""
## 1. Build the clean dataset
Keeps only `gun` / `knife`, drops the dataset's vague `weapon` and `stick-rod` labels, **re-splits by source photo**
(the original split leaks augmented copies of one photo across train and test), downloads and verifies the images,
and adds your webcam negatives as images with no boxes.
"""),
    code("""
!python prepare_dataset.py --ndjson weapon-dataset-main-yolo.ndjson --out /content/weapons --negatives /content/negatives --max-train 12000 --workers 32
!cat /content/weapons/manifest.json
"""),
    md("""
## 2. Train
Starts from COCO-pretrained YOLO11n. Settings favour what went wrong with earlier models: strong scale and
brightness variation (knives held close to a webcam look huge and blurry) and many empty frames as negatives.
"""),
    code("""
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.train(
    data="/content/weapons/data.yaml", epochs=40, imgsz=640, batch=32, patience=10, seed=0,
    project="/content/runs", name="weapon", cos_lr=True, close_mosaic=8, workers=4,
    scale=0.8, degrees=8, hsv_s=0.6, hsv_v=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0,
)
"""),
    md("""
## 3. Honest score on the clean test split
`test` here is the re-split, so no photo in it (or an augmented copy of it) was seen in training.
"""),
    code("""
best = YOLO("/content/runs/weapon/weights/best.pt")
m = best.val(data="/content/weapons/data.yaml", split="test", imgsz=640)
for i, name in best.names.items():
    print(f"{name:6s} precision {m.box.p[i]:.3f}  recall {m.box.r[i]:.3f}  mAP50 {m.box.ap50[i]:.3f}  mAP50-95 {m.box.ap[i]:.3f}")
print("overall mAP50", round(float(m.box.map50), 3), "| mAP50-95", round(float(m.box.map), 3))
"""),
    md("""
## 4. Export and download
ONNX is the file to use in SafeGuard: it is a plain computation graph, so loading it cannot run arbitrary code the
way a `.pt` pickle can. Keep `best.pt` only as a backup.
"""),
    code("""
best.export(format="onnx", imgsz=640, opset=12, simplify=True)
!cp /content/runs/weapon/weights/best.onnx /content/Weapon_own.onnx
!cp /content/runs/weapon/results.csv /content/results.csv
!cp /content/runs/weapon/args.yaml /content/args.yaml
!cp /content/weapons/manifest.json /content/manifest.json
!zip -q weapon_model.zip Weapon_own.onnx results.csv args.yaml manifest.json /content/runs/weapon/weights/best.pt
files.download("weapon_model.zip")
"""),
    md("""
## Back on your computer
1. Unzip `weapon_model.zip`; copy `Weapon_own.onnx` to `models/Weapon_own.onnx`.
2. Score it on your held-out webcam frames: `python scripts/weapons/eval_webcam.py models/Weapon_own.onnx`
   (compare with the two community models: knife recall <= 9%).
3. Only if recall is high and the false-alarm rate is near zero, enable it with
   `SAFEGUARD_WEAPON_ENABLED=true` and `SAFEGUARD_WEAPON_WEIGHTS=Weapon_own.onnx`.
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "train_weapon_detector.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
