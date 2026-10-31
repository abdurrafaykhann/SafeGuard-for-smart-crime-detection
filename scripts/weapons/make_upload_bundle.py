"""Collect the three small files the Colab notebook needs into instance/weapons_upload/.

    python scripts/weapons/make_upload_bundle.py --ndjson "C:/path/weapon-dataset-main-yolo.ndjson"

negatives.zip contains your webcam frames (they show your face). Nothing is uploaded by this script; you
choose to upload the folder's files to Colab yourself.
"""
import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--ndjson", type=Path, required=True)
a = ap.parse_args()

out = ROOT / "instance" / "weapons_upload"
out.mkdir(parents=True, exist_ok=True)
shutil.copyfile(a.ndjson, out / "weapon-dataset-main-yolo.ndjson")
shutil.copyfile(ROOT / "scripts" / "weapons" / "prepare_dataset.py", out / "prepare_dataset.py")
neg = sorted((ROOT / "instance" / "weapons_site" / "negatives").glob("*.jpg"))
with zipfile.ZipFile(out / "negatives.zip", "w", zipfile.ZIP_STORED) as z:  # JPEGs do not compress further
    for p in neg:
        z.write(p, p.name)
for p in sorted(out.iterdir()):
    print(f"{p.name:34s} {p.stat().st_size / 1e6:7.1f} MB")
print(f"negatives in zip: {len(neg)}  ->  upload these files from: {out}")
