import importlib.util
from pathlib import Path

import numpy as np
import pytest

from safeguard.services.clip_features import CLIP_FEATURES, FRAME_FEATURES, FrameFeaturizer, clip_vector

SPEC = importlib.util.spec_from_file_location("train_clip", Path(__file__).resolve().parent.parent / "scripts" / "train_clip_classifier.py")
tr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tr)


def test_feature_names_match_vector_lengths():
    assert len(CLIP_FEATURES) == len(FRAME_FEATURES) * 6 + 1
    assert clip_vector(np.zeros((12, len(FRAME_FEATURES)))).shape == (len(CLIP_FEATURES),)
    assert clip_vector(np.zeros((1, len(FRAME_FEATURES)))).shape == (len(CLIP_FEATURES),)  # too short: safe zeros


def test_motion_features_separate_calm_from_chaotic_scenes():
    rng = np.random.default_rng(0)
    calm_f, wild_f = FrameFeaturizer(), FrameFeaturizer()
    base = np.full((240, 320, 3), 90, np.uint8)
    calm = [calm_f(base) for _ in range(6)]
    wild = [wild_f(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)) for _ in range(6)]
    i = FRAME_FEATURES.index
    assert calm[-1][i("moving_fraction")] == 0 and calm[-1][i("flow_mean")] < 0.01
    assert wild[-1][i("moving_fraction")] > 0.03 and wild[-1][i("flow_mean")] > calm[-1][i("flow_mean")]
    assert not np.isnan(np.stack(wild)).any()


def test_flow_coherence_is_high_for_uniform_translation():
    f = FrameFeaturizer()
    rng = np.random.default_rng(3)
    canvas = cv2_blur(rng.integers(0, 255, (260, 400, 3), dtype=np.uint8))
    f(canvas[10:250, 10:330])
    row = f(canvas[10:250, 14:334])  # whole scene shifts 4 px to one side
    assert row[FRAME_FEATURES.index("flow_coherence")] > 0.6  # coordinated motion, e.g. people walking one way


def cv2_blur(img):
    import cv2
    return cv2.GaussianBlur(img, (9, 9), 0)


def test_threshold_picker_and_metrics():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.45, 0.4, 0.6, 0.7, 0.9])
    t = tr.pick_threshold(y, p)
    m = tr.metrics(y, p, t)
    assert 0.1 <= t <= 0.9 and m["f1"] >= tr.metrics(y, p, 0.5)["f1"] - 1e-9
    assert tr.metrics(y, p, 0.5)["tp"] == 3 and tr.metrics(y, p, 0.5)["fn"] == 1


def test_trainer_runs_on_synthetic_features(tmp_path):
    rng = np.random.default_rng(1)
    for split, n in (("train", 40), ("val", 20)):
        frames, lengths, labels, names = [], [], [], []
        for k in range(n):
            lab = k % 2
            rows = rng.normal(0, 0.1, (15, len(FRAME_FEATURES))).astype(np.float32) + lab * 0.6
            frames.append(rows); lengths.append(len(rows)); labels.append(lab); names.append(f"c{k}")
        np.savez(tmp_path / f"{split}_0.npz", frames=np.concatenate(frames), lengths=np.array(lengths),
                 labels=np.array(labels), names=np.array(names))
    import sys
    argv = sys.argv
    sys.argv = ["x", str(tmp_path), "--out", str(tmp_path / "m.joblib"), "--report", str(tmp_path / "r.json")]
    try:
        assert tr.main() == 0
    finally:
        sys.argv = argv
    import json
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["validation_once"]["auc"] > 0.9 and (tmp_path / "m.joblib").exists()
