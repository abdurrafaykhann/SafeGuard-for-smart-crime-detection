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


# ---------------------------------------------------------------- streaming clip scorer
from safeguard.services import clip_model as cm  # noqa: E402
from safeguard.services.violence import ViolenceDetector  # noqa: E402


class StubClip:
    """Stands in for the trained model: calls a clip violent when its mean optical flow is high."""
    available, threshold, error = True, 0.5, None

    def prob(self, rows):
        return 0.9 if rows[1:, FRAME_FEATURES.index("flow_mean")].mean() > 1.0 else 0.1


def _row(flow):
    r = np.zeros(len(FRAME_FEATURES), np.float32)
    r[FRAME_FEATURES.index("flow_mean")] = flow
    return r


def test_stream_waits_for_enough_rows_then_requires_two_consecutive_high_scores():
    s = cm.ClipStream(StubClip())
    for i in range(cm.MIN_ROWS - 1):
        s.add_row(_row(3.0))
    assert not s.active and not s.violent                  # not enough history yet
    s.add_row(_row(3.0))                                   # 9th row: first evaluation
    assert s.active and s.prob == 0.9 and s.streak == 1 and not s.violent   # one high score is not enough
    for _ in range(cm.EVAL_EVERY):
        s.add_row(_row(3.0))                               # next evaluation, one second later
    assert s.streak == 2 and s.violent
    for _ in range(cm.WINDOW):                             # calm returns: the window refills with calm rows
        s.add_row(_row(0.0))
    assert not s.violent and s.streak == 0


def test_stream_samples_at_the_trained_rate_regardless_of_frame_rate():
    s = cm.ClipStream(StubClip())
    frame = np.full((120, 160, 3), 90, np.uint8)
    for i in range(80):                                    # an 8 fps source for 10 s of video time
        s.push(frame, i * 0.125, (0.0, []))
    assert len(s.rows) == cm.WINDOW                        # 8 fps source: every 2nd frame = 4 samples/s
    s2 = cm.ClipStream(StubClip())
    for i in range(30):                                    # a 30 fps source for 1 s
        s2.push(frame, i / 30, (0.0, []))
    assert 4 <= len(s2.rows) <= 5                          # still ~4 samples per second (min spacing 0.23 s)


def test_detector_uses_clip_model_once_active_and_falls_back_before_that():
    det = ViolenceDetector(threshold=0.2, min_frames=3, clip=StubClip())
    assert det.mode.endswith("+clip") and det.active_threshold == 0.2        # fusion decides at first
    rng = np.random.default_rng(0)
    results = []
    for i in range(120):                                   # noisy frames -> high flow; 8 fps video time
        results.append(det.analyze(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8), i * 0.125))
    assert results[0].source == "fusion"                   # before 9 rows exist
    assert results[-1].source == "clip" and results[-1].clip_prob is not None
    assert any(r.violent and r.source == "clip" and r.label == "Violence (trained classifier)" for r in results)
    assert det.active_threshold == 0.5 and "clip" in results[-1].cues


def test_detector_without_clip_model_is_unchanged():
    det = ViolenceDetector(threshold=0.2)
    r = det.analyze(np.full((120, 160, 3), 90, np.uint8))
    assert r.source == "fusion" and "clip" not in r.cues and det.clip is None


def test_clip_model_reports_missing_or_incompatible_files(tmp_path):
    assert not cm.ClipModel(tmp_path / "nope.joblib").available
    assert "no trained clip model" in cm.ClipModel(tmp_path / "nope.joblib").error
    assert "disabled" in cm.ClipModel(None, enabled=False).error


def test_clip_threshold_override_changes_the_decision_point(tmp_path):
    import joblib
    from safeguard.services.clip_features import CLIP_FEATURES
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy="prior").fit(np.zeros((4, 3)), [0, 1, 0, 1])
    path = tmp_path / "m.joblib"
    joblib.dump({"model": clf, "threshold": 0.38, "features": CLIP_FEATURES}, path)
    assert cm.ClipModel(path).threshold == 0.38
    over = cm.ClipModel(path, threshold_override=0.55)
    assert over.threshold == 0.55 and over.trained_threshold == 0.38
