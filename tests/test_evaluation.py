"""The evaluation script must compute correct metrics; verified on synthetic clips and by hand."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("evaluate_violence", Path(__file__).resolve().parent.parent / "scripts" / "evaluate_violence.py")
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def test_metrics_by_hand():
    y, pred = [1, 1, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 0, 1]
    m = ev.metrics(y, pred)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 3, 1)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.75) and m["f1"] == pytest.approx(0.75)


def test_roc_auc_known_values():
    assert ev.roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0      # perfectly separable
    assert ev.roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0      # perfectly wrong
    assert ev.roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5      # no information
    assert ev.roc_auc([1, 1], [0.3, 0.4]) is None                     # one class only


def test_label_folders():
    assert ev.label_of("Fight") == 1 and ev.label_of("NonFight") == 0
    assert ev.label_of("non_violence") == 0 and ev.label_of("Violence") == 1
    assert ev.label_of("random") is None


def _clip(path: Path, noisy: bool):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    rng = np.random.default_rng(0)
    for i in range(30):
        w.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) if noisy else np.full((120, 160, 3), 80, np.uint8))
    w.release()


def test_end_to_end_on_synthetic_dataset(tmp_path):
    for sub, noisy in (("Fight", True), ("NonFight", False)):
        (tmp_path / sub).mkdir()
        for i in range(2):
            _clip(tmp_path / sub / f"{i}.mp4", noisy)
    r = ev.evaluate(tmp_path, limit=None, use_model=False)
    assert r["videos"] == 4 and r["positives"] == 2 and r["negatives"] == 2
    assert r["roc_auc"] == 1.0  # noisy clips score strictly higher than static ones
    assert len(r["sweep"]) == 19 and "f1" in r["best_f1_threshold"]
    # honest by construction: heuristics alone never reach the alert threshold, so nothing fires
    assert r["at_production_settings"]["tp"] == 0 and r["mode"] == "heuristics-only"
