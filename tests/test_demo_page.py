import json
from pathlib import Path

import numpy as np

from safeguard.services.violence import WeaponBackend


def test_demo_page_and_replay_data(client):
    r = client.get("/demo")
    assert r.status_code == 200 and b"demo.js" in r.data
    data = json.loads((Path(__file__).resolve().parent.parent / "safeguard" / "static" / "demo" / "replay.json").read_text())
    assert len(data["clips"]) == 5
    for c in data["clips"]:
        assert c["samples"] and (Path(__file__).resolve().parent.parent / "safeguard" / "static" / "demo" / c["video"]).exists()
        for s in c["samples"]:
            for w in s["weapons"]:
                assert w["box"][2] > 0 and w["box"][3] > 0  # the zero-size ONNX box bug must not come back


def test_weapon_backend_without_weights_is_inert():
    wb = WeaponBackend(None)
    assert not wb.available and wb.error
