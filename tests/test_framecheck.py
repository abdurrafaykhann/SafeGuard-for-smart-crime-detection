import cv2
import numpy as np

from safeguard.services.framecheck import NOISE_LIMIT, is_corrupt, noise_level


def test_pure_noise_is_corrupt_but_ordinary_scenes_are_not():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert is_corrupt(noise)
    scene = np.full((480, 640, 3), 90, np.uint8)
    cv2.rectangle(scene, (100, 80), (300, 400), (30, 160, 220), -1)          # hard edges
    cv2.putText(scene, "ENTRANCE 01", (330, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    scene = cv2.add(scene, rng.normal(0, 6, scene.shape).clip(0, 255).astype(np.uint8))   # normal sensor grain
    assert not is_corrupt(scene) and noise_level(scene) < NOISE_LIMIT / 3


def test_measurement_does_not_depend_on_frame_size():
    rng = np.random.default_rng(1)
    big = cv2.resize(rng.integers(0, 255, (360, 480, 3), dtype=np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST)
    assert is_corrupt(big) and is_corrupt(big[::2, ::2])


def test_pipeline_ignores_corrupted_frames(app):
    from types import SimpleNamespace
    from safeguard.services.pipeline import CameraPipeline
    cam = SimpleNamespace(id=1, name="c", source="0", detect_violence=True, detect_faces=False)
    sg = app.extensions["safeguard"]
    pipe = CameraPipeline(app, cam, sg.yolo, sg.faces, sg.weapons, sg.cnn, sg.clip)
    rng = np.random.default_rng(2)
    noisy = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    class Once:
        connected, fps = True, 8.0
        def __init__(self): self.n = 0
        def wait_for_frame(self, last_id, timeout=2.0):
            self.n += 1
            if self.n > 3:
                pipe._stop.set()
            return noisy, self.n
        def start(self): pass
        def stop(self): pass

    pipe.stream = Once()
    pipe._run()
    assert pipe.bad_frames >= 3 and pipe.state["label"].startswith("Bad video frame") and pipe.state["violent"] is False
