"""How does face identification behave as the watchlist grows? Accuracy, stranger false matches and search time.

    python scripts/evaluate_gallery_scale.py LFW_DIR [--out docs/evaluation_gallery_scale.json]

LFW_DIR is a folder-per-person dataset (Labeled Faces in the Wild, `lfw_funneled`). For each gallery size N (100 ... 1000) N
people are enrolled with ONE photo each (the hardest realistic enrolment); other photos of the same people are the "enrolled"
queries, and photos of people who were never enrolled are the "stranger" queries. Reported per N and threshold:
top-1 accuracy among enrolled queries, the share accepted as the right person / as the wrong person / rejected, and the
share of strangers wrongly matched. Search time is measured for numpy brute force and FAISS (flat inner product), also at
100k and 1M synthetic vectors to show where an index starts to matter. Embeddings are cached next to the dataset."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from safeguard.config import Config  # noqa: E402
from safeguard.services.faces import FaceEngine  # noqa: E402

SIZES = [100, 250, 500, 1000]
THRESHOLDS = [0.40, 0.45, 0.50, 0.55]


def embeddings(root: Path) -> tuple[np.ndarray, np.ndarray]:
    cache = root.parent / "embeddings_min2.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return d["X"], d["ids"]
    eng = FaceEngine(Config.MODELS_DIR, Config.FACE_MATCH_THRESHOLD)
    vecs, ids, skipped, n = [], [], 0, 0
    t0 = time.time()
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.jpg"))
        if len(files) < 2:
            continue
        for f in files:
            n += 1
            img = cv2.imread(str(f))
            face = eng.largest_face(img) if img is not None else None
            e = eng.embed(img, face) if face is not None else None
            if e is None:
                skipped += 1
                continue
            vecs.append(e)
            ids.append(folder.name)
            if n % 1000 == 0:
                print(f"  embedded {n} images ({time.time() - t0:.0f}s)", flush=True)
    X, ids = np.stack(vecs), np.array(ids)
    np.savez_compressed(cache, X=X, ids=ids)
    print(f"embedded {len(ids)} faces ({skipped} images skipped: no face found) in {time.time() - t0:.0f}s", flush=True)
    return X, ids


def timing(n_vectors: int, queries: int = 300) -> dict:
    import faiss
    rng = np.random.default_rng(1)
    G = rng.normal(size=(n_vectors, 128)).astype(np.float32)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    Q = G[rng.integers(0, n_vectors, queries)] + 0.05 * rng.normal(size=(queries, 128)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    t = time.perf_counter()
    for q in Q:
        int(np.argmax(G @ q))
    numpy_ms = (time.perf_counter() - t) / queries * 1000
    index = faiss.IndexFlatIP(128)
    index.add(G)
    t = time.perf_counter()
    for q in Q:
        index.search(q[None], 1)
    faiss_ms = (time.perf_counter() - t) / queries * 1000
    return {"vectors": n_vectors, "numpy_ms_per_query": round(numpy_ms, 3), "faiss_flat_ms_per_query": round(faiss_ms, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lfw", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    X, ids = embeddings(a.lfw)
    people = np.array(sorted(set(ids)))
    print(f"{len(people)} people, {len(ids)} faces", flush=True)
    first = {p: int(np.where(ids == p)[0][0]) for p in people}
    out = {"people_total": int(len(people)), "faces": int(len(ids)), "sizes": []}
    for n in SIZES:
        agg = {t: [0, 0, 0] for t in THRESHOLDS}
        strangers = {t: 0 for t in THRESHOLDS}
        top1 = n_gen = n_str = 0
        for rep in range(a.repeats):
            rng = np.random.default_rng(rep)
            chosen = rng.permutation(len(people))[:n]
            enrolled = set(people[chosen])
            gal_rows = np.array([first[p] for p in people[chosen]])
            G, gal_ids = X[gal_rows], ids[gal_rows]
            others = np.setdiff1d(np.arange(len(ids)), gal_rows)
            for i in others:
                sims = G @ X[i]
                j = int(np.argmax(sims))
                s = float(sims[j])
                if ids[i] in enrolled:
                    n_gen += 1
                    top1 += gal_ids[j] == ids[i]
                    for t in THRESHOLDS:
                        if s < t:
                            agg[t][2] += 1
                        elif gal_ids[j] == ids[i]:
                            agg[t][0] += 1
                        else:
                            agg[t][1] += 1
                else:
                    n_str += 1
                    for t in THRESHOLDS:
                        strangers[t] += s >= t
        row = {"gallery_people": n, "repeats": a.repeats, "enrolled_queries": n_gen, "stranger_queries": n_str,
               "top1_enrolled": round(top1 / n_gen, 4),
               "by_threshold": {str(t): {"enrolled_right": round(agg[t][0] / n_gen, 4), "enrolled_wrong_person": round(agg[t][1] / n_gen, 4),
                                         "enrolled_rejected": round(agg[t][2] / n_gen, 4), "stranger_false_match": round(strangers[t] / n_str, 5)}
                                for t in THRESHOLDS}}
        out["sizes"].append(row)
        b = row["by_threshold"]
        print(f"N={n:4d}: top-1 {row['top1_enrolled']:.1%} | thr 0.40: stranger matched {b['0.4']['stranger_false_match']:.2%}, right {b['0.4']['enrolled_right']:.1%}"
              f" | thr 0.45: stranger {b['0.45']['stranger_false_match']:.2%}, right {b['0.45']['enrolled_right']:.1%}"
              f" | thr 0.50: stranger {b['0.5']['stranger_false_match']:.2%}, right {b['0.5']['enrolled_right']:.1%}", flush=True)
    out["search_time"] = [timing(n) for n in (1000, 10_000, 100_000, 1_000_000)]
    for r in out["search_time"]:
        print(f"search over {r['vectors']:>9,} vectors: numpy {r['numpy_ms_per_query']:.3f} ms, FAISS flat {r['faiss_flat_ms_per_query']:.3f} ms per query", flush=True)
    if a.out:
        a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
