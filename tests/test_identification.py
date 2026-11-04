import json
from types import SimpleNamespace

import numpy as np

from safeguard.models import WatchlistEntry, WatchlistPhoto
from safeguard.services.faces import FaceEngine


def _vec(*lead):
    v = np.zeros(128, np.float32)
    v[:len(lead)] = lead
    return v / np.linalg.norm(v)


def _entry(i, label, *vectors):
    e = WatchlistEntry(id=i, label=label, embedding=vectors[0].tobytes(), consent_basis="x", record=json.dumps({"demo": True, "case_number": f"D{i}"}))
    e.photos = [WatchlistPhoto(embedding=v.tobytes()) for v in vectors[1:]]
    return e


def test_ranking_keeps_one_row_per_person_using_their_best_photo():
    eng = FaceEngine.__new__(FaceEngine)
    eng.match_threshold = 0.4
    eng.load_gallery([_entry(1, "A", _vec(1, 0), _vec(0, 1)), _entry(2, "B", _vec(1, 1))])
    top = eng.top_matches(_vec(0, 1), 5)
    assert [m.entry_id for m in top] == [1, 2] and len(top) == 2        # A once, not once per photo
    assert top[0].similarity > 0.99                                        # via A's second photo
    assert eng.match(_vec(0, 1)).label == "A" and eng.match(_vec(-1, -1)) is None


def test_case_file_round_trip_and_bad_json():
    e = _entry(1, "A", _vec(1, 0))
    assert e.to_dict(detail=True)["record"]["case_number"] == "D1" and "record" not in e.to_dict()
    e.record = "not json"
    assert e.case_file() == {}


def test_case_file_endpoint_lists_photos_searches_and_sightings(admin, app):
    import json as _json
    from safeguard.extensions import db
    from safeguard.models import Alert, AuditLog, WatchlistPhoto
    with app.app_context():
        e = _entry(7, "Case Person", _vec(1, 0), _vec(0, 1))
        e.photos[0].photo_path = "x.jpg"
        db.session.add(e)
        db.session.add(AuditLog(username="op", action="face_search_hit", detail="entry:7 sim:0.812"))
        db.session.add(AuditLog(username="op", action="face_search_hit", detail="entry:70 sim:0.900"))  # a different person (id 70)
        db.session.add(Alert(kind="watchlist", severity="high", title="m", watchlist_id=7, camera_name="Door", confidence=0.7))
        db.session.commit()
    d = admin.get("/api/watchlist/7?history=1").get_json()
    assert d["record"]["case_number"] == "D7" and len(d["photo_ids"]) == 1
    assert [s["similarity"] for s in d["searches"]] == [0.812]          # entry 70 must not leak into entry 7
    assert d["sightings"][0]["camera"] == "Door"
    assert admin.get("/console/watchlist/7").status_code == 200
    assert admin.get("/api/watchlist/7/photos/999").status_code == 404
    assert "searches" not in admin.get("/api/watchlist/7").get_json()
