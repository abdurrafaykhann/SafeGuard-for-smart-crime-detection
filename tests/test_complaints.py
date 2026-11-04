from datetime import datetime

from safeguard.extensions import db
from safeguard.models import Complaint
from safeguard.services import complaints as cp


def _add(app, name, status="reported", domain=None, cat="Investment scam", summary="took a deposit"):
    with app.app_context():
        n = Complaint.query.count() + 1
        db.session.add(Complaint(reference=f"FR-T-{n:03d}", entity_name=name, name_key=cp.name_key(name), entity_domain=domain,
                                 category=cat, summary=summary, status=status, filed_at=datetime(2026, 1, n)))
        db.session.commit()


def test_name_key_ignores_case_punctuation_and_legal_suffixes():
    assert cp.name_key("Brightpath Holdings, LTD.") == cp.name_key("brightpath   holdings") == "brightpath holdings"
    assert cp.name_key("Inc") == "inc"  # a name made only of a suffix is kept


def test_verdicts(app):
    _add(app, "Brightpath Holdings Ltd", "confirmed", "brightpath.example")
    _add(app, "Willow Furniture", "dismissed")
    with app.app_context():
        assert cp.lookup("brightpath holdings")["verdict"] == "unsafe"
        assert cp.lookup("https://www.Brightpath.example/pay?x=1")["verdict"] == "unsafe"      # by web address
        assert cp.lookup("Brightpath Holdngs")["verdict"] == "caution"                          # typo: similar name
        assert cp.lookup("willow furniture")["verdict"] == "caution"                            # only dismissed complaints
        none = cp.lookup("Completely Unknown Traders")
        assert none["verdict"] == "no_record" and none["matches"] == []
        assert cp.lookup("a")["valid"] is False


def test_public_home_page_checker_needs_no_login_and_shows_verdicts(client, app):
    _add(app, "Zenith Crypto Exchange", "under_investigation", "zenith.example", "Cryptocurrency scam", "withdrawals blocked")
    home = client.get("/").data
    assert b"Public fraud check" in home
    unsafe = client.get("/?q=Zenith+Crypto+Exchange")
    assert unsafe.status_code == 200 and b"UNSAFE" in unsafe.data and b"withdrawals blocked" in unsafe.data
    assert b"No complaints found" in client.get("/?q=Nobody+Heard+Of+Us").data
    xss = client.get("/?q=<script>alert(1)</script>").data
    assert b"<script>alert(1)</script>" not in xss
    assert client.get("/api/public/check?q=zenith.example").get_json()["verdict"] == "unsafe"


def test_public_result_never_exposes_complainant_or_internal_fields(client, app):
    _add(app, "Kestrel Repairs", "reported")
    body = client.get("/api/public/check?q=kestrel+repairs").get_data(as_text=True)
    assert "created_by" not in body and "complainant" not in body


def test_filing_a_complaint_is_admin_only(client, admin, app):
    anon = client.application.test_client()
    assert anon.post("/api/complaints", data={"entity_name": "X Y", "category": "c"}).status_code in (302, 401, 403)
    r = admin.post("/api/complaints", data={"entity_name": "Fresh Scam Co", "category": "Phishing", "entity_domain": "fresh-scam.example"})
    assert r.status_code == 201
    with app.app_context():
        assert cp.lookup("fresh-scam.example")["verdict"] == "unsafe"
