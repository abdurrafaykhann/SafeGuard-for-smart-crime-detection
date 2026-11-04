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


def test_staff_can_list_search_and_update_status(client, admin, app):
    _add(app, "Orchid Travel", "reported", "orchid.example")
    rows = admin.get("/api/complaints?q=orchid").get_json()
    assert len(rows) == 1 and rows[0]["entity_name"] == "Orchid Travel"
    assert admin.get("/api/complaints?status=confirmed").get_json() == []
    assert admin.patch(f"/api/complaints/{rows[0]['id']}", json={"status": "dismissed"}).status_code == 200
    assert admin.patch(f"/api/complaints/{rows[0]['id']}", json={"status": "bogus"}).status_code == 400
    with app.app_context():
        assert cp.lookup("orchid.example")["verdict"] == "caution"          # dismissed no longer counts as unsafe
    assert admin.get("/console/complaints").status_code == 200
    assert client.application.test_client().get("/api/complaints").status_code in (302, 401)


def test_public_report_is_held_until_staff_approve_it(client, admin, app):
    pub = app.test_client()
    r = pub.post("/report", data={"entity_name": "Sunrise Gold Traders", "category": "Investment scam", "entity_domain": "https://sunrise-gold.example/x",
                                  "description": "asked for upfront fees"})
    assert r.status_code == 303 and "reported=1" in r.headers["Location"]
    with app.app_context():
        assert cp.lookup("Sunrise Gold Traders")["verdict"] == "no_record"            # a tip alone changes nothing
    pending = admin.get("/api/reports").get_json()
    assert len(pending) == 1 and pending[0]["entity_domain"] == "sunrise-gold.example"
    assert admin.post(f"/api/reports/{pending[0]['id']}/approve").get_json()["reference"].startswith("FR-")
    with app.app_context():
        assert cp.lookup("sunrise-gold.example")["verdict"] == "unsafe"
    assert admin.post(f"/api/reports/{pending[0]['id']}/reject").status_code == 400    # already reviewed
    assert pub.get("/api/reports").status_code in (302, 401)


def test_report_form_rejects_bots_and_bad_input(client, app):
    pub = app.test_client()
    pub.post("/report", data={"entity_name": "Bot Corp", "category": "x", "website_url": "http://spam"})
    r = pub.post("/report", data={"entity_name": "", "category": ""})
    assert "report_error=1" in r.headers["Location"]
    admin = app.test_client()
    from tests.conftest import login
    login(admin)
    assert admin.get("/api/reports").get_json() == []


def test_home_page_has_report_form_and_bookmarklet(client):
    html = client.get("/").get_data(as_text=True)
    assert "Report a business" in html and "javascript:(function()" in html and "Check this site" in html


def test_demo_mode_seeds_fictional_complaints_once(tmp_path):
    from safeguard import create_app
    from safeguard.config import TestConfig

    class Demo(TestConfig):
        INSTANCE_DIR = tmp_path
        DATA_DIR = tmp_path / "data"
        DEMO_SEED = True

    app = create_app(Demo)
    with app.app_context():
        first = Complaint.query.count()
        assert first > 50 and cp.lookup("zenith-crypto.example")["verdict"] == "unsafe"
    app2 = create_app(Demo)                    # second start must not duplicate
    with app2.app_context():
        assert Complaint.query.count() == first
    app.extensions["safeguard"].manager.stop_all()
