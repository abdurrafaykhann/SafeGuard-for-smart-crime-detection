"""Record the SafeGuard demo video end to end: fake IP cameras -> live app -> scripted browser session -> narrated MP4.

    python scripts/demo_video/make_demo.py [--phase all|prepare|record|assemble] [--dry]

Everything runs on this machine against a throw-away instance (instance/video_demo), never your real data:
  * stock clips (Pexels licence) are served as MJPEG "cameras" by stream_server.py
  * the app runs on :5058 with its own database, a demo admin and the fictional complaints
  * a Playwright-driven Chrome records the screen; captions/cursor/quiz overlays are drawn in the page
  * Windows speech narrates; ffmpeg mixes narration and encodes H.264 + SRT captions
Face photos are blurred by CSS during recording; no real face is ever visible."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import narration  # noqa: E402

MEDIA = ROOT / "instance" / "demo_media"
BUILD = ROOT / "instance" / "video_demo_build"
INST = ROOT / "instance" / "video_demo"
PY = sys.executable
APP_PORT, STREAM_PORT = 5058, 8099
BASE = f"http://127.0.0.1:{APP_PORT}"
PASSWORD = "demo-video-pass-2026"
DATASET = Path(r"C:\Users\Hp\Desktop\smart crime detection system - Copy (10) - Copy\facial_recognition_crime (1) - working project\dataset")
W, H = 1280, 720          # video size
VW, VH = 1440, 810        # browser viewport (a little roomier than the video; scaled down on capture)
EST_TOTAL = 215.0  # only drives the progress bar
HOOK_CSS = ("#kpis{display:none!important}.live-grid{grid-template-columns:1fr!important}.live-grid>aside{display:none!important}"
            ".tiles{grid-template-columns:repeat(3,minmax(0,1fr))!important}")
WALL_CSS = ".tiles{grid-template-columns:repeat(2,minmax(0,1fr))!important}"


# ------------------------------------------------------------------------------------------- prepare
def prepare() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    scenes = MEDIA / "scenes"
    scenes.mkdir(exist_ok=True)
    calm = MEDIA / "normal" / "6307658.mp4"
    run = lambda *a: subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *a], check=True)
    # Portrait stock clips stay letterboxed (stream_server pads them). Cropping or blurred-padding them to fill the tile
    # was tried and made the calm clip raise violence/weapon alerts, so framing matters to these models (see README).
    enc = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart"]
    run("-i", str(calm), "-t", "6", *enc, str(scenes / "calm_6s.mp4"))
    run("-i", str(calm), "-vf", "hflip", *enc, str(scenes / "calm_flipped.mp4"))
    (MEDIA / "scenes.json").write_text(json.dumps({
        "lobby": ["normal/6307658.mp4"],
        "gym": ["scenes/calm_6s.mp4", "fight/6296278.mp4"],
        "carpark": ["scenes/calm_flipped.mp4"],
        "entrance": ["scenes/calm_6s.mp4", "gun/6092110.mp4", "knife/30166089.mp4", "knife/5735079.mp4"],
    }, indent=2), encoding="utf-8")
    # fresh instance with the fictional watchlist (public-figure photos get invented names and are blurred on screen)
    shutil.rmtree(INST, ignore_errors=True)
    INST.mkdir(parents=True)
    env = app_env()
    subprocess.run([PY, str(ROOT / "scripts" / "enroll_demo_dataset.py"), str(DATASET), "--replace", "--holdout", "--exclude", "me abdul rafay"], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("narration...", flush=True)
    d = narration.synth_all(BUILD / "audio")
    (BUILD / "durations.json").write_text(json.dumps(d), encoding="utf-8")
    print(f"prepared; narration {sum(d.values()):.0f}s", flush=True)


def app_env() -> dict:
    env = dict(os.environ)
    env.update({"SAFEGUARD_INSTANCE_DIR": str(INST), "SAFEGUARD_ADMIN_PASSWORD": PASSWORD, "SAFEGUARD_DEMO_SEED": "true",
                "SAFEGUARD_ALERT_COOLDOWN": "20"})
    return env


# ------------------------------------------------------------------------------------------- recording
class Rec:
    def __init__(self, page, durations: dict[str, float], t0: float) -> None:
        self.page, self.dur, self.t0 = page, durations, t0
        self.events: list[dict] = []
        self.cuts: list[tuple[float, float]] = []  # dead time (waiting on analysis jobs ...) removed in assemble()

    def now(self) -> float:
        return time.time() - self.t0

    def ov(self, **patch) -> None:
        patch.setdefault("prog", min(self.now() / EST_TOTAL, 1.0))
        self.page.evaluate("p => window.__ov && window.__ov.set(p)", patch)

    @contextmanager
    def line(self, key: str, chapter: str | None = None, caption: bool = True, tail: float = 0.5):
        start = self.now()
        print(f"[{start:6.1f}s] line {key}", flush=True)
        patch = {"cap": narration.LINES[key] if caption else ""}
        if chapter is not None:
            patch["chap"] = chapter
        self.ov(**patch)
        self.events.append({"key": key, "t": start, "dur": self.dur[key], "text": narration.LINES[key]})
        yield
        remaining = start + self.dur[key] + tail - self.now()
        if remaining > 0:
            time.sleep(remaining)
        self.ov(cap="")

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)
        self.ov()

    def move_click(self, selector: str, nth: int = 0, pause: float = 0.35) -> None:
        print(f"        click {selector} #{nth}", flush=True)
        loc = self.page.locator(selector).nth(nth)
        try:
            loc.scroll_into_view_if_needed(timeout=6000)
            box = loc.bounding_box(timeout=6000)
            self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=28)
            time.sleep(pause)
            self.page.mouse.down()
            self.page.mouse.up()
        except Exception as exc:  # fall back to a plain DOM click so the tour never stalls
            print(f"        click fallback for {selector}: {str(exc)[:80]}", flush=True)
            self.page.evaluate("([s, n]) => document.querySelectorAll(s)[n] && document.querySelectorAll(s)[n].click()", [selector, nth])
        time.sleep(0.25)

    def type(self, selector: str, text: str, delay: int = 55) -> None:
        loc = self.page.locator(selector).first
        box = loc.bounding_box()
        self.page.mouse.move(box["x"] + 40, box["y"] + box["height"] / 2, steps=20)
        self.page.mouse.down()
        self.page.mouse.up()
        loc.fill("")
        loc.type(text, delay=delay)

    def start_tile(self, name: str) -> None:
        """Click the Start button on a camera tile, like an operator would."""
        btn = self.page.locator(".tile", has_text=name).locator("button").first
        box = btn.bounding_box(timeout=8000)
        self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=22)
        time.sleep(0.25)
        self.page.mouse.down()
        self.page.mouse.up()
        time.sleep(0.3)

    def cut_wait(self, js: str, timeout: float, lead: float = 1.0) -> bool:
        """Wait for something the machine is slow to produce, and cut the waiting out of the final video."""
        time.sleep(lead)
        self.ov()
        c0 = self.now()
        ok = wait_for(self.page, js, timeout)
        c1 = self.now()
        if c1 - c0 > 1.5:
            self.cuts.append((c0, c1 - 0.4))
        return ok

    def settle(self, scroll_check: bool = True) -> None:
        """After a click that navigates: let the new page load, then scroll the checker into view."""
        time.sleep(1.0)
        self.page.wait_for_load_state("load")
        for _ in range(5):
            try:
                if scroll_check:
                    self.page.evaluate("document.getElementById('check').scrollIntoView()")
                return
            except Exception:
                time.sleep(0.6)

    def card(self, html: str | None) -> None:
        self.ov(card=html)

    def quiz(self, question: str, options: list[str], seconds: int, answer: int | None = None, answer_text: str = "") -> None:
        def render(left: int | None, reveal: bool) -> str:
            opts = "".join(f'<div class="opt{(" right" if i == answer else " wrong") if reveal else ""}">{o}</div>' for i, o in enumerate(options))
            tail = f'<div class="ans">{answer_text}</div>' if reveal else f'<div class="count">{left}</div>'
            return f'<div class="q">{question}</div><div class="opts">{opts}</div>{tail}'
        for left in range(seconds, 0, -1):
            self.ov(quiz=render(left, False))
            time.sleep(1.0)
        self.ov(quiz=render(None, True))


def api(page, method: str, path: str, body=None):
    return page.evaluate("async ([m, p, b]) => (await window.SG.api(p, { method: m, json: b }))", [method, path, body])


def wait_for(page, js: str, timeout: float, poll: float = 0.5):
    end = time.time() + timeout
    print(f"        wait_for {js[:70]!r} (<= {timeout}s)", flush=True)
    while time.time() < end:
        try:
            if page.evaluate(js):
                return True
        except Exception:  # page mid-navigation
            pass
        time.sleep(poll)
    return False


def record(args) -> None:
    from playwright.sync_api import sync_playwright

    durations = json.loads((BUILD / "durations.json").read_text(encoding="utf-8"))
    overlay = (Path(__file__).parent / "overlay.js").read_text(encoding="utf-8")
    raw = BUILD / "raw"
    shutil.rmtree(raw, ignore_errors=True)
    shutil.rmtree(INST / "data", ignore_errors=True) if False else None

    import socket
    for port in (APP_PORT, STREAM_PORT):
        with socket.socket() as sk:
            if sk.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"port {port} is already in use (a previous demo run left a process behind?)")
    procs = []
    try:
        procs.append(subprocess.Popen([PY, str(Path(__file__).parent / "stream_server.py"), str(MEDIA / "scenes.json"), "--port", str(STREAM_PORT)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen([PY, str(ROOT / "run.py"), "--port", str(APP_PORT)], env=app_env(), cwd=str(ROOT),
                                      stdout=open(BUILD / "app.log", "w"), stderr=subprocess.STDOUT))
        for _ in range(120):
            try:
                import urllib.request
                urllib.request.urlopen(BASE + "/healthz", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("app did not start; see instance/video_demo_build/app.log")
        print("app up; waiting for the models to warm up (cold start can take a couple of minutes)...", flush=True)
        for _ in range(360):
            if "models warmed up" in (BUILD / "app.log").read_text(encoding="utf-8", errors="ignore"):
                break
            time.sleep(1)
        else:
            raise SystemExit("models never finished warming up; see app.log")
        print("models warm", flush=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
            # 1) log in off camera, keep the session
            ctx0 = browser.new_context(viewport={"width": VW, "height": VH}, bypass_csp=True)
            pg0 = ctx0.new_page()
            pg0.goto(BASE + "/login")
            pg0.fill("#username", "admin")
            pg0.fill("#password", PASSWORD)
            pg0.click("button[type=submit]")
            pg0.wait_for_url("**/console/**")
            state = ctx0.storage_state()
            # cameras (created off camera, started on camera)
            ids = {}
            for name, scene, loc in (("Lobby", "lobby", "Ground floor"), ("Gym floor", "gym", "Level 1"), ("Car park", "carpark", "Outside")):
                ids[name] = api(pg0, "POST", "/api/cameras", {"name": name, "source": f"http://127.0.0.1:{STREAM_PORT}/{scene}.mjpg", "location": loc,
                                                              "detect_violence": True, "detect_faces": False})["id"]
            ctx0.close()

            # 2) the recorded session
            ctx = browser.new_context(viewport={"width": VW, "height": VH}, storage_state=state, bypass_csp=True,
                                      record_video_dir=str(raw), record_video_size={"width": W, "height": H}, accept_downloads=True)
            ctx.add_init_script(overlay)
            page = ctx.new_page()
            t0 = time.time()
            rec = Rec(page, durations, t0)
            run_script(rec, page, ids)
            rec.ov(cap="", chap="", card=None, quiz=None)
            ctx.close()
            browser.close()
        (BUILD / "events.json").write_text(json.dumps({"events": rec.events, "cuts": rec.cuts, "total": time.time() - t0}, indent=2), encoding="utf-8")
        print("recorded", flush=True)
    finally:
        for pr in procs:
            pr.terminate()


def violence_row(page) -> int:
    i = page.evaluate("[...document.querySelectorAll('#rows tr')].findIndex(r => /violence|sustained/i.test(r.innerText))")
    return max(i, 0)


def title_card(title: str, sub: str = "", extra: str = "") -> str:
    return f'<h1>{title}</h1><p>{sub}</p>{extra}'


def run_script(rec: Rec, page, ids: dict) -> None:
    """The scripted tour. Anything the narration says is on screen and real."""
    # ---------------------------------------------------------------- HOOK: spot the incident (live wall)
    page.goto(BASE + "/console/")
    page.add_style_tag(content=HOOK_CSS)
    rec.ov(chap="Live wall", zoom=1.12)
    wait_for(page, "document.querySelectorAll('.tile').length >= 3", 15)
    t_live = time.time()
    with rec.line("hook", chapter="Spot the incident"):
        for name in ("Lobby", "Car park", "Gym floor"):
            rec.start_tile(name)
        rec.wait(0.6)
        rec.quiz("Which camera has an incident?", ["Lobby", "Gym floor", "Car park"], 6, None, "Time is up. Watching the cameras...")
    ok = rec.cut_wait("document.querySelectorAll('#feed .feed-item').length > 0", 240, lead=0.5)
    print("gym incident visible after", round(time.time() - t_live, 1), "s", ok, flush=True)
    rec.quiz("Which camera has an incident?", ["Lobby", "Gym floor", "Car park"], 0, 1, "Gym floor: flagged by the AI")
    with rec.line("hook_reveal"):
        rec.wait(1.0)
    rec.ov(quiz=None)

    # ---------------------------------------------------------------- INTRO
    rec.card(title_card("Safe<span>Guard</span>", "AI surveillance console · public fraud check",
                        '<p style="font-size:20px">Final year project · University of Hertfordshire · Abdul Rafay Khan</p>'))
    with rec.line("intro", chapter="Intro"):
        rec.wait(1.0)
    rec.card(None)

    # ---------------------------------------------------------------- LIVE WALL + incident feed
    page.goto(BASE + "/console/")
    page.add_style_tag(content=WALL_CSS)
    with rec.line("live", chapter="Live wall"):
        page.mouse.move(640, 300, steps=30)
        rec.wait(1.0)
    with rec.line("feed"):
        page.mouse.move(1060, 420, steps=30)
        rec.wait(0.5)

    # ---------------------------------------------------------------- ALERT DETAIL, ACK, EVIDENCE REPORT
    for name in ("Lobby", "Gym floor", "Car park"):  # stop live updates so list rows stay still while we click
        api(page, "POST", f"/api/cameras/{ids[name]}/stop")
    with rec.line("alert", chapter="Alerts"):
        rec.move_click("a[href='/console/alerts']")
        page.wait_for_url("**/console/alerts")
        wait_for(page, "document.querySelectorAll('#rows img').length > 0", 10)
        rec.wait(0.5)
        rec.move_click("#rows img", nth=violence_row(page))
        wait_for(page, "document.getElementById('viewer-clip') && !document.getElementById('viewer-clip').hidden", 15)
        page.evaluate("document.getElementById('viewer-clip').play().catch(() => {})")
        rec.wait(3.0)
    with rec.line("ack"):
        rec.move_click("#viewer-close")
        rec.wait(0.4)
        ack = page.locator("#rows tr").nth(violence_row(page)).locator("button")
        if ack.count():
            box = ack.first.bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=25)
            page.mouse.down(); page.mouse.up()
            rec.wait(1.2)
        rec.move_click("#rows img", nth=violence_row(page))
        rec.wait(1.0)
        pdf_png = None
        try:
            box = page.locator("#viewer-report").bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=25)
            with page.expect_download(timeout=20000) as dl:
                page.mouse.down(); page.mouse.up()
            pdf = BUILD / "report.pdf"
            dl.value.save_as(str(pdf))
            import fitz
            pix = fitz.open(str(pdf))[0].get_pixmap(dpi=90)
            pdf_png = BUILD / "report.png"
            pix.save(str(pdf_png))
        except Exception as exc:
            print("pdf capture failed:", exc, flush=True)
    if pdf_png:
        import base64
        b64 = base64.b64encode(pdf_png.read_bytes()).decode()
        rec.card(f'<p style="font-size:22px">Evidence report (PDF)</p><img src="data:image/png;base64,{b64}" style="max-height:640px">')
        rec.wait(5.0)
        rec.card(None)
    rec.move_click("#viewer-close")

    # ---------------------------------------------------------------- WEAPON DETECTION
    ids["Entrance"] = api(page, "POST", "/api/cameras", {"name": "Entrance", "source": f"http://127.0.0.1:{STREAM_PORT}/entrance.mjpg",
                                                         "location": "Front door", "detect_violence": True, "detect_faces": False})["id"]
    page.goto(BASE + "/console/")
    page.add_style_tag(content=WALL_CSS)
    rec.ov(chap="Weapon detection")
    wait_for(page, "document.querySelectorAll('.tile').length >= 4", 15)
    page.evaluate("[...document.querySelectorAll('.tile')].forEach(t => { if (!/Entrance/.test(t.innerText)) t.style.display = 'none'; })")
    rec.start_tile("Entrance")
    rec.cut_wait("[...document.querySelectorAll('#feed .feed-item')].some(e => /weapon detected/i.test(e.textContent))", 240, lead=1.0)
    with rec.line("weapon"):
        rec.wait(2.0)
    for name in ("Lobby", "Gym floor", "Car park", "Entrance"):
        api(page, "POST", f"/api/cameras/{ids[name]}/stop")
    rec.wait(5.5)  # the 5 s after-the-alert part of the evidence clip finishes recording
    page.goto(BASE + "/console/alerts")
    wait_for(page, "document.querySelectorAll('#rows img').length > 0", 10)
    widx = page.evaluate("[...document.querySelectorAll('#rows tr')].findIndex(r => /weapon/i.test(r.innerText))")
    rec.move_click("#rows img", nth=max(widx, 0))
    wait_for(page, "document.getElementById('viewer-clip') && !document.getElementById('viewer-clip').hidden", 15)
    page.evaluate("document.getElementById('viewer-clip').play().catch(() => {})")
    rec.wait(4.0)
    rec.move_click("#viewer-close")

    # ---------------------------------------------------------------- ANALYTICS
    with rec.line("analytics", chapter="Analytics"):
        page.goto(BASE + "/console/analytics")
        rec.wait(1.0)
        page.mouse.move(700, 380, steps=30)
        rec.wait(1.0)

    # ---------------------------------------------------------------- VIDEO ANALYSIS
    with rec.line("video", chapter="Video analysis"):
        page.goto(BASE + "/console/analysis")
        rec.wait(0.8)
        page.set_input_files("#file", str(MEDIA / "fight" / "6296278.mp4"))
        rec.wait(0.8)
        rec.move_click("#go")
        rec.wait(2.5)
    rec.cut_wait("document.getElementById('report') && /peak risk/i.test(document.getElementById('report').innerText)", 240, lead=0.5)
    with rec.line("video_result"):
        rec.wait(2.5)

    # ---------------------------------------------------------------- FACE SEARCH (faces blurred)
    rec.ov(blur=True, chap="Face search")
    page.goto(BASE + "/console/search")
    ident = json.loads((INST / "demo_identity_map.json").read_text(encoding="utf-8"))
    folder = "MuhammadFahmi"  # checked beforehand: its held-out photo ranks first at ~80% similarity
    info = ident[folder]
    query = DATASET / folder / info["held_out"]
    with rec.line("faces", chapter="Face search"):
        rec.wait(0.8)
        page.set_input_files("#file", str(query))
        wait_for(page, "document.getElementById('results').innerText.includes('similar')", 20)
        rec.wait(2.5)
        page.mouse.move(640, 420, steps=25)
        rec.wait(1.0)
    with rec.line("faces_acc"):
        rec.move_click("#results a")
        wait_for(page, "document.getElementById('p-summary') && document.getElementById('p-summary').innerText.length > 20", 15)
        rec.wait(1.0)
    rec.ov(blur=False)

    # ---------------------------------------------------------------- FRAUD CHECK (public home page)
    page.goto(BASE + "/")
    rec.ov(chap="Fraud check", zoom=1.3)
    with rec.line("quiz_fraud"):
        page.evaluate("document.getElementById('check').scrollIntoView({behavior: 'smooth'})")
        rec.wait(1.0)
        rec.type("#check input[name=q]", "zenith-crypto.example", 60)
    rec.quiz("Is this website safe to pay?", ["Safe", "Not safe"], 3, None)
    rec.move_click("#check button[type=submit]")
    rec.settle()
    rec.quiz("Is this website safe to pay?", ["Safe", "Not safe"], 0, 1, "Not safe: 13 complaints on record")
    with rec.line("fraud_reveal"):
        rec.wait(1.5)
    rec.ov(quiz=None)
    with rec.line("fraud_caution"):
        rec.type("#check input[name=q]", "Brightpath Holdngs", 60)
        rec.move_click("#check button[type=submit]")
        rec.settle()
        rec.wait(2.5)
        rec.type("#check input[name=q]", "Sunrise Gold Traders", 60)
        rec.move_click("#check button[type=submit]")
        rec.settle()
        rec.wait(2.0)
    with rec.line("fraud_report"):
        rec.move_click("#report summary")
        rec.wait(0.5)
        rec.type("#report input[name=entity_name]", "Sunrise Gold Traders", 45)
        rec.type("#report input[name=entity_domain]", "sunrise-gold.example", 40)
        rec.type("#report input[name=category]", "Investment scam", 40)
        rec.type("#report textarea[name=description]", "Asked for upfront fees before releasing returns.", 25)
        rec.move_click("#report button[type=submit]")
        rec.settle(scroll_check=False)
        rec.wait(1.5)
    # staff approves
    page.goto(BASE + "/console/complaints")
    wait_for(page, "document.querySelectorAll('#r-list button').length > 0", 10)
    rec.ov(chap="Staff review", zoom=1.12)
    rec.wait(1.0)
    rec.move_click("#r-list button", nth=0)
    rec.wait(1.5)
    page.goto(BASE + "/?q=Sunrise+Gold+Traders")
    page.evaluate("document.getElementById('check').scrollIntoView()")
    rec.ov(chap="Fraud check", zoom=1.3)
    with rec.line("fraud_after"):
        rec.wait(2.0)

    # ---------------------------------------------------------------- UNDER THE HOOD + OUTRO
    rec.card(title_card("Built to be trusted", "", '<div class="kv"><div><b>83</b><small>automated tests</small></div><div><b>scrypt</b><small>hashed passwords</small></div>'
                        '<div><b>CSRF</b><small>+ rate limits</small></div><div><b>Audit log</b><small>every action</small></div>'
                        '<div><b>Retention</b><small>auto-delete</small></div><div><b>SHA-256</b><small>evidence hashing</small></div></div>'))
    with rec.line("tech", chapter="Engineering"):
        rec.wait(1.0)
    rec.card(title_card("Measured, not guessed", "Violence classifier on 400 held-out validation clips",
                        '<div class="kv"><div><b>79%</b><small>recall</small></div><div><b>81%</b><small>precision</small></div>'
                        '<div><b>97.7%</b><small>face top-1 (54 people, held-out photos)</small></div></div>'
                        '<p style="font-size:18px;margin-top:22px">Limits (close-range webcams, staged footage) are documented in the README.</p>'))
    with rec.line("honest"):
        rec.wait(1.0)
    rec.card(title_card("Thanks for watching", "SafeGuard · github.com/abdurrafaykhann/SafeGuard-for-smart-crime-detection",
                        '<p style="font-size:20px;margin-top:20px">Footage: Pexels (free licence) · Faces blurred · All records fictional</p>'
                        '<p style="font-size:30px;color:#46e1ff;margin-top:24px">Which feature would you build next?</p>'))
    with rec.line("outro", chapter="Thanks"):
        rec.wait(1.0)


# ------------------------------------------------------------------------------------------- assemble
def assemble() -> None:
    ev = json.loads((BUILD / "events.json").read_text(encoding="utf-8"))
    raw = sorted((BUILD / "raw").glob("*.webm"))
    if not raw:
        raise SystemExit("no raw recording")
    src = raw[-1]
    out = ROOT / "docs" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    lead = -0.4  # measured: captions appear ~0.4 s (video clock) before the script clock says so; video time = script time + lead
    cuts = sorted((a + lead, b + lead) for a, b in ev.get("cuts", []))
    total = ev["total"] + lead
    ack = next((e for e in ev["events"] if e["key"] == "ack"), None)
    card = make_pdf_card() if ack else None
    card_at = None
    if ack and card:
        ack_end = ack["t"] + ack["dur"]
        nxt = min((a_ for a_, b_ in ev.get("cuts", []) if a_ > ack_end), default=None)
        card_at = ack_end + 1.2
        if nxt is not None and nxt - ack_end > 10:  # the failed download wait: drop everything after the card
            cuts.append((ack_end + 7.4 + lead, nxt - 1.0 + lead))
            cuts.sort()

    def shifted(t: float) -> float:
        t += lead
        return t - sum(min(max(t - a, 0.0), b - a) for a, b in cuts)

    # video: keep everything except the cut ranges
    keep, cur = [], 0.0
    for a, b in cuts:
        keep.append((cur, a))
        cur = b
    keep.append((cur, total + 5))
    vfilters = [f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]" for i, (a, b) in enumerate(keep)]
    if card_at is not None:
        t1 = shifted(card_at)
        vfilters.append("".join(f"[v{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=1:a=0[vcat]")
        vfilters.append(f"[vcat][{len(ev['events']) + 1}:v]overlay=0:0:enable='between(t,{t1:.2f},{t1 + 6:.2f})'[vout]")
    else:
        vfilters.append("".join(f"[v{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=1:a=0[vout]")

    inputs, afilters, labels = [], [], []
    for i, e in enumerate(ev["events"]):
        inputs += ["-i", str(BUILD / "audio" / f"{e['key']}.wav")]
        ms = int(shifted(e["t"]) * 1000)
        afilters.append(f"[{i + 1}:a]adelay={ms}|{ms},volume=1.0[a{i}]")
        labels.append(f"[a{i}]")
    afilters.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0,alimiter=limit=0.95[aout]")
    mp4 = out / "safeguard-demo.mp4"
    if card_at is not None:
        inputs += ["-loop", "1", "-i", str(card)]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), *inputs, "-filter_complex", ";".join(vfilters + afilters),
                    "-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p", "-r", "30",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(mp4)], check=True)

    def ts(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, sec = divmod(rem, 60)
        return f"{int(h):02}:{int(m):02}:{int(sec):02},{int((sec % 1) * 1000):03}"
    srt = []
    for n, e in enumerate(ev["events"], 1):
        t = shifted(e["t"])
        srt.append(f"{n}\n{ts(t)} --> {ts(t + e['dur'])}\n{e['text']}\n")
    (out / "safeguard-demo.srt").write_text("\n".join(srt), encoding="utf-8")
    chapters = {}
    for e in ev["events"]:
        chapters[e["key"]] = shifted(e["t"])
    (out / "chapters.json").write_text(json.dumps(chapters, indent=2), encoding="utf-8")
    vtt = ["WEBVTT", ""]
    for e in ev["events"]:
        t = shifted(e["t"])
        vtt += [f"{ts(t).replace(',', '.')} --> {ts(t + e['dur']).replace(',', '.')}", e["text"], ""]
    (out / "safeguard-demo.vtt").write_text(chr(10).join(vtt), encoding="utf-8")
    titles = {"hook": "Spot the incident (quiz)", "hook_reveal": "The reveal", "intro": "What is SafeGuard", "live": "Live wall", "feed": "Incident feed",
              "alert": "Alerts with video evidence", "ack": "Acknowledge and PDF report", "weapon": "Weapon detection", "analytics": "Analytics",
              "video": "Video analysis", "video_result": "Analysis results", "faces": "Face search", "faces_acc": "Case file and accuracy",
              "quiz_fraud": "Fraud check quiz", "fraud_reveal": "Fraud check result", "fraud_caution": "Near misses and unknown names",
              "fraud_report": "Public reports", "fraud_after": "Staff approval", "tech": "Security and engineering", "honest": "Measured accuracy",
              "outro": "Thanks and next steps"}
    chap_list = [{"key": k, "title": titles.get(k, k), "t": round(v, 1)} for k, v in chapters.items()]
    html = (Path(__file__).parent / "page_template.html").read_text(encoding="utf-8").replace("/*CHAPTERS*/[]", json.dumps(chap_list))
    (out / "index.html").write_text(html, encoding="utf-8")
    print("wrote", mp4)


def make_pdf_card() -> Path | None:
    """The live recording did not capture the PDF download, so show a real report the app produced earlier (same code, an earlier alert)."""
    src = BUILD / "report.png"
    if not src.exists():
        return None
    from PIL import Image, ImageDraw, ImageFont
    W_, H_ = 1280, 720
    card = Image.new("RGB", (W_, H_), (6, 10, 18))
    d = ImageDraw.Draw(card)
    try:
        f1, f2 = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 30), ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 18)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    d.text((W_ // 2, 34), "One click: the evidence report (PDF)", font=f1, fill=(232, 238, 247), anchor="mt")
    page = Image.open(src).convert("RGB")
    h = 556
    page = page.resize((int(page.width * h / page.height), h), Image.LANCZOS)
    card.paste(page, ((W_ - page.width) // 2, 86))
    d.text((W_ // 2, 698), "Sample report from the same app (earlier alert): snapshot, SHA-256, cue scores, who acknowledged it", font=f2, fill=(159, 176, 198), anchor="mb")
    out = BUILD / "pdf_card.png"
    card.save(out)
    return out


SHORT_CUT = [  # (event, seconds of picture to keep after its narration ends)
    ("hook", 0.4), ("hook_reveal", 0.5), ("intro", 0.3), ("alert", 0.4), ("ack", 6.5), ("weapon", 1.5),
    ("faces", 1.2), ("quiz_fraud", 0.2), ("fraud_reveal", 0.5), ("outro", 0.6)]


def assemble_short() -> None:
    """A ~90 second highlights cut made from the same recording: quiz hook, alert + evidence, weapon, face search, fraud check."""
    ev = json.loads((BUILD / "events.json").read_text(encoding="utf-8"))
    src = sorted((BUILD / "raw").glob("*.webm"))[-1]
    out = ROOT / "docs" / "demo" / "safeguard-demo-90s.mp4"
    lead = -0.4
    by_key = {e["key"]: e for e in ev["events"]}
    segs, t_out = [], 0.0
    for key, extra in SHORT_CUT:
        e = by_key[key]
        a0 = max(e["t"] + lead - 0.3, 0.0)
        b0 = e["t"] + lead + e["dur"] + extra
        segs.append({"key": key, "a": a0, "b": b0, "out": t_out + 0.3})
        t_out += b0 - a0
    vf = [f"[0:v]trim=start={g['a']:.3f}:end={g['b']:.3f},setpts=PTS-STARTPTS[v{i}]" for i, g in enumerate(segs)]
    vf.append("".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[vout]")
    inputs, af, labels = [], [], []
    for i, g in enumerate(segs):
        inputs += ["-i", str(BUILD / "audio" / f"{g['key']}.wav")]
        ms = int(g["out"] * 1000)
        af.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
        labels.append(f"[a{i}]")
    af.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0,alimiter=limit=0.95[aout]")
    card = make_pdf_card()
    if card:
        ack_seg = next(g for g in segs if g["key"] == "ack")
        t1 = ack_seg["out"] + by_key["ack"]["dur"] + 0.8
        vf[-1] = vf[-1].replace("[vout]", "[vcat]")
        vf.append(f"[vcat][{len(segs) + 1}:v]overlay=0:0:enable='between(t,{t1:.2f},{t1 + 5:.2f})'[vout]")
        inputs += ["-loop", "1", "-i", str(card)]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), *inputs, "-filter_complex", ";".join(vf + af),
                    "-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p", "-r", "30",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)], check=True)
    print("wrote", out, f"({t_out:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "prepare", "record", "assemble", "short"])
    a = ap.parse_args()
    if a.phase in ("all", "prepare"):
        prepare()
    if a.phase in ("all", "record"):
        record(a)
    if a.phase in ("all", "assemble"):
        assemble()
    if a.phase in ("all", "short"):
        assemble_short()


if __name__ == "__main__":
    main()
