"""Capture a full set of project screenshots (public site + operator console) for posts and the README.

    python scripts/demo_video/make_screenshots.py [OUT_DIR]

Runs the app on a COPY of the demo instance (instance/video_demo: fictional complaints and watchlist, never your real data), with the
stock-footage fake cameras, then drives system Chrome. Face photos are blurred by CSS so no real face appears. Output goes to
docs/linkedin/screens (git-ignored)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PORT, STREAM = 5073, 8099
BASE = f"http://127.0.0.1:{PORT}"
PASSWORD = "demo-video-pass-2026"  # throw-away instance only
BLUR = ".match-row img, #preview, #p-photos img, #p-summary img, .person img, .cards img, .thumb { filter: blur(16px) !important; }"
CLIPS = [("fight", 5.0), ("knife", 5.5), ("gun", 7.2), ("rifle", 6.5), ("normal", 5.0)]


def wait(url: str, tries: int = 150) -> None:
    for _ in range(tries):
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise SystemExit(f"not up: {url}")


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "linkedin" / "screens"
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="sg_shots_"))
    inst = tmp / "inst"
    shutil.copytree(ROOT / "instance" / "video_demo", inst)
    env = dict(os.environ, SAFEGUARD_INSTANCE_DIR=str(inst), SAFEGUARD_DATA_DIR=str(inst / "data"), SAFEGUARD_AUTOSTART_CAMERAS="false")
    log = (tmp / "app.log").open("w")
    stream = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "demo_video" / "stream_server.py"), str(ROOT / "instance" / "demo_media" / "scenes.json"), "--port", str(STREAM)])
    app = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--dev", "--port", str(PORT)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    n = [0]

    def shot(pg, name: str, full: bool = False) -> None:
        n[0] += 1
        pg.screenshot(path=str(out / f"{n[0]:02d}-{name}.png"), full_page=full)
        print("shot", n[0], name, flush=True)

    try:
        wait(BASE + "/healthz")
        for _ in range(300):  # models warm up
            if "models warmed up" in (tmp / "app.log").read_text(errors="ignore"):
                break
            time.sleep(1)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required", "--hide-scrollbars"])
            ctx = b.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5, bypass_csp=True)
            ctx.set_default_navigation_timeout(120000)
            ctx.set_default_timeout(120000)
            pg = ctx.new_page()

            # ---------------- public site
            pg.goto(BASE + "/", wait_until="domcontentloaded")
            pg.wait_for_load_state("networkidle")
            pg.evaluate("document.querySelectorAll('.reveal').forEach(e=>e.classList.add('in'))")
            time.sleep(1.2)
            shot(pg, "home-hero")
            pg.evaluate("document.documentElement.style.scrollBehavior='auto'; document.querySelector('#features').scrollIntoView({block:'start'}); window.scrollBy(0,-110)")
            time.sleep(1)
            shot(pg, "home-capabilities")
            pg.evaluate("document.querySelector('#responsible').scrollIntoView({block:'start'}); window.scrollBy(0,-110)")
            time.sleep(1)
            shot(pg, "home-responsible-ai")

            # ---------------- interactive demo, one frame per clip at the alert moment
            pg.goto(BASE + "/demo", wait_until="domcontentloaded")
            pg.wait_for_selector(".demo-tile")
            pg.wait_for_function("[...document.querySelectorAll('.demo-tile video')].every(v => v.readyState >= 2)")
            time.sleep(1)
            for i, (name, at) in enumerate(CLIPS):
                pg.locator(".demo-tile").nth(i).click()
                pg.wait_for_function("document.getElementById('vid').readyState >= 3")
                pg.evaluate(f"(()=>{{const v=document.getElementById('vid'); v.currentTime={at};}})()")
                time.sleep(2.2)
                pg.evaluate("document.getElementById('vid').pause(); window.scrollTo(0, document.querySelector('.demo-stage').offsetTop - 96)")
                time.sleep(0.8)
                shot(pg, f"demo-{name}")

            # ---------------- public fraud check
            pg.goto(BASE + "/?q=Brightpath+Holdings+Ltd#check", wait_until="domcontentloaded")
            pg.wait_for_load_state("networkidle")
            pg.evaluate("document.querySelectorAll('.reveal').forEach(e=>e.classList.add('in'))")
            pg.evaluate("(document.querySelector('#check .result')||document.querySelector('#check .panel')).scrollIntoView({block:'center'})")
            time.sleep(1)
            shot(pg, "fraud-check-unsafe")

            # ---------------- operator console
            pg.goto(BASE + "/login", wait_until="domcontentloaded")
            pg.fill("#username", "admin")
            pg.fill("#password", PASSWORD)
            pg.click("button[type=submit]")
            pg.wait_for_url("**/console/**")
            pg.wait_for_selector(".tile button", timeout=60000)
            time.sleep(1.5)
            for i in range(pg.locator(".tile button:has-text('Start')").count()):
                pg.locator(".tile button:has-text('Start')").first.click()
                time.sleep(1.5)
            pg.add_style_tag(content=BLUR)
            time.sleep(70)  # let the cameras play long enough for alerts
            try:  # wait until a camera is actually in alarm so the wall shows a live incident
                pg.wait_for_selector(".tile.violent", timeout=150000)
                time.sleep(3)
            except Exception:
                pass
            pg.add_style_tag(content=".tiles{grid-template-columns:repeat(2,minmax(0,1fr))!important}")
            pg.evaluate("window.scrollTo(0, 0)")
            time.sleep(1.5)
            shot(pg, "console-live-wall")
            pg.goto(BASE + "/console/alerts", wait_until="domcontentloaded")
            pg.add_style_tag(content=BLUR)
            time.sleep(3)
            shot(pg, "console-alerts")
            pg.goto(BASE + "/console/analytics", wait_until="domcontentloaded")
            pg.add_style_tag(content=BLUR)
            time.sleep(2.5)
            shot(pg, "console-analytics")
            pg.goto(BASE + "/console/analysis", wait_until="domcontentloaded")
            pg.set_input_files("#file", str(ROOT / "instance" / "demo_media" / "fight" / "6296278.mp4"))
            pg.click("#go")
            for _ in range(240):
                txt = pg.inner_text("#report").lower()
                if "peak" in txt and "scanning" not in txt:
                    break
                time.sleep(1)
            time.sleep(3)
            shot(pg, "console-video-analysis")
            pg.goto(BASE + "/console/watchlist", wait_until="domcontentloaded")
            pg.add_style_tag(content=BLUR)
            time.sleep(2.5)
            shot(pg, "console-watchlist")
            first = pg.locator(".person a, .cards a").first
            if first.count():
                first.click()
                pg.add_style_tag(content=BLUR)
                time.sleep(2.5)
                shot(pg, "console-case-file")
            pg.goto(BASE + "/console/complaints", wait_until="domcontentloaded")
            time.sleep(2.5)
            shot(pg, "console-complaints")
            pg.goto(BASE + "/console/cameras", wait_until="domcontentloaded")
            time.sleep(2)
            shot(pg, "console-cameras")
            ctx.close()

            # ---------------- phone
            m = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True, bypass_csp=True)
            m.set_default_navigation_timeout(120000)
            m.set_default_timeout(120000)
            mp = m.new_page()
            mp.goto(BASE + "/", wait_until="domcontentloaded")
            mp.wait_for_load_state("networkidle")
            mp.evaluate("document.querySelectorAll('.reveal').forEach(e=>e.classList.add('in'))")
            time.sleep(1.2)
            shot(mp, "mobile-home")
            mp.goto(BASE + "/demo", wait_until="domcontentloaded")
            mp.wait_for_selector(".demo-tile")
            mp.locator(".demo-tile").nth(2).click()
            mp.wait_for_function("document.getElementById('vid').readyState >= 3")
            mp.evaluate("document.getElementById('vid').currentTime=9")
            time.sleep(2)
            mp.evaluate("document.getElementById('vid').pause(); window.scrollTo(0, document.querySelector('.demo-stage').offsetTop - 70)")
            time.sleep(0.8)
            shot(mp, "mobile-demo")
            b.close()
    finally:
        for pr in (app, stream):
            pr.terminate()
        shutil.rmtree(tmp, ignore_errors=True)
    print("done:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
