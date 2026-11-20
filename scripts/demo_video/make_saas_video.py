"""Record a polished, narrated product-demo video of the website's /demo replay, home page and fraud check.

    python scripts/demo_video/make_saas_video.py

Starts the app on a throw-away database (demo data only, no heavy models needed: the replay is pre-recorded detector output),
drives system Chrome with Playwright at 1920x1080, adds captions/title cards, narrates with the offline Windows voice and
assembles an H.264 MP4 with ffmpeg. Output: safeguard/static/demo/safeguard-product-demo.mp4 (+ .vtt captions).
Everything on screen is real: the replayed detections are the recorded output of the real detectors on stock clips."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration import synth  # noqa: E402

PORT = 5072
BASE = f"http://127.0.0.1:{PORT}"
W, H = 1920, 1080
OUT = ROOT / "safeguard" / "static" / "demo"
BUILD = Path(tempfile.gettempdir()) / "sg_saas_build"
FFMPEG = "ffmpeg"
LEAD = 0.15  # seconds the audio starts after a scene begins

LINES = {
    "title": "SafeGuard. A real-time AI surveillance console that flags incidents, and leaves every decision to a human.",
    "home": "It watches multiple camera feeds, scores risk with a trained classifier, and looks for weapons with a dedicated detector. Let's watch it work on real footage.",
    "fight": "First, a sparring session. The risk score climbs, and SafeGuard raises a violence alert within three seconds.",
    "knife": "Next, a knife. The weapon detector confirms it, boxes it, and raises a critical alert.",
    "gun": "A handgun in a car park. The weapon alert fires first, then a violence alert as the scene escalates.",
    "rifle": "A rifle. Detected, boxed and tracked as it moves.",
    "normal": "And ordinary activity stays quiet. No weapon, no violence, no alert.",
    "fraud": "SafeGuard also includes a public fraud check. Type a business name, and if complaints are on record, it is flagged unsafe. No sign-in needed.",
    "resp": "It is built for responsible use: documented consent for every enrolled face, a full audit trail, and a human decision on every alert.",
    "outro": "This replays the system's recorded output on selected stock clips. Measured results, including the failures, are in the project report. SafeGuard. Eyes on every frame, judgement stays human.",
}
CAPS = {
    "title": "", "home": "Multi-camera analysis · trained violence classifier · weapon detector",
    "fight": "Violence detection · alert in under 3 seconds", "knife": "Weapon detection · knife confirmed and tracked",
    "gun": "Weapon alert first, then violence", "rifle": "Long-gun detection and tracking", "normal": "Quiet scene · no alerts raised",
    "fraud": "Public fraud check · no sign-in required", "resp": "Consent · audit trail · human review on every alert",
    "outro": "",
}

OVERLAY = r"""
(() => {
  const install = () => {
  if (window.__ovInstalled) return; window.__ovInstalled = true;
  const css = document.createElement('style'); css.textContent = `
    #ov-cap{position:fixed;left:50%;bottom:30px;transform:translateX(-50%) translateY(14px);z-index:99999;max-width:1000px;padding:11px 22px;border-radius:14px;
      font:600 20px/1.25 Inter,system-ui,sans-serif;color:#fff;text-align:center;letter-spacing:-.01em;background:rgba(8,12,22,.78);backdrop-filter:blur(18px);
      box-shadow:0 20px 60px -10px rgba(0,0,0,.7),0 0 0 1px rgba(255,255,255,.14) inset;opacity:0;transition:opacity .5s,transform .5s}
    #ov-cap.on{opacity:1;transform:translateX(-50%) translateY(0)}
    #ov-card{position:fixed;inset:0;z-index:100000;display:grid;place-items:center;text-align:center;background:radial-gradient(900px 500px at 70% 10%,rgba(93,230,255,.22),transparent 60%),radial-gradient(800px 500px at 10% 90%,rgba(138,125,255,.25),transparent 60%),#05070d;
      opacity:0;pointer-events:none;transition:opacity .7s}
    #ov-card.on{opacity:1}
    #ov-card .m{width:66px;height:66px;border-radius:19px;margin:0 auto 24px;display:grid;place-items:center;background:linear-gradient(135deg,#5de6ff,#8a7dff);box-shadow:0 20px 70px -10px rgba(93,230,255,.7)}
    #ov-card h1{font:700 62px/1.04 "Space Grotesk",Inter,sans-serif;letter-spacing:-.045em;margin:0 0 16px;color:#fff}
    #ov-card h1 span{background:linear-gradient(110deg,#fff,#5de6ff 40%,#8a7dff 75%,#d58bff);-webkit-background-clip:text;color:transparent}
    #ov-card p{font:500 20px/1.4 Inter,sans-serif;color:#a6b2c7;margin:0 auto;max-width:760px}
    #ov-card small{display:block;margin-top:24px;font:500 15px "JetBrains Mono",monospace;color:#5de6ff;letter-spacing:.06em}
  `; document.head.appendChild(css);
  const cap = document.createElement('div'); cap.id = 'ov-cap'; document.body.appendChild(cap);
  const card = document.createElement('div'); card.id = 'ov-card'; document.body.appendChild(card);
  window.__cap = (t) => { if (!t) { cap.classList.remove('on'); return; } cap.textContent = t; cap.classList.add('on'); };
  window.__card = (h, sub, small) => { if (h === null) { card.classList.remove('on'); return; }
    card.replaceChildren(); const wrap = document.createElement('div');
    const m = document.createElement('div'); m.className = 'm'; m.innerHTML = '<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#02161c" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 4.5 6v5.5c0 4.6 3.1 8.2 7.5 9.5 4.4-1.3 7.5-4.9 7.5-9.5V6L12 3Z"/><path d="m9 12 2 2 4-4"/></svg>';
    const hh = document.createElement('h1'); hh.innerHTML = h; const p = document.createElement('p'); p.textContent = sub; wrap.append(m, hh, p);
    if (small) { const s = document.createElement('small'); s.textContent = small; wrap.append(s); }
    card.append(wrap); card.classList.add('on'); };
  };
  if (document.body) install(); else document.addEventListener('DOMContentLoaded', install);
})();
"""


def wait_up() -> None:
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE + "/healthz", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("app did not start")


def main() -> int:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "audio").mkdir(parents=True)
    (BUILD / "raw").mkdir()
    # narration first so scene lengths follow the voice
    dur = {k: synth(t, BUILD / "audio" / f"{k}.wav") for k, t in LINES.items()}
    print("narration:", {k: round(v, 1) for k, v in dur.items()}, flush=True)

    env = dict(os.environ, SAFEGUARD_INSTANCE_DIR=str(BUILD / "inst"), SAFEGUARD_DATA_DIR=str(BUILD / "inst" / "data"), SAFEGUARD_DEMO_SEED="true",
               SAFEGUARD_WEAPON_ENABLED="false", SAFEGUARD_AUTOSTART_CAMERAS="false", SAFEGUARD_ADMIN_PASSWORD="Demo-Only-98765")
    srv = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--dev", "--port", str(PORT)], cwd=ROOT, env=env,
                           stdout=(BUILD / "app.log").open("w"), stderr=subprocess.STDOUT)
    starts: dict[str, float] = {}
    try:
        wait_up()
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required", "--hide-scrollbars"])
            ctx = b.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5, record_video_dir=str(BUILD / "raw"), record_video_size={"width": W, "height": H}, bypass_csp=True)
            ctx.add_init_script(OVERLAY)
            page = ctx.new_page()
            t0 = time.time()

            def scene(key: str, min_len: float = 0.0):
                """Mark a scene start, show its caption, return a function that sleeps out the rest of the scene."""
                starts[key] = time.time() - t0
                page.evaluate(f"window.__cap && window.__cap({json.dumps(CAPS[key])})")
                end = time.time() + max(dur[key] + 0.7, min_len)
                return lambda: time.sleep(max(0, end - time.time()))

            # --- title card over the home page
            page.goto(BASE + "/")
            page.wait_for_load_state("networkidle")
            page.evaluate("window.__card('Eyes on every frame.<br><span>Judgement stays human.</span>', 'AI-assisted video surveillance with explainable alerts and a full audit trail', 'SAFEGUARD · FINAL-YEAR PROJECT')")
            starts["title"] = time.time() - t0
            time.sleep(dur["title"] + 0.9)
            # --- home
            fin = scene("home")
            page.evaluate("window.__card(null)")
            time.sleep(2.4)
            page.evaluate("document.documentElement.style.scrollBehavior='smooth'; document.querySelector('#features').scrollIntoView({block:'start'})")
            time.sleep(3.2)
            page.evaluate("window.scrollTo({top: 0})")
            fin()
            # --- demo replay, one clip at a time
            page.goto(BASE + "/demo")
            page.wait_for_selector(".demo-tile")
            time.sleep(0.8)
            page.evaluate("window.scrollTo({top: document.querySelector('.demo-stage').offsetTop - 96})")
            for i, (key, hold) in enumerate([("fight", 7.5), ("knife", 8.0), ("gun", 10.5), ("rifle", 8.0), ("normal", 7.5)]):
                page.locator(".demo-tile").nth(i).click()
                page.wait_for_function("document.getElementById('vid').readyState >= 3")
                fin = scene(key, hold)
                fin()
            # --- fraud check
            page.goto(BASE + "/#check")
            page.wait_for_load_state("networkidle")
            page.evaluate("document.querySelector('#check .panel').scrollIntoView({block:'center'})")
            time.sleep(1.3)
            fin = scene("fraud", 11.0)
            time.sleep(1.2)
            page.click("#check input[name=q]")
            page.keyboard.type("Brightpath Holdings Ltd", delay=85)
            time.sleep(0.5)
            page.click("#check button[type=submit]")
            page.wait_for_load_state("networkidle")
            page.evaluate("document.documentElement.style.scrollBehavior='auto'; (document.querySelector('#check .result')||document.querySelector('#check .panel')).scrollIntoView({block:'center'})")
            time.sleep(0.4)
            fin()
            # --- responsible use
            page.goto(BASE + "/#responsible")
            page.wait_for_load_state("networkidle")
            page.evaluate("document.documentElement.style.scrollBehavior='auto'; document.querySelector('#responsible').scrollIntoView({block:'start'}); window.scrollBy(0,-60)")
            time.sleep(1.3)
            fin = scene("resp", 6.5)
            fin()
            # --- outro
            starts["outro"] = time.time() - t0
            page.evaluate("window.__cap('')")
            page.evaluate("window.__card('SafeGuard', 'Violence and weapon detection · public fraud check · evidence reports · human review', 'safeguard-demo-puce.vercel.app')")
            time.sleep(dur["outro"] + 1.2)
            total = time.time() - t0
            vpath = page.video.path()
            ctx.close()
            b.close()
    finally:
        srv.terminate()
    print("recorded", round(total, 1), "s", flush=True)
    (BUILD / "starts.json").write_text(json.dumps({"starts": starts, "total": total}), encoding="utf-8")

    # --- assemble: delay each narration clip to its scene start, mix, encode
    inputs, filt, labels = [], [], []
    for n, key in enumerate(LINES):
        inputs += ["-i", str(BUILD / "audio" / f"{key}.wav")]
        ms = int((starts[key] + LEAD) * 1000)
        filt.append(f"[{n + 1}:a]adelay={ms}|{ms},volume=1.15[a{n}]")
        labels.append(f"[a{n}]")
    filt.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0,apad[aout]")
    out = OUT / "safeguard-product-demo.mp4"
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(vpath), *inputs, "-filter_complex", ";".join(filt), "-map", "0:v", "-map", "[aout]",
           "-t", f"{total:.2f}", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-vf", "fade=t=in:st=0:d=0.6,fade=t=out:st=%.2f:d=0.9" % (total - 0.9),
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    # captions (WebVTT) from the narration lines
    def ts(x: float) -> str:
        return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{x % 60:06.3f}"
    vtt = ["WEBVTT", ""]
    for key, text in LINES.items():
        s = starts[key] + LEAD
        vtt += [f"{ts(s)} --> {ts(s + dur[key])}", text, ""]
    (OUT / "safeguard-product-demo.vtt").write_text("\n".join(vtt), encoding="utf-8")
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
