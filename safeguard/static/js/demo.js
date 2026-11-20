/* Recorded-replay demo. Plays a stock clip and draws the REAL detector output that was recorded for it
   (static/demo/replay.json, made by scripts/demo_video/make_replay.py). No analysis happens in the browser.
   DOM is built with createElement/textContent only. */
(() => {
  const $ = (id) => document.getElementById(id);
  const vid = $('vid'), ov = $('ov'), ctx = ov.getContext('2d');
  const BASE = '/static/demo/';
  const THRESH = 0.5;
  let geo = { ox: 0, oy: 0, cw: 1, ch: 1 };
  let data, clip, cur = -1, tour = false, shown = new Map(), alerts = 0;

  const fmt = (t) => `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };

  function sampleAt(t) {
    const s = clip.samples; let lo = 0, hi = s.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (s[m].t <= t) lo = m; else hi = m - 1; }
    return s[lo];
  }

  function buildTiles() {
    const box = $('tiles');
    data.clips.forEach((c, i) => {
      const b = el('button', 'demo-tile'); b.type = 'button'; b.setAttribute('role', 'tab');
      const v = el('video'); v.src = BASE + c.video + '#t=1'; v.muted = true; v.preload = 'metadata'; v.setAttribute('aria-hidden', 'true');
      const meta = el('span', 'demo-tile-meta');
      meta.append(el('b', '', c.title), el('small', '', `${c.camera} · ${c.location}`));
      b.append(v, meta); b.addEventListener('click', () => { tour = false; $('tour').classList.remove('on'); select(i, true); });
      box.append(b);
    });
  }

  function select(i, play) {
    cur = i; clip = data.clips[i]; shown = new Map(); alerts = 0;
    [...$('tiles').children].forEach((t, k) => t.classList.toggle('active', k === i));
    $('camname').textContent = `${clip.camera} · ${clip.location}`;
    $('blurb').textContent = clip.blurb;
    const feed = $('feed'); feed.replaceChildren(el('li', 'empty-feed', 'Alerts appear here as they are raised.'));
    $('feedcount').textContent = '0 alerts'; $('banner').hidden = true;
    vid.src = BASE + clip.video; vid.currentTime = 0; drawChart();
    if (play) vid.play().catch(() => {});
    render();
  }

  function drawChart() {
    const W = 320, H = 110, d = clip.seconds || 1;
    const pts = clip.samples.map((s) => [(s.t / d) * W, H - Math.min(s.risk, 1) * (H - 6) - 3]);
    const path = pts.map((p, k) => `${k ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
    $('line').setAttribute('d', path);
    $('area').setAttribute('d', `${path} L${W} ${H} L0 ${H} Z`);
    const y = H - THRESH * (H - 6) - 3; const thr = $('thr'); thr.setAttribute('y1', y); thr.setAttribute('y2', y);
  }

  function addAlert(kind, t, text, sev) {
    if (shown.has(kind) && t - shown.get(kind) < 8) return; shown.set(kind, t); alerts++;
    const feed = $('feed'); if (feed.querySelector('.empty-feed')) feed.replaceChildren();
    const li = el('li', `sev-${sev}`);
    li.append(el('span', 'badge ' + (sev === 'critical' ? 'sev-critical' : 'sev-high'), sev), el('b', '', kind), el('small', '', `${clip.camera} · ${fmt(t)} · ${text}`));
    feed.prepend(li); $('feedcount').textContent = `${alerts} alert${alerts === 1 ? '' : 's'}`;
    const bn = $('banner'); bn.textContent = `${kind.toUpperCase()}: ${text}`; bn.hidden = false; bn.dataset.sev = sev;
    clearTimeout(addAlert.t); addAlert.t = setTimeout(() => { bn.hidden = true; }, 2600);
  }

  function resize() {
    const r = vid.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    ov.width = Math.round(r.width * dpr); ov.height = Math.round(r.height * dpr); ov.style.width = r.width + 'px'; ov.style.height = r.height + 'px';
  }

  function box(x, y, w, h, color, label, solid) {
    const W = ov.width, k = Math.max(W / 640, 1), X = geo.ox + x * geo.cw, Y = geo.oy + y * geo.ch, BW = w * geo.cw, BH = h * geo.ch;
    ctx.strokeStyle = color; ctx.lineWidth = 2 * k; ctx.strokeRect(X, Y, BW, BH);
    const c = 14 * k; ctx.lineWidth = 3.5 * k; ctx.beginPath();
    ctx.moveTo(X, Y + c); ctx.lineTo(X, Y); ctx.lineTo(X + c, Y); ctx.moveTo(X + BW - c, Y); ctx.lineTo(X + BW, Y); ctx.lineTo(X + BW, Y + c);
    ctx.moveTo(X, Y + BH - c); ctx.lineTo(X, Y + BH); ctx.lineTo(X + c, Y + BH); ctx.moveTo(X + BW - c, Y + BH); ctx.lineTo(X + BW, Y + BH); ctx.lineTo(X + BW, Y + BH - c); ctx.stroke();
    if (label) {
      ctx.font = `600 ${12 * k}px "JetBrains Mono", monospace`; const tw = ctx.measureText(label).width + 12 * k, th = 20 * k, ly = Y - th - 2 * k < 0 ? Y + 2 * k : Y - th - 2 * k;
      ctx.fillStyle = solid ? color : 'rgba(6,10,18,.82)'; ctx.fillRect(X, ly, tw, th); ctx.fillStyle = solid ? '#02121a' : color; ctx.textBaseline = 'middle'; ctx.fillText(label, X + 6 * k, ly + th / 2);
    }
  }

  function render() {
    if (!clip) return;
    const t = vid.currentTime || 0, s = sampleAt(t);
    ctx.clearRect(0, 0, ov.width, ov.height);
    const ready = vid.readyState >= 2 && vid.videoWidth > 0;
    const vr = (vid.videoWidth || 16) / (vid.videoHeight || 9), er = ov.width / ov.height;
    if (vr < er) { geo.ch = ov.height; geo.cw = ov.height * vr; geo.ox = (ov.width - geo.cw) / 2; geo.oy = 0; }
    else { geo.cw = ov.width; geo.ch = ov.width / vr; geo.ox = 0; geo.oy = (ov.height - geo.ch) / 2; }
    if (ready) s.people.forEach((p) => box(p[0], p[1], p[2], p[3], 'rgba(56,224,160,.9)', null));
    // a weapon is "confirmed" once the detector's sustained-sighting rule first fires; from then on it is tracked,
    // holding the last box for up to a second between detections. Clips that never confirm a weapon never show a box.
    const confirmed = clip.firstWeapon !== null && t >= clip.firstWeapon - 0.01;
    let wp = [];
    if (confirmed && ready) {
      const i = clip.samples.indexOf(s);
      for (let k = i; k >= 0 && s.t - clip.samples[k].t <= 1.0; k--) { const f = clip.samples[k].weapons.filter((w) => w.conf >= 0.45); if (f.length) { wp = f; break; } }
    }
    if (ready) wp.forEach((w) => box(w.box[0], w.box[1], w.box[2], w.box[3], '#ff4d6a', `${w.name.toUpperCase()} ${(w.conf * 100).toFixed(0)}%`, true));
    if (wp.length) addAlert('Weapon', t, `${wp[0].name} · ${(wp[0].conf * 100).toFixed(0)}% confidence`, 'critical');
    if (s.violent) addAlert('Violence', t, `risk ${s.risk.toFixed(2)}`, 'high');
    $('clock').textContent = fmt(t); $('k-risk').textContent = s.risk.toFixed(2); $('k-people').textContent = String(s.people.length);
    $('k-weapon').textContent = wp.length ? wp[0].name : 'none';
    const st = $('state'); const bad = wp.length > 0 || s.violent; st.className = 'chip ' + (bad ? 'bad' : 'ok'); st.lastChild.textContent = wp.length ? 'weapon' : s.violent ? 'violence' : 'clear';
    const d = clip.seconds || 1; $('head').setAttribute('x1', (t / d) * 320); $('head').setAttribute('x2', (t / d) * 320);
    $('scrubfill').style.width = (t / d) * 100 + '%'; $('scrubhead').style.left = (t / d) * 100 + '%';
    $('stage').classList.toggle('alarm', bad);
  }

  function loop() { render(); requestAnimationFrame(loop); }
  function syncBtn() { const p = !vid.paused; $('pp').lastChild.textContent = p ? 'Pause' : 'Play'; $('bigplay').hidden = p; }

  async function init() {
    data = await (await fetch(BASE + 'replay.json')).json();
    buildTiles(); window.addEventListener('resize', resize); vid.addEventListener('loadedmetadata', resize);
    vid.addEventListener('play', syncBtn); vid.addEventListener('pause', syncBtn);
    vid.addEventListener('ended', () => { if (tour && cur < data.clips.length - 1) select(cur + 1, true); else { tour = false; $('tour').classList.remove('on'); } syncBtn(); });
    $('pp').addEventListener('click', () => (vid.paused ? vid.play() : vid.pause()));
    $('bigplay').addEventListener('click', () => vid.play());
    $('tour').addEventListener('click', () => { tour = true; $('tour').classList.add('on'); select(0, true); });
    const seek = (e) => { const r = $('scrub').getBoundingClientRect(); vid.currentTime = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * (clip.seconds || vid.duration || 1); };
    $('scrub').addEventListener('click', seek);
    $('scrub').addEventListener('keydown', (e) => { if (e.key === 'ArrowRight') vid.currentTime += 2; if (e.key === 'ArrowLeft') vid.currentTime -= 2; });
    select(0, false); resize(); syncBtn(); loop();
    const q = new URLSearchParams(location.search); if (q.has('tour')) $('tour').click();
  }
  init().catch((e) => console.error('demo failed', e));
})();
