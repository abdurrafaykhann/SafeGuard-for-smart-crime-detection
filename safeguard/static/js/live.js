(() => {
  const { el, icon, api, ago, toast, sevBadge, pct } = window.SG;
  const tiles = document.getElementById('tiles');
  const feed = document.getElementById('feed');
  const isAdmin = document.querySelector('.shell').dataset.role === 'admin';
  const tileMap = new Map();
  const people = new Map(); // camera id -> people currently in view

  function buildTile(cam) {
    const img = el('img', { alt: `Live feed: ${cam.name}`, hidden: true });
    const placeholder = el('div', { class: 'placeholder' }, el('div', {}, icon('camera'), el('div', { text: cam.running ? 'Connecting…' : 'Camera is stopped' })));
    const meterFill = el('i');
    const meterText = el('span', { text: 'risk 0%' });
    const meter = el('div', { class: 'tile-meter', hidden: true }, meterText, el('div', { class: 'meter' }, meterFill));
    const status = el('small', { text: cam.location || '—' });
    const btn = el('button', { class: 'btn btn-sm', type: 'button' });
    const root = el('article', { class: 'tile' }, img, placeholder, meter,
      el('div', { class: 'tile-bar' }, el('div', {}, el('b', { text: cam.name }), status), btn));
    const t = { cam, root, img, placeholder, meterFill, meterText, meter, status, btn, running: false };
    btn.addEventListener('click', () => toggle(t));
    setRunning(t, cam.running);
    return t;
  }

  function setRunning(t, running) {
    t.running = running;
    t.btn.replaceChildren(icon(running ? 'stop' : 'play'), running ? 'Stop' : 'Start');
    t.meter.hidden = !running;
    if (running) {
      t.img.src = `/api/cameras/${t.cam.id}/feed?t=${Date.now()}`;
      t.img.hidden = false;
      t.img.onload = () => { t.placeholder.hidden = true; };
    } else {
      t.img.hidden = true; t.img.removeAttribute('src'); t.placeholder.hidden = false;
      t.placeholder.querySelector('div > div').textContent = 'Camera is stopped';
      t.root.classList.remove('violent');
      people.delete(t.cam.id);
      document.getElementById('k-people').textContent = [...people.values()].reduce((a, b) => a + b, 0);
    }
  }

  async function toggle(t) {
    t.btn.disabled = true;
    try {
      await api(`/api/cameras/${t.cam.id}/${t.running ? 'stop' : 'start'}`, { method: 'POST' });
      setRunning(t, !t.running);
      refreshKpis();
    } catch (e) { toast('Could not change camera', e.message, 'error'); }
    t.btn.disabled = false;
  }

  async function loadCameras() {
    const cams = await api('/api/cameras');
    document.getElementById('no-cameras').hidden = cams.length > 0;
    tiles.replaceChildren();
    cams.forEach((c) => { const t = buildTile(c); tileMap.set(c.id, t); tiles.append(t.root); });
    refreshKpis();
  }

  function feedItem(a) {
    const thumb = a.has_snapshot
      ? el('img', { class: 'thumb', src: `/api/alerts/${a.id}/snapshot`, alt: '', loading: 'lazy' })
      : el('div', { class: 'thumb' });
    return el('div', { class: 'feed-item', onclick: () => { location.href = '/console/alerts'; } }, thumb,
      el('div', {}, el('b', { text: a.title }), el('small', { text: `${a.camera || 'Camera'} · ${ago(a.created_at)} · ${pct(a.confidence)}` }), el('div', { style: 'margin-top:6px' }, sevBadge(a.severity))));
  }
  function pushFeed(a, prepend = true) {
    document.getElementById('feed-empty')?.remove();
    const node = feedItem(a);
    prepend ? feed.prepend(node) : feed.append(node);
    while (feed.children.length > 30) feed.lastElementChild.remove();
  }

  async function refreshKpis() {
    const running = [...tileMap.values()].filter((t) => t.running).length;
    document.getElementById('k-live').textContent = `${running}/${tileMap.size}`;
    document.getElementById('k-live-sub').textContent = running ? 'analysing in real time' : 'start a camera to begin';
    try {
      const [open, s] = await Promise.all([api('/api/alerts?status=open&limit=1'), api('/api/analytics/summary?days=1')]);
      document.getElementById('k-open').textContent = open.total;
      document.getElementById('k-24h').textContent = s.total;
      document.getElementById('k-24h-sub').textContent = s.critical ? `${s.critical} critical` : 'no critical incidents';
    } catch { /* leave placeholders */ }
  }

  SG.on('camera', (s) => {
    const t = tileMap.get(s.id);
    if (!t || !t.running) return;
    t.root.classList.toggle('violent', !!s.violent);
    t.meterFill.style.width = `${Math.round((s.score || 0) * 100)}%`;
    t.meterText.textContent = `risk ${Math.round((s.score || 0) * 100)}%`;
    const parts = [s.connected ? `${s.fps} fps` : 'no signal', `${s.persons} people`];
    if (s.faces) parts.push(`${s.faces} face${s.faces > 1 ? 's' : ''}`);
    t.status.textContent = parts.join(' · ');
    people.set(s.id, s.persons || 0);
    document.getElementById('k-people').textContent = [...people.values()].reduce((a, b) => a + b, 0);
  });
  SG.on('alert', (a) => { pushFeed(a); refreshKpis(); });

  (async () => {
    try {
      await loadCameras();
      const recent = await api('/api/alerts?limit=12');
      recent.items.reverse().forEach((a) => pushFeed(a));
    } catch (e) { toast('Failed to load', e.message, 'error'); }
  })();
})();
