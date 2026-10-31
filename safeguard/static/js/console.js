/* Console shell: one SSE connection per tab, fanned out to page scripts via SG.on(). */
(() => {
  const { toast, api } = window.SG;
  const handlers = {};
  const chip = document.getElementById('chip-link');
  const setChip = (cls, text) => {
    if (!chip) return;
    chip.className = `chip ${cls}`;
    chip.lastElementChild.textContent = text;
  };

  window.SG.on = (event, fn) => { (handlers[event] ||= []).push(fn); };

  async function refreshOpenCount() {
    try {
      const data = await api('/api/alerts?status=open&limit=1');
      const badge = document.getElementById('open-count');
      if (badge) { badge.textContent = data.total > 99 ? '99+' : data.total; badge.hidden = data.total === 0; }
    } catch { /* non-critical */ }
  }

  function connect() {
    const es = new EventSource('/api/stream');
    es.onopen = () => setChip('ok', 'live');
    es.onerror = () => setChip('bad', 'reconnecting');
    for (const name of ['alert', 'alert_ack', 'camera']) {
      es.addEventListener(name, (e) => {
        let data; try { data = JSON.parse(e.data); } catch { return; }
        (handlers[name] || []).forEach((fn) => fn(data));
      });
    }
    window.addEventListener('beforeunload', () => es.close());
  }

  window.SG.on('alert', (a) => {
    refreshOpenCount();
    toast(a.title, `${a.camera || 'Camera'} · ${a.severity.toUpperCase()}`, 'alert', 8000);
  });
  window.SG.on('alert_ack', refreshOpenCount);
  refreshOpenCount();
  connect();
})();
