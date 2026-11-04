(() => {
  const { el, icon, api, ago, stamp, pct, sevBadge, toast, debounce } = window.SG;
  const rows = document.getElementById('rows');
  const state = { page: 1, pages: 1 };
  const $ = (id) => document.getElementById(id);

  const query = (extra = {}) => {
    const p = new URLSearchParams({ page: state.page, limit: 15, ...extra });
    const map = { q: 'f-q', kind: 'f-kind', severity: 'f-sev', status: 'f-status' };
    for (const [k, id] of Object.entries(map)) if ($(id).value) p.set(k, $(id).value);
    return p;
  };

  const KIND = { violence: 'Violence', persistent: 'Sustained', weapon: 'Weapon', watchlist: 'Watchlist' };

  function row(a) {
    const thumb = a.has_snapshot
      ? el('img', { title: a.has_clip ? 'Has video clip' : '', src: `/api/alerts/${a.id}/snapshot`, alt: 'Snapshot', loading: 'lazy', style: 'width:84px;height:54px;object-fit:cover;border-radius:8px;cursor:pointer;background:#000;border:1px solid var(--line)', onclick: () => openViewer(a) })
      : el('span', { class: 'td-sub', text: 'none' });
    const ackCell = a.acknowledged_at
      ? el('span', { class: 'td-sub' }, icon('check'), ` ${a.acknowledged_by}`)
      : el('button', { class: 'btn btn-sm', type: 'button', onclick: (e) => ack(a, e.currentTarget) }, icon('check'), 'Acknowledge');
    return el('tr', { dataset: { id: a.id } },
      el('td', {}, thumb),
      el('td', {}, el('div', { class: 'td-title', text: a.title }), el('div', { class: `td-sub kind-${a.kind}`, text: KIND[a.kind] || a.kind })),
      el('td', {}, sevBadge(a.severity)),
      el('td', { text: a.camera || '—' }),
      el('td', { class: 'mono', text: pct(a.confidence) }),
      el('td', { title: stamp(a.created_at) }, el('div', { text: ago(a.created_at) }), el('div', { class: 'td-sub', text: stamp(a.created_at) })),
      el('td', {}, ackCell));
  }

  async function load() {
    try {
      const data = await api('/api/alerts?' + query());
      state.pages = data.pages;
      rows.replaceChildren(...data.items.map(row));
      $('empty').hidden = data.items.length > 0;
      $('count').textContent = `${data.total} alert${data.total === 1 ? '' : 's'} · page ${data.page} of ${data.pages}`;
      $('prev').disabled = data.page <= 1;
      $('next').disabled = data.page >= data.pages;
      $('export').href = '/api/alerts/export.csv?' + query({ limit: 100 });
    } catch (e) { toast('Could not load alerts', e.message, 'error'); }
  }

  async function ack(a, btn) {
    btn.disabled = true;
    try { await api(`/api/alerts/${a.id}/ack`, { method: 'POST' }); load(); }
    catch (e) { toast('Acknowledge failed', e.message, 'error'); btn.disabled = false; }
  }

  function openViewer(a) {
    $('viewer-img').src = `/api/alerts/${a.id}/snapshot`;
    const clip = $('viewer-clip');
    $('viewer-report').href = `/api/alerts/${a.id}/report.pdf`;
    clip.hidden = !a.has_clip;
    $('viewer-img').hidden = !!a.has_clip;
    if (a.has_clip) clip.src = `/api/alerts/${a.id}/clip`; else clip.removeAttribute('src');
    const cues = a.meta?.cues ? Object.entries(a.meta.cues) : [];
    $('viewer-meta').replaceChildren(...[
      el('h3', { text: a.title }),
      el('p', { text: `${a.camera || 'Camera'} · ${stamp(a.created_at)} · confidence ${pct(a.confidence)}` }),
      cues.length ? el('div', {}, el('div', { class: 'label', text: 'Why this fired', style: 'margin-bottom:8px' }),
        ...cues.map(([k, v]) => el('div', { class: 'bar-row', style: 'grid-template-columns:100px 1fr 44px' },
          el('span', { text: k }), el('div', { class: 'track' }, el('i', { style: `width:${Math.round(v * 100)}%` })), el('span', { text: pct(v) })))) : null].filter(Boolean));
    $('viewer').showModal();
  }
  $('viewer').addEventListener('close', () => $('viewer-clip').pause());
  $('viewer-close').addEventListener('click', () => $('viewer').close());
  $('viewer').addEventListener('click', (e) => { if (e.target === $('viewer')) $('viewer').close(); });

  const reload = () => { state.page = 1; load(); };
  ['f-kind', 'f-sev', 'f-status'].forEach((id) => $(id).addEventListener('change', reload));
  $('f-q').addEventListener('input', debounce(reload));
  $('prev').addEventListener('click', () => { state.page = Math.max(1, state.page - 1); load(); });
  $('next').addEventListener('click', () => { state.page = Math.min(state.pages, state.page + 1); load(); });
  SG.on('alert', () => { if (state.page === 1) load(); });
  SG.on('alert_ack', load);
  load();
})();
