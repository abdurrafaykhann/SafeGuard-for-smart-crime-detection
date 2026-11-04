(() => {
  const { el, api, toast, empty, stamp, debounce } = window.SG;
  const list = document.getElementById('c-list');
  const isAdmin = document.querySelector('.shell').dataset.role === 'admin';
  const STATUSES = [['reported', 'Reported'], ['under_investigation', 'Under investigation'], ['confirmed', 'Confirmed'], ['dismissed', 'Dismissed']];

  function row(c) {
    const status = isAdmin
      ? el('select', { 'aria-label': `Status of ${c.reference}`, onchange: (e) => setStatus(c, e.target.value) },
          ...STATUSES.map(([v, t]) => el('option', { value: v, text: t, ...(v === c.status ? { selected: 'selected' } : {}) })))
      : el('span', { class: 'badge', text: c.status.replace('_', ' ') });
    return el('div', { class: 'match-row', style: 'grid-template-columns:1fr auto' },
      el('div', {}, el('b', { text: c.entity_name }), c.entity_domain ? el('span', { class: 'mono td-sub', text: `  ${c.entity_domain}` }) : null,
        el('div', { class: 'td-sub', text: `${c.reference} · ${c.entity_type} · ${c.category} · filed ${stamp(c.filed_at)}` }),
        c.summary ? el('div', { class: 'td-sub', text: c.summary }) : null),
      status);
  }

  async function load() {
    const params = new URLSearchParams({ q: document.getElementById('c-q').value, status: document.getElementById('c-filter').value });
    const rows = await api(`/api/complaints?${params}`);
    list.replaceChildren(...(rows.length ? rows.map(row) : [empty('flag', 'No complaints match', 'Try a different search or filter.')]));
  }

  async function setStatus(c, status) {
    try {
      await api(`/api/complaints/${c.id}`, { method: 'PATCH', json: { status } });
      toast('Updated', `${c.reference} is now ${status.replace('_', ' ')}.`, 'success');
    } catch (e) { toast('Could not update', e.message, 'error'); load(); }
  }

  document.getElementById('c-q').addEventListener('input', debounce(load, 250));
  document.getElementById('c-filter').addEventListener('change', load);
  document.getElementById('file-complaint')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    try {
      await api('/api/complaints', { method: 'POST', form: new FormData(form) });
      toast('Filed', 'The complaint is now live on the public fraud check.', 'success');
      form.reset(); load();
    } catch (err) { toast('Could not file', err.message, 'error', 8000); }
  });
  async function loadReports() {
    const host = document.getElementById('r-list');
    if (!host) return;
    const rows = await api('/api/reports');
    host.replaceChildren(...(rows.length ? rows.map((r) => el('div', { class: 'match-row', style: 'grid-template-columns:1fr auto' },
      el('div', {}, el('b', { text: r.entity_name }), r.entity_domain ? el('span', { class: 'mono td-sub', text: `  ${r.entity_domain}` }) : null,
        el('div', { class: 'td-sub', text: `${r.category} · received ${stamp(r.created_at)}` }),
        r.description ? el('div', { class: 'td-sub', text: r.description }) : null),
      el('div', { style: 'display:flex;gap:6px' },
        el('button', { class: 'btn btn-sm btn-primary', type: 'button', onclick: () => decide(r, 'approve') }, 'Approve'),
        el('button', { class: 'btn btn-sm', type: 'button', onclick: () => decide(r, 'reject') }, 'Reject'))))
      : [empty('flag', 'No reports waiting', 'Tips from the public appear here for review.')]));
  }

  async function decide(r, decision) {
    try {
      await api(`/api/reports/${r.id}/${decision}`, { method: 'POST' });
      toast(decision === 'approve' ? 'Approved' : 'Rejected', r.entity_name, 'success');
      loadReports(); load();
    } catch (e) { toast('Could not update', e.message, 'error'); }
  }

  loadReports().catch(() => {});
  load().catch((e) => toast('Failed to load', e.message, 'error'));
})();
