(() => {
  const { el, icon, api, toast } = window.SG;
  const rows = document.getElementById('rows');

  function row(c) {
    const flags = [c.detect_violence && 'violence', c.detect_faces && 'faces', c.autostart && 'autostart'].filter(Boolean).join(' · ');
    return el('tr', {},
      el('td', {}, el('div', { class: 'td-title', text: c.name }), el('div', { class: 'td-sub', text: c.location || '—' })),
      el('td', { class: 'mono', text: c.source_kind }),
      el('td', { class: 'td-sub', text: flags || 'off' }),
      el('td', {}, el('span', { class: `chip ${c.running ? 'ok' : ''}` }, el('span', { class: 'dot' }), c.running ? 'running' : 'stopped')),
      el('td', {}, el('button', { class: 'btn btn-danger btn-sm', type: 'button', onclick: () => remove(c) }, icon('trash'), 'Delete')));
  }

  async function load() {
    const list = await api('/api/cameras');
    document.getElementById('empty').hidden = list.length > 0;
    rows.replaceChildren(...list.map(row));
  }

  async function remove(c) {
    if (!confirm(`Delete camera "${c.name}"? Its past alerts are kept.`)) return;
    try { await api(`/api/cameras/${c.id}`, { method: 'DELETE' }); load(); }
    catch (e) { toast('Could not delete', e.message, 'error'); }
  }

  document.getElementById('add').addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.currentTarget, v = new FormData(f);
    try {
      await api('/api/cameras', { method: 'POST', json: {
        name: v.get('name'), source: v.get('source'), location: v.get('location'),
        detect_violence: v.has('detect_violence'), detect_faces: v.has('detect_faces'), autostart: v.has('autostart') } });
      toast('Camera added', 'Open the Live wall to start it.', 'success');
      f.reset(); f.querySelectorAll('input[type=checkbox]:not([name=autostart])').forEach((c) => { c.checked = true; });
      load();
    } catch (err) { toast('Could not add camera', err.message, 'error', 7000); }
  });
  load();
})();
