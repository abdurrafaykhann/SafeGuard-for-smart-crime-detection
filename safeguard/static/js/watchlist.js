(() => {
  const { el, icon, api, toast, stamp } = window.SG;
  const cards = document.getElementById('cards');
  const isAdmin = document.querySelector('.shell').dataset.role === 'admin';

  function card(p) {
    const img = p.has_photo
      ? el('img', { src: `/api/watchlist/${p.id}/photo`, alt: `Reference crop for ${p.label}`, loading: 'lazy' })
      : el('div', { style: 'aspect-ratio:1;background:#000' });
    const remove = isAdmin ? el('button', { class: 'btn btn-danger btn-sm', type: 'button', onclick: () => del(p) }, icon('trash'), 'Remove') : null;
    return el('article', { class: 'panel person' }, img, el('div', { class: 'body' },
      el('a', { href: `/console/watchlist/${p.id}`, style: 'font-weight:700', text: p.label }),
      el('div', { class: 'td-sub', text: p.category }),
      el('div', { style: 'margin:10px 0' }, el('span', { class: `badge risk-${p.risk}`, text: `${p.risk} risk`, style: 'background:var(--surface-2)' })),
      el('div', { class: 'td-sub', text: `Basis: ${p.consent_basis}` }),
      el('div', { class: 'td-sub', text: `Added by ${p.created_by} · ${stamp(p.created_at)}`, style: 'margin-bottom:12px' }),
      remove));
  }

  async function load() {
    const list = await api('/api/watchlist');
    document.getElementById('empty').hidden = list.length > 0;
    cards.replaceChildren(...list.map(card));
  }

  async function del(p) {
    if (!confirm(`Remove ${p.label}? Their face signature and crop will be deleted immediately.`)) return;
    try { await api(`/api/watchlist/${p.id}`, { method: 'DELETE' }); toast('Removed', p.label, 'success'); load(); }
    catch (e) { toast('Could not remove', e.message, 'error'); }
  }

  document.getElementById('enrol')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.currentTarget, btn = form.querySelector('button[type=submit]');
    btn.disabled = true;
    try {
      await api('/api/watchlist', { method: 'POST', form: new FormData(form) });
      toast('Enrolled', 'The person has been added to the watchlist.', 'success');
      form.reset(); load();
    } catch (err) { toast('Enrolment failed', err.message, 'error', 8000); }
    btn.disabled = false;
  });
  load().catch((e) => toast('Failed to load', e.message, 'error'));
})();
