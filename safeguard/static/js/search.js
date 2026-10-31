(() => {
  const { el, api, toast, empty, icon } = window.SG;
  const drop = document.getElementById('drop');
  const input = document.getElementById('file');
  const preview = document.getElementById('preview');
  const results = document.getElementById('results');

  async function run(file) {
    if (!file) return;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    results.replaceChildren(el('div', { class: 'empty' }, el('div', { class: 'skeleton', style: 'width:60%;height:18px' })));
    const form = new FormData();
    form.append('image', file);
    try {
      const data = await api('/api/search', { method: 'POST', form });
      document.getElementById('threshold').textContent = `match threshold ${data.threshold}`;
      const closed = data.closed_set ? el('div', { class: 'panel-pad', style: 'border-top:1px solid var(--line)' },
        el('div', { class: 'label', text: 'Closed-set CNN (original FYP model, fixed identity list)', style: 'margin-bottom:8px' }),
        ...data.closed_set.predictions.map((p) => el('div', { class: 'td-sub', text: `${p.label} · ${Math.round(p.confidence * 100)}% ${p.above_floor ? '(above' : '(below'} ${Math.round(data.closed_set.min_conf * 100)}% floor)` })),
        el('div', { class: 'hint', text: 'A guess among known identities only; it cannot say "none of these". Verify manually.', style: 'margin-top:6px' })) : null;
      if (!data.matches.length) { results.replaceChildren(...[closed].filter(Boolean), empty('users', 'Watchlist is empty', 'Enrol people to match against.')); return; }
      results.replaceChildren(...[closed].filter(Boolean), ...data.matches.map((m) => el('div', { class: 'match-row' },
        el('img', { src: `/api/watchlist/${m.id}/photo`, alt: '' }),
        el('div', {}, el('b', { text: m.label }), el('div', { class: `td-sub risk-${m.risk}`, text: `${m.risk} risk` })),
        el('div', { style: 'text-align:right' },
          el('div', { class: 'mono', text: `${Math.round(m.similarity * 100)}% similar`, style: 'font-size:15px' }),
          el('span', { class: `badge ${m.is_match ? 'sev-high' : 'sev-medium'}`, text: m.is_match ? 'possible match' : 'below threshold' })))));
    } catch (e) {
      results.replaceChildren(empty('alert', 'Search failed', e.message));
      toast('Search failed', e.message, 'error');
    }
  }

  input.addEventListener('change', () => run(input.files[0]));
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('drag'); }));
  drop.addEventListener('drop', (e) => run(e.dataTransfer.files[0]));
})();
