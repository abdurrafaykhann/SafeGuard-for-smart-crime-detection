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
      if (!data.matches.length) { results.replaceChildren(empty('users', 'Watchlist is empty', 'Enrol people first.')); return; }
      results.replaceChildren(...data.matches.map((m) => el('div', { class: 'match-row' },
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
