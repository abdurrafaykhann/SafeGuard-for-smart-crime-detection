(() => {
  const { el, api, toast, empty, stamp, pct } = window.SG;
  const id = document.getElementById('person').dataset.id;
  const $ = (x) => document.getElementById(x);
  const kv = (k, v) => (v === undefined || v === null || v === '' ? null
    : el('div', { class: 'td-sub' }, el('b', { text: `${k}: ` }), document.createTextNode(String(v))));

  function photo(src, alt) {
    return el('a', { href: src, target: '_blank', rel: 'noopener' },
      el('img', { src, alt, loading: 'lazy', style: 'width:104px;height:104px;object-fit:cover;border-radius:10px;border:1px solid var(--line);background:#000' }));
  }

  function render(d) {
    const r = d.record || {};
    document.title = `${d.label} · Case file · SafeGuard`;
    $('p-summary').replaceChildren(
      el('div', { style: 'display:flex;gap:14px;align-items:center' },
        d.has_photo ? el('img', { src: `/api/watchlist/${d.id}/photo`, alt: '', style: 'width:96px;height:96px;object-fit:cover;border-radius:12px;background:#000' }) : null,
        el('div', {}, el('h3', { text: d.label, style: 'margin:0' }),
          el('div', { class: `td-sub risk-${d.risk}`, text: `${d.risk} risk · ${d.category}` }))),
      r.demo ? el('div', { class: 'hint', text: 'FICTIONAL DEMO RECORD — not real data', style: 'margin-top:10px;color:var(--warn,#e0a030)' }) : null,
      el('div', { style: 'display:grid;gap:4px;margin-top:10px' },
        kv('Case number', r.case_number), kv('Status', r.status), kv('Aliases', (r.aliases || []).join(', ')),
        kv('Age', r.age), kv('Gender', r.gender), kv('Nationality', r.nationality), kv('Height', r.height_cm ? `${r.height_cm} cm` : ''),
        kv('Last known location', r.last_known_location), kv('Summary', r.summary || d.notes),
        kv('Basis for record', d.consent_basis), kv('Added', `${d.created_by || '?'} · ${stamp(d.created_at)}`)));

    const photos = [photo(`/api/watchlist/${d.id}/photo`, `${d.label} photo 1`), ...(d.photo_ids || []).map((pid, i) => photo(`/api/watchlist/${d.id}/photos/${pid}`, `${d.label} photo ${i + 2}`))];
    $('p-photos').replaceChildren(...(d.has_photo ? photos : [el('span', { class: 'td-sub', text: 'No photos' })]));

    const offences = [...(r.offences || [])].sort((a, b) => (a.date < b.date ? -1 : 1));
    $('p-offences').replaceChildren(...(offences.length ? offences.map((o) => el('div', { style: 'border-left:2px solid var(--line);padding:0 0 14px 12px;margin-left:4px' },
      el('div', { class: 'mono td-sub', text: o.date }), el('b', { text: o.offence }),
      el('div', { class: 'td-sub', text: `${o.location} — ${o.outcome}` }), o.note ? el('div', { class: 'td-sub', text: o.note }) : null))
      : [el('span', { class: 'td-sub', text: 'No offences recorded.' })]));

    $('p-searches').replaceChildren(...((d.searches || []).length ? d.searches.map((s) => el('div', { class: 'match-row', style: 'grid-template-columns:1fr auto' },
      el('div', {}, el('b', { text: s.by || 'unknown' }), el('div', { class: 'td-sub', text: stamp(s.at) })), el('span', { class: 'mono', text: `${Math.round(s.similarity * 100)}% similar` })))
      : [empty('search', 'No searches yet', 'Face searches that match this person are listed here.')]));

    $('p-sightings').replaceChildren(...((d.sightings || []).length ? d.sightings.map((s) => el('div', { class: 'match-row', style: 'grid-template-columns:1fr auto' },
      el('div', {}, el('b', { text: s.camera || 'Camera' }), el('div', { class: 'td-sub', text: stamp(s.at) })),
      el('a', { href: `/console/alerts?q=${encodeURIComponent(d.label)}`, text: `${pct(s.confidence)} · view alerts` })))
      : [empty('live', 'Not seen on any camera', 'Live matches against this person appear here.')]));
  }

  api(`/api/watchlist/${id}?history=1`).then(render).catch((e) => toast('Could not load case file', e.message, 'error'));
})();
