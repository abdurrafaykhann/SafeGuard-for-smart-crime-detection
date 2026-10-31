(() => {
  const { el, icon, api, toast, empty, pct, stamp } = window.SG;
  const $ = (id) => document.getElementById(id);
  let file = null, timer = null;

  const fmt = (t) => `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

  function pick(f) {
    file = f || null;
    $('picked').textContent = f ? `${f.name} · ${(f.size / 1048576).toFixed(1)} MB` : '';
    $('go').disabled = !f;
  }
  $('file').addEventListener('change', (e) => pick(e.target.files[0]));
  const drop = $('drop');
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('drag'); }));
  drop.addEventListener('drop', (e) => pick(e.dataTransfer.files[0]));

  $('go').addEventListener('click', async () => {
    const form = new FormData();
    form.append('video', file);
    $('go').disabled = true;
    try {
      const job = await api('/api/analysis', { method: 'POST', form });
      toast('Upload accepted', 'Analysis started.', 'success');
      pick(null); $('file').value = '';
      track(job.id);
    } catch (e) { toast('Upload failed', e.message, 'error', 8000); $('go').disabled = false; }
  });

  function timeline(points, peak) {
    const W = 640, H = 120, NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const maxT = points.length ? points[points.length - 1][0] || 1 : 1;
    const d = points.map(([t, s], i) => `${i ? 'L' : 'M'}${(t / maxT) * W},${H - 6 - s * (H - 14)}`).join(' ');
    const mk = (tag, a) => { const n = document.createElementNS(NS, tag); for (const k in a) n.setAttribute(k, a[k]); return n; };
    svg.append(mk('path', { d: `${d} L${W},${H} L0,${H} Z`, fill: 'rgba(255,77,106,.14)' }));
    svg.append(mk('path', { d, fill: 'none', stroke: '#ff4d6a', 'stroke-width': 2, 'stroke-linejoin': 'round' }));
    svg.append(mk('line', { x1: 0, x2: W, y1: H - 6 - 0.62 * (H - 14), y2: H - 6 - 0.62 * (H - 14), stroke: 'rgba(255,255,255,.25)', 'stroke-dasharray': '4 4' }));
    return svg;
  }

  function render(job) {
    const host = $('report');
    if (job.status === 'failed') { host.replaceChildren(empty('alert', 'Analysis failed', job.error || 'Unknown error')); return; }
    if (job.status !== 'done') {
      host.replaceChildren(el('div', { class: 'panel-pad' },
        el('h3', { text: `Analysing ${job.filename}` }), el('p', { text: job.status === 'queued' ? 'Queued…' : 'Scanning frames…' }),
        el('div', { class: 'progress' }, el('i', { style: `width:${Math.round(job.progress * 100)}%` }))));
      return;
    }
    const r = job.result;
    const kpi = (label, value, hot) => el('div', { class: `panel kpi ${hot ? 'hot' : ''}` }, el('span', { class: 'label', text: label }), el('b', { text: value }));
    host.replaceChildren(
      el('div', { class: 'panel-head' }, el('h3', { text: job.filename }), el('span', { class: 'chip', text: r.mode })),
      el('div', { class: 'kpis', style: 'padding:16px;margin:0;grid-template-columns:repeat(3,1fr)' },
        kpi('Duration', fmt(r.duration_s)), kpi('Incidents', r.episodes.length, r.episodes.length > 0), kpi('Peak risk', pct(r.peak_score), r.peak_score >= 0.62)),
      el('div', { class: 'chart', style: 'padding:0 20px 8px' }, el('div', { class: 'label', text: 'Risk over time', style: 'margin-bottom:8px' }), timeline(r.timeline, r.peak_score)),
      r.episodes.length
        ? el('div', { class: 'table-wrap' }, el('table', {}, el('thead', {}, el('tr', {}, ...['From', 'To', 'Peak', 'Assessment'].map((h) => el('th', { text: h })))),
          el('tbody', {}, ...r.episodes.map((e) => el('tr', {}, el('td', { class: 'mono', text: fmt(e.start) }), el('td', { class: 'mono', text: fmt(e.end) }), el('td', { class: 'mono', text: pct(e.peak) }), el('td', { text: e.label }))))))
        : el('div', { class: 'empty' }, icon('clear'), el('b', { text: 'No incidents detected' }), el('span', { text: 'No frame crossed the violence threshold.' })),
      Object.keys(r.watchlist_hits).length
        ? el('div', { class: 'panel-pad' }, el('div', { class: 'label', text: 'Watchlist hits', style: 'margin-bottom:10px' }),
          ...Object.entries(r.watchlist_hits).map(([name, h]) => el('div', { class: 'td-sub', text: `${name} at ${fmt(h.t)} · ${Math.round(h.similarity * 100)}% similar · ${h.risk} risk` })))
        : null,
      r.evidence.length ? el('div', { class: 'evidence' }, ...r.evidence.filter((x) => x.file).map((x) => el('figure', { style: 'margin:0' }, el('img', { src: `/api/evidence/${x.file}`, alt: `Evidence at ${fmt(x.t)}`, loading: 'lazy' }), el('figcaption', { class: 'td-sub', text: fmt(x.t) })))) : null);
  }

  function track(id) {
    clearInterval(timer);
    const tick = async () => {
      try {
        const job = await api(`/api/analysis/${id}`);
        render(job);
        if (job.status === 'done' || job.status === 'failed') { clearInterval(timer); loadJobs(); }
      } catch { clearInterval(timer); }
    };
    tick();
    timer = setInterval(tick, 1500);
  }

  async function loadJobs() {
    const jobs = await api('/api/analysis');
    $('jobs').replaceChildren(...(jobs.length ? jobs.map((j) => el('button', { class: 'btn btn-sm', type: 'button', style: 'width:100%;justify-content:space-between;margin-bottom:6px', onclick: () => track(j.id) },
      el('span', { text: j.filename, style: 'overflow:hidden;text-overflow:ellipsis' }), el('span', { class: 'td-sub', text: j.status })))
      : [el('div', { class: 'hint', text: 'No jobs yet.' })]));
  }
  loadJobs();
})();
