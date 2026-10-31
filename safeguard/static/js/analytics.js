/* Dependency-free SVG charts so the theme stays fully under our control. */
(() => {
  const { el, api, toast, empty } = window.SG;
  const NS = 'http://www.w3.org/2000/svg';
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const svg = (tag, attrs = {}, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    return n;
  };

  function stackedBars(host, series) {
    const W = 720, H = 260, P = { l: 30, r: 8, t: 12, b: 28 };
    const keys = [['violence', css('--orange')], ['persistent', css('--crit')], ['watchlist', css('--accent-2')]];
    const max = Math.max(1, ...series.map((d) => keys.reduce((s, [k]) => s + d[k], 0)));
    const step = (W - P.l - P.r) / series.length, bw = Math.max(Math.min(step * 0.62, 34), 4);
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Alerts per day, stacked by type' });
    for (let i = 0; i <= 4; i++) {
      const y = P.t + (H - P.t - P.b) * (i / 4);
      root.append(svg('line', { x1: P.l, x2: W - P.r, y1: y, y2: y, stroke: 'rgba(255,255,255,.07)' }));
      root.append(svg('text', { x: P.l - 8, y: y + 3, 'text-anchor': 'end' }, Math.round(max * (1 - i / 4))));
    }
    series.forEach((d, i) => {
      let y = H - P.b;
      const x = P.l + step * i + (step - bw) / 2;
      keys.forEach(([k, colour]) => {
        const h = (d[k] / max) * (H - P.t - P.b);
        if (h > 0) {
          y -= h;
          const r = svg('rect', { x, y, width: bw, height: h, rx: 3, fill: colour });
          r.append(svg('title', {}, `${d.date}: ${d[k]} ${k}`));
          root.append(r);
        }
      });
      if (series.length <= 16 || i % Math.ceil(series.length / 12) === 0) {
        root.append(svg('text', { x: x + bw / 2, y: H - 8, 'text-anchor': 'middle' }, d.date.slice(5)));
      }
    });
    host.replaceChildren(root);
  }

  function donut(host, data) {
    const entries = Object.entries(data);
    const total = entries.reduce((s, [, v]) => s + v, 0);
    if (!total) { host.replaceChildren(empty('chart', 'No data', 'Nothing in this period.')); return; }
    const colours = { violence: css('--orange'), persistent: css('--crit'), watchlist: css('--accent-2') };
    const R = 70, C = 2 * Math.PI * R;
    const root = svg('svg', { viewBox: '0 0 200 200', width: 190, height: 190, role: 'img', 'aria-label': 'Alerts by type' });
    let offset = 0;
    entries.forEach(([k, v]) => {
      const len = (v / total) * C;
      root.append(svg('circle', { cx: 100, cy: 100, r: R, fill: 'none', stroke: colours[k] || css('--accent'), 'stroke-width': 22,
        'stroke-dasharray': `${Math.max(len - 3, 0)} ${C}`, 'stroke-dashoffset': -offset, transform: 'rotate(-90 100 100)' }));
      offset += len;
    });
    root.append(svg('text', { x: 100, y: 104, 'text-anchor': 'middle', style: 'font:600 30px var(--display);fill:var(--text)' }, total));
    root.append(svg('text', { x: 100, y: 124, 'text-anchor': 'middle' }, 'ALERTS'));
    const legend = el('div', { class: 'legend', style: 'margin-top:14px;justify-content:center' },
      ...entries.map(([k, v]) => el('span', {}, el('i', { style: `background:${colours[k]}` }), `${k} ${v}`)));
    host.replaceChildren(root, legend);
  }

  function hours(host, values) {
    const W = 520, H = 200, P = { l: 24, r: 4, t: 10, b: 24 };
    const max = Math.max(1, ...values), bw = (W - P.l - P.r) / 24;
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Alerts by hour of day' });
    values.forEach((v, h) => {
      const bh = (v / max) * (H - P.t - P.b);
      const r = svg('rect', { x: P.l + bw * h + 2, y: H - P.b - bh, width: bw - 4, height: Math.max(bh, 1.5), rx: 3,
        fill: v === max && v > 0 ? css('--accent') : 'rgba(70,225,255,.28)' });
      r.append(svg('title', {}, `${String(h).padStart(2, '0')}:00 - ${v} alert${v === 1 ? '' : 's'}`));
      root.append(r);
      if (h % 3 === 0) root.append(svg('text', { x: P.l + bw * h + bw / 2, y: H - 6, 'text-anchor': 'middle' }, String(h).padStart(2, '0')));
    });
    host.replaceChildren(root);
  }

  function barList(host, pairs, noun) {
    if (!pairs.length) { host.replaceChildren(empty('chart', 'Nothing yet', `No ${noun} in this period.`)); return; }
    const max = Math.max(...pairs.map(([, n]) => n));
    host.replaceChildren(...pairs.map(([name, n]) => el('div', { class: 'bar-row' },
      el('span', { text: name, title: name }), el('div', { class: 'track' }, el('i', { style: `width:${(n / max) * 100}%` })), el('span', { text: n }))));
  }

  const fmtDuration = (s) => s == null ? '–' : s < 90 ? `${Math.round(s)}s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;

  async function load() {
    try {
      const d = await api(`/api/analytics/summary?days=${$('range').value}`);
      $('a-total').textContent = d.total;
      $('a-crit').textContent = d.critical;
      $('a-ack').textContent = d.ack_rate == null ? '–' : `${Math.round(d.ack_rate * 100)}%`;
      $('a-open').textContent = `${d.open} still open`;
      $('a-mtta').textContent = fmtDuration(d.mean_ack_seconds);
      stackedBars($('c-days'), d.series);
      donut($('c-kind'), d.by_kind);
      hours($('c-hours'), d.by_hour);
      barList($('c-cams'), d.by_camera, 'camera activity');
      barList($('c-people'), d.top_watchlist, 'watchlist matches');
    } catch (e) { toast('Could not load analytics', e.message, 'error'); }
  }
  $('range').addEventListener('change', load);
  load();
})();
