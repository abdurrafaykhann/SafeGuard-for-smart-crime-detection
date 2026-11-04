/* Shared helpers. DOM is always built with createElement/textContent, never innerHTML,
   so data coming back from the API cannot inject markup. */
(() => {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const csrf = () => document.querySelector('meta[name="csrf-token"]')?.content || '';

  function el(tag, attrs = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'dataset') Object.assign(node.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }

  function icon(name) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('class', 'icon');
    const use = document.createElementNS(SVG_NS, 'use');
    use.setAttribute('href', `#i-${name}`);
    svg.append(use);
    return svg;
  }

  async function api(path, { method = 'GET', json, form } = {}) {
    const headers = { 'X-CSRFToken': csrf(), Accept: 'application/json' };
    let body;
    if (json !== undefined) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(json); }
    if (form) body = form;
    const res = await fetch(path, { method, headers, body, credentials: 'same-origin' });
    if (res.status === 401) { location.href = '/login?next=' + encodeURIComponent(location.pathname); throw new Error('Signed out'); }
    if (res.status === 204) return null;
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON body */ }
    if (!res.ok) throw new Error(data?.message || `Request failed (${res.status})`);
    return data;
  }

  function toast(title, message = '', kind = 'info', ttl = 5000) {
    const host = document.getElementById('toasts');
    if (!host) return;
    const node = el('div', { class: `toast ${kind}` }, el('div', {}, el('b', { text: title }), message ? el('small', { text: message }) : null));
    host.append(node);
    setTimeout(() => node.remove(), ttl);
    return node;
  }

  const rtf = typeof Intl !== 'undefined' && Intl.RelativeTimeFormat ? new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' }) : null;
  function ago(iso) {
    if (!iso) return '';
    const s = (new Date(iso) - Date.now()) / 1000;
    const abs = Math.abs(s);
    if (!rtf) return new Date(iso).toLocaleString();
    if (abs < 60) return rtf.format(Math.round(s), 'second');
    if (abs < 3600) return rtf.format(Math.round(s / 60), 'minute');
    if (abs < 86400) return rtf.format(Math.round(s / 3600), 'hour');
    return rtf.format(Math.round(s / 86400), 'day');
  }
  const stamp = (iso) => iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '';
  const pct = (v) => `${Math.round((v || 0) * 100)}%`;
  const debounce = (fn, ms = 250) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  function sevBadge(sev) { return el('span', { class: `badge sev-${sev}`, text: sev }); }
  function empty(iconName, title, text) {
    return el('div', { class: 'empty' }, icon(iconName), el('b', { text: title }), el('span', { text }));
  }

  window.SG = { el, icon, api, toast, ago, stamp, pct, debounce, sevBadge, empty, csrf };

  // public nav toggle
  const t = document.getElementById('nav-toggle');
  t?.addEventListener('click', () => document.getElementById('nav')?.classList.toggle('open'));
})();
