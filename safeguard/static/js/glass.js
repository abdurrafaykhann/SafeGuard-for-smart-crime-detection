/* Visual polish for the public pages: scroll reveals and the panel cursor-spotlight. Purely cosmetic:
   every page works unchanged when this script or JavaScript is unavailable. */
document.documentElement.classList.add('js');
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* scroll reveal */
const targets = document.querySelectorAll('.feature, .step, .principle, .stat-strip > div, .section-head, .cta-band');
targets.forEach((el, i) => { el.classList.add('reveal'); el.style.setProperty('--i', String(i % 6)); });
if (reduce || !('IntersectionObserver' in window)) targets.forEach((el) => el.classList.add('in'));
else {
  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
  }), { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
  targets.forEach((el) => io.observe(el));
}

/* cursor spotlight on panels */
document.addEventListener('pointermove', (e) => {
  const p = e.target.closest && e.target.closest('.panel');
  if (!p) return;
  const r = p.getBoundingClientRect();
  p.style.setProperty('--mx', `${e.clientX - r.left}px`);
  p.style.setProperty('--my', `${e.clientY - r.top}px`);
}, { passive: true });
