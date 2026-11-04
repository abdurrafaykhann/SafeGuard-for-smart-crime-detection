// Recording overlay for the demo video: captions, chapter tag, progress bar, fake cursor, title cards, quiz card.
// Injected into every page with add_init_script; state lives in sessionStorage so it survives navigation.
(() => {
  if (window.__ov) return;
  const KEY = '__ov_state';
  const state = Object.assign({ cap: '', chap: '', prog: 0, blur: false, zoom: 1, cx: 640, cy: 360, card: null, quiz: null },
    (() => { try { return JSON.parse(sessionStorage.getItem(KEY) || '{}'); } catch (e) { return {}; } })());
  const save = () => { try { sessionStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* ignore */ } };

  const css = `
  #__ov, #__ov * { box-sizing: border-box; pointer-events: none; }
  #__ov { position: fixed; inset: 0; z-index: 2147483647; font-family: Inter, 'Segoe UI', system-ui, sans-serif; }
  body > *:not(#__ov) { zoom: var(--ovz, 1); }
  #__cap { position: absolute; left: 50%; bottom: 34px; transform: translateX(-50%); max-width: 1020px; width: max-content;
    padding: 12px 22px; border-radius: 12px; font-size: 24px; line-height: 1.3; font-weight: 600; color: #fff; text-align: center;
    background: rgba(6, 10, 18, .86); border: 1px solid rgba(70, 225, 255, .35); box-shadow: 0 10px 40px rgba(0,0,0,.5); }
  #__cap:empty { display: none; }
  #__chap { position: absolute; right: 16px; top: 12px; padding: 5px 12px; border-radius: 999px; font: 700 12px/1 'JetBrains Mono', Consolas, monospace;
    letter-spacing: .08em; color: #46e1ff; background: rgba(6,10,18,.85); border: 1px solid rgba(70,225,255,.45); text-transform: uppercase; }
  #__chap:empty { display: none; }
  #__prog { position: absolute; left: 0; bottom: 0; height: 4px; background: linear-gradient(90deg, #46e1ff, #8b7bff); transition: width .4s linear; }
  #__cursor { position: absolute; width: 22px; height: 22px; margin: -3px 0 0 -3px; transition: left .02s linear, top .02s linear; }
  #__cursor svg { width: 22px; height: 22px; filter: drop-shadow(0 2px 3px rgba(0,0,0,.6)); }
  .__ripple { position: absolute; width: 14px; height: 14px; margin: -7px 0 0 -7px; border-radius: 50%; border: 3px solid #46e1ff; animation: __rip .55s ease-out forwards; }
  @keyframes __rip { to { transform: scale(4); opacity: 0; } }
  #__card { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; flex-direction: column; text-align: center; padding: 60px;
    background: radial-gradient(circle at 30% 20%, #10243d 0%, #060a12 65%); color: #fff; }
  #__card h1 { font-size: 64px; line-height: 1.05; margin: 0 0 14px; letter-spacing: -.02em; }
  #__card h1 span { background: linear-gradient(90deg, #46e1ff, #8b7bff); -webkit-background-clip: text; background-clip: text; color: transparent; }
  #__card p { font-size: 26px; color: #b7c4d6; margin: 6px 0; max-width: 980px; }
  #__card .kv { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; margin-top: 26px; width: 1040px; }
  #__card .kv div { background: rgba(255,255,255,.06); border: 1px solid rgba(70,225,255,.25); border-radius: 14px; padding: 18px 12px; }
  #__card .kv b { display: block; font-size: 38px; color: #46e1ff; }
  #__card .kv small { color: #b7c4d6; font-size: 16px; }
  #__card img { max-height: 560px; border-radius: 12px; box-shadow: 0 20px 60px rgba(0,0,0,.6); }
  #__quiz { position: absolute; left: 50%; top: 86px; transform: translateX(-50%); display: none; width: 760px; padding: 20px 26px; border-radius: 18px; text-align: center;
    background: rgba(6,10,18,.94); border: 2px solid #46e1ff; box-shadow: 0 20px 60px rgba(0,0,0,.6); color: #fff; }
  #__quiz .q { font-size: 28px; font-weight: 700; margin-bottom: 12px; }
  #__quiz .opts { display: flex; gap: 12px; justify-content: center; margin: 8px 0 12px; }
  #__quiz .opt { flex: 1; padding: 12px 8px; border-radius: 12px; font-size: 22px; font-weight: 700; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.2); }
  #__quiz .opt.right { background: rgba(56, 224, 160, .22); border-color: #38e0a0; color: #38e0a0; }
  #__quiz .opt.wrong { opacity: .45; }
  #__quiz .count { font: 800 54px/1 'JetBrains Mono', Consolas, monospace; color: #46e1ff; }
  #__quiz .ans { font-size: 24px; font-weight: 700; margin-top: 4px; }
  `;
  const blurCss = `.match-row img, #preview, #p-photos img, #p-summary img, .person img, .cards img { filter: blur(15px) !important; }`;

  function build() {
    if (document.getElementById('__ov') || !document.body) return;
    const st = document.createElement('style'); st.id = '__ovcss'; st.textContent = css; document.head.append(st);
    const bl = document.createElement('style'); bl.id = '__ovblur'; document.head.append(bl);
    const root = document.createElement('div'); root.id = '__ov';
    root.innerHTML = '<div id="__prog"></div><div id="__chap"></div><div id="__cap"></div><div id="__quiz"></div><div id="__card"></div>' +
      '<div id="__cursor"><svg viewBox="0 0 24 24"><path d="M3 2l7.5 19 2.7-7.6L21 10.7z" fill="#fff" stroke="#0a0f1a" stroke-width="1.6" stroke-linejoin="round"/></svg></div>';
    document.body.append(root);
    render();
  }
  const $ = (id) => document.getElementById(id);
  function render() {
    if (!$('__ov')) return;
    $('__cap').textContent = state.cap;
    $('__chap').textContent = state.chap;
    $('__prog').style.width = `${Math.min(100, state.prog * 100)}%`;
    $('__cursor').style.left = `${state.cx}px`; $('__cursor').style.top = `${state.cy}px`;
    $('__ovblur').textContent = state.blur ? blurCss : '';
    document.documentElement.style.setProperty('--ovz', String(state.zoom || 1));
    const card = $('__card');
    if (state.card) { card.innerHTML = state.card; card.style.display = 'flex'; } else { card.style.display = 'none'; card.innerHTML = ''; }
    const quiz = $('__quiz');
    if (state.quiz) { quiz.style.display = 'block'; quiz.innerHTML = state.quiz; } else { quiz.style.display = 'none'; quiz.innerHTML = ''; }
  }
  window.__ov = {
    set(patch) { Object.assign(state, patch); save(); render(); },
    click(x, y) {
      const r = document.createElement('div'); r.className = '__ripple'; r.style.left = x + 'px'; r.style.top = y + 'px';
      $('__ov') && $('__ov').append(r); setTimeout(() => r.remove(), 700);
    },
  };
  document.addEventListener('mousemove', (e) => { state.cx = e.clientX; state.cy = e.clientY; save(); const c = $('__cursor'); if (c) { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; } }, true);
  document.addEventListener('mousedown', (e) => window.__ov.click(e.clientX, e.clientY), true);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build); else build();
})();
