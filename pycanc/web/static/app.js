/* PyCanc — workstation front-end */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const D = 200, H = 256, W = 256, PX = 1.40625, SL = 2.5;
  const STAGES = [
    ["prep", "Preprocess · window, resample, crop"],
    ["stem", "Encoder stem"],
    ["layer1", "Residual stage 1"],
    ["layer2", "Residual stage 2"],
    ["layer3", "Residual stage 3"],
    ["layer4", "Residual stage 4"],
    ["pool", "Multi-attention pooling"],
    ["head", "Cumulative hazard head"],
    ["cal", "Ensemble + calibration"],
  ];

  const S = {
    status: null, cases: [], caseId: null, vol: null, valid: [0, D - 1],
    cur: { z: 100, y: 128, x: 128 }, attn: null, result: null,
    win: "model", ov: true, op: 0.7, gt: true, nModels: 1, tta: 1, patients: {}, clinical: null, poll: null, active: "axial", hover: null,
  };

  // ------------------------------------------------------------------ utils
  const toast = (msg, ms = 2600) => { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), ms); };
  const api = async (url, opt = {}) => {
    const r = await fetch(url, opt);
    if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch { } throw new Error(m); }
    return r;
  };
  const pct = (p, d = 1) => (p * 100).toFixed(p < 0.1 ? Math.max(d, 1) : d);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function inferno(t) {
    t = clamp(t, 0, 1);
    const c = [[0.0002189, 0.0016510, -0.0194809], [0.1065134, 0.5639564, 3.9327124], [11.6024931, -3.9728540, -15.9423941],
      [-41.7039961, 17.4363989, 44.3541452], [77.1629357, -33.4023589, -81.8073093], [-71.3194282, 32.6260643, 73.2095199], [25.1311262, -12.2426690, -23.0703250]];
    return [0, 1, 2].map(i => { let v = c[6][i]; for (let k = 5; k >= 0; k--) v = c[k][i] + t * v; return clamp(Math.round(v * 255), 0, 255); });
  }
  const INF = Array.from({ length: 256 }, (_, i) => inferno(i / 255));

  // ------------------------------------------------------------------ LUTs
  function lut() {
    const L = new Uint8ClampedArray(256);
    if (S.win === "model") { for (let i = 0; i < 256; i++) L[i] = i; return L; }
    const [wl, ww] = { lung: [-600, 1500], medi: [40, 400], bone: [400, 1800] }[S.win];
    for (let i = 0; i < 256; i++) { const hu = -1000 + i * 2500 / 255; L[i] = clamp((hu - (wl - ww / 2)) / ww * 255, 0, 255); }
    return L;
  }
  const srcVol = () => (S.win === "model" ? S.vol.lung : S.vol.wide);

  // ------------------------------------------------------------------ attention sampling
  function attnAt(z, y, x) {
    const A = S.attn; if (!A) return 0;
    const [T, h, w] = A.shape;
    const gz = clamp((z + 0.5) * T / D - 0.5, 0, T - 1), gy = clamp((y + 0.5) * h / H - 0.5, 0, h - 1), gx = clamp((x + 0.5) * w / W - 0.5, 0, w - 1);
    const z0 = Math.floor(gz), y0 = Math.floor(gy), x0 = Math.floor(gx);
    const z1 = Math.min(z0 + 1, T - 1), y1 = Math.min(y0 + 1, h - 1), x1 = Math.min(x0 + 1, w - 1);
    const fz = gz - z0, fy = gy - y0, fx = gx - x0, d = A.data;
    const g = (a, b, c) => d[(a * h + b) * w + c];
    const c00 = g(z0, y0, x0) * (1 - fx) + g(z0, y0, x1) * fx, c01 = g(z0, y1, x0) * (1 - fx) + g(z0, y1, x1) * fx;
    const c10 = g(z1, y0, x0) * (1 - fx) + g(z1, y0, x1) * fx, c11 = g(z1, y1, x0) * (1 - fx) + g(z1, y1, x1) * fx;
    return ((c00 * (1 - fy) + c01 * fy) * (1 - fz) + (c10 * (1 - fy) + c11 * fy) * fz);
  }

  // ------------------------------------------------------------------ 2D views
  const views = {};
  for (const el of $$(".vp")) {
    const kind = el.dataset.view;
    if (kind === "3d") continue;
    const canvas = el.querySelector("canvas");
    views[kind] = { el, kind, canvas, ctx: canvas.getContext("2d"), off: document.createElement("canvas"), offA: document.createElement("canvas"), rect: null };
  }

  function planeDims(kind) {
    const rows = S.valid[1] - S.valid[0] + 1;
    if (kind === "axial") return { w: W, h: H, pw: W * PX, ph: H * PX };
    return { w: W, h: rows, pw: W * PX, ph: rows * SL };
  }

  function renderPlane(v) {
    if (!S.vol) return;
    const { w, h } = planeDims(v.kind), vol = srcVol(), L = lut();
    v.off.width = w; v.off.height = h; v.offA.width = w; v.offA.height = h;
    const img = v.off.getContext("2d").createImageData(w, h), ia = v.offA.getContext("2d").createImageData(w, h);
    const { z, y, x } = S.cur, z0 = S.valid[0];
    const withA = S.attn && S.ov;
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      let vz, vy, vx;
      if (v.kind === "axial") { vz = z; vy = r; vx = c; }
      else if (v.kind === "coronal") { vz = z0 + r; vy = y; vx = c; }
      else { vz = z0 + r; vy = c; vx = x; }
      const g = L[vol[vz * 65536 + vy * 256 + vx]], o = (r * w + c) * 4;
      img.data[o] = g; img.data[o + 1] = g; img.data[o + 2] = g; img.data[o + 3] = 255;
      if (withA) {
        const a = Math.pow(attnAt(vz, vy, vx), 0.55);
        if (a > 0.06) {
          const col = INF[Math.round(clamp(0.25 + 0.75 * a, 0, 1) * 255)];
          ia.data[o] = col[0]; ia.data[o + 1] = col[1]; ia.data[o + 2] = col[2];
          ia.data[o + 3] = 255 * S.op * clamp((a - 0.06) / 0.25, 0, 1) * (0.35 + 0.65 * a);
        }
      }
    }
    v.off.getContext("2d").putImageData(img, 0, 0);
    v.offA.getContext("2d").putImageData(ia, 0, 0);
    drawView(v);
  }

  function drawView(v) {
    const c = v.canvas, ctx = v.ctx, dpr = window.devicePixelRatio || 1;
    const cw = c.clientWidth, ch = c.clientHeight;
    if (!cw || !ch) return;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    const { pw, ph, w, h } = planeDims(v.kind);
    const pad = 26, s = Math.min((cw - pad * 2) / pw, (ch - pad * 2) / ph);
    const sw = pw * s, sh = ph * s, ox = (cw - sw) / 2, oy = (ch - sh) / 2;
    v.rect = { ox, oy, sw, sh, w, h };
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(v.off, ox, oy, sw, sh);
    if (S.attn && S.ov) { ctx.globalCompositeOperation = "screen"; ctx.drawImage(v.offA, ox, oy, sw, sh); ctx.globalCompositeOperation = "source-over"; ctx.drawImage(v.offA, ox, oy, sw, sh); }

    const { z, y, x } = S.cur, z0 = S.valid[0];
    const P = (col, row) => [ox + (col + 0.5) / w * sw, oy + (row + 0.5) / h * sh];
    let ch_ = v.kind === "axial" ? P(x, y) : v.kind === "coronal" ? P(x, z - z0) : P(y, z - z0);
    // crosshair
    ctx.strokeStyle = "rgba(59,224,255,0.55)"; ctx.lineWidth = 1;
    const gap = 12;
    ctx.beginPath();
    ctx.moveTo(ox, ch_[1]); ctx.lineTo(ch_[0] - gap, ch_[1]); ctx.moveTo(ch_[0] + gap, ch_[1]); ctx.lineTo(ox + sw, ch_[1]);
    ctx.moveTo(ch_[0], oy); ctx.lineTo(ch_[0], ch_[1] - gap); ctx.moveTo(ch_[0], ch_[1] + gap); ctx.lineTo(ch_[0], oy + sh);
    ctx.stroke();
    ctx.fillStyle = "#3be0ff"; ctx.beginPath(); ctx.arc(ch_[0], ch_[1], 2, 0, 7); ctx.fill();

    // ground-truth nodule
    const nod = currentCase()?.meta?.nodule;
    if (S.gt && nod?.model_zyx) {
      const [nz, ny, nx] = nod.model_zyx, rvox = nod.diameter_mm / 2;
      const dist = v.kind === "axial" ? Math.abs(nz - z) * SL : v.kind === "coronal" ? Math.abs(ny - y) * PX : Math.abs(nx - x) * PX;
      if (dist < rvox * 1.6) {
        const p = v.kind === "axial" ? P(nx, ny) : v.kind === "coronal" ? P(nx, nz - z0) : P(ny, nz - z0);
        const rr = (rvox * 1.7) * s + 6;
        ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(61,255,168,0.9)"; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(p[0], p[1], rr, 0, 7); ctx.stroke(); ctx.restore();
        ctx.font = "600 10px JetBrains Mono, monospace"; ctx.fillStyle = "rgba(61,255,168,0.95)";
        ctx.fillText("TRUE NODULE", p[0] + rr + 4, p[1] - rr + 6);
      }
    }
    // hotspots
    if (S.result && S.ov) S.result.hotspots.forEach((hs, i) => {
      const dist = v.kind === "axial" ? Math.abs(hs.z - z) * SL : v.kind === "coronal" ? Math.abs(hs.y - y) * PX : Math.abs(hs.x - x) * PX;
      if (dist > 14 || hs.score < 0.08) return;
      const p = v.kind === "axial" ? P(hs.x, hs.y) : v.kind === "coronal" ? P(hs.x, hs.z - z0) : P(hs.y, hs.z - z0);
      const rr = 16 * s + 6;
      ctx.strokeStyle = `rgba(255,181,71,${0.9 - dist / 20})`; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let k = 0; k < 4; k++) { const a0 = k * Math.PI / 2 + 0.25, a1 = a0 + Math.PI / 2 - 0.5; ctx.moveTo(p[0] + rr * Math.cos(a0), p[1] + rr * Math.sin(a0)); ctx.arc(p[0], p[1], rr, a0, a1); }
      ctx.stroke();
      ctx.fillStyle = "#ffb547"; ctx.font = "700 11px JetBrains Mono, monospace"; ctx.fillText(String(i + 1), p[0] - rr - 4, p[1] - rr - 2);
    });
    // scale bar (5 cm)
    const bar = 50 * s;
    ctx.strokeStyle = "rgba(255,255,255,0.5)"; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(cw - 18 - bar, ch - 16); ctx.lineTo(cw - 18, ch - 16); ctx.moveTo(cw - 18 - bar, ch - 20); ctx.lineTo(cw - 18 - bar, ch - 12); ctx.moveTo(cw - 18, ch - 20); ctx.lineTo(cw - 18, ch - 12); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.55)"; ctx.font = "10px JetBrains Mono, monospace"; ctx.fillText("5 cm", cw - 18 - bar / 2 - 12, ch - 22);
    updateInfo(v);
  }

  function updateInfo(v) {
    const { z, y, x } = S.cur, z0 = S.valid[0], rows = S.valid[1] - z0 + 1;
    let t = v.kind === "axial" ? `IM ${z - z0 + 1}/${rows} · ${((z - z0) * SL).toFixed(1)} mm`
      : v.kind === "coronal" ? `ROW ${y + 1}/256 · ${(y * PX).toFixed(1)} mm` : `COL ${x + 1}/256 · ${(x * PX).toFixed(1)} mm`;
    if (S.hover && S.hover.view === v.kind && S.vol) {
      const { hz, hy, hx } = S.hover, hu = Math.round(-1000 + S.vol.wide[hz * 65536 + hy * 256 + hx] * 2500 / 255);
      t += `<br>HU ${hu}${S.attn ? ` · attn ${attnAt(hz, hy, hx).toFixed(2)}` : ""}`;
    }
    v.el.querySelector(".inf").innerHTML = t;
  }

  const renderAll = () => { Object.values(views).forEach(renderPlane); r3d?.setCursor(S.cur.z, S.cur.y, S.cur.x); drawProfile(); };
  let rafPending = false;
  const scheduleRender = () => { if (rafPending) return; rafPending = true; requestAnimationFrame(() => { rafPending = false; renderAll(); }); };

  function setActive(kind) { S.active = kind; $$(".vp").forEach(e => e.classList.toggle("active", e.dataset.view === kind)); }

  function eventToVoxel(v, e) {
    const r = v.canvas.getBoundingClientRect(), R = v.rect; if (!R) return null;
    const px = (e.clientX - r.left - R.ox) / R.sw * R.w, py = (e.clientY - r.top - R.oy) / R.sh * R.h;
    if (px < 0 || py < 0 || px >= R.w || py >= R.h) return null;
    const c = Math.floor(px), row = Math.floor(py), z0 = S.valid[0];
    if (v.kind === "axial") return { hz: S.cur.z, hy: row, hx: c };
    if (v.kind === "coronal") return { hz: z0 + row, hy: S.cur.y, hx: c };
    return { hz: z0 + row, hy: c, hx: S.cur.x };
  }

  Object.values(views).forEach(v => {
    let down = false;
    const apply = (e) => {
      const p = eventToVoxel(v, e); if (!p) return;
      if (v.kind === "axial") { S.cur.y = p.hy; S.cur.x = p.hx; }
      else if (v.kind === "coronal") { S.cur.z = p.hz; S.cur.x = p.hx; }
      else { S.cur.z = p.hz; S.cur.y = p.hy; }
      scheduleRender();
    };
    v.canvas.addEventListener("pointerdown", e => { down = true; setActive(v.kind); v.canvas.setPointerCapture(e.pointerId); apply(e); });
    v.canvas.addEventListener("pointermove", e => {
      if (down) apply(e);
      const p = eventToVoxel(v, e); S.hover = p ? { ...p, view: v.kind } : null; updateInfo(v);
    });
    v.canvas.addEventListener("pointerup", () => down = false);
    v.canvas.addEventListener("pointerleave", () => { S.hover = null; updateInfo(v); });
    v.canvas.addEventListener("wheel", e => {
      e.preventDefault(); if (!S.vol) return;
      const d = Math.sign(e.deltaY);
      if (v.kind === "axial") S.cur.z = clamp(S.cur.z + d, S.valid[0], S.valid[1]);
      else if (v.kind === "coronal") S.cur.y = clamp(S.cur.y + d, 0, H - 1);
      else S.cur.x = clamp(S.cur.x + d, 0, W - 1);
      scheduleRender();
    }, { passive: false });
    v.el.addEventListener("dblclick", () => toggleFocus(v.kind));
  });
  $('.vp[data-view="3d"]').addEventListener("dblclick", () => toggleFocus("3d"));
  window.addEventListener("keydown", e => {
    if (!S.vol || e.target.tagName === "INPUT") return;
    const d = { ArrowUp: -1, ArrowDown: 1, PageUp: -5, PageDown: 5 }[e.key]; if (!d) return;
    e.preventDefault(); S.cur.z = clamp(S.cur.z + d, S.valid[0], S.valid[1]); scheduleRender();
  });
  window.addEventListener("resize", () => Object.values(views).forEach(drawView));

  function toggleFocus(kind) {
    const g = $("#grid"), single = !g.classList.contains("single") || !$(`.vp[data-view="${kind}"]`).classList.contains("focus");
    $$(".vp").forEach(e => e.classList.toggle("focus", e.dataset.view === kind));
    g.classList.toggle("single", single);
    $("#lay-one").classList.toggle("on", single); $("#lay-grid").classList.toggle("on", !single);
    requestAnimationFrame(() => { Object.values(views).forEach(drawView); if (r3d) r3d.dirty = true; });
  }
  $("#lay-grid").onclick = () => { $("#grid").classList.remove("single"); $("#lay-grid").classList.add("on"); $("#lay-one").classList.remove("on"); requestAnimationFrame(() => { Object.values(views).forEach(drawView); if (r3d) r3d.dirty = true; }); };
  $("#lay-one").onclick = () => toggleFocus(S.active);

  // ------------------------------------------------------------------ 3D
  let r3d = null;
  try { r3d = new VolumeRenderer($('.vp[data-view="3d"] canvas')); if (r3d.failed) throw new Error("no webgl2"); }
  catch (e) { console.warn(e); r3d = null; $('.vp[data-view="3d"] .inf').textContent = "WebGL2 unavailable in this browser"; }
  $("#mode3d").onchange = e => { if (r3d) { r3d.mode = +e.target.value; r3d.dirty = true; } };
  $("#rot").onclick = e => { if (!r3d) return; r3d.autoRotate = !r3d.autoRotate; e.currentTarget.classList.toggle("on", r3d.autoRotate); r3d.dirty = true; };

  // ------------------------------------------------------------------ toolbar
  $("#win").onchange = e => { S.win = e.target.value; scheduleRender(); };
  $("#ov").onchange = e => { S.ov = e.target.checked; if (r3d) { r3d.attnOn = S.ov; r3d.dirty = true; } scheduleRender(); };
  $("#ov-op").oninput = e => { S.op = +e.target.value; if (r3d) { r3d.attnGain = 0.3 + S.op * 1.2; r3d.dirty = true; } scheduleRender(); };
  $("#gt").onchange = e => { S.gt = e.target.checked; Object.values(views).forEach(drawView); };

  // ------------------------------------------------------------------ phantom controls
  const segVal = id => $(`#${id} .on`)?.dataset.v;
  $$(".seg").forEach(seg => seg.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    seg.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    if (seg.id === "ens") { S.nModels = +b.dataset.v; updateEta(); }
    if (seg.id === "tta") { S.tta = +b.dataset.v; updateEta(); }
  }));
  $("#p-mm").oninput = e => $("#o-mm").textContent = `${e.target.value} mm`;
  $("#p-lvl").oninput = e => { const v = +e.target.value; $("#o-lvl").textContent = v < 0.34 ? "upper" : v < 0.67 ? "middle" : "lower"; };
  $("#p-nod").onchange = e => $("#nod-ctrls").style.opacity = e.target.checked ? 1 : 0.35;
  $("#dice").onclick = () => $("#p-seed").value = Math.floor(Math.random() * 9999);
  $("#gen").onclick = () => createPhantom();

  async function createPhantom(params) {
    const body = params || {
      seed: +$("#p-seed").value || 0, nodule: $("#p-nod").checked, nodule_mm: +$("#p-mm").value,
      nodule_type: segVal("p-type"), spiculated: $("#p-spic").checked, nodule_side: segVal("p-side"), nodule_level: +$("#p-lvl").value,
    };
    showEmpty("Synthesising chest CT…");
    $("#gen").disabled = true;
    try {
      const c = await (await api("/api/cases/phantom", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
      await refreshCases(); await openCase(c.id);
    } catch (e) { toast(e.message); hideEmpty(); }
    $("#gen").disabled = false;
  }

  // ------------------------------------------------------------------ upload
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));
  $("#file").onchange = e => e.target.files[0] && upload(e.target.files[0]);
  async function upload(f) {
    showEmpty(`Reading ${f.name}…`);
    const fd = new FormData(); fd.append("file", f);
    try { const c = await (await api("/api/cases/upload", { method: "POST", body: fd })).json(); await refreshCases(); await openCase(c.id); toast(`Loaded ${f.name}`); }
    catch (e) { toast(e.message, 5000); hideEmpty(); }
  }

  // ------------------------------------------------------------------ cases
  const currentCase = () => S.cases.find(c => c.id === S.caseId);
  async function refreshCases() {
    S.cases = await (await api("/api/cases")).json();
    const el = $("#cases");
    if (!S.cases.length) { el.innerHTML = `<div class="placeholder">No cases yet</div>`; return; }
    el.innerHTML = S.cases.map(c => {
      const r = c.job?.result?.risk?.[5];
      const st = c.job?.state === "running" || c.job?.state === "queued" ? "analysing…" : r != null ? "" : "not analysed";
      return `<div class="case ${c.id === S.caseId ? "active" : ""}" data-id="${c.id}">
        <div class="n">${c.name}</div><div class="r">${r != null ? pct(r) + "%" : ""}</div>
        <div class="m">${c.meta?.source || ""} · ${c.shape[0]} slices ${st ? "· " + st : ""}</div></div>`;
    }).join("");
    $$(".case", el).forEach(e => e.onclick = () => openCase(e.dataset.id));
  }

  function showEmpty(msg) { const e = $("#empty"); e.classList.remove("hidden"); e.querySelector("div").innerHTML = `<div class="spinner"></div>${msg}`; }
  function hideEmpty() { $("#empty").classList.add("hidden"); }

  async function openCase(id) {
    S.caseId = id; stopPoll();
    const c = currentCase();
    showEmpty("Streaming volume…");
    const get = k => api(`/api/cases/${id}/volume/${k}`).then(r => r.arrayBuffer()).then(b => new Uint8Array(b));
    const [lung, wide, mask] = await Promise.all([get("lung"), get("wide"), get("mask")]);
    if (S.caseId !== id) return;
    S.vol = { lung, wide, mask }; S.valid = c.valid;
    const nod = c.meta?.nodule?.model_zyx;
    S.cur = nod ? { z: nod[0], y: nod[1], x: nod[2] } : { z: Math.round((c.valid[0] + c.valid[1]) / 2), y: 128, x: 128 };
    S.attn = null; S.result = null;
    if (r3d) { r3d.setVolume(wide, mask, [D, H, W], c.valid); r3d.clearAttention(); }
    hideEmpty();
    await refreshCases();
    renderResult(null);
    renderAll();
    if (c.job?.state === "done") applyResult(c.job.result);
    else if (c.job?.state === "running" || c.job?.state === "queued") startPoll();
    else setPipe(null);
  }

  // ------------------------------------------------------------------ predict
  $("#run").onclick = async () => {
    if (!S.caseId) return toast("Load or generate a CT first");
    try {
      await api(`/api/cases/${S.caseId}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ n_models: S.nModels, tta: S.tta }) });
      startPoll();
    } catch (e) { toast(e.message); }
  };
  function stopPoll() { clearInterval(S.poll); S.poll = null; $("#run").disabled = false; $(".scan").classList.add("hidden"); $$(".arch .blk").forEach(b => b.classList.remove("live")); }
  function startPoll() {
    stopPoll();
    $("#run").disabled = true; $(".scan").classList.remove("hidden");
    const id = S.caseId, t0 = Date.now();
    S.poll = setInterval(async () => {
      if (S.caseId !== id) return stopPoll();
      const j = await (await api(`/api/cases/${id}/job`)).json();
      if (j.state === "done") { stopPoll(); applyResult(j.result); refreshCases(); toast(`PyCanc finished in ${j.result.seconds}s`); }
      else if (j.state === "error") { stopPoll(); setPipe(null); toast("Error: " + j.error, 6000); }
      else setPipe(j.progress, (Date.now() - t0) / 1000);
    }, 600);
    setPipe({}, 0);
  }

  function setPipe(p, elapsed) {
    const el = $("#pipe");
    let idx = -1, frac = 0;
    if (p === "done") { idx = STAGES.length; frac = 1; }
    else if (p) {
      const si = p.stage ? STAGES.findIndex(s => s[0] === p.stage) : 0;
      idx = Math.max(0, si);
      const n = (p.of || S.nModels) * (p.passes || 1), m = (p.pass || 0) * (p.of || 1) + (p.model || 0);
      frac = p.stage ? (m + (si - 1) / 7) / n : 0.01;
    }
    el.innerHTML = STAGES.map(([k, label], i) => `<div class="s ${i < idx ? "done" : i === idx ? "run" : ""}"><i></i>${label}</div>`).join("");
    $("#prog").style.width = `${clamp(frac, 0, 1) * 100}%`;
    $("#pipe-msg").textContent = p && p !== "done" && p.of ? `Model ${p.model + 1} of ${p.of}${p.passes > 1 ? ` · pass ${p.pass + 1}/${p.passes}` : ""} · ${elapsed?.toFixed(0)}s elapsed` : p === "done" ? "" : p ? "Queued…" : "";
    $$(".arch .blk").forEach(b => b.classList.toggle("live", p && p !== "done" && b.dataset.stage === (p.stage || "")));
  }

  function applyResult(res) {
    S.result = res;
    const bin = atob(res.attention_b64), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    S.attn = { shape: res.attention_shape, data: Float32Array.from(u8, v => v / 255) };
    if (r3d) r3d.setAttention(u8, res.attention_shape);
    const h = res.hotspots[0];
    if (h) S.cur = { z: h.z, y: h.y, x: h.x };
    renderResult(res); setPipe("done"); renderAll();
  }

  // ------------------------------------------------------------------ results panel
  const SQ = p => Math.sqrt(clamp(p, 0, 1));
  (function ticks() {
    const g = $("#ticks"); let html = "";
    [0.01, 0.05, 0.1, 0.25, 0.5, 1].forEach(p => {
      const a = (135 + 270 * SQ(p)) * Math.PI / 180, r1 = 106, r2 = 112, rl = 122;
      html += `<line x1="${120 + r1 * Math.cos(a)}" y1="${120 + r1 * Math.sin(a)}" x2="${120 + r2 * Math.cos(a)}" y2="${120 + r2 * Math.sin(a)}"/>`;
      html += `<text x="${120 + rl * Math.cos(a)}" y="${120 + rl * Math.sin(a) + 3}" text-anchor="middle" fill="#56617f" font-size="8.5" font-family="JetBrains Mono">${Math.round(p * 100)}%</text>`;
    });
    g.innerHTML = html;
  })();

  function animateNumber(el, to, ms = 1300) {
    const from = parseFloat(el.dataset.v || 0), t0 = performance.now();
    el.dataset.v = to;
    const step = t => { const k = clamp((t - t0) / ms, 0, 1), e = 1 - Math.pow(1 - k, 3), v = from + (to - from) * e; el.innerHTML = `${v.toFixed(to < 100 ? 1 : 0)}<small>%</small>`; if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }

  function renderResult(res) {
    const yrs = $("#years");
    if (!res) {
      $("#arc").setAttribute("stroke-dasharray", "0 578"); $("#arc").style.opacity = 0; $("#big").innerHTML = "—"; $("#big").dataset.v = 0;
      $("#big-sub").textContent = S.caseId ? "press ▶ to run PyCanc" : "load a CT";
      yrs.innerHTML = Array.from({ length: 6 }, (_, i) => `<div class="year"><b>—</b><span>YR ${i + 1}</span></div>`).join("");
      $("#curve").innerHTML = ""; $("#hot").innerHTML = `<div class="placeholder">Hotspots appear after prediction</div>`;
      renderWarn(null); renderClinical(); return;
    }
    const r6 = res.risk[5];
    $("#arc").style.opacity = 1;
    $("#arc").setAttribute("stroke-dasharray", `${433.5 * SQ(r6)} 578`);
    animateNumber($("#big"), r6 * 100);
    const spread = res.risk_low && (res.models_used > 1 || res.tta_passes > 1);
    $("#big-sub").innerHTML = (spread ? `<div class="range">95% range ${pct(res.risk_low[5])}–${pct(res.risk_high[5])}%</div>` : "")
      + `${res.models_used} model${res.models_used > 1 ? "s" : ""}${res.tta_passes > 1 ? ` × ${res.tta_passes} TTA` : ""} · ${Math.round(res.seconds)} s`;
    $("#big-sub").title = res.calibrated ? `${res.calibration} calibration` : "uncalibrated";
    yrs.innerHTML = res.risk.map((p, i) => `<div class="year" title="P(cancer within ${i + 1} yr)${spread ? ` · 95% range ${pct(res.risk_low[i])}–${pct(res.risk_high[i])}%` : ""}"><div class="fill" style="height:0%"></div><b>${pct(p)}%</b><span>YR ${i + 1}</span></div>`).join("");
    requestAnimationFrame(() => $$(".year .fill").forEach((f, i) => f.style.height = `${SQ(res.risk[i]) * 100}%`));
    drawCurve(res); drawHot(res); renderWarn(res); renderClinical();
  }

  function renderWarn(res) {
    const el = $("#warn"), st = S.status;
    if (st && !st.official_weights) {
      el.innerHTML = `<div class="warn-box"><b>Untrained weights.</b> The network architecture is exact, but no official checkpoints were found, so these numbers are random. Load the MIT ensemble to get real predictions.<div style="margin-top:8px"><button class="btn ghost" id="dl2">Get official weights (~700 MB)</button></div></div>`;
      $("#dl2").onclick = downloadWeights;
    } else if (res && currentCase()?.meta?.source === "Synthetic phantom") {
      el.innerHTML = `<div class="warn-box" style="border-color:rgba(59,224,255,.3);background:rgba(59,224,255,.06);color:#bfefff"><b style="color:var(--cyan)">Phantom scan.</b> Real MIT Sybil weights on a synthetic CT: useful for seeing how the model reacts to nodule size, density and location, not as a real risk estimate.</div>`;
    } else el.innerHTML = "";
    const qw = currentCase()?.warnings || [];
    if (qw.length) el.innerHTML += `<div class="warn-box" style="margin-top:8px"><b>Scan check.</b> ${qw.join(" ")}</div>`;
  }

  function drawCurve(res) {
    const svg = $("#curve"), Wd = 330, Hd = 190, L = 34, R = 14, T = 12, B = 24;
    const all = [...res.risk, ...res.raw_ensemble, ...res.per_model.flat(), ...(res.risk_high || [])];
    const nice = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1];
    const ymax = nice.find(v => v >= Math.max(...all) * 1.1) || 1;
    const X = t => L + t / 6 * (Wd - L - R), Y = p => Hd - B - p / ymax * (Hd - T - B);
    const path = arr => "M" + [[0, 0], ...arr.map((p, i) => [i + 1, p])].map(([t, p]) => `${X(t).toFixed(1)},${Y(p).toFixed(1)}`).join("L");
    let g = `<defs><linearGradient id="lineGrad" x1="0" x2="1"><stop offset="0" stop-color="#8b6bff"/><stop offset="1" stop-color="#ff4d7d"/></linearGradient>
      <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff4d7d" stop-opacity=".35"/><stop offset="1" stop-color="#8b6bff" stop-opacity="0"/></linearGradient></defs><g class="grid">`;
    for (let k = 0; k <= 4; k++) { const p = ymax * k / 4; g += `<line x1="${L}" x2="${Wd - R}" y1="${Y(p)}" y2="${Y(p)}"/><text x="${L - 6}" y="${Y(p) + 3}" text-anchor="end">${(p * 100).toFixed(ymax < 0.1 ? 1 : 0)}%</text>`; }
    for (let t = 0; t <= 6; t++) g += `<text x="${X(t)}" y="${Hd - 8}" text-anchor="middle">${t}y</text>`;
    g += `</g>`;
    if (res.per_model.length > 1) res.per_model.forEach(m => g += `<path class="member" d="${path(m)}"/>`);
    g += `<path class="rawl" d="${path(res.raw_ensemble)}"/>`;
    g += `<path d="${path(res.risk)}L${X(6)},${Y(0)}Z" fill="url(#areaGrad)"/>`;
    if (res.risk_low && (res.models_used > 1 || res.tta_passes > 1)) {
      const up = res.risk_high.map((p, i) => `${X(i + 1).toFixed(1)},${Y(p).toFixed(1)}`), lo = res.risk_low.map((p, i) => `${X(i + 1).toFixed(1)},${Y(p).toFixed(1)}`).reverse();
      g += `<path class="band" d="M${X(0)},${Y(0)}L${up.join("L")}L${lo.join("L")}Z"><title>95% uncertainty range</title></path>`;
    }
    g += `<path class="main" id="mainline" d="${path(res.risk)}"/>`;
    res.risk.forEach((p, i) => g += `<circle class="pt" cx="${X(i + 1)}" cy="${Y(p)}" r="3.2"><title>Year ${i + 1}: ${pct(p, 2)}%</title></circle>`);
    g += `<text x="${X(6) - 4}" y="${Y(res.risk[5]) - 9}" text-anchor="end" style="fill:#fff;font-weight:700">${pct(res.risk[5])}%</text>`;
    svg.innerHTML = g;
    const ml = $("#mainline"), len = ml.getTotalLength();
    ml.style.strokeDasharray = len; ml.style.strokeDashoffset = len; ml.getBoundingClientRect();
    ml.style.transition = "stroke-dashoffset 1.4s cubic-bezier(.2,.8,.2,1)"; ml.style.strokeDashoffset = 0;
  }

  function drawProfile() {
    const svg = $("#profile"), Wd = 330, Hd = 70;
    if (!S.result) { svg.innerHTML = `<text x="165" y="40" text-anchor="middle" fill="#56617f" font-size="11">attention across slices appears here</text>`; return; }
    const sa = S.result.slice_attention, n = sa.length, [z0, z1] = S.valid;
    const X = z => (z - z0) / (z1 - z0 + 1) * Wd;
    let g = "";
    sa.forEach((v, i) => {
      const za = i * D / n, zb = (i + 1) * D / n;
      if (zb < z0 || za > z1) return;
      const x0 = X(Math.max(za, z0)), x1 = X(Math.min(zb, z1 + 1)), h = Math.max(1.5, v * (Hd - 12));
      const c = INF[Math.round((0.2 + 0.8 * v) * 255)];
      g += `<rect x="${x0 + 0.6}" y="${Hd - 4 - h}" width="${Math.max(1, x1 - x0 - 1.2)}" height="${h}" rx="2" fill="rgb(${c})" opacity="${0.45 + 0.55 * v}"/>`;
    });
    g += `<line x1="${X(S.cur.z + 0.5)}" x2="${X(S.cur.z + 0.5)}" y1="0" y2="${Hd}" stroke="#3be0ff" stroke-width="1.5"/>`;
    g += `<text x="2" y="10" fill="#56617f" font-size="9" font-family="JetBrains Mono">apex</text><text x="${Wd - 2}" y="10" text-anchor="end" fill="#56617f" font-size="9" font-family="JetBrains Mono">base</text>`;
    svg.innerHTML = g;
  }
  $("#profile").addEventListener("click", e => {
    if (!S.vol) return;
    const r = e.currentTarget.getBoundingClientRect(), [z0, z1] = S.valid;
    S.cur.z = clamp(Math.floor(z0 + (e.clientX - r.left) / r.width * (z1 - z0 + 1)), z0, z1); scheduleRender();
  });

  function drawHot(res) {
    const nod = currentCase()?.meta?.nodule?.model_zyx;
    $("#hot").innerHTML = res.hotspots.map((h, i) => {
      let gt = "";
      if (nod) { const d = Math.hypot((h.z - nod[0]) * SL, (h.y - nod[1]) * PX, (h.x - nod[2]) * PX); gt = d < 25 ? ` · <span class="gt">on true nodule (${d.toFixed(0)} mm)</span>` : ""; }
      return `<div class="h" data-i="${i}"><div class="k">${i + 1}</div>
        <div><div class="bar"><i style="width:${h.score * 100}%"></i></div><div class="loc">IM ${h.z - S.valid[0] + 1} · ${h.x < 128 ? "right" : "left"} lung${gt}</div></div>
        <div class="loc" style="text-align:right">${(h.score * 100).toFixed(0)}%</div></div>`;
    }).join("");
    $$("#hot .h").forEach(e => e.onclick = () => { const h = res.hotspots[+e.dataset.i]; S.cur = { z: h.z, y: h.y, x: h.x }; scheduleRender(); });
  }

  // ------------------------------------------------------------------ status / weights
  async function loadStatus() {
    S.status = await (await api("/api/status")).json();
    const st = S.status, el = $("#status");
    const dl = st.download?.state === "running";
    el.innerHTML = `
      ${st.official_weights ? `<span class="chip ok"><span class="dot"></span>MIT weights · ${st.num_models} models</span>`
        : `<span class="chip warn"><span class="dot"></span>${dl ? "Downloading weights…" : "Untrained weights"}</span>${dl ? "" : `<button class="btn ghost" id="dl">Get official weights</button>`}`}
      <span class="chip info"><span class="dot"></span>${st.calibrated ? "calibrated" : "uncalibrated"}</span>
      <span class="chip"><span class="dot"></span>${st.device === "mps" ? "APPLE GPU" : st.device.toUpperCase()}${st.device === "cpu" ? ` · ${st.threads} threads` : ""}${st.half_precision ? " · FP16" : ""}</span>
      ${st.local_calibration ? `<span class="chip ok"><span class="dot"></span>local calibration</span>` : ""}`;
    $("#dl") && ($("#dl").onclick = downloadWeights);
    $$("#ens button").forEach(b => b.disabled = +b.dataset.v > st.num_models);
    if (S.nModels > st.num_models) { S.nModels = st.num_models; $$("#ens button").forEach(b => b.classList.toggle("on", +b.dataset.v === S.nModels)); }
    updateEta(); renderWarn(S.result);
    if (dl) setTimeout(loadStatus, 3000);
    else if (st.download?.state === "error") toast("Download failed: " + st.download.message, 6000);
  }
  async function downloadWeights() { await api("/api/weights/download", { method: "POST" }); toast("Downloading official MIT checkpoints…"); loadStatus(); }
  function updateEta() {
    const runs = S.nModels * S.tta, dev = S.status?.device;
    const per = dev === "cuda" ? 3 : dev === "mps" ? 15 : 60;
    const secs = runs * per;
    $("#eta").textContent = `${runs} forward pass${runs > 1 ? "es" : ""} · ≈ ${secs < 90 ? secs + " s" : Math.round(secs / 60) + " min"} on ${dev === "mps" ? "Apple GPU" : (dev || "cpu").toUpperCase()}`;
  }

  // ------------------------------------------------------------------ clinical (PLCOm2012)
  const cv = id => $(id).value;
  $("#c-smk").onchange = () => $("#c-quit-l").style.opacity = cv("#c-smk") === "former" ? 1 : 0.4;
  $("#c-smk").onchange();
  function patientForm() {
    return {
      age: +cv("#c-age"), bmi: +cv("#c-bmi"), smoking_status: cv("#c-smk"), cigarettes_per_day: +cv("#c-cpd"),
      smoking_years: +cv("#c-yrs"), years_since_quit: cv("#c-smk") === "former" ? +cv("#c-quit") : 0,
      education: cv("#c-edu"), race: cv("#c-race"), copd: $("#c-copd").checked,
      family_lung_cancer: $("#c-fam").checked, personal_cancer_history: $("#c-hx").checked,
    };
  }
  $("#c-go").onclick = async () => {
    if (!S.caseId) return toast("Load a case first");
    S.patients[S.caseId] = patientForm();
    await renderClinical();
  };
  async function renderClinical() {
    const el = $("#clin"), pt = S.patients[S.caseId];
    if (!pt) { el.innerHTML = `<div class="placeholder">Add patient history on the left to compare image-based and clinical risk</div>`; return; }
    let r;
    try { r = await (await api("/api/clinical", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ patient: pt, image_risk: S.result?.risk || null }) })).json(); }
    catch (e) { el.innerHTML = `<div class="placeholder">${e.message}</div>`; return; }
    if (!r.applicable) { el.innerHTML = `<div class="small">${r.note}</div>`; return; }
    const max = Math.max(0.1, r.risk * 1.3, S.result ? S.result.risk[5] * 1.3 : 0);
    const img = S.result ? S.result.risk[5] : null;
    el.innerHTML = `
      <div class="crow"><b>${pct(r.risk, 2)}%</b><span class="tag ${r.eligible ? "hi" : "lo"}">${r.eligible ? "≥ 1.51% · screening-eligible" : "below 1.51% threshold"}</span></div>
      <div class="meter"><i style="width:0"></i><u style="left:${r.threshold / max * 100}%" title="1.51% PLCOm2012 screening threshold"></u></div>
      <div class="cmp">
        <div><b>${img != null ? pct(img) + "%" : "—"}</b><span>Image</span></div>
        <div><b>${pct(r.risk)}%</b><span>Clinical</span></div>
        <div><b>${r.fused ? pct(r.fused[5]) + "%" : "—"}</b><span>Combined</span></div>
      </div>
      <div class="small">${r.fused ? "Combined = image + clinical fusion fitted on your local outcomes." : r.fusion_available ? "Run a prediction to see the combined score." : "A combined score appears once a fusion model is fitted on local outcomes (python -m pycanc.evaluate --fit-fusion)."} ${r.note || ""}</div>`;
    requestAnimationFrame(() => el.querySelector(".meter i").style.width = `${Math.min(100, r.risk / max * 100)}%`);
  }

  // ------------------------------------------------------------------ snapshot / report
  function composite() {
    const vs = ["axial", "3d", "coronal", "sagittal"].map(k => $(`.vp[data-view="${k}"] canvas`));
    if (r3d) r3d.render();
    const w = 900, h = 900, out = document.createElement("canvas"); out.width = w * 2; out.height = h * 2;
    const ctx = out.getContext("2d"); ctx.fillStyle = "#02040a"; ctx.fillRect(0, 0, out.width, out.height);
    vs.forEach((c, i) => {
      const x = (i % 2) * w, y = Math.floor(i / 2) * h, s = Math.min(w / c.width, h / c.height);
      ctx.drawImage(c, x + (w - c.width * s) / 2, y + (h - c.height * s) / 2, c.width * s, c.height * s);
    });
    return out;
  }
  $("#shot").onclick = () => { const a = document.createElement("a"); a.download = `pycanc-${S.caseId || "view"}.png`; a.href = composite().toDataURL("image/png"); a.click(); };
  $("#report").onclick = () => {
    const res = S.result, c = currentCase(); if (!res) return toast("Run a prediction first");
    const img = composite().toDataURL("image/jpeg", 0.9), date = new Date().toLocaleString();
    const w = window.open("", "_blank");
    w.document.write(`<!doctype html><html><head><title>PyCanc report — ${c.name}</title><style>
      body{font:14px Inter,system-ui,sans-serif;color:#111;max-width:860px;margin:30px auto;padding:0 20px}
      h1{font-size:22px;margin:0}h2{font-size:15px;margin:26px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
      table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px 8px;text-align:center;font-size:13px}
      th{background:#f4f6fb}.big{font-size:40px;font-weight:800}.muted{color:#666;font-size:12px}img{width:100%;border-radius:8px}
      .warn{background:#fff6e5;border:1px solid #f0c27a;padding:10px;border-radius:8px;font-size:12.5px}</style></head><body>
      <h1>PyCanc — lung cancer risk report</h1><div class="muted">${c.name} · ${c.meta?.source || ""} · generated ${date}</div>
      <h2>Predicted risk</h2><div class="big">${pct(res.risk[5])}%</div><div class="muted">probability of lung cancer diagnosis within 6 years (${res.models_used}-model ensemble, ${res.calibrated ? "calibrated" : "uncalibrated"})</div>
      <table style="margin-top:12px"><tr><th></th>${res.risk.map((_, i) => `<th>Year ${i + 1}</th>`).join("")}</tr>
      <tr><td>Calibrated</td>${res.risk.map(p => `<td>${pct(p, 2)}%</td>`).join("")}</tr>
      ${res.risk_low && (res.models_used > 1 || res.tta_passes > 1) ? `<tr><td>95% range</td>${res.risk.map((_, i) => `<td>${pct(res.risk_low[i])}–${pct(res.risk_high[i])}%</td>`).join("")}</tr>` : ""}
      <tr><td>Raw score</td>${res.raw_ensemble.map(p => `<td>${pct(p, 2)}%</td>`).join("")}</tr></table>
      ${$("#clin .crow b") ? `<h2>Clinical risk (PLCOm2012)</h2><div>6-year risk from smoking and clinical history: <b>${$("#clin .crow b").textContent}</b> · ${$("#clin .tag").textContent}</div>` : ""}
      <h2>Model attention</h2><img src="${img}"><table style="margin-top:10px"><tr><th>#</th><th>Slice</th><th>Side</th><th>Relative attention</th></tr>
      ${res.hotspots.map((h, i) => `<tr><td>${i + 1}</td><td>${h.z - S.valid[0] + 1}</td><td>${h.x < 128 ? "right" : "left"}</td><td>${(h.score * 100).toFixed(0)}%</td></tr>`).join("")}</table>
      <h2>Notes</h2><div class="warn">${res.official_weights ? "" : "<b>Untrained weights were used — values are not meaningful.</b> "}Research replica of Sybil (Mikhael et al., J Clin Oncol 2023). Not a medical device; not for diagnosis or clinical decision-making.</div>
      <script>setTimeout(()=>print(),400)<\/script></body></html>`);
    w.document.close();
  };

  // ------------------------------------------------------------------ pages
  function goPage(page) {
    $$(".tabs button").forEach(x => x.classList.toggle("active", x.dataset.page === page));
    ["home", "workspace", "model", "about"].forEach(p => $(`#page-${p}`).classList.toggle("hidden", p !== page));
    window.scrollTo({ top: 0 });
    if (page === "workspace") requestAnimationFrame(() => { Object.values(views).forEach(drawView); if (r3d) r3d.dirty = true; });
  }
  $$(".tabs button").forEach(b => b.onclick = () => goPage(b.dataset.page));
  $$("[data-go]").forEach(b => b.onclick = () => {
    const g = b.dataset.go;
    goPage(g === "demo" ? "workspace" : g);
    if (g === "demo") { const t = setInterval(() => { if (S.vol) { clearInterval(t); if (!S.result) $("#run").click(); } }, 300); }
  });

  // welcome: count-up stats + terminal lines
  (function welcome() {
    $$("[data-count]").forEach(el => {
      const to = +el.dataset.count, dec = +(el.dataset.dec || 0), t0 = performance.now() + 1500;
      const step = t => { const k = clamp((t - t0) / 1400, 0, 1), e = 1 - Math.pow(1 - k, 3); el.textContent = (to * e).toFixed(dec); if (k < 1) requestAnimationFrame(step); };
      requestAnimationFrame(step);
    });
    const term = $("#term"); if (!term) return;
    const lines = ["> pycanc.init()", "  loading 3D ResNet-18 encoder … ok", "  attention pooling · cumulative hazard head … ok"];
    let i = 0, waits = 0;
    const next = () => {
      if (i < lines.length) { term.innerHTML += `<div>${lines[i++]}</div>`; setTimeout(next, 650); return; }
      const st = S.status;
      if (!st && waits++ < 8) { setTimeout(next, 700); return; }
      term.innerHTML += st ? `<div class="${st.official_weights ? "ok" : "warn"}">  ${st.official_weights ? `✓ ${st.num_models} official MIT models ready on ${st.device === "mps" ? "Apple GPU" : st.device.toUpperCase()}` : "! no trained weights found · click “Get official weights”"}</div>`
        : `<div class="warn">  ! server offline</div>`;
    };
    setTimeout(next, 1700);
  })();

  (function stack() {
    let s = "";
    for (let i = 7; i >= 0; i--) s += `<rect x="${i * 5}" y="${i * 6}" width="72" height="72" rx="6" fill="url(#vol)" stroke="rgba(59,224,255,${0.15 + (7 - i) * 0.06})"/>`;
    s += `<ellipse cx="22" cy="36" rx="9" ry="16" fill="none" stroke="rgba(59,224,255,.6)"/><ellipse cx="50" cy="36" rx="9" ry="16" fill="none" stroke="rgba(59,224,255,.6)"/><circle cx="25" cy="30" r="3" fill="#ff4d7d"/>`;
    $("#stack").innerHTML = s;
  })();

  // ------------------------------------------------------------------ boot
  (async () => {
    setPipe(null); renderResult(null); drawProfile();
    try { await loadStatus(); } catch (e) { $("#status").innerHTML = `<span class="chip warn"><span class="dot"></span>Server offline</span>`; return; }
    await refreshCases();
    if (S.cases.length) openCase(S.cases[0].id); else createPhantom();
  })();
})();
