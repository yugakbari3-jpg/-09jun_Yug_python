/* PyCanc — welcome hero: rotating particle lungs, bronchial tree, scanning beam, glowing nodule. */
(function () {
  const canvas = document.getElementById("hero-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const rand = (() => { let s = 1234567; return () => (s = (s * 16807) % 2147483647) / 2147483647; })();

  // ---------------- geometry (unit space: x left-right, y up, z towards viewer)
  const pts = [];
  const prof = t => Math.pow(Math.sin(Math.PI / 2 * Math.min(1, (t + 0.02) / 0.85)), 0.55);
  const lungAt = (side, t) => ({ cx: side * (0.30 + 0.17 * prof(t)), rx: (side < 0 ? 0.41 : 0.38) * prof(t), rz: 0.34 * prof(t) });
  const domeY = r2 => -0.72 + 0.3 * (1 - r2);              // diaphragm, higher in the centre
  for (const side of [-1, 1]) {
    let n = 0;
    while (n < 4600) {
      const t = rand(), L = lungAt(side, t), a = rand() * Math.PI * 2;
      const shell = 0.84 + 0.16 * Math.sqrt(rand());
      let dx = Math.cos(a) * L.rx * shell;
      if (dx * side < 0) dx *= 0.62;                        // flatter medial surface
      const z = Math.sin(a) * L.rz * shell, x = L.cx + dx, y = 0.92 - t * 1.62;
      const r2 = (dx / L.rx) ** 2 + (z / L.rz) ** 2;
      if (y < domeY(r2)) continue;
      if (side > 0 && t > 0.42 && z > -0.02 && dx < -L.rx * 0.15) continue;   // cardiac notch (patient left)
      pts.push({ x, y, z, w: 0.5 + rand() * 0.5 });
      n++;
    }
    for (let k = 0; k < 900; k++) {                         // diaphragm surface
      const r = Math.sqrt(rand()) * 0.95, a = rand() * Math.PI * 2, L = lungAt(side, 0.85);
      let dx = Math.cos(a) * L.rx * r; if (dx * side < 0) dx *= 0.62;
      pts.push({ x: L.cx + dx, y: domeY(r * r), z: Math.sin(a) * L.rz * r, w: 0.35 + rand() * 0.3 });
    }
  }
  // bronchial tree
  const segs = [];
  (function branch(p, d, len, depth) {
    const q = [p[0] + d[0] * len, p[1] + d[1] * len, p[2] + d[2] * len];
    segs.push([p, q, depth]);
    if (depth > 6) return;
    for (const k of [-1, 1]) {
      const spread = 0.5 + rand() * 0.35;
      const nd = [d[0] + k * spread * (depth === 0 ? 1.4 : 0.8) + (rand() - 0.5) * 0.4, d[1] - 0.15 + (rand() - 0.5) * 0.5, d[2] + (rand() - 0.5) * 0.9];
      const l = Math.hypot(...nd);
      branch(q, nd.map(v => v / l), len * 0.74, depth + 1);
    }
  })([0, 1.12, 0], [0, -1, 0], 0.4, 0);
  const NODULE = { x: -0.55, y: 0.4, z: 0.1 };
  const dust = Array.from({ length: 140 }, () => ({ x: (rand() - 0.5) * 4, y: (rand() - 0.5) * 3, z: (rand() - 0.5) * 3, v: 0.02 + rand() * 0.05 }));

  let W = 0, H = 0, dpr = 1, mx = 0, my = 0, visible = true;
  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas.clientWidth; H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
  }
  window.addEventListener("resize", resize);
  window.addEventListener("pointermove", e => { mx = e.clientX / innerWidth - 0.5; my = e.clientY / innerHeight - 0.5; });
  new IntersectionObserver(es => visible = es[0].isIntersecting).observe(canvas);
  resize();

  function project(p, rot, tilt, cx, cy, scale) {
    const c = Math.cos(rot), s = Math.sin(rot);
    let x = p.x * c + p.z * s, z = -p.x * s + p.z * c, y = p.y;
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    const y2 = y * ct - z * st; z = y * st + z * ct;
    const f = 3.2 / (3.2 - z);
    return [cx + x * scale * f, cy - y2 * scale * f, z, f];
  }

  function frame(ts) {
    requestAnimationFrame(frame);
    if (!visible || !W) return;
    const t = ts * 0.001;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const wide = W > 900;
    const cx = wide ? W * 0.68 : W * 0.5, cy = wide ? H * 0.5 : H * 0.27, scale = wide ? Math.min(W * 0.27, H * 0.36) : Math.min(W * 0.4, H * 0.2);
    const rot = t * 0.22 + mx * 0.6, tilt = 0.18 + my * 0.25;
    const scanY = Math.sin(t * 0.7) * 0.95 - 0.05;          // beam sweeps apex <-> base

    // halo
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, scale * 1.8);
    g.addColorStop(0, "rgba(59,224,255,0.10)"); g.addColorStop(0.5, "rgba(139,107,255,0.06)"); g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

    ctx.globalCompositeOperation = "lighter";
    // dust
    for (const d of dust) {
      d.y += d.v * 0.016; if (d.y > 1.6) d.y = -1.6;
      const p = project(d, rot * 0.3, tilt, cx, cy, scale);
      ctx.fillStyle = `rgba(120,170,255,${0.10 * p[3]})`;
      ctx.fillRect(p[0], p[1], 1.4, 1.4);
    }
    // bronchial tree
    ctx.lineCap = "round";
    for (const [a, b, depth] of segs) {
      const pa = project({ x: a[0], y: a[1], z: a[2] }, rot, tilt, cx, cy, scale);
      const pb = project({ x: b[0], y: b[1], z: b[2] }, rot, tilt, cx, cy, scale);
      const near = Math.max(0, 1 - Math.abs((a[1] + b[1]) / 2 - scanY) * 4);
      ctx.strokeStyle = `rgba(${150 + 105 * near},${190 + 60 * near},255,${0.20 + 0.25 * near})`;
      ctx.lineWidth = Math.max(0.6, (3.2 - depth * 0.42)) * (pa[3] + pb[3]) / 2;
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    }
    // lung particles
    for (const p of pts) {
      const q = project(p, rot, tilt, cx, cy, scale);
      const depth = (q[2] + 0.6) / 1.2;
      const near = Math.max(0, 1 - Math.abs(p.y - scanY) * 9);
      const r = 40 + 80 * (1 - depth) + 215 * near, gg = 150 + 70 * depth + 105 * near, bb = 255;
      const a = (0.16 + 0.32 * depth) * p.w + near * 0.6;
      ctx.fillStyle = `rgba(${r | 0},${gg | 0},${bb},${a})`;
      const sz = (1.1 + near * 1.3) * q[3];
      ctx.fillRect(q[0], q[1], sz, sz);
    }
    // scan ring
    const ring = [];
    for (let i = 0; i <= 64; i++) { const a = i / 64 * Math.PI * 2; ring.push(project({ x: Math.cos(a) * 1.15, y: scanY, z: Math.sin(a) * 0.62 }, rot, tilt, cx, cy, scale)); }
    ctx.strokeStyle = "rgba(59,224,255,0.55)"; ctx.lineWidth = 1.4;
    ctx.beginPath(); ring.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke();
    ctx.fillStyle = "rgba(59,224,255,0.04)"; ctx.fill();

    // nodule
    const n = project(NODULE, rot, tilt, cx, cy, scale);
    const pulse = 0.5 + 0.5 * Math.sin(t * 3);
    const ng = ctx.createRadialGradient(n[0], n[1], 0, n[0], n[1], 34 * n[3]);
    ng.addColorStop(0, "rgba(255,240,200,0.95)"); ng.addColorStop(0.18, "rgba(255,170,60,0.85)");
    ng.addColorStop(0.45, `rgba(255,77,125,${0.35 + 0.2 * pulse})`); ng.addColorStop(1, "rgba(255,77,125,0)");
    ctx.fillStyle = ng; ctx.beginPath(); ctx.arc(n[0], n[1], 34 * n[3], 0, 7); ctx.fill();
    ctx.globalCompositeOperation = "source-over";
    const ringR = ((t * 0.8) % 1) * 46 + 8;
    ctx.strokeStyle = `rgba(255,181,71,${1 - ringR / 54})`; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(n[0], n[1], ringR, 0, 7); ctx.stroke();

    // HUD callout
    const lx = n[0] + (wide ? -170 : 50), ly = n[1] - 80;
    ctx.fillStyle = "rgba(4,6,12,0.72)"; ctx.fillRect(lx - 6, ly - 4, 132, 46);
    ctx.strokeStyle = "rgba(255,181,71,0.7)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(n[0] - 6, n[1] - 6); ctx.lineTo(lx + 120, ly + 18); ctx.lineTo(lx, ly + 18); ctx.stroke();
    ctx.fillStyle = "rgba(255,181,71,0.95)"; ctx.font = "700 11px JetBrains Mono, monospace";
    ctx.fillText("ATTENTION PEAK", lx, ly + 10);
    ctx.fillStyle = "rgba(230,236,255,0.75)"; ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText(`R upper lobe · ${(0.92 + 0.08 * pulse).toFixed(2)}`, lx, ly + 34);
  }
  requestAnimationFrame(frame);
})();
