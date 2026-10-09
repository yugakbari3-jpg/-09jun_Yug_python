/* Sybil·X — WebGL2 volume ray-marcher.
 * Textures: CT (HU -1000..1500 → 0..255), lung mask, attention (25×16×16), all trilinear.
 * Texture axes: x = image column, y = image row (anterior→posterior), z = slice (superior→inferior).
 * World axes:   x = image column,  y = superior (up),                 z = anterior (towards viewer).
 */
(function () {
  const VS = `#version 300 es
  in vec2 aPos; out vec2 vUv;
  void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }`;

  const FS = `#version 300 es
  precision highp float; precision highp sampler3D;
  in vec2 vUv; out vec4 outColor;
  uniform sampler3D uVol, uMask, uAttn;
  uniform mat4 uInvVP; uniform vec3 uCam;
  uniform vec3 uBox;           // half extents (world)
  uniform vec2 uZRange;        // valid texture-z range
  uniform int uMode; uniform float uAttnOn; uniform float uAttnGain;
  uniform vec3 uCursor;        // texture coords of the crosshair
  uniform float uTime; uniform vec2 uRes; uniform int uSteps;

  vec3 inferno(float t){
    t = clamp(t,0.0,1.0);
    const vec3 c0=vec3(0.0002189403691192265,0.001651004631001012,-0.01948089843709184);
    const vec3 c1=vec3(0.1065134194856116,0.5639564367884091,3.932712388889277);
    const vec3 c2=vec3(11.60249308247187,-3.972853965665698,-15.9423941062914);
    const vec3 c3=vec3(-41.70399613139459,17.43639888205313,44.35414519872813);
    const vec3 c4=vec3(77.162935699427,-33.40235894210092,-81.80730925738993);
    const vec3 c5=vec3(-71.31942824499214,32.62606426397723,73.20951985803202);
    const vec3 c6=vec3(25.13112622477341,-12.24266895238567,-23.07032500287172);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
  }
  float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453); }

  vec3 toTex(vec3 p){
    vec3 u = (p + uBox) / (2.0*uBox);
    return vec3(u.x, 1.0 - u.z, mix(uZRange.y, uZRange.x, u.y));
  }
  float huAt(vec3 tc){ return -1000.0 + texture(uVol, tc).r * 2500.0; }

  // transfer function: returns premultiplied-ready rgb + alpha per unit step
  vec4 transfer(float hu, float m, vec3 tc){
    if (uMode == 0) {                       // bronchovascular
      float inLung = smoothstep(0.35, 0.65, m);
      float vessel = smoothstep(-420.0, -120.0, hu) * inLung;
      float shell = (1.0 - abs(m - 0.5) * 2.0);
      shell = pow(max(shell,0.0), 3.0);
      float bone = smoothstep(220.0, 520.0, hu) * (1.0 - inLung);
      vec3 vcol = mix(vec3(0.95,0.18,0.30), vec3(1.0,0.86,0.86), smoothstep(-150.0, 120.0, hu));
      vec4 c = vec4(0.0);
      c += vec4(vcol, 0.55) * vessel;
      c += vec4(0.25,0.75,1.0, 0.035) * shell;
      c += vec4(0.92,0.90,0.82, 0.022) * bone;
      c.a += 0.0025 * smoothstep(-500.0,-200.0,hu) * (1.0 - inLung);  // ghost body
      c.rgb += vec3(0.3,0.4,0.6) * 0.0025 * smoothstep(-500.0,-200.0,hu) * (1.0 - inLung);
      return c;
    } else if (uMode == 1) {                // skeleton + lungs
      float bone = smoothstep(180.0, 480.0, hu);
      float shell = pow(max(1.0 - abs(m - 0.5) * 2.0, 0.0), 3.0);
      vec4 c = vec4(0.96,0.92,0.84, 0.75) * bone;
      c += vec4(0.3,0.75,1.0, 0.06) * shell;
      c += vec4(0.6,0.5,0.5, 0.004) * smoothstep(-300.0,-50.0,hu) * (1.0 - bone);
      return c;
    } else if (uMode == 2) {                // soft tissue
      float skin = smoothstep(-300.0, -80.0, hu) * (1.0 - smoothstep(-80.0, 0.0, hu));
      float soft = smoothstep(-20.0, 30.0, hu) * (1.0 - smoothstep(150.0, 250.0, hu));
      float bone = smoothstep(200.0, 500.0, hu);
      vec4 c = vec4(1.0,0.78,0.55, 0.035) * skin;
      c += vec4(0.92,0.36,0.32, 0.12) * soft;
      c += vec4(1.0,0.97,0.9, 0.6) * bone;
      return c;
    } else if (uMode == 4) {                // attention focus: dim anatomy
      float vessel = smoothstep(-420.0, -120.0, hu) * smoothstep(0.35, 0.65, m);
      float shell = pow(max(1.0 - abs(m - 0.5) * 2.0, 0.0), 3.0);
      return vec4(0.55,0.65,0.85, 0.25) * vessel + vec4(0.3,0.6,1.0,0.05) * shell + vec4(0.5,0.5,0.6,0.012) * smoothstep(180.0,480.0,hu);
    }
    return vec4(0.0);
  }

  void main(){
    vec4 ndc = vec4(vUv*2.0-1.0, 1.0, 1.0);
    vec4 wp = uInvVP * ndc; wp /= wp.w;
    vec3 ro = uCam, rd = normalize(wp.xyz - uCam);

    // background: deep vignette
    vec2 q = gl_FragCoord.xy / uRes;
    vec3 bg = mix(vec3(0.012,0.02,0.045), vec3(0.03,0.05,0.10), 1.0 - length(q-vec2(0.5,0.6))*1.3);

    vec3 inv = 1.0/rd;
    vec3 t0 = (-uBox - ro)*inv, t1 = (uBox - ro)*inv;
    vec3 tmn = min(t0,t1), tmx = max(t0,t1);
    float tn = max(max(tmn.x,tmn.y),tmn.z), tf = min(min(tmx.x,tmx.y),tmx.z);
    if (tn > tf || tf < 0.0) { outColor = vec4(bg,1.0); return; }
    tn = max(tn, 0.0);

    float dt = 2.0*length(uBox) / float(uSteps);
    float t = tn + hash(gl_FragCoord.xy + fract(uTime)) * dt;
    vec4 acc = vec4(0.0);
    float mip = 0.0; float mipA = 0.0;
    vec3 L1 = normalize(vec3(0.5,0.8,0.6));
    vec3 g = vec3(1.0/256.0, 1.0/256.0, 1.0/200.0);

    for (int i = 0; i < 1024; i++) {
      if (i >= uSteps || t > tf || acc.a > 0.985) break;
      vec3 p = ro + rd*t;
      vec3 tc = toTex(p);
      float hu = huAt(tc);
      if (uMode == 3) {
        float v = clamp((hu + 1000.0)/1800.0, 0.0, 1.0);
        mip = max(mip, v);
      } else {
        float m = texture(uMask, tc).r;
        vec4 c = transfer(hu, m, tc);
        if (c.a > 0.004) {
          vec3 n = vec3(
            huAt(tc + vec3(g.x,0,0)) - huAt(tc - vec3(g.x,0,0)),
            huAt(tc - vec3(0,g.y,0)) - huAt(tc + vec3(0,g.y,0)),
            huAt(tc - vec3(0,0,g.z)) - huAt(tc + vec3(0,0,g.z)));
          // texture->world: x same, world y = -tex z, world z = -tex y
          n = vec3(n.x, n.z, n.y);
          float glen = length(n);
          if (glen > 1.0) {
            n = -n / glen;
            vec3 V = -rd;
            float diff = abs(dot(n, L1))*0.65 + abs(dot(n, V))*0.35;
            float spec = pow(max(dot(reflect(-L1, n), V), 0.0), 28.0);
            c.rgb = c.rgb*(0.28 + 0.9*diff) + vec3(spec)*0.45*c.a*4.0;
          }
          float a = 1.0 - pow(1.0 - clamp(c.a,0.0,0.99), dt*180.0);
          acc.rgb += (1.0 - acc.a) * a * c.rgb;
          acc.a   += (1.0 - acc.a) * a;
        }
      }
      if (uAttnOn > 0.5) {
        float at = texture(uAttn, tc).r;
        float inside = step(uZRange.x, tc.z) * step(tc.z, uZRange.y);
        float e = pow(at, 1.6) * uAttnGain * inside;
        if (e > 0.002) {
          float a = 1.0 - pow(1.0 - clamp(e*0.18,0.0,0.95), dt*180.0);
          vec3 col = inferno(0.35 + 0.65*at) * 1.6;
          acc.rgb += (1.0 - acc.a) * a * col;
          acc.a   += (1.0 - acc.a) * a * 0.55;
        }
      }
      // crosshair plane (axial)
      if (abs(tc.z - uCursor.z) < 0.0028) {
        float a = 0.022;
        acc.rgb += (1.0 - acc.a) * a * vec3(0.23,0.88,1.0);
        acc.a += (1.0 - acc.a) * a * 0.5;
      }
      t += dt;
    }
    if (uMode == 3) {
      float v = pow(mip, 1.6);
      acc.rgb = vec3(v)*vec3(0.85,0.93,1.0) + acc.rgb;
      acc.a = max(acc.a, v);
    }
    // box wireframe glow
    vec3 pe = ro + rd*tn;
    vec3 d = abs(abs(pe) - uBox);
    float edge = 0.0;
    edge += (1.0-smoothstep(0.0,0.006,d.x+d.y)) + (1.0-smoothstep(0.0,0.006,d.y+d.z)) + (1.0-smoothstep(0.0,0.006,d.x+d.z));
    vec3 col = acc.rgb + bg*(1.0-acc.a) + vec3(0.23,0.88,1.0)*edge*0.35;
    col = col / (1.0 + col*0.25);              // soft tonemap
    col = pow(col, vec3(0.95));
    outColor = vec4(col, 1.0);
  }`;

  function mat4mul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s;
    }
    return o;
  }
  function perspective(fovy, asp, n, f) {
    const t = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
    o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o;
  }
  function lookAt(e, c, up) {
    const z = norm(sub(e, c)), x = norm(cross(up, z)), y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, e), -dot(y, e), -dot(z, e), 1]);
  }
  function invert(m) {
    const inv = new Float32Array(16);
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    det = 1 / det; for (let i = 0; i < 16; i++) inv[i] *= det; return inv;
  }
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

  class VolumeRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true, powerPreference: "high-performance" });
      this.gl = gl;
      if (!gl) { this.failed = true; return; }
      this.prog = this._program(VS, FS);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);
      const loc = gl.getAttribLocation(this.prog, "aPos");
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      this.u = {};
      for (const n of ["uVol", "uMask", "uAttn", "uInvVP", "uCam", "uBox", "uZRange", "uMode", "uAttnOn", "uAttnGain", "uCursor", "uTime", "uRes", "uSteps"])
        this.u[n] = gl.getUniformLocation(this.prog, n);
      this.tex = { vol: null, mask: null, attn: this._tex3d(new Uint8Array(1), 1, 1, 1) };
      this.yaw = 0.5; this.pitch = 0.18; this.dist = 2.5;
      this.mode = 0; this.attnOn = true; this.attnGain = 1.0; this.autoRotate = true;
      this.cursor = [0.5, 0.5, 0.5]; this.zRange = [0, 1];
      this.box = [0.45, 0.6, 0.45];
      this.interacting = false; this.dirty = true; this.ready = false;
      this._bind();
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
    }
    _program(vs, fs) {
      const gl = this.gl;
      const mk = (t, s) => { const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh); if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh)); return sh; };
      const p = gl.createProgram();
      gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      return p;
    }
    _tex3d(data, w, h, d, old) {
      const gl = this.gl;
      if (old) gl.deleteTexture(old);
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_3D, t);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, w, h, d, 0, gl.RED, gl.UNSIGNED_BYTE, data);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_3D, p, gl.LINEAR);
      for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
      return t;
    }
    setVolume(wide, mask, dims, valid) {
      if (this.failed) return;
      const [D, H, W] = dims;
      this.tex.vol = this._tex3d(wide, W, H, D, this.tex.vol);
      this.tex.mask = this._tex3d(mask, W, H, D, this.tex.mask);
      this.zRange = [(valid[0] + 0.5) / D, (valid[1] + 0.5) / D];
      // physical: in-plane 256 × 1.40625 mm = 360 mm, slices 2.5 mm
      const zmm = (valid[1] - valid[0] + 1) * 2.5, xmm = 360, s = 1 / 600;
      this.box = [xmm * s, zmm * s, xmm * s];
      this.ready = true; this.dirty = true;
    }
    setAttention(data, dims) {
      if (this.failed) return;
      const [D, H, W] = dims;
      this.tex.attn = this._tex3d(data, W, H, D, this.tex.attn);
      this.dirty = true;
    }
    clearAttention() { this.tex.attn = this._tex3d(new Uint8Array(1), 1, 1, 1, this.tex.attn); this.dirty = true; }
    setCursor(z, y, x) { this.cursor = [(x + 0.5) / 256, (y + 0.5) / 256, (z + 0.5) / 200]; this.dirty = true; }
    _bind() {
      const c = this.canvas; let last = null;
      c.addEventListener("pointerdown", (e) => { last = [e.clientX, e.clientY]; this.interacting = true; this.autoRotateWas = this.autoRotate; c.setPointerCapture(e.pointerId); });
      c.addEventListener("pointermove", (e) => {
        if (!last) return;
        this.yaw -= (e.clientX - last[0]) * 0.008;
        this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + (e.clientY - last[1]) * 0.006));
        last = [e.clientX, e.clientY]; this.dirty = true;
      });
      const up = () => { last = null; this.interacting = false; this.dirty = true; };
      c.addEventListener("pointerup", up); c.addEventListener("pointercancel", up);
      c.addEventListener("wheel", (e) => { e.preventDefault(); this.dist = Math.max(1.0, Math.min(6, this.dist * (1 + e.deltaY * 0.001))); this.dirty = true; }, { passive: false });
    }
    _loop(ts) {
      requestAnimationFrame(this._loop);
      if (!this.ready || this.failed) return;
      if (this.autoRotate && !this.interacting) { this.yaw += 0.0035; this.dirty = true; }
      if (!this.dirty) return;
      this.dirty = false;
      this.render(ts);
    }
    render(ts = 0) {
      const gl = this.gl, c = this.canvas;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const scale = (this.interacting || this.autoRotate) ? 0.75 : 1.0;
      const w = Math.max(2, Math.round(c.clientWidth * dpr * scale)), h = Math.max(2, Math.round(c.clientHeight * dpr * scale));
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
      gl.viewport(0, 0, w, h);
      const eye = [this.dist * Math.sin(this.yaw) * Math.cos(this.pitch), this.dist * Math.sin(this.pitch), this.dist * Math.cos(this.yaw) * Math.cos(this.pitch)];
      const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);
      const proj = perspective(0.62, w / h, 0.05, 20);
      const inv = invert(mat4mul(proj, view));
      gl.useProgram(this.prog);
      gl.bindVertexArray(this.vao);
      gl.uniformMatrix4fv(this.u.uInvVP, false, inv);
      gl.uniform3fv(this.u.uCam, eye);
      gl.uniform3fv(this.u.uBox, this.box);
      gl.uniform2fv(this.u.uZRange, this.zRange);
      gl.uniform1i(this.u.uMode, this.mode);
      gl.uniform1f(this.u.uAttnOn, this.attnOn ? 1 : 0);
      gl.uniform1f(this.u.uAttnGain, this.attnGain);
      gl.uniform3fv(this.u.uCursor, this.cursor);
      gl.uniform1f(this.u.uTime, (ts || 0) * 0.001);
      gl.uniform2f(this.u.uRes, w, h);
      gl.uniform1i(this.u.uSteps, this.interacting ? 220 : 420);
      [["uVol", this.tex.vol], ["uMask", this.tex.mask], ["uAttn", this.tex.attn]].forEach(([n, t], i) => {
        gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_3D, t); gl.uniform1i(this.u[n], i);
      });
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }
  window.VolumeRenderer = VolumeRenderer;
})();
