// Lab framework: the shared building blocks every algorithm's interactive lab
// is assembled from. An algorithm describes *what* to show (frames of its
// training run, an architecture graph, the charts it tracks); this file owns
// *how*: the player, the scrubber, narration with typeset maths, the dataset
// loader, live charts, raster decision fields and the architecture diagram.
const Lab = (() => {
  const DOMAIN = [-10, 10];
  // The neural labs use tiny networks; the CPU backend avoids GPU read-back
  // stalls on every snapshot and behaves identically on every device.
  try { if (window.tf && tf.setBackend) tf.setBackend("cpu"); } catch (e) {}
  let uidN = 0;
  const uid = (p = "l") => `${p}${++uidN}`;
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const fmt = (v, d = 3) => (v == null || !isFinite(v) ? "–" : Math.abs(v) >= 1e4 ? v.toExponential(2) : (+v).toFixed(d));

  // ---------------------------------------------------------------- colour
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  const col = (i) => `var(--series-${(((i | 0) % 7) + 7) % 7 + 1})`;
  let palCache = null;
  function pal() {
    if (!palCache) palCache = [1, 2, 3, 4, 5, 6, 7].map((i) => cssVar(`--series-${i}`) || MLU.palette[i - 1]);
    return palCache;
  }
  function rgb(c) { const k = d3.color(c); return k ? [k.r, k.g, k.b] : [128, 128, 128]; }
  function palRGB() { return pal().map(rgb); }
  window.addEventListener("ml-theme", () => { palCache = null; });

  // ---------------------------------------------------------------- maths typesetting
  // KaTeX renders HTML when its stylesheet is present, and falls back to native
  // MathML otherwise, so formulas stay typeset even where the CSS cannot load.
  let texHTMLMode = null;
  function texMode() {
    if (texHTMLMode !== null) return texHTMLMode;
    const probe = document.createElement("span");
    probe.className = "katex";
    probe.style.position = "absolute"; probe.style.visibility = "hidden";
    document.body.appendChild(probe);
    texHTMLMode = /KaTeX/i.test(getComputedStyle(probe).fontFamily);
    probe.remove();
    return texHTMLMode;
  }
  function texHTML(src, display = true) {
    if (!src) return "";
    if (!window.katex) return `<code class="tex-fallback">${esc(src)}</code>`;
    try {
      return katex.renderToString(src, { throwOnError: false, displayMode: display, output: texMode() ? "htmlAndMathml" : "mathml", strict: "ignore" });
    } catch (e) { return `<code class="tex-fallback">${esc(src)}</code>`; }
  }
  // Replaces $...$ (inline) and $$...$$ (display) inside a string of HTML.
  function richText(s) {
    if (!s) return "";
    return String(s)
      .replace(/\$\$([^$]+)\$\$/g, (_, t) => texHTML(t, true))
      .replace(/\$([^$]+)\$/g, (_, t) => texHTML(t, false));
  }

  // ---------------------------------------------------------------- icons
  const ICON = {
    first: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 5v14M19 5l-9 7 9 7z"/></svg>`,
    prev: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5l-7 7 7 7"/></svg>`,
    play: `<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M7 4.5v15l12-7.5z"/></svg>`,
    pause: `<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><rect x="6" y="4.5" width="4" height="15" rx="1"/><rect x="14" y="4.5" width="4" height="15" rx="1"/></svg>`,
    next: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5l7 7-7 7"/></svg>`,
    last: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 5v14M5 5l9 7-9 7z"/></svg>`,
    loop: `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 2l3 3-3 3"/><path d="M4 11V9a4 4 0 0 1 4-4h12"/><path d="M7 22l-3-3 3-3"/><path d="M20 13v2a4 4 0 0 1-4 4H4"/></svg>`,
  };

  // ---------------------------------------------------------------- datasets
  // Every generator draws from MLU's seeded RNG, so "reseed" is reproducible.
  const rr = (a, b) => MLU.randRange(a, b);
  const rn = () => MLU.randn();
  const clip = (v) => clamp(v, -9.8, 9.8);
  function blobsAt(centers, n, spread) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const c = i % centers.length;
      const s = Array.isArray(spread) ? spread[c] : spread;
      pts.push({ x: clip(centers[c][0] + rn() * s), y: clip(centers[c][1] + rn() * s), label: c });
    }
    return pts;
  }
  function ringCenters(k, r = 5.6) {
    const off = rr(0, Math.PI * 2);
    return Array.from({ length: k }, (_, i) => [r * Math.cos(off + (i * 2 * Math.PI) / k), r * Math.sin(off + (i * 2 * Math.PI) / k)]);
  }
  const DATA = {
    // ---- classification / clustering (labelled 2D points)
    blobs2: { name: "2 gaussian blobs", kinds: ["class", "cluster"], gen: ({ n, noise }) => blobsAt(ringCenters(2, 4.5), n, 0.7 + noise * 2.6) },
    blobs3: { name: "3 gaussian blobs", kinds: ["class", "cluster"], gen: ({ n, noise }) => blobsAt(ringCenters(3), n, 0.6 + noise * 2.4) },
    blobs4: { name: "4 gaussian blobs", kinds: ["class", "cluster"], gen: ({ n, noise }) => blobsAt(ringCenters(4, 6), n, 0.6 + noise * 2.2) },
    blobs5: { name: "5 gaussian blobs", kinds: ["cluster"], gen: ({ n, noise }) => blobsAt(ringCenters(5, 6.4), n, 0.5 + noise * 1.8) },
    linear: { name: "linearly separable", kinds: ["class"], gen: ({ n, noise }) => blobsAt([[-3.5, -2.5], [3.5, 2.5]], n, 1 + noise * 2.6) },
    overlap: { name: "overlapping classes", kinds: ["class"], gen: ({ n, noise }) => blobsAt([[-1.6, -0.8], [1.6, 0.8]], n, 2 + noise * 1.6) },
    moons: {
      name: "two moons", kinds: ["class", "cluster"], gen: ({ n, noise }) => Array.from({ length: n }, (_, i) => {
        const l = i % 2, t = rr(0, Math.PI), s = 0.06 + noise * 0.32;
        const u = l === 0 ? Math.cos(t) : 1 - Math.cos(t), v = l === 0 ? Math.sin(t) : 0.5 - Math.sin(t);
        return { x: clip((u - 0.5 + rn() * s) * 6.4), y: clip((v - 0.25 + rn() * s) * 6.4), label: l };
      }),
    },
    circles: {
      name: "concentric circles", kinds: ["class", "cluster"], gen: ({ n, noise }) => Array.from({ length: n }, (_, i) => {
        const l = i % 2, a = rr(0, Math.PI * 2), r = (l === 0 ? 3 : 7) + rn() * (0.25 + noise * 1.4);
        return { x: clip(r * Math.cos(a)), y: clip(r * Math.sin(a)), label: l };
      }),
    },
    xor: {
      name: "XOR quadrants", kinds: ["class"], gen: ({ n, noise }) => Array.from({ length: n }, () => {
        let x = rr(-9, 9), y = rr(-9, 9);
        if (Math.abs(x) < 0.8) x += Math.sign(x || 1) * 0.8;
        if (Math.abs(y) < 0.8) y += Math.sign(y || 1) * 0.8;
        const label = (x > 0) === (y > 0) ? 0 : 1;
        return { x: clip(x + rn() * noise * 2), y: clip(y + rn() * noise * 2), label };
      }),
    },
    spiral: {
      name: "two spirals", kinds: ["class", "cluster"], gen: ({ n, noise }) => Array.from({ length: n }, (_, i) => {
        const l = i % 2, t = rr(0.08, 1), a = t * 3.4 * Math.PI + l * Math.PI, r = t * 8.6;
        return { x: clip(r * Math.cos(a) + rn() * noise * 1.1), y: clip(r * Math.sin(a) + rn() * noise * 1.1), label: l };
      }),
    },
    imbalanced: {
      name: "imbalanced (85 / 15)", kinds: ["class"], gen: ({ n, noise }) => {
        const n1 = Math.max(3, Math.round(n * 0.15));
        const a = blobsAt([[-1.5, -1]], n - n1, 2.4 + noise * 1.2).map((p) => ({ ...p, label: 0 }));
        const b = blobsAt([[3.5, 3]], n1, 1 + noise * 1.2).map((p) => ({ ...p, label: 1 }));
        return a.concat(b);
      },
    },
    aniso: {
      name: "anisotropic blobs", kinds: ["cluster", "class"], gen: ({ n, noise }) => blobsAt(ringCenters(3, 4.2), n, 0.7 + noise * 1.6)
        .map((p) => ({ x: clip(0.95 * p.x - 0.95 * p.y * 0.62), y: clip(-0.5 * p.x + 1.1 * p.y * 0.62 + p.x * 0.45), label: p.label })),
    },
    varied: { name: "varied density", kinds: ["cluster"], gen: ({ n, noise }) => blobsAt([[-5, 4], [4, 4], [0, -4.5]], n, [0.5 + noise, 1.9 + noise * 1.4, 1 + noise]) },
    outliers: {
      name: "blobs + outliers", kinds: ["cluster", "class"], gen: ({ n, noise }) => {
        const no = Math.round(n * 0.1);
        const b = blobsAt(ringCenters(3, 5), n - no, 0.7 + noise * 1.4);
        for (let i = 0; i < no; i++) b.push({ x: rr(-9.5, 9.5), y: rr(-9.5, 9.5), label: MLU.randInt(3), outlier: true });
        return b;
      },
    },
    uniform: { name: "uniform noise (no structure)", kinds: ["cluster", "class"], gen: ({ n }) => Array.from({ length: n }, () => ({ x: rr(-9, 9), y: rr(-9, 9), label: MLU.randInt(2) })) },
    // ---- regression (y as a function of x)
    rlinear: { name: "linear + noise", kinds: ["reg"], gen: ({ n, noise }) => Array.from({ length: n }, () => { const x = rr(-9, 9); return { x, y: clip(0.75 * x - 1 + rn() * (0.4 + noise * 3.2)) }; }) },
    quad: { name: "quadratic", kinds: ["reg"], gen: ({ n, noise }) => Array.from({ length: n }, () => { const x = rr(-9, 9); return { x, y: clip(0.13 * x * x - 4.5 + rn() * (0.3 + noise * 2.4)) }; }) },
    sine: { name: "sine wave", kinds: ["reg"], gen: ({ n, noise }) => Array.from({ length: n }, () => { const x = rr(-9, 9); return { x, y: clip(5 * Math.sin(0.62 * x) + rn() * (0.3 + noise * 2.4)) }; }) },
    cubic: { name: "cubic", kinds: ["reg"], gen: ({ n, noise }) => Array.from({ length: n }, () => { const x = rr(-9, 9); return { x, y: clip(0.011 * x ** 3 - 0.35 * x + rn() * (0.3 + noise * 2.4)) }; }) },
    step: { name: "step function", kinds: ["reg"], gen: ({ n, noise }) => Array.from({ length: n }, () => { const x = rr(-9, 9); return { x, y: clip((x < -3 ? -4.5 : x < 3 ? 3 : -1) + rn() * (0.3 + noise * 2)) }; }) },
    routliers: {
      name: "linear + outliers", kinds: ["reg"], gen: ({ n, noise }) => Array.from({ length: n }, (_, i) => {
        const x = rr(-9, 9); const out = i % 11 === 5;
        return { x, y: clip(0.6 * x + rn() * (0.4 + noise * 2) + (out ? 7.5 : 0)), outlier: out };
      }),
    },
    hetero: { name: "heteroscedastic", kinds: ["reg"], gen: ({ n, noise }) => Array.from({ length: n }, () => { const x = rr(-9, 9); return { x, y: clip(0.5 * x + rn() * (0.2 + (Math.abs(x) / 9) * (1.5 + noise * 3))) }; }) },
  };
  function generate(key, { n = 80, noise = 0.3, seed } = {}) {
    if (seed != null) MLU.seedRng(seed);
    const d = DATA[key] || DATA.blobs3;
    return MLU.shuffle(d.gen({ n, noise }));
  }

  // Parses pasted or uploaded CSV text into points rescaled to the canvas.
  function parseCSV(text, kind) {
    const rows = text.trim().split(/\r?\n/).map((l) => l.split(/[,;\t]/).map((s) => s.trim())).filter((r) => r.length >= 2);
    if (rows.length && rows[0].some((c) => c !== "" && isNaN(+c) && rows[0].length <= 3 && isNaN(+rows[0][0]))) rows.shift();
    const raw = rows.map((r) => ({ x: +r[0], y: +r[1], l: r[2] })).filter((p) => isFinite(p.x) && isFinite(p.y));
    if (raw.length < 2) return null;
    const ext = (k) => { const v = raw.map((p) => p[k]); return [Math.min(...v), Math.max(...v)]; };
    const [ax, bx] = ext("x"), [ay, by] = ext("y");
    const sc = (v, a, b) => (b - a < 1e-12 ? 0 : -9 + ((v - a) / (b - a)) * 18);
    const labels = [...new Set(raw.map((p) => p.l).filter((l) => l != null && l !== ""))];
    return raw.map((p) => ({ x: sc(p.x, ax, bx), y: sc(p.y, ay, by), label: labels.length ? Math.max(0, labels.indexOf(p.l)) % 7 : 0 }));
  }
  function downloadCSV(points, name = "dataset.csv") {
    const hasLabel = points.some((p) => p.label != null);
    const text = (hasLabel ? "x,y,label\n" : "x,y\n") + points.map((p) => [p.x.toFixed(4), p.y.toFixed(4)].concat(hasLabel ? [p.label ?? 0] : []).join(",")).join("\n");
    try {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
      a.download = name; document.body.appendChild(a); a.click(); a.remove();
    } catch (e) { /* downloads may be blocked inside sandboxed frames */ }
  }

  // ---------------------------------------------------------------- 2D plane
  function plane(svgEl, { w = 640, h = 440, pad = 34, xd = DOMAIN, yd = DOMAIN, ticks = true, xLabel, yLabel } = {}) {
    const svg = d3.select(svgEl).attr("viewBox", `0 0 ${w} ${h}`).attr("preserveAspectRatio", "xMidYMid meet");
    svg.selectAll("*").remove();
    const x = d3.scaleLinear().domain(xd).range([pad, w - pad]);
    const y = d3.scaleLinear().domain(yd).range([h - pad, pad]);
    const layers = {};
    // Clip drawings to just inside the frame: contour polygons otherwise trace
    // the plot border, and curves would spill past the axes.
    const clipId = uid("clip");
    svg.append("defs").append("clipPath").attr("id", clipId).append("rect")
      .attr("x", pad + 1.5).attr("y", pad + 1.5).attr("width", w - 2 * pad - 3).attr("height", h - 2 * pad - 3);
    layers.bg = svg.append("g").attr("class", "ly-bg");
    const axes = svg.append("g").attr("class", "lab-axes");
    for (const k of ["under", "mid", "links", "pts", "top", "ui"]) layers[k] = svg.append("g").attr("class", "ly-" + k);
    for (const k of ["under", "mid", "links"]) layers[k].attr("clip-path", `url(#${clipId})`);
    function drawAxes() {
      axes.selectAll("*").remove();
      axes.append("rect").attr("x", pad).attr("y", pad).attr("width", w - 2 * pad).attr("height", h - 2 * pad).attr("class", "ax-frame");
      if (!ticks) return;
      for (const t of x.ticks(8)) {
        axes.append("line").attr("class", t === 0 ? "ax-zero" : "ax-grid").attr("x1", x(t)).attr("x2", x(t)).attr("y1", pad).attr("y2", h - pad);
        axes.append("text").attr("class", "ax-tick").attr("x", x(t)).attr("y", h - pad + 13).attr("text-anchor", "middle").text(+t.toFixed(2));
      }
      for (const t of y.ticks(6)) {
        axes.append("line").attr("class", t === 0 ? "ax-zero" : "ax-grid").attr("y1", y(t)).attr("y2", y(t)).attr("x1", pad).attr("x2", w - pad);
        axes.append("text").attr("class", "ax-tick").attr("x", pad - 5).attr("y", y(t) + 3).attr("text-anchor", "end").text(+t.toFixed(2));
      }
      if (xLabel) axes.append("text").attr("class", "ax-label").attr("x", w - pad).attr("y", h - pad - 5).attr("text-anchor", "end").text(xLabel);
      if (yLabel) axes.append("text").attr("class", "ax-label").attr("x", pad + 5).attr("y", pad + 12).text(yLabel);
    }
    drawAxes();
    return {
      svg, x, y, w, h, pad, layers,
      setDomain(nxd, nyd) { x.domain(nxd); y.domain(nyd); drawAxes(); },
      inside(px, py) { return px >= pad && px <= w - pad && py >= pad && py <= h - pad; },
    };
  }

  // Samples fn over the visible plane into a grid (row 0 = top).
  function sampleGrid(pl, fn, res = 80) {
    const [x0, x1] = pl.x.domain(), [y0, y1] = pl.y.domain();
    const cols = res, rows = Math.max(2, Math.round((res * (pl.h - 2 * pl.pad)) / (pl.w - 2 * pl.pad)));
    const values = new Array(cols * rows);
    for (let j = 0; j < rows; j++) {
      const yv = y1 - ((j + 0.5) / rows) * (y1 - y0);
      for (let i = 0; i < cols; i++) values[j * cols + i] = fn(x0 + ((i + 0.5) / cols) * (x1 - x0), yv);
    }
    return { cols, rows, values };
  }
  // Paints a grid of values through colorFn(v) -> [r,g,b,alpha 0..1] as a
  // single smoothly-upscaled image: far cheaper than thousands of rects.
  function paintGrid(g, pl, grid, colorFn, { opacity = 0.9, smooth = true, cls = "raster" } = {}) {
    const cv = document.createElement("canvas");
    cv.width = grid.cols; cv.height = grid.rows;
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(grid.cols, grid.rows);
    for (let k = 0; k < grid.values.length; k++) {
      const c = colorFn(grid.values[k]);
      img.data[k * 4] = c[0]; img.data[k * 4 + 1] = c[1]; img.data[k * 4 + 2] = c[2];
      img.data[k * 4 + 3] = Math.round((c[3] == null ? 1 : c[3]) * 255);
    }
    ctx.putImageData(img, 0, 0);
    let im = g.select(`image.${cls}`);
    if (im.empty()) im = g.append("image").attr("class", cls).attr("preserveAspectRatio", "none");
    im.attr("x", pl.pad).attr("y", pl.pad).attr("width", pl.w - 2 * pl.pad).attr("height", pl.h - 2 * pl.pad)
      .attr("href", cv.toDataURL()).style("opacity", opacity).style("image-rendering", smooth ? "auto" : "pixelated");
    return im;
  }
  function field(g, pl, fn, colorFn, opts = {}) { return paintGrid(g, pl, sampleGrid(pl, fn, opts.res || 80), colorFn, opts); }

  // Ready-made colour maps for decision fields.
  function probColor(p, a = 0.5) { // binary probability of class 1
    const P = palRGB(), c0 = P[0], c1 = P[1];
    const t = clamp(p, 0, 1);
    return [c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t, a * (0.25 + Math.abs(t - 0.5) * 1.5)];
  }
  function classColor(k, conf = 1, a = 0.42) {
    if (k == null || k < 0) return [0, 0, 0, 0];
    const c = palRGB()[k % 7];
    return [c[0], c[1], c[2], a * (0.35 + 0.65 * clamp(conf, 0, 1))];
  }
  function divergeColor(v, scale = 1, a = 0.55) { // signed values: class-0 hue for negative, class-1 for positive
    const P = palRGB(); const t = clamp(v / scale, -1, 1); const c = t < 0 ? P[0] : P[1];
    return [c[0], c[1], c[2], a * Math.abs(t)];
  }

  function contour(g, pl, grid, thresholds, { stroke = "var(--text)", width = 1.6, dash = null, cls = "iso" } = {}) {
    const paths = d3.contours().size([grid.cols, grid.rows]).thresholds(thresholds)(grid.values.map((v) => (isFinite(v) ? v : 0)));
    const sx = (pl.w - 2 * pl.pad) / grid.cols, sy = (pl.h - 2 * pl.pad) / grid.rows;
    const geo = d3.geoPath(d3.geoTransform({ point(px, py) { this.stream.point(pl.pad + px * sx, pl.pad + py * sy); } }));
    const sel = g.selectAll(`path.${cls}`).data(paths);
    sel.exit().remove();
    return sel.enter().append("path").attr("class", cls).merge(sel)
      .attr("d", geo).attr("fill", "none").attr("stroke", stroke).attr("stroke-width", width).attr("stroke-dasharray", dash);
  }

  function ellipsePath(pl, mu, cov, nsig = 2, steps = 64) {
    const e = MLU.jacobiEigen(cov);
    const a = Math.sqrt(Math.max(e.values[0], 1e-9)) * nsig, b = Math.sqrt(Math.max(e.values[1], 1e-9)) * nsig;
    const v = e.vectors[0];
    let d = "";
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * Math.PI * 2;
      const ux = a * Math.cos(t), uy = b * Math.sin(t);
      const px = mu[0] + ux * v[0] - uy * v[1], py = mu[1] + ux * v[1] + uy * v[0];
      d += (i ? "L" : "M") + pl.x(px).toFixed(1) + " " + pl.y(py).toFixed(1);
    }
    return d + "Z";
  }

  // ---------------------------------------------------------------- generic xy plot
  function xy(el, { w = 360, h = 130, m = { l: 38, r: 10, t: 10, b: 20 }, xd = [0, 1], yd = [0, 1], xLabel = "", yLabel = "", xTicks = 5, yTicks = 4, logY = false, band = false } = {}) {
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("class", "xy-svg");
    const x = band ? d3.scaleBand().domain(xd).range([m.l, w - m.r]).padding(0.18) : d3.scaleLinear().domain(xd).range([m.l, w - m.r]);
    const y = (logY ? d3.scaleLog() : d3.scaleLinear()).domain(yd).range([h - m.b, m.t]);
    const ax = svg.append("g").attr("class", "lab-axes");
    const g = svg.append("g");
    // Clip data drawings to the axes frame so lines and contours can never
    // spill over the border, labels or neighbouring cards.
    const xclip = uid("xclip");
    svg.append("defs").append("clipPath").attr("id", xclip).append("rect")
      .attr("x", m.l).attr("y", m.t).attr("width", Math.max(1, w - m.l - m.r)).attr("height", Math.max(1, h - m.t - m.b));
    g.attr("clip-path", `url(#${xclip})`);
    const top = svg.append("g");
    function axes() {
      ax.selectAll("*").remove();
      const yt = logY ? y.ticks(3) : y.ticks(yTicks);
      for (const t of yt) {
        ax.append("line").attr("class", "ax-grid").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(t)).attr("y2", y(t));
        ax.append("text").attr("class", "ax-tick").attr("x", m.l - 4).attr("y", y(t) + 3).attr("text-anchor", "end").text(d3.format(logY ? ".0e" : "~s")(t) === "0" ? "0" : shortNum(t));
      }
      if (!band) for (const t of x.ticks(xTicks)) ax.append("text").attr("class", "ax-tick").attr("x", x(t)).attr("y", h - m.b + 12).attr("text-anchor", "middle").text(shortNum(t));
      else for (const t of x.domain()) ax.append("text").attr("class", "ax-tick").attr("x", x(t) + x.bandwidth() / 2).attr("y", h - m.b + 12).attr("text-anchor", "middle").text(t);
      ax.append("line").attr("class", "ax-base").attr("x1", m.l).attr("x2", w - m.r).attr("y1", h - m.b).attr("y2", h - m.b);
      if (xLabel) ax.append("text").attr("class", "ax-label").attr("x", w - m.r).attr("y", h - m.b - 4).attr("text-anchor", "end").text(xLabel);
      if (yLabel) ax.append("text").attr("class", "ax-label").attr("x", m.l + 4).attr("y", m.t + 9).text(yLabel);
    }
    axes();
    const api = {
      svg, x, y, g, top, w, h, m,
      domain(nx, ny) { if (nx) x.domain(nx); if (ny) y.domain(ny); axes(); return api; },
      line(key, pts, { color = col(0), width = 2, dash = null, curve = d3.curveMonotoneX, area = false } = {}) {
        let p = g.select(`path.ln-${key}`);
        if (p.empty()) p = g.append("path").attr("class", `ln ln-${key}`);
        const good = pts.filter((d) => d && d[0] != null && d[1] != null && isFinite(d[0]) && isFinite(d[1]) && (!logY || d[1] > 0));
        if (area) {
          p.attr("d", d3.area().curve(curve).x((d) => x(d[0])).y0(y(y.domain()[0])).y1((d) => y(d[1]))(good) || "")
            .attr("fill", color).attr("fill-opacity", 0.16).attr("stroke", "none");
        } else {
          p.attr("d", d3.line().curve(curve).x((d) => x(d[0])).y((d) => y(d[1]))(good) || "")
            .attr("fill", "none").attr("stroke", color).attr("stroke-width", width).attr("stroke-dasharray", dash);
        }
        return p;
      },
      clear() { g.selectAll("*").remove(); top.selectAll("*").remove(); },
      vline(xv, cls = "xy-cursor") {
        let l = top.select(`line.${cls}`);
        if (l.empty()) l = top.append("line").attr("class", cls);
        if (xv == null || !isFinite(xv)) { l.style("display", "none"); return; }
        l.style("display", null).attr("x1", x(xv)).attr("x2", x(xv)).attr("y1", m.t).attr("y2", h - m.b);
      },
    };
    return api;
  }
  function shortNum(t) {
    const a = Math.abs(t);
    if (a === 0) return "0";
    if (a >= 1000) return d3.format(".2s")(t);
    if (a >= 10) return d3.format(".0f")(t);
    if (a >= 1) return d3.format(".1~f")(t);
    if (a >= 0.01) return d3.format(".2~f")(t);
    return d3.format(".0e")(t);
  }

  // A metric chart fed by frames: rows are per-frame objects of numbers.
  function chart(el, { series, h = 120, w = 360, logY = false, yd = null, xLabel = "step" }) {
    const legend = series.map((s) => `<span class="legend-item"><span class="swatch" style="background:${s.color}"></span>${s.label}</span>`).join("");
    const head = el.closest(".lab-card") && el.closest(".lab-card").querySelector(".chart-legend");
    if (head) head.innerHTML = legend;
    const P = xy(el, { w, h, logY, xLabel });
    const vals = {};
    let rows = [];
    const api = {
      set(nrows) {
        rows = nrows || [];
        const n = Math.max(2, rows.length);
        let lo = Infinity, hi = -Infinity;
        for (const s of series) for (const r of rows) {
          const v = r && r[s.key];
          if (v == null || !isFinite(v) || (logY && v <= 0)) continue;
          lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
        if (!isFinite(lo)) { lo = logY ? 0.01 : 0; hi = 1; }
        if (yd) { lo = yd[0] ?? lo; hi = yd[1] ?? hi; }
        if (hi - lo < 1e-9) { hi = lo + (logY ? lo : 1); }
        if (!logY && !yd) { const padv = (hi - lo) * 0.08; lo = lo >= 0 && lo - padv < 0 ? 0 : lo - padv; hi += padv; }
        P.domain([0, n - 1], [logY ? Math.max(lo, 1e-6) : lo, hi]);
        for (const s of series) P.line(s.key, rows.map((r, i) => [i, r ? r[s.key] : null]), { color: s.color, dash: s.dash, curve: d3.curveLinear });
      },
      cursor(i) {
        P.vline(rows.length ? i : null);
        const dots = P.top.selectAll("circle.cdot").data(series.filter((s) => rows[i] && rows[i][s.key] != null && isFinite(rows[i][s.key]) && (!logY || rows[i][s.key] > 0)));
        dots.exit().remove();
        dots.enter().append("circle").attr("class", "cdot").attr("r", 3.2).merge(dots)
          .attr("cx", P.x(i)).attr("cy", (s) => P.y(rows[i][s.key])).attr("fill", (s) => s.color);
        if (head) head.innerHTML = series.map((s) => `<span class="legend-item"><span class="swatch" style="background:${s.color}"></span>${s.label}${rows[i] && rows[i][s.key] != null && isFinite(rows[i][s.key]) ? ` <b class="num">${fmt(rows[i][s.key], s.digits ?? 3)}</b>` : ""}</span>`).join("");
      },
      xy: P,
    };
    return api;
  }

  // Simple bar chart (votes, weights, explained variance...).
  function bars(el, { w = 360, h = 130, yd = null, signed = false, xLabel = "", yLabel = "" } = {}) {
    let P = null;
    return {
      set(items, opts = {}) {
        const labels = items.map((d) => d.label);
        const max = yd ? yd[1] : Math.max(1e-9, ...items.map((d) => Math.abs(d.value)));
        const lo = yd ? yd[0] : signed ? -max : 0;
        d3.select(el).selectAll("svg").remove();
        P = xy(el, { w, h, band: true, xd: labels, yd: [lo, max * 1.08], xLabel, yLabel });
        P.g.selectAll("rect").data(items).enter().append("rect")
          .attr("x", (d) => P.x(d.label)).attr("width", P.x.bandwidth())
          .attr("y", (d) => P.y(Math.max(0, d.value))).attr("height", (d) => Math.max(0.5, Math.abs(P.y(d.value) - P.y(0))))
          .attr("fill", (d, i) => d.color || col(i)).attr("rx", 2).attr("opacity", (d) => d.dim ? 0.35 : 0.9);
        if (opts.values !== false) P.g.selectAll("text.bv").data(items).enter().append("text").attr("class", "bar-val")
          .attr("x", (d) => P.x(d.label) + P.x.bandwidth() / 2).attr("y", (d) => P.y(Math.max(0, d.value)) - 3).attr("text-anchor", "middle")
          .text((d) => d.text != null ? d.text : fmt(d.value, opts.digits ?? 2));
      },
    };
  }

  // Matrix heatmap (covariance, attention, Q-table, confusion...).
  function matrix(el, { cell = 26, rowLabels = [], colLabels = [], fmtv = (v) => fmt(v, 2), color = null, showValues = true, labelW = 46, labelH = 18, maxW = null } = {}) {
    const wrap = d3.select(el);
    return {
      set(M, { hl = null, range = null, rowL = rowLabels, colL = colLabels } = {}) {
        wrap.selectAll("svg").remove();
        const R = M.length, C = R ? M[0].length : 0;
        let c = cell;
        if (maxW && C * c + labelW > maxW) c = Math.max(10, Math.floor((maxW - labelW) / C));
        const W = labelW + C * c + 4, H = labelH + R * c + 4;
        const svg = wrap.append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("class", "mx-svg").style("max-width", W + "px");
        let lo = Infinity, hi = -Infinity;
        for (const r of M) for (const v of r) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
        if (range) [lo, hi] = range;
        const signed = lo < 0 && hi > 0;
        const mx = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
        const P = palRGB();
        const fill = color || ((v) => {
          if (!isFinite(v)) return "transparent";
          if (signed) { const t = clamp(v / mx, -1, 1); const k = t < 0 ? P[0] : P[1]; return `rgba(${k[0]},${k[1]},${k[2]},${Math.abs(t) * 0.85 + 0.05})`; }
          const t = hi - lo < 1e-12 ? 0.5 : (v - lo) / (hi - lo); const k = P[0]; return `rgba(${k[0]},${k[1]},${k[2]},${t * 0.85 + 0.06})`;
        });
        const g = svg.append("g").attr("transform", `translate(${labelW},${labelH})`);
        for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
          const on = hl && ((hl.r === i && (hl.c == null || hl.c === j)) || (hl.c === j && hl.r == null) || (hl.cells && hl.cells.some((q) => q[0] === i && q[1] === j)));
          g.append("rect").attr("x", j * c).attr("y", i * c).attr("width", c - 1.5).attr("height", c - 1.5).attr("rx", 2)
            .attr("fill", fill(M[i][j])).attr("class", on ? "mx-cell on" : "mx-cell");
          if (showValues && c >= 22) g.append("text").attr("class", "mx-val").attr("x", j * c + c / 2 - 0.75).attr("y", i * c + c / 2 + 3.5).attr("text-anchor", "middle").text(fmtv(M[i][j]));
        }
        rowL.forEach((t, i) => svg.append("text").attr("class", "mx-lab").attr("x", labelW - 5).attr("y", labelH + i * c + c / 2 + 3.5).attr("text-anchor", "end").text(t));
        colL.forEach((t, j) => svg.append("text").attr("class", "mx-lab").attr("x", labelW + j * c + c / 2).attr("y", labelH - 6).attr("text-anchor", "middle").text(t));
      },
    };
  }

  // ---------------------------------------------------------------- architecture diagram
  // spec: { w, h, nodes: [{id,label,sub,x,y,w,h,kind,info}], edges: [{from,to,label,bend,dash}] }
  function arch(el, spec, { onNode } = {}) {
    const W = spec.w || 380, H = spec.h || 210;
    const id = uid("arch");
    el.classList.add("arch-wrap");
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("class", "arch-svg");
    const defs = svg.append("defs");
    for (const [k, c] of [["", "var(--text-faint)"], ["-on", "var(--accent)"]]) {
      defs.append("marker").attr("id", `${id}-arr${k}`).attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5)
        .attr("markerWidth", 6.5).attr("markerHeight", 6.5).attr("orient", "auto-start-reverse")
        .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", c);
    }
    const nodes = new Map(spec.nodes.map((n) => {
      const o = { w: 86, h: 34, kind: "op", ...n };
      // widen to fit the text so labels never spill out of their box
      o.w = Math.max(o.w, (o.label || "").length * 6.7 + 16, (o.sub || "").length * 5.3 + 14);
      return [n.id, o];
    }));
    const anchor = (a, tx, ty) => {
      const dx = tx - a.x, dy = ty - a.y;
      const s = Math.min((a.w / 2 + 3) / Math.max(Math.abs(dx), 1e-9), (a.h / 2 + 3) / Math.max(Math.abs(dy), 1e-9));
      return [a.x + dx * s, a.y + dy * s];
    };
    const eg = svg.append("g");
    const ng = svg.append("g");
    const edges = (spec.edges || []).map((e) => {
      const a = nodes.get(e.from), b = nodes.get(e.to);
      if (!a || !b) return null;
      let d, mx, my;
      if (e.bend) {
        const midx = (a.x + b.x) / 2, midy = (a.y + b.y) / 2;
        const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
        const cx = midx - ((b.y - a.y) / len) * e.bend, cy = midy + ((b.x - a.x) / len) * e.bend;
        const p0 = anchor(a, cx, cy), p1 = anchor(b, cx, cy);
        d = `M${p0[0]},${p0[1]} Q${cx},${cy} ${p1[0]},${p1[1]}`;
        mx = 0.25 * p0[0] + 0.5 * cx + 0.25 * p1[0]; my = 0.25 * p0[1] + 0.5 * cy + 0.25 * p1[1];
      } else {
        const p0 = anchor(a, b.x, b.y), p1 = anchor(b, a.x, a.y);
        d = `M${p0[0]},${p0[1]} L${p1[0]},${p1[1]}`;
        mx = (p0[0] + p1[0]) / 2; my = (p0[1] + p1[1]) / 2;
      }
      const gEl = eg.append("g").attr("class", "arch-edge" + (e.dash ? " dashed" : ""));
      gEl.append("path").attr("d", d).attr("class", "ae-line").attr("marker-end", `url(#${id}-arr)`);
      gEl.append("path").attr("d", d).attr("class", "ae-flow");
      if (e.label) gEl.append("text").attr("class", "ae-label").attr("x", mx).attr("y", my - 4).attr("text-anchor", "middle").text(e.label);
      return { ...e, g: gEl };
    }).filter(Boolean);

    const tip = document.createElement("div");
    tip.className = "arch-tip"; tip.hidden = true;
    el.appendChild(tip);

    for (const n of nodes.values()) {
      const g = ng.append("g").attr("class", `arch-node k-${n.kind}`).attr("transform", `translate(${n.x - n.w / 2},${n.y - n.h / 2})`).attr("tabindex", 0);
      g.append("rect").attr("width", n.w).attr("height", n.h).attr("rx", n.kind === "param" ? n.h / 2 : 6);
      const hasSub = !!n.sub;
      g.append("text").attr("class", "an-label").attr("x", n.w / 2).attr("y", hasSub ? n.h / 2 - 2 : n.h / 2 + 4).attr("text-anchor", "middle").text(n.label);
      if (hasSub) g.append("text").attr("class", "an-sub").attr("x", n.w / 2).attr("y", n.h / 2 + 10).attr("text-anchor", "middle").text(n.sub);
      n.g = g;
      const show = () => {
        if (!n.info && !n.tex) return;
        tip.innerHTML = `<b>${esc(n.label)}</b>${n.info ? `<div>${n.info}</div>` : ""}${n.tex ? `<div class="tip-tex">${texHTML(n.tex, false)}</div>` : ""}`;
        tip.hidden = false;
        const box = el.getBoundingClientRect(), sb = svg.node().getBoundingClientRect();
        const k = sb.width / W;
        const left = (n.x + n.w / 2) * k + (sb.left - box.left) + 8;
        tip.style.left = Math.min(left, box.width - 230) + "px";
        tip.style.top = Math.max(0, (n.y - n.h / 2) * k + (sb.top - box.top)) + "px";
      };
      g.on("mouseenter", show).on("focus", show).on("mouseleave", () => (tip.hidden = true)).on("blur", () => (tip.hidden = true));
      g.on("click", () => onNode && onNode(n.id));
      g.on("keydown", (ev) => { if (ev.key === "Enter") onNode && onNode(n.id); });
    }
    const api = {
      highlight(ids = []) {
        const set = new Set(ids);
        for (const n of nodes.values()) n.g.classed("on", set.has(n.id));
        for (const e of edges) {
          const on = set.has(e.from) && set.has(e.to) || set.has(`${e.from}>${e.to}`);
          e.g.classed("on", on);
          e.g.select(".ae-line").attr("marker-end", `url(#${id}-arr${on ? "-on" : ""})`);
        }
      },
      setSub(nid, text) { const n = nodes.get(nid); if (n) n.g.select(".an-sub").text(text); },
      nodes,
    };
    return api;
  }

  // ---------------------------------------------------------------- neural network diagram
  function net(el, { layers, w = 380, h = 200, labels = null, onNeuron = null, r = 8 }) {
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("class", "net-svg");
    const L = layers.length;
    const pos = layers.map((n, l) => Array.from({ length: n }, (_, i) => {
      const x = 28 + (l * (w - 56)) / Math.max(1, L - 1);
      const span = Math.min(h - 30, (n - 1) * (r * 2 + 6));
      return [x, h / 2 - span / 2 + (n > 1 ? (i * span) / (n - 1) : 0)];
    }));
    const eg = svg.append("g"), ng = svg.append("g"), lg = svg.append("g");
    if (labels) labels.forEach((t, l) => lg.append("text").attr("class", "net-lab").attr("x", pos[l][0][0]).attr("y", h - 4).attr("text-anchor", "middle").text(t));
    const edges = [];
    for (let l = 0; l < L - 1; l++) for (let i = 0; i < layers[l]; i++) for (let j = 0; j < layers[l + 1]; j++) {
      edges.push({ l, i, j, el: eg.append("line").attr("x1", pos[l][i][0]).attr("y1", pos[l][i][1]).attr("x2", pos[l + 1][j][0]).attr("y2", pos[l + 1][j][1]).attr("class", "net-edge") });
    }
    const neurons = [];
    pos.forEach((layer, l) => layer.forEach(([cx, cy], i) => {
      const c = ng.append("circle").attr("cx", cx).attr("cy", cy).attr("r", r).attr("class", "net-node");
      if (onNeuron) c.style("cursor", "pointer").on("mouseenter", () => onNeuron(l, i)).on("click", () => onNeuron(l, i, true));
      neurons.push({ l, i, el: c });
    }));
    return {
      // weights[l][i][j]: from unit i in layer l to unit j in layer l+1
      weights(Wts) {
        let mx = 1e-9;
        for (const e of edges) { const v = Wts[e.l] && Wts[e.l][e.i] ? Wts[e.l][e.i][e.j] : 0; mx = Math.max(mx, Math.abs(v)); }
        for (const e of edges) {
          const v = Wts[e.l] && Wts[e.l][e.i] ? Wts[e.l][e.i][e.j] : 0;
          e.el.style("stroke", v >= 0 ? "var(--series-1)" : "var(--series-2)").style("stroke-width", 0.3 + 3 * Math.abs(v) / mx).style("stroke-opacity", 0.18 + 0.75 * Math.abs(v) / mx);
        }
      },
      activations(A) { // A[l][i] in roughly [-1, 1] or [0, 1]
        for (const n of neurons) {
          const v = A && A[n.l] ? A[n.l][n.i] : null;
          n.el.style("fill", v == null ? "var(--bg-inset)" : v >= 0 ? `color-mix(in srgb, var(--series-1) ${Math.round(clamp(v, 0, 1) * 85)}%, var(--bg-inset))` : `color-mix(in srgb, var(--series-2) ${Math.round(clamp(-v, 0, 1) * 85)}%, var(--bg-inset))`);
        }
      },
      highlightLayer(ls) { const s = new Set(ls); for (const n of neurons) n.el.classed("on", s.has(n.l)); for (const e of edges) e.el.classed("on", s.has(e.l) && s.has(e.l + 1)); },
      select(l, i) { for (const n of neurons) n.el.classed("sel", n.l === l && n.i === i); },
      pos,
    };
  }

  // ---------------------------------------------------------------- player
  function player(el, { onFrame, fps = 5 }) {
    el.innerHTML = `
      <div class="player-btns">
        <button data-p="first" title="first step (Home)" aria-label="first step">${ICON.first}</button>
        <button data-p="prev" title="step back (←)" aria-label="step back">${ICON.prev}</button>
        <button data-p="play" class="primary play-btn" title="play / pause (space)" aria-label="play">${ICON.play}</button>
        <button data-p="next" title="step forward (→)" aria-label="step forward">${ICON.next}</button>
        <button data-p="last" title="last step (End)" aria-label="last step">${ICON.last}</button>
      </div>
      <div class="player-track"><input type="range" data-p="scrub" min="0" max="0" value="0" aria-label="scrub through training steps" /><div class="player-ticks" data-p="ticks"></div></div>
      <div class="player-meta">
        <span class="player-pos" data-p="pos">0 / 0</span>
        <select data-p="speed" aria-label="playback speed" title="playback speed">
          <option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option><option value="20">max</option>
        </select>
        <button data-p="loop" class="icon-toggle" title="loop" aria-label="loop" aria-pressed="false">${ICON.loop}</button>
      </div>`;
    const q = (k) => el.querySelector(`[data-p="${k}"]`);
    dropdown(q("speed"), { compact: true });
    let frames = [], idx = 0, playing = false, timer = null, loop = false, live = false;
    const listeners = [];
    function render() {
      q("scrub").max = Math.max(0, frames.length - 1);
      q("scrub").value = idx;
      q("pos").textContent = frames.length ? `${idx + 1} / ${frames.length}` : "0 / 0";
      q("play").innerHTML = playing ? ICON.pause : ICON.play;
      q("play").setAttribute("aria-label", playing ? "pause" : "play");
      const pct = frames.length > 1 ? (idx / (frames.length - 1)) * 100 : 0;
      q("scrub").style.setProperty("--pct", pct + "%");
    }
    function ticks() {
      // marks where the phase changes, so the scrubber reads like chapters
      const t = q("ticks"); t.innerHTML = "";
      if (frames.length < 2 || frames.length > 600) return;
      let prev = null;
      frames.forEach((f, i) => {
        if (f.mark || (f.phase !== prev && f.major)) {
          const s = document.createElement("span");
          s.style.left = (i / (frames.length - 1)) * 100 + "%";
          s.title = f.title || f.phase || "";
          t.appendChild(s);
        }
        prev = f.phase;
      });
    }
    function show(i) {
      if (!frames.length) { render(); return; }
      idx = clamp(i, 0, frames.length - 1);
      render();
      try { onFrame(frames[idx], idx, frames); } catch (e) { console.error(e); }
      listeners.forEach((fn) => fn(frames[idx], idx, frames));
    }
    function speed() { return +q("speed").value; }
    function tick() {
      timer = null;
      if (!playing) return;
      if (idx >= frames.length - 1) {
        if (live) { timer = setTimeout(tick, 120); return; }
        if (loop && frames.length > 1) show(0);
        else { playing = false; render(); return; }
      } else {
        const jump = speed() >= 20 ? Math.max(1, Math.ceil(frames.length / 120)) : 1;
        show(idx + jump);
      }
      timer = setTimeout(tick, speed() >= 20 ? 16 : 1000 / (fps * speed()));
    }
    function play() { if (!frames.length) return; if (idx >= frames.length - 1 && !live) show(0); playing = true; render(); clearTimeout(timer); timer = setTimeout(tick, 1000 / (fps * speed())); }
    function pause() { playing = false; clearTimeout(timer); timer = null; render(); }
    q("play").onclick = () => (playing ? pause() : play());
    q("first").onclick = () => { pause(); show(0); };
    q("last").onclick = () => { pause(); show(frames.length - 1); };
    q("prev").onclick = () => { pause(); show(idx - 1); };
    q("next").onclick = () => { pause(); show(idx + 1); };
    q("scrub").oninput = (e) => { pause(); show(+e.target.value); };
    q("loop").onclick = () => { loop = !loop; q("loop").classList.toggle("on", loop); q("loop").setAttribute("aria-pressed", String(loop)); };
    render();
    return {
      load(fr, { at = "start", play: autoplay = false } = {}) {
        frames = fr || [];
        ticks();
        show(at === "end" ? frames.length - 1 : typeof at === "number" ? at : 0);
        if (autoplay) play(); else if (!live) pause();
      },
      append(f) {
        frames.push(f);
        const follow = idx >= frames.length - 2;
        if (follow) show(frames.length - 1); else render();
      },
      replaceAll(fr) { frames = fr; idx = Math.min(idx, frames.length - 1); render(); },
      setLive(v) { live = v; },
      play, pause, show, redraw() { show(idx); },
      step(d) { pause(); show(idx + d); },
      toggle() { playing ? pause() : play(); },
      get frames() { return frames; }, get index() { return idx; }, get playing() { return playing; },
      onShow(fn) { listeners.push(fn); },
      destroy() { pause(); },
    };
  }

  // ---------------------------------------------------------------- custom dropdown
  // Replaces native <select> controls with a small modal-style listbox. The
  // native select stays in the DOM (visually hidden) as the source of truth,
  // so lab.val / lab.watch / lab.ctl keep working untouched: picking an
  // option sets select.value and dispatches input + change.
  let ddOpen = null;
  function ddGlobal() {
    if (ddGlobal.done) return;
    ddGlobal.done = true;
    document.addEventListener("pointerdown", (e) => {
      if (ddOpen && e.target instanceof Node && !ddOpen.root.contains(e.target)) ddCloseAll(false);
    }, true);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && ddOpen) { e.stopPropagation(); ddCloseAll(true); }
    }, true);
    window.addEventListener("scroll", (e) => {
      if (ddOpen && e.target instanceof Node && !ddOpen.root.contains(e.target)) ddCloseAll(false);
    }, true);
  }
  function ddCloseAll(refocus) {
    const api = ddOpen; ddOpen = null;
    if (!api) return;
    api.list.hidden = true;
    api.btn.setAttribute("aria-expanded", "false");
    api.root.classList.remove("open");
    if (refocus && api.btn.isConnected) api.btn.focus({ preventScroll: true });
  }
  function dropdown(select, { compact = false } = {}) {
    ddGlobal();
    if (!select || select.dataset.ddDone) return null;
    select.dataset.ddDone = "1";
    const opts = [...select.options].map((o) => ({ v: o.value, t: o.textContent }));
    if (!opts.length) return null;
    const root = document.createElement("div");
    root.className = "dd" + (compact ? " dd-compact" : "");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dd-btn";
    btn.setAttribute("aria-haspopup", "listbox");
    btn.setAttribute("aria-expanded", "false");
    const val = document.createElement("span");
    val.className = "dd-val";
    const chev = document.createElement("span");
    chev.className = "dd-chev"; chev.setAttribute("aria-hidden", "true"); chev.textContent = "▾";
    btn.append(val, chev);
    const list = document.createElement("div");
    list.className = "dd-list"; list.setAttribute("role", "listbox"); list.tabIndex = -1; list.hidden = true;
    root.append(btn, list);
    select.after(root);
    select.classList.add("dd-native");
    select.tabIndex = -1;
    select.setAttribute("aria-hidden", "true");
    const optEls = opts.map((o) => {
      const d = document.createElement("div");
      d.className = "dd-opt"; d.setAttribute("role", "option"); d.dataset.v = o.v; d.tabIndex = -1;
      const ck = document.createElement("span"); ck.className = "dd-check"; ck.setAttribute("aria-hidden", "true"); ck.textContent = "✓";
      const tx = document.createElement("span"); tx.textContent = o.t;
      d.append(ck, tx);
      return d;
    });
    list.append(...optEls);
    const labelOf = (v) => { const o = opts.find((o) => o.v === v); return o ? o.t : v; };
    function paint() {
      val.textContent = labelOf(select.value);
      btn.title = labelOf(select.value);
      optEls.forEach((el) => {
        const on = el.dataset.v === select.value;
        el.classList.toggle("on", on);
        el.setAttribute("aria-selected", String(on));
      });
    }
    function pick(v, refocus = true) {
      ddCloseAll(false);
      if (select.value !== v) {
        select.value = v;
        select.dispatchEvent(new Event("input", { bubbles: true }));
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }
      paint();
      if (refocus && btn.isConnected) btn.focus({ preventScroll: true });
    }
    let hi = 0;
    function hl(i) {
      hi = ((i % optEls.length) + optEls.length) % optEls.length;
      optEls.forEach((el, k) => el.classList.toggle("hl", k === hi));
      optEls[hi].scrollIntoView({ block: "nearest" });
    }
    function open() {
      if (ddOpen && ddOpen.root === root) { ddCloseAll(true); return; }
      ddCloseAll(false);
      paint();
      list.hidden = false;
      btn.setAttribute("aria-expanded", "true");
      root.classList.add("open");
      const r = root.getBoundingClientRect();
      root.classList.toggle("up", r.bottom + 230 > window.innerHeight && r.top > 250);
      ddOpen = { root, btn, list };
      hi = Math.max(0, opts.findIndex((o) => o.v === select.value));
      hl(hi);
      list.focus({ preventScroll: true });
    }
    // Enter / Space toggle through the native button click; only arrows and
    // Space-scroll need keydown handling here.
    btn.addEventListener("click", () => (list.hidden ? open() : ddCloseAll(true)));
    btn.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); if (list.hidden) open(); }
      else if (e.key === " ") e.preventDefault();
    });
    list.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); hl(hi + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); hl(hi - 1); }
      else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(optEls[hi].dataset.v); }
      else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); ddCloseAll(true); }
      else if (e.key === "Tab") ddCloseAll(false);
    });
    list.addEventListener("mousemove", (e) => {
      const o = e.target && e.target.closest ? e.target.closest(".dd-opt") : null;
      if (o) hl(optEls.indexOf(o));
    });
    list.addEventListener("click", (e) => {
      const o = e.target && e.target.closest ? e.target.closest(".dd-opt") : null;
      if (o) pick(o.dataset.v);
    });
    select.addEventListener("change", paint);
    select.addEventListener("input", paint);
    paint();
    return { root, btn, list, sync: paint };
  }

  // ---------------------------------------------------------------- dataset card
  function datasetCardHTML(d) {
    const opts = d.presets.map((k) => `<option value="${k}"${k === d.def ? " selected" : ""}>${(d.names && d.names[k]) || (DATA[k] ? DATA[k].name : k)}</option>`).join("");
    return `
      <div class="control-card ds-card">
        <h3>dataset</h3>
        <div class="field"><label>example data</label><select data-ds="preset">${opts}</select></div>
        ${d.fixedN ? "" : `<div class="field"><label>samples <span class="val" data-ds="n-val">${d.n}</span></label><input type="range" data-ds="n" min="${d.nRange ? d.nRange[0] : 10}" max="${d.nRange ? d.nRange[1] : 300}" step="5" value="${d.n}" /></div>`}
        ${d.noNoise ? "" : `<div class="field"><label>noise <span class="val" data-ds="noise-val">${d.noise.toFixed(2)}</span></label><input type="range" data-ds="noise" min="0" max="1" step="0.05" value="${d.noise}" /></div>`}
        <div class="btn-row">
          <button data-ds="reseed" title="new random sample from the same distribution">reseed</button>
          ${d.csv === false ? "" : `<label class="file-btn" title="load x,y[,label] from a CSV file">upload csv<input type="file" accept=".csv,.txt,text/csv" data-ds="file" hidden /></label>
          <button data-ds="export" title="download the current points as CSV">export</button>`}
          ${d.clear === false ? "" : `<button data-ds="clear">clear</button>`}
        </div>
        ${d.hint ? `<div class="note">${d.hint}</div>` : ""}
      </div>`;
  }

  // ---------------------------------------------------------------- studio: the standard lab layout
  // cfg: { id, legend, tools, stageHTML, hint, plane:{...}|false, data:{kind,presets,def,n,noise},
  //        classes, side:[{title,html}], arch, charts:[{id,title,series,logY,custom}], panels:[{id,title,html,wide}],
  //        note, onFrame(frame, i), onData(points), fps, phases:{key:{label,color}} }
  function studio(root, cfg) {
    const classButtons = cfg.classes
      ? `<div class="class-pick" data-r="classes">${Array.from({ length: cfg.classes }, (_, i) => `<button data-c="${i}" class="${i === 0 ? "on" : ""}" title="add points as class ${i}${i === 1 ? " (or shift-click)" : ""}"><span class="swatch" style="background:${col(i)}"></span>${cfg.classNames ? cfg.classNames[i] : i}</button>`).join("")}</div>`
      : "";
    const chartsHTML = (cfg.charts || []).map((c) => `
      <div class="lab-card lab-chart${c.wide ? " wide" : ""}"><div class="lab-card-head"><span>${c.title}</span><span class="chart-legend"></span></div><div data-chart="${c.id}"></div>${c.note ? `<div class="lab-card-note">${c.note}</div>` : ""}</div>`).join("");
    const panelsHTML = (cfg.panels || []).map((p) => `
      <div class="lab-card${p.wide ? " wide" : ""}" data-panel="${p.id}"><div class="lab-card-head"><span>${p.title}</span><span class="lab-card-sub">${p.sub || ""}</span></div><div data-r="panel-${p.id}">${p.html || ""}</div></div>`).join("");
    root.innerHTML = `
      <div class="panel lab">
        <div class="lab-main">
          <div class="stage">
            <div class="stage-toolbar">
              <div class="legend" data-r="legend">${cfg.legend || ""}${classButtons}</div>
              <div class="btn-row" data-r="tools">${cfg.tools || ""}</div>
            </div>
            <div class="stage-body" data-r="stage">${cfg.stageHTML != null ? cfg.stageHTML : `<svg data-r="svg" class="lab-svg" role="img" aria-label="interactive plot"></svg>`}</div>
            ${cfg.hint ? `<div class="stage-hint" data-r="hint">${cfg.hint}</div>` : ""}
          </div>
          ${cfg.player === false ? "" : `<div class="player" data-r="player"></div>`}
          <div class="lab-lower">
            <div class="lab-card lab-narr" data-r="narr">
              <div class="narr-head"><span class="phase-chip" data-n="phase">ready</span><span class="narr-step" data-n="step"></span></div>
              <div class="narr-title" data-n="title">Press play to watch the algorithm work</div>
              <div class="narr-text" data-n="text">Each step of training is recorded as a frame. Play it, scrub it, or step with ← →.</div>
              <div class="narr-tex" data-n="tex"></div>
            </div>
            ${cfg.arch ? `<div class="lab-card lab-arch"><div class="lab-card-head"><span>${cfg.archTitle || "architecture"}</span><span class="lab-card-sub">hover a block · click to jump there</span></div><div data-r="arch"></div></div>` : ""}
            ${panelsHTML}
            ${chartsHTML}
          </div>
        </div>
        <div class="controls">
          ${cfg.data ? datasetCardHTML({ n: 80, noise: 0.3, ...cfg.data }) : ""}
          ${(cfg.side || []).map((c) => `<div class="control-card"${c.id ? ` data-side="${c.id}"` : ""}><h3>${c.title}</h3>${c.html}</div>`).join("")}
          <div class="control-card"><h3>readout</h3><div class="readout" data-r="readout">–</div>${cfg.note ? `<div class="note">${cfg.note}</div>` : ""}</div>
        </div>
      </div>`;
    const R = (k) => root.querySelector(`[data-r="${k}"]`);
    const q = (sel) => root.querySelector(sel);
    const lab = { root, R, q, cfg, points: [], cleanup: [] };

    // ---- custom dropdowns replace every native select (controls + dataset)
    root.querySelectorAll("select[data-k]").forEach((s) => dropdown(s));
    if (cfg.data) { const ps = root.querySelector('select[data-ds="preset"]'); if (ps) dropdown(ps); }

    // ---- plane
    if (cfg.plane !== false && R("svg")) lab.plane = plane(R("svg"), cfg.plane || {});

    // ---- narration
    const narr = R("narr");
    const N = (k) => narr.querySelector(`[data-n="${k}"]`);
    lab.narrate = (f, i, all) => {
      if (!f) return;
      const ph = cfg.phases && cfg.phases[f.phase];
      const chip = N("phase");
      chip.textContent = (ph && ph.label) || f.phase || "step";
      chip.style.setProperty("--chip", ph && ph.color != null ? col(ph.color) : "var(--text-faint)");
      N("step").textContent = all ? `step ${i + 1} of ${all.length}` : "";
      N("title").innerHTML = richText(f.title || "");
      N("text").innerHTML = richText(f.text || "");
      N("tex").innerHTML = f.tex ? texHTML(f.tex, true) : "";
    };

    // ---- architecture
    if (cfg.arch) {
      lab.arch = arch(R("arch"), cfg.arch, {
        onNode: (nid) => {
          const fr = lab.player ? lab.player.frames : [];
          if (!fr.length) return;
          const start = lab.player.index;
          for (let k = 1; k <= fr.length; k++) {
            const j = (start + k) % fr.length;
            if ((fr[j].arch || []).includes(nid)) { lab.player.pause(); lab.player.show(j); return; }
          }
        },
      });
    }

    // ---- charts
    lab.charts = {};
    for (const c of cfg.charts || []) {
      const el = root.querySelector(`[data-chart="${c.id}"]`);
      lab.charts[c.id] = c.custom ? { el } : chart(el, c);
    }

    // ---- player
    if (cfg.player !== false) {
      lab.player = player(R("player"), {
        fps: cfg.fps || 5,
        onFrame: (f, i, all) => {
          lab.narrate(f, i, all);
          if (lab.arch) lab.arch.highlight(f.arch || []);
          for (const c of cfg.charts || []) if (!c.custom && lab.charts[c.id]) lab.charts[c.id].cursor(i);
          if (cfg.onFrame) cfg.onFrame(f, i, all);
          if (typeof MLApp !== "undefined" && MLApp.status) MLApp.status({ phase: (cfg.phases && cfg.phases[f.phase] && cfg.phases[f.phase].label) || f.phase, frame: `${i + 1}/${all.length}` });
        },
      });
    }
    // Load frames and refresh the metric charts from each frame's `m` field.
    lab.load = (frames, opts = {}) => {
      for (const c of cfg.charts || []) if (!c.custom && lab.charts[c.id]) lab.charts[c.id].set(frames.map((f) => f.m || {}));
      lab.player.load(frames, opts);
    };
    lab.readout = (html) => { R("readout").innerHTML = html; };

    // ---- class picker
    let curClass = 0;
    if (cfg.classes) {
      R("classes").addEventListener("click", (e) => {
        const b = e.target.closest("button"); if (!b) return;
        curClass = +b.dataset.c;
        R("classes").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      });
    }
    lab.cls = (ev) => (ev && ev.shiftKey ? 1 : curClass);

    // ---- dataset
    if (cfg.data) {
      const D = (k) => root.querySelector(`[data-ds="${k}"]`);
      let seed = 7;
      const params = () => ({ n: D("n") ? +D("n").value : cfg.data.n || 80, noise: D("noise") ? +D("noise").value : cfg.data.noise || 0.3 });
      lab.loadPreset = (key, { reseed = false } = {}) => {
        if (reseed) seed = (seed * 9301 + 49297) % 233280;
        const k = key || D("preset").value;
        if (cfg.data.custom && cfg.data.custom[k]) { MLU.seedRng(seed); lab.points = MLU.shuffle(cfg.data.custom[k](params(), seed)); }
        else lab.points = generate(k, { ...params(), seed });
        if (cfg.onData) cfg.onData(lab.points, k);
      };
      lab.preset = () => D("preset").value;
      D("preset").onchange = () => lab.loadPreset();
      if (D("n")) D("n").oninput = () => { D("n-val").textContent = D("n").value; };
      if (D("n")) D("n").onchange = () => lab.loadPreset();
      if (D("noise")) D("noise").oninput = () => { D("noise-val").textContent = (+D("noise").value).toFixed(2); };
      if (D("noise")) D("noise").onchange = () => lab.loadPreset();
      D("reseed").onclick = () => lab.loadPreset(null, { reseed: true });
      if (D("clear")) D("clear").onclick = () => { lab.points = []; if (cfg.onData) cfg.onData(lab.points, "custom"); };
      if (D("export")) D("export").onclick = () => downloadCSV(lab.points, `${cfg.id || "lab"}-data.csv`);
      if (D("file")) D("file").onchange = (e) => {
        const file = e.target.files && e.target.files[0]; if (!file) return;
        const rd = new FileReader();
        rd.onload = () => {
          const pts = parseCSV(String(rd.result), cfg.data.kind);
          if (pts) { lab.points = cfg.data.kind === "reg" ? pts.map((p) => ({ x: p.x, y: p.y })) : pts; if (cfg.onData) cfg.onData(lab.points, "csv"); }
          else lab.readout("could not read that file: expected rows of x,y[,label]");
        };
        rd.readAsText(file);
        e.target.value = "";
      };
    }

    // ---- editable points on the plane
    // newPoint(x, y, event) returns the point to add (or null).
    lab.editable = ({ newPoint, onChange, live = false, drag = true } = {}) => {
      const pl = lab.plane; if (!pl) return;
      pl.svg.on("click.edit", (event) => {
        if (event.target.closest && event.target.closest(".pt-edit, .no-add")) return;
        const [px, py] = d3.pointer(event, pl.svg.node());
        if (!pl.inside(px, py)) return;
        const p = newPoint ? newPoint(pl.x.invert(px), pl.y.invert(py), event) : { x: pl.x.invert(px), y: pl.y.invert(py), label: lab.cls(event) };
        if (!p) return;
        lab.points.push(p);
        onChange && onChange("add", p);
      });
      lab.bindPoints = (sel) => {
        sel.classed("pt-edit", true)
          .on("dblclick", (event, d) => { event.stopPropagation(); lab.points = lab.points.filter((p) => p !== d); onChange && onChange("remove", d); });
        if (drag) sel.call(d3.drag()
          .on("start", function () { d3.select(this).classed("dragging", true); })
          .on("drag", function (event, d) {
            d.x = clamp(pl.x.invert(event.x), pl.x.domain()[0], pl.x.domain()[1]);
            d.y = clamp(pl.y.invert(event.y), pl.y.domain()[0], pl.y.domain()[1]);
            d3.select(this).attr("cx", pl.x(d.x)).attr("cy", pl.y(d.y));
            if (live) onChange && onChange("drag", d);
          })
          .on("end", function (event, d) { d3.select(this).classed("dragging", false); onChange && onChange("move", d); }));
        return sel;
      };
    };
    // Draws lab.points (or given pts) on the plane's point layer.
    lab.drawPoints = ({ pts = lab.points, color = (d) => col(d.label || 0), r = 4.6, opacity = 1, stroke = "var(--bg)", sw = 1.2, cls = "pt", layer = "pts", edit = true, title = null } = {}) => {
      const g = lab.plane.layers[layer];
      const sel = g.selectAll(`circle.${cls}`).data(pts);
      sel.exit().remove();
      const m = sel.enter().append("circle").attr("class", cls).merge(sel)
        .attr("cx", (d) => lab.plane.x(d.x)).attr("cy", (d) => lab.plane.y(d.y))
        .attr("r", r).attr("fill", color).attr("stroke", stroke).attr("stroke-width", sw).attr("opacity", opacity);
      if (title) { m.selectAll("title").remove(); m.append("title").text(title); }
      if (edit && lab.bindPoints) lab.bindPoints(m);
      return m;
    };

    // ---- keyboard: space / arrows drive the player
    const onKey = (e) => {
      if (!lab.player) return;
      const t = (e.target.tagName || "").toLowerCase();
      if (["input", "select", "textarea"].includes(t) || e.target.isContentEditable) return;
      if (e.target.closest && e.target.closest(".dd")) return; // custom dropdown handles its own keys
      if (t === "button" && (e.key === " " || e.key === "Enter")) return;
      if (!root.isConnected || root.offsetParent === null) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === " ") { e.preventDefault(); lab.player.toggle(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); lab.player.step(1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); lab.player.step(-1); }
      else if (e.key === "Home") { e.preventDefault(); lab.player.pause(); lab.player.show(0); }
      else if (e.key === "End") { e.preventDefault(); lab.player.pause(); lab.player.show(lab.player.frames.length - 1); }
    };
    document.addEventListener("keydown", onKey);
    const onTheme = () => { if (lab.player) lab.player.redraw(); if (cfg.onTheme) cfg.onTheme(); };
    window.addEventListener("ml-theme", onTheme);
    lab.destroy = () => {
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("ml-theme", onTheme);
      if (ddOpen && root.contains(ddOpen.root)) ddCloseAll(false);
      if (lab.player) lab.player.destroy();
      lab.cleanup.forEach((fn) => { try { fn(); } catch (e) {} });
    };
    // Reads a control by data-k attribute: <input data-k="lr">.
    lab.val = (k) => { const el = root.querySelector(`[data-k="${k}"]`); if (!el) return null; return el.type === "checkbox" ? el.checked : el.tagName === "SELECT" ? el.value : +el.value; };
    lab.ctl = (k) => root.querySelector(`[data-k="${k}"]`);
    // Wire range/select controls: shows value in [data-v=k] and calls fn on change.
    lab.watch = (keys, fn, { live = false, format = {} } = {}) => {
      for (const k of keys) {
        const el = lab.ctl(k); if (!el) continue;
        const upd = () => { const v = root.querySelector(`[data-v="${k}"]`); if (v) v.textContent = format[k] ? format[k](lab.val(k)) : lab.val(k); };
        upd();
        el.addEventListener("input", () => { upd(); if (live) fn(k); });
        if (!live) el.addEventListener("change", () => fn(k));
      }
    };
    return lab;
  }

  // Control-markup helpers so algorithm files stay terse.
  const ui = {
    range: (k, label, min, max, step, value, show = value) => `<div class="field"><label>${label} <span class="val" data-v="${k}">${show}</span></label><input type="range" data-k="${k}" min="${min}" max="${max}" step="${step}" value="${value}" /></div>`,
    select: (k, label, opts, value) => `<div class="field"><label>${label}</label><select data-k="${k}">${opts.map((o) => { const [v, t] = Array.isArray(o) ? o : [o, o]; return `<option value="${v}"${v == value ? " selected" : ""}>${t}</option>`; }).join("")}</select></div>`,
    check: (k, label, on = false) => `<label class="check"><input type="checkbox" data-k="${k}"${on ? " checked" : ""} /> ${label}</label>`,
    legend: (items) => items.map(([c, t, shape]) => `<span class="legend-item"><span class="swatch${shape ? " " + shape : ""}" style="background:${c}"></span>${t}</span>`).join(""),
    btn: (k, label, primary = false, title = "") => `<button data-b="${k}"${primary ? ' class="primary"' : ""}${title ? ` title="${title}"` : ""}>${label}</button>`,
  };

  // Epoch indices to record: dense at the start (where most change happens), sparser later.
  function checkpoints(total, maxFrames = 70) {
    const set = new Set([0, total]);
    for (let i = 0; i < maxFrames; i++) set.add(Math.round(Math.pow(total, i / (maxFrames - 1)) - 1 + (i ? 1 : 0)));
    return [...set].filter((v) => v >= 0 && v <= total).sort((a, b) => a - b);
  }
  // Mixes palette colours by weights (soft assignments) into a CSS colour.
  function mixColor(ws) {
    const P = palRGB(); let r = 0, g = 0, b = 0, t = 0;
    ws.forEach((w, k) => { r += w * P[k % 7][0]; g += w * P[k % 7][1]; b += w * P[k % 7][2]; t += w; });
    t = t || 1;
    return `rgb(${Math.round(r / t)},${Math.round(g / t)},${Math.round(b / t)})`;
  }
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  return {
    checkpoints, mixColor, sigmoid,
    DOMAIN, DATA, generate, parseCSV, downloadCSV, plane, sampleGrid, paintGrid, field, contour, ellipsePath,
    probColor, classColor, divergeColor, xy, chart, bars, matrix, arch, net, player, studio, ui, dropdown,
    texHTML, richText, col, pal, palRGB, rgb, esc, clamp, fmt, uid, ICON,
  };
})();
