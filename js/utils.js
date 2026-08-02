// Shared math + data-generation helpers used across algorithm modules.
// Deliberately dependency-free (pure JS) so every algorithm and guide module
// can rely on it without pulling in a numerical library.
const MLU = (() => {
  let seed = 42;
  function rng() {
    // mulberry32 - deterministic so demos are reproducible on reload
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function seedRng(s) { seed = s; }
  function randRange(a, b) { return a + rng() * (b - a); }
  function randn() {
    // Box-Muller
    const u = Math.max(rng(), 1e-9), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function randInt(n) { return Math.floor(rng() * n); }
  function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  // ---------- linear algebra (small, dense) ----------
  function zeros(n, m) { return Array.from({ length: n }, () => new Array(m).fill(0)); }
  function matMul(A, B) {
    const n = A.length, k = B.length, m = B[0].length;
    const out = zeros(n, m);
    for (let i = 0; i < n; i++)
      for (let p = 0; p < k; p++) {
        const a = A[i][p];
        if (a === 0) continue;
        for (let j = 0; j < m; j++) out[i][j] += a * B[p][j];
      }
    return out;
  }
  function transpose(A) {
    const n = A.length, m = A[0].length;
    const T = zeros(m, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j];
    return T;
  }
  function mean(vals) { return vals.reduce((a, b) => a + b, 0) / vals.length; }
  function std(vals) {
    const m = mean(vals);
    return Math.sqrt(mean(vals.map((v) => (v - m) ** 2))) || 1e-9;
  }
  function meanVec(points) {
    const d = points[0].length;
    const m = new Array(d).fill(0);
    for (const p of points) for (let i = 0; i < d; i++) m[i] += p[i] / points.length;
    return m;
  }
  function covMatrix(points, mu) {
    const d = points[0].length;
    const C = zeros(d, d);
    for (const p of points) {
      for (let i = 0; i < d; i++)
        for (let j = 0; j < d; j++) C[i][j] += (p[i] - mu[i]) * (p[j] - mu[j]) / points.length;
    }
    return C;
  }
  // Jacobi eigenvalue algorithm for small symmetric matrices (used by PCA / GMM ellipses)
  function jacobiEigen(Ain, maxSweeps = 100) {
    const n = Ain.length;
    const A = Ain.map((r) => r.slice());
    let V = zeros(n, n);
    for (let i = 0; i < n; i++) V[i][i] = 1;
    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-12) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          if (Math.abs(A[p][q]) < 1e-12) continue;
          const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          const app = A[p][p], aqq = A[q][q], apq = A[p][q];
          A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
          A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
          A[p][q] = A[q][p] = 0;
          for (let k = 0; k < n; k++) {
            if (k !== p && k !== q) {
              const akp = A[k][p], akq = A[k][q];
              A[k][p] = A[p][k] = c * akp - s * akq;
              A[k][q] = A[q][k] = s * akp + c * akq;
            }
          }
          for (let k = 0; k < n; k++) {
            const vkp = V[k][p], vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const eigVals = A.map((row, i) => row[i]);
    const eigVecs = V[0].map((_, j) => V.map((row) => row[j])); // columns -> rows of eigenvectors
    const order = eigVals.map((v, i) => i).sort((a, b) => eigVals[b] - eigVals[a]);
    return { values: order.map((i) => eigVals[i]), vectors: order.map((i) => eigVecs[i]) };
  }
  function inverse2x2(A) {
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    const d = Math.abs(det) < 1e-9 ? 1e-9 : det;
    return [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]];
  }
  function det2x2(A) { return A[0][0] * A[1][1] - A[0][1] * A[1][0]; }

  // General N x N inverse via Gauss-Jordan with partial pivoting.
  function inverseN(Ain) {
    const n = Ain.length;
    const A = Ain.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
      if (Math.abs(A[piv][col]) < 1e-12) A[piv][col] += 1e-9;
      [A[col], A[piv]] = [A[piv], A[col]];
      const pv = A[col][col];
      for (let j = 0; j < 2 * n; j++) A[col][j] /= pv;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = A[r][col];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[col][j];
      }
    }
    return A.map((row) => row.slice(n));
  }
  function solveLinearSystem(A, b) {
    // solves A x = b for square A via inverse (fine for the small N used here)
    const Ainv = inverseN(A);
    return Ainv.map((row) => row.reduce((s, v, j) => s + v * b[j], 0));
  }

  // ---------- distances ----------
  function dist2(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return s; }
  function dist(a, b) { return Math.sqrt(dist2(a, b)); }

  // ---------- synthetic dataset generators ----------
  function makeBlobs({ n = 90, clusters = 3, spread = 0.9, xr = [-8, 8], yr = [-8, 8] } = {}) {
    const centers = Array.from({ length: clusters }, () => [randRange(xr[0], xr[1]), randRange(yr[0], yr[1])]);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const c = centers[i % clusters];
      pts.push({ x: c[0] + randn() * spread, y: c[1] + randn() * spread, label: i % clusters });
    }
    return shuffle(pts);
  }
  function makeLinear({ n = 40, slope = 1.4, intercept = -2, noise = 2.2, xr = [-8, 8] } = {}) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const x = randRange(xr[0], xr[1]);
      pts.push({ x, y: slope * x + intercept + randn() * noise });
    }
    return pts;
  }
  function makeTwoClass({ n = 80, mode = "linear" } = {}) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      if (mode === "circles") {
        const label = i % 2;
        const r = label === 0 ? randRange(0, 3) : randRange(5, 8);
        const a = randRange(0, Math.PI * 2);
        pts.push({ x: r * Math.cos(a) + randn() * 0.5, y: r * Math.sin(a) + randn() * 0.5, label });
      } else if (mode === "xor") {
        const x = randRange(-8, 8), y = randRange(-8, 8);
        pts.push({ x, y, label: (x > 0) === (y > 0) ? 0 : 1 });
      } else {
        const label = i % 2;
        const cx = label === 0 ? -3 : 3;
        pts.push({ x: cx + randn() * 2.4, y: randn() * 2.4, label });
      }
    }
    return shuffle(pts);
  }

  // ---------- SVG helpers ----------
  const NS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs = {}) {
    const el = document.createElementNS(NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }
  function clearNode(el) { while (el.firstChild) el.removeChild(el.firstChild); }

  const palette = ["#7dd3fc", "#fca5a5", "#86efac", "#fcd34d", "#c4b5fd", "#fda4af", "#93c5fd"];

  function makeScales(width, height, pad, domain) {
    const [x0, x1, y0, y1] = domain;
    const sx = (x) => pad + ((x - x0) / (x1 - x0)) * (width - 2 * pad);
    const sy = (y) => height - pad - ((y - y0) / (y1 - y0)) * (height - 2 * pad);
    const isx = (px) => x0 + ((px - pad) / (width - 2 * pad)) * (x1 - x0);
    const isy = (py) => y0 + ((height - pad - py) / (height - 2 * pad)) * (y1 - y0);
    return { sx, sy, isx, isy };
  }

  // Small static function plot used by the per-algorithm "Reference" tab
  // (loss curves, activation shapes, impurity curves, etc.)
  function plotFn(container, { fn, domain, yDomain, width = 280, height = 130, color = "var(--accent)", fn2, color2 } = {}) {
    if (!container) return;
    clearNode(container);
    const pad = 20;
    const [x0, x1] = domain;
    let [y0, y1] = yDomain || (() => {
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i <= 100; i++) {
        const v = fn(x0 + (i / 100) * (x1 - x0));
        lo = Math.min(lo, v); hi = Math.max(hi, v);
      }
      return [lo, hi];
    })();
    if (y1 - y0 < 1e-6) y1 = y0 + 1;
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, style: "display:block" });
    const sx = (v) => pad + ((v - x0) / (x1 - x0)) * (width - 2 * pad);
    const sy = (v) => height - pad - ((v - y0) / (y1 - y0)) * (height - 2 * pad);
    svg.appendChild(svgEl("rect", { x: pad, y: pad, width: width - 2 * pad, height: height - 2 * pad, fill: "var(--bg-inset)", stroke: "var(--border-soft)" }));
    if (y0 < 0 && y1 > 0) svg.appendChild(svgEl("line", { x1: pad, x2: width - pad, y1: sy(0), y2: sy(0), stroke: "var(--border)", "stroke-dasharray": "3,3" }));
    if (x0 < 0 && x1 > 0) svg.appendChild(svgEl("line", { x1: sx(0), x2: sx(0), y1: pad, y2: height - pad, stroke: "var(--border)", "stroke-dasharray": "3,3" }));
    function pathFor(f) {
      let d = "";
      for (let i = 0; i <= 100; i++) {
        const xv = x0 + (i / 100) * (x1 - x0);
        const yv = Math.max(y0, Math.min(y1, f(xv)));
        d += (i === 0 ? "M" : "L") + sx(xv).toFixed(1) + " " + sy(yv).toFixed(1) + " ";
      }
      return d;
    }
    svg.appendChild(svgEl("path", { d: pathFor(fn), fill: "none", stroke: color, "stroke-width": 2 }));
    if (fn2) svg.appendChild(svgEl("path", { d: pathFor(fn2), fill: "none", stroke: color2 || "var(--text-faint)", "stroke-width": 2, "stroke-dasharray": "4,3" }));
    container.appendChild(svg);
  }

  function drawAxes(svg, width, height, pad, domain) {
    const { sx, sy } = makeScales(width, height, pad, domain);
    const g = svgEl("g");
    const zx = sx(0), zy = sy(0);
    if (zx >= pad && zx <= width - pad) {
      g.appendChild(svgEl("line", { x1: zx, y1: pad, x2: zx, y2: height - pad, stroke: "var(--border)", "stroke-dasharray": "3,3" }));
    }
    if (zy >= pad && zy <= height - pad) {
      g.appendChild(svgEl("line", { x1: pad, y1: zy, x2: width - pad, y2: zy, stroke: "var(--border)", "stroke-dasharray": "3,3" }));
    }
    g.appendChild(svgEl("rect", { x: pad, y: pad, width: width - 2 * pad, height: height - 2 * pad, fill: "none", stroke: "var(--border-soft)" }));
    svg.appendChild(g);
    return { sx, sy };
  }

  return {
    rng, seedRng, randRange, randn, randInt, shuffle,
    zeros, matMul, transpose, mean, std, meanVec, covMatrix, jacobiEigen, inverse2x2, det2x2, inverseN, solveLinearSystem,
    dist, dist2, makeBlobs, makeLinear, makeTwoClass,
    svgEl, clearNode, palette, makeScales, drawAxes, plotFn, NS,
  };
})();
