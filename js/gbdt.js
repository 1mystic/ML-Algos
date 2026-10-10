// Shared second-order gradient-boosted tree core used by the XGBoost and
// LightGBM labs: logistic loss gradients, regularised split gain, depth-wise
// (XGBoost) and leaf-wise (LightGBM) growth, and histogram binning.
const GBDT = (() => {
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  function gradHess(pts, F) {
    const g = [], h = [];
    for (let i = 0; i < pts.length; i++) { const p = sigmoid(F[i]); g.push(p - pts[i].label); h.push(Math.max(p * (1 - p), 1e-6)); }
    return { g, h };
  }
  const score = (G, H, lam) => (G * G) / (H + lam);
  const gain = (GL, HL, GR, HR, lam, gamma) => 0.5 * (score(GL, HL, lam) + score(GR, HR, lam) - score(GL + GR, HL + HR, lam)) - gamma;
  const leafW = (G, H, lam) => -G / (H + lam);

  // Quantile bin edges per feature (LightGBM-style histograms).
  function makeBins(pts, maxBin) {
    const out = {};
    for (const f of ["x", "y"]) {
      const v = pts.map((p) => p[f]).sort((a, b) => a - b);
      const edges = [];
      for (let b = 1; b < maxBin; b++) { const e = v[Math.floor((b / maxBin) * v.length)]; if (e != null && (!edges.length || e > edges[edges.length - 1] + 1e-9)) edges.push(e); }
      out[f] = edges;
    }
    return out;
  }
  const binOf = (edges, v) => { let lo = 0, hi = edges.length; while (lo < hi) { const m = (lo + hi) >> 1; if (v < edges[m]) hi = m; else lo = m + 1; } return lo; };

  function bestSplit(pts, idx, g, h, w, P, bins) {
    let G = 0, H = 0;
    for (const i of idx) { G += g[i] * w[i]; H += h[i] * w[i]; }
    let best = null;
    for (const f of ["x", "y"]) {
      if (bins) {
        const E = bins[f], nb = E.length + 1;
        const hg = new Float64Array(nb), hh = new Float64Array(nb);
        for (const i of idx) { const b = binOf(E, pts[i][f]); hg[b] += g[i] * w[i]; hh[b] += h[i] * w[i]; }
        let GL = 0, HL = 0;
        for (let b = 0; b < nb - 1; b++) {
          GL += hg[b]; HL += hh[b];
          const GR = G - GL, HR = H - HL;
          if (HL < P.minH || HR < P.minH) continue;
          const gn = gain(GL, HL, GR, HR, P.lam, P.gamma);
          if (!best || gn > best.gain) best = { f, thr: E[b] - 1e-9, gain: gn, GL, HL, GR, HR };
        }
      } else {
        const s = idx.slice().sort((a, b) => pts[a][f] - pts[b][f]);
        let GL = 0, HL = 0;
        for (let k = 0; k < s.length - 1; k++) {
          GL += g[s[k]] * w[s[k]]; HL += h[s[k]] * w[s[k]];
          if (pts[s[k + 1]][f] - pts[s[k]][f] < 1e-9) continue;
          const GR = G - GL, HR = H - HL;
          if (HL < P.minH || HR < P.minH) continue;
          const gn = gain(GL, HL, GR, HR, P.lam, P.gamma);
          if (!best || gn > best.gain) best = { f, thr: (pts[s[k]][f] + pts[s[k + 1]][f]) / 2, gain: gn, GL, HL, GR, HR };
        }
      }
    }
    return { G, H, best };
  }

  function makeNode(nodes, pts, idx, g, h, w, P, bounds, depth, parent) {
    const nd = { id: nodes.length, idx, bounds, depth, parent, leaf: true };
    const r = bestSplit(pts, idx, g, h, w, P, P.bins);
    nd.G = r.G; nd.H = r.H; nd.w = leafW(r.G, r.H, P.lam); nd.cand = r.best;
    nodes.push(nd);
    return nd;
  }
  function applySplit(nodes, pts, nd, g, h, w, P) {
    const s = nd.cand;
    const [x0, x1, y0, y1] = nd.bounds;
    const L = nd.idx.filter((i) => pts[i][s.f] <= s.thr), R = nd.idx.filter((i) => pts[i][s.f] > s.thr);
    Object.assign(nd, { leaf: false, f: s.f, thr: s.thr, gain: s.gain });
    nd.left = makeNode(nodes, pts, L, g, h, w, P, s.f === "x" ? [x0, s.thr, y0, y1] : [x0, x1, y0, s.thr], nd.depth + 1, nd.id).id;
    nd.right = makeNode(nodes, pts, R, g, h, w, P, s.f === "x" ? [s.thr, x1, y0, y1] : [x0, x1, s.thr, y1], nd.depth + 1, nd.id).id;
  }

  // XGBoost: grow level by level to max_depth, then prune splits whose gain
  // (already net of gamma) is negative, bottom-up.
  function growDepthwise(pts, idx, g, h, w, P) {
    const nodes = [];
    makeNode(nodes, pts, idx, g, h, w, P, [-10, 10, -10, 10], 0, null);
    const queue = [0];
    while (queue.length) {
      const nd = nodes[queue.shift()];
      if (nd.depth >= P.maxDepth || !nd.cand || nd.idx.length < 2) continue;
      if (nd.cand.gain + P.gamma <= 1e-12) continue;
      applySplit(nodes, pts, nd, g, h, w, P);
      queue.push(nd.left, nd.right);
    }
    const pruned = [];
    for (let k = nodes.length - 1; k >= 0; k--) {
      const nd = nodes[k];
      if (nd.leaf) continue;
      if (nodes[nd.left].leaf && nodes[nd.right].leaf && nd.gain < 0) { pruned.push({ ...nd }); nd.leaf = true; nd.pruned = true; }
    }
    return { nodes, pruned };
  }

  // LightGBM: always split the leaf with the largest gain, up to num_leaves.
  function growLeafwise(pts, idx, g, h, w, P, record) {
    const nodes = [];
    makeNode(nodes, pts, idx, g, h, w, P, [-10, 10, -10, 10], 0, null);
    let nLeaves = 1;
    while (nLeaves < P.numLeaves) {
      const frontier = nodes.filter((n) => n.leaf && n.cand && n.cand.gain > 0 && n.idx.length >= 2 && n.depth < (P.maxDepth || 99));
      if (!frontier.length) break;
      frontier.sort((a, b) => b.cand.gain - a.cand.gain);
      const pick = frontier[0];
      if (record) record(nodes, frontier, pick);
      applySplit(nodes, pts, pick, g, h, w, P);
      nLeaves++;
    }
    return { nodes };
  }

  function leafFor(nodes, p) { let nd = nodes[0]; while (!nd.leaf) nd = nodes[p[nd.f] <= nd.thr ? nd.left : nd.right]; return nd; }
  function logloss(pts, F) { let s = 0; for (let i = 0; i < pts.length; i++) { const p = sigmoid(F[i]); s -= pts[i].label ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12); } return pts.length ? s / pts.length : 0; }
  function accuracy(pts, F) { return pts.length ? pts.reduce((s, p, i) => s + ((F[i] >= 0 ? 1 : 0) === p.label), 0) / pts.length : 0; }

  return { sigmoid, gradHess, gain, leafW, makeBins, binOf, growDepthwise, growLeafwise, leafFor, logloss, accuracy };
})();
