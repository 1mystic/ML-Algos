(() => {
  const { ui, col, fmt } = Lab;
  const H = (i) => Math.log(i) + 0.5772156649;
  const c = (n) => (n > 2 ? 2 * H(n - 1) - (2 * (n - 1)) / n : n === 2 ? 1 : 0);

  function iTree(pts, idx, depth, maxDepth, bounds, rec) {
    if (idx.length <= 1 || depth >= maxDepth) return { leaf: true, size: idx.length, depth };
    const f = MLU.rng() < 0.5 ? "x" : "y";
    let lo = Infinity, hi = -Infinity;
    for (const i of idx) { lo = Math.min(lo, pts[i][f]); hi = Math.max(hi, pts[i][f]); }
    if (hi - lo < 1e-9) return { leaf: true, size: idx.length, depth };
    const thr = lo + MLU.rng() * (hi - lo);
    const L = idx.filter((i) => pts[i][f] < thr), R = idx.filter((i) => pts[i][f] >= thr);
    const [x0, x1, y0, y1] = bounds;
    const bL = f === "x" ? [x0, thr, y0, y1] : [x0, x1, y0, thr], bR = f === "x" ? [thr, x1, y0, y1] : [x0, x1, thr, y1];
    if (rec) rec({ f, thr, bounds, idx, L, R, bL, bR, depth });
    return { leaf: false, f, thr, left: iTree(pts, L, depth + 1, maxDepth, bL, rec && rec.follow(L) ? rec : null), right: iTree(pts, R, depth + 1, maxDepth, bR, rec && rec.follow(R) ? rec : null) };
  }
  function pathLen(t, p) { let d = 0; while (!t.leaf) { t = p[t.f] < t.thr ? t.left : t.right; d++; } return d + c(t.size); }
  const score = (Eh, psi) => Math.pow(2, -Eh / c(psi));

  function forest(pts, T, psi) {
    const trees = [], maxD = Math.ceil(Math.log2(Math.max(2, psi)));
    for (let t = 0; t < T; t++) {
      const idx = MLU.shuffle([...pts.keys()]).slice(0, Math.min(psi, pts.length));
      trees.push(iTree(pts, idx, 0, maxD, [-10, 10, -10, 10], null));
    }
    return trees;
  }

  function isolationFrames(pts, target, who) {
    const frames = [];
    const steps = [];
    const rec = (s) => steps.push(s);
    rec.follow = (idx) => idx.includes(target);
    iTree(pts, [...pts.keys()], 0, 50, [-10, 10, -10, 10], rec);
    const path = steps.filter((s) => s.idx.includes(target));
    path.forEach((s, k) => frames.push({ phase: "iso", arch: ["sub", "split", "rec"], title: `Isolating the ${who} point: split ${k + 1} on ${s.f} at ${s.thr.toFixed(2)}`, text: `A random feature and a random threshold between the min and max of the remaining points. The ${who} point keeps the side it falls on; <b>${(s.L.includes(target) ? s.L : s.R).length}</b> point${(s.L.includes(target) ? s.L : s.R).length === 1 ? "" : "s"} remain with it.`, tex: "f\\sim\\mathcal U\\{1..d\\},\\quad t\\sim\\mathcal U\\big[\\min x_f,\\;\\max x_f\\big]", st: { box: s.L.includes(target) ? s.bL : s.bR, prevBox: s.bounds, split: s, target, keep: s.L.includes(target) ? s.L : s.R } }));
    const depth = path.length;
    frames.push({ phase: "iso", major: true, arch: ["rec", "h"], title: `The ${who} point was isolated after ${depth} split${depth === 1 ? "" : "s"}`, text: who === "anomalous" ? "Points far from the crowd get separated by the first few random cuts, so their path is short." : "A point inside a dense cluster needs many cuts before nothing else shares its box, so its path is long.", tex: "h(x)=\\text{number of edges from the root to the leaf containing }x", st: { box: path.length ? (path[path.length - 1].L.includes(target) ? path[path.length - 1].bL : path[path.length - 1].bR) : [-10, 10, -10, 10], target, keep: [target], depth } });
    return { frames, depth };
  }

  function buildFrames(pts, T, psi, target, normal) {
    if (pts.length < 4) return { frames: [{ phase: "idle", title: "Add more points", text: "", st: {} }], trees: [] };
    const a = isolationFrames(pts, target, "anomalous");
    const b = isolationFrames(pts, normal, "typical");
    const frames = [{ phase: "intro", major: true, arch: ["sub"], title: "Anomalies are few and different, so they are easy to isolate", text: "Isolation Forest never models what normal data looks like. It just counts how many random axis-aligned cuts it takes to separate a point from everyone else.", st: { target, normal } }, ...a.frames, ...b.frames];
    const trees = forest(pts, T, psi);
    const show = new Set(Lab.checkpoints(T, 30));
    for (let t = 1; t <= T; t++) {
      if (!show.has(t)) continue;
      const Eh = (p) => trees.slice(0, t).reduce((s, tr) => s + pathLen(tr, p), 0) / t;
      frames.push({ phase: "forest", major: t === 1, arch: ["sub", "h", "avg", "score"], title: `Forest of ${t} tree${t === 1 ? "" : "s"}: average path length → anomaly score`, text: `Each tree uses a random subsample of ψ = ${Math.min(psi, pts.length)} points. Averaging path lengths over trees removes the luck of individual cuts. Selected point: E[h] = <b>${Eh(pts[target]).toFixed(2)}</b>, score <b>${score(Eh(pts[target]), Math.min(psi, pts.length)).toFixed(3)}</b>; typical point: E[h] = <b>${Eh(pts[normal]).toFixed(2)}</b>.`, tex: "s(x)=2^{-\\,E[h(x)]/c(\\psi)},\\qquad c(\\psi)=2H(\\psi-1)-\\tfrac{2(\\psi-1)}{\\psi}", st: { T: t, target, normal, field: true }, m: { anom: Eh(pts[target]), norm: Eh(pts[normal]) } });
    }
    frames.push({ ...frames[frames.length - 1], phase: "done", major: true, arch: ["score", "out"], title: "Threshold the score", text: "Scores near 1 mean very short paths (anomalies); around 0.5 or below means normal. The contamination setting picks the threshold so that fraction of points is flagged (ringed)." });
    return { frames, trees };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "iforest", fps: 2.5,
      legend: ui.legend([["var(--series-2)", "flagged anomaly", "ring"], ["var(--series-4)", "current box", "ring"], ["var(--accent)", "selected point ◆"]]),
      tools: ui.btn("replay", "replay", true) + ui.btn("regrow", "new random trees"),
      hint: "click a point to make it the one being isolated · click empty space to add · drag / double-click to edit",
      data: { kind: "cluster", presets: ["outliers", "blobs3", "varied", "moons", "uniform"], def: "outliers", n: 160, noise: 0.25 },
      side: [{ title: "forest", html: ui.range("T", "trees", 1, 150, 1, 100) + ui.range("psi", "subsample ψ", 16, 256, 16, 128) + ui.range("cont", "contamination (flag top %)", 1, 30, 1, 8) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "sub", label: "subsample ψ", sub: "per tree", x: 58, y: 40, kind: "param" },
          { id: "split", label: "random split", sub: "feature + threshold", x: 190, y: 40, kind: "op" },
          { id: "rec", label: "recurse", sub: "until isolated", x: 318, y: 40, kind: "op" },
          { id: "h", label: "path length h(x)", sub: "+ c(leaf size)", x: 300, y: 112, kind: "op" },
          { id: "avg", label: "E[h(x)]", sub: "mean over trees", x: 170, y: 112, kind: "op" },
          { id: "score", label: "s = 2^(−E[h]/c(ψ))", sub: "anomaly score", x: 170, y: 168, kind: "loss" },
          { id: "out", label: "flag", sub: "top contamination %", x: 322, y: 168, kind: "out" },
        ],
        edges: [{ from: "sub", to: "split" }, { from: "split", to: "rec" }, { from: "rec", to: "split", bend: -24, dash: true }, { from: "rec", to: "h" }, { from: "h", to: "avg" }, { from: "avg", to: "score" }, { from: "score", to: "out" }],
      },
      phases: { intro: { label: "idea", color: 3 }, iso: { label: "isolate", color: 4 }, forest: { label: "forest", color: 0 }, done: { label: "flag", color: 1 } },
      charts: [{ id: "h", title: "average path length as trees are added", series: [{ key: "anom", label: "selected point", color: "var(--series-2)" }, { key: "norm", label: "typical point", color: "var(--series-1)", dash: "4 3" }] }],
      panels: [{ id: "hist", title: "score distribution", sub: "dashed = threshold" }],
      onFrame: render, onData: () => { target = null; replay(); },
    });
    lab.editable({ newPoint: (x, y) => ({ x, y, label: 0 }), onChange: (kind) => { if (kind === "remove") target = null; replay({ at: "end" }); } });
    const pl = lab.plane, L = pl.layers;
    let target = null, normal = 0, trees = [], seed = 5;
    const psi = () => Math.min(lab.val("psi"), lab.points.length);
    function pickDefaults() {
      MLU.seedRng(1234);
      const tr = forest(lab.points, 60, psi());
      const s = lab.points.map((p) => tr.reduce((a, t) => a + pathLen(t, p), 0) / tr.length);
      if (target == null || target >= lab.points.length) target = s.indexOf(Math.min(...s));
      normal = s.indexOf(Math.max(...s));
    }
    function replay({ at = "start" } = {}) {
      if (lab.points.length < 4) { lab.load([{ phase: "idle", title: "Add more points", text: "", st: {} }]); return; }
      pickDefaults();
      MLU.seedRng(seed);
      const r = buildFrames(lab.points, lab.val("T"), psi(), target, normal);
      trees = r.trees;
      lab.load(r.frames, { at, play: at === "start" });
    }
    const boxG = L.mid.append("g"), flagG = L.top.append("g"), tgG = L.top.append("g");
    const histEl = lab.R("panel-hist");
    function render(f) {
      const st = f.st, pts = lab.points;
      boxG.selectAll("*").remove(); flagG.selectAll("*").remove(); tgG.selectAll("*").remove();
      if (st.target == null) { lab.drawPoints({}); return; }
      const ps = psi();
      let scores = null, thr = null;
      if (st.field) {
        const use = trees.slice(0, st.T);
        const sc = (p) => score(use.reduce((a, t) => a + pathLen(t, p), 0) / use.length, ps);
        Lab.field(L.bg, pl, (x, y) => sc({ x, y }), (v) => { const P = Lab.palRGB(); const t = Math.max(0, Math.min(1, (v - 0.35) / 0.4)); const a = P[0], b = P[1]; return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, 0.12 + 0.5 * t]; }, { res: 70 });
        scores = pts.map(sc);
        const sorted = scores.slice().sort((a, b) => b - a);
        thr = sorted[Math.max(0, Math.round((lab.val("cont") / 100) * pts.length) - 1)];
        scores.forEach((s, i) => { if (s >= thr) flagG.append("circle").attr("cx", pl.x(pts[i].x)).attr("cy", pl.y(pts[i].y)).attr("r", 8.5).attr("fill", "none").attr("stroke", "var(--series-2)").attr("stroke-width", 2).attr("pointer-events", "none"); });
      } else L.bg.selectAll("*").remove();
      if (st.box) {
        if (st.split) {
          const s = st.split, [x0, x1, y0, y1] = s.bounds;
          const ln = s.f === "x" ? [s.thr, y0, s.thr, y1] : [x0, s.thr, x1, s.thr];
          boxG.append("rect").attr("x", pl.x(x0)).attr("y", pl.y(y1)).attr("width", pl.x(x1) - pl.x(x0)).attr("height", pl.y(y0) - pl.y(y1)).attr("fill", "none").attr("stroke", "var(--text-faint)").attr("stroke-dasharray", "3 3");
          boxG.append("line").attr("x1", pl.x(ln[0])).attr("y1", pl.y(ln[1])).attr("x2", pl.x(ln[2])).attr("y2", pl.y(ln[3])).attr("stroke", "var(--text)").attr("stroke-width", 2);
        }
        const [x0, x1, y0, y1] = st.box;
        boxG.append("rect").attr("x", pl.x(x0)).attr("y", pl.y(y1)).attr("width", Math.max(1, pl.x(x1) - pl.x(x0))).attr("height", Math.max(1, pl.y(y0) - pl.y(y1))).attr("fill", "var(--series-4)").attr("fill-opacity", 0.1).attr("stroke", "var(--series-4)").attr("stroke-width", 2.4);
      }
      const keep = st.keep ? new Set(st.keep) : null;
      const sel = lab.drawPoints({ color: (d, i) => (scores ? (scores[i] >= thr ? col(1) : col(0)) : "var(--series-1)"), opacity: (d, i) => (keep && !keep.has(i) ? 0.25 : 0.9), r: 4 });
      sel.on("click.sel", (ev, d) => { ev.stopPropagation(); target = pts.indexOf(d); replay(); });
      for (const [i, c2] of [[st.target, "var(--accent)"], [st.normal != null ? st.normal : null, "var(--text-dim)"]]) {
        if (i == null) continue;
        tgG.append("rect").attr("x", -7).attr("y", -7).attr("width", 14).attr("height", 14).attr("transform", `translate(${pl.x(pts[i].x)},${pl.y(pts[i].y)}) rotate(45)`).attr("fill", c2).attr("stroke", "var(--bg)").attr("stroke-width", 1.5).attr("pointer-events", "none");
      }
      histEl.innerHTML = "";
      if (scores) {
        const P = Lab.xy(histEl, { w: 360, h: 130, xd: [0.3, 0.9], yd: [0, 1], xLabel: "anomaly score" });
        const bins = d3.bin().domain([0.3, 0.9]).thresholds(24)(scores.map((s) => Math.min(0.899, Math.max(0.3, s))));
        const ym = Math.max(...bins.map((b) => b.length), 1);
        P.domain(null, [0, ym]);
        P.g.selectAll("rect").data(bins).enter().append("rect").attr("x", (b) => P.x(b.x0) + 0.5).attr("width", (b) => Math.max(0, P.x(b.x1) - P.x(b.x0) - 1)).attr("y", (b) => P.y(b.length)).attr("height", (b) => P.y(0) - P.y(b.length)).attr("fill", (b) => (b.x0 >= thr ? col(1) : col(0))).attr("opacity", 0.75);
        P.top.append("line").attr("x1", P.x(thr)).attr("x2", P.x(thr)).attr("y1", P.m.t).attr("y2", P.h - P.m.b).attr("stroke", "var(--text)").attr("stroke-dasharray", "4 3");
      } else histEl.innerHTML = `<div class="lab-card-note">Scores appear once the forest is built.</div>`;
      lab.readout(`<div class="kv"><span>points</span><b>${pts.length}</b><span>ψ · c(ψ)</span><b>${ps} · ${c(ps).toFixed(2)}</b><span>flagged</span><b>${scores ? scores.filter((s) => s >= thr).length : "–"}</b><span>selected score</span><b>${scores ? scores[st.target].toFixed(3) : st.depth != null ? "depth " + st.depth : "–"}</b></div>`);
    }
    lab.watch(["T", "psi"], () => replay());
    lab.watch(["cont"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.q('[data-b="regrow"]').onclick = () => { seed = (seed * 17 + 3) % 9973; replay(); };
    lab.loadPreset("outliers");
    return lab.destroy;
  }

  MLApp.register({
    id: "isolation-forest",
    name: "Isolation Forest",
    category: "Unsupervised - Anomaly & Association",
    tagline: "anomalies have short random paths",
    isNew: true,
    description: "Detect outliers without modelling normal data: random axis-aligned cuts isolate rare, distant points in just a few splits. Watch one outlier and one typical point get isolated side by side, then see path lengths averaged over a forest into an anomaly-score map.",
    info: {
      type: "Unsupervised - Anomaly detection. Ensemble of random isolation trees; non-parametric.",
      scenario: "Fraud, intrusion and defect detection on tabular data where anomalies are rare and labels are scarce or absent.",
      inputs: "Unlabeled feature vectors; number of trees, subsample size ψ, and an expected contamination rate.",
      output: "An anomaly score in (0, 1] per point, and binary flags at a chosen threshold.",
      intuition: {
        definition: "Pick a random feature and a random cut, and keep cutting the side that contains your point. An outlier sits alone in empty space, so it gets cut off almost immediately. A normal point is surrounded by neighbours and needs many cuts. Average the number of cuts over many random trees and you have an anomaly score.",
        steps: ["Draw a random subsample of ψ points for each tree.", "Split recursively on random features at random thresholds until each point is alone (or a depth limit).", "For each point, record its path length in every tree.", "Convert the average path length into a score; flag the highest scores."],
        applications: ["Credit card and payment fraud detection", "Network intrusion and log anomaly detection", "Manufacturing sensor fault detection", "Data cleaning: finding corrupted rows"],
      },
      math: [
        { title: "Path length", formula: "h(x) = depth of x's leaf + c(leaf size)", note: "The c(·) term accounts for points left unsplit when the depth limit stops the tree." },
        { title: "Average path of an unsuccessful BST search", formula: "c(n) = 2H(n−1) − 2(n−1)/n,  H(i) ≈ ln i + 0.5772", note: "The expected path length for n points; used to normalise h." },
        { title: "Anomaly score", formula: "s(x, ψ) = 2^(−E[h(x)] / c(ψ))", note: "s → 1 when E[h] → 0 (anomaly), s = 0.5 when E[h] = c(ψ) (average), s → 0 for very deep points." },
        { title: "Depth limit", formula: "ℓ = ⌈log₂ ψ⌉", note: "Anomalies are found high in the tree; growing deeper only refines normal points, so trees stop early." },
      ],
      pipeline: [{ label: "Subsample ψ" }, { label: "Random splits", note: "until isolated" }, { label: "Path length h(x)" }, { label: "Average over trees" }, { label: "Score & threshold", accent: "red" }],
      decisionFunction: { text: "anomaly(x) = s(x, ψ) ≥ τ   where τ is set by the contamination rate", mechanism: "Short average paths give high scores. The threshold is usually the score quantile matching the expected anomaly fraction.", plot: { fn: (h) => Math.pow(2, -h / c(256)), domain: [0, 20], caption: "score vs average path length for ψ = 256 (c(ψ) ≈ 10.2): s = 0.5 exactly at the average depth" } },
      lossFunction: { text: "No loss is optimised: trees are completely random", mechanism: "Isolation Forest is not trained toward any objective. Its power comes from the statistical fact that random partitions isolate sparse regions faster than dense ones." },
      optimization: [{ title: "Training cost", formula: "O(T · ψ log ψ)", note: "Independent of n beyond sampling, so it scales to huge datasets." }, { title: "Scoring cost", formula: "O(T · log ψ) per point", note: "Tiny; suitable for streaming." }, { title: "Extended Isolation Forest", formula: "random hyperplane cuts", note: "Removes the axis-aligned artefacts in the score map." }],
      assumptions: [{ name: "Anomalies are few", why: "If they form a dense group they mask each other.", check: "Keep ψ small (256) to reduce masking and swamping." }, { name: "Anomalies are different in feature space", why: "Isolation is about being far from others.", check: "Engineer features where anomalies stand out." }],
      hyperparameters: [{ name: "n_estimators", range: "50 - 500", increasing: "More stable scores.", strategy: "100 is usually enough; check score stability." }, { name: "max_samples ψ", range: "64 - 512", increasing: "Deeper trees, more masking of clustered anomalies.", strategy: "256 is the paper's default and works broadly." }, { name: "contamination", range: "0.001 - 0.2", increasing: "More points flagged.", strategy: "Set from domain knowledge; or rank by score instead of thresholding." }, { name: "max_features", range: "0.5 - 1.0", increasing: "Uses more features per tree.", strategy: "Lower for many irrelevant features." }],
      metrics: ["ROC-AUC / PR-AUC (if some labels exist)", "Precision at k", "Score distribution shape"],
      typicalUses: ["Fraud and intrusion detection", "Sensor fault detection", "Outlier removal before modelling"],
      diagnostics: ["Plot the score histogram: a clean right tail means separable anomalies.", "Inspect top-scored points manually.", "Vary ψ: stable rankings mean robust detection."],
      advantages: ["Linear-time training, tiny memory.", "No distance computations, works in high dimensions.", "No assumptions about the distribution of normal data.", "Few hyperparameters."],
      limitations: [{ name: "Axis-aligned artefacts", note: "score maps show cross-shaped ghosts", fix: "Extended Isolation Forest." }, { name: "Local anomalies", note: "points anomalous only relative to a nearby dense cluster can be missed", fix: "Local Outlier Factor." }, { name: "Masking", note: "clusters of anomalies look normal", fix: "smaller ψ." }],
      alternatives: [{ name: "Local Outlier Factor", when: "Anomalies are defined relative to local density." }, { name: "One-class SVM", when: "Smaller data and a smooth boundary of normality." }, { name: "Autoencoder reconstruction error", when: "High-dimensional structured data (images, sequences)." }],
      pitfalls: [{ problem: "Too many points flagged", solution: "Contamination set too high; use score ranking instead." }, { problem: "Obvious clustered outliers not flagged", solution: "Masking; reduce ψ." }, { problem: "Scores driven by one noisy feature", solution: "Feature selection or scaling checks." }],
      quickRef: [{ name: "Score", formula: "s = 2^(−E[h]/c(ψ))" }, { name: "c(n)", formula: "2H(n−1) − 2(n−1)/n" }, { name: "Depth limit", formula: "⌈log₂ ψ⌉" }, { name: "Cost", formula: "O(T ψ log ψ)" }],
      code: `from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=200, max_samples=256,
                      contamination=0.02, random_state=0).fit(X)
scores = -iso.score_samples(X)        # higher = more anomalous
flags = iso.predict(X) == -1`,
      whyChain: [
        { q: "Why do anomalies get short paths?", a: "A uniformly random threshold is likely to fall in the large empty gap between an isolated point and the rest of the data, separating it in one cut. A point in a dense cluster can only be separated by thresholds landing in the tiny gaps around it." },
        { q: "Why subsample instead of using all data?", a: "Small subsamples reduce swamping (normal points near anomalies) and masking (dense groups of anomalies), and make training cost independent of n." },
        { q: "Why normalise by c(ψ)?", a: "Path lengths grow with sample size. Dividing by the expected path length of a random binary search tree of ψ points makes scores comparable across ψ and gives the 0.5 reference." },
      ],
      workedExample: {
        setup: "ψ = 256, so c(ψ) ≈ 10.24. Point A averages 4.1 splits, point B averages 11.5.",
        steps: ["s(A) = 2^(−4.1/10.24) = 2^(−0.40) ≈ 0.76.", "s(B) = 2^(−11.5/10.24) = 2^(−1.12) ≈ 0.46.", "With threshold 0.6, A is flagged and B is normal."],
        result: "A is an anomaly (0.76), B is normal (0.46)",
      },
    },
    mount,
  });
})();
