(() => {
  const { ui, col, fmt } = Lab;

  function buildFrames(pts, P) {
    const n = pts.length;
    if (!pts.some((p) => p.label === 0) || !pts.some((p) => p.label === 1)) return [{ phase: "idle", title: "Need both classes", text: "", st: { trees: [], F: [] } }];
    const bins = GBDT.makeBins(pts, P.maxBin);
    const base = Math.log((pts.filter((p) => p.label).length + 0.5) / (pts.filter((p) => !p.label).length + 0.5));
    const F = new Array(n).fill(base);
    const trees = [];
    const frames = [{ phase: "bin", major: true, arch: ["bin"], title: `Histogram binning: each feature bucketed into ≤ ${P.maxBin} bins`, text: `Before any tree is grown, every feature value is replaced by its quantile bin (grid lines). Split search then scans ${bins.x.length + 1} + ${bins.y.length + 1} bin boundaries instead of ${n - 1} × 2 sorted values, using per-bin sums of g and h. This is the main reason LightGBM is fast.`, tex: "\\text{cost per split: }O(\\#\\text{bins})\\text{ instead of }O(n)", st: { trees: [], F: F.slice(), bins }, m: { loss: GBDT.logloss(pts, F), acc: GBDT.accuracy(pts, F) } }];
    for (let t = 1; t <= P.rounds; t++) {
      const { g, h } = GBDT.gradHess(pts, F);
      // GOSS: keep the top-a fraction by |g|, sample b of the rest and up-weight them
      let idx = pts.map((_, i) => i), w = new Array(n).fill(1), goss = null;
      if (P.goss) {
        const order = idx.slice().sort((a, b) => Math.abs(g[b]) - Math.abs(g[a]));
        const nTop = Math.max(1, Math.round(0.2 * n)), nRand = Math.max(1, Math.round(0.15 * n));
        const top = order.slice(0, nTop), rest = MLU.shuffle(order.slice(nTop)).slice(0, nRand);
        idx = top.concat(rest);
        for (const i of rest) w[i] = (1 - 0.2) / 0.15;
        goss = { top: new Set(top), rand: new Set(rest) };
        if (t <= 2) frames.push({ phase: "goss", arch: ["gh", "goss"], title: `Round ${t}: GOSS keeps ${nTop} large-gradient points + ${nRand} random others`, text: "Gradient-based One-Side Sampling: points with small gradients are already well fit, so most of them are dropped. The random survivors are up-weighted by (1 − a)/b to keep the gradient sums unbiased. Faded points are skipped this round.", tex: "\\tilde G=\\sum_{i\\in A}g_i+\\frac{1-a}{b}\\sum_{i\\in B}g_i", st: { trees: trees.slice(), F: F.slice(), bins, goss, g }, m: { loss: GBDT.logloss(pts, F), acc: GBDT.accuracy(pts, F) } });
      }
      const detail = t <= 2;
      const tree = GBDT.growLeafwise(pts, idx, g, h, w, { numLeaves: P.numLeaves, lam: P.lam, gamma: 0, minH: 0.5, bins }, detail ? (nodes, frontier, pick) => {
        frames.push({ phase: "leaf", arch: ["hist", "pick", "split"], title: `Round ${t}: split the leaf with the largest gain (${pick.cand.gain.toFixed(3)})`, text: `Leaf-wise growth: of the ${frontier.length} current leaves, only the most promising one is split, wherever it is in the tree. Level-wise (XGBoost default) would split every leaf at this depth. Gains are listed in the leaf queue.`, tex: "\\ell^\\star=\\arg\\max_{\\ell\\in\\text{leaves}}\\text{Gain}(\\ell)", st: { trees: trees.slice(), F: F.slice(), bins, goss, grow: { nodes: nodes.map((nd) => ({ ...nd })), frontier: frontier.map((nd) => ({ id: nd.id, gain: nd.cand.gain, f: nd.cand.f, thr: nd.cand.thr, n: nd.idx.length, bounds: nd.bounds })), pick: pick.id } }, m: { loss: GBDT.logloss(pts, F), acc: GBDT.accuracy(pts, F) } });
      } : null);
      for (let i = 0; i < n; i++) F[i] += P.eta * GBDT.leafFor(tree.nodes, pts[i]).w;
      trees.push(tree.nodes);
      const show = t <= 4 || t % Math.max(1, Math.round(P.rounds / 30)) === 0 || t === P.rounds;
      if (show) frames.push({ phase: "add", major: t === 1, arch: ["split", "add"], title: `Round ${t}: tree with ${tree.nodes.filter((nd) => nd.leaf).length} leaves added`, text: `Log-loss <b>${GBDT.logloss(pts, F).toFixed(4)}</b>, accuracy <b>${(GBDT.accuracy(pts, F) * 100).toFixed(1)}%</b>. Leaf-wise trees are deep and lopsided where the data is hard, shallow where it is easy.`, tex: "F_t=F_{t-1}+\\eta\\,f_t", st: { trees: trees.slice(), F: F.slice(), bins, last: tree.nodes }, m: { loss: GBDT.logloss(pts, F), acc: GBDT.accuracy(pts, F) } });
    }
    frames.push({ ...frames[frames.length - 1], phase: "done", major: true, arch: ["add"], title: `${P.rounds} rounds complete`, text: "Same second-order objective as XGBoost; the differences are engineering: histograms, leaf-wise growth with num_leaves as the main capacity knob, GOSS sampling and exclusive feature bundling for sparse data." });
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "lgbm", fps: 2.5, classes: 2,
      legend: ui.legend([["var(--text-faint)", "bin boundaries", "line"], ["var(--series-4)", "leaf chosen to split", "ring"]]),
      tools: ui.btn("retrain", "retrain", true),
      hint: "click adds the selected class · drag / double-click to edit",
      data: { kind: "class", presets: ["spiral", "moons", "circles", "xor", "overlap", "imbalanced"], def: "spiral", n: 160, noise: 0.15, nRange: [20, 300] },
      side: [{ title: "booster", html: ui.range("R", "rounds", 1, 80, 1, 30) + ui.range("eta", "learning_rate", 0.05, 1, 0.05, 0.3) + ui.range("leaves", "num_leaves", 2, 31, 1, 8) + ui.range("bins", "max_bin", 4, 64, 1, 16) + ui.range("lam", "lambda_l2", 0, 10, 0.5, 1) + ui.check("goss", "GOSS sampling", false) + ui.check("grid", "show bin grid", true) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "bin", label: "binning", sub: "quantile histograms", x: 62, y: 40, kind: "param" },
          { id: "gh", label: "g, h", sub: "per point", x: 50, y: 108, w: 60, kind: "loss" },
          { id: "goss", label: "GOSS", sub: "sample by |g|", x: 50, y: 166, w: 74, kind: "param" },
          { id: "hist", label: "Σg, Σh per bin", sub: "per leaf", x: 186, y: 40, kind: "op" },
          { id: "pick", label: "best leaf", sub: "max gain", x: 186, y: 108, kind: "op" },
          { id: "split", label: "split it", sub: "leaf-wise", x: 300, y: 108, kind: "op" },
          { id: "add", label: "F += η f", sub: "add tree", x: 300, y: 166, kind: "out" },
        ],
        edges: [{ from: "bin", to: "hist" }, { from: "gh", to: "hist" }, { from: "goss", to: "gh", dash: true }, { from: "hist", to: "pick" }, { from: "pick", to: "split" }, { from: "split", to: "pick", bend: 20, dash: true, label: "until num_leaves" }, { from: "split", to: "add" }],
      },
      phases: { bin: { label: "binning", color: 3 }, goss: { label: "GOSS", color: 4 }, leaf: { label: "leaf-wise split", color: 0 }, add: { label: "add tree", color: 2 }, done: { label: "done", color: 2 } },
      panels: [{ id: "queue", title: "leaf queue (candidate gains)" }],
      charts: [{ id: "loss", title: "training log-loss & accuracy", series: [{ key: "loss", label: "log-loss", color: "var(--series-1)" }, { key: "acc", label: "accuracy", color: "var(--series-3)", dash: "4 3" }] }],
      onFrame: render, onData: () => retrain(),
    });
    lab.editable({ onChange: () => retrain({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    const P = () => ({ rounds: lab.val("R"), eta: lab.val("eta"), numLeaves: lab.val("leaves"), maxBin: lab.val("bins"), lam: lab.val("lam"), goss: lab.val("goss") });
    let base = 0;
    function retrain({ at = "start" } = {}) {
      const pts = lab.points;
      base = Math.log((pts.filter((p) => p.label).length + 0.5) / (pts.filter((p) => !p.label).length + 0.5));
      MLU.seedRng(21);
      lab.load(buildFrames(pts, P()), { at, play: at === "start" });
    }
    const gridG = L.under.append("g"), leafG = L.mid.append("g");
    function render(f) {
      const st = f.st, p = P();
      if (!st.F.length) { L.bg.selectAll("*").remove(); lab.drawPoints({}); return; }
      const pred = (q) => base + st.trees.reduce((s, nodes) => s + p.eta * GBDT.leafFor(nodes, q).w, 0);
      const grid = Lab.sampleGrid(pl, (x, y) => GBDT.sigmoid(pred({ x, y })), 110);
      Lab.paintGrid(L.bg, pl, grid, (v) => Lab.probColor(v, 0.5), { smooth: false });
      Lab.contour(L.under, pl, grid, [0.5], { cls: "iso-main", width: 2 });
      gridG.selectAll("*").remove();
      if (lab.val("grid") && st.bins) {
        for (const e of st.bins.x) gridG.append("line").attr("x1", pl.x(e)).attr("x2", pl.x(e)).attr("y1", pl.y(-10)).attr("y2", pl.y(10)).attr("stroke", "var(--text-faint)").attr("stroke-opacity", 0.45).attr("stroke-dasharray", "2 3");
        for (const e of st.bins.y) gridG.append("line").attr("y1", pl.y(e)).attr("y2", pl.y(e)).attr("x1", pl.x(-10)).attr("x2", pl.x(10)).attr("stroke", "var(--text-faint)").attr("stroke-opacity", 0.45).attr("stroke-dasharray", "2 3");
      }
      leafG.selectAll("*").remove();
      const rect = (b, attrs) => { const [x0, x1, y0, y1] = b; const r = leafG.append("rect").attr("x", pl.x(x0)).attr("y", pl.y(y1)).attr("width", pl.x(x1) - pl.x(x0)).attr("height", pl.y(y0) - pl.y(y1)); for (const k in attrs) r.attr(k, attrs[k]); return r; };
      const q = lab.R("panel-queue");
      if (st.grow) {
        for (const nd of st.grow.nodes) if (nd.leaf) rect(nd.bounds, { fill: "none", stroke: "var(--text)", "stroke-opacity": 0.4, "stroke-dasharray": "3 3" });
        const pk = st.grow.frontier.find((c) => c.id === st.grow.pick);
        rect(pk.bounds, { fill: "var(--series-4)", "fill-opacity": 0.08, stroke: "var(--series-4)", "stroke-width": 2.6 });
        const [x0, x1, y0, y1] = pk.bounds;
        const ln = pk.f === "x" ? [pk.thr, y0, pk.thr, y1] : [x0, pk.thr, x1, pk.thr];
        leafG.append("line").attr("x1", pl.x(ln[0])).attr("y1", pl.y(ln[1])).attr("x2", pl.x(ln[2])).attr("y2", pl.y(ln[3])).attr("stroke", "var(--series-4)").attr("stroke-width", 2.4).attr("stroke-dasharray", "6 4");
        q.innerHTML = `<table class="mini-table"><tr><th>leaf</th><th>points</th><th>best split</th><th>gain</th></tr>${st.grow.frontier.map((c) => `<tr class="${c.id === st.grow.pick ? "on" : ""}"><td>#${c.id}</td><td>${c.n}</td><td>${c.f} ≤ ${c.thr.toFixed(2)}</td><td>${c.gain.toFixed(3)}</td></tr>`).join("")}</table>`;
      } else if (st.last) {
        for (const nd of st.last) if (nd.leaf) rect(nd.bounds, { fill: "none", stroke: "var(--text)", "stroke-opacity": 0.35, "stroke-dasharray": "3 3" });
        q.innerHTML = `<div class="lab-card-note">Leaf-wise growth is recorded in detail for the first two rounds. This tree has <b>${st.last.filter((nd) => nd.leaf).length}</b> leaves at depths ${[...new Set(st.last.filter((nd) => nd.leaf).map((nd) => nd.depth))].sort().join(", ")}: unbalanced, unlike a level-wise tree.</div>`;
      } else q.innerHTML = `<div class="lab-card-note">The queue appears while the first trees are grown.</div>`;
      lab.drawPoints({ color: (d) => col(d.label), opacity: (d, i) => (st.goss && !st.goss.top.has(i) && !st.goss.rand.has(i) ? 0.18 : 1), r: (d, i) => (st.goss && st.goss.rand.has(i) ? 6 : 4.2), stroke: (d, i) => (st.goss && st.goss.top.has(i) ? "var(--text)" : "var(--bg)") });
      const m = f.m || {};
      lab.readout(`<div class="kv"><span>trees</span><b>${st.trees.length}</b><span>bins (x / y)</span><b>${st.bins ? st.bins.x.length + 1 + " / " + (st.bins.y.length + 1) : "–"}</b><span>log-loss</span><b>${fmt(m.loss)}</b><span>accuracy</span><b>${m.acc != null ? (m.acc * 100).toFixed(1) + "%" : "–"}</b></div>`);
    }
    lab.watch(["R", "eta", "leaves", "bins", "lam", "goss"], () => retrain());
    lab.watch(["grid"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    lab.loadPreset("spiral");
    return lab.destroy;
  }

  MLApp.register({
    id: "lightgbm",
    name: "LightGBM",
    category: "Supervised - Trees & Ensembles",
    tagline: "histograms, leaf-wise growth, GOSS",
    description: "The engineering behind LightGBM's speed: quantile histogram bins, leaf-wise growth that always splits the single best leaf, and gradient-based one-side sampling, each animated on a spiral dataset.",
    info: {
      type: "Supervised - Regression/Classification. Histogram-based, leaf-wise (best-first) gradient-boosted tree ensemble.",
      scenario: "Very large or high-dimensional tabular datasets where training speed and memory matter - LightGBM trades some tree 'balance' for much faster training.",
      inputs: "Feature vectors x (often large-n / high-cardinality) and targets y.",
      intuition: {
        definition: "The same boosted ensemble as XGBoost, made fast by two changes: bin every feature into a <b>histogram</b> once so split search costs O(bins) instead of O(n), and grow each tree <b>leaf-wise</b>, always splitting whichever leaf anywhere in the tree promises the biggest error reduction.",
        steps: [
          "Bucket continuous features into a few hundred bins up front.",
          "Search splits over bins, not raw sorted values.",
          "Split the single best leaf in the whole tree, not a whole level.",
          "Constrain num_leaves and min_data_in_leaf so the deep branches do not memorise.",
        ],
        applications: [
          "Tabular datasets with millions of rows where XGBoost is too slow",
          "High-cardinality categorical features, handled natively",
          "Low-latency retraining pipelines",
          "Learning-to-rank with LambdaRank objectives",
          "Click-through and conversion prediction at ad scale",
        ],
      },
      math: [
        { title: "Same objective as XGBoost", formula: "Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ,   w* = −G/(H+λ)", note: "The mathematics of what makes a good split is unchanged. Everything LightGBM does differently is about how that search is executed." },
        { title: "Histogram binning", formula: "split search: O(m·p) → O(bins·p) per node", note: "Each feature is discretized once into at most max_bin buckets. Gradient and Hessian sums are accumulated per bucket, so the per-node cost stops depending on the row count." },
        { title: "Histogram subtraction", formula: "hist(child_right) = hist(parent) − hist(child_left)", note: "Only the smaller child's histogram is built from data; the sibling comes free by subtraction. This roughly halves the work at every node." },
        { title: "Leaf-wise growth", formula: "split argmax_{leaf ∈ tree} Gain(leaf),  until num_leaves reached", note: "Best-first rather than level-by-level. For a fixed leaf budget this reaches lower training loss, because leaves are spent where the error is, not spread evenly." },
        { title: "GOSS", formula: "keep the top a% by |gradient|, sample b% of the rest, upweight them by (1−a)/b", note: "Gradient-based One-Side Sampling. Well-fitted points have small gradients and contribute little, so most of them can be dropped with a correction factor that keeps the gain estimate unbiased." },
        { title: "EFB", formula: "bundle mutually exclusive sparse features into one", note: "Exclusive Feature Bundling. In sparse one-hot data most feature pairs are never non-zero together, so they can share a bin range and cut the effective feature count." },
      ],
      pipeline: [
        { label: "Bin features", note: "once, max_bin buckets" },
        { label: "Accumulate g,h", note: "per bin" },
        { label: "Best leaf", note: "max gain anywhere" },
        { label: "Split + subtract", note: "sibling hist free" },
        { label: "num_leaves reached", note: "next round", accent: "green" },
      ],
      decisionFunction: {
        text: "Same additive ensemble ŷ(x) = F₀ + η·Σₘ treeₘ(x) as GBM/XGBoost - the difference is how each tree is grown.",
        mechanism: "Leaf-wise (best-first) growth repeatedly splits whichever current leaf anywhere in the tree gives the single largest error reduction - not depth-by-depth like standard CART/GBM/XGBoost - producing deeper, more asymmetric trees for the same leaf budget.",
      },
      lossFunction: {
        text: "Same gradient/hessian Gain criterion as XGBoost, applied to histogram-binned feature values",
        mechanism: "Continuous features are discretized into a fixed number of histogram bins up front, so split search costs O(bins) per feature instead of O(n) - a large constant-factor speedup at the cost of slightly coarser split thresholds.",
      },
      optimization: [
        { title: "Where the speed comes from", formula: "histograms + subtraction + GOSS + EFB", note: "No single trick dominates. Binning removes the row-count dependence, subtraction halves node work, GOSS cuts rows, and EFB cuts columns." },
        { title: "Binning is a one-off cost", formula: "O(m·p) once, then O(bins·p) per node forever", note: "The up-front discretization is amortised across every node of every tree, which is why the advantage grows with ensemble size." },
        { title: "Leaf-wise risk", formula: "unbounded depth for a fixed leaf budget", note: "Best-first growth can drive one branch very deep on a handful of rows. num_leaves and min_data_in_leaf are the guards, and they matter far more here than in XGBoost." },
        { title: "Depth relationship", formula: "num_leaves ≤ 2^max_depth", note: "If you set both, keep num_leaves comfortably below the level-wise equivalent. Setting num_leaves = 2^max_depth reproduces level-wise behaviour and gives up the advantage." },
        { title: "Native categoricals", formula: "sort categories by Σg/Σh, then split the sorted order", note: "Avoids one-hot explosion. Finds a near-optimal partition of category values in O(k log k) rather than trying all 2^(k−1) subsets." },
      ],
      output: "A continuous predicted value (sum of the ensemble's tree contributions).",
      assumptions: [
        { name: "Enough rows per leaf", why: "Leaf-wise growth will happily isolate tiny groups, which is the main way LightGBM overfits.", check: "Set min_data_in_leaf to at least 20, and much higher on small datasets." },
        { name: "Binning resolution is adequate", why: "Split thresholds are restricted to bin edges, so very fine distinctions can be lost.", check: "Raise max_bin if accuracy is limited; lower it for speed." },
        { name: "Dataset is reasonably large", why: "The histogram and sampling machinery has fixed overhead that only pays off at scale.", check: "Below a few thousand rows, XGBoost or a random forest is often simpler and just as good." },
        { name: "Tabular, structured features", why: "No notion of spatial or sequential structure.", check: "Use a neural network for unstructured data." },
        { name: "Validation split available", why: "Round count and leaf budget both need empirical selection.", check: "Always pass a valid_sets for early stopping." },
      ],
      regularization: [
        { name: "num_leaves", formula: "leaf count ≤ L", note: "The primary capacity control, replacing max_depth. This is the knob to tune first." },
        { name: "min_data_in_leaf", formula: "n_leaf ≥ k", note: "Critical under leaf-wise growth. Without it, deep branches will isolate individual rows." },
        { name: "min_sum_hessian_in_leaf", formula: "H_leaf ≥ h_min", note: "The curvature-weighted counterpart, equivalent to XGBoost's min_child_weight." },
        { name: "lambda_l1 / lambda_l2", formula: "+ α Σ|wⱼ| + ½λ Σwⱼ²", note: "Same leaf-weight penalties as XGBoost, entering the same w* and gain formulas." },
        { name: "feature_fraction", formula: "random subset of columns per tree", note: "Decorrelates trees and speeds up split search." },
        { name: "bagging_fraction", formula: "random subset of rows per iteration", note: "Requires bagging_freq to be set, otherwise it is silently ignored." },
        { name: "max_bin", formula: "histogram resolution", note: "Fewer bins is itself a regularizer: coarser thresholds cannot carve out noise as precisely." },
      ],
      hyperparameters: [
        { name: "num_leaves", range: "15 - 255", increasing: "Much more capacity and the fastest route to overfitting.", strategy: "The single most important parameter. Start at 31 and keep it well under 2^max_depth." },
        { name: "learning_rate", range: "0.01 - 0.3", increasing: "Fewer rounds required, higher overfitting risk.", strategy: "Fix at 0.05 and let early stopping choose num_iterations." },
        { name: "num_iterations", range: "100 - 10000", increasing: "More capacity, eventual overfitting.", strategy: "Set high, early-stop. Do not grid-search." },
        { name: "min_data_in_leaf", range: "20 - 500", increasing: "Stronger regularization, shallower effective trees.", strategy: "Raise aggressively on small or noisy data. This is the main defence against leaf-wise overfitting." },
        { name: "max_bin", range: "63 - 511", increasing: "Finer split thresholds, better accuracy, slower training and more memory.", strategy: "255 is the default and usually right. Drop to 63 for a quick speed win." },
        { name: "feature_fraction", range: "0.4 - 1.0", increasing: "Trees see more features and correlate more.", strategy: "0.8 default; lower with many correlated columns." },
        { name: "bagging_fraction", range: "0.4 - 1.0", increasing: "Less regularization from sampling.", strategy: "0.8 with bagging_freq=1. Remember both must be set." },
        { name: "lambda_l2", range: "0 - 10", increasing: "Leaf values shrink toward zero.", strategy: "Log-scale search jointly with num_leaves." },
        { name: "max_depth", range: "-1 (none) or 3 - 12", increasing: "Bounds how deep leaf-wise growth can run.", strategy: "Leave unlimited and control with num_leaves; set it only as a safety net on small data." },
      ],
      metrics: ["RMSE / MAE / R² (regression)", "Log-loss / AUC (classification)", "NDCG / MAP (ranking)", "Training time & memory footprint (a key differentiator at scale)"],
      typicalUses: ["Very large tabular datasets", "Low-latency training pipelines", "Ranking/recommendation (e.g. LambdaRank objectives)", "Anywhere GBM-quality accuracy is needed but training speed/memory is the bottleneck"],
      diagnostics: [
        "If training accuracy is near perfect while validation lags badly, num_leaves is too high or min_data_in_leaf too low. Fix those before touching anything else.",
        "Inspect the actual tree depths. Leaf-wise growth producing depth-30 branches on a small dataset is a warning sign.",
        "Watch the early-stopping curve. LightGBM converges in fewer rounds than XGBoost at the same learning rate because leaf-wise growth fits faster per tree.",
        "If bagging appears to do nothing, check that bagging_freq is set. bagging_fraction alone is ignored.",
        "On small datasets, compare against XGBoost. LightGBM's advantages are scale advantages and can invert below a few thousand rows.",
      ],
      advantages: [
        "Dramatically faster training than exact-split boosting, often by an order of magnitude on large data.",
        "Much lower memory use, since binned features are stored compactly as small integers.",
        "Leaf-wise growth reaches lower loss for the same number of leaves.",
        "Categorical features are handled natively without one-hot encoding.",
        "Scales to millions of rows and high-dimensional sparse data via GOSS and EFB.",
        "Distributed and GPU training are supported out of the box.",
      ],
      limitations: [
        { name: "Overfits small datasets readily", note: "leaf-wise growth isolates tiny groups", fix: "raise min_data_in_leaf, lower num_leaves, or use XGBoost instead." },
        { name: "num_leaves is easy to misconfigure", note: "it is not interchangeable with max_depth", fix: "keep num_leaves well below 2^max_depth." },
        { name: "Binning loses precision", note: "split thresholds are limited to bin edges", fix: "raise max_bin when accuracy matters more than speed." },
        { name: "Unbalanced trees", note: "deep asymmetric branches are harder to reason about", fix: "cap max_depth as a safety net." },
        { name: "Sensitive to parameter interactions", note: "bagging_fraction silently does nothing without bagging_freq", fix: "read the parameter aliases carefully." },
        { name: "Cannot extrapolate", note: "standard tree limitation", fix: "detrend or model the trend separately." },
      ],
      alternatives: [
        { name: "XGBoost", when: "Smaller datasets, or you prefer level-wise growth's more conservative behaviour." },
        { name: "CatBoost", when: "Categorical features dominate, or you want strong results with minimal tuning." },
        { name: "Random forest", when: "You want robustness with essentially no tuning." },
        { name: "Linear model", when: "Interpretability or extrapolation is required." },
      ],
      pitfalls: [
        { problem: "Severe overfitting on a small dataset", solution: "Lower num_leaves, raise min_data_in_leaf substantially, and consider XGBoost." },
        { problem: "Setting num_leaves = 2^max_depth", solution: "That reproduces level-wise growth and discards the leaf-wise advantage. Keep it much lower." },
        { problem: "Bagging has no effect", solution: "bagging_freq must be set to a positive integer alongside bagging_fraction." },
        { problem: "Warning about no further splits", solution: "min_data_in_leaf or min_gain_to_split is too strict for the data volume." },
        { problem: "Categorical feature performs poorly", solution: "Pass it via categorical_feature rather than one-hot encoding, and beware high-cardinality overfitting." },
        { problem: "Results differ run to run", solution: "Sampling and multithreading introduce nondeterminism. Set seeds and deterministic=True." },
      ],
      quickRef: [
        { name: "Split gain", formula: "same as XGBoost" },
        { name: "Leaf weight", formula: "w* = −G/(H+λ)" },
        { name: "Growth", formula: "leaf-wise (best-first)" },
        { name: "Split cost", formula: "O(bins·p) per node" },
        { name: "Hist subtraction", formula: "sibling = parent − child" },
        { name: "Leaf budget", formula: "num_leaves ≤ 2^max_depth" },
        { name: "GOSS", formula: "keep top-a% |g|, sample b% rest" },
        { name: "EFB", formula: "bundle exclusive sparse features" },
      ],
      code: `import lightgbm as lgb

train_set = lgb.Dataset(X_fit, y_fit, categorical_feature=cat_cols)
valid_set = lgb.Dataset(X_val, y_val, reference=train_set)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,            # the main capacity lever, not max_depth
    "min_data_in_leaf": 50,      # critical guard for leaf-wise growth
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,           # without this, bagging_fraction is ignored
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbosity": -1,
    "seed": 42,
}

model = lgb.train(
    params, train_set,
    num_boost_round=5000,        # upper bound; early stopping decides
    valid_sets=[valid_set],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
print("best iteration:", model.best_iteration)`,
      whyChain: [
        { q: "What is the difference between leaf-wise and level-wise growth?", a: "Level-wise splits every node at the current depth before descending, producing balanced trees. Leaf-wise repeatedly splits whichever single leaf in the whole tree offers the largest gain, producing deep asymmetric trees. For the same leaf count, leaf-wise reaches lower training loss." },
        { q: "So why does leaf-wise overfit more?", a: "Because nothing stops it concentrating every remaining leaf on one hard corner of the data. It can drive a branch thirty levels deep to isolate a handful of noisy rows. Level-wise growth is implicitly regularized by having to spend leaves evenly." },
        { q: "Why does binning make it so much faster?", a: "Exact split search must sort and scan every value, which is O(m log m) per feature per node. With histograms you accumulate gradient sums into a fixed number of buckets, so the per-node cost becomes O(bins) and stops depending on the row count entirely." },
        { q: "Does binning hurt accuracy?", a: "Barely. Split thresholds are restricted to bin edges, but with 255 bins the loss of precision is usually below the noise floor. Coarser bins even act as a mild regularizer." },
        { q: "What is histogram subtraction?", a: "After building the histogram for one child, the sibling's histogram is just the parent's minus that child's. So you only ever build the smaller child from raw data, which roughly halves the work per node." },
        { q: "What is the intuition behind GOSS?", a: "Points with small gradients are already well fitted and contribute little to the split gain. GOSS keeps all the large-gradient points, randomly samples the rest, and upweights the survivors so the gain estimate stays unbiased. You get most of the information from a fraction of the rows." },
        { q: "Why not just set num_leaves = 2^max_depth?", a: "That gives leaf-wise growth exactly the budget a balanced tree would use, so it behaves like level-wise growth and you lose the benefit. The point is to use fewer leaves, placed where they matter." },
        { q: "When should you prefer XGBoost?", a: "On smaller datasets, where LightGBM's overhead does not pay off and its leaf-wise growth overfits more readily. XGBoost with tree_method='hist' also closes much of the speed gap." },
      ],
      parameters: [
        { name: "num_leaves", effect: "Direct leaf-wise complexity control (used instead of max_depth) - the main lever for model capacity." },
        { name: "learning rate", effect: "Shrinks each tree's contribution, same role as in GBM/XGBoost." },
        { name: "max_bin", effect: "Histogram resolution. More bins → more precise splits but slower and more memory." },
        { name: "min_data_in_leaf", effect: "Regularizes against overfitting on tiny leaves - important since leaf-wise growth can otherwise carve out very small, deep leaves." },
      ],
      metrics: ["RMSE / MAE / R² (regression)", "Log-loss / AUC (classification)", "Training time & memory footprint (a key differentiator at scale)"],
      typicalUses: ["Very large tabular datasets", "Low-latency training pipelines", "Ranking/recommendation (e.g. LambdaRank objectives)", "Anywhere GBM-quality accuracy is needed but training speed/memory is the bottleneck"],
      workedExample: {
        setup: "Grow a tree to a budget of 5 leaves on a small dataset, comparing leaf-wise vs level-wise growth.",
        steps: [
          "Level-wise can only add a full layer at a time: 1 leaf → split root → 2 leaves → split both → 4 leaves. Adding a 5th leaf would require a full next level (8 leaves), which exceeds the budget - so level-wise stops at 4 leaves.",
          "Leaf-wise instead asks 'which single leaf, if split, reduces error the most?' at every step: split root (1→2 leaves), then split whichever of those 2 has the best next split (2→3), then again (3→4), then again (4→5) - reaching the full budget of 5.",
          "Because leaf-wise always spends its next leaf where the error reduction is largest, its 5 leaves are concentrated where the data is hardest to fit, rather than spread evenly.",
        ],
        result: "For the same leaf budget, leaf-wise typically reaches lower training error than level-wise - exactly the comparison the playground above shows live",
      },
    },
    mount,
  });
})();
