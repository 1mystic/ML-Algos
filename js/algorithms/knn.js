(() => {
  const { ui, col, fmt } = Lab;
  const METRIC = {
    euclidean: (a, b) => Math.hypot(a.x - b.x, a.y - b.y),
    manhattan: (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y),
    chebyshev: (a, b) => Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y)),
  };
  const TEX = {
    euclidean: "d(x,x')=\\sqrt{\\textstyle\\sum_j (x_j-x'_j)^2}",
    manhattan: "d(x,x')=\\textstyle\\sum_j |x_j-x'_j|",
    chebyshev: "d(x,x')=\\max_j |x_j-x'_j|",
  };
  function classify(pts, q, k, metric, weighted, nc, skip = -1) {
    const d = [];
    for (let i = 0; i < pts.length; i++) if (i !== skip) d.push({ i, d: METRIC[metric](pts[i], q) });
    d.sort((a, b) => a.d - b.d);
    const votes = new Array(nc).fill(0);
    for (const n of d.slice(0, k)) votes[pts[n.i].label] += weighted ? 1 / (n.d + 1e-6) : 1;
    let best = 0; for (let c = 1; c < nc; c++) if (votes[c] > votes[best]) best = c;
    const tot = votes.reduce((a, b) => a + b, 0) || 1;
    return { label: best, conf: votes[best] / tot, votes, sorted: d };
  }

  function buildFrames(pts, q, { k, metric, weighted }, nc) {
    if (!pts.length) return [{ phase: "idle", title: "No training points", text: "Load a dataset or click to add points.", st: { k: 0, votes: new Array(nc).fill(0) } }];
    k = Math.min(k, pts.length);
    const res = classify(pts, q, k, metric, weighted, nc);
    const frames = [];
    frames.push({ phase: "query", arch: ["q"], title: "A new query point arrives", text: "KNN has no training step at all: the model <b>is</b> the stored data. All the work happens now, at prediction time. Drag the ◆ to query anywhere.", st: { k: 0, votes: new Array(nc).fill(0) } });
    frames.push({ phase: "dist", major: true, arch: ["q", "data", "dist"], title: `Measure the ${metric} distance to all ${pts.length} points`, text: "Every stored point is compared with the query. Line opacity shows closeness. This O(n·d) scan per query is why KNN is slow at inference on big data (KD-trees and ANN indexes help).", tex: TEX[metric], st: { k: 0, allLines: true, votes: new Array(nc).fill(0) } });
    const votes = new Array(nc).fill(0);
    for (let j = 0; j < k; j++) {
      const n = res.sorted[j];
      votes[pts[n.i].label] += weighted ? 1 / (n.d + 1e-6) : 1;
      frames.push({
        phase: "neigh", arch: ["dist", "sort", "topk"],
        title: `Neighbour ${j + 1} of ${k}: class ${pts[n.i].label} at distance ${n.d.toFixed(2)}`,
        text: `The ${metric} ball grows until it holds ${j + 1} point${j ? "s" : ""}. ${weighted ? "Distance weighting: closer neighbours carry more vote (1/d)." : "Each neighbour casts one equal vote."}`,
        st: { k: j + 1, votes: votes.slice(), radius: n.d },
      });
    }
    const tot = votes.reduce((a, b) => a + b, 0) || 1;
    frames.push({
      phase: "vote", major: true, arch: ["topk", "vote", "out"],
      title: `Majority vote → class ${res.label}`,
      text: `Votes: ${votes.map((v, c) => `class ${c}: <b>${weighted ? v.toFixed(2) : v}</b>`).join(", ")}. The vote share (${(res.conf * 100).toFixed(0)}%) is a crude probability estimate.`,
      tex: weighted ? "\\hat y=\\arg\\max_c\\sum_{i\\in N_k(x)}\\frac{\\mathbb 1[y_i=c]}{d(x,x_i)}" : "\\hat y=\\arg\\max_c\\sum_{i\\in N_k(x)}\\mathbb 1[y_i=c]",
      st: { k, votes: votes.slice(), radius: res.sorted[k - 1].d, decided: res.label },
    });
    frames.push({
      phase: "field", major: true, arch: ["out"],
      title: "Repeat for every location: the decision regions",
      text: `Running the same query everywhere paints the decision boundary. Small k gives jagged islands (high variance); large k smooths them away (high bias). k = ${k} here.`,
      st: { k, votes: votes.slice(), radius: res.sorted[k - 1].d, decided: res.label, field: true },
    });
    return frames;
  }

  function mount(root) {
    const NC = 3;
    const lab = Lab.studio(root, {
      id: "knn", fps: 3, classes: NC,
      legend: ui.legend([["var(--accent)", "query ◆"], ["var(--text)", "k-NN ball", "ring"]]),
      tools: ui.btn("replay", "replay query", true) + ui.btn("loo", "scan k (LOO)", false, "leave-one-out accuracy for every k"),
      hint: "drag the ◆ query · click adds the selected class · drag / double-click points to edit",
      data: { kind: "class", presets: ["blobs3", "moons", "circles", "xor", "spiral", "overlap", "imbalanced"], def: "blobs3", n: 90, noise: 0.35 },
      side: [{ title: "model", html: ui.range("k", "k (neighbours)", 1, 30, 1, 7) + ui.select("metric", "distance", ["euclidean", "manhattan", "chebyshev"], "euclidean") + ui.check("w", "distance-weighted votes", false) + ui.check("field", "always show decision regions", false) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "q", label: "query x", sub: "new point", x: 46, y: 40, kind: "data" },
          { id: "data", label: "stored data", sub: "all (xᵢ, yᵢ)", x: 46, y: 140, kind: "data", info: "Training = memorising the dataset. Nothing is fitted." },
          { id: "dist", label: "distances", sub: "d(x, xᵢ) ∀i", x: 160, y: 90, kind: "op" },
          { id: "sort", label: "sort", sub: "ascending", x: 250, y: 40, w: 70, kind: "op" },
          { id: "topk", label: "top-k", sub: "N_k(x)", x: 250, y: 140, w: 70, kind: "op" },
          { id: "vote", label: "vote", sub: "majority", x: 338, y: 140, w: 64, kind: "op" },
          { id: "out", label: "ŷ", sub: "class", x: 338, y: 40, w: 56, kind: "out" },
        ],
        edges: [{ from: "q", to: "dist" }, { from: "data", to: "dist" }, { from: "dist", to: "sort" }, { from: "sort", to: "topk" }, { from: "topk", to: "vote" }, { from: "vote", to: "out" }],
      },
      phases: { query: { label: "query", color: 3 }, dist: { label: "distances", color: 4 }, neigh: { label: "neighbours", color: 0 }, vote: { label: "vote", color: 2 }, field: { label: "regions", color: 2 } },
      panels: [{ id: "votes", title: "vote tally" }],
      charts: [{ id: "loo", title: "accuracy vs k (leave-one-out)", custom: true, note: "Leave-one-out: classify each point using all the others. The peak marks a good k; k = 1 always scores 100% on the training set itself." }],
      onFrame: render, onData: () => { looData = null; drawLoo(); replay(); },
    });
    lab.editable({ onChange: () => { looData = null; drawLoo(); replay({ at: "end" }); } });
    const pl = lab.plane, L = pl.layers;
    const q = { x: 1.2, y: 0.8 };
    const params = () => ({ k: lab.val("k"), metric: lab.val("metric"), weighted: lab.val("w") });
    function replay({ at = "start" } = {}) { lab.load(buildFrames(lab.points, q, params(), NC), { at, play: at === "start" }); }

    const votes = Lab.bars(lab.R("panel-votes"), { w: 360, h: 140 });
    let looData = null;
    function drawLoo() {
      const el = lab.charts.loo.el; el.innerHTML = "";
      const P = Lab.xy(el, { w: 360, h: 130, xd: [1, 30], yd: [0, 1], xLabel: "k" });
      if (!looData) { P.g.append("text").attr("class", "ax-label").attr("x", 190).attr("y", 60).attr("text-anchor", "middle").text("press “scan k” to compute"); return; }
      P.line("acc", looData.map((d) => [d.k, d.acc]), { color: "var(--series-3)" });
      const best = looData.reduce((a, b) => (b.acc > a.acc ? b : a));
      P.g.append("circle").attr("cx", P.x(best.k)).attr("cy", P.y(best.acc)).attr("r", 4.5).attr("fill", "var(--series-3)");
      P.vline(lab.val("k"));
      el.closest(".lab-card").querySelector(".chart-legend").innerHTML = `best k = <b>${best.k}</b> (${(best.acc * 100).toFixed(1)}%)`;
    }
    lab.q('[data-b="loo"]').onclick = () => {
      const pts = lab.points, p = params();
      looData = d3.range(1, 31).map((k) => ({ k, acc: pts.reduce((s, pt, i) => s + (classify(pts, pt, Math.min(k, pts.length - 1), p.metric, p.weighted, NC, i).label === pt.label), 0) / Math.max(1, pts.length) }));
      drawLoo();
    };
    drawLoo();

    const ballG = L.mid.append("g"), qG = L.top.append("g").attr("class", "no-add").style("cursor", "move");
    qG.append("rect").attr("x", -9).attr("y", -9).attr("width", 18).attr("height", 18).attr("transform", "rotate(45)").attr("fill", "var(--accent)").attr("stroke", "var(--bg)").attr("stroke-width", 2);
    qG.call(d3.drag().on("drag", (ev) => {
      q.x = Math.max(-10, Math.min(10, pl.x.invert(ev.x))); q.y = Math.max(-10, Math.min(10, pl.y.invert(ev.y)));
      lab.load(buildFrames(lab.points, q, params(), NC), { at: "end" });
    }));

    function ballPath(r) {
      const m = lab.val("metric");
      if (m === "euclidean") return null;
      const cx = q.x, cy = q.y;
      const pts = m === "manhattan" ? [[cx + r, cy], [cx, cy + r], [cx - r, cy], [cx, cy - r]] : [[cx + r, cy + r], [cx - r, cy + r], [cx - r, cy - r], [cx + r, cy - r]];
      return "M" + pts.map((p) => `${pl.x(p[0])},${pl.y(p[1])}`).join("L") + "Z";
    }

    function render(f) {
      const p = params();
      const st = f.st;
      const res = lab.points.length ? classify(lab.points, q, Math.min(p.k, lab.points.length), p.metric, p.weighted, NC) : null;
      if ((st.field || lab.val("field")) && lab.points.length) {
        Lab.field(L.bg, pl, (x, y) => { const r = classify(lab.points, { x, y }, Math.min(p.k, lab.points.length), p.metric, p.weighted, NC); return r.label + r.conf * 0.999; },
          (v) => Lab.classColor(Math.floor(v), (v % 1) * 1.6 - 0.5, 0.36), { res: 70 });
      } else L.bg.selectAll("*").remove();
      const inK = new Set(res && st.k ? res.sorted.slice(0, st.k).map((n) => n.i) : []);
      const maxD = res ? res.sorted[res.sorted.length - 1].d : 1;
      const lines = L.links.selectAll("line").data(res && (st.allLines || st.k) ? res.sorted.filter((n) => st.allLines || inK.has(n.i)) : []);
      lines.exit().remove();
      lines.enter().append("line").merge(lines)
        .attr("x1", pl.x(q.x)).attr("y1", pl.y(q.y)).attr("x2", (n) => pl.x(lab.points[n.i].x)).attr("y2", (n) => pl.y(lab.points[n.i].y))
        .attr("stroke", (n) => (inK.has(n.i) ? col(lab.points[n.i].label) : "var(--text-faint)"))
        .attr("stroke-width", (n) => (inK.has(n.i) ? 2 : 0.6)).attr("stroke-opacity", (n) => (inK.has(n.i) ? 0.9 : 0.5 * (1 - n.d / maxD) + 0.04));
      // neighbourhood ball
      ballG.selectAll("*").remove();
      if (st.radius) {
        const bp = ballPath(st.radius);
        const sh = bp ? ballG.append("path").attr("d", bp) : ballG.append("ellipse").attr("cx", pl.x(q.x)).attr("cy", pl.y(q.y)).attr("rx", Math.abs(pl.x(st.radius) - pl.x(0))).attr("ry", Math.abs(pl.y(st.radius) - pl.y(0)));
        sh.attr("fill", st.decided != null ? col(st.decided) : "var(--text)").attr("fill-opacity", 0.07).attr("stroke", "var(--text)").attr("stroke-dasharray", "5 4").attr("stroke-width", 1.4);
      }
      lab.drawPoints({ color: (d) => col(d.label), r: (d, i) => (inK.has(i) ? 6.5 : 4.4), stroke: (d, i) => (inK.has(i) ? "var(--text)" : "var(--bg)"), sw: (d, i) => (inK.has(i) ? 1.8 : 1.1) });
      qG.attr("transform", `translate(${pl.x(q.x)},${pl.y(q.y)})`);
      qG.select("rect").attr("fill", st.decided != null ? col(st.decided) : "var(--accent)");
      votes.set(st.votes.map((v, c) => ({ label: "class " + c, value: v, color: col(c), text: p.weighted ? v.toFixed(2) : String(v) })));
      lab.readout(`<div class="kv"><span>query</span><b>(${q.x.toFixed(2)}, ${q.y.toFixed(2)})</b><span>prediction</span><b>${res ? "class " + res.label : "–"}</b><span>vote share</span><b>${res ? (res.conf * 100).toFixed(0) + "%" : "–"}</b><span>training cost</span><b>0 (lazy)</b></div>`);
      if (looData) { const P = lab.charts.loo.el.querySelector("svg"); if (P) drawLoo(); }
    }
    lab.watch(["k", "metric", "w"], () => replay({ at: "end" }));
    lab.watch(["field"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.loadPreset("blobs3");
    return lab.destroy;
  }

  MLApp.register({
    id: "knn",
    name: "K-Nearest Neighbors",
    category: "Supervised - Classification",
    tagline: "lazy learner: distances, neighbours, vote",
    description: "No training at all: the data is the model. Drag a query point and watch distances get measured, the neighbour ball grow one point at a time, and the vote decide; then scan k with leave-one-out accuracy.",
    info: {
      type: "Supervised - Classification (a regression variant also exists). Non-parametric, instance-based ('lazy') learning.",
      scenario: "Irregular / non-linear decision boundaries where enough labeled data exists near any likely query point, and a training phase isn't wanted or needed.",
      inputs: "A set of labeled training points {(xᵢ, yᵢ)}, a distance metric, and a query point x.",
      intuition: {
        definition: "KNN has no model. It memorises the training set, and to classify a new point it simply asks the <b>k closest stored points</b> what they are and takes a vote. All the work happens at prediction time.",
        steps: [
          "Store every training point as-is. Training cost is zero.",
          "For a query, measure the distance to all stored points.",
          "Keep the k nearest and let them vote (or average, for regression).",
          "k controls smoothness: it is the only real knob.",
        ],
        applications: [
          "Recommenders: find users nearest to you and suggest what they liked",
          "Anomaly detection: points with distant neighbours are suspicious",
          "Handwritten digit recognition baselines",
          "Imputing missing values from similar rows",
          "Geospatial or sensor lookups where nearness is literally meaningful",
        ],
      },
      math: [
        { title: "Distance metric", formula: "d(x, xᵢ) = ‖x − xᵢ‖₂ = √( Σⱼ (xⱼ − xᵢⱼ)² )", note: "Euclidean by default. Manhattan (L1) resists outliers better; cosine suits text and other direction-only data." },
        { title: "Neighbourhood", formula: "N_k(x) = the k training points with smallest d(x, xᵢ)", note: "Found by a full scan here, or in roughly O(log m) using a KD-tree or ball-tree in low dimensions." },
        { title: "Classification rule", formula: "ŷ(x) = argmax_c Σ_{i ∈ N_k(x)} 1[yᵢ = c]", note: "A plain majority vote. Ties are broken by the nearest member or by shrinking k." },
        { title: "Distance weighting", formula: "ŷ(x) = argmax_c Σ_{i ∈ N_k(x)} (1 / d(x, xᵢ)) · 1[yᵢ = c]", note: "Closer neighbours count more, which softens the jump as a point crosses in or out of the neighbourhood." },
        { title: "Regression variant", formula: "ŷ(x) = (1/k) Σ_{i ∈ N_k(x)} yᵢ", note: "Same neighbourhood, averaged instead of voted. Produces a piecewise-constant surface." },
      ],
      pipeline: [
        { label: "Store data", note: "no training" },
        { label: "Query x", note: "new point" },
        { label: "Distances", note: "to all m points" },
        { label: "Take k nearest", note: "partial sort" },
        { label: "Vote", note: "majority label", accent: "green" },
      ],
      decisionFunction: {
        text: "ŷ(x) = majority label among the k training points closest to x",
        mechanism: "At prediction time, compute the distance from x to every training point, sort, keep the k smallest, and vote - there are no trained parameters at all, just stored data.",
      },
      lossFunction: {
        text: "No training-time objective - KNN never fits parameters.",
        mechanism: "There's nothing to minimize during 'training' (which is just storing the data); the only real design choices, k and the distance metric, are instead chosen to minimize held-out validation error.",
        plot: { fn: (k) => 0.02 * (k - 8) ** 2 + 0.05, domain: [1, 25], yDomain: [0, 1.3], color: "var(--accent)", caption: "typical validation-error curve vs k: small k → high variance (fits noise), large k → high bias (over-smooths)" },
      },
      optimization: [
        { title: "Cost profile", formula: "train: O(1)   predict: O(m·p) per query", note: "The opposite of most models. Training is instant, but every prediction touches the whole dataset, so serving cost grows with the data you keep." },
        { title: "Speeding up search", formula: "KD-tree / ball-tree: O(log m)   LSH / HNSW: approximate", note: "Trees help while p is under roughly 20. Past that they degrade to a linear scan and approximate methods take over." },
        { title: "Choosing k", formula: "k* = argmin_k CV-error(k)", note: "The only fitting that happens. Use odd k for two classes to avoid ties; a common starting guess is k ≈ √m." },
      ],
      output: "A predicted class label for the query point, plus the vote breakdown as an informal confidence.",
      assumptions: [
        { name: "Locality holds", why: "The whole method rests on nearby points sharing a label. If the target flips rapidly over short distances, no k works.", check: "Compare validation accuracy against a global model such as logistic regression." },
        { name: "Features are comparably scaled", why: "Euclidean distance is dominated by whichever feature has the largest numeric range.", check: "Always standardize or min-max scale before fitting." },
        { name: "Low-to-moderate dimensionality", why: "In high dimensions all pairwise distances converge, so 'nearest' stops meaning anything.", check: "If p is large, reduce with PCA first or switch models." },
        { name: "Roughly balanced classes", why: "A dominant class wins votes purely by being numerous in every neighbourhood.", check: "Compare class counts; use distance weighting or resampling." },
        { name: "Enough data density", why: "Sparse regions give neighbours that are not actually near.", check: "Inspect the distance to the k-th neighbour; large values mean extrapolation." },
      ],
      hyperparameters: [
        { name: "k", range: "1 - 50", increasing: "Smoother boundary, higher bias, lower variance. k = m predicts the global majority.", strategy: "Sweep k with cross-validation and pick the minimum of the U-shaped error curve. Keep k odd for binary problems." },
        { name: "distance metric", range: "euclidean / manhattan / cosine", increasing: "Not applicable", strategy: "Euclidean for dense numeric data, Manhattan when outliers or mixed scales bite, cosine for text and embeddings." },
        { name: "weights", range: "uniform / distance", increasing: "Distance weighting sharpens the boundary near dense clusters.", strategy: "Try 'distance' when a large k is needed for stability but you do not want far neighbours to dilute the vote." },
        { name: "algorithm", range: "brute / kd_tree / ball_tree", increasing: "Not applicable", strategy: "Let the library auto-select. Force brute force above roughly 20 dimensions." },
        { name: "p (Minkowski)", range: "1 - 2", increasing: "Moves from Manhattan (1) toward Euclidean (2) and beyond.", strategy: "Rarely worth tuning past choosing 1 or 2." },
      ],
      metrics: ["Accuracy", "Confusion matrix", "F1-score", "MSE (for the regression variant)"],
      typicalUses: ["Recommendation systems", "Anomaly detection", "Handwriting / image recognition baselines", "Any modest-sized dataset where local similarity is meaningful"],
      diagnostics: [
        "Plot validation error against k. The curve should be U-shaped; if it only falls, your k range is too small.",
        "At k = 1 training accuracy is always 100% because every point is its own nearest neighbour. That number is meaningless.",
        "Inspect the distance to the k-th neighbour at prediction time. Unusually large values flag a query outside the training distribution.",
        "If accuracy jumps sharply after scaling, distance was previously being driven by one dominant feature.",
      ],
      advantages: [
        "No training phase at all, so new data can be added instantly without refitting.",
        "Makes no assumption about the shape of the boundary, so it can carve arbitrarily irregular regions.",
        "Naturally multi-class with no extra machinery.",
        "Conceptually transparent: you can point at the exact neighbours that drove any prediction.",
        "A single hyperparameter (k) does most of the work.",
      ],
      limitations: [
        { name: "Slow, memory-hungry predictions", note: "every query scans the stored dataset", fix: "KD-trees, approximate nearest neighbour indexes, or prototype selection." },
        { name: "Curse of dimensionality", note: "distances become uninformative as features multiply", fix: "PCA, feature selection, or a learned metric." },
        { name: "Scale-sensitive", note: "an unscaled large-range feature silently dominates the metric", fix: "standardize every feature." },
        { name: "Skewed by class imbalance", note: "the majority class wins neighbourhoods by sheer count", fix: "distance weighting or resampling." },
        { name: "No interpretable model", note: "there are no coefficients or rules to inspect globally", fix: "use a tree or linear model if you need one." },
      ],
      alternatives: [
        { name: "SVM with RBF kernel", when: "You want a non-linear boundary but fast predictions and a compact model." },
        { name: "Random forest", when: "Mixed feature types, higher dimensionality, or you need feature importances." },
        { name: "Approximate NN (FAISS, HNSW)", when: "The KNN idea is right but the dataset is too large for exact search." },
        { name: "Logistic regression", when: "The boundary is roughly linear; it will be faster and better calibrated." },
      ],
      pitfalls: [
        { problem: "Accuracy is poor despite plenty of data", solution: "Features are almost certainly unscaled. Standardize and retry." },
        { problem: "Boundary looks like noise", solution: "k is too small. Raise it until validation error bottoms out." },
        { problem: "Everything is predicted as one class", solution: "Class imbalance. Use distance weighting or rebalance the training set." },
        { problem: "Predictions are too slow in production", solution: "Build a KD-tree, or move to an approximate index." },
        { problem: "Perfect training accuracy reported", solution: "You measured k = 1 on the training set. Always evaluate on held-out data." },
      ],
      quickRef: [
        { name: "Euclidean distance", formula: "d = √Σⱼ(xⱼ − xᵢⱼ)²" },
        { name: "Manhattan distance", formula: "d = Σⱼ|xⱼ − xᵢⱼ|" },
        { name: "Classification", formula: "ŷ = mode{ yᵢ : i ∈ N_k(x) }" },
        { name: "Regression", formula: "ŷ = (1/k) Σ_{i∈N_k} yᵢ" },
        { name: "Distance weight", formula: "wᵢ = 1 / d(x, xᵢ)" },
        { name: "Predict cost", formula: "O(m·p) brute force" },
        { name: "Rule of thumb", formula: "k ≈ √m, odd for 2 classes" },
      ],
      code: `from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Scaling is not optional: the metric is only as good as the units.
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())

grid = GridSearchCV(
    pipe,
    {"kneighborsclassifier__n_neighbors": [1, 3, 5, 9, 15, 25],
     "kneighborsclassifier__weights": ["uniform", "distance"],
     "kneighborsclassifier__metric": ["euclidean", "manhattan"]},
    cv=5,
).fit(X_train, y_train)

print(grid.best_params_, grid.score(X_test, y_test))`,
      whyChain: [
        { q: "Why is KNN called a lazy learner?", a: "It does no work at training time beyond storing the data. All computation is deferred to prediction, which is the reverse of eager learners like logistic regression." },
        { q: "What actually happens as k grows?", a: "The boundary smooths out. Small k chases individual points (low bias, high variance); large k averages over a wide region (high bias, low variance). k = m degenerates to always predicting the majority class." },
        { q: "Why does KNN collapse in high dimensions?", a: "As p grows, the ratio between the nearest and farthest distances tends to 1. Every point becomes roughly equidistant, so the k nearest neighbours are no longer meaningfully nearer than anything else." },
        { q: "Why must you scale features?", a: "Euclidean distance sums squared differences across features. A feature measured in thousands contributes far more than one in the range 0 to 1, so it silently becomes the only feature that matters." },
        { q: "Is 100% training accuracy at k = 1 a good sign?", a: "No, it is a tautology. Each training point is its own nearest neighbour at distance zero, so it always predicts itself. Only held-out accuracy is informative." },
        { q: "When is distance weighting worth it?", a: "When you need a large k for stability but the neighbourhood spans regions of different density. Weighting lets the genuinely close points dominate the vote." },
        { q: "How do you make KNN practical on millions of rows?", a: "Give up exactness. Approximate indexes such as HNSW or LSH find near-neighbours in sublinear time with negligible accuracy loss." },
      ],
      parameters: [
        { name: "k", effect: "Neighborhood size. Small k → jagged, high-variance boundary; large k → smoother, higher-bias boundary." },
        { name: "distance metric", effect: "Defines 'closeness' (Euclidean here). Different metrics suit different feature geometries/scales." },
        { name: "vote weighting", effect: "Uniform vs distance-weighted votes - weighting lets very close neighbors count more than borderline ones." },
      ],
      workedExample: {
        setup: "Query point (3,2), k=3, training set A(1,1,red) B(2,1,red) C(4,3,blue) D(5,4,blue) E(1,2,red).",
        steps: [
          "Squared distances to (3,2): A=(2²+1²)=5, B=(1²+1²)=2, C=(1²+1²)=2, D=(2²+2²)=8, E=(2²+0²)=4.",
          "Sorted by distance: B (2, red), C (2, blue), E (4, red), A (5, red), D (8, blue).",
          "Take the k=3 nearest: B(red), C(blue), E(red).",
          "Vote count: red=2, blue=1.",
        ],
        result: "Majority vote → predicted class = red",
      },
    },
    mount,
  });
})();
