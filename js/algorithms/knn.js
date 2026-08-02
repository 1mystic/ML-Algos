(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];
  const N_CLASSES = 3;

  function classify(pt, points, k) {
    if (!points.length) return { label: 0, votes: [] };
    const withD = points.map((p) => ({ p, d: MLU.dist2(pt, p) })).sort((a, b) => a.d - b.d).slice(0, Math.min(k, points.length));
    const counts = new Array(N_CLASSES).fill(0);
    for (const { p } of withD) counts[p.label]++;
    let best = 0;
    for (let i = 1; i < N_CLASSES; i++) if (counts[i] > counts[best]) best = i;
    return { label: best, votes: counts };
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">
              ${[0, 1, 2].map((i) => `<span class="legend-item"><span class="swatch" style="background:${MLU.palette[i]}"></span>class ${i}</span>`).join("")}
            </div>
            <div class="btn-row"><button id="knn-regen">regenerate data</button><button id="knn-clear">clear</button></div>
          </div>
          <svg id="knn-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">pick a class below, then click the plot to add points · move the mouse to preview a live classification</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>add points as</h3>
            <div class="btn-row" id="knn-class-buttons">
              ${[0, 1, 2].map((i) => `<button data-c="${i}" style="border-color:${MLU.palette[i]}">${i}</button>`).join("")}
            </div>
          </div>
          <div class="control-card">
            <h3>k = <span class="val" id="knn-k-val">5</span></h3>
            <input type="range" id="knn-k" min="1" max="25" step="1" value="5" />
          </div>
          <div class="control-card">
            <h3>live query</h3>
            <div class="readout" id="knn-readout">move the mouse over the plot</div>
            <div class="note">Background shading = majority vote of the k nearest neighbors at every grid cell, recomputed as you add points or change k - the same brute-force vote as <code>mla/knn.py</code>.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#knn-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeBlobs({ n: 60, clusters: 3 }).map((p) => ({ x: p.x, y: p.y, label: p.label }));
    let currentClass = 0;
    const bgG = svg.append("g");
    const ptsG = svg.append("g");
    const cursor = svg.append("circle").attr("r", 7).attr("fill", "none").attr("stroke", "var(--text)").attr("stroke-width", 1.5).style("opacity", 0);

    function k() { return +document.getElementById("knn-k").value; }

    function render() {
      document.getElementById("knn-k-val").textContent = k();
      const cell = 14;
      const cells = [];
      for (let px = PAD; px < W - PAD; px += cell)
        for (let py = PAD; py < H - PAD; py += cell) {
          const xv = x.invert(px + cell / 2), yv = y.invert(py + cell / 2);
          const { label } = classify({ x: xv, y: yv }, points, k());
          cells.push({ px, py, label });
        }
      const rects = bgG.selectAll("rect").data(cells);
      rects.enter().append("rect").attr("width", cell).attr("height", cell)
        .merge(rects)
        .attr("x", (d) => d.px).attr("y", (d) => d.py)
        .attr("fill", (d) => MLU.palette[d.label]).attr("opacity", points.length ? 0.16 : 0);
      rects.exit().remove();

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle")
        .attr("r", 5).attr("stroke", "var(--bg)").attr("stroke-width", 1).style("cursor", "grab")
        .merge(sel)
        .attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y))
        .attr("fill", (d) => MLU.palette[d.label])
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); render(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          render();
        }));
      sel.exit().remove();
    }

    svg.on("click", (event) => {
      if (event.target.tagName === "circle") return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py), label: currentClass });
      render();
    });
    svg.on("mousemove", (event) => {
      const [px, py] = d3.pointer(event);
      if (px < PAD || px > W - PAD || py < PAD || py > H - PAD) { cursor.style("opacity", 0); return; }
      const pt = { x: x.invert(px), y: y.invert(py) };
      const { label, votes } = classify(pt, points, k());
      cursor.attr("cx", px).attr("cy", py).attr("stroke", MLU.palette[label]).style("opacity", points.length ? 1 : 0);
      if (points.length) {
        document.getElementById("knn-readout").innerHTML =
          `query → class <b style="color:${MLU.palette[label]}">${label}</b><br>votes: ${votes.map((v, i) => `<span style="color:${MLU.palette[i]}">${v}</span>`).join(" · ")}`;
      }
    });
    document.getElementById("knn-class-buttons").addEventListener("click", (e) => {
      const btn = e.target.closest("button"); if (!btn) return;
      currentClass = +btn.dataset.c;
      document.querySelectorAll("#knn-class-buttons button").forEach((b) => b.classList.toggle("primary", +b.dataset.c === currentClass));
    });
    document.querySelector("#knn-class-buttons button").classList.add("primary");
    document.getElementById("knn-k").addEventListener("input", render);
    document.getElementById("knn-regen").addEventListener("click", () => {
      points = MLU.makeBlobs({ n: 60, clusters: 3 }).map((p) => ({ x: p.x, y: p.y, label: p.label }));
      render();
    });
    document.getElementById("knn-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "knn",
    name: "K-Nearest Neighbors",
    category: "Supervised - Classification",
    tagline: "brute-force vote, live query",
    description: "Classifies a query point by majority vote among its k closest training points. Move the mouse to query live; the shaded background is that same vote evaluated on a grid.",
    sourceFile: "mla/knn.py",
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
