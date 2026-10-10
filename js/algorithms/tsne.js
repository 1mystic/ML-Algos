(() => {
  const { ui, col, fmt } = Lab;

  // ---- high-dimensional example data (labels are only used for colour)
  function makeData(kind, n) {
    const X = [], lab = [];
    if (kind === "clusters6" || kind === "clusters3") {
      const k = kind === "clusters6" ? 6 : 3, D = kind === "clusters6" ? 10 : 8;
      const C = Array.from({ length: k }, () => Array.from({ length: D }, () => MLU.randn() * 4));
      const spread = kind === "clusters3" ? [0.5, 1.5, 3] : new Array(k).fill(1);
      for (let i = 0; i < n; i++) { const c = i % k; X.push(C[c].map((v) => v + MLU.randn() * spread[c])); lab.push(c); }
    } else if (kind === "swiss") {
      for (let i = 0; i < n; i++) { const t = 1.5 * Math.PI * (1 + 2 * MLU.rng()), h = MLU.rng() * 12; X.push([t * Math.cos(t), h, t * Math.sin(t)]); lab.push(Math.min(6, Math.floor(((t - 1.5 * Math.PI) / (3 * Math.PI)) * 7))); }
    } else { // shells
      for (let i = 0; i < n; i++) { const c = i % 2, r = c ? 6 : 2; const v = [MLU.randn(), MLU.randn(), MLU.randn()]; const s = Math.hypot(...v); X.push(v.map((q) => (q / s) * r + MLU.randn() * 0.15)); lab.push(c); }
    }
    return { X, lab, D: X[0].length };
  }

  function affinities(X, perp) {
    const n = X.length;
    const D = X.map((a) => X.map((b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0)));
    const P = MLU.zeros(n, n), sig = [];
    const H = Math.log(perp);
    for (let i = 0; i < n; i++) {
      let lo = 1e-20, hi = 1e20, beta = 1;
      for (let it = 0; it < 60; it++) {
        let sum = 0, sumDP = 0;
        for (let j = 0; j < n; j++) if (j !== i) { const p = Math.exp(-D[i][j] * beta); P[i][j] = p; sum += p; sumDP += D[i][j] * p; }
        sum = sum || 1e-12;
        const h = Math.log(sum) + (beta * sumDP) / sum;
        for (let j = 0; j < n; j++) P[i][j] /= sum;
        if (Math.abs(h - H) < 1e-5) break;
        if (h > H) { lo = beta; beta = hi === 1e20 ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = (beta + lo) / 2; }
      }
      sig.push(Math.sqrt(1 / (2 * beta)));
    }
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = (P[i][j] + P[j][i]) / (2 * n); P[i][j] = P[j][i] = Math.max(v, 1e-12); }
    return { P, sig };
  }

  function run(X, { perp, lr, iters, exag }) {
    const n = X.length;
    const { P, sig } = affinities(X, perp);
    let Y = Array.from({ length: n }, () => [MLU.randn() * 1e-2, MLU.randn() * 1e-2]);
    const V = Y.map(() => [0, 0]), G = Y.map(() => [1, 1]);
    const frames = [];
    const every = Math.max(1, Math.round(iters / 70));
    for (let t = 0; t <= iters; t++) {
      const ex = exag && t < 100 ? 12 : 1;
      const num = MLU.zeros(n, n); let Z = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const q = 1 / (1 + (Y[i][0] - Y[j][0]) ** 2 + (Y[i][1] - Y[j][1]) ** 2); num[i][j] = num[j][i] = q; Z += 2 * q; }
      if (t % every === 0 || t === iters) {
        let kl = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) kl += P[i][j] * Math.log(P[i][j] / Math.max(num[i][j] / Z, 1e-12));
        frames.push({ Y: Y.map((y) => y.slice()), t, ex, kl });
      }
      const mom = t < 250 ? 0.5 : 0.8;
      // compute every gradient from the same snapshot, then update all points
      const grad = Y.map((yi, i) => {
        let gx = 0, gy = 0;
        for (let j = 0; j < n; j++) if (i !== j) { const m = (ex * P[i][j] - num[i][j] / Z) * num[i][j]; gx += 4 * m * (yi[0] - Y[j][0]); gy += 4 * m * (yi[1] - Y[j][1]); }
        return [gx, gy];
      });
      for (let i = 0; i < n; i++) for (let d = 0; d < 2; d++) {
        const g = grad[i][d];
        G[i][d] = Math.sign(g) !== Math.sign(V[i][d]) ? G[i][d] + 0.2 : Math.max(0.01, G[i][d] * 0.8);
        V[i][d] = mom * V[i][d] - lr * G[i][d] * g;
        Y[i][d] += V[i][d];
      }
      const my = [MLU.mean(Y.map((y) => y[0])), MLU.mean(Y.map((y) => y[1]))];
      Y = Y.map((y) => [y[0] - my[0], y[1] - my[1]]);
    }
    return { frames, P, sig };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "tsne", fps: 8, plane: { ticks: false },
      legend: ui.legend([["var(--text-faint)", "colour = true group (never shown to t-SNE)"]]),
      tools: ui.btn("rerun", "rerun (new init)", true),
      hint: "the axes of a t-SNE map have no meaning; only neighbourhoods do",
      side: [
        { title: "input data", html: ui.select("ds", "high-dimensional dataset", [["clusters6", "6 clusters in 10-D"], ["clusters3", "3 clusters, unequal spread, 8-D"], ["swiss", "swiss roll (3-D manifold)"], ["shells", "two nested spheres (3-D)"]], "clusters6") + ui.range("n", "points", 40, 200, 10, 120) },
        { title: "t-SNE", html: ui.range("perp", "perplexity", 2, 50, 1, 20) + ui.range("lr", "learning rate", 10, 500, 10, 150) + ui.range("it", "iterations", 100, 1000, 50, 500) + ui.check("exag", "early exaggeration (×12, first 100 its)", true) },
      ],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "X", label: "X", sub: "n × D", x: 34, y: 40, w: 56, kind: "data" },
          { id: "sig", label: "σᵢ search", sub: "match perplexity", x: 140, y: 40, kind: "param" },
          { id: "P", label: "P (Gaussian)", sub: "high-D affinities", x: 262, y: 40, kind: "op" },
          { id: "Y", label: "Y ~ N(0, 10⁻⁴)", sub: "2-D init", x: 70, y: 112, kind: "param" },
          { id: "Q", label: "Q (Student-t)", sub: "low-D affinities", x: 200, y: 112, kind: "op" },
          { id: "kl", label: "KL(P‖Q)", sub: "gradient", x: 318, y: 112, kind: "loss" },
          { id: "upd", label: "Y ← Y − η∇", sub: "momentum + gains", x: 200, y: 168, kind: "op" },
        ],
        edges: [{ from: "X", to: "sig" }, { from: "sig", to: "P" }, { from: "P", to: "kl" }, { from: "Y", to: "Q" }, { from: "Q", to: "kl" }, { from: "kl", to: "upd" }, { from: "upd", to: "Q", dash: true, label: "iterate" }],
      },
      phases: { exag: { label: "early exaggeration", color: 3 }, opt: { label: "optimise", color: 0 }, done: { label: "done", color: 2 } },
      panels: [{ id: "P", title: "input affinities P", sub: "rows sorted by group" }, { id: "pca", title: "for comparison: PCA of the same data", sub: "linear projection" }],
      charts: [{ id: "kl", title: "KL divergence", series: [{ key: "kl", label: "KL(P‖Q)", color: "var(--series-1)" }] }],
      onFrame: render,
    });
    const pl = lab.plane, L = pl.layers;
    let data = null, R = null, seed = 3;
    function regenerate() { MLU.seedRng(41); data = makeData(lab.val("ds"), lab.val("n")); rerun(); }
    function rerun() {
      MLU.seedRng(seed);
      R = run(data.X, { perp: Math.min(lab.val("perp"), (data.X.length - 1) / 3), lr: lab.val("lr"), iters: lab.val("it"), exag: lab.val("exag") });
      const frames = R.frames.map((fr, k) => ({
        phase: fr.t === lab.val("it") ? "done" : fr.ex > 1 ? "exag" : "opt", major: k === 0 || (fr.ex === 1 && R.frames[k - 1] && R.frames[k - 1].ex > 1),
        arch: fr.t === 0 ? ["X", "sig", "P", "Y"] : ["Q", "kl", "upd"],
        title: fr.t === 0 ? `Iteration 0: points start in a tiny random cloud` : fr.ex > 1 ? `Iteration ${fr.t}: early exaggeration` : fr.t === lab.val("it") ? `Done after ${fr.t} iterations` : `Iteration ${fr.t}: refining the map`,
        text: fr.t === 0 ? `Each point's Gaussian width σᵢ was tuned so its neighbourhood has perplexity ≈ ${lab.val("perp")} (median σ = ${d3.median(R.sig).toFixed(2)}). P holds those high-dimensional similarities.` : fr.ex > 1 ? "P is temporarily multiplied by 12, so true neighbours attract very strongly. Clusters form early and get room to move past each other before fine structure is fixed." : `Attraction (P) pulls neighbours together; the heavy-tailed Student-t in Q lets dissimilar points sit far apart without much penalty, which avoids the crowding problem. KL = <b>${fr.kl.toFixed(3)}</b>.`,
        tex: fr.t === 0 ? "p_{j|i}=\\frac{e^{-\\lVert x_i-x_j\\rVert^2/2\\sigma_i^2}}{\\sum_{k\\ne i}e^{-\\lVert x_i-x_k\\rVert^2/2\\sigma_i^2}},\\quad p_{ij}=\\frac{p_{j|i}+p_{i|j}}{2n}" : "\\frac{\\partial C}{\\partial y_i}=4\\sum_j (p_{ij}-q_{ij})\\,\\frac{y_i-y_j}{1+\\lVert y_i-y_j\\rVert^2},\\quad q_{ij}=\\frac{(1+\\lVert y_i-y_j\\rVert^2)^{-1}}{\\sum_{k\\ne l}(1+\\lVert y_k-y_l\\rVert^2)^{-1}}",
        st: fr, m: { kl: fr.kl },
      }));
      drawP(); drawPCA();
      lab.load(frames, { play: true });
    }
    function drawP() {
      const el = lab.R("panel-P"); el.innerHTML = "";
      const n = data.X.length, order = [...Array(n).keys()].sort((a, b) => data.lab[a] - data.lab[b]);
      const cv = document.createElement("canvas"); cv.width = n; cv.height = n; cv.style.width = "100%"; cv.style.maxWidth = "260px"; cv.style.imageRendering = "pixelated"; cv.style.borderRadius = "4px";
      const ctx = cv.getContext("2d"), img = ctx.createImageData(n, n);
      let mx = 0; for (const r of R.P) for (const v of r) mx = Math.max(mx, v);
      const c = Lab.palRGB()[0];
      const bg = Lab.rgb(getComputedStyle(document.documentElement).getPropertyValue("--bg-inset").trim() || "#202020");
      order.forEach((i, a) => order.forEach((j, b) => { const t = Math.sqrt(R.P[i][j] / mx); const k = (a * n + b) * 4; for (let q = 0; q < 3; q++) img.data[k + q] = bg[q] + (c[q] - bg[q]) * t; img.data[k + 3] = 255; }));
      ctx.putImageData(img, 0, 0);
      el.appendChild(cv);
      const note = document.createElement("div"); note.className = "lab-card-note"; note.textContent = "Bright blocks on the diagonal = groups of mutual neighbours in the original space. t-SNE tries to reproduce this matrix in 2-D.";
      el.appendChild(note);
    }
    function drawPCA() {
      const el = lab.R("panel-pca"); el.innerHTML = "";
      const X = data.X, mu = MLU.meanVec(X), Z = X.map((x) => x.map((v, i) => v - mu[i]));
      const C = MLU.covMatrix(Z, new Array(Z[0].length).fill(0));
      const E = MLU.jacobiEigen(C);
      const pr = Z.map((z) => [E.vectors[0], E.vectors[1]].map((v) => v.reduce((s, w, i) => s + w * z[i], 0)));
      const ex = d3.extent(pr, (p) => p[0]), ey = d3.extent(pr, (p) => p[1]);
      const P = Lab.xy(el, { w: 300, h: 180, xd: ex, yd: ey, xTicks: 0, yTicks: 0, m: { l: 8, r: 8, t: 8, b: 8 } });
      P.g.selectAll("circle").data(pr).enter().append("circle").attr("cx", (p) => P.x(p[0])).attr("cy", (p) => P.y(p[1])).attr("r", 2.6).attr("fill", (p, i) => col(data.lab[i])).attr("opacity", 0.85);
    }
    const ptG = L.pts.append("g");
    function render(f) {
      const Y = f.st.Y;
      const ext = Math.max(1e-3, ...Y.map((y) => Math.max(Math.abs(y[0]), Math.abs(y[1])))) * 1.1;
      const sx = (v) => (v / ext) * 9.5;
      const c = ptG.selectAll("circle").data(Y);
      c.exit().remove();
      c.enter().append("circle").attr("r", 4.2).attr("stroke", "var(--bg)").attr("stroke-width", 1).merge(c)
        .transition().duration(110).ease(d3.easeLinear)
        .attr("cx", (y) => pl.x(sx(y[0]))).attr("cy", (y) => pl.y(sx(y[1]))).attr("fill", (y, i) => col(data.lab[i]));
      lab.readout(`<div class="kv"><span>points × dims</span><b>${data.X.length} × ${data.D}</b><span>iteration</span><b>${f.st.t}</b><span>KL(P‖Q)</span><b>${fmt(f.st.kl)}</b><span>perplexity</span><b>${lab.val("perp")}</b></div>`);
    }
    lab.watch(["ds", "n"], () => regenerate());
    lab.watch(["perp", "lr", "it", "exag"], () => rerun());
    lab.q('[data-b="rerun"]').onclick = () => { seed = (seed * 31 + 7) % 9973; rerun(); };
    regenerate();
    return lab.destroy;
  }

  MLApp.register({
    id: "tsne",
    name: "t-SNE",
    category: "Unsupervised - Dimensionality Reduction",
    tagline: "perplexity, P vs Q, animated embedding",
    description: "Embed high-dimensional data in 2-D by matching neighbour probabilities. Watch points spread from a tiny random cloud through early exaggeration into clusters, compare with PCA of the same data, and see the input affinity matrix t-SNE is trying to reproduce.",
    info: {
      type: "Unsupervised - Non-linear dimensionality reduction, visualization-focused (no out-of-sample mapping).",
      scenario: "Visualizing high-dimensional data (embeddings, gene expression, learned features) in 2D/3D so that clusters close in high-D stay close in the 2D map.",
      inputs: "A set of high-dimensional points - unsupervised; labels, if any, are only used afterward to color the plot.",
      intuition: {
        definition: "Turn distances into <b>neighbour probabilities</b> in both the original space and the 2D map, then move the 2D points around until the two probability distributions match. It is a layout algorithm optimised for one thing: keeping neighbours together.",
        steps: [
          "In high-D, convert distances to probabilities with a per-point Gaussian.",
          "In 2D, do the same with a heavy-tailed Student-t kernel.",
          "Measure the mismatch with KL divergence.",
          "Gradient-descend on the 2D coordinates until it settles.",
        ],
        applications: [
          "Visualising word and sentence embeddings",
          "Single-cell RNA sequencing cluster maps",
          "Inspecting what a neural network's hidden layers learned",
          "Exploratory discovery of subgroups before formal clustering",
          "Sanity-checking whether classes are separable at all",
        ],
      },
      math: [
        { title: "High-D conditional affinity", formula: "p_{j|i} = exp(−‖xᵢ−xⱼ‖²/2σᵢ²) / Σ_{k≠i} exp(−‖xᵢ−xₖ‖²/2σᵢ²)", note: "A Gaussian centred on each point, converted to a probability distribution over its neighbours. Note σᵢ varies per point." },
        { title: "Perplexity calibration", formula: "Perp(Pᵢ) = 2^{H(Pᵢ)} = target,   H(Pᵢ) = −Σⱼ p_{j|i} log₂ p_{j|i}", note: "σᵢ is binary-searched so every point has the same effective neighbour count. This is what adapts the method to varying local density." },
        { title: "Symmetrise", formula: "pᵢⱼ = (p_{j|i} + p_{i|j}) / (2m)", note: "Makes the affinity matrix symmetric and guarantees every point contributes to the cost, so outliers cannot be ignored." },
        { title: "Low-D affinity", formula: "qᵢⱼ = (1 + ‖yᵢ−yⱼ‖²)⁻¹ / Σ_{k≠l}(1 + ‖yₖ−y_l‖²)⁻¹", note: "A Student-t with one degree of freedom. Its heavy tail is the key design choice, and it is what the 't' in t-SNE refers to." },
        { title: "Cost", formula: "KL(P‖Q) = Σᵢ Σⱼ pᵢⱼ · log(pᵢⱼ / qᵢⱼ)", note: "Asymmetric on purpose: a large p paired with a small q is punished heavily, but a small p with large q costs little. That is why local structure is preserved and global structure is not." },
        { title: "Gradient", formula: "∂C/∂yᵢ = 4 Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ − yⱼ)(1 + ‖yᵢ−yⱼ‖²)⁻¹", note: "Reads as a spring system: attraction where p exceeds q, repulsion where q exceeds p." },
      ],
      pipeline: [
        { label: "High-D points", note: "often PCA'd to 50D" },
        { label: "P from Gaussians", note: "perplexity-calibrated" },
        { label: "Random 2D init", note: "or PCA init" },
        { label: "Q from Student-t", note: "heavy tail" },
        { label: "Descend KL(P‖Q)", note: "2D map", accent: "green" },
      ],
      decisionFunction: {
        text: "yᵢ is found by gradient descent, not computed by a formula, so that low-D neighbor probabilities Q match high-D neighbor probabilities P",
        mechanism: "High-D neighbor probability Pᵢⱼ uses a Gaussian kernel calibrated per-point to a target perplexity; low-D probability Qᵢⱼ uses a heavier-tailed Student-t kernel - the heavy tail is what lets moderately-distant 2D points still represent moderately-similar high-D points without everything collapsing into one blob.",
      },
      lossFunction: {
        text: "KL(P‖Q) = Σᵢⱼ Pᵢⱼ·log(Pᵢⱼ/Qᵢⱼ)",
        mechanism: "Minimized by gradient descent with momentum directly on the 2D coordinates: points that should be neighbors (high P) but currently aren't (low Q) get pulled together; points that shouldn't be neighbors get pushed apart.",
      },
      optimization: [
        { title: "Early exaggeration", formula: "multiply P by ~12 for the first ~250 iterations", note: "Inflating the attractive forces early forces tight clusters to form with plenty of empty space between them, so they do not get trapped overlapping." },
        { title: "Momentum schedule", formula: "0.5 early, 0.8 after exaggeration ends", note: "Low momentum while the layout is chaotic, higher once it is settling, which speeds convergence without oscillation." },
        { title: "Barnes-Hut approximation", formula: "O(m²) → O(m log m)", note: "Distant points are grouped into a quadtree cell and treated as one repulsive body. Standard for datasets above a few thousand points." },
        { title: "Cost is non-convex", formula: "different seeds give different maps", note: "There is no single correct layout. Run it more than once and only trust structure that reappears." },
        { title: "PCA first", formula: "reduce to ~50 dimensions before t-SNE", note: "Cuts noise and cost substantially, and is standard practice. It also makes the initial neighbour graph more reliable." },
      ],
      output: "A 2D (or 3D) coordinate for every input point, suitable for a scatter plot - not a reusable function for new points.",
      assumptions: [
        { name: "Only local structure matters", why: "The KL cost barely penalises misplaced distant points, so between-cluster distances in the map are not meaningful.", check: "Never interpret the gap between two clusters as a similarity measure." },
        { name: "You want a picture, not features", why: "There is no learned mapping, so the output cannot be applied to new data.", check: "For a reusable embedding use PCA, UMAP, or an autoencoder." },
        { name: "Perplexity suits the data size", why: "Perplexity is an effective neighbour count and must be well below the number of points.", check: "Keep perplexity under m/3; typical values are 5 to 50." },
        { name: "Dimensionality pre-reduced", why: "Raw high-dimensional distances are noisy and slow to compute.", check: "Apply PCA to roughly 50 dimensions first." },
        { name: "Enough iterations", why: "Stopping early leaves a half-formed layout that looks like structure but is not.", check: "Run at least 1000 iterations and confirm the KL divergence has plateaued." },
      ],
      hyperparameters: [
        { name: "perplexity", range: "5 - 50", increasing: "Wider neighbourhoods, more global structure retained, clusters merge and soften.", strategy: "The main knob. Try 5, 30, and 50 and only trust structure that survives all three. Must be less than the point count." },
        { name: "n_iter", range: "1000 - 5000", increasing: "More refined layout, diminishing returns once KL plateaus.", strategy: "1000 minimum. Increase if the map still looks unsettled." },
        { name: "learning_rate", range: "10 - 1000 (or 'auto')", increasing: "Faster separation, risk of points flying apart into a diffuse ball.", strategy: "Use 'auto' (roughly m/12). A dense ball with everything compressed usually means it is too low." },
        { name: "early_exaggeration", range: "4 - 24", increasing: "Larger gaps between clusters, more distinct separation.", strategy: "12 is the default and rarely needs changing." },
        { name: "init", range: "pca / random", increasing: "Not applicable", strategy: "Use PCA initialisation. It is more reproducible and preserves more global structure than a random start." },
        { name: "metric", range: "euclidean / cosine / correlation", increasing: "Not applicable", strategy: "Cosine for embeddings and text, euclidean for most other numeric data." },
      ],
      metrics: ["No single 'accuracy' - trustworthiness / continuity scores", "Final KL divergence (compare only across runs on the same data)", "Qualitative check: do known clusters/classes separate in the map?"],
      typicalUses: ["Visualizing word/sentence embeddings", "Single-cell genomics cluster visualization", "Inspecting a neural network's learned feature space", "Exploratory cluster discovery"],
      diagnostics: [
        "Run several perplexities and several seeds. Structure that appears in all of them is real; structure that appears once is an artifact.",
        "A uniformly spread ball of points usually means the learning rate is too low or the iteration count too small.",
        "Many tiny scattered clumps often means perplexity is too low and the algorithm is fitting noise.",
        "Check that KL divergence has plateaued. A still-falling curve means the layout is unfinished.",
        "Never measure cluster sizes or between-cluster distances off the map. Both are artifacts of the algorithm, not properties of the data.",
      ],
      advantages: [
        "Exceptionally good at revealing local cluster structure that linear methods miss entirely.",
        "Handles non-linear manifolds that PCA flattens incorrectly.",
        "Per-point perplexity calibration adapts automatically to regions of different density.",
        "The Student-t kernel solves the crowding problem, so clusters separate cleanly instead of collapsing.",
        "Produces genuinely useful exploratory pictures of embeddings and genomics data.",
      ],
      limitations: [
        { name: "No out-of-sample mapping", note: "new points require rerunning the whole optimisation", fix: "UMAP, parametric t-SNE, or an autoencoder." },
        { name: "Global structure is not preserved", note: "distances between clusters are meaningless", fix: "UMAP retains more global structure; check PCA for the overall geometry." },
        { name: "Cluster sizes are meaningless", note: "the algorithm equalises density, so a tight cluster may render large", fix: "never read size off the plot." },
        { name: "Stochastic", note: "different seeds produce different maps", fix: "PCA initialisation and multiple runs." },
        { name: "Slow", note: "O(m²) naive, and still costly with Barnes-Hut", fix: "PCA first, or use UMAP which is much faster." },
        { name: "Can invent apparent clusters", note: "random data will still produce clumps at low perplexity", fix: "vary perplexity and check the structure persists." },
      ],
      alternatives: [
        { name: "UMAP", when: "Almost always the better default now: faster, preserves more global structure, and supports transforming new points." },
        { name: "PCA", when: "You need a fast, deterministic, interpretable, reusable linear projection." },
        { name: "Autoencoder", when: "You want a learned non-linear embedding usable as features on new data." },
        { name: "PHATE / diffusion maps", when: "The data has continuous trajectory structure rather than discrete clusters." },
      ],
      pitfalls: [
        { problem: "Reading meaning into gaps between clusters", solution: "Between-cluster distance is not preserved. This is the single most common misinterpretation." },
        { problem: "Comparing cluster sizes in the map", solution: "Also meaningless. t-SNE equalises local density by construction." },
        { problem: "Map is a featureless ball", solution: "Learning rate too low or too few iterations. Use 'auto' and at least 1000 steps." },
        { problem: "Different results every run", solution: "Expected, since the cost is non-convex. Use PCA init, fix the seed, and confirm structure across runs." },
        { problem: "Trying to transform a test set", solution: "There is no transform method by design. Use UMAP if you need one." },
        { problem: "Running t-SNE on raw 10000-dimensional data", solution: "Apply PCA to about 50 dimensions first, for both speed and noise reduction." },
      ],
      quickRef: [
        { name: "High-D affinity", formula: "p_{j|i} ∝ exp(−‖xᵢ−xⱼ‖²/2σᵢ²)" },
        { name: "Symmetrised", formula: "pᵢⱼ = (p_{j|i}+p_{i|j})/2m" },
        { name: "Perplexity", formula: "Perp = 2^{H(Pᵢ)}" },
        { name: "Low-D affinity", formula: "qᵢⱼ ∝ (1+‖yᵢ−yⱼ‖²)⁻¹" },
        { name: "Cost", formula: "KL(P‖Q) = Σ pᵢⱼ log(pᵢⱼ/qᵢⱼ)" },
        { name: "Gradient", formula: "4Σⱼ(pᵢⱼ−qᵢⱼ)(yᵢ−yⱼ)(1+‖yᵢ−yⱼ‖²)⁻¹" },
        { name: "Barnes-Hut", formula: "O(m log m)" },
        { name: "Meaningless in the map", formula: "cluster size, inter-cluster distance" },
      ],
      code: `from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Standard practice: PCA down to ~50D first for speed and denoising.
X50 = PCA(n_components=50, random_state=42).fit_transform(X)

# Vary perplexity: only trust structure that survives all of them.
for perp in (5, 30, 50):
    Y = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",     # roughly m/12
        init="pca",               # more reproducible than random
        n_iter=1000,
        random_state=42,
    ).fit_transform(X50)
    # scatter(Y[:, 0], Y[:, 1], c=labels)

# There is deliberately no .transform(): t-SNE cannot embed new points.
# If you need that, use UMAP:
# import umap; um = umap.UMAP(n_neighbors=15, min_dist=0.1).fit(X50)
# um.transform(X_new)`,
      whyChain: [
        { q: "Why use a Student-t distribution in the low-dimensional map?", a: "It solves the crowding problem. Two dimensions have far less room than the original space, so moderately distant points cannot all be placed at proportionate distances. The t-distribution's heavy tail assigns a reasonable probability even to fairly separated 2D points, so clusters can spread apart instead of collapsing into one blob." },
        { q: "What is the crowding problem exactly?", a: "The volume of a ball grows exponentially with dimension. In 50 dimensions a point can have many neighbours all roughly equidistant; in 2D there is simply not enough area to place them all at that distance. Something has to give, and with a Gaussian in both spaces everything gets crushed together." },
        { q: "What does perplexity actually control?", a: "It is a smooth measure of the effective number of neighbours each point considers. The algorithm binary-searches a separate bandwidth σᵢ per point so that every point achieves the same perplexity, which is how it adapts to regions of differing density." },
        { q: "Why is KL divergence asymmetric, and does it matter?", a: "It matters enormously. KL(P‖Q) heavily penalises placing true neighbours far apart, but barely penalises placing distant points close together. That asymmetry is exactly why local structure is faithful and global structure is not." },
        { q: "Why can't you interpret the distance between two clusters?", a: "Because the cost function does not care. Once two points are far apart in the map, moving them further apart or closer together changes the KL divergence very little. The layout of clusters relative to each other is effectively arbitrary." },
        { q: "Why does t-SNE have no transform method?", a: "It does not learn a function. It optimises the coordinates of each specific point directly, so there is no mapping to apply to unseen data. Embedding a new point would require rerunning the optimisation with it included." },
        { q: "What is early exaggeration for?", a: "Multiplying P early in training inflates the attractive forces, so clusters form tightly and push apart from each other, leaving empty space. Without it, clusters tend to get stuck overlapping in a local optimum they cannot escape." },
        { q: "t-SNE or UMAP?", a: "UMAP for most purposes now. It is substantially faster, preserves more global structure, and can transform new data. t-SNE remains a strong, well-understood choice for pure local-structure visualisation." },
      ],
      parameters: [
        { name: "perplexity", effect: "Roughly 'how many neighbors' each point's Gaussian considers. Small → emphasizes very local structure; large → considers broader neighborhoods." },
        { name: "iterations", effect: "More iterations refine the layout further, with diminishing returns after clusters separate." },
        { name: "learning rate / momentum", effect: "Affect how quickly and how distinctly clusters separate early in training." },
      ],
      metrics: ["No single 'accuracy' - trustworthiness / continuity scores", "Qualitative check: do known clusters/classes separate in the map?"],
      typicalUses: ["Visualizing word/sentence embeddings", "Single-cell genomics cluster visualization", "Inspecting a neural network's learned feature space", "Exploratory cluster discovery"],
      workedExample: {
        setup: "3 points in 1D: x=0, x=1, x=5. Compute p_{j|0} (neighbor probability of points 1 and 2, given point 0) with β=1/(2σ²)=0.5.",
        steps: [
          "Squared distances from point 0: d(0,1)²=1, d(0,2)²=25.",
          "Unnormalized affinities: exp(−0.5×1)=0.6065, exp(−0.5×25)=exp(−12.5)≈3.7×10⁻⁶.",
          "Sum = 0.6065 + 0.0000037 ≈ 0.60651.",
          "p_{1|0} = 0.6065/0.60651 ≈ 0.99994. p_{2|0} = 0.0000037/0.60651 ≈ 0.0000061.",
        ],
        result: "Point 1 (close) gets ~99.99% of point 0's neighbor probability mass; point 2 (far) gets essentially none - exactly the local-neighborhood emphasis t-SNE is built on",
      },
    },
    mount,
  });
})();
