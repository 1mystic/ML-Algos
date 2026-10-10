(() => {
  const { ui, col, fmt } = Lab;

  function analyse(pts, standardise) {
    const X = pts.map((p) => [p.x, p.y]);
    const mu = MLU.meanVec(X);
    let Z = X.map((x) => [x[0] - mu[0], x[1] - mu[1]]);
    let sd = [1, 1];
    if (standardise) { sd = [0, 1].map((j) => Math.sqrt(MLU.mean(Z.map((z) => z[j] ** 2))) || 1); Z = Z.map((z) => [z[0] / sd[0] * 3, z[1] / sd[1] * 3]); }
    const C = MLU.covMatrix(Z, [0, 0]);
    const E = MLU.jacobiEigen(C);
    let v1 = E.vectors[0]; if (v1[0] < 0) v1 = [-v1[0], -v1[1]];
    const v2 = [-v1[1], v1[0]];
    return { mu, Z, C, vals: E.values, v1, v2, sd };
  }
  const projVar = (Z, w) => MLU.mean(Z.map((z) => (z[0] * w[0] + z[1] * w[1]) ** 2));

  function buildFrames(pts, standardise) {
    if (pts.length < 3) return { frames: [{ phase: "idle", title: "Add at least three points", text: "", st: {} }], A: null };
    const A = analyse(pts, standardise);
    const tot = A.vals[0] + A.vals[1];
    const frames = [];
    frames.push({ phase: "raw", major: true, arch: ["X"], title: "Raw data with its mean (✚)", text: "PCA looks for the directions along which the data varies most. Variance is measured around the mean, so the first job is to centre the cloud.", st: { t: 0 } });
    for (const t of [0.34, 0.67, 1]) frames.push({ phase: "center", arch: ["X", "center"], title: "Centre the data: subtract the mean", text: `Shifting every point by −μ = (${(-A.mu[0]).toFixed(2)}, ${(-A.mu[1]).toFixed(2)}) puts the mean at the origin.${standardise ? " Standardisation also divides each feature by its standard deviation, so features in large units cannot dominate." : ""}`, tex: standardise ? "z_{ij}=\\frac{x_{ij}-\\mu_j}{\\sigma_j}" : "z_i=x_i-\\mu", st: { t } });
    frames.push({ phase: "cov", major: true, arch: ["center", "cov"], title: "Covariance matrix: the shape of the cloud", text: `Diagonal entries are each feature's variance; off-diagonals measure how they move together (here ${A.C[0][1] >= 0 ? "positively" : "negatively"} correlated). The ellipse is the 2σ contour of a Gaussian with this covariance.`, tex: `\\Sigma=\\tfrac1n Z^\\top Z=\\begin{bmatrix}${A.C[0][0].toFixed(2)} & ${A.C[0][1].toFixed(2)}\\\\ ${A.C[1][0].toFixed(2)} & ${A.C[1][1].toFixed(2)}\\end{bmatrix}`, st: { t: 1, ell: true } });
    for (let deg = 0; deg <= 180; deg += 6) {
      const th = (deg * Math.PI) / 180, w = [Math.cos(th), Math.sin(th)];
      const v = projVar(A.Z, w);
      frames.push({ phase: "scan", arch: ["cov", "obj"], title: `Try a direction: θ = ${deg}°`, text: `Project every point onto the line. Variance along it: <b>${v.toFixed(2)}</b> (${((v / tot) * 100).toFixed(0)}% of total). The red sticks are what is lost: their mean square is the reconstruction error, and the two always add up to the total variance, so maximising one minimises the other.`, tex: "\\operatorname{Var}(w)=\\frac1n\\sum_i (w^\\top z_i)^2=w^\\top\\Sigma w,\\qquad \\lVert w\\rVert=1", st: { t: 1, ell: true, w, th: deg } });
    }
    const deg1 = ((Math.atan2(A.v1[1], A.v1[0]) * 180) / Math.PI + 180) % 180;
    frames.push({ phase: "eig", major: true, arch: ["obj", "eig", "sort"], title: `The answer is the top eigenvector: PC1 at ${deg1.toFixed(1)}°`, text: `Maximising wᵀΣw subject to ‖w‖ = 1 gives Σw = λw. PC1 has λ₁ = <b>${A.vals[0].toFixed(2)}</b> (${((A.vals[0] / tot) * 100).toFixed(1)}% of variance); PC2 is orthogonal with λ₂ = <b>${A.vals[1].toFixed(2)}</b>. Arrow lengths are 2√λ.`, tex: "\\Sigma\\,v_k=\\lambda_k v_k,\\qquad \\lambda_1\\ge\\lambda_2,\\qquad \\text{explained ratio}_k=\\frac{\\lambda_k}{\\sum_j\\lambda_j}", st: { t: 1, ell: true, pcs: true, w: A.v1, th: deg1 } });
    for (const s of [0.25, 0.5, 0.75, 1]) frames.push({ phase: "proj", arch: ["eig", "proj"], title: "Keep 1 component: project onto PC1", text: `Each point is replaced by its coordinate along PC1 (one number instead of two). Reconstruction error: <b>${A.vals[1].toFixed(2)}</b> per point, exactly the discarded eigenvalue.`, tex: "\\hat z_i=(v_1^\\top z_i)\\,v_1,\\qquad \\frac1n\\sum_i\\lVert z_i-\\hat z_i\\rVert^2=\\lambda_2", st: { t: 1, pcs: true, w: A.v1, collapse: s } });
    frames.push({ phase: "done", major: true, arch: ["proj", "out"], title: "PCA = rotate to the eigenbasis, then drop the small axes", text: "In d dimensions the recipe is identical: the top-k eigenvectors of the covariance span the k-dimensional subspace that keeps the most variance. In practice it is computed with an SVD of the centred data matrix, which is more stable than forming Σ.", tex: "Z=U S V^\\top,\\qquad \\text{PCs}=V,\\qquad \\lambda_k=s_k^2/n", st: { t: 1, pcs: true, w: A.v1, collapse: 1 } });
    return { frames, A };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "pca", fps: 4,
      legend: ui.legend([["var(--accent)", "candidate direction", "line"], ["var(--series-2)", "residual", "line"], ["var(--series-1)", "PC1"], ["var(--series-5)", "PC2"]]),
      tools: ui.btn("replay", "replay", true),
      hint: "click to add points · drag / double-click to edit",
      data: {
        kind: "cluster", presets: ["corr", "line", "aniso", "blobs2", "uniform", "moons"], def: "corr", n: 80, noise: 0.3, names: { corr: "correlated Gaussian", line: "nearly one-dimensional" },
        custom: {
          corr: ({ n, noise }) => Array.from({ length: n }, () => { const a = MLU.randn() * 3.6, b = MLU.randn() * (0.6 + noise * 2.4); return { x: 1 + a * 0.83 - b * 0.56, y: -0.5 + a * 0.56 + b * 0.83, label: 0 }; }),
          line: ({ n, noise }) => Array.from({ length: n }, () => { const a = MLU.randRange(-8, 8), b = MLU.randn() * (0.15 + noise * 0.8); return { x: a * 0.94 - b * 0.34, y: a * 0.34 + b * 0.94 - 1, label: 0 }; }),
        },
      },
      side: [{ title: "options", html: ui.check("std", "standardise features first", false) + ui.check("resid", "show residual sticks", true) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "X", label: "X", sub: "n × d", x: 30, y: 44, w: 48, kind: "data" },
          { id: "center", label: "centre", sub: "X − μ", x: 112, y: 44, kind: "op" },
          { id: "cov", label: "Σ = ZᵀZ / n", sub: "covariance", x: 222, y: 44, kind: "op" },
          { id: "obj", label: "max wᵀΣw", sub: "‖w‖ = 1", x: 330, y: 44, kind: "loss" },
          { id: "eig", label: "eigendecompose", sub: "Σv = λv", x: 300, y: 120, kind: "op" },
          { id: "sort", label: "sort by λ", sub: "keep top k", x: 170, y: 120, kind: "param" },
          { id: "proj", label: "Z·V_k", sub: "project", x: 60, y: 120, kind: "op" },
          { id: "out", label: "scores", sub: "n × k", x: 60, y: 172, kind: "out" },
        ],
        edges: [{ from: "X", to: "center" }, { from: "center", to: "cov" }, { from: "cov", to: "obj" }, { from: "obj", to: "eig" }, { from: "eig", to: "sort" }, { from: "sort", to: "proj" }, { from: "proj", to: "out" }],
      },
      phases: { raw: { label: "raw data", color: 3 }, center: { label: "centre", color: 3 }, cov: { label: "covariance", color: 4 }, scan: { label: "direction scan", color: 0 }, eig: { label: "eigenvectors", color: 2 }, proj: { label: "project", color: 2 }, done: { label: "done", color: 2 } },
      panels: [{ id: "cov", title: "covariance matrix & explained variance" }],
      charts: [{ id: "var", title: "variance captured vs lost, by direction θ", custom: true }],
      onFrame: render, onData: () => replay(),
    });
    lab.editable({ newPoint: (x, y) => ({ x, y, label: 0 }), onChange: () => replay({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    let A = null, VP = null;
    function replay({ at = "start" } = {}) {
      const r = buildFrames(lab.points, lab.val("std"));
      A = r.A;
      drawVar(); drawCov();
      lab.load(r.frames, { at, play: at === "start" });
    }
    function drawVar() {
      const el = lab.charts.var.el; el.innerHTML = "";
      if (!A) return;
      const tot = A.vals[0] + A.vals[1];
      VP = Lab.xy(el, { w: 360, h: 130, xd: [0, 180], yd: [0, tot * 1.05], xLabel: "θ (deg)" });
      const th = d3.range(0, 180.5, 2);
      VP.line("cap", th.map((t) => [t, projVar(A.Z, [Math.cos((t * Math.PI) / 180), Math.sin((t * Math.PI) / 180)])]), { color: "var(--series-1)" });
      VP.line("lost", th.map((t) => [t, tot - projVar(A.Z, [Math.cos((t * Math.PI) / 180), Math.sin((t * Math.PI) / 180)])]), { color: "var(--series-2)", dash: "4 3" });
      el.closest(".lab-card").querySelector(".chart-legend").innerHTML = `<span class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>variance kept</span><span class="legend-item"><span class="swatch" style="background:var(--series-2)"></span>reconstruction error</span>`;
    }
    function drawCov() {
      const el = lab.R("panel-cov"); el.innerHTML = `<div style="display:grid;grid-template-columns:150px 1fr;gap:12px;align-items:center"><div data-m></div><div data-b></div></div>`;
      if (!A) return;
      Lab.matrix(el.querySelector("[data-m]"), { cell: 50, rowLabels: ["x", "y"], colLabels: ["x", "y"], labelW: 16 }).set(A.C);
      const tot = A.vals[0] + A.vals[1];
      Lab.bars(el.querySelector("[data-b]"), { w: 240, h: 120, yd: [0, 1] }).set([{ label: "PC1", value: A.vals[0] / tot, color: "var(--series-1)", text: ((A.vals[0] / tot) * 100).toFixed(1) + "%" }, { label: "PC2", value: A.vals[1] / tot, color: "var(--series-5)", text: ((A.vals[1] / tot) * 100).toFixed(1) + "%" }]);
    }
    const ellG = L.mid.append("g"), dirG = L.mid.append("g"), pcG = L.top.append("g"), muG = L.top.append("g");
    const defs = pl.svg.append("defs");
    for (const [id, c] of [["pca-a1", "var(--series-1)"], ["pca-a2", "var(--series-5)"]]) defs.append("marker").attr("id", id).attr("viewBox", "0 0 10 10").attr("refX", 8).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto").append("path").attr("d", "M0,0L10,5L0,10z").attr("fill", c);
    function render(f) {
      const st = f.st;
      ellG.selectAll("*").remove(); dirG.selectAll("*").remove(); pcG.selectAll("*").remove(); muG.selectAll("*").remove(); L.links.selectAll("*").remove();
      if (!A) { lab.drawPoints({}); return; }
      const raw = lab.points.map((p) => [p.x, p.y]);
      const pos = A.Z.map((z, i) => {
        let q = [raw[i][0] + (z[0] - raw[i][0]) * st.t, raw[i][1] + (z[1] - raw[i][1]) * st.t];
        if (st.collapse) { const s = z[0] * A.v1[0] + z[1] * A.v1[1]; const pr = [s * A.v1[0], s * A.v1[1]]; q = [z[0] + (pr[0] - z[0]) * st.collapse, z[1] + (pr[1] - z[1]) * st.collapse]; }
        return q;
      });
      const m = [A.mu[0] * (1 - st.t), A.mu[1] * (1 - st.t)];
      muG.append("path").attr("d", d3.symbol(d3.symbolCross, 170)()).attr("transform", `translate(${pl.x(m[0])},${pl.y(m[1])})`).attr("fill", "var(--accent)").attr("stroke", "var(--bg)");
      if (st.ell) ellG.append("path").attr("d", Lab.ellipsePath(pl, [0, 0], A.C, 2)).attr("fill", "var(--series-1)").attr("fill-opacity", 0.05).attr("stroke", "var(--text-faint)").attr("stroke-dasharray", "5 4");
      if (st.w && !st.collapse) {
        const w = st.w;
        dirG.append("line").attr("x1", pl.x(-w[0] * 15)).attr("y1", pl.y(-w[1] * 15)).attr("x2", pl.x(w[0] * 15)).attr("y2", pl.y(w[1] * 15)).attr("stroke", "var(--accent)").attr("stroke-width", 2);
        if (lab.val("resid")) A.Z.forEach((z) => { const s = z[0] * w[0] + z[1] * w[1]; L.links.append("line").attr("x1", pl.x(z[0])).attr("y1", pl.y(z[1])).attr("x2", pl.x(s * w[0])).attr("y2", pl.y(s * w[1])).attr("stroke", "var(--series-2)").attr("stroke-opacity", 0.55); });
        A.Z.forEach((z) => { const s = z[0] * w[0] + z[1] * w[1]; dirG.append("circle").attr("cx", pl.x(s * w[0])).attr("cy", pl.y(s * w[1])).attr("r", 2.2).attr("fill", "var(--accent)"); });
      }
      if (st.pcs) {
        [[A.v1, A.vals[0], "pca-a1", "var(--series-1)", "PC1"], [A.v2, A.vals[1], "pca-a2", "var(--series-5)", "PC2"]].forEach(([v, lam, mk, c, t]) => {
          const L2 = 2 * Math.sqrt(lam);
          pcG.append("line").attr("x1", pl.x(0)).attr("y1", pl.y(0)).attr("x2", pl.x(v[0] * L2)).attr("y2", pl.y(v[1] * L2)).attr("stroke", c).attr("stroke-width", 3.2).attr("marker-end", `url(#${mk})`);
          pcG.append("text").attr("class", "lbl-strong halo").attr("x", pl.x(v[0] * (L2 + 0.8))).attr("y", pl.y(v[1] * (L2 + 0.8))).attr("text-anchor", "middle").text(t);
        });
      }
      const sel = lab.drawPoints({ color: "var(--series-1)", opacity: st.collapse ? 0.85 : 0.8, edit: st.t === 0 });
      sel.attr("cx", (d, i) => pl.x(pos[i][0])).attr("cy", (d, i) => pl.y(pos[i][1]));
      if (VP) VP.vline(st.th != null ? st.th : null);
      const tot = A.vals[0] + A.vals[1];
      lab.readout(`<div class="kv"><span>λ₁ · λ₂</span><b>${fmt(A.vals[0], 2)} · ${fmt(A.vals[1], 2)}</b><span>PC1 explains</span><b>${((A.vals[0] / tot) * 100).toFixed(1)}%</b><span>PC1 direction</span><b>(${fmt(A.v1[0], 2)}, ${fmt(A.v1[1], 2)})</b></div>`);
    }
    lab.watch(["std"], () => replay({ at: "end" }));
    lab.watch(["resid"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.loadPreset("corr");
    return lab.destroy;
  }

  MLApp.register({
    id: "pca",
    name: "Principal Component Analysis",
    category: "Unsupervised - Dimensionality Reduction",
    tagline: "variance-maximising rotation, step by step",
    description: "Centre the data, build the covariance matrix, then sweep a direction around the circle to watch variance kept and reconstruction error trade off, until the eigenvector appears at the peak and the cloud collapses onto PC1.",
    info: {
      type: "Unsupervised - Dimensionality reduction / linear transformation.",
      scenario: "Compressing correlated features into fewer uncorrelated components, visualizing high-dimensional data, denoising, or as pre-processing before another model.",
      inputs: "A matrix of numeric features X, typically centered (and often standardized) first.",
      intuition: {
        definition: "Find the directions along which the data <b>varies most</b>, and describe each point by its coordinates along just a few of them. Those directions are orthogonal, ordered by variance, and are exactly the eigenvectors of the covariance matrix.",
        steps: [
          "Centre the data so the mean sits at the origin.",
          "Compute the covariance matrix of the features.",
          "Take its eigenvectors: these are the principal axes.",
          "Keep the top k by eigenvalue and project onto them.",
        ],
        applications: [
          "Compressing correlated sensor or survey features",
          "Plotting high-dimensional data in two dimensions",
          "Denoising by discarding low-variance components",
          "Decorrelating features before regression or clustering",
          "Eigenfaces and other classical image compression",
        ],
      },
      math: [
        { title: "Centre the data", formula: "X̃ = X − μ,  μ = column means", note: "Non-negotiable. Without centring, the first component points at the mean rather than the direction of spread." },
        { title: "Covariance matrix", formula: "C = (1/(m−1))·X̃ᵀX̃    (p × p, symmetric)", note: "Entry Cⱼₖ is the covariance between features j and k. Symmetry guarantees real eigenvalues and orthogonal eigenvectors." },
        { title: "Eigendecomposition", formula: "C·vⱼ = λⱼ·vⱼ,   λ₁ ≥ λ₂ ≥ … ≥ λ_p ≥ 0", note: "Each eigenvector vⱼ is a principal axis; its eigenvalue λⱼ is the variance of the data along that axis." },
        { title: "Variance maximisation", formula: "v₁ = argmax_{‖v‖=1} Var(X̃v) = argmax vᵀCv", note: "The first component is the unit direction of maximum projected variance. Each later one repeats this subject to being orthogonal to all previous." },
        { title: "Projection", formula: "Z = X̃·V_k    (m × k scores)", note: "V_k holds the top k eigenvectors as columns. Z gives each point's coordinates in the new basis." },
        { title: "Reconstruction", formula: "X̂ = Z·V_kᵀ + μ,   error = Σ_{j>k} λⱼ", note: "The discarded eigenvalues are exactly the reconstruction error. Among all rank-k linear maps, this is provably optimal (Eckart-Young)." },
        { title: "SVD route", formula: "X̃ = UΣVᵀ  ⟹  same V, with λⱼ = σⱼ²/(m−1)", note: "How it is actually computed. SVD avoids forming C, which squares the condition number and loses precision." },
      ],
      pipeline: [
        { label: "Standardize", note: "centre, often scale" },
        { label: "Covariance C", note: "p × p matrix" },
        { label: "Eigen / SVD", note: "get V, λ" },
        { label: "Keep top k", note: "by variance" },
        { label: "Scores Z", note: "m × k projection", accent: "green" },
      ],
      decisionFunction: {
        text: "project(x) = Vᵀ(x − μ), using the top eigenvectors V of the covariance matrix",
        mechanism: "Eigendecompose the centered covariance matrix: the eigenvector with the largest eigenvalue is the direction of maximum variance (PC1); the next-largest orthogonal direction is PC2, and so on.",
      },
      lossFunction: {
        text: "min Σᵢ ‖xᵢ − reconstruct(project(xᵢ))‖²  (reconstruction error)",
        mechanism: "Among all rank-k linear projections, PCA's is provably the one minimizing this reconstruction error - solved exactly by eigendecomposition, with no optimization loop.",
        plot: { fn: (k) => Math.exp(-0.6 * k), domain: [1, 6], color: "var(--accent)", caption: "typical 'scree plot' shape: variance explained per component usually drops off fast" },
      },
      optimization: [
        { title: "No iteration", formula: "one eigendecomposition, closed form", note: "PCA is a matrix factorisation, not a learned model. There is no learning rate, no initialisation, and no local optimum." },
        { title: "Cost", formula: "O(m·p² + p³) via covariance, O(m·p·k) via randomized SVD", note: "The p³ eigendecomposition dominates in wide data. Randomized SVD is far cheaper when only a few components are wanted." },
        { title: "Choosing k", formula: "smallest k with Σ_{j≤k} λⱼ / Σ_j λⱼ ≥ target", note: "Typically 90 to 95% cumulative variance. Alternatives are the scree elbow or Kaiser's rule of keeping λ > 1 on standardized data." },
        { title: "Whitening", formula: "Z_white = Z / √λ", note: "Rescales components to unit variance. Useful before distance-based models, harmful if the variance ordering itself is informative." },
        { title: "Incremental PCA", formula: "update the basis batch by batch", note: "For data too large to hold in memory. Trades a little accuracy for constant memory." },
      ],
      output: "A lower-dimensional projected representation, plus the principal axes (eigenvectors) and their eigenvalues (variance explained).",
      assumptions: [
        { name: "Structure is linear", why: "PCA can only rotate and project. Data on a curved manifold is flattened incorrectly.", check: "If a 2D PCA plot looks structureless but t-SNE or UMAP shows clusters, the structure is non-linear." },
        { name: "Variance means information", why: "Components are ranked purely by variance, which need not align with what you care about.", check: "For a supervised task compare against LDA, which ranks by class separation instead." },
        { name: "Data is centred", why: "Uncentred data makes PC1 point toward the mean rather than along the spread.", check: "Every library centres automatically. Do not disable it." },
        { name: "Features are comparably scaled", why: "Covariance is scale-dependent, so a feature in large units dominates regardless of relevance.", check: "Standardize unless all features already share meaningful units." },
        { name: "Components need not be interpretable", why: "Each is a dense mixture of all original features.", check: "Use sparse PCA or factor analysis if you need readable loadings." },
      ],
      hyperparameters: [
        { name: "n_components k", range: "1 - min(m, p)", increasing: "Less information lost, less compression. k = p is a lossless rotation.", strategy: "Pick by cumulative explained variance (90 to 95%), the scree elbow, or downstream validation score. Pass a float such as 0.95 to let the library choose." },
        { name: "standardize first", range: "true / false", increasing: "Not applicable", strategy: "Standardize whenever feature units differ. Skip only when all features are already on one comparable scale, such as pixel intensities." },
        { name: "whiten", range: "true / false", increasing: "Not applicable", strategy: "Enable before distance-based models such as KNN or SVM; leave off when the variance ordering carries meaning." },
        { name: "svd_solver", range: "auto / full / randomized / arpack", increasing: "Not applicable", strategy: "auto is right nearly always. randomized is much faster when k is far smaller than p." },
      ],
      metrics: ["Cumulative % variance explained", "Reconstruction error (MSE)", "Scree plot elbow position", "Downstream model performance after projection"],
      typicalUses: ["Visualizing high-dimensional data in 2D/3D", "Noise reduction", "Decorrelating features before regression/clustering", "Data compression"],
      diagnostics: [
        "Plot the scree curve of explained variance per component. A sharp drop means a few components capture the structure; a flat curve means PCA is not helping.",
        "Check cumulative variance. If you need 40 of 50 components to reach 90%, the features were not very correlated and compression will cost you.",
        "Inspect the loadings of the top components to see which original features drive each axis.",
        "Compare a PCA scatter against t-SNE or UMAP. If they disagree sharply, the structure is non-linear.",
        "Verify the pipeline scales inside cross-validation. Fitting PCA on the full dataset before splitting leaks test information.",
      ],
      advantages: [
        "Exact closed-form solution with no hyperparameter search and no random seed.",
        "Provably the optimal linear projection for reconstruction error.",
        "Removes multicollinearity outright, since components are orthogonal by construction.",
        "Denoises effectively when noise is spread across low-variance directions.",
        "Fast, and scalable via randomized or incremental variants.",
        "Fully invertible, so you can project down and reconstruct back up.",
      ],
      limitations: [
        { name: "Linear only", note: "cannot unfold curved manifolds such as a swiss roll", fix: "kernel PCA, UMAP, t-SNE, or an autoencoder." },
        { name: "Components are uninterpretable", note: "each mixes every original feature", fix: "sparse PCA, or factor analysis with rotation." },
        { name: "Variance is not relevance", note: "a low-variance feature may be the only predictive one", fix: "use LDA or PLS when labels are available." },
        { name: "Scale-sensitive", note: "unstandardized features hijack the components", fix: "standardize first." },
        { name: "Outlier-sensitive", note: "covariance is driven by extreme points", fix: "robust PCA, or remove outliers." },
        { name: "Assumes Gaussian-ish structure", note: "only second-order statistics are used", fix: "ICA if you need independent rather than merely uncorrelated components." },
      ],
      alternatives: [
        { name: "t-SNE / UMAP", when: "The goal is visualisation and the structure is non-linear." },
        { name: "Kernel PCA", when: "You want PCA's framework but a non-linear manifold." },
        { name: "LDA", when: "Labels exist and you want axes that separate classes, not axes of maximum variance." },
        { name: "Autoencoder", when: "Large datasets and complex non-linear compression, and you need an out-of-sample mapping." },
        { name: "Truncated SVD", when: "Sparse data such as TF-IDF, where centring would destroy sparsity." },
      ],
      pitfalls: [
        { problem: "One feature dominates PC1", solution: "Unstandardized units. Scale before fitting." },
        { problem: "PCA plot shows no structure", solution: "Structure may be non-linear. Try UMAP or t-SNE before concluding there is none." },
        { problem: "Accuracy drops after PCA", solution: "You discarded low-variance but predictive directions. Keep more components, or use a supervised reduction." },
        { problem: "Fitting PCA before the train/test split", solution: "Leakage. Put PCA inside a Pipeline so it is fitted per fold." },
        { problem: "Component signs flip between runs", solution: "Eigenvector sign is arbitrary. Harmless, but fix it if you compare loadings across runs." },
        { problem: "Centring a huge sparse matrix blows up memory", solution: "Use TruncatedSVD, which skips centring and preserves sparsity." },
      ],
      quickRef: [
        { name: "Centre", formula: "X̃ = X − μ" },
        { name: "Covariance", formula: "C = X̃ᵀX̃ / (m−1)" },
        { name: "Eigenproblem", formula: "C v = λ v" },
        { name: "Projection", formula: "Z = X̃ V_k" },
        { name: "Reconstruction", formula: "X̂ = Z V_kᵀ + μ" },
        { name: "Explained variance", formula: "λⱼ / Σ λ" },
        { name: "Recon. error", formula: "Σ_{j>k} λⱼ" },
        { name: "Via SVD", formula: "X̃ = UΣVᵀ, λ = σ²/(m−1)" },
      ],
      code: `from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Pass a float to let PCA pick k by cumulative explained variance.
pca = make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=42))
Z = pca.fit_transform(X_train)

p = pca.named_steps["pca"]
print("components kept:", p.n_components_)
print("cumulative variance:", np.cumsum(p.explained_variance_ratio_))

# Which original features drive each axis:
loadings = p.components_          # shape (k, p)

# Inside a model, PCA must be fitted per fold or you leak test data.
from sklearn.linear_model import LogisticRegression
clf = make_pipeline(StandardScaler(), PCA(n_components=20), LogisticRegression())

# Sparse text data: skip centring so sparsity survives.
svd = TruncatedSVD(n_components=100).fit_transform(tfidf_matrix)`,
      whyChain: [
        { q: "Why are the principal components the eigenvectors of the covariance matrix?", a: "The variance of the data projected onto a unit vector v is vᵀCv. Maximising that subject to ‖v‖ = 1 is a Rayleigh quotient problem, and its solution is the eigenvector with the largest eigenvalue. The eigenvalue is the variance itself." },
        { q: "Why must the data be centred first?", a: "Variance is measured about the mean. If the data is not centred, the direction that maximises the raw second moment points from the origin toward the data cloud, which describes the location of the data rather than its spread." },
        { q: "Why are the components always orthogonal?", a: "The covariance matrix is real and symmetric, and the spectral theorem guarantees such a matrix has an orthogonal eigenbasis. Geometrically, each component captures variance that the previous ones have not, so it must be perpendicular to them." },
        { q: "Why standardize before PCA?", a: "Covariance depends on units. A feature measured in millimetres has a variance a million times larger than the same feature in metres, so it would capture PC1 for reasons that have nothing to do with information content." },
        { q: "Why is SVD preferred over eigendecomposing the covariance matrix?", a: "Forming XᵀX squares the condition number, so precision is lost for ill-conditioned data. SVD works directly on X and returns the same V with better numerical stability." },
        { q: "Does a high-variance component mean an important one?", a: "Not necessarily. PCA is unsupervised, so it has no idea what you want to predict. A low-variance direction can carry all the class signal, which is why PCA sometimes hurts downstream accuracy. LDA optimises separation instead." },
        { q: "Why can PCA not unfold a swiss roll?", a: "PCA applies a single linear map, so it can rotate and project but never bend. A swiss roll needs different transformations in different regions, which requires a non-linear method." },
        { q: "How does PCA denoise data?", a: "Signal usually concentrates in a few high-variance directions while noise spreads thinly across all of them. Discarding the low-variance components removes proportionally more noise than signal, and the reconstruction is cleaner than the original." },
      ],
      parameters: [
        { name: "number of components", effect: "How many axes to keep. Fewer = more compression / more information lost." },
        { name: "standardize features?", effect: "Important when features are on very different scales - otherwise high-variance features dominate the principal axes for reasons unrelated to importance." },
      ],
      metrics: ["Cumulative % variance explained", "Reconstruction error (MSE)"],
      typicalUses: ["Visualizing high-dimensional data in 2D/3D", "Noise reduction", "Decorrelating features before regression/clustering", "Data compression"],
      workedExample: {
        setup: "Points (1,2), (3,4), (5,6) - perfectly correlated (they lie exactly on a line). Find PC1.",
        steps: [
          "Mean = (3, 4). Deviations: (−2,−2), (0,0), (2,2).",
          "Sxx = (4+0+4)/3 = 2.667, Syy = (4+0+4)/3 = 2.667, Sxy = (4+0+4)/3 = 2.667.",
          "Covariance matrix = [[2.667, 2.667], [2.667, 2.667]] - trace = 5.333, determinant = 2.667×2.667 − 2.667×2.667 = 0.",
          "Determinant 0 means one eigenvalue is 0; since trace = sum of eigenvalues, the other eigenvalue is 5.333.",
          "The eigenvector for λ=5.333 is the direction (1,1)/√2 - a 45° line.",
        ],
        result: "PC1 points along (1,1)/√2 and explains 100% of the variance - exactly the line y=x+1 the data lies on, with 0% left for PC2",
      },
    },
    mount,
  });
})();
