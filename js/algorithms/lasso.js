(() => {
  const { ui, col, fmt } = Lab;
  const soft = (z, g) => (z > g ? z - g : z < -g ? z + g : 0);

  function design(pts, deg) {
    const raw = pts.map((p) => Array.from({ length: deg }, (_, d) => (p.x / 9) ** (d + 1)));
    const mu = Array.from({ length: deg }, (_, j) => MLU.mean(raw.map((r) => r[j])));
    const sd = Array.from({ length: deg }, (_, j) => Math.sqrt(MLU.mean(raw.map((r) => (r[j] - mu[j]) ** 2))) || 1);
    const X = raw.map((r) => r.map((v, j) => (v - mu[j]) / sd[j]));
    const ym = MLU.mean(pts.map((p) => p.y));
    return { X, y: pts.map((p) => p.y - ym), ym, mu, sd };
  }
  const objective = (X, y, w, lam, a) => { const n = y.length; let r = 0; for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < w.length; j++) s += X[i][j] * w[j]; r += (y[i] - s) ** 2; } return r / (2 * n) + lam * (a * w.reduce((s, v) => s + Math.abs(v), 0) + ((1 - a) / 2) * w.reduce((s, v) => s + v * v, 0)); };

  // Coordinate descent; record(w, j, sweep) is called after each coordinate update.
  function cd(D, lam, a, { maxSweeps = 200, tol = 1e-6, w0 = null, record = null } = {}) {
    const { X, y } = D, n = y.length, p = X[0].length;
    const w = w0 ? w0.slice() : new Array(p).fill(0);
    const r = y.map((yi, i) => yi - X[i].reduce((s, v, j) => s + v * w[j], 0));
    const z = Array.from({ length: p }, (_, j) => MLU.mean(X.map((row) => row[j] * row[j])));
    for (let s = 1; s <= maxSweeps; s++) {
      let maxd = 0;
      for (let j = 0; j < p; j++) {
        let rho = 0;
        for (let i = 0; i < n; i++) rho += X[i][j] * (r[i] + X[i][j] * w[j]);
        rho /= n;
        const nw = soft(rho, lam * a) / (z[j] + lam * (1 - a));
        const d = nw - w[j];
        if (d) { for (let i = 0; i < n; i++) r[i] -= X[i][j] * d; w[j] = nw; }
        maxd = Math.max(maxd, Math.abs(d));
        if (record) record(w, j, s, rho);
      }
      if (maxd < tol) break;
    }
    return w;
  }

  function buildFrames(pts, P) {
    if (pts.length < 3) return { frames: [{ phase: "idle", title: "Add more points", text: "", st: { w: new Array(P.deg).fill(0) } }], D: null };
    const D = design(pts, P.deg);
    const frames = [{ phase: "init", major: true, arch: ["phi", "std"], title: `${P.deg} polynomial features, standardised; start at w = 0`, text: "Coordinate descent optimises one coefficient at a time while holding the others fixed. For the L1 penalty each one-dimensional problem has a closed-form answer: the soft-threshold.", tex: "\\min_w\\;\\frac{1}{2n}\\lVert y-Xw\\rVert^2+\\lambda\\Big(\\alpha\\lVert w\\rVert_1+\\tfrac{1-\\alpha}{2}\\lVert w\\rVert_2^2\\Big)", st: { w: new Array(P.deg).fill(0) }, m: { obj: objective(D.X, D.y, new Array(P.deg).fill(0), P.lam, P.a) } }];
    let lastSweep = 0;
    cd(D, P.lam, P.a, {
      maxSweeps: 60,
      record: (w, j, s, rho) => {
        if (s <= 2) frames.push({ phase: "coord", major: s === 1 && j === 0, arch: ["rho", "soft", "upd"], title: `Sweep ${s}, coordinate w${j + 1}: ρ = ${rho.toFixed(3)} → w${j + 1} = ${w[j].toFixed(3)}`, text: Math.abs(rho) <= P.lam * P.a ? `|ρ| ≤ λα = ${(P.lam * P.a).toFixed(3)}, so the soft-threshold sets this coefficient <b>exactly to zero</b>. That is how the lasso does feature selection.` : `ρ is this feature's correlation with the partial residual. It is shrunk toward 0 by λα and divided by 1 + λ(1 − α).`, tex: "w_j\\leftarrow\\frac{S(\\rho_j,\\lambda\\alpha)}{\\tfrac1n\\lVert x_j\\rVert^2+\\lambda(1-\\alpha)},\\quad S(z,\\gamma)=\\operatorname{sign}(z)\\max(|z|-\\gamma,0)", st: { w: w.slice(), cur: j }, m: { obj: objective(D.X, D.y, w, P.lam, P.a) } });
        else if (j === w.length - 1 && s !== lastSweep) { lastSweep = s; frames.push({ phase: "sweep", arch: ["upd", "conv"], title: `Sweep ${s} complete`, text: `${w.filter((v) => v === 0).length} of ${w.length} coefficients are exactly zero.`, st: { w: w.slice() }, m: { obj: objective(D.X, D.y, w, P.lam, P.a) } }); }
      },
    });
    const w = frames[frames.length - 1].st.w;
    frames.push({ phase: "done", major: true, arch: ["conv", "out"], title: `Converged: ${w.filter((v) => v !== 0).length} non-zero coefficients`, text: P.a > 0 ? "Increase λ to drive more coefficients to zero; the path chart shows every coefficient's value as λ varies. Lower α toward 0 and the solution turns into ridge: everything shrinks but nothing hits zero." : "With α = 0 this is ridge regression: coefficients shrink smoothly but never become exactly zero.", st: { w }, m: { obj: objective(D.X, D.y, w, P.lam, P.a) } });
    return { frames, D };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "lasso", fps: 6, plane: { xLabel: "x", yLabel: "y" },
      legend: ui.legend([["var(--accent)", "fitted polynomial", "line"], ["var(--series-1)", "data"]]),
      tools: ui.btn("retrain", "retrain", true),
      hint: "click to add points · drag / double-click to edit",
      data: { kind: "reg", presets: ["sine", "cubic", "rlinear", "step", "quad"], def: "sine", n: 30, noise: 0.25 },
      side: [{ title: "regularisation", html: ui.range("lam", "λ (log₁₀)", -4, 1, 0.1, -1.5, "0.032") + ui.range("a", "α: 0 = ridge · 1 = lasso", 0, 1, 0.05, 1) + ui.range("deg", "polynomial degree", 2, 12, 1, 10) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "phi", label: "φ(x)", sub: "poly features", x: 50, y: 40, kind: "data" },
          { id: "std", label: "standardise", sub: "zero mean, unit var", x: 180, y: 40, kind: "op" },
          { id: "rho", label: "ρⱼ", sub: "residual corr.", x: 318, y: 40, kind: "op" },
          { id: "soft", label: "soft-threshold", sub: "S(ρ, λα)", x: 316, y: 112, kind: "loss" },
          { id: "upd", label: "update wⱼ", sub: "÷ (1 + λ(1−α))", x: 180, y: 112, kind: "op" },
          { id: "conv", label: "converged?", sub: "max |Δw| < tol", x: 56, y: 112, kind: "param" },
          { id: "out", label: "sparse w", x: 56, y: 170, kind: "out" },
        ],
        edges: [{ from: "phi", to: "std" }, { from: "std", to: "rho" }, { from: "rho", to: "soft" }, { from: "soft", to: "upd" }, { from: "upd", to: "rho", dash: true, label: "next j", bend: -22 }, { from: "upd", to: "conv" }, { from: "conv", to: "out" }],
      },
      phases: { init: { label: "init", color: 3 }, coord: { label: "coordinate step", color: 0 }, sweep: { label: "sweep", color: 4 }, done: { label: "converged", color: 2 } },
      panels: [{ id: "coef", title: "coefficients", sub: "faded = exactly zero" }, { id: "geo", title: "why L1 gives zeros", sub: "loss contours meet the constraint" }],
      charts: [{ id: "path", title: "regularisation path: coefficients vs λ", custom: true, wide: true }],
      onFrame: render, onData: () => retrain(),
    });
    lab.editable({ newPoint: (x, y) => ({ x, y }), onChange: () => retrain({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    const P = () => ({ lam: Math.pow(10, lab.val("lam")), a: lab.val("a"), deg: lab.val("deg") });
    let D = null, PP = null;
    function retrain({ at = "start" } = {}) {
      const r = buildFrames(lab.points, P());
      D = r.D;
      drawPath(); drawGeo();
      lab.load(r.frames, { at, play: at === "start" });
    }
    function drawPath() {
      const el = lab.charts.path.el; el.innerHTML = "";
      if (!D) return;
      const lams = d3.range(1, -4.01, -0.125).map((v) => Math.pow(10, v));
      let w = null;
      const path = lams.map((lam) => { w = cd(D, lam, P().a, { w0: w, maxSweeps: 300 }); return w.slice(); });
      const mx = Math.max(0.1, ...path.flat().map(Math.abs));
      PP = Lab.xy(el, { w: 720, h: 170, xd: [-4, 1], yd: [-mx * 1.05, mx * 1.05], xLabel: "log₁₀ λ  (strong regularisation →)" });
      for (let j = 0; j < D.X[0].length; j++) PP.line("c" + j, lams.map((lam, k) => [Math.log10(lam), path[k][j]]), { color: col(j), width: 1.8, curve: d3.curveLinear });
      PP.vline(Math.log10(P().lam));
    }
    function drawGeo() {
      const el = lab.R("panel-geo"); el.innerHTML = "";
      const G = Lab.xy(el, { w: 360, h: 200, xd: [-2.4, 2.4], yd: [-1.6, 1.6], xLabel: "w₁", yLabel: "w₂", xTicks: 5, yTicks: 3 });
      const a = P().a, c0 = [1.5, 0.9], H = [[1, 0.6], [0.6, 1.4]];
      const loss = (x, y) => { const dx = x - c0[0], dy = y - c0[1]; return 0.5 * (dx * (H[0][0] * dx + H[0][1] * dy) + dy * (H[1][0] * dx + H[1][1] * dy)); };
      const pen = (x, y) => a * (Math.abs(x) + Math.abs(y)) + ((1 - a) / 2) * (x * x + y * y);
      // constrained optimum via coordinate descent on the 2-D problem
      const lam2 = Math.min(3, P().lam * 12 + 0.25);
      let w = [0, 0];
      for (let it = 0; it < 400; it++) for (let j = 0; j < 2; j++) { const o = 1 - j; const rho = H[j][j] * c0[j] + H[j][o] * (c0[o] - w[o]); w[j] = soft(rho, lam2 * a) / (H[j][j] + lam2 * (1 - a)); }
      const res = 70, vals = [], vp = [];
      for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) { const x = -2.4 + ((i + 0.5) / res) * 4.8, y = 1.6 - ((j + 0.5) / res) * 3.2; vals.push(loss(x, y)); vp.push(pen(x, y)); }
      const geo = d3.geoPath(d3.geoTransform({ point(px, py) { this.stream.point(G.x.range()[0] + (px / res) * (G.x.range()[1] - G.x.range()[0]), G.y.range()[1] + (py / res) * (G.y.range()[0] - G.y.range()[1])); } }));
      const lw = loss(w[0], w[1]);
      G.g.selectAll("path.lc").data(d3.contours().size([res, res]).thresholds([lw * 0.25, lw * 0.6, lw, lw * 1.6])(vals)).enter().append("path").attr("d", geo).attr("fill", "none").attr("stroke", "var(--series-1)").attr("stroke-width", (d) => (Math.abs(d.value - lw) < 1e-9 ? 2.2 : 1)).attr("opacity", 0.8);
      const pw = pen(w[0], w[1]) || 1e-6;
      G.g.selectAll("path.pc").data(d3.contours().size([res, res]).thresholds([pw])(vp)).enter().append("path").attr("d", geo).attr("fill", "var(--series-4)").attr("fill-opacity", 0.15).attr("stroke", "var(--series-4)").attr("stroke-width", 2);
      G.g.append("circle").attr("cx", G.x(c0[0])).attr("cy", G.y(c0[1])).attr("r", 3.5).attr("fill", "var(--series-1)");
      G.g.append("circle").attr("cx", G.x(w[0])).attr("cy", G.y(w[1])).attr("r", 5).attr("fill", "var(--accent)");
      G.g.append("text").attr("class", "lbl").attr("x", G.x(w[0]) + 8).attr("y", G.y(w[1]) - 6).text(`w* = (${w[0].toFixed(2)}, ${w[1].toFixed(2)})`).style("font-size", "9px").attr("fill", "var(--text)");
    }
    const curve = L.mid.append("path").attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2.4);
    const coef = Lab.bars(lab.R("panel-coef"), { w: 360, h: 140, signed: true });
    function render(f) {
      const w = f.st.w;
      if (D) {
        const pred = (x) => D.ym + w.reduce((s, wj, j) => s + wj * (((x / 9) ** (j + 1) - D.mu[j]) / D.sd[j]), 0);
        curve.attr("d", d3.line().x((x) => pl.x(x)).y((x) => pl.y(Math.max(-14, Math.min(14, pred(x)))))(d3.range(-10, 10.01, 0.05)));
      }
      lab.drawPoints({ color: "var(--series-1)" });
      coef.set(w.map((v, j) => ({ label: "x" + (j + 1 > 1 ? (j + 1).toString().split("").map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]).join("") : ""), value: v, color: col(j), dim: v === 0 || f.st.cur != null && f.st.cur !== j })), { values: false });
      const nz = w.filter((v) => v !== 0).length;
      lab.readout(`<div class="kv"><span>λ · α</span><b>${fmt(P().lam, 4)} · ${P().a}</b><span>non-zero coefficients</span><b>${nz} / ${w.length}</b><span>objective</span><b>${fmt(f.m && f.m.obj, 4)}</b></div>`);
    }
    lab.watch(["lam", "a", "deg"], () => retrain({ at: "end" }), { format: { lam: (v) => fmt(Math.pow(10, v), 4) } });
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    lab.loadPreset("sine");
    return lab.destroy;
  }

  MLApp.register({
    id: "lasso",
    name: "Lasso & Elastic Net",
    category: "Supervised - Regression",
    tagline: "L1 sparsity via coordinate descent",
    isNew: true,
    description: "Linear regression with an L1 (and optionally L2) penalty that drives coefficients exactly to zero. Watch coordinate descent soft-threshold one coefficient at a time, see the full regularisation path as λ varies, and the geometry of why a diamond-shaped constraint produces sparse solutions.",
    info: {
      type: "Supervised - Regression. Linear, regularised; performs embedded feature selection.",
      scenario: "Many candidate features, some irrelevant or correlated, where a sparse, interpretable model is wanted: genomics, text, sensor selection, economics.",
      inputs: "Feature matrix X (standardised) and continuous target y; penalty strength λ and mixing α.",
      output: "A coefficient vector w, typically with many exact zeros.",
      intuition: {
        definition: "Ordinary least squares plus a price on the absolute size of every coefficient. A feature only gets a non-zero weight if it reduces the error by more than its price, so weak features are switched off entirely. Elastic net blends in a ridge penalty to handle groups of correlated features gracefully.",
        steps: ["Standardise features so the penalty treats them equally.", "Cycle through coefficients, solving each one-dimensional problem exactly.", "Soft-threshold: shrink toward zero by λα and clip at zero.", "Repeat sweeps until coefficients stop changing; trace the path over many λ."],
        applications: ["Gene selection from thousands of expression levels", "Sparse text regression and interpretable scoring models", "Compressed sensing and signal recovery", "Selecting sensors or survey questions"],
      },
      math: [
        { title: "Objective", formula: "min_w (1/2n)‖y − Xw‖² + λ[α‖w‖₁ + (1−α)/2 ‖w‖²]", note: "α = 1 is the lasso, α = 0 is ridge, in between is the elastic net." },
        { title: "Soft-thresholding", formula: "S(z, γ) = sign(z)·max(|z| − γ, 0)", note: "The exact solution of the one-dimensional L1 problem: anything with |z| ≤ γ becomes exactly 0." },
        { title: "Coordinate update", formula: "wⱼ = S(ρⱼ, λα) / ((1/n)‖xⱼ‖² + λ(1−α))", note: "ρⱼ = (1/n) xⱼᵀ(y − X₋ⱼw₋ⱼ) is the correlation of feature j with the residual that excludes it." },
        { title: "Why L1 gives zeros", formula: "the ‖w‖₁ ball has corners on the axes", note: "Elliptical loss contours usually first touch a diamond at a corner, where some coordinates are 0; a circle (L2) has no corners." },
        { title: "λ_max", formula: "λ_max = max_j |xⱼᵀy| / (nα)", note: "Above this value every coefficient is zero; paths are computed from λ_max downward with warm starts." },
      ],
      pipeline: [{ label: "Standardise X" }, { label: "Partial residual ρⱼ" }, { label: "Soft-threshold" }, { label: "Sweep until stable" }, { label: "Sparse w", accent: "green" }],
      decisionFunction: { text: "ŷ = w₀ + Σⱼ wⱼ xⱼ   (many wⱼ = 0)", mechanism: "Prediction is ordinary linear regression; the sparsity means only the selected features are used." },
      lossFunction: { text: "L(w) = (1/2n)‖y − Xw‖² + λα‖w‖₁ + λ(1−α)/2 ‖w‖²", mechanism: "Convex but not differentiable at zero (the kink of |w| is what creates exact zeros), so plain gradient descent is replaced by coordinate descent or proximal methods.", plot: { fn: (z) => (z > 1 ? z - 1 : z < -1 ? z + 1 : 0), domain: [-3, 3], fn2: (z) => z / 2, caption: "soft-threshold S(z, 1) (solid, flat zero region) vs ridge shrinkage z/(1+λ) (dashed, never exactly zero)" } },
      optimization: [{ title: "Cyclic coordinate descent", formula: "O(np) per sweep", note: "Each coordinate update is closed form; glmnet's workhorse." }, { title: "Warm starts along the path", formula: "solve λ₁ > λ₂ > … reusing w", note: "The full path costs little more than a single fit." }, { title: "Strong rules / active sets", formula: "skip coefficients that will stay zero", note: "Huge speedups when p is large." }, { title: "Alternatives", formula: "LARS, proximal gradient (ISTA/FISTA)", note: "LARS computes the exact piecewise-linear lasso path." }],
      assumptions: [{ name: "Approximately sparse truth", why: "L1 works best when only some features matter.", check: "Compare against ridge with cross-validation." }, { name: "Standardised features", why: "The penalty is not scale invariant.", check: "Always standardise (or the fit is arbitrary)." }, { name: "Not too correlated (pure lasso)", why: "Lasso picks one feature from a correlated group somewhat arbitrarily.", check: "Use elastic net (α < 1) for grouped features." }],
      hyperparameters: [{ name: "λ (alpha in sklearn)", range: "1e-4 - λ_max", increasing: "More coefficients zero, more bias.", strategy: "Cross-validate along the path (LassoCV); consider the 1-SE rule." }, { name: "α (l1_ratio)", range: "0 - 1", increasing: "Sparser, less grouping.", strategy: "0.5–0.9 when features are correlated." }, { name: "tol / max_iter", range: "1e-4, 1000", increasing: "More precise convergence.", strategy: "Increase max_iter if a convergence warning appears." }],
      metrics: ["Cross-validated MSE / R²", "Number of selected features", "Stability of selection across resamples"],
      typicalUses: ["Feature selection", "High-dimensional regression (p > n)", "Interpretable models"],
      diagnostics: ["Plot the regularisation path to see the order in which features enter.", "Check selection stability with bootstrap (stability selection).", "Compare coefficients with ridge to spot correlated groups."],
      advantages: ["Built-in feature selection.", "Works when p > n.", "Convex: unique optimum (for lasso when X is in general position).", "Fast path algorithms."],
      limitations: [{ name: "Picks one of correlated features", note: "unstable selection", fix: "elastic net." }, { name: "At most n non-zeros (lasso)", note: "when p > n", fix: "elastic net." }, { name: "Biased large coefficients", note: "shrinkage applies to everything", fix: "relaxed lasso or refit OLS on the selected set." }],
      alternatives: [{ name: "Ridge regression", when: "Many small effects; prediction matters more than selection." }, { name: "Gradient boosting", when: "Non-linear effects and interactions." }, { name: "Best-subset / stepwise", when: "Very few features and an exact subset is required." }],
      pitfalls: [{ problem: "Selected features change with every resample", solution: "Correlated predictors; use elastic net or stability selection." }, { problem: "Forgot to standardise", solution: "Large-scale features are under-penalised; standardise first." }, { problem: "Interpreting zero as 'irrelevant'", solution: "It may be represented by a correlated feature that was kept." }],
      quickRef: [{ name: "Objective", formula: "‖y−Xw‖²/2n + λ(α‖w‖₁ + (1−α)‖w‖²/2)" }, { name: "Soft-threshold", formula: "sign(z)(|z|−γ)₊" }, { name: "Update", formula: "S(ρⱼ, λα)/(1 + λ(1−α))" }, { name: "λ_max", formula: "max|xⱼᵀy|/(nα)" }],
      code: `from sklearn.linear_model import LassoCV, ElasticNetCV, lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.5, .8, 1.0], cv=5))
model.fit(X, y)
coef = model[-1].coef_          # many exact zeros
alphas, coefs, _ = lasso_path(StandardScaler().fit_transform(X), y)`,
      whyChain: [
        { q: "Why does L1 produce exact zeros but L2 does not?", a: "The L1 penalty has a kink at zero with subgradient [−λ, λ]: any feature whose correlation with the residual is smaller than λ is held exactly at zero. The L2 penalty is smooth, so its minimum only scales coefficients down." },
        { q: "Why use coordinate descent?", a: "The objective is not differentiable at zero, but each one-dimensional subproblem has a closed-form soft-threshold solution. Cycling through them is simple, needs no step size and converges for this convex problem." },
        { q: "What does elastic net fix?", a: "With correlated features the lasso tends to keep one and drop the rest. The added L2 term makes the problem strictly convex and spreads weight across the group, so correlated features enter together." },
      ],
      workedExample: {
        setup: "One coordinate: ρ = 0.5, standardised feature, λ = 0.3, α = 1.",
        steps: ["Soft-threshold: S(0.5, 0.3) = 0.2.", "Denominator 1 + λ(1 − α) = 1.", "If λ were 0.6, then |ρ| ≤ λ and the coefficient would be exactly 0."],
        result: "w = 0.2 at λ = 0.3; w = 0 for any λ ≥ 0.5",
      },
    },
    mount,
  });
})();
