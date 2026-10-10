(() => {
  const { ui, col, fmt } = Lab;
  // Inputs are scaled to [-1, 1] before building the polynomial basis so that
  // gradient descent is well conditioned at every degree.
  const S = 9;
  const phi = (x, deg) => { const u = x / S; const r = [1]; for (let d = 1; d <= deg; d++) r.push(u ** d); return r; };
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

  function mse(pts, w, deg) { let s = 0; for (const p of pts) s += (dot(w, phi(p.x, deg)) - p.y) ** 2; return pts.length ? s / pts.length : 0; }
  function closedForm(pts, deg, lam) {
    const P = deg + 1;
    const A = MLU.zeros(P, P), b = new Array(P).fill(0);
    for (const p of pts) { const f = phi(p.x, deg); for (let i = 0; i < P; i++) { b[i] += f[i] * p.y; for (let j = 0; j < P; j++) A[i][j] += f[i] * f[j]; } }
    for (let i = 1; i < P; i++) A[i][i] += lam * pts.length; // bias is not penalised
    return { w: MLU.solveLinearSystem(A, b), A, b };
  }
  function r2(pts, w, deg) {
    const m = MLU.mean(pts.map((p) => p.y));
    const ss = pts.reduce((s, p) => s + (p.y - m) ** 2, 0) || 1;
    return 1 - (mse(pts, w, deg) * pts.length) / ss;
  }

  function buildFrames(pts, { deg, lam, solver, lr, epochs }) {
    const P = deg + 1;
    if (pts.length < 2) return [{ phase: "idle", title: "Add at least two points", text: "Click the canvas or load an example dataset.", st: { w: new Array(P).fill(0) } }];
    const frames = [];
    const reg = (w) => lam * w.slice(1).reduce((s, v) => s + v * v, 0);
    if (solver === "normal") {
      const { w } = closedForm(pts, deg, lam);
      frames.push({ phase: "basis", arch: ["x", "basis"], title: "Build the design matrix Φ", text: `Each x becomes a row of ${P} basis features [1, x, x², …]. With ${pts.length} points, Φ is ${pts.length} × ${P}.`, tex: "\\Phi=\\begin{bmatrix}1 & x_1 & \\cdots & x_1^{d}\\\\ \\vdots & & & \\vdots\\\\ 1 & x_n & \\cdots & x_n^{d}\\end{bmatrix}", st: { w: new Array(P).fill(0), hide: true }, m: {} });
      frames.push({ phase: "solve", arch: ["basis", "model", "loss"], title: "Set the gradient to zero", text: "Squared error is a convex bowl in w, so its minimum is where the gradient vanishes. That gives the normal equations; ridge adds λI to the diagonal, which also makes the matrix invertible.", tex: "\\nabla_w\\Big(\\tfrac1n\\lVert \\Phi w-y\\rVert^2+\\lambda\\lVert w\\rVert^2\\Big)=0\\;\\Rightarrow\\;(\\Phi^\\top\\Phi+n\\lambda I)\\,w=\\Phi^\\top y", st: { w: new Array(P).fill(0), hide: true }, m: {} });
      frames.push({ phase: "done", major: true, arch: ["model", "yhat", "loss"], title: "Solved in one shot", text: `MSE <b>${mse(pts, w, deg).toFixed(3)}</b>, R² <b>${r2(pts, w, deg).toFixed(3)}</b>. No iterations, no learning rate, but cost grows as O(p³) in the number of features.`, tex: "\\hat w=(\\Phi^\\top\\Phi+n\\lambda I)^{-1}\\Phi^\\top y", st: { w }, m: { mse: mse(pts, w, deg) } });
      return frames;
    }
    let w = new Array(P).fill(0);
    const keep = new Set(Lab.checkpoints(epochs, 80));
    for (let e = 0; e <= epochs; e++) {
      const L = mse(pts, w, deg) + reg(w);
      if (!isFinite(L) || L > 1e8) {
        frames.push({ phase: "diverged", major: true, arch: ["update"], title: `Diverged at epoch ${e}`, text: "The learning rate is too large: every step overshoots the minimum by more than the last, so the loss explodes. Lower η.", st: { w: frames.length ? frames[frames.length - 1].st.w : w }, m: {} });
        break;
      }
      const g = new Array(P).fill(0);
      for (const p of pts) { const f = phi(p.x, deg); const err = dot(w, f) - p.y; for (let i = 0; i < P; i++) g[i] += (2 * err * f[i]) / pts.length; }
      for (let i = 1; i < P; i++) g[i] += 2 * lam * w[i];
      if (keep.has(e)) {
        const gn = Math.sqrt(g.reduce((s, v) => s + v * v, 0));
        frames.push({
          phase: e === 0 ? "init" : "gd", major: e === 0, arch: e === 0 ? ["x", "basis", "model"] : ["model", "yhat", "loss", "grad", "update"],
          title: e === 0 ? "Start from w = 0 (a flat line)" : `Epoch ${e} · gradient step`,
          text: e === 0 ? "Gradient descent starts anywhere and walks downhill on the loss surface. The residual sticks show each point's error." : `Loss <b>${L.toFixed(3)}</b>, gradient norm <b>${gn.toFixed(3)}</b>. Each step moves w against the gradient, scaled by η = ${lr}.`,
          tex: e === 0 ? "\\hat y=w^\\top\\phi(x),\\qquad L(w)=\\tfrac1n\\sum_i(\\hat y_i-y_i)^2+\\lambda\\lVert w_{1:}\\rVert^2" : "w\\leftarrow w-\\eta\\,\\nabla L=w-\\eta\\Big(\\tfrac{2}{n}\\Phi^\\top(\\Phi w-y)+2\\lambda w\\Big)",
          st: { w: w.slice(), g: g.slice() }, m: { mse: mse(pts, w, deg), gnorm: gn },
        });
      }
      for (let i = 0; i < P; i++) w[i] -= lr * g[i];
    }
    const last = frames[frames.length - 1];
    if (last.phase !== "diverged") {
      const opt = closedForm(pts, deg, lam).w;
      const gap = mse(pts, last.st.w, deg) - mse(pts, opt, deg);
      frames.push({ ...last, phase: "done", major: true, arch: ["yhat", "loss"], title: "Training finished", text: `Final MSE <b>${mse(pts, last.st.w, deg).toFixed(3)}</b>, R² <b>${r2(pts, last.st.w, deg).toFixed(3)}</b>. The exact closed-form optimum is ${gap < 1e-3 ? "reached" : `still <b>${gap.toFixed(3)}</b> lower: run more epochs or raise η`}.` });
    }
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "linreg",
      fps: 8,
      legend: ui.legend([["var(--accent)", "fitted curve", "line"], ["var(--series-2)", "residual", "line"], ["var(--series-1)", "data"]]),
      tools: ui.btn("retrain", "retrain", true),
      hint: "click to add points · drag to move · double-click to delete",
      plane: { xLabel: "x", yLabel: "y" },
      data: { kind: "reg", presets: ["rlinear", "quad", "sine", "cubic", "step", "routliers", "hetero"], def: "rlinear", n: 40, noise: 0.3 },
      side: [
        { title: "model", html: ui.range("deg", "polynomial degree", 1, 9, 1, 1) + ui.range("lam", "ridge λ (log₁₀, −5 = off)", -5, 1, 0.25, -5, "off") },
        { title: "solver", html: ui.select("solver", "method", [["gd", "gradient descent"], ["normal", "normal equation"]], "gd") + ui.range("lr", "learning rate η", 0.01, 1.2, 0.01, 0.3) + ui.range("ep", "epochs", 20, 1500, 10, 300) + ui.check("resid", "residual sticks", true) },
      ],
      arch: {
        w: 380, h: 200,
        nodes: [
          { id: "x", label: "x", sub: "input", x: 34, y: 42, w: 46, kind: "data", info: "A single numeric feature (scaled to [−1, 1] internally)." },
          { id: "basis", label: "φ(x)", sub: "1, x, …, x^d", x: 126, y: 42, kind: "op", info: "Basis expansion. A polynomial of degree d is still linear in its weights, so the same solver works.", tex: "\\phi(x)=[1,x,x^2,\\dots,x^d]" },
          { id: "model", label: "wᵀφ", sub: "linear model", x: 236, y: 42, kind: "op", info: "A weighted sum of basis features.", tex: "\\hat y=w^\\top\\phi(x)" },
          { id: "yhat", label: "ŷ", sub: "prediction", x: 336, y: 42, w: 56, kind: "out" },
          { id: "loss", label: "MSE + λ‖w‖²", sub: "objective", x: 300, y: 140, kind: "loss", info: "Mean squared error plus the ridge penalty. Convex, so there is one global minimum.", tex: "L=\\tfrac1n\\lVert\\Phi w-y\\rVert^2+\\lambda\\lVert w\\rVert^2" },
          { id: "grad", label: "∇L", sub: "gradient", x: 186, y: 140, kind: "op", tex: "\\nabla L=\\tfrac2n\\Phi^\\top(\\Phi w-y)+2\\lambda w" },
          { id: "update", label: "w ← w − η∇L", sub: "step", x: 70, y: 140, kind: "param" },
        ],
        edges: [{ from: "x", to: "basis" }, { from: "basis", to: "model" }, { from: "model", to: "yhat" }, { from: "yhat", to: "loss" }, { from: "loss", to: "grad" }, { from: "grad", to: "update" }, { from: "update", to: "model", dash: true, label: "w" }],
      },
      phases: { init: { label: "init", color: 3 }, gd: { label: "gradient step", color: 0 }, basis: { label: "design matrix", color: 3 }, solve: { label: "normal eq.", color: 4 }, done: { label: "done", color: 2 }, diverged: { label: "diverged", color: 1 } },
      panels: [{ id: "land", title: "loss landscape", sub: "w₀ (bias) × w₁ (slope) · degree 1 only" }],
      charts: [{ id: "loss", title: "training loss", series: [{ key: "mse", label: "MSE", color: "var(--series-1)" }, { key: "gnorm", label: "‖∇L‖", color: "var(--series-4)", dash: "4 3" }], logY: true }],
      onFrame: render,
      onData: () => retrain(),
    });
    lab.editable({ newPoint: (x, y) => ({ x, y }), onChange: () => retrain({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    const params = () => ({ deg: lab.val("deg"), lam: lab.val("lam") <= -5 ? 0 : Math.pow(10, lab.val("lam")), solver: lab.val("solver"), lr: lab.val("lr"), epochs: lab.val("ep") });
    let frames = [];
    function retrain({ at = "start" } = {}) {
      frames = buildFrames(lab.points, params());
      lab.load(frames, { at, play: at === "start" });
      drawLandscape();
    }

    // ---- loss landscape (degree 1 only)
    const landEl = lab.R("panel-land");
    let land = null;
    function drawLandscape() {
      landEl.innerHTML = "";
      land = null;
      const p = params();
      if (p.deg !== 1 || lab.points.length < 2) {
        landEl.innerHTML = `<div class="lab-card-note">${p.deg !== 1 ? "With degree > 1 the loss lives in " + (p.deg + 1) + " dimensions. The coefficient bars below show where gradient descent is heading." : "Add data to see the landscape."}</div><div data-coef></div>`;
        land = { bars: Lab.bars(landEl.querySelector("[data-coef]"), { w: 360, h: 150, signed: true }) };
        return;
      }
      const opt = closedForm(lab.points, 1, p.lam).w;
      const span = Math.max(4, Math.abs(opt[0]) * 1.6, Math.abs(opt[1]) * 1.6);
      const P = Lab.xy(landEl, { w: 360, h: 210, xd: [-span, span], yd: [-span, span], xLabel: "w₀ bias", yLabel: "w₁ slope", m: { l: 34, r: 8, t: 8, b: 20 } });
      const res = 70, vals = [];
      for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
        const w0 = -span + ((i + 0.5) / res) * 2 * span, w1 = span - ((j + 0.5) / res) * 2 * span;
        vals.push(Math.log(mse(lab.points, [w0, w1], 1) + p.lam * w1 * w1 + 1e-6));
      }
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const cs = d3.contours().size([res, res]).thresholds(d3.range(lo, hi, (hi - lo) / 16))(vals);
      const sx = (P.x.range()[1] - P.x.range()[0]) / res, sy = (P.y.range()[0] - P.y.range()[1]) / res;
      const geo = d3.geoPath(d3.geoTransform({ point(px, py) { this.stream.point(P.x.range()[0] + px * sx, P.y.range()[1] + py * sy); } }));
      P.g.selectAll("path.ct").data(cs).enter().append("path").attr("class", "ct").attr("d", geo).attr("fill", "none")
        .attr("stroke", (d, i) => d3.interpolateRgb(Lab.pal()[0], Lab.pal()[1])(i / cs.length)).attr("stroke-opacity", 0.55).attr("stroke-width", 1);
      P.g.append("path").attr("d", d3.symbol(d3.symbolStar, 70)()).attr("transform", `translate(${P.x(opt[0])},${P.y(opt[1])})`).attr("fill", "var(--series-3)");
      land = { P };
    }
    function drawLandPath(i) {
      if (!land) return;
      const f = frames[i];
      if (land.bars) { land.bars.set(f.st.w.map((v, k) => ({ label: "w" + k, value: v, color: v >= 0 ? col(0) : col(1) }))); return; }
      const path = frames.slice(0, i + 1).filter((fr) => fr.st.w.length === 2).map((fr) => fr.st.w);
      land.P.line("path", path, { color: "var(--accent)", width: 1.6, curve: d3.curveLinear });
      const dots = land.P.top.selectAll("circle").data(path.length ? [path[path.length - 1]] : []);
      dots.exit().remove();
      dots.enter().append("circle").attr("r", 4.5).merge(dots).attr("cx", (d) => land.P.x(d[0])).attr("cy", (d) => land.P.y(d[1])).attr("fill", "var(--accent)");
    }

    const curve = L.mid.append("path").attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2.4);
    function render(f, i) {
      const { deg } = params();
      const w = f.st.w;
      const xs = d3.range(-10, 10.01, 0.1);
      curve.attr("d", f.st.hide ? null : d3.line().x((x) => pl.x(x)).y((x) => pl.y(Math.max(-14, Math.min(14, dot(w, phi(x, deg))))))(xs));
      const res = L.links.selectAll("line").data(lab.val("resid") && !f.st.hide ? lab.points : []);
      res.exit().remove();
      res.enter().append("line").merge(res)
        .attr("x1", (d) => pl.x(d.x)).attr("x2", (d) => pl.x(d.x)).attr("y1", (d) => pl.y(d.y)).attr("y2", (d) => pl.y(Math.max(-10, Math.min(10, dot(w, phi(d.x, deg))))))
        .attr("stroke", "var(--series-2)").attr("stroke-width", 1.3).attr("stroke-opacity", 0.75);
      lab.drawPoints({ color: (d) => (d.outlier ? "var(--series-4)" : "var(--series-1)") });
      drawLandPath(i);
      const pts = lab.points;
      const eq = deg === 1 ? `ŷ = ${fmt(w[1] / S, 3)}·x ${w[0] >= 0 ? "+" : "−"} ${fmt(Math.abs(w[0]), 3)}` : `degree-${deg} polynomial`;
      lab.readout(`<div class="kv"><span>fit</span><b>${eq}</b><span>MSE</span><b>${pts.length ? fmt(mse(pts, w, deg)) : "–"}</b><span>R²</span><b>${pts.length > 1 ? fmt(r2(pts, w, deg)) : "–"}</b><span>‖w‖</span><b>${fmt(Math.hypot(...w.slice(1)))}</b></div>`);
    }

    lab.watch(["deg", "lam", "solver", "lr", "ep"], () => retrain(), { format: { lam: (v) => (v <= -5 ? "off" : "1e" + v) } });
    lab.watch(["resid"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    lab.loadPreset("rlinear");
    return lab.destroy;
  }

  MLApp.register({
    id: "linear-regression",
    name: "Linear Regression",
    category: "Supervised - Regression",
    tagline: "polynomial + ridge, GD vs normal equation",
    description: "Fit a line or polynomial by least squares. Watch gradient descent walk down the loss surface frame by frame, compare it with the one-shot normal equation, and see ridge regularisation tame a high-degree fit.",
    info: {
      type: "Supervised - Regression. Parametric linear model (generalized here with a polynomial basis and an L2/ridge penalty).",
      scenario: "You need a simple, interpretable baseline for predicting a continuous target from numeric features - trend estimation, forecasting with few features, or as a first model before trying anything non-linear.",
      inputs: "A feature value x (expanded into a polynomial basis [1, x, x², …, x^d]) and a continuous target y for each training point.",
      intuition: {
        definition: "Fit the straight line (or polynomial curve) that makes the <b>squared vertical distances</b> to the training points as small as possible. Because the loss is a convex quadratic, the best fit has a closed-form solution - no iteration required.",
        steps: [
          "Assume the target is a weighted sum of the features plus noise.",
          "Measure fit by summed squared residuals (y − ŷ)².",
          "Set the gradient to zero → the normal equations solve for β directly.",
          "A polynomial basis keeps the model linear in β while bending the curve.",
        ],
        applications: [
          "Sales / demand forecasting from spend and seasonality",
          "Estimating the price effect of a house's square footage",
          "Dose–response curves in biology",
          "Calibrating a sensor against a reference instrument",
          "A reference baseline before trying any non-linear model",
        ],
      },
      math: [
        { title: "Design matrix", formula: "X = [1, x, x², …, x^d]  (one row per point)", note: "Polynomial features go in the columns, so the model stays <i>linear in the parameters</i> even when the curve is not." },
        { title: "Model", formula: "ŷ = Xβ,   y = Xβ + ε,  ε ~ N(0, σ²)", note: "Gaussian noise is what makes least squares the maximum-likelihood estimator." },
        { title: "Residual sum of squares", formula: "RSS(β) = ‖y − Xβ‖² = Σᵢ(yᵢ − ŷᵢ)²", note: "Convex quadratic bowl in β - one minimum, no local traps." },
        { title: "Normal equations", formula: "∇RSS = −2Xᵀ(y − Xβ) = 0  ⟹  XᵀXβ = Xᵀy", note: "Setting the gradient to zero gives a linear system." },
        { title: "Closed-form solution", formula: "β = (XᵀX + λI)⁻¹ Xᵀy", note: "λ = 0 is ordinary least squares. Any λ > 0 (ridge) also makes XᵀX invertible when features are collinear." },
      ],
      pipeline: [
        { label: "Raw feature x", note: "scalar input" },
        { label: "Basis expansion", note: "[1, x, …, x^d]" },
        { label: "Normal equations", note: "(XᵀX+λI)⁻¹Xᵀy" },
        { label: "Coefficients β", note: "fitted once" },
        { label: "Prediction ŷ", note: "ŷ = xᵀβ", accent: "green" },
      ],
      decisionFunction: {
        text: "ŷ = β₀ + β₁x + β₂x² + … + β_d·x^d",
        mechanism: "Prediction is a fixed weighted sum of the (possibly polynomial-transformed) input - a direct evaluation, with no iterative inference step once β is fit.",
        plot: { fn: (r) => r * r, domain: [-4, 4], color: "var(--accent)", caption: "squared error vs residual (y − ŷ) - this exact shape is what's being minimized below" },
      },
      lossFunction: {
        text: "L(β) = Σᵢ(yᵢ − ŷᵢ)² + λ·Σβₖ²",
        mechanism: "A convex quadratic in β, so it has one closed-form minimizer via the normal equations β = (XᵀX + λI)⁻¹Xᵀy - no gradient descent needed.",
      },
      optimization: [
        { title: "Closed form vs. gradient descent", formula: "O(p³) inversion  vs.  O(m·p) per step", note: "The normal equations win for a few thousand features; beyond that the p×p inversion dominates and gradient descent (or SGD) is cheaper." },
        { title: "Gradient (if solved iteratively)", formula: "∇L = −(2/m)·Xᵀ(y − Xβ) + 2λβ", note: "Same 'error × feature' form as logistic regression, because both are generalized linear models." },
        { title: "Conditioning", formula: "κ(XᵀX) = κ(X)²", note: "Squaring the condition number is why a raw normal-equation solve is numerically fragile; production solvers use QR or SVD instead." },
      ],
      output: "A continuous predicted value ŷ for any input x, plus the fitted coefficient vector β.",
      assumptions: [
        { name: "Linearity in parameters", why: "A genuinely curved relationship is systematically mis-fit.", check: "Plot residuals against fitted values - any pattern means the form is wrong." },
        { name: "Independent errors", why: "Autocorrelated residuals (common in time series) collapse the standard errors.", check: "Durbin–Watson statistic; plot residuals in collection order." },
        { name: "Homoscedasticity", why: "Non-constant error variance makes OLS inefficient and the confidence intervals wrong.", check: "Residual-vs-fitted plot should be a flat band, not a fan." },
        { name: "Normal errors", why: "Needed for exact t/F inference - not for the point estimate itself.", check: "Q–Q plot of residuals." },
        { name: "No perfect multicollinearity", why: "XᵀX becomes singular and β is undefined.", check: "VIF per feature; ridge fixes it outright." },
      ],
      regularization: [
        { name: "OLS", formula: "‖y − Xβ‖²", note: "No penalty. Unbiased, but high variance when features are many or correlated." },
        { name: "Ridge (L2)", formula: "‖y − Xβ‖² + λ‖β‖²", note: "Shrinks coefficients smoothly; keeps every feature. Best when predictors are correlated." },
        { name: "Lasso (L1)", formula: "‖y − Xβ‖² + λ‖β‖₁", note: "Drives coefficients to exactly zero - automatic feature selection. No closed form; solved by coordinate descent." },
        { name: "Elastic net", formula: "‖y − Xβ‖² + λ₁‖β‖₁ + λ₂‖β‖²", note: "Selects features while keeping correlated groups intact." },
      ],
      hyperparameters: [
        { name: "degree", range: "1 – 10", increasing: "More flexible curve; variance and overfitting rise sharply.", strategy: "Pick the smallest degree whose validation error stops improving; watch the ends of the range where polynomials swing wildly." },
        { name: "λ (ridge)", range: "1e-4 – 100", increasing: "Stronger shrinkage → higher bias, lower variance; β → 0 in the limit.", strategy: "Log-scale grid with k-fold CV. Always standardize features first - the penalty is scale-dependent." },
        { name: "fit intercept", range: "true / false", increasing: "-", strategy: "Keep it on unless the data is already centred. The intercept is never penalized." },
      ],
      metrics: ["R² (coefficient of determination)", "Adjusted R² (penalizes extra features)", "Mean Squared Error (MSE) / RMSE", "Mean Absolute Error (MAE)", "AIC / BIC for model selection"],
      typicalUses: ["Trend estimation and forecasting", "Interpretable baseline before non-linear models", "Quantifying a feature's effect via its coefficient", "Denoising a noisy 1D signal"],
      diagnostics: [
        "Residual-vs-fitted plot: should be structureless noise. A curve means the degree is too low; a fan means heteroscedasticity.",
        "Q–Q plot of residuals to check the normality assumption behind the confidence intervals.",
        "Cook's distance to find single points that are dragging the whole fit.",
        "A large gap between R² and adjusted R² means the extra features are not earning their keep.",
      ],
      advantages: [
        "Closed-form solution - deterministic, fast, and no hyperparameter search for plain OLS.",
        "Coefficients read directly as 'unit change in y per unit change in xᵢ'.",
        "Well-understood statistical theory: confidence intervals, p-values, and prediction intervals all come free.",
        "Extremely cheap at prediction time; trivial to deploy anywhere.",
        "Ridge makes it stable even with more features than observations.",
      ],
      limitations: [
        { name: "Only linear in the parameters", note: "genuinely non-linear structure needs a basis expansion you chose in advance", fix: "polynomial/spline features, or a tree ensemble." },
        { name: "Very outlier-sensitive", note: "squared error means one bad point can dominate the fit", fix: "Huber loss, RANSAC, or winsorizing." },
        { name: "Polynomial instability", note: "high-degree fits oscillate violently near the edges of the data (Runge's phenomenon)", fix: "splines, or keep the degree low and add ridge." },
        { name: "Breaks under multicollinearity", note: "coefficients become huge and sign-unstable", fix: "ridge, PCA, or drop redundant features." },
        { name: "Extrapolates badly", note: "predictions outside the training range are unconstrained", fix: "don't extrapolate; flag out-of-range inputs." },
      ],
      alternatives: [
        { name: "Ridge / lasso", when: "Many features, or features that are correlated." },
        { name: "Gradient boosting", when: "Tabular data with interactions and non-linearity, and interpretability is secondary." },
        { name: "Splines / GAM", when: "Smooth non-linearity, but you still want an interpretable additive model." },
        { name: "Huber / quantile regression", when: "Heavy-tailed noise or influential outliers." },
      ],
      pitfalls: [
        { problem: "R² keeps rising as you add features", solution: "R² never decreases with more features - use adjusted R² or a held-out split." },
        { problem: "Coefficients flip sign when a feature is added", solution: "Classic multicollinearity. Check VIF and apply ridge." },
        { problem: "Great train fit, poor test fit", solution: "Degree too high. Lower it or raise λ." },
        { problem: "Ridge penalty behaves erratically", solution: "Standardize features - the L2 penalty is not scale-invariant." },
        { problem: "Singular matrix error", solution: "Duplicate or perfectly collinear columns; drop them or use a nonzero λ." },
      ],
      quickRef: [
        { name: "Model", formula: "ŷ = Xβ" },
        { name: "Loss (OLS)", formula: "RSS = ‖y − Xβ‖²" },
        { name: "Normal equations", formula: "XᵀXβ = Xᵀy" },
        { name: "Ridge solution", formula: "β = (XᵀX + λI)⁻¹Xᵀy" },
        { name: "Gradient", formula: "∇L = −2Xᵀ(y − Xβ) + 2λβ" },
        { name: "R²", formula: "1 − RSS/TSS" },
        { name: "Adjusted R²", formula: "1 − (1−R²)(m−1)/(m−p−1)" },
        { name: "Slope (simple case)", formula: "β₁ = Sxy / Sxx" },
      ],
      code: `from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

pipe = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),          # the L2 penalty is scale-sensitive
    Ridge(alpha=1.0),          # alpha == lambda; 0 would be plain OLS
)

grid = GridSearchCV(
    pipe,
    {"polynomialfeatures__degree": [1, 2, 3, 5],
     "ridge__alpha": [1e-3, 1e-2, 0.1, 1, 10]},
    cv=5, scoring="neg_mean_squared_error",
).fit(X_train, y_train)

pred = grid.predict(X_test)
print(grid.best_params_)
print("R2:", r2_score(y_test, pred),
      "RMSE:", mean_squared_error(y_test, pred, squared=False))`,
      whyChain: [
        { q: "Why squared error rather than absolute error?", a: "Squaring makes the loss differentiable everywhere and convex-quadratic, which yields the closed-form normal equations. It's also the maximum-likelihood estimator under Gaussian noise." },
        { q: "So why would you ever use absolute error?", a: "Squared error weights a residual of 10 a hundred times more than a residual of 1, so outliers dominate. MAE (or Huber) is robust to them - you trade the closed form for robustness." },
        { q: "How can a polynomial fit still be 'linear' regression?", a: "Linearity refers to the parameters, not the input. ŷ = β₀ + β₁x + β₂x² is linear in β, so the same normal equations apply to the expanded design matrix." },
        { q: "Why does ridge fix multicollinearity?", a: "Collinear columns make XᵀX near-singular, so its inverse blows up. Adding λI lifts every eigenvalue by λ, guaranteeing invertibility and bounding ‖β‖." },
        { q: "Is ridge biased?", a: "Yes - deliberately. It accepts a little bias for a large drop in variance, which usually lowers total test error. That's the bias–variance trade-off made explicit." },
        { q: "Why must you standardize before ridge but not before OLS?", a: "OLS is equivariant to feature scaling; the fit is unchanged. Ridge penalizes Σβⱼ², so a feature measured in millimetres gets a far larger coefficient - and therefore a far larger penalty - than the same feature in metres." },
        { q: "Adding a useless feature raised my R². Is the model better?", a: "No. R² is non-decreasing in the number of features by construction. Use adjusted R², AIC, or a held-out test set." },
        { q: "Why avoid inverting XᵀX directly in production?", a: "Forming XᵀX squares the condition number, so precision is lost. QR decomposition or SVD solves the same least-squares problem far more stably." },
      ],
      parameters: [
        { name: "degree", effect: "Order of the polynomial basis. Higher degree fits more flexible curves but raises variance/overfitting risk, especially with few points." },
        { name: "λ (ridge)", effect: "L2 penalty strength. Larger λ shrinks coefficients toward zero, trading a bit of bias for lower variance." },
      ],
      workedExample: {
        setup: "Fit ŷ = β₀ + β₁x (degree 1, no ridge) to points (1,2), (2,3), (3,5), (4,4).",
        steps: [
          "Means: x̄ = (1+2+3+4)/4 = 2.5, ȳ = (2+3+5+4)/4 = 3.5.",
          "Deviations (x−x̄): −1.5, −0.5, 0.5, 1.5. Deviations (y−ȳ): −1.5, −0.5, 1.5, 0.5.",
          "Sxy = Σ(x−x̄)(y−ȳ) = 2.25 + 0.25 + 0.75 + 0.75 = 4.0.",
          "Sxx = Σ(x−x̄)² = 2.25 + 0.25 + 0.25 + 2.25 = 5.0.",
          "β₁ = Sxy / Sxx = 4.0 / 5.0 = 0.8.",
          "β₀ = ȳ − β₁x̄ = 3.5 − 0.8×2.5 = 1.5.",
        ],
        result: "ŷ = 1.5 + 0.8x → prediction at x=5: ŷ = 1.5 + 0.8×5 = 5.5",
      },
    },
    mount,
  });
})();
