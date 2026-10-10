(() => {
  const { ui, col, fmt } = Lab;
  const KER = {
    rbf: { name: "RBF (squared exponential)", k: (a, b, l, s) => s * s * Math.exp(-((a - b) ** 2) / (2 * l * l)), tex: "k(x,x')=\\sigma_f^2\\exp\\!\\Big(-\\frac{(x-x')^2}{2\\ell^2}\\Big)" },
    matern: { name: "Matérn 5/2", k: (a, b, l, s) => { const r = Math.abs(a - b) / l, q = Math.sqrt(5) * r; return s * s * (1 + q + (5 * r * r) / 3) * Math.exp(-q); }, tex: "k=\\sigma_f^2\\Big(1+\\tfrac{\\sqrt5 r}{\\ell}+\\tfrac{5r^2}{3\\ell^2}\\Big)e^{-\\sqrt5 r/\\ell}" },
    periodic: { name: "periodic", k: (a, b, l, s) => s * s * Math.exp((-2 * Math.sin((Math.PI * Math.abs(a - b)) / 6) ** 2) / (l * l / 4)), tex: "k=\\sigma_f^2\\exp\\!\\Big(-\\frac{2\\sin^2(\\pi|x-x'|/p)}{\\ell^2}\\Big)" },
    linrbf: { name: "linear + RBF", k: (a, b, l, s) => 0.02 * a * b + s * s * Math.exp(-((a - b) ** 2) / (2 * l * l)), tex: "k=\\sigma_b^2xx'+\\sigma_f^2\\exp\\!\\big(-\\tfrac{(x-x')^2}{2\\ell^2}\\big)" },
  };
  function chol(A) {
    const n = A.length, L = MLU.zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-10)) : s / L[j][j];
    }
    return L;
  }
  const fwd = (L, b) => { const x = []; for (let i = 0; i < L.length; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * x[k]; x.push(s / L[i][i]); } return x; };
  const bwd = (L, b) => { const n = L.length, x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; };

  const GRID = d3.range(-10, 10.001, 0.2);
  function posterior(X, Y, P) {
    const k = (a, b) => KER[P.ker].k(a, b, P.l, P.sf);
    const n = X.length, m = GRID.length;
    const ym = 0;
    if (!n) {
      const cov = GRID.map((a) => GRID.map((b) => k(a, b)));
      return { mu: GRID.map(() => 0), cov, lml: null };
    }
    const K = X.map((a, i) => X.map((b, j) => k(a, b) + (i === j ? P.sn * P.sn : 0)));
    const L = chol(K);
    const alpha = bwd(L, fwd(L, Y.map((y) => y - ym)));
    const Ks = X.map((a) => GRID.map((b) => k(a, b)));
    const mu = GRID.map((_, j) => ym + X.reduce((s, _, i) => s + Ks[i][j] * alpha[i], 0));
    const V = GRID.map((_, j) => fwd(L, X.map((_, i) => Ks[i][j])));
    const cov = GRID.map((a, p) => GRID.map((b, q) => k(a, b) - V[p].reduce((s, v, t) => s + v * V[q][t], 0)));
    const lml = -0.5 * Y.reduce((s, y, i) => s + (y - ym) * alpha[i], 0) - L.reduce((s, r, i) => s + Math.log(r[i]), 0) - (n / 2) * Math.log(2 * Math.PI);
    return { mu, cov, lml, K };
  }
  function samples(post, Z) {
    const m = GRID.length;
    const C = post.cov.map((r, i) => r.map((v, j) => (i === j ? v + 1e-6 : v)));
    const L = chol(C);
    return Z.map((z) => GRID.map((_, i) => post.mu[i] + L[i].reduce((s, v, k) => s + (k <= i ? v * z[k] : 0), 0)));
  }

  function buildFrames(pts, P, Z) {
    const frames = [];
    const order = pts.slice();
    for (let k = 0; k <= order.length; k++) {
      const sub = order.slice(0, k);
      const post = posterior(sub.map((p) => p.x), sub.map((p) => p.y), P);
      const sd = post.cov.map((r, i) => Math.sqrt(Math.max(0, r[i])));
      frames.push({
        phase: k === 0 ? "prior" : k === order.length ? "post" : "cond", major: k === 0 || k === order.length,
        arch: k === 0 ? ["ker", "prior"] : ["ker", "K", "chol", "mean", "var"],
        title: k === 0 ? "Prior: a distribution over functions" : k === order.length ? `Posterior after all ${k} observations` : `Condition on observation ${k}: (${sub[k - 1].x.toFixed(2)}, ${sub[k - 1].y.toFixed(2)})`,
        text: k === 0 ? "Before seeing data, a GP says every function value is Gaussian with mean 0, and nearby inputs are correlated according to the kernel. The thin lines are random functions drawn from this prior." : k === order.length ? `The band is the 95% credible interval: narrow near data, widening back to the prior far away. Log marginal likelihood <b>${fmt(post.lml, 2)}</b> scores how well these hyperparameters explain the data.` : "Each new point pins the functions down near it; uncertainty collapses there and the sampled functions all pass close to it. No parameters are fitted: this is exact Bayesian conditioning.",
        tex: k === 0 ? `f\\sim\\mathcal{GP}\\big(0,\\,k(x,x')\\big),\\qquad ${KER[P.ker].tex}` : "\\mu_*=k_*^\\top(K+\\sigma_n^2I)^{-1}y,\\qquad \\sigma_*^2=k_{**}-k_*^\\top(K+\\sigma_n^2I)^{-1}k_*",
        st: { k, mu: post.mu, sd, samp: samples(post, Z), K: post.K, lml: post.lml }, m: { lml: post.lml != null ? post.lml / Math.max(1, k) : null },
      });
    }
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "gp", fps: 1.6, plane: { xLabel: "x", yLabel: "f(x)" },
      legend: ui.legend([["var(--accent)", "posterior mean", "line"], ["var(--series-1)", "95% band"], ["var(--series-5)", "sampled functions", "line"]]),
      tools: ui.btn("replay", "replay", true) + ui.btn("opt", "optimise ℓ", false, "maximise the log marginal likelihood over the length scale"),
      hint: "click to add an observation · drag / double-click to edit",
      data: { kind: "reg", presets: ["sine", "step", "cubic", "rlinear", "quad"], def: "sine", n: 8, noise: 0.05, nRange: [0, 40] },
      side: [{ title: "kernel", html: ui.select("ker", "covariance function", Object.entries(KER).map(([k, v]) => [k, v.name]), "rbf") + ui.range("l", "length scale ℓ (log₁₀)", -0.7, 1.2, 0.05, 0.25, "1.78") + ui.range("sf", "signal σ_f", 0.5, 8, 0.25, 4) + ui.range("sn", "noise σ_n", 0.01, 2, 0.01, 0.3) + ui.range("ns", "sampled functions", 0, 6, 1, 3) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "ker", label: "kernel k(x, x′)", sub: "ℓ, σ_f", x: 70, y: 40, kind: "param" },
          { id: "prior", label: "GP prior", sub: "f ~ GP(0, k)", x: 70, y: 110, kind: "op" },
          { id: "K", label: "K + σₙ²I", sub: "train covariance", x: 200, y: 40, kind: "op" },
          { id: "chol", label: "Cholesky L", sub: "O(n³)", x: 318, y: 40, kind: "loss" },
          { id: "mean", label: "mean k*ᵀα", sub: "α = K⁻¹y", x: 318, y: 120, kind: "out" },
          { id: "var", label: "variance", sub: "k** − vᵀv", x: 200, y: 120, kind: "out" },
          { id: "lml", label: "log p(y | X)", sub: "tune ℓ, σ", x: 200, y: 172, kind: "param" },
        ],
        edges: [{ from: "ker", to: "prior" }, { from: "ker", to: "K" }, { from: "K", to: "chol" }, { from: "chol", to: "mean" }, { from: "chol", to: "var" }, { from: "chol", to: "lml", bend: 30 }],
      },
      phases: { prior: { label: "prior", color: 3 }, cond: { label: "condition", color: 0 }, post: { label: "posterior", color: 2 } },
      panels: [{ id: "km", title: "kernel matrix of the observations", sub: "correlation between data points" }, { id: "kf", title: "kernel shape k(0, r)" }],
      charts: [{ id: "lml", title: "log marginal likelihood vs length scale", custom: true }],
      onFrame: render, onData: () => replay(),
    });
    lab.editable({ newPoint: (x, y) => ({ x, y }), onChange: () => replay({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    const P = () => ({ ker: lab.val("ker"), l: Math.pow(10, lab.val("l")), sf: lab.val("sf"), sn: lab.val("sn") });
    MLU.seedRng(77);
    const Zall = Array.from({ length: 6 }, () => GRID.map(() => MLU.randn()));
    function replay({ at = "start" } = {}) { lab.load(buildFrames(lab.points, P(), Zall.slice(0, lab.val("ns"))), { at, play: at === "start" }); drawLml(); drawKf(); }
    let LP = null;
    function drawLml() {
      const el = lab.charts.lml.el; el.innerHTML = "";
      if (lab.points.length < 2) return;
      const ls = d3.range(-0.7, 1.21, 0.05);
      const v = ls.map((g) => posterior(lab.points.map((p) => p.x), lab.points.map((p) => p.y), { ...P(), l: Math.pow(10, g) }).lml);
      LP = Lab.xy(el, { w: 360, h: 130, xd: [-0.7, 1.2], yd: [Math.min(...v), Math.max(...v) + 0.5], xLabel: "log₁₀ ℓ" });
      LP.line("v", ls.map((g, i) => [g, v[i]]), { color: "var(--series-5)" });
      LP.vline(lab.val("l"));
      const b = v.indexOf(Math.max(...v));
      LP.g.append("circle").attr("cx", LP.x(ls[b])).attr("cy", LP.y(v[b])).attr("r", 4).attr("fill", "var(--series-3)");
      return ls[b];
    }
    function drawKf() {
      const el = lab.R("panel-kf"); el.innerHTML = "";
      const p = P(), K = KER[p.ker];
      const KP = Lab.xy(el, { w: 360, h: 120, xd: [0, 10], yd: [0, p.sf * p.sf * 1.1 + 0.2], xLabel: "distance r" });
      KP.line("k", d3.range(0, 10.01, 0.05).map((r) => [r, K.k(0, r, p.l, p.sf)]), { color: "var(--series-1)" });
    }
    const band = L.under.append("path"), meanP = L.mid.append("path"), sampG = L.mid.append("g");
    const km = Lab.matrix(lab.R("panel-km"), { cell: 22, showValues: false, labelW: 22, maxW: 340 });
    function render(f) {
      const st = f.st;
      band.attr("d", d3.area().x((x, i) => pl.x(x)).y0((x, i) => pl.y(st.mu[i] - 1.96 * st.sd[i])).y1((x, i) => pl.y(st.mu[i] + 1.96 * st.sd[i]))(GRID)).attr("fill", "var(--series-1)").attr("fill-opacity", 0.18).attr("stroke", "none");
      meanP.attr("d", d3.line().x((x) => pl.x(x)).y((x, i) => pl.y(st.mu[i]))(GRID)).attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2.6);
      const s = sampG.selectAll("path").data(st.samp);
      s.exit().remove();
      s.enter().append("path").attr("fill", "none").merge(s).attr("d", (y) => d3.line().x((x) => pl.x(x)).y((x, i) => pl.y(y[i]))(GRID)).attr("stroke", (d, i) => col(4 + (i % 3))).attr("stroke-width", 1.2).attr("opacity", 0.75);
      lab.drawPoints({ color: (d, i) => (i < st.k ? "var(--accent)" : "var(--text-faint)"), r: (d, i) => (i === st.k - 1 ? 6.5 : 4.6), opacity: (d, i) => (i < st.k ? 1 : 0.35), stroke: (d, i) => (i === st.k - 1 ? "var(--series-4)" : "var(--bg)"), sw: (d, i) => (i === st.k - 1 ? 2.5 : 1) });
      if (st.K) km.set(st.K, { rowL: st.K.map((_, i) => i + 1), colL: st.K.map((_, i) => i + 1) }); else km.set([[0]], { rowL: ["–"], colL: ["–"] });
      lab.readout(`<div class="kv"><span>observations used</span><b>${st.k} / ${lab.points.length}</b><span>log marginal lik.</span><b>${fmt(st.lml, 2)}</b><span>ℓ · σ_f · σ_n</span><b>${P().l.toFixed(2)} · ${P().sf} · ${P().sn}</b></div><div class="note">Cost O(n³) for the Cholesky factorisation: exact GPs suit up to a few thousand points.</div>`);
    }
    lab.watch(["ker", "l", "sf", "sn", "ns"], () => replay({ at: "end" }), { format: { l: (v) => Math.pow(10, v).toFixed(2) } });
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.q('[data-b="opt"]').onclick = () => { const best = drawLml(); if (best != null) { lab.ctl("l").value = best; lab.ctl("l").dispatchEvent(new Event("input")); replay({ at: "end" }); } };
    lab.loadPreset("sine");
    return lab.destroy;
  }

  MLApp.register({
    id: "gaussian-process",
    name: "Gaussian Process Regression",
    category: "Supervised - Regression",
    tagline: "Bayesian functions with uncertainty bands",
    isNew: true,
    description: "Regression that returns a whole distribution over functions. Start from the prior, then watch each observation pin the sampled functions down and shrink the 95% band around it; change the kernel and length scale to change what 'smooth' means, and tune them by maximising the marginal likelihood.",
    info: {
      type: "Supervised - Regression (also classification with approximations). Non-parametric, Bayesian, kernel-based.",
      scenario: "Small to medium datasets where calibrated uncertainty matters: Bayesian optimisation of expensive experiments, geostatistics (kriging), surrogate models, time series with known structure.",
      inputs: "Training inputs X with noisy targets y; a kernel with hyperparameters (length scale, signal and noise variance).",
      output: "For any test input, a Gaussian predictive distribution: mean and variance.",
      intuition: {
        definition: "Instead of fitting one function, put a probability distribution over all functions, specified by how strongly any two inputs' outputs should be correlated (the kernel). Observing data conditions this distribution: functions inconsistent with the data become improbable, leaving a posterior mean and an honest uncertainty band that widens away from the data.",
        steps: ["Choose a kernel encoding smoothness, periodicity or trends.", "Build the covariance matrix of the training inputs, plus observation noise.", "Solve a linear system to get the posterior mean at test points.", "Subtract explained covariance to get the posterior variance.", "Tune hyperparameters by maximising the marginal likelihood."],
        applications: ["Bayesian optimisation of hyperparameters and lab experiments", "Kriging in geostatistics and mining", "Surrogate models for expensive simulations", "Robotics and control with uncertainty-aware dynamics"],
      },
      math: [
        { title: "GP prior", formula: "f ~ GP(m(x), k(x, x′))", note: "Any finite set of function values is jointly Gaussian with covariance given by the kernel." },
        { title: "Posterior mean", formula: "μ* = k*ᵀ(K + σₙ²I)⁻¹y", note: "A weighted combination of training targets; the weights come from kernel similarities." },
        { title: "Posterior variance", formula: "σ*² = k(x*, x*) − k*ᵀ(K + σₙ²I)⁻¹k*", note: "Prior variance minus what the data explains; does not depend on y at all." },
        { title: "Log marginal likelihood", formula: "log p(y|X) = −½yᵀ(K+σₙ²I)⁻¹y − ½log|K+σₙ²I| − (n/2)log 2π", note: "Data fit minus a complexity penalty: an automatic Occam's razor for hyperparameters." },
        { title: "Kernel as feature space", formula: "k(x, x′) = φ(x)ᵀφ(x′)", note: "A GP equals Bayesian linear regression in the (possibly infinite) feature space of the kernel." },
      ],
      pipeline: [{ label: "Kernel k" }, { label: "K + σₙ²I" }, { label: "Cholesky" }, { label: "Mean & variance" }, { label: "Tune by log ML", accent: "green" }],
      decisionFunction: { text: "p(f* | X, y, x*) = N(μ*, σ*²)", mechanism: "Predictions are full Gaussian distributions. Use the mean as a point prediction and the variance for confidence bands, active learning or acquisition functions." },
      lossFunction: { text: "maximise log p(y | X, θ) over kernel hyperparameters θ", mechanism: "There are no weights to train; only a handful of hyperparameters, chosen by gradient ascent on the marginal likelihood (or by cross-validation).", plot: { fn: (r) => Math.exp(-(r * r) / 2), domain: [0, 4], fn2: (r) => (1 + Math.sqrt(5) * r + (5 * r * r) / 3) * Math.exp(-Math.sqrt(5) * r), caption: "RBF (solid) vs Matérn 5/2 (dashed) correlation as a function of distance r/ℓ" } },
      optimization: [{ title: "Cholesky factorisation", formula: "O(n³) time, O(n²) memory", note: "Numerically stable; reused for mean, variance and likelihood." }, { title: "Sparse / inducing points", formula: "O(nm²) with m ≪ n inducing points", note: "SVGP and friends scale GPs to large data." }, { title: "Hyperparameter search", formula: "L-BFGS on log ML with restarts", note: "The likelihood surface can be multimodal." }],
      assumptions: [{ name: "Gaussian noise", why: "Gives closed-form posteriors.", check: "Use a Student-t likelihood (approximate) for outliers." }, { name: "Kernel matches the function", why: "The kernel is the model.", check: "Compare kernels by marginal likelihood." }, { name: "Stationarity (for RBF/Matérn)", why: "Same smoothness everywhere.", check: "Non-stationary kernels or input warping otherwise." }],
      hyperparameters: [{ name: "length scale ℓ", range: "data scale", increasing: "Smoother, slower-varying functions.", strategy: "Maximise the marginal likelihood." }, { name: "signal variance σ_f²", range: "≈ Var(y)", increasing: "Larger function amplitude, wider bands.", strategy: "Initialise from the variance of y." }, { name: "noise variance σₙ²", range: "≥ 0", increasing: "Smoother fit that ignores small wiggles.", strategy: "Fit by marginal likelihood; set a floor for stability." }, { name: "kernel family", range: "RBF / Matérn / periodic / sums", increasing: "Not applicable", strategy: "Matérn 5/2 is a robust default; add a periodic term for seasonality." }],
      metrics: ["Negative log predictive density", "RMSE", "Calibration of predictive intervals (coverage)"],
      typicalUses: ["Bayesian optimisation", "Spatial interpolation", "Small-data regression with uncertainty"],
      diagnostics: ["Check 95% interval coverage on held-out points.", "Inspect fitted length scales (automatic relevance determination across features).", "Watch for the noise variance collapsing to 0 (overfitting)."],
      advantages: ["Calibrated uncertainty out of the box.", "Very data-efficient.", "Hyperparameters chosen by marginal likelihood, no validation split needed.", "Kernels compose to encode prior knowledge."],
      limitations: [{ name: "Cubic scaling", note: "exact GPs cost O(n³)", fix: "sparse / inducing-point approximations." }, { name: "Kernel choice matters", note: "wrong kernel, wrong uncertainty", fix: "compare by marginal likelihood." }, { name: "High-dimensional inputs", note: "distance-based kernels weaken", fix: "ARD kernels, deep kernel learning." }],
      alternatives: [{ name: "Bayesian linear regression", when: "A fixed set of basis functions is enough." }, { name: "Random forests / boosting", when: "Large data and point predictions suffice." }, { name: "Neural networks with ensembles", when: "Huge data and approximate uncertainty is acceptable." }],
      pitfalls: [{ problem: "Cholesky fails (not positive definite)", solution: "Add jitter or increase the noise variance." }, { problem: "Band too narrow far from data", solution: "Length scale too large or σ_f too small; refit by marginal likelihood." }, { problem: "Slow on 10k+ points", solution: "Use a sparse GP." }],
      quickRef: [{ name: "Mean", formula: "k*ᵀ(K+σ²I)⁻¹y" }, { name: "Variance", formula: "k** − k*ᵀ(K+σ²I)⁻¹k*" }, { name: "RBF", formula: "σ_f² exp(−r²/2ℓ²)" }, { name: "Cost", formula: "O(n³)" }],
      code: `from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=5, normalize_y=True).fit(X, y)
mean, std = gp.predict(X_test, return_std=True)
print(gp.kernel_)          # hyperparameters tuned by marginal likelihood`,
      whyChain: [
        { q: "Why does the variance not depend on y?", a: "In the Gaussian conditioning formula the posterior covariance only involves the input locations and the kernel. Where you have measured determines how certain you are; what you measured determines the mean." },
        { q: "Why does the marginal likelihood avoid overfitting?", a: "It has a data-fit term and a log-determinant term that penalises flexible kernels (short length scales, large variances) which could explain many datasets. The best hyperparameters balance the two." },
        { q: "How do GPs relate to neural networks?", a: "A single-hidden-layer network with infinitely many random units converges to a GP (Neal, 1996); infinitely wide deep networks correspond to GPs with the neural tangent or NNGP kernels." },
      ],
      workedExample: {
        setup: "One observation y = 2 at x = 0, RBF kernel σ_f = 1, ℓ = 1, noise σₙ² = 0.1. Predict at x* = 1.",
        steps: ["K + σₙ² = 1 + 0.1 = 1.1; k* = exp(−½) ≈ 0.607.", "Mean: 0.607 · 2 / 1.1 ≈ 1.10.", "Variance: 1 − 0.607² / 1.1 ≈ 1 − 0.335 = 0.665."],
        result: "f(1) ~ N(1.10, 0.665): pulled toward the observation, still uncertain",
      },
    },
    mount,
  });
})();
