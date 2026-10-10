(() => {
  const { ui, col, fmt } = Lab;
  const S = 9;
  function kernelFn(type, gamma, degree) {
    if (type === "linear") return (a, b) => (a.x * b.x + a.y * b.y) / (S * S);
    if (type === "poly") return (a, b) => ((a.x * b.x + a.y * b.y) / (S * S) + 1) ** degree;
    return (a, b) => Math.exp(-gamma * ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) / (S * S) * 4);
  }
  const KTEX = {
    linear: "K(x,x')=x^\\top x'",
    poly: "K(x,x')=(x^\\top x'+1)^{d}",
    rbf: "K(x,x')=\\exp\\!\\big(-\\gamma\\lVert x-x'\\rVert^2\\big)",
  };

  // Simplified SMO (Platt 1998, as taught in CS229): repeatedly pick a pair of
  // multipliers that violate the KKT conditions and optimise them jointly in
  // closed form, clipping to the box [0, C].
  function smoFrames(pts, K, C, maxFrames = 140) {
    const n = pts.length, y = pts.map((p) => (p.label === 1 ? 1 : -1));
    const Km = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => K(pts[i], pts[j])));
    const a = new Float64Array(n);
    let b = 0;
    const f = (i) => { let s = b; for (let j = 0; j < n; j++) if (a[j]) s += a[j] * y[j] * Km[j][i]; return s; };
    const dual = () => { let s = 0; for (let i = 0; i < n; i++) s += a[i]; let q = 0; for (let i = 0; i < n; i++) if (a[i]) for (let j = 0; j < n; j++) if (a[j]) q += a[i] * a[j] * y[i] * y[j] * Km[i][j]; return s - q / 2; };
    const snap = (extra) => ({ a: Array.from(a), b, ...extra });
    const frames = [{ phase: "init", major: true, arch: ["k", "dual"], title: "All multipliers αᵢ start at 0", text: "The dual problem has one multiplier per training point. A point only influences the boundary if its α > 0; those points are the <b>support vectors</b>. Right now there are none.", tex: "\\max_{\\alpha}\\;\\sum_i\\alpha_i-\\tfrac12\\sum_{i,j}\\alpha_i\\alpha_j y_iy_jK(x_i,x_j)\\quad\\text{s.t. }0\\le\\alpha_i\\le C,\\;\\sum_i\\alpha_iy_i=0", st: snap({}), m: { dual: 0, sv: 0 } }];
    const tol = 1e-3;
    let passes = 0, iter = 0, updates = 0;
    const every = Math.max(1, Math.round(n / 12));
    while (passes < 6 && iter < 80) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        const Ei = f(i) - y[i];
        if ((y[i] * Ei < -tol && a[i] < C) || (y[i] * Ei > tol && a[i] > 0)) {
          let j = MLU.randInt(n - 1); if (j >= i) j++;
          const Ej = f(j) - y[j];
          const ai = a[i], aj = a[j];
          let Lo, Hi;
          if (y[i] !== y[j]) { Lo = Math.max(0, aj - ai); Hi = Math.min(C, C + aj - ai); } else { Lo = Math.max(0, ai + aj - C); Hi = Math.min(C, ai + aj); }
          if (Hi - Lo < 1e-9) continue;
          const eta = 2 * Km[i][j] - Km[i][i] - Km[j][j];
          if (eta >= 0) continue;
          let nj = aj - (y[j] * (Ei - Ej)) / eta;
          nj = Math.min(Hi, Math.max(Lo, nj));
          if (Math.abs(nj - aj) < 1e-5) continue;
          const ni = ai + y[i] * y[j] * (aj - nj);
          a[i] = ni; a[j] = nj;
          const b1 = b - Ei - y[i] * (ni - ai) * Km[i][i] - y[j] * (nj - aj) * Km[i][j];
          const b2 = b - Ej - y[i] * (ni - ai) * Km[i][j] - y[j] * (nj - aj) * Km[j][j];
          b = ni > 0 && ni < C ? b1 : nj > 0 && nj < C ? b2 : (b1 + b2) / 2;
          changed++; updates++;
          if (updates % every === 0 && frames.length < maxFrames) {
            const sv = Array.from(a).filter((v) => v > 1e-6).length;
            frames.push({ phase: "smo", arch: ["pair", "clip", "bias", "dual"], title: `SMO update ${updates}: optimise α${i} and α${j} together`, text: `Pick a pair violating the KKT conditions, solve their 2-variable QP exactly, clip to the box [0, C], then fix b. Support vectors so far: <b>${sv}</b>. The dual objective only ever rises.`, tex: "\\alpha_j^{new}=\\mathrm{clip}_{[L,H]}\\Big(\\alpha_j-\\frac{y_j(E_i-E_j)}{\\eta}\\Big),\\quad \\eta=2K_{ij}-K_{ii}-K_{jj}", st: snap({ pair: [i, j] }), m: { dual: dual(), sv } });
          }
        }
      }
      iter++;
      passes = changed === 0 ? passes + 1 : 0;
    }
    const sv = Array.from(a).filter((v) => v > 1e-6).length;
    const bounded = Array.from(a).filter((v) => v > C - 1e-6).length;
    frames.push({ phase: "done", major: true, arch: ["k", "sum", "out"], title: `Converged: ${sv} support vectors`, text: `Only the ringed points define the model; delete any other point and the boundary will not move. ${bounded} of them sit at α = C (inside the margin or misclassified). The dashed lines are f(x) = ±1, the margin.`, tex: "f(x)=\\sum_{i\\in SV}\\alpha_iy_iK(x_i,x)+b,\\qquad \\hat y=\\operatorname{sign}f(x)", st: snap({}), m: { dual: dual(), sv } });
    return { frames, y };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "svm", fps: 5, classes: 2, classNames: ["−1", "+1"],
      legend: ui.legend([["var(--text)", "support vector", "ring"], ["var(--text)", "f(x)=0", "line"], ["var(--text)", "margin f=±1", "dash"]]),
      tools: ui.btn("retrain", "retrain", true),
      hint: "click adds the selected class · drag points across the margin and watch which ones matter",
      data: { kind: "class", presets: ["linear", "circles", "moons", "xor", "overlap", "spiral"], def: "linear", n: 60, noise: 0.3, nRange: [10, 140] },
      side: [{ title: "kernel", html: ui.select("kern", "kernel", [["linear", "linear"], ["poly", "polynomial"], ["rbf", "RBF (Gaussian)"]], "linear") + ui.range("C", "C (log₁₀)", -2, 3, 0.25, 0.5, "3.16") + ui.range("gamma", "γ for RBF (log₁₀)", -1.5, 2, 0.25, 0.5, "3.16") + ui.range("deg", "degree for poly", 2, 6, 1, 3) }],
      arch: {
        w: 380, h: 200,
        nodes: [
          { id: "x", label: "x", sub: "query", x: 30, y: 44, w: 44, kind: "data" },
          { id: "k", label: "K(xᵢ, x)", sub: "kernel trick", x: 124, y: 44, kind: "op", info: "Similarity to each support vector. The kernel computes an inner product in a (possibly infinite) feature space without ever building it." },
          { id: "sum", label: "Σ αᵢyᵢK + b", sub: "weighted vote", x: 238, y: 44, kind: "op" },
          { id: "out", label: "sign", sub: "±1", x: 340, y: 44, w: 56, kind: "out" },
          { id: "dual", label: "dual QP", sub: "max W(α)", x: 46, y: 150, kind: "loss", tex: "W(\\alpha)=\\sum\\alpha_i-\\tfrac12\\sum\\alpha_i\\alpha_jy_iy_jK_{ij}" },
          { id: "pair", label: "pick (i, j)", sub: "KKT violators", x: 140, y: 150, kind: "op" },
          { id: "clip", label: "clip to [0,C]", sub: "box constraint", x: 240, y: 150, kind: "param" },
          { id: "bias", label: "update b", x: 336, y: 150, w: 64, kind: "param" },
        ],
        edges: [{ from: "x", to: "k" }, { from: "k", to: "sum" }, { from: "sum", to: "out" }, { from: "dual", to: "pair" }, { from: "pair", to: "clip" }, { from: "clip", to: "bias" }, { from: "bias", to: "sum", dash: true, label: "α, b" }],
      },
      phases: { init: { label: "init", color: 3 }, smo: { label: "SMO step", color: 0 }, done: { label: "converged", color: 2 } },
      panels: [{ id: "alpha", title: "multipliers α of the support vectors" }],
      charts: [{ id: "dual", title: "dual objective & support vectors", series: [{ key: "dual", label: "W(α)", color: "var(--series-1)" }, { key: "sv", label: "# SV", color: "var(--series-4)", dash: "4 3", digits: 0 }] }],
      onFrame: render, onData: () => retrain(),
    });
    lab.editable({ onChange: () => retrain({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    const P = () => ({ kern: lab.val("kern"), C: Math.pow(10, lab.val("C")), gamma: Math.pow(10, lab.val("gamma")), deg: lab.val("deg") });
    let K = null, ys = [];
    function retrain({ at = "start" } = {}) {
      const p = P();
      K = kernelFn(p.kern, p.gamma, p.deg);
      lab.ctl("gamma").closest(".field").style.display = p.kern === "rbf" ? "" : "none";
      lab.ctl("deg").closest(".field").style.display = p.kern === "poly" ? "" : "none";
      if (!lab.points.some((q) => q.label === 0) || !lab.points.some((q) => q.label === 1)) {
        lab.load([{ phase: "idle", title: "Need both classes", text: "Add points of each class.", st: { a: [], b: 0 } }]);
        return;
      }
      MLU.seedRng(5);
      const r = smoFrames(lab.points, K, p.C);
      ys = r.y;
      lab.load(r.frames, { at, play: at === "start" });
    }
    const alphaBars = Lab.bars(lab.R("panel-alpha"), { w: 360, h: 130 });
    const ringG = L.top.append("g"), pairG = L.top.append("g");
    function render(f) {
      const st = f.st, pts = lab.points;
      if (!st.a.length) { L.bg.selectAll("*").remove(); lab.drawPoints({}); return; }
      const sv = st.a.map((v, i) => i).filter((i) => st.a[i] > 1e-6);
      const dec = (x, y) => { let s = st.b; for (const i of sv) s += st.a[i] * ys[i] * K(pts[i], { x, y }); return s; };
      const grid = Lab.sampleGrid(pl, dec, 80);
      Lab.paintGrid(L.bg, pl, grid, (v) => Lab.divergeColor(v, 2.2, 0.5));
      Lab.contour(L.under, pl, grid, [0], { cls: "iso-main", width: 2.2 });
      Lab.contour(L.under, pl, grid, [-1, 1], { cls: "iso-dash", width: 1.2, dash: "5 4", stroke: "var(--text-dim)" });
      lab.drawPoints({ color: (d) => col(d.label) });
      const C = P().C;
      const rings = ringG.selectAll("circle").data(sv);
      rings.exit().remove();
      rings.enter().append("circle").attr("fill", "none").attr("pointer-events", "none").merge(rings)
        .attr("cx", (i) => pl.x(pts[i].x)).attr("cy", (i) => pl.y(pts[i].y)).attr("r", 9)
        .attr("stroke", "var(--text)").attr("stroke-width", (i) => (st.a[i] > C - 1e-6 ? 1.2 : 2)).attr("stroke-dasharray", (i) => (st.a[i] > C - 1e-6 ? "2 2" : null));
      pairG.selectAll("*").remove();
      if (st.pair) {
        const [i, j] = st.pair;
        pairG.append("line").attr("x1", pl.x(pts[i].x)).attr("y1", pl.y(pts[i].y)).attr("x2", pl.x(pts[j].x)).attr("y2", pl.y(pts[j].y)).attr("stroke", "var(--series-4)").attr("stroke-width", 2).attr("stroke-dasharray", "4 3");
        for (const k of [i, j]) pairG.append("circle").attr("cx", pl.x(pts[k].x)).attr("cy", pl.y(pts[k].y)).attr("r", 13).attr("fill", "none").attr("stroke", "var(--series-4)").attr("stroke-width", 2.4);
      }
      const svs = sv.slice().sort((p, q) => st.a[q] - st.a[p]).slice(0, 24);
      alphaBars.set(svs.map((i) => ({ label: "#" + i, value: st.a[i], color: col(pts[i].label) })), { values: false });
      let ok = 0; for (let i = 0; i < pts.length; i++) if ((dec(pts[i].x, pts[i].y) >= 0 ? 1 : 0) === pts[i].label) ok++;
      lab.readout(`<div class="kv"><span>support vectors</span><b>${sv.length} / ${pts.length}</b><span>at bound α = C</span><b>${sv.filter((i) => st.a[i] > C - 1e-6).length}</b><span>bias b</span><b>${fmt(st.b)}</b><span>train accuracy</span><b>${((ok / pts.length) * 100).toFixed(1)}%</b></div><div class="note">${Lab.texHTML(KTEX[P().kern], false)} · dotted rings mark α = C</div>`);
    }
    lab.watch(["kern", "C", "gamma", "deg"], () => retrain(), { format: { C: (v) => fmt(Math.pow(10, v), 2), gamma: (v) => fmt(Math.pow(10, v), 2) } });
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    lab.loadPreset("linear");
    return lab.destroy;
  }

  MLApp.register({
    id: "svm",
    name: "Support Vector Machine",
    category: "Supervised - Classification",
    tagline: "SMO solver, margins & support vectors",
    description: "Maximum-margin classification with linear, polynomial and RBF kernels. Watch the SMO solver optimise pairs of multipliers, support vectors light up, and the margin settle; drag points to see that only support vectors matter.",
    info: {
      type: "Supervised - Classification (an SVR regression variant also exists). Max-margin kernel method.",
      scenario: "Classification with a clear or kernel-separable margin, on small-to-medium datasets, when a robust, theoretically-grounded margin-based classifier is wanted; kernels let it handle non-linear boundaries.",
      inputs: "Feature vectors x, binary labels y ∈ {−1, +1}, and a choice of kernel (linear / poly / RBF).",
      intuition: {
        definition: "Of all the hyperplanes that separate the classes, pick the one with the <b>widest margin</b>. Only the points sitting on that margin (the support vectors) matter, and a kernel lets the same machinery draw curved boundaries by working in a higher-dimensional space it never explicitly builds.",
        steps: [
          "Find the separating hyperplane with maximum distance to the nearest point.",
          "Allow a few violations, paid for at rate C (soft margin).",
          "Only points on or inside the margin get non-zero weight.",
          "Swap the dot product for a kernel to bend the boundary.",
        ],
        applications: [
          "Text and document classification with thousands of sparse features",
          "Gene expression and other wide, low-sample bioinformatics data",
          "Image classification before deep learning took over",
          "Handwriting recognition",
          "Novelty detection via the one-class variant",
        ],
      },
      math: [
        { title: "Margin", formula: "margin = 2 / ‖w‖,  subject to  yᵢ(wᵀxᵢ + b) ≥ 1", note: "Maximising the margin is the same as minimising ‖w‖. The constraint says every point sits on the correct side, at least one unit away in scaled terms." },
        { title: "Hard-margin primal", formula: "min (1/2)‖w‖²   s.t.  yᵢ(wᵀxᵢ + b) ≥ 1", note: "A convex quadratic program with a unique solution, but it has no answer at all if the data is not perfectly separable." },
        { title: "Soft margin", formula: "min (1/2)‖w‖² + C·Σᵢ ξᵢ   s.t.  yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0", note: "Slack variables ξᵢ buy permission to violate the margin. C sets the price." },
        { title: "Dual form", formula: "max Σᵢαᵢ − (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ·K(xᵢ,xⱼ)   s.t.  0 ≤ αᵢ ≤ C,  Σᵢαᵢyᵢ = 0", note: "The data appears only inside inner products, which is precisely what makes the kernel trick possible." },
        { title: "Kernel trick", formula: "K(xᵢ,xⱼ) = φ(xᵢ)ᵀφ(xⱼ)", note: "Compute the inner product in a high-dimensional space without ever forming φ(x). The RBF kernel corresponds to an infinite-dimensional space." },
        { title: "Decision function", formula: "f(x) = Σ_{i ∈ SV} αᵢyᵢK(xᵢ, x) + b,  ŷ = sign f(x)", note: "Only support vectors contribute, so the stored model is usually far smaller than the training set." },
      ],
      pipeline: [
        { label: "Features x", note: "scaled" },
        { label: "Kernel K", note: "linear / poly / RBF" },
        { label: "Solve dual QP", note: "get α, b" },
        { label: "Support vectors", note: "αᵢ > 0 only" },
        { label: "sign f(x)", note: "class ±1", accent: "green" },
      ],
      decisionFunction: {
        text: "ŷ(x) = sign( Σⱼ αⱼyⱼK(xⱼ, x) + b )",
        mechanism: "The decision is a weighted sum of kernel similarities to the support vectors (points with αⱼ>0) rather than an explicit weight vector - this is what lets the model work in the implicit feature space a kernel induces.",
      },
      lossFunction: {
        text: "L = Σᵢ max(0, 1 − yᵢf(xᵢ)) + (1/2C)‖w‖²  (hinge loss, soft-margin)",
        mechanism: "Hinge loss is exactly zero once a point is correctly classified with margin ≥1, so only points near/inside the margin influence the solution - trained here via the kernelized Pegasos online sub-gradient method rather than full quadratic programming.",
        plot: { fn: (z) => Math.max(0, 1 - z), domain: [-2, 3], yDomain: [0, 3], color: "var(--accent)", fn2: (z) => Math.log(1 + Math.exp(-z)), color2: "var(--text-faint)", caption: "hinge loss (solid) vs margin z=y·f(x) - zero past margin 1; log-loss (dashed) never reaches exactly zero, which is why SVM solutions are sparse" },
      },
      optimization: [
        { title: "Why the dual", formula: "primal: p unknowns    dual: m unknowns", note: "Solving the dual is what exposes the kernel. It also wins outright when features vastly outnumber samples, which is the usual text-classification shape." },
        { title: "SMO", formula: "optimise two αᵢ at a time, analytically", note: "Sequential Minimal Optimization is the standard solver (libsvm). Each two-variable subproblem has a closed form, so no general QP library is needed." },
        { title: "Pegasos (used in this demo)", formula: "w := (1 − 1/t)·w + η·yᵢ·xᵢ  when the margin is violated", note: "A stochastic sub-gradient method on the hinge loss. Converges in time independent of the dataset size, which makes it practical online." },
        { title: "KKT conditions", formula: "αᵢ = 0 → outside margin;  0 < αᵢ < C → on margin;  αᵢ = C → violating", note: "These three cases classify every training point and are how solvers decide when they are done." },
        { title: "Cost", formula: "train: O(m²) to O(m³)   predict: O(n_SV · p)", note: "The quadratic-to-cubic training cost is why kernel SVMs stall past roughly 100k rows." },
      ],
      output: "A predicted class label, plus the signed score f(x) (larger magnitude = more confident).",
      assumptions: [
        { name: "Features are scaled", why: "Both ‖w‖² and the RBF distance are dominated by large-range features.", check: "Standardize every feature. This matters more for SVM than for almost any other model." },
        { name: "A margin exists in some space", why: "If classes overlap heavily everywhere, no kernel recovers a clean separation and C just trades one error for another.", check: "Look at the fraction of points that end up as support vectors. Near 100% means no real margin." },
        { name: "Moderate dataset size", why: "Training is at least quadratic in the number of samples.", check: "Past roughly 100k rows switch to LinearSVC or SGDClassifier." },
        { name: "Binary labels", why: "The formulation is intrinsically two-class.", check: "Multi-class is built by one-vs-one or one-vs-rest wrappers, which multiply training cost." },
        { name: "Probabilities not required", why: "SVM outputs a signed distance, not a probability.", check: "Platt scaling can add probabilities but needs an extra internal cross-validation." },
      ],
      regularization: [
        { name: "Soft margin (C)", formula: "(1/2)‖w‖² + C·Σξᵢ", note: "C is the inverse regularization strength. Small C means a wide margin and heavy regularization; large C forces the model to fit the training points." },
        { name: "Linear kernel", formula: "K(a,b) = aᵀb", note: "No extra capacity. The right default when p is large, as in text." },
        { name: "Polynomial kernel", formula: "K(a,b) = (γ·aᵀb + r)^d", note: "Captures feature interactions up to order d. Numerically touchy for d above about 3." },
        { name: "RBF kernel", formula: "K(a,b) = exp(−γ‖a − b‖²)", note: "The general-purpose default. γ sets how far one point's influence reaches; large γ gives tight islands around each point." },
      ],
      hyperparameters: [
        { name: "C", range: "0.01 - 1000", increasing: "Narrower margin, fewer violations tolerated, higher variance and overfitting risk.", strategy: "Log-scale grid jointly with γ. C and γ interact strongly, so never tune them separately." },
        { name: "kernel", range: "linear / poly / rbf", increasing: "Not applicable", strategy: "Try linear first when p is large or m is large. RBF when the boundary is clearly curved and the data is modest." },
        { name: "γ (RBF)", range: "1e-4 - 10", increasing: "Each point's influence shrinks, the boundary tightens around individual points, and overfitting rises sharply.", strategy: "Start from 'scale' (1 / (p · var(X))) and search a log grid around it." },
        { name: "degree (poly)", range: "2 - 5", increasing: "Higher-order interactions, sharply rising cost and numerical instability.", strategy: "Rarely beats RBF. Keep at 2 or 3 if used at all." },
        { name: "class_weight", range: "None / balanced", increasing: "Raises the cost of misclassifying the minority class.", strategy: "Use 'balanced' on skewed data; it effectively gives each class its own C." },
        { name: "probability", range: "true / false", increasing: "Not applicable", strategy: "Leave off unless needed. Turning it on adds an internal 5-fold Platt calibration and multiplies training time." },
      ],
      metrics: ["Accuracy", "F1-score", "ROC-AUC", "Number of support vectors (model complexity)"],
      typicalUses: ["Text/document classification", "Bioinformatics (gene-expression classification)", "Image classification (pre-deep-learning era)", "Small/medium datasets needing a strong, controllable non-linear classifier"],
      diagnostics: [
        "Track the support-vector fraction. A model where most training points are support vectors is memorising, not generalising: lower C or γ.",
        "Plot a validation heatmap over the C by γ grid. The good region is usually a narrow diagonal band, not a single point.",
        "If training never finishes, the data is too large for a kernel SVM. Move to LinearSVC or SGDClassifier with a hinge loss.",
        "A model that is perfect on train and poor on test almost always means γ is too high.",
      ],
      advantages: [
        "Maximising the margin gives strong generalisation guarantees and good behaviour on small datasets.",
        "The kernel trick reaches highly non-linear boundaries without ever materialising the feature space.",
        "Effective when features outnumber samples, where most models overfit immediately.",
        "The final model depends only on the support vectors, so it is often compact.",
        "The optimisation is convex, so there is a single global optimum and no seed sensitivity.",
      ],
      limitations: [
        { name: "Scales badly", note: "training is quadratic to cubic in the number of samples", fix: "LinearSVC, SGDClassifier, or the Nystroem kernel approximation." },
        { name: "No native probabilities", note: "the output is a signed distance", fix: "Platt scaling, at the cost of an internal cross-validation." },
        { name: "Very sensitive to C and γ", note: "performance swings wildly across the grid", fix: "systematic joint log-scale search." },
        { name: "Requires scaling", note: "unscaled features silently wreck the kernel", fix: "standardize inside the pipeline." },
        { name: "Hard to interpret", note: "with a non-linear kernel there are no coefficients to read", fix: "use a linear kernel, or explain with SHAP." },
        { name: "Multi-class is bolted on", note: "one-vs-one trains K(K−1)/2 models", fix: "acceptable for few classes; otherwise pick another model." },
      ],
      alternatives: [
        { name: "LinearSVC / SGDClassifier", when: "Same hinge-loss objective, but linear and scalable to millions of rows." },
        { name: "Gradient boosting", when: "Tabular data, mixed types, and you want feature importances." },
        { name: "Logistic regression", when: "You need calibrated probabilities and the boundary is close to linear." },
        { name: "Neural network", when: "Very large datasets where the quadratic training cost is fatal." },
      ],
      pitfalls: [
        { problem: "Training hangs on a large dataset", solution: "Kernel SVM is at least O(m²). Switch to LinearSVC or approximate the kernel with Nystroem." },
        { problem: "Accuracy is near chance", solution: "Almost always unscaled features. Put StandardScaler in the pipeline." },
        { problem: "Perfect train accuracy, poor test", solution: "γ or C too large. Move both down a few orders of magnitude." },
        { problem: "Nearly every point is a support vector", solution: "The model is memorising. Reduce C, reduce γ, or accept the classes are not separable." },
        { problem: "predict_proba is unavailable", solution: "Construct with probability=True, or rank by decision_function instead." },
        { problem: "Minority class is ignored", solution: "Set class_weight='balanced'." },
      ],
      quickRef: [
        { name: "Margin width", formula: "2 / ‖w‖" },
        { name: "Primal (soft)", formula: "min ½‖w‖² + C·Σξᵢ" },
        { name: "Hinge loss", formula: "max(0, 1 − y·f(x))" },
        { name: "Dual constraint", formula: "0 ≤ αᵢ ≤ C,  Σαᵢyᵢ = 0" },
        { name: "Decision function", formula: "f(x) = Σαᵢyᵢ K(xᵢ,x) + b" },
        { name: "RBF kernel", formula: "K = exp(−γ‖a−b‖²)" },
        { name: "Poly kernel", formula: "K = (γ·aᵀb + r)^d" },
        { name: "Support vector", formula: "any point with αᵢ > 0" },
      ],
      code: `from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Scaling is mandatory: the kernel is a distance.
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

# C and gamma interact, so search them together on a log grid.
grid = GridSearchCV(
    pipe,
    {"svc__C":     [0.1, 1, 10, 100],
     "svc__gamma": [1e-3, 1e-2, 0.1, "scale"],
     "svc__class_weight": [None, "balanced"]},
    cv=5, n_jobs=-1,
).fit(X_train, y_train)

print(grid.best_params_)
print("support vectors:", grid.best_estimator_[-1].n_support_)

# Above ~100k rows, drop the kernel and use the linear solver instead.
fast = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto"))`,
      whyChain: [
        { q: "Why maximise the margin rather than just separate the classes?", a: "Infinitely many hyperplanes separate a separable dataset. The widest-margin one is furthest from every training point, so small perturbations to the data are least likely to flip a prediction. That translates into a tighter generalisation bound." },
        { q: "What is the kernel trick actually doing?", a: "In the dual, the data appears only as inner products xᵢᵀxⱼ. Replacing that with K(xᵢ,xⱼ) computes the inner product in some higher-dimensional space φ, so you get a linear separator there, which is a curved one here, without ever computing φ(x)." },
        { q: "Why can RBF handle any boundary?", a: "The RBF kernel corresponds to an infinite-dimensional feature space. With enough support vectors it can approximate any continuous decision boundary, which is also exactly why it overfits so readily." },
        { q: "What do C and γ each control?", a: "C is the penalty for violating the margin: it trades margin width against training errors. γ is the reach of a single training point in the RBF kernel: large γ makes each point influence only its immediate surroundings, producing tight islands." },
        { q: "Why is the solution sparse when logistic regression's is not?", a: "Hinge loss is exactly zero once a point is correctly classified beyond the margin, so those points contribute nothing to the gradient and get α = 0. Log-loss is always strictly positive, so every point keeps some influence forever." },
        { q: "Why does SVM struggle on huge datasets?", a: "The dual has one variable per training sample and the kernel matrix is m by m. Both memory and time grow at least quadratically, so the kernel version becomes impractical well before a linear model does." },
        { q: "How do you get probabilities out of an SVM?", a: "Platt scaling: fit a one-dimensional logistic regression to the decision-function values on held-out folds. It is a post-hoc patch, not part of the SVM objective." },
        { q: "Linear or RBF kernel for text?", a: "Linear. Text is already extremely high-dimensional and usually close to linearly separable there, so RBF adds cost and overfitting risk for no gain." },
      ],
      parameters: [
        { name: "kernel", effect: "Linear / poly / RBF - controls what shapes of decision boundary are reachable at all." },
        { name: "C (soft-margin penalty)", effect: "Small C → wider margin, tolerates more training-point violations. Large C → fits training points harder, narrower margin." },
        { name: "γ (RBF)", effect: "How far a single point's influence reaches. Small γ → smooth/far-reaching boundary. Large γ → tight, local boundary (risk of overfitting)." },
        { name: "degree (poly)", effect: "Order of feature interactions the polynomial kernel can represent." },
      ],
      metrics: ["Accuracy", "F1-score", "ROC-AUC", "Number of support vectors (model complexity)"],
      typicalUses: ["Text/document classification", "Bioinformatics (gene-expression classification)", "Image classification (pre-deep-learning era)", "Small/medium datasets needing a strong, controllable non-linear classifier"],
      workedExample: {
        setup: "Linear kernel. Two support vectors: x1=(1,1), y1=+1, α1=0.5; x2=(−1,−1), y2=−1, α2=0.5; b=0. Classify query x=(2,2).",
        steps: [
          "K(x1,x) = x1·x = 1×2+1×2 = 4.",
          "K(x2,x) = x2·x = −1×2+−1×2 = −4.",
          "f(x) = α1·y1·K(x1,x) + α2·y2·K(x2,x) + b = 0.5×1×4 + 0.5×(−1)×(−4) + 0 = 2 + 2 = 4.",
        ],
        result: "f(x) = 4 > 0 → predicted class = +1",
      },
    },
    mount,
  });
})();
