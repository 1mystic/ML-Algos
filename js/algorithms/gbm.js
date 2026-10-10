(() => {
  const { ui, col, fmt } = Lab;
  const median = (a) => { if (!a.length) return 0; const s = a.slice().sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
  function regTree(xs, rs, idx, depth, maxDepth, minLeaf) {
    const vals = idx.map((i) => rs[i]);
    const mean = MLU.mean(vals);
    if (depth >= maxDepth || idx.length < 2 * minLeaf) return { leaf: true, idx, value: mean };
    const sorted = idx.slice().sort((a, b) => xs[a] - xs[b]);
    let sL = 0, sL2 = 0, sT = 0, sT2 = 0;
    for (const i of sorted) { sT += rs[i]; sT2 += rs[i] ** 2; }
    let best = null;
    for (let k = 0; k < sorted.length - 1; k++) {
      const r = rs[sorted[k]]; sL += r; sL2 += r * r;
      const nl = k + 1, nr = sorted.length - nl;
      if (nl < minLeaf || nr < minLeaf || xs[sorted[k + 1]] - xs[sorted[k]] < 1e-9) continue;
      const sse = sL2 - (sL * sL) / nl + (sT2 - sL2) - ((sT - sL) ** 2) / nr;
      if (!best || sse < best.sse) best = { sse, thr: (xs[sorted[k]] + xs[sorted[k + 1]]) / 2 };
    }
    if (!best) return { leaf: true, idx, value: mean };
    return { leaf: false, thr: best.thr, left: regTree(xs, rs, idx.filter((i) => xs[i] <= best.thr), depth + 1, maxDepth, minLeaf), right: regTree(xs, rs, idx.filter((i) => xs[i] > best.thr), depth + 1, maxDepth, minLeaf) };
  }
  const predT = (t, x) => (t.leaf ? t.value : predT(x <= t.thr ? t.left : t.right, x));
  const leaves = (t) => (t.leaf ? [t] : leaves(t.left).concat(leaves(t.right)));
  const LOSS = {
    squared: { grad: (y, F) => y - F, name: "squared error", tex: "r_i=y_i-F(x_i)", leaf: null },
    absolute: { grad: (y, F) => Math.sign(y - F), name: "absolute error", tex: "r_i=\\operatorname{sign}(y_i-F(x_i))", leaf: (res) => median(res) },
    huber: { grad: (y, F) => Math.max(-1.5, Math.min(1.5, y - F)), name: "Huber (δ = 1.5)", tex: "r_i=\\operatorname{clip}(y_i-F(x_i),-\\delta,\\delta)", leaf: (res) => { const m = median(res); return m + MLU.mean(res.map((v) => Math.max(-1.5, Math.min(1.5, v - m)))); } },
  };

  function buildFrames(pts, { depth, lr, rounds, loss }) {
    const train = pts.filter((p) => !p.test), test = pts.filter((p) => p.test);
    if (train.length < 3) return [{ phase: "idle", title: "Add more points", text: "", st: { trees: [], F0: 0 } }];
    const xs = train.map((p) => p.x), ys = train.map((p) => p.y);
    const L = LOSS[loss];
    const F0 = loss === "squared" ? MLU.mean(ys) : median(ys);
    const trees = [];
    const Fx = (x) => F0 + trees.reduce((s, t) => s + lr * predT(t, x), 0);
    const mse = (set) => (set.length ? MLU.mean(set.map((p) => (p.y - Fx(p.x)) ** 2)) : null);
    const frames = [{ phase: "init", major: true, arch: ["f0"], title: `Start with a constant: F₀ = ${F0.toFixed(2)}`, text: `The best constant under ${L.name} is the ${loss === "squared" ? "mean" : "median"} of y. Every later tree only has to explain what this misses.`, tex: loss === "squared" ? "F_0=\\arg\\min_c\\sum_i (y_i-c)^2=\\bar y" : "F_0=\\operatorname{median}(y)", st: { trees: [], F0 }, m: { train: mse(train), test: mse(test) } }];
    for (let m = 1; m <= rounds; m++) {
      const F = xs.map((x) => Fx(x));
      const r = ys.map((y, i) => L.grad(y, F[i]));
      const idx = xs.map((_, i) => i);
      const tree = regTree(xs, r, idx, 0, depth, 2);
      if (L.leaf) for (const lf of leaves(tree)) lf.value = L.leaf(lf.idx.map((i) => ys[i] - F[i]));
      const show = m <= 4 || m % Math.max(1, Math.round(rounds / 25)) === 0 || m === rounds;
      if (show) {
        frames.push({ phase: "resid", arch: ["F", "loss", "res"], title: `Round ${m}: compute pseudo-residuals`, text: `The negative gradient of the loss at the current prediction. ${loss === "squared" ? "For squared error that is simply the residual y − F." : "Under this loss it is a bounded version of the residual, so outliers cannot dominate."} Mean |r| = <b>${MLU.mean(r.map(Math.abs)).toFixed(3)}</b>.`, tex: `${L.tex}=-\\left.\\frac{\\partial L(y_i,F)}{\\partial F}\\right|_{F=F_{m-1}(x_i)}`, st: { trees: trees.slice(), F0, r, showR: true }, m: { train: mse(train), test: mse(test) } });
        frames.push({ phase: "fit", arch: ["res", "tree"], title: `Round ${m}: fit a depth-${depth} tree to the residuals`, text: `A small regression tree h<sub>${m}</sub> approximates the residual pattern with ${leaves(tree).length} piecewise-constant steps (orange, lower plot).`, tex: "h_m=\\arg\\min_h\\sum_i\\big(r_i-h(x_i)\\big)^2", st: { trees: trees.slice(), F0, r, showR: true, h: tree }, m: { train: mse(train), test: mse(test) } });
      }
      trees.push(tree);
      if (show) frames.push({ phase: "add", major: m === 1, arch: ["tree", "shrink", "F"], title: `Round ${m}: add the shrunken tree`, text: `F ← F + ν·h with ν = ${lr}. Small steps mean each tree corrects only part of the error: slower, but it generalises better. Train MSE <b>${mse(train).toFixed(3)}</b>${test.length ? `, test MSE <b>${mse(test).toFixed(3)}</b>` : ""}.`, tex: "F_m(x)=F_{m-1}(x)+\\nu\\,h_m(x)", st: { trees: trees.slice(), F0, h: tree, prevN: trees.length - 1 }, m: { train: mse(train), test: mse(test) } });
    }
    frames.push({ ...frames[frames.length - 1], phase: "done", major: true, arch: ["F", "out"], title: `${rounds} trees added`, text: "Gradient boosting is gradient descent in function space: each tree is one step along the negative gradient. Watch the test curve: with a high learning rate or many deep trees it eventually turns up while training error keeps falling." });
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "gbm", fps: 4,
      legend: ui.legend([["var(--accent)", "ensemble F(x)", "line"], ["var(--series-1)", "train"], ["var(--text-faint)", "test (hollow)", "ring"], ["var(--series-7)", "new tree hₘ", "line"]]),
      tools: ui.btn("retrain", "retrain", true),
      stageHTML: `<svg data-r="svg" class="lab-svg"></svg><div class="sub-cap" style="margin-top:8px">pseudo-residuals rᵢ and the tree fitted to them</div><svg data-r="res" class="lab-svg"></svg>`,
      plane: { h: 330, xLabel: "x", yLabel: "y" },
      hint: "every 4th point is held out as test data · click to add training points · drag / double-click to edit",
      data: { kind: "reg", presets: ["sine", "step", "cubic", "quad", "rlinear", "routliers"], def: "sine", n: 60, noise: 0.3 },
      side: [{ title: "boosting", html: ui.range("depth", "tree depth", 1, 5, 1, 2) + ui.range("lr", "learning rate ν", 0.05, 1, 0.05, 0.3) + ui.range("M", "rounds M", 1, 150, 1, 40) + ui.select("loss", "loss", [["squared", "squared error"], ["absolute", "absolute error (median)"], ["huber", "Huber"]], "squared") }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "f0", label: "F₀", sub: "constant", x: 36, y: 40, w: 54, kind: "param" },
          { id: "F", label: "Fₘ₋₁(x)", sub: "current model", x: 140, y: 40, kind: "op" },
          { id: "loss", label: "∂L/∂F", sub: "loss gradient", x: 260, y: 40, kind: "loss" },
          { id: "res", label: "rᵢ", sub: "pseudo-residuals", x: 330, y: 112, kind: "op" },
          { id: "tree", label: "tree hₘ", sub: "fit to rᵢ", x: 214, y: 112, kind: "op" },
          { id: "shrink", label: "× ν", sub: "shrinkage", x: 110, y: 112, w: 64, kind: "param" },
          { id: "out", label: "F_M(x)", sub: "prediction", x: 210, y: 168, kind: "out" },
        ],
        edges: [{ from: "f0", to: "F" }, { from: "F", to: "loss" }, { from: "loss", to: "res" }, { from: "res", to: "tree" }, { from: "tree", to: "shrink" }, { from: "shrink", to: "F", label: "add", dash: true }, { from: "F", to: "out", bend: -40 }],
      },
      phases: { init: { label: "init", color: 3 }, resid: { label: "residuals", color: 1 }, fit: { label: "fit tree", color: 6 }, add: { label: "add tree", color: 0 }, done: { label: "done", color: 2 } },
      charts: [{ id: "mse", title: "train vs test MSE", series: [{ key: "train", label: "train", color: "var(--series-1)" }, { key: "test", label: "test", color: "var(--series-2)", dash: "4 3" }], logY: true }],
      onFrame: render, onData: () => { lab.points.forEach((p, i) => (p.test = i % 4 === 3)); retrain(); },
    });
    lab.editable({ newPoint: (x, y) => ({ x, y, test: false }), onChange: () => retrain({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    const rp = Lab.plane(lab.R("res"), { h: 190, yd: [-6, 6], xLabel: "x", yLabel: "r" });
    const params = () => ({ depth: lab.val("depth"), lr: lab.val("lr"), rounds: lab.val("M"), loss: lab.val("loss") });
    function retrain({ at = "start" } = {}) { lab.load(buildFrames(lab.points, params()), { at, play: at === "start" }); }
    const fPath = L.mid.append("path").attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2.6);
    const fPrev = L.mid.append("path").attr("fill", "none").attr("stroke", "var(--text-faint)").attr("stroke-width", 1.4).attr("stroke-dasharray", "4 4");
    const hPath = rp.layers.mid.append("path").attr("fill", "none").attr("stroke", "var(--series-7)").attr("stroke-width", 2.6);
    const xs = d3.range(-10, 10.001, 0.05);
    const line = (P, f) => d3.line().x((x) => P.x(x)).y((x) => P.y(Math.max(-12, Math.min(12, f(x)))))(xs);
    function render(f) {
      const st = f.st, lr = lab.val("lr");
      const FN = (trees) => (x) => st.F0 + trees.reduce((s, t) => s + lr * predT(t, x), 0);
      const cur = FN(st.trees);
      fPath.attr("d", line(pl, cur));
      fPrev.attr("d", st.prevN != null ? line(pl, FN(st.trees.slice(0, st.prevN))) : null);
      const train = lab.points.filter((p) => !p.test);
      const sticks = L.links.selectAll("line").data(st.showR ? train : []);
      sticks.exit().remove();
      sticks.enter().append("line").merge(sticks).attr("x1", (d) => pl.x(d.x)).attr("x2", (d) => pl.x(d.x)).attr("y1", (d) => pl.y(d.y)).attr("y2", (d) => pl.y(cur(d.x))).attr("stroke", "var(--series-2)").attr("stroke-opacity", 0.7);
      lab.drawPoints({ color: (d) => (d.test ? "transparent" : "var(--series-1)"), stroke: (d) => (d.test ? "var(--text-faint)" : "var(--bg)"), sw: (d) => (d.test ? 1.6 : 1.1) });
      // residual panel
      const rr = st.r || train.map((p) => p.y - cur(p.x));
      const rd = rp.layers.pts.selectAll("circle").data(train.map((p, i) => ({ x: p.x, r: rr[i] })));
      rd.exit().remove();
      rd.enter().append("circle").attr("r", 3.6).merge(rd).attr("cx", (d) => rp.x(d.x)).attr("cy", (d) => rp.y(Math.max(-6, Math.min(6, d.r)))).attr("fill", "var(--series-2)").attr("opacity", st.r ? 0.9 : 0.35);
      hPath.attr("d", st.h ? d3.line().curve(d3.curveStepAfter).x((x) => rp.x(x)).y((x) => rp.y(Math.max(-6, Math.min(6, predT(st.h, x)))))(xs) : null);
      const mtr = f.m && f.m.train, mte = f.m && f.m.test;
      lab.readout(`<div class="kv"><span>trees</span><b>${st.trees.length}</b><span>train MSE</span><b>${fmt(mtr)}</b><span>test MSE</span><b>${fmt(mte)}</b><span>F₀</span><b>${fmt(st.F0, 2)}</b></div>`);
    }
    lab.watch(["depth", "lr", "M", "loss"], () => retrain());
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    lab.loadPreset("sine");
    return lab.destroy;
  }

  MLApp.register({
    id: "gbm",
    name: "Gradient Boosting Trees",
    category: "Supervised - Trees & Ensembles",
    tagline: "stagewise residual fitting, train vs test",
    description: "Gradient descent in function space: compute pseudo-residuals, fit a small tree to them, add a shrunken copy, repeat. A held-out test set shows exactly when more trees start to overfit.",
    info: {
      type: "Supervised - Regression (classification variants swap in a different loss). Additive ensemble, sequential (stagewise) boosting.",
      scenario: "High predictive accuracy on tabular data when you can trade some interpretability and training time for it - the idea behind XGBoost/LightGBM.",
      inputs: "Feature vectors x and continuous targets y.",
      intuition: {
        definition: "Build the model one small tree at a time. Each new tree is fitted to <b>what the ensemble still gets wrong</b>, then shrunk before being added. Thousands of weak learners, each nudging the prediction slightly, compose into a very strong one.",
        steps: [
          "Start with a constant prediction, usually the mean.",
          "Compute the residual (the negative gradient) at every point.",
          "Fit a shallow tree to those residuals.",
          "Add it scaled by the learning rate, then repeat.",
        ],
        applications: [
          "Tabular competition winners, almost universally",
          "Credit and insurance risk scoring",
          "Demand and sales forecasting",
          "Search ranking and click-through-rate prediction",
          "Fraud detection on structured transaction data",
        ],
      },
      math: [
        { title: "Additive model", formula: "F_M(x) = F₀(x) + η·Σ_{m=1}^{M} h_m(x)", note: "A sum of shrunk weak learners. Each h_m is a shallow regression tree, and η holds each one back so no single tree dominates." },
        { title: "Initialisation", formula: "F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)", note: "The best constant prediction: the mean for squared error, the median for absolute error, the log-odds for log-loss." },
        { title: "Pseudo-residuals", formula: "rᵢₘ = −[ ∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ) ]_{F = F_{m−1}}", note: "The negative gradient of the loss with respect to the current prediction. This is the general form; the residual is just the special case for squared error." },
        { title: "Why residuals for squared error", formula: "L = ½(y − F)²  ⟹  −∂L/∂F = y − F", note: "The gradient is literally the residual, which is why the classic description 'fit the next tree to the errors' is exactly right for regression." },
        { title: "Fit and shrink", formula: "h_m = tree fit to {(xᵢ, rᵢₘ)},   F_m = F_{m−1} + η·h_m", note: "Gradient descent in function space: each tree is a step in the direction that most reduces the loss, and η is the step size." },
        { title: "Leaf values", formula: "γ_j = argmin_γ Σ_{xᵢ ∈ leaf j} L(yᵢ, F_{m−1}(xᵢ) + γ)", note: "After the tree structure is fixed, each leaf's output is re-solved against the true loss rather than the gradient approximation." },
      ],
      pipeline: [
        { label: "F₀ = mean(y)", note: "constant start" },
        { label: "Residuals", note: "r = y − F" },
        { label: "Fit stump", note: "tree on r" },
        { label: "Shrink", note: "× learning rate η" },
        { label: "F += η·h", note: "repeat M times", accent: "green" },
      ],
      decisionFunction: {
        text: "ŷ(x) = F₀ + η · Σ_{m=1}^{M} treeₘ(x)",
        mechanism: "The prediction is the sum of an initial constant plus many small, shrunk regression trees added one at a time.",
      },
      lossFunction: {
        text: "L = Σᵢ (yᵢ − F(xᵢ))²",
        mechanism: "Each new tree is fit to the negative gradient of the loss w.r.t. current predictions - for squared error that gradient is exactly the residual, so every round is 'fit a small tree to what's still wrong, then shrink it in'.",
        plot: { fn: (r) => r * r, domain: [-4, 4], color: "var(--accent)", caption: "same squared-error shape as linear regression - each new tree chases this residual" },
      },
      optimization: [
        { title: "Gradient descent in function space", formula: "F_m = F_{m−1} − η·∇_F L", note: "Ordinary gradient descent updates a parameter vector. Boosting updates a function, and the tree is how that gradient step is represented on new inputs." },
        { title: "The core trade-off", formula: "η · M ≈ constant", note: "Halving the learning rate roughly doubles the trees needed. Small η with many trees generalises better because each step is a smaller, less noisy commitment." },
        { title: "Early stopping", formula: "stop when validation loss has not improved for k rounds", note: "The single most useful regularizer. Boosting will happily keep overfitting forever, so let the validation curve choose M." },
        { title: "Stochastic boosting", formula: "fit each tree on a random fraction of rows (subsample < 1)", note: "Injects noise, decorrelates successive trees, and usually improves generalisation while speeding up training." },
        { title: "Swapping the loss", formula: "L = ½(y−F)² | |y−F| | Huber | log-loss", note: "Only the pseudo-residual formula changes. This is why the same engine handles regression, classification, and ranking." },
      ],
      output: "A continuous predicted value: the sum of every tree's shrunk contribution.",
      assumptions: [
        { name: "Loss is differentiable", why: "The whole method needs a gradient with respect to the prediction.", check: "Any standard loss qualifies. Non-differentiable objectives need a surrogate." },
        { name: "Weak learners stay weak", why: "Deep trees fit the residuals entirely in one round, which turns boosting into a single overfitted tree.", check: "Keep max_depth between 3 and 8. Depth 1 stumps are a valid extreme." },
        { name: "Clean-ish labels", why: "Boosting concentrates on the hardest points, so mislabelled rows attract disproportionate attention.", check: "Inspect the rows with the largest final residuals; they are often data errors." },
        { name: "Held-out data available", why: "Without a validation set there is no principled way to choose the number of rounds.", check: "Always reserve an early-stopping split." },
        { name: "Sequential training is acceptable", why: "Each tree depends on the previous one, so rounds cannot be parallelised the way a forest can.", check: "Parallelism is available within a split search, not across rounds." },
      ],
      regularization: [
        { name: "Shrinkage", formula: "F += η·h,  η ∈ [0.01, 0.3]", note: "The primary control. Lower η means each tree commits less, so the ensemble explores more carefully." },
        { name: "Tree depth", formula: "depth ≤ 3 to 8", note: "Caps the interaction order any single tree can express. Depth d captures interactions among at most d features." },
        { name: "Row subsampling", formula: "subsample ∈ [0.5, 1.0]", note: "Stochastic gradient boosting. Adds variance reduction through decorrelation, much like bagging." },
        { name: "Column subsampling", formula: "colsample ∈ [0.5, 1.0]", note: "Borrowed from random forests. Prevents one strong feature from being used in every tree." },
        { name: "Early stopping", formula: "M* = argmin validation loss", note: "Effectively regularizes by truncating the sum before it starts fitting noise." },
        { name: "Leaf penalties", formula: "+ γ·|leaves| + ½λ‖w‖²", note: "Explicit in XGBoost and LightGBM; classic GBM relies on depth and shrinkage alone." },
      ],
      hyperparameters: [
        { name: "learning_rate η", range: "0.01 - 0.3", increasing: "Faster fitting per tree, fewer trees needed, higher risk of overshooting and overfitting.", strategy: "Fix a small value such as 0.05, then let early stopping choose the tree count. Tune η last." },
        { name: "n_estimators M", range: "100 - 5000", increasing: "Keeps reducing training loss indefinitely and will eventually overfit.", strategy: "Do not tune by grid search. Set it high and use early stopping on a validation split." },
        { name: "max_depth", range: "2 - 8", increasing: "Each tree captures higher-order interactions; overfits quickly past 8.", strategy: "Start at 3. Raise only if training loss plateaus too early." },
        { name: "subsample", range: "0.5 - 1.0", increasing: "Less noise injection, less regularization.", strategy: "0.8 is a reliable default; lower it if overfitting persists." },
        { name: "colsample_bytree", range: "0.5 - 1.0", increasing: "Trees see more features and become more correlated.", strategy: "0.8 by default. Lower it when features are many and correlated." },
        { name: "min_samples_leaf", range: "1 - 100", increasing: "More conservative leaves, smoother fit.", strategy: "Raise on noisy data to stop leaves chasing individual points." },
        { name: "loss", range: "squared / absolute / huber / quantile", increasing: "Not applicable", strategy: "Huber or absolute error when outliers are present; quantile when you need prediction intervals." },
      ],
      metrics: ["MSE / RMSE", "MAE", "R²", "(classification variants: log-loss / AUC)", "Validation loss curve versus boosting round"],
      typicalUses: ["Tabular prediction competitions", "Risk scoring", "Demand forecasting", "Ranking / click-through-rate prediction"],
      diagnostics: [
        "Plot training and validation loss against boosting round. Training loss falls monotonically; the round where validation loss turns upward is your M.",
        "If validation loss never turns upward, you have not run enough rounds or η is too small.",
        "If it turns upward almost immediately, η or depth is too high.",
        "Examine the largest remaining residuals. Persistent extreme errors usually indicate label noise or a missing feature.",
        "Compare against a random forest. If boosting is not clearly ahead, it is probably under-tuned.",
      ],
      advantages: [
        "State-of-the-art accuracy on structured and tabular data, consistently ahead of forests and linear models.",
        "Any differentiable loss plugs straight in, so regression, classification, ranking, and quantile estimation share one engine.",
        "Captures non-linearity and interactions automatically, with no feature scaling required.",
        "Handles mixed feature types and, in modern implementations, missing values natively.",
        "Produces useful feature importances and works well with SHAP for explanation.",
      ],
      limitations: [
        { name: "Overfits without care", note: "unlike a forest, more rounds genuinely hurt", fix: "early stopping on a validation split, always." },
        { name: "Sequential by nature", note: "rounds cannot be parallelised across trees", fix: "histogram-based splitting (LightGBM) or GPU training." },
        { name: "Many interacting hyperparameters", note: "η, depth, and M all pull against each other", fix: "fix η low, early-stop for M, then tune depth." },
        { name: "Sensitive to label noise", note: "boosting deliberately focuses on the hardest points, which may be the wrong ones", fix: "robust losses such as Huber, and clean the data." },
        { name: "Not interpretable as a whole", note: "thousands of trees are not a readable model", fix: "SHAP values or partial dependence plots." },
        { name: "Cannot extrapolate", note: "inherits the flat-beyond-range behaviour of trees", fix: "detrend first, or use a linear model for the trend component." },
      ],
      alternatives: [
        { name: "XGBoost / LightGBM / CatBoost", when: "Essentially always in production. Same idea, far faster, with explicit regularization." },
        { name: "Random forest", when: "You want good accuracy with almost no tuning and parallel training." },
        { name: "Linear or ridge regression", when: "The relationship is close to linear, or you must extrapolate." },
        { name: "Neural network", when: "Unstructured data such as images, audio, or text." },
      ],
      pitfalls: [
        { problem: "Training loss keeps falling but test loss rises", solution: "Classic boosting overfit. Enable early stopping and lower the learning rate." },
        { problem: "Model underfits badly", solution: "η too small for the round budget, or depth too shallow. Raise one of them." },
        { problem: "Training takes forever", solution: "Depth or round count too high. Use a histogram-based implementation such as LightGBM." },
        { problem: "Grid-searching n_estimators", solution: "Wasted effort. Set it high once and early-stop instead." },
        { problem: "A few outliers dominate the fit", solution: "Squared error amplifies them. Switch to Huber or absolute loss." },
        { problem: "Results are not reproducible", solution: "Row and column subsampling are stochastic. Fix the random seed." },
      ],
      quickRef: [
        { name: "Additive model", formula: "F_M = F₀ + η·Σ h_m" },
        { name: "Pseudo-residual", formula: "r = −∂L/∂F" },
        { name: "Squared-error case", formula: "r = y − F" },
        { name: "Update", formula: "F_m = F_{m−1} + η·h_m" },
        { name: "Init (MSE)", formula: "F₀ = mean(y)" },
        { name: "Init (log-loss)", formula: "F₀ = log(p/(1−p))" },
        { name: "Rule of thumb", formula: "halve η → double M" },
        { name: "Depth d captures", formula: "interactions of order ≤ d" },
      ],
      code: `from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Fix a small learning rate, set the round budget high, and let
# early stopping decide where to stop. Never grid-search n_estimators.
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=5000,             # an upper bound, not a target
    max_depth=4,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=50,       # patience
    random_state=42,
).fit(X_fit, y_fit)

print("rounds actually used:", model.n_iter_)

# Robust loss when outliers are present:
robust = GradientBoostingRegressor(loss="huber", alpha=0.9, learning_rate=0.05)`,
      whyChain: [
        { q: "How does boosting differ from bagging?", a: "Bagging trains independent trees in parallel on resampled data and averages them, which reduces variance. Boosting trains trees sequentially, each correcting the previous ensemble, which reduces bias. Their failure modes are opposite: extra bagged trees never hurt, extra boosted rounds eventually do." },
        { q: "Why is it called gradient boosting when we just fit residuals?", a: "The residual is the negative gradient of squared error with respect to the prediction. Generalising from 'residual' to 'negative gradient' is what allows any differentiable loss to be boosted with the same algorithm." },
        { q: "Why shrink each tree by a learning rate?", a: "Without shrinkage each tree fully corrects the current errors, so the ensemble commits hard to early, noisy signal. Small steps let later trees revise earlier decisions, which reliably generalises better." },
        { q: "Why use shallow trees rather than deep ones?", a: "Each learner only needs to be slightly better than random; the sum provides the power. A deep tree would absorb all the residual in one round, leaving nothing to boost and reproducing a single overfitted tree." },
        { q: "Why does more boosting rounds overfit when more forest trees do not?", a: "A forest averages, so its prediction converges to an expectation as trees are added. Boosting sums, so every extra tree adds capacity and will eventually start fitting noise." },
        { q: "What does max_depth control conceptually?", a: "The maximum interaction order. Depth 1 stumps give a purely additive model with no interactions; depth 3 can express three-way interactions." },
        { q: "Why does boosting struggle with noisy labels?", a: "Each round redirects attention to the largest remaining errors. A mislabelled row is permanently wrong, so it keeps attracting effort and successive trees contort themselves around it." },
        { q: "What is the right tuning order?", a: "Fix η small (0.05), set M high with early stopping, then tune depth and subsampling. Tune η last, and only if the round budget is a problem." },
      ],
      parameters: [
        { name: "number of trees / rounds", effect: "More rounds fit the training data more closely, raising overfitting risk without regularization elsewhere." },
        { name: "tree depth", effect: "How much each individual tree can model. Boosting typically uses shallow trees (depth 1–4) and lets many of them combine." },
        { name: "learning rate η (shrinkage)", effect: "Down-weights each tree's contribution. Lower η with more rounds usually generalizes better but costs more compute." },
      ],
      metrics: ["MSE / RMSE", "MAE", "R²", "(classification variants: log-loss / AUC)"],
      typicalUses: ["Tabular prediction competitions", "Risk scoring", "Demand forecasting", "Ranking / click-through-rate prediction"],
      workedExample: {
        setup: "Targets y=[1,2,3] at x=[1,2,3]. F₀=mean(y)=2, learning rate η=0.5. Fit one stump splitting at x≤1.5.",
        steps: [
          "Residuals = y − F₀ = [1−2, 2−2, 3−2] = [−1, 0, 1].",
          "Stump split at x≤1.5: left={x=1, resid=−1} → leaf value −1. Right={x=2,3, resid=0,1} → leaf value mean=0.5.",
          "Shrink and add: F(1) = 2 + 0.5×(−1) = 1.5. F(2) = 2 + 0.5×0.5 = 2.25. F(3) = 2 + 0.5×0.5 = 2.25.",
          "New residuals = y − F = [1−1.5, 2−2.25, 3−2.25] = [−0.5, −0.25, 0.75].",
        ],
        result: "Residual magnitudes shrank from [1,0,1] to [0.5,0.25,0.75] after one boosting round - the next tree fits what's left",
      },
    },
    mount,
  });
})();
