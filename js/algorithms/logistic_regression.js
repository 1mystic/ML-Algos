(() => {
  const { ui, col, fmt, sigmoid } = Lab;
  const S = 9;
  function feats(x, y, kind) {
    const u = x / S, v = y / S;
    return kind === "quad" ? [1, u, v, u * u, v * v, u * v] : [1, u, v];
  }
  const NAMES = { lin: ["b", "w₁·x", "w₂·y"], quad: ["b", "x", "y", "x²", "y²", "xy"] };
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

  function stats(pts, w, kind, thr) {
    let loss = 0, ok = 0;
    for (const p of pts) {
      const pr = sigmoid(dot(w, feats(p.x, p.y, kind)));
      loss += -(p.label ? Math.log(pr + 1e-12) : Math.log(1 - pr + 1e-12));
      if ((pr >= thr ? 1 : 0) === p.label) ok++;
    }
    return { loss: pts.length ? loss / pts.length : 0, acc: pts.length ? ok / pts.length : 0 };
  }

  function buildFrames(pts, { kind, lr, epochs, lam, thr }) {
    const P = feats(0, 0, kind).length;
    if (!pts.some((p) => p.label === 0) || !pts.some((p) => p.label === 1))
      return [{ phase: "idle", title: "Need both classes", text: "Add points of class 0 and class 1 (shift-click adds class 1).", st: { w: new Array(P).fill(0) } }];
    const frames = [];
    let w = new Array(P).fill(0);
    const keep = new Set(Lab.checkpoints(epochs, 80));
    for (let e = 0; e <= epochs; e++) {
      const g = new Array(P).fill(0);
      for (const p of pts) {
        const f = feats(p.x, p.y, kind);
        const err = sigmoid(dot(w, f)) - p.label;
        for (let i = 0; i < P; i++) g[i] += (err * f[i]) / pts.length;
      }
      for (let i = 1; i < P; i++) g[i] += lam * w[i];
      if (keep.has(e)) {
        const s = stats(pts, w, kind, thr);
        frames.push({
          phase: e === 0 ? "init" : "gd", major: e === 0,
          arch: e === 0 ? ["x", "feat", "z", "sig"] : ["z", "sig", "loss", "grad", "upd"],
          title: e === 0 ? "w = 0: every point gets p = 0.5" : `Epoch ${e} · gradient step`,
          text: e === 0
            ? "With zero weights the model is maximally unsure. The field shows p(class 1 | x): saturated colour means confidence, pale means doubt."
            : `Log-loss <b>${s.loss.toFixed(3)}</b>, accuracy <b>${(s.acc * 100).toFixed(1)}%</b>. The gradient is just each point's error (p − y) times its features: confident mistakes pull hardest.`,
          tex: e === 0 ? "p(y{=}1\\mid x)=\\sigma(w^\\top\\phi(x))=\\frac{1}{1+e^{-w^\\top\\phi(x)}}" : "\\nabla L=\\frac1n\\sum_i\\big(\\sigma(w^\\top\\phi_i)-y_i\\big)\\,\\phi_i+\\lambda w",
          st: { w: w.slice() }, m: { loss: s.loss, acc: s.acc },
        });
      }
      for (let i = 0; i < P; i++) w[i] -= lr * g[i];
    }
    const last = frames[frames.length - 1];
    const s = stats(pts, last.st.w, kind, thr);
    frames.push({
      ...last, phase: "done", major: true, arch: ["sig", "out"], title: "Trained",
      text: `Final log-loss <b>${s.loss.toFixed(3)}</b>, accuracy <b>${(s.acc * 100).toFixed(1)}%</b>. The solid line is where p = ${thr}; the dashed lines are p = 0.25 and 0.75. ${kind === "lin" ? "A linear model can only draw a straight boundary, so rings and XOR defeat it. Switch the features to quadratic." : "Quadratic features let a linear model draw conics: circles, ellipses and hyperbolas."}`,
    });
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "logreg", fps: 8, classes: 2,
      legend: ui.legend([["var(--text)", "p = threshold", "line"], ["var(--text)", "p = 0.25 / 0.75", "dash"]]),
      tools: ui.btn("retrain", "retrain", true),
      hint: "click adds the selected class · shift-click adds class 1 · drag / double-click to edit",
      data: { kind: "class", presets: ["linear", "overlap", "blobs2", "imbalanced", "circles", "moons", "xor"], def: "linear", n: 80, noise: 0.3 },
      side: [
        { title: "features", html: ui.select("kind", "basis φ(x)", [["lin", "linear: x, y"], ["quad", "quadratic: + x², y², xy"]], "lin") + ui.range("lam", "L2 strength λ", 0, 0.2, 0.005, 0) },
        { title: "optimiser", html: ui.range("lr", "learning rate η", 0.05, 8, 0.05, 2) + ui.range("ep", "epochs", 20, 1500, 10, 400) + ui.range("thr", "decision threshold", 0.05, 0.95, 0.05, 0.5) },
      ],
      arch: {
        w: 380, h: 200,
        nodes: [
          { id: "x", label: "x", sub: "(x₁, x₂)", x: 32, y: 42, w: 50, kind: "data" },
          { id: "feat", label: "φ(x)", sub: "features", x: 112, y: 42, kind: "op", info: "Optional basis expansion. Quadratic terms make curved boundaries possible while the model stays linear in w." },
          { id: "z", label: "z = wᵀφ", sub: "logit", x: 210, y: 42, kind: "op", info: "The log-odds. The decision boundary is the set where z = logit(threshold)." },
          { id: "sig", label: "σ(z)", sub: "probability", x: 310, y: 42, kind: "op", tex: "\\sigma(z)=1/(1+e^{-z})" },
          { id: "out", label: "p ≥ t ?", sub: "class", x: 340, y: 120, w: 64, kind: "out" },
          { id: "loss", label: "log-loss", sub: "−y log p − (1−y) log(1−p)", x: 220, y: 160, kind: "loss", tex: "L=-\\tfrac1n\\sum y\\log p+(1-y)\\log(1-p)" },
          { id: "grad", label: "∇L", sub: "Σ(p−y)φ", x: 110, y: 160, w: 70, kind: "op" },
          { id: "upd", label: "w ← w − η∇L", x: 50, y: 112, kind: "param" },
        ],
        edges: [{ from: "x", to: "feat" }, { from: "feat", to: "z" }, { from: "z", to: "sig" }, { from: "sig", to: "out" }, { from: "sig", to: "loss" }, { from: "loss", to: "grad" }, { from: "grad", to: "upd" }, { from: "upd", to: "z", dash: true, label: "w" }],
      },
      phases: { init: { label: "init", color: 3 }, gd: { label: "gradient step", color: 0 }, done: { label: "trained", color: 2 } },
      panels: [{ id: "sig", title: "every point on the sigmoid", sub: "x-axis: its logit z · y: its label" }],
      charts: [{ id: "loss", title: "loss & accuracy", series: [{ key: "loss", label: "log-loss", color: "var(--series-1)" }, { key: "acc", label: "accuracy", color: "var(--series-3)", dash: "4 3" }] }],
      onFrame: render, onData: () => retrain(),
    });
    lab.editable({ onChange: () => retrain({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    const params = () => ({ kind: lab.val("kind"), lr: lab.val("lr"), epochs: lab.val("ep"), lam: lab.val("lam"), thr: lab.val("thr") });
    function retrain({ at = "start" } = {}) { lab.load(buildFrames(lab.points, params()), { at, play: at === "start" }); }

    const sigEl = lab.R("panel-sig");
    const SP = Lab.xy(sigEl, { w: 360, h: 150, xd: [-8, 8], yd: [-0.1, 1.1], xLabel: "z", yLabel: "p" });
    SP.line("sig", d3.range(-8, 8.01, 0.1).map((z) => [z, sigmoid(z)]), { color: "var(--text-dim)", width: 1.6 });

    function render(f) {
      const { kind, thr } = params();
      const w = f.st.w;
      const prob = (x, y) => sigmoid(dot(w, feats(x, y, kind)));
      const grid = Lab.sampleGrid(pl, prob, 90);
      Lab.paintGrid(L.bg, pl, grid, (p) => Lab.probColor(p, 0.5));
      Lab.contour(L.under, pl, grid, [thr], { cls: "iso-main", width: 2 });
      Lab.contour(L.under, pl, grid, [0.25, 0.75], { cls: "iso-dash", width: 1, dash: "4 4", stroke: "var(--text-dim)" });
      lab.drawPoints({
        color: (d) => col(d.label),
        stroke: (d) => ((prob(d.x, d.y) >= thr ? 1 : 0) !== d.label ? "var(--text)" : "var(--bg)"),
        sw: (d) => ((prob(d.x, d.y) >= thr ? 1 : 0) !== d.label ? 2 : 1.1),
      });
      // sigmoid panel: each point at its logit
      const zs = lab.points.map((p, i) => ({ z: Math.max(-8, Math.min(8, dot(w, feats(p.x, p.y, kind)))), l: p.label, j: ((i * 37) % 11) / 110 }));
      const dots = SP.g.selectAll("circle").data(zs);
      dots.exit().remove();
      dots.enter().append("circle").attr("r", 3).merge(dots).transition().duration(120)
        .attr("cx", (d) => SP.x(d.z)).attr("cy", (d) => SP.y(d.l ? 1 - d.j : d.j)).attr("fill", (d) => col(d.l)).attr("opacity", 0.75);
      const zt = Math.log(thr / (1 - thr));
      SP.vline(zt, "thr-line");
      const s = stats(lab.points, w, kind, thr);
      const names = NAMES[kind];
      lab.readout(`<div class="kv">${w.map((v, k) => `<span>${names[k]}</span><b>${fmt(v)}</b>`).join("")}<span>log-loss</span><b>${fmt(s.loss)}</b><span>accuracy</span><b>${(s.acc * 100).toFixed(1)}%</b></div><div class="note">outlined points are misclassified at the current threshold</div>`);
    }
    lab.watch(["kind", "lr", "ep", "lam"], () => retrain());
    lab.watch(["thr"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    lab.loadPreset("linear");
    return lab.destroy;
  }

  MLApp.register({
    id: "logistic-regression",
    name: "Logistic Regression",
    category: "Supervised - Classification",
    tagline: "sigmoid + log-loss, animated gradient descent",
    description: "A linear model for class probabilities. Watch the probability field sharpen epoch by epoch, see every point slide along the sigmoid, and add quadratic features to bend the boundary around rings and XOR.",
    info: {
      type: "Supervised - Binary classification. Linear discriminative model (a generalized linear model with a logit/sigmoid link).",
      scenario: "Binary outcomes where you want calibrated probabilities and an interpretable, linear decision boundary - e.g. churn, click-through, disease presence, credit default.",
      inputs: "A feature vector x = (x₁, x₂) and a binary label y ∈ {0, 1}.",
      intuition: {
        definition: "Despite the name, logistic regression is a <b>classifier</b>. It takes a linear combination of the features, squashes it through a sigmoid into a probability, and thresholds that probability into a class.",
        steps: [
          "Score the point with a linear function: z = wᵀx + b.",
          "Squash z into (0, 1) with the sigmoid to get a probability.",
          "Threshold at 0.5 (or wherever the cost of errors dictates).",
          "Fit w by minimizing log-loss - the boundary is always a hyperplane.",
        ],
        applications: [
          "Spam vs. not-spam email filtering",
          "Disease present / absent from a screening panel",
          "Customer churn prediction",
          "Credit default risk scoring",
          "Ad click-through-rate estimation",
        ],
      },
      math: [
        { title: "Linear predictor (log-odds)", formula: "z = wᵀx + b = w₁x₁ + w₂x₂ + … + b", note: "An unbounded real score. Points far on the positive side are confidently class 1." },
        { title: "Sigmoid transform", formula: "σ(z) = 1 / (1 + e⁻ᶻ)", note: "Monotonic, differentiable, S-shaped, centred at σ(0) = 0.5. Maps (−∞, ∞) → (0, 1)." },
        { title: "Probability model", formula: "P(y=1 | x) = σ(wᵀx + b),  P(y=0 | x) = 1 − σ(wᵀx + b)", note: "The two class probabilities always sum to 1." },
        { title: "Logit link", formula: "log( p / (1 − p) ) = wᵀx + b", note: "This is what makes it a linear model: the log-odds are linear in x. A one-unit rise in xᵢ multiplies the odds by e^wᵢ." },
        { title: "Decision boundary", formula: "predict 1 ⟺ σ(z) ≥ 0.5 ⟺ z ≥ 0", note: "The boundary is exactly the hyperplane wᵀx + b = 0 - always linear in feature space." },
      ],
      pipeline: [
        { label: "Features x", note: "x₁ … xₚ" },
        { label: "Linear score", note: "z = wᵀx + b" },
        { label: "Sigmoid", note: "σ(z) ∈ (0,1)" },
        { label: "Threshold", note: "≥ 0.5 ?" },
        { label: "Class label", note: "0 or 1", accent: "green" },
      ],
      decisionFunction: {
        text: "P(y=1 | x) = σ(w₀ + w₁x₁ + w₂x₂), predict 1 if P ≥ 0.5",
        mechanism: "A linear score is passed through the sigmoid to squash it into (0, 1); the decision boundary is exactly the hyperplane where the linear score is zero.",
      },
      lossFunction: {
        text: "L(w) = −Σᵢ[yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ)]",
        mechanism: "Binary cross-entropy - convex in w, but has no closed form because of the sigmoid, so it's minimized iteratively with (batch) gradient descent.",
        plot: { fn: (z) => -Math.log(1 / (1 + Math.exp(-z))), domain: [-6, 6], yDomain: [0, 6], color: "var(--accent)", caption: "log-loss for the true class as its linear score z moves from very wrong (left) to very right (correct & confident)" },
      },
      optimization: [
        { title: "Why cross-entropy, not MSE", formula: "MSE ∘ σ ⇒ non-convex", note: "Squared error composed with the sigmoid gives a non-convex surface with plateaus where the gradient vanishes. Cross-entropy is the negative log-likelihood and is strictly convex, so gradient descent reaches the global optimum." },
        { title: "Gradient", formula: "∂L/∂wⱼ = (1/m) Σᵢ (ŷᵢ − yᵢ)·xᵢⱼ   →   ∇L = (1/m)·Xᵀ(ŷ − y)", note: "The sigmoid derivative cancels against the log, leaving a strikingly simple 'error × feature' form - identical in shape to linear regression's gradient." },
        { title: "Update rule", formula: "w := w − α·∇L(w)", note: "Batch gradient descent. Each step is O(m·p); with feature scaling it converges in a few hundred iterations." },
      ],
      output: "A probability P(y=1|x) in [0,1], thresholded (usually at 0.5) into a class label.",
      assumptions: [
        { name: "Binary target", why: "The likelihood is Bernoulli; a continuous or ordered target needs a different link.", check: "Count the distinct values of y." },
        { name: "Linear log-odds", why: "The model can only bend the probability surface, not the boundary - a curved boundary is unreachable.", check: "Plot the empirical logit against each continuous feature; add polynomial or interaction terms if it curves." },
        { name: "No severe multicollinearity", why: "Correlated features make the coefficients unstable and uninterpretable, even when accuracy is fine.", check: "VIF < 5–10 per feature." },
        { name: "Independent observations", why: "Clustered or repeated-measure data understates the standard errors.", check: "Inspect how the data was collected; use mixed models if grouped." },
        { name: "Enough events per feature", why: "Too few positives per predictor overfits and can cause perfect separation.", check: "Aim for 10–20 minority-class events per predictor." },
      ],
      regularization: [
        { name: "L2 (ridge)", formula: "L + λ·Σ wⱼ²", note: "Shrinks all weights toward zero without reaching it. Stabilizes correlated features and cures perfect separation. The default." },
        { name: "L1 (lasso)", formula: "L + λ·Σ |wⱼ|", note: "The constraint region has corners on the axes, so the optimum lands on them - weights hit exactly zero and features drop out. Built-in feature selection." },
        { name: "Elastic net", formula: "L + λ₁·Σ|wⱼ| + λ₂·Σwⱼ²", note: "Selects features while keeping correlated groups together instead of arbitrarily picking one." },
      ],
      hyperparameters: [
        { name: "learning rate α", range: "1e-4 – 1", increasing: "Bigger steps; too large oscillates or diverges.", strategy: "Log-scale search; scale features first so one rate suits all weights." },
        { name: "L2 penalty λ", range: "1e-4 – 10", increasing: "More shrinkage → higher bias, lower variance.", strategy: "Log-scale grid with cross-validation. (sklearn exposes C = 1/λ, so larger C means <i>less</i> regularization.)" },
        { name: "iterations", range: "100 – 10 000", increasing: "Closer to the optimum, then no further benefit.", strategy: "Raise until the loss curve flattens; watch for a convergence warning." },
        { name: "penalty type", range: "l1 / l2 / elasticnet", increasing: "-", strategy: "l2 by default; l1 when you suspect many irrelevant features." },
        { name: "class weight", range: "None / balanced", increasing: "Upweights the minority class.", strategy: "Use 'balanced' when the positive rate is below roughly 10%." },
        { name: "threshold", range: "0 – 1", increasing: "Higher precision, lower recall.", strategy: "Tune on the PR curve to match the real cost of FP vs FN - not automatically 0.5." },
      ],
      metrics: ["Accuracy", "Precision / Recall / F1", "ROC-AUC", "PR-AUC (imbalanced data)", "Log-loss", "Brier score (calibration)"],
      typicalUses: ["Churn / spam / click prediction", "Medical screening (binary)", "Credit risk scoring", "Any interpretable binary-classification baseline"],
      diagnostics: [
        "Compare train vs. validation log-loss - a widening gap means overfitting; raise λ.",
        "Calibration curve: bin the predicted probabilities and plot against observed frequency. A good model sits on the diagonal.",
        "Coefficient signs should match domain intuition; a flipped sign usually means multicollinearity.",
        "Exploding weights during training signal perfectly separable data - add L2.",
      ],
      advantages: [
        "Outputs genuinely calibrated probabilities, not just labels.",
        "Coefficients are directly interpretable as log-odds ratios (e^wᵢ is the odds ratio).",
        "Convex loss - one global optimum, no restart lottery.",
        "Trains and predicts fast, and scales to millions of sparse features.",
        "Regularization and online/incremental updates come for free.",
      ],
      limitations: [
        { name: "Linear boundary only", note: "cannot separate XOR-shaped or ring-shaped classes", fix: "add polynomial/interaction terms, or switch to SVM-RBF or a tree ensemble." },
        { name: "Sensitive to outliers", note: "extreme feature values drag the boundary", fix: "robust scaling or winsorizing." },
        { name: "Perfect separation", note: "weights diverge to infinity when classes are cleanly split", fix: "any nonzero L2 penalty." },
        { name: "Struggles with class imbalance", note: "predicts the majority class by default", fix: "class weights, resampling, or threshold tuning." },
        { name: "No automatic interactions", note: "feature crosses must be built by hand", fix: "explicit interaction terms or a tree model." },
      ],
      alternatives: [
        { name: "SVM (RBF kernel)", when: "The boundary is clearly non-linear and you don't need probabilities." },
        { name: "Gradient boosting", when: "Tabular data with interactions and non-linearity; accuracy matters more than interpretability." },
        { name: "Naive Bayes", when: "Very little training data, or very high-dimensional text features." },
        { name: "Neural network", when: "Large data with complex structure (images, sequences)." },
      ],
      pitfalls: [
        { problem: "Loss won't converge", solution: "Standardize features, lower the learning rate, raise the iteration cap." },
        { problem: "Perfect separation / huge weights", solution: "Add L2 regularization, or drop the leaking feature." },
        { problem: "Accuracy looks great on imbalanced data", solution: "Report PR-AUC and the confusion matrix instead of accuracy." },
        { problem: "Coefficients not comparable", solution: "Standardize features before comparing weight magnitudes." },
        { problem: "Probabilities are badly calibrated", solution: "Platt scaling or isotonic regression on a held-out split." },
      ],
      quickRef: [
        { name: "Linear predictor", formula: "z = wᵀx + b" },
        { name: "Sigmoid", formula: "σ(z) = 1 / (1 + e⁻ᶻ)" },
        { name: "Prediction", formula: "ŷ = 1[σ(z) ≥ 0.5]" },
        { name: "Log-odds", formula: "log(p / (1−p)) = wᵀx + b" },
        { name: "Log-loss", formula: "L = −(1/m)Σ[y·log ŷ + (1−y)·log(1−ŷ)]" },
        { name: "Gradient", formula: "∇L = (1/m)·Xᵀ(ŷ − y)" },
        { name: "Odds ratio", formula: "OR = e^wᵢ" },
        { name: "Softmax (K classes)", formula: "P(y=k) = e^{zₖ} / Σⱼ e^{zⱼ}" },
      ],
      code: `from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_auc_score

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=1.0,                   # inverse regularization: smaller = stronger
        penalty="l2",            # "l1" for feature selection
        solver="lbfgs",          # "saga" for l1/elasticnet on big data
        max_iter=1000,
        class_weight="balanced", # for skewed positive rates
    ),
)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_prob >= 0.5))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))`,
      whyChain: [
        { q: "What actually separates logistic from linear regression?", a: "Linear regression predicts an unbounded continuous value with squared error; logistic passes the same linear score through a sigmoid to get a probability and fits it with log-loss." },
        { q: "Why model the log-odds instead of the probability directly?", a: "Probabilities live in [0,1], which a linear function can leave. The logit maps them to (−∞, ∞), so a linear model is well-posed and the coefficients become clean odds ratios." },
        { q: "Why not just use MSE as the loss?", a: "Composed with the sigmoid, MSE is non-convex: it has plateaus where the gradient vanishes, so gradient descent stalls at bad solutions. Cross-entropy is convex here." },
        { q: "Where does cross-entropy come from?", a: "It's the negative log-likelihood of the Bernoulli model. Minimizing it is exactly maximum-likelihood estimation." },
        { q: "Why does L1 zero out coefficients but L2 doesn't?", a: "The L1 constraint region is a diamond with corners on the axes. Elliptical loss contours touch it at a corner, which is a point where some coordinates are exactly zero. L2's ball is smooth, so contact happens off-axis." },
        { q: "Your model is 97% accurate on a 3% positive rate. Good?", a: "No - predicting all-negative also scores 97%. Look at PR-AUC, recall, and the confusion matrix." },
        { q: "The weights blew up during training. Why?", a: "The classes are perfectly separable, so the likelihood keeps improving as ‖w‖ → ∞. Any L2 penalty bounds it." },
        { q: "How do you compare feature importance?", a: "Standardize the features first, then compare |wᵢ| - or use permutation importance, which doesn't assume linearity." },
      ],
      workedExample: {
        setup: "One gradient-descent step from w=(0,0,0) [bias,w1,w2], learning rate=1, on a single point x=(1,2), y=1.",
        steps: [
          "Score z = w0 + w1·1 + w2·2 = 0.",
          "Prediction ŷ = σ(0) = 0.5.",
          "Error = ŷ − y = 0.5 − 1 = −0.5.",
          "Gradient = error × [1, x1, x2] = −0.5 × [1, 1, 2] = [−0.5, −0.5, −1.0].",
          "Update: w_new = w − lr × gradient = [0,0,0] − [−0.5,−0.5,−1.0] = [0.5, 0.5, 1.0].",
        ],
        result: "After one step: w = [0.5, 0.5, 1.0] - the score for this point is now 0.5+0.5×1+1.0×2 = 3.0, so ŷ=σ(3.0)≈0.953, much closer to the true label 1.",
      },
    },
    mount,
  });
})();
