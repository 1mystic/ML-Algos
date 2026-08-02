(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];

  function kernel(a, b, type, p) {
    const dot = a.x * b.x + a.y * b.y;
    if (type === "linear") return dot;
    if (type === "poly") return (dot + 1) ** p.degree;
    if (type === "rbf") { const d2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2; return Math.exp(-p.gamma * d2); }
    return dot;
  }

  // Kernelized Pegasos (Shalev-Shwartz et al.): a light, real online SVM
  // solver - well suited to a small in-browser demo with linear/poly/rbf kernels.
  function trainSVM(points, type, params, C) {
    const n = points.length;
    const alpha = new Array(n).fill(0);
    if (n < 2) return { alpha, decide: () => 0 };
    const lambda = 1 / (C * n);
    const T = Math.max(400, n * 60);
    // precompute kernel matrix once (n is small)
    const K = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => kernel(points[i], points[j], type, params)));
    for (let t = 1; t <= T; t++) {
      const i = MLU.randInt(n);
      let s = 0;
      for (let j = 0; j < n; j++) if (alpha[j]) s += alpha[j] * points[j].label * K[i][j];
      const margin = points[i].label * (s / (lambda * t));
      if (margin < 1) alpha[i] += 1;
    }
    const scale = 1 / (lambda * T);
    function decide(pt) {
      let s = 0;
      for (let j = 0; j < n; j++) if (alpha[j]) s += alpha[j] * points[j].label * kernel(points[j], pt, type, params);
      return s * scale;
    }
    return { alpha, decide };
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[0]}"></span>class -1</span>
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[1]}"></span>class +1</span>
              <span class="legend-item">ring = support vector</span>
            </div>
            <div class="btn-row">
              <button id="svm-preset-linear">linear data</button>
              <button id="svm-preset-circles">circular data</button>
              <button id="svm-clear">clear</button>
            </div>
          </div>
          <svg id="svm-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">click to add class −1 · shift-click for class +1 · retrains automatically</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>kernel</h3>
            <div class="field"><label>type</label>
              <select id="svm-kernel"><option value="linear">linear</option><option value="rbf" selected>RBF</option><option value="poly">poly</option></select>
            </div>
            <div class="field" id="svm-gamma-field"><label>&gamma; (RBF) <span class="val" id="svm-gamma-val">0.5</span></label>
              <input type="range" id="svm-gamma" min="1" max="200" step="1" value="50" /></div>
            <div class="field" id="svm-degree-field" style="display:none"><label>degree (poly) <span class="val" id="svm-degree-val">3</span></label>
              <input type="range" id="svm-degree" min="2" max="5" step="1" value="3" /></div>
            <div class="field"><label>C (soft-margin) <span class="val" id="svm-c-val">1.0</span></label>
              <input type="range" id="svm-c" min="1" max="100" step="1" value="10" /></div>
          </div>
          <div class="control-card">
            <h3>fit</h3>
            <div class="readout" id="svm-readout">–</div>
            <div class="note">Kernelized Pegasos solver on the dual soft-margin SVM objective - same Linear / Poly / RBF kernel choice as <code>mla/svm/svm.py</code>, trained online rather than via full QP for speed in-browser.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#svm-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    function presetLinear() { return MLU.makeTwoClass({ n: 50, mode: "linear" }).map((p) => ({ ...p, label: p.label === 0 ? -1 : 1 })); }
    function presetCircles() { return MLU.makeTwoClass({ n: 70, mode: "circles" }).map((p) => ({ ...p, label: p.label === 0 ? -1 : 1 })); }
    let points = presetCircles();

    const bgG = svg.append("g");
    const ptsG = svg.append("g");

    function kType() { return document.getElementById("svm-kernel").value; }
    function params() { return { gamma: +document.getElementById("svm-gamma").value / 100, degree: +document.getElementById("svm-degree").value }; }
    function Cval() { return +document.getElementById("svm-c").value / 10; }

    function render() {
      document.getElementById("svm-gamma-val").textContent = params().gamma.toFixed(2);
      document.getElementById("svm-degree-val").textContent = params().degree;
      document.getElementById("svm-c-val").textContent = Cval().toFixed(1);
      document.getElementById("svm-gamma-field").style.display = kType() === "rbf" ? "" : "none";
      document.getElementById("svm-degree-field").style.display = kType() === "poly" ? "" : "none";

      const hasBoth = points.some((p) => p.label === -1) && points.some((p) => p.label === 1);
      let model = null;
      if (hasBoth) model = trainSVM(points, kType(), params(), Cval());

      const cell = 14, cells = [];
      if (model) {
        for (let px = PAD; px < W - PAD; px += cell)
          for (let py = PAD; py < H - PAD; py += cell) {
            const xv = x.invert(px + cell / 2), yv = y.invert(py + cell / 2);
            cells.push({ px, py, f: model.decide({ x: xv, y: yv }) });
          }
      }
      const rects = bgG.selectAll("rect").data(cells);
      rects.enter().append("rect").attr("width", cell).attr("height", cell)
        .merge(rects).attr("x", (d) => d.px).attr("y", (d) => d.py)
        .attr("fill", (d) => (d.f >= 0 ? MLU.palette[1] : MLU.palette[0]))
        .attr("opacity", (d) => Math.min(0.32, 0.08 + Math.abs(Math.tanh(d.f)) * 0.22));
      rects.exit().remove();

      const sel = ptsG.selectAll("g.pt").data(points);
      const enter = sel.enter().append("g").attr("class", "pt");
      enter.append("circle").attr("class", "sv-ring").attr("r", 9).attr("fill", "none").attr("stroke", "var(--text)").attr("stroke-width", 1.2);
      enter.append("circle").attr("class", "dot").attr("r", 5).attr("stroke", "var(--bg)").attr("stroke-width", 1);
      const merged = enter.merge(sel);
      merged.attr("transform", (d) => `translate(${x(d.x)},${y(d.y)})`).style("cursor", "grab");
      merged.select(".dot").attr("fill", (d) => (d.label === 1 ? MLU.palette[1] : MLU.palette[0]));
      merged.select(".sv-ring").style("opacity", (d, i) => (model && model.alpha[i] > 0 ? 1 : 0));
      merged
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); render(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          render();
        }));
      sel.exit().remove();

      let correct = 0;
      if (model) for (const p of points) if (Math.sign(model.decide(p) || 1) === p.label) correct++;
      document.getElementById("svm-readout").innerHTML = hasBoth
        ? `points: <b>${points.length}</b><br>support vectors: <b>${model.alpha.filter((a) => a > 0).length}</b><br>train accuracy: <b class="num">${((correct / points.length) * 100).toFixed(1)}%</b>`
        : "need points from both classes";
    }

    svg.on("click", (event) => {
      if (event.target.closest(".pt")) return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py), label: event.shiftKey ? 1 : -1 });
      render();
    });
    ["svm-kernel", "svm-gamma", "svm-degree", "svm-c"].forEach((id) => document.getElementById(id).addEventListener("input", render));
    document.getElementById("svm-preset-linear").addEventListener("click", () => { points = presetLinear(); render(); });
    document.getElementById("svm-preset-circles").addEventListener("click", () => { points = presetCircles(); render(); });
    document.getElementById("svm-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "svm",
    name: "Support Vector Machine",
    category: "Supervised - Classification",
    tagline: "linear / poly / RBF kernels",
    description: "A kernelized soft-margin SVM trained with the kernel-Pegasos online solver. Switch kernels to see why RBF handles the circular dataset that a linear kernel can't.",
    sourceFile: "mla/svm/svm.py",
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
