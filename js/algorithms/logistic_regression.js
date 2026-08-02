(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  function trainGD(points, lr, iters, l2) {
    let w = [0, 0, 0]; // bias, w1, w2
    const n = points.length;
    if (!n) return w;
    for (let it = 0; it < iters; it++) {
      const grad = [0, 0, 0];
      for (const p of points) {
        const z = w[0] + w[1] * p.x + w[2] * p.y;
        const err = sigmoid(z) - p.label;
        grad[0] += err; grad[1] += err * p.x; grad[2] += err * p.y;
      }
      w[0] -= lr * (grad[0] / n);
      w[1] -= lr * (grad[1] / n + l2 * w[1]);
      w[2] -= lr * (grad[2] / n + l2 * w[2]);
    }
    return w;
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[0]}"></span>class 0</span>
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[1]}"></span>class 1</span>
            </div>
            <div class="btn-row">
              <button id="lg-regen">regenerate data</button>
              <button id="lg-clear">clear</button>
            </div>
          </div>
          <svg id="lg-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">click to add class <b>0</b> · shift-click to add class <b>1</b> · double-click a point to remove</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>training</h3>
            <div class="field"><label>learning rate <span class="val" id="lg-lr-val"></span></label>
              <input type="range" id="lg-lr" min="1" max="50" step="1" value="20" /></div>
            <div class="field"><label>L2 penalty <span class="val" id="lg-l2-val"></span></label>
              <input type="range" id="lg-l2" min="0" max="50" step="1" value="0" /></div>
          </div>
          <div class="control-card">
            <h3>fit</h3>
            <div class="readout" id="lg-readout">–</div>
            <div class="note">Batch gradient descent on the log-loss, same objective as <code>mla/linear_models.py</code>'s <code>LogisticRegression</code>. The background shading is P(class 1) = &sigma;(w&middot;x); the solid line is the 0.5 decision boundary.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#lg-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeTwoClass({ n: 60, mode: "linear" });
    const bgG = svg.append("g");
    const boundary = svg.append("line").attr("stroke", "var(--accent)").attr("stroke-width", 2);
    const ptsG = svg.append("g");

    function lr() { return +document.getElementById("lg-lr").value / 100; }
    function l2() { return +document.getElementById("lg-l2").value / 1000; }

    function render() {
      document.getElementById("lg-lr-val").textContent = lr().toFixed(2);
      document.getElementById("lg-l2-val").textContent = l2().toFixed(3);
      const w = trainGD(points, lr(), 400, l2());

      const cell = 16;
      const cells = [];
      for (let px = PAD; px < W - PAD; px += cell)
        for (let py = PAD; py < H - PAD; py += cell) {
          const xv = x.invert(px + cell / 2), yv = y.invert(py + cell / 2);
          const p = sigmoid(w[0] + w[1] * xv + w[2] * yv);
          cells.push({ px, py, p });
        }
      const rects = bgG.selectAll("rect").data(cells);
      rects.enter().append("rect").attr("width", cell).attr("height", cell)
        .merge(rects)
        .attr("x", (d) => d.px).attr("y", (d) => d.py)
        .attr("fill", (d) => d3.interpolateRgb(MLU.palette[0], MLU.palette[1])(d.p))
        .attr("opacity", 0.16);
      rects.exit().remove();

      if (Math.abs(w[2]) > 1e-6) {
        const x1 = DOMAIN[0], x2 = DOMAIN[1];
        const y1 = -(w[0] + w[1] * x1) / w[2], y2 = -(w[0] + w[1] * x2) / w[2];
        boundary.attr("x1", x(x1)).attr("y1", y(y1)).attr("x2", x(x2)).attr("y2", y(y2)).style("opacity", 1);
      } else boundary.style("opacity", 0);

      let correct = 0;
      for (const p of points) if ((sigmoid(w[0] + w[1] * p.x + w[2] * p.y) >= 0.5 ? 1 : 0) === p.label) correct++;
      document.getElementById("lg-readout").innerHTML =
        `points: <b>${points.length}</b><br>train accuracy: <b class="num">${points.length ? ((correct / points.length) * 100).toFixed(1) : "–"}%</b><br>` +
        `w: <span class="num">[${w.map((v) => v.toFixed(2)).join(", ")}]</span>`;

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
      points.push({ x: x.invert(px), y: y.invert(py), label: event.shiftKey ? 1 : 0 });
      render();
    });
    document.getElementById("lg-lr").addEventListener("input", render);
    document.getElementById("lg-l2").addEventListener("input", render);
    document.getElementById("lg-regen").addEventListener("click", () => { points = MLU.makeTwoClass({ n: 60, mode: "linear" }); render(); });
    document.getElementById("lg-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "logistic-regression",
    name: "Logistic Regression",
    category: "Supervised - Classification",
    tagline: "gradient descent, linear boundary",
    description: "Binary classifier trained with batch gradient descent on the log-loss. Background shading shows the predicted probability surface; the line is the 0.5 decision boundary.",
    sourceFile: "mla/linear_models.py",
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
