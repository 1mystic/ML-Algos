(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];

  function sse(vals) {
    if (!vals.length) return 0;
    const m = MLU.mean(vals);
    return vals.reduce((s, v) => s + (v - m) ** 2, 0);
  }
  function buildRegTree(pts, depth, maxDepth) {
    if (depth >= maxDepth || pts.length < 4) return { leaf: true, value: MLU.mean(pts.map((p) => p.r)) };
    const xs = [...new Set(pts.map((p) => p.x))].sort((a, b) => a - b);
    let best = null;
    for (let i = 0; i < xs.length - 1; i++) {
      const thr = (xs[i] + xs[i + 1]) / 2;
      const left = pts.filter((p) => p.x <= thr), right = pts.filter((p) => p.x > thr);
      if (!left.length || !right.length) continue;
      const cost = sse(left.map((p) => p.r)) + sse(right.map((p) => p.r));
      if (!best || cost < best.cost) best = { cost, thr, left, right };
    }
    if (!best) return { leaf: true, value: MLU.mean(pts.map((p) => p.r)) };
    return { leaf: false, thr: best.thr, left: buildRegTree(best.left, depth + 1, maxDepth), right: buildRegTree(best.right, depth + 1, maxDepth) };
  }
  function predictTree(node, x) { return node.leaf ? node.value : predictTree(x <= node.thr ? node.left : node.right, x); }

  function targetFn(xv) { return 2.4 * Math.sin(xv * 0.7) + 0.15 * xv; }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint" style="font-size:11.5px">trees in ensemble: <b id="gbm-count">0</b></span></div>
            <div class="btn-row">
              <button id="gbm-step" class="primary">+ 1 tree</button>
              <button id="gbm-step10">+ 10 trees</button>
              <button id="gbm-reset">reset ensemble</button>
              <button id="gbm-regen">regenerate data</button>
            </div>
          </div>
          <svg id="gbm-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">each click fits one shallow tree to the current residuals and shrinks it into the ensemble</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>hyperparameters</h3>
            <div class="field"><label>tree depth <span class="val" id="gbm-depth-val">2</span></label>
              <input type="range" id="gbm-depth" min="1" max="4" step="1" value="2" /></div>
            <div class="field"><label>learning rate <span class="val" id="gbm-lr-val">0.30</span></label>
              <input type="range" id="gbm-lr" min="5" max="100" step="5" value="30" /></div>
          </div>
          <div class="control-card">
            <h3>fit</h3>
            <div class="readout" id="gbm-readout">–</div>
            <div class="note">Gradient boosting on squared error: F<sub>0</sub> = mean(y); each new shallow tree is fit to the current residual y − F(x), then added in as F += &eta;&middot;tree(x) - the loop in <code>mla/ensemble/gbm.py</code>.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#gbm-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    function genData() {
      const pts = [];
      for (let i = 0; i < 45; i++) { const xv = MLU.randRange(...DOMAIN); pts.push({ x: xv, y: targetFn(xv) + MLU.randn() * 0.6 }); }
      return pts;
    }
    let points = genData();
    let trees = [], F0 = 0;

    function depth() { return +document.getElementById("gbm-depth").value; }
    function lr() { return +document.getElementById("gbm-lr").value / 100; }

    function predict(xv) { return F0 + trees.reduce((s, t) => s + lr() * predictTree(t, xv), 0); }

    const path = svg.append("path").attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2.2);
    const ptsG = svg.append("g");

    function render() {
      document.getElementById("gbm-count").textContent = trees.length;
      document.getElementById("gbm-depth-val").textContent = depth();
      document.getElementById("gbm-lr-val").textContent = lr().toFixed(2);

      const steps = 160;
      const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
      const curve = Array.from({ length: steps + 1 }, (_, i) => {
        const xv = DOMAIN[0] + (i / steps) * (DOMAIN[1] - DOMAIN[0]);
        return [xv, predict(xv)];
      });
      path.attr("d", line(curve));

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle").attr("r", 4.5).attr("fill", MLU.palette[0]).attr("stroke", "var(--bg)").attr("stroke-width", 1)
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y));
      sel.exit().remove();

      const mse = points.length ? MLU.mean(points.map((p) => (p.y - predict(p.x)) ** 2)) : 0;
      document.getElementById("gbm-readout").innerHTML =
        `points: <b>${points.length}</b><br>trees: <b>${trees.length}</b><br>train MSE: <b class="num">${mse.toFixed(3)}</b>`;
    }

    function boost() {
      const residPts = points.map((p) => ({ x: p.x, r: p.y - predict(p.x) }));
      trees.push(buildRegTree(residPts, 0, depth()));
    }

    document.getElementById("gbm-step").addEventListener("click", () => { boost(); render(); });
    document.getElementById("gbm-step10").addEventListener("click", () => { for (let i = 0; i < 10; i++) boost(); render(); });
    document.getElementById("gbm-reset").addEventListener("click", () => { trees = []; F0 = points.length ? MLU.mean(points.map((p) => p.y)) : 0; render(); });
    document.getElementById("gbm-regen").addEventListener("click", () => {
      points = genData(); trees = []; F0 = MLU.mean(points.map((p) => p.y)); render();
    });
    document.getElementById("gbm-depth").addEventListener("input", render);
    document.getElementById("gbm-lr").addEventListener("input", render);

    F0 = MLU.mean(points.map((p) => p.y));
    render();
    return () => {};
  }

  MLApp.register({
    id: "gbm",
    name: "Gradient Boosting Trees",
    category: "Supervised - Trees & Ensembles",
    tagline: "stagewise residual fitting",
    description: "Builds an ensemble one shallow regression tree at a time, each fit to the previous ensemble's residuals and shrunk in by a learning rate. Step through boosting rounds to watch the fit tighten.",
    sourceFile: "mla/ensemble/gbm.py",
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
      output: "A continuous predicted value: the sum of every tree's shrunk contribution.",
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
