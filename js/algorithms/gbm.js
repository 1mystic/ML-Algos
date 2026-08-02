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
