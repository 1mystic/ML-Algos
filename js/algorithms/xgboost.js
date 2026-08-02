(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];

  // Regularized, 2nd-order (Newton) regression tree: splits are scored with the
  // XGBoost gain formula and leaves use the closed-form regularized weight −G/(H+λ).
  function buildXGBTree(points, depth, maxDepth, lambda, gamma) {
    const G = points.reduce((s, p) => s + p.g, 0);
    const H = points.reduce((s, p) => s + p.h, 0);
    const leafValue = -G / (H + lambda);
    if (depth >= maxDepth || points.length < 4) return { leaf: true, value: leafValue };
    const xs = [...new Set(points.map((p) => p.x))].sort((a, b) => a - b);
    let best = null;
    for (let i = 0; i < xs.length - 1; i++) {
      const thr = (xs[i] + xs[i + 1]) / 2;
      const left = points.filter((p) => p.x <= thr), right = points.filter((p) => p.x > thr);
      if (!left.length || !right.length) continue;
      const GL = left.reduce((s, p) => s + p.g, 0), HL = left.reduce((s, p) => s + p.h, 0);
      const GR = right.reduce((s, p) => s + p.g, 0), HR = right.reduce((s, p) => s + p.h, 0);
      const gain = 0.5 * ((GL * GL) / (HL + lambda) + (GR * GR) / (HR + lambda) - (G * G) / (H + lambda)) - gamma;
      if (!best || gain > best.gain) best = { gain, thr, left, right };
    }
    if (!best || best.gain <= 0) return { leaf: true, value: leafValue }; // gain-based pruning
    return {
      leaf: false, thr: best.thr, gain: best.gain,
      left: buildXGBTree(best.left, depth + 1, maxDepth, lambda, gamma),
      right: buildXGBTree(best.right, depth + 1, maxDepth, lambda, gamma),
    };
  }
  function predictTree(node, x) { return node.leaf ? node.value : predictTree(x <= node.thr ? node.left : node.right, x); }

  function targetFn(xv) { return 2.4 * Math.sin(xv * 0.7) + 0.15 * xv; }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint">trees in ensemble: <b id="xgb-count">0</b></span></div>
            <div class="btn-row">
              <button id="xgb-step" class="primary">+ 1 tree</button>
              <button id="xgb-step10">+ 10 trees</button>
              <button id="xgb-reset">reset ensemble</button>
              <button id="xgb-regen">regenerate data</button>
            </div>
          </div>
          <svg id="xgb-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">each round: compute gradient g / hessian h per point, grow one gain-regularized tree, shrink it in</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>hyperparameters</h3>
            <div class="field"><label>tree depth <span class="val" id="xgb-depth-val">2</span></label>
              <input type="range" id="xgb-depth" min="1" max="4" step="1" value="2" /></div>
            <div class="field"><label>learning rate &eta; <span class="val" id="xgb-lr-val">0.30</span></label>
              <input type="range" id="xgb-lr" min="5" max="100" step="5" value="30" /></div>
            <div class="field"><label>&lambda; (L2 leaf reg.) <span class="val" id="xgb-lambda-val">1.0</span></label>
              <input type="range" id="xgb-lambda" min="0" max="100" step="1" value="10" /></div>
            <div class="field"><label>&gamma; (min. split gain) <span class="val" id="xgb-gamma-val">0.0</span></label>
              <input type="range" id="xgb-gamma" min="0" max="100" step="1" value="0" /></div>
          </div>
          <div class="control-card">
            <h3>fit</h3>
            <div class="readout" id="xgb-readout">–</div>
            <div class="note">Unlike plain gradient boosting, each split is only kept if its Gain exceeds &gamma; (a real prune step), and every leaf's value is shrunk toward 0 by &lambda; - try raising &gamma; and watch shallow, low-gain splits stop happening.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#xgb-svg");
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

    function depth() { return +document.getElementById("xgb-depth").value; }
    function lr() { return +document.getElementById("xgb-lr").value / 100; }
    function lambda() { return +document.getElementById("xgb-lambda").value / 10; }
    function gamma() { return +document.getElementById("xgb-gamma").value / 20; }

    function predict(xv) { return F0 + trees.reduce((s, t) => s + lr() * predictTree(t, xv), 0); }

    const path = svg.append("path").attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2.2);
    const ptsG = svg.append("g");

    function render() {
      document.getElementById("xgb-count").textContent = trees.length;
      document.getElementById("xgb-depth-val").textContent = depth();
      document.getElementById("xgb-lr-val").textContent = lr().toFixed(2);
      document.getElementById("xgb-lambda-val").textContent = lambda().toFixed(1);
      document.getElementById("xgb-gamma-val").textContent = gamma().toFixed(2);

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
      document.getElementById("xgb-readout").innerHTML =
        `points: <b>${points.length}</b><br>trees: <b>${trees.length}</b><br>train MSE: <b class="num">${mse.toFixed(3)}</b>`;
    }

    function boost() {
      // squared-error loss ⇒ gradient g = ŷ−y, hessian h = 1 per point
      const withGrad = points.map((p) => { const pred = predict(p.x); return { x: p.x, g: pred - p.y, h: 1 }; });
      trees.push(buildXGBTree(withGrad, 0, depth(), lambda(), gamma()));
    }

    document.getElementById("xgb-step").addEventListener("click", () => { boost(); render(); });
    document.getElementById("xgb-step10").addEventListener("click", () => { for (let i = 0; i < 10; i++) boost(); render(); });
    document.getElementById("xgb-reset").addEventListener("click", () => { trees = []; F0 = points.length ? MLU.mean(points.map((p) => p.y)) : 0; render(); });
    document.getElementById("xgb-regen").addEventListener("click", () => { points = genData(); trees = []; F0 = MLU.mean(points.map((p) => p.y)); render(); });
    ["xgb-depth", "xgb-lr", "xgb-lambda", "xgb-gamma"].forEach((id) => document.getElementById(id).addEventListener("input", render));

    F0 = MLU.mean(points.map((p) => p.y));
    render();
    return () => {};
  }

  MLApp.register({
    id: "xgboost",
    name: "XGBoost",
    category: "Supervised - Trees & Ensembles",
    tagline: "regularized, 2nd-order boosting",
    description: "Gradient boosting with a Newton (2nd-order, gradient+hessian) approximation to the loss, an explicit per-leaf L2 penalty, and a minimum-gain threshold that prunes low-value splits automatically.",
    info: {
      type: "Supervised - Regression/Classification. Regularized, second-order (Newton) gradient-boosted tree ensemble.",
      scenario: "Top predictive accuracy on tabular data with finer control over overfitting than plain gradient boosting - historically the dominant tabular-competition baseline.",
      inputs: "Feature vectors x and targets y.",
      intuition: {
        definition: "Gradient boosting rebuilt around an <b>explicit regularized objective</b>. Instead of fitting trees to residuals and hoping depth limits are enough, XGBoost writes down loss plus complexity penalty, expands it to second order, and derives the optimal leaf value and split gain in closed form.",
        steps: [
          "Take a second-order Taylor expansion of the loss, not just the gradient.",
          "Add explicit penalties for leaf count (γ) and leaf magnitude (λ).",
          "Solve for the optimal leaf weight analytically.",
          "Only accept a split if its gain beats γ, which prunes automatically.",
        ],
        applications: [
          "Kaggle tabular competitions, where it dominated for years",
          "Credit default and insurance risk models",
          "Fraud and anomaly scoring on transaction data",
          "Learning-to-rank for search and recommendation",
          "Click-through-rate prediction at ad-serving scale",
        ],
      },
      math: [
        { title: "Regularized objective", formula: "Obj = Σᵢ l(yᵢ, ŷᵢ) + Σₘ Ω(fₘ),   Ω(f) = γ·T + ½λ·Σⱼ wⱼ²", note: "T is the leaf count and wⱼ the leaf weights. Complexity is part of the objective, not an afterthought imposed by a depth cap." },
        { title: "Second-order expansion", formula: "Obj⁽ᵐ⁾ ≈ Σᵢ [ gᵢ·fₘ(xᵢ) + ½hᵢ·fₘ(xᵢ)² ] + Ω(fₘ)", note: "gᵢ = ∂l/∂ŷ and hᵢ = ∂²l/∂ŷ². Using curvature as well as slope is a Newton step, which converges faster and more stably than plain gradient descent." },
        { title: "Optimal leaf weight", formula: "wⱼ* = −Gⱼ / (Hⱼ + λ),   Gⱼ = Σ gᵢ,  Hⱼ = Σ hᵢ  over leaf j", note: "Derived by setting the derivative of the quadratic to zero. The +λ in the denominator is what shrinks leaves built from few or low-curvature points." },
        { title: "Split gain", formula: "Gain = ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ", note: "The objective improvement from splitting. A negative gain means the split is rejected, so γ acts as a built-in pruning threshold rather than a post-hoc step." },
        { title: "Squared-error case", formula: "gᵢ = ŷᵢ − yᵢ,  hᵢ = 1", note: "Hessians are all 1, so H is just the sample count and w* = −G/(n+λ), the shrunk mean residual. For log-loss, hᵢ = p(1−p), so confident points contribute little curvature." },
        { title: "Sparsity-aware splits", formula: "learn a default direction per node for missing values", note: "Missing entries are routed whichever way improves the gain, so missingness is handled natively instead of being imputed." },
      ],
      pipeline: [
        { label: "Current ŷ", note: "ensemble so far" },
        { label: "g and h", note: "grad + hessian" },
        { label: "Split gain", note: "prune if < γ" },
        { label: "Leaf w*", note: "−G/(H+λ)" },
        { label: "ŷ += η·tree", note: "next round", accent: "green" },
      ],
      decisionFunction: {
        text: "ŷ(x) = F₀ + η·Σₘ treeₘ(x), same additive form as plain GBM",
        mechanism: "The structure is identical to gradient boosting, but each tree's splits AND leaf values are chosen using a 2nd-order Taylor approximation of the loss (per-sample gradient g and hessian h), not just the 1st-order residual.",
      },
      lossFunction: {
        text: "Obj = Σᵢ l(yᵢ,ŷᵢ) + Σₘ[γT + ½λΣ_leaf w²]  →  leaf weight w* = −G/(H+λ),  Gain = ½[GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ)] − γ",
        mechanism: "For squared error, gᵢ=ŷᵢ−yᵢ and hᵢ=1, so the optimal leaf value has a closed form; a candidate split is only taken if its Gain is positive, which is what lets XGBoost prune low-value splits automatically via γ rather than relying solely on a depth limit.",
        plot: { fn: (lam) => 2 / (1 + lam), domain: [0, 10], color: "var(--accent)", caption: "regularized leaf weight w*=−G/(H+λ) for fixed G=−2,H=1 - larger λ shrinks the leaf value toward 0" },
      },
      optimization: [
        { title: "Newton boosting", formula: "step = −g/(h+λ)  rather than  −η·g", note: "Dividing by the curvature automatically scales the step: flat regions of the loss get large steps, sharply curved regions get small ones. Plain GBM has no such adaptation." },
        { title: "Approximate split finding", formula: "weighted quantile sketch over candidate thresholds", note: "Exact greedy search costs O(m log m) per feature. Bucketing into quantile candidates weighted by hᵢ keeps most of the accuracy at a fraction of the cost." },
        { title: "Column blocks", formula: "pre-sorted, compressed column storage", note: "Features are sorted once and reused across all rounds, which is what makes split search parallelisable across features despite boosting being sequential across rounds." },
        { title: "Cache-aware access", formula: "prefetch gradient statistics into per-thread buffers", note: "Much of XGBoost's real-world speed advantage is systems engineering rather than a change to the algorithm." },
        { title: "Early stopping", formula: "halt when validation metric stalls for k rounds", note: "Still essential. The explicit regularizers reduce but do not eliminate overfitting." },
      ],
      output: "A continuous predicted value (sum of shrunk, regularized tree contributions).",
      assumptions: [
        { name: "Loss is twice differentiable", why: "The method needs a Hessian, not just a gradient.", check: "Standard losses qualify. A custom objective must return both g and h." },
        { name: "Positive curvature", why: "A zero or negative Hessian makes the leaf weight undefined or unstable.", check: "λ > 0 guarantees a positive denominator. Never set λ = 0 with a custom loss." },
        { name: "Tabular, structured data", why: "Tree ensembles have no notion of spatial or sequential structure.", check: "For images, audio, or text, use a neural network." },
        { name: "Validation split available", why: "Round count must be chosen empirically.", check: "Always pass an eval_set." },
        { name: "Reasonable feature cardinality", why: "One-hot encoding a very high-cardinality categorical explodes the feature count.", check: "Use target encoding, or switch to LightGBM or CatBoost which handle categoricals natively." },
      ],
      regularization: [
        { name: "λ (reg_lambda)", formula: "+ ½λ Σⱼ wⱼ²", note: "L2 on leaf weights. Appears directly in the denominator of w*, so it shrinks leaves supported by little curvature. The default of 1 is already non-zero." },
        { name: "α (reg_alpha)", formula: "+ α Σⱼ |wⱼ|", note: "L1 on leaf weights. Can drive leaf values to exactly zero, effectively removing them." },
        { name: "γ (min_split_loss)", formula: "Gain − γ > 0 required", note: "A minimum improvement threshold. This is genuine pre-pruning derived from the objective, not a heuristic." },
        { name: "max_depth", formula: "depth ≤ D", note: "Still present as a hard cap. XGBoost grows level-wise, so depth directly bounds leaf count at 2^D." },
        { name: "min_child_weight", formula: "H_leaf ≥ threshold", note: "A minimum summed Hessian per leaf. For squared error that is a sample count; for log-loss it is an effective count that discounts confident points." },
        { name: "subsample / colsample", formula: "row and column fractions per tree", note: "Stochastic regularization inherited from bagging and stochastic gradient boosting." },
      ],
      hyperparameters: [
        { name: "learning_rate η", range: "0.01 - 0.3", increasing: "Fewer rounds needed, higher overfitting risk.", strategy: "Fix at 0.05, let early stopping pick the round count, and only revisit if training is too slow." },
        { name: "n_estimators", range: "100 - 10000", increasing: "More capacity, eventual overfitting.", strategy: "Set high and early-stop. Never grid-search it." },
        { name: "max_depth", range: "3 - 10", increasing: "Higher interaction order, exponentially more leaves, rapid overfitting.", strategy: "Start at 6. This is the strongest capacity lever after the round count." },
        { name: "min_child_weight", range: "1 - 100", increasing: "More conservative leaves, stronger regularization.", strategy: "Raise it on noisy or imbalanced data. Often more effective than lowering depth." },
        { name: "γ (min_split_loss)", range: "0 - 5", increasing: "More aggressive pruning, smaller trees.", strategy: "Leave at 0 initially. Raise it if trees are large and overfitting despite depth limits." },
        { name: "λ (reg_lambda)", range: "0.1 - 10", increasing: "Leaf values shrink toward zero.", strategy: "Default 1 is sensible. Log-scale search alongside depth." },
        { name: "subsample", range: "0.5 - 1.0", increasing: "Less noise injection, weaker regularization.", strategy: "0.8 is a solid default." },
        { name: "colsample_bytree", range: "0.3 - 1.0", increasing: "Trees see more features and correlate more.", strategy: "0.8 by default; lower with many correlated features." },
        { name: "scale_pos_weight", range: "1 - ratio", increasing: "Upweights the positive class.", strategy: "Set to negatives/positives for imbalanced binary classification." },
      ],
      metrics: ["RMSE / MAE / R² (regression)", "Log-loss / AUC (classification)", "Validation-set early-stopping curve", "SHAP values for per-prediction attribution"],
      typicalUses: ["Tabular ML competitions", "Credit risk / fraud scoring", "Click-through-rate prediction", "Ranking (e.g. LambdaMART-style objectives)"],
      diagnostics: [
        "Watch the eval_set curve each round. The gap between training and validation loss is your overfitting gauge.",
        "best_iteration after early stopping tells you the effective model size; if it equals your cap, raise the cap.",
        "Use SHAP rather than the built-in importances, and note that gain, weight, and cover importances often disagree.",
        "If gamma pruning is removing almost every split, it is set too high and the model will underfit.",
        "Compare against LightGBM. A large speed gap at similar accuracy means you should switch.",
      ],
      advantages: [
        "Regularization is part of the objective, so overfitting control is principled rather than heuristic.",
        "Second-order information makes each step better scaled and convergence faster than first-order boosting.",
        "Missing values are handled natively by learning a default branch direction per node.",
        "Heavily optimised implementation with parallel split finding, out-of-core training, and GPU support.",
        "Mature ecosystem: sklearn API, SHAP integration, ranking objectives, and custom loss support.",
      ],
      limitations: [
        { name: "Large hyperparameter surface", note: "roughly ten interacting knobs", fix: "tune in order: depth, then min_child_weight, then subsampling, then λ and γ." },
        { name: "Slower than LightGBM at scale", note: "level-wise growth and exact splits cost more", fix: "use tree_method='hist', or switch to LightGBM." },
        { name: "Categoricals need encoding", note: "no native categorical support in older versions", fix: "target encoding, or use CatBoost or LightGBM." },
        { name: "Still overfits noisy data", note: "regularization reduces but does not remove the tendency", fix: "early stopping plus higher min_child_weight." },
        { name: "Not interpretable directly", note: "hundreds of interacting trees", fix: "SHAP or partial dependence." },
        { name: "Cannot extrapolate", note: "tree predictions are flat beyond the training range", fix: "detrend, or model the trend separately." },
      ],
      alternatives: [
        { name: "LightGBM", when: "Large datasets where training speed and memory dominate. Usually faster at comparable accuracy." },
        { name: "CatBoost", when: "Many high-cardinality categorical features, or you want strong defaults with minimal tuning." },
        { name: "Random forest", when: "You want decent accuracy with almost no tuning." },
        { name: "Linear model", when: "Interpretability is required, or you must extrapolate." },
      ],
      pitfalls: [
        { problem: "Validation loss rises after a few rounds", solution: "Lower the learning rate and depth, and rely on early stopping." },
        { problem: "Model underfits with high gamma", solution: "γ is pruning valuable splits. Reduce it toward 0." },
        { problem: "Training is very slow", solution: "Set tree_method='hist', lower max_bin, or move to LightGBM." },
        { problem: "Minority class is ignored", solution: "Set scale_pos_weight to the negative-to-positive ratio." },
        { problem: "Feature importances disagree between types", solution: "Expected. Gain, weight, and cover measure different things. Use SHAP for a consistent answer." },
        { problem: "Custom objective produces NaN", solution: "The Hessian is probably zero or negative somewhere. Clip it to a small positive floor." },
      ],
      quickRef: [
        { name: "Objective", formula: "Σ l(y,ŷ) + γT + ½λΣw²" },
        { name: "Gradient", formula: "gᵢ = ∂l/∂ŷᵢ" },
        { name: "Hessian", formula: "hᵢ = ∂²l/∂ŷᵢ²" },
        { name: "Optimal leaf", formula: "w* = −G/(H+λ)" },
        { name: "Split gain", formula: "½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ" },
        { name: "MSE case", formula: "g = ŷ − y,  h = 1" },
        { name: "Log-loss case", formula: "g = p − y,  h = p(1−p)" },
        { name: "Update", formula: "ŷ += η · treeₘ(x)" },
      ],
      code: `import xgboost as xgb
from sklearn.model_selection import train_test_split

X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2)

model = xgb.XGBClassifier(
    n_estimators=5000,          # an upper bound; early stopping decides
    learning_rate=0.05,
    max_depth=6,                # main capacity lever
    min_child_weight=1,         # min summed hessian per leaf
    gamma=0,                    # min split gain; raise to prune harder
    reg_lambda=1.0,             # L2 on leaf weights
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",         # much faster than "exact"
    eval_metric="auc",
    early_stopping_rounds=50,
    scale_pos_weight=(y_fit == 0).sum() / (y_fit == 1).sum(),
    random_state=42,
)
model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=100)
print("best iteration:", model.best_iteration)

# Prefer SHAP to the built-in importances, which disagree by type.
import shap
values = shap.TreeExplainer(model).shap_values(X_val)`,
      whyChain: [
        { q: "What does XGBoost add over plain gradient boosting?", a: "Three things: an explicit complexity penalty inside the objective, a second-order Taylor expansion so splits and leaf values use curvature as well as slope, and a heavily engineered implementation. The additive tree structure itself is unchanged." },
        { q: "Why use the Hessian at all?", a: "It is a Newton step rather than a gradient step. Dividing by curvature scales the update automatically, so flat regions of the loss get bigger moves and sharp regions get smaller ones. That converges in fewer rounds and is more stable." },
        { q: "Where does the leaf weight formula come from?", a: "Once the tree structure is fixed, the objective for a leaf is a quadratic in w: G·w + ½(H+λ)w². Setting the derivative to zero gives w* = −G/(H+λ). It is not a heuristic, it is the exact minimiser." },
        { q: "How does γ prune trees?", a: "The gain formula already subtracts γ. If splitting a node does not improve the regularized objective by more than γ, the gain is negative and the split is rejected. Pruning is therefore built into split selection rather than applied afterwards." },
        { q: "What is min_child_weight actually measuring?", a: "The summed Hessian in a leaf, not the row count. For squared error the Hessian is 1 per row so it is a count, but for log-loss it is p(1−p), so confidently classified rows contribute almost nothing. It is an effective sample size." },
        { q: "Why does λ shrink small leaves more?", a: "w* = −G/(H+λ). A leaf with few points has small H, so λ dominates the denominator and drags the weight toward zero. Large leaves have H much bigger than λ and are barely affected." },
        { q: "How are missing values handled without imputation?", a: "At each node XGBoost tries sending all missing rows left and all right, and keeps whichever direction gives higher gain. That default direction is stored in the node and used at prediction time." },
        { q: "XGBoost or LightGBM?", a: "LightGBM is generally faster on large data because it grows leaf-wise on histograms. XGBoost with tree_method='hist' closes much of the gap and its level-wise growth is a little more conservative. Accuracy is usually close; pick on speed and categorical handling." },
      ],
      parameters: [
        { name: "learning rate η", effect: "Shrinks every tree's contribution, same role as in plain GBM." },
        { name: "λ (L2 on leaf weights)", effect: "Larger λ pulls every leaf's value toward 0, damping the influence of leaves with little data (low H)." },
        { name: "γ (min split gain)", effect: "A split is only made if it improves the objective by more than γ - larger γ prunes more aggressively, producing simpler trees." },
        { name: "max depth", effect: "Upper bound on tree depth, same role as in plain GBM/decision trees." },
      ],
      metrics: ["RMSE / MAE / R² (regression)", "Log-loss / AUC (classification)", "Validation-set early-stopping curve"],
      typicalUses: ["Tabular ML competitions", "Credit risk / fraud scoring", "Click-through-rate prediction", "Ranking (e.g. LambdaMART-style objectives)"],
      workedExample: {
        setup: "A leaf with summed gradient G=−2, hessian H=4, λ=1. A candidate split divides it into left (GL=−3, HL=2) and right (GR=1, HR=2), γ=0.1.",
        steps: [
          "Leaf weight before splitting: w* = −G/(H+λ) = −(−2)/(4+1) = 2/5 = 0.4.",
          "Gain = ½[GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ)] − γ.",
          "= ½[9/3 + 1/3 − 4/5] − 0.1 = ½[3 + 0.333 − 0.8] − 0.1 = ½(2.533) − 0.1 = 1.267 − 0.1.",
        ],
        result: "Gain ≈ 1.167 > 0 → the split is taken (it improves the regularized objective)",
      },
    },
    mount,
  });
})();
