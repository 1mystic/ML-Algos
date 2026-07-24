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
            <div class="note">Unlike plain gradient boosting, each split is only kept if its Gain exceeds &gamma; (a real prune step), and every leaf's value is shrunk toward 0 by &lambda; — try raising &gamma; and watch shallow, low-gain splits stop happening.</div>
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
    category: "Supervised — Trees & Ensembles",
    tagline: "regularized, 2nd-order boosting",
    description: "Gradient boosting with a Newton (2nd-order, gradient+hessian) approximation to the loss, an explicit per-leaf L2 penalty, and a minimum-gain threshold that prunes low-value splits automatically.",
    sourceFile: "not in the original repo — added as a widely-used, more regularized extension of mla/ensemble/gbm.py's plain gradient boosting",
    info: {
      type: "Supervised — Regression/Classification. Regularized, second-order (Newton) gradient-boosted tree ensemble.",
      scenario: "Top predictive accuracy on tabular data with finer control over overfitting than plain gradient boosting — historically the dominant tabular-competition baseline.",
      inputs: "Feature vectors x and targets y.",
      decisionFunction: {
        text: "ŷ(x) = F₀ + η·Σₘ treeₘ(x), same additive form as plain GBM",
        mechanism: "The structure is identical to gradient boosting, but each tree's splits AND leaf values are chosen using a 2nd-order Taylor approximation of the loss (per-sample gradient g and hessian h), not just the 1st-order residual.",
      },
      lossFunction: {
        text: "Obj = Σᵢ l(yᵢ,ŷᵢ) + Σₘ[γT + ½λΣ_leaf w²]  →  leaf weight w* = −G/(H+λ),  Gain = ½[GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ)] − γ",
        mechanism: "For squared error, gᵢ=ŷᵢ−yᵢ and hᵢ=1, so the optimal leaf value has a closed form; a candidate split is only taken if its Gain is positive, which is what lets XGBoost prune low-value splits automatically via γ rather than relying solely on a depth limit.",
        plot: { fn: (lam) => 2 / (1 + lam), domain: [0, 10], color: "var(--accent)", caption: "regularized leaf weight w*=−G/(H+λ) for fixed G=−2,H=1 — larger λ shrinks the leaf value toward 0" },
      },
      output: "A continuous predicted value (sum of shrunk, regularized tree contributions).",
      parameters: [
        { name: "learning rate η", effect: "Shrinks every tree's contribution, same role as in plain GBM." },
        { name: "λ (L2 on leaf weights)", effect: "Larger λ pulls every leaf's value toward 0, damping the influence of leaves with little data (low H)." },
        { name: "γ (min split gain)", effect: "A split is only made if it improves the objective by more than γ — larger γ prunes more aggressively, producing simpler trees." },
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
