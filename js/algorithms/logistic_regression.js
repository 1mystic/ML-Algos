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
    category: "Supervised — Classification",
    tagline: "gradient descent, linear boundary",
    description: "Binary classifier trained with batch gradient descent on the log-loss. Background shading shows the predicted probability surface; the line is the 0.5 decision boundary.",
    sourceFile: "mla/linear_models.py",
    info: {
      type: "Supervised — Binary classification. Linear discriminative model (a generalized linear model with a logit/sigmoid link).",
      scenario: "Binary outcomes where you want calibrated probabilities and an interpretable, linear decision boundary — e.g. churn, click-through, disease presence, credit default.",
      inputs: "A feature vector x = (x₁, x₂) and a binary label y ∈ {0, 1}.",
      decisionFunction: {
        text: "P(y=1 | x) = σ(w₀ + w₁x₁ + w₂x₂), predict 1 if P ≥ 0.5",
        mechanism: "A linear score is passed through the sigmoid to squash it into (0, 1); the decision boundary is exactly the hyperplane where the linear score is zero.",
      },
      lossFunction: {
        text: "L(w) = −Σᵢ[yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ)]",
        mechanism: "Binary cross-entropy — convex in w, but has no closed form because of the sigmoid, so it's minimized iteratively with (batch) gradient descent.",
        plot: { fn: (z) => -Math.log(1 / (1 + Math.exp(-z))), domain: [-6, 6], yDomain: [0, 6], color: "var(--accent)", caption: "log-loss for the true class as its linear score z moves from very wrong (left) to very right (correct & confident)" },
      },
      output: "A probability P(y=1|x) in [0,1], thresholded (usually at 0.5) into a class label.",
      parameters: [
        { name: "learning rate", effect: "Gradient descent step size. Too high can overshoot/diverge; too low trains very slowly." },
        { name: "L2 penalty", effect: "Regularization strength on the weights (not the bias). Larger values shrink w, reducing overfitting/variance." },
        { name: "iterations", effect: "How many gradient steps are taken. Too few underfits; plenty are cheap since each step is O(n)." },
      ],
      metrics: ["Accuracy", "Precision / Recall / F1", "ROC-AUC", "Log-loss"],
      typicalUses: ["Churn / spam / click prediction", "Medical screening (binary)", "Credit risk scoring", "Any interpretable binary-classification baseline"],
      workedExample: {
        setup: "One gradient-descent step from w=(0,0,0) [bias,w1,w2], learning rate=1, on a single point x=(1,2), y=1.",
        steps: [
          "Score z = w0 + w1·1 + w2·2 = 0.",
          "Prediction ŷ = σ(0) = 0.5.",
          "Error = ŷ − y = 0.5 − 1 = −0.5.",
          "Gradient = error × [1, x1, x2] = −0.5 × [1, 1, 2] = [−0.5, −0.5, −1.0].",
          "Update: w_new = w − lr × gradient = [0,0,0] − [−0.5,−0.5,−1.0] = [0.5, 0.5, 1.0].",
        ],
        result: "After one step: w = [0.5, 0.5, 1.0] — the score for this point is now 0.5+0.5×1+1.0×2 = 3.0, so ŷ=σ(3.0)≈0.953, much closer to the true label 1.",
      },
    },
    mount,
  });
})();
