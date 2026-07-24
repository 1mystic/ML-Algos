(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];

  function fitRidge(points, degree, lambda) {
    // design matrix with polynomial basis [1, x, x^2, ... x^degree]
    const X = points.map((p) => Array.from({ length: degree + 1 }, (_, k) => p.x ** k));
    const y = points.map((p) => p.y);
    const Xt = MLU.transpose(X);
    const XtX = MLU.matMul(Xt, X);
    for (let i = 0; i < XtX.length; i++) XtX[i][i] += lambda;
    const Xty = Xt.map((row) => row.reduce((s, v, j) => s + v * y[j], 0));
    return MLU.solveLinearSystem(XtX, Xty);
  }
  function predict(coeffs, x) { return coeffs.reduce((s, c, k) => s + c * x ** k, 0); }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="legend-item"><span class="swatch" style="background:${MLU.palette[0]}"></span>data point</span></div>
            <div class="btn-row">
              <button id="lr-regen">regenerate data</button>
              <button id="lr-clear">clear</button>
            </div>
          </div>
          <svg id="lr-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">click the plot to add a point · drag a point to move it · double-click a point to remove it</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>model</h3>
            <div class="field">
              <label>polynomial degree <span class="val" id="lr-deg-val">1</span></label>
              <input type="range" id="lr-degree" min="1" max="6" step="1" value="1" />
            </div>
            <div class="field">
              <label>ridge &lambda; (L2) <span class="val" id="lr-lambda-val">0.00</span></label>
              <input type="range" id="lr-lambda" min="0" max="300" step="1" value="0" />
            </div>
          </div>
          <div class="control-card">
            <h3>fit</h3>
            <div class="readout" id="lr-readout">–</div>
            <div class="note">Closed-form ridge regression: &beta; = (XᵀX + &lambda;I)⁻¹Xᵀy, recomputed on every edit — same normal-equation approach as <code>mla/linear_models.py</code>, generalized to a polynomial basis so you can see over/underfitting live.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#lr-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeLinear({ n: 26, slope: 0.9, intercept: -1.5, noise: 2 })
      .map((p) => ({ x: Math.max(DOMAIN[0], Math.min(DOMAIN[1], p.x)), y: Math.max(DOMAIN[0], Math.min(DOMAIN[1], p.y)) }));

    const path = svg.append("path").attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2);
    const ptsG = svg.append("g");

    function degree() { return +document.getElementById("lr-degree").value; }
    function lambda() { return +document.getElementById("lr-lambda").value / 10; }

    function render() {
      document.getElementById("lr-deg-val").textContent = degree();
      document.getElementById("lr-lambda-val").textContent = lambda().toFixed(1);

      let coeffs = [0, 0];
      let r2 = 0;
      if (points.length > degree()) {
        coeffs = fitRidge(points, degree(), lambda());
        const yMean = MLU.mean(points.map((p) => p.y));
        let ssRes = 0, ssTot = 0;
        for (const p of points) {
          const pred = predict(coeffs, p.x);
          ssRes += (p.y - pred) ** 2;
          ssTot += (p.y - yMean) ** 2;
        }
        r2 = ssTot > 1e-9 ? 1 - ssRes / ssTot : 1;
        const steps = 80;
        const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
        const pts2 = Array.from({ length: steps + 1 }, (_, i) => {
          const xv = DOMAIN[0] + (i / steps) * (DOMAIN[1] - DOMAIN[0]);
          return [xv, Math.max(DOMAIN[0] * 1.5, Math.min(DOMAIN[1] * 1.5, predict(coeffs, xv)))];
        });
        path.attr("d", line(pts2)).style("opacity", 1);
      } else {
        path.style("opacity", 0);
      }

      document.getElementById("lr-readout").innerHTML =
        `points: <b>${points.length}</b><br>R²: <b class="num">${points.length > degree() ? r2.toFixed(3) : "–"}</b><br>` +
        `&beta;: <span class="num">[${coeffs.map((c) => c.toFixed(2)).join(", ")}]</span>`;

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle")
        .attr("r", 5).attr("fill", MLU.palette[0]).attr("stroke", "var(--bg)").attr("stroke-width", 1)
        .style("cursor", "grab")
        .merge(sel)
        .attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y))
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
      points.push({ x: x.invert(px), y: y.invert(py) });
      render();
    });
    document.getElementById("lr-degree").addEventListener("input", render);
    document.getElementById("lr-lambda").addEventListener("input", render);
    document.getElementById("lr-regen").addEventListener("click", () => {
      points = MLU.makeLinear({ n: 26, slope: MLU.randRange(-1.5, 1.5), intercept: MLU.randRange(-3, 3), noise: 2 })
        .map((p) => ({ x: Math.max(DOMAIN[0], Math.min(DOMAIN[1], p.x)), y: Math.max(DOMAIN[0], Math.min(DOMAIN[1], p.y)) }));
      render();
    });
    document.getElementById("lr-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "linear-regression",
    name: "Linear Regression",
    category: "Supervised — Regression",
    tagline: "ridge + polynomial basis",
    description: "Ordinary/ridge regression fit via the normal equations, extended with a polynomial basis so you can watch under- and overfitting happen as you drag points and change degree/&lambda; live.",
    sourceFile: "mla/linear_models.py",
    info: {
      type: "Supervised — Regression. Parametric linear model (generalized here with a polynomial basis and an L2/ridge penalty).",
      scenario: "You need a simple, interpretable baseline for predicting a continuous target from numeric features — trend estimation, forecasting with few features, or as a first model before trying anything non-linear.",
      inputs: "A feature value x (expanded into a polynomial basis [1, x, x², …, x^d]) and a continuous target y for each training point.",
      decisionFunction: {
        text: "ŷ = β₀ + β₁x + β₂x² + … + β_d·x^d",
        mechanism: "Prediction is a fixed weighted sum of the (possibly polynomial-transformed) input — a direct evaluation, with no iterative inference step once β is fit.",
        plot: { fn: (r) => r * r, domain: [-4, 4], color: "var(--accent)", caption: "squared error vs residual (y − ŷ) — this exact shape is what's being minimized below" },
      },
      lossFunction: {
        text: "L(β) = Σᵢ(yᵢ − ŷᵢ)² + λ·Σβₖ²",
        mechanism: "A convex quadratic in β, so it has one closed-form minimizer via the normal equations β = (XᵀX + λI)⁻¹Xᵀy — no gradient descent needed.",
      },
      output: "A continuous predicted value ŷ for any input x, plus the fitted coefficient vector β.",
      parameters: [
        { name: "degree", effect: "Order of the polynomial basis. Higher degree fits more flexible curves but raises variance/overfitting risk, especially with few points." },
        { name: "λ (ridge)", effect: "L2 penalty strength. Larger λ shrinks coefficients toward zero, trading a bit of bias for lower variance." },
      ],
      metrics: ["R² (coefficient of determination)", "Mean Squared Error (MSE) / RMSE", "Mean Absolute Error (MAE)"],
      typicalUses: ["Trend estimation and forecasting", "Interpretable baseline before non-linear models", "Quantifying a feature's effect via its coefficient", "Denoising a noisy 1D signal"],
      workedExample: {
        setup: "Fit ŷ = β₀ + β₁x (degree 1, no ridge) to points (1,2), (2,3), (3,5), (4,4).",
        steps: [
          "Means: x̄ = (1+2+3+4)/4 = 2.5, ȳ = (2+3+5+4)/4 = 3.5.",
          "Deviations (x−x̄): −1.5, −0.5, 0.5, 1.5. Deviations (y−ȳ): −1.5, −0.5, 1.5, 0.5.",
          "Sxy = Σ(x−x̄)(y−ȳ) = 2.25 + 0.25 + 0.75 + 0.75 = 4.0.",
          "Sxx = Σ(x−x̄)² = 2.25 + 0.25 + 0.25 + 2.25 = 5.0.",
          "β₁ = Sxy / Sxx = 4.0 / 5.0 = 0.8.",
          "β₀ = ȳ − β₁x̄ = 3.5 − 0.8×2.5 = 1.5.",
        ],
        result: "ŷ = 1.5 + 0.8x → prediction at x=5: ŷ = 1.5 + 0.8×5 = 5.5",
      },
    },
    mount,
  });
})();
