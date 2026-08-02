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
            <div class="note">Closed-form ridge regression: &beta; = (XᵀX + &lambda;I)⁻¹Xᵀy, recomputed on every edit, generalized to a polynomial basis so you can see over/underfitting live.</div>
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
    category: "Supervised - Regression",
    tagline: "ridge + polynomial basis",
    description: "Ordinary/ridge regression fit via the normal equations, extended with a polynomial basis so you can watch under- and overfitting happen as you drag points and change degree/&lambda; live.",
    info: {
      type: "Supervised - Regression. Parametric linear model (generalized here with a polynomial basis and an L2/ridge penalty).",
      scenario: "You need a simple, interpretable baseline for predicting a continuous target from numeric features - trend estimation, forecasting with few features, or as a first model before trying anything non-linear.",
      inputs: "A feature value x (expanded into a polynomial basis [1, x, x², …, x^d]) and a continuous target y for each training point.",
      intuition: {
        definition: "Fit the straight line (or polynomial curve) that makes the <b>squared vertical distances</b> to the training points as small as possible. Because the loss is a convex quadratic, the best fit has a closed-form solution - no iteration required.",
        steps: [
          "Assume the target is a weighted sum of the features plus noise.",
          "Measure fit by summed squared residuals (y − ŷ)².",
          "Set the gradient to zero → the normal equations solve for β directly.",
          "A polynomial basis keeps the model linear in β while bending the curve.",
        ],
        applications: [
          "Sales / demand forecasting from spend and seasonality",
          "Estimating the price effect of a house's square footage",
          "Dose–response curves in biology",
          "Calibrating a sensor against a reference instrument",
          "A reference baseline before trying any non-linear model",
        ],
      },
      math: [
        { title: "Design matrix", formula: "X = [1, x, x², …, x^d]  (one row per point)", note: "Polynomial features go in the columns, so the model stays <i>linear in the parameters</i> even when the curve is not." },
        { title: "Model", formula: "ŷ = Xβ,   y = Xβ + ε,  ε ~ N(0, σ²)", note: "Gaussian noise is what makes least squares the maximum-likelihood estimator." },
        { title: "Residual sum of squares", formula: "RSS(β) = ‖y − Xβ‖² = Σᵢ(yᵢ − ŷᵢ)²", note: "Convex quadratic bowl in β - one minimum, no local traps." },
        { title: "Normal equations", formula: "∇RSS = −2Xᵀ(y − Xβ) = 0  ⟹  XᵀXβ = Xᵀy", note: "Setting the gradient to zero gives a linear system." },
        { title: "Closed-form solution", formula: "β = (XᵀX + λI)⁻¹ Xᵀy", note: "λ = 0 is ordinary least squares. Any λ > 0 (ridge) also makes XᵀX invertible when features are collinear." },
      ],
      pipeline: [
        { label: "Raw feature x", note: "scalar input" },
        { label: "Basis expansion", note: "[1, x, …, x^d]" },
        { label: "Normal equations", note: "(XᵀX+λI)⁻¹Xᵀy" },
        { label: "Coefficients β", note: "fitted once" },
        { label: "Prediction ŷ", note: "ŷ = xᵀβ", accent: "green" },
      ],
      decisionFunction: {
        text: "ŷ = β₀ + β₁x + β₂x² + … + β_d·x^d",
        mechanism: "Prediction is a fixed weighted sum of the (possibly polynomial-transformed) input - a direct evaluation, with no iterative inference step once β is fit.",
        plot: { fn: (r) => r * r, domain: [-4, 4], color: "var(--accent)", caption: "squared error vs residual (y − ŷ) - this exact shape is what's being minimized below" },
      },
      lossFunction: {
        text: "L(β) = Σᵢ(yᵢ − ŷᵢ)² + λ·Σβₖ²",
        mechanism: "A convex quadratic in β, so it has one closed-form minimizer via the normal equations β = (XᵀX + λI)⁻¹Xᵀy - no gradient descent needed.",
      },
      optimization: [
        { title: "Closed form vs. gradient descent", formula: "O(p³) inversion  vs.  O(m·p) per step", note: "The normal equations win for a few thousand features; beyond that the p×p inversion dominates and gradient descent (or SGD) is cheaper." },
        { title: "Gradient (if solved iteratively)", formula: "∇L = −(2/m)·Xᵀ(y − Xβ) + 2λβ", note: "Same 'error × feature' form as logistic regression, because both are generalized linear models." },
        { title: "Conditioning", formula: "κ(XᵀX) = κ(X)²", note: "Squaring the condition number is why a raw normal-equation solve is numerically fragile; production solvers use QR or SVD instead." },
      ],
      output: "A continuous predicted value ŷ for any input x, plus the fitted coefficient vector β.",
      assumptions: [
        { name: "Linearity in parameters", why: "A genuinely curved relationship is systematically mis-fit.", check: "Plot residuals against fitted values - any pattern means the form is wrong." },
        { name: "Independent errors", why: "Autocorrelated residuals (common in time series) collapse the standard errors.", check: "Durbin–Watson statistic; plot residuals in collection order." },
        { name: "Homoscedasticity", why: "Non-constant error variance makes OLS inefficient and the confidence intervals wrong.", check: "Residual-vs-fitted plot should be a flat band, not a fan." },
        { name: "Normal errors", why: "Needed for exact t/F inference - not for the point estimate itself.", check: "Q–Q plot of residuals." },
        { name: "No perfect multicollinearity", why: "XᵀX becomes singular and β is undefined.", check: "VIF per feature; ridge fixes it outright." },
      ],
      regularization: [
        { name: "OLS", formula: "‖y − Xβ‖²", note: "No penalty. Unbiased, but high variance when features are many or correlated." },
        { name: "Ridge (L2)", formula: "‖y − Xβ‖² + λ‖β‖²", note: "Shrinks coefficients smoothly; keeps every feature. Best when predictors are correlated." },
        { name: "Lasso (L1)", formula: "‖y − Xβ‖² + λ‖β‖₁", note: "Drives coefficients to exactly zero - automatic feature selection. No closed form; solved by coordinate descent." },
        { name: "Elastic net", formula: "‖y − Xβ‖² + λ₁‖β‖₁ + λ₂‖β‖²", note: "Selects features while keeping correlated groups intact." },
      ],
      hyperparameters: [
        { name: "degree", range: "1 – 10", increasing: "More flexible curve; variance and overfitting rise sharply.", strategy: "Pick the smallest degree whose validation error stops improving; watch the ends of the range where polynomials swing wildly." },
        { name: "λ (ridge)", range: "1e-4 – 100", increasing: "Stronger shrinkage → higher bias, lower variance; β → 0 in the limit.", strategy: "Log-scale grid with k-fold CV. Always standardize features first - the penalty is scale-dependent." },
        { name: "fit intercept", range: "true / false", increasing: "-", strategy: "Keep it on unless the data is already centred. The intercept is never penalized." },
      ],
      metrics: ["R² (coefficient of determination)", "Adjusted R² (penalizes extra features)", "Mean Squared Error (MSE) / RMSE", "Mean Absolute Error (MAE)", "AIC / BIC for model selection"],
      typicalUses: ["Trend estimation and forecasting", "Interpretable baseline before non-linear models", "Quantifying a feature's effect via its coefficient", "Denoising a noisy 1D signal"],
      diagnostics: [
        "Residual-vs-fitted plot: should be structureless noise. A curve means the degree is too low; a fan means heteroscedasticity.",
        "Q–Q plot of residuals to check the normality assumption behind the confidence intervals.",
        "Cook's distance to find single points that are dragging the whole fit.",
        "A large gap between R² and adjusted R² means the extra features are not earning their keep.",
      ],
      advantages: [
        "Closed-form solution - deterministic, fast, and no hyperparameter search for plain OLS.",
        "Coefficients read directly as 'unit change in y per unit change in xᵢ'.",
        "Well-understood statistical theory: confidence intervals, p-values, and prediction intervals all come free.",
        "Extremely cheap at prediction time; trivial to deploy anywhere.",
        "Ridge makes it stable even with more features than observations.",
      ],
      limitations: [
        { name: "Only linear in the parameters", note: "genuinely non-linear structure needs a basis expansion you chose in advance", fix: "polynomial/spline features, or a tree ensemble." },
        { name: "Very outlier-sensitive", note: "squared error means one bad point can dominate the fit", fix: "Huber loss, RANSAC, or winsorizing." },
        { name: "Polynomial instability", note: "high-degree fits oscillate violently near the edges of the data (Runge's phenomenon)", fix: "splines, or keep the degree low and add ridge." },
        { name: "Breaks under multicollinearity", note: "coefficients become huge and sign-unstable", fix: "ridge, PCA, or drop redundant features." },
        { name: "Extrapolates badly", note: "predictions outside the training range are unconstrained", fix: "don't extrapolate; flag out-of-range inputs." },
      ],
      alternatives: [
        { name: "Ridge / lasso", when: "Many features, or features that are correlated." },
        { name: "Gradient boosting", when: "Tabular data with interactions and non-linearity, and interpretability is secondary." },
        { name: "Splines / GAM", when: "Smooth non-linearity, but you still want an interpretable additive model." },
        { name: "Huber / quantile regression", when: "Heavy-tailed noise or influential outliers." },
      ],
      pitfalls: [
        { problem: "R² keeps rising as you add features", solution: "R² never decreases with more features - use adjusted R² or a held-out split." },
        { problem: "Coefficients flip sign when a feature is added", solution: "Classic multicollinearity. Check VIF and apply ridge." },
        { problem: "Great train fit, poor test fit", solution: "Degree too high. Lower it or raise λ." },
        { problem: "Ridge penalty behaves erratically", solution: "Standardize features - the L2 penalty is not scale-invariant." },
        { problem: "Singular matrix error", solution: "Duplicate or perfectly collinear columns; drop them or use a nonzero λ." },
      ],
      quickRef: [
        { name: "Model", formula: "ŷ = Xβ" },
        { name: "Loss (OLS)", formula: "RSS = ‖y − Xβ‖²" },
        { name: "Normal equations", formula: "XᵀXβ = Xᵀy" },
        { name: "Ridge solution", formula: "β = (XᵀX + λI)⁻¹Xᵀy" },
        { name: "Gradient", formula: "∇L = −2Xᵀ(y − Xβ) + 2λβ" },
        { name: "R²", formula: "1 − RSS/TSS" },
        { name: "Adjusted R²", formula: "1 − (1−R²)(m−1)/(m−p−1)" },
        { name: "Slope (simple case)", formula: "β₁ = Sxy / Sxx" },
      ],
      code: `from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

pipe = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),          # the L2 penalty is scale-sensitive
    Ridge(alpha=1.0),          # alpha == lambda; 0 would be plain OLS
)

grid = GridSearchCV(
    pipe,
    {"polynomialfeatures__degree": [1, 2, 3, 5],
     "ridge__alpha": [1e-3, 1e-2, 0.1, 1, 10]},
    cv=5, scoring="neg_mean_squared_error",
).fit(X_train, y_train)

pred = grid.predict(X_test)
print(grid.best_params_)
print("R2:", r2_score(y_test, pred),
      "RMSE:", mean_squared_error(y_test, pred, squared=False))`,
      whyChain: [
        { q: "Why squared error rather than absolute error?", a: "Squaring makes the loss differentiable everywhere and convex-quadratic, which yields the closed-form normal equations. It's also the maximum-likelihood estimator under Gaussian noise." },
        { q: "So why would you ever use absolute error?", a: "Squared error weights a residual of 10 a hundred times more than a residual of 1, so outliers dominate. MAE (or Huber) is robust to them - you trade the closed form for robustness." },
        { q: "How can a polynomial fit still be 'linear' regression?", a: "Linearity refers to the parameters, not the input. ŷ = β₀ + β₁x + β₂x² is linear in β, so the same normal equations apply to the expanded design matrix." },
        { q: "Why does ridge fix multicollinearity?", a: "Collinear columns make XᵀX near-singular, so its inverse blows up. Adding λI lifts every eigenvalue by λ, guaranteeing invertibility and bounding ‖β‖." },
        { q: "Is ridge biased?", a: "Yes - deliberately. It accepts a little bias for a large drop in variance, which usually lowers total test error. That's the bias–variance trade-off made explicit." },
        { q: "Why must you standardize before ridge but not before OLS?", a: "OLS is equivariant to feature scaling; the fit is unchanged. Ridge penalizes Σβⱼ², so a feature measured in millimetres gets a far larger coefficient - and therefore a far larger penalty - than the same feature in metres." },
        { q: "Adding a useless feature raised my R². Is the model better?", a: "No. R² is non-decreasing in the number of features by construction. Use adjusted R², AIC, or a held-out test set." },
        { q: "Why avoid inverting XᵀX directly in production?", a: "Forming XᵀX squares the condition number, so precision is lost. QR decomposition or SVD solves the same least-squares problem far more stably." },
      ],
      parameters: [
        { name: "degree", effect: "Order of the polynomial basis. Higher degree fits more flexible curves but raises variance/overfitting risk, especially with few points." },
        { name: "λ (ridge)", effect: "L2 penalty strength. Larger λ shrinks coefficients toward zero, trading a bit of bias for lower variance." },
      ],
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
