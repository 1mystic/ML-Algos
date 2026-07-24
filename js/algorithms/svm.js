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
  // solver — well suited to a small in-browser demo with linear/poly/rbf kernels.
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
            <div class="note">Kernelized Pegasos solver on the dual soft-margin SVM objective — same Linear / Poly / RBF kernel choice as <code>mla/svm/svm.py</code>, trained online rather than via full QP for speed in-browser.</div>
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
    category: "Supervised — Classification",
    tagline: "linear / poly / RBF kernels",
    description: "A kernelized soft-margin SVM trained with the kernel-Pegasos online solver. Switch kernels to see why RBF handles the circular dataset that a linear kernel can't.",
    sourceFile: "mla/svm/svm.py",
    info: {
      type: "Supervised — Classification (an SVR regression variant also exists). Max-margin kernel method.",
      scenario: "Classification with a clear or kernel-separable margin, on small-to-medium datasets, when a robust, theoretically-grounded margin-based classifier is wanted; kernels let it handle non-linear boundaries.",
      inputs: "Feature vectors x, binary labels y ∈ {−1, +1}, and a choice of kernel (linear / poly / RBF).",
      decisionFunction: {
        text: "ŷ(x) = sign( Σⱼ αⱼyⱼK(xⱼ, x) + b )",
        mechanism: "The decision is a weighted sum of kernel similarities to the support vectors (points with αⱼ>0) rather than an explicit weight vector — this is what lets the model work in the implicit feature space a kernel induces.",
      },
      lossFunction: {
        text: "L = Σᵢ max(0, 1 − yᵢf(xᵢ)) + (1/2C)‖w‖²  (hinge loss, soft-margin)",
        mechanism: "Hinge loss is exactly zero once a point is correctly classified with margin ≥1, so only points near/inside the margin influence the solution — trained here via the kernelized Pegasos online sub-gradient method rather than full quadratic programming.",
        plot: { fn: (z) => Math.max(0, 1 - z), domain: [-2, 3], yDomain: [0, 3], color: "var(--accent)", fn2: (z) => Math.log(1 + Math.exp(-z)), color2: "var(--text-faint)", caption: "hinge loss (solid) vs margin z=y·f(x) — zero past margin 1; log-loss (dashed) never reaches exactly zero, which is why SVM solutions are sparse" },
      },
      output: "A predicted class label, plus the signed score f(x) (larger magnitude = more confident).",
      parameters: [
        { name: "kernel", effect: "Linear / poly / RBF — controls what shapes of decision boundary are reachable at all." },
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
