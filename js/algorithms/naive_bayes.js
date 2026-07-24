(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];
  const N_CLASSES = 3;

  function fitGaussianNB(points) {
    const classes = [];
    for (let c = 0; c < N_CLASSES; c++) {
      const pts = points.filter((p) => p.label === c);
      if (!pts.length) { classes.push(null); continue; }
      const mx = MLU.mean(pts.map((p) => p.x)), my = MLU.mean(pts.map((p) => p.y));
      const sx = MLU.std(pts.map((p) => p.x)), sy = MLU.std(pts.map((p) => p.y));
      classes.push({ mx, my, sx: Math.max(sx, 0.4), sy: Math.max(sy, 0.4), prior: pts.length / points.length });
    }
    return classes;
  }
  function logGauss(v, m, s) { return -0.5 * Math.log(2 * Math.PI * s * s) - ((v - m) ** 2) / (2 * s * s); }
  function predict(model, pt) {
    let best = -1, bestScore = -Infinity, scores = [];
    for (let c = 0; c < N_CLASSES; c++) {
      const m = model[c];
      if (!m) { scores.push(-Infinity); continue; }
      const score = Math.log(m.prior) + logGauss(pt.x, m.mx, m.sx) + logGauss(pt.y, m.my, m.sy);
      scores.push(score);
      if (score > bestScore) { bestScore = score; best = c; }
    }
    return { label: best, scores };
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">${[0, 1, 2].map((i) => `<span class="legend-item"><span class="swatch" style="background:${MLU.palette[i]}"></span>class ${i}</span>`).join("")}</div>
            <div class="btn-row"><button id="nb-regen">regenerate data</button><button id="nb-clear">clear</button></div>
          </div>
          <svg id="nb-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">pick a class, click to add points · dashed ellipses = 1&sigma; of each class's fitted Gaussian</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>add points as</h3>
            <div class="btn-row" id="nb-class-buttons">
              ${[0, 1, 2].map((i) => `<button data-c="${i}" style="border-color:${MLU.palette[i]}">${i}</button>`).join("")}
            </div>
          </div>
          <div class="control-card">
            <h3>model</h3>
            <div class="readout" id="nb-readout">–</div>
            <div class="note">Gaussian Naive Bayes: each feature modeled with an independent per-class normal distribution; prediction = argmax of log-prior + log-likelihood, matching <code>mla/naive_bayes.py</code>.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#nb-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeBlobs({ n: 60, clusters: 3 }).map((p) => ({ x: p.x, y: p.y, label: p.label }));
    let currentClass = 0;
    const bgG = svg.append("g");
    const ellG = svg.append("g");
    const ptsG = svg.append("g");

    function render() {
      const model = fitGaussianNB(points);
      const cell = 14, cells = [];
      for (let px = PAD; px < W - PAD; px += cell)
        for (let py = PAD; py < H - PAD; py += cell) {
          const xv = x.invert(px + cell / 2), yv = y.invert(py + cell / 2);
          const { label } = predict(model, { x: xv, y: yv });
          cells.push({ px, py, label });
        }
      const rects = bgG.selectAll("rect").data(cells);
      rects.enter().append("rect").attr("width", cell).attr("height", cell)
        .merge(rects).attr("x", (d) => d.px).attr("y", (d) => d.py)
        .attr("fill", (d) => (d.label >= 0 ? MLU.palette[d.label] : "transparent")).attr("opacity", points.length ? 0.16 : 0);
      rects.exit().remove();

      const ellData = model.filter(Boolean).map((m, i) => m && { ...m, c: model.indexOf(m) });
      const ell = ellG.selectAll("ellipse").data(model.map((m, i) => (m ? { ...m, c: i } : null)).filter(Boolean));
      ell.enter().append("ellipse").attr("fill", "none").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3")
        .merge(ell)
        .attr("cx", (d) => x(d.mx)).attr("cy", (d) => y(d.my))
        .attr("rx", (d) => Math.abs(x(d.mx + d.sx) - x(d.mx))).attr("ry", (d) => Math.abs(y(d.my - d.sy) - y(d.my)))
        .attr("stroke", (d) => MLU.palette[d.c]);
      ell.exit().remove();

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle")
        .attr("r", 5).attr("stroke", "var(--bg)").attr("stroke-width", 1).style("cursor", "grab")
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("fill", (d) => MLU.palette[d.label])
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); render(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          render();
        }));
      sel.exit().remove();

      document.getElementById("nb-readout").innerHTML = model.map((m, i) => m
        ? `class ${i}: &mu;=(${m.mx.toFixed(1)}, ${m.my.toFixed(1)}) &sigma;=(${m.sx.toFixed(1)}, ${m.sy.toFixed(1)}) prior=${m.prior.toFixed(2)}`
        : `class ${i}: no data`).join("<br>");
    }

    svg.on("click", (event) => {
      if (event.target.tagName === "circle") return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py), label: currentClass });
      render();
    });
    document.getElementById("nb-class-buttons").addEventListener("click", (e) => {
      const btn = e.target.closest("button"); if (!btn) return;
      currentClass = +btn.dataset.c;
      document.querySelectorAll("#nb-class-buttons button").forEach((b) => b.classList.toggle("primary", +b.dataset.c === currentClass));
    });
    document.querySelector("#nb-class-buttons button").classList.add("primary");
    document.getElementById("nb-regen").addEventListener("click", () => {
      points = MLU.makeBlobs({ n: 60, clusters: 3 }).map((p) => ({ x: p.x, y: p.y, label: p.label }));
      render();
    });
    document.getElementById("nb-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "naive-bayes",
    name: "Gaussian Naive Bayes",
    category: "Supervised — Classification",
    tagline: "independent per-class gaussians",
    description: "Fits an independent 1D normal distribution per feature per class, then classifies by the highest log-prior + log-likelihood. Ellipses show each class's fitted 1σ Gaussian.",
    sourceFile: "mla/naive_bayes.py",
    info: {
      type: "Supervised — Classification. Generative probabilistic model (Gaussian Naive Bayes).",
      scenario: "A fast, simple baseline classifier, especially effective with high-dimensional or sparse features (text), or whenever a cheap probabilistic model is preferable to an iteratively-trained one.",
      inputs: "Feature vector x and class label y; each feature is assumed conditionally independent and normally distributed given the class.",
      decisionFunction: {
        text: "ŷ = argmax_c [ log P(c) + Σ_features log N(xᵢ; μ_{c,i}, σ_{c,i}) ]",
        mechanism: "Bayes' rule under a 'naive' feature-independence assumption: each feature's likelihood under a class's fitted Gaussian is combined (summed in log-space) with that class's prior, and the highest-scoring class wins.",
        plot: { fn: (x) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI), domain: [-4, 4], color: "var(--accent)", caption: "the per-feature Gaussian each class's likelihood is built from (standard normal shown)" },
      },
      lossFunction: {
        text: "No iterative loss — closed-form maximum-likelihood estimation.",
        mechanism: "Each class's mean, variance, and prior are computed directly from the training data in a single pass; there's no optimization loop to converge.",
      },
      output: "A predicted class label, plus a posterior probability / log-score for every class.",
      parameters: [
        { name: "variance smoothing", effect: "A small ε added to every variance estimate to avoid division-by-zero when a class's feature has near-zero spread." },
        { name: "likelihood family", effect: "Gaussian here; Multinomial/Bernoulli variants exist for count/binary features (e.g. text)." },
      ],
      metrics: ["Accuracy", "Precision / Recall / F1", "Log-loss"],
      typicalUses: ["Spam filtering", "Document/text classification", "Medical screening", "Real-time classification where speed matters more than squeezing out max accuracy"],
      workedExample: {
        setup: "Class A heights {170,172,168}, Class B heights {180,182,178}, equal priors. Classify x=173.",
        steps: [
          "Both classes: mean=170 (A) / 180 (B); variance = ((0)²+(2)²+(−2)²)/3 = 8/3 ≈ 2.667 for each.",
          "Gaussian pdf coefficient: 1/√(2π·2.667) = 1/4.094 = 0.2443.",
          "P(x|A): exponent = −(173−170)²/(2·2.667) = −9/5.334 = −1.687 → e⁻¹·⁶⁸⁷ ≈ 0.185 → pdf_A = 0.2443×0.185 ≈ 0.0452.",
          "P(x|B): exponent = −(173−180)²/(2·2.667) = −49/5.334 = −9.187 → e⁻⁹·¹⁸⁷ ≈ 1.02×10⁻⁴ → pdf_B ≈ 0.2443×1.02×10⁻⁴ ≈ 2.50×10⁻⁵.",
          "Posterior ∝ prior × likelihood (prior 0.5 cancels in the ratio): unnormalized A=0.0452×0.5=0.0226, B=2.50×10⁻⁵×0.5=1.25×10⁻⁵.",
          "Normalize: P(A|x) = 0.0226/(0.0226+0.0000125) ≈ 0.9994, P(B|x) ≈ 0.0006.",
        ],
        result: "Predict class A (P≈99.9%) — x=173 is far closer to A's mean (170) than B's (180) relative to their shared spread",
      },
    },
    mount,
  });
})();
