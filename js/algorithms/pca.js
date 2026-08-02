(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[0]}"></span>data</span>
              <span class="legend-item"><span class="swatch" style="background:var(--accent)"></span>PC1</span>
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[4]}"></span>PC2</span>
            </div>
            <div class="btn-row">
              <button id="pca-regen">regenerate data</button>
              <button id="pca-clear">clear</button>
            </div>
          </div>
          <svg id="pca-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">click to add points · drag to move them · arrows are the principal axes, scaled by &radic;eigenvalue</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>options</h3>
            <div class="field"><label><input type="checkbox" id="pca-proj" checked style="width:auto;margin-right:6px" />show projections onto PC1</label></div>
          </div>
          <div class="control-card">
            <h3>variance</h3>
            <div class="readout" id="pca-readout">–</div>
            <div class="note">Eigendecomposition of the centered covariance matrix (via Jacobi rotation), same objective as <code>mla/pca.py</code>. PC1 is the direction of maximum variance; projecting onto it is the best 1D lossy compression of this data.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#pca-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    function genData() {
      const angle = MLU.randRange(0, Math.PI);
      const n = 60, pts = [];
      const cx = MLU.randRange(-2, 2), cy = MLU.randRange(-2, 2);
      for (let i = 0; i < n; i++) {
        const a = MLU.randn() * 4, b = MLU.randn() * 1.1;
        pts.push({
          x: cx + a * Math.cos(angle) - b * Math.sin(angle),
          y: cy + a * Math.sin(angle) + b * Math.cos(angle),
        });
      }
      return pts;
    }
    let points = genData();

    const projG = svg.append("g");
    const axesG = svg.append("g");
    const ptsG = svg.append("g");

    function render() {
      let mu = { x: 0, y: 0 }, eig = { values: [1, 0], vectors: [[1, 0], [0, 1]] };
      if (points.length >= 2) {
        mu = MLU.meanVec(points.map((p) => [p.x, p.y]));
        mu = { x: mu[0], y: mu[1] };
        const cov = MLU.covMatrix(points.map((p) => [p.x, p.y]), [mu.x, mu.y]);
        eig = MLU.jacobiEigen(cov);
      }
      const showProj = document.getElementById("pca-proj").checked;

      if (showProj && points.length >= 2) {
        const [v0, v1] = eig.vectors[0];
        const projLines = points.map((p) => {
          const dx = p.x - mu.x, dy = p.y - mu.y;
          const t = dx * v0 + dy * v1;
          return { x1: p.x, y1: p.y, x2: mu.x + t * v0, y2: mu.y + t * v1 };
        });
        const sel = projG.selectAll("line").data(projLines);
        sel.enter().append("line").attr("stroke", "var(--text-faint)").attr("stroke-width", 1).attr("stroke-dasharray", "2,2")
          .merge(sel).attr("x1", (d) => x(d.x1)).attr("y1", (d) => y(d.y1)).attr("x2", (d) => x(d.x2)).attr("y2", (d) => y(d.y2));
        sel.exit().remove();
      } else projG.selectAll("line").remove();

      const arrows = points.length >= 2 ? [
        { v: eig.vectors[0], val: eig.values[0], color: "var(--accent)" },
        { v: eig.vectors[1], val: eig.values[1], color: MLU.palette[4] },
      ] : [];
      const asel = axesG.selectAll("line.axis").data(arrows);
      asel.enter().append("line").attr("class", "axis").attr("stroke-width", 2.5)
        .merge(asel)
        .attr("x1", (d) => x(mu.x - d.v[0] * Math.sqrt(Math.max(d.val, 0))))
        .attr("y1", (d) => y(mu.y - d.v[1] * Math.sqrt(Math.max(d.val, 0))))
        .attr("x2", (d) => x(mu.x + d.v[0] * Math.sqrt(Math.max(d.val, 0))))
        .attr("y2", (d) => y(mu.y + d.v[1] * Math.sqrt(Math.max(d.val, 0))))
        .attr("stroke", (d) => d.color);
      asel.exit().remove();

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle").attr("r", 5).attr("fill", MLU.palette[0]).attr("stroke", "var(--bg)").attr("stroke-width", 1).style("cursor", "grab")
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y))
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); render(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          render();
        }));
      sel.exit().remove();

      const total = eig.values[0] + eig.values[1] || 1;
      document.getElementById("pca-readout").innerHTML = points.length >= 2
        ? `points: <b>${points.length}</b><br>PC1 explains: <b class="num">${((eig.values[0] / total) * 100).toFixed(1)}%</b><br>PC2 explains: <b class="num">${((eig.values[1] / total) * 100).toFixed(1)}%</b>`
        : "add at least 2 points";
    }

    svg.on("click", (event) => {
      if (event.target.tagName === "circle") return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py) });
      render();
    });
    document.getElementById("pca-proj").addEventListener("change", render);
    document.getElementById("pca-regen").addEventListener("click", () => { points = genData(); render(); });
    document.getElementById("pca-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "pca",
    name: "Principal Component Analysis",
    category: "Unsupervised - Dimensionality Reduction",
    tagline: "eigendecomposition of the covariance",
    description: "Finds the orthogonal directions of maximum variance in the data via eigendecomposition of the covariance matrix, and shows what's lost when you project onto just the first component.",
    sourceFile: "mla/pca.py",
    info: {
      type: "Unsupervised - Dimensionality reduction / linear transformation.",
      scenario: "Compressing correlated features into fewer uncorrelated components, visualizing high-dimensional data, denoising, or as pre-processing before another model.",
      inputs: "A matrix of numeric features X, typically centered (and often standardized) first.",
      decisionFunction: {
        text: "project(x) = Vᵀ(x − μ), using the top eigenvectors V of the covariance matrix",
        mechanism: "Eigendecompose the centered covariance matrix: the eigenvector with the largest eigenvalue is the direction of maximum variance (PC1); the next-largest orthogonal direction is PC2, and so on.",
      },
      lossFunction: {
        text: "min Σᵢ ‖xᵢ − reconstruct(project(xᵢ))‖²  (reconstruction error)",
        mechanism: "Among all rank-k linear projections, PCA's is provably the one minimizing this reconstruction error - solved exactly by eigendecomposition, with no optimization loop.",
        plot: { fn: (k) => Math.exp(-0.6 * k), domain: [1, 6], color: "var(--accent)", caption: "typical 'scree plot' shape: variance explained per component usually drops off fast" },
      },
      output: "A lower-dimensional projected representation, plus the principal axes (eigenvectors) and their eigenvalues (variance explained).",
      parameters: [
        { name: "number of components", effect: "How many axes to keep. Fewer = more compression / more information lost." },
        { name: "standardize features?", effect: "Important when features are on very different scales - otherwise high-variance features dominate the principal axes for reasons unrelated to importance." },
      ],
      metrics: ["Cumulative % variance explained", "Reconstruction error (MSE)"],
      typicalUses: ["Visualizing high-dimensional data in 2D/3D", "Noise reduction", "Decorrelating features before regression/clustering", "Data compression"],
      workedExample: {
        setup: "Points (1,2), (3,4), (5,6) - perfectly correlated (they lie exactly on a line). Find PC1.",
        steps: [
          "Mean = (3, 4). Deviations: (−2,−2), (0,0), (2,2).",
          "Sxx = (4+0+4)/3 = 2.667, Syy = (4+0+4)/3 = 2.667, Sxy = (4+0+4)/3 = 2.667.",
          "Covariance matrix = [[2.667, 2.667], [2.667, 2.667]] - trace = 5.333, determinant = 2.667×2.667 − 2.667×2.667 = 0.",
          "Determinant 0 means one eigenvalue is 0; since trace = sum of eigenvalues, the other eigenvalue is 5.333.",
          "The eigenvector for λ=5.333 is the direction (1,1)/√2 - a 45° line.",
        ],
        result: "PC1 points along (1,1)/√2 and explains 100% of the variance - exactly the line y=x+1 the data lies on, with 0% left for PC2",
      },
    },
    mount,
  });
})();
