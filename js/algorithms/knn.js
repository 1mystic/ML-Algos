(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];
  const N_CLASSES = 3;

  function classify(pt, points, k) {
    if (!points.length) return { label: 0, votes: [] };
    const withD = points.map((p) => ({ p, d: MLU.dist2(pt, p) })).sort((a, b) => a.d - b.d).slice(0, Math.min(k, points.length));
    const counts = new Array(N_CLASSES).fill(0);
    for (const { p } of withD) counts[p.label]++;
    let best = 0;
    for (let i = 1; i < N_CLASSES; i++) if (counts[i] > counts[best]) best = i;
    return { label: best, votes: counts };
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">
              ${[0, 1, 2].map((i) => `<span class="legend-item"><span class="swatch" style="background:${MLU.palette[i]}"></span>class ${i}</span>`).join("")}
            </div>
            <div class="btn-row"><button id="knn-regen">regenerate data</button><button id="knn-clear">clear</button></div>
          </div>
          <svg id="knn-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">pick a class below, then click the plot to add points · move the mouse to preview a live classification</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>add points as</h3>
            <div class="btn-row" id="knn-class-buttons">
              ${[0, 1, 2].map((i) => `<button data-c="${i}" style="border-color:${MLU.palette[i]}">${i}</button>`).join("")}
            </div>
          </div>
          <div class="control-card">
            <h3>k = <span class="val" id="knn-k-val">5</span></h3>
            <input type="range" id="knn-k" min="1" max="25" step="1" value="5" />
          </div>
          <div class="control-card">
            <h3>live query</h3>
            <div class="readout" id="knn-readout">move the mouse over the plot</div>
            <div class="note">Background shading = majority vote of the k nearest neighbors at every grid cell, recomputed as you add points or change k — the same brute-force vote as <code>mla/knn.py</code>.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#knn-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeBlobs({ n: 60, clusters: 3 }).map((p) => ({ x: p.x, y: p.y, label: p.label }));
    let currentClass = 0;
    const bgG = svg.append("g");
    const ptsG = svg.append("g");
    const cursor = svg.append("circle").attr("r", 7).attr("fill", "none").attr("stroke", "var(--text)").attr("stroke-width", 1.5).style("opacity", 0);

    function k() { return +document.getElementById("knn-k").value; }

    function render() {
      document.getElementById("knn-k-val").textContent = k();
      const cell = 14;
      const cells = [];
      for (let px = PAD; px < W - PAD; px += cell)
        for (let py = PAD; py < H - PAD; py += cell) {
          const xv = x.invert(px + cell / 2), yv = y.invert(py + cell / 2);
          const { label } = classify({ x: xv, y: yv }, points, k());
          cells.push({ px, py, label });
        }
      const rects = bgG.selectAll("rect").data(cells);
      rects.enter().append("rect").attr("width", cell).attr("height", cell)
        .merge(rects)
        .attr("x", (d) => d.px).attr("y", (d) => d.py)
        .attr("fill", (d) => MLU.palette[d.label]).attr("opacity", points.length ? 0.16 : 0);
      rects.exit().remove();

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
      points.push({ x: x.invert(px), y: y.invert(py), label: currentClass });
      render();
    });
    svg.on("mousemove", (event) => {
      const [px, py] = d3.pointer(event);
      if (px < PAD || px > W - PAD || py < PAD || py > H - PAD) { cursor.style("opacity", 0); return; }
      const pt = { x: x.invert(px), y: y.invert(py) };
      const { label, votes } = classify(pt, points, k());
      cursor.attr("cx", px).attr("cy", py).attr("stroke", MLU.palette[label]).style("opacity", points.length ? 1 : 0);
      if (points.length) {
        document.getElementById("knn-readout").innerHTML =
          `query → class <b style="color:${MLU.palette[label]}">${label}</b><br>votes: ${votes.map((v, i) => `<span style="color:${MLU.palette[i]}">${v}</span>`).join(" · ")}`;
      }
    });
    document.getElementById("knn-class-buttons").addEventListener("click", (e) => {
      const btn = e.target.closest("button"); if (!btn) return;
      currentClass = +btn.dataset.c;
      document.querySelectorAll("#knn-class-buttons button").forEach((b) => b.classList.toggle("primary", +b.dataset.c === currentClass));
    });
    document.querySelector("#knn-class-buttons button").classList.add("primary");
    document.getElementById("knn-k").addEventListener("input", render);
    document.getElementById("knn-regen").addEventListener("click", () => {
      points = MLU.makeBlobs({ n: 60, clusters: 3 }).map((p) => ({ x: p.x, y: p.y, label: p.label }));
      render();
    });
    document.getElementById("knn-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "knn",
    name: "K-Nearest Neighbors",
    category: "Supervised — Classification",
    tagline: "brute-force vote, live query",
    description: "Classifies a query point by majority vote among its k closest training points. Move the mouse to query live; the shaded background is that same vote evaluated on a grid.",
    sourceFile: "mla/knn.py",
    info: {
      type: "Supervised — Classification (a regression variant also exists). Non-parametric, instance-based ('lazy') learning.",
      scenario: "Irregular / non-linear decision boundaries where enough labeled data exists near any likely query point, and a training phase isn't wanted or needed.",
      inputs: "A set of labeled training points {(xᵢ, yᵢ)}, a distance metric, and a query point x.",
      decisionFunction: {
        text: "ŷ(x) = majority label among the k training points closest to x",
        mechanism: "At prediction time, compute the distance from x to every training point, sort, keep the k smallest, and vote — there are no trained parameters at all, just stored data.",
      },
      lossFunction: {
        text: "No training-time objective — KNN never fits parameters.",
        mechanism: "There's nothing to minimize during 'training' (which is just storing the data); the only real design choices, k and the distance metric, are instead chosen to minimize held-out validation error.",
        plot: { fn: (k) => 0.02 * (k - 8) ** 2 + 0.05, domain: [1, 25], yDomain: [0, 1.3], color: "var(--accent)", caption: "typical validation-error curve vs k: small k → high variance (fits noise), large k → high bias (over-smooths)" },
      },
      output: "A predicted class label for the query point, plus the vote breakdown as an informal confidence.",
      parameters: [
        { name: "k", effect: "Neighborhood size. Small k → jagged, high-variance boundary; large k → smoother, higher-bias boundary." },
        { name: "distance metric", effect: "Defines 'closeness' (Euclidean here). Different metrics suit different feature geometries/scales." },
        { name: "vote weighting", effect: "Uniform vs distance-weighted votes — weighting lets very close neighbors count more than borderline ones." },
      ],
      metrics: ["Accuracy", "Confusion matrix", "F1-score", "MSE (for the regression variant)"],
      typicalUses: ["Recommendation systems", "Anomaly detection", "Handwriting / image recognition baselines", "Any modest-sized dataset where local similarity is meaningful"],
      workedExample: {
        setup: "Query point (3,2), k=3, training set A(1,1,red) B(2,1,red) C(4,3,blue) D(5,4,blue) E(1,2,red).",
        steps: [
          "Squared distances to (3,2): A=(2²+1²)=5, B=(1²+1²)=2, C=(1²+1²)=2, D=(2²+2²)=8, E=(2²+0²)=4.",
          "Sorted by distance: B (2, red), C (2, blue), E (4, red), A (5, red), D (8, blue).",
          "Take the k=3 nearest: B(red), C(blue), E(red).",
          "Vote count: red=2, blue=1.",
        ],
        result: "Majority vote → predicted class = red",
      },
    },
    mount,
  });
})();
