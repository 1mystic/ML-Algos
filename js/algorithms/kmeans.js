(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint" style="font-size:11.5px">iteration <b id="km-iter">0</b></span></div>
            <div class="btn-row">
              <button id="km-step" class="primary">step</button>
              <button id="km-run">run to convergence</button>
              <button id="km-reset-c">reset centroids</button>
              <button id="km-regen">regenerate data</button>
              <button id="km-clear">clear</button>
            </div>
          </div>
          <svg id="km-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">click to add data points · each "step" = one Lloyd iteration (reassign, then re-average)</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>k = <span class="val" id="km-k-val">3</span></h3>
            <input type="range" id="km-k" min="1" max="7" step="1" value="3" />
          </div>
          <div class="control-card">
            <h3>state</h3>
            <div class="readout" id="km-readout">–</div>
            <div class="note">Lloyd's algorithm: assign each point to its closest centroid, then move each centroid to the mean of its assigned points, repeat — the same loop as <code>mla/kmeans.py</code>.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#km-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeBlobs({ n: 70, clusters: 4 }).map((p) => ({ x: p.x, y: p.y, c: 0 }));
    let centroids = [];
    let iter = 0;

    function k() { return +document.getElementById("km-k").value; }
    function initCentroids() {
      const shuffled = MLU.shuffle(points);
      centroids = Array.from({ length: k() }, (_, i) => shuffled[i % Math.max(shuffled.length, 1)]
        ? { x: shuffled[i % shuffled.length].x, y: shuffled[i % shuffled.length].y }
        : { x: MLU.randRange(...DOMAIN), y: MLU.randRange(...DOMAIN) });
      iter = 0;
    }
    initCentroids();

    function assign() {
      let moved = false;
      for (const p of points) {
        let best = 0, bestD = Infinity;
        for (let i = 0; i < centroids.length; i++) {
          const d = MLU.dist2(p, centroids[i]);
          if (d < bestD) { bestD = d; best = i; }
        }
        if (p.c !== best) moved = true;
        p.c = best;
      }
      return moved;
    }
    function update() {
      let maxShift = 0;
      for (let i = 0; i < centroids.length; i++) {
        const members = points.filter((p) => p.c === i);
        if (!members.length) continue;
        const nx = MLU.mean(members.map((p) => p.x)), ny = MLU.mean(members.map((p) => p.y));
        maxShift = Math.max(maxShift, MLU.dist({ x: nx, y: ny }, centroids[i]));
        centroids[i].x = nx; centroids[i].y = ny;
      }
      return maxShift;
    }
    function inertia() {
      return points.reduce((s, p) => s + (centroids[p.c] ? MLU.dist2(p, centroids[p.c]) : 0), 0);
    }

    const ptsG = svg.append("g");
    const cenG = svg.append("g");

    function render() {
      document.getElementById("km-k-val").textContent = k();
      document.getElementById("km-iter").textContent = iter;

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle").attr("r", 5).attr("stroke", "var(--bg)").attr("stroke-width", 1).style("cursor", "grab")
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y))
        .attr("fill", (d) => MLU.palette[d.c % MLU.palette.length])
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); render(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          render();
        }));
      sel.exit().remove();

      const csel = cenG.selectAll("g.centroid").data(centroids);
      const enter = csel.enter().append("g").attr("class", "centroid");
      enter.append("circle").attr("r", 10).attr("fill", "none").attr("stroke-width", 2.5);
      enter.append("circle").attr("r", 3).attr("stroke", "none");
      const merged = enter.merge(csel);
      merged.attr("transform", (d) => `translate(${x(d.x)},${y(d.y)})`);
      merged.select("circle:nth-child(1)").attr("stroke", (d, i) => MLU.palette[i % MLU.palette.length]);
      merged.select("circle:nth-child(2)").attr("fill", (d, i) => MLU.palette[i % MLU.palette.length]);
      csel.exit().remove();

      document.getElementById("km-readout").innerHTML =
        `points: <b>${points.length}</b><br>k: <b>${centroids.length}</b><br>inertia (SSE): <b class="num">${points.length ? inertia().toFixed(1) : "–"}</b>`;
    }

    svg.on("click", (event) => {
      if (event.target.tagName === "circle") return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py), c: 0 });
      render();
    });
    document.getElementById("km-k").addEventListener("input", () => { initCentroids(); render(); });
    document.getElementById("km-step").addEventListener("click", () => { assign(); update(); iter++; render(); });
    document.getElementById("km-run").addEventListener("click", () => {
      for (let i = 0; i < 100; i++) { const moved = assign(); const shift = update(); iter++; if (!moved && shift < 1e-3) break; }
      render();
    });
    document.getElementById("km-reset-c").addEventListener("click", () => { initCentroids(); render(); });
    document.getElementById("km-regen").addEventListener("click", () => {
      points = MLU.makeBlobs({ n: 70, clusters: 4 }).map((p) => ({ x: p.x, y: p.y, c: 0 }));
      initCentroids(); render();
    });
    document.getElementById("km-clear").addEventListener("click", () => { points = []; initCentroids(); render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "kmeans",
    name: "K-Means",
    category: "Unsupervised — Clustering",
    tagline: "Lloyd's algorithm, step-through",
    description: "Hard clustering via alternating assignment and centroid-averaging. Step through iterations one at a time to see exactly how the centroids converge.",
    sourceFile: "mla/kmeans.py",
    info: {
      type: "Unsupervised — Clustering. Centroid-based, hard assignment.",
      scenario: "Partitioning unlabeled data into k compact, roughly spherical groups — customer segmentation, color quantization, or as an unsupervised pre-processing step.",
      inputs: "Unlabeled points {xᵢ} and a chosen number of clusters k.",
      decisionFunction: {
        text: "assign(x) = argmin_j ‖x − cⱼ‖²",
        mechanism: "Lloyd's algorithm alternates: (1) assign every point to its nearest centroid, (2) move each centroid to the mean of its assigned points — repeated until assignments stop changing.",
      },
      lossFunction: {
        text: "J = Σᵢ Σ_{x∈Cᵢ} ‖x − cᵢ‖²  (inertia / within-cluster sum of squares)",
        mechanism: "Each Lloyd step provably never increases J, so it converges — but only to a local minimum, which is why the result depends on how centroids were initialized.",
        plot: { fn: (k) => 10 / (k + 0.6) + 0.3, domain: [1, 8], color: "var(--accent)", caption: "typical inertia-vs-k 'elbow' curve — pick k near where it stops dropping sharply" },
      },
      output: "A cluster index for every point, plus the k centroid coordinates.",
      parameters: [
        { name: "k", effect: "Number of clusters — the main lever, usually chosen via the elbow method or silhouette score." },
        { name: "initialization", effect: "Random vs k-means++ starting centroids — strongly affects which local minimum is found." },
        { name: "restarts", effect: "Running Lloyd's algorithm from several initializations and keeping the lowest-inertia result." },
      ],
      metrics: ["Inertia / WCSS", "Silhouette score", "Davies–Bouldin index", "Adjusted Rand Index / NMI (if ground-truth labels exist)"],
      typicalUses: ["Customer/market segmentation", "Image color quantization", "Document/topic clustering", "Vector quantization"],
      workedExample: {
        setup: "Points (1,1), (1,2), (4,4), (5,5); k=2; initial centroids C1=(1,1), C2=(5,5).",
        steps: [
          "Assign (1,1): dist to C1=0, to C2=√32≈5.66 → C1.",
          "Assign (1,2): dist to C1=1, to C2=√25=5 → C1.",
          "Assign (4,4): dist to C1=√18≈4.24, to C2=√2≈1.41 → C2.",
          "Assign (5,5): dist to C1=√32≈5.66, to C2=0 → C2.",
          "Update C1 = mean of (1,1),(1,2) = (1, 1.5). Update C2 = mean of (4,4),(5,5) = (4.5, 4.5).",
        ],
        result: "After one iteration: C1=(1, 1.5), C2=(4.5, 4.5) — already close to the final converged centroids",
      },
    },
    mount,
  });
})();
