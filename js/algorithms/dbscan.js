(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];
  const NOISE = -1, UNVISITED = -2;

  function regionQuery(points, i, eps2) {
    const neigh = [];
    for (let j = 0; j < points.length; j++) if (MLU.dist2(points[i], points[j]) <= eps2) neigh.push(j);
    return neigh; // includes i itself
  }
  function runDBSCAN(points, eps, minPts) {
    const n = points.length;
    const labels = new Array(n).fill(UNVISITED);
    const eps2 = eps * eps;
    let cluster = 0;
    for (let i = 0; i < n; i++) {
      if (labels[i] !== UNVISITED) continue;
      const neighbors = regionQuery(points, i, eps2);
      if (neighbors.length < minPts) { labels[i] = NOISE; continue; }
      labels[i] = cluster;
      const seeds = neighbors.filter((j) => j !== i);
      for (let s = 0; s < seeds.length; s++) {
        const j = seeds[s];
        if (labels[j] === NOISE) labels[j] = cluster;
        if (labels[j] !== UNVISITED) continue;
        labels[j] = cluster;
        const jNeighbors = regionQuery(points, j, eps2);
        if (jNeighbors.length >= minPts) for (const k of jNeighbors) if (!seeds.includes(k)) seeds.push(k);
      }
      cluster++;
    }
    return { labels, nClusters: cluster };
  }
  function isCore(points, i, labels, eps, minPts) {
    return regionQuery(points, i, eps * eps).length >= minPts;
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint">clusters found: <b id="db-nclusters">0</b> · noise points: <b id="db-noise">0</b></span></div>
            <div class="btn-row">
              <button id="db-preset-blobs">blob data</button>
              <button id="db-preset-moons">arbitrary-shape data</button>
              <button id="db-clear">clear</button>
            </div>
          </div>
          <svg id="db-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">click to add points · hover a point to preview its &epsilon;-neighborhood · filled = core, ring = border, × = noise</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>&epsilon; (neighborhood radius) <span class="val" id="db-eps-val">1.2</span></h3>
            <input type="range" id="db-eps" min="3" max="40" step="1" value="12" />
          </div>
          <div class="control-card">
            <h3>minPts <span class="val" id="db-minpts-val">4</span></h3>
            <input type="range" id="db-minpts" min="2" max="12" step="1" value="4" />
          </div>
          <div class="control-card">
            <h3>fit</h3>
            <div class="readout" id="db-readout">–</div>
            <div class="note">No k to choose in advance - clusters emerge from density connectivity, and points too sparse to belong anywhere are left as noise rather than forced into a cluster, unlike K-Means.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#db-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    function presetBlobs() {
      const pts = MLU.makeBlobs({ n: 70, clusters: 3, spread: 0.7 }).map((p) => ({ x: p.x, y: p.y }));
      for (let i = 0; i < 8; i++) pts.push({ x: MLU.randRange(...DOMAIN), y: MLU.randRange(...DOMAIN) });
      return pts;
    }
    function presetMoons() {
      const pts = [];
      for (let i = 0; i < 45; i++) { const a = MLU.randRange(0, Math.PI); pts.push({ x: 5 * Math.cos(a) + MLU.randn() * 0.3, y: 5 * Math.sin(a) + MLU.randn() * 0.3 - 1 }); }
      for (let i = 0; i < 45; i++) { const a = MLU.randRange(0, Math.PI); pts.push({ x: 5 * Math.cos(a) + MLU.randn() * 0.3 + 3, y: -5 * Math.sin(a) + MLU.randn() * 0.3 + 2 }); }
      return pts;
    }
    let points = presetBlobs();

    const eyeCircle = svg.append("circle").attr("r", 0).attr("fill", "none").attr("stroke", "var(--text-faint)").attr("stroke-dasharray", "3,3");
    const ptsG = svg.append("g");

    function eps() { return +document.getElementById("db-eps").value / 10; }
    function minPts() { return +document.getElementById("db-minpts").value; }

    function render() {
      document.getElementById("db-eps-val").textContent = eps().toFixed(1);
      document.getElementById("db-minpts-val").textContent = minPts();
      const { labels, nClusters } = runDBSCAN(points, eps(), minPts());
      const noiseCount = labels.filter((l) => l === NOISE).length;
      document.getElementById("db-nclusters").textContent = nClusters;
      document.getElementById("db-noise").textContent = noiseCount;
      document.getElementById("db-readout").innerHTML = `points: <b>${points.length}</b><br>clusters: <b class="num">${nClusters}</b><br>noise: <b class="num">${noiseCount}</b> (${points.length ? ((noiseCount / points.length) * 100).toFixed(0) : 0}%)`;

      const data = points.map((p, i) => ({ ...p, label: labels[i], core: isCore(points, i, labels, eps(), minPts()) }));
      const sel = ptsG.selectAll("g.pt").data(data);
      const enter = sel.enter().append("g").attr("class", "pt");
      enter.append("circle").attr("class", "dot").attr("r", 5.5);
      enter.append("path").attr("class", "noise-x").attr("d", "M-4,-4 L4,4 M4,-4 L-4,4").attr("stroke-width", 1.6);
      const merged = enter.merge(sel);
      merged.attr("transform", (d) => `translate(${x(d.x)},${y(d.y)})`).style("cursor", "grab");
      merged.select(".dot")
        .style("opacity", (d) => (d.label === NOISE ? 0 : 1))
        .attr("fill", (d) => (d.core ? MLU.palette[((d.label % MLU.palette.length) + MLU.palette.length) % MLU.palette.length] : "none"))
        .attr("stroke", (d) => MLU.palette[((d.label % MLU.palette.length) + MLU.palette.length) % MLU.palette.length])
        .attr("stroke-width", 1.6);
      merged.select(".noise-x").style("opacity", (d) => (d.label === NOISE ? 1 : 0)).attr("stroke", "var(--text-faint)");
      merged
        .on("mouseenter", (event, d) => { eyeCircle.attr("cx", x(d.x)).attr("cy", y(d.y)).attr("r", Math.abs(x(eps()) - x(0))).style("opacity", 1); })
        .on("mouseleave", () => eyeCircle.style("opacity", 0))
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); render(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          render();
        }));
      sel.exit().remove();
    }

    svg.on("click", (event) => {
      if (event.target.closest(".pt")) return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py) });
      render();
    });
    document.getElementById("db-eps").addEventListener("input", render);
    document.getElementById("db-minpts").addEventListener("input", render);
    document.getElementById("db-preset-blobs").addEventListener("click", () => { points = presetBlobs(); render(); });
    document.getElementById("db-preset-moons").addEventListener("click", () => { points = presetMoons(); render(); });
    document.getElementById("db-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "dbscan",
    name: "DBSCAN",
    category: "Unsupervised - Clustering",
    tagline: "density-connectivity, no fixed k",
    description: "Density-Based Spatial Clustering: groups points that are densely connected through overlapping neighborhoods, and leaves sparse points labeled as noise instead of forcing them into a cluster.",
    sourceFile: "not in the original repo - added as a commonly-requested density-based clustering method",
    info: {
      type: "Unsupervised - Clustering. Density-based, does not require choosing the number of clusters in advance.",
      scenario: "Clusters have arbitrary (non-spherical) shapes, contain noise/outliers that shouldn't be forced into any cluster, or you don't want to pre-specify k - the classic limitation of K-Means this addresses.",
      inputs: "Unlabeled points, a neighborhood radius ε, and a minimum neighborhood size minPts.",
      decisionFunction: {
        text: "cluster(x) = the group reached by chaining together overlapping ε-neighborhoods of core points; unreachable points → noise (label −1)",
        mechanism: "A point is a 'core point' if at least minPts points (including itself) lie within ε of it. Clusters grow by repeatedly absorbing any point within ε of an already-included core point ('density-reachability'); points that are never absorbed this way are labeled noise.",
        plot: { fn: (r) => 7 / (1 + Math.exp(-(r - 1.5) * 3)), domain: [0, 3], yDomain: [0, 8], color: "var(--accent)", caption: "illustrative: number of points within radius r of a sample point - ε is chosen near where this crosses minPts" },
      },
      lossFunction: {
        text: "No explicit objective is minimized - a direct, deterministic neighborhood-expansion procedure.",
        mechanism: "Like KNN, DBSCAN has no training loss; ε and minPts are instead chosen by inspecting the resulting clusters/noise ratio, or via a k-distance plot heuristic.",
      },
      output: "A cluster label (or noise, −1) for every point - the number of clusters is discovered automatically, not specified up front.",
      parameters: [
        { name: "ε (eps)", effect: "Neighborhood radius. Too small → almost everything is noise. Too large → distinct clusters merge into one." },
        { name: "minPts", effect: "Minimum neighborhood size to count as a core point. Higher values demand denser regions and are more robust to noise, but can miss small real clusters." },
        { name: "distance metric", effect: "Defines the neighborhood shape (Euclidean here)." },
      ],
      metrics: ["Silhouette score (computed excluding noise)", "Number of clusters found", "Noise ratio"],
      typicalUses: ["Geospatial/GPS clustering", "Anomaly/outlier detection (the noise points ARE the anomalies)", "Clusters of arbitrary, non-convex shape", "Image segmentation"],
      workedExample: {
        setup: "Points A(0,0) B(1,0) C(2,0) D(2,1), and an isolated E(8,8). ε=1.5, minPts=2 (neighborhood count includes the point itself).",
        steps: [
          "Neighbors of A within ε=1.5 (incl. self): {A, B} (dist 1) → size 2 ≥ minPts → A is core.",
          "Neighbors of B: {A (1), B, C (1), D (√2≈1.41)} → size 4 ≥ minPts → B is core.",
          "Neighbors of C: {B (1), C, D (1)} → size 3 ≥ minPts → C is core. Neighbors of D: {C (1), B (1.41), D} → size 3 ≥ minPts → D is core.",
          "Neighbors of E: {E} only (everything else is far away) → size 1 < minPts → E is noise.",
          "Expand from A (core): add B. B is core → add its neighbors C, D. C and D are core but their neighbors (B, C, D) are already included.",
        ],
        result: "Cluster 0 = {A, B, C, D}; E is labeled noise (−1) - found without ever specifying a number of clusters",
      },
    },
    mount,
  });
})();
