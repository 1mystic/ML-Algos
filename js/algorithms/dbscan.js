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
    info: {
      type: "Unsupervised - Clustering. Density-based, does not require choosing the number of clusters in advance.",
      scenario: "Clusters have arbitrary (non-spherical) shapes, contain noise/outliers that shouldn't be forced into any cluster, or you don't want to pre-specify k - the classic limitation of K-Means this addresses.",
      inputs: "Unlabeled points, a neighborhood radius ε, and a minimum neighborhood size minPts.",
      intuition: {
        definition: "Clusters are <b>dense regions separated by sparse ones</b>. A point in a crowded neighbourhood is a core point; core points that are close together chain into a cluster of any shape; points in no dense region are labelled noise rather than forced into a cluster.",
        steps: [
          "Count how many points lie within ε of each point.",
          "Points with at least minPts neighbours are core points.",
          "Chain overlapping core neighbourhoods into clusters.",
          "Anything never reached is noise, not a cluster member.",
        ],
        applications: [
          "GPS and geospatial clustering, where density is literal",
          "Outlier and fraud detection, since noise is a first-class output",
          "Image segmentation of irregular regions",
          "Grouping sensor readings with unknown group count",
          "Any data with crescent, ring, or filament-shaped clusters",
        ],
      },
      math: [
        { title: "ε-neighbourhood", formula: "N_ε(x) = { q ∈ D : dist(x, q) ≤ ε }", note: "The set of points within radius ε, including x itself in the usual convention." },
        { title: "Core point", formula: "x is core ⟺ |N_ε(x)| ≥ minPts", note: "The density test. Everything else in the algorithm is defined in terms of core points." },
        { title: "Directly density-reachable", formula: "q is directly reachable from x ⟺ x is core and q ∈ N_ε(x)", note: "Note the asymmetry: a border point is reachable from a core point but not the reverse, because it cannot extend the cluster." },
        { title: "Density-reachable", formula: "chain x = p₁, p₂, …, pₙ = q where each pᵢ₊₁ is directly reachable from pᵢ", note: "Transitive closure over core points. This chaining is what lets a cluster follow an arbitrarily curved shape." },
        { title: "Density-connected", formula: "p and q are connected if some core o reaches both", note: "The symmetric relation that actually defines a cluster. A cluster is a maximal density-connected set." },
        { title: "Point taxonomy", formula: "core: ≥ minPts neighbours | border: in a core's neighbourhood but not core | noise: neither", note: "Only three categories. Border points join whichever cluster reached them first, which is the algorithm's one order-dependent quirk." },
      ],
      pipeline: [
        { label: "Pick unvisited x", note: "any order" },
        { label: "Count N_ε(x)", note: "range query" },
        { label: "Core?", note: "≥ minPts" },
        { label: "Expand chain", note: "absorb reachable" },
        { label: "Labels or noise", note: "k discovered", accent: "green" },
      ],
      decisionFunction: {
        text: "cluster(x) = the group reached by chaining together overlapping ε-neighborhoods of core points; unreachable points → noise (label −1)",
        mechanism: "A point is a 'core point' if at least minPts points (including itself) lie within ε of it. Clusters grow by repeatedly absorbing any point within ε of an already-included core point ('density-reachability'); points that are never absorbed this way are labeled noise.",
        plot: { fn: (r) => 7 / (1 + Math.exp(-(r - 1.5) * 3)), domain: [0, 3], yDomain: [0, 8], color: "var(--accent)", caption: "illustrative: number of points within radius r of a sample point - ε is chosen near where this crosses minPts" },
      },
      lossFunction: {
        text: "No explicit objective is minimized - a direct, deterministic neighborhood-expansion procedure.",
        mechanism: "Like KNN, DBSCAN has no training loss; ε and minPts are instead chosen by inspecting the resulting clusters/noise ratio, or via a k-distance plot heuristic.",
      },
      optimization: [
        { title: "Cost", formula: "O(m²) naive, O(m log m) with a spatial index", note: "The whole algorithm is a sequence of range queries. A KD-tree or ball-tree makes each one logarithmic, but only while dimensionality stays low." },
        { title: "Choosing ε from the data", formula: "sort the distance to each point's k-th nearest neighbour, look for the knee", note: "The k-distance plot with k = minPts. The elbow marks where points stop being in dense regions, which is a principled starting value for ε." },
        { title: "Choosing minPts", formula: "minPts ≥ p + 1, commonly 2p", note: "Must exceed the dimensionality or almost everything looks dense. A value of 4 is the classic default for 2D data." },
        { title: "Order dependence", formula: "border points join the first cluster that reaches them", note: "Core points and noise labels are fully deterministic; only border-point assignment can vary with scan order. In practice this affects very few points." },
        { title: "HDBSCAN", formula: "vary ε continuously, extract the most stable clusters", note: "Removes the hardest hyperparameter by building a hierarchy over all ε values and selecting clusters that persist longest." },
      ],
      output: "A cluster label (or noise, −1) for every point - the number of clusters is discovered automatically, not specified up front.",
      assumptions: [
        { name: "Clusters have similar density", why: "A single global ε cannot describe both a dense cluster and a sparse one. One will merge, the other will fragment.", check: "If the k-distance plot has several knees, use HDBSCAN or OPTICS instead." },
        { name: "Clusters are separated by sparse regions", why: "The definition of a cluster boundary is a drop in density. Touching clusters of equal density will merge.", check: "Inspect whether the merged clusters have a genuine density gap." },
        { name: "Features are scaled", why: "ε is a single radius applied across all dimensions.", check: "Standardize before clustering. ε is meaningless otherwise." },
        { name: "Low to moderate dimensionality", why: "Distance concentration makes all neighbourhoods look alike, and spatial indexes degrade to linear scans.", check: "Above roughly 10 to 20 features, reduce with PCA or UMAP first." },
        { name: "Meaningful distance metric", why: "Density is defined entirely through the metric.", check: "Use cosine for text embeddings, haversine for geographic coordinates." },
      ],
      hyperparameters: [
        { name: "ε (eps)", range: "data-dependent", increasing: "Neighbourhoods grow, clusters merge, noise shrinks. Too large and everything becomes one cluster.", strategy: "Read it off the knee of the k-distance plot with k = minPts. This is the parameter that matters most and it cannot be guessed." },
        { name: "minPts", range: "3 - 50", increasing: "Denser regions required, more points labelled noise, more robust to spurious clusters.", strategy: "Start at 2·p for p-dimensional data, minimum p+1. Raise it if you get many small dubious clusters." },
        { name: "metric", range: "euclidean / manhattan / cosine / haversine", increasing: "Not applicable", strategy: "Match the data. Haversine for lat/long, cosine for embeddings." },
        { name: "algorithm", range: "auto / kd_tree / ball_tree / brute", increasing: "Not applicable", strategy: "Leave on auto. It falls back to brute force in high dimensions, which is correct." },
        { name: "min_samples (sklearn)", range: "same as minPts", increasing: "Same effect as minPts.", strategy: "Note that sklearn counts the point itself, so the semantics match the classic definition." },
      ],
      metrics: ["Silhouette score (computed excluding noise)", "Number of clusters found", "Noise ratio", "Adjusted Rand Index (when ground truth exists)"],
      typicalUses: ["Geospatial/GPS clustering", "Anomaly/outlier detection (the noise points ARE the anomalies)", "Clusters of arbitrary, non-convex shape", "Image segmentation"],
      diagnostics: [
        "Always plot the k-distance curve before choosing ε. Guessing wastes far more time than the plot costs.",
        "Track the noise ratio. Above roughly 30% usually means ε is too small; near 0% often means it is too large.",
        "If you get one giant cluster plus a few tiny ones, ε has bridged genuinely separate groups.",
        "Compute silhouette on non-noise points only. Including noise makes the score meaningless.",
        "If results vary noticeably between runs, that is border-point order dependence, which is expected and usually harmless.",
      ],
      advantages: [
        "The number of clusters is discovered rather than specified.",
        "Finds arbitrarily shaped clusters, including rings and crescents that centroid methods cannot represent.",
        "Outliers are an explicit output rather than being forced into a cluster.",
        "Robust to outliers by construction, since noise never influences cluster shape.",
        "Only two hyperparameters, and one of them can be read off a plot.",
        "Core-point and noise labels are deterministic regardless of scan order.",
      ],
      limitations: [
        { name: "Cannot handle varying density", note: "one global ε cannot fit both dense and sparse clusters", fix: "HDBSCAN or OPTICS." },
        { name: "Very sensitive to ε", note: "small changes flip between one cluster and all noise", fix: "choose it from the k-distance knee rather than by trial and error." },
        { name: "Struggles in high dimensions", note: "distance concentration destroys the density signal", fix: "PCA or UMAP before clustering." },
        { name: "Border points are order-dependent", note: "they join whichever cluster reaches them first", fix: "usually negligible; DBSCAN* excludes border points entirely." },
        { name: "No predict for new points", note: "clusters are defined by the training data's density", fix: "fit a classifier on the labels, or use HDBSCAN's approximate_predict." },
        { name: "Requires scaling", note: "ε is a single radius across all features", fix: "standardize first." },
      ],
      alternatives: [
        { name: "HDBSCAN", when: "Cluster densities vary, or you want to avoid choosing ε entirely. Usually the better default today." },
        { name: "OPTICS", when: "You want an ordering that reveals structure across all density levels." },
        { name: "K-means", when: "Clusters are spherical, similarly sized, and k is known. Far faster at scale." },
        { name: "Spectral clustering", when: "Clusters are connected and non-convex but density is uniform." },
      ],
      pitfalls: [
        { problem: "Everything is labelled noise", solution: "ε is too small or minPts too large. Check the k-distance plot." },
        { problem: "Everything is one cluster", solution: "ε is too large and has bridged the gaps. Reduce it." },
        { problem: "Results are nonsense despite reasonable ε", solution: "Features are unscaled, so ε means different things per dimension. Standardize." },
        { problem: "Dense clusters found, sparse ones lost", solution: "Inherent to a single global ε. Move to HDBSCAN." },
        { problem: "Very slow on a large dataset", solution: "The spatial index has degraded. Reduce dimensionality, or sample." },
        { problem: "No predict method available", solution: "DBSCAN is transductive. Train a classifier on its labels to score new points." },
      ],
      quickRef: [
        { name: "Neighbourhood", formula: "N_ε(x) = {q : d(x,q) ≤ ε}" },
        { name: "Core point", formula: "|N_ε(x)| ≥ minPts" },
        { name: "Border point", formula: "non-core, but in some core's N_ε" },
        { name: "Noise", formula: "neither core nor border, label −1" },
        { name: "Cluster", formula: "maximal density-connected set" },
        { name: "minPts rule", formula: "≥ p+1, typically 2p" },
        { name: "ε heuristic", formula: "knee of the k-distance plot" },
        { name: "Cost", formula: "O(m log m) with an index" },
      ],
      code: `import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)   # eps is meaningless unscaled
min_pts = 2 * Xs.shape[1]                # rule of thumb: 2 x dimensions

# Read eps off the knee of the sorted k-distance curve rather than guessing.
dists, _ = NearestNeighbors(n_neighbors=min_pts).fit(Xs).kneighbors(Xs)
k_dist = np.sort(dists[:, -1])
# plt.plot(k_dist)  -> the elbow y-value is your eps

db = DBSCAN(eps=0.5, min_samples=min_pts).fit(Xs)
labels = db.labels_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_ratio = (labels == -1).mean()
print(n_clusters, round(noise_ratio, 3))

# Varying density? HDBSCAN removes the eps choice entirely.
# import hdbscan; hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(Xs)`,
      whyChain: [
        { q: "What is DBSCAN's main advantage over k-means?", a: "Two things. It discovers the number of clusters instead of requiring it, and it can find clusters of any shape because clusters are defined by density connectivity rather than distance to a centre. It also isolates outliers instead of assigning them." },
        { q: "What is the difference between a core and a border point?", a: "A core point has at least minPts neighbours within ε and can extend a cluster. A border point is inside some core point's neighbourhood but is not itself dense enough, so it joins the cluster but cannot grow it further. That asymmetry is what stops clusters leaking through sparse regions." },
        { q: "Why is DBSCAN unable to handle varying density?", a: "ε and minPts together define one absolute density threshold applied everywhere. If you set it for the sparse cluster, the dense clusters merge; if you set it for the dense ones, the sparse cluster becomes noise. There is no single value that works for both." },
        { q: "How do you actually choose ε?", a: "Compute each point's distance to its minPts-th nearest neighbour, sort those distances, and plot them. Points inside clusters have small values and noise has large ones, so the curve has a knee. The y-value at the knee is a well-motivated ε." },
        { q: "Why must minPts exceed the number of dimensions?", a: "In p dimensions you need at least p+1 points to define a volume. With fewer, neighbourhoods are degenerate and nearly every point qualifies as core, so the density test stops discriminating." },
        { q: "Is DBSCAN deterministic?", a: "Core points and noise are fully deterministic. Only border points can vary, because they join whichever cluster reaches them first, and that depends on scan order. The variation is usually confined to a handful of points." },
        { q: "Why does DBSCAN break down in high dimensions?", a: "Distances concentrate, so every point's ε-neighbourhood contains roughly the same count and the density contrast that the algorithm relies on disappears. Spatial indexes also stop helping, pushing you back to O(m²)." },
        { q: "Why is there no predict method?", a: "A cluster is defined as a density-connected region of the training data. A new point has no defined membership without recomputing connectivity. HDBSCAN offers an approximate_predict; otherwise train a classifier on the cluster labels." },
      ],
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
