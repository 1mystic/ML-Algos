(() => {
  const { ui, col, fmt } = Lab;
  const UNVIS = -2, NOISE = -1;

  function buildFrames(pts, eps, minPts) {
    const n = pts.length;
    if (!n) return [{ phase: "idle", title: "No data", text: "", st: { lab: [], core: [] } }];
    const nb = pts.map((p) => { const r = []; for (let j = 0; j < n; j++) if (Math.hypot(p.x - pts[j].x, p.y - pts[j].y) <= eps) r.push(j); return r; });
    const core = nb.map((r) => r.length >= minPts);
    const labels = new Array(n).fill(UNVIS);
    const frames = [];
    const snap = (extra) => ({ lab: labels.slice(), core, ...extra });
    frames.push({ phase: "core", major: true, arch: ["data", "rq", "core"], title: `Core points: at least ${minPts} neighbours within ε = ${eps.toFixed(2)}`, text: `Count each point's ε-neighbourhood (itself included). <b>${core.filter(Boolean).length}</b> of ${n} points are <b>core</b> (large dots): they sit in dense regions. Clusters will be grown only through core points.`, tex: "N_\\varepsilon(p)=\\{q:\\lVert p-q\\rVert\\le\\varepsilon\\},\\qquad p\\text{ is core}\\iff |N_\\varepsilon(p)|\\ge \\text{minPts}", st: snap({ showCore: true }), m: { clusters: 0, noise: 0 } });
    let C = 0, pendingNoise = [];
    const flushNoise = () => {
      if (!pendingNoise.length) return;
      frames.push({ phase: "noise", arch: ["rq", "noise"], title: `${pendingNoise.length} non-core point${pendingNoise.length > 1 ? "s" : ""} not reachable yet → tentatively noise`, text: "A point that is not core and not yet claimed by any cluster is marked noise. It can still be adopted later as a <b>border</b> point if a cluster's expansion reaches it.", st: snap({ focus: pendingNoise.slice() }), m: stats() });
      pendingNoise = [];
    };
    const stats = () => ({ clusters: C, noise: labels.filter((l) => l === NOISE).length });
    for (let i = 0; i < n; i++) {
      if (labels[i] !== UNVIS) continue;
      if (!core[i]) { labels[i] = NOISE; pendingNoise.push(i); continue; }
      flushNoise();
      labels[i] = C;
      frames.push({ phase: "seed", major: true, arch: ["core", "expand"], title: `New cluster ${C} seeded at core point #${i}`, text: "An unvisited core point starts a new cluster. Everything density-reachable from it will join.", st: snap({ cur: i, circles: [i] }), m: stats() });
      let wave = nb[i].filter((j) => j !== i), wn = 1;
      for (const j of wave) if (labels[j] === UNVIS || labels[j] === NOISE) labels[j] = C;
      while (wave.length) {
        const next = new Set();
        for (const j of wave) if (core[j]) for (const q of nb[j]) if (labels[q] === UNVIS || labels[q] === NOISE) { labels[q] = C; next.add(q); }
        frames.push({ phase: "expand", arch: ["expand", "border"], title: `Cluster ${C}: expansion wave ${wn} adds ${wave.length} point${wave.length > 1 ? "s" : ""}`, text: `Breadth-first search: every newly reached <b>core</b> point contributes its own ε-neighbourhood (circles); reached non-core points become <b>border</b> points and stop the spread there.`, st: snap({ wave: wave.slice(), circles: wave.filter((j) => core[j]) }), m: stats() });
        wave = [...next]; wn++;
      }
      C++;
    }
    flushNoise();
    frames.push({ phase: "done", major: true, arch: ["out"], title: `Done: ${C} cluster${C === 1 ? "" : "s"}, ${labels.filter((l) => l === NOISE).length} noise points`, text: "No k was given: the number of clusters fell out of the density threshold. Clusters can take any shape. Note the weakness: one global ε cannot fit clusters of very different density (try the varied-density data).", st: snap({}), m: stats() });
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "dbscan", fps: 2.5,
      legend: ui.legend([["var(--text)", "core (large)"], ["var(--text-faint)", "noise ×"], ["var(--series-4)", "ε-neighbourhood", "ring"]]),
      tools: ui.btn("replay", "replay", true),
      hint: "click to add points · drag / double-click to edit · click the k-distance chart to set ε",
      data: { kind: "cluster", presets: ["moons", "circles", "blobs3", "varied", "outliers", "spiral", "uniform"], def: "moons", n: 180, noise: 0.15 },
      side: [{ title: "density", html: ui.range("eps", "ε (radius)", 0.3, 4, 0.05, 1.2) + ui.range("mp", "minPts", 2, 15, 1, 5) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "data", label: "points", x: 40, y: 40, w: 62, kind: "data" },
          { id: "rq", label: "region query", sub: "N_ε(p)", x: 150, y: 40, kind: "op" },
          { id: "core", label: "core?", sub: "|N_ε| ≥ minPts", x: 280, y: 40, kind: "loss" },
          { id: "expand", label: "expand cluster", sub: "BFS via cores", x: 280, y: 112, kind: "op" },
          { id: "border", label: "border", sub: "reached, not core", x: 150, y: 112, kind: "param" },
          { id: "noise", label: "noise", sub: "unreachable", x: 52, y: 112, w: 70, kind: "param" },
          { id: "out", label: "labels", sub: "clusters + noise", x: 216, y: 170, kind: "out" },
        ],
        edges: [{ from: "data", to: "rq" }, { from: "rq", to: "core" }, { from: "core", to: "expand", label: "yes" }, { from: "expand", to: "border" }, { from: "rq", to: "noise", label: "no" }, { from: "expand", to: "out" }],
      },
      phases: { core: { label: "core points", color: 3 }, seed: { label: "new cluster", color: 0 }, expand: { label: "expand", color: 2 }, noise: { label: "noise", color: 1 }, done: { label: "done", color: 2 } },
      charts: [{ id: "kd", title: "k-distance plot (choosing ε)", custom: true, note: "Each point's distance to its minPts-th nearest neighbour, sorted. The knee marks a good ε: below it, points sit in dense regions." }, { id: "cnt", title: "clusters & noise", series: [{ key: "clusters", label: "clusters", color: "var(--series-1)", digits: 0 }, { key: "noise", label: "noise", color: "var(--series-2)", dash: "4 3", digits: 0 }] }],
      onFrame: render, onData: () => replay(),
    });
    lab.editable({ newPoint: (x, y) => ({ x, y, label: 0 }), onChange: () => replay({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    function replay({ at = "start" } = {}) { lab.load(buildFrames(lab.points, lab.val("eps"), lab.val("mp")), { at, play: at === "start" }); drawKD(); }
    function drawKD() {
      const el = lab.charts.kd.el; el.innerHTML = "";
      const k = lab.val("mp"), pts = lab.points;
      if (pts.length <= k) return;
      const kd = pts.map((p) => pts.map((q) => Math.hypot(p.x - q.x, p.y - q.y)).sort((a, b) => a - b)[Math.min(k - 1, pts.length - 1)]).sort((a, b) => a - b);
      const P = Lab.xy(el, { w: 360, h: 130, xd: [0, kd.length - 1], yd: [0, Math.max(...kd) * 1.05], xLabel: "points sorted", yLabel: "dist" });
      P.line("kd", kd.map((v, i) => [i, v]), { color: "var(--series-5)" });
      P.top.append("line").attr("x1", P.m.l).attr("x2", P.w - P.m.r).attr("y1", P.y(lab.val("eps"))).attr("y2", P.y(lab.val("eps"))).attr("stroke", "var(--series-4)").attr("stroke-dasharray", "5 4").attr("stroke-width", 1.6);
      P.svg.style("cursor", "crosshair").on("click", (ev) => {
        const [, py] = d3.pointer(ev);
        const v = Math.max(0.3, Math.min(4, P.y.invert(py)));
        lab.ctl("eps").value = v.toFixed(2); lab.ctl("eps").dispatchEvent(new Event("input")); lab.ctl("eps").dispatchEvent(new Event("change"));
      });
      el.closest(".lab-card").querySelector(".chart-legend").innerHTML = `ε = <b>${lab.val("eps").toFixed(2)}</b>`;
    }
    const circG = L.mid.append("g"), noiseG = L.top.append("g");
    function render(f) {
      const st = f.st;
      if (!st.lab.length) { lab.drawPoints({}); return; }
      const eps = lab.val("eps");
      circG.selectAll("*").remove();
      for (const i of st.circles || []) {
        const p = lab.points[i];
        circG.append("ellipse").attr("cx", pl.x(p.x)).attr("cy", pl.y(p.y)).attr("rx", pl.x(eps) - pl.x(0)).attr("ry", pl.y(0) - pl.y(eps)).attr("fill", "var(--series-4)").attr("fill-opacity", 0.06).attr("stroke", "var(--series-4)").attr("stroke-opacity", 0.7).attr("stroke-width", 1.2);
      }
      const wave = new Set(st.wave || []), focus = new Set(st.focus || []);
      const visible = lab.points.map((_, i) => st.lab[i] !== NOISE);
      lab.drawPoints({
        pts: lab.points,
        color: (d, i) => (st.lab[i] >= 0 ? col(st.lab[i]) : st.lab[i] === NOISE ? "transparent" : st.showCore && st.core[i] ? "var(--text-dim)" : "var(--text-faint)"),
        r: (d, i) => (st.core[i] ? 5.2 : 3.2),
        stroke: (d, i) => (wave.has(i) || i === st.cur ? "var(--accent)" : st.lab[i] === NOISE ? "transparent" : "var(--bg)"),
        sw: (d, i) => (wave.has(i) || i === st.cur ? 2 : 1), opacity: (d, i) => (st.lab[i] === UNVIS && !st.showCore ? 0.45 : 1),
      });
      const nz = noiseG.selectAll("path").data(lab.points.map((p, i) => ({ p, i })).filter((d) => st.lab[d.i] === NOISE));
      nz.exit().remove();
      nz.enter().append("path").attr("d", "M-4,-4L4,4M-4,4L4,-4").attr("pointer-events", "none").merge(nz)
        .attr("transform", (d) => `translate(${pl.x(d.p.x)},${pl.y(d.p.y)})`).attr("stroke", (d) => (focus.has(d.i) ? "var(--series-2)" : "var(--text-faint)")).attr("stroke-width", (d) => (focus.has(d.i) ? 2.4 : 1.6));
      const nC = new Set(st.lab.filter((l) => l >= 0)).size;
      const border = st.lab.filter((l, i) => l >= 0 && !st.core[i]).length;
      lab.readout(`<div class="kv"><span>clusters</span><b>${nC}</b><span>core points</span><b>${st.core.filter(Boolean).length}</b><span>border points</span><b>${border}</b><span>noise</span><b>${st.lab.filter((l) => l === NOISE).length}</b></div>`);
    }
    lab.watch(["eps", "mp"], () => replay({ at: "end" }), { format: { eps: (v) => (+v).toFixed(2) } });
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.loadPreset("moons");
    return lab.destroy;
  }

  MLApp.register({
    id: "dbscan",
    name: "DBSCAN",
    category: "Unsupervised - Clustering",
    tagline: "core points, BFS expansion, noise",
    description: "Density-based clustering with no k. Watch core points get identified, clusters spread outward wave by wave through ε-neighbourhoods, border points get adopted and noise get marked; choose ε from the k-distance plot.",
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
