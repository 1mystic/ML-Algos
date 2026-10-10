(() => {
  const { ui, col, fmt } = Lab;

  function affinity(pts, mode, k, sigma) {
    const n = pts.length;
    const D = pts.map((p) => pts.map((q) => Math.hypot(p.x - q.x, p.y - q.y)));
    const W = MLU.zeros(n, n);
    if (mode === "knn") {
      for (let i = 0; i < n; i++) {
        const nn = D[i].map((d, j) => [d, j]).sort((a, b) => a[0] - b[0]).slice(1, k + 1);
        for (const [, j] of nn) { W[i][j] = 1; W[j][i] = 1; }
      }
    } else {
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) W[i][j] = Math.exp(-(D[i][j] ** 2) / (2 * sigma * sigma));
    }
    return W;
  }
  function kmeansRows(U, k) {
    let best = null;
    for (let r = 0; r < 6; r++) {
      let C = MLU.shuffle([...U.keys()]).slice(0, k).map((i) => U[i].slice());
      let lab = new Array(U.length).fill(0);
      for (let it = 0; it < 50; it++) {
        let ch = 0;
        lab = U.map((u, i) => { let b = 0, bd = Infinity; C.forEach((c, j) => { const d = c.reduce((s, v, t) => s + (v - u[t]) ** 2, 0); if (d < bd) { bd = d; b = j; } }); if (b !== lab[i]) ch++; return b; });
        C = C.map((c, j) => { const m = U.filter((_, i) => lab[i] === j); return m.length ? c.map((_, t) => MLU.mean(m.map((u) => u[t]))) : c; });
        if (!ch) break;
      }
      const J = U.reduce((s, u, i) => s + C[lab[i]].reduce((a, v, t) => a + (v - u[t]) ** 2, 0), 0);
      if (!best || J < best.J) best = { lab, J };
    }
    return best.lab;
  }
  function plainKMeans(pts, k) { return kmeansRows(pts.map((p) => [p.x, p.y]), k); }

  function compute(pts, P) {
    const n = pts.length;
    const W = affinity(pts, P.aff, P.knn, P.sigma);
    const deg = W.map((r) => r.reduce((a, b) => a + b, 0) + 1e-12);
    const L = W.map((r, i) => r.map((w, j) => (i === j ? 1 : 0) - w / Math.sqrt(deg[i] * deg[j])));
    const eig = MLU.jacobiEigen(L, 40); // sorted descending
    const vals = eig.values.slice().reverse(), vecs = eig.vectors.slice().reverse();
    const k = P.k;
    const U = Array.from({ length: n }, (_, i) => vecs.slice(0, k).map((v) => v[i]));
    const Un = U.map((u) => { const s = Math.hypot(...u) || 1; return u.map((v) => v / s); });
    const lab = kmeansRows(Un, k);
    return { W, deg, vals, vecs, U, Un, lab, km: plainKMeans(pts, k) };
  }

  function buildFrames(pts, P) {
    if (pts.length < P.k + 2) return { frames: [{ phase: "idle", title: "Add more points", text: "", st: {} }], R: null };
    const R = compute(pts, P);
    const edges = R.W.reduce((s, r) => s + r.filter((w) => w > 0.05).length, 0) / 2;
    const gap = R.vals[P.k] - R.vals[P.k - 1];
    const frames = [
      { phase: "data", major: true, arch: ["data"], title: "Raw points: k-means would cut these with straight lines", text: "Shown in the faint colours: plain k-means on the raw coordinates. On rings and moons it slices straight through the true groups, because it can only make convex clusters.", st: { view: "data", km: true } },
      { phase: "graph", major: true, arch: ["data", "graph"], title: `Build a similarity graph (${Math.round(edges)} edges)`, text: P.aff === "knn" ? `Connect each point to its ${P.knn} nearest neighbours. Points in the same ring are linked through chains of neighbours even when far apart in a straight line.` : `Fully connected graph with Gaussian weights, σ = ${P.sigma}. Line opacity shows weight.`, tex: P.aff === "knn" ? "W_{ij}=1\\text{ if }j\\in\\text{kNN}(i)\\text{ or }i\\in\\text{kNN}(j)" : "W_{ij}=\\exp\\!\\big(-\\lVert x_i-x_j\\rVert^2/2\\sigma^2\\big)", st: { view: "data", edges: true } },
      { phase: "lap", arch: ["graph", "lap"], title: "Normalised graph Laplacian", text: "The Laplacian measures how much a function on the nodes varies across edges. Its smallest eigenvectors are the smoothest possible functions on the graph: roughly constant inside each connected group, changing only across weak links. The panel shows the affinity matrix, rows ordered by the final clusters.", tex: "L_{sym}=I-D^{-1/2}WD^{-1/2},\\qquad D_{ii}=\\sum_j W_{ij}", st: { view: "data", edges: true, mat: true } },
      { phase: "eig", arch: ["lap", "eig"], title: `Take the ${P.k} smallest eigenvectors (eigengap ${gap.toFixed(3)})`, text: "Each point gets new coordinates: its entries in the first k eigenvectors. A large gap after the k-th eigenvalue (chart) means the graph really has about k well-separated groups.", tex: "L_{sym}\\,u_j=\\lambda_j u_j,\\quad 0=\\lambda_1\\le\\lambda_2\\le\\dots,\\qquad U=[u_1,\\dots,u_k]", st: { view: "data", edges: true, mat: true } },
    ];
    for (let t = 1; t <= 10; t++) frames.push({ phase: "embed", arch: ["eig", "norm"], title: "Move every point to its spectral coordinates", text: "Animating from the original plane into the embedding (eigenvectors 2 and 3 shown, rows normalised). Groups that were tangled rings become tight, separate clumps.", tex: "y_i=\\frac{U_{i,:}}{\\lVert U_{i,:}\\rVert}", st: { view: "morph", t: t / 10 } });
    frames.push({ phase: "km", major: true, arch: ["norm", "km"], title: "Run k-means in the embedding", text: "Now the clusters are compact blobs, exactly what k-means is good at.", st: { view: "embed", colored: true } });
    for (let t = 1; t <= 8; t++) frames.push({ phase: "back", arch: ["km", "out"], title: "Carry the labels back to the original points", text: "Same labels, original coordinates: the rings and moons come out whole.", st: { view: "morph", t: 1 - t / 8, colored: true } });
    frames.push({ phase: "done", major: true, arch: ["out"], title: "Spectral clustering result", text: `Compare with plain k-means (toggle in the sidebar). Agreement between the two: <b>${(agree(R.lab, R.km, P.k) * 100).toFixed(0)}%</b>. Spectral clustering solves a relaxed version of the normalised graph cut problem.`, tex: "\\min_{A_1..A_k}\\;\\sum_j\\frac{\\mathrm{cut}(A_j,\\bar A_j)}{\\mathrm{vol}(A_j)}", st: { view: "data", colored: true, mat: true } });
    return { frames, R };
  }
  function agree(a, b, k) { // best label permutation agreement (greedy)
    const used = new Set(); let ok = 0;
    for (let c = 0; c < k; c++) {
      let best = -1, bc = 0;
      for (let d = 0; d < k; d++) if (!used.has(d)) { const m = a.reduce((s, v, i) => s + (v === c && b[i] === d), 0); if (m > bc || best < 0) { bc = m; best = d; } }
      used.add(best); ok += bc;
    }
    return ok / a.length;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "spectral", fps: 4,
      legend: ui.legend([["var(--text-faint)", "graph edge", "line"]]),
      tools: ui.btn("replay", "replay", true),
      hint: "click to add points · drag / double-click to edit · keep n ≤ 160 (eigendecomposition is O(n³))",
      data: { kind: "cluster", presets: ["circles", "moons", "spiral", "blobs3", "aniso"], def: "circles", n: 150, noise: 0.05, nRange: [20, 180] },
      side: [{ title: "model", html: ui.range("k", "clusters k", 2, 6, 1, 2) + ui.select("aff", "affinity", [["knn", "k-nearest-neighbour graph"], ["rbf", "Gaussian (RBF) kernel"]], "knn") + ui.range("knn", "neighbours", 3, 20, 1, 5) + ui.range("sigma", "σ for RBF", 0.3, 4, 0.1, 1) + ui.check("km", "show plain k-means result", false) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "data", label: "points", x: 40, y: 40, w: 60, kind: "data" },
          { id: "graph", label: "affinity W", sub: "kNN or RBF", x: 140, y: 40, kind: "op" },
          { id: "lap", label: "Laplacian", sub: "I − D^-½ W D^-½", x: 262, y: 40, kind: "op" },
          { id: "eig", label: "k smallest", sub: "eigenvectors U", x: 330, y: 112, kind: "loss" },
          { id: "norm", label: "normalise rows", x: 206, y: 112, kind: "param" },
          { id: "km", label: "k-means", sub: "on rows of U", x: 92, y: 112, kind: "op" },
          { id: "out", label: "labels", x: 92, y: 168, w: 60, kind: "out" },
        ],
        edges: [{ from: "data", to: "graph" }, { from: "graph", to: "lap" }, { from: "lap", to: "eig" }, { from: "eig", to: "norm" }, { from: "norm", to: "km" }, { from: "km", to: "out" }],
      },
      phases: { data: { label: "data", color: 3 }, graph: { label: "graph", color: 4 }, lap: { label: "Laplacian", color: 4 }, eig: { label: "eigenvectors", color: 0 }, embed: { label: "embed", color: 0 }, km: { label: "k-means", color: 2 }, back: { label: "map back", color: 2 }, done: { label: "done", color: 2 } },
      panels: [{ id: "mat", title: "affinity matrix W", sub: "ordered by cluster" }],
      charts: [{ id: "eig", title: "eigenvalues of L (eigengap heuristic)", custom: true }],
      onFrame: render, onData: () => replay(),
    });
    lab.editable({ newPoint: (x, y) => ({ x, y, label: 0 }), onChange: () => replay({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    let R = null;
    const P = () => ({ k: lab.val("k"), aff: lab.val("aff"), knn: lab.val("knn"), sigma: lab.val("sigma") });
    const mat = Lab.matrix(lab.R("panel-mat"), { cell: 3, showValues: false, labelW: 4, labelH: 4 });
    function replay({ at = "start" } = {}) {
      const r = buildFrames(lab.points, P());
      R = r.R;
      lab.ctl("knn").closest(".field").style.display = P().aff === "knn" ? "" : "none";
      lab.ctl("sigma").closest(".field").style.display = P().aff === "rbf" ? "" : "none";
      drawEig();
      if (R) {
        const order = [...R.lab.keys()].sort((a, b) => R.lab[a] - R.lab[b]);
        mat.set(order.map((i) => order.map((j) => R.W[i][j])), { range: [0, 1] });
      }
      lab.load(r.frames, { at, play: at === "start" });
    }
    function drawEig() {
      const el = lab.charts.eig.el; el.innerHTML = "";
      if (!R) return;
      const v = R.vals.slice(0, 10);
      const b = Lab.bars(el, { w: 360, h: 130, yd: [0, Math.max(...v) * 1.1 + 1e-3] });
      b.set(v.map((x, i) => ({ label: "λ" + (i + 1), value: x, color: i < P().k ? "var(--series-1)" : "var(--text-faint)" })), { digits: 3 });
    }
    // embedding coordinates scaled into the plane
    function embedXY() {
      const E = R.Un.map((u) => (u.length >= 3 ? [u[1], u[2]] : [u[0], u[1]]));
      const ex = d3.extent(E, (e) => e[0]), ey = d3.extent(E, (e) => e[1]);
      const sx = (v) => -8 + ((v - ex[0]) / (ex[1] - ex[0] || 1)) * 16, sy = (v) => -8 + ((v - ey[0]) / (ey[1] - ey[0] || 1)) * 16;
      return E.map((e) => [sx(e[0]), sy(e[1])]);
    }
    const edgeG = L.links.append("g");
    function render(f) {
      const st = f.st;
      edgeG.selectAll("*").remove();
      if (!R) { lab.drawPoints({}); return; }
      const E = embedXY();
      const pos = lab.points.map((p, i) => (st.view === "embed" ? E[i] : st.view === "morph" ? [p.x + (E[i][0] - p.x) * st.t, p.y + (E[i][1] - p.y) * st.t] : [p.x, p.y]));
      if (st.edges) {
        const n = lab.points.length;
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (R.W[i][j] > 0.05) edgeG.append("line").attr("x1", pl.x(pos[i][0])).attr("y1", pl.y(pos[i][1])).attr("x2", pl.x(pos[j][0])).attr("y2", pl.y(pos[j][1])).attr("stroke", "var(--text-faint)").attr("stroke-opacity", 0.15 + 0.4 * R.W[i][j]).attr("stroke-width", 0.8);
      }
      const showKm = lab.val("km") || st.km;
      const sel = lab.drawPoints({ color: (d, i) => (showKm && !st.colored ? col(R.km[i]) : st.colored ? col(R.lab[i]) : "var(--text-faint)"), opacity: showKm && !st.colored ? 0.6 : 1, edit: st.view === "data" });
      sel.attr("cx", (d, i) => pl.x(pos[i][0])).attr("cy", (d, i) => pl.y(pos[i][1]));
      if (lab.val("km") && st.colored) sel.attr("stroke", (d, i) => col(R.km[i])).attr("stroke-width", 2);
      lab.readout(`<div class="kv"><span>points</span><b>${lab.points.length}</b><span>eigengap λₖ₊₁ − λₖ</span><b>${fmt(R.vals[P().k] - R.vals[P().k - 1])}</b><span>agreement with k-means</span><b>${(agree(R.lab, R.km, P().k) * 100).toFixed(0)}%</b></div><div class="note">${lab.val("km") ? "ring colour = plain k-means label" : ""}</div>`);
    }
    lab.watch(["k", "aff", "knn", "sigma"], () => replay({ at: "end" }), { format: { sigma: (v) => (+v).toFixed(1) } });
    lab.watch(["km"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.loadPreset("circles");
    return lab.destroy;
  }

  MLApp.register({
    id: "spectral",
    name: "Spectral Clustering",
    category: "Unsupervised - Clustering",
    tagline: "graph Laplacian eigenvectors → k-means",
    isNew: true,
    description: "Build a similarity graph, take the smallest eigenvectors of its Laplacian, and cluster in that new space. Watch rings and moons that defeat k-means morph into separable clumps in the spectral embedding, then map back whole.",
    info: {
      type: "Unsupervised - Clustering. Graph-based, uses a spectral (eigenvector) embedding; relaxes normalised graph cuts.",
      scenario: "Clusters defined by connectivity rather than compactness (rings, moons, manifolds), moderate n, or data that is naturally a graph (social networks, image pixels).",
      inputs: "Points with a similarity function (k-NN graph or RBF kernel), or a precomputed affinity matrix; the number of clusters k.",
      output: "A cluster label per point, plus a low-dimensional spectral embedding.",
      intuition: {
        definition: "Turn the data into a graph where similar points are connected. Clusters are groups of nodes with many edges inside and few edges between. The eigenvectors of the graph Laplacian with the smallest eigenvalues are smooth over the graph, so they take nearly constant values on each group. Use them as new coordinates and the clusters become easy.",
        steps: ["Build an affinity matrix W (k-NN or Gaussian kernel).", "Form the normalised Laplacian L = I − D^-½ W D^-½.", "Compute its k smallest eigenvectors; stack them as columns of U.", "Normalise the rows of U and run k-means on them.", "Assign each original point the label of its row."],
        applications: ["Image segmentation (normalised cuts, Shi & Malik)", "Community detection in networks", "Clustering on manifolds and non-convex shapes", "Speaker diarisation and gene-network modules"],
      },
      math: [
        { title: "Graph Laplacian", formula: "L = D − W,  L_sym = I − D^-½ W D^-½", note: "D is the diagonal degree matrix. L is positive semi-definite." },
        { title: "Smoothness", formula: "fᵀ L f = ½ Σᵢⱼ Wᵢⱼ (fᵢ − fⱼ)²", note: "Small values mean f barely changes across strong edges: such f are near-constant within clusters." },
        { title: "Zero eigenvalues", formula: "#{λ = 0} = number of connected components", note: "With perfectly separated groups, the indicator vectors of the components are exact eigenvectors." },
        { title: "Normalised cut", formula: "Ncut(A₁..A_k) = Σⱼ cut(Aⱼ, Āⱼ)/vol(Aⱼ)", note: "NP-hard to minimise exactly; relaxing indicator vectors to real values turns it into the eigenproblem." },
        { title: "Eigengap heuristic", formula: "choose k maximising λ_{k+1} − λ_k", note: "A big jump after λ_k signals k well-separated groups." },
      ],
      pipeline: [{ label: "Affinity graph" }, { label: "Laplacian" }, { label: "k eigenvectors" }, { label: "Row-normalise" }, { label: "k-means", accent: "green" }],
      decisionFunction: { text: "label(xᵢ) = kmeans( U_{i,:} / ‖U_{i,:}‖ )", mechanism: "Clustering happens in the spectral embedding; labels transfer back to the original points one to one. New points need an out-of-sample extension (e.g. Nyström)." },
      lossFunction: { text: "min_{A₁..A_k} Σⱼ cut(Aⱼ, Āⱼ) / vol(Aⱼ)   (relaxed: min tr(Uᵀ L_sym U) s.t. UᵀU = I)", mechanism: "The relaxed problem is solved exactly by the k smallest eigenvectors (Rayleigh–Ritz); k-means then rounds the continuous solution back to discrete labels." },
      optimization: [{ title: "Eigensolver", formula: "O(n³) dense, far less with sparse k-NN graphs + Lanczos", note: "ARPACK / LOBPCG find only the few eigenvectors needed." }, { title: "Nyström / landmarks", formula: "approximate W from m ≪ n landmarks", note: "Scales spectral clustering to large n." }],
      assumptions: [{ name: "Clusters are well connected internally", why: "The method finds weakly linked groups in the graph.", check: "Visualise the graph; adjust neighbours or σ." }, { name: "Good affinity scale", why: "Too few neighbours fragments clusters; too many bridges them.", check: "Sweep k-NN / σ and watch the eigengap." }, { name: "k known (or eigengap visible)", why: "k-means at the end needs k.", check: "Eigengap heuristic." }],
      hyperparameters: [{ name: "n_clusters", range: "2 - 20", increasing: "More, smaller groups.", strategy: "Eigengap." }, { name: "affinity", range: "nearest_neighbors / rbf", increasing: "Not applicable", strategy: "k-NN graphs are sparse and robust to scale." }, { name: "n_neighbors", range: "5 - 30", increasing: "Denser graph, risk of bridging clusters.", strategy: "Smallest value that keeps each cluster connected." }, { name: "gamma / σ", range: "data scale", increasing: "Wider kernel, more global connections.", strategy: "Median pairwise distance is a good start." }],
      metrics: ["Silhouette (in embedding space)", "Normalised cut value", "ARI / NMI against labels"],
      typicalUses: ["Non-convex clusters", "Image segmentation", "Graph community detection"],
      diagnostics: ["Plot the eigenvalues: look for the gap.", "Plot the embedding: clusters should be tight blobs.", "Check graph connectivity: isolated points produce extra zero eigenvalues."],
      advantages: ["Finds non-convex clusters k-means cannot.", "Works directly on graphs and similarity matrices.", "Solid theory (graph cuts, random walks).", "Only a standard eigenproblem plus k-means."],
      limitations: [{ name: "Expensive for large n", note: "dense eigendecomposition is cubic", fix: "sparse graphs, Nyström, or HDBSCAN." }, { name: "Sensitive to graph construction", note: "neighbours / σ change the result", fix: "sweep and use the eigengap." }, { name: "k still required", note: "", fix: "eigengap heuristic." }],
      alternatives: [{ name: "DBSCAN", when: "Non-convex clusters plus noise, larger n." }, { name: "K-Means", when: "Clusters are compact blobs." }, { name: "Hierarchical (single linkage)", when: "Connectivity-based clusters with a dendrogram." }],
      pitfalls: [{ problem: "Result looks like k-means", solution: "The graph is too dense (too many neighbours / too wide σ)." }, { problem: "Many tiny clusters", solution: "The graph is disconnected; add neighbours." }, { problem: "Out of memory", solution: "Use a sparse k-NN affinity." }],
      quickRef: [{ name: "Laplacian", formula: "L_sym = I − D^-½WD^-½" }, { name: "Embedding", formula: "k smallest eigenvectors" }, { name: "Rounding", formula: "k-means on normalised rows" }, { name: "Choose k", formula: "argmax eigengap" }],
      code: `from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                        n_neighbors=10, assign_labels="kmeans", random_state=0)
labels = sc.fit_predict(X)`,
      whyChain: [
        { q: "Why do Laplacian eigenvectors reveal clusters?", a: "fᵀLf sums squared differences of f across edges. The eigenvectors with the smallest eigenvalues minimise this, so they vary as little as possible along edges: nearly constant within densely connected groups and changing only across sparse cuts." },
        { q: "Why run k-means at the end?", a: "The eigenvectors are a continuous relaxation of the discrete cluster indicators. In the embedding each cluster becomes a tight blob, and k-means is a convenient way to round back to discrete labels." },
        { q: "Why normalise the Laplacian?", a: "The unnormalised cut favours cutting off tiny groups. Normalising by volume (sum of degrees) balances cluster sizes and corresponds to random-walk transition probabilities." },
      ],
      workedExample: {
        setup: "Two disconnected pairs of nodes: {1,2} joined with weight 1, {3,4} joined with weight 1, nothing in between.",
        steps: ["Degrees are all 1, so L = I − W.", "λ = 0 has multiplicity 2 (two components).", "Eigenvectors: (1,1,0,0)/√2 and (0,0,1,1)/√2.", "Rows of U: nodes 1,2 → (0.71, 0), nodes 3,4 → (0, 0.71)."],
        result: "k-means on the rows separates {1,2} from {3,4} perfectly",
      },
    },
    mount,
  });
})();
