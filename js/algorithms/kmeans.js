(() => {
  const { ui, col } = Lab;

  const d2 = (p, c) => (p.x - c[0]) ** 2 + (p.y - c[1]) ** 2;
  function inertia(pts, cents, assign) {
    let s = 0;
    for (let i = 0; i < pts.length; i++) if (assign[i] >= 0) s += d2(pts[i], cents[assign[i]]);
    return s;
  }
  function nearest(p, cents) {
    let b = 0, bd = Infinity;
    for (let j = 0; j < cents.length; j++) { const d = d2(p, cents[j]); if (d < bd) { bd = d; b = j; } }
    return b;
  }
  function seedPlusPlus(pts, k, record) {
    const n = pts.length;
    const first = MLU.randInt(n);
    const cents = [[pts[first].x, pts[first].y]];
    if (record) record(cents, null, first);
    while (cents.length < k) {
      const D = pts.map((p) => Math.min(...cents.map((c) => d2(p, c))));
      const tot = D.reduce((a, b) => a + b, 0) || 1;
      let r = MLU.rng() * tot, idx = 0;
      for (; idx < n - 1; idx++) { r -= D[idx]; if (r <= 0) break; }
      if (record) record(cents, D.map((v) => v / tot), idx);
      cents.push([pts[idx].x, pts[idx].y]);
    }
    return cents;
  }
  // Full Lloyd run without recording, used by the elbow scan.
  function quickKMeans(pts, k) {
    let cents = seedPlusPlus(pts, k);
    let assign = new Int16Array(pts.length).fill(-1);
    for (let it = 0; it < 60; it++) {
      let changed = 0;
      for (let i = 0; i < pts.length; i++) { const b = nearest(pts[i], cents); if (b !== assign[i]) { changed++; assign[i] = b; } }
      cents = cents.map((c, j) => {
        let sx = 0, sy = 0, m = 0;
        for (let i = 0; i < pts.length; i++) if (assign[i] === j) { sx += pts[i].x; sy += pts[i].y; m++; }
        return m ? [sx / m, sy / m] : c;
      });
      if (!changed) break;
    }
    return { cents, assign, J: inertia(pts, cents, assign) };
  }
  function silhouette(pts, assign, k) {
    if (k < 2 || pts.length < 3) return null;
    let tot = 0;
    for (let i = 0; i < pts.length; i++) {
      const sums = new Array(k).fill(0), cnt = new Array(k).fill(0);
      for (let j = 0; j < pts.length; j++) if (i !== j) { const d = Math.sqrt(d2(pts[i], [pts[j].x, pts[j].y])); sums[assign[j]] += d; cnt[assign[j]]++; }
      const own = assign[i];
      const a = cnt[own] ? sums[own] / cnt[own] : 0;
      let b = Infinity;
      for (let c = 0; c < k; c++) if (c !== own && cnt[c]) b = Math.min(b, sums[c] / cnt[c]);
      if (!isFinite(b)) continue;
      tot += (b - a) / Math.max(a, b, 1e-9);
    }
    return tot / pts.length;
  }

  function buildFrames(pts, k, init) {
    const n = pts.length;
    if (!n) return [{ phase: "idle", title: "No data yet", text: "Pick an example dataset or click the canvas to add points.", arch: ["data"], st: { cents: [], assign: new Int16Array(0) } }];
    k = Math.min(k, n);
    const frames = [];
    const none = new Int16Array(n).fill(-1);
    let cents;
    if (init === "kmeans++") {
      cents = seedPlusPlus(pts, k, (cs, w, pick) => {
        frames.push({
          phase: "seed", arch: ["data", "seed"],
          title: cs.length === 1 && !w ? "k-means++: first centroid is a uniformly random point" : `k-means++: choosing centroid ${cs.length + 1} of ${k}`,
          text: w ? "Each point's size is its probability of being picked next. Points far from every existing centroid are much more likely, which spreads the seeds out." : "One data point is chosen at random to be the first centre.",
          tex: w ? "P(x)=\\dfrac{D(x)^2}{\\sum_{x'} D(x')^2},\\quad D(x)=\\min_j \\lVert x-\\mu_j\\rVert" : null,
          st: { cents: cs.map((c) => c.slice()), assign: none, w, pick },
        });
      });
    } else {
      const idx = MLU.shuffle([...Array(n).keys()]).slice(0, k);
      cents = idx.map((i) => [pts[i].x, pts[i].y]);
    }
    frames.push({
      phase: "seed", major: true, arch: ["seed"],
      title: `${k} initial centroids placed`, text: init === "kmeans++" ? "Seeding done. Now Lloyd's algorithm alternates two steps until nothing changes." : "Random init: k data points picked uniformly. Try a few reruns to see how often this lands in a poor local minimum.",
      st: { cents: cents.map((c) => c.slice()), assign: none },
    });
    let assign = none;
    for (let it = 1; it <= 60; it++) {
      const na = new Int16Array(n);
      let changed = 0;
      for (let i = 0; i < n; i++) { na[i] = nearest(pts[i], cents); if (na[i] !== assign[i]) changed++; }
      assign = na;
      const J = inertia(pts, cents, assign);
      frames.push({
        phase: "assign", major: true, arch: ["assign", "data"],
        title: `Iteration ${it} · assignment step`,
        text: `Every point joins its nearest centroid. <b>${changed}</b> point${changed === 1 ? "" : "s"} changed cluster. Inertia is now <b>${J.toFixed(1)}</b>.`,
        tex: "c_i=\\arg\\min_{j}\\;\\lVert x_i-\\mu_j\\rVert^2",
        st: { cents: cents.map((c) => c.slice()), assign: na, links: true }, m: { inertia: J, changed },
      });
      if (changed === 0) {
        frames.push({
          phase: "done", major: true, arch: ["check", "out"],
          title: `Converged after ${it - 1} update${it - 1 === 1 ? "" : "s"}`,
          text: `No point changed cluster, so the next update would leave every centroid where it is. Final inertia <b>${J.toFixed(1)}</b>. This is a local minimum: rerun with a different seed to see whether a better one exists.`,
          tex: "J=\\sum_{j=1}^{k}\\sum_{x_i\\in C_j}\\lVert x_i-\\mu_j\\rVert^2",
          st: { cents: cents.map((c) => c.slice()), assign: na }, m: { inertia: J, changed: 0 },
        });
        break;
      }
      const prev = cents.map((c) => c.slice());
      cents = cents.map((c, j) => {
        let sx = 0, sy = 0, m = 0;
        for (let i = 0; i < n; i++) if (na[i] === j) { sx += pts[i].x; sy += pts[i].y; m++; }
        return m ? [sx / m, sy / m] : c;
      });
      const shift = Math.max(...cents.map((c, j) => Math.hypot(c[0] - prev[j][0], c[1] - prev[j][1])));
      const J2 = inertia(pts, cents, na);
      frames.push({
        phase: "update", arch: ["update"],
        title: `Iteration ${it} · update step`,
        text: `Each centroid jumps to the mean of its members. Largest move: <b>${shift.toFixed(2)}</b>. Inertia drops from ${J.toFixed(1)} to <b>${J2.toFixed(1)}</b>, and it can never rise.`,
        tex: "\\mu_j \\leftarrow \\frac{1}{|C_j|}\\sum_{x_i\\in C_j} x_i",
        st: { cents: cents.map((c) => c.slice()), prev, assign: na, links: true }, m: { inertia: J2, changed: 0 },
      });
    }
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "kmeans",
      fps: 2.2,
      legend: ui.legend([["var(--text)", "centroid", "ring"], ["var(--text-faint)", "trail", "line"]]),
      tools: ui.btn("rerun", "rerun (new seed)", true, "re-initialise the centroids and replay") + ui.btn("elbow", "elbow scan", false, "run k = 1…8 and plot inertia and silhouette"),
      hint: "click to add points · drag to move · double-click to delete · the run is re-recorded after every edit",
      data: { kind: "cluster", presets: ["blobs4", "blobs3", "blobs5", "aniso", "varied", "moons", "circles", "outliers", "uniform"], def: "blobs4", n: 140, noise: 0.25 },
      side: [
        { title: "model", html: ui.range("k", "k (clusters)", 1, 8, 1, 4) + ui.select("init", "initialisation", [["kmeans++", "k-means++"], ["random", "random points"]], "kmeans++") },
        { title: "view", html: ui.check("vor", "Voronoi regions", true) + ui.check("links", "assignment links", true) + ui.check("trail", "centroid trails", true) },
      ],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "data", label: "data X", sub: "n × 2, unlabeled", x: 52, y: 40, kind: "data", info: "The raw, unlabeled points. Features should be on comparable scales: k-means is pure distance." },
          { id: "seed", label: "seed", sub: "k-means++", x: 52, y: 130, kind: "param", info: "Choose k starting centroids. k-means++ samples each new seed with probability ∝ squared distance to the nearest existing seed.", tex: "P(x)\\propto D(x)^2" },
          { id: "assign", label: "assign", sub: "argmin distance", x: 190, y: 40, kind: "op", info: "Hold centroids fixed and give every point to its nearest one. This is the optimal assignment for fixed centroids.", tex: "c_i=\\arg\\min_j\\lVert x_i-\\mu_j\\rVert^2" },
          { id: "update", label: "update", sub: "mean of members", x: 190, y: 130, kind: "op", info: "Hold assignments fixed and move each centroid to its members' mean, the point minimising summed squared distance.", tex: "\\mu_j=\\tfrac{1}{|C_j|}\\textstyle\\sum_{C_j}x_i" },
          { id: "check", label: "converged?", sub: "no reassignment", x: 318, y: 130, kind: "loss", info: "Stop when an assignment step changes nothing. Inertia J decreased monotonically on the way here." },
          { id: "out", label: "labels + μ", sub: "partition", x: 318, y: 40, kind: "out", info: "One cluster id per point and k centroids. Clusters are convex Voronoi cells." },
        ],
        edges: [
          { from: "data", to: "assign" }, { from: "seed", to: "assign", label: "μ⁰" }, { from: "assign", to: "update" },
          { from: "update", to: "assign", bend: -26, dash: true, label: "repeat" }, { from: "update", to: "check" }, { from: "check", to: "out", label: "yes" },
        ],
      },
      phases: { seed: { label: "seed", color: 3 }, assign: { label: "assign", color: 0 }, update: { label: "update", color: 2 }, done: { label: "converged", color: 2 }, idle: { label: "idle" } },
      charts: [
        { id: "inertia", title: "inertia J per step", series: [{ key: "inertia", label: "J", color: "var(--series-1)", digits: 1 }] },
        { id: "elbow", title: "choosing k: elbow & silhouette", custom: true, note: "Inertia always falls as k grows; look for the bend. Silhouette (higher is better) peaks near the natural k." },
      ],
      onFrame: render,
      onData: () => rerun(),
    });
    lab.editable({ newPoint: (x, y) => ({ x, y, label: 0 }), onChange: () => rerun({ at: "end" }) });

    const pl = lab.plane;
    const L = pl.layers;
    let seed = 11;
    let elbowData = null;
    const elbowEl = lab.charts.elbow.el;

    function rerun({ at = "start", play = true } = {}) {
      MLU.seedRng(seed);
      const frames = buildFrames(lab.points, lab.val("k"), lab.val("init"));
      lab.load(frames, { at, play: play && at === "start" });
      drawElbow();
    }

    function drawElbow() {
      elbowEl.innerHTML = "";
      const P = Lab.xy(elbowEl, { w: 360, h: 130, xd: [1, 8], yd: [0, 1.05], xLabel: "k" });
      if (!elbowData) { P.g.append("text").attr("class", "ax-label").attr("x", 180).attr("y", 62).attr("text-anchor", "middle").text("press “elbow scan” to compute"); return; }
      const J1 = elbowData[0].J || 1;
      P.line("j", elbowData.map((d) => [d.k, d.J / J1]), { color: "var(--series-1)" });
      P.line("s", elbowData.filter((d) => d.s != null).map((d) => [d.k, Math.max(0, d.s)]), { color: "var(--series-3)", dash: "4 3" });
      P.g.selectAll("circle").data(elbowData).enter().append("circle").attr("cx", (d) => P.x(d.k)).attr("cy", (d) => P.y(d.J / J1)).attr("r", (d) => (d.k === lab.val("k") ? 5 : 2.8))
        .attr("fill", (d) => (d.k === lab.val("k") ? "var(--accent)" : "var(--series-1)"));
      const best = elbowData.filter((d) => d.s != null).sort((a, b) => b.s - a.s)[0];
      const head = elbowEl.closest(".lab-card").querySelector(".chart-legend");
      head.innerHTML = `<span class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>J / J₁</span><span class="legend-item"><span class="swatch" style="background:var(--series-3)"></span>silhouette · best k=<b>${best ? best.k : "–"}</b></span>`;
    }
    function elbowScan() {
      if (lab.points.length < 3) return;
      MLU.seedRng(99);
      elbowData = [];
      for (let k = 1; k <= 8; k++) {
        let best = null;
        for (let r = 0; r < 4; r++) { const res = quickKMeans(lab.points, Math.min(k, lab.points.length)); if (!best || res.J < best.J) best = res; }
        elbowData.push({ k, J: best.J, s: k > 1 ? silhouette(lab.points, best.assign, k) : null });
      }
      drawElbow();
    }

    const trailG = L.mid.append("g"), linkG = L.links, cenG = L.top.append("g");
    function render(f, i, all) {
      const st = f.st;
      const n = lab.points.length;
      // Voronoi regions behind everything
      if (lab.val("vor") && st.cents.length) {
        Lab.field(L.bg, pl, (x, y) => nearest({ x, y }, st.cents), (k) => Lab.classColor(k, 1, 0.16), { res: 200 });
      } else L.bg.selectAll("*").remove();
      // assignment links
      const showLinks = lab.val("links") && st.links;
      const lk = linkG.selectAll("line").data(showLinks ? lab.points.map((p, idx) => ({ p, c: st.assign[idx] })).filter((d) => d.c >= 0) : []);
      lk.exit().remove();
      lk.enter().append("line").merge(lk)
        .attr("x1", (d) => pl.x(d.p.x)).attr("y1", (d) => pl.y(d.p.y))
        .attr("x2", (d) => pl.x(st.cents[d.c][0])).attr("y2", (d) => pl.y(st.cents[d.c][1]))
        .attr("stroke", (d) => col(d.c)).attr("stroke-opacity", 0.22).attr("stroke-width", 0.8);
      // points
      lab.drawPoints({
        color: (d, idx) => (st.assign[idx] >= 0 ? col(st.assign[idx]) : "var(--text-faint)"),
        r: (d, idx) => (st.w ? 2.5 + Math.min(9, Math.sqrt(st.w[idx] * n) * 3.2) : 4.4),
        stroke: (d, idx) => (st.pick === idx && st.w ? "var(--accent)" : "var(--bg)"),
        sw: (d, idx) => (st.pick === idx && st.w ? 2.5 : 1.1),
      });
      // centroid trails up to this frame
      const trails = st.cents.map((_, j) => all.slice(0, i + 1).filter((fr) => fr.st.cents[j]).map((fr) => fr.st.cents[j]));
      const tr = trailG.selectAll("path").data(lab.val("trail") ? trails : []);
      tr.exit().remove();
      tr.enter().append("path").merge(tr)
        .attr("d", (t) => d3.line().x((c) => pl.x(c[0])).y((c) => pl.y(c[1]))(t.filter((c, q) => q === 0 || c[0] !== t[q - 1][0] || c[1] !== t[q - 1][1])))
        .attr("fill", "none").attr("stroke", (t, j) => col(j)).attr("stroke-width", 1.6).attr("stroke-dasharray", "3 3").attr("opacity", 0.8);
      // centroids (animate from previous position on update steps)
      const cs = cenG.selectAll("g.cent").data(st.cents);
      cs.exit().remove();
      const ent = cs.enter().append("g").attr("class", "cent no-add");
      ent.append("circle").attr("r", 11).attr("fill", "var(--bg-inset)").attr("fill-opacity", 0.55).attr("stroke-width", 2.6);
      ent.append("path").attr("d", "M-5,0H5M0,-5V5").attr("stroke-width", 2);
      const m = ent.merge(cs);
      m.select("circle").attr("stroke", (d, j) => col(j));
      m.select("path").attr("stroke", (d, j) => col(j));
      if (st.prev) {
        m.attr("transform", (d, j) => `translate(${pl.x(st.prev[j][0])},${pl.y(st.prev[j][1])})`)
          .transition().duration(520).ease(d3.easeCubicInOut).attr("transform", (d) => `translate(${pl.x(d[0])},${pl.y(d[1])})`);
      } else m.interrupt().attr("transform", (d) => `translate(${pl.x(d[0])},${pl.y(d[1])})`);

      const J = st.assign[0] >= 0 ? inertia(lab.points, st.cents, st.assign) : null;
      const sizes = st.cents.map((_, j) => st.assign.reduce((s, a) => s + (a === j), 0));
      lab.readout(`<div class="kv"><span>points</span><b>${n}</b><span>k</span><b>${st.cents.length}</b><span>inertia J</span><b>${J == null ? "–" : J.toFixed(1)}</b><span>cluster sizes</span><b>${st.assign[0] >= 0 ? sizes.join(" · ") : "–"}</b></div>`);
    }

    lab.watch(["k", "init"], () => { elbowData && drawElbow(); rerun(); });
    lab.watch(["vor", "links", "trail"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="rerun"]').onclick = () => { seed = (seed * 7 + 13) % 10007; rerun(); };
    lab.q('[data-b="elbow"]').onclick = elbowScan;

    lab.loadPreset("blobs4");
    return lab.destroy;
  }

  MLApp.register({
    id: "kmeans",
    name: "K-Means",
    category: "Unsupervised - Clustering",
    tagline: "k-means++ seeding, animated Lloyd steps",
    description: "Hard clustering by alternating assignment and centroid-averaging. Watch k-means++ spread its seeds, then step through every assign and update move, with Voronoi regions, centroid trails and an elbow/silhouette scan for choosing k.",
    info: {
      type: "Unsupervised - Clustering. Centroid-based, hard assignment.",
      scenario: "Partitioning unlabeled data into k compact, roughly spherical groups - customer segmentation, color quantization, or as an unsupervised pre-processing step.",
      inputs: "Unlabeled points {xᵢ} and a chosen number of clusters k.",
      intuition: {
        definition: "Pick k centres, then repeat two steps until nothing moves: <b>assign</b> every point to its nearest centre, and <b>move</b> each centre to the average of the points that chose it. Simple, fast, and guaranteed to converge, though not necessarily to the best answer.",
        steps: [
          "Choose k starting centroids, ideally spread out via k-means++.",
          "Assign each point to the nearest centroid.",
          "Recompute each centroid as the mean of its members.",
          "Repeat until assignments stop changing.",
        ],
        applications: [
          "Customer and market segmentation",
          "Image colour quantization down to k palette entries",
          "Compressing feature vectors into codebooks",
          "Document grouping as a fast topic proxy",
          "Initialising more expensive models such as GMM",
        ],
      },
      math: [
        { title: "Objective", formula: "J = Σ_{j=1}^{k} Σ_{x ∈ Cⱼ} ‖x − cⱼ‖²", note: "Within-cluster sum of squares, also called inertia. Minimising it exactly over all partitions is NP-hard, which is why we use an alternating heuristic." },
        { title: "Assignment step", formula: "Cⱼ = { x : j = argmin_l ‖x − c_l‖² }", note: "Holding centroids fixed, this is the assignment that minimises J. Ties are broken arbitrarily." },
        { title: "Update step", formula: "cⱼ = (1/|Cⱼ|) Σ_{x ∈ Cⱼ} x", note: "Holding assignments fixed, the mean is the point minimising summed squared distance. This is why the algorithm requires squared Euclidean distance specifically." },
        { title: "Monotone convergence", formula: "J(t+1) ≤ J(t)", note: "Both steps weakly decrease J, and there are finitely many partitions, so the algorithm must terminate. It stops at a local minimum, not the global one." },
        { title: "k-means++ seeding", formula: "P(pick x as next centre) ∝ D(x)², where D(x) = distance to the nearest chosen centre", note: "Spreads the initial centroids apart. Gives an expected O(log k) approximation guarantee and dramatically reduces bad local minima." },
        { title: "Voronoi geometry", formula: "boundaries are perpendicular bisectors between centroids", note: "The clusters are always convex polytopes. No amount of iteration can produce a crescent or ring-shaped cluster." },
      ],
      pipeline: [
        { label: "Seed centroids", note: "k-means++" },
        { label: "Assign", note: "nearest centroid" },
        { label: "Update", note: "mean of members" },
        { label: "Converged?", note: "assignments stable" },
        { label: "Labels + centroids", note: "final output", accent: "green" },
      ],
      decisionFunction: {
        text: "assign(x) = argmin_j ‖x − cⱼ‖²",
        mechanism: "Lloyd's algorithm alternates: (1) assign every point to its nearest centroid, (2) move each centroid to the mean of its assigned points - repeated until assignments stop changing.",
      },
      lossFunction: {
        text: "J = Σᵢ Σ_{x∈Cᵢ} ‖x − cᵢ‖²  (inertia / within-cluster sum of squares)",
        mechanism: "Each Lloyd step provably never increases J, so it converges - but only to a local minimum, which is why the result depends on how centroids were initialized.",
        plot: { fn: (k) => 10 / (k + 0.6) + 0.3, domain: [1, 8], color: "var(--accent)", caption: "typical inertia-vs-k 'elbow' curve - pick k near where it stops dropping sharply" },
      },
      optimization: [
        { title: "Lloyd's algorithm", formula: "alternate assign and update until stable", note: "A coordinate-descent scheme: each step optimises one set of variables exactly while holding the other fixed. That is what makes the monotone decrease guaranteed." },
        { title: "Cost", formula: "O(m · k · p) per iteration", note: "Linear in every dimension, which is why k-means scales to very large datasets where hierarchical or density methods do not." },
        { title: "Restarts", formula: "run n_init times, keep the lowest J", note: "The cheapest defence against bad local minima. With k-means++ seeding, 10 restarts is usually plenty." },
        { title: "Mini-batch variant", formula: "update centroids from random subsets of size b", note: "Trades a small amount of quality for a large speedup on datasets too big to scan repeatedly." },
        { title: "Choosing k", formula: "elbow on J(k), or argmax silhouette(k)", note: "J always decreases with k, so it cannot pick k on its own. The elbow is where the marginal gain drops off; silhouette and gap statistic are more principled." },
      ],
      output: "A cluster index for every point, plus the k centroid coordinates.",
      assumptions: [
        { name: "Clusters are roughly spherical", why: "Squared Euclidean distance to a centre implies isotropic, equal-variance groups.", check: "Plot the clusters. Elongated or curved groups will be split wrongly; use GMM or DBSCAN." },
        { name: "Clusters are similarly sized", why: "The objective penalises large clusters, so a big group tends to get split while a small one gets absorbed.", check: "Inspect cluster populations for large imbalances." },
        { name: "Features are scaled", why: "Distance is dominated by the largest-range feature.", check: "Standardize before clustering, always." },
        { name: "k is known or discoverable", why: "The algorithm cannot decide how many clusters exist.", check: "Elbow plot, silhouette scan, or gap statistic." },
        { name: "Mean is meaningful", why: "The update step averages coordinates, which requires continuous numeric features.", check: "For categorical data use k-modes; for arbitrary metrics use k-medoids." },
      ],
      hyperparameters: [
        { name: "k", range: "2 - 50", increasing: "Inertia always falls; clusters become smaller and less meaningful, eventually one per point.", strategy: "Never pick by inertia alone. Use the elbow plus silhouette score, and sanity-check the clusters against domain knowledge." },
        { name: "init", range: "k-means++ / random", increasing: "Not applicable", strategy: "Always k-means++. Random init is meaningfully worse and there is no reason to use it." },
        { name: "n_init", range: "1 - 50", increasing: "Better chance of finding a good local minimum, linear cost increase.", strategy: "10 is the usual default and sufficient with k-means++." },
        { name: "max_iter", range: "100 - 1000", increasing: "More iterations before giving up.", strategy: "300 is almost always enough. Convergence is usually reached in well under 50." },
        { name: "tol", range: "1e-6 - 1e-3", increasing: "Stops earlier, slightly less converged.", strategy: "Rarely worth changing." },
        { name: "algorithm", range: "lloyd / elkan", increasing: "Not applicable", strategy: "Elkan uses the triangle inequality to skip distance computations and is faster on dense, low-dimensional data." },
      ],
      metrics: ["Inertia / WCSS", "Silhouette score", "Davies-Bouldin index", "Calinski-Harabasz index", "Adjusted Rand Index / NMI (if ground-truth labels exist)"],
      typicalUses: ["Customer/market segmentation", "Image color quantization", "Document/topic clustering", "Vector quantization"],
      diagnostics: [
        "Plot inertia against k. The elbow is a heuristic, not a proof, and is often ambiguous.",
        "Silhouette score gives a per-point measure of how well it fits its cluster versus the next nearest. Values near 0 mean the point sits on a boundary.",
        "Run with several random seeds. If the clusters change substantially, the structure is weak or k is wrong.",
        "Check cluster sizes. A cluster with two members usually means k is too high or an outlier captured a centroid.",
        "Project to 2D with PCA and colour by cluster to see whether the partition looks plausible.",
      ],
      advantages: [
        "Very fast and linear in the number of points, so it scales to millions of rows.",
        "Simple to implement, explain, and reason about.",
        "Guaranteed to converge in a finite number of steps.",
        "Centroids are directly interpretable as prototype examples of each group.",
        "Extends naturally to streaming and mini-batch settings.",
      ],
      limitations: [
        { name: "k must be chosen in advance", note: "the algorithm cannot discover the cluster count", fix: "elbow, silhouette, or gap statistic; or use DBSCAN which infers it." },
        { name: "Only convex, isotropic clusters", note: "Voronoi cells cannot represent rings or crescents", fix: "GMM for elliptical shapes, DBSCAN or spectral clustering for arbitrary ones." },
        { name: "Local minima", note: "the result depends on initialisation", fix: "k-means++ seeding with multiple restarts." },
        { name: "Sensitive to outliers", note: "means are dragged by extreme points", fix: "k-medoids, or remove outliers first." },
        { name: "Assumes similar cluster sizes", note: "large clusters get split, small ones absorbed", fix: "GMM with free mixing weights." },
        { name: "Needs scaled features", note: "unscaled data makes one feature dominate distance", fix: "standardize." },
        { name: "Hard assignments only", note: "no notion of a point belonging partly to two clusters", fix: "GMM gives soft responsibilities." },
      ],
      alternatives: [
        { name: "Gaussian mixture model", when: "Clusters overlap, differ in shape or size, or you want membership probabilities." },
        { name: "DBSCAN", when: "Clusters are irregularly shaped, there are outliers, or k is unknown." },
        { name: "Hierarchical clustering", when: "You want a dendrogram and nested structure rather than a flat partition." },
        { name: "Spectral clustering", when: "Clusters are connected but non-convex, such as concentric rings." },
      ],
      pitfalls: [
        { problem: "Clusters look arbitrary", solution: "Features are probably unscaled. Standardize and re-run." },
        { problem: "Different results every run", solution: "Local minima. Use k-means++ with n_init at 10 or more, and fix the seed." },
        { problem: "Elbow plot has no clear elbow", solution: "There may be no natural cluster structure. Check the silhouette score; if it is near 0, clustering is not telling you anything." },
        { problem: "One cluster holds almost every point", solution: "Outliers have captured the other centroids. Remove them or use k-medoids." },
        { problem: "Ring-shaped data is split down the middle", solution: "Fundamental limitation. Use DBSCAN or spectral clustering." },
        { problem: "Picking k by minimum inertia", solution: "Inertia is monotone decreasing in k, so this always selects the maximum k. Use silhouette instead." },
      ],
      quickRef: [
        { name: "Objective", formula: "J = Σⱼ Σ_{x∈Cⱼ} ‖x − cⱼ‖²" },
        { name: "Assignment", formula: "argmin_j ‖x − cⱼ‖²" },
        { name: "Update", formula: "cⱼ = mean(Cⱼ)" },
        { name: "k-means++", formula: "P(x) ∝ D(x)²" },
        { name: "Cost", formula: "O(m·k·p) per iteration" },
        { name: "Silhouette", formula: "s = (b − a) / max(a, b)" },
        { name: "Cluster shape", formula: "Voronoi cells, always convex" },
      ],
      code: `from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

Xs = StandardScaler().fit_transform(X)   # never skip this

# Inertia falls monotonically with k, so choose with silhouette too.
for k in range(2, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    print(k, round(km.inertia_, 1), round(silhouette_score(Xs, labels), 3))

final = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
labels = final.fit_predict(Xs)

# For datasets too large to scan repeatedly:
fast = MiniBatchKMeans(n_clusters=4, batch_size=4096, n_init=10)`,
      whyChain: [
        { q: "Why does k-means always converge?", a: "Both steps weakly decrease the objective, and there are only finitely many ways to partition the points. A monotonically decreasing sequence over a finite set must terminate." },
        { q: "If it always converges, why do restarts help?", a: "Converging is not the same as converging well. The objective has many local minima, and which one you land in is determined entirely by the initial centroids. Restarts sample several basins and keep the best." },
        { q: "Why use squared distance rather than plain Euclidean?", a: "Because the mean is the minimiser of summed squared distance. If you used unsquared distance the correct update would be the geometric median, not the mean, which has no closed form. The squaring is what makes the update step trivial." },
        { q: "What does k-means++ actually do?", a: "It picks each new centroid with probability proportional to the squared distance from the nearest existing centroid, so centres start far apart. This alone turns a method with no approximation guarantee into one with an expected O(log k) bound." },
        { q: "Why can k-means never find a ring-shaped cluster?", a: "Assignment is by nearest centroid, so cluster boundaries are perpendicular bisectors between centres. That makes every cluster a convex Voronoi cell, and a ring is not convex." },
        { q: "Why can't you pick k by minimising inertia?", a: "Inertia decreases monotonically as k grows and reaches zero when every point is its own cluster. It measures fit, not parsimony. Silhouette, the gap statistic, or BIC under a GMM trade fit against complexity." },
        { q: "How does k-means relate to GMM?", a: "K-means is the limiting case of a GMM with spherical, equal-variance components as the variance goes to zero, and with hard rather than soft assignments. EM becomes Lloyd's algorithm." },
        { q: "Why is scaling more critical here than for a tree?", a: "The entire method is distance-based, and squared Euclidean distance sums contributions across features. A feature measured in thousands contributes millions to the sum while one in [0,1] contributes fractions, so it alone determines the clusters." },
      ],
      parameters: [
        { name: "k", effect: "Number of clusters - the main lever, usually chosen via the elbow method or silhouette score." },
        { name: "initialization", effect: "Random vs k-means++ starting centroids - strongly affects which local minimum is found." },
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
        result: "After one iteration: C1=(1, 1.5), C2=(4.5, 4.5) - already close to the final converged centroids",
      },
    },
    mount,
  });
})();
