(() => {
  const { ui, col, fmt } = Lab;

  function shiftOnce(x, pts, bw, kernel) {
    let sx = 0, sy = 0, sw = 0;
    for (const p of pts) {
      const d2 = (p.x - x[0]) ** 2 + (p.y - x[1]) ** 2;
      const w = kernel === "flat" ? (d2 <= bw * bw ? 1 : 0) : Math.exp(-d2 / (2 * bw * bw));
      sx += w * p.x; sy += w * p.y; sw += w;
    }
    return sw ? [sx / sw, sy / sw] : x.slice();
  }
  function kde(x, y, pts, bw) { let s = 0; for (const p of pts) s += Math.exp(-((p.x - x) ** 2 + (p.y - y) ** 2) / (2 * bw * bw)); return s / Math.max(1, pts.length); }

  function buildFrames(pts, bw, kernel, maxIt) {
    if (!pts.length) return [{ phase: "idle", title: "No data", text: "", st: { pos: [] } }];
    let pos = pts.map((p) => [p.x, p.y]);
    const frames = [{ phase: "init", major: true, arch: ["data", "seeds"], title: "Every point starts as a seed", text: `Mean shift assumes the data was drawn from a density with hills (modes). Each seed will climb uphill until it reaches the top of its hill. Bandwidth h = <b>${bw.toFixed(2)}</b> sets the kernel size, and therefore how many hills there are.`, tex: "\\hat f(x)=\\frac1n\\sum_i K\\!\\left(\\frac{x-x_i}{h}\\right)", st: { pos: pos.map((q) => q.slice()) }, m: { shift: null } }];
    for (let it = 1; it <= maxIt; it++) {
      const next = pos.map((q) => shiftOnce(q, pts, bw, kernel));
      const shift = Math.max(...next.map((q, i) => Math.hypot(q[0] - pos[i][0], q[1] - pos[i][1])));
      frames.push({ phase: "shift", major: it === 1, arch: ["seeds", "kernel", "mean", "move"], title: `Iteration ${it}: every seed jumps to the weighted mean of its window`, text: `The mean-shift vector m(x) − x points up the density gradient (the background contours). Largest move this round: <b>${shift.toFixed(3)}</b>. The circle shows one seed's kernel window.`, tex: "m(x)=\\frac{\\sum_i K(x-x_i)\\,x_i}{\\sum_i K(x-x_i)},\\qquad x\\leftarrow m(x)", st: { pos: next.map((q) => q.slice()), prev: pos }, m: { shift } });
      pos = next;
      if (shift < 1e-3) break;
    }
    // merge converged seeds into modes
    const modes = [], lab = [];
    for (const q of pos) {
      let k = modes.findIndex((m) => Math.hypot(m[0] - q[0], m[1] - q[1]) < bw / 2);
      if (k < 0) { modes.push(q.slice()); k = modes.length - 1; }
      lab.push(k);
    }
    frames.push({ phase: "merge", major: true, arch: ["move", "modes", "out"], title: `Converged: ${modes.length} mode${modes.length === 1 ? "" : "s"} → ${modes.length} cluster${modes.length === 1 ? "" : "s"}`, text: "Seeds that ended on the same peak (within h/2) are merged; each original point takes the label of the mode it climbed to. The number of clusters was never specified: it came from the bandwidth.", st: { pos: pos.map((q) => q.slice()), modes, lab }, m: { shift: 0 } });
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "ms", fps: 3,
      legend: ui.legend([["var(--text-faint)", "trajectory", "line"], ["var(--accent)", "mode ★"], ["var(--series-4)", "kernel window", "ring"]]),
      tools: ui.btn("replay", "replay", true),
      hint: "click a point to watch its kernel window · click empty space to add a point · drag / double-click to edit",
      data: { kind: "cluster", presets: ["blobs3", "varied", "blobs4", "moons", "aniso", "outliers"], def: "blobs3", n: 90, noise: 0.35, nRange: [10, 200] },
      side: [{ title: "model", html: ui.range("bw", "bandwidth h", 0.4, 5, 0.1, 2) + ui.select("kern", "kernel", [["gauss", "Gaussian"], ["flat", "flat (uniform disc)"]], "gauss") + ui.range("it", "max iterations", 5, 60, 1, 30) + ui.check("kde", "density contours", true) + ui.check("trail", "trajectories", true) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "data", label: "points xᵢ", x: 50, y: 40, kind: "data" },
          { id: "seeds", label: "seeds", sub: "start at each xᵢ", x: 50, y: 112, kind: "param" },
          { id: "kernel", label: "kernel K", sub: "bandwidth h", x: 170, y: 40, kind: "op" },
          { id: "mean", label: "weighted mean", sub: "m(x)", x: 300, y: 40, kind: "op" },
          { id: "move", label: "x ← m(x)", sub: "climb density", x: 300, y: 112, kind: "op" },
          { id: "modes", label: "merge modes", sub: "within h/2", x: 170, y: 168, kind: "loss" },
          { id: "out", label: "labels", sub: "mode per point", x: 310, y: 168, kind: "out" },
        ],
        edges: [{ from: "data", to: "seeds" }, { from: "data", to: "kernel" }, { from: "seeds", to: "kernel" }, { from: "kernel", to: "mean" }, { from: "mean", to: "move" }, { from: "move", to: "kernel", dash: true, label: "repeat" }, { from: "move", to: "modes" }, { from: "modes", to: "out" }],
      },
      phases: { init: { label: "seeds", color: 3 }, shift: { label: "shift", color: 0 }, merge: { label: "merge modes", color: 2 } },
      charts: [{ id: "shift", title: "largest shift per iteration", series: [{ key: "shift", label: "max ‖m(x) − x‖", color: "var(--series-1)" }], logY: true }],
      onFrame: render, onData: () => replay(),
    });
    let watched = 0;
    lab.editable({ newPoint: (x, y) => ({ x, y, label: 0 }), onChange: () => replay({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    let frames = [];
    function replay({ at = "start" } = {}) { frames = buildFrames(lab.points, lab.val("bw"), lab.val("kern"), lab.val("it")); lab.load(frames, { at, play: at === "start" }); }
    const trailG = L.mid.append("g"), seedG = L.top.append("g"), winG = L.mid.append("g");
    function render(f, i) {
      const st = f.st, bw = lab.val("bw");
      if (!st.pos.length) { lab.drawPoints({}); return; }
      if (lab.val("kde")) {
        const grid = Lab.sampleGrid(pl, (x, y) => kde(x, y, lab.points, bw), 70);
        const hi = Math.max(...grid.values);
        L.bg.selectAll("*").remove();
        Lab.contour(L.under, pl, grid, d3.range(hi * 0.08, hi, hi / 9), { cls: "kde", width: 0.9, stroke: "var(--text-faint)" });
      } else L.under.selectAll("path.kde").remove();
      trailG.selectAll("*").remove();
      if (lab.val("trail")) {
        const hist = frames.slice(0, i + 1).map((fr) => fr.st.pos);
        lab.points.forEach((p, k) => {
          trailG.append("path").attr("d", d3.line().x((q) => pl.x(q[0])).y((q) => pl.y(q[1]))(hist.map((h) => h[k]))).attr("fill", "none").attr("stroke", st.lab ? col(st.lab[k]) : "var(--text-faint)").attr("stroke-opacity", k === watched ? 0.95 : 0.35).attr("stroke-width", k === watched ? 2 : 1);
        });
      }
      lab.drawPoints({ color: (d, k) => (st.lab ? col(st.lab[k]) : "var(--text-faint)"), opacity: 0.55, r: 3.4 }).on("click.watch", (ev, d) => { ev.stopPropagation(); watched = lab.points.indexOf(d); lab.player.redraw(); });
      const seeds = seedG.selectAll("circle").data(st.pos);
      seeds.exit().remove();
      const se = seeds.enter().append("circle").attr("r", 2.6).attr("pointer-events", "none").merge(seeds).attr("fill", (q, k) => (st.lab ? col(st.lab[k]) : "var(--accent)"));
      if (st.prev) se.attr("cx", (q, k) => pl.x(st.prev[k][0])).attr("cy", (q, k) => pl.y(st.prev[k][1])).transition().duration(320).attr("cx", (q) => pl.x(q[0])).attr("cy", (q) => pl.y(q[1]));
      else se.attr("cx", (q) => pl.x(q[0])).attr("cy", (q) => pl.y(q[1]));
      winG.selectAll("*").remove();
      const w = (st.prev || st.pos)[watched];
      if (w) {
        const r = lab.val("kern") === "flat" ? bw : bw * 2;
        winG.append("ellipse").attr("cx", pl.x(w[0])).attr("cy", pl.y(w[1])).attr("rx", pl.x(r) - pl.x(0)).attr("ry", pl.y(0) - pl.y(r)).attr("fill", "var(--series-4)").attr("fill-opacity", 0.07).attr("stroke", "var(--series-4)").attr("stroke-width", 1.6).attr("stroke-dasharray", "5 4");
        const t = st.pos[watched];
        winG.append("line").attr("x1", pl.x(w[0])).attr("y1", pl.y(w[1])).attr("x2", pl.x(t[0])).attr("y2", pl.y(t[1])).attr("stroke", "var(--series-4)").attr("stroke-width", 2.4);
      }
      seedG.selectAll("path.mode").remove();
      if (st.modes) for (const [k, m] of st.modes.entries()) seedG.append("path").attr("class", "mode").attr("d", d3.symbol(d3.symbolStar, 230)()).attr("transform", `translate(${pl.x(m[0])},${pl.y(m[1])})`).attr("fill", col(k)).attr("stroke", "var(--bg)").attr("stroke-width", 1.5);
      lab.readout(`<div class="kv"><span>points</span><b>${lab.points.length}</b><span>bandwidth h</span><b>${bw.toFixed(2)}</b><span>modes</span><b>${st.modes ? st.modes.length : "…"}</b><span>last shift</span><b>${fmt(f.m && f.m.shift, 4)}</b></div>`);
    }
    lab.watch(["bw", "kern", "it"], () => replay(), { format: { bw: (v) => (+v).toFixed(1) } });
    lab.watch(["kde", "trail"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.loadPreset("blobs3");
    return lab.destroy;
  }

  MLApp.register({
    id: "mean-shift",
    name: "Mean Shift",
    category: "Unsupervised - Clustering",
    tagline: "hill-climbing to density modes",
    isNew: true,
    description: "Every point climbs the kernel density estimate until it reaches a peak; points that share a peak share a cluster. Watch the seeds flow uphill along their trajectories, and see how the bandwidth alone decides how many clusters exist.",
    info: {
      type: "Unsupervised - Clustering (mode seeking). Non-parametric, density-based, no k required.",
      scenario: "Low-dimensional data with blob-like clusters of unknown number; image segmentation and object tracking (CamShift).",
      inputs: "Points and a kernel bandwidth h.",
      output: "Cluster modes (density peaks) and a label per point.",
      intuition: {
        definition: "Imagine the points as sand; smoothing it with a kernel gives a landscape of hills. Place a marble on every point and let it roll uphill: each marble ends on a hilltop. Marbles on the same hilltop form a cluster.",
        steps: ["Place a seed at every data point.", "Move each seed to the kernel-weighted mean of the points around it.", "Repeat until seeds stop moving.", "Merge seeds that converged to the same peak; label points by their peak."],
        applications: ["Image segmentation in joint colour-space coordinates", "Visual object tracking (mean-shift / CamShift)", "Finding hotspots in geographic data", "Mode finding in density estimation"],
      },
      math: [
        { title: "Kernel density estimate", formula: "f̂(x) = (1/nhᵈ) Σᵢ K((x − xᵢ)/h)", note: "Smooths the empirical distribution; h controls smoothness." },
        { title: "Mean-shift vector", formula: "m(x) − x = Σ K(x − xᵢ) xᵢ / Σ K(x − xᵢ) − x", note: "Proportional to ∇f̂(x)/f̂(x), so it always points uphill on the density." },
        { title: "Update", formula: "x ← m(x)", note: "A gradient ascent with an adaptive step size: large steps in sparse regions, small near the peak." },
        { title: "Convergence", formula: "f̂(x⁽ᵗ⁺¹⁾) ≥ f̂(x⁽ᵗ⁾)", note: "For convex, monotonically decreasing kernel profiles the density along the trajectory never decreases." },
      ],
      pipeline: [{ label: "Seeds = points" }, { label: "Kernel window" }, { label: "Weighted mean" }, { label: "Shift, repeat" }, { label: "Merge modes", accent: "green" }],
      decisionFunction: { text: "label(x) = mode reached by iterating x ← m(x)", mechanism: "Basins of attraction of the density peaks partition the space; a new point is labelled by running mean shift from it." },
      lossFunction: { text: "maximise f̂(x) locally for each seed", mechanism: "Each seed performs gradient ascent on the kernel density estimate. There is no global loss over the partition." },
      optimization: [
        { title: "Cost", formula: "O(T · n²) naive", note: "Every seed looks at every point each iteration." },
        { title: "Bin seeding", formula: "seeds on a coarse grid", note: "Far fewer seeds than points; sklearn's bin_seeding." },
        { title: "Bandwidth estimation", formula: "quantile of pairwise distances", note: "sklearn's estimate_bandwidth picks h from nearest-neighbour distances." },
      ],
      assumptions: [{ name: "Clusters are density peaks", why: "Each cluster must have its own mode.", check: "Plot the KDE; flat or multi-peaked clusters split or merge." }, { name: "One bandwidth fits all", why: "A single h is used everywhere.", check: "Varied densities need adaptive bandwidths." }, { name: "Low dimension", why: "Kernel density estimation degrades quickly with dimension.", check: "Reduce dimension first." }],
      hyperparameters: [{ name: "bandwidth h", range: "data-dependent", increasing: "Smoother density, fewer and larger clusters.", strategy: "Start from estimate_bandwidth, then adjust by eye or silhouette." }, { name: "kernel", range: "flat / Gaussian", increasing: "Not applicable", strategy: "Gaussian gives smoother trajectories; flat is faster." }, { name: "bin_seeding", range: "on / off", increasing: "Not applicable", strategy: "On for large n." }],
      metrics: ["Silhouette score", "Number of modes vs bandwidth (stability)", "ARI / NMI if labels exist"],
      typicalUses: ["Image segmentation", "Tracking", "Unknown-k clustering in 2-3D"],
      diagnostics: ["Sweep h and plot the number of modes: plateaus indicate stable structure.", "Check for singleton modes: outliers become their own clusters with small h."],
      advantages: ["No k required.", "Arbitrary cluster shapes along density ridges.", "One intuitive parameter.", "Robust to outliers in the sense that they do not move modes much."],
      limitations: [{ name: "Slow", note: "quadratic per iteration", fix: "bin seeding, KD-trees, subsampling." }, { name: "Bandwidth sensitive", note: "h determines everything", fix: "estimate it; check stability." }, { name: "Poor in high dimensions", note: "KDE suffers from the curse of dimensionality", fix: "PCA first." }],
      alternatives: [{ name: "DBSCAN / HDBSCAN", when: "Arbitrary shapes, noise labelling, larger data." }, { name: "GMM", when: "You want a parametric density with soft labels." }, { name: "K-Means", when: "k is known and speed matters." }],
      pitfalls: [{ problem: "Everything becomes one cluster", solution: "Bandwidth too large; reduce h." }, { problem: "Dozens of tiny clusters", solution: "Bandwidth too small; increase h or merge small modes." }, { problem: "Very slow", solution: "Use bin seeding or a subsample." }],
      quickRef: [{ name: "KDE", formula: "f̂(x) = (1/n) Σ K((x − xᵢ)/h)" }, { name: "Shift", formula: "m(x) = ΣKxᵢ / ΣK" }, { name: "Update", formula: "x ← m(x)" }],
      code: `from sklearn.cluster import MeanShift, estimate_bandwidth

h = estimate_bandwidth(X, quantile=0.2)
ms = MeanShift(bandwidth=h, bin_seeding=True).fit(X)
labels, modes = ms.labels_, ms.cluster_centers_`,
      whyChain: [
        { q: "Why does the mean-shift vector point uphill?", a: "Differentiating the kernel density estimate gives a term proportional to (weighted mean − x). So moving to the weighted mean is a gradient ascent step with an automatically scaled step size." },
        { q: "How does bandwidth relate to k?", a: "The number of modes of a KDE decreases as h grows. Choosing h is effectively choosing the scale at which you look for clusters." },
        { q: "How is mean shift related to k-means?", a: "With a flat kernel and seeds at centroids, each update is a local mean, similar to k-means' update step, but mean shift does not fix k and lets seeds merge." },
      ],
      workedExample: {
        setup: "1-D points 0, 1, 2, 10, 11 with a flat kernel of radius 3, seed at 2.",
        steps: ["Window around 2 covers 0, 1, 2 → mean 1.", "Window around 1 covers 0, 1, 2 → mean 1: converged.", "Seed at 10: window covers 10, 11 → mean 10.5, stays."],
        result: "Two modes, at 1 and 10.5: two clusters",
      },
    },
    mount,
  });
})();
