(() => {
  const { ui, col, fmt } = Lab;
  const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  // Agglomerative clustering. Returns the merge list (scipy-style linkage).
  function agglomerate(pts, linkage) {
    const n = pts.length;
    let clusters = pts.map((p, i) => ({ id: i, members: [i], cx: p.x, cy: p.y }));
    const merges = [];
    const dist = (A, B) => {
      if (linkage === "ward") { const na = A.members.length, nb = B.members.length; return Math.sqrt(((2 * na * nb) / (na + nb))) * Math.hypot(A.cx - B.cx, A.cy - B.cy); }
      let best = linkage === "single" ? Infinity : linkage === "complete" ? -Infinity : 0, pair = null;
      for (const i of A.members) for (const j of B.members) {
        const v = d(pts[i], pts[j]);
        if (linkage === "single" ? v < best : linkage === "complete" ? v > best : false) { best = v; pair = [i, j]; }
        if (linkage === "average") best += v / (A.members.length * B.members.length);
      }
      return linkage === "average" ? best : { v: best, pair };
    };
    let next = n;
    while (clusters.length > 1) {
      let best = null;
      for (let a = 0; a < clusters.length; a++) for (let b = a + 1; b < clusters.length; b++) {
        const r = dist(clusters[a], clusters[b]);
        const v = typeof r === "number" ? r : r.v;
        if (!best || v < best.v) best = { a, b, v, pair: typeof r === "number" ? null : r.pair };
      }
      const A = clusters[best.a], B = clusters[best.b];
      const na = A.members.length, nb = B.members.length;
      const M = { id: next++, members: A.members.concat(B.members), cx: (A.cx * na + B.cx * nb) / (na + nb), cy: (A.cy * na + B.cy * nb) / (na + nb) };
      merges.push({ a: A.id, b: B.id, id: M.id, h: best.v, size: M.members.length, pair: best.pair, ca: [A.cx, A.cy], cb: [B.cx, B.cy], members: M.members.slice() });
      clusters = clusters.filter((_, k) => k !== best.a && k !== best.b).concat([M]);
    }
    return merges;
  }
  // Flat labels when the tree is cut into k clusters (undo the last k-1 merges).
  function cutLabels(n, merges, k, upto = merges.length) {
    const parent = new Map();
    const use = merges.slice(0, Math.max(0, Math.min(upto, merges.length - (k - 1))));
    const find = (x) => { while (parent.has(x)) x = parent.get(x); return x; };
    for (const m of use) { parent.set(m.a, m.id); parent.set(m.b, m.id); }
    const roots = new Map(); const lab = [];
    for (let i = 0; i < n; i++) { const r = find(i); if (!roots.has(r)) roots.set(r, roots.size); lab.push(roots.get(r)); }
    return lab;
  }

  const TEX = {
    single: "d(A,B)=\\min_{a\\in A,b\\in B}\\lVert a-b\\rVert",
    complete: "d(A,B)=\\max_{a\\in A,b\\in B}\\lVert a-b\\rVert",
    average: "d(A,B)=\\frac{1}{|A||B|}\\sum_{a\\in A}\\sum_{b\\in B}\\lVert a-b\\rVert",
    ward: "d(A,B)=\\sqrt{\\tfrac{2|A||B|}{|A|+|B|}}\\,\\lVert\\mu_A-\\mu_B\\rVert",
  };

  function buildFrames(pts, linkage) {
    const n = pts.length;
    if (n < 2) return { frames: [{ phase: "idle", title: "Add at least two points", text: "", st: { step: 0 } }], merges: [] };
    const merges = agglomerate(pts, linkage);
    const frames = [{ phase: "init", major: true, arch: ["single", "dist"], title: `Start: ${n} singleton clusters`, text: "Agglomerative clustering starts with every point as its own cluster and repeatedly merges the two closest clusters. 'Closest' is defined by the linkage rule.", tex: TEX[linkage], st: { step: 0 }, m: { h: 0 } }];
    const every = Math.max(1, Math.floor(n / 45));
    const sizeOf = (id) => (id < n ? 1 : merges.find((x) => x.id === id).size);
    merges.forEach((m, k) => {
      const left = n - k - 1;
      if (k % every && left > 12 && k !== merges.length - 1) return;
      frames.push({ phase: "merge", major: left === 1, arch: ["dist", "pick", "merge", "update"], title: `Merge ${k + 1}: clusters of size ${sizeOf(m.a)} + ${sizeOf(m.b)} at height ${m.h.toFixed(2)}`, text: `The closest pair under <b>${linkage}</b> linkage is joined; ${left} cluster${left === 1 ? "" : "s"} remain. The merge appears in the dendrogram at its linkage distance. Big jumps in merge height suggest natural places to cut.`, tex: TEX[linkage], st: { step: k + 1 }, m: { h: m.h } });
    });
    frames.push({ phase: "cut", major: true, arch: ["tree", "cut"], title: "Cut the dendrogram to get flat clusters", text: "The full merge history is a tree. Any horizontal cut gives a flat clustering: choose k with the slider and watch where the line falls. Single linkage chains along thin bridges (good for moons, bad for noise); Ward favours compact, equal-sized blobs.", st: { step: merges.length, cut: true }, m: { h: merges[merges.length - 1].h } });
    return { frames, merges };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "hc", fps: 4,
      legend: ui.legend([["var(--series-4)", "latest merge", "line"], ["var(--text-faint)", "cluster hull", "ring"]]),
      tools: ui.btn("replay", "replay", true),
      hint: "colours show the clusters you get at the current cut · click to add points · drag / double-click to edit",
      data: { kind: "cluster", presets: ["blobs3", "moons", "varied", "blobs4", "aniso", "outliers"], def: "blobs3", n: 40, noise: 0.3, nRange: [6, 90] },
      side: [{ title: "model", html: ui.select("link", "linkage", [["ward", "Ward (min variance increase)"], ["single", "single (nearest pair)"], ["complete", "complete (farthest pair)"], ["average", "average (UPGMA)"]], "ward") + ui.range("k", "cut into k clusters", 1, 10, 1, 3) + ui.check("hull", "draw cluster hulls", true) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "single", label: "n singletons", x: 60, y: 40, kind: "data" },
          { id: "dist", label: "linkage d(A,B)", sub: "all cluster pairs", x: 190, y: 40, kind: "op" },
          { id: "pick", label: "closest pair", x: 318, y: 40, kind: "loss" },
          { id: "merge", label: "merge", sub: "record height", x: 318, y: 110, kind: "op" },
          { id: "update", label: "update", sub: "Lance–Williams", x: 190, y: 110, kind: "op" },
          { id: "tree", label: "dendrogram", sub: "full hierarchy", x: 72, y: 110, kind: "out" },
          { id: "cut", label: "cut at k", sub: "flat labels", x: 72, y: 168, kind: "param" },
        ],
        edges: [{ from: "single", to: "dist" }, { from: "dist", to: "pick" }, { from: "pick", to: "merge" }, { from: "merge", to: "update" }, { from: "update", to: "dist", dash: true, label: "until 1 left" }, { from: "update", to: "tree" }, { from: "tree", to: "cut" }],
      },
      phases: { init: { label: "singletons", color: 3 }, merge: { label: "merge", color: 0 }, cut: { label: "cut", color: 2 } },
      panels: [{ id: "dendro", title: "dendrogram", sub: "dashed line = current cut", wide: true }],
      charts: [{ id: "h", title: "merge height per step", series: [{ key: "h", label: "linkage distance", color: "var(--series-5)" }] }],
      onFrame: render, onData: () => replay(),
    });
    lab.editable({ newPoint: (x, y) => ({ x, y, label: 0 }), onChange: () => replay({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    let merges = [];
    function replay({ at = "start" } = {}) {
      const r = buildFrames(lab.points, lab.val("link"));
      merges = r.merges;
      lab.load(r.frames, { at, play: at === "start" });
    }
    const hullG = L.mid.append("g"), mergeG = L.top.append("g");
    const dEl = lab.R("panel-dendro");

    function dendrogram(step, k) {
      dEl.innerHTML = "";
      const n = lab.points.length; if (n < 2) return;
      // leaf order from the final tree
      const kids = new Map(merges.map((m) => [m.id, [m.a, m.b]]));
      const order = [];
      const walk = (id) => { if (id < n) order.push(id); else kids.get(id).forEach(walk); };
      walk(merges[merges.length - 1].id);
      const pos = new Map(order.map((id, i) => [id, i]));
      const H = Math.max(...merges.map((m) => m.h)) || 1;
      const W = 720, Ht = 210, ml = 34, mb = 14;
      const svg = d3.select(dEl).append("svg").attr("viewBox", `0 0 ${W} ${Ht}`).attr("class", "xy-svg");
      const x = d3.scaleLinear().domain([0, n - 1]).range([ml, W - 10]);
      const y = d3.scaleLinear().domain([0, H * 1.05]).range([Ht - mb, 8]);
      for (const t of y.ticks(4)) { svg.append("line").attr("class", "ax-grid").attr("x1", ml).attr("x2", W - 10).attr("y1", y(t)).attr("y2", y(t)).attr("stroke", "var(--border-soft)"); svg.append("text").attr("class", "ax-tick").attr("x", ml - 4).attr("y", y(t) + 3).attr("text-anchor", "end").attr("fill", "var(--text-faint)").style("font-size", "9px").text(+t.toFixed(1)); }
      const labels = cutLabels(n, merges, k);
      const height = new Map(), xpos = new Map(pos);
      merges.forEach((m, i) => { height.set(m.id, m.h); xpos.set(m.id, (xpos.get(m.a) + xpos.get(m.b)) / 2); });
      const hOf = (id) => (id < n ? 0 : height.get(id));
      const cutH = k > 1 && merges.length >= k - 1 ? (merges[merges.length - k].h + merges[merges.length - k + 1].h) / 2 : H * 1.02;
      merges.slice(0, step).forEach((m, i) => {
        const latest = i === step - 1;
        const leafA = m.members[0];
        const c = m.h <= cutH ? col(labels[leafA]) : "var(--text-faint)";
        svg.append("path").attr("d", `M${x(xpos.get(m.a))},${y(hOf(m.a))}V${y(m.h)}H${x(xpos.get(m.b))}V${y(hOf(m.b))}`).attr("fill", "none").attr("stroke", latest ? "var(--series-4)" : c).attr("stroke-width", latest ? 2.6 : 1.4);
      });
      order.forEach((id, i) => svg.append("circle").attr("cx", x(i)).attr("cy", y(0)).attr("r", 2.4).attr("fill", col(labels[id])));
      if (step >= merges.length && k > 1) svg.append("line").attr("x1", ml).attr("x2", W - 10).attr("y1", y(cutH)).attr("y2", y(cutH)).attr("stroke", "var(--accent)").attr("stroke-dasharray", "6 4").attr("stroke-width", 1.6);
    }

    function render(f) {
      const st = f.st, n = lab.points.length;
      hullG.selectAll("*").remove(); mergeG.selectAll("*").remove();
      if (!merges.length) { lab.drawPoints({}); return; }
      const k = lab.val("k");
      const finalLab = cutLabels(n, merges, k);
      // current clusters at this step
      const parent = new Map();
      for (const m of merges.slice(0, st.step)) { parent.set(m.a, m.id); parent.set(m.b, m.id); }
      const find = (x) => { while (parent.has(x)) x = parent.get(x); return x; };
      const groups = new Map();
      for (let i = 0; i < n; i++) { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(i); }
      if (lab.val("hull")) for (const [, mem] of groups) {
        if (mem.length < 2) continue;
        const ptsXY = mem.map((i) => [pl.x(lab.points[i].x), pl.y(lab.points[i].y)]);
        const hull = mem.length >= 3 ? d3.polygonHull(ptsXY) : ptsXY;
        if (!hull) continue;
        hullG.append("path").attr("d", (mem.length >= 3 ? "M" + hull.join("L") + "Z" : `M${hull[0]}L${hull[1]}`)).attr("fill", col(finalLab[mem[0]])).attr("fill-opacity", 0.08).attr("stroke", col(finalLab[mem[0]])).attr("stroke-opacity", 0.55).attr("stroke-width", 1.4).attr("stroke-linejoin", "round");
      }
      const last = st.step > 0 && !st.cut ? merges[st.step - 1] : null;
      if (last) {
        const [p, q] = last.pair ? [lab.points[last.pair[0]], lab.points[last.pair[1]]].map((o) => [o.x, o.y]) : [last.ca, last.cb];
        mergeG.append("line").attr("x1", pl.x(p[0])).attr("y1", pl.y(p[1])).attr("x2", pl.x(q[0])).attr("y2", pl.y(q[1])).attr("stroke", "var(--series-4)").attr("stroke-width", 3).attr("stroke-linecap", "round");
      }
      const memLast = last ? new Set(last.members) : null;
      lab.drawPoints({ color: (dd, i) => col(finalLab[i]), stroke: (dd, i) => (memLast && memLast.has(i) ? "var(--series-4)" : "var(--bg)"), sw: (dd, i) => (memLast && memLast.has(i) ? 2.4 : 1.1) });
      dendrogram(st.step, k);
      lab.readout(`<div class="kv"><span>points</span><b>${n}</b><span>clusters now</span><b>${groups.size}</b><span>merges done</span><b>${st.step} / ${merges.length}</b><span>cut height</span><b>${k > 1 && merges.length >= k ? fmt((merges[merges.length - k].h + merges[merges.length - k + 1].h) / 2, 2) : "–"}</b></div>`);
    }
    lab.watch(["link"], () => replay());
    lab.watch(["k", "hull"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.loadPreset("blobs3");
    return lab.destroy;
  }

  MLApp.register({
    id: "hierarchical",
    name: "Hierarchical Clustering",
    category: "Unsupervised - Clustering",
    tagline: "agglomerative merges → dendrogram",
    isNew: true,
    description: "Start with every point alone and keep merging the two closest clusters until one remains. Watch merges happen on the map while the dendrogram grows beside it, compare single, complete, average and Ward linkage, and cut the tree at any k.",
    info: {
      type: "Unsupervised - Clustering. Agglomerative (bottom-up) hierarchical, deterministic.",
      scenario: "Small to medium datasets where the nesting of clusters matters, k is unknown, or a dendrogram is a useful deliverable (taxonomies, gene expression, document hierarchies).",
      inputs: "Points (or any pairwise distance matrix) and a linkage criterion.",
      output: "A full merge tree (dendrogram); flat clusters for any chosen cut height or k.",
      intuition: {
        definition: "Treat every point as a cluster. Find the two clusters that are closest, merge them, and record how far apart they were. Repeat until everything is one cluster. The record of merges is a tree; cutting it at a height gives clusters.",
        steps: ["Compute pairwise distances between all points.", "Merge the closest pair of clusters under the linkage rule.", "Update distances from the new cluster to all others.", "Repeat n − 1 times; draw the dendrogram; cut it."],
        applications: ["Phylogenetic trees and gene-expression heatmaps", "Customer or product taxonomies", "Document and topic hierarchies", "Image segmentation with connectivity constraints"],
      },
      math: [
        { title: "Single linkage", formula: "d(A,B) = min_{a∈A,b∈B} ‖a − b‖", note: "Equivalent to a minimum spanning tree. Finds elongated, chained clusters but suffers from chaining through noise." },
        { title: "Complete linkage", formula: "d(A,B) = max_{a∈A,b∈B} ‖a − b‖", note: "Favours compact clusters of similar diameter; sensitive to outliers." },
        { title: "Average linkage (UPGMA)", formula: "d(A,B) = (1/|A||B|) Σ_a Σ_b ‖a − b‖", note: "A compromise between single and complete." },
        { title: "Ward linkage", formula: "Δ(A,B) = (|A||B|/(|A|+|B|)) ‖μ_A − μ_B‖²", note: "Merges the pair that increases total within-cluster variance least: the hierarchical cousin of k-means." },
        { title: "Lance–Williams update", formula: "d(k, i∪j) = αᵢd(k,i) + αⱼd(k,j) + βd(i,j) + γ|d(k,i) − d(k,j)|", note: "All four linkages are special cases, so distances to a merged cluster are updated in O(n) without revisiting points." },
        { title: "Cophenetic distance", formula: "c(x, y) = height at which x and y first merge", note: "The cophenetic correlation with the original distances measures how faithfully the dendrogram preserves them." },
      ],
      pipeline: [{ label: "Distances" }, { label: "Merge closest", note: "linkage" }, { label: "Update", note: "Lance–Williams" }, { label: "Dendrogram" }, { label: "Cut", note: "k or height", accent: "green" }],
      decisionFunction: { text: "labels = cut(dendrogram, height h or k clusters)", mechanism: "There is no model to apply to new points: the result is a partition of the training data. A new point can be assigned to the nearest cluster after the fact." },
      lossFunction: { text: "Greedy: at each step merge argmin_{A,B} d(A,B)   (Ward: minimise the increase in Σ within-cluster SSE)", mechanism: "There is no global objective being optimised; each merge is locally optimal and never undone. Ward's criterion is the closest to an explicit loss (it greedily minimises the k-means objective)." },
      optimization: [
        { title: "Naive algorithm", formula: "O(n³) time, O(n²) memory", note: "Recompute the closest pair each step." },
        { title: "Priority queues / NN-chain", formula: "O(n²) time", note: "The nearest-neighbour chain algorithm achieves O(n²) for Ward, complete and average linkage." },
        { title: "Single linkage via MST", formula: "O(n²) or better with spatial indexes", note: "Kruskal's algorithm on the distance graph." },
      ],
      assumptions: [
        { name: "A meaningful distance", why: "Everything is driven by pairwise distances.", check: "Scale features; pick a metric that fits the data." },
        { name: "Nested structure is plausible", why: "The output is a strict hierarchy.", check: "Check the cophenetic correlation." },
        { name: "Small enough n", why: "O(n²) memory for the distance matrix.", check: "Beyond ~20k points use BIRCH, mini-batch k-means, or HDBSCAN." },
      ],
      hyperparameters: [
        { name: "linkage", range: "ward / complete / average / single", increasing: "Not applicable", strategy: "Ward for compact blobs, single for chains and arbitrary shapes, average as a robust default." },
        { name: "metric", range: "euclidean / manhattan / cosine …", increasing: "Not applicable", strategy: "Ward requires Euclidean. Cosine for text embeddings." },
        { name: "n_clusters or distance_threshold", range: "k ≥ 1 or h > 0", increasing: "More (or fewer) clusters.", strategy: "Look for the largest vertical gap in the dendrogram." },
        { name: "connectivity", range: "k-NN graph", increasing: "Restricts merges to neighbours.", strategy: "Use for spatial or image data to get contiguous clusters." },
      ],
      metrics: ["Cophenetic correlation", "Silhouette score at the chosen cut", "Adjusted Rand Index (if labels exist)"],
      typicalUses: ["Dendrograms for exploration", "Clustered heatmaps", "Taxonomy building"],
      diagnostics: ["Read the merge-height plot: a sharp jump means a natural cut just below it.", "Compare linkages: if they agree, the structure is robust.", "Watch for chaining with single linkage: one giant cluster plus singletons."],
      advantages: ["No k needed up front; one run gives every k.", "Deterministic: no random initialisation.", "Dendrogram is interpretable and shows nesting.", "Works with any distance or similarity."],
      limitations: [{ name: "Quadratic memory, cubic time naive", note: "does not scale past tens of thousands of points", fix: "BIRCH, HDBSCAN, or k-means." }, { name: "Greedy and irreversible", note: "an early bad merge is never undone", fix: "try several linkages." }, { name: "Linkage choice changes everything", note: "single chains, complete splits large clusters", fix: "match linkage to expected cluster shape." }],
      alternatives: [{ name: "K-Means", when: "Large n and roughly spherical clusters." }, { name: "DBSCAN / HDBSCAN", when: "Arbitrary shapes with noise; HDBSCAN also gives a hierarchy." }, { name: "GMM", when: "You want soft, probabilistic membership." }],
      pitfalls: [{ problem: "One huge cluster and many singletons", solution: "Single-linkage chaining. Use Ward or average." }, { problem: "Out of memory", solution: "The distance matrix is n². Subsample or use a scalable method." }, { problem: "Unscaled features", solution: "Standardise before computing distances." }],
      quickRef: [{ name: "Single", formula: "min pairwise distance" }, { name: "Complete", formula: "max pairwise distance" }, { name: "Average", formula: "mean pairwise distance" }, { name: "Ward", formula: "min increase in SSE" }, { name: "Cost", formula: "O(n²) memory, O(n²)–O(n³) time" }],
      code: `from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

Z = linkage(X, method="ward")            # (n-1) x 4 merge table
c, _ = cophenet(Z, pdist(X))             # how well the tree preserves distances
labels = fcluster(Z, t=3, criterion="maxclust")
dendrogram(Z)

from sklearn.cluster import AgglomerativeClustering
AgglomerativeClustering(n_clusters=3, linkage="average").fit_predict(X)`,
      whyChain: [
        { q: "Why does single linkage chain?", a: "A merge only needs one close pair of points. A thin line of noise points provides a bridge between otherwise distant groups, so they merge early into one long cluster." },
        { q: "Why is Ward similar to k-means?", a: "Ward merges the pair that least increases the within-cluster sum of squares, which is exactly the k-means objective. It is a greedy hierarchical way of reducing it." },
        { q: "How do you choose where to cut?", a: "Look for the largest jump in merge height: merges below it join close points, the one above it would join genuinely separate groups. Silhouette at candidate cuts gives a quantitative check." },
        { q: "Why can't you predict on new points directly?", a: "The method outputs a partition of the training points, not a function. You can assign new points to the cluster of their nearest neighbour or centroid, but that is a separate step." },
      ],
      workedExample: {
        setup: "Points on a line at 0, 1, 5, 6, 20 with single linkage.",
        steps: ["Closest pair: {0},{1} at distance 1 (or {5},{6}, tie). Merge {0,1}.", "Merge {5,6} at distance 1.", "Next closest: {0,1} to {5,6} at 5 − 1 = 4. Merge.", "Finally {0,1,5,6} with {20} at 20 − 6 = 14."],
        result: "Heights 1, 1, 4, 14: the big jump from 4 to 14 suggests 2 clusters",
      },
    },
    mount,
  });
})();
