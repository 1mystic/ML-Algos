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
            <div class="note">Lloyd's algorithm: assign each point to its closest centroid, then move each centroid to the mean of its assigned points, repeat - the same loop as <code>mla/kmeans.py</code>.</div>
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
    category: "Unsupervised - Clustering",
    tagline: "Lloyd's algorithm, step-through",
    description: "Hard clustering via alternating assignment and centroid-averaging. Step through iterations one at a time to see exactly how the centroids converge.",
    sourceFile: "mla/kmeans.py",
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
