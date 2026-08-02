(() => {
  const W = 640, H = 460, PAD = 36;

  function pairwiseSq(X) {
    const n = X.length;
    const D = MLU.zeros(n, n);
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        let s = 0; for (let k = 0; k < X[i].length; k++) s += (X[i][k] - X[j][k]) ** 2;
        D[i][j] = D[j][i] = s;
      }
    return D;
  }
  function computeP(D, perplexity) {
    const n = D.length;
    const P = MLU.zeros(n, n);
    const target = Math.log(perplexity);
    for (let i = 0; i < n; i++) {
      let betaMin = -Infinity, betaMax = Infinity, beta = 1;
      for (let iter = 0; iter < 50; iter++) {
        let sumP = 0, H = 0;
        const row = new Array(n).fill(0);
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          row[j] = Math.exp(-D[i][j] * beta);
          sumP += row[j];
        }
        sumP = sumP || 1e-12;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const pj = row[j] / sumP;
          if (pj > 1e-12) H -= pj * Math.log(pj);
        }
        const diff = H - target;
        if (Math.abs(diff) < 1e-5) { for (let j = 0; j < n; j++) P[i][j] = row[j] / sumP; break; }
        if (diff > 0) { betaMin = beta; beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2; }
        else { betaMax = beta; beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2; }
        if (iter === 49) for (let j = 0; j < n; j++) P[i][j] = row[j] / sumP;
      }
    }
    // symmetrize
    const Ps = MLU.zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Ps[i][j] = Math.max((P[i][j] + P[j][i]) / (2 * n), 1e-12);
    return Ps;
  }

  function makeClusterData() {
    const dims = 5, clusters = 4, perCluster = 15;
    const centers = Array.from({ length: clusters }, () => Array.from({ length: dims }, () => MLU.randRange(-6, 6)));
    const X = [], labels = [];
    for (let c = 0; c < clusters; c++)
      for (let i = 0; i < perCluster; i++) {
        X.push(centers[c].map((v) => v + MLU.randn() * 1.1));
        labels.push(c);
      }
    return { X, labels };
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint">iteration <b id="tsne-iter">0</b> · KL divergence <b id="tsne-kl">–</b></span></div>
            <div class="btn-row">
              <button id="tsne-step" class="primary">+ 20 iterations</button>
              <button id="tsne-reset">reset embedding</button>
              <button id="tsne-regen">new dataset</button>
            </div>
          </div>
          <svg id="tsne-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">source data is 4 gaussian clusters in 5D - colors show the true (hidden-from-the-algorithm) cluster</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>perplexity <span class="val" id="tsne-perp-val">20</span></h3>
            <input type="range" id="tsne-perp" min="5" max="45" step="1" value="20" />
          </div>
          <div class="control-card">
            <h3>fit</h3>
            <div class="readout" id="tsne-readout">–</div>
            <div class="note">Full (non-Barnes-Hut) t-SNE: high-D affinities P come from a per-point Gaussian kernel calibrated to the target perplexity via binary search; low-D affinities Q use a Student-t kernel; gradient descent with momentum minimizes KL(P‖Q).</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#tsne-svg");
    let { X, labels } = makeClusterData();
    let P, Y, Ydelta, iter;

    function initEmbedding() {
      P = computeP(pairwiseSq(X), perplexity());
      Y = X.map(() => [MLU.randn() * 1, MLU.randn() * 1]);
      Ydelta = Y.map(() => [0, 0]);
      iter = 0;
    }
    function perplexity() { return +document.getElementById("tsne-perp").value; }
    initEmbedding();

    function tsneStep() {
      const n = Y.length;
      const Dq = MLU.zeros(n, n), numer = MLU.zeros(n, n);
      let Z = 0;
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) {
          const d2 = (Y[i][0] - Y[j][0]) ** 2 + (Y[i][1] - Y[j][1]) ** 2;
          const num = 1 / (1 + d2);
          numer[i][j] = numer[j][i] = num;
          Z += 2 * num;
        }
      const Q = MLU.zeros(n, n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) Q[i][j] = Math.max(numer[i][j] / Z, 1e-12);

      const momentum = iter < 20 ? 0.5 : 0.8;
      const lr = 100;
      let kl = 0;
      for (let i = 0; i < n; i++) {
        const grad = [0, 0];
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const mult = (P[i][j] - Q[i][j]) * numer[i][j];
          grad[0] += mult * (Y[i][0] - Y[j][0]);
          grad[1] += mult * (Y[i][1] - Y[j][1]);
          kl += P[i][j] * Math.log(P[i][j] / Q[i][j]);
        }
        Ydelta[i][0] = momentum * Ydelta[i][0] - lr * 4 * grad[0];
        Ydelta[i][1] = momentum * Ydelta[i][1] - lr * 4 * grad[1];
      }
      for (let i = 0; i < n; i++) { Y[i][0] += Ydelta[i][0] / n; Y[i][1] += Ydelta[i][1] / n; }
      iter++;
      return kl / 2;
    }

    const ptsG = svg.append("g");
    function render(kl) {
      document.getElementById("tsne-iter").textContent = iter;
      document.getElementById("tsne-perp-val").textContent = perplexity();
      if (kl !== undefined) document.getElementById("tsne-kl").textContent = kl.toFixed(3);

      const ys = Y.map((p) => p[0]), xs = Y.map((p) => p[1]);
      const ext = Math.max(1, ...Y.flat().map(Math.abs)) * 1.15;
      const x = d3.scaleLinear().domain([-ext, ext]).range([PAD, W - PAD]);
      const y = d3.scaleLinear().domain([-ext, ext]).range([H - PAD, PAD]);
      svg.selectAll("g.axes").remove();
      const axesG = svg.insert("g", ":first-child").attr("class", "axes");
      axesG.append("rect").attr("x", PAD).attr("y", PAD).attr("width", W - 2 * PAD).attr("height", H - 2 * PAD).attr("fill", "none").attr("stroke", "var(--border-soft)");

      const data = Y.map((p, i) => ({ x: p[0], y: p[1], label: labels[i] }));
      const sel = ptsG.selectAll("circle").data(data);
      sel.enter().append("circle").attr("r", 5).attr("stroke", "var(--bg)").attr("stroke-width", 1)
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("fill", (d) => MLU.palette[d.label]);
      sel.exit().remove();

      document.getElementById("tsne-readout").innerHTML = `points: <b>${data.length}</b> (5D → 2D)<br>iterations: <b>${iter}</b>`;
    }

    document.getElementById("tsne-step").addEventListener("click", () => {
      let kl = 0; for (let i = 0; i < 20; i++) kl = tsneStep();
      render(kl);
    });
    document.getElementById("tsne-reset").addEventListener("click", () => { initEmbedding(); render(); });
    document.getElementById("tsne-regen").addEventListener("click", () => { ({ X, labels } = makeClusterData()); initEmbedding(); render(); });
    document.getElementById("tsne-perp").addEventListener("change", () => { initEmbedding(); render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "tsne",
    name: "t-SNE",
    category: "Unsupervised - Dimensionality Reduction",
    tagline: "high-D neighborhoods → 2D map",
    description: "Embeds 5-dimensional clustered data into 2D by matching high-dimensional neighbor probabilities to a low-dimensional Student-t distribution via gradient descent. Step through iterations to watch clusters separate.",
    info: {
      type: "Unsupervised - Non-linear dimensionality reduction, visualization-focused (no out-of-sample mapping).",
      scenario: "Visualizing high-dimensional data (embeddings, gene expression, learned features) in 2D/3D so that clusters close in high-D stay close in the 2D map.",
      inputs: "A set of high-dimensional points - unsupervised; labels, if any, are only used afterward to color the plot.",
      intuition: {
        definition: "Turn distances into <b>neighbour probabilities</b> in both the original space and the 2D map, then move the 2D points around until the two probability distributions match. It is a layout algorithm optimised for one thing: keeping neighbours together.",
        steps: [
          "In high-D, convert distances to probabilities with a per-point Gaussian.",
          "In 2D, do the same with a heavy-tailed Student-t kernel.",
          "Measure the mismatch with KL divergence.",
          "Gradient-descend on the 2D coordinates until it settles.",
        ],
        applications: [
          "Visualising word and sentence embeddings",
          "Single-cell RNA sequencing cluster maps",
          "Inspecting what a neural network's hidden layers learned",
          "Exploratory discovery of subgroups before formal clustering",
          "Sanity-checking whether classes are separable at all",
        ],
      },
      math: [
        { title: "High-D conditional affinity", formula: "p_{j|i} = exp(−‖xᵢ−xⱼ‖²/2σᵢ²) / Σ_{k≠i} exp(−‖xᵢ−xₖ‖²/2σᵢ²)", note: "A Gaussian centred on each point, converted to a probability distribution over its neighbours. Note σᵢ varies per point." },
        { title: "Perplexity calibration", formula: "Perp(Pᵢ) = 2^{H(Pᵢ)} = target,   H(Pᵢ) = −Σⱼ p_{j|i} log₂ p_{j|i}", note: "σᵢ is binary-searched so every point has the same effective neighbour count. This is what adapts the method to varying local density." },
        { title: "Symmetrise", formula: "pᵢⱼ = (p_{j|i} + p_{i|j}) / (2m)", note: "Makes the affinity matrix symmetric and guarantees every point contributes to the cost, so outliers cannot be ignored." },
        { title: "Low-D affinity", formula: "qᵢⱼ = (1 + ‖yᵢ−yⱼ‖²)⁻¹ / Σ_{k≠l}(1 + ‖yₖ−y_l‖²)⁻¹", note: "A Student-t with one degree of freedom. Its heavy tail is the key design choice, and it is what the 't' in t-SNE refers to." },
        { title: "Cost", formula: "KL(P‖Q) = Σᵢ Σⱼ pᵢⱼ · log(pᵢⱼ / qᵢⱼ)", note: "Asymmetric on purpose: a large p paired with a small q is punished heavily, but a small p with large q costs little. That is why local structure is preserved and global structure is not." },
        { title: "Gradient", formula: "∂C/∂yᵢ = 4 Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ − yⱼ)(1 + ‖yᵢ−yⱼ‖²)⁻¹", note: "Reads as a spring system: attraction where p exceeds q, repulsion where q exceeds p." },
      ],
      pipeline: [
        { label: "High-D points", note: "often PCA'd to 50D" },
        { label: "P from Gaussians", note: "perplexity-calibrated" },
        { label: "Random 2D init", note: "or PCA init" },
        { label: "Q from Student-t", note: "heavy tail" },
        { label: "Descend KL(P‖Q)", note: "2D map", accent: "green" },
      ],
      decisionFunction: {
        text: "yᵢ is found by gradient descent, not computed by a formula, so that low-D neighbor probabilities Q match high-D neighbor probabilities P",
        mechanism: "High-D neighbor probability Pᵢⱼ uses a Gaussian kernel calibrated per-point to a target perplexity; low-D probability Qᵢⱼ uses a heavier-tailed Student-t kernel - the heavy tail is what lets moderately-distant 2D points still represent moderately-similar high-D points without everything collapsing into one blob.",
      },
      lossFunction: {
        text: "KL(P‖Q) = Σᵢⱼ Pᵢⱼ·log(Pᵢⱼ/Qᵢⱼ)",
        mechanism: "Minimized by gradient descent with momentum directly on the 2D coordinates: points that should be neighbors (high P) but currently aren't (low Q) get pulled together; points that shouldn't be neighbors get pushed apart.",
      },
      optimization: [
        { title: "Early exaggeration", formula: "multiply P by ~12 for the first ~250 iterations", note: "Inflating the attractive forces early forces tight clusters to form with plenty of empty space between them, so they do not get trapped overlapping." },
        { title: "Momentum schedule", formula: "0.5 early, 0.8 after exaggeration ends", note: "Low momentum while the layout is chaotic, higher once it is settling, which speeds convergence without oscillation." },
        { title: "Barnes-Hut approximation", formula: "O(m²) → O(m log m)", note: "Distant points are grouped into a quadtree cell and treated as one repulsive body. Standard for datasets above a few thousand points." },
        { title: "Cost is non-convex", formula: "different seeds give different maps", note: "There is no single correct layout. Run it more than once and only trust structure that reappears." },
        { title: "PCA first", formula: "reduce to ~50 dimensions before t-SNE", note: "Cuts noise and cost substantially, and is standard practice. It also makes the initial neighbour graph more reliable." },
      ],
      output: "A 2D (or 3D) coordinate for every input point, suitable for a scatter plot - not a reusable function for new points.",
      assumptions: [
        { name: "Only local structure matters", why: "The KL cost barely penalises misplaced distant points, so between-cluster distances in the map are not meaningful.", check: "Never interpret the gap between two clusters as a similarity measure." },
        { name: "You want a picture, not features", why: "There is no learned mapping, so the output cannot be applied to new data.", check: "For a reusable embedding use PCA, UMAP, or an autoencoder." },
        { name: "Perplexity suits the data size", why: "Perplexity is an effective neighbour count and must be well below the number of points.", check: "Keep perplexity under m/3; typical values are 5 to 50." },
        { name: "Dimensionality pre-reduced", why: "Raw high-dimensional distances are noisy and slow to compute.", check: "Apply PCA to roughly 50 dimensions first." },
        { name: "Enough iterations", why: "Stopping early leaves a half-formed layout that looks like structure but is not.", check: "Run at least 1000 iterations and confirm the KL divergence has plateaued." },
      ],
      hyperparameters: [
        { name: "perplexity", range: "5 - 50", increasing: "Wider neighbourhoods, more global structure retained, clusters merge and soften.", strategy: "The main knob. Try 5, 30, and 50 and only trust structure that survives all three. Must be less than the point count." },
        { name: "n_iter", range: "1000 - 5000", increasing: "More refined layout, diminishing returns once KL plateaus.", strategy: "1000 minimum. Increase if the map still looks unsettled." },
        { name: "learning_rate", range: "10 - 1000 (or 'auto')", increasing: "Faster separation, risk of points flying apart into a diffuse ball.", strategy: "Use 'auto' (roughly m/12). A dense ball with everything compressed usually means it is too low." },
        { name: "early_exaggeration", range: "4 - 24", increasing: "Larger gaps between clusters, more distinct separation.", strategy: "12 is the default and rarely needs changing." },
        { name: "init", range: "pca / random", increasing: "Not applicable", strategy: "Use PCA initialisation. It is more reproducible and preserves more global structure than a random start." },
        { name: "metric", range: "euclidean / cosine / correlation", increasing: "Not applicable", strategy: "Cosine for embeddings and text, euclidean for most other numeric data." },
      ],
      metrics: ["No single 'accuracy' - trustworthiness / continuity scores", "Final KL divergence (compare only across runs on the same data)", "Qualitative check: do known clusters/classes separate in the map?"],
      typicalUses: ["Visualizing word/sentence embeddings", "Single-cell genomics cluster visualization", "Inspecting a neural network's learned feature space", "Exploratory cluster discovery"],
      diagnostics: [
        "Run several perplexities and several seeds. Structure that appears in all of them is real; structure that appears once is an artifact.",
        "A uniformly spread ball of points usually means the learning rate is too low or the iteration count too small.",
        "Many tiny scattered clumps often means perplexity is too low and the algorithm is fitting noise.",
        "Check that KL divergence has plateaued. A still-falling curve means the layout is unfinished.",
        "Never measure cluster sizes or between-cluster distances off the map. Both are artifacts of the algorithm, not properties of the data.",
      ],
      advantages: [
        "Exceptionally good at revealing local cluster structure that linear methods miss entirely.",
        "Handles non-linear manifolds that PCA flattens incorrectly.",
        "Per-point perplexity calibration adapts automatically to regions of different density.",
        "The Student-t kernel solves the crowding problem, so clusters separate cleanly instead of collapsing.",
        "Produces genuinely useful exploratory pictures of embeddings and genomics data.",
      ],
      limitations: [
        { name: "No out-of-sample mapping", note: "new points require rerunning the whole optimisation", fix: "UMAP, parametric t-SNE, or an autoencoder." },
        { name: "Global structure is not preserved", note: "distances between clusters are meaningless", fix: "UMAP retains more global structure; check PCA for the overall geometry." },
        { name: "Cluster sizes are meaningless", note: "the algorithm equalises density, so a tight cluster may render large", fix: "never read size off the plot." },
        { name: "Stochastic", note: "different seeds produce different maps", fix: "PCA initialisation and multiple runs." },
        { name: "Slow", note: "O(m²) naive, and still costly with Barnes-Hut", fix: "PCA first, or use UMAP which is much faster." },
        { name: "Can invent apparent clusters", note: "random data will still produce clumps at low perplexity", fix: "vary perplexity and check the structure persists." },
      ],
      alternatives: [
        { name: "UMAP", when: "Almost always the better default now: faster, preserves more global structure, and supports transforming new points." },
        { name: "PCA", when: "You need a fast, deterministic, interpretable, reusable linear projection." },
        { name: "Autoencoder", when: "You want a learned non-linear embedding usable as features on new data." },
        { name: "PHATE / diffusion maps", when: "The data has continuous trajectory structure rather than discrete clusters." },
      ],
      pitfalls: [
        { problem: "Reading meaning into gaps between clusters", solution: "Between-cluster distance is not preserved. This is the single most common misinterpretation." },
        { problem: "Comparing cluster sizes in the map", solution: "Also meaningless. t-SNE equalises local density by construction." },
        { problem: "Map is a featureless ball", solution: "Learning rate too low or too few iterations. Use 'auto' and at least 1000 steps." },
        { problem: "Different results every run", solution: "Expected, since the cost is non-convex. Use PCA init, fix the seed, and confirm structure across runs." },
        { problem: "Trying to transform a test set", solution: "There is no transform method by design. Use UMAP if you need one." },
        { problem: "Running t-SNE on raw 10000-dimensional data", solution: "Apply PCA to about 50 dimensions first, for both speed and noise reduction." },
      ],
      quickRef: [
        { name: "High-D affinity", formula: "p_{j|i} ∝ exp(−‖xᵢ−xⱼ‖²/2σᵢ²)" },
        { name: "Symmetrised", formula: "pᵢⱼ = (p_{j|i}+p_{i|j})/2m" },
        { name: "Perplexity", formula: "Perp = 2^{H(Pᵢ)}" },
        { name: "Low-D affinity", formula: "qᵢⱼ ∝ (1+‖yᵢ−yⱼ‖²)⁻¹" },
        { name: "Cost", formula: "KL(P‖Q) = Σ pᵢⱼ log(pᵢⱼ/qᵢⱼ)" },
        { name: "Gradient", formula: "4Σⱼ(pᵢⱼ−qᵢⱼ)(yᵢ−yⱼ)(1+‖yᵢ−yⱼ‖²)⁻¹" },
        { name: "Barnes-Hut", formula: "O(m log m)" },
        { name: "Meaningless in the map", formula: "cluster size, inter-cluster distance" },
      ],
      code: `from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# Standard practice: PCA down to ~50D first for speed and denoising.
X50 = PCA(n_components=50, random_state=42).fit_transform(X)

# Vary perplexity: only trust structure that survives all of them.
for perp in (5, 30, 50):
    Y = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",     # roughly m/12
        init="pca",               # more reproducible than random
        n_iter=1000,
        random_state=42,
    ).fit_transform(X50)
    # scatter(Y[:, 0], Y[:, 1], c=labels)

# There is deliberately no .transform(): t-SNE cannot embed new points.
# If you need that, use UMAP:
# import umap; um = umap.UMAP(n_neighbors=15, min_dist=0.1).fit(X50)
# um.transform(X_new)`,
      whyChain: [
        { q: "Why use a Student-t distribution in the low-dimensional map?", a: "It solves the crowding problem. Two dimensions have far less room than the original space, so moderately distant points cannot all be placed at proportionate distances. The t-distribution's heavy tail assigns a reasonable probability even to fairly separated 2D points, so clusters can spread apart instead of collapsing into one blob." },
        { q: "What is the crowding problem exactly?", a: "The volume of a ball grows exponentially with dimension. In 50 dimensions a point can have many neighbours all roughly equidistant; in 2D there is simply not enough area to place them all at that distance. Something has to give, and with a Gaussian in both spaces everything gets crushed together." },
        { q: "What does perplexity actually control?", a: "It is a smooth measure of the effective number of neighbours each point considers. The algorithm binary-searches a separate bandwidth σᵢ per point so that every point achieves the same perplexity, which is how it adapts to regions of differing density." },
        { q: "Why is KL divergence asymmetric, and does it matter?", a: "It matters enormously. KL(P‖Q) heavily penalises placing true neighbours far apart, but barely penalises placing distant points close together. That asymmetry is exactly why local structure is faithful and global structure is not." },
        { q: "Why can't you interpret the distance between two clusters?", a: "Because the cost function does not care. Once two points are far apart in the map, moving them further apart or closer together changes the KL divergence very little. The layout of clusters relative to each other is effectively arbitrary." },
        { q: "Why does t-SNE have no transform method?", a: "It does not learn a function. It optimises the coordinates of each specific point directly, so there is no mapping to apply to unseen data. Embedding a new point would require rerunning the optimisation with it included." },
        { q: "What is early exaggeration for?", a: "Multiplying P early in training inflates the attractive forces, so clusters form tightly and push apart from each other, leaving empty space. Without it, clusters tend to get stuck overlapping in a local optimum they cannot escape." },
        { q: "t-SNE or UMAP?", a: "UMAP for most purposes now. It is substantially faster, preserves more global structure, and can transform new data. t-SNE remains a strong, well-understood choice for pure local-structure visualisation." },
      ],
      parameters: [
        { name: "perplexity", effect: "Roughly 'how many neighbors' each point's Gaussian considers. Small → emphasizes very local structure; large → considers broader neighborhoods." },
        { name: "iterations", effect: "More iterations refine the layout further, with diminishing returns after clusters separate." },
        { name: "learning rate / momentum", effect: "Affect how quickly and how distinctly clusters separate early in training." },
      ],
      metrics: ["No single 'accuracy' - trustworthiness / continuity scores", "Qualitative check: do known clusters/classes separate in the map?"],
      typicalUses: ["Visualizing word/sentence embeddings", "Single-cell genomics cluster visualization", "Inspecting a neural network's learned feature space", "Exploratory cluster discovery"],
      workedExample: {
        setup: "3 points in 1D: x=0, x=1, x=5. Compute p_{j|0} (neighbor probability of points 1 and 2, given point 0) with β=1/(2σ²)=0.5.",
        steps: [
          "Squared distances from point 0: d(0,1)²=1, d(0,2)²=25.",
          "Unnormalized affinities: exp(−0.5×1)=0.6065, exp(−0.5×25)=exp(−12.5)≈3.7×10⁻⁶.",
          "Sum = 0.6065 + 0.0000037 ≈ 0.60651.",
          "p_{1|0} = 0.6065/0.60651 ≈ 0.99994. p_{2|0} = 0.0000037/0.60651 ≈ 0.0000061.",
        ],
        result: "Point 1 (close) gets ~99.99% of point 0's neighbor probability mass; point 2 (far) gets essentially none - exactly the local-neighborhood emphasis t-SNE is built on",
      },
    },
    mount,
  });
})();
