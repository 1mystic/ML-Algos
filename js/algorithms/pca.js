(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[0]}"></span>data</span>
              <span class="legend-item"><span class="swatch" style="background:var(--accent)"></span>PC1</span>
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[4]}"></span>PC2</span>
            </div>
            <div class="btn-row">
              <button id="pca-regen">regenerate data</button>
              <button id="pca-clear">clear</button>
            </div>
          </div>
          <svg id="pca-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">click to add points · drag to move them · arrows are the principal axes, scaled by &radic;eigenvalue</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>options</h3>
            <div class="field"><label><input type="checkbox" id="pca-proj" checked style="width:auto;margin-right:6px" />show projections onto PC1</label></div>
          </div>
          <div class="control-card">
            <h3>variance</h3>
            <div class="readout" id="pca-readout">–</div>
            <div class="note">Eigendecomposition of the centered covariance matrix (via Jacobi rotation). PC1 is the direction of maximum variance; projecting onto it is the best 1D lossy compression of this data.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#pca-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    function genData() {
      const angle = MLU.randRange(0, Math.PI);
      const n = 60, pts = [];
      const cx = MLU.randRange(-2, 2), cy = MLU.randRange(-2, 2);
      for (let i = 0; i < n; i++) {
        const a = MLU.randn() * 4, b = MLU.randn() * 1.1;
        pts.push({
          x: cx + a * Math.cos(angle) - b * Math.sin(angle),
          y: cy + a * Math.sin(angle) + b * Math.cos(angle),
        });
      }
      return pts;
    }
    let points = genData();

    const projG = svg.append("g");
    const axesG = svg.append("g");
    const ptsG = svg.append("g");

    function render() {
      let mu = { x: 0, y: 0 }, eig = { values: [1, 0], vectors: [[1, 0], [0, 1]] };
      if (points.length >= 2) {
        mu = MLU.meanVec(points.map((p) => [p.x, p.y]));
        mu = { x: mu[0], y: mu[1] };
        const cov = MLU.covMatrix(points.map((p) => [p.x, p.y]), [mu.x, mu.y]);
        eig = MLU.jacobiEigen(cov);
      }
      const showProj = document.getElementById("pca-proj").checked;

      if (showProj && points.length >= 2) {
        const [v0, v1] = eig.vectors[0];
        const projLines = points.map((p) => {
          const dx = p.x - mu.x, dy = p.y - mu.y;
          const t = dx * v0 + dy * v1;
          return { x1: p.x, y1: p.y, x2: mu.x + t * v0, y2: mu.y + t * v1 };
        });
        const sel = projG.selectAll("line").data(projLines);
        sel.enter().append("line").attr("stroke", "var(--text-faint)").attr("stroke-width", 1).attr("stroke-dasharray", "2,2")
          .merge(sel).attr("x1", (d) => x(d.x1)).attr("y1", (d) => y(d.y1)).attr("x2", (d) => x(d.x2)).attr("y2", (d) => y(d.y2));
        sel.exit().remove();
      } else projG.selectAll("line").remove();

      const arrows = points.length >= 2 ? [
        { v: eig.vectors[0], val: eig.values[0], color: "var(--accent)" },
        { v: eig.vectors[1], val: eig.values[1], color: MLU.palette[4] },
      ] : [];
      const asel = axesG.selectAll("line.axis").data(arrows);
      asel.enter().append("line").attr("class", "axis").attr("stroke-width", 2.5)
        .merge(asel)
        .attr("x1", (d) => x(mu.x - d.v[0] * Math.sqrt(Math.max(d.val, 0))))
        .attr("y1", (d) => y(mu.y - d.v[1] * Math.sqrt(Math.max(d.val, 0))))
        .attr("x2", (d) => x(mu.x + d.v[0] * Math.sqrt(Math.max(d.val, 0))))
        .attr("y2", (d) => y(mu.y + d.v[1] * Math.sqrt(Math.max(d.val, 0))))
        .attr("stroke", (d) => d.color);
      asel.exit().remove();

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle").attr("r", 5).attr("fill", MLU.palette[0]).attr("stroke", "var(--bg)").attr("stroke-width", 1).style("cursor", "grab")
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y))
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); render(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          render();
        }));
      sel.exit().remove();

      const total = eig.values[0] + eig.values[1] || 1;
      document.getElementById("pca-readout").innerHTML = points.length >= 2
        ? `points: <b>${points.length}</b><br>PC1 explains: <b class="num">${((eig.values[0] / total) * 100).toFixed(1)}%</b><br>PC2 explains: <b class="num">${((eig.values[1] / total) * 100).toFixed(1)}%</b>`
        : "add at least 2 points";
    }

    svg.on("click", (event) => {
      if (event.target.tagName === "circle") return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py) });
      render();
    });
    document.getElementById("pca-proj").addEventListener("change", render);
    document.getElementById("pca-regen").addEventListener("click", () => { points = genData(); render(); });
    document.getElementById("pca-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "pca",
    name: "Principal Component Analysis",
    category: "Unsupervised - Dimensionality Reduction",
    tagline: "eigendecomposition of the covariance",
    description: "Finds the orthogonal directions of maximum variance in the data via eigendecomposition of the covariance matrix, and shows what's lost when you project onto just the first component.",
    info: {
      type: "Unsupervised - Dimensionality reduction / linear transformation.",
      scenario: "Compressing correlated features into fewer uncorrelated components, visualizing high-dimensional data, denoising, or as pre-processing before another model.",
      inputs: "A matrix of numeric features X, typically centered (and often standardized) first.",
      intuition: {
        definition: "Find the directions along which the data <b>varies most</b>, and describe each point by its coordinates along just a few of them. Those directions are orthogonal, ordered by variance, and are exactly the eigenvectors of the covariance matrix.",
        steps: [
          "Centre the data so the mean sits at the origin.",
          "Compute the covariance matrix of the features.",
          "Take its eigenvectors: these are the principal axes.",
          "Keep the top k by eigenvalue and project onto them.",
        ],
        applications: [
          "Compressing correlated sensor or survey features",
          "Plotting high-dimensional data in two dimensions",
          "Denoising by discarding low-variance components",
          "Decorrelating features before regression or clustering",
          "Eigenfaces and other classical image compression",
        ],
      },
      math: [
        { title: "Centre the data", formula: "X̃ = X − μ,  μ = column means", note: "Non-negotiable. Without centring, the first component points at the mean rather than the direction of spread." },
        { title: "Covariance matrix", formula: "C = (1/(m−1))·X̃ᵀX̃    (p × p, symmetric)", note: "Entry Cⱼₖ is the covariance between features j and k. Symmetry guarantees real eigenvalues and orthogonal eigenvectors." },
        { title: "Eigendecomposition", formula: "C·vⱼ = λⱼ·vⱼ,   λ₁ ≥ λ₂ ≥ … ≥ λ_p ≥ 0", note: "Each eigenvector vⱼ is a principal axis; its eigenvalue λⱼ is the variance of the data along that axis." },
        { title: "Variance maximisation", formula: "v₁ = argmax_{‖v‖=1} Var(X̃v) = argmax vᵀCv", note: "The first component is the unit direction of maximum projected variance. Each later one repeats this subject to being orthogonal to all previous." },
        { title: "Projection", formula: "Z = X̃·V_k    (m × k scores)", note: "V_k holds the top k eigenvectors as columns. Z gives each point's coordinates in the new basis." },
        { title: "Reconstruction", formula: "X̂ = Z·V_kᵀ + μ,   error = Σ_{j>k} λⱼ", note: "The discarded eigenvalues are exactly the reconstruction error. Among all rank-k linear maps, this is provably optimal (Eckart-Young)." },
        { title: "SVD route", formula: "X̃ = UΣVᵀ  ⟹  same V, with λⱼ = σⱼ²/(m−1)", note: "How it is actually computed. SVD avoids forming C, which squares the condition number and loses precision." },
      ],
      pipeline: [
        { label: "Standardize", note: "centre, often scale" },
        { label: "Covariance C", note: "p × p matrix" },
        { label: "Eigen / SVD", note: "get V, λ" },
        { label: "Keep top k", note: "by variance" },
        { label: "Scores Z", note: "m × k projection", accent: "green" },
      ],
      decisionFunction: {
        text: "project(x) = Vᵀ(x − μ), using the top eigenvectors V of the covariance matrix",
        mechanism: "Eigendecompose the centered covariance matrix: the eigenvector with the largest eigenvalue is the direction of maximum variance (PC1); the next-largest orthogonal direction is PC2, and so on.",
      },
      lossFunction: {
        text: "min Σᵢ ‖xᵢ − reconstruct(project(xᵢ))‖²  (reconstruction error)",
        mechanism: "Among all rank-k linear projections, PCA's is provably the one minimizing this reconstruction error - solved exactly by eigendecomposition, with no optimization loop.",
        plot: { fn: (k) => Math.exp(-0.6 * k), domain: [1, 6], color: "var(--accent)", caption: "typical 'scree plot' shape: variance explained per component usually drops off fast" },
      },
      optimization: [
        { title: "No iteration", formula: "one eigendecomposition, closed form", note: "PCA is a matrix factorisation, not a learned model. There is no learning rate, no initialisation, and no local optimum." },
        { title: "Cost", formula: "O(m·p² + p³) via covariance, O(m·p·k) via randomized SVD", note: "The p³ eigendecomposition dominates in wide data. Randomized SVD is far cheaper when only a few components are wanted." },
        { title: "Choosing k", formula: "smallest k with Σ_{j≤k} λⱼ / Σ_j λⱼ ≥ target", note: "Typically 90 to 95% cumulative variance. Alternatives are the scree elbow or Kaiser's rule of keeping λ > 1 on standardized data." },
        { title: "Whitening", formula: "Z_white = Z / √λ", note: "Rescales components to unit variance. Useful before distance-based models, harmful if the variance ordering itself is informative." },
        { title: "Incremental PCA", formula: "update the basis batch by batch", note: "For data too large to hold in memory. Trades a little accuracy for constant memory." },
      ],
      output: "A lower-dimensional projected representation, plus the principal axes (eigenvectors) and their eigenvalues (variance explained).",
      assumptions: [
        { name: "Structure is linear", why: "PCA can only rotate and project. Data on a curved manifold is flattened incorrectly.", check: "If a 2D PCA plot looks structureless but t-SNE or UMAP shows clusters, the structure is non-linear." },
        { name: "Variance means information", why: "Components are ranked purely by variance, which need not align with what you care about.", check: "For a supervised task compare against LDA, which ranks by class separation instead." },
        { name: "Data is centred", why: "Uncentred data makes PC1 point toward the mean rather than along the spread.", check: "Every library centres automatically. Do not disable it." },
        { name: "Features are comparably scaled", why: "Covariance is scale-dependent, so a feature in large units dominates regardless of relevance.", check: "Standardize unless all features already share meaningful units." },
        { name: "Components need not be interpretable", why: "Each is a dense mixture of all original features.", check: "Use sparse PCA or factor analysis if you need readable loadings." },
      ],
      hyperparameters: [
        { name: "n_components k", range: "1 - min(m, p)", increasing: "Less information lost, less compression. k = p is a lossless rotation.", strategy: "Pick by cumulative explained variance (90 to 95%), the scree elbow, or downstream validation score. Pass a float such as 0.95 to let the library choose." },
        { name: "standardize first", range: "true / false", increasing: "Not applicable", strategy: "Standardize whenever feature units differ. Skip only when all features are already on one comparable scale, such as pixel intensities." },
        { name: "whiten", range: "true / false", increasing: "Not applicable", strategy: "Enable before distance-based models such as KNN or SVM; leave off when the variance ordering carries meaning." },
        { name: "svd_solver", range: "auto / full / randomized / arpack", increasing: "Not applicable", strategy: "auto is right nearly always. randomized is much faster when k is far smaller than p." },
      ],
      metrics: ["Cumulative % variance explained", "Reconstruction error (MSE)", "Scree plot elbow position", "Downstream model performance after projection"],
      typicalUses: ["Visualizing high-dimensional data in 2D/3D", "Noise reduction", "Decorrelating features before regression/clustering", "Data compression"],
      diagnostics: [
        "Plot the scree curve of explained variance per component. A sharp drop means a few components capture the structure; a flat curve means PCA is not helping.",
        "Check cumulative variance. If you need 40 of 50 components to reach 90%, the features were not very correlated and compression will cost you.",
        "Inspect the loadings of the top components to see which original features drive each axis.",
        "Compare a PCA scatter against t-SNE or UMAP. If they disagree sharply, the structure is non-linear.",
        "Verify the pipeline scales inside cross-validation. Fitting PCA on the full dataset before splitting leaks test information.",
      ],
      advantages: [
        "Exact closed-form solution with no hyperparameter search and no random seed.",
        "Provably the optimal linear projection for reconstruction error.",
        "Removes multicollinearity outright, since components are orthogonal by construction.",
        "Denoises effectively when noise is spread across low-variance directions.",
        "Fast, and scalable via randomized or incremental variants.",
        "Fully invertible, so you can project down and reconstruct back up.",
      ],
      limitations: [
        { name: "Linear only", note: "cannot unfold curved manifolds such as a swiss roll", fix: "kernel PCA, UMAP, t-SNE, or an autoencoder." },
        { name: "Components are uninterpretable", note: "each mixes every original feature", fix: "sparse PCA, or factor analysis with rotation." },
        { name: "Variance is not relevance", note: "a low-variance feature may be the only predictive one", fix: "use LDA or PLS when labels are available." },
        { name: "Scale-sensitive", note: "unstandardized features hijack the components", fix: "standardize first." },
        { name: "Outlier-sensitive", note: "covariance is driven by extreme points", fix: "robust PCA, or remove outliers." },
        { name: "Assumes Gaussian-ish structure", note: "only second-order statistics are used", fix: "ICA if you need independent rather than merely uncorrelated components." },
      ],
      alternatives: [
        { name: "t-SNE / UMAP", when: "The goal is visualisation and the structure is non-linear." },
        { name: "Kernel PCA", when: "You want PCA's framework but a non-linear manifold." },
        { name: "LDA", when: "Labels exist and you want axes that separate classes, not axes of maximum variance." },
        { name: "Autoencoder", when: "Large datasets and complex non-linear compression, and you need an out-of-sample mapping." },
        { name: "Truncated SVD", when: "Sparse data such as TF-IDF, where centring would destroy sparsity." },
      ],
      pitfalls: [
        { problem: "One feature dominates PC1", solution: "Unstandardized units. Scale before fitting." },
        { problem: "PCA plot shows no structure", solution: "Structure may be non-linear. Try UMAP or t-SNE before concluding there is none." },
        { problem: "Accuracy drops after PCA", solution: "You discarded low-variance but predictive directions. Keep more components, or use a supervised reduction." },
        { problem: "Fitting PCA before the train/test split", solution: "Leakage. Put PCA inside a Pipeline so it is fitted per fold." },
        { problem: "Component signs flip between runs", solution: "Eigenvector sign is arbitrary. Harmless, but fix it if you compare loadings across runs." },
        { problem: "Centring a huge sparse matrix blows up memory", solution: "Use TruncatedSVD, which skips centring and preserves sparsity." },
      ],
      quickRef: [
        { name: "Centre", formula: "X̃ = X − μ" },
        { name: "Covariance", formula: "C = X̃ᵀX̃ / (m−1)" },
        { name: "Eigenproblem", formula: "C v = λ v" },
        { name: "Projection", formula: "Z = X̃ V_k" },
        { name: "Reconstruction", formula: "X̂ = Z V_kᵀ + μ" },
        { name: "Explained variance", formula: "λⱼ / Σ λ" },
        { name: "Recon. error", formula: "Σ_{j>k} λⱼ" },
        { name: "Via SVD", formula: "X̃ = UΣVᵀ, λ = σ²/(m−1)" },
      ],
      code: `from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Pass a float to let PCA pick k by cumulative explained variance.
pca = make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=42))
Z = pca.fit_transform(X_train)

p = pca.named_steps["pca"]
print("components kept:", p.n_components_)
print("cumulative variance:", np.cumsum(p.explained_variance_ratio_))

# Which original features drive each axis:
loadings = p.components_          # shape (k, p)

# Inside a model, PCA must be fitted per fold or you leak test data.
from sklearn.linear_model import LogisticRegression
clf = make_pipeline(StandardScaler(), PCA(n_components=20), LogisticRegression())

# Sparse text data: skip centring so sparsity survives.
svd = TruncatedSVD(n_components=100).fit_transform(tfidf_matrix)`,
      whyChain: [
        { q: "Why are the principal components the eigenvectors of the covariance matrix?", a: "The variance of the data projected onto a unit vector v is vᵀCv. Maximising that subject to ‖v‖ = 1 is a Rayleigh quotient problem, and its solution is the eigenvector with the largest eigenvalue. The eigenvalue is the variance itself." },
        { q: "Why must the data be centred first?", a: "Variance is measured about the mean. If the data is not centred, the direction that maximises the raw second moment points from the origin toward the data cloud, which describes the location of the data rather than its spread." },
        { q: "Why are the components always orthogonal?", a: "The covariance matrix is real and symmetric, and the spectral theorem guarantees such a matrix has an orthogonal eigenbasis. Geometrically, each component captures variance that the previous ones have not, so it must be perpendicular to them." },
        { q: "Why standardize before PCA?", a: "Covariance depends on units. A feature measured in millimetres has a variance a million times larger than the same feature in metres, so it would capture PC1 for reasons that have nothing to do with information content." },
        { q: "Why is SVD preferred over eigendecomposing the covariance matrix?", a: "Forming XᵀX squares the condition number, so precision is lost for ill-conditioned data. SVD works directly on X and returns the same V with better numerical stability." },
        { q: "Does a high-variance component mean an important one?", a: "Not necessarily. PCA is unsupervised, so it has no idea what you want to predict. A low-variance direction can carry all the class signal, which is why PCA sometimes hurts downstream accuracy. LDA optimises separation instead." },
        { q: "Why can PCA not unfold a swiss roll?", a: "PCA applies a single linear map, so it can rotate and project but never bend. A swiss roll needs different transformations in different regions, which requires a non-linear method." },
        { q: "How does PCA denoise data?", a: "Signal usually concentrates in a few high-variance directions while noise spreads thinly across all of them. Discarding the low-variance components removes proportionally more noise than signal, and the reconstruction is cleaner than the original." },
      ],
      parameters: [
        { name: "number of components", effect: "How many axes to keep. Fewer = more compression / more information lost." },
        { name: "standardize features?", effect: "Important when features are on very different scales - otherwise high-variance features dominate the principal axes for reasons unrelated to importance." },
      ],
      metrics: ["Cumulative % variance explained", "Reconstruction error (MSE)"],
      typicalUses: ["Visualizing high-dimensional data in 2D/3D", "Noise reduction", "Decorrelating features before regression/clustering", "Data compression"],
      workedExample: {
        setup: "Points (1,2), (3,4), (5,6) - perfectly correlated (they lie exactly on a line). Find PC1.",
        steps: [
          "Mean = (3, 4). Deviations: (−2,−2), (0,0), (2,2).",
          "Sxx = (4+0+4)/3 = 2.667, Syy = (4+0+4)/3 = 2.667, Sxy = (4+0+4)/3 = 2.667.",
          "Covariance matrix = [[2.667, 2.667], [2.667, 2.667]] - trace = 5.333, determinant = 2.667×2.667 − 2.667×2.667 = 0.",
          "Determinant 0 means one eigenvalue is 0; since trace = sum of eigenvalues, the other eigenvalue is 5.333.",
          "The eigenvector for λ=5.333 is the direction (1,1)/√2 - a 45° line.",
        ],
        result: "PC1 points along (1,1)/√2 and explains 100% of the variance - exactly the line y=x+1 the data lies on, with 0% left for PC2",
      },
    },
    mount,
  });
})();
