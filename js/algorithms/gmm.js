(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];
  const TWO_PI = 2 * Math.PI;

  function gaussPdf(p, mu, cov) {
    const dx = p.x - mu.x, dy = p.y - mu.y;
    const det = Math.max(MLU.det2x2(cov), 1e-6);
    const inv = MLU.inverse2x2(cov);
    const m = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy);
    return Math.exp(-0.5 * m) / (TWO_PI * Math.sqrt(det));
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint" style="font-size:11.5px">EM iteration <b id="gm-iter">0</b></span></div>
            <div class="btn-row">
              <button id="gm-step" class="primary">step (E+M)</button>
              <button id="gm-run">run to convergence</button>
              <button id="gm-reset">reset components</button>
              <button id="gm-regen">regenerate data</button>
            </div>
          </div>
          <svg id="gm-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">points are colored by their most-likely component · ellipses = 1&sigma; contour of each Gaussian</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>components k = <span class="val" id="gm-k-val">3</span></h3>
            <input type="range" id="gm-k" min="1" max="6" step="1" value="3" />
          </div>
          <div class="control-card">
            <h3>state</h3>
            <div class="readout" id="gm-readout">–</div>
            <div class="note">Expectation-Maximization: E-step computes soft responsibilities under each Gaussian, M-step re-estimates each component's mean/covariance/weight from those responsibilities - as in <code>mla/gaussian_mixture.py</code>.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#gm-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeBlobs({ n: 80, clusters: 3, spread: 1.1 }).map((p) => ({ x: p.x, y: p.y, c: 0 }));
    let comps = [];
    let iter = 0, loglik = 0;

    function k() { return +document.getElementById("gm-k").value; }
    function initComps() {
      const shuffled = MLU.shuffle(points);
      comps = Array.from({ length: k() }, (_, i) => ({
        mu: shuffled.length ? { x: shuffled[i % shuffled.length].x, y: shuffled[i % shuffled.length].y } : { x: MLU.randRange(...DOMAIN), y: MLU.randRange(...DOMAIN) },
        cov: [[4, 0], [0, 4]],
        weight: 1 / k(),
      }));
      iter = 0;
    }
    initComps();

    function eStep() {
      const resp = points.map(() => new Array(comps.length).fill(0));
      let ll = 0;
      points.forEach((p, i) => {
        const ws = comps.map((c) => c.weight * gaussPdf(p, c.mu, c.cov));
        const s = ws.reduce((a, b) => a + b, 0) || 1e-9;
        resp[i] = ws.map((w) => w / s);
        ll += Math.log(s);
        p.c = ws.indexOf(Math.max(...ws));
      });
      return { resp, ll };
    }
    function mStep(resp) {
      const n = points.length;
      comps.forEach((comp, k2) => {
        let Nk = 0;
        for (let i = 0; i < n; i++) Nk += resp[i][k2];
        if (Nk < 1e-6) return;
        let mx = 0, my = 0;
        for (let i = 0; i < n; i++) { mx += resp[i][k2] * points[i].x; my += resp[i][k2] * points[i].y; }
        mx /= Nk; my /= Nk;
        let cxx = 0, cyy = 0, cxy = 0;
        for (let i = 0; i < n; i++) {
          const dx = points[i].x - mx, dy = points[i].y - my;
          cxx += resp[i][k2] * dx * dx; cyy += resp[i][k2] * dy * dy; cxy += resp[i][k2] * dx * dy;
        }
        cxx = cxx / Nk + 0.15; cyy = cyy / Nk + 0.15; cxy /= Nk;
        comp.mu = { x: mx, y: my };
        comp.cov = [[cxx, cxy], [cxy, cyy]];
        comp.weight = Nk / n;
      });
    }

    const ptsG = svg.append("g");
    const ellG = svg.append("g");

    function render() {
      document.getElementById("gm-k-val").textContent = k();
      document.getElementById("gm-iter").textContent = iter;

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

      const esel = ellG.selectAll("ellipse").data(comps);
      esel.enter().append("ellipse").attr("fill", "none").attr("stroke-width", 2)
        .merge(esel)
        .attr("cx", (d) => x(d.mu.x)).attr("cy", (d) => y(d.mu.y))
        .attr("stroke", (d, i) => MLU.palette[i % MLU.palette.length])
        .attr("transform", (d, i) => {
          const eig = MLU.jacobiEigen(d.cov);
          const [v0x, v0y] = eig.vectors[0];
          const angle = Math.atan2(-v0y, v0x) * 180 / Math.PI; // flip y for screen coords
          return `rotate(${angle} ${x(d.mu.x)} ${y(d.mu.y)})`;
        })
        .each(function (d) {
          const eig = MLU.jacobiEigen(d.cov);
          const scaleX = Math.abs(x(1) - x(0));
          d3.select(this)
            .attr("rx", Math.sqrt(Math.max(eig.values[0], 0.01)) * scaleX)
            .attr("ry", Math.sqrt(Math.max(eig.values[1], 0.01)) * scaleX);
        });
      esel.exit().remove();

      document.getElementById("gm-readout").innerHTML =
        `points: <b>${points.length}</b><br>log-likelihood: <b class="num">${points.length ? loglik.toFixed(1) : "–"}</b><br>` +
        comps.map((c, i) => `<span style="color:${MLU.palette[i]}">&pi;${i}=${c.weight.toFixed(2)}</span>`).join(" ");
    }

    svg.on("click", (event) => {
      if (event.target.tagName === "circle") return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py), c: 0 });
      render();
    });
    document.getElementById("gm-k").addEventListener("input", () => { initComps(); render(); });
    document.getElementById("gm-step").addEventListener("click", () => {
      const { resp, ll } = eStep(); mStep(resp); loglik = ll; iter++; render();
    });
    document.getElementById("gm-run").addEventListener("click", () => {
      let prev = -Infinity;
      for (let i = 0; i < 100; i++) {
        const { resp, ll } = eStep(); mStep(resp); loglik = ll; iter++;
        if (Math.abs(ll - prev) < 1e-3) break;
        prev = ll;
      }
      render();
    });
    document.getElementById("gm-reset").addEventListener("click", () => { initComps(); render(); });
    document.getElementById("gm-regen").addEventListener("click", () => {
      points = MLU.makeBlobs({ n: 80, clusters: 3, spread: 1.1 }).map((p) => ({ x: p.x, y: p.y, c: 0 }));
      initComps(); render();
    });

    render();
    return () => {};
  }

  MLApp.register({
    id: "gmm",
    name: "Gaussian Mixture Model",
    category: "Unsupervised - Clustering",
    tagline: "EM algorithm, soft clusters",
    description: "Soft clustering via Expectation-Maximization: components are full 2D Gaussians (with orientation), fit by alternating responsibility estimation (E) and weighted re-fitting (M).",
    sourceFile: "mla/gaussian_mixture.py",
    info: {
      type: "Unsupervised - Clustering (soft/probabilistic). Generative mixture model.",
      scenario: "Clusters that overlap or have different sizes/shapes/orientations, or when you need membership probabilities rather than hard labels - a probabilistic generalization of K-Means.",
      inputs: "Unlabeled points and a chosen number of mixture components k.",
      intuition: {
        definition: "Assume the data was generated by picking one of k Gaussians at random and drawing a point from it. Fitting the model means recovering each Gaussian's <b>weight, mean, and covariance</b>. Because the component labels are hidden, we alternate between guessing them softly and re-fitting the Gaussians.",
        steps: [
          "E-step: given current Gaussians, compute each point's probability of belonging to each one.",
          "M-step: re-fit each Gaussian using those probabilities as weights.",
          "Repeat. The likelihood never decreases.",
          "Full covariances let clusters be stretched and rotated, unlike k-means.",
        ],
        applications: [
          "Segmentation where clusters genuinely overlap",
          "Anomaly detection by flagging low-likelihood points",
          "Density estimation as a smooth alternative to histograms",
          "Speaker identification and acoustic modelling",
          "Modelling subpopulations in biology and finance",
        ],
      },
      math: [
        { title: "Mixture density", formula: "p(x) = Σ_{k=1}^{K} π_k · N(x; μ_k, Σ_k),   Σ_k π_k = 1", note: "A weighted sum of Gaussian bumps. With enough components this can approximate essentially any continuous density." },
        { title: "Multivariate Gaussian", formula: "N(x; μ, Σ) = (2π)^(−p/2)|Σ|^(−1/2) · exp( −½(x−μ)ᵀΣ⁻¹(x−μ) )", note: "The covariance matrix Σ is what allows elliptical, rotated clusters. Its eigenvectors give the axis directions and eigenvalues the squared radii." },
        { title: "E-step (responsibilities)", formula: "γ_{ik} = π_k·N(xᵢ; μ_k, Σ_k) / Σ_j π_j·N(xᵢ; μ_j, Σ_j)", note: "The posterior probability that point i came from component k. Each row of γ sums to 1, which is the soft assignment." },
        { title: "M-step (re-fit)", formula: "N_k = Σᵢ γ_{ik},   π_k = N_k/m,   μ_k = (1/N_k)Σᵢ γ_{ik}xᵢ,   Σ_k = (1/N_k)Σᵢ γ_{ik}(xᵢ−μ_k)(xᵢ−μ_k)ᵀ", note: "Exactly the ordinary sample statistics, but weighted by responsibility instead of hard membership." },
        { title: "Objective", formula: "log L = Σᵢ log( Σ_k π_k·N(xᵢ; μ_k, Σ_k) )", note: "EM maximises this. The log of a sum has no closed-form maximiser, which is precisely why the latent-variable trick is needed." },
        { title: "Model selection", formula: "BIC = −2·log L + d·log m,    AIC = −2·log L + 2d", note: "Unlike k-means inertia, these penalise parameter count, so they can actually choose K. Pick the K minimising BIC." },
      ],
      pipeline: [
        { label: "Init", note: "often k-means" },
        { label: "E-step", note: "γ = responsibilities" },
        { label: "M-step", note: "re-fit π, μ, Σ" },
        { label: "log L plateau?", note: "convergence check" },
        { label: "Soft clusters", note: "γ + ellipses", accent: "green" },
      ],
      decisionFunction: {
        text: "γ_{ik} = P(component k | x) ∝ π_k · N(x; μ_k, Σ_k)",
        mechanism: "Each component is a full Gaussian (its own mean, covariance, and weight); a point's responsibility for a component is proportional to that component's density at the point, times its mixing weight.",
        plot: { fn: (x) => 0.5 * Math.exp(-((x + 2) ** 2) / 2) + 0.5 * Math.exp(-((x - 2) ** 2) / (2 * 1.5 ** 2)), domain: [-6, 6], color: "var(--accent)", caption: "a 1D illustration: a mixture density is a weighted sum of Gaussian bumps" },
      },
      lossFunction: {
        text: "−Σᵢ log( Σ_k π_k · N(xᵢ; μ_k, Σ_k) )  (negative log-likelihood)",
        mechanism: "Minimized via Expectation-Maximization: the E-step computes responsibilities under current parameters, the M-step re-fits each component's mean/covariance/weight as the responsibility-weighted statistics - each step never decreases the likelihood.",
      },
      optimization: [
        { title: "Why EM rather than gradient ascent", formula: "log Σ_k π_k N(...) has no closed-form maximiser", note: "Introducing the hidden component label turns the log-of-a-sum into a sum-of-logs, which does have closed-form updates. EM exploits that at the cost of iterating." },
        { title: "Monotone likelihood", formula: "log L(t+1) ≥ log L(t)", note: "Each EM iteration maximises a lower bound that touches the true likelihood at the current parameters, so the likelihood can never decrease. Convergence is to a local optimum." },
        { title: "Cost", formula: "O(m · K · p²) per iteration (full covariance)", note: "The p² term comes from the covariance matrices. Diagonal covariance drops it to O(m·K·p), which matters in high dimensions." },
        { title: "Degeneracy", formula: "log L → ∞ as a component collapses onto one point", note: "A component with zero variance on a single point gives infinite density. The likelihood is genuinely unbounded, so a covariance floor (reg_covar) is not optional." },
        { title: "Initialisation", formula: "usually k-means, then EM", note: "EM is sensitive to starting values. Running k-means first gives sensible means and avoids many bad local optima." },
      ],
      output: "Soft responsibilities per point per component, plus each component's mean, covariance, and weight.",
      assumptions: [
        { name: "Components are Gaussian", why: "The whole model is a sum of normals; heavily skewed or bounded data is badly represented.", check: "Inspect the fitted ellipses against the data. Consider transforming skewed features first." },
        { name: "K is known", why: "Like k-means, the component count is an input.", check: "Scan K and pick the minimum BIC, or use a Dirichlet process mixture which infers it." },
        { name: "Enough data per component", why: "A full covariance needs p(p+1)/2 parameters per component, so high dimensions demand a lot of data.", check: "Compare m to K·p². If it is tight, use diagonal or tied covariance." },
        { name: "No exact duplicates or singular directions", why: "A collapsing component drives the likelihood to infinity.", check: "Keep reg_covar above zero; deduplicate identical rows." },
        { name: "Features scaled comparably", why: "Not strictly required with full covariance, but it improves conditioning and convergence.", check: "Standardize unless the units are already meaningful." },
      ],
      regularization: [
        { name: "reg_covar", formula: "Σ_k := Σ_k + εI", note: "A small ridge on every covariance diagonal. Prevents singular matrices and the infinite-likelihood degeneracy. Never set it to zero." },
        { name: "Full covariance", formula: "Kp(p+1)/2 parameters", note: "Most flexible: each component gets its own shape and orientation. Needs the most data." },
        { name: "Tied covariance", formula: "one shared Σ for all components", note: "All clusters share a shape but differ in position. A good middle ground when data is limited." },
        { name: "Diagonal covariance", formula: "Kp parameters, axis-aligned ellipses", note: "No rotation allowed. Much cheaper and far more stable in high dimensions." },
        { name: "Spherical covariance", formula: "K parameters, circles only", note: "The most constrained. This is essentially soft k-means." },
        { name: "Bayesian GMM", formula: "Dirichlet prior on π", note: "Lets unnecessary components shrink toward zero weight, effectively selecting K automatically." },
      ],
      hyperparameters: [
        { name: "n_components K", range: "1 - 20", increasing: "Better likelihood always, eventually one component per point.", strategy: "Scan K and select the minimum BIC. Unlike inertia, BIC genuinely penalises complexity." },
        { name: "covariance_type", range: "full / tied / diag / spherical", increasing: "Not applicable", strategy: "Full when data is plentiful and p is small. Move to diag as p grows or m shrinks." },
        { name: "reg_covar", range: "1e-6 - 1e-3", increasing: "More smoothing, rounder components, better conditioning.", strategy: "Raise it if you hit singular-matrix errors or see collapsed components." },
        { name: "n_init", range: "1 - 20", increasing: "Better chance of a good local optimum.", strategy: "Use at least 5. EM is more initialisation-sensitive than k-means." },
        { name: "init_params", range: "kmeans / k-means++ / random", increasing: "Not applicable", strategy: "k-means initialisation is the sensible default and rarely worth changing." },
        { name: "max_iter", range: "100 - 1000", increasing: "More iterations before stopping.", strategy: "100 is often too few for full covariance. Check that it converged rather than hit the cap." },
        { name: "tol", range: "1e-5 - 1e-3", increasing: "Stops sooner on smaller likelihood gains.", strategy: "Default is fine; loosen only if training time matters." },
      ],
      metrics: ["Log-likelihood", "BIC / AIC (for choosing k)", "Silhouette score on the resulting hard assignments", "Per-point log-likelihood as an anomaly score"],
      typicalUses: ["Soft/overlapping cluster segmentation", "Anomaly detection (low-likelihood points)", "Density estimation", "Speaker/topic modeling"],
      diagnostics: [
        "Plot BIC against K. The minimum is your model-selection answer, and unlike the k-means elbow it is usually unambiguous.",
        "Check whether EM converged or simply hit max_iter. Hitting the cap means the reported fit is not a local optimum.",
        "Look for components with near-zero weight or a collapsed covariance. Both indicate K is too high or reg_covar too low.",
        "Examine the responsibility matrix. Rows near uniform mean the point is genuinely ambiguous; that information is lost if you take a hard argmax.",
        "Run several initialisations and compare final log-likelihoods. A wide spread means the optimisation surface is rough.",
      ],
      advantages: [
        "Soft assignments quantify genuine ambiguity instead of forcing every point into one bucket.",
        "Full covariances capture elongated and rotated clusters that k-means cannot represent.",
        "It is a proper generative model, so you can sample new data and evaluate density at any point.",
        "BIC and AIC give a principled way to choose the number of components.",
        "Per-point likelihood doubles as a natural anomaly score.",
        "Different covariance types provide a clean dial between flexibility and stability.",
      ],
      limitations: [
        { name: "Assumes Gaussian components", note: "poor fit for skewed, bounded, or heavy-tailed data", fix: "transform features, or use a mixture of t-distributions." },
        { name: "Local optima", note: "EM converges to whichever basin it started in", fix: "k-means initialisation and multiple restarts." },
        { name: "Degenerate solutions", note: "a component can collapse onto a single point for infinite likelihood", fix: "reg_covar must stay above zero." },
        { name: "Parameter-hungry in high dimensions", note: "full covariance needs p(p+1)/2 numbers per component", fix: "diagonal or tied covariance, or reduce dimensions with PCA first." },
        { name: "K must be specified", note: "the model cannot infer component count on its own", fix: "BIC scan, or a Bayesian GMM with a Dirichlet prior." },
        { name: "Slower than k-means", note: "covariance estimation and inversion each iteration", fix: "use diagonal covariance, or k-means if soft assignment is not needed." },
      ],
      alternatives: [
        { name: "K-means", when: "Clusters are spherical and similarly sized, and you only need hard labels quickly." },
        { name: "DBSCAN", when: "Clusters are arbitrarily shaped or the data has substantial noise." },
        { name: "Bayesian GMM", when: "You want the component count inferred rather than chosen." },
        { name: "Kernel density estimation", when: "You need density estimation but no cluster structure." },
      ],
      pitfalls: [
        { problem: "Singular covariance matrix error", solution: "A component has collapsed. Raise reg_covar, lower K, or switch to diagonal covariance." },
        { problem: "Log-likelihood keeps rising with K", solution: "Expected. Likelihood always improves with more parameters. Use BIC, which penalises them." },
        { problem: "Very different results between runs", solution: "EM local optima. Raise n_init and use k-means initialisation." },
        { problem: "Fit is poor in high dimensions", solution: "Too many covariance parameters for the data. Use diag or tied, or apply PCA first." },
        { problem: "A component has almost no weight", solution: "K is too high. Reduce it, or use a Bayesian GMM which prunes automatically." },
        { problem: "Convergence warning at max_iter", solution: "Raise max_iter. Full covariance often needs several hundred iterations." },
      ],
      quickRef: [
        { name: "Mixture density", formula: "p(x) = Σ_k π_k N(x; μ_k, Σ_k)" },
        { name: "Responsibility", formula: "γ_{ik} = π_k N_k(xᵢ) / Σ_j π_j N_j(xᵢ)" },
        { name: "Effective count", formula: "N_k = Σᵢ γ_{ik}" },
        { name: "Weight update", formula: "π_k = N_k / m" },
        { name: "Mean update", formula: "μ_k = Σᵢ γ_{ik} xᵢ / N_k" },
        { name: "Covariance update", formula: "Σ_k = Σᵢ γ_{ik}(xᵢ−μ_k)(xᵢ−μ_k)ᵀ / N_k" },
        { name: "Objective", formula: "log L = Σᵢ log Σ_k π_k N_k(xᵢ)" },
        { name: "BIC", formula: "−2 log L + d log m" },
      ],
      code: `from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np

# BIC actually penalises complexity, so unlike k-means inertia
# it can be minimised directly to choose K.
scores = {}
for k in range(1, 11):
    gm = GaussianMixture(
        n_components=k, covariance_type="full",
        n_init=5, reg_covar=1e-6, random_state=42,
    ).fit(X)
    scores[k] = gm.bic(X)
best_k = min(scores, key=scores.get)

gm = GaussianMixture(n_components=best_k, covariance_type="full",
                     n_init=5, max_iter=500, random_state=42).fit(X)

resp   = gm.predict_proba(X)     # soft responsibilities, not just labels
logp   = gm.score_samples(X)     # per-point log density
outlier = logp < np.percentile(logp, 1)   # low likelihood = anomaly

# Let the model prune unneeded components itself:
bgm = BayesianGaussianMixture(n_components=20,
                              weight_concentration_prior=0.01).fit(X)`,
      whyChain: [
        { q: "How is a GMM different from k-means?", a: "Three ways: assignments are soft probabilities rather than hard labels, each cluster has its own covariance so it can be stretched and rotated, and each has its own mixing weight so clusters can differ in size. K-means is the limiting case with spherical equal covariance and hard assignment." },
        { q: "Why do we need EM at all?", a: "The log-likelihood contains a log of a sum over components, which has no closed-form maximiser. If you knew which component each point came from, the sum would vanish and the maximum-likelihood estimates would be plain weighted means and covariances. EM alternates between estimating that hidden membership and using it." },
        { q: "Why is the likelihood guaranteed not to decrease?", a: "The E-step constructs a lower bound on the log-likelihood that touches it exactly at the current parameters. The M-step maximises that bound. So the true likelihood is at least as high as the bound's new value, which is at least the old one." },
        { q: "Why can the likelihood become infinite?", a: "If a component's mean sits exactly on a data point and its covariance shrinks to zero, the density at that point diverges. The likelihood is genuinely unbounded above, so the global maximum is a degenerate solution. reg_covar puts a floor under the variance to rule it out." },
        { q: "Why does BIC work for choosing K when inertia does not?", a: "Log-likelihood, like inertia, always improves with more components. BIC subtracts d·log(m) where d is the parameter count, so adding a component must improve the fit by more than it costs in complexity. That creates a genuine minimum." },
        { q: "When would you use diagonal instead of full covariance?", a: "When p is large or m is small. Full covariance needs p(p+1)/2 parameters per component, so in 50 dimensions that is 1275 numbers per cluster. Diagonal needs 50 and is far more stable, at the cost of forcing axis-aligned ellipses." },
        { q: "How does a GMM detect anomalies?", a: "It is a density model, so score_samples gives log p(x) for any point. Points in low-density regions have very negative scores. Thresholding that is a principled outlier test, which k-means cannot provide." },
        { q: "What does a responsibility of 0.5 mean?", a: "The point is genuinely equidistant in probability between two components, so the model is telling you it is ambiguous. Taking a hard argmax discards exactly this information, which is often the most interesting part." },
      ],
      parameters: [
        { name: "k (components)", effect: "Number of Gaussians. Too few underfits multi-modal data; too many overfits / creates degenerate tiny components." },
        { name: "covariance shape", effect: "Full covariance (used here) lets clusters be elongated/rotated; diagonal or spherical covariance is a more restrictive, faster alternative." },
        { name: "EM iterations", effect: "How long to alternate E/M steps before stopping; convergence is judged by the log-likelihood plateauing." },
      ],
      metrics: ["Log-likelihood", "BIC / AIC (for choosing k)", "Silhouette score on the resulting hard assignments"],
      typicalUses: ["Soft/overlapping cluster segmentation", "Anomaly detection (low-likelihood points)", "Density estimation", "Speaker/topic modeling"],
      workedExample: {
        setup: "Two fixed 1D components: N(0,1) and N(4,1), both weight 0.5. Compute the E-step responsibility for x=1.",
        steps: [
          "pdf₁(1) = (1/√2π)·e^(−(1−0)²/2) = 0.3989×e^−0.5 = 0.3989×0.6065 ≈ 0.2420.",
          "pdf₂(1) = (1/√2π)·e^(−(1−4)²/2) = 0.3989×e^−4.5 = 0.3989×0.01111 ≈ 0.00443.",
          "Unnormalized responsibilities: r₁ = 0.5×0.2420 = 0.1210, r₂ = 0.5×0.00443 = 0.00222.",
          "Normalize: γ₁ = 0.1210/(0.1210+0.00222) ≈ 0.982, γ₂ ≈ 0.018.",
        ],
        result: "x=1 is assigned ~98.2% responsibility to component 1, 1.8% to component 2 - soft, not all-or-nothing like K-Means",
      },
    },
    mount,
  });
})();
