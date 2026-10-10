(() => {
  const { ui, col, fmt } = Lab;
  function logN(x, mu, S) {
    const d = MLU.det2x2(S), inv = MLU.inverse2x2(S);
    const a = x[0] - mu[0], b = x[1] - mu[1];
    return -0.5 * (a * (inv[0][0] * a + inv[0][1] * b) + b * (inv[1][0] * a + inv[1][1] * b)) - Math.log(2 * Math.PI) - 0.5 * Math.log(Math.max(d, 1e-12));
  }
  const lse = (a) => { const m = Math.max(...a); return m + Math.log(a.reduce((s, v) => s + Math.exp(v - m), 0)); };
  function constrain(S, type) {
    if (type === "diag") return [[S[0][0], 0], [0, S[1][1]]];
    if (type === "spherical") { const v = (S[0][0] + S[1][1]) / 2; return [[v, 0], [0, v]]; }
    return S;
  }
  function estep(X, comp) {
    let ll = 0;
    const R = X.map((x) => {
      const l = comp.map((c) => Math.log(c.pi + 1e-12) + logN(x, c.mu, c.S));
      const z = lse(l); ll += z;
      return l.map((v) => Math.exp(v - z));
    });
    return { R, ll };
  }
  function mstep(X, R, k, type) {
    const n = X.length;
    return Array.from({ length: k }, (_, j) => {
      const Nk = R.reduce((s, r) => s + r[j], 0) + 1e-9;
      const mu = [0, 0];
      X.forEach((x, i) => { mu[0] += (R[i][j] * x[0]) / Nk; mu[1] += (R[i][j] * x[1]) / Nk; });
      const S = [[0, 0], [0, 0]];
      X.forEach((x, i) => { const a = x[0] - mu[0], b = x[1] - mu[1]; S[0][0] += (R[i][j] * a * a) / Nk; S[0][1] += (R[i][j] * a * b) / Nk; S[1][1] += (R[i][j] * b * b) / Nk; });
      S[1][0] = S[0][1]; S[0][0] += 1e-3; S[1][1] += 1e-3;
      return { pi: Nk / n, mu, S: constrain(S, type) };
    });
  }
  function initComp(X, k) {
    const idx = MLU.shuffle([...X.keys()]).slice(0, k);
    const v = MLU.covMatrix(X, MLU.meanVec(X));
    const s = (v[0][0] + v[1][1]) / 2 / Math.max(1, k);
    return idx.map((i) => ({ pi: 1 / k, mu: X[i].slice(), S: [[s, 0], [0, s]] }));
  }
  function nParams(k, type) { const cov = type === "full" ? 3 : type === "diag" ? 2 : 1; return k * (2 + cov) + (k - 1); }
  function fitQuick(X, k, type) {
    let comp = initComp(X, k), ll = -Infinity;
    for (let it = 0; it < 80; it++) { const e = estep(X, comp); comp = mstep(X, e.R, k, type); if (Math.abs(e.ll - ll) < 1e-4) { ll = e.ll; break; } ll = e.ll; }
    return { comp, ll: estep(X, comp).ll };
  }

  function buildFrames(pts, k, type, iters) {
    const X = pts.map((p) => [p.x, p.y]);
    if (X.length < k + 1) return [{ phase: "idle", title: "Need more points", text: "", st: { comp: [], R: null } }];
    let comp = initComp(X, k);
    const frames = [{ phase: "init", major: true, arch: ["init"], title: `Initialise ${k} Gaussians`, text: "Means start on random data points, covariances as broad circles, mixing weights equal. Like k-means, EM only finds a local optimum, so initialisation matters.", tex: "p(x)=\\sum_{j=1}^{k}\\pi_j\\,\\mathcal N(x\\mid\\mu_j,\\Sigma_j)", st: { comp: JSON.parse(JSON.stringify(comp)), R: null }, m: {} }];
    let prev = -Infinity;
    for (let it = 1; it <= iters; it++) {
      const { R, ll } = estep(X, comp);
      frames.push({ phase: "e", major: it === 1, arch: ["e", "ll"], title: `Iteration ${it} · E-step: soft responsibilities`, text: `Each point is shared among the components in proportion to how well each explains it. Colours are blended by responsibility, so points between clusters look mixed. Log-likelihood <b>${ll.toFixed(2)}</b>.`, tex: "\\gamma_{ij}=\\frac{\\pi_j\\,\\mathcal N(x_i\\mid\\mu_j,\\Sigma_j)}{\\sum_l \\pi_l\\,\\mathcal N(x_i\\mid\\mu_l,\\Sigma_l)}", st: { comp: JSON.parse(JSON.stringify(comp)), R }, m: { ll: ll / X.length } });
      const old = comp;
      comp = mstep(X, R, k, type);
      frames.push({ phase: "m", arch: ["m"], title: `Iteration ${it} · M-step: re-fit each Gaussian`, text: `Weighted means, covariances and mixing weights, using the responsibilities as soft counts. Mixing weights: ${comp.map((c) => c.pi.toFixed(2)).join(", ")}.`, tex: "N_j=\\sum_i\\gamma_{ij},\\quad \\mu_j=\\frac{1}{N_j}\\sum_i\\gamma_{ij}x_i,\\quad \\Sigma_j=\\frac1{N_j}\\sum_i\\gamma_{ij}(x_i-\\mu_j)(x_i-\\mu_j)^\\top,\\quad \\pi_j=\\frac{N_j}{n}", st: { comp: JSON.parse(JSON.stringify(comp)), prev: old, R }, m: { ll: ll / X.length } });
      if (Math.abs(ll - prev) < 1e-3) {
        frames.push({ phase: "done", major: true, arch: ["ll", "out"], title: `Converged after ${it} iterations`, text: `The log-likelihood stopped improving (it never decreases under EM). Each point gets a probability of belonging to every cluster, and clusters can be elliptical, tilted and of different sizes, which k-means cannot do.`, st: { comp: JSON.parse(JSON.stringify(comp)), R }, m: { ll: ll / X.length } });
        break;
      }
      prev = ll;
    }
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "gmm", fps: 2.5,
      legend: ui.legend([["var(--text)", "1σ / 2σ ellipses", "ring"], ["var(--text)", "mean", "ring"]]),
      tools: ui.btn("rerun", "rerun (new init)", true) + ui.btn("bic", "BIC scan", false, "fit k = 1…7 and compare BIC"),
      hint: "point colour is a blend of responsibilities · click to add points · drag / double-click to edit",
      data: { kind: "cluster", presets: ["aniso", "blobs3", "varied", "blobs4", "moons", "outliers"], def: "aniso", n: 150, noise: 0.3 },
      side: [{ title: "model", html: ui.range("k", "components k", 1, 7, 1, 3) + ui.select("cov", "covariance type", [["full", "full (tilted ellipses)"], ["diag", "diagonal (axis-aligned)"], ["spherical", "spherical (circles)"]], "full") + ui.range("it", "max EM iterations", 5, 100, 5, 40) + ui.check("dens", "show density contours", true) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "init", label: "init π, μ, Σ", sub: "random means", x: 56, y: 44, kind: "param" },
          { id: "e", label: "E-step", sub: "responsibilities γ", x: 184, y: 44, kind: "op", tex: "\\gamma_{ij}\\propto\\pi_j\\mathcal N(x_i|\\mu_j,\\Sigma_j)" },
          { id: "m", label: "M-step", sub: "weighted MLE", x: 184, y: 140, kind: "op" },
          { id: "ll", label: "log p(X)", sub: "never decreases", x: 316, y: 44, kind: "loss", tex: "\\sum_i\\log\\sum_j\\pi_j\\mathcal N(x_i|\\mu_j,\\Sigma_j)" },
          { id: "out", label: "soft labels", sub: "P(cluster | x)", x: 316, y: 140, kind: "out" },
        ],
        edges: [{ from: "init", to: "e" }, { from: "e", to: "m" }, { from: "m", to: "e", bend: -30, dash: true, label: "repeat" }, { from: "e", to: "ll" }, { from: "ll", to: "out", label: "converged" }],
      },
      phases: { init: { label: "init", color: 3 }, e: { label: "E-step", color: 0 }, m: { label: "M-step", color: 2 }, done: { label: "converged", color: 2 } },
      panels: [{ id: "resp", title: "responsibilities of a sample point", sub: "hover a point" }],
      charts: [{ id: "ll", title: "log-likelihood per point", series: [{ key: "ll", label: "ℓ / n", color: "var(--series-1)" }] }, { id: "bic", title: "model selection: BIC vs k (lower is better)", custom: true }],
      onFrame: render, onData: () => { bic = null; drawBic(); rerun(); },
    });
    lab.editable({ newPoint: (x, y) => ({ x, y, label: 0 }), onChange: () => rerun({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    let seed = 4, bic = null, hover = 0, noAnim = false;
    function rerun({ at = "start" } = {}) { MLU.seedRng(seed); lab.load(buildFrames(lab.points, lab.val("k"), lab.val("cov"), lab.val("it")), { at, play: at === "start" }); }
    function drawBic() {
      const el = lab.charts.bic.el; el.innerHTML = "";
      const P = Lab.xy(el, { w: 360, h: 130, xd: [1, 7], yd: bic ? [Math.min(...bic.map((b) => b.v)) * 0.98, Math.max(...bic.map((b) => b.v)) * 1.02] : [0, 1], xLabel: "k" });
      if (!bic) { P.g.append("text").attr("class", "ax-label").attr("x", 190).attr("y", 60).attr("text-anchor", "middle").text("press “BIC scan” to compute"); return; }
      P.line("bic", bic.map((b) => [b.k, b.v]), { color: "var(--series-5)" });
      const best = bic.reduce((a, b) => (b.v < a.v ? b : a));
      P.g.append("circle").attr("cx", P.x(best.k)).attr("cy", P.y(best.v)).attr("r", 5).attr("fill", "var(--series-5)");
      el.closest(".lab-card").querySelector(".chart-legend").innerHTML = `best k = <b>${best.k}</b>`;
    }
    lab.q('[data-b="bic"]').onclick = () => {
      const X = lab.points.map((p) => [p.x, p.y]); if (X.length < 10) return;
      MLU.seedRng(77);
      bic = d3.range(1, 8).map((k) => { let best = null; for (let r = 0; r < 3; r++) { const f = fitQuick(X, k, lab.val("cov")); if (!best || f.ll > best.ll) best = f; } return { k, v: -2 * best.ll + nParams(k, lab.val("cov")) * Math.log(X.length) }; });
      drawBic();
    };
    drawBic();
    const ellG = L.mid.append("g"), muG = L.top.append("g");
    const respBars = Lab.bars(lab.R("panel-resp"), { w: 360, h: 120, yd: [0, 1] });
    function render(f) {
      const st = f.st, comp = st.comp;
      if (!comp.length) { L.bg.selectAll("*").remove(); lab.drawPoints({}); return; }
      if (lab.val("dens")) {
        const grid = Lab.sampleGrid(pl, (x, y) => lse(comp.map((c) => Math.log(c.pi + 1e-12) + logN([x, y], c.mu, c.S))), 80);
        L.bg.selectAll("*").remove();
        const vals = grid.values.filter(isFinite); const hi = Math.max(...vals);
        Lab.contour(L.under, pl, grid, d3.range(hi - 7, hi, 0.9), { cls: "dens", width: 0.8, stroke: "var(--text-faint)" });
      } else { L.bg.selectAll("*").remove(); L.under.selectAll("path.dens").remove(); }
      ellG.selectAll("*").remove(); muG.selectAll("*").remove();
      comp.forEach((c, j) => {
        for (const s of [1, 2]) {
          const pth = ellG.append("path").attr("fill", s === 1 ? col(j) : "none").attr("fill-opacity", 0.07 + c.pi * 0.15).attr("stroke", col(j)).attr("stroke-width", s === 1 ? 2.2 : 1.2).attr("stroke-dasharray", s === 2 ? "5 4" : null);
          if (st.prev && st.prev[j] && !noAnim) pth.attr("d", Lab.ellipsePath(pl, st.prev[j].mu, st.prev[j].S, s)).transition().duration(450).attr("d", Lab.ellipsePath(pl, c.mu, c.S, s));
          else pth.attr("d", Lab.ellipsePath(pl, c.mu, c.S, s));
        }
        const src = st.prev && st.prev[j] && !noAnim ? st.prev[j] : c;
        const m = muG.append("g").attr("transform", `translate(${pl.x(src.mu[0])},${pl.y(src.mu[1])})`);
        m.append("circle").attr("r", 7).attr("fill", "var(--bg-inset)").attr("stroke", col(j)).attr("stroke-width", 2.4);
        m.append("text").attr("class", "lbl-strong").attr("text-anchor", "middle").attr("y", -11).text(`π=${c.pi.toFixed(2)}`);
        m.transition().duration(450).attr("transform", `translate(${pl.x(c.mu[0])},${pl.y(c.mu[1])})`);
      });
      const R = st.R;
      const pts = lab.drawPoints({ color: (d, i) => (R ? Lab.mixColor(R[i]) : "var(--text-faint)"), r: (d, i) => (i === hover ? 7 : 4.3), stroke: (d, i) => (i === hover ? "var(--text)" : "var(--bg)"), sw: (d, i) => (i === hover ? 2 : 1) });
      pts.on("mouseenter.r", (ev, d) => { hover = lab.points.indexOf(d); noAnim = true; lab.player.redraw(); noAnim = false; });
      const r = R && R[hover] ? R[hover] : comp.map(() => 0);
      respBars.set(r.map((v, j) => ({ label: "comp " + j, value: v, color: col(j), text: (v * 100).toFixed(0) + "%" })));
      const ll = R ? estep(lab.points.map((p) => [p.x, p.y]), comp).ll : null;
      lab.readout(`<div class="kv"><span>log-likelihood</span><b>${fmt(ll, 1)}</b><span>BIC</span><b>${ll == null ? "–" : (-2 * ll + nParams(comp.length, lab.val("cov")) * Math.log(lab.points.length)).toFixed(1)}</b><span>parameters</span><b>${nParams(comp.length, lab.val("cov"))}</b></div>`);
    }
    lab.watch(["k", "cov", "it"], () => rerun());
    lab.watch(["dens"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="rerun"]').onclick = () => { seed = (seed * 13 + 7) % 9973; rerun(); };
    lab.loadPreset("aniso");
    return lab.destroy;
  }

  MLApp.register({
    id: "gmm",
    name: "Gaussian Mixture Model",
    category: "Unsupervised - Clustering",
    tagline: "EM with soft responsibilities, BIC",
    description: "A mixture of Gaussians fitted by Expectation-Maximisation. Watch responsibilities blend point colours in the E-step and ellipses glide to new shapes in the M-step, track the log-likelihood, and pick k with a BIC scan.",
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
