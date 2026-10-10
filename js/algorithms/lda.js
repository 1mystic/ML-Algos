(() => {
  const { ui, col, fmt } = Lab;

  function stats(pts) {
    const classes = [...new Set(pts.map((p) => p.label))].sort();
    const mu = MLU.meanVec(pts.map((p) => [p.x, p.y]));
    const per = classes.map((c) => {
      const m = pts.filter((p) => p.label === c).map((p) => [p.x, p.y]);
      const mc = MLU.meanVec(m);
      return { c, n: m.length, mu: mc, cov: MLU.covMatrix(m, mc) };
    });
    const Sw = [[0, 0], [0, 0]], Sb = [[0, 0], [0, 0]];
    for (const k of per) {
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
        Sw[i][j] += k.cov[i][j] * k.n;
        Sb[i][j] += k.n * (k.mu[i] - mu[i]) * (k.mu[j] - mu[j]);
      }
    }
    const tot = MLU.covMatrix(pts.map((p) => [p.x, p.y]), mu);
    return { classes, mu, per, Sw, Sb, tot };
  }
  const quad = (M, v) => v[0] * (M[0][0] * v[0] + M[0][1] * v[1]) + v[1] * (M[1][0] * v[0] + M[1][1] * v[1]);
  function fisherDir(s) {
    // largest eigenvector of Sw^-1 Sb
    const Swi = MLU.inverse2x2([[s.Sw[0][0] + 1e-6, s.Sw[0][1]], [s.Sw[1][0], s.Sw[1][1] + 1e-6]]);
    const M = MLU.matMul(Swi, s.Sb);
    // power iteration (M is not symmetric)
    let v = [1, 0.3];
    for (let i = 0; i < 100; i++) { const nv = [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]; const n = Math.hypot(...nv) || 1; v = [nv[0] / n, nv[1] / n]; }
    return v;
  }

  function buildFrames(pts) {
    const s = stats(pts);
    if (s.classes.length < 2 || s.per.some((k) => k.n < 2)) return { frames: [{ phase: "idle", title: "Need two classes with at least two points each", text: "Load an example or add points.", st: {} }], s: null };
    const frames = [];
    frames.push({ phase: "means", major: true, arch: ["data", "means"], title: "Compute each class mean", text: `Class means: ${s.per.map((k) => `μ${k.c} = (${k.mu[0].toFixed(2)}, ${k.mu[1].toFixed(2)})`).join(", ")}. LDA wants a direction along which these means are far apart <i>relative to</i> how spread out each class is.`, tex: "\\mu_c=\\frac1{N_c}\\sum_{i:y_i=c}x_i", st: { means: true } });
    frames.push({ phase: "scatter", arch: ["means", "sw", "sb"], title: "Within-class and between-class scatter", text: "S<sub>W</sub> sums each class's covariance (ellipses): the noise. S<sub>B</sub> measures how far the class means are from the overall mean: the signal. LDA assumes all classes share one covariance, so it pools them.", tex: "S_W=\\sum_c\\sum_{i\\in c}(x_i-\\mu_c)(x_i-\\mu_c)^\\top,\\qquad S_B=\\sum_c N_c(\\mu_c-\\mu)(\\mu_c-\\mu)^\\top", st: { means: true, ell: true } });
    const steps = 36;
    for (let k = 0; k <= steps; k++) {
      const th = (k / steps) * Math.PI;
      const w = [Math.cos(th), Math.sin(th)];
      const J = quad(s.Sb, w) / Math.max(1e-9, quad(s.Sw, w));
      frames.push({ phase: "scan", arch: ["sw", "sb", "ratio"], title: `Try direction θ = ${((th * 180) / Math.PI).toFixed(0)}°`, text: `Project every point onto the line. Fisher's ratio J(w) = <b>${J.toFixed(3)}</b>: between-class variance of the projections divided by within-class variance. The histograms below show how well the classes separate on this line.`, tex: "J(w)=\\frac{w^\\top S_B\\,w}{w^\\top S_W\\,w}", st: { means: true, ell: true, w, th } });
    }
    const w = fisherDir(s);
    const th = Math.atan2(w[1], w[0]);
    const Jb = quad(s.Sb, w) / quad(s.Sw, w);
    frames.push({ phase: "solve", major: true, arch: ["ratio", "eig", "proj"], title: `Closed form: w ∝ S_W⁻¹(μ₁ − μ₀), J = ${Jb.toFixed(3)}`, text: "No search is needed: maximising the Rayleigh quotient is a generalised eigenproblem. For two classes the answer is simply S_W⁻¹ times the mean difference, so LDA trains in one shot.", tex: "S_W^{-1}S_B\\,w=\\lambda w\\;\\;\\Rightarrow\\;\\; w^\\star\\propto S_W^{-1}(\\mu_1-\\mu_0)", st: { means: true, ell: true, w, th: ((th % Math.PI) + Math.PI) % Math.PI, best: true } });
    frames.push({ phase: "classify", major: true, arch: ["proj", "out"], title: "Classify: threshold the projection (or use the Gaussian posterior)", text: "With shared covariance Σ, the class posteriors are logistic in x, so the boundary is a straight line, perpendicular to Σ⁻¹-adjusted mean difference. The dotted line shows the PCA direction for comparison: maximum variance is not maximum separation.", tex: "\\delta_c(x)=x^\\top\\Sigma^{-1}\\mu_c-\\tfrac12\\mu_c^\\top\\Sigma^{-1}\\mu_c+\\log\\pi_c", st: { means: true, ell: true, w, th: ((th % Math.PI) + Math.PI) % Math.PI, best: true, field: true } });
    return { frames, s };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "lda", fps: 4, classes: 3,
      legend: ui.legend([["var(--accent)", "projection direction w", "line"], ["var(--text-faint)", "PCA direction", "dash"]]),
      tools: ui.btn("replay", "replay", true),
      hint: "click adds the selected class · drag points and the optimal direction updates",
      data: {
        kind: "class", presets: ["elong", "linear", "blobs3", "overlap", "aniso"], def: "elong", n: 120, noise: 0.3, names: { elong: "elongated parallel classes" },
        custom: {
          elong: ({ n, noise }) => Array.from({ length: n }, (_, i) => {
            const l = i % 2, t = MLU.randn() * (4 + noise * 2), u = MLU.randn() * (0.7 + noise);
            return { x: Math.max(-9.8, Math.min(9.8, t * 0.8 - u * 0.6 + (l ? 1.3 : -1.3))), y: Math.max(-9.8, Math.min(9.8, t * 0.6 + u * 0.8 + (l ? -1.6 : 1.6))), label: l };
          }),
        },
      },
      side: [{ title: "view", html: ui.check("pca", "show PCA direction", true) + ui.check("feet", "projection lines", true) }],
      arch: {
        w: 380, h: 200,
        nodes: [
          { id: "data", label: "X, y", sub: "labelled", x: 34, y: 95, w: 56, kind: "data" },
          { id: "means", label: "class means μ_c", x: 130, y: 36, kind: "op" },
          { id: "sw", label: "S_W", sub: "within-class", x: 130, y: 95, w: 74, kind: "op" },
          { id: "sb", label: "S_B", sub: "between-class", x: 130, y: 154, w: 74, kind: "op" },
          { id: "ratio", label: "J(w)", sub: "Fisher ratio", x: 222, y: 112, w: 70, kind: "loss", tex: "J=\\frac{w^\\top S_Bw}{w^\\top S_Ww}" },
          { id: "eig", label: "eig(S_W⁻¹S_B)", sub: "closed form", x: 312, y: 36, kind: "param" },
          { id: "proj", label: "z = wᵀx", sub: "project", x: 318, y: 112, w: 70, kind: "op" },
          { id: "out", label: "class", sub: "threshold", x: 318, y: 168, w: 64, kind: "out" },
        ],
        edges: [{ from: "data", to: "means" }, { from: "data", to: "sw" }, { from: "data", to: "sb" }, { from: "sw", to: "ratio" }, { from: "sb", to: "ratio" }, { from: "ratio", to: "eig" }, { from: "eig", to: "proj" }, { from: "proj", to: "out" }],
      },
      phases: { means: { label: "means", color: 3 }, scatter: { label: "scatter", color: 4 }, scan: { label: "direction scan", color: 0 }, solve: { label: "closed form", color: 2 }, classify: { label: "classify", color: 2 } },
      panels: [{ id: "hist", title: "projected 1-D distributions" }],
      charts: [{ id: "J", title: "Fisher ratio J(θ) vs projected variance", custom: true }],
      onFrame: render, onData: () => replay(),
    });
    lab.editable({ onChange: () => replay({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    let S = null, frames = [];
    function replay({ at = "start" } = {}) {
      const r = buildFrames(lab.points);
      S = r.s; frames = r.frames;
      drawJ();
      lab.load(frames, { at, play: at === "start" });
    }
    let JP = null;
    function drawJ() {
      const el = lab.charts.J.el; el.innerHTML = "";
      JP = Lab.xy(el, { w: 360, h: 130, xd: [0, 180], yd: [0, 1.05], xLabel: "θ (deg)" });
      if (!S) return;
      const ths = d3.range(0, 180.5, 2);
      const Js = ths.map((t) => { const w = [Math.cos((t * Math.PI) / 180), Math.sin((t * Math.PI) / 180)]; return quad(S.Sb, w) / quad(S.Sw, w); });
      const Vs = ths.map((t) => { const w = [Math.cos((t * Math.PI) / 180), Math.sin((t * Math.PI) / 180)]; return quad(S.tot, w); });
      const mj = Math.max(...Js), mv = Math.max(...Vs);
      JP.line("J", ths.map((t, i) => [t, Js[i] / mj]), { color: "var(--series-3)" });
      JP.line("V", ths.map((t, i) => [t, Vs[i] / mv]), { color: "var(--text-faint)", dash: "4 3" });
      el.closest(".lab-card").querySelector(".chart-legend").innerHTML = `<span class="legend-item"><span class="swatch" style="background:var(--series-3)"></span>J(θ) / max</span><span class="legend-item"><span class="swatch" style="background:var(--text-faint)"></span>variance (PCA objective)</span>`;
    }
    const histEl = lab.R("panel-hist");
    const ellG = L.mid.append("g"), lineG = L.mid.append("g"), meanG = L.top.append("g");
    function render(f) {
      const st = f.st;
      ellG.selectAll("*").remove(); lineG.selectAll("*").remove(); meanG.selectAll("*").remove(); L.links.selectAll("*").remove();
      if (!S) { L.bg.selectAll("*").remove(); lab.drawPoints({}); return; }
      if (st.field) {
        const pooled = [[S.Sw[0][0] / lab.points.length, S.Sw[0][1] / lab.points.length], [S.Sw[1][0] / lab.points.length, S.Sw[1][1] / lab.points.length]];
        const Pi = MLU.inverse2x2(pooled);
        const disc = (x, y) => S.per.map((k) => { const a = [Pi[0][0] * k.mu[0] + Pi[0][1] * k.mu[1], Pi[1][0] * k.mu[0] + Pi[1][1] * k.mu[1]]; return x * a[0] + y * a[1] - 0.5 * (k.mu[0] * a[0] + k.mu[1] * a[1]) + Math.log(k.n / lab.points.length); });
        Lab.field(L.bg, pl, (x, y) => { const d = disc(x, y); return S.per[d.indexOf(Math.max(...d))].c; }, (c) => Lab.classColor(c, 1, 0.18), { res: 120, smooth: false });
      } else L.bg.selectAll("*").remove();
      if (st.ell) for (const k of S.per) for (const s of [1, 2]) ellG.append("path").attr("d", Lab.ellipsePath(pl, k.mu, k.cov, s)).attr("fill", "none").attr("stroke", col(k.c)).attr("stroke-width", s === 1 ? 1.8 : 1).attr("stroke-dasharray", s === 2 ? "4 4" : null);
      const mu = S.mu;
      if (st.w) {
        const w = st.w;
        lineG.append("line").attr("x1", pl.x(mu[0] - w[0] * 14)).attr("y1", pl.y(mu[1] - w[1] * 14)).attr("x2", pl.x(mu[0] + w[0] * 14)).attr("y2", pl.y(mu[1] + w[1] * 14)).attr("stroke", "var(--accent)").attr("stroke-width", st.best ? 2.8 : 2);
        if (lab.val("feet")) {
          L.links.selectAll("line").data(lab.points).enter().append("line")
            .attr("x1", (p) => pl.x(p.x)).attr("y1", (p) => pl.y(p.y))
            .attr("x2", (p) => { const t = (p.x - mu[0]) * w[0] + (p.y - mu[1]) * w[1]; return pl.x(mu[0] + t * w[0]); })
            .attr("y2", (p) => { const t = (p.x - mu[0]) * w[0] + (p.y - mu[1]) * w[1]; return pl.y(mu[1] + t * w[1]); })
            .attr("stroke", (p) => col(p.label)).attr("stroke-opacity", 0.25);
          lineG.selectAll("circle").data(lab.points).enter().append("circle").attr("r", 2.4)
            .attr("cx", (p) => { const t = (p.x - mu[0]) * w[0] + (p.y - mu[1]) * w[1]; return pl.x(mu[0] + t * w[0]); })
            .attr("cy", (p) => { const t = (p.x - mu[0]) * w[0] + (p.y - mu[1]) * w[1]; return pl.y(mu[1] + t * w[1]); })
            .attr("fill", (p) => col(p.label));
        }
        // histograms of projections
        const proj = lab.points.map((p) => ({ z: (p.x - mu[0]) * w[0] + (p.y - mu[1]) * w[1], c: p.label }));
        histEl.innerHTML = "";
        const zmax = Math.max(1, ...proj.map((d) => Math.abs(d.z)));
        const bins = d3.bin().domain([-zmax, zmax]).thresholds(24);
        const groups = S.classes.map((c) => ({ c, b: bins(proj.filter((d) => d.c === c).map((d) => d.z)) }));
        const ymax = Math.max(1, ...groups.flatMap((g) => g.b.map((b) => b.length)));
        const HP = Lab.xy(histEl, { w: 360, h: 130, xd: [-zmax, zmax], yd: [0, ymax], xLabel: "z = wᵀ(x − μ)" });
        for (const g of groups) HP.g.selectAll(null).data(g.b).enter().append("rect").attr("x", (b) => HP.x(b.x0) + 0.5).attr("width", (b) => Math.max(0, HP.x(b.x1) - HP.x(b.x0) - 1)).attr("y", (b) => HP.y(b.length)).attr("height", (b) => HP.y(0) - HP.y(b.length)).attr("fill", col(g.c)).attr("opacity", 0.45);
        if (JP && st.th != null) JP.vline((st.th * 180) / Math.PI);
      }
      if (lab.val("pca")) {
        const e = MLU.jacobiEigen(S.tot).vectors[0];
        lineG.append("line").attr("x1", pl.x(mu[0] - e[0] * 14)).attr("y1", pl.y(mu[1] - e[1] * 14)).attr("x2", pl.x(mu[0] + e[0] * 14)).attr("y2", pl.y(mu[1] + e[1] * 14)).attr("stroke", "var(--text-faint)").attr("stroke-width", 1.6).attr("stroke-dasharray", "6 5");
      }
      lab.drawPoints({ color: (d) => col(d.label), opacity: st.w && lab.val("feet") ? 0.75 : 1 });
      if (st.means) for (const k of S.per) meanG.append("path").attr("d", d3.symbol(d3.symbolCross, 160)()).attr("transform", `translate(${pl.x(k.mu[0])},${pl.y(k.mu[1])}) rotate(45)`).attr("fill", col(k.c)).attr("stroke", "var(--bg)").attr("stroke-width", 1.5);
      const w = fisherDir(S);
      lab.readout(`<div class="kv"><span>optimal w</span><b>(${fmt(w[0], 2)}, ${fmt(w[1], 2)})</b><span>best J</span><b>${fmt(quad(S.Sb, w) / quad(S.Sw, w))}</b><span>classes</span><b>${S.classes.length}</b><span>max components</span><b>${Math.min(S.classes.length - 1, 2)}</b></div>`);
    }
    lab.watch(["pca", "feet"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.loadPreset("elong");
    return lab.destroy;
  }

  MLApp.register({
    id: "lda",
    name: "Linear Discriminant Analysis",
    category: "Supervised - Classification",
    tagline: "Fisher's projection, shared covariance",
    isNew: true,
    description: "Find the direction that best separates the classes, not the one with the most variance. Watch a candidate line sweep around while Fisher's ratio is scored at every angle, then see the closed-form answer and how it differs from PCA.",
    info: {
      type: "Supervised - Classification and supervised dimensionality reduction. Linear, generative (Gaussian classes with a shared covariance).",
      scenario: "Classes that are roughly Gaussian with similar spread; as a fast, stable classifier on small data, or as a supervised projection to ≤ C − 1 dimensions before another model.",
      inputs: "Feature vectors x ∈ ℝᵈ with class labels y ∈ {1..C}.",
      output: "A projection matrix W (d × (C−1)) and class predictions from linear discriminant scores.",
      intuition: {
        definition: "LDA looks for a line onto which the classes project into tight, well-separated bumps. It scores a direction by the ratio of between-class spread to within-class spread and maximises that ratio in closed form. Read generatively, it models each class as a Gaussian with the same covariance, which makes the optimal boundary linear.",
        steps: ["Compute each class mean and the pooled within-class scatter S_W.", "Compute the between-class scatter S_B from the means.", "Solve the eigenproblem S_W⁻¹S_B w = λw.", "Project onto the top C−1 eigenvectors, or classify with the discriminant scores."],
        applications: ["Face recognition (Fisherfaces)", "Credit scoring and medical diagnosis on small tabular data", "Supervised dimensionality reduction before k-NN or clustering", "Brain-computer interfaces (EEG signal classification)"],
      },
      math: [
        { title: "Fisher criterion", formula: "J(w) = (wᵀS_B w) / (wᵀS_W w)", note: "A Rayleigh quotient: separation of projected means relative to projected class spread." },
        { title: "Scatter matrices", formula: "S_W = Σ_c Σ_{i∈c}(xᵢ−μ_c)(xᵢ−μ_c)ᵀ,  S_B = Σ_c N_c(μ_c−μ)(μ_c−μ)ᵀ", note: "S_B has rank at most C−1, which caps the number of useful discriminant directions." },
        { title: "Solution", formula: "S_W⁻¹S_B w = λw;  for two classes w ∝ S_W⁻¹(μ₁ − μ₀)", note: "No iterative optimisation: one matrix inverse and an eigen-decomposition." },
        { title: "Generative view", formula: "x | y=c ~ N(μ_c, Σ) with shared Σ", note: "Bayes' rule with a shared covariance cancels the quadratic terms, leaving linear discriminants." },
        { title: "Discriminant function", formula: "δ_c(x) = xᵀΣ⁻¹μ_c − ½μ_cᵀΣ⁻¹μ_c + log π_c", note: "Predict argmax_c δ_c(x). Allowing each class its own Σ_c gives QDA with quadratic boundaries." },
      ],
      pipeline: [{ label: "Class means" }, { label: "S_W, S_B" }, { label: "eig(S_W⁻¹S_B)" }, { label: "Project", note: "≤ C−1 dims" }, { label: "Classify", note: "linear discriminants", accent: "green" }],
      decisionFunction: { text: "ŷ = argmax_c [ xᵀΣ⁻¹μ_c − ½μ_cᵀΣ⁻¹μ_c + log π_c ]", mechanism: "Each class gets a linear score; boundaries between classes are hyperplanes. Equivalently, project with w and threshold midway between the projected means (adjusted for priors)." },
      lossFunction: { text: "maximise J(w) = wᵀS_B w / wᵀS_W w   (equivalently: Gaussian maximum likelihood with shared Σ)", mechanism: "There is no iterative loss minimisation; the optimum comes from a generalised eigenproblem. The scan in the lab simply shows that the eigenvector really is the peak of J(θ)." },
      optimization: [
        { title: "Closed form", formula: "w = S_W⁻¹(μ₁ − μ₀)", note: "O(d³) for the inverse; trivial for modest d." },
        { title: "Shrinkage", formula: "Σ̂ = (1−α)Σ + αI·tr(Σ)/d", note: "When d is large relative to n, S_W is singular or noisy. Ledoit–Wolf shrinkage stabilises it." },
        { title: "SVD solver", formula: "avoid forming S_W explicitly", note: "Numerically safer for high-dimensional data." },
      ],
      assumptions: [
        { name: "Gaussian classes", why: "The optimality of the linear boundary comes from the Gaussian model.", check: "Histograms / QQ plots per class; LDA is fairly robust if roughly unimodal." },
        { name: "Equal covariance across classes", why: "Different covariances make the true boundary quadratic.", check: "Compare class covariance matrices; if very different, use QDA." },
        { name: "Classes differ in means", why: "LDA separates by mean differences only.", check: "If classes share a mean but differ in spread (rings), LDA fails." },
        { name: "S_W invertible", why: "Needed for the solution.", check: "Use shrinkage when n is small relative to d." },
      ],
      hyperparameters: [
        { name: "solver", range: "svd / lsqr / eigen", increasing: "Not applicable", strategy: "svd by default; lsqr or eigen to enable shrinkage." },
        { name: "shrinkage", range: "0 - 1 or 'auto'", increasing: "More regularised covariance, lower variance, more bias.", strategy: "'auto' (Ledoit–Wolf) when d is large." },
        { name: "n_components", range: "1 - C−1", increasing: "More discriminant directions kept.", strategy: "C−1 for classification; 2 for visualisation." },
        { name: "priors", range: "class frequencies", increasing: "Shifts the thresholds toward rarer classes.", strategy: "Set explicitly when training and deployment class balances differ." },
      ],
      metrics: ["Accuracy / balanced accuracy", "Log-loss (it outputs probabilities)", "Fisher ratio of the projection", "Explained between-class variance per component"],
      typicalUses: ["Small-sample classification", "Supervised dimensionality reduction", "Baseline before more flexible models"],
      diagnostics: ["Compare LDA with QDA: a big gain for QDA means unequal covariances.", "Plot the 1-D projections per class: overlap is directly visible.", "Inspect the scalings to see which features drive separation."],
      advantages: ["Closed-form, fast and deterministic.", "Works well with few samples thanks to its strong assumptions.", "Gives both a classifier and a supervised projection.", "Calibrated-ish probabilities under its model."],
      limitations: [{ name: "Linear boundaries only", note: "cannot capture curved boundaries", fix: "QDA, kernel LDA, or a non-linear model." }, { name: "At most C−1 components", note: "two classes give a single direction", fix: "use PCA for unsupervised extra dimensions." }, { name: "Sensitive to outliers", note: "means and covariances are not robust", fix: "robust covariance estimates." }, { name: "Fails when classes share a mean", note: "e.g. concentric rings", fix: "non-linear methods." }],
      alternatives: [{ name: "QDA", when: "Classes have different covariances." }, { name: "Logistic regression", when: "You do not want to assume Gaussian classes." }, { name: "PCA", when: "There are no labels; you want maximum variance instead of separation." }],
      pitfalls: [{ problem: "Using PCA before classification and losing the signal", solution: "PCA keeps variance, not separability. Use LDA, or keep more PCA components." }, { problem: "Singular S_W with many features", solution: "Shrinkage or PCA first." }, { problem: "Rings or XOR patterns", solution: "LDA separates means only; switch to a non-linear model." }],
      quickRef: [{ name: "Fisher ratio", formula: "J = wᵀS_Bw / wᵀS_Ww" }, { name: "Two-class solution", formula: "w ∝ S_W⁻¹(μ₁−μ₀)" }, { name: "Discriminant", formula: "δ_c = xᵀΣ⁻¹μ_c − ½μ_cᵀΣ⁻¹μ_c + log π_c" }, { name: "Max components", formula: "min(C−1, d)" }],
      code: `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(solver="lsqr", shrinkage="auto")   # shrinkage for many features
lda.fit(X_train, y_train)
print(lda.score(X_test, y_test))

# As a supervised projection (at most C-1 dimensions)
Z = LDA(n_components=2).fit_transform(X, y)`,
      whyChain: [
        { q: "Why does LDA beat PCA for classification?", a: "PCA maximises total variance and never looks at labels; the highest-variance direction can be exactly the one along which classes overlap. LDA maximises separation relative to within-class spread." },
        { q: "Why is the boundary linear?", a: "With a shared covariance, the quadratic term xᵀΣ⁻¹x is identical for every class and cancels when comparing scores, leaving a linear function of x." },
        { q: "Why at most C−1 directions?", a: "S_B is built from C mean differences around the overall mean, which sum to zero, so its rank is at most C−1; only that many eigenvalues are non-zero." },
        { q: "How does LDA relate to logistic regression?", a: "Both give linear boundaries. LDA estimates them through a Gaussian generative model (more efficient if the model is right); logistic regression fits the boundary directly (more robust if it is wrong)." },
      ],
      workedExample: {
        setup: "Two classes with μ₀ = (0, 0), μ₁ = (2, 0) and pooled S_W = [[1, 0.8],[0.8, 1]].",
        steps: ["Mean difference d = (2, 0).", "S_W⁻¹ = 1/0.36 · [[1, −0.8],[−0.8, 1]].", "w ∝ S_W⁻¹ d = (2, −1.6)/0.36 ∝ (1, −0.8).", "The direction is tilted away from the x-axis to cancel the shared correlation."],
        result: "w ∝ (1, −0.8): not the naive mean-difference direction (1, 0)",
      },
    },
    mount,
  });
})();
