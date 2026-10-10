(() => {
  const { ui, col, fmt } = Lab;
  const NC = 3;
  const gauss = (x, m, v) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
  const lgauss = (x, m, v) => -((x - m) ** 2) / (2 * v) - 0.5 * Math.log(2 * Math.PI * v);

  function fit(pts, smooth, uniformPrior) {
    const allVar = Math.max(MLU.std(pts.map((p) => p.x)) ** 2, MLU.std(pts.map((p) => p.y)) ** 2);
    const eps = smooth * allVar;
    const cls = [];
    for (let c = 0; c < NC; c++) {
      const m = pts.filter((p) => p.label === c);
      if (!m.length) { cls.push(null); continue; }
      const mx = MLU.mean(m.map((p) => p.x)), my = MLU.mean(m.map((p) => p.y));
      const vx = MLU.mean(m.map((p) => (p.x - mx) ** 2)) + eps, vy = MLU.mean(m.map((p) => (p.y - my) ** 2)) + eps;
      cls.push({ c, n: m.length, prior: m.length / pts.length, mx, my, vx, vy });
    }
    const present = cls.filter(Boolean).length;
    if (uniformPrior) cls.forEach((k) => k && (k.prior = 1 / present));
    return cls;
  }
  function posterior(model, x, y, upto = NC) {
    const lp = model.map((k, c) => (k && c < upto ? Math.log(k.prior) + lgauss(x, k.mx, k.vx) + lgauss(y, k.my, k.vy) : -Infinity));
    const mx = Math.max(...lp);
    const e = lp.map((v) => (isFinite(v) ? Math.exp(v - mx) : 0));
    const s = e.reduce((a, b) => a + b, 0) || 1;
    return { post: e.map((v) => v / s), lp };
  }

  function buildFrames(pts, q, smooth, uni) {
    const present = [...new Set(pts.map((p) => p.label))];
    if (pts.length < 2 || present.length < 2) return [{ phase: "idle", title: "Need points from at least two classes", text: "Load an example dataset or click to add points.", st: { upto: 0 } }];
    const model = fit(pts, smooth, uni);
    const frames = [];
    frames.push({
      phase: "prior", major: true, arch: ["data", "prior"], title: "Count the classes: the prior P(y)",
      text: `Before looking at any feature, the best guess is the class frequency. ${model.map((k, c) => (k ? `P(y=${c}) = ${k.n}/${pts.length} = <b>${k.prior.toFixed(2)}</b>` : "")).filter(Boolean).join(", ")}${uni ? " (uniform priors are on, so these are overridden to be equal)" : ""}.`,
      tex: "P(y=c)=\\frac{N_c}{N}", st: { upto: 0, model },
    });
    model.forEach((k, c) => {
      if (!k) return;
      frames.push({
        phase: "fit", arch: ["data", "lik"], title: `Fit class ${c}: one 1-D Gaussian per feature`,
        text: `For class ${c}: x ~ N(<b>${k.mx.toFixed(2)}</b>, ${Math.sqrt(k.vx).toFixed(2)}²) and y ~ N(<b>${k.my.toFixed(2)}</b>, ${Math.sqrt(k.vy).toFixed(2)}²). Fitting is just a mean and a variance per feature per class, which is why Naive Bayes trains in a single pass.`,
        tex: `\\mu_{${c}j}=\\tfrac{1}{N_{${c}}}\\textstyle\\sum_{i:y_i=${c}} x_{ij},\\quad \\sigma^2_{${c}j}=\\tfrac{1}{N_{${c}}}\\sum_{i:y_i=${c}}(x_{ij}-\\mu_{${c}j})^2`,
        st: { upto: c + 1, model, focus: c },
      });
    });
    frames.push({
      phase: "indep", major: true, arch: ["lik"], title: "The naive step: multiply the per-feature likelihoods",
      text: "Assuming features are independent given the class, the joint density is a product of the 1-D Gaussians. Geometrically each class becomes an <b>axis-aligned</b> ellipse: no tilt is possible. Load the anisotropic dataset to watch that assumption break.",
      tex: "p(x\\mid y=c)=\\prod_{j=1}^{d}\\mathcal N\\!\\left(x_j;\\,\\mu_{cj},\\sigma^2_{cj}\\right)",
      st: { upto: NC, model, ell: true },
    });
    frames.push({
      phase: "post", major: true, arch: ["lik", "prior", "bayes", "out"], title: "Bayes' rule turns likelihoods into posteriors",
      text: "Multiply each class's likelihood by its prior and normalise. Colour shows the winning class, saturation its posterior probability. Boundaries between Gaussians of unequal variance are curved (quadratic).",
      tex: "P(y=c\\mid x)=\\frac{P(y=c)\\prod_j p(x_j\\mid c)}{\\sum_{c'}P(y=c')\\prod_j p(x_j\\mid c')}",
      st: { upto: NC, model, ell: true, field: true },
    });
    const { post, lp } = posterior(model, q.x, q.y);
    const win = post.indexOf(Math.max(...post));
    frames.push({
      phase: "query", major: true, arch: ["bayes", "out"], title: `Query ◆ → class ${win} with probability ${(post[win] * 100).toFixed(1)}%`,
      text: `Working in log space avoids underflow: ${lp.map((v, c) => (isFinite(v) ? `log score(${c}) = <b>${v.toFixed(2)}</b>` : "")).filter(Boolean).join(", ")}. Softmax of these gives the posterior bars. Drag the ◆ to probe anywhere.`,
      tex: "\\hat y=\\arg\\max_c\\Big[\\log P(c)+\\sum_j\\log\\mathcal N(x_j;\\mu_{cj},\\sigma^2_{cj})\\Big]",
      st: { upto: NC, model, ell: true, field: true, query: true },
    });
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "nb", fps: 0.9, classes: NC,
      legend: ui.legend([["var(--accent)", "query ◆"], ["var(--text)", "1σ / 2σ ellipse", "ring"]]),
      tools: ui.btn("replay", "replay", true),
      hint: "curves along the bottom and left edges are each class's per-feature Gaussians · drag the ◆ to query",
      data: { kind: "class", presets: ["blobs3", "aniso", "overlap", "imbalanced", "moons", "circles"], def: "blobs3", n: 120, noise: 0.35 },
      side: [{ title: "model", html: ui.range("sm", "variance smoothing (log₁₀)", -9, 0, 0.5, -9) + ui.check("uni", "uniform class priors", false) + ui.check("marg", "show marginal densities", true) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "data", label: "X, y", sub: "labelled data", x: 44, y: 95, w: 64, kind: "data" },
          { id: "prior", label: "P(y)", sub: "class counts", x: 150, y: 38, kind: "param" },
          { id: "lik", label: "∏ⱼ N(xⱼ|μ,σ²)", sub: "per-feature Gaussians", x: 150, y: 152, kind: "op", info: "One mean and variance per feature per class. The product is the 'naive' independence assumption." },
          { id: "bayes", label: "Bayes' rule", sub: "prior × likelihood", x: 268, y: 95, kind: "op", tex: "P(c|x)\\propto P(c)\\,p(x|c)" },
          { id: "out", label: "argmax", sub: "class", x: 346, y: 165, w: 60, kind: "out" },
        ],
        edges: [{ from: "data", to: "prior" }, { from: "data", to: "lik" }, { from: "prior", to: "bayes" }, { from: "lik", to: "bayes" }, { from: "bayes", to: "out" }],
      },
      phases: { prior: { label: "prior", color: 3 }, fit: { label: "fit Gaussians", color: 0 }, indep: { label: "independence", color: 4 }, post: { label: "posterior", color: 2 }, query: { label: "query", color: 2 } },
      panels: [{ id: "post", title: "query posterior P(y | x)" }, { id: "params", title: "learned parameters" }],
      onFrame: render, onData: () => replay(),
    });
    lab.editable({ onChange: () => replay({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    const q = { x: 0.5, y: 0.5 };
    const smooth = () => Math.pow(10, lab.val("sm"));
    function replay({ at = "start" } = {}) { lab.load(buildFrames(lab.points, q, smooth(), lab.val("uni")), { at, play: at === "start" }); }
    const postBars = Lab.bars(lab.R("panel-post"), { w: 360, h: 130, yd: [0, 1] });
    const ellG = L.mid.append("g"), margG = L.mid.append("g");
    const qG = L.top.append("g").attr("class", "no-add").style("cursor", "move");
    qG.append("rect").attr("x", -8).attr("y", -8).attr("width", 16).attr("height", 16).attr("transform", "rotate(45)").attr("stroke", "var(--bg)").attr("stroke-width", 2);
    qG.call(d3.drag().on("drag", (ev) => { q.x = Math.max(-10, Math.min(10, pl.x.invert(ev.x))); q.y = Math.max(-10, Math.min(10, pl.y.invert(ev.y))); lab.load(buildFrames(lab.points, q, smooth(), lab.val("uni")), { at: "end" }); }));

    function render(f) {
      const st = f.st, model = st.model;
      if (st.field && model) Lab.field(L.bg, pl, (x, y) => { const p = posterior(model, x, y).post; const k = p.indexOf(Math.max(...p)); return k + p[k] * 0.999; }, (v) => Lab.classColor(Math.floor(v), ((v % 1) - 0.34) * 1.5, 0.42), { res: 90 });
      else L.bg.selectAll("*").remove();
      ellG.selectAll("*").remove(); margG.selectAll("*").remove();
      if (model) {
        model.forEach((k, c) => {
          if (!k || c >= st.upto) return;
          const dim = st.focus != null && st.focus !== c;
          if (st.ell || st.focus === c) for (const s of [1, 2]) ellG.append("path").attr("d", Lab.ellipsePath(pl, [k.mx, k.my], [[k.vx, 0], [0, k.vy]], s)).attr("fill", s === 1 ? col(c) : "none").attr("fill-opacity", 0.08).attr("stroke", col(c)).attr("stroke-width", s === 1 ? 2 : 1.2).attr("stroke-dasharray", s === 2 ? "5 4" : null).attr("opacity", dim ? 0.35 : 1);
          if (lab.val("marg")) {
            const H = 70, Wd = 70;
            const PX = Math.max(...model.filter(Boolean).map((m) => gauss(m.mx, m.mx, m.vx)));
            const PY = Math.max(...model.filter(Boolean).map((m) => gauss(m.my, m.my, m.vy)));
            const scx = H / PX, scy = Wd / PY;
            const xs = d3.range(-10, 10.01, 0.1);
            margG.append("path").attr("d", d3.line().x((x) => pl.x(x)).y((x) => pl.y(-10) - Math.min(H, gauss(x, k.mx, k.vx) * scx))(xs))
              .attr("fill", col(c)).attr("fill-opacity", 0.14).attr("stroke", col(c)).attr("stroke-width", 1.8).attr("opacity", dim ? 0.3 : 1);
            margG.append("path").attr("d", d3.line().y((y) => pl.y(y)).x((y) => pl.x(-10) + Math.min(Wd, gauss(y, k.my, k.vy) * scy))(xs))
              .attr("fill", col(c)).attr("fill-opacity", 0.14).attr("stroke", col(c)).attr("stroke-width", 1.8).attr("opacity", dim ? 0.3 : 1);
          }
        });
      }
      lab.drawPoints({ color: (d) => col(d.label), opacity: (d) => (st.focus != null && d.label !== st.focus ? 0.25 : 1) });
      qG.style("display", st.query ? null : "none").attr("transform", `translate(${pl.x(q.x)},${pl.y(q.y)})`);
      if (model && st.query) {
        const { post } = posterior(model, q.x, q.y);
        qG.select("rect").attr("fill", col(post.indexOf(Math.max(...post))));
        postBars.set(post.map((p, c) => ({ label: "class " + c, value: p, color: col(c), text: (p * 100).toFixed(1) + "%" })));
      } else postBars.set([0, 1, 2].map((c) => ({ label: "class " + c, value: 0, color: col(c), text: "–" })));
      lab.R("panel-params").innerHTML = model ? `<table class="mini-table"><tr><th>class</th><th>prior</th><th>μₓ</th><th>σₓ</th><th>μᵧ</th><th>σᵧ</th></tr>${model.map((k, c) => (k ? `<tr class="${st.focus === c ? "on" : ""}"><td><span class="swatch" style="background:${col(c)}"></span> ${c}</td><td>${c < st.upto || f.phase === "prior" ? k.prior.toFixed(2) : "·"}</td><td>${c < st.upto ? fmt(k.mx, 2) : "·"}</td><td>${c < st.upto ? fmt(Math.sqrt(k.vx), 2) : "·"}</td><td>${c < st.upto ? fmt(k.my, 2) : "·"}</td><td>${c < st.upto ? fmt(Math.sqrt(k.vy), 2) : "·"}</td></tr>` : "")).join("")}</table><div class="lab-card-note">Only 2 × d + 1 numbers per class. That tiny parameter count is why Naive Bayes needs so little data.</div>` : "";
      let acc = null;
      if (model) acc = lab.points.filter((p) => { const po = posterior(model, p.x, p.y).post; return po.indexOf(Math.max(...po)) === p.label; }).length / lab.points.length;
      lab.readout(`<div class="kv"><span>points</span><b>${lab.points.length}</b><span>train accuracy</span><b>${acc == null ? "–" : (acc * 100).toFixed(1) + "%"}</b><span>parameters</span><b>${model ? model.filter(Boolean).length * 5 : "–"}</b></div>`);
    }
    lab.watch(["sm", "uni"], () => replay({ at: "end" }), { format: { sm: (v) => "1e" + v } });
    lab.watch(["marg"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.loadPreset("blobs3");
    return lab.destroy;
  }

  MLApp.register({
    id: "naive-bayes",
    name: "Gaussian Naive Bayes",
    category: "Supervised - Classification",
    tagline: "per-class Gaussians + Bayes' rule",
    description: "Fit one Gaussian per feature per class, multiply them under the naive independence assumption, and apply Bayes' rule. Watch priors, marginal densities, axis-aligned ellipses and the posterior field build up, then drag a query to read its posterior.",
    info: {
      type: "Supervised - Classification. Generative probabilistic model (Gaussian Naive Bayes).",
      scenario: "A fast, simple baseline classifier, especially effective with high-dimensional or sparse features (text), or whenever a cheap probabilistic model is preferable to an iteratively-trained one.",
      inputs: "Feature vector x and class label y; each feature is assumed conditionally independent and normally distributed given the class.",
      intuition: {
        definition: "Apply Bayes' rule, then make one deliberately wrong simplification: pretend every feature is <b>independent given the class</b>. That turns an intractable joint distribution into a product of cheap one-dimensional ones, and the resulting classifier is fast, stable, and surprisingly hard to beat.",
        steps: [
          "For each class, learn how each feature is distributed on its own.",
          "For a new point, score every class by prior times likelihood.",
          "Multiply the per-feature likelihoods together (the naive step).",
          "Work in log space and take the highest-scoring class.",
        ],
        applications: [
          "Spam and phishing filters, the classic use case",
          "Topic and sentiment labelling of documents",
          "Language identification from character frequencies",
          "Real-time triage where latency matters more than the last point of accuracy",
          "A strong first baseline on any high-dimensional sparse problem",
        ],
      },
      math: [
        { title: "Bayes' rule", formula: "P(c | x) = P(x | c)·P(c) / P(x)", note: "P(x) is the same for every class, so it can be dropped when comparing them." },
        { title: "The naive assumption", formula: "P(x | c) = Π_j P(xⱼ | c)", note: "Assumes features are conditionally independent given the class. Almost never true, yet the argmax often survives the error." },
        { title: "Gaussian likelihood", formula: "P(xⱼ | c) = (1 / √(2πσ²_{c,j})) · exp( −(xⱼ − μ_{c,j})² / (2σ²_{c,j}) )", note: "One mean and one variance per feature per class. For counts use Multinomial, for binary use Bernoulli." },
        { title: "Log-space scoring", formula: "score(c) = log P(c) + Σ_j log P(xⱼ | c)", note: "Multiplying hundreds of small probabilities underflows to zero in floating point. Summing logs is numerically safe and faster." },
        { title: "Prediction", formula: "ŷ = argmax_c score(c)", note: "Convert scores back to posteriors with a softmax if you need calibrated probabilities." },
      ],
      pipeline: [
        { label: "Training data", note: "x with labels" },
        { label: "Per-class stats", note: "μ, σ² per feature" },
        { label: "Priors P(c)", note: "class frequencies" },
        { label: "Log scores", note: "log P(c) + Σ log P(xⱼ|c)" },
        { label: "argmax", note: "predicted class", accent: "green" },
      ],
      decisionFunction: {
        text: "ŷ = argmax_c [ log P(c) + Σ_features log N(xᵢ; μ_{c,i}, σ_{c,i}) ]",
        mechanism: "Bayes' rule under a 'naive' feature-independence assumption: each feature's likelihood under a class's fitted Gaussian is combined (summed in log-space) with that class's prior, and the highest-scoring class wins.",
        plot: { fn: (x) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI), domain: [-4, 4], color: "var(--accent)", caption: "the per-feature Gaussian each class's likelihood is built from (standard normal shown)" },
      },
      lossFunction: {
        text: "No iterative loss - closed-form maximum-likelihood estimation.",
        mechanism: "Each class's mean, variance, and prior are computed directly from the training data in a single pass; there's no optimization loop to converge.",
      },
      optimization: [
        { title: "Closed-form MLE", formula: "μ_{c,j} = mean of xⱼ over class c,  σ²_{c,j} = variance of xⱼ over class c,  P(c) = mₑ / m", note: "One pass over the data computes everything. There is no learning rate, no convergence criterion, and no randomness." },
        { title: "Cost", formula: "train: O(m·p)   predict: O(K·p)", note: "Linear in the data and independent of it at prediction time. This is about as cheap as supervised learning gets." },
        { title: "Smoothing", formula: "σ² := σ² + ε   (Gaussian)    P(xⱼ|c) = (count + α) / (total + α·V)   (Laplace, discrete)", note: "Prevents a single unseen feature value from multiplying the whole class score by zero." },
      ],
      output: "A predicted class label, plus a posterior probability / log-score for every class.",
      assumptions: [
        { name: "Conditional independence", why: "The core simplification. Correlated features get counted repeatedly, which pushes posteriors toward 0 or 1.", check: "Inspect the feature correlation matrix. Heavy correlation means the probabilities are overconfident even if the labels stay right." },
        { name: "Correct likelihood family", why: "Gaussian NB assumes each feature is bell-shaped within a class; skewed features are badly modelled.", check: "Histogram each feature per class. Log-transform skewed ones, or switch to Multinomial or Bernoulli." },
        { name: "Non-zero variance", why: "A feature that is constant within a class gives σ² = 0 and the density blows up.", check: "Variance smoothing handles it automatically; otherwise drop the constant feature." },
        { name: "Representative priors", why: "Class priors come straight from training frequencies, which may not match deployment.", check: "Override the priors explicitly if the real-world base rate differs." },
        { name: "No unseen categories", why: "In the discrete variants an unseen value has zero likelihood and annihilates the class score.", check: "Apply Laplace smoothing with α > 0." },
      ],
      regularization: [
        { name: "Variance smoothing", formula: "σ² + ε,  ε ≈ 1e-9 · max variance", note: "The Gaussian equivalent of Laplace smoothing. Stops near-constant features from producing infinite densities." },
        { name: "Laplace / additive", formula: "(count + α) / (total + α·V)", note: "For Multinomial and Bernoulli NB. α = 1 is Laplace, α < 1 is Lidstone. Guarantees every value has non-zero probability." },
        { name: "Complement NB", formula: "weights from the complement of each class", note: "A variant that corrects the bias Multinomial NB shows on imbalanced text data." },
      ],
      hyperparameters: [
        { name: "var_smoothing (Gaussian)", range: "1e-12 - 1e-6", increasing: "Widens every fitted Gaussian, smoothing the boundary and reducing overconfidence.", strategy: "Log-scale search. The default rarely needs changing unless features are near-constant." },
        { name: "alpha (Multinomial)", range: "0.01 - 10", increasing: "More smoothing toward the uniform distribution, higher bias.", strategy: "Log-scale CV. α = 1 is a safe default for text." },
        { name: "likelihood family", range: "Gaussian / Multinomial / Bernoulli", increasing: "Not applicable", strategy: "Gaussian for continuous features, Multinomial for counts or TF-IDF, Bernoulli for binary presence flags." },
        { name: "fit_prior", range: "true / false", increasing: "Not applicable", strategy: "Set false to force uniform priors when training frequencies do not reflect deployment." },
        { name: "class_prior", range: "vector summing to 1", increasing: "Not applicable", strategy: "Supply explicitly when you know the true base rate." },
      ],
      metrics: ["Accuracy", "Precision / Recall / F1", "Log-loss", "ROC-AUC (ranking quality, which survives poor calibration)"],
      typicalUses: ["Spam filtering", "Document/text classification", "Medical screening", "Real-time classification where speed matters more than squeezing out max accuracy"],
      diagnostics: [
        "Check the calibration curve. Naive Bayes usually predicts probabilities near 0 or 1 even when it is wrong, because correlated features multiply the same evidence repeatedly.",
        "Compare accuracy against ROC-AUC. High AUC with poor log-loss is the signature of good ranking but bad calibration.",
        "Plot per-feature histograms by class to confirm the Gaussian assumption before trusting Gaussian NB.",
        "If one class is never predicted, inspect the priors and consider setting them uniform.",
      ],
      advantages: [
        "Trains in a single pass, with no iteration, no learning rate, and no random seed.",
        "Works well with far more features than samples, which is exactly the text-classification regime.",
        "Handles many classes at once with no extra cost or one-vs-rest machinery.",
        "Needs very little training data to produce a usable model.",
        "Predictions are cheap and constant-time, which suits real-time filtering.",
      ],
      limitations: [
        { name: "Independence assumption is false", note: "correlated features double-count evidence", fix: "de-correlate with PCA, drop redundant features, or accept it and use only the ranking." },
        { name: "Badly calibrated probabilities", note: "posteriors are pushed to the extremes", fix: "Platt scaling or isotonic regression if you need real probabilities." },
        { name: "Zero-frequency problem", note: "an unseen value can zero out an entire class", fix: "Laplace or variance smoothing." },
        { name: "Gaussian assumption often wrong", note: "skewed or multi-modal features are poorly captured", fix: "transform features, discretize, or use kernel density NB." },
        { name: "Cannot learn interactions", note: "the model is additive in log space by construction", fix: "add explicit interaction features, or use a tree ensemble." },
      ],
      alternatives: [
        { name: "Logistic regression", when: "You want calibrated probabilities and features are correlated. It is the discriminative counterpart to NB." },
        { name: "Linear SVM", when: "Text classification where you want maximum accuracy and do not need probabilities." },
        { name: "Gradient boosting", when: "Tabular data with genuine feature interactions." },
        { name: "Complement NB", when: "Multinomial NB is underperforming on imbalanced text." },
      ],
      pitfalls: [
        { problem: "Probabilities are all 0.999 or 0.001", solution: "Expected behaviour with correlated features. Calibrate, or use only the ranking." },
        { problem: "A class score becomes zero or NaN", solution: "Zero-frequency or zero-variance. Turn on smoothing." },
        { problem: "Numeric underflow on long documents", solution: "Score in log space rather than multiplying raw probabilities." },
        { problem: "Gaussian NB performs poorly on counts", solution: "Wrong likelihood family. Use Multinomial NB." },
        { problem: "Accuracy drops after adding similar features", solution: "Duplicated evidence under the independence assumption. Remove redundant features." },
      ],
      quickRef: [
        { name: "Bayes' rule", formula: "P(c|x) ∝ P(c)·P(x|c)" },
        { name: "Naive factorization", formula: "P(x|c) = Π_j P(xⱼ|c)" },
        { name: "Gaussian likelihood", formula: "N(xⱼ; μ_{c,j}, σ²_{c,j})" },
        { name: "Log score", formula: "log P(c) + Σ_j log P(xⱼ|c)" },
        { name: "Prediction", formula: "ŷ = argmax_c score(c)" },
        { name: "Laplace smoothing", formula: "(count + α)/(total + α·V)" },
        { name: "Train cost", formula: "O(m·p), single pass" },
      ],
      code: `from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

# Continuous features
gnb = GaussianNB(var_smoothing=1e-9).fit(X_train, y_train)

# Text: counts or TF-IDF want the multinomial variant
text_clf = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    MultinomialNB(alpha=1.0),      # alpha=1 is Laplace smoothing
).fit(docs_train, y_train)

# NB probabilities are poorly calibrated; wrap it if you need real ones.
calibrated = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)
calibrated.fit(X_train, y_train)

print(classification_report(y_test, text_clf.predict(docs_test)))`,
      whyChain: [
        { q: "Why is it called 'naive'?", a: "Because of the conditional independence assumption: it pretends that, within a class, knowing one feature tells you nothing about another. In real data that is essentially always false." },
        { q: "If the assumption is false, why does it still work?", a: "Classification only needs the argmax to be right, not the probabilities. The independence error tends to inflate all class scores in a similar direction, so the ranking between classes often survives even when the magnitudes are badly wrong." },
        { q: "So what does break?", a: "Calibration. Correlated features count the same evidence multiple times, driving posteriors to 0 or 1. Use NB for the label, not for the confidence, unless you calibrate it." },
        { q: "Why score in log space?", a: "Multiplying hundreds of probabilities each below 1 underflows to exactly zero in floating point. Summing logs is numerically stable and turns multiplications into cheaper additions." },
        { q: "What is the zero-frequency problem?", a: "If a feature value never appeared with a class in training, its estimated likelihood is 0, and multiplying by zero wipes out that class no matter how much other evidence supports it. Laplace smoothing adds a pseudo-count so nothing is ever impossible." },
        { q: "Generative or discriminative, and why does it matter?", a: "Generative: it models P(x|c) and can in principle sample new data. Logistic regression models P(c|x) directly. Generative models need less data to converge; discriminative ones usually win asymptotically." },
        { q: "How do you pick between the NB variants?", a: "By what the features are. Gaussian for continuous measurements, Multinomial for word counts or TF-IDF, Bernoulli for binary presence indicators." },
      ],
      parameters: [
        { name: "variance smoothing", effect: "A small ε added to every variance estimate to avoid division-by-zero when a class's feature has near-zero spread." },
        { name: "likelihood family", effect: "Gaussian here; Multinomial/Bernoulli variants exist for count/binary features (e.g. text)." },
      ],
      workedExample: {
        setup: "Class A heights {170,172,168}, Class B heights {180,182,178}, equal priors. Classify x=173.",
        steps: [
          "Both classes: mean=170 (A) / 180 (B); variance = ((0)²+(2)²+(−2)²)/3 = 8/3 ≈ 2.667 for each.",
          "Gaussian pdf coefficient: 1/√(2π·2.667) = 1/4.094 = 0.2443.",
          "P(x|A): exponent = −(173−170)²/(2·2.667) = −9/5.334 = −1.687 → e⁻¹·⁶⁸⁷ ≈ 0.185 → pdf_A = 0.2443×0.185 ≈ 0.0452.",
          "P(x|B): exponent = −(173−180)²/(2·2.667) = −49/5.334 = −9.187 → e⁻⁹·¹⁸⁷ ≈ 1.02×10⁻⁴ → pdf_B ≈ 0.2443×1.02×10⁻⁴ ≈ 2.50×10⁻⁵.",
          "Posterior ∝ prior × likelihood (prior 0.5 cancels in the ratio): unnormalized A=0.0452×0.5=0.0226, B=2.50×10⁻⁵×0.5=1.25×10⁻⁵.",
          "Normalize: P(A|x) = 0.0226/(0.0226+0.0000125) ≈ 0.9994, P(B|x) ≈ 0.0006.",
        ],
        result: "Predict class A (P≈99.9%) - x=173 is far closer to A's mean (170) than B's (180) relative to their shared spread",
      },
    },
    mount,
  });
})();
