(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];
  const N_CLASSES = 3;

  function fitGaussianNB(points) {
    const classes = [];
    for (let c = 0; c < N_CLASSES; c++) {
      const pts = points.filter((p) => p.label === c);
      if (!pts.length) { classes.push(null); continue; }
      const mx = MLU.mean(pts.map((p) => p.x)), my = MLU.mean(pts.map((p) => p.y));
      const sx = MLU.std(pts.map((p) => p.x)), sy = MLU.std(pts.map((p) => p.y));
      classes.push({ mx, my, sx: Math.max(sx, 0.4), sy: Math.max(sy, 0.4), prior: pts.length / points.length });
    }
    return classes;
  }
  function logGauss(v, m, s) { return -0.5 * Math.log(2 * Math.PI * s * s) - ((v - m) ** 2) / (2 * s * s); }
  function predict(model, pt) {
    let best = -1, bestScore = -Infinity, scores = [];
    for (let c = 0; c < N_CLASSES; c++) {
      const m = model[c];
      if (!m) { scores.push(-Infinity); continue; }
      const score = Math.log(m.prior) + logGauss(pt.x, m.mx, m.sx) + logGauss(pt.y, m.my, m.sy);
      scores.push(score);
      if (score > bestScore) { bestScore = score; best = c; }
    }
    return { label: best, scores };
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">${[0, 1, 2].map((i) => `<span class="legend-item"><span class="swatch" style="background:${MLU.palette[i]}"></span>class ${i}</span>`).join("")}</div>
            <div class="btn-row"><button id="nb-regen">regenerate data</button><button id="nb-clear">clear</button></div>
          </div>
          <svg id="nb-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">pick a class, click to add points · dashed ellipses = 1&sigma; of each class's fitted Gaussian</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>add points as</h3>
            <div class="btn-row" id="nb-class-buttons">
              ${[0, 1, 2].map((i) => `<button data-c="${i}" style="border-color:${MLU.palette[i]}">${i}</button>`).join("")}
            </div>
          </div>
          <div class="control-card">
            <h3>model</h3>
            <div class="readout" id="nb-readout">–</div>
            <div class="note">Gaussian Naive Bayes: each feature modeled with an independent per-class normal distribution; prediction = argmax of log-prior + log-likelihood.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#nb-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeBlobs({ n: 60, clusters: 3 }).map((p) => ({ x: p.x, y: p.y, label: p.label }));
    let currentClass = 0;
    const bgG = svg.append("g");
    const ellG = svg.append("g");
    const ptsG = svg.append("g");

    function render() {
      const model = fitGaussianNB(points);
      const cell = 14, cells = [];
      for (let px = PAD; px < W - PAD; px += cell)
        for (let py = PAD; py < H - PAD; py += cell) {
          const xv = x.invert(px + cell / 2), yv = y.invert(py + cell / 2);
          const { label } = predict(model, { x: xv, y: yv });
          cells.push({ px, py, label });
        }
      const rects = bgG.selectAll("rect").data(cells);
      rects.enter().append("rect").attr("width", cell).attr("height", cell)
        .merge(rects).attr("x", (d) => d.px).attr("y", (d) => d.py)
        .attr("fill", (d) => (d.label >= 0 ? MLU.palette[d.label] : "transparent")).attr("opacity", points.length ? 0.16 : 0);
      rects.exit().remove();

      const ellData = model.filter(Boolean).map((m, i) => m && { ...m, c: model.indexOf(m) });
      const ell = ellG.selectAll("ellipse").data(model.map((m, i) => (m ? { ...m, c: i } : null)).filter(Boolean));
      ell.enter().append("ellipse").attr("fill", "none").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3")
        .merge(ell)
        .attr("cx", (d) => x(d.mx)).attr("cy", (d) => y(d.my))
        .attr("rx", (d) => Math.abs(x(d.mx + d.sx) - x(d.mx))).attr("ry", (d) => Math.abs(y(d.my - d.sy) - y(d.my)))
        .attr("stroke", (d) => MLU.palette[d.c]);
      ell.exit().remove();

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle")
        .attr("r", 5).attr("stroke", "var(--bg)").attr("stroke-width", 1).style("cursor", "grab")
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("fill", (d) => MLU.palette[d.label])
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); render(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          render();
        }));
      sel.exit().remove();

      document.getElementById("nb-readout").innerHTML = model.map((m, i) => m
        ? `class ${i}: &mu;=(${m.mx.toFixed(1)}, ${m.my.toFixed(1)}) &sigma;=(${m.sx.toFixed(1)}, ${m.sy.toFixed(1)}) prior=${m.prior.toFixed(2)}`
        : `class ${i}: no data`).join("<br>");
    }

    svg.on("click", (event) => {
      if (event.target.tagName === "circle") return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py), label: currentClass });
      render();
    });
    document.getElementById("nb-class-buttons").addEventListener("click", (e) => {
      const btn = e.target.closest("button"); if (!btn) return;
      currentClass = +btn.dataset.c;
      document.querySelectorAll("#nb-class-buttons button").forEach((b) => b.classList.toggle("primary", +b.dataset.c === currentClass));
    });
    document.querySelector("#nb-class-buttons button").classList.add("primary");
    document.getElementById("nb-regen").addEventListener("click", () => {
      points = MLU.makeBlobs({ n: 60, clusters: 3 }).map((p) => ({ x: p.x, y: p.y, label: p.label }));
      render();
    });
    document.getElementById("nb-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "naive-bayes",
    name: "Gaussian Naive Bayes",
    category: "Supervised - Classification",
    tagline: "independent per-class gaussians",
    description: "Fits an independent 1D normal distribution per feature per class, then classifies by the highest log-prior + log-likelihood. Ellipses show each class's fitted 1σ Gaussian.",
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
