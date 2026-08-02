(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];
  const N_CLASSES = 3;

  function gini(labels) {
    if (!labels.length) return 0;
    const counts = {};
    for (const l of labels) counts[l] = (counts[l] || 0) + 1;
    let s = 0;
    for (const c in counts) s += (counts[c] / labels.length) ** 2;
    return 1 - s;
  }
  function majority(labels) {
    const counts = {};
    for (const l of labels) counts[l] = (counts[l] || 0) + 1;
    return +Object.keys(counts).reduce((a, b) => (counts[a] > counts[b] ? a : b));
  }
  function buildTree(pts, depth, maxDepth) {
    const labels = pts.map((p) => p.label);
    if (depth >= maxDepth || pts.length < 4 || gini(labels) === 0) return { leaf: true, label: majority(labels) };
    let best = null;
    for (const feat of ["x", "y"]) {
      const vals = [...new Set(pts.map((p) => p[feat]))].sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const thr = (vals[i] + vals[i + 1]) / 2;
        const left = pts.filter((p) => p[feat] <= thr), right = pts.filter((p) => p[feat] > thr);
        if (!left.length || !right.length) continue;
        const g = (left.length * gini(left.map((p) => p.label)) + right.length * gini(right.map((p) => p.label))) / pts.length;
        if (!best || g < best.g) best = { g, feat, thr, left, right };
      }
    }
    if (!best) return { leaf: true, label: majority(labels) };
    return {
      leaf: false, feat: best.feat, thr: best.thr,
      left: buildTree(best.left, depth + 1, maxDepth),
      right: buildTree(best.right, depth + 1, maxDepth),
    };
  }
  function predictTree(node, p) {
    if (node.leaf) return node.label;
    return predictTree(p[node.feat] <= node.thr ? node.left : node.right, p);
  }
  function bootstrapSample(pts) {
    return Array.from({ length: pts.length }, () => pts[MLU.randInt(pts.length)]);
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">${[0, 1, 2].map((i) => `<span class="legend-item"><span class="swatch" style="background:${MLU.palette[i]}"></span>class ${i}</span>`).join("")}</div>
            <div class="btn-row"><button id="dt-regen">regenerate data</button><button id="dt-clear">clear</button></div>
          </div>
          <svg id="dt-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">pick a class, click to add points · boundary is rebuilt after every change</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>add points as</h3>
            <div class="btn-row" id="dt-class-buttons">
              ${[0, 1, 2].map((i) => `<button data-c="${i}" style="border-color:${MLU.palette[i]}">${i}</button>`).join("")}
            </div>
          </div>
          <div class="control-card">
            <h3>model</h3>
            <div class="field"><label>mode</label>
              <select id="dt-mode"><option value="tree">single decision tree</option><option value="forest">random forest (bagging)</option></select>
            </div>
            <div class="field"><label>max depth <span class="val" id="dt-depth-val">4</span></label>
              <input type="range" id="dt-depth" min="1" max="10" step="1" value="4" /></div>
            <div class="field" id="dt-nest-field" style="display:none"><label>trees (n_estimators) <span class="val" id="dt-nest-val">15</span></label>
              <input type="range" id="dt-nest" min="3" max="40" step="1" value="15" /></div>
          </div>
          <div class="control-card">
            <h3>fit</h3>
            <div class="readout" id="dt-readout">–</div>
            <div class="note" id="dt-note">Recursive Gini-impurity splitting on axis-aligned thresholds, as in <code>mla/ensemble/random_forest.py</code>'s base tree.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#dt-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeBlobs({ n: 70, clusters: 3 }).map((p) => ({ x: p.x, y: p.y, label: p.label }));
    let currentClass = 0;
    const bgG = svg.append("g");
    const ptsG = svg.append("g");

    function depth() { return +document.getElementById("dt-depth").value; }
    function nest() { return +document.getElementById("dt-nest").value; }
    function mode() { return document.getElementById("dt-mode").value; }

    function render() {
      document.getElementById("dt-depth-val").textContent = depth();
      document.getElementById("dt-nest-val").textContent = nest();
      document.getElementById("dt-nest-field").style.display = mode() === "forest" ? "" : "none";
      document.getElementById("dt-note").innerHTML = mode() === "forest"
        ? "Bagging: each tree trains on a bootstrap resample of the points; the boundary shown is the ensemble's majority vote - same idea as <code>mla/ensemble/random_forest.py</code>."
        : "Recursive Gini-impurity splitting on axis-aligned thresholds, as in <code>mla/ensemble/random_forest.py</code>'s base tree.";

      let trees = [];
      if (points.length >= 4) {
        if (mode() === "forest") {
          for (let i = 0; i < nest(); i++) trees.push(buildTree(bootstrapSample(points), 0, depth()));
        } else {
          trees = [buildTree(points, 0, depth())];
        }
      }
      function predict(p) {
        if (!trees.length) return -1;
        const votes = new Array(N_CLASSES).fill(0);
        for (const t of trees) votes[predictTree(t, p)]++;
        let best = 0; for (let i = 1; i < N_CLASSES; i++) if (votes[i] > votes[best]) best = i;
        return best;
      }

      const cell = 12, cells = [];
      for (let px = PAD; px < W - PAD; px += cell)
        for (let py = PAD; py < H - PAD; py += cell) {
          const xv = x.invert(px + cell / 2), yv = y.invert(py + cell / 2);
          cells.push({ px, py, label: predict({ x: xv, y: yv }) });
        }
      const rects = bgG.selectAll("rect").data(cells);
      rects.enter().append("rect").attr("width", cell).attr("height", cell)
        .merge(rects).attr("x", (d) => d.px).attr("y", (d) => d.py)
        .attr("fill", (d) => (d.label >= 0 ? MLU.palette[d.label] : "transparent")).attr("opacity", 0.16);
      rects.exit().remove();

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle").attr("r", 5).attr("stroke", "var(--bg)").attr("stroke-width", 1).style("cursor", "grab")
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("fill", (d) => MLU.palette[d.label])
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); render(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          render();
        }));
      sel.exit().remove();

      let correct = 0;
      for (const p of points) if (predict(p) === p.label) correct++;
      document.getElementById("dt-readout").innerHTML =
        `points: <b>${points.length}</b><br>trees: <b>${trees.length}</b><br>train accuracy: <b class="num">${points.length ? ((correct / points.length) * 100).toFixed(1) : "–"}%</b>`;
    }

    svg.on("click", (event) => {
      if (event.target.tagName === "circle") return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py), label: currentClass });
      render();
    });
    document.getElementById("dt-class-buttons").addEventListener("click", (e) => {
      const btn = e.target.closest("button"); if (!btn) return;
      currentClass = +btn.dataset.c;
      document.querySelectorAll("#dt-class-buttons button").forEach((b) => b.classList.toggle("primary", +b.dataset.c === currentClass));
    });
    document.querySelector("#dt-class-buttons button").classList.add("primary");
    ["dt-depth", "dt-nest", "dt-mode"].forEach((id) => document.getElementById(id).addEventListener("input", render));
    document.getElementById("dt-regen").addEventListener("click", () => {
      points = MLU.makeBlobs({ n: 70, clusters: 3 }).map((p) => ({ x: p.x, y: p.y, label: p.label }));
      render();
    });
    document.getElementById("dt-clear").addEventListener("click", () => { points = []; render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "decision-tree",
    name: "Decision Tree & Random Forest",
    category: "Supervised - Trees & Ensembles",
    tagline: "Gini splits, optional bagging",
    description: "A CART-style classifier that recursively splits on the axis-aligned threshold that most reduces Gini impurity. Switch to random forest mode to see bagged trees smooth out the blocky boundary.",
    sourceFile: "mla/ensemble/random_forest.py",
    info: {
      type: "Supervised - Classification/Regression. Non-parametric, tree-based (single tree); bagged ensemble in random-forest mode.",
      scenario: "You need an interpretable, non-linear model that handles mixed feature types without scaling (single tree); switch to random forest when you want more accuracy/robustness and can trade away some interpretability.",
      inputs: "Feature vectors x and class labels y.",
      intuition: {
        definition: "Ask a sequence of yes/no questions about single features, each chosen to make the resulting groups as <b>pure</b> as possible, until every leaf is dominated by one class. A random forest builds hundreds of such trees on resampled data and averages away their individual mistakes.",
        steps: [
          "At each node, try every feature and threshold.",
          "Keep the split that most reduces impurity in the children.",
          "Recurse until the node is pure or a stopping rule fires.",
          "A forest repeats this on bootstrap samples and votes.",
        ],
        applications: [
          "Credit approval rules that a regulator can actually read",
          "Clinical decision support and triage protocols",
          "Churn driver analysis via feature importance",
          "Tabular benchmarks where forests are still the default baseline",
          "Segmenting customers into interpretable buckets",
        ],
      },
      math: [
        { title: "Gini impurity", formula: "Gini(S) = 1 − Σ_c p_c²", note: "The probability that two items drawn at random from the node have different labels. Zero when the node is pure, 0.5 at a two-class even split." },
        { title: "Entropy", formula: "H(S) = −Σ_c p_c·log₂ p_c", note: "The alternative criterion. Peaks at 1 bit for a two-class even split. In practice it and Gini pick nearly the same trees; Gini is cheaper because it avoids the logarithm." },
        { title: "Split quality", formula: "ΔI = I(parent) − [ (n_L/n)·I(left) + (n_R/n)·I(right) ]", note: "Information gain: the weighted impurity drop. The greedy algorithm takes the split with the largest ΔI at every node." },
        { title: "Regression criterion", formula: "MSE(S) = (1/n) Σᵢ (yᵢ − ȳ_S)²", note: "For regression trees the leaf predicts the mean and impurity is variance. The same weighted-reduction rule applies." },
        { title: "Bagging (forest)", formula: "ŷ(x) = majority { treeₜ(x) : t = 1…T },  each tree on a bootstrap sample", note: "Averaging T decorrelated trees cuts variance by roughly a factor of T without raising bias." },
        { title: "Feature subsampling", formula: "at each split, consider only √p random features", note: "The critical second ingredient. Without it every tree picks the same dominant feature first and the trees stay correlated, so averaging gains little." },
      ],
      pipeline: [
        { label: "Node data", note: "all samples here" },
        { label: "Try splits", note: "every feature × threshold" },
        { label: "Max ΔI", note: "best impurity drop" },
        { label: "Recurse", note: "left and right child" },
        { label: "Leaf", note: "majority label", accent: "green" },
      ],
      decisionFunction: {
        text: "ŷ(x) = label at the leaf reached by recursively testing 'feature_j ≤ threshold?' (forest: majority vote across trees)",
        mechanism: "Each internal node routes a point left or right based on one feature/threshold; random forest trains many such trees on bootstrap-resampled data and votes.",
      },
      lossFunction: {
        text: "Gini(S) = 1 − Σ_c p_c²  (impurity), split chosen to minimize the weighted impurity of the two children",
        mechanism: "A greedy, recursive minimization - every split is locally optimal (not globally), and recursion stops at a depth/size limit or once a node is pure.",
        plot: { fn: (p) => 1 - (p * p + (1 - p) * (1 - p)), domain: [0, 1], yDomain: [0, 0.55], color: "var(--accent)", fn2: (p) => (-p * Math.log2(Math.max(p, 1e-9)) - (1 - p) * Math.log2(Math.max(1 - p, 1e-9))) / 2, color2: "var(--text-faint)", caption: "impurity of a 2-class node vs its class balance p - Gini (solid) and entropy/2 (dashed) both peak at a 50/50 split and hit 0 when pure" },
      },
      optimization: [
        { title: "Greedy, not optimal", formula: "locally best split at each node", note: "Finding the globally optimal tree is NP-complete, so every practical algorithm is greedy. A split that looks poor now but enables a great pair of children later will never be found." },
        { title: "Cost", formula: "train: O(p · m log m)   predict: O(depth)", note: "Sorting each feature once per node dominates training. Prediction is just a walk down the tree, which is why trees serve so fast." },
        { title: "Pre-pruning", formula: "stop when depth ≥ D, or n < min_samples, or ΔI < ε", note: "Cheap and the usual choice, but short-sighted: it can stop just before a valuable split." },
        { title: "Post-pruning", formula: "R_α(T) = R(T) + α·|leaves(T)|", note: "Cost-complexity pruning. Grow the tree fully, then collapse the subtrees whose accuracy gain does not justify their leaf count. α is chosen by cross-validation." },
        { title: "Out-of-bag estimate", formula: "each bootstrap leaves out ≈ 37% of rows", note: "Those held-out rows give a free validation score for a forest, with no separate split required." },
      ],
      output: "A predicted class label (single tree: one leaf; forest: majority vote), optionally with class probabilities.",
      assumptions: [
        { name: "Axis-aligned splits suffice", why: "Every boundary is a step function perpendicular to a feature axis. A diagonal boundary needs a staircase of many splits.", check: "If a linear model beats a deep tree, the true boundary is probably oblique." },
        { name: "No scaling needed", why: "Splits are based on order, not distance, so monotone transforms of a feature change nothing.", check: "Genuinely nothing to do here, unlike SVM or KNN." },
        { name: "Enough samples per leaf", why: "Leaves built from two or three points memorise noise.", check: "Set min_samples_leaf and inspect the leaf-size distribution." },
        { name: "Features are not wildly high-cardinality", why: "Impurity criteria are biased toward features with many distinct values, which offer more split points.", check: "Prefer permutation importance over the built-in Gini importance when cardinality varies." },
        { name: "Reasonably stationary data", why: "Trees extrapolate as a flat constant beyond the training range.", check: "Never use a tree for trend extrapolation; use a linear model or explicitly detrend." },
      ],
      regularization: [
        { name: "max_depth", formula: "depth ≤ D", note: "The bluntest and most effective control. Depth 1 is a stump; each extra level doubles the potential leaf count." },
        { name: "min_samples_leaf", formula: "n_leaf ≥ k", note: "Guarantees every prediction is backed by k observations. Often more robust than tuning depth." },
        { name: "ccp_alpha", formula: "R(T) + α·|leaves|", note: "Post-pruning penalty. Grows the full tree then trims it back, which is principled but slower." },
        { name: "Bagging", formula: "average of T bootstrap trees", note: "Does not restrict any single tree; reduces variance by averaging instead." },
        { name: "max_features", formula: "√p per split (classification)", note: "Decorrelates the trees in a forest. The single most important forest hyperparameter after tree count." },
      ],
      hyperparameters: [
        { name: "max_depth", range: "1 - 30 (None)", increasing: "More splits, lower bias, sharply higher variance. Unlimited depth on clean data reaches 100% training accuracy.", strategy: "For a single tree keep it small enough to read, roughly 3 to 6. For a forest leave it unlimited and control variance with tree count." },
        { name: "min_samples_leaf", range: "1 - 50", increasing: "Smoother, more conservative boundaries and fewer leaves.", strategy: "Raise it when leaves are tiny. Values of 5 to 20 are a good default on noisy data." },
        { name: "min_samples_split", range: "2 - 50", increasing: "Fewer splits attempted, shallower tree.", strategy: "Usually redundant with min_samples_leaf. Tune one of the two." },
        { name: "n_estimators (forest)", range: "100 - 1000", increasing: "Lower variance with strongly diminishing returns; it never causes overfitting, only cost.", strategy: "Raise until the OOB error curve flattens, then stop. More is never harmful, just slower." },
        { name: "max_features (forest)", range: "√p, log₂p, 0.3 - 1.0", increasing: "Trees become stronger individually but more correlated, so averaging helps less.", strategy: "√p for classification, p/3 for regression. Lower it if the trees look too similar." },
        { name: "criterion", range: "gini / entropy", increasing: "Not applicable", strategy: "Rarely matters. Gini is slightly faster and is the default." },
        { name: "ccp_alpha", range: "0 - 0.05", increasing: "More aggressive pruning, smaller tree, higher bias.", strategy: "Use the cost-complexity pruning path and pick α by cross-validation." },
      ],
      metrics: ["Accuracy", "F1-score", "Gini/feature importance", "Out-of-bag error (forest)", "Permutation importance (unbiased alternative)"],
      typicalUses: ["Credit scoring", "Medical diagnosis rule extraction", "Tabular-data benchmarks", "Feature-importance analysis"],
      diagnostics: [
        "For a forest, the out-of-bag score is a free validation estimate. If it is far above your test score, something leaked.",
        "Plot training against validation accuracy versus depth. The gap opening up marks the point where the tree starts memorising.",
        "Prefer permutation importance to the built-in impurity importance, which is biased toward high-cardinality and continuous features.",
        "A single tree that changes completely when you resample the data is telling you the variance is too high to trust it. Use a forest.",
        "Very deep trees with single-sample leaves are memorising and will not generalise.",
      ],
      advantages: [
        "A single shallow tree is genuinely readable: the decision path is the explanation.",
        "No scaling, centring, or normalisation is ever required.",
        "Handles numeric and categorical features together, and missing values with surrogate splits.",
        "Captures non-linearity and feature interactions automatically, with no manual feature crosses.",
        "Random forests are hard to misconfigure: they resist overfitting and work well at default settings.",
        "Prediction is extremely fast, being a single walk down the tree.",
      ],
      limitations: [
        { name: "High variance", note: "a small change in the data can restructure the whole tree", fix: "bagging, or a random forest." },
        { name: "Axis-aligned boundaries only", note: "diagonal boundaries need a staircase of splits", fix: "rotate features with PCA, or use an oblique tree or a linear model." },
        { name: "Cannot extrapolate", note: "predictions are constant beyond the training range", fix: "use a linear model for trends." },
        { name: "Greedy construction", note: "no guarantee of the best overall tree", fix: "usually acceptable; ensembles paper over it." },
        { name: "Biased impurity importance", note: "favours high-cardinality features", fix: "permutation importance or SHAP." },
        { name: "Forests lose interpretability", note: "hundreds of trees are no longer a readable rule set", fix: "SHAP values, or keep one shallow tree as a surrogate explanation." },
      ],
      alternatives: [
        { name: "Gradient boosting (XGBoost, LightGBM)", when: "You want the best tabular accuracy and can afford tuning. Usually beats a forest." },
        { name: "Logistic or linear regression", when: "The relationship is close to linear, or you must extrapolate." },
        { name: "Extremely randomized trees", when: "You want even lower variance and faster training than a random forest." },
        { name: "Rule-list models", when: "Interpretability is a hard requirement and a tree is still too complex." },
      ],
      pitfalls: [
        { problem: "100% training accuracy", solution: "The tree is unpruned and memorising. Cap depth or raise min_samples_leaf." },
        { problem: "Predictions change a lot between runs", solution: "Single-tree variance. Set a random_state, or move to a forest." },
        { problem: "One feature dominates the importances", solution: "Likely high cardinality or leakage. Check with permutation importance." },
        { problem: "Model does badly on a linear trend", solution: "Trees approximate slopes with staircases and cannot extrapolate. Use a linear model." },
        { problem: "Forest is slow to train", solution: "Lower n_estimators, cap depth, or set n_jobs=-1. Trees are embarrassingly parallel." },
        { problem: "Minority class never predicted", solution: "Set class_weight='balanced'." },
      ],
      quickRef: [
        { name: "Gini impurity", formula: "1 − Σ_c p_c²" },
        { name: "Entropy", formula: "−Σ_c p_c log₂ p_c" },
        { name: "Information gain", formula: "I(parent) − Σ (nₖ/n)·I(childₖ)" },
        { name: "Regression impurity", formula: "MSE = (1/n)Σ(yᵢ − ȳ)²" },
        { name: "Cost-complexity", formula: "R_α(T) = R(T) + α·|leaves|" },
        { name: "Forest prediction", formula: "majority vote over T trees" },
        { name: "OOB fraction", formula: "≈ 1/e ≈ 37% held out per tree" },
        { name: "Train cost", formula: "O(p · m log m)" },
      ],
      code: `from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# A single shallow tree: use it when the rules must be readable.
tree = DecisionTreeClassifier(
    max_depth=4,             # keep small enough to print
    min_samples_leaf=10,     # no leaf backed by fewer than 10 rows
    criterion="gini",
    class_weight="balanced",
    random_state=42,
).fit(X_train, y_train)
print(export_text(tree, feature_names=list(feature_names)))

# A forest: more accuracy, no longer readable.
forest = RandomForestClassifier(
    n_estimators=500,        # more is never worse, only slower
    max_features="sqrt",     # decorrelates the trees
    oob_score=True,          # free validation estimate
    n_jobs=-1, random_state=42,
).fit(X_train, y_train)
print("OOB:", forest.oob_score_)

# Built-in importances are cardinality-biased; permute instead.
imp = permutation_importance(forest, X_test, y_test, n_repeats=10)`,
      whyChain: [
        { q: "Why Gini rather than entropy?", a: "They almost always choose the same splits. Gini avoids computing a logarithm, so it is slightly faster, which is why it is the default. The choice is not worth tuning." },
        { q: "Why is a single decision tree unstable?", a: "Splits are chosen greedily and hierarchically. If a different split narrowly wins at the root, every subtree below it is built on different data, so the whole structure changes. Small data perturbations therefore produce very different trees." },
        { q: "How does a random forest fix that?", a: "By averaging. Individual high-variance, low-bias trees make uncorrelated errors, so averaging T of them cuts variance by roughly T while leaving bias unchanged." },
        { q: "Why sample features at each split, not just rows?", a: "Bootstrapping rows alone is not enough. If one feature is strongly predictive, every tree splits on it first and they all end up correlated, so averaging barely helps. Restricting each split to a random √p subset forces structural diversity." },
        { q: "Can adding more trees to a forest cause overfitting?", a: "No. The forest prediction converges to an expectation as T grows, so extra trees only cost compute. This is the opposite of boosting, where extra rounds absolutely can overfit." },
        { q: "What is out-of-bag error?", a: "Each bootstrap sample omits about 37% of rows. Scoring each tree on the rows it never saw gives an unbiased validation estimate for free, without holding out a separate set." },
        { q: "Why can a tree not extrapolate?", a: "Every leaf predicts a constant fitted from training rows. Any input beyond the training range falls into an edge leaf and receives that leaf's constant, so the prediction flatlines instead of continuing a trend." },
        { q: "Why is the built-in feature importance misleading?", a: "Impurity importance rewards features that offer more split points, so continuous and high-cardinality features look important even when they are random noise. Permutation importance measures actual predictive contribution instead." },
      ],
      parameters: [
        { name: "max depth", effect: "Limits tree complexity. Deeper trees fit more detail but overfit more readily." },
        { name: "min samples per split/leaf", effect: "Prevents tiny, noise-fitting leaves from forming." },
        { name: "n_estimators (forest)", effect: "Number of bagged trees. More trees reduce variance, with diminishing returns and higher compute cost." },
        { name: "bootstrap sample (forest)", effect: "Each tree trains on a random resample-with-replacement of the data, which is what makes the trees diverse enough to average usefully." },
      ],
      metrics: ["Accuracy", "F1-score", "Gini/feature importance", "Out-of-bag error (forest)"],
      typicalUses: ["Credit scoring", "Medical diagnosis rule extraction", "Tabular-data benchmarks", "Feature-importance analysis"],
      workedExample: {
        setup: "1D points x=1(A), x=2(A), x=3(B), x=4(B). Compare candidate splits at threshold 1.5 vs 2.5 using Gini.",
        steps: [
          "Split at 2.5 → left={x=1,2} both class A: Gini_left = 1−(1²+0²) = 0. Right={x=3,4} both class B: Gini_right = 0.",
          "Weighted Gini at 2.5 = (2/4)×0 + (2/4)×0 = 0.",
          "Split at 1.5 → left={x=1} class A: Gini_left = 0. Right={x=2,3,4} = {A,B,B}: p_A=1/3, p_B=2/3 → Gini_right = 1−(1/9+4/9) = 4/9 ≈ 0.444.",
          "Weighted Gini at 1.5 = (1/4)×0 + (3/4)×0.444 ≈ 0.333.",
        ],
        result: "Threshold 2.5 (weighted Gini 0) beats threshold 1.5 (weighted Gini 0.333) - the tree picks 2.5, a perfect split",
      },
    },
    mount,
  });
})();
