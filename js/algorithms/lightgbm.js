(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];

  function sse(vals) { if (!vals.length) return 0; const m = MLU.mean(vals); return vals.reduce((s, v) => s + (v - m) ** 2, 0); }
  function makeLeaf(points) { return { leaf: true, points, value: points.length ? MLU.mean(points.map((p) => p.y)) : 0 }; }
  function bestSplit(points) {
    if (points.length < 4) return null;
    const xs = [...new Set(points.map((p) => p.x))].sort((a, b) => a - b);
    if (xs.length < 2) return null;
    const base = sse(points.map((p) => p.y));
    let best = null;
    for (let i = 0; i < xs.length - 1; i++) {
      const thr = (xs[i] + xs[i + 1]) / 2;
      const left = points.filter((p) => p.x <= thr), right = points.filter((p) => p.x > thr);
      if (!left.length || !right.length) continue;
      const gain = base - (sse(left.map((p) => p.y)) + sse(right.map((p) => p.y)));
      if (!best || gain > best.gain) best = { thr, left, right, gain };
    }
    return best && best.gain > 1e-9 ? best : null;
  }
  // Leaf-wise (best-first): always split whichever current leaf gives the single largest gain.
  function buildLeafWise(points, budget) {
    const root = makeLeaf(points);
    let leaves = [root];
    while (leaves.length < budget) {
      let bestIdx = -1, bestInfo = null;
      leaves.forEach((leaf, i) => {
        const s = bestSplit(leaf.points);
        if (s && (!bestInfo || s.gain > bestInfo.gain)) { bestInfo = s; bestIdx = i; }
      });
      if (bestIdx === -1) break;
      const leaf = leaves[bestIdx];
      leaf.leaf = false; leaf.thr = bestInfo.thr;
      leaf.left = makeLeaf(bestInfo.left); leaf.right = makeLeaf(bestInfo.right);
      leaves.splice(bestIdx, 1, leaf.left, leaf.right);
    }
    return { root, leafCount: leaves.length };
  }
  // Level-wise (traditional): expand every current leaf together, one full level at a time.
  function buildLevelWise(points, budget) {
    const root = makeLeaf(points);
    let leaves = [root];
    while (leaves.length * 2 <= budget) {
      const current = leaves.slice();
      const next = [];
      let didSplit = false;
      for (const leaf of current) {
        const s = bestSplit(leaf.points);
        if (s) {
          leaf.leaf = false; leaf.thr = s.thr;
          leaf.left = makeLeaf(s.left); leaf.right = makeLeaf(s.right);
          next.push(leaf.left, leaf.right); didSplit = true;
        } else next.push(leaf);
      }
      leaves = next;
      if (!didSplit) break;
    }
    return { root, leafCount: leaves.length };
  }
  function predictTree(node, x) { return node.leaf ? node.value : predictTree(x <= node.thr ? node.left : node.right, x); }
  function targetFn(xv) { return 2.4 * Math.sin(xv * 0.7) + 0.15 * xv; }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">
              <span class="legend-item"><span class="swatch" style="background:var(--accent)"></span>leaf-wise (LightGBM-style)</span>
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[1]}"></span>level-wise (traditional)</span>
            </div>
            <div class="btn-row"><button id="lgb-regen">regenerate data</button></div>
          </div>
          <svg id="lgb-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">both trees are grown to (at most) the same leaf budget - compare the resulting fit and MSE</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>leaf budget <span class="val" id="lgb-budget-val">8</span></h3>
            <input type="range" id="lgb-budget" min="2" max="24" step="1" value="8" />
          </div>
          <div class="control-card">
            <h3>result</h3>
            <div class="readout" id="lgb-readout">–</div>
            <div class="note">Level-wise growth can only add a full layer at a time, so its leaf count jumps in powers of 2; leaf-wise growth spends every leaf where the single biggest error reduction is, hitting the exact budget and usually reaching lower training error for the same number of leaves - LightGBM's core real-world speed/accuracy trade-off.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#lgb-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    function genData() {
      const pts = [];
      for (let i = 0; i < 60; i++) { const xv = MLU.randRange(...DOMAIN); pts.push({ x: xv, y: targetFn(xv) + MLU.randn() * 0.6 }); }
      return pts;
    }
    let points = genData();

    const pathLeaf = svg.append("path").attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2.2);
    const pathLevel = svg.append("path").attr("fill", "none").attr("stroke", MLU.palette[1]).attr("stroke-width", 2.2).attr("stroke-dasharray", "5,3");
    const ptsG = svg.append("g");

    function budget() { return +document.getElementById("lgb-budget").value; }

    function curveFor(predictFn) {
      const steps = 160;
      const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
      return line(Array.from({ length: steps + 1 }, (_, i) => {
        const xv = DOMAIN[0] + (i / steps) * (DOMAIN[1] - DOMAIN[0]);
        return [xv, predictFn(xv)];
      }));
    }

    function render() {
      document.getElementById("lgb-budget-val").textContent = budget();
      const leafTree = buildLeafWise(points, budget());
      const levelTree = buildLevelWise(points, budget());
      const predLeaf = (xv) => predictTree(leafTree.root, xv);
      const predLevel = (xv) => predictTree(levelTree.root, xv);

      pathLeaf.attr("d", curveFor(predLeaf));
      pathLevel.attr("d", curveFor(predLevel));

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle").attr("r", 4.5).attr("fill", MLU.palette[2]).attr("stroke", "var(--bg)").attr("stroke-width", 1)
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y));
      sel.exit().remove();

      const mseLeaf = MLU.mean(points.map((p) => (p.y - predLeaf(p.x)) ** 2));
      const mseLevel = MLU.mean(points.map((p) => (p.y - predLevel(p.x)) ** 2));
      document.getElementById("lgb-readout").innerHTML =
        `leaf-wise: <b>${leafTree.leafCount}</b> leaves, MSE <b class="num">${mseLeaf.toFixed(3)}</b><br>` +
        `level-wise: <b>${levelTree.leafCount}</b> leaves, MSE <b class="num">${mseLevel.toFixed(3)}</b>`;
    }

    document.getElementById("lgb-budget").addEventListener("input", render);
    document.getElementById("lgb-regen").addEventListener("click", () => { points = genData(); render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "lightgbm",
    name: "LightGBM",
    category: "Supervised - Trees & Ensembles",
    tagline: "leaf-wise growth, histogram splits",
    description: "A gradient-boosting variant built for speed at scale: it grows each tree leaf-wise (always splitting whichever leaf helps most) instead of level-wise, and bins continuous features into histograms so split-finding is fast.",
    info: {
      type: "Supervised - Regression/Classification. Histogram-based, leaf-wise (best-first) gradient-boosted tree ensemble.",
      scenario: "Very large or high-dimensional tabular datasets where training speed and memory matter - LightGBM trades some tree 'balance' for much faster training.",
      inputs: "Feature vectors x (often large-n / high-cardinality) and targets y.",
      intuition: {
        definition: "The same boosted ensemble as XGBoost, made fast by two changes: bin every feature into a <b>histogram</b> once so split search costs O(bins) instead of O(n), and grow each tree <b>leaf-wise</b>, always splitting whichever leaf anywhere in the tree promises the biggest error reduction.",
        steps: [
          "Bucket continuous features into a few hundred bins up front.",
          "Search splits over bins, not raw sorted values.",
          "Split the single best leaf in the whole tree, not a whole level.",
          "Constrain num_leaves and min_data_in_leaf so the deep branches do not memorise.",
        ],
        applications: [
          "Tabular datasets with millions of rows where XGBoost is too slow",
          "High-cardinality categorical features, handled natively",
          "Low-latency retraining pipelines",
          "Learning-to-rank with LambdaRank objectives",
          "Click-through and conversion prediction at ad scale",
        ],
      },
      math: [
        { title: "Same objective as XGBoost", formula: "Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ,   w* = −G/(H+λ)", note: "The mathematics of what makes a good split is unchanged. Everything LightGBM does differently is about how that search is executed." },
        { title: "Histogram binning", formula: "split search: O(m·p) → O(bins·p) per node", note: "Each feature is discretized once into at most max_bin buckets. Gradient and Hessian sums are accumulated per bucket, so the per-node cost stops depending on the row count." },
        { title: "Histogram subtraction", formula: "hist(child_right) = hist(parent) − hist(child_left)", note: "Only the smaller child's histogram is built from data; the sibling comes free by subtraction. This roughly halves the work at every node." },
        { title: "Leaf-wise growth", formula: "split argmax_{leaf ∈ tree} Gain(leaf),  until num_leaves reached", note: "Best-first rather than level-by-level. For a fixed leaf budget this reaches lower training loss, because leaves are spent where the error is, not spread evenly." },
        { title: "GOSS", formula: "keep the top a% by |gradient|, sample b% of the rest, upweight them by (1−a)/b", note: "Gradient-based One-Side Sampling. Well-fitted points have small gradients and contribute little, so most of them can be dropped with a correction factor that keeps the gain estimate unbiased." },
        { title: "EFB", formula: "bundle mutually exclusive sparse features into one", note: "Exclusive Feature Bundling. In sparse one-hot data most feature pairs are never non-zero together, so they can share a bin range and cut the effective feature count." },
      ],
      pipeline: [
        { label: "Bin features", note: "once, max_bin buckets" },
        { label: "Accumulate g,h", note: "per bin" },
        { label: "Best leaf", note: "max gain anywhere" },
        { label: "Split + subtract", note: "sibling hist free" },
        { label: "num_leaves reached", note: "next round", accent: "green" },
      ],
      decisionFunction: {
        text: "Same additive ensemble ŷ(x) = F₀ + η·Σₘ treeₘ(x) as GBM/XGBoost - the difference is how each tree is grown.",
        mechanism: "Leaf-wise (best-first) growth repeatedly splits whichever current leaf anywhere in the tree gives the single largest error reduction - not depth-by-depth like standard CART/GBM/XGBoost - producing deeper, more asymmetric trees for the same leaf budget.",
      },
      lossFunction: {
        text: "Same gradient/hessian Gain criterion as XGBoost, applied to histogram-binned feature values",
        mechanism: "Continuous features are discretized into a fixed number of histogram bins up front, so split search costs O(bins) per feature instead of O(n) - a large constant-factor speedup at the cost of slightly coarser split thresholds.",
      },
      optimization: [
        { title: "Where the speed comes from", formula: "histograms + subtraction + GOSS + EFB", note: "No single trick dominates. Binning removes the row-count dependence, subtraction halves node work, GOSS cuts rows, and EFB cuts columns." },
        { title: "Binning is a one-off cost", formula: "O(m·p) once, then O(bins·p) per node forever", note: "The up-front discretization is amortised across every node of every tree, which is why the advantage grows with ensemble size." },
        { title: "Leaf-wise risk", formula: "unbounded depth for a fixed leaf budget", note: "Best-first growth can drive one branch very deep on a handful of rows. num_leaves and min_data_in_leaf are the guards, and they matter far more here than in XGBoost." },
        { title: "Depth relationship", formula: "num_leaves ≤ 2^max_depth", note: "If you set both, keep num_leaves comfortably below the level-wise equivalent. Setting num_leaves = 2^max_depth reproduces level-wise behaviour and gives up the advantage." },
        { title: "Native categoricals", formula: "sort categories by Σg/Σh, then split the sorted order", note: "Avoids one-hot explosion. Finds a near-optimal partition of category values in O(k log k) rather than trying all 2^(k−1) subsets." },
      ],
      output: "A continuous predicted value (sum of the ensemble's tree contributions).",
      assumptions: [
        { name: "Enough rows per leaf", why: "Leaf-wise growth will happily isolate tiny groups, which is the main way LightGBM overfits.", check: "Set min_data_in_leaf to at least 20, and much higher on small datasets." },
        { name: "Binning resolution is adequate", why: "Split thresholds are restricted to bin edges, so very fine distinctions can be lost.", check: "Raise max_bin if accuracy is limited; lower it for speed." },
        { name: "Dataset is reasonably large", why: "The histogram and sampling machinery has fixed overhead that only pays off at scale.", check: "Below a few thousand rows, XGBoost or a random forest is often simpler and just as good." },
        { name: "Tabular, structured features", why: "No notion of spatial or sequential structure.", check: "Use a neural network for unstructured data." },
        { name: "Validation split available", why: "Round count and leaf budget both need empirical selection.", check: "Always pass a valid_sets for early stopping." },
      ],
      regularization: [
        { name: "num_leaves", formula: "leaf count ≤ L", note: "The primary capacity control, replacing max_depth. This is the knob to tune first." },
        { name: "min_data_in_leaf", formula: "n_leaf ≥ k", note: "Critical under leaf-wise growth. Without it, deep branches will isolate individual rows." },
        { name: "min_sum_hessian_in_leaf", formula: "H_leaf ≥ h_min", note: "The curvature-weighted counterpart, equivalent to XGBoost's min_child_weight." },
        { name: "lambda_l1 / lambda_l2", formula: "+ α Σ|wⱼ| + ½λ Σwⱼ²", note: "Same leaf-weight penalties as XGBoost, entering the same w* and gain formulas." },
        { name: "feature_fraction", formula: "random subset of columns per tree", note: "Decorrelates trees and speeds up split search." },
        { name: "bagging_fraction", formula: "random subset of rows per iteration", note: "Requires bagging_freq to be set, otherwise it is silently ignored." },
        { name: "max_bin", formula: "histogram resolution", note: "Fewer bins is itself a regularizer: coarser thresholds cannot carve out noise as precisely." },
      ],
      hyperparameters: [
        { name: "num_leaves", range: "15 - 255", increasing: "Much more capacity and the fastest route to overfitting.", strategy: "The single most important parameter. Start at 31 and keep it well under 2^max_depth." },
        { name: "learning_rate", range: "0.01 - 0.3", increasing: "Fewer rounds required, higher overfitting risk.", strategy: "Fix at 0.05 and let early stopping choose num_iterations." },
        { name: "num_iterations", range: "100 - 10000", increasing: "More capacity, eventual overfitting.", strategy: "Set high, early-stop. Do not grid-search." },
        { name: "min_data_in_leaf", range: "20 - 500", increasing: "Stronger regularization, shallower effective trees.", strategy: "Raise aggressively on small or noisy data. This is the main defence against leaf-wise overfitting." },
        { name: "max_bin", range: "63 - 511", increasing: "Finer split thresholds, better accuracy, slower training and more memory.", strategy: "255 is the default and usually right. Drop to 63 for a quick speed win." },
        { name: "feature_fraction", range: "0.4 - 1.0", increasing: "Trees see more features and correlate more.", strategy: "0.8 default; lower with many correlated columns." },
        { name: "bagging_fraction", range: "0.4 - 1.0", increasing: "Less regularization from sampling.", strategy: "0.8 with bagging_freq=1. Remember both must be set." },
        { name: "lambda_l2", range: "0 - 10", increasing: "Leaf values shrink toward zero.", strategy: "Log-scale search jointly with num_leaves." },
        { name: "max_depth", range: "-1 (none) or 3 - 12", increasing: "Bounds how deep leaf-wise growth can run.", strategy: "Leave unlimited and control with num_leaves; set it only as a safety net on small data." },
      ],
      metrics: ["RMSE / MAE / R² (regression)", "Log-loss / AUC (classification)", "NDCG / MAP (ranking)", "Training time & memory footprint (a key differentiator at scale)"],
      typicalUses: ["Very large tabular datasets", "Low-latency training pipelines", "Ranking/recommendation (e.g. LambdaRank objectives)", "Anywhere GBM-quality accuracy is needed but training speed/memory is the bottleneck"],
      diagnostics: [
        "If training accuracy is near perfect while validation lags badly, num_leaves is too high or min_data_in_leaf too low. Fix those before touching anything else.",
        "Inspect the actual tree depths. Leaf-wise growth producing depth-30 branches on a small dataset is a warning sign.",
        "Watch the early-stopping curve. LightGBM converges in fewer rounds than XGBoost at the same learning rate because leaf-wise growth fits faster per tree.",
        "If bagging appears to do nothing, check that bagging_freq is set. bagging_fraction alone is ignored.",
        "On small datasets, compare against XGBoost. LightGBM's advantages are scale advantages and can invert below a few thousand rows.",
      ],
      advantages: [
        "Dramatically faster training than exact-split boosting, often by an order of magnitude on large data.",
        "Much lower memory use, since binned features are stored compactly as small integers.",
        "Leaf-wise growth reaches lower loss for the same number of leaves.",
        "Categorical features are handled natively without one-hot encoding.",
        "Scales to millions of rows and high-dimensional sparse data via GOSS and EFB.",
        "Distributed and GPU training are supported out of the box.",
      ],
      limitations: [
        { name: "Overfits small datasets readily", note: "leaf-wise growth isolates tiny groups", fix: "raise min_data_in_leaf, lower num_leaves, or use XGBoost instead." },
        { name: "num_leaves is easy to misconfigure", note: "it is not interchangeable with max_depth", fix: "keep num_leaves well below 2^max_depth." },
        { name: "Binning loses precision", note: "split thresholds are limited to bin edges", fix: "raise max_bin when accuracy matters more than speed." },
        { name: "Unbalanced trees", note: "deep asymmetric branches are harder to reason about", fix: "cap max_depth as a safety net." },
        { name: "Sensitive to parameter interactions", note: "bagging_fraction silently does nothing without bagging_freq", fix: "read the parameter aliases carefully." },
        { name: "Cannot extrapolate", note: "standard tree limitation", fix: "detrend or model the trend separately." },
      ],
      alternatives: [
        { name: "XGBoost", when: "Smaller datasets, or you prefer level-wise growth's more conservative behaviour." },
        { name: "CatBoost", when: "Categorical features dominate, or you want strong results with minimal tuning." },
        { name: "Random forest", when: "You want robustness with essentially no tuning." },
        { name: "Linear model", when: "Interpretability or extrapolation is required." },
      ],
      pitfalls: [
        { problem: "Severe overfitting on a small dataset", solution: "Lower num_leaves, raise min_data_in_leaf substantially, and consider XGBoost." },
        { problem: "Setting num_leaves = 2^max_depth", solution: "That reproduces level-wise growth and discards the leaf-wise advantage. Keep it much lower." },
        { problem: "Bagging has no effect", solution: "bagging_freq must be set to a positive integer alongside bagging_fraction." },
        { problem: "Warning about no further splits", solution: "min_data_in_leaf or min_gain_to_split is too strict for the data volume." },
        { problem: "Categorical feature performs poorly", solution: "Pass it via categorical_feature rather than one-hot encoding, and beware high-cardinality overfitting." },
        { problem: "Results differ run to run", solution: "Sampling and multithreading introduce nondeterminism. Set seeds and deterministic=True." },
      ],
      quickRef: [
        { name: "Split gain", formula: "same as XGBoost" },
        { name: "Leaf weight", formula: "w* = −G/(H+λ)" },
        { name: "Growth", formula: "leaf-wise (best-first)" },
        { name: "Split cost", formula: "O(bins·p) per node" },
        { name: "Hist subtraction", formula: "sibling = parent − child" },
        { name: "Leaf budget", formula: "num_leaves ≤ 2^max_depth" },
        { name: "GOSS", formula: "keep top-a% |g|, sample b% rest" },
        { name: "EFB", formula: "bundle exclusive sparse features" },
      ],
      code: `import lightgbm as lgb

train_set = lgb.Dataset(X_fit, y_fit, categorical_feature=cat_cols)
valid_set = lgb.Dataset(X_val, y_val, reference=train_set)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,            # the main capacity lever, not max_depth
    "min_data_in_leaf": 50,      # critical guard for leaf-wise growth
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,           # without this, bagging_fraction is ignored
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbosity": -1,
    "seed": 42,
}

model = lgb.train(
    params, train_set,
    num_boost_round=5000,        # upper bound; early stopping decides
    valid_sets=[valid_set],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
print("best iteration:", model.best_iteration)`,
      whyChain: [
        { q: "What is the difference between leaf-wise and level-wise growth?", a: "Level-wise splits every node at the current depth before descending, producing balanced trees. Leaf-wise repeatedly splits whichever single leaf in the whole tree offers the largest gain, producing deep asymmetric trees. For the same leaf count, leaf-wise reaches lower training loss." },
        { q: "So why does leaf-wise overfit more?", a: "Because nothing stops it concentrating every remaining leaf on one hard corner of the data. It can drive a branch thirty levels deep to isolate a handful of noisy rows. Level-wise growth is implicitly regularized by having to spend leaves evenly." },
        { q: "Why does binning make it so much faster?", a: "Exact split search must sort and scan every value, which is O(m log m) per feature per node. With histograms you accumulate gradient sums into a fixed number of buckets, so the per-node cost becomes O(bins) and stops depending on the row count entirely." },
        { q: "Does binning hurt accuracy?", a: "Barely. Split thresholds are restricted to bin edges, but with 255 bins the loss of precision is usually below the noise floor. Coarser bins even act as a mild regularizer." },
        { q: "What is histogram subtraction?", a: "After building the histogram for one child, the sibling's histogram is just the parent's minus that child's. So you only ever build the smaller child from raw data, which roughly halves the work per node." },
        { q: "What is the intuition behind GOSS?", a: "Points with small gradients are already well fitted and contribute little to the split gain. GOSS keeps all the large-gradient points, randomly samples the rest, and upweights the survivors so the gain estimate stays unbiased. You get most of the information from a fraction of the rows." },
        { q: "Why not just set num_leaves = 2^max_depth?", a: "That gives leaf-wise growth exactly the budget a balanced tree would use, so it behaves like level-wise growth and you lose the benefit. The point is to use fewer leaves, placed where they matter." },
        { q: "When should you prefer XGBoost?", a: "On smaller datasets, where LightGBM's overhead does not pay off and its leaf-wise growth overfits more readily. XGBoost with tree_method='hist' also closes much of the speed gap." },
      ],
      parameters: [
        { name: "num_leaves", effect: "Direct leaf-wise complexity control (used instead of max_depth) - the main lever for model capacity." },
        { name: "learning rate", effect: "Shrinks each tree's contribution, same role as in GBM/XGBoost." },
        { name: "max_bin", effect: "Histogram resolution. More bins → more precise splits but slower and more memory." },
        { name: "min_data_in_leaf", effect: "Regularizes against overfitting on tiny leaves - important since leaf-wise growth can otherwise carve out very small, deep leaves." },
      ],
      metrics: ["RMSE / MAE / R² (regression)", "Log-loss / AUC (classification)", "Training time & memory footprint (a key differentiator at scale)"],
      typicalUses: ["Very large tabular datasets", "Low-latency training pipelines", "Ranking/recommendation (e.g. LambdaRank objectives)", "Anywhere GBM-quality accuracy is needed but training speed/memory is the bottleneck"],
      workedExample: {
        setup: "Grow a tree to a budget of 5 leaves on a small dataset, comparing leaf-wise vs level-wise growth.",
        steps: [
          "Level-wise can only add a full layer at a time: 1 leaf → split root → 2 leaves → split both → 4 leaves. Adding a 5th leaf would require a full next level (8 leaves), which exceeds the budget - so level-wise stops at 4 leaves.",
          "Leaf-wise instead asks 'which single leaf, if split, reduces error the most?' at every step: split root (1→2 leaves), then split whichever of those 2 has the best next split (2→3), then again (3→4), then again (4→5) - reaching the full budget of 5.",
          "Because leaf-wise always spends its next leaf where the error reduction is largest, its 5 leaves are concentrated where the data is hardest to fit, rather than spread evenly.",
        ],
        result: "For the same leaf budget, leaf-wise typically reaches lower training error than level-wise - exactly the comparison the playground above shows live",
      },
    },
    mount,
  });
})();
