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
          <div class="stage-hint">both trees are grown to (at most) the same leaf budget — compare the resulting fit and MSE</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>leaf budget <span class="val" id="lgb-budget-val">8</span></h3>
            <input type="range" id="lgb-budget" min="2" max="24" step="1" value="8" />
          </div>
          <div class="control-card">
            <h3>result</h3>
            <div class="readout" id="lgb-readout">–</div>
            <div class="note">Level-wise growth can only add a full layer at a time, so its leaf count jumps in powers of 2; leaf-wise growth spends every leaf where the single biggest error reduction is, hitting the exact budget and usually reaching lower training error for the same number of leaves — LightGBM's core real-world speed/accuracy trade-off.</div>
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
    category: "Supervised — Trees & Ensembles",
    tagline: "leaf-wise growth, histogram splits",
    description: "A gradient-boosting variant built for speed at scale: it grows each tree leaf-wise (always splitting whichever leaf helps most) instead of level-wise, and bins continuous features into histograms so split-finding is fast.",
    sourceFile: "not in the original repo — added as a widely-used, high-performance extension of mla/ensemble/gbm.py's plain gradient boosting",
    info: {
      type: "Supervised — Regression/Classification. Histogram-based, leaf-wise (best-first) gradient-boosted tree ensemble.",
      scenario: "Very large or high-dimensional tabular datasets where training speed and memory matter — LightGBM trades some tree 'balance' for much faster training.",
      inputs: "Feature vectors x (often large-n / high-cardinality) and targets y.",
      decisionFunction: {
        text: "Same additive ensemble ŷ(x) = F₀ + η·Σₘ treeₘ(x) as GBM/XGBoost — the difference is how each tree is grown.",
        mechanism: "Leaf-wise (best-first) growth repeatedly splits whichever current leaf anywhere in the tree gives the single largest error reduction — not depth-by-depth like standard CART/GBM/XGBoost — producing deeper, more asymmetric trees for the same leaf budget.",
      },
      lossFunction: {
        text: "Same gradient/hessian Gain criterion as XGBoost, applied to histogram-binned feature values",
        mechanism: "Continuous features are discretized into a fixed number of histogram bins up front, so split search costs O(bins) per feature instead of O(n) — a large constant-factor speedup at the cost of slightly coarser split thresholds.",
      },
      output: "A continuous predicted value (sum of the ensemble's tree contributions).",
      parameters: [
        { name: "num_leaves", effect: "Direct leaf-wise complexity control (used instead of max_depth) — the main lever for model capacity." },
        { name: "learning rate", effect: "Shrinks each tree's contribution, same role as in GBM/XGBoost." },
        { name: "max_bin", effect: "Histogram resolution. More bins → more precise splits but slower and more memory." },
        { name: "min_data_in_leaf", effect: "Regularizes against overfitting on tiny leaves — important since leaf-wise growth can otherwise carve out very small, deep leaves." },
      ],
      metrics: ["RMSE / MAE / R² (regression)", "Log-loss / AUC (classification)", "Training time & memory footprint (a key differentiator at scale)"],
      typicalUses: ["Very large tabular datasets", "Low-latency training pipelines", "Ranking/recommendation (e.g. LambdaRank objectives)", "Anywhere GBM-quality accuracy is needed but training speed/memory is the bottleneck"],
      workedExample: {
        setup: "Grow a tree to a budget of 5 leaves on a small dataset, comparing leaf-wise vs level-wise growth.",
        steps: [
          "Level-wise can only add a full layer at a time: 1 leaf → split root → 2 leaves → split both → 4 leaves. Adding a 5th leaf would require a full next level (8 leaves), which exceeds the budget — so level-wise stops at 4 leaves.",
          "Leaf-wise instead asks 'which single leaf, if split, reduces error the most?' at every step: split root (1→2 leaves), then split whichever of those 2 has the best next split (2→3), then again (3→4), then again (4→5) — reaching the full budget of 5.",
          "Because leaf-wise always spends its next leaf where the error reduction is largest, its 5 leaves are concentrated where the data is hardest to fit, rather than spread evenly.",
        ],
        result: "For the same leaf budget, leaf-wise typically reaches lower training error than level-wise — exactly the comparison the playground above shows live",
      },
    },
    mount,
  });
})();
