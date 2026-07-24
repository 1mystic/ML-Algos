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
        ? "Bagging: each tree trains on a bootstrap resample of the points; the boundary shown is the ensemble's majority vote — same idea as <code>mla/ensemble/random_forest.py</code>."
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
    category: "Supervised — Trees & Ensembles",
    tagline: "Gini splits, optional bagging",
    description: "A CART-style classifier that recursively splits on the axis-aligned threshold that most reduces Gini impurity. Switch to random forest mode to see bagged trees smooth out the blocky boundary.",
    sourceFile: "mla/ensemble/random_forest.py",
    info: {
      type: "Supervised — Classification/Regression. Non-parametric, tree-based (single tree); bagged ensemble in random-forest mode.",
      scenario: "You need an interpretable, non-linear model that handles mixed feature types without scaling (single tree); switch to random forest when you want more accuracy/robustness and can trade away some interpretability.",
      inputs: "Feature vectors x and class labels y.",
      decisionFunction: {
        text: "ŷ(x) = label at the leaf reached by recursively testing 'feature_j ≤ threshold?' (forest: majority vote across trees)",
        mechanism: "Each internal node routes a point left or right based on one feature/threshold; random forest trains many such trees on bootstrap-resampled data and votes.",
      },
      lossFunction: {
        text: "Gini(S) = 1 − Σ_c p_c²  (impurity), split chosen to minimize the weighted impurity of the two children",
        mechanism: "A greedy, recursive minimization — every split is locally optimal (not globally), and recursion stops at a depth/size limit or once a node is pure.",
        plot: { fn: (p) => 1 - (p * p + (1 - p) * (1 - p)), domain: [0, 1], yDomain: [0, 0.55], color: "var(--accent)", fn2: (p) => (-p * Math.log2(Math.max(p, 1e-9)) - (1 - p) * Math.log2(Math.max(1 - p, 1e-9))) / 2, color2: "var(--text-faint)", caption: "impurity of a 2-class node vs its class balance p — Gini (solid) and entropy/2 (dashed) both peak at a 50/50 split and hit 0 when pure" },
      },
      output: "A predicted class label (single tree: one leaf; forest: majority vote), optionally with class probabilities.",
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
        result: "Threshold 2.5 (weighted Gini 0) beats threshold 1.5 (weighted Gini 0.333) — the tree picks 2.5, a perfect split",
      },
    },
    mount,
  });
})();
