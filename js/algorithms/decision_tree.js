(() => {
  const { ui, col, fmt } = Lab;
  const NC = 3;
  const counts = (pts, idx) => { const c = new Array(NC).fill(0); for (const i of idx) c[pts[i].label]++; return c; };
  const IMP = {
    gini: (c, n) => (n ? 1 - c.reduce((s, v) => s + (v / n) ** 2, 0) : 0),
    entropy: (c, n) => (n ? -c.reduce((s, v) => s + (v ? (v / n) * Math.log2(v / n) : 0), 0) : 0),
  };
  const argmax = (a) => a.indexOf(Math.max(...a));

  // Best axis-aligned split of a node; also returns the impurity curve per feature.
  function bestSplit(pts, idx, crit, feats, minLeaf) {
    const n = idx.length, parent = IMP[crit](counts(pts, idx), n);
    let best = null; const curves = {};
    for (const f of feats) {
      const sorted = idx.slice().sort((a, b) => pts[a][f] - pts[b][f]);
      const left = new Array(NC).fill(0), right = counts(pts, idx);
      curves[f] = [];
      for (let k = 0; k < n - 1; k++) {
        const l = pts[sorted[k]].label; left[l]++; right[l]--;
        const v0 = pts[sorted[k]][f], v1 = pts[sorted[k + 1]][f];
        if (v1 - v0 < 1e-9) continue;
        const nl = k + 1, nr = n - nl;
        const w = (nl * IMP[crit](left, nl) + nr * IMP[crit](right, nr)) / n;
        const thr = (v0 + v1) / 2;
        curves[f].push([thr, w]);
        if (nl < minLeaf || nr < minLeaf) continue;
        if (!best || w < best.w - 1e-12) best = { f, thr, w, gain: parent - w };
      }
    }
    return { best, curves, parent };
  }

  function growTree(pts, idx, { maxDepth, minLeaf, crit, featPick }) {
    const nodes = [];
    const root = { id: 0, depth: 0, idx, bounds: [-10, 10, -10, 10] };
    nodes.push(root);
    const queue = [root];
    let order = 0;
    while (queue.length) {
      const nd = queue.shift();
      nd.counts = counts(pts, nd.idx);
      nd.n = nd.idx.length;
      nd.imp = IMP[crit](nd.counts, nd.n);
      nd.cls = argmax(nd.counts);
      if (nd.depth >= maxDepth || nd.n < 2 * minLeaf || nd.imp < 1e-12) { nd.leaf = true; continue; }
      const sp = bestSplit(pts, nd.idx, crit, featPick(), minLeaf);
      if (!sp.best || sp.best.gain <= 1e-12) { nd.leaf = true; continue; }
      Object.assign(nd, { f: sp.best.f, thr: sp.best.thr, gain: sp.best.gain, curves: sp.curves, order: order++ });
      const [x0, x1, y0, y1] = nd.bounds;
      const L = { id: nodes.length, depth: nd.depth + 1, parent: nd.id, idx: nd.idx.filter((i) => pts[i][nd.f] <= nd.thr), bounds: nd.f === "x" ? [x0, nd.thr, y0, y1] : [x0, x1, y0, nd.thr] };
      nodes.push(L);
      const R = { id: nodes.length, depth: nd.depth + 1, parent: nd.id, idx: nd.idx.filter((i) => pts[i][nd.f] > nd.thr), bounds: nd.f === "x" ? [nd.thr, x1, y0, y1] : [x0, x1, nd.thr, y1] };
      nodes.push(R);
      nd.left = L.id; nd.right = R.id;
      queue.push(L, R);
    }
    return nodes;
  }
  // Predict with only the first `s` splits applied (for animation).
  function leafOf(nodes, p, s = Infinity) {
    let nd = nodes[0];
    while (nd.order != null && nd.order < s) nd = nodes[p[nd.f] <= nd.thr ? nd.left : nd.right];
    return nd;
  }

  function treeFrames(pts, P) {
    const nodes = growTree(pts, pts.map((_, i) => i), { ...P, featPick: () => ["x", "y"] });
    const splits = nodes.filter((n) => n.order != null).sort((a, b) => a.order - b.order);
    const frames = [{ phase: "root", major: true, arch: ["data", "node"], title: `Root: all ${pts.length} points, ${P.crit} = ${nodes[0].imp.toFixed(3)}`, text: "A tree starts with every point in one region and predicts the majority class. Impurity measures how mixed the region is: 0 means pure.", tex: P.crit === "gini" ? "G=1-\\sum_k p_k^2" : "H=-\\sum_k p_k\\log_2 p_k", st: { s: 0, nodes, cur: 0 }, m: { acc: acc(nodes, pts, 0) } }];
    splits.forEach((nd, k) => {
      if (P.scan && k < 3) {
        const curve = nd.curves[nd.f];
        const picks = [0.15, 0.4, 0.65, 0.9].map((q) => curve[Math.floor(q * (curve.length - 1))]).filter(Boolean);
        for (const c of picks) frames.push({ phase: "scan", arch: ["node", "scan"], title: `Scanning thresholds on ${nd.f} for node ${nd.id}`, text: `Candidate ${nd.f} ≤ ${c[0].toFixed(2)} gives weighted child impurity <b>${c[1].toFixed(3)}</b>. Every midpoint between sorted values is tried, on every feature.`, tex: "\\text{score}(t)=\\tfrac{n_L}{n}\\,I(L)+\\tfrac{n_R}{n}\\,I(R)", st: { s: k, nodes, cur: nd.id, cand: { f: nd.f, thr: c[0] } }, m: { acc: acc(nodes, pts, k) } });
      }
      frames.push({
        phase: "split", major: k === 0, arch: ["scan", "split", "rec"],
        title: `Split ${k + 1}: ${nd.f} ≤ ${nd.thr.toFixed(2)} (depth ${nd.depth})`,
        text: `Best of all candidates: impurity drops from ${nd.imp.toFixed(3)} to ${(nd.imp - nd.gain).toFixed(3)} (gain <b>${nd.gain.toFixed(3)}</b>). The region is cut in two and each child is handled recursively.`,
        tex: "\\Delta I = I(\\text{parent})-\\Big(\\tfrac{n_L}{n}I(L)+\\tfrac{n_R}{n}I(R)\\Big)",
        st: { s: k + 1, nodes, cur: nd.id }, m: { acc: acc(nodes, pts, k + 1) },
      });
    });
    const leaves = nodes.filter((n) => n.leaf).length;
    frames.push({ phase: "done", major: true, arch: ["leaf", "out"], title: `Tree complete: ${splits.length} splits, ${leaves} leaves`, text: `Each leaf predicts its majority class. Training accuracy <b>${(acc(nodes, pts, Infinity) * 100).toFixed(1)}%</b>. Deep trees carve tiny boxes around individual points: that is overfitting, which is what forests fix.`, st: { s: Infinity, nodes, cur: null }, m: { acc: acc(nodes, pts, Infinity) } });
    return frames;
  }
  function acc(nodes, pts, s) { return pts.length ? pts.filter((p) => leafOf(nodes, p, s).cls === p.label).length / pts.length : 0; }

  function forestFrames(pts, P) {
    const n = pts.length, trees = [], inBag = [];
    const frames = [];
    const votes = pts.map(() => new Array(NC).fill(0));
    for (let t = 0; t < P.nTrees; t++) {
      const idx = Array.from({ length: n }, () => MLU.randInt(n));
      const bag = new Set(idx);
      const nodes = growTree(pts, idx, { ...P, featPick: () => (P.mf === "one" ? [MLU.rng() < 0.5 ? "x" : "y"] : ["x", "y"]) });
      trees.push(nodes); inBag.push(bag);
      // out-of-bag error using trees that did not see each point
      for (let i = 0; i < n; i++) if (!bag.has(i)) votes[i][leafOf(nodes, pts[i]).cls]++;
      let oobN = 0, oobOk = 0;
      for (let i = 0; i < n; i++) if (votes[i].some((v) => v)) { oobN++; if (argmax(votes[i]) === pts[i].label) oobOk++; }
      const mult = new Array(n).fill(0); idx.forEach((i) => mult[i]++);
      frames.push({
        phase: t === 0 ? "bag" : "tree", major: t === 0, arch: t === 0 ? ["data", "boot", "tree1"] : ["boot", "tree1", "vote"],
        title: `Tree ${t + 1} of ${P.nTrees}: bootstrap sample, ${n - bag.size} points out-of-bag`,
        text: `Each tree trains on n draws <i>with replacement</i> (big dots = drawn several times, faded = left out). ${P.mf === "one" ? "Each split also only sees one randomly chosen feature, " : ""}which decorrelates the trees. The field shows the majority vote of all ${t + 1} trees. OOB error: <b>${oobN ? ((1 - oobOk / oobN) * 100).toFixed(1) + "%" : "–"}</b>.`,
        tex: "\\hat y(x)=\\operatorname{mode}\\{\\,h_1(x),\\dots,h_T(x)\\,\\}",
        st: { forest: true, T: t + 1, trees, mult, last: nodes }, m: { oob: oobN ? 1 - oobOk / oobN : null, acc: forestAcc(trees.slice(0, t + 1), pts) },
      });
    }
    frames.push({ ...frames[frames.length - 1], phase: "done", major: true, arch: ["vote", "out"], title: `Forest of ${P.nTrees} trees`, text: "Averaging many high-variance, decorrelated trees keeps their low bias while cancelling much of their variance: the boundary is smoother and the OOB error (a free validation estimate) settles." });
    return frames;
  }
  function forestVote(trees, p) { const v = new Array(NC).fill(0); for (const t of trees) v[leafOf(t, p).cls]++; return v; }
  function forestAcc(trees, pts) { return pts.filter((p) => argmax(forestVote(trees, p)) === p.label).length / Math.max(1, pts.length); }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "dt", fps: 2.5, classes: NC,
      legend: ui.legend([["var(--series-4)", "node being split", "ring"], ["var(--text)", "split line", "line"]]),
      tools: `<div class="seg" data-r="mode"><button data-m="tree" class="on">decision tree</button><button data-m="forest">random forest</button></div>` + ui.btn("retrain", "retrain", true),
      hint: "click adds the selected class · drag / double-click to edit",
      data: { kind: "class", presets: ["blobs3", "moons", "xor", "circles", "spiral", "overlap"], def: "blobs3", n: 90, noise: 0.35 },
      side: [
        { title: "tree", html: ui.range("depth", "max depth", 1, 10, 1, 4) + ui.range("leaf", "min samples per leaf", 1, 15, 1, 2) + ui.select("crit", "criterion", [["gini", "Gini impurity"], ["entropy", "entropy (information gain)"]], "gini") + ui.check("scan", "animate threshold scan", true) },
        { title: "forest", id: "forest", html: ui.range("nt", "trees", 2, 60, 1, 25) + ui.select("mf", "features per split", [["one", "1 random (max_features = 1)"], ["all", "both (bagging only)"]], "one") },
      ],
      arch: {
        w: 380, h: 200,
        nodes: [
          { id: "data", label: "data", sub: "(X, y)", x: 40, y: 40, w: 56, kind: "data" },
          { id: "boot", label: "bootstrap", sub: "forest only", x: 44, y: 150, kind: "param" },
          { id: "node", label: "node", sub: "region + points", x: 132, y: 40, kind: "op" },
          { id: "scan", label: "scan splits", sub: "features × thresholds", x: 254, y: 40, kind: "loss" },
          { id: "split", label: "best split", sub: "max impurity drop", x: 322, y: 100, kind: "op" },
          { id: "rec", label: "recurse", sub: "children", x: 214, y: 100, kind: "op" },
          { id: "leaf", label: "leaf", sub: "majority class", x: 116, y: 100, kind: "out" },
          { id: "tree1", label: "tree hₜ", sub: "fully grown", x: 152, y: 162, kind: "op" },
          { id: "vote", label: "vote", sub: "mode of T trees", x: 262, y: 162, kind: "op" },
          { id: "out", label: "ŷ", x: 346, y: 162, w: 44, kind: "out" },
        ],
        edges: [{ from: "data", to: "node" }, { from: "node", to: "scan" }, { from: "scan", to: "split" }, { from: "split", to: "rec" }, { from: "rec", to: "node", dash: true }, { from: "rec", to: "leaf", label: "stop" }, { from: "data", to: "boot" }, { from: "boot", to: "tree1" }, { from: "tree1", to: "vote" }, { from: "vote", to: "out" }, { from: "leaf", to: "out", bend: 30 }],
      },
      phases: { root: { label: "root", color: 3 }, scan: { label: "threshold scan", color: 4 }, split: { label: "split", color: 0 }, bag: { label: "bootstrap", color: 3 }, tree: { label: "add tree", color: 0 }, done: { label: "done", color: 2 } },
      panels: [{ id: "tree", title: "tree structure", sub: "colour = majority class · current node outlined", wide: true }, { id: "curve", title: "impurity vs threshold at this node" }],
      charts: [{ id: "acc", title: "accuracy", series: [{ key: "acc", label: "train acc", color: "var(--series-3)" }, { key: "oob", label: "OOB error", color: "var(--series-2)", dash: "4 3" }] }],
      onFrame: render, onData: () => retrain(),
    });
    lab.editable({ onChange: () => retrain({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    let mode = "tree";
    const params = () => ({ maxDepth: lab.val("depth"), minLeaf: lab.val("leaf"), crit: lab.val("crit"), scan: lab.val("scan"), nTrees: lab.val("nt"), mf: lab.val("mf") });
    function retrain({ at = "start" } = {}) {
      lab.q('[data-side="forest"]').style.opacity = mode === "forest" ? 1 : 0.45;
      if (lab.points.length < 2) { lab.load([{ phase: "root", title: "Add some points", text: "", st: { s: 0, nodes: null } }]); return; }
      MLU.seedRng(17);
      lab.load(mode === "tree" ? treeFrames(lab.points, params()) : forestFrames(lab.points, params()), { at, play: at === "start" });
    }
    lab.R("mode").onclick = (e) => {
      const b = e.target.closest("button"); if (!b) return;
      mode = b.dataset.m;
      lab.R("mode").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      retrain();
    };
    const splitG = L.mid.append("g"), regionG = L.mid.append("g");
    const treeEl = lab.R("panel-tree"), curveEl = lab.R("panel-curve");

    function drawTree(nodes, s, cur) {
      treeEl.innerHTML = "";
      if (!nodes) return;
      const vis = (nd) => nd.parent == null || (nodes[nd.parent].order != null && nodes[nd.parent].order < s);
      const build = (nd) => ({ nd, children: nd.order != null && nd.order < s ? [build(nodes[nd.left]), build(nodes[nd.right])] : [] });
      const h = d3.hierarchy(build(nodes[0]), (d) => d.children);
      const depth = h.height + 1, leaves = h.leaves().length;
      const W = Math.max(360, leaves * 58), H = 30 + depth * 54;
      d3.tree().size([W - 60, H - 46])(h);
      const svg = d3.select(treeEl).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("class", "arch-svg").style("max-height", "300px");
      const g = svg.append("g").attr("transform", "translate(30,22)");
      g.selectAll("path").data(h.links()).enter().append("path").attr("d", d3.linkVertical().x((d) => d.x).y((d) => d.y)).attr("fill", "none").attr("stroke", "var(--border)").attr("stroke-width", 1.4);
      g.selectAll("text.edge").data(h.links()).enter().append("text").attr("class", "ae-label").attr("x", (d) => (d.source.x + d.target.x) / 2 + (d.target.x < d.source.x ? -6 : 6)).attr("y", (d) => (d.source.y + d.target.y) / 2).attr("text-anchor", (d) => (d.target.x < d.source.x ? "end" : "start")).text((d) => (d.target.x < d.source.x ? "yes" : "no"));
      const ng = g.selectAll("g.tn").data(h.descendants()).enter().append("g").attr("transform", (d) => `translate(${d.x},${d.y})`);
      ng.append("rect").attr("x", -27).attr("y", -13).attr("width", 54).attr("height", 26).attr("rx", 5)
        .attr("fill", (d) => `color-mix(in srgb, ${col(d.data.nd.cls)} ${Math.round((1 - d.data.nd.imp) * 45 + 10)}%, var(--bg-inset))`)
        .attr("stroke", (d) => (d.data.nd.id === cur ? "var(--series-4)" : "var(--border)")).attr("stroke-width", (d) => (d.data.nd.id === cur ? 2.6 : 1));
      ng.append("text").attr("class", "an-label").attr("y", -1).attr("text-anchor", "middle").style("font-size", "9px").attr("fill", "var(--text)")
        .text((d) => (d.children ? `${d.data.nd.f} ≤ ${d.data.nd.thr.toFixed(1)}` : `class ${d.data.nd.cls}`));
      ng.append("text").attr("y", 9).attr("text-anchor", "middle").style("font-size", "8px").attr("fill", "var(--text-faint)").text((d) => d.data.nd.counts.join("/"));
    }
    function drawCurve(nd) {
      curveEl.innerHTML = "";
      if (!nd || !nd.curves) { curveEl.innerHTML = `<div class="lab-card-note">${mode === "forest" ? "Each tree in the forest is grown exactly like the single tree, on its own bootstrap sample." : "Pick a step that splits a node to see its threshold scan."}</div>`; return; }
      const all = [...(nd.curves.x || []), ...(nd.curves.y || [])];
      if (!all.length) return;
      const P = Lab.xy(curveEl, { w: 360, h: 140, xd: [-10, 10], yd: [0, Math.max(...all.map((c) => c[1])) * 1.1], xLabel: "threshold" });
      if (nd.curves.x) P.line("x", nd.curves.x, { color: "var(--series-1)", curve: d3.curveStepAfter });
      if (nd.curves.y) P.line("y", nd.curves.y, { color: "var(--series-5)", curve: d3.curveStepAfter });
      P.vline(nd.thr);
      curveEl.closest(".lab-card").querySelector(".lab-card-sub").innerHTML = `<span class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>split on x</span> <span class="legend-item"><span class="swatch" style="background:var(--series-5)"></span>split on y</span>`;
    }

    function render(f) {
      const st = f.st, pts = lab.points;
      splitG.selectAll("*").remove(); regionG.selectAll("*").remove();
      if (!st.nodes && !st.forest) { L.bg.selectAll("*").remove(); lab.drawPoints({}); return; }
      if (st.forest) {
        Lab.field(L.bg, pl, (x, y) => { const v = forestVote(st.trees.slice(0, st.T), { x, y }); const k = argmax(v); return k + (v[k] / st.T) * 0.999; }, (v) => Lab.classColor(Math.floor(v), ((v % 1) - 0.33) * 1.5, 0.42), { res: 80 });
        // outline the newest tree's partition
        for (const nd of st.last) if (nd.leaf) {
          const [x0, x1, y0, y1] = nd.bounds;
          regionG.append("rect").attr("x", pl.x(x0)).attr("y", pl.y(y1)).attr("width", pl.x(x1) - pl.x(x0)).attr("height", pl.y(y0) - pl.y(y1)).attr("fill", "none").attr("stroke", "var(--text)").attr("stroke-opacity", 0.22);
        }
        lab.drawPoints({ color: (d) => col(d.label), r: (d, i) => (st.mult[i] ? 3.2 + Math.min(4, st.mult[i]) * 1.4 : 3.4), opacity: (d, i) => (st.mult[i] ? 1 : 0.3) });
        drawTree(st.last, Infinity, null);
        drawCurve(null);
        lab.readout(`<div class="kv"><span>trees</span><b>${st.T}</b><span>train accuracy</span><b>${((f.m.acc || 0) * 100).toFixed(1)}%</b><span>OOB error</span><b>${f.m.oob != null ? (f.m.oob * 100).toFixed(1) + "%" : "–"}</b></div>`);
        return;
      }
      const nodes = st.nodes, s = st.s;
      Lab.field(L.bg, pl, (x, y) => { const nd = leafOf(nodes, { x, y }, s); return nd.cls + (nd.counts[nd.cls] / Math.max(1, nd.n)) * 0.999; }, (v) => Lab.classColor(Math.floor(v), ((v % 1) - 0.33) * 1.5, 0.4), { res: 160, smooth: false });
      for (const nd of nodes) if (nd.order != null && nd.order < s) {
        const [x0, x1, y0, y1] = nd.bounds;
        const line = nd.f === "x" ? [nd.thr, y0, nd.thr, y1] : [x0, nd.thr, x1, nd.thr];
        splitG.append("line").attr("x1", pl.x(line[0])).attr("y1", pl.y(line[1])).attr("x2", pl.x(line[2])).attr("y2", pl.y(line[3])).attr("stroke", "var(--text)").attr("stroke-width", nd.id === st.cur ? 2.8 : 1.4).attr("opacity", 0.85);
      }
      const curNode = st.cur != null ? nodes[st.cur] : null;
      if (curNode) {
        const [x0, x1, y0, y1] = curNode.bounds;
        regionG.append("rect").attr("x", pl.x(x0)).attr("y", pl.y(y1)).attr("width", pl.x(x1) - pl.x(x0)).attr("height", pl.y(y0) - pl.y(y1)).attr("fill", "var(--series-4)").attr("fill-opacity", 0.06).attr("stroke", "var(--series-4)").attr("stroke-width", 2.4).attr("stroke-dasharray", "6 4");
        if (st.cand) {
          const c = st.cand;
          const line = c.f === "x" ? [c.thr, y0, c.thr, y1] : [x0, c.thr, x1, c.thr];
          splitG.append("line").attr("x1", pl.x(line[0])).attr("y1", pl.y(line[1])).attr("x2", pl.x(line[2])).attr("y2", pl.y(line[3])).attr("stroke", "var(--series-4)").attr("stroke-width", 2).attr("stroke-dasharray", "4 3");
        }
      }
      const inCur = curNode ? new Set(curNode.idx) : null;
      lab.drawPoints({ color: (d) => col(d.label), opacity: (d, i) => (inCur && !inCur.has(i) ? 0.35 : 1) });
      drawTree(nodes, s, st.cur);
      drawCurve(curNode);
      const leaves = nodes.filter((nd) => leafOf(nodes, { x: (nd.bounds[0] + nd.bounds[1]) / 2, y: (nd.bounds[2] + nd.bounds[3]) / 2 }, s) === nd).length;
      lab.readout(`<div class="kv"><span>splits applied</span><b>${Math.min(s, nodes.filter((n) => n.order != null).length)}</b><span>train accuracy</span><b>${((f.m.acc || 0) * 100).toFixed(1)}%</b><span>max depth</span><b>${lab.val("depth")}</b></div>`);
    }
    lab.watch(["depth", "leaf", "crit", "scan", "nt", "mf"], () => retrain());
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    lab.loadPreset("blobs3");
    return lab.destroy;
  }

  MLApp.register({
    id: "decision-tree",
    name: "Decision Tree & Random Forest",
    category: "Supervised - Trees & Ensembles",
    tagline: "Gini splits grow live, bagging + OOB",
    description: "Watch a tree scan every threshold, pick the purest split and recurse, with the tree diagram growing in sync. Switch to a random forest to see bootstrap samples, feature subsampling, out-of-bag error and the vote smoothing the boundary.",
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
