(() => {
  const { ui, col, fmt } = Lab;
  const S = 9;

  function buildFrames(pts, { eta, epochs, pocket, order }) {
    const y = pts.map((p) => (p.label === 1 ? 1 : -1));
    if (!y.includes(1) || !y.includes(-1)) return [{ phase: "idle", title: "Need both classes", text: "Add points of each class (shift-click adds class 1).", st: { w: [0, 0, 0] } }];
    let w = [0, 0, 0]; // [b, w1, w2] on scaled inputs
    let best = w.slice(), bestErr = Infinity;
    const errs = (v) => pts.reduce((s, p, i) => s + (y[i] * (v[0] + v[1] * p.x / S + v[2] * p.y / S) <= 0), 0);
    const frames = [{ phase: "init", major: true, arch: ["x", "sum"], title: "Start with w = 0", text: "The perceptron is a single linear threshold unit. It learns online: visit points one at a time and only change anything when it gets one wrong.", tex: "\\hat y=\\operatorname{sign}(w^\\top x+b)", st: { w: w.slice() }, m: { mistakes: errs(w) } }];
    let total = 0;
    for (let e = 1; e <= epochs; e++) {
      let mistakes = 0, okRun = 0;
      const idx = order === "shuffle" ? MLU.shuffle([...pts.keys()]) : [...pts.keys()];
      for (const i of idx) {
        const p = pts[i];
        const a = w[0] + w[1] * p.x / S + w[2] * p.y / S;
        if (y[i] * a <= 0) {
          const prev = w.slice();
          w = [w[0] + eta * y[i], w[1] + eta * y[i] * p.x / S, w[2] + eta * y[i] * p.y / S];
          mistakes++; total++;
          const E = errs(w);
          if (E < bestErr) { bestErr = E; best = w.slice(); }
          if (frames.length < 260) frames.push({
            phase: "update", arch: ["sum", "sign", "cmp", "upd"],
            title: `Epoch ${e}: point #${i} is misclassified → update`,
            text: `${okRun ? `${okRun} point${okRun > 1 ? "s" : ""} before it were already correct (no change). ` : ""}This one has y = ${y[i] > 0 ? "+1" : "−1"} but w·x + b = ${a.toFixed(2)}. Add ${y[i] > 0 ? "" : "minus "}η·x to w: the weight vector (arrow) swings toward the point, rotating the boundary. Mistakes this epoch: <b>${mistakes}</b>.`,
            tex: "w\\leftarrow w+\\eta\\,y_i\\,x_i,\\qquad b\\leftarrow b+\\eta\\,y_i",
            st: { w: w.slice(), prev, cur: i }, m: { mistakes: E },
          });
          okRun = 0;
        } else okRun++;
      }
      frames.push({
        phase: mistakes ? "epoch" : "done", major: true, arch: mistakes ? ["cmp"] : ["sign", "out"],
        title: mistakes ? `End of epoch ${e}: ${mistakes} mistake${mistakes > 1 ? "s" : ""}` : `Converged in epoch ${e}: a full pass with zero mistakes`,
        text: mistakes ? "Another pass over the data. If the classes are linearly separable, the convergence theorem guarantees this stops after at most (R/γ)² updates." : `Every point is on the right side. Total updates: <b>${total}</b>. Note the boundary is just <i>a</i> separator, not the maximum-margin one an SVM would find.`,
        st: { w: w.slice() }, m: { mistakes: errs(w) },
      });
      if (!mistakes) break;
    }
    const last = frames[frames.length - 1];
    if (last.phase !== "done") {
      frames.push({ phase: "done", major: true, arch: ["out"], title: pocket ? `No convergence: keep the pocket weights (${bestErr} errors)` : "No convergence after the epoch budget", text: pocket ? "The data is not linearly separable, so the perceptron would cycle forever. The pocket algorithm keeps the best weights seen so far and returns those." : "The data is not linearly separable, so updates never stop and the boundary keeps jumping. Turn on the pocket variant, or use logistic regression / an SVM with a soft margin.", st: { w: pocket ? best : w.slice() }, m: { mistakes: pocket ? bestErr : errs(w) } });
    }
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "perceptron", fps: 4, classes: 2,
      legend: ui.legend([["var(--accent)", "weight vector w", "line"], ["var(--series-4)", "point being corrected"]]),
      tools: ui.btn("retrain", "retrain", true),
      hint: "click adds the selected class · drag / double-click to edit · try the overlapping data to see it never settle",
      data: { kind: "class", presets: ["linear", "blobs2", "overlap", "xor", "circles"], def: "linear", n: 50, noise: 0.25 },
      side: [{ title: "training", html: ui.range("eta", "learning rate η", 0.1, 2, 0.1, 1) + ui.range("ep", "max epochs", 1, 40, 1, 15) + ui.select("order", "visit order", [["fixed", "fixed"], ["shuffle", "shuffle each epoch"]], "fixed") + ui.check("pocket", "pocket algorithm (keep best)", true) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "x", label: "x", sub: "(x₁, x₂, 1)", x: 40, y: 46, w: 58, kind: "data" },
          { id: "sum", label: "Σ wⱼxⱼ + b", sub: "weighted sum", x: 140, y: 46, kind: "op" },
          { id: "sign", label: "sign", sub: "step activation", x: 250, y: 46, kind: "op", info: "A hard threshold. Non-differentiable, which is why the perceptron uses a mistake-driven rule rather than gradient descent." },
          { id: "out", label: "ŷ ∈ {±1}", x: 344, y: 46, w: 64, kind: "out" },
          { id: "cmp", label: "ŷ = y ?", sub: "compare", x: 300, y: 145, kind: "loss" },
          { id: "upd", label: "w += η y x", sub: "only on mistakes", x: 140, y: 145, kind: "param" },
        ],
        edges: [{ from: "x", to: "sum" }, { from: "sum", to: "sign" }, { from: "sign", to: "out" }, { from: "sign", to: "cmp" }, { from: "cmp", to: "upd", label: "no" }, { from: "upd", to: "sum", dash: true, label: "w" }],
      },
      phases: { init: { label: "init", color: 3 }, update: { label: "mistake → update", color: 1 }, epoch: { label: "epoch end", color: 4 }, done: { label: "done", color: 2 } },
      charts: [{ id: "m", title: "training errors after each update", series: [{ key: "mistakes", label: "misclassified", color: "var(--series-2)", digits: 0 }] }],
      panels: [{ id: "w", title: "weights" }],
      onFrame: render, onData: () => retrain(),
    });
    lab.editable({ onChange: () => retrain({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    const params = () => ({ eta: lab.val("eta"), epochs: lab.val("ep"), pocket: lab.val("pocket"), order: lab.val("order") });
    function retrain({ at = "start" } = {}) { MLU.seedRng(3); lab.load(buildFrames(lab.points, params()), { at, play: at === "start" }); }
    const arrowG = L.top.append("g"), curG = L.top.append("g");
    const defs = pl.svg.append("defs");
    defs.append("marker").attr("id", "pc-arr").attr("viewBox", "0 0 10 10").attr("refX", 8).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto").append("path").attr("d", "M0,0L10,5L0,10z").attr("fill", "var(--accent)");
    const wBars = Lab.bars(lab.R("panel-w"), { w: 360, h: 120, signed: true });

    function render(f) {
      const w = f.st.w;
      const act = (x, y) => w[0] + w[1] * x / S + w[2] * y / S;
      if (w[1] || w[2] || w[0]) {
        const grid = Lab.sampleGrid(pl, act, 60);
        Lab.paintGrid(L.bg, pl, grid, (v) => Lab.classColor(v >= 0 ? 1 : 0, 1, 0.14), { smooth: false });
        Lab.contour(L.under, pl, grid, [0], { cls: "iso-main", width: 2.2 });
      } else { L.bg.selectAll("*").remove(); L.under.selectAll("*").remove(); }
      if (f.st.prev && (f.st.prev[1] || f.st.prev[2])) {
        const pg = Lab.sampleGrid(pl, (x, y) => f.st.prev[0] + f.st.prev[1] * x / S + f.st.prev[2] * y / S, 60);
        Lab.contour(L.under, pl, pg, [0], { cls: "iso-prev", width: 1.2, dash: "4 4", stroke: "var(--text-faint)" });
      } else L.under.selectAll("path.iso-prev").remove();
      lab.drawPoints({ color: (d) => col(d.label), stroke: (d, i) => (i === f.st.cur ? "var(--series-4)" : "var(--bg)"), sw: (d, i) => (i === f.st.cur ? 3 : 1.1), r: (d, i) => (i === f.st.cur ? 7 : 4.6) });
      // weight vector drawn from the boundary point closest to the origin
      arrowG.selectAll("*").remove();
      const nrm = Math.hypot(w[1], w[2]);
      if (nrm > 1e-9) {
        const ux = w[1] / nrm, uy = w[2] / nrm;
        const d0 = (-w[0] / nrm) * S; // signed distance of boundary from origin (data units)
        const bx = ux * d0, by = uy * d0;
        arrowG.append("line").attr("x1", pl.x(bx)).attr("y1", pl.y(by)).attr("x2", pl.x(bx + ux * 3.2)).attr("y2", pl.y(by + uy * 3.2)).attr("stroke", "var(--accent)").attr("stroke-width", 2.4).attr("marker-end", "url(#pc-arr)");
        arrowG.append("text").attr("class", "lbl-strong halo").attr("x", pl.x(bx + ux * 3.6)).attr("y", pl.y(by + uy * 3.6)).text("w");
      }
      wBars.set([{ label: "b", value: w[0] }, { label: "w₁", value: w[1] }, { label: "w₂", value: w[2] }].map((d) => ({ ...d, color: d.value >= 0 ? col(1) : col(0) })));
      const err = lab.points.reduce((s, p) => s + ((act(p.x, p.y) >= 0 ? 1 : 0) !== p.label), 0);
      lab.readout(`<div class="kv"><span>weights</span><b>(${fmt(w[1], 2)}, ${fmt(w[2], 2)})</b><span>bias</span><b>${fmt(w[0], 2)}</b><span>training errors</span><b>${err} / ${lab.points.length}</b></div>`);
    }
    lab.watch(["eta", "ep", "pocket", "order"], () => retrain());
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    lab.loadPreset("linear");
    return lab.destroy;
  }

  MLApp.register({
    id: "perceptron",
    name: "Perceptron",
    category: "Supervised - Classification",
    tagline: "mistake-driven linear threshold unit",
    isNew: true,
    description: "Rosenblatt's 1958 single neuron: a weighted sum and a hard threshold, trained by nudging the weights only when it makes a mistake. Watch the weight vector swing toward each misclassified point, and see why it never settles on overlapping data.",
    info: {
      type: "Supervised - Binary classification. Linear, online, mistake-driven; the ancestor of every neural network.",
      scenario: "Linearly separable data where a fast, simple online learner is enough, or as the teaching model for neural networks and online learning theory.",
      inputs: "Feature vectors x ∈ ℝᵈ with labels y ∈ {−1, +1}.",
      output: "A weight vector w and bias b; prediction is sign(w·x + b).",
      intuition: {
        definition: "A perceptron draws a line (hyperplane) and checks which side each point lands on. When it gets a point wrong it tilts the line toward that point by adding the point (times its label) to the weights. If a perfect line exists, this provably finds one in a finite number of mistakes.",
        steps: ["Start with all weights at zero.", "Visit training points one by one.", "If a point is on the wrong side, add η·y·x to w and η·y to b.", "Repeat passes until a full pass makes no mistakes."],
        applications: ["Historical foundation of neural networks (Mark I Perceptron, 1958)", "Online learning for streaming, high-dimensional sparse data", "Averaged and structured perceptrons for NLP tagging and parsing", "A building block for understanding margins, kernels and SGD"],
      },
      math: [
        { title: "Decision rule", formula: "ŷ = sign(w·x + b)", note: "A linear threshold unit: the boundary is the hyperplane w·x + b = 0, and w is its normal vector." },
        { title: "Update rule", formula: "if y(w·x + b) ≤ 0:  w ← w + η y x,  b ← b + η y", note: "Correct points cause no change. A mistake adds the point to w, increasing y(w·x) by η‖x‖² for that point." },
        { title: "Perceptron criterion", formula: "L(w) = Σᵢ max(0, −yᵢ(w·xᵢ + b))", note: "The update is exactly stochastic subgradient descent on this loss. It is zero for every correctly classified point, however close to the boundary." },
        { title: "Convergence theorem (Novikoff 1962)", formula: "#mistakes ≤ (R / γ)²", note: "R bounds ‖x‖ and γ is the best achievable margin. The bound does not depend on the number of points or the dimension." },
        { title: "Non-separable data", formula: "updates never stop; w cycles", note: "Without a perfect separator there is always a misclassified point. The pocket algorithm keeps the best w seen; the averaged perceptron averages all of them." },
        { title: "Dual / kernel form", formula: "w = Σᵢ αᵢ yᵢ xᵢ,  f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x)", note: "αᵢ counts how many times point i triggered an update, which makes the kernel perceptron possible." },
      ],
      pipeline: [{ label: "Init w = 0" }, { label: "Visit point", note: "online" }, { label: "Wrong side?", note: "y(w·x+b) ≤ 0" }, { label: "w += ηyx" }, { label: "Pass with 0 mistakes", note: "stop", accent: "green" }],
      decisionFunction: { text: "ŷ = sign(w·x + b)", mechanism: "Points on the side w points toward are +1, the rest −1. Only the direction of w matters for classification; its length only matters relative to b.", plot: { fn: (z) => (z >= 0 ? 1 : -1), domain: [-3, 3], yDomain: [-1.4, 1.4], caption: "the step activation: no gradient anywhere, so learning is driven by mistakes instead" } },
      lossFunction: { text: "L(w) = Σᵢ max(0, −yᵢ(w·xᵢ + b))", mechanism: "The perceptron criterion. Its subgradient at a misclassified point is −yᵢxᵢ, so a unit-step SGD update reproduces the classic rule. Unlike hinge loss there is no margin term, so the solution can sit arbitrarily close to points.", plot: { fn: (m) => Math.max(0, -m), domain: [-3, 3], fn2: (m) => Math.max(0, 1 - m), caption: "perceptron loss max(0, −m) (solid) vs SVM hinge max(0, 1 − m) (dashed), as functions of the margin m = y·f(x)" } },
      optimization: [
        { title: "Online updates", formula: "one point at a time", note: "No learning rate tuning is needed for separable data with w₀ = 0: scaling η just scales w, which does not change the boundary." },
        { title: "Pocket algorithm", formula: "keep w* with the lowest training error", note: "Makes the perceptron usable on non-separable data at the cost of an error evaluation per update." },
        { title: "Averaged perceptron", formula: "w̄ = (1/T) Σₜ wₜ", note: "Averaging all intermediate weight vectors approximates a large-margin solution and generalises much better in practice." },
        { title: "Order matters", formula: "shuffle each epoch", note: "Different visiting orders give different final separators. Shuffling avoids systematic cycles." },
      ],
      assumptions: [
        { name: "Linear separability (for convergence)", why: "The guarantee only holds if a perfect hyperplane exists.", check: "If training error never reaches zero, the data is not separable in this feature space." },
        { name: "Binary labels in {−1, +1}", why: "The update uses y as a sign.", check: "Encode classes accordingly; use one-vs-rest for multiclass." },
        { name: "Comparable feature scales", why: "The update adds raw x, so large-scale features dominate the direction of w.", check: "Standardise features." },
      ],
      hyperparameters: [
        { name: "η (learning rate)", range: "0.01 - 1", increasing: "Larger steps; with w₀ = 0 on separable data it only rescales w.", strategy: "Leave at 1 unless you start from non-zero weights." },
        { name: "max epochs", range: "5 - 1000", increasing: "More chances to converge; on non-separable data just more cycling.", strategy: "Set a cap and use pocket or averaging." },
        { name: "shuffle", range: "on / off", increasing: "Not applicable", strategy: "Shuffle each epoch to avoid order artefacts." },
        { name: "averaging / pocket", range: "on / off", increasing: "Not applicable", strategy: "Always use one of them on real data." },
      ],
      metrics: ["Accuracy", "Number of mistakes per epoch", "Margin of the final separator"],
      typicalUses: ["Teaching and theory of online learning", "Large sparse linear problems (with averaging)", "Structured prediction (structured perceptron)"],
      diagnostics: ["Plot mistakes per epoch: zero means separable and converged; a noisy plateau means not separable.", "Compare the final boundary with an SVM: the perceptron's can sit very close to some points.", "Check the weight norm: it grows without bound on non-separable data."],
      advantages: ["Extremely simple and fast; one multiply-add per feature per point.", "Online: learns from streams with constant memory.", "Provable finite convergence on separable data.", "Kernelisable via its dual form."],
      limitations: [{ name: "Linear only", note: "cannot solve XOR", fix: "feature maps, kernels, or a multilayer network." }, { name: "Never converges on non-separable data", note: "updates cycle forever", fix: "pocket, averaging, or a soft-margin loss." }, { name: "No probabilities", note: "outputs only a sign", fix: "logistic regression." }, { name: "Arbitrary separator", note: "no margin maximisation", fix: "SVM or averaged perceptron." }],
      alternatives: [{ name: "Logistic regression", when: "You want probabilities and a smooth, always-convergent objective." }, { name: "Linear SVM", when: "You want the maximum-margin separator." }, { name: "MLP", when: "The boundary is non-linear." }],
      pitfalls: [{ problem: "Training never stops", solution: "The data is not separable. Cap epochs and use the pocket or averaged variant." }, { problem: "Boundary jumps around between runs", solution: "Order dependence. Shuffle and average." }, { problem: "One feature dominates", solution: "Standardise the inputs." }],
      quickRef: [{ name: "Predict", formula: "sign(w·x + b)" }, { name: "Update", formula: "w += ηyx, b += ηy on mistakes" }, { name: "Loss", formula: "max(0, −y f(x))" }, { name: "Mistake bound", formula: "(R/γ)²" }],
      code: `import numpy as np
from sklearn.linear_model import Perceptron

clf = Perceptron(eta0=1.0, max_iter=50, shuffle=True, random_state=0)
clf.fit(X, y)

# From scratch
w, b = np.zeros(X.shape[1]), 0.0
for epoch in range(50):
    mistakes = 0
    for xi, yi in zip(X, y):            # yi in {-1, +1}
        if yi * (xi @ w + b) <= 0:
            w += yi * xi; b += yi; mistakes += 1
    if mistakes == 0:
        break`,
      whyChain: [
        { q: "Why does the update move the boundary the right way?", a: "After w ← w + yx, the new score on that point is y(w + yx)·x = y w·x + ‖x‖², so it increases by ‖x‖² > 0. The point is pushed toward the correct side." },
        { q: "Why is there no learning-rate tuning?", a: "Starting from zero, every w is a sum of η·y·x terms, so η just scales w by a constant. The sign of w·x, and hence every prediction, is unchanged." },
        { q: "Why can't it solve XOR?", a: "XOR is not linearly separable: no single line puts (0,0),(1,1) on one side and (0,1),(1,0) on the other. Minsky and Papert's 1969 analysis of this stalled neural network research for a decade." },
        { q: "How is the perceptron related to SGD?", a: "Its update is stochastic subgradient descent with step 1 on max(0, −y f(x)). Replace that with the hinge loss max(0, 1 − y f(x)) plus L2 and you get the SVM; replace it with log-loss and you get logistic regression." },
      ],
      workedExample: {
        setup: "Points x₁=(2,1) y=+1, x₂=(−1,−2) y=−1. Start w=(0,0), b=0, η=1.",
        steps: ["x₁: score 0 ≤ 0 → mistake. w = (2,1), b = 1.", "x₂: score 2(−1)+1(−2)+1 = −3, y·score = 3 > 0 → correct.", "Second pass: x₁ score 2·2+1·1+1 = 6 > 0 correct; x₂ correct.", "A full pass with no mistakes: converged."],
        result: "w = (2, 1), b = 1 after a single update",
      },
    },
    mount,
  });
})();
