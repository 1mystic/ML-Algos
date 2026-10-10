(() => {
  const { ui, col, fmt } = Lab;

  function bestStump(pts, y, w) {
    let best = null;
    for (const f of ["x", "y"]) {
      const order = pts.map((_, i) => i).sort((a, b) => pts[a][f] - pts[b][f]);
      // error of "predict +1 when value > thr" as thr sweeps upward
      let errPos = 0; for (let i = 0; i < pts.length; i++) if (y[i] === -1) errPos += w[i]; // thr below all: everything predicted +1
      const consider = (thr, e) => {
        if (!best || e < best.err) best = { f, thr, s: 1, err: e };
        if (1 - e < best.err) best = { f, thr, s: -1, err: 1 - e };
      };
      consider(pts[order[0]][f] - 0.5, errPos);
      for (let k = 0; k < order.length; k++) {
        const i = order[k];
        errPos += y[i] === 1 ? w[i] : -w[i]; // point i now predicted -1
        const v0 = pts[i][f], v1 = k + 1 < order.length ? pts[order[k + 1]][f] : v0 + 1;
        if (v1 - v0 < 1e-9) continue;
        consider((v0 + v1) / 2, errPos);
      }
    }
    return best;
  }
  const stumpPred = (st, p) => (p[st.f] > st.thr ? st.s : -st.s);

  function buildFrames(pts, T) {
    const n = pts.length, y = pts.map((p) => (p.label === 1 ? 1 : -1));
    if (!y.includes(1) || !y.includes(-1)) return [{ phase: "idle", title: "Need both classes", text: "Add points of each class.", st: { stumps: [], w: [] } }];
    let w = new Array(n).fill(1 / n);
    const stumps = [];
    const F = new Array(n).fill(0);
    const frames = [{ phase: "init", major: true, arch: ["w"], title: "Every point starts with equal weight 1/n", text: "AdaBoost trains a sequence of weak learners (here: one-split decision stumps). After each one it re-weights the data so the next learner focuses on what is still wrong. Dot size = weight.", tex: "w_i^{(1)}=\\tfrac1n", st: { stumps: [], w: w.slice() }, m: {} }];
    for (let t = 1; t <= T; t++) {
      const st = bestStump(pts, y, w);
      const err = Math.min(Math.max(st.err, 1e-10), 1 - 1e-10);
      const alpha = 0.5 * Math.log((1 - err) / err);
      const stump = { ...st, alpha };
      const wrong = pts.map((p, i) => stumpPred(stump, p) !== y[i]);
      frames.push({ phase: "fit", arch: ["w", "stump", "eps"], title: `Round ${t}: best stump ${st.f} ${st.s > 0 ? ">" : "≤"} ${st.thr.toFixed(2)} → class +1`, text: `Among all axis-aligned single splits, this one has the lowest <b>weighted</b> error ε = <b>${err.toFixed(3)}</b>. It only has to beat a coin flip (ε < 0.5) on the current weights. Its say in the final vote: α = ½ ln((1−ε)/ε) = <b>${alpha.toFixed(3)}</b>.`, tex: "\\varepsilon_t=\\sum_i w_i\\,\\mathbb 1[h_t(x_i)\\ne y_i],\\qquad \\alpha_t=\\tfrac12\\ln\\frac{1-\\varepsilon_t}{\\varepsilon_t}", st: { stumps: stumps.slice(), w: w.slice(), cand: stump, wrong }, m: { eps: err, err: trainErr(F, y) } });
      stumps.push(stump);
      let Z = 0;
      for (let i = 0; i < n; i++) { F[i] += alpha * stumpPred(stump, pts[i]); w[i] *= Math.exp(-alpha * y[i] * stumpPred(stump, pts[i])); Z += w[i]; }
      w = w.map((v) => v / Z);
      const expLoss = F.reduce((s, f, i) => s + Math.exp(-y[i] * f), 0) / n;
      frames.push({ phase: "reweight", major: t === 1, arch: ["alpha", "rew", "ens"], title: `Round ${t}: re-weight and add to the ensemble`, text: `Misclassified points are multiplied by e^{α} = ${Math.exp(alpha).toFixed(2)}, correct ones by e^{−α} = ${Math.exp(-alpha).toFixed(2)}, then everything is renormalised. The ensemble F = Σ αₜhₜ now has training error <b>${(trainErr(F, y) * 100).toFixed(1)}%</b>.`, tex: "w_i\\leftarrow\\frac{w_i\\,e^{-\\alpha_t y_i h_t(x_i)}}{Z_t},\\qquad F(x)=\\sum_{s\\le t}\\alpha_s h_s(x)", st: { stumps: stumps.slice(), w: w.slice(), wrong }, m: { eps: err, err: trainErr(F, y), exp: expLoss } });
      if (err < 1e-9) break;
    }
    const last = frames[frames.length - 1];
    frames.push({ ...last, phase: "done", major: true, arch: ["ens", "out"], title: `Final ensemble of ${stumps.length} stumps`, text: `Training error <b>${(last.m.err * 100).toFixed(1)}%</b>. Stumps alone can only cut along one axis, yet their weighted sum builds a staircase boundary around rings and moons. Training error typically keeps falling long after test error flattens, because the margins keep growing.`, st: { stumps: stumps.slice(), w: last.st.w } });
    return frames;
  }
  function trainErr(F, y) { return F.reduce((s, f, i) => s + ((f >= 0 ? 1 : -1) !== y[i]), 0) / F.length; }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "ada", fps: 2.5, classes: 2,
      legend: ui.legend([["var(--series-4)", "current stump", "line"], ["var(--text)", "ensemble boundary", "line"]]),
      tools: ui.btn("retrain", "retrain", true),
      hint: "dot area ∝ weight · click adds the selected class · drag / double-click to edit",
      data: { kind: "class", presets: ["circles", "moons", "xor", "linear", "overlap", "spiral"], def: "circles", n: 80, noise: 0.2 },
      side: [{ title: "boosting", html: ui.range("T", "rounds T", 1, 80, 1, 30) + ui.check("ens", "show ensemble field", true) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "w", label: "weights wᵢ", sub: "start 1/n", x: 50, y: 40, kind: "param" },
          { id: "stump", label: "fit stump hₜ", sub: "min weighted error", x: 172, y: 40, kind: "op", info: "A depth-1 tree: a single threshold on a single feature." },
          { id: "eps", label: "εₜ", sub: "weighted error", x: 290, y: 40, kind: "loss" },
          { id: "alpha", label: "αₜ", sub: "½ln((1−ε)/ε)", x: 310, y: 112, kind: "param" },
          { id: "rew", label: "re-weight", sub: "wᵢ·e^(−αyh)", x: 172, y: 112, kind: "op" },
          { id: "ens", label: "F = Σαₜhₜ", sub: "ensemble", x: 160, y: 168, kind: "op" },
          { id: "out", label: "sign F", x: 300, y: 168, w: 64, kind: "out" },
        ],
        edges: [{ from: "w", to: "stump" }, { from: "stump", to: "eps" }, { from: "eps", to: "alpha" }, { from: "alpha", to: "rew" }, { from: "rew", to: "w", dash: true, label: "next round" }, { from: "alpha", to: "ens" }, { from: "ens", to: "out" }],
      },
      phases: { init: { label: "init", color: 3 }, fit: { label: "fit stump", color: 0 }, reweight: { label: "re-weight", color: 1 }, done: { label: "done", color: 2 } },
      panels: [{ id: "alpha", title: "stump weights αₜ" }],
      charts: [{ id: "err", title: "errors per round", series: [{ key: "err", label: "ensemble train error", color: "var(--series-1)" }, { key: "eps", label: "stump ε (weighted)", color: "var(--series-2)", dash: "4 3" }, { key: "exp", label: "exp-loss", color: "var(--series-4)" }] }],
      onFrame: render, onData: () => retrain(),
    });
    lab.editable({ onChange: () => retrain({ at: "end" }) });
    const pl = lab.plane, L = pl.layers;
    function retrain({ at = "start" } = {}) { lab.load(buildFrames(lab.points, lab.val("T")), { at, play: at === "start" }); }
    const stumpG = L.mid.append("g");
    const aBars = Lab.bars(lab.R("panel-alpha"), { w: 360, h: 130 });
    function render(f) {
      const st = f.st, n = lab.points.length;
      const F = (x, y) => st.stumps.reduce((s, h) => s + h.alpha * stumpPred(h, { x, y }), 0);
      if (lab.val("ens") && st.stumps.length) {
        const grid = Lab.sampleGrid(pl, F, 100);
        const sc = Math.max(0.5, st.stumps.reduce((s, h) => s + h.alpha, 0) * 0.6);
        Lab.paintGrid(L.bg, pl, grid, (v) => Lab.divergeColor(v, sc, 0.5), { smooth: false });
        Lab.contour(L.under, pl, grid, [0], { cls: "iso-main", width: 2 });
      } else { L.bg.selectAll("*").remove(); L.under.selectAll("*").remove(); }
      stumpG.selectAll("*").remove();
      const cur = st.cand || null;
      if (cur) {
        const p0 = cur.f === "x" ? [cur.thr, -10] : [-10, cur.thr], p1 = cur.f === "x" ? [cur.thr, 10] : [10, cur.thr];
        stumpG.append("line").attr("x1", pl.x(p0[0])).attr("y1", pl.y(p0[1])).attr("x2", pl.x(p1[0])).attr("y2", pl.y(p1[1])).attr("stroke", "var(--series-4)").attr("stroke-width", 3).attr("stroke-dasharray", "8 5");
        // shade the side predicted +1
        const r = cur.f === "x" ? (cur.s > 0 ? [cur.thr, 10, -10, 10] : [-10, cur.thr, -10, 10]) : (cur.s > 0 ? [-10, 10, cur.thr, 10] : [-10, 10, -10, cur.thr]);
        stumpG.append("rect").attr("x", pl.x(r[0])).attr("y", pl.y(r[3])).attr("width", pl.x(r[1]) - pl.x(r[0])).attr("height", pl.y(r[2]) - pl.y(r[3])).attr("fill", col(1)).attr("fill-opacity", 0.1);
      }
      const wmax = Math.max(...st.w, 1 / Math.max(1, n));
      lab.drawPoints({
        color: (d) => col(d.label), r: (d, i) => (st.w.length ? 2 + 11 * Math.sqrt(st.w[i] / wmax) : 4.5),
        stroke: (d, i) => (st.wrong && st.wrong[i] ? "var(--text)" : "var(--bg)"), sw: (d, i) => (st.wrong && st.wrong[i] ? 2 : 1), opacity: 0.88,
      });
      const shown = st.stumps.slice(-24);
      aBars.set(shown.map((h, k) => ({ label: String(st.stumps.length - shown.length + k + 1), value: h.alpha, color: h.f === "x" ? col(0) : col(4) })), { values: false });
      lab.readout(`<div class="kv"><span>stumps</span><b>${st.stumps.length}</b><span>ensemble error</span><b>${f.m && f.m.err != null ? (f.m.err * 100).toFixed(1) + "%" : "–"}</b><span>max weight</span><b>${st.w.length ? wmax.toFixed(3) : "–"}</b></div><div class="note">bars: blue = split on x, violet = split on y</div>`);
    }
    lab.watch(["T"], () => retrain());
    lab.watch(["ens"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    lab.loadPreset("circles");
    return lab.destroy;
  }

  MLApp.register({
    id: "adaboost",
    name: "AdaBoost",
    category: "Supervised - Trees & Ensembles",
    tagline: "re-weighted decision stumps",
    isNew: true,
    description: "The original boosting algorithm: fit a one-split stump, give it a vote proportional to how good it is, then up-weight the points it got wrong so the next stump focuses on them. Watch dots swell as they become hard, and a staircase boundary emerge from axis-aligned cuts.",
    info: {
      type: "Supervised - Classification (binary; SAMME extends to multiclass). Sequential ensemble of weak learners.",
      scenario: "Clean tabular data where a simple, strong baseline ensemble is wanted; historically the Viola-Jones face detector.",
      inputs: "Feature vectors with labels y ∈ {−1, +1}; a weak learner family (usually decision stumps).",
      output: "A weighted vote F(x) = Σ αₜ hₜ(x); prediction sign(F(x)).",
      intuition: {
        definition: "Train a weak model, see which examples it gets wrong, make those examples heavier, and train the next weak model on the re-weighted data. Each model gets a vote weighted by its accuracy. Many barely-better-than-chance learners add up to a strong one.",
        steps: ["Start with equal weights on all points.", "Fit the weak learner that minimises weighted error ε.", "Give it vote α = ½ ln((1−ε)/ε).", "Multiply weights of misclassified points by e^α, correct ones by e^−α, renormalise.", "Repeat T rounds; predict with the sign of the weighted vote."],
        applications: ["Viola-Jones real-time face detection (boosted Haar features)", "Baseline classifier on tabular data", "Ranking (RankBoost) and cost-sensitive classification", "A conceptual stepping stone to gradient boosting"],
      },
      math: [
        { title: "Weighted error", formula: "εₜ = Σᵢ wᵢ 𝟙[hₜ(xᵢ) ≠ yᵢ]", note: "Weights sum to 1, so ε is between 0 and 1. A weak learner only needs ε < 0.5." },
        { title: "Learner weight", formula: "αₜ = ½ ln((1 − εₜ)/εₜ)", note: "Zero at ε = 0.5 (useless), large as ε → 0, negative if worse than chance (its vote is flipped)." },
        { title: "Weight update", formula: "wᵢ ← wᵢ exp(−αₜ yᵢ hₜ(xᵢ)) / Zₜ", note: "After the update, the current learner has exactly 50% weighted error: the next learner must find something new." },
        { title: "Exponential loss", formula: "L(F) = Σᵢ exp(−yᵢ F(xᵢ))", note: "AdaBoost is forward stagewise additive modelling under this loss; α is the exact line-search step." },
        { title: "Training error bound", formula: "err ≤ Πₜ 2√(εₜ(1−εₜ)) ≤ exp(−2 Σₜ γₜ²)", note: "With γₜ = 0.5 − εₜ, training error falls exponentially fast in the number of rounds." },
        { title: "Margin theory", formula: "margin(xᵢ) = yᵢ F(xᵢ) / Σ αₜ", note: "Rounds after zero training error still increase margins, which explains why test error often keeps improving." },
      ],
      pipeline: [{ label: "Equal weights" }, { label: "Fit stump", note: "min weighted error" }, { label: "α from ε" }, { label: "Re-weight", note: "boost mistakes" }, { label: "Weighted vote", note: "sign Σαh", accent: "green" }],
      decisionFunction: { text: "ŷ = sign( Σₜ αₜ hₜ(x) )", mechanism: "Each stump votes ±1 on its side of a threshold; votes are weighted by α and summed. The boundary is a union of axis-aligned pieces, so it looks like a staircase." },
      lossFunction: { text: "L = Σᵢ exp(−yᵢ F(xᵢ))", mechanism: "The exponential loss punishes confident mistakes very heavily, which is why AdaBoost is sensitive to label noise and outliers: a mislabelled point's weight grows exponentially.", plot: { fn: (m) => Math.exp(-m), domain: [-2, 3], fn2: (m) => Math.log(1 + Math.exp(-m)) / Math.LN2, yDomain: [0, 5], caption: "exponential loss (solid) vs logistic loss (dashed) by margin y·F(x): the exponential grows much faster for wrong-side points" } },
      optimization: [
        { title: "Forward stagewise", formula: "Fₜ = Fₜ₋₁ + αₜhₜ, earlier terms frozen", note: "Greedy: never revisits past learners." },
        { title: "Exact step size", formula: "α minimises Σ wᵢ exp(−α yᵢ hₜ(xᵢ))", note: "Setting the derivative to zero gives the ½ ln((1−ε)/ε) formula." },
        { title: "Shrinkage", formula: "F += ν αₜ hₜ, ν < 1", note: "A learning rate trades more rounds for better generalisation, as in gradient boosting." },
      ],
      assumptions: [
        { name: "Weak learnability", why: "Each learner must beat chance on the weighted data.", check: "If ε hits 0.5, boosting stalls." },
        { name: "Low label noise", why: "Mislabelled points get exponentially large weights and dominate later rounds.", check: "Inspect points with the largest final weights: they are often errors." },
      ],
      hyperparameters: [
        { name: "n_estimators (T)", range: "50 - 1000", increasing: "Lower training error; test error usually plateaus, can rise with noise.", strategy: "Tune with validation, together with the learning rate." },
        { name: "learning_rate (ν)", range: "0.01 - 1", increasing: "Bigger steps per learner, fewer rounds needed.", strategy: "Smaller ν with more rounds generalises better." },
        { name: "base estimator depth", range: "1 - 3", increasing: "Stronger learners capture interactions, overfit sooner.", strategy: "Stumps (depth 1) are the classic choice; depth 2-3 for interactions." },
      ],
      metrics: ["Accuracy / F1", "Training vs validation error per round", "Margin distribution"],
      typicalUses: ["Tabular classification baselines", "Feature-selection-by-boosting (each stump picks one feature)", "Object detection cascades"],
      diagnostics: ["Plot training and validation error against rounds.", "List the points with the highest weights: likely label noise or genuine hard cases.", "Check the distribution of selected features across stumps."],
      advantages: ["Simple, few hyperparameters.", "Turns any weak learner into a strong one.", "Implicit feature selection with stumps.", "Strong theory: exponential decrease of training error, margin bounds."],
      limitations: [{ name: "Sensitive to noise and outliers", note: "exponential loss lets bad points dominate", fix: "gradient boosting with logistic or Huber loss." }, { name: "Sequential", note: "cannot train learners in parallel", fix: "random forests if parallelism matters." }, { name: "Superseded on tabular data", note: "gradient boosting libraries are usually stronger", fix: "XGBoost / LightGBM." }],
      alternatives: [{ name: "Gradient boosting", when: "You want any differentiable loss and stronger performance." }, { name: "Random forest", when: "Noisy labels, or you want parallel training." }],
      pitfalls: [{ problem: "Validation error rises with more rounds", solution: "Label noise. Lower the learning rate, cap rounds, or switch loss." }, { problem: "ε reaches 0 in round 1", solution: "The base learner is too strong; boosting adds nothing. Use stumps." }, { problem: "Huge weight on a few points", solution: "Check those labels." }],
      quickRef: [{ name: "Error", formula: "ε = Σ w 𝟙[h ≠ y]" }, { name: "Vote", formula: "α = ½ ln((1−ε)/ε)" }, { name: "Update", formula: "w ← w e^(−αyh) / Z" }, { name: "Predict", formula: "sign Σ αh" }, { name: "Loss", formula: "Σ e^(−yF)" }],
      code: `from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),   # stumps
    n_estimators=300, learning_rate=0.5, random_state=0)
ada.fit(X_train, y_train)

# staged predictions show test error round by round
errs = [1 - (p == y_test).mean() for p in ada.staged_predict(X_test)]`,
      whyChain: [
        { q: "Why does each re-weighting make the last learner exactly 50% wrong?", a: "The update multiplies correct weights by e^−α and wrong ones by e^α. With α = ½ ln((1−ε)/ε), the total weight on the wrong points becomes ε·e^α = √(ε(1−ε)), equal to the total on correct points. So the next learner cannot reuse the same split." },
        { q: "Why can test error keep dropping after training error reaches zero?", a: "Additional rounds increase the margins yᵢF(xᵢ)/Σα of training points. Generalisation bounds depend on the margin distribution, not the number of learners." },
        { q: "How is AdaBoost related to gradient boosting?", a: "AdaBoost is gradient boosting with the exponential loss: the re-weighting is the gradient of that loss, and α is an exact line search. Gradient boosting generalises the idea to any differentiable loss." },
        { q: "Why are stumps enough?", a: "Each stump only needs to beat chance on the current weights. Adding many axis-aligned stumps builds an additive model that can approximate complex boundaries, although with depth-1 stumps it cannot represent feature interactions directly." },
      ],
      workedExample: {
        setup: "4 points, weights 0.25 each. The first stump misclassifies one point.",
        steps: ["ε = 0.25, so α = ½ ln(0.75/0.25) = ½ ln 3 ≈ 0.549.", "Wrong point: 0.25·e^0.549 ≈ 0.433. Correct points: 0.25·e^−0.549 ≈ 0.144 each.", "Z = 0.433 + 3·0.144 = 0.866.", "New weights: wrong ≈ 0.5, correct ≈ 0.167 each."],
        result: "The misclassified point now carries half of all the weight",
      },
    },
    mount,
  });
})();
