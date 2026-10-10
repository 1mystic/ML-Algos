(() => {
  const { ui, col, fmt } = Lab;
  const SURF = {
    bowl: { name: "ill-conditioned bowl", xd: [-10, 10], yd: [-6, 6], f: (x, y) => 0.04 * x * x + 2 * y * y, g: (x, y) => [0.08 * x, 4 * y], start: [-9, 3], opt: [[0, 0]], lr: [0.2, 0.6] },
    rosen: { name: "Rosenbrock banana", xd: [-2, 2], yd: [-1, 3], f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2, g: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)], start: [-1.6, 2.4], opt: [[1, 1]], lr: [0.0009, 0.08] },
    saddle: { name: "saddle point", xd: [-4, 4], yd: [-4, 4], f: (x, y) => x * x - y * y + 0.06 * y ** 4, g: (x, y) => [2 * x, -2 * y + 0.24 * y ** 3], start: [-3.5, 0.02], opt: [[0, 4.08], [0, -4.08]], lr: [0.05, 0.12] },
    himmel: { name: "Himmelblau (4 minima)", xd: [-5, 5], yd: [-5, 5], f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2, g: (x, y) => [4 * x * (x * x + y - 11) + 2 * (x + y * y - 7), 2 * (x * x + y - 11) + 4 * y * (x + y * y - 7)], start: [-0.3, -0.9], opt: [[3, 2], [-2.805, 3.131], [-3.779, -3.283], [3.584, -1.848]], lr: [0.006, 0.15] },
    bumpy: { name: "bumpy (many local minima)", xd: [-6, 6], yd: [-6, 6], f: (x, y) => 0.08 * (x * x + y * y) + 1.4 * (2 - Math.cos(2 * x) - Math.cos(2 * y)), g: (x, y) => [0.16 * x + 2.8 * Math.sin(2 * x), 0.16 * y + 2.8 * Math.sin(2 * y)], start: [5.2, 4.6], opt: [[0, 0]], lr: [0.08, 0.3] },
  };
  const OPTS = [
    { id: "sgd", name: "SGD", adaptive: false, tex: "\\theta\\leftarrow\\theta-\\eta g" },
    { id: "mom", name: "Momentum", adaptive: false, tex: "v\\leftarrow\\beta v+g,\\;\\theta\\leftarrow\\theta-\\eta v" },
    { id: "nag", name: "Nesterov", adaptive: false, tex: "v\\leftarrow\\beta v+\\nabla f(\\theta-\\eta\\beta v),\\;\\theta\\leftarrow\\theta-\\eta v" },
    { id: "adagrad", name: "AdaGrad", adaptive: true, tex: "G\\leftarrow G+g^2,\\;\\theta\\leftarrow\\theta-\\eta\\,g/\\sqrt{G}" },
    { id: "rmsprop", name: "RMSProp", adaptive: true, tex: "s\\leftarrow\\rho s+(1-\\rho)g^2,\\;\\theta\\leftarrow\\theta-\\eta\\,g/\\sqrt{s}" },
    { id: "adam", name: "Adam", adaptive: true, tex: "\\theta\\leftarrow\\theta-\\eta\\,\\hat m/(\\sqrt{\\hat v}+\\epsilon)" },
  ];

  function run(S, opt, start, T, lrMul, noise, beta) {
    let th = start.slice(), v = [0, 0], m = [0, 0], s = [0, 0], G = [0, 0];
    const lr = (opt.adaptive ? S.lr[1] : S.lr[0]) * lrMul;
    const path = [th.slice()];
    for (let t = 1; t <= T; t++) {
      const look = opt.id === "nag" ? [th[0] - lr * beta * v[0], th[1] - lr * beta * v[1]] : th;
      let g = S.g(look[0], look[1]);
      if (noise) g = g.map((q) => q + MLU.randn() * noise * (1 + Math.abs(q) * 0.2));
      g = g.map((q) => Math.max(-1e4, Math.min(1e4, q)));
      for (let d = 0; d < 2; d++) {
        if (opt.id === "sgd") th[d] -= lr * g[d];
        else if (opt.id === "mom" || opt.id === "nag") { v[d] = beta * v[d] + g[d]; th[d] -= lr * v[d]; }
        else if (opt.id === "adagrad") { G[d] += g[d] * g[d]; th[d] -= (lr * g[d]) / (Math.sqrt(G[d]) + 1e-8); }
        else if (opt.id === "rmsprop") { s[d] = 0.9 * s[d] + 0.1 * g[d] * g[d]; th[d] -= (lr * g[d]) / (Math.sqrt(s[d]) + 1e-8); }
        else { m[d] = 0.9 * m[d] + 0.1 * g[d]; s[d] = 0.999 * s[d] + 0.001 * g[d] * g[d]; const mh = m[d] / (1 - 0.9 ** t), vh = s[d] / (1 - 0.999 ** t); th[d] -= (lr * mh) / (Math.sqrt(vh) + 1e-8); }
      }
      if (!isFinite(th[0]) || !isFinite(th[1]) || Math.abs(th[0]) > 1e6) { th = [NaN, NaN]; }
      path.push(th.slice());
    }
    return path;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "optim", fps: 14, plane: { h: 420 },
      legend: OPTS.map((o, i) => `<span class="legend-item"><span class="swatch" style="background:${col(i)}"></span>${o.name}</span>`).join("") + ui.legend([["var(--series-3)", "minimum ★"]]),
      tools: ui.btn("race", "race again", true),
      hint: "click anywhere on the surface to move the starting point · contours are log-spaced",
      side: [
        { title: "loss surface", html: ui.select("surf", "function", Object.entries(SURF).map(([k, v]) => [k, v.name]), "rosen") + ui.range("T", "iterations", 50, 600, 10, 300) },
        { title: "optimisers", html: OPTS.map((o, i) => `<label class="check"><input type="checkbox" data-o="${o.id}" checked /> <span class="swatch" style="background:${col(i)}"></span> ${o.name}</label>`).join("") + ui.range("lr", "learning-rate multiplier (log₁₀)", -1, 1, 0.1, 0, "1.0") + ui.range("beta", "momentum β", 0, 0.99, 0.01, 0.9) + ui.range("noise", "gradient noise (stochastic)", 0, 3, 0.1, 0) },
      ],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "th", label: "θₜ", sub: "parameters", x: 40, y: 50, w: 60, kind: "param" },
          { id: "g", label: "g = ∇f(θ)", sub: "+ noise if stochastic", x: 158, y: 50, kind: "op" },
          { id: "m", label: "1st moment", sub: "momentum / m", x: 290, y: 50, kind: "op", tex: "m\\leftarrow\\beta_1 m+(1-\\beta_1)g" },
          { id: "v", label: "2nd moment", sub: "Σg² / RMS", x: 290, y: 120, kind: "op", tex: "v\\leftarrow\\beta_2 v+(1-\\beta_2)g^2" },
          { id: "upd", label: "update rule", sub: "step = η · direction", x: 158, y: 120, kind: "loss" },
          { id: "next", label: "θₜ₊₁", x: 40, y: 120, w: 60, kind: "out" },
        ],
        edges: [{ from: "th", to: "g" }, { from: "g", to: "m" }, { from: "g", to: "v" }, { from: "m", to: "upd" }, { from: "v", to: "upd" }, { from: "upd", to: "next" }, { from: "next", to: "th", dash: true, label: "repeat" }],
      },
      phases: { start: { label: "start", color: 3 }, run: { label: "descending", color: 0 }, done: { label: "finished", color: 2 } },
      panels: [{ id: "tab", title: "scoreboard" }],
      charts: [{ id: "loss", title: "loss f(θ) per iteration (log scale)", custom: true }],
      onFrame: render,
    });
    const pl = lab.plane, L = pl.layers;
    let S = SURF.rosen, start = S.start.slice(), paths = {}, fmin = 0;
    const active = () => OPTS.filter((o) => lab.q(`[data-o="${o.id}"]`).checked);
    function surface() {
      S = SURF[lab.val("surf")];
      start = S.start.slice();
      pl.setDomain(S.xd, S.yd);
      fmin = Math.min(...S.opt.map((o) => S.f(o[0], o[1])));
      const grid = Lab.sampleGrid(pl, (x, y) => Math.log(S.f(x, y) - fmin + 1e-3), 120);
      L.bg.selectAll("*").remove(); L.under.selectAll("*").remove();
      const lo = Math.min(...grid.values), hi = Math.max(...grid.values);
      Lab.paintGrid(L.bg, pl, grid, (v) => { const t = (v - lo) / (hi - lo); const P = Lab.palRGB(); const a = P[4], b = P[0]; return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, 0.08 + 0.2 * (1 - t)]; });
      Lab.contour(L.under, pl, grid, d3.range(lo, hi, (hi - lo) / 22), { cls: "fc", width: 0.8, stroke: "var(--text-faint)" });
      L.under.selectAll("path.fc").attr("stroke-opacity", 0.5);
      race();
    }
    function race() {
      const T = lab.val("T");
      MLU.seedRng(4);
      paths = {};
      for (const o of OPTS) paths[o.id] = run(S, o, start, T, Math.pow(10, lab.val("lr")), lab.val("noise"), lab.val("beta"));
      const every = Math.max(1, Math.round(T / 100));
      const frames = [];
      for (let t = 0; t <= T; t += every) {
        const act = active();
        const best = act.map((o) => ({ o, l: S.f(...paths[o.id][t]) })).filter((d) => isFinite(d.l)).sort((a, b) => a.l - b.l)[0];
        frames.push({ phase: t === 0 ? "start" : t >= T - every + 1 ? "done" : "run", major: t === 0, arch: t === 0 ? ["th"] : ["th", "g", "m", "v", "upd", "next"], title: t === 0 ? "All optimisers start from the same point" : `Iteration ${t}`, text: t === 0 ? "Same surface, same start, same number of steps. The only difference is how each method turns the gradient into a step." : `Leader: <b>${best ? best.o.name : "–"}</b> at f = ${best ? fmt(best.l, 4) : "–"}. ${S === SURF.bowl ? "Plain SGD zig-zags across the narrow valley; momentum and adaptive methods rescale the steep direction." : S === SURF.saddle ? "Near the saddle the gradient is tiny: methods that normalise by gradient size (RMSProp, Adam) escape fastest." : S === SURF.rosen ? "The banana valley is easy to find and slow to follow; momentum carries speed along the curved floor." : "Each method lands in a basin determined by its own dynamics."}`, tex: best ? best.o.tex : "g=\\nabla f(\\theta)", st: { t } });
      }
      lab.load(frames, { play: true });
      drawChart();
    }
    let CP = null;
    function drawChart() {
      const el = lab.charts.loss.el; el.innerHTML = "";
      const T = lab.val("T");
      const vals = [];
      for (const o of active()) for (const p of paths[o.id]) { const l = S.f(p[0], p[1]) - fmin; if (isFinite(l) && l > 0) vals.push(l); }
      const f0 = Math.max(1e-3, S.f(start[0], start[1]) - fmin);
      CP = Lab.xy(el, { w: 360, h: 140, xd: [0, T], yd: [Math.max(1e-6, Math.min(...vals, 1)), Math.min(Math.max(...vals, 1e-3), f0 * 30)], logY: true, xLabel: "iteration" });
      const cap = CP.y.domain()[1];
      OPTS.forEach((o, i) => { if (active().includes(o)) CP.line(o.id, paths[o.id].map((p, t) => [t, Math.min(cap, Math.max(1e-6, S.f(p[0], p[1]) - fmin))]), { color: col(i), width: 1.8, curve: d3.curveLinear }); });
    }
    const pathG = L.mid.append("g"), dotG = L.top.append("g"), startG = L.top.append("g"), optG = L.top.append("g");
    function render(f) {
      const t = f.st.t;
      pathG.selectAll("*").remove(); dotG.selectAll("*").remove(); startG.selectAll("*").remove(); optG.selectAll("*").remove();
      for (const o of S.opt) optG.append("path").attr("d", d3.symbol(d3.symbolStar, 160)()).attr("transform", `translate(${pl.x(o[0])},${pl.y(o[1])})`).attr("fill", "var(--series-3)").attr("stroke", "var(--bg)");
      startG.append("circle").attr("cx", pl.x(start[0])).attr("cy", pl.y(start[1])).attr("r", 7).attr("fill", "none").attr("stroke", "var(--text)").attr("stroke-width", 2);
      const rows = [];
      OPTS.forEach((o, i) => {
        if (!active().includes(o)) return;
        const p = paths[o.id].slice(0, t + 1).filter((q) => isFinite(q[0]));
        pathG.append("path").attr("d", d3.line().x((q) => pl.x(Math.max(S.xd[0] - 5, Math.min(S.xd[1] + 5, q[0])))).y((q) => pl.y(Math.max(S.yd[0] - 5, Math.min(S.yd[1] + 5, q[1]))))(p)).attr("fill", "none").attr("stroke", col(i)).attr("stroke-width", 2).attr("stroke-opacity", 0.9);
        const cur = paths[o.id][t];
        if (isFinite(cur[0])) dotG.append("circle").attr("cx", pl.x(cur[0])).attr("cy", pl.y(cur[1])).attr("r", 5.5).attr("fill", col(i)).attr("stroke", "var(--bg)").attr("stroke-width", 1.5);
        const l = S.f(cur[0], cur[1]);
        const d = Math.min(...S.opt.map((q) => Math.hypot(q[0] - cur[0], q[1] - cur[1])));
        rows.push({ o, i, l, d });
      });
      if (CP) CP.vline(t);
      rows.sort((a, b) => (isFinite(a.l) ? a.l : Infinity) - (isFinite(b.l) ? b.l : Infinity));
      lab.R("panel-tab").innerHTML = `<table class="mini-table"><tr><th>#</th><th>optimiser</th><th>f(θ)</th><th>dist. to min</th></tr>${rows.map((r, k) => `<tr><td>${k + 1}</td><td><span class="swatch" style="background:${col(r.i)}"></span> ${r.o.name}</td><td>${isFinite(r.l) ? fmt(r.l, 4) : "diverged"}</td><td>${isFinite(r.d) ? fmt(r.d, 3) : "–"}</td></tr>`).join("")}</table>`;
      lab.readout(`<div class="kv"><span>surface</span><b>${S.name}</b><span>start</span><b>(${start[0].toFixed(2)}, ${start[1].toFixed(2)})</b><span>iteration</span><b>${t}</b><span>base η (SGD / adaptive)</span><b>${S.lr[0]} / ${S.lr[1]}</b></div>`);
    }
    pl.svg.on("click.start", (ev) => {
      const [px, py] = d3.pointer(ev, pl.svg.node());
      if (!pl.inside(px, py)) return;
      start = [pl.x.invert(px), pl.y.invert(py)];
      race();
    });
    lab.watch(["surf"], () => surface());
    lab.watch(["T", "lr", "beta", "noise"], () => race(), { format: { lr: (v) => Math.pow(10, v).toFixed(2), noise: (v) => (+v).toFixed(1) } });
    lab.q('[data-r="stage"]').closest(".lab").querySelectorAll("[data-o]").forEach((c) => c.addEventListener("change", () => { drawChart(); lab.player.redraw(); }));
    lab.q('[data-b="race"]').onclick = () => race();
    surface();
    return lab.destroy;
  }

  MLApp.register({
    id: "optimizers",
    name: "Gradient Descent Optimisers",
    category: "Optimization",
    tagline: "SGD · Momentum · Nesterov · AdaGrad · RMSProp · Adam race",
    isNew: true,
    description: "The engines inside every model's training loop, racing on the same loss surface. Watch plain SGD zig-zag down a narrow valley, momentum carry speed around the Rosenbrock banana, adaptive methods escape a saddle point, and noise turn descent stochastic. Click anywhere to change the starting point.",
    info: {
      type: "Optimisation - First-order iterative methods for minimising differentiable loss functions; the training algorithms of machine learning.",
      scenario: "Any model trained by gradients: neural networks, logistic regression, matrix factorisation, embeddings.",
      inputs: "A differentiable objective f(θ), its gradient (often a noisy minibatch estimate), a starting point θ₀ and hyperparameters (learning rate, momentum, decay rates).",
      output: "A sequence θ₀, θ₁, … converging toward a (local) minimum.",
      intuition: {
        definition: "Gradient descent repeatedly steps downhill. Its variants differ in how they turn a gradient into a step: momentum remembers past directions like a heavy ball, adaptive methods give each parameter its own step size based on how large its gradients have been, and Adam does both.",
        steps: ["Compute the gradient g at the current parameters (possibly from a minibatch).", "Update running statistics (velocity, squared-gradient averages).", "Compute the step from those statistics and the learning rate.", "Move the parameters and repeat."],
        applications: ["Training deep networks (Adam / AdamW, SGD with momentum)", "Large-scale linear models (SGD)", "Sparse features and embeddings (AdaGrad)", "Recurrent networks (RMSProp historically)"],
      },
      math: [
        { title: "Gradient descent", formula: "θₜ₊₁ = θₜ − η ∇f(θₜ)", note: "Converges for convex smooth f when η < 2/L (L = largest curvature)." },
        { title: "Momentum (heavy ball)", formula: "vₜ = βvₜ₋₁ + gₜ,  θₜ₊₁ = θₜ − ηvₜ", note: "Averages gradients over ~1/(1−β) steps, damping oscillation across valleys and accelerating along them." },
        { title: "Nesterov accelerated gradient", formula: "vₜ = βvₜ₋₁ + ∇f(θₜ − ηβvₜ₋₁)", note: "Evaluates the gradient at the look-ahead point; optimal O(1/t²) rate on smooth convex problems." },
        { title: "AdaGrad", formula: "Gₜ = Gₜ₋₁ + gₜ²,  θ −= η gₜ / √Gₜ", note: "Per-parameter step sizes; great for sparse features, but steps shrink forever." },
        { title: "RMSProp", formula: "sₜ = ρsₜ₋₁ + (1−ρ)gₜ²,  θ −= η gₜ / √sₜ", note: "Exponential average instead of a sum, so the step size stops decaying." },
        { title: "Adam", formula: "m = β₁m + (1−β₁)g,  v = β₂v + (1−β₂)g²,  θ −= η m̂/(√v̂ + ε)", note: "Momentum plus RMSProp with bias correction for the zero initialisation of m and v." },
      ],
      pipeline: [{ label: "Gradient g" }, { label: "Moments m, v" }, { label: "Step direction" }, { label: "× η" }, { label: "Update θ", accent: "green" }],
      decisionFunction: { text: "θₜ₊₁ = θₜ − η · direction(gₜ, history)", mechanism: "SGD uses g itself; momentum a running average of g; adaptive methods divide by a running RMS of g, making steps roughly scale-invariant per parameter." },
      lossFunction: { text: "minimise f(θ)   (in ML: f = average training loss over examples)", mechanism: "The optimiser does not change the objective, only the path taken to reduce it. Different paths can end in different minima with different generalisation." },
      optimization: [{ title: "Condition number", formula: "κ = L / μ", note: "Plain GD needs O(κ log 1/ε) steps; momentum O(√κ log 1/ε). The bowl surface has κ = 50." }, { title: "Learning-rate schedules", formula: "warmup, step decay, cosine", note: "Large steps early for progress, small steps late for precision." }, { title: "Weight decay", formula: "AdamW: θ −= ηλθ separately", note: "Decoupled from the adaptive scaling, unlike L2 inside Adam." }],
      assumptions: [{ name: "Differentiable objective", why: "Gradients drive every method here.", check: "Use subgradients or smoothing for kinks." }, { name: "Reasonable scaling", why: "Plain SGD is sensitive to feature scales; adaptive methods less so.", check: "Normalise inputs anyway." }],
      hyperparameters: [{ name: "learning rate η", range: "1e-5 - 1", increasing: "Faster, then oscillation and divergence.", strategy: "The single most important knob: LR range test, then a schedule." }, { name: "momentum β / β₁", range: "0.8 - 0.99", increasing: "Smoother, faster in valleys, overshoots more.", strategy: "0.9 default." }, { name: "β₂ (Adam)", range: "0.98 - 0.9999", increasing: "Slower-adapting step sizes.", strategy: "0.999 default; 0.95 for some large models." }, { name: "ε", range: "1e-8 - 1e-6", increasing: "Damps adaptive scaling.", strategy: "Rarely tuned." }, { name: "batch size", range: "16 - 4096", increasing: "Less gradient noise.", strategy: "Scale η with batch size." }],
      metrics: ["Training loss vs steps and vs wall-clock", "Validation loss (generalisation)", "Gradient norm"],
      typicalUses: ["Neural network training", "Online learning", "Any differentiable fitting problem"],
      diagnostics: ["Loss oscillates or explodes: lower η.", "Loss plateaus early: raise η or add momentum.", "Adam trains fast but validates worse: try SGD + momentum or AdamW with tuned decay."],
      advantages: ["Cheap per step: only first derivatives.", "Scale to billions of parameters.", "Work with noisy minibatch gradients.", "Adaptive methods need little tuning."],
      limitations: [{ name: "Local minima and saddles", note: "non-convex objectives", fix: "momentum, noise, restarts." }, { name: "Sensitive to learning rate", note: "", fix: "schedules, adaptive methods." }, { name: "Ill-conditioning slows convergence", note: "zig-zagging", fix: "momentum, preconditioning, second-order methods." }],
      alternatives: [{ name: "Newton / L-BFGS", when: "Small to medium deterministic problems where curvature information pays off." }, { name: "Coordinate descent", when: "L1-regularised linear models (see Lasso)." }, { name: "Evolution strategies", when: "Gradients unavailable." }],
      pitfalls: [{ problem: "Using L2 regularisation inside Adam", solution: "Use AdamW (decoupled weight decay)." }, { problem: "No learning-rate warmup on transformers", solution: "Adaptive estimates are poor early; warm up." }, { problem: "Comparing optimisers at one learning rate", solution: "Tune η for each; defaults favour different scales." }],
      quickRef: [{ name: "SGD", formula: "θ −= ηg" }, { name: "Momentum", formula: "v = βv + g; θ −= ηv" }, { name: "RMSProp", formula: "θ −= ηg/√s" }, { name: "Adam", formula: "θ −= η m̂/(√v̂ + ε)" }],
      code: `import torch
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=10_000)
for x, y in loader:
    loss = loss_fn(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()`,
      whyChain: [
        { q: "Why does SGD zig-zag in narrow valleys?", a: "The gradient points mostly across the valley (steep direction) rather than along it. A learning rate small enough to be stable across the valley is tiny along it, so progress alternates side to side." },
        { q: "Why does momentum help?", a: "Across the valley successive gradients alternate in sign and cancel in the running average; along the valley they agree and accumulate. The effective step grows in the useful direction and shrinks in the oscillating one." },
        { q: "Why does Adam need bias correction?", a: "m and v start at zero, so early averages are biased toward zero, especially v with β₂ = 0.999. Dividing by 1 − βᵗ removes that bias so early steps are not tiny or erratic." },
        { q: "Why do adaptive methods escape saddles faster?", a: "Near a saddle gradients are small; dividing by their RMS rescales them to a normal step size, so the iterate keeps moving instead of crawling." },
      ],
      workedExample: {
        setup: "f(θ) = θ², start θ = 4, η = 0.1.",
        steps: ["Gradient 2θ = 8; GD step to 4 − 0.8 = 3.2.", "Next gradient 6.4; step to 2.56. Each step multiplies θ by 0.8.", "With η = 1.1 the factor is 1 − 2.2 = −1.2: |θ| grows and GD diverges.", "Adam's first step is ≈ η·sign(g) regardless of the gradient size."],
        result: "GD converges geometrically when |1 − 2η| < 1, i.e. η < 1",
      },
    },
    mount,
  });
})();
