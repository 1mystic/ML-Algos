(() => {
  const { ui, col, fmt } = Lab;
  const STRATS = [
    { id: "eps", name: "ε-greedy" },
    { id: "ucb", name: "UCB1" },
    { id: "ts", name: "Thompson sampling" },
  ];
  // Beta(a, b) sample via two gamma draws (Marsaglia-Tsang)
  function gamma(k) {
    if (k < 1) return gamma(k + 1) * Math.pow(MLU.rng(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) { let x, v; do { x = MLU.randn(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = MLU.rng(); if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
  }
  const beta = (a, b) => { const x = gamma(a), y = gamma(b); return x / (x + y); };
  function betaPdf(x, a, b) { const lb = lgamma(a) + lgamma(b) - lgamma(a + b); return Math.exp((a - 1) * Math.log(Math.max(x, 1e-9)) + (b - 1) * Math.log(Math.max(1 - x, 1e-9)) - lb); }
  function lgamma(z) { const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61503916999185, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]; if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z); z -= 1; let x = c[0]; for (let i = 1; i < g + 2; i++) x += c[i] / (z + i); const t = z + g + 0.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x); }

  function simulate(P) {
    MLU.seedRng(P.seed);
    const K = P.K;
    const best = 0.35 + 0.4 * MLU.rng();
    const probs = Array.from({ length: K }, (_, i) => (i === 0 ? best : Math.max(0.02, best - P.gap * (0.4 + 0.6 * MLU.rng()) - MLU.rng() * 0.15)));
    const order = MLU.shuffle([...probs.keys()]);
    const p = order.map((i) => probs[i]);
    const pStar = Math.max(...p);
    const table = Array.from({ length: P.T }, () => p.map((q) => (MLU.rng() < q ? 1 : 0)));
    const S = STRATS.map(() => ({ n: new Array(K).fill(0), s: new Array(K).fill(0), regret: 0, opt: 0 }));
    const frames = [];
    MLU.seedRng(P.seed + 1);
    for (let t = 0; t < P.T; t++) {
      const pulls = [];
      STRATS.forEach((st, k) => {
        const s = S[k];
        let a;
        if (st.id === "eps") a = MLU.rng() < P.eps ? MLU.randInt(K) : (() => { const m = s.n.map((n, i) => (n ? s.s[i] / n : Infinity)); return m.indexOf(Math.max(...m)); })();
        else if (st.id === "ucb") { const u = s.n.map((n, i) => (n ? s.s[i] / n + P.c * Math.sqrt((2 * Math.log(t + 1)) / n) : Infinity)); a = u.indexOf(Math.max(...u)); }
        else { const th = s.n.map((n, i) => beta(1 + s.s[i], 1 + n - s.s[i])); a = th.indexOf(Math.max(...th)); }
        const r = table[t][a];
        s.n[a]++; s.s[a] += r; s.regret += pStar - p[a]; if (p[a] === pStar) s.opt++;
        pulls.push({ a, r });
      });
      const rec = t < 12 || (t + 1) % Math.max(1, Math.round(P.T / 90)) === 0 || t === P.T - 1;
      if (rec) frames.push({
        phase: t < 12 ? "pull" : "run", major: t === 0 || t === 12, arch: ["est", "sel", "pull", "upd"],
        title: t < 12 ? `Round ${t + 1}: each strategy pulls one arm` : `After ${t + 1} rounds`,
        text: t < 12 ? STRATS.map((st, k) => `${st.name} → arm ${pulls[k].a + 1} (${pulls[k].r ? "win" : "loss"})`).join(" · ") + ". Every strategy sees the same reward if it picks the same arm at the same time, so the comparison is fair." : `Cumulative regret: ${STRATS.map((st, k) => `${st.name} <b>${S[k].regret.toFixed(1)}</b>`).join(", ")}. Regret = reward lost versus always pulling the best arm. Logarithmic growth is the best possible (Lai–Robbins).`,
        tex: t < 12 ? "\\text{UCB1: }a_t=\\arg\\max_a\\Big[\\hat\\mu_a+c\\sqrt{\\tfrac{2\\ln t}{n_a}}\\Big],\\qquad \\text{TS: }\\theta_a\\sim\\mathrm{Beta}(1+s_a,\\,1+n_a-s_a)" : "R_T=T\\mu^\\star-\\sum_{t=1}^{T}\\mu_{a_t}",
        st: { t: t + 1, S: S.map((s) => ({ n: s.n.slice(), s: s.s.slice(), regret: s.regret, opt: s.opt })), pulls },
        m: Object.fromEntries(STRATS.flatMap((st, k) => [[st.id, S[k].regret], [st.id + "_o", S[k].opt / (t + 1)]])),
      });
    }
    return { frames, p };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "bandit", fps: 6, plane: false,
      stageHTML: `<svg data-r="arms" class="lab-svg" viewBox="0 0 720 330"></svg>`,
      legend: STRATS.map((s, i) => `<span class="legend-item"><span class="swatch" style="background:${col(i)}"></span>${s.name}</span>`).join("") + ui.legend([["var(--text)", "true win rate", "dash"]]),
      tools: ui.btn("new", "new bandit", true),
      hint: "bars = each strategy's estimated win rate per arm · numbers = pulls · last pull outlined",
      side: [
        { title: "casino", html: ui.range("K", "arms", 2, 10, 1, 5) + ui.range("gap", "gap to best arm", 0.02, 0.5, 0.02, 0.15) + ui.range("T", "rounds", 100, 2000, 100, 600) + ui.check("reveal", "reveal true win rates", true) },
        { title: "strategies", html: ui.range("eps", "ε for ε-greedy", 0, 0.5, 0.01, 0.1) + ui.range("c", "UCB exploration c", 0.1, 3, 0.1, 1) },
      ],
      arch: {
        w: 380, h: 180,
        nodes: [
          { id: "est", label: "estimates", sub: "μ̂ₐ, nₐ or Beta(α, β)", x: 64, y: 44, kind: "param" },
          { id: "sel", label: "select arm", sub: "explore vs exploit", x: 210, y: 44, kind: "op" },
          { id: "pull", label: "pull", sub: "reward r ∈ {0, 1}", x: 320, y: 110, kind: "data" },
          { id: "upd", label: "update", sub: "counts / posterior", x: 150, y: 120, kind: "op" },
        ],
        edges: [{ from: "est", to: "sel" }, { from: "sel", to: "pull" }, { from: "pull", to: "upd" }, { from: "upd", to: "est" }],
      },
      phases: { pull: { label: "single pulls", color: 3 }, run: { label: "running", color: 0 } },
      panels: [{ id: "beta", title: "Thompson sampling posteriors", sub: "Beta(1 + wins, 1 + losses) per arm" }],
      charts: [{ id: "reg", title: "cumulative regret", series: STRATS.map((s, i) => ({ key: s.id, label: s.name, color: col(i), digits: 1 })) }, { id: "opt", title: "fraction of pulls on the best arm", series: STRATS.map((s, i) => ({ key: s.id + "_o", label: s.name, color: col(i), digits: 2 })), yd: [0, 1] }],
      onFrame: render,
    });
    let seed = 3, sim = null;
    const P = () => ({ K: lab.val("K"), gap: lab.val("gap"), T: lab.val("T"), eps: lab.val("eps"), c: lab.val("c"), seed });
    function run() { sim = simulate(P()); lab.load(sim.frames, { play: true }); }
    const svg = d3.select(lab.R("arms"));
    function render(f) {
      const st = f.st, K = sim.p.length;
      svg.selectAll("*").remove();
      const rowH = 100, x0 = 120, W = 720 - x0 - 10, bw = Math.min(70, W / K - 10);
      STRATS.forEach((sg, k) => {
        const y0 = 12 + k * rowH, s = st.S[k];
        svg.append("text").attr("x", 8).attr("y", y0 + 30).style("font-size", "11px").style("font-weight", 600).attr("fill", col(k)).text(sg.name);
        svg.append("text").attr("x", 8).attr("y", y0 + 46).attr("class", "lbl").text(`regret ${s.regret.toFixed(1)}`);
        svg.append("text").attr("x", 8).attr("y", y0 + 60).attr("class", "lbl").text(`best arm ${(s.opt / st.t * 100).toFixed(0)}%`);
        for (let a = 0; a < K; a++) {
          const bx = x0 + a * (W / K) + (W / K - bw) / 2;
          const mu = s.n[a] ? s.s[a] / s.n[a] : 0;
          const h = 70;
          svg.append("rect").attr("x", bx).attr("y", y0 + 4).attr("width", bw).attr("height", h).attr("fill", "var(--bg-inset)").attr("rx", 3);
          svg.append("rect").attr("x", bx).attr("y", y0 + 4 + h * (1 - mu)).attr("width", bw).attr("height", h * mu).attr("fill", col(k)).attr("opacity", 0.75).attr("rx", 3);
          if (sg.id === "ucb" && s.n[a]) { const bonus = lab.val("c") * Math.sqrt((2 * Math.log(st.t)) / s.n[a]); const top = Math.min(1, mu + bonus); svg.append("line").attr("x1", bx + bw / 2).attr("x2", bx + bw / 2).attr("y1", y0 + 4 + h * (1 - mu)).attr("y2", y0 + 4 + h * (1 - top)).attr("stroke", "var(--text)").attr("stroke-width", 1.4); svg.append("line").attr("x1", bx + bw / 2 - 5).attr("x2", bx + bw / 2 + 5).attr("y1", y0 + 4 + h * (1 - top)).attr("y2", y0 + 4 + h * (1 - top)).attr("stroke", "var(--text)"); }
          if (lab.val("reveal")) svg.append("line").attr("x1", bx - 3).attr("x2", bx + bw + 3).attr("y1", y0 + 4 + h * (1 - sim.p[a])).attr("y2", y0 + 4 + h * (1 - sim.p[a])).attr("stroke", "var(--text)").attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
          if (st.pulls && st.pulls[k].a === a) svg.append("rect").attr("x", bx - 2).attr("y", y0 + 2).attr("width", bw + 4).attr("height", h + 4).attr("fill", "none").attr("stroke", st.pulls[k].r ? "var(--series-3)" : "var(--series-2)").attr("stroke-width", 2.4).attr("rx", 4);
          svg.append("text").attr("x", bx + bw / 2).attr("y", y0 + h + 18).attr("text-anchor", "middle").style("font-size", "9.5px").attr("fill", "var(--text-dim)").text(`${s.n[a]}`);
          if (k === 0) svg.append("text").attr("x", bx + bw / 2).attr("y", 326).attr("text-anchor", "middle").attr("class", "lbl").text(`arm ${a + 1}${lab.val("reveal") ? ` (${(sim.p[a] * 100).toFixed(0)}%)` : ""}`);
        }
      });
      // Thompson posteriors
      const el = lab.R("panel-beta"); el.innerHTML = "";
      const s = st.S[2];
      const xs = d3.range(0.005, 1, 0.01);
      const curves = sim.p.map((_, a) => xs.map((x) => [x, betaPdf(x, 1 + s.s[a], 1 + s.n[a] - s.s[a])]));
      const ym = Math.min(40, Math.max(...curves.flat().map((c) => c[1])));
      const BP = Lab.xy(el, { w: 360, h: 140, xd: [0, 1], yd: [0, ym * 1.05], xLabel: "win rate" });
      curves.forEach((c, a) => BP.line("b" + a, c.map((q) => [q[0], Math.min(ym, q[1])]), { color: col(a + 3), width: 1.8 }));
      lab.readout(`<div class="kv"><span>round</span><b>${st.t}</b><span>best arm</span><b>#${sim.p.indexOf(Math.max(...sim.p)) + 1} (${(Math.max(...sim.p) * 100).toFixed(0)}%)</b>${STRATS.map((sg, k) => `<span>${sg.name} regret</span><b>${st.S[k].regret.toFixed(1)}</b>`).join("")}</div>`);
    }
    lab.watch(["K", "gap", "T", "eps", "c"], () => run());
    lab.watch(["reveal"], () => lab.player.redraw(), { live: true });
    lab.q('[data-b="new"]').onclick = () => { seed = (seed * 53 + 17) % 99991; run(); };
    run();
    return lab.destroy;
  }

  MLApp.register({
    id: "bandits",
    name: "Multi-Armed Bandits",
    category: "Reinforcement Learning",
    tagline: "ε-greedy vs UCB1 vs Thompson sampling",
    isNew: true,
    description: "The purest form of the exploration–exploitation trade-off: slot machines with unknown win rates. Three strategies play the same machines side by side, with estimates, confidence bonuses and Bayesian posteriors updating each round, while cumulative regret shows who wastes the fewest pulls.",
    info: {
      type: "Reinforcement learning - Sequential decision making with a single state. Online learning under partial feedback.",
      scenario: "Repeated choices between options with unknown payoffs where you learn while you earn: A/B/n testing, ad and recommendation selection, clinical trial allocation, hyperparameter search.",
      inputs: "A set of K actions (arms); after each pull only that arm's reward is observed.",
      output: "A policy for choosing arms that maximises total reward (equivalently, minimises regret).",
      intuition: {
        definition: "Every pull either exploits (choose the arm that looks best) or explores (try an arm you are unsure about, in case it is better). Pure exploitation can lock onto a mediocre arm forever; pure exploration wastes pulls. Good strategies explore exactly as much as their uncertainty justifies.",
        steps: ["Keep an estimate (and uncertainty) of each arm's mean reward.", "Choose an arm using a rule that balances estimate and uncertainty.", "Observe the reward and update that arm's estimate.", "Repeat; regret should grow only logarithmically."],
        applications: ["Website A/B/n testing that shifts traffic to winners", "News, ad and product recommendation", "Adaptive clinical trials", "Hyperparameter tuning (Hyperband, successive halving)"],
      },
      math: [
        { title: "Regret", formula: "R_T = Tμ* − Σₜ μ_{aₜ}", note: "Reward lost against always playing the best arm." },
        { title: "Lower bound (Lai–Robbins)", formula: "R_T ≥ Σ_{a: Δₐ>0} (Δₐ / KL(μₐ ‖ μ*)) log T", note: "No algorithm can do better than logarithmic regret on every problem." },
        { title: "ε-greedy", formula: "explore uniformly with prob. ε, else argmax μ̂", note: "Simple, but constant ε gives linear regret: it keeps exploring obviously bad arms." },
        { title: "UCB1", formula: "aₜ = argmax μ̂ₐ + √(2 ln t / nₐ)", note: "Optimism in the face of uncertainty: the bonus shrinks as an arm is tried. O(log T) regret." },
        { title: "Thompson sampling", formula: "θₐ ~ Beta(1 + sₐ, 1 + nₐ − sₐ),  aₜ = argmax θₐ", note: "Sample a plausible world from the posterior and act optimally in it; matches the Lai–Robbins bound asymptotically." },
      ],
      pipeline: [{ label: "Estimates" }, { label: "Select arm", note: "explore / exploit" }, { label: "Pull" }, { label: "Update", accent: "green" }],
      decisionFunction: { text: "aₜ = argmaxₐ score(a)   with score = μ̂ₐ (+ bonus) or a posterior sample", mechanism: "All three strategies are argmax rules; they differ in what they add to the empirical mean: random exploration, a deterministic confidence bonus, or posterior randomness." },
      lossFunction: { text: "minimise expected regret E[R_T]", mechanism: "There is no gradient-based training; the update is just counting. The objective is evaluated through the regret curve." },
      optimization: [{ title: "Incremental means", formula: "μ̂ ← μ̂ + (r − μ̂)/n", note: "Constant memory per arm." }, { title: "Decaying ε", formula: "εₜ ∝ 1/t", note: "Fixes ε-greedy's linear regret but needs tuning." }, { title: "Contextual bandits", formula: "LinUCB, neural Thompson", note: "Use features of the user or situation to share information across arms." }],
      assumptions: [{ name: "Stationary rewards", why: "Estimates assume arm means do not drift.", check: "Use sliding windows or discounted bandits if they do." }, { name: "Independent pulls", why: "No state carries over between rounds.", check: "If actions change future rewards, use full RL." }],
      hyperparameters: [{ name: "ε (ε-greedy)", range: "0.01 - 0.2", increasing: "More exploration, more regret on easy problems.", strategy: "Decay it over time." }, { name: "c (UCB)", range: "0.1 - 2", increasing: "Wider confidence bonus, more exploration.", strategy: "√2 is the theoretical value for rewards in [0, 1]; smaller often works better in practice." }, { name: "prior (Thompson)", range: "Beta(1, 1) …", increasing: "Stronger priors need more evidence to move.", strategy: "Uniform unless you have real prior knowledge." }],
      metrics: ["Cumulative regret", "Fraction of pulls on the best arm", "Total reward"],
      typicalUses: ["Online experiments", "Recommendation", "Resource allocation"],
      diagnostics: ["Regret curve should bend over (log T); a straight line means linear regret.", "Check how many pulls clearly-bad arms still receive."],
      advantages: ["Simple, cheap, online.", "Strong theoretical guarantees (UCB, Thompson).", "Wastes far less traffic than fixed A/B splits."],
      limitations: [{ name: "No state or long-term consequences", note: "", fix: "full RL (Q-learning)." }, { name: "Non-stationarity breaks estimates", note: "", fix: "discounted or sliding-window bandits." }, { name: "Many arms explore slowly", note: "", fix: "contextual / structured bandits." }],
      alternatives: [{ name: "Classical A/B testing", when: "You need clean statistical inference more than reward during the test." }, { name: "Q-learning", when: "Actions change the state and future rewards." }, { name: "Contextual bandits", when: "The best arm depends on observable context." }],
      pitfalls: [{ problem: "Greedy locks onto a bad arm", solution: "It never explores; use UCB or Thompson." }, { problem: "Constant ε wastes traffic forever", solution: "Decay ε or switch method." }, { problem: "Rewards drift over time", solution: "Discount old observations." }],
      quickRef: [{ name: "Regret", formula: "Tμ* − Σ μ_{aₜ}" }, { name: "UCB1", formula: "μ̂ + √(2 ln t / n)" }, { name: "Thompson", formula: "argmax θ, θ ~ Beta(1+s, 1+f)" }, { name: "Best rate", formula: "O(log T)" }],
      code: `import numpy as np
K, T = 5, 1000
wins, pulls = np.zeros(K), np.zeros(K)
for t in range(T):
    theta = np.random.beta(1 + wins, 1 + pulls - wins)   # Thompson sampling
    a = int(np.argmax(theta))
    r = np.random.rand() < true_p[a]
    pulls[a] += 1; wins[a] += r`,
      whyChain: [
        { q: "Why does constant ε-greedy have linear regret?", a: "It keeps choosing a random arm a fixed fraction ε of the time forever, including arms it knows are bad. Each of those pulls costs a constant expected amount, so regret grows proportionally to T." },
        { q: "Why does UCB's optimism work?", a: "An arm is pulled either because its estimate is truly high (exploitation) or because it has been tried so rarely that its bonus is large (exploration). Each pull shrinks the bonus, so bad arms are revisited only about log T times." },
        { q: "Why is Thompson sampling so effective?", a: "It explores in proportion to the probability that each arm is the best. Clearly bad arms almost never win a posterior draw, while genuinely uncertain ones do, which matches exploration to real uncertainty." },
      ],
      workedExample: {
        setup: "UCB1 at t = 100. Arm A: 60 pulls, 30 wins. Arm B: 5 pulls, 2 wins.",
        steps: ["A: 0.50 + √(2·ln 100 / 60) = 0.50 + 0.39 = 0.89.", "B: 0.40 + √(2·ln 100 / 5) = 0.40 + 1.36 = 1.76.", "UCB pulls B: its estimate is lower but far less certain."],
        result: "UCB explores arm B (score 1.76 vs 0.89)",
      },
    },
    mount,
  });
})();
