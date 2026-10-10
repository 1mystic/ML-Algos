(() => {
  const { ui, col, fmt } = Lab;
  const C = 9, R = 6;
  const ACTS = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // up, right, down, left (dx, dy) with y down
  const ANAME = ["↑", "→", "↓", "←"];
  const LAYOUTS = {
    cliff: () => { const g = blank(); g[R - 1][0] = "S"; g[R - 1][C - 1] = "G"; for (let c = 1; c < C - 1; c++) g[R - 1][c] = "P"; return g; },
    maze: () => { const g = blank(); g[0][0] = "S"; g[R - 1][C - 1] = "G"; for (let r = 0; r < 4; r++) g[r][2] = "W"; for (let r = 2; r < R; r++) g[r][5] = "W"; g[2][7] = "P"; g[4][3] = "P"; return g; },
    open: () => { const g = blank(); g[2][0] = "S"; g[2][C - 1] = "G"; g[1][4] = "P"; g[3][4] = "P"; g[2][4] = "W"; return g; },
  };
  function blank() { return Array.from({ length: R }, () => new Array(C).fill(".")); }
  const REW = { G: 10, P: -10, ".": -0.1, S: -0.1 };

  function simulate(grid, P) {
    let start = [0, 0];
    grid.forEach((row, y) => row.forEach((v, x) => { if (v === "S") start = [x, y]; }));
    const Q = Array.from({ length: R }, () => Array.from({ length: C }, () => [0, 0, 0, 0]));
    const frames = [], returns = [];
    const step = (x, y, a) => {
      if (MLU.rng() < P.slip) a = (a + (MLU.rng() < 0.5 ? 1 : 3)) % 4;
      let nx = x + ACTS[a][0], ny = y + ACTS[a][1];
      if (nx < 0 || ny < 0 || nx >= C || ny >= R || grid[ny][nx] === "W") { nx = x; ny = y; }
      const cell = grid[ny][nx];
      return { nx, ny, r: REW[cell] ?? -0.1, done: cell === "G" || cell === "P" };
    };
    const greedy = (q) => { const m = Math.max(...q); const best = [0, 1, 2, 3].filter((a) => q[a] === m); return best[MLU.randInt(best.length)]; };
    const pick = (x, y, eps) => (MLU.rng() < eps ? MLU.randInt(4) : greedy(Q[y][x]));
    const snapQ = () => Q.map((row) => row.map((q) => q.slice()));
    const detailEps = 2;
    for (let ep = 1; ep <= P.episodes; ep++) {
      const eps = Math.max(0.02, P.eps * Math.pow(P.decay, ep - 1));
      let [x, y] = start, a = pick(x, y, eps), G = 0, path = [[x, y]], n = 0;
      while (n < 120) {
        const { nx, ny, r, done } = step(x, y, a);
        const a2 = pick(nx, ny, eps);
        const target = done ? r : r + P.gamma * (P.algo === "sarsa" ? Q[ny][nx][a2] : Math.max(...Q[ny][nx]));
        const old = Q[y][x][a];
        Q[y][x][a] += P.alpha * (target - old);
        G += r; n++;
        path.push([nx, ny]);
        if (ep <= detailEps && n <= 40) frames.push({ phase: "step", major: ep === 1 && n === 1, arch: ["s", "pol", "env", "td", "upd"], title: `Episode ${ep}, step ${n}: ${ANAME[a]} from (${x},${y}) → (${nx},${ny}), reward ${r}`, text: `TD target = ${done ? `r = ${r} (terminal)` : `r + γ·${P.algo === "sarsa" ? `Q(s′, a′=${ANAME[a2]})` : "max Q(s′,·)"} = ${r} + ${P.gamma}·${fmt((target - r) / P.gamma, 3)} = ${fmt(target, 3)}`}. Q(s, ${ANAME[a]}) moves from ${fmt(old, 3)} to <b>${fmt(Q[y][x][a], 3)}</b>. ε = ${eps.toFixed(2)}: early on the agent mostly explores at random.`, tex: P.algo === "sarsa" ? "Q(s,a)\\leftarrow Q(s,a)+\\alpha\\big[r+\\gamma Q(s',a')-Q(s,a)\\big]" : "Q(s,a)\\leftarrow Q(s,a)+\\alpha\\big[r+\\gamma\\max_{a'}Q(s',a')-Q(s,a)\\big]", st: { Q: snapQ(), agent: [nx, ny], path: path.slice(), upd: [x, y, a] }, m: {} });
        x = nx; y = ny; a = a2;
        if (done) break;
      }
      returns.push(G);
      const show = ep > detailEps && (ep <= 12 || ep % Math.max(1, Math.round(P.episodes / 60)) === 0 || ep === P.episodes);
      const m = { ret: G, avg: MLU.mean(returns.slice(-10)), len: n };
      if (show || ep <= detailEps) frames.push({ phase: "episode", major: ep === detailEps + 1, arch: ["upd", "s"], title: `Episode ${ep} finished: return ${G.toFixed(1)} in ${n} steps`, text: `ε decays to ${eps.toFixed(2)}. Arrows show the current greedy policy (best action per cell); triangle colours show each action's Q-value. Value spreads backwards from the goal one experience at a time.`, tex: "V(s)=\\max_a Q(s,a),\\qquad \\pi(s)=\\arg\\max_a Q(s,a)", st: { Q: snapQ(), agent: [x, y], path, eps }, m });
    }
    // greedy rollout
    let [x, y] = start; const path = [[x, y]];
    for (let n = 0; n < 40; n++) {
      const a = greedy(Q[y][x]);
      let nx = x + ACTS[a][0], ny = y + ACTS[a][1];
      if (nx < 0 || ny < 0 || nx >= C || ny >= R || grid[ny][nx] === "W") { nx = x; ny = y; }
      x = nx; y = ny; path.push([x, y]);
      frames.push({ phase: "greedy", arch: ["pol", "env"], title: `Greedy rollout, step ${n + 1}`, text: "Exploration off: the agent follows argmax Q. " + (P.algo === "sarsa" ? "SARSA learns the value of the policy it actually follows (with exploration), so it prefers a safer route away from pits." : "Q-learning learns the optimal greedy policy, so it hugs the shortest route even past the pits."), st: { Q: snapQ(), agent: [x, y], path: path.slice(), greedy: true }, m: {} });
      if (grid[y][x] === "G" || grid[y][x] === "P") break;
    }
    return { frames, returns };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "ql", fps: 6, plane: false,
      stageHTML: `<svg data-r="grid" class="lab-svg" viewBox="0 0 640 430"></svg>`,
      legend: ui.legend([["var(--series-3)", "goal +10"], ["var(--series-2)", "pit −10"], ["var(--text-faint)", "wall"], ["var(--series-4)", "agent"]]),
      tools: `<div class="seg" data-r="paint"><button data-p="W" class="on">wall</button><button data-p="P">pit</button><button data-p="G">goal</button><button data-p="S">start</button><button data-p=".">erase</button></div>` + ui.btn("train", "train", true),
      hint: "click cells to paint the selected tile · every step costs −0.1",
      side: [
        { title: "world", html: ui.select("layout", "layout", [["cliff", "cliff walk"], ["maze", "maze"], ["open", "open field"]], "cliff") + ui.range("slip", "slip probability (stochastic moves)", 0, 0.4, 0.05, 0) },
        { title: "agent", html: ui.select("algo", "algorithm", [["q", "Q-learning (off-policy)"], ["sarsa", "SARSA (on-policy)"]], "q") + ui.range("alpha", "learning rate α", 0.05, 1, 0.05, 0.5) + ui.range("gamma", "discount γ", 0.5, 0.99, 0.01, 0.95) + ui.range("eps", "initial ε (exploration)", 0, 1, 0.05, 0.5) + ui.range("decay", "ε decay per episode", 0.9, 1, 0.005, 0.97) + ui.range("ep", "episodes", 10, 500, 10, 200) },
      ],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "s", label: "state s", sub: "grid cell", x: 46, y: 40, kind: "data" },
          { id: "pol", label: "ε-greedy π", sub: "explore / exploit", x: 170, y: 40, kind: "param" },
          { id: "env", label: "environment", sub: "r, s′", x: 310, y: 40, kind: "op" },
          { id: "td", label: "TD target", sub: "r + γ max Q(s′,·)", x: 300, y: 120, kind: "loss" },
          { id: "upd", label: "Q(s,a) += α δ", sub: "table update", x: 140, y: 120, kind: "op" },
          { id: "q", label: "Q table", sub: "|S| × |A|", x: 140, y: 172, kind: "out" },
        ],
        edges: [{ from: "s", to: "pol" }, { from: "pol", to: "env", label: "a" }, { from: "env", to: "td" }, { from: "td", to: "upd", label: "δ" }, { from: "upd", to: "q" }, { from: "upd", to: "s", dash: true, label: "s ← s′" }],
      },
      phases: { step: { label: "TD step", color: 0 }, episode: { label: "episode", color: 4 }, greedy: { label: "greedy rollout", color: 2 } },
      charts: [{ id: "ret", title: "return per episode", custom: true }],
      panels: [{ id: "qs", title: "Q-values at the agent's cell" }],
      onFrame: render,
    });
    let grid = LAYOUTS.cliff(), paint = "W", sim = null;
    const P = () => ({ algo: lab.val("algo"), alpha: lab.val("alpha"), gamma: lab.val("gamma"), eps: lab.val("eps"), decay: lab.val("decay"), episodes: lab.val("ep"), slip: lab.val("slip") });
    function train() { MLU.seedRng(13); sim = simulate(grid, P()); drawRet(); lab.load(sim.frames, { play: true }); }
    function drawRet() {
      const el = lab.charts.ret.el; el.innerHTML = "";
      const r = sim.returns;
      const Pp = Lab.xy(el, { w: 360, h: 130, xd: [1, r.length], yd: [Math.min(...r), Math.max(...r, 0) + 0.5], xLabel: "episode" });
      Pp.line("r", r.map((v, i) => [i + 1, v]), { color: "var(--series-1)", width: 1, curve: d3.curveLinear });
      Pp.line("avg", r.map((v, i) => [i + 1, MLU.mean(r.slice(Math.max(0, i - 9), i + 1))]), { color: "var(--series-4)", width: 2.2 });
      el.closest(".lab-card").querySelector(".chart-legend").innerHTML = `<span class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>return</span><span class="legend-item"><span class="swatch" style="background:var(--series-4)"></span>10-episode mean</span>`;
    }
    lab.R("paint").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; paint = b.dataset.p; lab.R("paint").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); };
    const svg = d3.select(lab.R("grid"));
    const qBars = Lab.bars(lab.R("panel-qs"), { w: 360, h: 120, signed: true });
    function render(f) {
      const st = f.st;
      svg.selectAll("*").remove();
      const cs = Math.min(620 / C, 410 / R), ox = (640 - cs * C) / 2, oy = 10;
      const allQ = st.Q.flat(2).map(Math.abs); const mx = Math.max(0.5, ...allQ);
      for (let y = 0; y < R; y++) for (let x = 0; x < C; x++) {
        const t = grid[y][x], X = ox + x * cs, Y = oy + y * cs;
        const g = svg.append("g").style("cursor", "pointer").on("click", () => {
          if (paint === "S") grid.forEach((row) => row.forEach((v, i) => { if (v === "S") row[i] = "."; }));
          grid[y][x] = paint; train();
        });
        g.append("rect").attr("x", X).attr("y", Y).attr("width", cs - 1).attr("height", cs - 1).attr("fill", t === "W" ? "var(--text-faint)" : t === "G" ? "color-mix(in srgb, var(--series-3) 45%, var(--bg-inset))" : t === "P" ? "color-mix(in srgb, var(--series-2) 45%, var(--bg-inset))" : "var(--bg-inset)").attr("stroke", "var(--border-soft)");
        if (t === "W" || t === "G" || t === "P") { g.append("text").attr("x", X + cs / 2).attr("y", Y + cs / 2 + 5).attr("text-anchor", "middle").style("font-size", "14px").attr("fill", "var(--text)").text(t === "G" ? "+10" : t === "P" ? "−10" : ""); continue; }
        const q = st.Q[y][x], cx = X + cs / 2, cy = Y + cs / 2;
        const tri = [[[X, Y], [X + cs - 1, Y]], [[X + cs - 1, Y], [X + cs - 1, Y + cs - 1]], [[X + cs - 1, Y + cs - 1], [X, Y + cs - 1]], [[X, Y + cs - 1], [X, Y]]];
        tri.forEach((e, a) => {
          const v = q[a];
          const on = st.upd && st.upd[0] === x && st.upd[1] === y && st.upd[2] === a;
          g.append("path").attr("d", `M${cx},${cy}L${e[0][0]},${e[0][1]}L${e[1][0]},${e[1][1]}Z`).attr("fill", v >= 0 ? `color-mix(in srgb, var(--series-3) ${Math.round(Math.min(1, v / mx) * 75)}%, transparent)` : `color-mix(in srgb, var(--series-2) ${Math.round(Math.min(1, -v / mx) * 75)}%, transparent)`).attr("stroke", on ? "var(--series-4)" : "var(--border-soft)").attr("stroke-width", on ? 3 : 0.5);
        });
        if (q.some((v) => v !== 0)) {
          const a = q.indexOf(Math.max(...q));
          const L2 = cs * 0.26;
          g.append("line").attr("x1", cx - ACTS[a][0] * L2 * 0.6).attr("y1", cy - ACTS[a][1] * L2 * 0.6).attr("x2", cx + ACTS[a][0] * L2).attr("y2", cy + ACTS[a][1] * L2).attr("stroke", "var(--text)").attr("stroke-width", 2.2).attr("marker-end", "url(#ql-arr)");
        }
        if (t === "S") g.append("text").attr("x", X + 4).attr("y", Y + 12).style("font-size", "9px").attr("fill", "var(--text-dim)").text("start");
      }
      const defs = svg.append("defs");
      defs.append("marker").attr("id", "ql-arr").attr("viewBox", "0 0 10 10").attr("refX", 6).attr("refY", 5).attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto").append("path").attr("d", "M0,0L10,5L0,10z").attr("fill", "var(--text)");
      if (st.path && st.path.length > 1) svg.append("path").attr("d", d3.line().x((p) => ox + p[0] * cs + cs / 2).y((p) => oy + p[1] * cs + cs / 2)(st.path)).attr("fill", "none").attr("stroke", "var(--series-4)").attr("stroke-width", 3).attr("stroke-opacity", 0.6).attr("pointer-events", "none");
      if (st.agent) svg.append("circle").attr("cx", ox + st.agent[0] * cs + cs / 2).attr("cy", oy + st.agent[1] * cs + cs / 2).attr("r", cs * 0.18).attr("fill", "var(--series-4)").attr("stroke", "var(--bg)").attr("stroke-width", 2).attr("pointer-events", "none");
      const q = st.agent ? st.Q[st.agent[1]][st.agent[0]] : [0, 0, 0, 0];
      qBars.set(q.map((v, a) => ({ label: ANAME[a], value: v, color: v >= 0 ? col(2) : col(1) })));
      const m = f.m || {};
      lab.readout(`<div class="kv"><span>algorithm</span><b>${P().algo === "sarsa" ? "SARSA" : "Q-learning"}</b><span>episodes</span><b>${sim.returns.length}</b><span>last return</span><b>${m.ret != null ? m.ret.toFixed(1) : "–"}</b><span>10-ep mean</span><b>${m.avg != null ? m.avg.toFixed(2) : "–"}</b></div>`);
    }
    lab.watch(["layout"], () => { grid = LAYOUTS[lab.val("layout")](); train(); });
    lab.watch(["algo", "alpha", "gamma", "eps", "decay", "ep", "slip"], () => train());
    lab.q('[data-b="train"]').onclick = () => train();
    train();
    return lab.destroy;
  }

  MLApp.register({
    id: "q-learning",
    name: "Q-Learning Gridworld",
    category: "Reinforcement Learning",
    tagline: "TD control: Q-learning vs SARSA",
    description: "An agent learns to reach the goal in an editable gridworld purely from rewards. Watch individual temporal-difference updates in the first episodes, then Q-values and the greedy policy spread back from the goal, and compare off-policy Q-learning with on-policy SARSA on the cliff.",
    info: {
      type: "Reinforcement Learning - value-based, model-free, off-policy, tabular temporal-difference control.",
      scenario: "Sequential decision-making where an agent must learn purely from trial-and-error reward signals (not labeled examples) which actions lead to good long-term outcomes under delayed rewards.",
      inputs: "The environment's current state s, the set of possible actions, and a reward r received after each action - no labeled 'correct action' is ever given.",
      intuition: {
        definition: "Keep a table estimating the <b>long-term value</b> of taking each action in each state. After every step, nudge the entry you just used toward the reward you got plus the best value available from where you landed. Value propagates backward from rewards until the whole table is consistent.",
        steps: [
          "Act, observe the reward and the next state.",
          "Compute a target: immediate reward plus discounted best future value.",
          "Move the current estimate a fraction α toward that target.",
          "Explore occasionally so you do not lock onto a mediocre route.",
        ],
        applications: [
          "Game-playing agents, the ancestor of Deep Q-Networks",
          "Robot navigation and control in simulation",
          "Inventory, pricing, and resource scheduling",
          "Adaptive traffic signal control",
          "Any simulatable sequential decision problem with delayed reward",
        ],
      },
      math: [
        { title: "Return", formula: "G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …", note: "Discounted cumulative future reward. γ below 1 keeps the sum finite and expresses a preference for sooner rewards." },
        { title: "Action-value function", formula: "Q^π(s,a) = E[ G_t | s_t = s, a_t = a, then follow π ]", note: "The expected return from taking a in s and behaving according to π thereafter. Q-learning targets the optimal Q* directly." },
        { title: "Bellman optimality", formula: "Q*(s,a) = E[ r + γ·max_{a'} Q*(s',a') ]", note: "The fixed-point equation the table converges to. The max is what makes it optimality rather than merely policy evaluation." },
        { title: "TD error", formula: "δ = r + γ·max_{a'} Q(s',a') − Q(s,a)", note: "The gap between the bootstrapped target and the current estimate. It is a surprise signal, not a supervised loss." },
        { title: "Update rule", formula: "Q(s,a) := Q(s,a) + α·δ", note: "An exponentially weighted running average toward the target. There is no gradient here in the tabular case, just incremental averaging." },
        { title: "ε-greedy policy", formula: "a = random with probability ε, else argmax_a Q(s,a)", note: "The simplest workable answer to exploration versus exploitation. Usually ε is decayed as the estimates become trustworthy." },
        { title: "Convergence condition", formula: "Σ_t α_t = ∞ and Σ_t α_t² < ∞, with every (s,a) visited infinitely often", note: "Under these Robbins-Monro conditions Q converges to Q* with probability 1, regardless of the policy used to explore." },
      ],
      pipeline: [
        { label: "State s", note: "observe" },
        { label: "ε-greedy action", note: "explore or exploit" },
        { label: "Get r, s'", note: "environment step" },
        { label: "TD error δ", note: "r + γ max Q(s',·) − Q" },
        { label: "Q += α·δ", note: "repeat", accent: "green" },
      ],
      decisionFunction: {
        text: "π(s) = argmax_a Q(s,a)",
        mechanism: "The agent keeps a table of estimated long-term value for every (state, action) pair; once trained, it simply acts on the highest-value action in the current state. During training it uses ε-greedy (mostly greedy, occasionally random) to keep exploring.",
      },
      lossFunction: {
        text: "δ = r + γ·max_{a'} Q(s',a') − Q(s,a);   Q(s,a) ← Q(s,a) + α·δ",
        mechanism: "Not a loss in the supervised sense - the temporal-difference error δ is used directly as the update. It bootstraps off the agent's own current value estimates rather than a ground-truth label, driving Q(s,a) toward consistency with the Bellman optimality equation.",
      },
      optimization: [
        { title: "Off-policy learning", formula: "target uses max_{a'} Q(s',a'), not the action actually taken", note: "The agent learns the value of the optimal policy while behaving with a different, exploratory one. That decoupling is what lets it learn from random exploration, demonstrations, or replayed history." },
        { title: "SARSA contrast", formula: "SARSA target: r + γ·Q(s', a'_actual)", note: "On-policy: it learns the value of the policy it is actually following, including its exploration mistakes. SARSA takes safer routes near cliffs; Q-learning learns the optimal but riskier one." },
        { title: "Bootstrapping", formula: "the target contains Q itself", note: "Learning from a guess. It is far more sample-efficient than waiting for full episode returns, but errors propagate and can amplify." },
        { title: "Maximisation bias", formula: "E[max Q̂] ≥ max E[Q̂]", note: "Taking a max over noisy estimates systematically overestimates. Double Q-learning fixes it by keeping two tables and using one to select the action and the other to evaluate it." },
        { title: "Exploration schedule", formula: "ε_t = max(ε_min, ε₀·decay^t)", note: "High exploration early when estimates are meaningless, low later when they can be trusted. A fixed ε either explores too little at first or acts suboptimally forever." },
        { title: "Scaling up", formula: "replace the table with Q(s,a; θ), a neural network", note: "This is DQN. It requires a replay buffer and a target network, because bootstrapping off a function approximator that is itself changing is unstable." },
      ],
      output: "A value Q(s,a) for every state-action pair, and the greedy policy it induces (the best action in every state).",
      assumptions: [
        { name: "Markov property", why: "The update assumes the current state fully determines the future; history beyond it is irrelevant.", check: "If the agent needs memory of earlier steps, the state representation is incomplete. Add features or use a recurrent policy." },
        { name: "Discrete, enumerable states and actions", why: "Tabular Q-learning stores one entry per pair.", check: "Continuous or huge spaces require function approximation, that is, DQN." },
        { name: "Every state-action pair is reachable", why: "Convergence requires visiting each infinitely often.", check: "Keep ε above zero long enough, and make sure the environment does not trap the agent." },
        { name: "Stationary environment", why: "Old experience must remain valid; a changing environment invalidates the table.", check: "Keep a floor under α so the agent can still adapt." },
        { name: "Rewards are informative", why: "With rewards only at the very end, credit takes an enormous number of episodes to propagate back.", check: "Consider reward shaping, but do it carefully or you change the optimal policy." },
      ],
      hyperparameters: [
        { name: "α (learning rate)", range: "0.01 - 0.5", increasing: "Faster adaptation, noisier and less stable estimates.", strategy: "0.1 is a common default. Decay it for convergence in a stationary environment; keep it fixed if the environment drifts." },
        { name: "γ (discount factor)", range: "0.8 - 0.999", increasing: "More far-sighted, but slower and less stable value propagation.", strategy: "0.99 for long-horizon tasks, lower for short episodic ones. Effective horizon is roughly 1/(1−γ)." },
        { name: "ε (exploration rate)", range: "1.0 → 0.01", increasing: "More exploration, slower exploitation of what is known.", strategy: "Start at 1.0 and decay to about 0.01. Insufficient exploration is the most common cause of a stuck agent." },
        { name: "ε decay rate", range: "0.99 - 0.9999", increasing: "Slower decay, more exploration overall.", strategy: "Tune so ε reaches its floor at roughly half the training budget." },
        { name: "episodes", range: "100 - 1e6", increasing: "More convergence, more compute.", strategy: "Train until the reward curve plateaus rather than for a fixed count." },
        { name: "Q initialisation", range: "0 or optimistic", increasing: "Optimistic values encourage systematic early exploration.", strategy: "Optimistic initialisation is a clean alternative to high ε: unvisited actions look attractive until tried." },
      ],
      metrics: ["Cumulative episode reward", "Steps to reach the goal", "Change in Q-table between episodes (convergence)", "Success rate under a greedy policy with ε set to 0"],
      typicalUses: ["Game-playing agents", "Robotics control", "Resource allocation / scheduling", "Any simulate-able sequential decision problem - the tabular ancestor of Deep Q-Networks (DQN)"],
      diagnostics: [
        "Plot episode reward over time. It should trend up and then flatten. A flat line from the start means the agent is not learning at all.",
        "Track the maximum absolute Q change per episode. It should shrink toward zero as the table converges.",
        "Evaluate periodically with ε set to zero. Training reward is depressed by exploration and understates the learned policy.",
        "Check the visit count for each state-action pair. Pairs never visited have meaningless values.",
        "If reward plateaus below optimum, ε probably decayed too fast and the agent settled on a local route.",
      ],
      advantages: [
        "Model-free: it needs no knowledge of transition probabilities or reward structure, only experience.",
        "Off-policy, so it can learn the optimal policy from random, human, or replayed behaviour.",
        "Provably converges to the optimal Q in the tabular case under mild conditions.",
        "Conceptually simple: one update rule, three hyperparameters.",
        "Handles delayed reward, propagating credit backward across many steps.",
        "Learns online, updating after every single step rather than at episode end.",
      ],
      limitations: [
        { name: "Table size explodes", note: "one entry per state-action pair is impossible for large or continuous spaces", fix: "function approximation, that is, DQN." },
        { name: "No generalisation", note: "learning about one state teaches nothing about a similar one", fix: "features plus a neural network." },
        { name: "Sample inefficient", note: "needs many visits per state-action pair", fix: "experience replay, model-based methods, or reward shaping." },
        { name: "Maximisation bias", note: "the max over noisy estimates overestimates values", fix: "Double Q-learning." },
        { name: "Exploration is crude", note: "ε-greedy explores uniformly at random with no sense of what is worth trying", fix: "optimistic initialisation, UCB, or count-based bonuses." },
        { name: "Sensitive to reward design", note: "poorly shaped rewards produce degenerate policies", fix: "use potential-based shaping, which provably preserves the optimal policy." },
      ],
      alternatives: [
        { name: "SARSA", when: "You want the agent to account for its own exploration risk, such as near a cliff." },
        { name: "DQN", when: "The state space is large or continuous. The direct neural extension of this algorithm." },
        { name: "Policy gradient / PPO", when: "Continuous action spaces, or you need a stochastic policy." },
        { name: "Actor-critic (A2C, SAC)", when: "You want value bootstrapping and a learned policy together, which is the modern default." },
        { name: "Monte Carlo control", when: "Episodes are short and you prefer unbiased returns to bootstrapped ones." },
      ],
      pitfalls: [
        { problem: "Agent gets stuck on a mediocre route", solution: "Exploration collapsed too early. Decay ε more slowly, or raise its floor." },
        { problem: "Q values diverge to huge numbers", solution: "γ at or above 1 in a non-episodic task, or α too high. Set γ below 1 and lower α." },
        { problem: "Agent learns nothing at all", solution: "Rewards may be too sparse, or the state representation may violate the Markov property." },
        { problem: "Training reward looks worse than expected", solution: "Exploration is polluting it. Evaluate separately with ε set to 0." },
        { problem: "Learned values are systematically too optimistic", solution: "Maximisation bias. Use Double Q-learning." },
        { problem: "Works in the small grid, fails in the large one", solution: "The table no longer fits and states are rarely revisited. Move to DQN." },
      ],
      quickRef: [
        { name: "Return", formula: "G_t = Σ_k γ^k r_{t+k}" },
        { name: "Bellman optimality", formula: "Q*(s,a) = E[r + γ max_{a'} Q*(s',a')]" },
        { name: "TD error", formula: "δ = r + γ max_{a'}Q(s',a') − Q(s,a)" },
        { name: "Update", formula: "Q(s,a) += α·δ" },
        { name: "Policy", formula: "π(s) = argmax_a Q(s,a)" },
        { name: "ε-greedy", formula: "random w.p. ε, else greedy" },
        { name: "SARSA target", formula: "r + γ·Q(s', a'_actual)" },
        { name: "Effective horizon", formula: "≈ 1/(1 − γ)" },
      ],
      code: `import numpy as np

Q = np.zeros((n_states, n_actions))
alpha, gamma = 0.1, 0.99
eps, eps_min, eps_decay = 1.0, 0.01, 0.995

for episode in range(5000):
    s, done = env.reset(), False
    while not done:
        # epsilon-greedy: explore early, exploit once values are trustworthy
        a = env.sample_action() if np.random.rand() < eps else Q[s].argmax()
        s2, r, done = env.step(a)

        # Off-policy: the target uses max over next actions, not the
        # action actually taken next. That is what makes it Q-learning.
        target = r + (0 if done else gamma * Q[s2].max())
        Q[s, a] += alpha * (target - Q[s, a])
        s = s2

    eps = max(eps_min, eps * eps_decay)

    # Evaluate greedily: training reward is depressed by exploration.
    if episode % 500 == 0:
        print(episode, round(eps, 3), evaluate(Q, env, epsilon=0.0))

policy = Q.argmax(axis=1)`,
      whyChain: [
        { q: "What does off-policy actually mean here?", a: "The update target uses max over next actions rather than the action the agent will actually take. So it learns the value of behaving optimally, even while behaving randomly. The behaviour policy and the policy being learned are decoupled." },
        { q: "How does Q-learning differ from SARSA in practice?", a: "SARSA's target uses the action actually taken next, so it learns the value of the exploring policy, including the risk of random moves. On a cliff-walking task Q-learning learns the optimal path right along the edge, while SARSA learns a safer detour, because it accounts for occasionally stepping off." },
        { q: "Why is bootstrapping both the strength and the weakness?", a: "Using the current estimate as part of the target means you can learn from a single step rather than waiting for the episode to end, which is enormously more sample efficient. But you are learning from a guess, so errors propagate and, once function approximation is involved, can amplify into divergence." },
        { q: "What does γ actually control?", a: "How far ahead the agent cares. The effective horizon is roughly 1/(1−γ), so γ = 0.99 means it weighs about 100 steps ahead. It also guarantees the infinite sum converges, which is why γ must stay below 1 in continuing tasks." },
        { q: "Why does maximisation bias happen?", a: "Q values are noisy estimates. Taking a max over several noisy numbers systematically selects the ones that happen to be overestimated, so the max of the estimates exceeds the estimate of the max. The bias compounds through bootstrapping." },
        { q: "How does Double Q-learning fix that?", a: "It keeps two independent tables. One selects the best action, the other supplies its value. Since the noise in the two is independent, the action chosen because it looks good in one table is not systematically overvalued in the other." },
        { q: "Why decay ε rather than keeping it fixed?", a: "Early on the Q values are meaningless, so acting greedily just locks in arbitrary choices. Late on the values are informative, so continued random actions only cost reward. Decaying shifts the balance as the estimates become trustworthy." },
        { q: "What has to change to get from here to DQN?", a: "Replace the table with a neural network mapping state to action values. That breaks tabular convergence guarantees, so you need a replay buffer to decorrelate consecutive samples and a slowly updated target network so the bootstrap target does not chase itself." },
      ],
      parameters: [
        { name: "α (learning rate)", effect: "How much each new experience overwrites the old value estimate. Higher = faster but noisier learning." },
        { name: "γ (discount factor)", effect: "How much future reward matters relative to immediate reward. Near 1 = far-sighted; near 0 = short-sighted." },
        { name: "ε (exploration rate)", effect: "Probability of a random action instead of the current best-known one - balances exploring the environment vs exploiting what's already learned." },
      ],
      metrics: ["Cumulative episode reward", "Steps to reach the goal", "Change in Q-table between episodes (convergence)"],
      typicalUses: ["Game-playing agents", "Robotics control", "Resource allocation / scheduling", "Any simulate-able sequential decision problem - the tabular ancestor of Deep Q-Networks (DQN)"],
      workedExample: {
        setup: "Q(s,a)=0 initially. Taking action a in state s gives reward r=−1 and lands in s′, where max_a′Q(s′,·)=5. α=0.5, γ=0.9.",
        steps: [
          "TD error δ = r + γ×max_a′Q(s′,·) − Q(s,a) = −1 + 0.9×5 − 0 = −1 + 4.5 = 3.5.",
          "Update: Q(s,a) ← Q(s,a) + α×δ = 0 + 0.5×3.5 = 1.75.",
        ],
        result: "Q(s,a) updates from 0 to 1.75 - even though the immediate reward was negative, the high value already learned at s′ propagates back and makes this action look good",
      },
    },
    mount,
  });
})();
