(() => {
  const GW = 9, GH = 6, CELL = 62;
  const ACTIONS = [[0, -1], [0, 1], [-1, 0], [1, 0]]; // up, down, left, right
  const ACTION_ANGLE = [-90, 90, 180, 0];

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint">episode <b id="ql-ep">0</b> · last reward <b id="ql-last">–</b></span></div>
            <div class="btn-row">
              <button id="ql-run1">run 1 episode</button>
              <button id="ql-run50" class="primary">run 50 episodes</button>
              <button id="ql-play">play greedy path</button>
              <button id="ql-reset-q">reset Q-table</button>
              <button id="ql-clear-grid">clear grid</button>
            </div>
          </div>
          <div class="hscroll" style="align-self:center;max-width:100%"><svg id="ql-svg" width="${GW * CELL}" height="${GH * CELL}"></svg></div>
          <div class="stage-hint">paint mode below, then click cells · S = start (fixed top-left) · shading = max Q-value · arrows = greedy action</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>paint</h3>
            <div class="btn-row" id="ql-paint-buttons">
              <button data-m="wall">wall</button>
              <button data-m="goal">goal (+10)</button>
              <button data-m="pit">pit (−10)</button>
              <button data-m="empty">erase</button>
            </div>
          </div>
          <div class="control-card">
            <h3>hyperparameters</h3>
            <div class="field"><label>&alpha; learning rate <span class="val" id="ql-alpha-val">0.3</span></label>
              <input type="range" id="ql-alpha" min="1" max="90" step="1" value="30" /></div>
            <div class="field"><label>&gamma; discount <span class="val" id="ql-gamma-val">0.9</span></label>
              <input type="range" id="ql-gamma" min="10" max="99" step="1" value="90" /></div>
            <div class="field"><label>&epsilon; exploration <span class="val" id="ql-eps-val">0.2</span></label>
              <input type="range" id="ql-eps" min="0" max="80" step="1" value="20" /></div>
          </div>
          <div class="control-card">
            <h3>state</h3>
            <div class="readout" id="ql-readout">–</div>
            <div class="note">Tabular Q-learning: Q(s,a) += &alpha;(r + &gamma;&middot;max<sub>a'</sub>Q(s',a') − Q(s,a)), with an &epsilon;-greedy behavior policy - the update rule behind <code>mla/rl/dqn.py</code>'s deep variant, done here with an exact table since the state space is small.</div>
          </div>
        </div>
      </div>
    `;

    let grid = Array.from({ length: GH }, () => new Array(GW).fill("empty"));
    grid[1][6] = "goal"; grid[4][3] = "pit";
    for (let i = 1; i < GH - 1; i++) grid[i][4] = "wall";
    grid[2][4] = "empty";
    const start = [0, 0];
    let Q = Array.from({ length: GH }, () => Array.from({ length: GW }, () => new Array(4).fill(0)));
    let episode = 0, paintMode = "wall";

    function alpha() { return +document.getElementById("ql-alpha").value / 100; }
    function gamma() { return +document.getElementById("ql-gamma").value / 100; }
    function eps() { return +document.getElementById("ql-eps").value / 100; }

    function step(pos, a) {
      const [dx, dy] = ACTIONS[a];
      let [px, py] = pos;
      const nx = px + dx, ny = py + dy;
      if (nx < 0 || nx >= GW || ny < 0 || ny >= GH || grid[ny][nx] === "wall") return { pos: [px, py], reward: -1, done: false };
      const cell = grid[ny][nx];
      if (cell === "goal") return { pos: [nx, ny], reward: 10, done: true };
      if (cell === "pit") return { pos: [nx, ny], reward: -10, done: true };
      return { pos: [nx, ny], reward: -0.1, done: false };
    }
    function runEpisode(explore) {
      let pos = [...start], total = 0;
      for (let t = 0; t < 200; t++) {
        const [px, py] = pos;
        let a;
        if (explore && MLU.rng() < eps()) a = MLU.randInt(4);
        else { const qs = Q[py][px]; a = qs.indexOf(Math.max(...qs)); }
        const { pos: npos, reward, done } = step(pos, a);
        const [nx, ny] = npos;
        const bestNext = Math.max(...Q[ny][nx]);
        Q[py][px][a] += alpha() * (reward + gamma() * bestNext - Q[py][px][a]);
        total += reward;
        pos = npos;
        if (done) return total;
      }
      return total;
    }

    const svg = d3.select("#ql-svg");
    const cellG = svg.append("g");
    const arrowG = svg.append("g");
    const agentMarker = svg.append("circle").attr("r", CELL * 0.28).attr("fill", "var(--accent)").style("opacity", 0);

    function maxQRange() {
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < GH; i++) for (let j = 0; j < GW; j++) if (grid[i][j] === "empty") {
        const m = Math.max(...Q[i][j]); lo = Math.min(lo, m); hi = Math.max(hi, m);
      }
      if (lo === Infinity) { lo = 0; hi = 1; }
      return [lo, hi];
    }

    function render() {
      document.getElementById("ql-ep").textContent = episode;
      document.getElementById("ql-alpha-val").textContent = alpha().toFixed(2);
      document.getElementById("ql-gamma-val").textContent = gamma().toFixed(2);
      document.getElementById("ql-eps-val").textContent = eps().toFixed(2);

      const [lo, hi] = maxQRange();
      const cells = [];
      for (let i = 0; i < GH; i++) for (let j = 0; j < GW; j++) cells.push({ i, j, type: grid[i][j] });
      const sel = cellG.selectAll("rect").data(cells);
      sel.enter().append("rect").attr("width", CELL - 2).attr("height", CELL - 2).attr("stroke", "var(--border-soft)")
        .merge(sel)
        .attr("x", (d) => d.j * CELL + 1).attr("y", (d) => d.i * CELL + 1)
        .attr("fill", (d) => {
          if (d.type === "wall") return "#3a3a3a";
          if (d.type === "goal") return "#4ade80";
          if (d.type === "pit") return "#f87171";
          const q = Math.max(...Q[d.i][d.j]);
          const t = hi > lo ? (q - lo) / (hi - lo) : 0.5;
          return d3.interpolateRgb("#161616", "#3b82f6")(t);
        })
        .on("click", (event, d) => { if (d.i === start[1] && d.j === start[0]) return; grid[d.i][d.j] = paintMode; render(); });
      sel.exit().remove();

      const arrows = cells.filter((d) => d.type === "empty" && !(d.i === start[1] && d.j === start[0]));
      const asel = arrowG.selectAll("path").data(arrows);
      asel.enter().append("path").attr("d", `M ${-CELL * 0.18} ${-CELL * 0.12} L ${CELL * 0.18} 0 L ${-CELL * 0.18} ${CELL * 0.12} Z`).attr("fill", "var(--text-dim)")
        .merge(asel)
        .attr("transform", (d) => {
          const qs = Q[d.i][d.j];
          const a = qs.indexOf(Math.max(...qs));
          const cx = d.j * CELL + CELL / 2, cy = d.i * CELL + CELL / 2;
          return `translate(${cx},${cy}) rotate(${ACTION_ANGLE[a]})`;
        })
        .style("opacity", (d) => (Math.max(...Q[d.i][d.j]) !== 0 ? 0.8 : 0));
      asel.exit().remove();

      svg.selectAll("text.start-label").remove();
      svg.append("text").attr("class", "start-label").attr("x", start[0] * CELL + CELL / 2).attr("y", start[1] * CELL + CELL / 2 + 4)
        .attr("text-anchor", "middle").attr("fill", "var(--text)").attr("font-size", 13).attr("font-weight", 700).text("S");

      document.getElementById("ql-readout").innerHTML = `grid: ${GW}×${GH}<br>episodes trained: <b>${episode}</b>`;
    }

    document.getElementById("ql-paint-buttons").addEventListener("click", (e) => {
      const btn = e.target.closest("button"); if (!btn) return;
      paintMode = btn.dataset.m;
      document.querySelectorAll("#ql-paint-buttons button").forEach((b) => b.classList.toggle("primary", b.dataset.m === paintMode));
    });
    document.querySelector('#ql-paint-buttons button[data-m="wall"]').classList.add("primary");

    document.getElementById("ql-run1").addEventListener("click", () => {
      const r = runEpisode(true); episode++; document.getElementById("ql-last").textContent = r.toFixed(1); render();
    });
    document.getElementById("ql-run50").addEventListener("click", () => {
      let r = 0; for (let i = 0; i < 50; i++) { r = runEpisode(true); episode++; }
      document.getElementById("ql-last").textContent = r.toFixed(1); render();
    });
    document.getElementById("ql-reset-q").addEventListener("click", () => {
      Q = Array.from({ length: GH }, () => Array.from({ length: GW }, () => new Array(4).fill(0)));
      episode = 0; document.getElementById("ql-last").textContent = "–"; render();
    });
    document.getElementById("ql-clear-grid").addEventListener("click", () => {
      grid = Array.from({ length: GH }, () => new Array(GW).fill("empty"));
      render();
    });
    ["ql-alpha", "ql-gamma", "ql-eps"].forEach((id) => document.getElementById(id).addEventListener("input", render));

    let playTimer = null;
    document.getElementById("ql-play").addEventListener("click", () => {
      if (playTimer) { clearInterval(playTimer); playTimer = null; }
      let pos = [...start], t = 0, visited = new Set();
      agentMarker.style("opacity", 1);
      playTimer = setInterval(() => {
        const [px, py] = pos;
        agentMarker.attr("cx", px * CELL + CELL / 2).attr("cy", py * CELL + CELL / 2);
        const key = px + "," + py;
        if (t++ > 40 || visited.has(key)) { clearInterval(playTimer); playTimer = null; agentMarker.style("opacity", 0); return; }
        visited.add(key);
        const qs = Q[py][px];
        const a = qs.indexOf(Math.max(...qs));
        const { pos: npos, done } = step(pos, a);
        pos = npos;
        if (done) {
          agentMarker.attr("cx", pos[0] * CELL + CELL / 2).attr("cy", pos[1] * CELL + CELL / 2);
          clearInterval(playTimer); playTimer = null;
          setTimeout(() => agentMarker.style("opacity", 0), 500);
        }
      }, 260);
    });

    render();
    return () => { if (playTimer) clearInterval(playTimer); };
  }

  MLApp.register({
    id: "q-learning",
    name: "Q-Learning Gridworld",
    category: "Reinforcement Learning",
    tagline: "tabular TD control",
    description: "An agent learns a value for every (state, action) pair purely from trial and error and a delayed reward, via the Q-learning temporal-difference update. Paint walls, a goal, and a pit, then train and watch it find the route.",
    sourceFile: "mla/rl/dqn.py",
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
