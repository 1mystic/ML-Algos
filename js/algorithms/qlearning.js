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
      decisionFunction: {
        text: "π(s) = argmax_a Q(s,a)",
        mechanism: "The agent keeps a table of estimated long-term value for every (state, action) pair; once trained, it simply acts on the highest-value action in the current state. During training it uses ε-greedy (mostly greedy, occasionally random) to keep exploring.",
      },
      lossFunction: {
        text: "δ = r + γ·max_{a'} Q(s',a') − Q(s,a);   Q(s,a) ← Q(s,a) + α·δ",
        mechanism: "Not a loss in the supervised sense - the temporal-difference error δ is used directly as the update. It bootstraps off the agent's own current value estimates rather than a ground-truth label, driving Q(s,a) toward consistency with the Bellman optimality equation.",
      },
      output: "A value Q(s,a) for every state-action pair, and the greedy policy it induces (the best action in every state).",
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
