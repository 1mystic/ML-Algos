(() => {
  const { ui, col, fmt } = Lab;
  const S = ["Fair", "Loaded"];

  function model(p, q, six) {
    return {
      pi: [0.5, 0.5],
      A: [[p, 1 - p], [1 - q, q]],
      B: [[1, 1, 1, 1, 1, 1].map((v) => v / 6), [1, 2, 3, 4, 5, 6].map((k) => (k === 6 ? six : (1 - six) / 5))],
    };
  }
  function sample(M, T) {
    const st = [], ob = [];
    let s = MLU.rng() < M.pi[0] ? 0 : 1;
    for (let t = 0; t < T; t++) {
      if (t) s = MLU.rng() < M.A[s][0] ? 0 : 1;
      let r = MLU.rng(), o = 0;
      for (; o < 5; o++) { r -= M.B[s][o]; if (r <= 0) break; }
      st.push(s); ob.push(o);
    }
    return { st, ob };
  }
  function forward(M, ob) {
    const T = ob.length, al = [], c = [];
    for (let t = 0; t < T; t++) {
      const a = [0, 1].map((j) => (t ? al[t - 1][0] * M.A[0][j] + al[t - 1][1] * M.A[1][j] : M.pi[j]) * M.B[j][ob[t]]);
      const s = a[0] + a[1]; c.push(s); al.push(a.map((v) => v / s));
    }
    return { al, ll: c.reduce((s, v) => s + Math.log(v), 0), c };
  }
  function backward(M, ob, c) {
    const T = ob.length, be = new Array(T);
    be[T - 1] = [1, 1];
    for (let t = T - 2; t >= 0; t--) be[t] = [0, 1].map((i) => [0, 1].reduce((s, j) => s + M.A[i][j] * M.B[j][ob[t + 1]] * be[t + 1][j], 0) / c[t + 1]);
    return be;
  }
  function viterbi(M, ob) {
    const T = ob.length, dl = [], psi = [];
    for (let t = 0; t < T; t++) {
      if (!t) { dl.push([0, 1].map((j) => Math.log(M.pi[j]) + Math.log(M.B[j][ob[0]]))); psi.push([0, 0]); continue; }
      const row = [], bp = [];
      for (let j = 0; j < 2; j++) {
        const cand = [0, 1].map((i) => dl[t - 1][i] + Math.log(M.A[i][j]));
        const b = cand[0] >= cand[1] ? 0 : 1;
        row.push(cand[b] + Math.log(M.B[j][ob[t]])); bp.push(b);
      }
      dl.push(row); psi.push(bp);
    }
    const path = new Array(T);
    path[T - 1] = dl[T - 1][0] >= dl[T - 1][1] ? 0 : 1;
    for (let t = T - 2; t >= 0; t--) path[t] = psi[t + 1][path[t + 1]];
    return { dl, psi, path };
  }

  function buildFrames(M, seq) {
    const { ob, st } = seq, T = ob.length;
    const F = forward(M, ob), Bk = backward(M, ob, F.c), V = viterbi(M, ob);
    const gamma = F.al.map((a, t) => { const g = [a[0] * Bk[t][0], a[1] * Bk[t][1]]; const s = g[0] + g[1]; return [g[0] / s, g[1] / s]; });
    const frames = [{ phase: "model", major: true, arch: ["pi", "A", "B"], title: "A hidden Markov model of a dishonest casino", text: `The casino secretly switches between a fair die and a loaded one (six with probability ${M.B[1][5].toFixed(2)}). We only see the rolls. Switching follows a Markov chain: stay fair with p = ${M.A[0][0]}, stay loaded with q = ${M.A[1][1]}.`, tex: "P(x_{1:T},z_{1:T})=\\pi_{z_1}\\prod_{t=2}^{T}A_{z_{t-1}z_t}\\prod_{t=1}^{T}B_{z_t}(x_t)", st: { stage: "model" } }];
    for (let t = 0; t < T; t++) frames.push({ phase: "fwd", major: t === 0, arch: ["A", "B", "fwd"], title: `Forward pass, t = ${t + 1}: roll ${ob[t] + 1}`, text: `α<sub>t</sub>(j) = probability of the rolls so far and being in state j now. Each column sums over both ways of arriving (edge thickness = contribution), then multiplies by how likely the roll is in that state. P(Loaded | rolls so far) = <b>${F.al[t][1].toFixed(2)}</b>.`, tex: "\\alpha_t(j)=B_j(x_t)\\sum_i \\alpha_{t-1}(i)\\,A_{ij}", st: { stage: "fwd", t } });
    frames.push({ phase: "fwd", arch: ["fwd", "lik"], title: `Sequence likelihood: log P(rolls) = ${F.ll.toFixed(2)}`, text: "Summing the last column gives the probability of the whole roll sequence under the model, in O(T·N²) instead of summing over all 2ᵀ hidden paths. Columns are rescaled to avoid underflow; the log-likelihood is the sum of the log scale factors.", tex: "P(x_{1:T})=\\sum_j\\alpha_T(j)", st: { stage: "fwd", t: T - 1 } });
    for (let t = T - 1; t >= 0; t -= 3) frames.push({ phase: "bwd", arch: ["bwd"], title: `Backward pass, t = ${t + 1}`, text: "β<sub>t</sub>(i) is the probability of all <i>future</i> rolls given state i now. It runs right to left with the same recursion shape.", tex: "\\beta_t(i)=\\sum_j A_{ij}\\,B_j(x_{t+1})\\,\\beta_{t+1}(j)", st: { stage: "bwd", t } });
    frames.push({ phase: "post", major: true, arch: ["fwd", "bwd", "post"], title: "Smoothed posterior: combine past and future evidence", text: `γ<sub>t</sub> uses every roll, before and after t. Shaded bars in the chart are the true hidden states; the curve is P(Loaded | all rolls). Posterior decoding gets <b>${(gamma.filter((g, t) => (g[1] > 0.5 ? 1 : 0) === st[t]).length / T * 100).toFixed(0)}%</b> of states right.`, tex: "\\gamma_t(j)=P(z_t=j\\mid x_{1:T})=\\frac{\\alpha_t(j)\\,\\beta_t(j)}{\\sum_k\\alpha_t(k)\\,\\beta_t(k)}", st: { stage: "post" } });
    for (let t = 0; t < T; t += 2) frames.push({ phase: "vit", major: t === 0, arch: ["A", "B", "vit"], title: `Viterbi, t = ${Math.min(T, t + 1)}: keep only the best way into each state`, text: "Same recursion as the forward pass but with max instead of sum. Each node remembers which predecessor gave its best score (bold edges are the back-pointers).", tex: "\\delta_t(j)=\\log B_j(x_t)+\\max_i\\big[\\delta_{t-1}(i)+\\log A_{ij}\\big],\\quad \\psi_t(j)=\\arg\\max_i[\\cdots]", st: { stage: "vit", t: Math.min(T - 1, t) } });
    for (let t = T - 1; t >= -1; t -= 4) frames.push({ phase: "back", arch: ["vit", "path"], title: "Backtrack the back-pointers from the best final state", text: "Following ψ from the end recovers the single most probable hidden path.", tex: "z_T^\\star=\\arg\\max_j\\delta_T(j),\\qquad z_t^\\star=\\psi_{t+1}(z_{t+1}^\\star)", st: { stage: "back", from: Math.max(0, t) } });
    const acc = V.path.filter((s, t) => s === st[t]).length / T;
    frames.push({ phase: "done", major: true, arch: ["path", "post"], title: `Viterbi path matches the truth on ${(acc * 100).toFixed(0)}% of rolls`, text: "Viterbi gives the best <i>whole path</i>; posterior decoding gives the best state <i>at each time</i>. They usually agree, but posterior decoding can produce paths with impossible transitions, while Viterbi cannot.", st: { stage: "back", from: 0, done: true } });
    return { frames, F, Bk, V, gamma };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "hmm", fps: 4, plane: false,
      stageHTML: `<div class="html-stage"><div class="sub-cap">trellis: top row = observed rolls (click to change) · rows = hidden states · bottom strip = true state</div><div data-r="trellis" class="hscroll"></div></div>`,
      legend: ui.legend([[col(0), "Fair"], [col(1), "Loaded"], ["var(--series-4)", "Viterbi path", "line"]]),
      tools: ui.btn("new", "new sequence", true),
      side: [{ title: "model", html: ui.range("p", "P(stay fair)", 0.5, 0.99, 0.01, 0.95) + ui.range("q", "P(stay loaded)", 0.5, 0.99, 0.01, 0.9) + ui.range("six", "loaded die: P(six)", 0.17, 0.9, 0.01, 0.5) + ui.range("T", "sequence length", 8, 40, 1, 26) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "pi", label: "π", sub: "initial", x: 30, y: 40, w: 50, kind: "param" },
          { id: "A", label: "A", sub: "transitions", x: 100, y: 40, w: 66, kind: "param" },
          { id: "B", label: "B", sub: "emissions", x: 176, y: 40, w: 64, kind: "param" },
          { id: "fwd", label: "forward α", sub: "sum-product", x: 90, y: 110, kind: "op" },
          { id: "bwd", label: "backward β", sub: "right to left", x: 210, y: 110, kind: "op" },
          { id: "lik", label: "P(x)", sub: "likelihood", x: 40, y: 168, w: 66, kind: "loss" },
          { id: "post", label: "γ = αβ", sub: "posterior", x: 150, y: 168, kind: "out" },
          { id: "vit", label: "Viterbi δ, ψ", sub: "max-product", x: 318, y: 60, kind: "op" },
          { id: "path", label: "best path z*", sub: "backtrack", x: 318, y: 150, kind: "out" },
        ],
        edges: [{ from: "pi", to: "fwd" }, { from: "A", to: "fwd" }, { from: "B", to: "fwd" }, { from: "A", to: "bwd" }, { from: "fwd", to: "lik" }, { from: "fwd", to: "post" }, { from: "bwd", to: "post" }, { from: "B", to: "vit" }, { from: "vit", to: "path" }],
      },
      phases: { model: { label: "model", color: 3 }, fwd: { label: "forward", color: 0 }, bwd: { label: "backward", color: 4 }, post: { label: "posterior", color: 2 }, vit: { label: "Viterbi", color: 6 }, back: { label: "backtrack", color: 6 }, done: { label: "decoded", color: 2 } },
      charts: [{ id: "post", title: "P(Loaded | rolls) over time", custom: true, wide: true, note: "Shaded columns = the casino really used the loaded die. Solid = smoothed posterior γ, dashed = filtered α (past only)." }],
      onFrame: render,
    });
    let M = null, seq = null, R = null, seed = 9;
    function regen() { MLU.seedRng(seed); M = model(lab.val("p"), lab.val("q"), lab.val("six")); seq = sample(M, lab.val("T")); rebuild(); }
    function rebuild({ at = "start" } = {}) { M = model(lab.val("p"), lab.val("q"), lab.val("six")); R = buildFrames(M, seq); drawPost(null); lab.load(R.frames, { at, play: at === "start" }); }
    let PP = null;
    function drawPost(stage) {
      const el = lab.charts.post.el; el.innerHTML = "";
      const T = seq.ob.length;
      PP = Lab.xy(el, { w: 720, h: 140, xd: [0.5, T + 0.5], yd: [0, 1], xLabel: "t" });
      seq.st.forEach((s, t) => { if (s) PP.g.append("rect").attr("x", PP.x(t + 0.5)).attr("width", PP.x(t + 1.5) - PP.x(t + 0.5)).attr("y", PP.m.t).attr("height", PP.h - PP.m.t - PP.m.b).attr("fill", col(1)).attr("opacity", 0.12); });
    }
    function render(f) {
      const st = f.st, T = seq.ob.length;
      const W = 50 + T * 34, H = 200;
      const el = lab.R("trellis"); el.innerHTML = "";
      const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).style("min-width", Math.min(W, 900) + "px").attr("class", "arch-svg");
      const x = (t) => 48 + t * 34, y = (s) => 82 + s * 62;
      S.forEach((name, s) => svg.append("text").attr("x", 4).attr("y", y(s) + 4).style("font-size", "10px").attr("fill", col(s)).text(name));
      const fwdUpto = st.stage === "fwd" ? st.t : st.stage === "model" ? -1 : T - 1;
      const showAlpha = st.stage !== "model";
      // edges
      for (let t = 1; t < T; t++) for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
        let w = 0.6, c = "var(--border)", op = 0.6;
        if (st.stage === "fwd" && t <= st.t) { w = 0.5 + 5 * R.F.al[t - 1][i] * M.A[i][j]; c = col(i); op = 0.55; }
        if ((st.stage === "vit" && t <= st.t) || st.stage === "back") { if (R.V.psi[t][j] === i) { w = 2.2; c = "var(--text-dim)"; op = 0.85; } }
        if (st.stage === "back" && t > st.from && R.V.path[t - 1] === i && R.V.path[t] === j) { w = 4; c = "var(--series-4)"; op = 1; }
        svg.append("line").attr("x1", x(t - 1) + 11).attr("y1", y(i)).attr("x2", x(t) - 11).attr("y2", y(j)).attr("stroke", c).attr("stroke-width", w).attr("opacity", op);
      }
      for (let t = 0; t < T; t++) {
        // die face
        const g = svg.append("g").attr("transform", `translate(${x(t)},${24})`).style("cursor", "pointer");
        g.append("rect").attr("x", -12).attr("y", -12).attr("width", 24).attr("height", 24).attr("rx", 5).attr("fill", seq.ob[t] === 5 ? "color-mix(in srgb, var(--series-2) 22%, var(--bg-raised))" : "var(--bg-raised)").attr("stroke", st.t === t && (st.stage === "fwd" || st.stage === "vit" || st.stage === "bwd") ? "var(--accent)" : "var(--border)").attr("stroke-width", 1.4);
        g.append("text").attr("text-anchor", "middle").attr("y", 4).style("font-size", "12px").style("font-weight", 600).attr("fill", "var(--text)").text(seq.ob[t] + 1);
        g.on("click", () => { seq.ob[t] = (seq.ob[t] + 1) % 6; rebuild({ at: "end" }); });
        for (let s = 0; s < 2; s++) {
          let v = null;
          if (st.stage === "fwd" && t <= st.t) v = R.F.al[t][s];
          else if (st.stage === "bwd") v = t >= st.t ? (() => { const b = R.Bk[t]; return b[s] / (b[0] + b[1]); })() : null;
          else if (st.stage === "post" || st.stage === "back") v = R.gamma[t][s];
          else if (st.stage === "vit" && t <= st.t) { const d = R.V.dl[t]; const m = Math.max(...d); v = Math.exp(d[s] - m) / (Math.exp(d[0] - m) + Math.exp(d[1] - m)); }
          const onPath = st.stage === "back" && t >= st.from && R.V.path[t] === s;
          svg.append("circle").attr("cx", x(t)).attr("cy", y(s)).attr("r", 11).attr("fill", v == null ? "var(--bg-inset)" : `color-mix(in srgb, ${col(s)} ${Math.round(v * 80 + 6)}%, var(--bg-inset))`).attr("stroke", onPath ? "var(--series-4)" : "var(--border)").attr("stroke-width", onPath ? 3 : 1);
          if (v != null && T <= 30) svg.append("text").attr("x", x(t)).attr("y", y(s) + 3).attr("text-anchor", "middle").style("font-size", "7.5px").attr("fill", "var(--text)").text(v.toFixed(2).replace(/^0/, ""));
        }
        svg.append("rect").attr("x", x(t) - 14).attr("y", H - 18).attr("width", 28).attr("height", 10).attr("rx", 2).attr("fill", col(seq.st[t])).attr("opacity", 0.75);
      }
      // posterior chart
      if (PP) {
        PP.g.selectAll("path.ln").remove();
        if (st.stage !== "model") PP.line("filt", R.F.al.slice(0, fwdUpto + 1).map((a, t) => [t + 1, a[1]]), { color: "var(--series-1)", dash: "4 3", curve: d3.curveLinear });
        if (st.stage === "post" || st.stage === "back") PP.line("post", R.gamma.map((g, t) => [t + 1, g[1]]), { color: "var(--series-2)", width: 2.4, curve: d3.curveLinear });
        if (st.stage === "back") PP.line("vit", R.V.path.map((s, t) => [t + 1, s ? 0.97 : 0.03]).filter((_, t) => t >= st.from), { color: "var(--series-4)", width: 2, curve: d3.curveStepAfter });
      }
      const accV = R.V.path.filter((s, t) => s === seq.st[t]).length / T;
      lab.readout(`<div class="kv"><span>rolls</span><b>${T}</b><span>sixes</span><b>${seq.ob.filter((o) => o === 5).length}</b><span>log P(rolls)</span><b>${fmt(R.F.ll, 2)}</b><span>Viterbi accuracy</span><b>${(accV * 100).toFixed(0)}%</b></div>`);
    }
    lab.watch(["p", "q", "six"], () => rebuild({ at: "end" }));
    lab.watch(["T"], () => regen());
    lab.q('[data-b="new"]').onclick = () => { seed = (seed * 37 + 11) % 99991; regen(); };
    regen();
    return lab.destroy;
  }

  MLApp.register({
    id: "hmm",
    name: "Hidden Markov Model",
    category: "Probabilistic & Sequential",
    tagline: "forward-backward & Viterbi on a trellis",
    isNew: true,
    description: "The dishonest casino: a hidden switch between a fair and a loaded die, seen only through the rolls. Watch the forward pass sweep the trellis, the backward pass return, posteriors combine both, and Viterbi trace the single most likely hidden path.",
    info: {
      type: "Probabilistic sequence model (generative, latent-variable). Inference by dynamic programming; learning by Baum-Welch (EM).",
      scenario: "Sequential data generated by an unobserved process that switches between a few regimes: speech, part-of-speech tagging, gene finding, regime detection in finance, activity recognition.",
      inputs: "An observation sequence x₁..x_T and model parameters (π, A, B), or many sequences to learn them from.",
      output: "Sequence likelihood, per-time state posteriors, and the most likely hidden state path.",
      intuition: {
        definition: "A hidden state evolves as a Markov chain (each state depends only on the previous one) and emits an observation at each step. We never see the states, only the emissions. Dynamic programming lets us reason about all exponentially many hidden paths in time linear in the sequence length.",
        steps: ["Forward: sweep left to right, accumulating the probability of the observations so far for each current state.", "Backward: sweep right to left with the probability of future observations.", "Posterior: multiply forward and backward messages to get P(state at t | everything).", "Viterbi: replace sum with max to find the single best path, then backtrack."],
        applications: ["Speech recognition (pre-deep-learning acoustic models)", "Part-of-speech tagging and named-entity recognition", "Gene and CpG-island finding in DNA", "Market regime and fault detection"],
      },
      math: [
        { title: "Joint probability", formula: "P(x, z) = π_{z₁} Πₜ A_{z_{t−1}z_t} Πₜ B_{z_t}(x_t)", note: "Markov transitions plus conditionally independent emissions." },
        { title: "Forward recursion", formula: "αₜ(j) = B_j(xₜ) Σᵢ αₜ₋₁(i) Aᵢⱼ", note: "O(T N²) instead of O(N^T). P(x) = Σⱼ α_T(j)." },
        { title: "Backward recursion", formula: "βₜ(i) = Σⱼ Aᵢⱼ B_j(xₜ₊₁) βₜ₊₁(j)", note: "Probability of the future given the present state." },
        { title: "Posterior (smoothing)", formula: "γₜ(j) = αₜ(j) βₜ(j) / P(x)", note: "Uses the whole sequence; also gives pairwise posteriors ξ for learning." },
        { title: "Viterbi", formula: "δₜ(j) = B_j(xₜ) maxᵢ δₜ₋₁(i) Aᵢⱼ", note: "Max-product version; back-pointers recover argmax_z P(z | x)." },
        { title: "Baum-Welch (EM)", formula: "Âᵢⱼ = Σₜ ξₜ(i,j) / Σₜ γₜ(i)", note: "E-step: forward-backward; M-step: re-estimate π, A, B from expected counts." },
      ],
      pipeline: [{ label: "Model π, A, B" }, { label: "Forward α" }, { label: "Backward β" }, { label: "Posterior γ" }, { label: "Viterbi path", accent: "green" }],
      decisionFunction: { text: "z* = argmax_z P(z | x)   (Viterbi)   or   ẑₜ = argmax_j γₜ(j)   (posterior decoding)", mechanism: "Viterbi maximises the probability of the whole path; posterior decoding maximises each position independently and minimises the expected number of wrong states." },
      lossFunction: { text: "Learning maximises log P(x | π, A, B) = log Σ_z P(x, z)", mechanism: "Baum-Welch is EM on this objective; each iteration never decreases the likelihood but converges to a local optimum, so initialisation matters." },
      optimization: [{ title: "Scaling", formula: "normalise αₜ each step, log P(x) = Σ log cₜ", note: "Raw α underflows after a few hundred steps." }, { title: "Log-space Viterbi", formula: "sums of logs instead of products", note: "Same reason." }, { title: "Multiple restarts", formula: "Baum-Welch from several inits", note: "Likelihood surface is multimodal." }],
      assumptions: [{ name: "Markov property", why: "State depends only on the previous state.", check: "If longer memory matters, use higher-order HMMs or RNNs." }, { name: "Conditional independence of emissions", why: "Each observation depends only on the current state.", check: "Correlated observations need richer emission models." }, { name: "Discrete, few states", why: "Cost is N² per step.", check: "Choose N by BIC or held-out likelihood." }],
      hyperparameters: [{ name: "number of states N", range: "2 - 50", increasing: "More expressive, more parameters, slower.", strategy: "Domain knowledge, BIC, or held-out likelihood." }, { name: "emission family", range: "categorical / Gaussian / GMM", increasing: "Not applicable", strategy: "Match the observation type." }, { name: "topology", range: "ergodic / left-right", increasing: "Not applicable", strategy: "Left-right for speech and processes with progression." }],
      metrics: ["Log-likelihood on held-out sequences", "State accuracy (if labels exist)", "BIC for model selection"],
      typicalUses: ["Sequence labelling", "Segmentation into regimes", "Anomaly detection by low likelihood"],
      diagnostics: ["Plot posteriors over time: confident switches vs ambiguous regions.", "Check learned transition matrix for near-zero rows (unused states).", "Compare Viterbi and posterior decoding paths."],
      advantages: ["Exact inference in linear time.", "Interpretable parameters (switching and emission probabilities).", "Handles variable-length sequences and missing data naturally.", "Generative: can sample and score sequences."],
      limitations: [{ name: "First-order memory", note: "cannot capture long-range dependencies", fix: "RNNs / transformers, or higher-order HMMs." }, { name: "Local optima in learning", note: "Baum-Welch is EM", fix: "multiple restarts, good initialisation." }, { name: "Independence assumptions", note: "emissions conditionally independent", fix: "CRFs for discriminative labelling." }],
      alternatives: [{ name: "Conditional Random Fields", when: "Supervised sequence labelling with rich overlapping features." }, { name: "RNN / LSTM", when: "Long-range dependencies and large data." }, { name: "Kalman filter", when: "Continuous hidden state with linear-Gaussian dynamics." }],
      pitfalls: [{ problem: "Probabilities underflow to 0", solution: "Scale α/β or work in log space." }, { problem: "Learned states have no meaning", solution: "EM found a different labelling or local optimum; constrain or initialise better." }, { problem: "Posterior path has impossible transitions", solution: "Use Viterbi when a consistent path is required." }],
      quickRef: [{ name: "Forward", formula: "αₜ(j) = B_j(xₜ) Σᵢ αₜ₋₁(i)Aᵢⱼ" }, { name: "Backward", formula: "βₜ(i) = Σⱼ Aᵢⱼ B_j(xₜ₊₁) βₜ₊₁(j)" }, { name: "Posterior", formula: "γ ∝ α β" }, { name: "Viterbi", formula: "δₜ(j) = B_j(xₜ) maxᵢ δₜ₋₁(i)Aᵢⱼ" }, { name: "Cost", formula: "O(T N²)" }],
      code: `from hmmlearn.hmm import CategoricalHMM
import numpy as np

hmm = CategoricalHMM(n_components=2, n_iter=100, random_state=0)
hmm.fit(rolls.reshape(-1, 1))                 # Baum-Welch
logp, path = hmm.decode(rolls.reshape(-1, 1)) # Viterbi
post = hmm.predict_proba(rolls.reshape(-1, 1))# forward-backward`,
      whyChain: [
        { q: "Why is the forward algorithm linear in T?", a: "The Markov property means the probability of being in state j at time t only needs the totals for each state at t−1, not the entire history. So each step costs N² regardless of how many paths led there." },
        { q: "Why do Viterbi and posterior decoding differ?", a: "Posterior decoding picks the most probable state at each time separately; the resulting sequence can include transitions with zero probability. Viterbi optimises the joint path probability." },
        { q: "How does Baum-Welch relate to EM for GMMs?", a: "Exactly the same idea: the E-step computes soft assignments of hidden variables (γ and ξ via forward-backward), the M-step re-estimates parameters as weighted counts. A GMM is an HMM with no transitions." },
      ],
      workedExample: {
        setup: "π = (0.5, 0.5); fair emits six with 1/6, loaded with 1/2. First roll is a six.",
        steps: ["α₁(Fair) = 0.5 · 1/6 = 0.083.", "α₁(Loaded) = 0.5 · 1/2 = 0.25.", "Normalise: P(Loaded | first roll) = 0.25 / 0.333 = 0.75."],
        result: "One six moves the belief in the loaded die from 50% to 75%",
      },
    },
    mount,
  });
})();
