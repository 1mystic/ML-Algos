(() => {
  const { ui, col, fmt } = Lab;
  const HID = 4;
  const sig = (v) => 1 / (1 + Math.exp(-v));
  const rv = (n, s) => Array.from({ length: n }, () => MLU.randn() * s);
  const rm = (n, m, s) => Array.from({ length: n }, () => rv(m, s));
  const mv = (M, v) => M.map((r) => r.reduce((s, w, j) => s + w * v[j], 0));

  function weights(type, rec, fbias) {
    MLU.seedRng(12);
    const gate = (b = 0) => ({ wx: rv(HID, 1.0), wh: rm(HID, HID, rec / Math.sqrt(HID)), b: rv(HID, 0.1).map((v) => v + b) });
    if (type === "lstm") return { f: gate(fbias), i: gate(), g: gate(), o: gate() };
    if (type === "gru") return { z: gate(), r: gate(), n: gate() };
    return { h: gate() };
  }
  const pre = (G, x, h) => mv(G.wh, h).map((v, k) => v + G.wx[k] * x + G.b[k]);
  function seq(kind, T) {
    if (kind === "pulse") return Array.from({ length: T }, (_, t) => (t === 0 ? 1 : 0));
    if (kind === "step") return Array.from({ length: T }, (_, t) => (t >= 3 ? 1 : 0));
    if (kind === "sine") return Array.from({ length: T }, (_, t) => +Math.sin(t * 0.8).toFixed(2));
    MLU.seedRng(5); return Array.from({ length: T }, () => (MLU.rng() < 0.5 ? 1 : 0));
  }

  function run(type, W, xs) {
    let h = new Array(HID).fill(0), c = new Array(HID).fill(0);
    const steps = [];
    let J = null; // running Jacobian (RNN) / scalar memory (gated)
    let mem = 1;
    xs.forEach((x, t) => {
      const s = { x, hPrev: h.slice(), cPrev: c.slice() };
      if (type === "lstm") {
        s.f = pre(W.f, x, h).map(sig); s.i = pre(W.i, x, h).map(sig); s.g = pre(W.g, x, h).map(Math.tanh); s.o = pre(W.o, x, h).map(sig);
        c = c.map((cv, k) => s.f[k] * cv + s.i[k] * s.g[k]); h = c.map((cv, k) => s.o[k] * Math.tanh(cv));
        if (t > 0) mem *= MLU.mean(s.f);
      } else if (type === "gru") {
        s.z = pre(W.z, x, h).map(sig); s.r = pre(W.r, x, h).map(sig);
        const rh = h.map((v, k) => v * s.r[k]);
        s.n = mv(W.n.wh, rh).map((v, k) => Math.tanh(v + W.n.wx[k] * x + W.n.b[k]));
        h = h.map((v, k) => (1 - s.z[k]) * s.n[k] + s.z[k] * v);
        if (t > 0) mem *= MLU.mean(s.z);
      } else {
        h = pre(W.h, x, h).map(Math.tanh);
        if (t > 0) {
          const D = h.map((v) => 1 - v * v);
          const Jt = W.h.wh.map((row, a) => row.map((w) => w * D[a]));
          J = J ? MLU.matMul(Jt, J) : Jt;
          mem = Math.sqrt(J.flat().reduce((q, v) => q + v * v, 0) / HID);
        }
      }
      s.h = h.slice(); s.c = c.slice(); s.mem = t === 0 ? 1 : mem;
      steps.push(s);
    });
    return steps;
  }

  const ARCH = {
    rnn: {
      w: 380, h: 190,
      nodes: [
        { id: "x", label: "xₜ", sub: "input", x: 40, y: 150, w: 50, kind: "data" },
        { id: "hp", label: "hₜ₋₁", sub: "previous state", x: 40, y: 50, kind: "param" },
        { id: "lin", label: "Wₓx + Wₕh + b", sub: "affine", x: 180, y: 100, kind: "op" },
        { id: "tanh", label: "tanh", x: 300, y: 100, w: 56, kind: "op" },
        { id: "h", label: "hₜ", sub: "new state", x: 300, y: 30, w: 66, kind: "out" },
      ],
      edges: [{ from: "x", to: "lin" }, { from: "hp", to: "lin" }, { from: "lin", to: "tanh" }, { from: "tanh", to: "h" }, { from: "h", to: "hp", dash: true, label: "next step" }],
    },
    lstm: {
      w: 380, h: 210,
      nodes: [
        { id: "x", label: "[hₜ₋₁, xₜ]", sub: "inputs", x: 46, y: 182, kind: "data" },
        { id: "cp", label: "cₜ₋₁", x: 34, y: 34, w: 48, kind: "param" },
        { id: "f", label: "f  σ", sub: "forget", x: 110, y: 120, w: 56, kind: "op", tex: "f=\\sigma(W_f[h,x]+b_f)" },
        { id: "i", label: "i  σ", sub: "input", x: 178, y: 120, w: 56, kind: "op", tex: "i=\\sigma(W_i[h,x]+b_i)" },
        { id: "g", label: "g  tanh", sub: "candidate", x: 246, y: 120, w: 62, kind: "op", tex: "g=\\tanh(W_g[h,x]+b_g)" },
        { id: "o", label: "o  σ", sub: "output", x: 322, y: 120, w: 56, kind: "op", tex: "o=\\sigma(W_o[h,x]+b_o)" },
        { id: "c", label: "c = f⊙c + i⊙g", sub: "cell state", x: 200, y: 34, kind: "loss" },
        { id: "h", label: "h = o⊙tanh(c)", sub: "hidden", x: 326, y: 70, kind: "out" },
      ],
      edges: [{ from: "x", to: "f" }, { from: "x", to: "i" }, { from: "x", to: "g" }, { from: "x", to: "o" }, { from: "cp", to: "c" }, { from: "f", to: "c" }, { from: "i", to: "c" }, { from: "g", to: "c" }, { from: "c", to: "h" }, { from: "o", to: "h" }],
    },
    gru: {
      w: 380, h: 200,
      nodes: [
        { id: "x", label: "[hₜ₋₁, xₜ]", sub: "inputs", x: 56, y: 166, kind: "data" },
        { id: "z", label: "z  σ", sub: "update gate", x: 70, y: 86, kind: "op", tex: "z=\\sigma(W_z[h,x]+b_z)" },
        { id: "r", label: "r  σ", sub: "reset gate", x: 180, y: 140, kind: "op", tex: "r=\\sigma(W_r[h,x]+b_r)" },
        { id: "n", label: "n  tanh", sub: "candidate", x: 200, y: 70, kind: "op", tex: "n=\\tanh(W_nx+U_n(r\\odot h)+b_n)" },
        { id: "h", label: "h = (1−z)⊙n + z⊙h", sub: "blend", x: 300, y: 24, kind: "out" },
      ],
      edges: [{ from: "x", to: "z" }, { from: "x", to: "r" }, { from: "r", to: "n" }, { from: "x", to: "n" }, { from: "z", to: "h" }, { from: "n", to: "h" }],
    },
  };

  function buildFrames(type, steps) {
    const frames = [{ phase: "init", major: true, arch: type === "rnn" ? ["hp"] : type === "lstm" ? ["cp"] : ["x"], title: "h₀ = 0" + (type === "lstm" ? ", c₀ = 0" : ""), text: "The same cell, with the same weights, is applied at every time step. Its state is the only thing carried forward: everything the network remembers must fit in it.", st: { t: -1 }, m: {} }];
    steps.forEach((s, t) => {
      const base = { x: s.x, t };
      const m = { h0: s.h[0], h1: s.h[1], mem: s.mem };
      if (type === "lstm") {
        frames.push({ phase: "forget", major: t === 0, arch: ["x", "f", "cp", "c"], title: `t = ${t + 1}: forget gate`, text: `How much of each cell unit to keep: f = (${s.f.map((v) => v.toFixed(2)).join(", ")}). Values near 1 preserve memory, near 0 erase it.`, tex: "f_t=\\sigma\\big(W_f[h_{t-1},x_t]+b_f\\big)", st: { ...base, show: ["f"] }, m });
        frames.push({ phase: "write", arch: ["x", "i", "g"], title: `t = ${t + 1}: input gate and candidate`, text: `g proposes new content (tanh, in [−1, 1]); i decides how much of it to write. Input xₜ = ${s.x}.`, tex: "i_t=\\sigma(W_i[h_{t-1},x_t]+b_i),\\quad g_t=\\tanh(W_g[h_{t-1},x_t]+b_g)", st: { ...base, show: ["f", "i", "g"] }, m });
        frames.push({ phase: "cell", arch: ["f", "i", "g", "c"], title: `t = ${t + 1}: update the cell state`, text: "The cell state changes only through element-wise multiply and add, no squashing. That additive path is the gradient highway that lets LSTMs remember over long spans.", tex: "c_t=f_t\\odot c_{t-1}+i_t\\odot g_t", st: { ...base, show: ["f", "i", "g", "c"] }, m });
        frames.push({ phase: "output", arch: ["o", "c", "h"], title: `t = ${t + 1}: output gate → hidden state`, text: `h = o ⊙ tanh(c) = (${s.h.map((v) => v.toFixed(2)).join(", ")}). The hidden state is a filtered view of the cell.`, tex: "o_t=\\sigma(W_o[h_{t-1},x_t]+b_o),\\quad h_t=o_t\\odot\\tanh(c_t)", st: { ...base, show: ["f", "i", "g", "c", "o", "h"] }, m });
      } else if (type === "gru") {
        frames.push({ phase: "gates", major: t === 0, arch: ["x", "z", "r"], title: `t = ${t + 1}: update and reset gates`, text: `z decides how much old state to keep, r how much of it the candidate may look at.`, tex: "z_t=\\sigma(W_z[h_{t-1},x_t]),\\quad r_t=\\sigma(W_r[h_{t-1},x_t])", st: { ...base, show: ["z", "r"] }, m });
        frames.push({ phase: "cand", arch: ["r", "n"], title: `t = ${t + 1}: candidate state`, text: "A tanh candidate built from the input and the reset-gated old state.", tex: "n_t=\\tanh\\big(W_nx_t+U_n(r_t\\odot h_{t-1})+b_n\\big)", st: { ...base, show: ["z", "r", "n"] }, m });
        frames.push({ phase: "blend", arch: ["z", "n", "h"], title: `t = ${t + 1}: blend old and new`, text: `h = (${s.h.map((v) => v.toFixed(2)).join(", ")}). When z ≈ 1 the state is copied through untouched, which is how GRUs carry memory.`, tex: "h_t=(1-z_t)\\odot n_t+z_t\\odot h_{t-1}", st: { ...base, show: ["z", "r", "n", "h"] }, m });
      } else {
        frames.push({ phase: "step", major: t === 0, arch: ["x", "hp", "lin", "tanh", "h"], title: `t = ${t + 1}: hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)`, text: `Input ${s.x}; new state (${s.h.map((v) => v.toFixed(2)).join(", ")}). Memory of t = 1 still present: <b>${s.mem.toExponential(1)}</b> (norm of ∂hₜ/∂h₁). Each step multiplies by Wₕ and a tanh slope ≤ 1, so this shrinks or explodes geometrically.`, tex: "h_t=\\tanh(W_xx_t+W_hh_{t-1}+b),\\qquad \\frac{\\partial h_t}{\\partial h_1}=\\prod_{k=2}^{t}\\operatorname{diag}(1-h_k^2)\\,W_h", st: { ...base, show: ["h"] }, m });
      }
    });
    const last = steps[steps.length - 1];
    frames.push({ phase: "done", major: true, arch: type === "lstm" ? ["c", "h"] : ["h"], title: `Sequence processed: memory of the first step = ${last.mem.toExponential(1)}`, text: type === "rnn" ? "A plain RNN's sensitivity to early inputs decays (or blows up) exponentially with distance: the vanishing / exploding gradient problem. Compare with the LSTM at a forget bias of +2." : type === "lstm" ? "With forget gates near 1, the cell carries information across many steps with almost no decay. Lower the forget bias to watch the memory fade." : "The update gate plays the role of the LSTM's forget gate with fewer parameters.", st: { x: last.x, t: steps.length - 1, show: ["f", "i", "g", "o", "c", "h", "z", "r", "n"] }, m: { h0: last.h[0], h1: last.h[1], mem: last.mem } });
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "rnn", fps: 3, plane: false,
      stageHTML: `<svg data-r="unroll" class="lab-svg" viewBox="0 0 760 250"></svg>`,
      legend: ui.legend([["var(--series-1)", "positive"], ["var(--series-2)", "negative"], ["var(--series-4)", "current step", "ring"]]),
      tools: ui.btn("replay", "replay", true),
      hint: "click an input bar to toggle it between 0 and 1 · the same weights are reused at every step",
      side: [
        { title: "cell", html: `<div class="seg" data-r="type" style="margin-bottom:12px"><button data-t="rnn">RNN</button><button data-t="lstm" class="on">LSTM</button><button data-t="gru">GRU</button></div>` + ui.range("fb", "forget-gate bias (LSTM)", -3, 4, 0.5, 2) + ui.range("rec", "recurrent weight scale", 0.2, 3, 0.1, 1.2) },
        { title: "input", html: ui.select("seq", "sequence", [["pulse", "single pulse at t = 1"], ["step", "step at t = 4"], ["sine", "sine wave"], ["bits", "random bits"]], "pulse") + ui.range("T", "length", 4, 16, 1, 10) },
      ],
      arch: ARCH.lstm, archTitle: "inside the cell",
      phases: { init: { label: "init", color: 3 }, forget: { label: "forget gate", color: 1 }, write: { label: "write", color: 0 }, cell: { label: "cell update", color: 3 }, output: { label: "output", color: 2 }, gates: { label: "gates", color: 1 }, cand: { label: "candidate", color: 0 }, blend: { label: "blend", color: 2 }, step: { label: "time step", color: 0 }, done: { label: "done", color: 2 } },
      panels: [{ id: "gates", title: "gate & state values at this step", sub: "rows = vectors, columns = 4 hidden units" }],
      charts: [{ id: "h", title: "hidden units over time", series: [{ key: "h0", label: "h[0]", color: "var(--series-1)" }, { key: "h1", label: "h[1]", color: "var(--series-5)" }] }, { id: "mem", title: "memory of t = 1 (gradient flow)", series: [{ key: "mem", label: "‖∂hₜ/∂h₁‖ (approx.)", color: "var(--series-2)", digits: 4 }], logY: true }],
      onFrame: render,
    });
    let type = "lstm", xs = seq("pulse", 10), steps = null;
    function rebuildArch() {
      const el = lab.R("arch"); el.innerHTML = "";
      lab.arch = Lab.arch(el, ARCH[type], { onNode: (nid) => { const fr = lab.player.frames; const j = fr.findIndex((f, k) => k > lab.player.index && (f.arch || []).includes(nid)); if (j >= 0) { lab.player.pause(); lab.player.show(j); } } });
    }
    function replay({ at = "start" } = {}) {
      lab.ctl("fb").closest(".field").style.opacity = type === "lstm" ? 1 : 0.4;
      steps = run(type, weights(type, lab.val("rec"), lab.val("fb")), xs);
      lab.load(buildFrames(type, steps), { at, play: at === "start" });
    }
    lab.R("type").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; type = b.dataset.t; lab.R("type").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); rebuildArch(); replay(); };
    const gm = Lab.matrix(lab.R("panel-gates"), { cell: 30, colLabels: ["u1", "u2", "u3", "u4"], labelW: 52, range: [-1, 1] });
    const svg = d3.select(lab.R("unroll"));
    function render(f) {
      const st = f.st, T = xs.length;
      svg.selectAll("*").remove();
      const W = 760, dx = Math.min(70, (W - 60) / T), x0 = 40;
      const cx = (t) => x0 + dx * t + dx / 2;
      const bar = (g, v, x, y, w, h) => g.append("rect").attr("x", x).attr("y", v >= 0 ? y - (h * Math.min(1, v)) : y).attr("width", w).attr("height", Math.max(0.5, h * Math.min(1, Math.abs(v)))).attr("fill", v >= 0 ? "var(--series-1)" : "var(--series-2)").attr("rx", 1);
      svg.append("text").attr("class", "lbl").attr("x", 4).attr("y", 215).text("xₜ");
      svg.append("text").attr("class", "lbl").attr("x", 4).attr("y", 52).text("hₜ");
      if (type === "lstm") svg.append("text").attr("class", "lbl").attr("x", 4).attr("y", 18).text("cₜ");
      for (let t = 0; t < T; t++) {
        const on = t === st.t, past = t <= st.t;
        // input bar
        const gx = svg.append("g").style("cursor", "pointer").on("click", () => { xs[t] = xs[t] ? 0 : 1; replay({ at: "end" }); });
        gx.append("rect").attr("x", cx(t) - 14).attr("y", 186).attr("width", 28).attr("height", 40).attr("fill", "transparent");
        gx.append("line").attr("x1", cx(t) - 14).attr("x2", cx(t) + 14).attr("y1", 222).attr("y2", 222).attr("stroke", "var(--border)");
        bar(gx, xs[t], cx(t) - 8, 222, 16, 34);
        svg.append("line").attr("x1", cx(t)).attr("x2", cx(t)).attr("y1", 184).attr("y2", 150).attr("stroke", past ? "var(--text-faint)" : "var(--border)");
        // cell box
        svg.append("rect").attr("x", cx(t) - dx * 0.36).attr("y", 104).attr("width", dx * 0.72).attr("height", 44).attr("rx", 7).attr("fill", on ? "color-mix(in srgb, var(--series-4) 16%, var(--bg-raised))" : past ? "var(--bg-raised)" : "var(--bg-inset)").attr("stroke", on ? "var(--series-4)" : "var(--border)").attr("stroke-width", on ? 2.4 : 1);
        svg.append("text").attr("x", cx(t)).attr("y", 130).attr("text-anchor", "middle").style("font-size", "10px").attr("fill", past ? "var(--text)" : "var(--text-faint)").text(type.toUpperCase());
        if (t < T - 1) svg.append("line").attr("x1", cx(t) + dx * 0.36).attr("x2", cx(t + 1) - dx * 0.36).attr("y1", 126).attr("y2", 126).attr("stroke", t < st.t ? "var(--text-dim)" : "var(--border)").attr("stroke-width", 1.6);
        // hidden state bars
        if (past && steps[t]) {
          const s = steps[t];
          s.h.forEach((v, k) => bar(svg, v, cx(t) - 16 + k * 8, 76, 6, 34));
          svg.append("line").attr("x1", cx(t) - 18).attr("x2", cx(t) + 18).attr("y1", 76).attr("y2", 76).attr("stroke", "var(--border)");
          if (type === "lstm") { s.c.forEach((v, k) => bar(svg, Math.tanh(v), cx(t) - 16 + k * 8, 26, 6, 18)); svg.append("line").attr("x1", cx(t) - 18).attr("x2", cx(t) + 18).attr("y1", 26).attr("y2", 26).attr("stroke", "var(--border)"); }
        }
        svg.append("text").attr("x", cx(t)).attr("y", 243).attr("text-anchor", "middle").style("font-size", "9px").attr("fill", "var(--text-faint)").text(`t=${t + 1}`);
      }
      if (type === "lstm") svg.append("line").attr("x1", x0).attr("x2", cx(Math.max(0, st.t)) + 18).attr("y1", 32).attr("y2", 32).attr("stroke", "var(--series-3)").attr("stroke-width", 1.4).attr("stroke-dasharray", "4 3").attr("opacity", st.t >= 0 ? 0.7 : 0);
      // gate matrix
      const s = st.t >= 0 ? steps[st.t] : null;
      const keys = type === "lstm" ? ["f", "i", "g", "o", "c", "h"] : type === "gru" ? ["z", "r", "n", "h"] : ["h"];
      const rows = keys.map((k) => (s && (st.show || []).includes(k) ? s[k].map((v) => (k === "c" ? Math.tanh(v) : v)) : [NaN, NaN, NaN, NaN]));
      gm.set(rows.map((r) => r.map((v) => (isFinite(v) ? v : 0))), { rowL: keys.map((k) => (k === "c" ? "tanh(c)" : k)), range: [-1, 1] });
      lab.readout(`<div class="kv"><span>cell</span><b>${type.toUpperCase()}</b><span>parameters</span><b>${(type === "lstm" ? 4 : type === "gru" ? 3 : 1) * (HID * HID + HID + HID)}</b><span>step</span><b>${Math.max(0, st.t + 1)} / ${T}</b><span>memory of t = 1</span><b>${s ? s.mem.toExponential(1) : "–"}</b></div><div class="note">Weights are fixed random draws (seeded), not trained: the point is the mechanics of the recurrence.</div>`);
    }
    lab.watch(["fb", "rec"], () => replay({ at: "end" }));
    lab.watch(["seq", "T"], () => { xs = seq(lab.val("seq"), lab.val("T")); replay(); });
    lab.q('[data-b="replay"]').onclick = () => replay();
    replay();
    return lab.destroy;
  }

  MLApp.register({
    id: "rnn-lstm",
    name: "RNN / LSTM Cell",
    category: "Deep Learning",
    tagline: "unrolled RNN / LSTM / GRU, gate by gate",
    description: "Unroll a recurrent cell over a sequence and step through every gate: forget, input, candidate, cell update and output for the LSTM, the two gates of the GRU, or the plain tanh RNN, while a gradient-flow chart shows why plain RNNs forget.",
    info: {
      type: "Building block of sequence models - recurrent, parametric, stateful (vanilla RNN or LSTM cell).",
      scenario: "Sequence data where order and (for LSTM) long-range dependencies matter - text, time series, audio - anywhere a fixed-size input vector isn't natural.",
      inputs: "A sequence x₁, …, x_T processed one timestep at a time, carrying forward a hidden state (and, for LSTM, a cell state).",
      intuition: {
        definition: "Process a sequence one step at a time, carrying a <b>hidden state</b> that summarises everything seen so far. The same weights are reused at every step. An LSTM adds a separate cell state plus three gates that decide what to forget, what to write, and what to expose, which is how it remembers across hundreds of steps.",
        steps: [
          "Combine the current input with the previous hidden state.",
          "Squash the result to produce the new hidden state.",
          "Reuse identical weights at every timestep.",
          "LSTM gates the flow so gradients can survive long spans.",
        ],
        applications: [
          "Language modelling and text generation before transformers",
          "Time-series forecasting for demand, sensors, and finance",
          "Speech recognition and audio processing",
          "Sequence labelling such as named-entity recognition",
          "Anomaly detection in streaming telemetry",
        ],
      },
      math: [
        { title: "Vanilla RNN step", formula: "hₜ = tanh(W_x·xₜ + W_h·hₜ₋₁ + b)", note: "One matrix for the input, one for the recurrent connection. The state is a lossy running summary of the entire prefix." },
        { title: "Why gradients vanish", formula: "∂h_T/∂h_t = Π_{k=t+1}^{T} W_hᵀ · diag(tanh'(·))", note: "A product of T−t terms. If the recurrent weight's spectral radius is below 1 the product decays exponentially; above 1 it explodes. Either way, learning across long spans fails." },
        { title: "LSTM gates", formula: "fₜ = σ(W_f·[hₜ₋₁, xₜ] + b_f)   iₜ = σ(W_i·…)   oₜ = σ(W_o·…)", note: "Three sigmoid gates producing soft switches in [0,1]: forget, input, and output. Each is a learned function of the current input and previous state." },
        { title: "Cell state update", formula: "g̃ₜ = tanh(W_g·[hₜ₋₁, xₜ] + b_g),   cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ g̃ₜ", note: "The crucial line. The old cell state is scaled and added to, not passed through a matrix multiply and a squashing function. That additive path is what preserves gradients." },
        { title: "Hidden output", formula: "hₜ = oₜ ⊙ tanh(cₜ)", note: "The cell state is the long-term memory; the hidden state is a filtered view of it exposed to the rest of the network." },
        { title: "Constant error carousel", formula: "∂cₜ/∂cₜ₋₁ = fₜ", note: "When the forget gate is near 1, the gradient passes through essentially unchanged. This is the whole reason LSTMs work, and why forget-gate bias is often initialised to 1." },
        { title: "GRU simplification", formula: "merges forget and input into one update gate, drops the separate cell state", note: "Roughly 25% fewer parameters, usually comparable accuracy, and faster to train." },
      ],
      pipeline: [
        { label: "xₜ + hₜ₋₁", note: "input and state" },
        { label: "Gates", note: "f, i, o via σ" },
        { label: "Candidate g̃ₜ", note: "tanh" },
        { label: "cₜ = f·cₜ₋₁ + i·g̃", note: "additive memory" },
        { label: "hₜ = o·tanh(cₜ)", note: "next step", accent: "green" },
      ],
      decisionFunction: {
        text: "RNN: hₜ=tanh(Wₓxₜ+W_hhₜ₋₁+b)   LSTM: fₜ,iₜ,oₜ=σ(·), gₜ=tanh(·), cₜ=fₜ·cₜ₋₁+iₜ·gₜ, hₜ=oₜ·tanh(cₜ)",
        mechanism: "The same small set of weights is reapplied at every timestep, using the previous hidden (and cell) state as extra input. LSTM's three gates learn when to forget old information, write new information, and expose the cell state - letting it retain information far longer than a vanilla RNN before gradients vanish.",
        plot: { fn: (z) => 1 / (1 + Math.exp(-z)), domain: [-6, 6], color: "var(--accent)", caption: "the sigmoid squashing every LSTM gate into a soft [0,1] switch - near 0 blocks information flow, near 1 passes it through" },
      },
      lossFunction: {
        text: "Task-dependent (cross-entropy for next-token prediction, MSE for sequence regression), summed over timesteps",
        mechanism: "Trained via backpropagation through time (BPTT) - the chain rule is unrolled across every timestep, which is exactly why vanishing/exploding gradients are a bigger problem here than in feed-forward nets, and exactly what LSTM's gates were designed to mitigate.",
      },
      optimization: [
        { title: "Backpropagation through time", formula: "unroll the network across T steps, then backpropagate normally", note: "An RNN unrolled is just a very deep feed-forward network with tied weights. Every classical depth problem therefore appears, but worse, because the same matrix is applied repeatedly." },
        { title: "Truncated BPTT", formula: "backpropagate only k steps, carry the state forward without gradient", note: "Full BPTT over a long sequence is memory-prohibitive. Truncation caps cost at the price of not learning dependencies longer than k." },
        { title: "Gradient clipping", formula: "if ‖g‖ > c then g := c·g/‖g‖", note: "Essential in practice. Exploding gradients are common and clipping the norm is a cheap, reliable fix." },
        { title: "Forget-gate bias init", formula: "b_f := 1", note: "Starts the forget gate open so information flows by default. A small change that noticeably improves long-range learning early in training." },
        { title: "Sequential bottleneck", formula: "step t cannot start before step t−1 finishes", note: "The fundamental limitation. Unlike a CNN or transformer, an RNN cannot parallelise across the time dimension, which is what ultimately made transformers win." },
      ],
      output: "A hidden state hₜ at every timestep, used directly or fed into a further output layer for predictions.",
      assumptions: [
        { name: "Order matters", why: "The entire architecture exists to model sequential dependency.", check: "If shuffling the sequence does not change the label, you do not need an RNN." },
        { name: "Same dynamics at every timestep", why: "Weights are shared across time, which assumes the generating process is stationary.", check: "For strongly regime-dependent series, add time features or segment the data." },
        { name: "Dependencies fit the memory horizon", why: "Even LSTMs degrade well before a thousand steps.", check: "If the span is very long, use attention." },
        { name: "Inputs are normalised", why: "Unscaled inputs destabilise recurrent dynamics badly, since errors compound across steps.", check: "Standardize, and scale targets for regression." },
        { name: "Sequential compute is acceptable", why: "Training cannot be parallelised across time.", check: "For long sequences and large data, a transformer will train far faster." },
      ],
      regularization: [
        { name: "Gradient clipping", formula: "clip ‖g‖ to c, typically 1 to 5", note: "Not optional. Recurrent nets explode frequently and this is the standard remedy." },
        { name: "Recurrent dropout", formula: "same dropout mask at every timestep", note: "Applying an independent mask each step injects noise that compounds and destroys memory. The shared-mask variational form is the correct one." },
        { name: "Dropout between layers", formula: "applied on the vertical connections only", note: "Safe on stacked RNN layers, unlike dropout on the recurrent connection itself." },
        { name: "Weight decay", formula: "L + λ‖W‖²", note: "Standard, applied to input and recurrent matrices." },
        { name: "Layer normalisation", formula: "normalise across features within each timestep", note: "Batch norm does not fit recurrence well because sequence lengths vary. Layer norm is the recurrent equivalent." },
        { name: "Teacher forcing", formula: "feed the true previous token during training", note: "Speeds convergence but creates exposure bias, since inference must use the model's own predictions. Scheduled sampling mitigates it." },
      ],
      hyperparameters: [
        { name: "hidden size", range: "32 - 1024", increasing: "More memory capacity, quadratically more recurrent parameters, more overfitting.", strategy: "128 to 256 is a solid starting range. Recurrent weight count grows with the square of hidden size." },
        { name: "cell type", range: "RNN / LSTM / GRU", increasing: "Not applicable", strategy: "Never use a vanilla RNN for anything beyond a demonstration. GRU first, since it is faster and usually equal; LSTM if GRU underperforms." },
        { name: "num_layers", range: "1 - 4", increasing: "More abstraction, harder training, slower.", strategy: "Two layers is a common sweet spot. Beyond three, gains are rare without residual connections." },
        { name: "sequence length / BPTT window", range: "20 - 500", increasing: "Longer dependencies learnable, more memory and worse gradient flow.", strategy: "Set from the domain: how far back does the signal genuinely reach? Then truncate to fit memory." },
        { name: "gradient clip norm", range: "0.5 - 10", increasing: "Less aggressive clipping.", strategy: "Start at 1.0. If loss spikes appear, lower it." },
        { name: "learning rate", range: "1e-4 - 1e-2", increasing: "Faster, then unstable. RNNs tolerate less than feed-forward nets.", strategy: "1e-3 with Adam is standard. Use a decay schedule." },
        { name: "dropout", range: "0 - 0.5", increasing: "More regularization, slower convergence.", strategy: "0.2 to 0.3 between layers. Use the variational form on recurrent connections." },
        { name: "bidirectional", range: "true / false", increasing: "Doubles parameters, gives access to future context.", strategy: "Only valid when the whole sequence is available up front. Never for real-time forecasting." },
      ],
      metrics: ["Perplexity / cross-entropy (language modeling)", "MSE / MAE (time-series forecasting)", "Accuracy / F1 (sequence classification)", "BLEU or ROUGE (sequence generation)"],
      typicalUses: ["Language modeling", "Machine translation (pre-Transformer era)", "Time-series forecasting", "Speech recognition"],
      diagnostics: [
        "Monitor the gradient norm across training. Repeated spikes mean clipping is doing real work; a norm decaying toward zero means vanishing gradients.",
        "Inspect the forget-gate activations. Values saturated near zero mean the cell is discarding memory every step and behaving like a vanilla RNN.",
        "Test whether the model actually uses long context by truncating the input history and checking whether performance drops. Often it does not.",
        "Watch for sudden loss spikes, the signature of exploding gradients. Lower the clip threshold.",
        "For forecasting, compare against a naive persistence baseline. RNNs frequently fail to beat 'predict the last value'.",
      ],
      advantages: [
        "Handles variable-length sequences naturally, with no fixed input size.",
        "Parameter count is independent of sequence length thanks to weight sharing.",
        "LSTM and GRU gating genuinely captures dependencies across hundreds of steps.",
        "Constant memory per step at inference, which suits streaming and real-time use.",
        "Well suited to online and incremental prediction, where a transformer must reprocess context.",
      ],
      limitations: [
        { name: "Cannot parallelise across time", note: "each step waits for the previous one", fix: "transformers, or convolutional sequence models." },
        { name: "Vanishing and exploding gradients", note: "inherent to repeated matrix application", fix: "LSTM/GRU gating, clipping, careful initialisation." },
        { name: "Limited effective memory", note: "even LSTMs degrade over very long spans", fix: "attention mechanisms." },
        { name: "Sequential inference latency", note: "generating n tokens takes n sequential steps", fix: "unavoidable for autoregressive models; use smaller models or caching." },
        { name: "Hard to train", note: "sensitive to learning rate, initialisation, and clipping", fix: "use well-tested recipes rather than tuning from scratch." },
        { name: "Superseded for most NLP", note: "transformers dominate", fix: "use RNNs where streaming or small footprint matters." },
      ],
      alternatives: [
        { name: "Transformer", when: "Long sequences, plentiful data, and parallel training matters. The default for NLP now." },
        { name: "Temporal convolutional network", when: "Fixed-horizon sequence modelling with parallel training and stable gradients." },
        { name: "Classical time series (ARIMA, ETS)", when: "Short univariate series with clear seasonality. Often beats deep models." },
        { name: "Gradient boosting on lag features", when: "Tabular forecasting. Frequently outperforms RNNs and is far cheaper." },
      ],
      pitfalls: [
        { problem: "Loss spikes to a huge value then recovers", solution: "Exploding gradients. Add or tighten gradient clipping." },
        { problem: "Model ignores distant context", solution: "Vanishing gradients or too small a BPTT window. Use LSTM/GRU, initialise forget bias to 1, widen the window." },
        { problem: "Dropout on recurrent connections destroys performance", solution: "Independent per-step masks compound noise. Use variational recurrent dropout with a shared mask." },
        { problem: "Great training loss, poor generation", solution: "Exposure bias from teacher forcing. Try scheduled sampling." },
        { problem: "Training is very slow", solution: "Inherent to sequential processing. Use cuDNN-fused layers, or switch to a transformer." },
        { problem: "Bidirectional model fails in production", solution: "It needs the full sequence, which real-time inference does not have. Use a unidirectional model." },
      ],
      quickRef: [
        { name: "RNN step", formula: "hₜ = tanh(W_x xₜ + W_h hₜ₋₁ + b)" },
        { name: "Forget gate", formula: "fₜ = σ(W_f[hₜ₋₁, xₜ] + b_f)" },
        { name: "Input gate", formula: "iₜ = σ(W_i[hₜ₋₁, xₜ] + b_i)" },
        { name: "Candidate", formula: "g̃ₜ = tanh(W_g[hₜ₋₁, xₜ] + b_g)" },
        { name: "Cell update", formula: "cₜ = fₜ⊙cₜ₋₁ + iₜ⊙g̃ₜ" },
        { name: "Hidden output", formula: "hₜ = oₜ ⊙ tanh(cₜ)" },
        { name: "Gradient path", formula: "∂cₜ/∂cₜ₋₁ = fₜ" },
        { name: "Clipping", formula: "g := c·g/‖g‖ if ‖g‖ > c" },
      ],
      code: `import torch, torch.nn as nn

class Forecaster(nn.Module):
    def __init__(self, n_features, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers,
            batch_first=True,
            dropout=0.2,          # applies between layers, not recurrently
            bidirectional=False,  # never bidirectional for live forecasting
        )
        self.head = nn.Linear(hidden, 1)
        # Open the forget gate at init so memory flows by default.
        for name, p in self.lstm.named_parameters():
            if "bias_ih" in name:
                n = p.shape[0] // 4
                p.data[n:2 * n].fill_(1.0)   # forget-gate slice

    def forward(self, x):
        out, _ = self.lstm(x)         # (batch, time, hidden)
        return self.head(out[:, -1])  # predict from the last step

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for xb, yb in loader:
    opt.zero_grad()
    nn.functional.mse_loss(model(xb), yb).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # not optional
    opt.step()`,
      whyChain: [
        { q: "Why do gradients vanish in a vanilla RNN?", a: "Backpropagating from step T to step t multiplies together T−t copies of the recurrent Jacobian. Each includes the recurrent weight matrix and a tanh derivative bounded by 1. If the resulting factor is below 1, the product decays exponentially, so early steps receive essentially no learning signal." },
        { q: "How does the LSTM cell state fix that?", a: "The update is cₜ = fₜ·cₜ₋₁ + iₜ·g̃ₜ, which is additive rather than a matrix multiply through a squashing function. The gradient of cₜ with respect to cₜ₋₁ is just the forget gate. When that gate is near 1, gradient flows back essentially unattenuated, which is the constant error carousel." },
        { q: "What does each LSTM gate actually decide?", a: "The forget gate decides how much of the old cell state to retain, the input gate how much of the new candidate to write, and the output gate how much of the cell state to expose as the hidden output. Each is a learned, input-dependent soft switch." },
        { q: "Why initialise the forget-gate bias to 1?", a: "A sigmoid at bias zero outputs 0.5, so the cell halves its memory every step and it decays geometrically before the model learns otherwise. Biasing toward 1 opens the gate at initialisation so information persists by default, and the model learns when to forget." },
        { q: "Why is a GRU often preferred over an LSTM?", a: "It merges the forget and input gates into one update gate and drops the separate cell state, giving roughly 25% fewer parameters and faster training. Accuracy is usually indistinguishable, so it is the cheaper default." },
        { q: "Why did transformers replace RNNs?", a: "Parallelism. An RNN must process step t before step t+1, so training time scales with sequence length and cannot use GPU parallelism across time. Self-attention sees all positions at once, trains in parallel, and gives every position direct access to every other rather than through a compressed state." },
        { q: "Why is gradient clipping essentially mandatory?", a: "The same recurrent matrix is applied repeatedly, so if its largest singular value exceeds 1 the gradient grows exponentially with sequence length. A single exploding batch can destroy the weights. Clipping the norm caps the step while preserving direction." },
        { q: "Why is per-step independent dropout harmful in an RNN?", a: "Fresh noise at every timestep compounds across the sequence and corrupts the hidden state's ability to carry information. Variational dropout uses the same mask at every step, so the noise is consistent and memory survives." },
      ],
      parameters: [
        { name: "hidden size", effect: "Memory capacity. More units can represent richer state but cost more compute and can overfit short sequences." },
        { name: "cell type", effect: "Vanilla RNN (simpler, faster) vs LSTM (more parameters, much better long-range memory and gradient stability)." },
        { name: "sequence length trained on", effect: "How far back the model is asked to remember during training - longer sequences stress gradient flow more." },
      ],
      metrics: ["Perplexity / cross-entropy (language modeling)", "MSE (time-series forecasting)", "Accuracy (sequence classification)"],
      typicalUses: ["Language modeling", "Machine translation (pre-Transformer era)", "Time-series forecasting", "Speech recognition"],
      workedExample: {
        setup: "Vanilla RNN, scalar hidden unit. h₀=0, Wx=0.5, Wh=0.8, b=0. Feed x₁=1 then x₂=0.",
        steps: [
          "Step 1: h₁ = tanh(0.5×1 + 0.8×0 + 0) = tanh(0.5) ≈ 0.4621.",
          "Step 2: h₂ = tanh(0.5×0 + 0.8×0.4621 + 0) = tanh(0.3697) ≈ 0.3538.",
        ],
        result: "Hidden state decays from 0.4621 toward 0 once the input stops - with no gating, memory of x₁ fades geometrically each step",
      },
    },
    mount,
  });
})();
