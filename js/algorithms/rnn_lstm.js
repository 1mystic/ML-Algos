(() => {
  const HID = 4;
  const sigmoid = (v) => 1 / (1 + Math.exp(-v));
  const tanh = Math.tanh;

  function randVec(n, scale) { return Array.from({ length: n }, () => MLU.randn() * scale); }
  function randMat(n, m, scale) { return Array.from({ length: n }, () => randVec(m, scale)); }
  function matVec(M, v) { return M.map((row) => row.reduce((s, w, j) => s + w * v[j], 0)); }
  function addV(...vs) { return vs[0].map((_, i) => vs.reduce((s, v) => s + v[i], 0)); }

  function makeWeights(hasGates) {
    const gates = hasGates ? ["f", "i", "o", "g"] : ["h"];
    const w = {};
    for (const g of gates) w[g] = { wx: randVec(HID, 0.9), wh: randMat(HID, HID, 0.4), b: randVec(HID, 0.2) };
    return w;
  }

  function stepLSTM(w, xt, hPrev, cPrev) {
    const gate = (name, act) => {
      const z = addV(w[name].wx.map((wx) => wx * xt), matVec(w[name].wh, hPrev), w[name].b);
      return z.map(act);
    };
    const f = gate("f", sigmoid), i = gate("i", sigmoid), o = gate("o", sigmoid), g = gate("g", tanh);
    const c = f.map((fv, k) => fv * cPrev[k] + i[k] * g[k]);
    const h = c.map((cv, k) => o[k] * tanh(cv));
    return { f, i, o, g, c, h };
  }
  function stepRNN(w, xt, hPrev) {
    const z = addV(w.h.wx.map((wx) => wx * xt), matVec(w.h.wh, hPrev), w.h.b);
    const h = z.map(tanh);
    return { h };
  }

  function bars(svgSel, values, range, color, w, h) {
    const n = values.length;
    const bw = w / n;
    const sel = svgSel.selectAll("rect").data(values);
    const zeroY = range[0] < 0 ? h / 2 : h;
    sel.enter().append("rect").attr("width", bw - 4)
      .merge(sel)
      .attr("x", (d, i) => i * bw + 2)
      .attr("fill", color)
      .attr("y", (d) => (d >= 0 ? zeroY - (d / range[1]) * (h / (range[0] < 0 ? 2 : 1)) : zeroY))
      .attr("height", (d) => Math.max(1, Math.abs(d / range[1]) * (h / (range[0] < 0 ? 2 : 1))));
    sel.exit().remove();
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint">timestep <b id="rnn-t">0</b> / <span id="rnn-len">10</span></span></div>
            <div class="btn-row">
              <button id="rnn-step" class="primary">step →</button>
              <button id="rnn-reset">reset</button>
              <button id="rnn-rand">randomize weights</button>
            </div>
          </div>
          <div style="padding:10px 4px">
            <div class="stage-hint" style="margin-bottom:6px">input sequence - click a step to toggle 0 / 1</div>
            <div id="rnn-seq" class="hscroll" style="display:flex;gap:4px;margin-bottom:16px"></div>
            <div id="rnn-gates" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:14px"></div>
            <div class="stage-hint" style="margin:16px 0 6px">hidden state h over time (unit 0)</div>
            <svg id="rnn-trace" width="100%" height="80" viewBox="0 0 600 80" preserveAspectRatio="none"></svg>
          </div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>cell type</h3>
            <select id="rnn-type"><option value="lstm">LSTM</option><option value="rnn">vanilla RNN</option></select>
          </div>
          <div class="control-card">
            <h3>mechanics</h3>
            <div class="note" id="rnn-note"></div>
          </div>
        </div>
      </div>
    `;

    const LSTM_NOTE = "At each step the LSTM computes forget (f), input (i) and output (o) gates plus a candidate (g), then updates the cell state c<sub>t</sub>=f&middot;c<sub>t-1</sub>+i&middot;g and hidden state h<sub>t</sub>=o&middot;tanh(c<sub>t</sub>) - the gating equations behind <code>mla/neuralnet</code>'s LSTM layer, run here with fixed random weights so you can watch the mechanics without a training loop.";
    const RNN_NOTE = "A vanilla RNN collapses all of that into one update: h<sub>t</sub> = tanh(W<sub>x</sub>x<sub>t</sub> + W<sub>h</sub>h<sub>t-1</sub> + b). Compare its trace to the LSTM's - notice how much faster it can saturate or forget.";

    let seq = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1];
    let weights = { lstm: makeWeights(true), rnn: makeWeights(false) };
    let t = 0, h = new Array(HID).fill(0), c = new Array(HID).fill(0);
    let trace = [];
    let lastGates = {};

    function type() { return document.getElementById("rnn-type").value; }

    function renderSeq() {
      const box = document.getElementById("rnn-seq");
      box.innerHTML = "";
      seq.forEach((v, i) => {
        const b = document.createElement("button");
        b.textContent = v;
        b.style.width = "30px";
        if (i === t) b.classList.add("primary");
        b.addEventListener("click", () => { seq[i] = seq[i] ? 0 : 1; renderSeq(); });
        box.appendChild(b);
      });
      document.getElementById("rnn-len").textContent = seq.length;
    }

    function renderGates() {
      const box = document.getElementById("rnn-gates");
      box.innerHTML = "";
      const specs = type() === "lstm"
        ? [["forget f", "f", [0, 1], "#fca5a5"], ["input i", "i", [0, 1], "#86efac"], ["output o", "o", [0, 1], "#7dd3fc"], ["candidate g", "g", [-1, 1], "#fcd34d"], ["cell c", "c", [-2, 2], "#c4b5fd"], ["hidden h", "h", [-1, 1], "#f5f5f4"]]
        : [["hidden h", "h", [-1, 1], "#f5f5f4"]];
      const last = lastGates || {};
      for (const [label, key, range, color] of specs) {
        const wrap = document.createElement("div");
        wrap.innerHTML = `<div class="stage-hint" style="margin-bottom:4px">${label}</div>`;
        const svg = d3.select(document.createElementNS(MLU.NS, "svg"));
        svg.attr("width", "100%").attr("height", 54).attr("viewBox", "0 0 120 54");
        wrap.appendChild(svg.node());
        box.appendChild(wrap);
        const vals = key === "h" ? h : (last[key] || new Array(HID).fill(0));
        bars(svg, vals, range, color, 120, 54);
      }
    }

    function renderTrace() {
      const svg = d3.select("#rnn-trace");
      svg.selectAll("*").remove();
      if (trace.length < 2) return;
      const xs = d3.scaleLinear().domain([0, seq.length - 1]).range([4, 596]);
      const ys = d3.scaleLinear().domain([-1, 1]).range([76, 4]);
      const line = d3.line().x((d, i) => xs(i)).y((d) => ys(d));
      svg.append("path").attr("d", line(trace)).attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2);
      svg.selectAll("circle").data(trace).enter().append("circle").attr("cx", (d, i) => xs(i)).attr("cy", (d) => ys(d)).attr("r", 3).attr("fill", "var(--accent)");
    }

    function render() {
      document.getElementById("rnn-t").textContent = t;
      document.getElementById("rnn-note").innerHTML = type() === "lstm" ? LSTM_NOTE : RNN_NOTE;
      renderSeq();
      renderGates();
      renderTrace();
    }

    function reset() {
      t = 0; h = new Array(HID).fill(0); c = new Array(HID).fill(0); trace = []; lastGates = {};
      render();
    }

    document.getElementById("rnn-step").addEventListener("click", () => {
      if (t >= seq.length) return;
      const xt = seq[t] ? 1 : -1;
      if (type() === "lstm") {
        const out = stepLSTM(weights.lstm, xt, h, c);
        h = out.h; c = out.c; lastGates = out;
      } else {
        const out = stepRNN(weights.rnn, xt, h);
        h = out.h; lastGates = out;
      }
      trace.push(h[0]);
      t++;
      render();
    });
    document.getElementById("rnn-reset").addEventListener("click", reset);
    document.getElementById("rnn-rand").addEventListener("click", () => { weights = { lstm: makeWeights(true), rnn: makeWeights(false) }; reset(); });
    document.getElementById("rnn-type").addEventListener("change", reset);

    reset();
    return () => {};
  }

  MLApp.register({
    id: "rnn-lstm",
    name: "RNN / LSTM Cell",
    category: "Deep Learning",
    tagline: "step through the gate equations",
    description: "Step a recurrent cell through a binary sequence one timestep at a time, watching every gate activation, the cell state, and the hidden state update live. Toggle between a vanilla RNN and an LSTM to compare.",
    sourceFile: "mla/neuralnet/rnn.py",
    info: {
      type: "Building block of sequence models - recurrent, parametric, stateful (vanilla RNN or LSTM cell).",
      scenario: "Sequence data where order and (for LSTM) long-range dependencies matter - text, time series, audio - anywhere a fixed-size input vector isn't natural.",
      inputs: "A sequence x₁, …, x_T processed one timestep at a time, carrying forward a hidden state (and, for LSTM, a cell state).",
      decisionFunction: {
        text: "RNN: hₜ=tanh(Wₓxₜ+W_hhₜ₋₁+b)   LSTM: fₜ,iₜ,oₜ=σ(·), gₜ=tanh(·), cₜ=fₜ·cₜ₋₁+iₜ·gₜ, hₜ=oₜ·tanh(cₜ)",
        mechanism: "The same small set of weights is reapplied at every timestep, using the previous hidden (and cell) state as extra input. LSTM's three gates learn when to forget old information, write new information, and expose the cell state - letting it retain information far longer than a vanilla RNN before gradients vanish.",
        plot: { fn: (z) => 1 / (1 + Math.exp(-z)), domain: [-6, 6], color: "var(--accent)", caption: "the sigmoid squashing every LSTM gate into a soft [0,1] switch - near 0 blocks information flow, near 1 passes it through" },
      },
      lossFunction: {
        text: "Task-dependent (cross-entropy for next-token prediction, MSE for sequence regression), summed over timesteps",
        mechanism: "Trained via backpropagation through time (BPTT) - the chain rule is unrolled across every timestep, which is exactly why vanishing/exploding gradients are a bigger problem here than in feed-forward nets, and exactly what LSTM's gates were designed to mitigate.",
      },
      output: "A hidden state hₜ at every timestep, used directly or fed into a further output layer for predictions.",
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
