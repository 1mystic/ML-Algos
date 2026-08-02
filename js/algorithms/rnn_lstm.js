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
