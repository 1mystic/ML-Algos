(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend">
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[0]}"></span>class 0</span>
              <span class="legend-item"><span class="swatch" style="background:${MLU.palette[1]}"></span>class 1</span>
              <span class="stage-hint">epoch <b id="nn-epoch">0</b></span>
            </div>
            <div class="btn-row">
              <button id="nn-train" class="primary">train</button>
              <button id="nn-reset">reset weights</button>
              <button id="nn-preset-xor">XOR data</button>
              <button id="nn-preset-circles">circular data</button>
            </div>
          </div>
          <svg id="nn-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">click class 0 / shift-click class 1 to add points · this is a real net training live via TensorFlow.js</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>architecture</h3>
            <div class="field"><label>hidden layer 1 units <span class="val" id="nn-h1-val">8</span></label>
              <input type="range" id="nn-h1" min="1" max="16" step="1" value="8" /></div>
            <div class="field"><label>hidden layer 2 units <span class="val" id="nn-h2-val">8</span></label>
              <input type="range" id="nn-h2" min="0" max="16" step="1" value="8" /></div>
            <div class="field"><label>activation</label>
              <select id="nn-act"><option value="relu">relu</option><option value="tanh">tanh</option></select></div>
            <div class="field"><label>learning rate <span class="val" id="nn-lr-val">0.03</span></label>
              <input type="range" id="nn-lr" min="1" max="30" step="1" value="8" /></div>
          </div>
          <div class="control-card">
            <h3>training</h3>
            <div class="readout" id="nn-readout">–</div>
            <div class="note">Feed-forward MLP with backprop via TensorFlow.js autodiff (Adam + binary cross-entropy) — the same architecture family as <code>mla/neuralnet</code>, trained continuously in the browser. Changing the architecture rebuilds the network.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#nn-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeTwoClass({ n: 90, mode: "xor" });
    let model = null, training = false, epoch = 0, rafId = null, stopped = false;

    function h1() { return +document.getElementById("nn-h1").value; }
    function h2() { return +document.getElementById("nn-h2").value; }
    function act() { return document.getElementById("nn-act").value; }
    function lrVal() { return +document.getElementById("nn-lr").value / 250; }

    function buildModel() {
      if (model) model.dispose();
      const m = tf.sequential();
      m.add(tf.layers.dense({ units: h1(), activation: act(), inputShape: [2] }));
      if (h2() > 0) m.add(tf.layers.dense({ units: h2(), activation: act() }));
      m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
      m.compile({ optimizer: tf.train.adam(lrVal()), loss: "binaryCrossentropy", metrics: ["accuracy"] });
      model = m;
      epoch = 0;
    }
    buildModel();

    const bgG = svg.append("g");
    const ptsG = svg.append("g");

    async function drawBoundary() {
      const cell = 16, xs = [], coords = [];
      for (let px = PAD; px < W - PAD; px += cell)
        for (let py = PAD; py < H - PAD; py += cell) {
          xs.push([x.invert(px + cell / 2) / 10, y.invert(py + cell / 2) / 10]);
          coords.push({ px, py });
        }
      const preds = tf.tidy(() => model.predict(tf.tensor2d(xs)).dataSync());
      const cells = coords.map((c, i) => ({ ...c, p: preds[i] }));
      const rects = bgG.selectAll("rect").data(cells);
      rects.enter().append("rect").attr("width", cell).attr("height", cell)
        .merge(rects).attr("x", (d) => d.px).attr("y", (d) => d.py)
        .attr("fill", (d) => d3.interpolateRgb(MLU.palette[0], MLU.palette[1])(d.p)).attr("opacity", 0.18);
      rects.exit().remove();
    }

    function drawPoints() {
      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle").attr("r", 5).attr("stroke", "var(--bg)").attr("stroke-width", 1).style("cursor", "grab")
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("fill", (d) => MLU.palette[d.label])
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); drawPoints(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          drawPoints();
        }));
      sel.exit().remove();
    }

    async function trainLoop() {
      if (stopped) return;
      if (training && points.length > 3) {
        const xsArr = points.map((p) => [p.x / 10, p.y / 10]);
        const ysArr = points.map((p) => [p.label]);
        const xt = tf.tensor2d(xsArr), yt = tf.tensor2d(ysArr);
        const h = await model.fit(xt, yt, { epochs: 3, batchSize: points.length, verbose: 0 });
        xt.dispose(); yt.dispose();
        epoch += 3;
        document.getElementById("nn-epoch").textContent = epoch;
        document.getElementById("nn-readout").innerHTML =
          `points: <b>${points.length}</b><br>loss: <b class="num">${h.history.loss.at(-1).toFixed(4)}</b><br>accuracy: <b class="num">${(h.history.acc ? h.history.acc.at(-1) : h.history.accuracy.at(-1)).toFixed(3)}</b>`;
        await drawBoundary();
      }
      rafId = requestAnimationFrame(trainLoop);
    }

    svg.on("click", (event) => {
      if (event.target.tagName === "circle") return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py), label: event.shiftKey ? 1 : 0 });
      drawPoints();
    });
    document.getElementById("nn-train").addEventListener("click", (e) => {
      training = !training;
      e.target.textContent = training ? "pause" : "train";
      e.target.classList.toggle("primary", training);
    });
    document.getElementById("nn-reset").addEventListener("click", () => { buildModel(); drawBoundary(); document.getElementById("nn-epoch").textContent = 0; });
    ["nn-h1", "nn-h2", "nn-act", "nn-lr"].forEach((id) => document.getElementById(id).addEventListener("change", () => {
      document.getElementById("nn-h1-val").textContent = h1();
      document.getElementById("nn-h2-val").textContent = h2();
      document.getElementById("nn-lr-val").textContent = lrVal().toFixed(3);
      buildModel(); drawBoundary();
    }));
    document.getElementById("nn-preset-xor").addEventListener("click", () => { points = MLU.makeTwoClass({ n: 90, mode: "xor" }); buildModel(); drawPoints(); drawBoundary(); });
    document.getElementById("nn-preset-circles").addEventListener("click", () => { points = MLU.makeTwoClass({ n: 90, mode: "circles" }); buildModel(); drawPoints(); drawBoundary(); });

    document.getElementById("nn-h1-val").textContent = h1();
    document.getElementById("nn-h2-val").textContent = h2();
    document.getElementById("nn-lr-val").textContent = lrVal().toFixed(3);
    drawPoints();
    drawBoundary();
    trainLoop();

    return () => { stopped = true; training = false; if (rafId) cancelAnimationFrame(rafId); if (model) model.dispose(); };
  }

  MLApp.register({
    id: "neural-net",
    name: "Neural Network (MLP)",
    category: "Deep Learning",
    tagline: "TensorFlow.js, trained live",
    description: "A configurable feed-forward network trained live in your browser with TensorFlow.js. Try XOR or circular data with a linear-only setup (0 units in layer 2, relu off) to see why depth matters.",
    sourceFile: "mla/neuralnet",
    info: {
      type: "Supervised — Classification/Regression. Parametric, non-linear (feed-forward multilayer perceptron).",
      scenario: "The true decision boundary is complex/non-linear and enough data exists to fit many parameters — the general-purpose workhorse of modern ML.",
      inputs: "A feature vector x (2D points with binary labels in this demo).",
      decisionFunction: {
        text: "ŷ = σ( W_L · act( … act(W₁x + b₁) … ) + b_L )",
        mechanism: "Input passes through alternating linear transforms and non-linear activations layer by layer; each hidden layer carves out more complex regions of input space than a single linear/logistic model could (e.g. it can solve XOR).",
        plot: { fn: (x) => Math.max(0, x), domain: [-4, 4], color: "var(--accent)", fn2: (x) => Math.tanh(x), color2: "var(--series-4)", caption: "the two activations available here — ReLU (solid, piecewise-linear) vs tanh (dashed, saturates smoothly at ±1)" },
      },
      lossFunction: {
        text: "L = −Σᵢ[yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ)]",
        mechanism: "Minimized via backpropagation: the chain rule computes the gradient of the loss w.r.t. every weight in every layer, and Adam uses those gradients to update all weights together, repeated over many epochs.",
      },
      output: "A predicted class probability (or a continuous value, for a regression output layer).",
      parameters: [
        { name: "hidden units / layers", effect: "Model capacity. More units/layers fit more complex functions but train slower and risk overfitting on small data." },
        { name: "activation", effect: "ReLU vs tanh — affects gradient flow and how easily units saturate." },
        { name: "learning rate", effect: "Adam's step size. Too high destabilizes training; too low trains very slowly." },
        { name: "epochs", effect: "How long to train. Watch validation loss to catch overfitting." },
      ],
      metrics: ["Accuracy", "Log-loss", "F1 / ROC-AUC (classification)", "MSE / MAE (regression)"],
      typicalUses: ["Image/speech/text classification", "Tabular deep learning", "Function approximation", "Any problem with enough data and a non-linear input-output relationship"],
      workedExample: {
        setup: "2 inputs → 2 ReLU hidden units → 1 sigmoid output. x=[1,0]. Hidden weights [[0.5,−0.5],[0.3,0.8]], biases 0. Output weights [1.0,−1.0], bias 0. True label y=1.",
        steps: [
          "Hidden unit 1: relu(0.5×1 + (−0.5)×0) = relu(0.5) = 0.5.",
          "Hidden unit 2: relu(0.3×1 + 0.8×0) = relu(0.3) = 0.3.",
          "Output pre-activation: z = 1.0×0.5 + (−1.0)×0.3 = 0.2.",
          "Output ŷ = σ(0.2) = 1/(1+e⁻⁰·²) = 1/1.819 ≈ 0.550.",
          "Loss = −log(ŷ) = −log(0.550) ≈ 0.598 (since true label is 1).",
        ],
        result: "Forward pass output ŷ ≈ 0.550, cross-entropy loss ≈ 0.598 — backprop would now push weights to raise ŷ toward 1",
      },
    },
    mount,
  });
})();
