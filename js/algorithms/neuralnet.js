(() => {
  const { ui, col, fmt } = Lab;
  const MAXF = 360;

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "mlp", fps: 12, classes: 2,
      legend: ui.legend([["var(--text-faint)", "test point (hollow)", "ring"], ["var(--text)", "p = 0.5", "line"]]) + `<span class="legend-item" data-r="showing" style="color:var(--series-4)"></span>`,
      tools: ui.btn("train", "train", true, "start / pause live training") + ui.btn("step", "+10 epochs") + ui.btn("reset", "reset weights"),
      hint: "every 5th point is held out as test data · click adds the selected class · hover a neuron in the network to see what it computes",
      data: { kind: "class", presets: ["xor", "circles", "moons", "spiral", "linear", "blobs2", "overlap"], def: "xor", n: 160, noise: 0.15 },
      side: [
        { title: "architecture", html: ui.range("L", "hidden layers", 1, 3, 1, 2) + ui.range("U", "units per layer", 1, 8, 1, 6) + ui.select("act", "activation", [["relu", "ReLU"], ["tanh", "tanh"], ["sigmoid", "sigmoid"], ["linear", "linear (no non-linearity!)"]], "tanh") },
        { title: "training", html: ui.range("lr", "learning rate (log₁₀)", -3.5, -0.5, 0.25, -1.5, "0.032") + ui.range("l2", "L2 weight decay (log₁₀, −6 = off)", -6, -1, 0.5, -6, "off") + ui.select("bs", "batch", [["full", "full batch"], ["16", "mini-batch 16"]], "full") },
      ],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "x", label: "x ∈ ℝ²", sub: "input", x: 36, y: 40, w: 60, kind: "data" },
          { id: "hid", label: "Wx + b → act", sub: "hidden layers", x: 160, y: 40, kind: "op", tex: "a^{(l)}=g\\big(W^{(l)}a^{(l-1)}+b^{(l)}\\big)" },
          { id: "out", label: "σ(z)", sub: "p(class 1)", x: 300, y: 40, w: 70, kind: "out" },
          { id: "loss", label: "BCE", sub: "+ λ‖W‖²", x: 300, y: 120, kind: "loss" },
          { id: "bp", label: "backprop", sub: "autodiff ∇", x: 172, y: 120, kind: "op", tex: "\\delta^{(l)}=(W^{(l+1)\\top}\\delta^{(l+1)})\\odot g'(z^{(l)})" },
          { id: "adam", label: "Adam step", sub: "per-weight lr", x: 50, y: 120, kind: "param" },
        ],
        edges: [{ from: "x", to: "hid" }, { from: "hid", to: "out" }, { from: "out", to: "loss" }, { from: "loss", to: "bp" }, { from: "bp", to: "adam" }, { from: "adam", to: "hid", dash: true, label: "update W" }],
      },
      phases: { init: { label: "init", color: 3 }, train: { label: "training", color: 0 }, paused: { label: "paused", color: 4 } },
      panels: [{ id: "net", title: "network", sub: "edge colour = sign, width = |weight| · hover a neuron", wide: true }],
      charts: [{ id: "loss", title: "loss (train / test)", series: [{ key: "loss", label: "train", color: "var(--series-1)" }, { key: "tloss", label: "test", color: "var(--series-2)", dash: "4 3" }] }, { id: "acc", title: "accuracy (train / test)", series: [{ key: "acc", label: "train", color: "var(--series-3)" }, { key: "tacc", label: "test", color: "var(--series-5)", dash: "4 3" }], yd: [0, 1] }],
      onFrame: render,
      onData: () => { lab.points.forEach((p, i) => (p.test = i % 5 === 4)); reset(); },
    });
    lab.editable({ newPoint: (x, y, ev) => ({ x, y, label: lab.cls(ev), test: false }), onChange: () => { if (!training) pushFrame(); } });
    const pl = lab.plane, L = pl.layers;
    lab.player.setLive(true);

    let model = null, opt = null, epoch = 0, training = false, timer = null, frames = [], gridT = null, gridDims = null, subModels = {}, focus = null, net = null, stopped = false;
    const P = () => ({ L: lab.val("L"), U: lab.val("U"), act: lab.val("act"), lr: Math.pow(10, lab.val("lr")), l2: lab.val("l2") <= -6 ? 0 : Math.pow(10, lab.val("l2")), bs: lab.val("bs") });

    function build() {
      if (model) model.dispose();
      if (opt) opt.dispose();
      subModels = {};
      const p = P();
      model = tf.sequential();
      for (let l = 0; l < p.L; l++) model.add(tf.layers.dense({ units: p.U, activation: p.act, inputShape: l ? undefined : [2], kernelInitializer: "glorotNormal" }));
      model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
      opt = tf.train.adam(p.lr);
      epoch = 0;
      const layers = [2, ...Array(p.L).fill(p.U), 1];
      const el = lab.R("panel-net"); el.innerHTML = "";
      net = Lab.net(el, { layers, w: 720, h: 190, r: 9, labels: ["input", ...Array(p.L).fill(0).map((_, i) => `hidden ${i + 1}`), "output"], onNeuron: (l, i, click) => { focus = click && focus && focus[0] === l && focus[1] === i ? null : [l, i]; net.select(...(focus || [-1, -1])); lab.player.redraw(); } });
    }
    function ensureGrid() {
      const res = 60;
      if (gridT && gridDims && gridDims.res === res) return;
      if (gridT) gridT.dispose();
      const g = Lab.sampleGrid(pl, (x, y) => [x / 10, y / 10], res);
      gridDims = { res, cols: g.cols, rows: g.rows };
      gridT = tf.tensor2d(g.values);
    }
    function data(set) {
      const pts = lab.points.filter((p) => (set === "test" ? p.test : !p.test));
      return pts;
    }
    function evalSet(pts) {
      if (!pts.length) return { loss: null, acc: null };
      return tf.tidy(() => {
        const xs = tf.tensor2d(pts.map((p) => [p.x / 10, p.y / 10])), ys = tf.tensor2d(pts.map((p) => [p.label]));
        const pr = model.predict(xs);
        const loss = tf.losses.logLoss(ys, pr).dataSync()[0];
        const acc = tf.mean(tf.equal(tf.greaterEqual(pr, 0.5), tf.cast(ys, "bool"))).dataSync()[0];
        return { loss, acc };
      });
    }
    function trainSteps(n) {
      const tr = data("train");
      if (tr.length < 2) return;
      const p = P();
      tf.tidy(() => {
        const X = tf.tensor2d(tr.map((q) => [q.x / 10, q.y / 10])), Y = tf.tensor2d(tr.map((q) => [q.label]));
        for (let s = 0; s < n; s++) {
          let xs = X, ys = Y;
          if (p.bs !== "full") { const idx = tf.tensor1d(Array.from({ length: 16 }, () => Math.floor(Math.random() * tr.length)), "int32"); xs = tf.gather(X, idx); ys = tf.gather(Y, idx); }
          opt.minimize(() => {
            let loss = tf.losses.logLoss(ys, model.apply(xs));
            if (p.l2) for (const w of model.trainableWeights) if (/kernel/.test(w.name)) loss = loss.add(tf.sum(tf.square(w.read())).mul(p.l2));
            return loss;
          });
          epoch++;
        }
      });
    }
    function snapshot() {
      ensureGrid();
      const grid = tf.tidy(() => model.predict(gridT).dataSync().slice());
      const W = model.layers.map((ly) => ly.getWeights()[0].arraySync());
      const tr = evalSet(data("train")), te = evalSet(data("test"));
      return { grid, W, epoch, m: { loss: tr.loss, acc: tr.acc, tloss: te.loss, tacc: te.acc } };
    }
    function pushFrame() {
      const s = snapshot();
      const fr = {
        phase: training ? "train" : epoch === 0 ? "init" : "paused", arch: training ? ["hid", "out", "loss", "bp", "adam"] : ["x", "hid", "out"],
        title: epoch === 0 ? "Random initial weights" : `Epoch ${epoch}`,
        text: epoch === 0 ? "Glorot-initialised weights give a nearly flat, uninformative output. Press <b>train</b>: every few steps a snapshot is recorded, so you can pause and scrub back through training." : `Train loss <b>${fmt(s.m.loss)}</b>, test loss <b>${fmt(s.m.tloss)}</b>. ${P().act === "linear" ? "With linear activations the whole network collapses to one linear map, so the boundary stays straight no matter how deep it is." : s.m.tloss != null && s.m.loss != null && s.m.tloss > s.m.loss * 1.6 && epoch > 300 ? "Test loss is pulling away from train loss: the network is starting to memorise. Try weight decay or fewer units." : "Each hidden unit learns a soft half-plane; later layers combine them into curved regions."}`,
        tex: epoch === 0 ? "\\hat p=\\sigma\\big(W^{(L)}g(\\cdots g(W^{(1)}x+b^{(1)})\\cdots)+b^{(L)}\\big)" : "W\\leftarrow W-\\eta\\,\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon},\\qquad L=-\\tfrac1n\\sum y\\log\\hat p+(1-y)\\log(1-\\hat p)",
        st: s, m: s.m,
      };
      frames.push(fr);
      if (frames.length > MAXF) frames = frames.filter((f, i) => i === 0 || i % 2 === 1 || i > frames.length - 40);
      for (const c of ["loss", "acc"]) lab.charts[c].set(frames.map((f) => f.m));
      const follow = lab.player.frames.length === 0 || lab.player.index >= lab.player.frames.length - 2;
      if (frames.length === 1) lab.load(frames, { at: "end" });
      else { lab.player.replaceAll(frames); if (follow) lab.player.show(frames.length - 1); }
    }
    function loop() {
      if (!training || stopped) return;
      trainSteps(P().bs === "full" ? 6 : 12);
      pushFrame();
      timer = setTimeout(loop, 40);
    }
    function setTraining(v) {
      training = v;
      const b = lab.q('[data-b="train"]');
      b.textContent = v ? "pause" : "train";
      if (v) loop(); else { clearTimeout(timer); pushFrame(); }
    }
    function reset() {
      clearTimeout(timer);
      build();
      frames = [];
      for (const c of ["loss", "acc"]) lab.charts[c].set([]);
      pushFrame();
      if (training) loop();
    }

    function neuronValues(l, i, isLatest, st) {
      if (l === 0) return Lab.sampleGrid(pl, (x, y) => (i === 0 ? x : y) / 10, 60).values;
      const p = P();
      if (l === p.L + 1) return st.grid;
      if (!isLatest) return null;
      ensureGrid();
      if (!subModels[l]) subModels[l] = tf.model({ inputs: model.inputs, outputs: model.layers[l - 1].output });
      return tf.tidy(() => { const out = subModels[l].predict(gridT); return out.slice([0, i], [-1, 1]).dataSync().slice(); });
    }

    const showEl = lab.R("legend").querySelector('[data-r="showing"]');
    function render(f, idx, all) {
      const st = f.st;
      if (!st || !st.grid) return;
      const isLatest = idx === all.length - 1;
      let vals = st.grid, label = "", isOut = true;
      if (focus) {
        const v = neuronValues(focus[0], focus[1], isLatest, st);
        if (v) { vals = v; isOut = focus[0] === P().L + 1; label = focus[0] === 0 ? `showing input ${focus[1] === 0 ? "x₁" : "x₂"}` : isOut ? "showing output neuron" : `showing hidden ${focus[0]} · unit ${focus[1] + 1}`; }
        else label = "neuron view needs the latest frame";
      }
      showEl.textContent = label;
      const grid = { cols: gridDims.cols, rows: gridDims.rows, values: Array.from(vals) };
      if (isOut) {
        Lab.paintGrid(L.bg, pl, grid, (p) => Lab.probColor(p, 0.55));
        Lab.contour(L.under, pl, grid, [0.5], { cls: "iso-main", width: 2 });
      } else {
        const mx = Math.max(1e-6, ...grid.values.map(Math.abs));
        Lab.paintGrid(L.bg, pl, grid, (v) => Lab.divergeColor(v, mx, 0.6));
        L.under.selectAll("path").remove();
      }
      lab.drawPoints({ color: (d) => (d.test ? "transparent" : col(d.label)), stroke: (d) => (d.test ? col(d.label) : "var(--bg)"), sw: (d) => (d.test ? 2 : 1.1), r: 4.4 });
      if (net) {
        net.weights(st.W);
      }
      const m = f.m || {};
      const nParams = st.W.reduce((s, w) => s + w.length * w[0].length + w[0].length, 0);
      lab.readout(`<div class="kv"><span>epoch</span><b>${st.epoch}</b><span>parameters</span><b>${nParams}</b><span>train loss / acc</span><b>${fmt(m.loss)} / ${m.acc != null ? (m.acc * 100).toFixed(0) + "%" : "–"}</b><span>test loss / acc</span><b>${fmt(m.tloss)} / ${m.tacc != null ? (m.tacc * 100).toFixed(0) + "%" : "–"}</b></div><div class="note">Real backprop via TensorFlow.js autodiff, Adam optimiser.</div>`);
    }

    lab.q('[data-b="train"]').onclick = () => setTraining(!training);
    lab.q('[data-b="step"]').onclick = () => { trainSteps(10); pushFrame(); };
    lab.q('[data-b="reset"]').onclick = () => reset();
    lab.watch(["L", "U", "act", "lr", "l2", "bs"], () => reset(), { format: { lr: (v) => fmt(Math.pow(10, v), 4), l2: (v) => (v <= -6 ? "off" : "1e" + v) } });
    lab.cleanup.push(() => { stopped = true; clearTimeout(timer); if (model) model.dispose(); if (opt) opt.dispose(); if (gridT) gridT.dispose(); });
    lab.loadPreset("xor");
    setTraining(true);
    return lab.destroy;
  }

  MLApp.register({
    id: "neural-net",
    name: "Neural Network (MLP)",
    category: "Deep Learning",
    tagline: "TensorFlow.js, live training, neuron views",
    description: "A configurable multilayer perceptron trained live in your browser with TensorFlow.js. Watch the decision field form, weights thicken in the network diagram, train and test loss diverge when it overfits, and hover any neuron to see the feature it has learned.",
    info: {
      type: "Supervised - Classification/Regression. Parametric, non-linear (feed-forward multilayer perceptron).",
      scenario: "The true decision boundary is complex/non-linear and enough data exists to fit many parameters - the general-purpose workhorse of modern ML.",
      inputs: "A feature vector x (2D points with binary labels in this demo).",
      intuition: {
        definition: "Stack linear transforms and non-linear activations. Each layer <b>re-represents</b> its input, so later layers work with features the earlier ones invented. Remove the non-linearities and the whole stack collapses back into a single linear model.",
        steps: [
          "Multiply by a weight matrix and add a bias.",
          "Apply a non-linear activation such as ReLU.",
          "Repeat for each layer, ending in the output shape you need.",
          "Backpropagate the loss gradient and update every weight.",
        ],
        applications: [
          "Image, speech, and text classification",
          "Function approximation where the form is unknown",
          "Recommendation and embedding models",
          "Tabular deep learning, though boosting often still wins there",
          "Any problem with abundant data and non-linear structure",
        ],
      },
      math: [
        { title: "Layer forward pass", formula: "z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾,   a⁽ˡ⁾ = g(z⁽ˡ⁾)", note: "a⁽⁰⁾ is the input. Each layer is an affine map followed by an elementwise non-linearity." },
        { title: "Why non-linearity is essential", formula: "W₂(W₁x) = (W₂W₁)x = Wx", note: "Composing linear maps yields another linear map. Without g, a hundred layers have exactly the expressive power of one, and XOR remains unsolvable." },
        { title: "Common activations", formula: "ReLU: max(0,z)   tanh: (e^z−e^−z)/(e^z+e^−z)   sigmoid: 1/(1+e^−z)", note: "ReLU is the default: its gradient is 1 for positive inputs, so it does not saturate. Tanh and sigmoid squash into a bounded range and flatten at the extremes." },
        { title: "Output layer and loss", formula: "binary: σ(z) + BCE    multi-class: softmax(z) + cross-entropy    regression: identity + MSE", note: "The activation and loss are chosen as a pair so the output gradient simplifies to the clean form (ŷ − y)." },
        { title: "Backpropagation", formula: "δ⁽ᴸ⁾ = ∇_a L ⊙ g'(z⁽ᴸ⁾),   δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ g'(z⁽ˡ⁾)", note: "The chain rule applied layer by layer, reusing each layer's intermediate result. This is what makes computing millions of gradients affordable." },
        { title: "Weight gradients", formula: "∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾·(a⁽ˡ⁻¹⁾)ᵀ,   ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾", note: "An outer product of the incoming activation and the outgoing error signal. Same shape as the weight matrix, as it must be." },
        { title: "Universal approximation", formula: "one hidden layer of sufficient width approximates any continuous function on a compact set", note: "An existence result only. It says nothing about how wide, whether gradient descent will find it, or how much data is needed." },
      ],
      pipeline: [
        { label: "Input x", note: "feature vector" },
        { label: "Linear Wx+b", note: "affine map" },
        { label: "Activation", note: "ReLU / tanh" },
        { label: "Repeat layers", note: "deeper features" },
        { label: "Output + loss", note: "backprop, update", accent: "green" },
      ],
      decisionFunction: {
        text: "ŷ = σ( W_L · act( … act(W₁x + b₁) … ) + b_L )",
        mechanism: "Input passes through alternating linear transforms and non-linear activations layer by layer; each hidden layer carves out more complex regions of input space than a single linear/logistic model could (e.g. it can solve XOR).",
        plot: { fn: (x) => Math.max(0, x), domain: [-4, 4], color: "var(--accent)", fn2: (x) => Math.tanh(x), color2: "var(--series-4)", caption: "the two activations available here - ReLU (solid, piecewise-linear) vs tanh (dashed, saturates smoothly at ±1)" },
      },
      lossFunction: {
        text: "L = −Σᵢ[yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ)]",
        mechanism: "Minimized via backpropagation: the chain rule computes the gradient of the loss w.r.t. every weight in every layer, and Adam uses those gradients to update all weights together, repeated over many epochs.",
      },
      optimization: [
        { title: "Stochastic gradient descent", formula: "W := W − η·∇_W L(minibatch)", note: "Full-batch gradients are too expensive and too smooth. Minibatch noise is not just a compromise, it actively helps escape sharp minima." },
        { title: "Momentum", formula: "v := βv + ∇L,   W := W − η·v", note: "Accumulates a velocity across steps, damping oscillation across narrow valleys and accelerating along consistent directions." },
        { title: "Adam", formula: "m̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ),  W := W − η·m̂/(√v̂ + ε)", note: "Momentum plus a per-parameter adaptive step size from the squared-gradient average. The default optimiser because it works acceptably with almost no tuning." },
        { title: "Vanishing gradients", formula: "sigmoid'(z) ≤ 0.25, so products across L layers shrink like 0.25^L", note: "Why deep sigmoid or tanh networks stopped training. ReLU has gradient exactly 1 on its positive side, which is the fix that made depth practical." },
        { title: "Weight initialisation", formula: "He: Var(W) = 2/n_in (ReLU)    Xavier: Var(W) = 1/n_in (tanh)", note: "Scaled to keep activation variance stable across layers. Initialising all weights to zero is fatal: every unit computes the same thing and receives the same gradient forever." },
        { title: "Batch normalisation", formula: "normalise each minibatch, then rescale by learned γ and β", note: "Stabilises the distribution entering each layer, allowing much higher learning rates and acting as a mild regularizer." },
      ],
      output: "A predicted class probability (or a continuous value, for a regression output layer).",
      assumptions: [
        { name: "Enough training data", why: "Millions of parameters need many examples, or the network memorises.", check: "If data is scarce, prefer a regularized linear model or gradient boosting." },
        { name: "Features are scaled", why: "Unscaled inputs produce wildly different gradient magnitudes per weight, which destabilises training.", check: "Standardize inputs. This matters as much here as for SVM." },
        { name: "The relationship is learnable by gradient descent", why: "The loss surface is non-convex, so there is no guarantee of finding a good optimum.", check: "Confirm the training loss actually falls. If it does not, the architecture or learning rate is wrong." },
        { name: "Sufficient compute", why: "Training is orders of magnitude more expensive than fitting a linear model.", check: "Budget for GPU time on anything non-trivial." },
        { name: "Interpretability is not required", why: "Millions of interacting weights are not explainable by inspection.", check: "Use SHAP or integrated gradients, or choose a simpler model." },
      ],
      regularization: [
        { name: "L2 / weight decay", formula: "L + λ‖W‖²", note: "Shrinks weights toward zero. In Adam use AdamW, which decouples decay from the adaptive step and behaves as intended." },
        { name: "Dropout", formula: "randomly zero a fraction p of activations during training", note: "Prevents units from co-adapting by forcing redundant representations. Disabled at inference, with activations rescaled to compensate." },
        { name: "Early stopping", formula: "halt when validation loss stops improving", note: "The simplest and often the most effective regularizer. Also saves compute." },
        { name: "Batch normalisation", formula: "normalise per minibatch", note: "Primarily an optimisation aid, but the minibatch noise regularizes as a side effect." },
        { name: "Data augmentation", formula: "train on label-preserving transforms of the input", note: "Effectively enlarges the dataset. The highest-leverage regularizer when applicable." },
        { name: "Label smoothing", formula: "target = 1−ε instead of 1", note: "Discourages overconfident outputs and improves calibration." },
      ],
      hyperparameters: [
        { name: "learning rate", range: "1e-5 - 1e-1", increasing: "Faster progress, then instability, divergence, and NaN losses.", strategy: "The single most important hyperparameter. Use an LR range test or start at 1e-3 with Adam, then add a decay schedule." },
        { name: "hidden units per layer", range: "16 - 2048", increasing: "More capacity, slower training, more overfitting risk.", strategy: "Start wide enough to overfit, then regularize back. Underfitting is harder to diagnose than overfitting." },
        { name: "depth (layers)", range: "1 - 100+", increasing: "More abstraction, harder optimisation without residual connections.", strategy: "Two or three hidden layers suffice for tabular data. Depth pays off mainly for images, audio, and text." },
        { name: "batch size", range: "16 - 1024", increasing: "Smoother gradients, faster wall-clock per epoch, often slightly worse generalisation.", strategy: "32 to 256 is standard. Scale the learning rate roughly with batch size." },
        { name: "activation", range: "ReLU / GELU / tanh", increasing: "Not applicable", strategy: "ReLU by default, GELU for transformers. Avoid sigmoid in hidden layers." },
        { name: "dropout rate", range: "0 - 0.5", increasing: "Stronger regularization, slower convergence, eventual underfitting.", strategy: "0.2 to 0.5 on dense layers. Reduce it if training loss will not fall." },
        { name: "epochs", range: "10 - 1000", increasing: "Better fit, then overfitting.", strategy: "Set high and use early stopping with patience rather than guessing." },
        { name: "weight decay", range: "1e-6 - 1e-2", increasing: "Smaller weights, smoother function.", strategy: "1e-4 is a reasonable default. Use AdamW rather than Adam plus L2." },
      ],
      metrics: ["Accuracy", "Log-loss", "F1 / ROC-AUC (classification)", "MSE / MAE (regression)", "Training and validation loss curves"],
      typicalUses: ["Image/speech/text classification", "Tabular deep learning", "Function approximation", "Any problem with enough data and a non-linear input-output relationship"],
      diagnostics: [
        "Always plot training and validation loss together. Their relationship diagnoses almost every problem: both high means underfitting, train low with validation high means overfitting, train not falling at all means a bug or a bad learning rate.",
        "Before anything else, verify the model can overfit a batch of ten examples to near-zero loss. If it cannot, there is a bug, not a tuning problem.",
        "Watch for dead ReLU units whose activations are always zero. A high learning rate is the usual cause; try LeakyReLU.",
        "Monitor gradient norms. Exploding values call for gradient clipping; vanishing ones call for better initialisation or residual connections.",
        "A loss that becomes NaN almost always means the learning rate is too high or there is a log(0) in the loss.",
      ],
      advantages: [
        "Universal approximation: with enough capacity it can represent essentially any continuous function.",
        "Learns its own features, removing the need for manual feature engineering.",
        "The dominant approach for images, audio, text, and any unstructured data.",
        "Scales with data and compute in a way most classical models do not.",
        "Highly flexible architecturally: one framework covers classification, regression, generation, and embedding.",
        "Supports transfer learning, so a pretrained model can be adapted with very little data.",
      ],
      limitations: [
        { name: "Data-hungry", note: "millions of parameters need many examples", fix: "transfer learning, augmentation, or a simpler model." },
        { name: "Expensive to train", note: "GPU time and energy", fix: "start from a pretrained model." },
        { name: "Non-convex optimisation", note: "no guarantee of a good optimum, and results vary by seed", fix: "good initialisation, adaptive optimisers, multiple runs." },
        { name: "Many hyperparameters", note: "learning rate, depth, width, regularization all interact", fix: "tune learning rate first; it dominates everything else." },
        { name: "Black box", note: "not interpretable by inspection", fix: "SHAP, integrated gradients, or attention visualisation." },
        { name: "Often beaten on tabular data", note: "gradient boosting usually wins on structured features", fix: "benchmark against XGBoost before committing." },
        { name: "Overconfident and poorly calibrated", note: "softmax outputs are far too certain", fix: "temperature scaling or label smoothing." },
      ],
      alternatives: [
        { name: "Gradient boosting", when: "Tabular or structured data. Usually more accurate and far cheaper." },
        { name: "Logistic regression", when: "The relationship is near-linear, data is limited, or interpretability is required." },
        { name: "Pretrained transformer or CNN", when: "Images or text. Fine-tuning beats training from scratch almost always." },
        { name: "SVM", when: "Small dataset with a non-linear boundary." },
      ],
      pitfalls: [
        { problem: "Loss becomes NaN", solution: "Learning rate too high, or a log(0). Lower the rate, clip gradients, add epsilon inside logs." },
        { problem: "Loss does not decrease at all", solution: "Check for zero-initialised weights, a wrong loss function, or unscaled inputs. Try overfitting ten samples first." },
        { problem: "Perfect train accuracy, poor validation", solution: "Overfitting. Add dropout, weight decay, augmentation, or stop earlier." },
        { problem: "Training is extremely slow to converge", solution: "Learning rate too low, or inputs unnormalised. Add batch normalisation." },
        { problem: "Many units are permanently dead", solution: "Dying ReLU from too high a learning rate. Lower it or switch to LeakyReLU." },
        { problem: "Results differ wildly between runs", solution: "Non-convex loss plus random initialisation. Fix seeds and average across runs." },
        { problem: "Validation accuracy exceeds training accuracy", solution: "Normal early on with dropout, since it is active only during training." },
      ],
      quickRef: [
        { name: "Forward pass", formula: "a⁽ˡ⁾ = g(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)" },
        { name: "ReLU", formula: "max(0, z)" },
        { name: "Softmax", formula: "e^{zₖ} / Σⱼ e^{zⱼ}" },
        { name: "Cross-entropy", formula: "−Σ y log ŷ" },
        { name: "Output delta", formula: "δ⁽ᴸ⁾ = ŷ − y" },
        { name: "Backprop", formula: "δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ g'(z⁽ˡ⁾)" },
        { name: "Weight gradient", formula: "∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ" },
        { name: "He init", formula: "Var(W) = 2 / n_in" },
        { name: "Adam", formula: "W −= η·m̂/(√v̂ + ε)" },
      ],
      code: `import torch, torch.nn as nn

model = nn.Sequential(
    nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, n_classes),   # raw logits; the loss applies softmax
)

# AdamW decouples weight decay from the adaptive step, unlike Adam + L2.
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(200):
    model.train()
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()                        # disables dropout
    with torch.no_grad():
        val = sum(loss_fn(model(xb), yb) for xb, yb in val_loader)
    sched.step(val)                     # early stopping goes here

# Sanity check before any real run: can it overfit 10 samples to ~0 loss?
# If not, you have a bug, not a tuning problem.`,
      whyChain: [
        { q: "Why do neural networks need non-linear activations?", a: "Because composing linear maps gives another linear map. W₂(W₁x) equals (W₂W₁)x, so without a non-linearity a network of any depth has exactly the expressive power of a single linear layer and cannot even solve XOR." },
        { q: "Why did ReLU replace sigmoid in hidden layers?", a: "The sigmoid's derivative peaks at 0.25 and approaches zero when saturated. Multiplying such factors across many layers makes the gradient vanish exponentially with depth. ReLU has derivative exactly 1 on its positive side, so gradients propagate undiminished." },
        { q: "What is the dying ReLU problem?", a: "If a large gradient step pushes a unit's pre-activation permanently negative for all inputs, its output is always zero and so is its gradient. The unit can never recover. Lower learning rates or LeakyReLU, which has a small negative slope, avoid it." },
        { q: "Why can't you initialise all weights to zero?", a: "Every unit in a layer would compute the same output and receive the same gradient, so they would stay identical forever. The layer would have the capacity of a single unit. Random initialisation breaks that symmetry." },
        { q: "Why does minibatch noise help rather than hurt?", a: "Beyond being cheaper, the noise in stochastic gradients helps escape sharp minima and saddle points. Sharp minima tend to generalise worse, so the noise acts as an implicit regularizer that biases training toward flatter, more robust solutions." },
        { q: "What exactly does dropout do?", a: "It randomly zeroes activations during training, so no unit can rely on any specific other unit being present. That forces redundant, distributed representations. It also approximates averaging over an exponential ensemble of subnetworks." },
        { q: "If one hidden layer is a universal approximator, why go deep?", a: "Universality is an existence theorem, not a practicality claim. The required width can be exponential in the input dimension. Depth lets features compose hierarchically, so the same function is representable with far fewer parameters and is easier to learn." },
        { q: "Why does Adam usually beat plain SGD out of the box?", a: "It maintains a per-parameter step size scaled by the recent squared-gradient average, so rarely updated parameters get larger steps. That makes it far less sensitive to the initial learning rate. Well-tuned SGD with momentum often generalises slightly better in the end." },
        { q: "How do you tell underfitting from overfitting?", a: "Compare the two loss curves. Both high and close together means underfitting: add capacity. Training low with validation high and diverging means overfitting: add regularization or data." },
      ],
      parameters: [
        { name: "hidden units / layers", effect: "Model capacity. More units/layers fit more complex functions but train slower and risk overfitting on small data." },
        { name: "activation", effect: "ReLU vs tanh - affects gradient flow and how easily units saturate." },
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
        result: "Forward pass output ŷ ≈ 0.550, cross-entropy loss ≈ 0.598 - backprop would now push weights to raise ŷ toward 1",
      },
    },
    mount,
  });
})();
