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
            <div class="note">Feed-forward MLP with backprop via TensorFlow.js autodiff (Adam + binary cross-entropy) - the same architecture family as <code>mla/neuralnet</code>, trained continuously in the browser. Changing the architecture rebuilds the network.</div>
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
