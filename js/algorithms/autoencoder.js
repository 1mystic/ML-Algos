(() => {
  const { ui, col, fmt } = Lab;
  const MAXF = 300;
  const curves = {
    arc: (t) => [-8 + 16 * t, 6 - 22 * (t - 0.5) ** 2 * 2.4 + 1],
    sine: (t) => [-8.5 + 17 * t, 4.5 * Math.sin(t * 2 * Math.PI * 1.1)],
    scurve: (t) => { const a = (t - 0.5) * 3 * Math.PI; return [5 * Math.sin(a), 3 * Math.sign(a) * (Math.cos(a) - 1)]; },
    spiral: (t) => { const a = 0.8 + t * 3.2 * Math.PI, r = 1 + t * 7.5; return [r * Math.cos(a), r * Math.sin(a)]; },
    circle: (t) => [6.5 * Math.cos(t * 2 * Math.PI), 6.5 * Math.sin(t * 2 * Math.PI)],
  };
  function genCurve(kind, n, noise) {
    return Array.from({ length: n }, () => { const t = MLU.rng(); const [x, y] = curves[kind](t); return { x: x + MLU.randn() * (0.15 + noise), y: y + MLU.randn() * (0.15 + noise), t, label: Math.min(6, Math.floor(t * 7)) }; });
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "ae", fps: 12,
      legend: ui.legend([["var(--accent)", "decoder curve x̂(z)", "line"], ["var(--text-faint)", "reconstruction error", "line"], ["var(--series-4)", "decoded latent probe"]]),
      tools: ui.btn("train", "train", true) + ui.btn("reset", "reset weights"),
      hint: "point colour = position along the true curve · hover the latent strip below to decode any z",
      data: {
        kind: "cluster", presets: ["arc", "sine", "scurve", "spiral", "circle"], def: "arc", n: 160, noise: 0.15, csv: false, names: { arc: "parabolic arc", sine: "sine curve", scurve: "S-curve", spiral: "spiral", circle: "circle (no 1-D chart exists)" },
        custom: Object.fromEntries(Object.keys(curves).map((k) => [k, ({ n, noise }) => genCurve(k, n, noise * 0.8)])),
      },
      side: [{ title: "architecture", html: ui.select("act", "activation", [["tanh", "tanh (non-linear)"], ["linear", "linear (= PCA)"]], "tanh") + ui.range("U", "units in encoder / decoder", 4, 32, 4, 16) + ui.range("lr", "learning rate (log₁₀)", -3.5, -1, 0.25, -2, "0.01") }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "x", label: "x ∈ ℝ²", sub: "input", x: 36, y: 60, w: 60, kind: "data" },
          { id: "enc", label: "encoder", sub: "2 → U → 1", x: 128, y: 60, kind: "op" },
          { id: "z", label: "z ∈ ℝ¹", sub: "bottleneck", x: 222, y: 60, w: 66, kind: "param", info: "The bottleneck forces the network to summarise each 2-D point with a single number: a coordinate along the curve." },
          { id: "dec", label: "decoder", sub: "1 → U → 2", x: 310, y: 60, kind: "op" },
          { id: "xh", label: "x̂", sub: "reconstruction", x: 310, y: 150, kind: "out" },
          { id: "loss", label: "‖x − x̂‖²", sub: "MSE", x: 174, y: 150, kind: "loss" },
          { id: "opt", label: "Adam", sub: "backprop", x: 56, y: 150, kind: "param" },
        ],
        edges: [{ from: "x", to: "enc" }, { from: "enc", to: "z" }, { from: "z", to: "dec" }, { from: "dec", to: "xh" }, { from: "xh", to: "loss" }, { from: "loss", to: "opt" }, { from: "opt", to: "enc", dash: true }],
      },
      phases: { init: { label: "init", color: 3 }, train: { label: "training", color: 0 }, paused: { label: "paused", color: 4 } },
      panels: [{ id: "lat", title: "latent space: every point's code z", sub: "hover to decode", wide: true }],
      charts: [{ id: "loss", title: "reconstruction MSE", series: [{ key: "loss", label: "MSE", color: "var(--series-1)", digits: 4 }], logY: true }],
      onFrame: render,
      onData: () => reset(),
    });
    const pl = lab.plane, L = pl.layers;
    lab.player.setLive(true);
    let enc = null, dec = null, ae = null, opt = null, training = false, timer = null, epoch = 0, frames = [], stopped = false, probe = null;
    const P = () => ({ act: lab.val("act"), U: lab.val("U"), lr: Math.pow(10, lab.val("lr")) });
    function build() {
      [enc, dec, opt].forEach((m) => m && m.dispose());
      const p = P();
      enc = tf.sequential({ layers: [tf.layers.dense({ units: p.U, activation: p.act, inputShape: [2] }), tf.layers.dense({ units: p.U, activation: p.act }), tf.layers.dense({ units: 1 })] });
      dec = tf.sequential({ layers: [tf.layers.dense({ units: p.U, activation: p.act, inputShape: [1] }), tf.layers.dense({ units: p.U, activation: p.act }), tf.layers.dense({ units: 2 })] });
      opt = tf.train.adam(p.lr);
      epoch = 0;
    }
    const X = () => lab.points.map((q) => [q.x / 10, q.y / 10]);
    function trainSteps(n) {
      tf.tidy(() => {
        const xs = tf.tensor2d(X());
        for (let s = 0; s < n; s++) { opt.minimize(() => tf.losses.meanSquaredError(xs, dec.apply(enc.apply(xs)))); epoch++; }
      });
    }
    function snapshot() {
      return tf.tidy(() => {
        const xs = tf.tensor2d(X());
        const z = enc.predict(xs);
        const xh = dec.predict(z);
        const zs = z.dataSync().slice();
        const zmin = Math.min(...zs), zmax = Math.max(...zs);
        const zr = tf.linspace(zmin - 0.1 * (zmax - zmin), zmax + 0.1 * (zmax - zmin), 120).reshape([120, 1]);
        const curve = dec.predict(zr).arraySync().map((p) => [p[0] * 10, p[1] * 10]);
        const loss = tf.losses.meanSquaredError(xs, xh).dataSync()[0] * 100;
        return { z: zs, xh: xh.arraySync().map((p) => [p[0] * 10, p[1] * 10]), curve, zmin, zmax, loss, epoch };
      });
    }
    function pushFrame() {
      if (!lab.points.length) return;
      const s = snapshot();
      frames.push({ phase: training ? "train" : epoch ? "paused" : "init", arch: training ? ["enc", "z", "dec", "xh", "loss", "opt"] : ["x", "enc", "z", "dec"], title: epoch ? `Epoch ${epoch}` : "Untrained: everything maps near one point", text: epoch ? `Reconstruction MSE <b>${s.loss.toFixed(4)}</b>. The bright curve is the decoder evaluated over the whole range of z: the 1-D manifold the network has learned to represent. ${P().act === "linear" ? "With linear layers the best it can do is a straight line, exactly PCA's first component." : "With non-linear layers it can bend to follow the data, a non-linear PCA."}` : "The encoder squeezes each 2-D point into one number z; the decoder must rebuild the point from z alone. Press train.", tex: "\\min_{\\phi,\\theta}\\;\\frac1n\\sum_i\\big\\lVert x_i-g_\\theta(f_\\phi(x_i))\\big\\rVert^2", st: s, m: { loss: s.loss } });
      if (frames.length > MAXF) frames = frames.filter((f, i) => i === 0 || i % 2 === 1 || i > frames.length - 40);
      lab.charts.loss.set(frames.map((f) => f.m));
      const follow = lab.player.frames.length === 0 || lab.player.index >= lab.player.frames.length - 2;
      if (frames.length === 1) lab.load(frames, { at: "end" }); else { lab.player.replaceAll(frames); if (follow) lab.player.show(frames.length - 1); }
    }
    function loop() { if (!training || stopped) return; trainSteps(8); pushFrame(); timer = setTimeout(loop, 40); }
    function setTraining(v) { training = v; lab.q('[data-b="train"]').textContent = v ? "pause" : "train"; if (v) loop(); else { clearTimeout(timer); pushFrame(); } }
    function reset() { clearTimeout(timer); build(); frames = []; lab.charts.loss.set([]); pushFrame(); if (training) loop(); }

    const errG = L.links.append("g"), curveP = L.mid.append("path").attr("fill", "none").attr("stroke-width", 3), recG = L.mid.append("g"), probeG = L.top.append("g");
    const latEl = lab.R("panel-lat");
    function render(f) {
      const st = f.st; if (!st) return;
      errG.selectAll("*").remove(); recG.selectAll("*").remove();
      lab.points.forEach((p, i) => {
        const r = st.xh[i]; if (!r) return;
        errG.append("line").attr("x1", pl.x(p.x)).attr("y1", pl.y(p.y)).attr("x2", pl.x(r[0])).attr("y2", pl.y(r[1])).attr("stroke", "var(--text-faint)").attr("stroke-opacity", 0.45);
        recG.append("circle").attr("cx", pl.x(r[0])).attr("cy", pl.y(r[1])).attr("r", 2.2).attr("fill", "var(--accent)").attr("opacity", 0.7);
      });
      curveP.attr("d", d3.line().x((p) => pl.x(p[0])).y((p) => pl.y(p[1]))(st.curve)).attr("stroke", "var(--accent)").attr("opacity", 0.85);
      lab.drawPoints({ color: (d) => col(d.label), r: 3.8, edit: false });
      // latent strip
      latEl.innerHTML = "";
      const LP = Lab.xy(latEl, { w: 720, h: 70, xd: [st.zmin - 0.1 * (st.zmax - st.zmin || 1), st.zmax + 0.1 * (st.zmax - st.zmin || 1)], yd: [0, 1], yTicks: 0, xLabel: "z", m: { l: 20, r: 12, t: 6, b: 18 } });
      LP.g.selectAll("circle").data(lab.points).enter().append("circle").attr("cx", (d, i) => LP.x(st.z[i])).attr("cy", (d, i) => LP.y(0.25 + ((i * 37) % 17) / 34)).attr("r", 3.2).attr("fill", (d) => col(d.label)).attr("opacity", 0.85);
      LP.svg.style("cursor", "crosshair").on("mousemove", (ev) => {
        const [px] = d3.pointer(ev); const z = LP.x.invert(px);
        const k = Math.max(0, Math.min(st.curve.length - 1, Math.round(((z - (st.zmin - 0.1 * (st.zmax - st.zmin))) / (1.2 * (st.zmax - st.zmin || 1))) * (st.curve.length - 1))));
        probe = st.curve[k];
        LP.vline(z);
        probeG.selectAll("*").remove();
        probeG.append("circle").attr("cx", pl.x(probe[0])).attr("cy", pl.y(probe[1])).attr("r", 8).attr("fill", "var(--series-4)").attr("stroke", "var(--bg)").attr("stroke-width", 2);
      }).on("mouseleave", () => { probeG.selectAll("*").remove(); LP.vline(null); });
      const nParams = (2 * P().U + P().U) + (P().U * P().U + P().U) + (P().U + 1);
      lab.readout(`<div class="kv"><span>epoch</span><b>${st.epoch}</b><span>recon. MSE</span><b>${st.loss.toFixed(4)}</b><span>bottleneck</span><b>1-D</b><span>encoder params</span><b>${nParams}</b></div><div class="note">The circle has no continuous 1-D chart: watch the decoder curve break it somewhere.</div>`);
    }
    lab.q('[data-b="train"]').onclick = () => setTraining(!training);
    lab.q('[data-b="reset"]').onclick = () => reset();
    lab.watch(["act", "U", "lr"], () => reset(), { format: { lr: (v) => fmt(Math.pow(10, v), 4) } });
    lab.cleanup.push(() => { stopped = true; clearTimeout(timer); [enc, dec, opt].forEach((m) => m && m.dispose()); });
    lab.loadPreset("arc");
    setTraining(true);
    return lab.destroy;
  }

  MLApp.register({
    id: "autoencoder",
    name: "Autoencoder",
    category: "Deep Learning",
    tagline: "bottleneck → learned manifold (TF.js)",
    isNew: true,
    description: "A network trained to copy its input through a one-number bottleneck. Watch it discover the 1-D curve hidden in 2-D data: the decoder sweeps out the learned manifold, reconstruction errors shrink, and hovering the latent strip decodes any code back into the plane. Switch to linear layers and it becomes PCA.",
    info: {
      type: "Unsupervised / self-supervised - Representation learning and dimensionality reduction. Neural, non-linear.",
      scenario: "Learning compact codes for compression, denoising, anomaly detection (high reconstruction error) or as features for downstream models; images, signals, tabular data.",
      inputs: "Unlabeled vectors x (the target is x itself).",
      output: "An encoder f(x) = z (low-dimensional code) and a decoder g(z) = x̂.",
      intuition: {
        definition: "Force data through a narrow middle layer and ask the network to rebuild the input on the other side. To succeed it must keep only the information that matters: the coordinates along the data's underlying structure. The middle layer becomes a learned, non-linear compression.",
        steps: ["Encoder layers map x down to a small code z.", "Decoder layers map z back up to x̂.", "Train both to minimise reconstruction error ‖x − x̂‖².", "Use z as features, or reconstruction error as an anomaly score."],
        applications: ["Image and signal denoising (denoising autoencoders)", "Anomaly detection in manufacturing and fraud", "Pre-training and feature learning", "Generative modelling via the variational autoencoder (VAE)"],
      },
      math: [
        { title: "Encoder and decoder", formula: "z = f_φ(x),  x̂ = g_θ(z)", note: "Usually mirror-image MLPs or CNNs; dim(z) ≪ dim(x) creates the bottleneck." },
        { title: "Objective", formula: "min Σᵢ ‖xᵢ − g_θ(f_φ(xᵢ))‖²", note: "Any reconstruction loss works: cross-entropy for binary pixels, MSE for real values." },
        { title: "Linear autoencoder = PCA", formula: "f, g linear ⇒ optimal code spans the top-k principal subspace", note: "Non-linear activations let the code follow curved manifolds instead." },
        { title: "Denoising autoencoder", formula: "min ‖x − g(f(x + ε))‖²", note: "Corrupt the input, reconstruct the clean version: learns robust features." },
        { title: "Variational autoencoder", formula: "max E_q[log p(x|z)] − KL(q(z|x) ‖ p(z))", note: "Makes the latent space a smooth probability distribution you can sample from." },
      ],
      pipeline: [{ label: "Input x" }, { label: "Encoder", note: "compress" }, { label: "Bottleneck z" }, { label: "Decoder", note: "expand" }, { label: "Reconstruction x̂", accent: "green" }],
      decisionFunction: { text: "z = f_φ(x)  (encode),   x̂ = g_θ(z)  (decode),   anomaly score = ‖x − x̂‖²", mechanism: "Points near the learned manifold reconstruct well; points far from it cannot be expressed through the bottleneck and reconstruct badly, which is why reconstruction error flags anomalies." },
      lossFunction: { text: "L = (1/n) Σᵢ ‖xᵢ − x̂ᵢ‖²", mechanism: "Gradients flow from the reconstruction back through the decoder, the bottleneck and the encoder; both halves are trained jointly end to end." },
      optimization: [{ title: "Adam + mini-batches", formula: "standard deep-learning training", note: "Same machinery as any MLP or CNN." }, { title: "Undercomplete vs regularised", formula: "dim(z) < dim(x), or sparsity / noise / contractive penalties", note: "Without some constraint an overcomplete autoencoder learns the identity." }, { title: "Tied weights", formula: "W_dec = W_encᵀ", note: "Halves parameters; common in linear and shallow AEs." }],
      assumptions: [{ name: "Data lies near a low-dimensional manifold", why: "Otherwise a small code cannot capture it.", check: "Reconstruction error vs code size curve." }, { name: "Training data is representative", why: "Anything unlike training data reconstructs poorly (useful for anomalies, harmful otherwise).", check: "Validate on held-out data." }],
      hyperparameters: [{ name: "code size", range: "1 - 512", increasing: "Better reconstruction, less compression, risk of identity.", strategy: "Smallest size meeting the reconstruction target." }, { name: "depth / width", range: "1 - 6 layers", increasing: "More flexible manifolds.", strategy: "Mirror encoder and decoder." }, { name: "activation", range: "ReLU / tanh / linear", increasing: "Not applicable", strategy: "Non-linear for curved structure; linear recovers PCA." }, { name: "noise level (denoising)", range: "0 - 0.5", increasing: "More robust features, blurrier output.", strategy: "Tune on downstream task." }],
      metrics: ["Reconstruction error (train / validation)", "Downstream task performance using z", "Anomaly-detection ROC-AUC"],
      typicalUses: ["Dimensionality reduction", "Denoising", "Anomaly detection", "Pre-training"],
      diagnostics: ["Plot reconstructions next to inputs.", "Visualise the latent space coloured by known attributes.", "Check whether validation error rises (overfitting)."],
      advantages: ["Learns non-linear structure that PCA cannot.", "Unsupervised: no labels needed.", "Flexible architecture (CNN, RNN, transformer encoders).", "Foundation of VAEs and masked autoencoders."],
      limitations: [{ name: "Latent space not organised by default", note: "nearby codes may decode to unrelated outputs", fix: "VAE or regularisation." }, { name: "Can learn the identity", note: "if the bottleneck is too wide", fix: "smaller code, noise, sparsity." }, { name: "Topology limits", note: "a 1-D code cannot cover a closed loop without a seam", fix: "larger code or periodic latent." }],
      alternatives: [{ name: "PCA", when: "Structure is linear; you want a closed-form, interpretable projection." }, { name: "t-SNE / UMAP", when: "You only need a visualisation, not an invertible mapping." }, { name: "VAE", when: "You want to sample new data from a well-behaved latent space." }],
      pitfalls: [{ problem: "Perfect reconstruction, useless code", solution: "Bottleneck too wide: the network copies. Shrink it or add noise." }, { problem: "Blurry outputs", solution: "MSE averages plausible outputs; use perceptual or adversarial losses for images." }, { problem: "Anomaly detector flags everything new", solution: "Training data did not cover normal variation." }],
      quickRef: [{ name: "Encode", formula: "z = f(x)" }, { name: "Decode", formula: "x̂ = g(z)" }, { name: "Loss", formula: "‖x − x̂‖²" }, { name: "Linear AE", formula: "≡ PCA subspace" }],
      code: `import torch.nn as nn
enc = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 16))
dec = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())
opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-3)
for x, _ in loader:
    x = x.view(len(x), -1)
    loss = nn.functional.binary_cross_entropy(dec(enc(x)), x)
    opt.zero_grad(); loss.backward(); opt.step()`,
      whyChain: [
        { q: "Why does a linear autoencoder recover PCA?", a: "With linear maps the reconstruction is a rank-k projection of the data, and the projection minimising squared error is onto the top-k principal subspace (Eckart–Young). The learned weights span that subspace, though not necessarily the individual eigenvectors." },
        { q: "Why is reconstruction error a good anomaly score?", a: "The network only learned to express points near the training manifold through the bottleneck. A point off the manifold has no good code, so its reconstruction lands on the manifold and the distance between them is large." },
        { q: "Why can't a 1-D code represent a circle cleanly?", a: "A continuous map from a line onto a closed loop must either fold or leave a seam. The decoder curve either misses a piece or jumps, which shows up as a cluster of large errors." },
      ],
      workedExample: {
        setup: "Data on the line y = x in 2-D, code size 1, linear autoencoder.",
        steps: ["Optimal encoder: z = (x + y)/√2 (projection onto the line).", "Optimal decoder: x̂ = z·(1/√2, 1/√2).", "Every point on the line reconstructs exactly; error is 0.", "A point (1, −1) off the line encodes to 0 and reconstructs to (0, 0): error 2."],
        result: "On-manifold points reconstruct perfectly; the off-manifold point has error 2",
      },
    },
    mount,
  });
})();
