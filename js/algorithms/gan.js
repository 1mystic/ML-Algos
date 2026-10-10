(() => {
  const { ui, col, fmt } = Lab;
  const MAXF = 300, SC = 5, NB = 60;
  const DIST = {
    two: { name: "two modes", sample: () => (MLU.rng() < 0.5 ? -4 + MLU.randn() * 0.7 : 3 + MLU.randn() * 1), pdf: (x) => 0.5 * gpdf(x, -4, 0.7) + 0.5 * gpdf(x, 3, 1) },
    three: { name: "three modes", sample: () => { const r = MLU.rng(); return r < 1 / 3 ? -5 + MLU.randn() * 0.6 : r < 2 / 3 ? 0 + MLU.randn() * 0.6 : 5 + MLU.randn() * 0.6; }, pdf: (x) => (gpdf(x, -5, 0.6) + gpdf(x, 0, 0.6) + gpdf(x, 5, 0.6)) / 3 },
    gauss: { name: "single Gaussian", sample: () => 2 + MLU.randn() * 1.5, pdf: (x) => gpdf(x, 2, 1.5) },
    uniform: { name: "uniform [−5, 5]", sample: () => MLU.randRange(-5, 5), pdf: (x) => (x >= -5 && x <= 5 ? 0.1 : 0) },
    skew: { name: "skewed (shifted exponential)", sample: () => -4 + -Math.log(Math.max(1e-9, MLU.rng())) * 2.2, pdf: (x) => (x >= -4 ? Math.exp(-(x + 4) / 2.2) / 2.2 : 0) },
  };
  function gpdf(x, m, s) { return Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI)); }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "gan", fps: 12, plane: { h: 340, yd: [0, 1.05], xLabel: "x", yLabel: "density / D(x)" },
      legend: ui.legend([["var(--series-1)", "real data p(x)"], ["var(--series-2)", "generator samples"], ["var(--accent)", "discriminator D(x)", "line"]]),
      tools: ui.btn("train", "train", true) + ui.btn("reset", "reset both networks"),
      hint: "the discriminator curve is the probability it assigns to 'real' at each x; at equilibrium it flattens to 0.5",
      side: [
        { title: "real data", html: ui.select("dist", "target distribution", Object.entries(DIST).map(([k, v]) => [k, v.name]), "two") },
        { title: "training", html: ui.range("lrd", "D learning rate (log₁₀)", -4, -1.5, 0.25, -2.5, "0.003") + ui.range("lrg", "G learning rate (log₁₀)", -4, -1.5, 0.25, -2.5, "0.003") + ui.range("k", "D steps per G step", 1, 5, 1, 1) + ui.range("U", "hidden units", 4, 32, 4, 16) + ui.select("gl", "generator loss", [["ns", "non-saturating −log D(G(z))"], ["mm", "minimax log(1 − D(G(z)))"]], "ns") },
      ],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "z", label: "z ~ N(0, 1)", sub: "noise", x: 54, y: 40, kind: "data" },
          { id: "G", label: "generator G", sub: "MLP", x: 170, y: 40, kind: "op" },
          { id: "fake", label: "x_fake", x: 280, y: 40, w: 66, kind: "param" },
          { id: "real", label: "x_real", sub: "data", x: 54, y: 120, kind: "data" },
          { id: "D", label: "discriminator D", sub: "real or fake?", x: 220, y: 120, kind: "op" },
          { id: "lossD", label: "D loss", sub: "BCE", x: 330, y: 120, w: 64, kind: "loss" },
          { id: "lossG", label: "G loss", sub: "fool D", x: 300, y: 172, w: 64, kind: "loss" },
        ],
        edges: [{ from: "z", to: "G" }, { from: "G", to: "fake" }, { from: "fake", to: "D" }, { from: "real", to: "D" }, { from: "D", to: "lossD" }, { from: "D", to: "lossG" }, { from: "lossG", to: "G", dash: true, label: "update G", bend: 40 }],
      },
      phases: { init: { label: "init", color: 3 }, train: { label: "adversarial training", color: 0 }, paused: { label: "paused", color: 4 } },
      panels: [{ id: "map", title: "generator mapping z → x", sub: "a learned transport of the noise" }],
      charts: [{ id: "loss", title: "losses", series: [{ key: "ld", label: "D", color: "var(--series-1)" }, { key: "lg", label: "G", color: "var(--series-2)", dash: "4 3" }] }, { id: "tv", title: "distance to real distribution", series: [{ key: "tv", label: "total variation", color: "var(--series-5)" }], yd: [0, 1] }],
      onFrame: render,
    });
    const pl = lab.plane, L = pl.layers;
    lab.player.setLive(true);
    let G = null, D = null, og = null, od = null, training = false, timer = null, step = 0, frames = [], stopped = false;
    const P = () => ({ dist: lab.val("dist"), lrd: Math.pow(10, lab.val("lrd")), lrg: Math.pow(10, lab.val("lrg")), k: lab.val("k"), U: lab.val("U"), gl: lab.val("gl") });
    function build() {
      [G, D, og, od].forEach((m) => m && m.dispose());
      const p = P();
      G = tf.sequential({ layers: [tf.layers.dense({ units: p.U, activation: "tanh", inputShape: [1] }), tf.layers.dense({ units: p.U, activation: "tanh" }), tf.layers.dense({ units: 1 })] });
      D = tf.sequential({ layers: [tf.layers.dense({ units: p.U, activation: "tanh", inputShape: [1] }), tf.layers.dense({ units: p.U, activation: "tanh" }), tf.layers.dense({ units: 1, activation: "sigmoid" })] });
      og = tf.train.adam(p.lrg, 0.5); od = tf.train.adam(p.lrd, 0.5);
      step = 0;
    }
    const realBatch = (n) => Array.from({ length: n }, () => [DIST[P().dist].sample() / SC]);
    let last = { ld: null, lg: null };
    function trainSteps(n) {
      const p = P(), B = 128;
      for (let s = 0; s < n; s++) {
        tf.tidy(() => {
          for (let k = 0; k < p.k; k++) {
            const xr = tf.tensor2d(realBatch(B)), z = tf.randomNormal([B, 1]);
            const xf = G.predict(z);
            const ld = od.minimize(() => {
              const dr = D.apply(xr), df = D.apply(xf);
              return tf.add(tf.losses.logLoss(tf.onesLike(dr), dr), tf.losses.logLoss(tf.zerosLike(df), df));
            }, true, D.trainableWeights.map((w) => w.val));
            last.ld = ld.dataSync()[0];
          }
          const z = tf.randomNormal([B, 1]);
          const lg = og.minimize(() => {
            const df = D.apply(G.apply(z));
            return p.gl === "ns" ? tf.losses.logLoss(tf.onesLike(df), df) : tf.neg(tf.losses.logLoss(tf.zerosLike(df), df));
          }, true, G.trainableWeights.map((w) => w.val));
          last.lg = lg.dataSync()[0];
        });
        step++;
      }
    }
    function snapshot() {
      return tf.tidy(() => {
        const z = tf.randomNormal([1500, 1]);
        const fake = G.predict(z).dataSync().map((v) => v * SC);
        const xs = d3.range(-10, 10.01, 0.2);
        const dcurve = D.predict(tf.tensor2d(xs.map((x) => [x / SC]))).dataSync();
        const zs = d3.range(-3, 3.01, 0.1);
        const gmap = G.predict(tf.tensor2d(zs.map((v) => [v]))).dataSync().map((v) => v * SC);
        const bins = d3.bin().domain([-10, 10]).thresholds(NB)(Array.from(fake).filter((v) => v > -10 && v < 10));
        const w = 20 / NB;
        const hist = bins.map((b) => ({ x0: b.x0, x1: b.x1, p: b.length / (1500 * w) }));
        const pdf = DIST[P().dist].pdf;
        const tv = 0.5 * hist.reduce((s, b) => s + Math.abs(b.p - pdf((b.x0 + b.x1) / 2)) * w, 0);
        return { hist, d: xs.map((x, i) => [x, dcurve[i]]), gmap: zs.map((v, i) => [v, gmap[i]]), tv, step, m: { ld: last.ld, lg: last.lg, tv } };
      });
    }
    function pushFrame() {
      const s = snapshot();
      frames.push({ phase: training ? "train" : step ? "paused" : "init", arch: training ? ["z", "G", "fake", "real", "D", "lossD", "lossG"] : ["z", "G", "fake"], title: step ? `Step ${step}` : "Untrained generator: noise mapped to a narrow blob", text: step ? `D loss <b>${fmt(last.ld)}</b>, G loss <b>${fmt(last.lg)}</b>, total-variation distance <b>${s.tv.toFixed(3)}</b>. ${s.tv < 0.15 ? "The generated histogram now covers the real density; D(x) hovers near 0.5 because it can no longer tell them apart." : "D rises where real data is denser than fakes, and G follows that gradient to move samples there. Watch for <b>mode collapse</b>: all fakes piling into one mode."}` : "The generator turns standard normal noise into samples. The discriminator is a classifier. They are trained against each other.", tex: "\\min_G\\max_D\\;\\mathbb E_{x\\sim p_{data}}[\\log D(x)]+\\mathbb E_{z\\sim p_z}[\\log(1-D(G(z)))]", st: s, m: s.m });
      if (frames.length > MAXF) frames = frames.filter((f, i) => i === 0 || i % 2 === 1 || i > frames.length - 40);
      for (const c of ["loss", "tv"]) lab.charts[c].set(frames.map((f) => f.m));
      const follow = lab.player.frames.length === 0 || lab.player.index >= lab.player.frames.length - 2;
      if (frames.length === 1) lab.load(frames, { at: "end" }); else { lab.player.replaceAll(frames); if (follow) lab.player.show(frames.length - 1); }
    }
    function loop() { if (!training || stopped) return; trainSteps(5); pushFrame(); timer = setTimeout(loop, 30); }
    function setTraining(v) { training = v; lab.q('[data-b="train"]').textContent = v ? "pause" : "train"; if (v) loop(); else { clearTimeout(timer); pushFrame(); } }
    function reset() { clearTimeout(timer); MLU.seedRng(3); build(); frames = []; last = { ld: null, lg: null }; for (const c of ["loss", "tv"]) lab.charts[c].set([]); pushFrame(); if (training) loop(); }
    const realP = L.under.append("path"), histG = L.mid.append("g"), dP = L.top.append("path").attr("fill", "none").attr("stroke", "var(--accent)").attr("stroke-width", 2.4);
    const halfP = L.under.append("line");
    const mapEl = lab.R("panel-map");
    function render(f) {
      const st = f.st; if (!st) return;
      const pdf = DIST[P().dist].pdf;
      const xs = d3.range(-10, 10.01, 0.05);
      const mx = Math.max(0.05, ...xs.map(pdf), ...st.hist.map((b) => b.p));
      const k = 0.9 / mx;
      realP.attr("d", d3.area().x((x) => pl.x(x)).y0(pl.y(0)).y1((x) => pl.y(pdf(x) * k))(xs)).attr("fill", "var(--series-1)").attr("fill-opacity", 0.25).attr("stroke", "var(--series-1)").attr("stroke-width", 1.6);
      const b = histG.selectAll("rect").data(st.hist);
      b.exit().remove();
      b.enter().append("rect").merge(b).attr("x", (d) => pl.x(d.x0) + 0.5).attr("width", (d) => Math.max(0, pl.x(d.x1) - pl.x(d.x0) - 1)).attr("y", (d) => pl.y(Math.min(1.02, d.p * k))).attr("height", (d) => pl.y(0) - pl.y(Math.min(1.02, d.p * k))).attr("fill", "var(--series-2)").attr("opacity", 0.55);
      dP.attr("d", d3.line().x((p) => pl.x(p[0])).y((p) => pl.y(p[1]))(st.d));
      halfP.attr("x1", pl.x(-10)).attr("x2", pl.x(10)).attr("y1", pl.y(0.5)).attr("y2", pl.y(0.5)).attr("stroke", "var(--text-faint)").attr("stroke-dasharray", "4 4");
      mapEl.innerHTML = "";
      const MP = Lab.xy(mapEl, { w: 360, h: 140, xd: [-3, 3], yd: [-10, 10], xLabel: "z", yLabel: "G(z)" });
      MP.line("g", st.gmap, { color: "var(--series-2)", width: 2.4 });
      lab.readout(`<div class="kv"><span>step</span><b>${st.step}</b><span>D loss</span><b>${fmt(f.m.ld)}</b><span>G loss</span><b>${fmt(f.m.lg)}</b><span>TV distance</span><b>${st.tv.toFixed(3)}</b></div><div class="note">At the theoretical optimum D(x) = ½ everywhere and the D loss is 2 ln 2 ≈ 1.386.</div>`);
    }
    lab.q('[data-b="train"]').onclick = () => setTraining(!training);
    lab.q('[data-b="reset"]').onclick = () => reset();
    lab.watch(["dist", "lrd", "lrg", "k", "U", "gl"], () => reset(), { format: { lrd: (v) => fmt(Math.pow(10, v), 4), lrg: (v) => fmt(Math.pow(10, v), 4) } });
    lab.cleanup.push(() => { stopped = true; clearTimeout(timer); [G, D, og, od].forEach((m) => m && m.dispose()); });
    reset();
    setTraining(true);
    return lab.destroy;
  }

  MLApp.register({
    id: "gan",
    name: "Generative Adversarial Network",
    category: "Deep Learning",
    tagline: "generator vs discriminator (TF.js)",
    isNew: true,
    description: "Two networks in a game: a generator turns noise into samples, a discriminator tries to tell them from real data. Watch the fake histogram chase the real density, the discriminator's belief curve flatten toward ½, and the mapping from noise to data take shape, including the occasional mode collapse.",
    info: {
      type: "Deep learning - Generative model. Implicit density, trained by a two-player minimax game.",
      scenario: "Generating realistic samples (images, audio, tabular data) when you need sharp outputs and do not need explicit likelihoods.",
      inputs: "Samples from the real data distribution; a noise distribution p(z) (usually Gaussian).",
      output: "A generator network G(z) that maps noise to new samples resembling the data.",
      intuition: {
        definition: "A forger (generator) and a detective (discriminator) train together. The detective learns to spot fakes; the forger learns from the detective's gradients how to make fakes more convincing. If both keep improving, the fakes become indistinguishable from real data.",
        steps: ["Sample real data and generate fakes from noise.", "Update D to label real as 1 and fake as 0.", "Update G so that D labels its fakes as real.", "Alternate until D is reduced to guessing."],
        applications: ["Photorealistic image synthesis (StyleGAN)", "Image-to-image translation (pix2pix, CycleGAN)", "Super-resolution and inpainting", "Synthetic tabular and medical data"],
      },
      math: [
        { title: "Minimax objective", formula: "min_G max_D E_x[log D(x)] + E_z[log(1 − D(G(z)))]", note: "D is a binary classifier; G is trained to make it fail." },
        { title: "Optimal discriminator", formula: "D*(x) = p_data(x) / (p_data(x) + p_g(x))", note: "For a fixed generator; equals ½ wherever the two densities agree." },
        { title: "What G minimises", formula: "C(G) = −log 4 + 2·JS(p_data ‖ p_g)", note: "With an optimal D the game minimises the Jensen–Shannon divergence; the global optimum is p_g = p_data." },
        { title: "Non-saturating generator loss", formula: "max_G E_z[log D(G(z))]", note: "Same fixed point, but strong gradients early when D easily rejects fakes." },
        { title: "Wasserstein GAN", formula: "min_G max_{‖f‖_L≤1} E[f(x)] − E[f(G(z))]", note: "Earth-mover distance gives useful gradients even when distributions do not overlap." },
      ],
      pipeline: [{ label: "Noise z" }, { label: "Generator" }, { label: "Real + fake" }, { label: "Discriminator" }, { label: "Alternate updates", accent: "green" }],
      decisionFunction: { text: "x_new = G(z),  z ~ N(0, I)", mechanism: "Sampling is a single forward pass of the generator: no iterative procedure, which makes GANs fast at inference compared with diffusion models." },
      lossFunction: { text: "L_D = −E[log D(x)] − E[log(1 − D(G(z)))],   L_G = −E[log D(G(z))]", mechanism: "D minimises binary cross-entropy; G minimises the non-saturating loss. Losses do not decrease monotonically: they oscillate as each player adapts to the other.", plot: { fn: (d) => -Math.log(Math.max(1e-4, d)), domain: [0.01, 1], fn2: (d) => Math.log(Math.max(1e-4, 1 - d)), yDomain: [-5, 5], caption: "generator loss as a function of D(G(z)): non-saturating −log D (solid) has a steep gradient when D rejects fakes; minimax log(1 − D) (dashed) is flat there" } },
      optimization: [{ title: "Alternating gradient steps", formula: "k steps of D, 1 step of G", note: "Simultaneous gradient descent on a game can cycle instead of converge." }, { title: "Adam with β₁ = 0.5", formula: "lower momentum", note: "The DCGAN recipe; high momentum amplifies oscillations." }, { title: "Stabilisers", formula: "spectral norm, gradient penalty, two time-scale updates", note: "Constrain D so its gradients stay informative." }],
      assumptions: [{ name: "Balanced players", why: "A too-strong D gives vanishing gradients; a too-weak D gives no signal.", check: "Watch both losses; neither should go to 0." }, { name: "Enough capacity", why: "G must be able to represent the target.", check: "Mode coverage on held-out data." }],
      hyperparameters: [{ name: "learning rates (G, D)", range: "1e-5 - 1e-3", increasing: "Faster but less stable.", strategy: "Two time-scale: D slightly faster." }, { name: "D steps per G step", range: "1 - 5", increasing: "Stronger D.", strategy: "1 for standard GANs, 5 for WGAN critics." }, { name: "latent dimension", range: "1 - 512", increasing: "More expressive generator.", strategy: "100–512 for images." }, { name: "batch size", range: "32 - 2048", increasing: "Smoother gradients, better diversity.", strategy: "As large as memory allows." }],
      metrics: ["FID (Fréchet Inception Distance)", "Inception Score", "Precision / recall of the generated distribution", "Visual inspection"],
      typicalUses: ["Image synthesis and editing", "Data augmentation", "Domain translation"],
      diagnostics: ["D loss near 0 and G loss exploding: D has won; lower its learning rate.", "All samples identical: mode collapse.", "Losses oscillating without settling is normal; judge by sample quality."],
      advantages: ["Very sharp samples.", "Fast single-pass sampling.", "No explicit likelihood needed.", "Flexible conditioning (class, image, text)."],
      limitations: [{ name: "Unstable training", note: "oscillation, divergence", fix: "WGAN-GP, spectral norm, careful learning rates." }, { name: "Mode collapse", note: "G covers only part of the data", fix: "minibatch discrimination, unrolled GANs, diffusion models." }, { name: "No likelihood", note: "hard to evaluate", fix: "FID / precision-recall." }],
      alternatives: [{ name: "Diffusion models", when: "Best quality and diversity, sampling speed less important." }, { name: "VAE", when: "Stable training and an explicit latent space with likelihood bounds." }, { name: "Normalising flows", when: "Exact likelihoods are required." }],
      pitfalls: [{ problem: "Generator produces the same output for all z", solution: "Mode collapse: lower G's learning rate, add noise, or switch to WGAN." }, { problem: "D wins instantly", solution: "Use the non-saturating loss and weaken D." }, { problem: "Judging training by the loss curve", solution: "GAN losses are not progress meters; inspect samples and FID." }],
      quickRef: [{ name: "Game", formula: "min_G max_D V(D, G)" }, { name: "Optimal D", formula: "p_data / (p_data + p_g)" }, { name: "G loss (NS)", formula: "−log D(G(z))" }, { name: "Equilibrium", formula: "p_g = p_data, D ≡ ½" }],
      code: `G = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
D = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1), nn.Sigmoid())
oG = torch.optim.Adam(G.parameters(), 3e-3, betas=(0.5, 0.999))
oD = torch.optim.Adam(D.parameters(), 3e-3, betas=(0.5, 0.999))
bce = nn.BCELoss()
for x in loader:
    z = torch.randn(len(x), 1)
    lossD = bce(D(x), torch.ones(len(x), 1)) + bce(D(G(z).detach()), torch.zeros(len(x), 1))
    oD.zero_grad(); lossD.backward(); oD.step()
    lossG = bce(D(G(z)), torch.ones(len(x), 1))      # non-saturating
    oG.zero_grad(); lossG.backward(); oG.step()`,
      whyChain: [
        { q: "Why does the discriminator end at ½?", a: "The optimal D for a fixed G is p_data/(p_data + p_g). When the generator matches the data exactly, the ratio is ½ everywhere: the best classifier can only guess." },
        { q: "Why use the non-saturating loss?", a: "Early on D rejects fakes confidently, so D(G(z)) ≈ 0 where log(1 − D) is flat and gives G almost no gradient. −log D is steep there, so G learns quickly from the same fixed point." },
        { q: "What is mode collapse?", a: "G finds a few outputs that fool the current D and maps many z to them. D then learns to reject those, G jumps to another mode, and the pair can cycle instead of covering the whole distribution." },
      ],
      workedExample: {
        setup: "At one x, p_data(x) = 0.3 and p_g(x) = 0.1.",
        steps: ["Optimal D(x) = 0.3 / (0.3 + 0.1) = 0.75.", "D says 'probably real' because real data is denser here.", "G's gradient pushes more samples toward x until p_g(x) = 0.3 and D(x) = 0.5."],
        result: "D*(x) = 0.75 now; 0.5 once the generator matches the data at x",
      },
    },
    mount,
  });
})();
