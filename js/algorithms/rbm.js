(() => {
  const { ui, col, fmt } = Lab;
  const S = 4, NV = S * S;
  const sig = (v) => 1 / (1 + Math.exp(-v));
  // Bars-and-stripes: any subset of rows on, or any subset of columns on.
  function barsAndStripes() {
    const out = new Map();
    for (let m = 0; m < 1 << S; m++) {
      const rows = Array.from({ length: NV }, (_, k) => ((m >> Math.floor(k / S)) & 1));
      const cols = Array.from({ length: NV }, (_, k) => ((m >> (k % S)) & 1));
      out.set(rows.join(""), rows); out.set(cols.join(""), cols);
    }
    return [...out.values()];
  }
  function init(nh) { MLU.seedRng(8); return { W: Array.from({ length: NV }, () => Array.from({ length: nh }, () => MLU.randn() * 0.1)), b: new Array(NV).fill(0), c: new Array(nh).fill(0), nh }; }
  const ph = (R, v) => Array.from({ length: R.nh }, (_, j) => sig(R.c[j] + v.reduce((s, x, i) => s + x * R.W[i][j], 0)));
  const pv = (R, h) => Array.from({ length: NV }, (_, i) => sig(R.b[i] + h.reduce((s, x, j) => s + x * R.W[i][j], 0)));
  const bern = (p) => p.map((q) => (MLU.rng() < q ? 1 : 0));
  const clone = (R) => ({ W: R.W.map((r) => r.slice()), b: R.b.slice(), c: R.c.slice(), nh: R.nh });

  function cdStep(R, batch, lr, k) {
    const dW = R.W.map((r) => r.map(() => 0)), db = new Array(NV).fill(0), dc = new Array(R.nh).fill(0);
    let err = 0;
    for (const v0 of batch) {
      const h0 = ph(R, v0);
      let hs = bern(h0), vk = null, hk = null;
      for (let s = 0; s < k; s++) { vk = pv(R, hs); hk = ph(R, vk); hs = bern(hk); }
      for (let i = 0; i < NV; i++) { for (let j = 0; j < R.nh; j++) dW[i][j] += v0[i] * h0[j] - vk[i] * hk[j]; db[i] += v0[i] - vk[i]; err += (v0[i] - vk[i]) ** 2; }
      for (let j = 0; j < R.nh; j++) dc[j] += h0[j] - hk[j];
    }
    const n = batch.length;
    for (let i = 0; i < NV; i++) { for (let j = 0; j < R.nh; j++) R.W[i][j] += (lr * dW[i][j]) / n; R.b[i] += (lr * db[i]) / n; }
    for (let j = 0; j < R.nh; j++) R.c[j] += (lr * dc[j]) / n;
    return err / (n * NV);
  }
  function fantasy(R, n, steps) {
    return Array.from({ length: n }, () => { let v = bern(new Array(NV).fill(0.5)); for (let s = 0; s < steps; s++) { const h = bern(ph(R, v)); v = bern(pv(R, h)); } return pv(R, bern(ph(R, v))); });
  }
  const isValid = (v, data) => data.some((d) => d.every((x, i) => x === (v[i] > 0.5 ? 1 : 0)));

  function buildFrames(P) {
    const data = barsAndStripes();
    const R = init(P.nh);
    const frames = [];
    // a detailed walk through one CD-1 update on one pattern
    MLU.seedRng(31);
    const v0 = data[5], h0 = ph(R, v0), hs = bern(h0), v1 = pv(R, hs), h1 = ph(R, v1);
    const base = { R: clone(R), sample: data.slice(0, 8) };
    frames.push({ phase: "data", major: true, arch: ["v"], title: `Training data: ${data.length} "bars and stripes" images (4 × 4)`, text: "Each image is all-horizontal stripes or all-vertical bars. An RBM must learn this structure with no labels: a generative model of what valid images look like.", st: { ...base } });
    frames.push({ phase: "pos", arch: ["v", "hgv"], title: "Positive phase: hidden activations driven by a real image", text: "Each hidden unit computes p(h = 1 | v) independently, because there are no hidden-hidden connections (that is the 'restricted' part).", tex: "p(h_j=1\\mid v)=\\sigma\\Big(c_j+\\sum_i v_iW_{ij}\\Big)", st: { ...base, cd: { v0, h0 } } });
    frames.push({ phase: "sample", arch: ["hgv", "hs"], title: "Sample binary hidden states", text: "Stochastic binary sampling acts as a strong regulariser: hidden units cannot pass fine-grained real values back.", tex: "h_j\\sim\\text{Bernoulli}\\big(p(h_j=1\\mid v)\\big)", st: { ...base, cd: { v0, h0, hs } } });
    frames.push({ phase: "recon", arch: ["hs", "vgh"], title: "Reconstruct the image from the hidden sample", text: "Run the same weights backwards. With random weights the reconstruction is grey noise; training will make it look like the input.", tex: "p(v_i=1\\mid h)=\\sigma\\Big(b_i+\\sum_j W_{ij}h_j\\Big)", st: { ...base, cd: { v0, h0, hs, v1 } } });
    frames.push({ phase: "neg", arch: ["vgh", "hgv2", "dw"], title: "Negative phase and the contrastive-divergence update", text: "Correlations measured on data minus correlations measured on the model's own reconstruction. This pushes probability toward real images and away from what the model currently imagines.", tex: "\\Delta W_{ij}=\\varepsilon\\big(\\langle v_ih_j\\rangle_{\\text{data}}-\\langle v_ih_j\\rangle_{\\text{recon}}\\big)", st: { ...base, cd: { v0, h0, hs, v1, h1 } } });
    const show = new Set(Lab.checkpoints(P.epochs, 60));
    for (let e = 1; e <= P.epochs; e++) {
      const err = cdStep(R, MLU.shuffle(data), P.lr, P.k);
      if (!show.has(e)) continue;
      const fant = fantasy(R, 8, 40);
      const valid = fant.filter((v) => isValid(v, data)).length;
      frames.push({ phase: "train", major: e === 1, arch: ["v", "hgv", "vgh", "dw"], title: `Epoch ${e}`, text: `Reconstruction error <b>${err.toFixed(4)}</b>. Bottom row: images the model <i>dreams</i> by running 40 steps of Gibbs sampling from noise. <b>${valid}/8</b> are valid bars-and-stripes patterns. Filters (right) show each hidden unit's weights: they become row and column detectors.`, tex: "E(v,h)=-b^\\top v-c^\\top h-v^\\top Wh,\\qquad p(v,h)=\\frac{e^{-E(v,h)}}{Z}", st: { R: clone(R), sample: data.slice(0, 8), recon: data.slice(0, 8).map((v) => pv(R, ph(R, v))), fant, valid }, m: { err, valid: valid / 8 } });
    }
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "rbm", fps: 4, plane: false,
      stageHTML: `<svg data-r="rbm" class="lab-svg" viewBox="0 0 760 330"></svg>`,
      legend: ui.legend([["var(--text)", "pixel on"], ["var(--series-1)", "positive weight"], ["var(--series-2)", "negative weight"]]),
      tools: ui.btn("retrain", "retrain", true),
      hint: "top: data and reconstructions · middle: one contrastive-divergence step · right: learned filters · bottom: samples dreamed by the model",
      side: [{ title: "model", html: ui.range("nh", "hidden units", 2, 16, 1, 12) + ui.range("lr", "learning rate ε", 0.01, 1, 0.01, 0.5) + ui.range("k", "CD-k Gibbs steps", 1, 5, 1, 1) + ui.range("ep", "epochs", 50, 2000, 50, 1000) }],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "v", label: "v (data)", sub: "16 visible", x: 54, y: 40, kind: "data" },
          { id: "hgv", label: "p(h | v)", sub: "positive phase", x: 180, y: 40, kind: "op" },
          { id: "hs", label: "sample h", sub: "Bernoulli", x: 310, y: 40, kind: "param" },
          { id: "vgh", label: "p(v | h)", sub: "reconstruct", x: 310, y: 112, kind: "op" },
          { id: "hgv2", label: "p(h | v′)", sub: "negative phase", x: 180, y: 112, kind: "op" },
          { id: "dw", label: "ΔW = ε(⟨vh⟩₀ − ⟨vh⟩₁)", sub: "contrastive divergence", x: 150, y: 170, kind: "loss" },
        ],
        edges: [{ from: "v", to: "hgv" }, { from: "hgv", to: "hs" }, { from: "hs", to: "vgh" }, { from: "vgh", to: "hgv2" }, { from: "hgv2", to: "dw" }, { from: "hgv", to: "dw", bend: 30 }],
      },
      phases: { data: { label: "data", color: 3 }, pos: { label: "positive phase", color: 0 }, sample: { label: "sample", color: 4 }, recon: { label: "reconstruct", color: 2 }, neg: { label: "CD update", color: 1 }, train: { label: "training", color: 0 } },
      charts: [{ id: "err", title: "reconstruction error & valid dreams", series: [{ key: "err", label: "recon. MSE", color: "var(--series-1)", digits: 4 }, { key: "valid", label: "valid samples", color: "var(--series-3)", dash: "4 3", digits: 2 }] }],
      onFrame: render,
    });
    function retrain() { lab.load(buildFrames({ nh: lab.val("nh"), lr: lab.val("lr"), k: lab.val("k"), epochs: lab.val("ep") }), { play: true }); }
    const svg = d3.select(lab.R("rbm"));
    function img(x, y, v, cs, opts = {}) {
      const g = svg.append("g").attr("transform", `translate(${x},${y})`);
      for (let i = 0; i < NV; i++) g.append("rect").attr("x", (i % S) * cs).attr("y", Math.floor(i / S) * cs).attr("width", cs - 1).attr("height", cs - 1).attr("rx", 1)
        .attr("fill", opts.signed ? (v[i] >= 0 ? `color-mix(in srgb, var(--series-1) ${Math.round(Math.min(1, v[i] / opts.mx) * 90 + 5)}%, var(--bg-raised))` : `color-mix(in srgb, var(--series-2) ${Math.round(Math.min(1, -v[i] / opts.mx) * 90 + 5)}%, var(--bg-raised))`) : `color-mix(in srgb, var(--text) ${Math.round(v[i] * 92 + 3)}%, var(--bg-raised))`);
      if (opts.ring) g.append("rect").attr("x", -3).attr("y", -3).attr("width", S * cs + 5).attr("height", S * cs + 5).attr("rx", 3).attr("fill", "none").attr("stroke", opts.ring).attr("stroke-width", 2);
      return g;
    }
    const lbl = (x, y, t) => svg.append("text").attr("class", "lbl").attr("x", x).attr("y", y).text(t);
    function render(f) {
      const st = f.st;
      svg.selectAll("*").remove();
      lbl(10, 14, "training images");
      st.sample.forEach((v, k) => img(10 + k * 46, 22, v, 10));
      if (st.recon) { lbl(10, 82, "their reconstructions"); st.recon.forEach((v, k) => img(10 + k * 46, 90, v, 10)); }
      if (st.cd) {
        const { v0, h0, hs, v1, h1 } = st.cd;
        lbl(10, 82, "one CD-1 step on one image");
        img(14, 96, v0, 14, { ring: "var(--series-4)" });
        svg.append("text").attr("class", "lbl").attr("x", 14).attr("y", 170).text("v₀");
        const hx = 120;
        if (h0) h0.forEach((p, j) => { svg.append("circle").attr("cx", hx + j * 22).attr("cy", 110).attr("r", 8).attr("fill", `color-mix(in srgb, var(--series-1) ${Math.round(p * 90)}%, var(--bg-raised))`).attr("stroke", "var(--border)"); svg.append("text").attr("x", hx + j * 22).attr("y", 96).attr("text-anchor", "middle").style("font-size", "8px").attr("fill", "var(--text-faint)").text(p.toFixed(2).replace(/^0/, "")); });
        if (hs) hs.forEach((b, j) => svg.append("rect").attr("x", hx + j * 22 - 6).attr("y", 126).attr("width", 12).attr("height", 12).attr("rx", 2).attr("fill", b ? "var(--text)" : "var(--bg-raised)").attr("stroke", "var(--border)"));
        svg.append("text").attr("class", "lbl").attr("x", hx - 10).attr("y", 160).text("p(h|v₀) and sampled h");
        if (v1) { img(hx + (h0.length) * 22 + 20, 96, v1, 14, { ring: "var(--series-5)" }); svg.append("text").attr("class", "lbl").attr("x", hx + h0.length * 22 + 20).attr("y", 170).text("v₁ (reconstruction)"); }
      }
      // filters
      const R = st.R, mx = Math.max(1e-6, ...R.W.flat().map(Math.abs));
      lbl(400, 14, "filters: each hidden unit's 16 weights");
      for (let j = 0; j < R.nh; j++) img(400 + (j % 6) * 58, 22 + Math.floor(j / 6) * 58, R.W.map((r) => r[j]), 12, { signed: true, mx });
      if (st.fant) {
        lbl(10, 196, `dreamed samples (Gibbs chain from noise) · ${st.valid}/8 valid`);
        st.fant.forEach((v, k) => img(10 + k * 70, 206, v, 15, { ring: isValid(v, barsAndStripes()) ? "var(--series-3)" : "var(--series-2)" }));
      }
      lab.readout(`<div class="kv"><span>visible × hidden</span><b>${NV} × ${R.nh}</b><span>weights</span><b>${NV * R.nh + NV + R.nh}</b><span>recon. error</span><b>${f.m && f.m.err != null ? f.m.err.toFixed(4) : "–"}</b></div><div class="note">Green ring = a valid bars-and-stripes image, red = not.</div>`);
    }
    lab.watch(["nh", "lr", "k", "ep"], () => retrain());
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    retrain();
    return lab.destroy;
  }

  MLApp.register({
    id: "rbm",
    name: "Restricted Boltzmann Machine",
    category: "Deep Learning",
    tagline: "contrastive divergence on bars & stripes",
    description: "An energy-based generative network trained with contrastive divergence. Step through one CD-1 update (positive phase, sampling, reconstruction, negative phase), then watch hidden-unit filters turn into row and column detectors while the model learns to dream valid bars-and-stripes images.",
    info: {
      type: "Unsupervised - Generative, energy-based, bipartite undirected graphical model.",
      scenario: "Learning a probability distribution over binary (or binarized) data to generate samples, extract latent features, or pretrain deeper networks (stacked RBMs historically formed deep belief networks).",
      inputs: "A binary (or [0,1]-valued) visible vector v; hidden vector h has no observed data - it's inferred.",
      intuition: {
        definition: "Assign every configuration of visible and hidden units an <b>energy</b>, and define probability as decreasing in that energy. Training means lowering the energy of patterns you actually observed and raising it everywhere else, so the model comes to prefer real data over everything it could imagine.",
        steps: [
          "Connect visible units to hidden units, with no within-layer links.",
          "That restriction makes both conditionals factorise into simple sigmoids.",
          "Sample hidden from visible and back again (block Gibbs).",
          "Push weights so real data has lower energy than the reconstruction.",
        ],
        applications: [
          "Collaborative filtering, notably in the Netflix Prize era",
          "Unsupervised feature learning before labelled fine-tuning",
          "Layer-wise pretraining of deep belief networks",
          "Generative modelling of binary patterns",
          "Dimensionality reduction as a non-linear alternative to PCA",
        ],
      },
      math: [
        { title: "Energy function", formula: "E(v,h) = −vᵀWh − bᵀv − cᵀh", note: "Bilinear in v and h. Low energy means a compatible configuration, which is what the model learns to assign to real data." },
        { title: "Boltzmann distribution", formula: "p(v,h) = e^{−E(v,h)} / Z,   Z = Σ_{v,h} e^{−E(v,h)}", note: "The partition function Z sums over every possible configuration, so it is exponentially large and intractable. Nearly every difficulty with RBMs traces back to Z." },
        { title: "The restriction", formula: "no v-v or h-h connections ⟹ p(h|v) = Π_j p(hⱼ|v)", note: "This is what 'restricted' means. Without it the conditionals would not factorise and sampling would require slow sequential updates." },
        { title: "Conditionals", formula: "p(hⱼ=1|v) = σ(Σᵢ Wᵢⱼvᵢ + cⱼ),   p(vᵢ=1|h) = σ(Σⱼ Wᵢⱼhⱼ + bᵢ)", note: "Independent per-unit sigmoids, so a whole layer can be sampled in one vectorised operation. This is block Gibbs sampling." },
        { title: "Log-likelihood gradient", formula: "∂log p(v)/∂Wᵢⱼ = ⟨vᵢhⱼ⟩_data − ⟨vᵢhⱼ⟩_model", note: "A positive phase that is easy to compute and a negative phase that requires an expectation over the model distribution, which needs Z and is therefore intractable." },
        { title: "Contrastive divergence", formula: "CD-k: ⟨vᵢhⱼ⟩_data − ⟨vᵢhⱼ⟩_k-step-Gibbs", note: "Approximate the negative phase by running only k Gibbs steps starting from the data rather than to equilibrium. k = 1 works remarkably well in practice despite being a biased estimator." },
        { title: "Free energy", formula: "F(v) = −bᵀv − Σⱼ log(1 + e^{cⱼ + Wⱼᵀv})", note: "Marginalises out the hidden units analytically. Useful for scoring configurations and for pseudo-likelihood evaluation." },
      ],
      pipeline: [
        { label: "Visible v₀", note: "real data" },
        { label: "Sample h₀", note: "σ(Wᵀv + c)" },
        { label: "Reconstruct v₁", note: "σ(Wh + b)" },
        { label: "Resample h₁", note: "one Gibbs step" },
        { label: "ΔW ∝ v₀h₀ᵀ − v₁h₁ᵀ", note: "CD-1 update", accent: "green" },
      ],
      decisionFunction: {
        text: "p(hⱼ=1|v)=σ(Σᵢ Wᵢⱼvᵢ+cⱼ)     p(vᵢ=1|h)=σ(Σⱼ Wᵢⱼhⱼ+bᵢ)",
        mechanism: "Because there are no visible-visible or hidden-hidden connections ('restricted'), both conditionals factorize into independent per-unit sigmoids - exactly what makes alternating block-Gibbs sampling between v and h tractable.",
        plot: { fn: (z) => 1 / (1 + Math.exp(-z)), domain: [-6, 6], color: "var(--accent)", caption: "both conditionals are independent per-unit sigmoids of a weighted sum" },
      },
      lossFunction: {
        text: "−log p(v), where p(v) ∝ Σ_h e^{−E(v,h)}, E(v,h) = −vᵀWh − bᵀv − cᵀh",
        mechanism: "The exact gradient needs an average over the whole distribution and is intractable, so RBMs train with Contrastive Divergence: run a few Gibbs steps from real data to get a 'negative' sample, then nudge weights to make real data more likely than that reconstruction.",
      },
      optimization: [
        { title: "Why the exact gradient is impossible", formula: "⟨·⟩_model requires summing 2^(n_v + n_h) terms", note: "The negative phase is an expectation under the model, which needs the partition function. For even 100 visible units that sum is astronomically large." },
        { title: "CD-1 in practice", formula: "v₀ → h₀ → v₁ → h₁, then ΔW = η(v₀h₀ᵀ − v₁h₁ᵀ)", note: "One up-down-up pass. It optimises the wrong objective, but the direction is close enough that it works well empirically." },
        { title: "Persistent CD", formula: "keep the Gibbs chain running across minibatches instead of restarting from data", note: "The chain gets closer to the true model distribution over time, giving a less biased negative phase and better generative samples." },
        { title: "Sampling detail", formula: "sample h binary, but use probabilities for v when updating", note: "Binary hidden samples act as a regularizing bottleneck; using probabilities for the visible reconstruction reduces sampling noise in the gradient." },
        { title: "Momentum and decay", formula: "momentum 0.5 → 0.9, weight decay ~1e-4", note: "Standard recipe from Hinton's practical guide. Start momentum low while weights are moving fast, raise it once training settles." },
      ],
      output: "Samples of the hidden layer (a learned latent/feature representation) or reconstructed visible samples.",
      assumptions: [
        { name: "Data is binary or in [0,1]", why: "The standard energy function assumes Bernoulli visible units.", check: "Binarize, scale to [0,1], or use a Gaussian-Bernoulli RBM for continuous inputs." },
        { name: "Hidden units are conditionally independent", why: "Guaranteed by the bipartite restriction, and required for tractable block sampling.", check: "Structural, so nothing to verify. It is also why an RBM is weaker than a full Boltzmann machine." },
        { name: "CD bias is tolerable", why: "CD-1 does not follow the true likelihood gradient.", check: "If generative sample quality matters, use persistent CD or a larger k." },
        { name: "Features are comparably scaled", why: "Weights are shared across a single global energy function.", check: "Normalize inputs to a common range." },
        { name: "Reconstruction error is only a proxy", why: "It can fall steadily while the actual likelihood worsens.", check: "Track pseudo-likelihood or annealed importance sampling instead for real evaluation." },
      ],
      regularization: [
        { name: "Weight decay", formula: "L + λ‖W‖²", note: "Keeps weights small so units stay in the sigmoid's responsive range. Typically 1e-4." },
        { name: "Sparsity target", formula: "penalise deviation of mean hidden activation from a small ρ", note: "Encourages each hidden unit to fire rarely, producing more interpretable, part-based features." },
        { name: "Momentum", formula: "0.5 rising to 0.9", note: "Smooths the noisy CD gradient estimate considerably." },
        { name: "Dropout on hidden units", formula: "randomly zero hidden activations", note: "Applicable but less common. The binary sampling already injects substantial noise." },
        { name: "Persistent CD", formula: "carry the chain across updates", note: "Acts as a regularizer by keeping the negative phase closer to the true model distribution." },
      ],
      hyperparameters: [
        { name: "n_hidden", range: "16 - 2000", increasing: "More representational capacity, higher risk of memorising, slower.", strategy: "Start well below the visible count for compression. Larger is acceptable when combined with a sparsity penalty." },
        { name: "CD steps k", range: "1 - 25", increasing: "Less biased gradient, proportionally more compute.", strategy: "CD-1 for feature learning, which is usually enough. Raise k or use persistent CD when generative quality matters." },
        { name: "learning rate", range: "1e-4 - 1e-1", increasing: "Faster learning, then unstable weights and saturated units.", strategy: "0.01 is a common starting point. Roughly, the weight update should be about 1e-3 of the weight magnitude." },
        { name: "batch size", range: "10 - 100", increasing: "Smoother gradient estimates, fewer updates per epoch.", strategy: "10 to 100 is the classic range. Small batches suit the noisy CD gradient." },
        { name: "momentum", range: "0.5 - 0.9", increasing: "More smoothing, faster progress along consistent directions.", strategy: "Start at 0.5 and raise to 0.9 after the first few epochs." },
        { name: "weight decay", range: "1e-5 - 1e-3", increasing: "Smaller weights, less overfitting, potential underfitting.", strategy: "1e-4 is the standard default." },
        { name: "sparsity target", range: "0.01 - 0.1", increasing: "Not applicable", strategy: "Set around 0.05 when you want part-based, interpretable features." },
      ],
      metrics: ["Reconstruction error", "Pseudo-likelihood (a tractable proxy for the true likelihood)", "Annealed importance sampling estimate of log Z", "Quality of downstream features when used for pretraining"],
      typicalUses: ["Collaborative filtering (the Netflix-prize-era RBM recommender)", "Unsupervised feature learning / pretraining", "Generative modeling of binary data"],
      diagnostics: [
        "Visualise the weight matrix as images. On MNIST-like data a healthy RBM shows stroke and edge detectors; unstructured noise means training failed.",
        "Monitor hidden-unit activation rates. Units always on or always off are dead and contribute nothing.",
        "Do not trust reconstruction error alone. It can fall while the model distribution gets worse, because CD optimises something other than likelihood.",
        "Watch weight magnitudes. Rapid growth means the learning rate is too high and units are saturating.",
        "Run a long Gibbs chain and look at the samples. If they collapse to a few patterns, the model has not captured the distribution.",
      ],
      advantages: [
        "The bipartite restriction makes exact block Gibbs sampling possible, which a general Boltzmann machine does not allow.",
        "A genuine generative model: you can sample new data, not just encode it.",
        "Learns useful unsupervised features from unlabelled data.",
        "Historically important as the layer-wise pretraining that first made deep networks trainable.",
        "Hidden activations serve as a non-linear dimensionality reduction.",
        "Handles missing inputs naturally, which is why it suited collaborative filtering.",
      ],
      limitations: [
        { name: "Intractable partition function", note: "the true likelihood cannot be computed or optimised directly", fix: "contrastive divergence, and AIS for evaluation." },
        { name: "CD is a biased estimator", note: "it does not follow the likelihood gradient", fix: "persistent CD, or larger k." },
        { name: "Binary units by default", note: "continuous data needs a modified energy function", fix: "Gaussian-Bernoulli RBM, or binarize." },
        { name: "Hard to evaluate", note: "reconstruction error is a poor proxy for likelihood", fix: "pseudo-likelihood or annealed importance sampling." },
        { name: "Fiddly to train", note: "many interacting hyperparameters and no clean convergence signal", fix: "follow an established practical recipe." },
        { name: "Largely superseded", note: "VAEs, GANs, and diffusion models dominate generative modelling", fix: "use RBMs for understanding, modern models for results." },
      ],
      alternatives: [
        { name: "Variational autoencoder", when: "You want a modern generative model with a tractable training objective and a usable latent space." },
        { name: "Autoencoder", when: "You only need feature learning or compression, not a probability model." },
        { name: "GAN or diffusion model", when: "High-fidelity generation, especially for images." },
        { name: "Matrix factorization", when: "Collaborative filtering. Simpler and usually just as strong." },
      ],
      pitfalls: [
        { problem: "Reconstruction error falls but samples are poor", solution: "Expected. Reconstruction error is not likelihood. Use persistent CD and evaluate with pseudo-likelihood." },
        { problem: "Weights grow without bound", solution: "Learning rate too high. Lower it and add weight decay." },
        { problem: "Hidden units are always on or always off", solution: "Dead units from saturation. Lower the learning rate, add a sparsity penalty, re-check initialisation." },
        { problem: "Continuous data trains badly", solution: "Bernoulli units assume binary input. Use a Gaussian-Bernoulli RBM or binarize." },
        { problem: "Samples collapse to a few patterns", solution: "The Gibbs chain is mixing poorly. Use persistent CD with more chains." },
        { problem: "Cannot tell whether training is working", solution: "There is no clean convergence signal. Visualise the filters, which is the most reliable check." },
      ],
      quickRef: [
        { name: "Energy", formula: "E(v,h) = −vᵀWh − bᵀv − cᵀh" },
        { name: "Joint probability", formula: "p(v,h) = e^{−E}/Z" },
        { name: "Hidden conditional", formula: "p(hⱼ=1|v) = σ(Wⱼᵀv + cⱼ)" },
        { name: "Visible conditional", formula: "p(vᵢ=1|h) = σ(Wᵢh + bᵢ)" },
        { name: "Free energy", formula: "F(v) = −bᵀv − Σⱼ log(1+e^{cⱼ+Wⱼᵀv})" },
        { name: "True gradient", formula: "⟨vh⟩_data − ⟨vh⟩_model" },
        { name: "CD-k update", formula: "ΔW = η(v₀h₀ᵀ − v_k h_kᵀ)" },
        { name: "Why 'restricted'", formula: "no v-v or h-h edges" },
      ],
      code: `from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# Bernoulli units expect inputs in [0, 1].
X01 = X / 16.0

rbm = BernoulliRBM(
    n_components=256,     # hidden units
    learning_rate=0.01,
    batch_size=32,
    n_iter=40,
    random_state=42,      # sklearn implements CD-1
)

# The classic use: unsupervised pretraining, then a supervised head.
clf = Pipeline([("rbm", rbm), ("logreg", LogisticRegression(max_iter=1000))])
clf.fit(X01_train, y_train)

# Filters are the most reliable diagnostic: they should look structured.
filters = rbm.components_          # (n_hidden, n_visible)

# Reconstruction error is a poor proxy; prefer pseudo-likelihood.
print(rbm.score_samples(X01_test).mean())`,
      whyChain: [
        { q: "What does 'restricted' actually restrict?", a: "Connections within a layer. A general Boltzmann machine allows visible-visible and hidden-hidden edges, which makes the conditionals depend on other units in the same layer and forces slow sequential sampling. Removing them makes each layer conditionally independent given the other, so a whole layer can be sampled at once." },
        { q: "Why is the log-likelihood gradient intractable?", a: "It is the difference between a data expectation and a model expectation. The data term is easy. The model term requires averaging over the model's own distribution, which needs the partition function Z, a sum over exponentially many configurations." },
        { q: "How does contrastive divergence get around that?", a: "Instead of sampling from the model to equilibrium, it starts a Gibbs chain at a real data point and runs it just k steps. The resulting sample stands in for the model expectation. It is biased, because the chain has not mixed, but the gradient direction is close enough to be useful." },
        { q: "Why does CD-1 work at all if it is so crude?", a: "The chain starts at the data, so after one step it has moved in the direction the model would drag the data. That difference is enough to tell the weights which way to push, even though the magnitude is wrong. It is a direction estimator, not a gradient estimator." },
        { q: "What does the energy function mean intuitively?", a: "It scores compatibility. Configurations where active visible units connect through positive weights to active hidden units get low energy and therefore high probability. Training lowers the energy of observed patterns and, through the negative phase, raises it for patterns the model currently prefers." },
        { q: "Why is reconstruction error a misleading metric?", a: "The model is trained to shape a probability distribution, not to reconstruct. An RBM can reconstruct its training data well while placing much of its probability mass on configurations that never occur. Reconstruction error measures a one-step round trip, not distributional fit." },
        { q: "Why were RBMs historically so important?", a: "Before good initialisation and ReLU, deep networks could not be trained end to end. Stacking RBMs and training each layer greedily gave a sensible starting point for the whole network, which is what made deep belief networks work and reignited interest in deep learning." },
        { q: "Why is nobody using them now?", a: "Better initialisation, ReLU, batch normalisation, and residual connections made pretraining unnecessary, and VAEs, GANs, and diffusion models generate far better samples with tractable objectives. RBMs remain valuable for understanding energy-based modelling." },
      ],
      parameters: [
        { name: "hidden units", effect: "Representational capacity - more hidden units can capture more complex visible-layer structure." },
        { name: "CD-k (Gibbs steps)", effect: "More contrastive-divergence steps give a better gradient estimate at higher compute cost." },
        { name: "learning rate", effect: "Size of the weight update during contrastive-divergence training." },
      ],
      metrics: ["Reconstruction error", "Pseudo-likelihood (a tractable proxy for the true likelihood)", "Quality of downstream features when used for pretraining"],
      typicalUses: ["Collaborative filtering (the Netflix-prize-era RBM recommender)", "Unsupervised feature learning / pretraining", "Generative modeling of binary data"],
      workedExample: {
        setup: "3 visible units v=[1,0,1]. Weights into hidden unit 1: W=[0.5,−0.3,0.2], bias c1=0.",
        steps: [
          "z = W·v + c1 = 0.5×1 + (−0.3)×0 + 0.2×1 + 0 = 0.7.",
          "p(h1=1|v) = σ(0.7) = 1/(1+e⁻⁰·⁷) = 1/(1+0.4966) = 1/1.4966.",
        ],
        result: "p(h1=1|v) ≈ 0.668 - hidden unit 1 is likely (but not certain) to activate given this visible pattern",
      },
    },
    mount,
  });
})();
