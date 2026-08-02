(() => {
  const VN = 8, HN = 6, CELL = 26;
  const sigmoid = (v) => 1 / (1 + Math.exp(-v));

  function makeRBM() {
    const W = Array.from({ length: VN * VN }, () => Array.from({ length: HN }, () => MLU.randn() * 0.35));
    const b = Array.from({ length: VN * VN }, () => MLU.randn() * 0.1);
    const c = Array.from({ length: HN }, () => MLU.randn() * 0.1);
    return { W, b, c };
  }
  function sampleHiddenProbs(rbm, v) {
    return Array.from({ length: HN }, (_, j) => sigmoid(v.reduce((s, vi, i) => s + vi * rbm.W[i][j], 0) + rbm.c[j]));
  }
  function sampleVisibleProbs(rbm, h) {
    return Array.from({ length: VN * VN }, (_, i) => sigmoid(h.reduce((s, hj, j) => s + hj * rbm.W[i][j], 0) + rbm.b[i]));
  }
  function bernoulliSample(probs) { return probs.map((p) => (MLU.rng() < p ? 1 : 0)); }

  function drawSquareGrid(svgSel, n, cell, values, colorFn) {
    const flat = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) flat.push({ i, j, v: values[i * n + j] });
    const sel = svgSel.selectAll("rect").data(flat);
    sel.enter().append("rect").attr("width", cell - 2).attr("height", cell - 2)
      .merge(sel).attr("x", (d) => d.j * cell).attr("y", (d) => d.i * cell).attr("fill", (d) => colorFn(d.v));
    sel.exit().remove();
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint">Gibbs step <b id="rbm-step-n">0</b></span></div>
            <div class="btn-row">
              <button id="rbm-step" class="primary">Gibbs step (v→h→v)</button>
              <button id="rbm-randw">randomize weights</button>
              <button id="rbm-clear">clear canvas</button>
            </div>
          </div>
          <div class="hscroll" style="display:flex;gap:26px;flex-wrap:wrap;justify-content:center;padding:14px 4px;flex:1;align-items:center">
            <div style="text-align:center">
              <div class="stage-hint" style="margin-bottom:6px">visible units (draw here)</div>
              <svg id="rbm-v" width="${VN * CELL}" height="${VN * CELL}"></svg>
            </div>
            <div style="text-align:center">
              <div class="stage-hint" style="margin-bottom:6px">hidden units p(h=1|v)</div>
              <svg id="rbm-h" width="${HN * 34}" height="34"></svg>
            </div>
            <div style="text-align:center">
              <div class="stage-hint" style="margin-bottom:6px">reconstruction p(v=1|h)</div>
              <svg id="rbm-recon" width="${VN * CELL}" height="${VN * CELL}"></svg>
            </div>
          </div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>energy-based sampling</h3>
            <div class="readout" id="rbm-readout">–</div>
            <div class="note">A Restricted Boltzmann Machine has no visible-visible or hidden-hidden connections, so both conditionals factorize: p(h<sub>j</sub>=1|v)=&sigma;(&sum;<sub>i</sub>W<sub>ij</sub>v<sub>i</sub>+c<sub>j</sub>) and p(v<sub>i</sub>=1|h)=&sigma;(&sum;<sub>j</sub>W<sub>ij</sub>h<sub>j</sub>+b<sub>i</sub>). Each "Gibbs step" samples h from v, then resamples v from that h - the block-Gibbs sweep used for contrastive divergence in <code>mla/rbm.py</code>, shown here with fixed random weights rather than trained ones.</div>
          </div>
        </div>
      </div>
    `;

    let rbm = makeRBM();
    let v = new Array(VN * VN).fill(0);
    let hProbs = new Array(HN).fill(0);
    let vRecon = new Array(VN * VN).fill(0);
    let stepN = 0;

    const vSvg = d3.select("#rbm-v"), hSvg = d3.select("#rbm-h"), reconSvg = d3.select("#rbm-recon");

    function render() {
      document.getElementById("rbm-step-n").textContent = stepN;
      drawSquareGrid(vSvg, VN, CELL, v, (val) => (val ? "#f5f5f4" : "#161616"));
      drawSquareGrid(reconSvg, VN, CELL, vRecon, (val) => `rgba(125,211,252,${0.15 + val * 0.75})`);
      const hsel = hSvg.selectAll("rect").data(hProbs);
      hsel.enter().append("rect").attr("width", 28).attr("y", 0)
        .merge(hsel).attr("x", (d, i) => i * 34).attr("height", (d) => Math.max(2, d * 34)).attr("y", (d) => 34 - Math.max(2, d * 34))
        .attr("fill", "#86efac");
      hsel.exit().remove();
      document.getElementById("rbm-readout").innerHTML =
        `visible units on: <b>${v.reduce((a, b) => a + b, 0)}</b> / ${VN * VN}<br>mean hidden activation: <b class="num">${hProbs.length ? MLU.mean(hProbs).toFixed(3) : "–"}</b>`;
    }

    let painting = false, eraseMode = false;
    function paintAt(event) {
      const rect = vSvg.node().getBoundingClientRect();
      const px = event.clientX - rect.left, py = event.clientY - rect.top;
      const j = Math.floor(px / CELL), i = Math.floor(py / CELL);
      if (i >= 0 && i < VN && j >= 0 && j < VN) { v[i * VN + j] = eraseMode ? 0 : 1; render(); }
    }
    vSvg.on("mousedown", (event) => { painting = true; eraseMode = event.button === 2; paintAt(event); });
    vSvg.on("mousemove", (event) => { if (painting) paintAt(event); });
    vSvg.on("contextmenu", (event) => event.preventDefault());
    window.addEventListener("mouseup", () => { painting = false; });

    document.getElementById("rbm-step").addEventListener("click", () => {
      hProbs = sampleHiddenProbs(rbm, v);
      const h = bernoulliSample(hProbs);
      vRecon = sampleVisibleProbs(rbm, h);
      v = bernoulliSample(vRecon);
      stepN++;
      render();
    });
    document.getElementById("rbm-randw").addEventListener("click", () => { rbm = makeRBM(); stepN = 0; render(); });
    document.getElementById("rbm-clear").addEventListener("click", () => { v = new Array(VN * VN).fill(0); hProbs = new Array(HN).fill(0); vRecon = new Array(VN * VN).fill(0); stepN = 0; render(); });

    // seed with a simple pattern so the demo isn't blank on load
    for (let i = 2; i < VN - 2; i++) { v[i * VN + i] = 1; v[i * VN + (VN - 1 - i)] = 1; }
    render();
    return () => {};
  }

  MLApp.register({
    id: "rbm",
    name: "Restricted Boltzmann Machine",
    category: "Deep Learning",
    tagline: "block-Gibbs sampling",
    description: "Draw a binary pattern on the visible layer and alternately sample hidden-given-visible and visible-given-hidden - the block-Gibbs sweep at the heart of RBM training and generation.",
    sourceFile: "mla/rbm.py",
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
