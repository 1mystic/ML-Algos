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
            <div class="note">A Restricted Boltzmann Machine has no visible-visible or hidden-hidden connections, so both conditionals factorize: p(h<sub>j</sub>=1|v)=&sigma;(&sum;<sub>i</sub>W<sub>ij</sub>v<sub>i</sub>+c<sub>j</sub>) and p(v<sub>i</sub>=1|h)=&sigma;(&sum;<sub>j</sub>W<sub>ij</sub>h<sub>j</sub>+b<sub>i</sub>). Each "Gibbs step" samples h from v, then resamples v from that h — the block-Gibbs sweep used for contrastive divergence in <code>mla/rbm.py</code>, shown here with fixed random weights rather than trained ones.</div>
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
    description: "Draw a binary pattern on the visible layer and alternately sample hidden-given-visible and visible-given-hidden — the block-Gibbs sweep at the heart of RBM training and generation.",
    sourceFile: "mla/rbm.py",
    info: {
      type: "Unsupervised — Generative, energy-based, bipartite undirected graphical model.",
      scenario: "Learning a probability distribution over binary (or binarized) data to generate samples, extract latent features, or pretrain deeper networks (stacked RBMs historically formed deep belief networks).",
      inputs: "A binary (or [0,1]-valued) visible vector v; hidden vector h has no observed data — it's inferred.",
      decisionFunction: {
        text: "p(hⱼ=1|v)=σ(Σᵢ Wᵢⱼvᵢ+cⱼ)     p(vᵢ=1|h)=σ(Σⱼ Wᵢⱼhⱼ+bᵢ)",
        mechanism: "Because there are no visible-visible or hidden-hidden connections ('restricted'), both conditionals factorize into independent per-unit sigmoids — exactly what makes alternating block-Gibbs sampling between v and h tractable.",
        plot: { fn: (z) => 1 / (1 + Math.exp(-z)), domain: [-6, 6], color: "var(--accent)", caption: "both conditionals are independent per-unit sigmoids of a weighted sum" },
      },
      lossFunction: {
        text: "−log p(v), where p(v) ∝ Σ_h e^{−E(v,h)}, E(v,h) = −vᵀWh − bᵀv − cᵀh",
        mechanism: "The exact gradient needs an average over the whole distribution and is intractable, so RBMs train with Contrastive Divergence: run a few Gibbs steps from real data to get a 'negative' sample, then nudge weights to make real data more likely than that reconstruction.",
      },
      output: "Samples of the hidden layer (a learned latent/feature representation) or reconstructed visible samples.",
      parameters: [
        { name: "hidden units", effect: "Representational capacity — more hidden units can capture more complex visible-layer structure." },
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
        result: "p(h1=1|v) ≈ 0.668 — hidden unit 1 is likely (but not certain) to activate given this visible pattern",
      },
    },
    mount,
  });
})();
