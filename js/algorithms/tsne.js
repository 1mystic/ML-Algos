(() => {
  const W = 640, H = 460, PAD = 36;

  function pairwiseSq(X) {
    const n = X.length;
    const D = MLU.zeros(n, n);
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        let s = 0; for (let k = 0; k < X[i].length; k++) s += (X[i][k] - X[j][k]) ** 2;
        D[i][j] = D[j][i] = s;
      }
    return D;
  }
  function computeP(D, perplexity) {
    const n = D.length;
    const P = MLU.zeros(n, n);
    const target = Math.log(perplexity);
    for (let i = 0; i < n; i++) {
      let betaMin = -Infinity, betaMax = Infinity, beta = 1;
      for (let iter = 0; iter < 50; iter++) {
        let sumP = 0, H = 0;
        const row = new Array(n).fill(0);
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          row[j] = Math.exp(-D[i][j] * beta);
          sumP += row[j];
        }
        sumP = sumP || 1e-12;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const pj = row[j] / sumP;
          if (pj > 1e-12) H -= pj * Math.log(pj);
        }
        const diff = H - target;
        if (Math.abs(diff) < 1e-5) { for (let j = 0; j < n; j++) P[i][j] = row[j] / sumP; break; }
        if (diff > 0) { betaMin = beta; beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2; }
        else { betaMax = beta; beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2; }
        if (iter === 49) for (let j = 0; j < n; j++) P[i][j] = row[j] / sumP;
      }
    }
    // symmetrize
    const Ps = MLU.zeros(n, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) Ps[i][j] = Math.max((P[i][j] + P[j][i]) / (2 * n), 1e-12);
    return Ps;
  }

  function makeClusterData() {
    const dims = 5, clusters = 4, perCluster = 15;
    const centers = Array.from({ length: clusters }, () => Array.from({ length: dims }, () => MLU.randRange(-6, 6)));
    const X = [], labels = [];
    for (let c = 0; c < clusters; c++)
      for (let i = 0; i < perCluster; i++) {
        X.push(centers[c].map((v) => v + MLU.randn() * 1.1));
        labels.push(c);
      }
    return { X, labels };
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint">iteration <b id="tsne-iter">0</b> · KL divergence <b id="tsne-kl">–</b></span></div>
            <div class="btn-row">
              <button id="tsne-step" class="primary">+ 20 iterations</button>
              <button id="tsne-reset">reset embedding</button>
              <button id="tsne-regen">new dataset</button>
            </div>
          </div>
          <svg id="tsne-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">source data is 4 gaussian clusters in 5D — colors show the true (hidden-from-the-algorithm) cluster</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>perplexity <span class="val" id="tsne-perp-val">20</span></h3>
            <input type="range" id="tsne-perp" min="5" max="45" step="1" value="20" />
          </div>
          <div class="control-card">
            <h3>fit</h3>
            <div class="readout" id="tsne-readout">–</div>
            <div class="note">Full (non-Barnes-Hut) t-SNE: high-D affinities P come from a per-point Gaussian kernel calibrated to the target perplexity via binary search; low-D affinities Q use a Student-t kernel; gradient descent with momentum minimizes KL(P‖Q) — the same objective as <code>mla/tsne.py</code>.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#tsne-svg");
    let { X, labels } = makeClusterData();
    let P, Y, Ydelta, iter;

    function initEmbedding() {
      P = computeP(pairwiseSq(X), perplexity());
      Y = X.map(() => [MLU.randn() * 1, MLU.randn() * 1]);
      Ydelta = Y.map(() => [0, 0]);
      iter = 0;
    }
    function perplexity() { return +document.getElementById("tsne-perp").value; }
    initEmbedding();

    function tsneStep() {
      const n = Y.length;
      const Dq = MLU.zeros(n, n), numer = MLU.zeros(n, n);
      let Z = 0;
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) {
          const d2 = (Y[i][0] - Y[j][0]) ** 2 + (Y[i][1] - Y[j][1]) ** 2;
          const num = 1 / (1 + d2);
          numer[i][j] = numer[j][i] = num;
          Z += 2 * num;
        }
      const Q = MLU.zeros(n, n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) Q[i][j] = Math.max(numer[i][j] / Z, 1e-12);

      const momentum = iter < 20 ? 0.5 : 0.8;
      const lr = 100;
      let kl = 0;
      for (let i = 0; i < n; i++) {
        const grad = [0, 0];
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const mult = (P[i][j] - Q[i][j]) * numer[i][j];
          grad[0] += mult * (Y[i][0] - Y[j][0]);
          grad[1] += mult * (Y[i][1] - Y[j][1]);
          kl += P[i][j] * Math.log(P[i][j] / Q[i][j]);
        }
        Ydelta[i][0] = momentum * Ydelta[i][0] - lr * 4 * grad[0];
        Ydelta[i][1] = momentum * Ydelta[i][1] - lr * 4 * grad[1];
      }
      for (let i = 0; i < n; i++) { Y[i][0] += Ydelta[i][0] / n; Y[i][1] += Ydelta[i][1] / n; }
      iter++;
      return kl / 2;
    }

    const ptsG = svg.append("g");
    function render(kl) {
      document.getElementById("tsne-iter").textContent = iter;
      document.getElementById("tsne-perp-val").textContent = perplexity();
      if (kl !== undefined) document.getElementById("tsne-kl").textContent = kl.toFixed(3);

      const ys = Y.map((p) => p[0]), xs = Y.map((p) => p[1]);
      const ext = Math.max(1, ...Y.flat().map(Math.abs)) * 1.15;
      const x = d3.scaleLinear().domain([-ext, ext]).range([PAD, W - PAD]);
      const y = d3.scaleLinear().domain([-ext, ext]).range([H - PAD, PAD]);
      svg.selectAll("g.axes").remove();
      const axesG = svg.insert("g", ":first-child").attr("class", "axes");
      axesG.append("rect").attr("x", PAD).attr("y", PAD).attr("width", W - 2 * PAD).attr("height", H - 2 * PAD).attr("fill", "none").attr("stroke", "var(--border-soft)");

      const data = Y.map((p, i) => ({ x: p[0], y: p[1], label: labels[i] }));
      const sel = ptsG.selectAll("circle").data(data);
      sel.enter().append("circle").attr("r", 5).attr("stroke", "var(--bg)").attr("stroke-width", 1)
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("fill", (d) => MLU.palette[d.label]);
      sel.exit().remove();

      document.getElementById("tsne-readout").innerHTML = `points: <b>${data.length}</b> (5D → 2D)<br>iterations: <b>${iter}</b>`;
    }

    document.getElementById("tsne-step").addEventListener("click", () => {
      let kl = 0; for (let i = 0; i < 20; i++) kl = tsneStep();
      render(kl);
    });
    document.getElementById("tsne-reset").addEventListener("click", () => { initEmbedding(); render(); });
    document.getElementById("tsne-regen").addEventListener("click", () => { ({ X, labels } = makeClusterData()); initEmbedding(); render(); });
    document.getElementById("tsne-perp").addEventListener("change", () => { initEmbedding(); render(); });

    render();
    return () => {};
  }

  MLApp.register({
    id: "tsne",
    name: "t-SNE",
    category: "Unsupervised — Dimensionality Reduction",
    tagline: "high-D neighborhoods → 2D map",
    description: "Embeds 5-dimensional clustered data into 2D by matching high-dimensional neighbor probabilities to a low-dimensional Student-t distribution via gradient descent. Step through iterations to watch clusters separate.",
    sourceFile: "mla/tsne.py",
    info: {
      type: "Unsupervised — Non-linear dimensionality reduction, visualization-focused (no out-of-sample mapping).",
      scenario: "Visualizing high-dimensional data (embeddings, gene expression, learned features) in 2D/3D so that clusters close in high-D stay close in the 2D map.",
      inputs: "A set of high-dimensional points — unsupervised; labels, if any, are only used afterward to color the plot.",
      decisionFunction: {
        text: "yᵢ is found by gradient descent, not computed by a formula, so that low-D neighbor probabilities Q match high-D neighbor probabilities P",
        mechanism: "High-D neighbor probability Pᵢⱼ uses a Gaussian kernel calibrated per-point to a target perplexity; low-D probability Qᵢⱼ uses a heavier-tailed Student-t kernel — the heavy tail is what lets moderately-distant 2D points still represent moderately-similar high-D points without everything collapsing into one blob.",
      },
      lossFunction: {
        text: "KL(P‖Q) = Σᵢⱼ Pᵢⱼ·log(Pᵢⱼ/Qᵢⱼ)",
        mechanism: "Minimized by gradient descent with momentum directly on the 2D coordinates: points that should be neighbors (high P) but currently aren't (low Q) get pulled together; points that shouldn't be neighbors get pushed apart.",
      },
      output: "A 2D (or 3D) coordinate for every input point, suitable for a scatter plot — not a reusable function for new points.",
      parameters: [
        { name: "perplexity", effect: "Roughly 'how many neighbors' each point's Gaussian considers. Small → emphasizes very local structure; large → considers broader neighborhoods." },
        { name: "iterations", effect: "More iterations refine the layout further, with diminishing returns after clusters separate." },
        { name: "learning rate / momentum", effect: "Affect how quickly and how distinctly clusters separate early in training." },
      ],
      metrics: ["No single 'accuracy' — trustworthiness / continuity scores", "Qualitative check: do known clusters/classes separate in the map?"],
      typicalUses: ["Visualizing word/sentence embeddings", "Single-cell genomics cluster visualization", "Inspecting a neural network's learned feature space", "Exploratory cluster discovery"],
      workedExample: {
        setup: "3 points in 1D: x=0, x=1, x=5. Compute p_{j|0} (neighbor probability of points 1 and 2, given point 0) with β=1/(2σ²)=0.5.",
        steps: [
          "Squared distances from point 0: d(0,1)²=1, d(0,2)²=25.",
          "Unnormalized affinities: exp(−0.5×1)=0.6065, exp(−0.5×25)=exp(−12.5)≈3.7×10⁻⁶.",
          "Sum = 0.6065 + 0.0000037 ≈ 0.60651.",
          "p_{1|0} = 0.6065/0.60651 ≈ 0.99994. p_{2|0} = 0.0000037/0.60651 ≈ 0.0000061.",
        ],
        result: "Point 1 (close) gets ~99.99% of point 0's neighbor probability mass; point 2 (far) gets essentially none — exactly the local-neighborhood emphasis t-SNE is built on",
      },
    },
    mount,
  });
})();
