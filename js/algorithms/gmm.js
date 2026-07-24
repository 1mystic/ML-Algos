(() => {
  const W = 640, H = 460, PAD = 36;
  const DOMAIN = [-10, 10];
  const TWO_PI = 2 * Math.PI;

  function gaussPdf(p, mu, cov) {
    const dx = p.x - mu.x, dy = p.y - mu.y;
    const det = Math.max(MLU.det2x2(cov), 1e-6);
    const inv = MLU.inverse2x2(cov);
    const m = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy);
    return Math.exp(-0.5 * m) / (TWO_PI * Math.sqrt(det));
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint" style="font-size:11.5px">EM iteration <b id="gm-iter">0</b></span></div>
            <div class="btn-row">
              <button id="gm-step" class="primary">step (E+M)</button>
              <button id="gm-run">run to convergence</button>
              <button id="gm-reset">reset components</button>
              <button id="gm-regen">regenerate data</button>
            </div>
          </div>
          <svg id="gm-svg" viewBox="0 0 ${W} ${H}" style="flex:1;width:100%"></svg>
          <div class="stage-hint">points are colored by their most-likely component · ellipses = 1&sigma; contour of each Gaussian</div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>components k = <span class="val" id="gm-k-val">3</span></h3>
            <input type="range" id="gm-k" min="1" max="6" step="1" value="3" />
          </div>
          <div class="control-card">
            <h3>state</h3>
            <div class="readout" id="gm-readout">–</div>
            <div class="note">Expectation-Maximization: E-step computes soft responsibilities under each Gaussian, M-step re-estimates each component's mean/covariance/weight from those responsibilities — as in <code>mla/gaussian_mixture.py</code>.</div>
          </div>
        </div>
      </div>
    `;

    const svg = d3.select("#gm-svg");
    const x = d3.scaleLinear().domain(DOMAIN).range([PAD, W - PAD]);
    const y = d3.scaleLinear().domain(DOMAIN).range([H - PAD, PAD]);
    MLU.drawAxes(svg.node(), W, H, PAD, [DOMAIN[0], DOMAIN[1], DOMAIN[0], DOMAIN[1]]);

    let points = MLU.makeBlobs({ n: 80, clusters: 3, spread: 1.1 }).map((p) => ({ x: p.x, y: p.y, c: 0 }));
    let comps = [];
    let iter = 0, loglik = 0;

    function k() { return +document.getElementById("gm-k").value; }
    function initComps() {
      const shuffled = MLU.shuffle(points);
      comps = Array.from({ length: k() }, (_, i) => ({
        mu: shuffled.length ? { x: shuffled[i % shuffled.length].x, y: shuffled[i % shuffled.length].y } : { x: MLU.randRange(...DOMAIN), y: MLU.randRange(...DOMAIN) },
        cov: [[4, 0], [0, 4]],
        weight: 1 / k(),
      }));
      iter = 0;
    }
    initComps();

    function eStep() {
      const resp = points.map(() => new Array(comps.length).fill(0));
      let ll = 0;
      points.forEach((p, i) => {
        const ws = comps.map((c) => c.weight * gaussPdf(p, c.mu, c.cov));
        const s = ws.reduce((a, b) => a + b, 0) || 1e-9;
        resp[i] = ws.map((w) => w / s);
        ll += Math.log(s);
        p.c = ws.indexOf(Math.max(...ws));
      });
      return { resp, ll };
    }
    function mStep(resp) {
      const n = points.length;
      comps.forEach((comp, k2) => {
        let Nk = 0;
        for (let i = 0; i < n; i++) Nk += resp[i][k2];
        if (Nk < 1e-6) return;
        let mx = 0, my = 0;
        for (let i = 0; i < n; i++) { mx += resp[i][k2] * points[i].x; my += resp[i][k2] * points[i].y; }
        mx /= Nk; my /= Nk;
        let cxx = 0, cyy = 0, cxy = 0;
        for (let i = 0; i < n; i++) {
          const dx = points[i].x - mx, dy = points[i].y - my;
          cxx += resp[i][k2] * dx * dx; cyy += resp[i][k2] * dy * dy; cxy += resp[i][k2] * dx * dy;
        }
        cxx = cxx / Nk + 0.15; cyy = cyy / Nk + 0.15; cxy /= Nk;
        comp.mu = { x: mx, y: my };
        comp.cov = [[cxx, cxy], [cxy, cyy]];
        comp.weight = Nk / n;
      });
    }

    const ptsG = svg.append("g");
    const ellG = svg.append("g");

    function render() {
      document.getElementById("gm-k-val").textContent = k();
      document.getElementById("gm-iter").textContent = iter;

      const sel = ptsG.selectAll("circle").data(points);
      sel.enter().append("circle").attr("r", 5).attr("stroke", "var(--bg)").attr("stroke-width", 1).style("cursor", "grab")
        .merge(sel).attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y))
        .attr("fill", (d) => MLU.palette[d.c % MLU.palette.length])
        .on("dblclick", (event, d) => { points = points.filter((p) => p !== d); render(); })
        .call(d3.drag().on("drag", function (event, d) {
          d.x = Math.max(DOMAIN[0], Math.min(DOMAIN[1], x.invert(event.x)));
          d.y = Math.max(DOMAIN[0], Math.min(DOMAIN[1], y.invert(event.y)));
          render();
        }));
      sel.exit().remove();

      const esel = ellG.selectAll("ellipse").data(comps);
      esel.enter().append("ellipse").attr("fill", "none").attr("stroke-width", 2)
        .merge(esel)
        .attr("cx", (d) => x(d.mu.x)).attr("cy", (d) => y(d.mu.y))
        .attr("stroke", (d, i) => MLU.palette[i % MLU.palette.length])
        .attr("transform", (d, i) => {
          const eig = MLU.jacobiEigen(d.cov);
          const [v0x, v0y] = eig.vectors[0];
          const angle = Math.atan2(-v0y, v0x) * 180 / Math.PI; // flip y for screen coords
          return `rotate(${angle} ${x(d.mu.x)} ${y(d.mu.y)})`;
        })
        .each(function (d) {
          const eig = MLU.jacobiEigen(d.cov);
          const scaleX = Math.abs(x(1) - x(0));
          d3.select(this)
            .attr("rx", Math.sqrt(Math.max(eig.values[0], 0.01)) * scaleX)
            .attr("ry", Math.sqrt(Math.max(eig.values[1], 0.01)) * scaleX);
        });
      esel.exit().remove();

      document.getElementById("gm-readout").innerHTML =
        `points: <b>${points.length}</b><br>log-likelihood: <b class="num">${points.length ? loglik.toFixed(1) : "–"}</b><br>` +
        comps.map((c, i) => `<span style="color:${MLU.palette[i]}">&pi;${i}=${c.weight.toFixed(2)}</span>`).join(" ");
    }

    svg.on("click", (event) => {
      if (event.target.tagName === "circle") return;
      const [px, py] = d3.pointer(event);
      points.push({ x: x.invert(px), y: y.invert(py), c: 0 });
      render();
    });
    document.getElementById("gm-k").addEventListener("input", () => { initComps(); render(); });
    document.getElementById("gm-step").addEventListener("click", () => {
      const { resp, ll } = eStep(); mStep(resp); loglik = ll; iter++; render();
    });
    document.getElementById("gm-run").addEventListener("click", () => {
      let prev = -Infinity;
      for (let i = 0; i < 100; i++) {
        const { resp, ll } = eStep(); mStep(resp); loglik = ll; iter++;
        if (Math.abs(ll - prev) < 1e-3) break;
        prev = ll;
      }
      render();
    });
    document.getElementById("gm-reset").addEventListener("click", () => { initComps(); render(); });
    document.getElementById("gm-regen").addEventListener("click", () => {
      points = MLU.makeBlobs({ n: 80, clusters: 3, spread: 1.1 }).map((p) => ({ x: p.x, y: p.y, c: 0 }));
      initComps(); render();
    });

    render();
    return () => {};
  }

  MLApp.register({
    id: "gmm",
    name: "Gaussian Mixture Model",
    category: "Unsupervised — Clustering",
    tagline: "EM algorithm, soft clusters",
    description: "Soft clustering via Expectation-Maximization: components are full 2D Gaussians (with orientation), fit by alternating responsibility estimation (E) and weighted re-fitting (M).",
    sourceFile: "mla/gaussian_mixture.py",
    info: {
      type: "Unsupervised — Clustering (soft/probabilistic). Generative mixture model.",
      scenario: "Clusters that overlap or have different sizes/shapes/orientations, or when you need membership probabilities rather than hard labels — a probabilistic generalization of K-Means.",
      inputs: "Unlabeled points and a chosen number of mixture components k.",
      decisionFunction: {
        text: "γ_{ik} = P(component k | x) ∝ π_k · N(x; μ_k, Σ_k)",
        mechanism: "Each component is a full Gaussian (its own mean, covariance, and weight); a point's responsibility for a component is proportional to that component's density at the point, times its mixing weight.",
        plot: { fn: (x) => 0.5 * Math.exp(-((x + 2) ** 2) / 2) + 0.5 * Math.exp(-((x - 2) ** 2) / (2 * 1.5 ** 2)), domain: [-6, 6], color: "var(--accent)", caption: "a 1D illustration: a mixture density is a weighted sum of Gaussian bumps" },
      },
      lossFunction: {
        text: "−Σᵢ log( Σ_k π_k · N(xᵢ; μ_k, Σ_k) )  (negative log-likelihood)",
        mechanism: "Minimized via Expectation-Maximization: the E-step computes responsibilities under current parameters, the M-step re-fits each component's mean/covariance/weight as the responsibility-weighted statistics — each step never decreases the likelihood.",
      },
      output: "Soft responsibilities per point per component, plus each component's mean, covariance, and weight.",
      parameters: [
        { name: "k (components)", effect: "Number of Gaussians. Too few underfits multi-modal data; too many overfits / creates degenerate tiny components." },
        { name: "covariance shape", effect: "Full covariance (used here) lets clusters be elongated/rotated; diagonal or spherical covariance is a more restrictive, faster alternative." },
        { name: "EM iterations", effect: "How long to alternate E/M steps before stopping; convergence is judged by the log-likelihood plateauing." },
      ],
      metrics: ["Log-likelihood", "BIC / AIC (for choosing k)", "Silhouette score on the resulting hard assignments"],
      typicalUses: ["Soft/overlapping cluster segmentation", "Anomaly detection (low-likelihood points)", "Density estimation", "Speaker/topic modeling"],
      workedExample: {
        setup: "Two fixed 1D components: N(0,1) and N(4,1), both weight 0.5. Compute the E-step responsibility for x=1.",
        steps: [
          "pdf₁(1) = (1/√2π)·e^(−(1−0)²/2) = 0.3989×e^−0.5 = 0.3989×0.6065 ≈ 0.2420.",
          "pdf₂(1) = (1/√2π)·e^(−(1−4)²/2) = 0.3989×e^−4.5 = 0.3989×0.01111 ≈ 0.00443.",
          "Unnormalized responsibilities: r₁ = 0.5×0.2420 = 0.1210, r₂ = 0.5×0.00443 = 0.00222.",
          "Normalize: γ₁ = 0.1210/(0.1210+0.00222) ≈ 0.982, γ₂ ≈ 0.018.",
        ],
        result: "x=1 is assigned ~98.2% responsibility to component 1, 1.8% to component 2 — soft, not all-or-nothing like K-Means",
      },
    },
    mount,
  });
})();
