(() => {
  const NU = 6, NI = 6, K = 2;
  const W2 = 460, H2 = 300, PAD = 34;

  function initParams() {
    return {
      w0: 0,
      wu: new Array(NU).fill(0),
      wi: new Array(NI).fill(0),
      vu: Array.from({ length: NU }, () => Array.from({ length: K }, () => MLU.randn() * 0.3)),
      vi: Array.from({ length: NI }, () => Array.from({ length: K }, () => MLU.randn() * 0.3)),
    };
  }
  function predict(p, u, i) {
    let dot = 0; for (let f = 0; f < K; f++) dot += p.vu[u][f] * p.vi[i][f];
    return p.w0 + p.wu[u] + p.wi[i] + dot;
  }
  function trainEpochs(p, ratings, epochs, lr, reg) {
    const entries = [];
    for (let u = 0; u < NU; u++) for (let i = 0; i < NI; i++) if (ratings[u][i] != null) entries.push([u, i, ratings[u][i]]);
    for (let e = 0; e < epochs; e++) {
      const order = MLU.shuffle(entries);
      for (const [u, i, r] of order) {
        const pred = predict(p, u, i);
        const err = pred - r;
        p.w0 -= lr * err;
        p.wu[u] -= lr * (err + reg * p.wu[u]);
        p.wi[i] -= lr * (err + reg * p.wi[i]);
        for (let f = 0; f < K; f++) {
          const vu = p.vu[u][f], vi = p.vi[i][f];
          p.vu[u][f] -= lr * (err * vi + reg * vu);
          p.vi[i][f] -= lr * (err * vu + reg * vi);
        }
      }
    }
    let sse = 0; for (const [u, i, r] of entries) sse += (predict(p, u, i) - r) ** 2;
    return entries.length ? Math.sqrt(sse / entries.length) : null;
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint">click a cell to set the rating shown below, right-click to clear it</span></div>
            <div class="btn-row"><button id="fm-train" class="primary">train 500 epochs</button><button id="fm-reset">reset model</button><button id="fm-randdata">random ratings</button></div>
          </div>
          <div style="padding:8px 4px">
            <div class="stage-hint" style="margin-bottom:6px">user × item rating matrix (bold = observed, faded = FM prediction)</div>
            <div class="hscroll"><table id="fm-table" style="border-collapse:collapse;font-size:12px"></table></div>
          </div>
          <div style="padding:8px 4px;flex:1">
            <div class="stage-hint" style="margin-bottom:6px">learned latent space (each user/item is a 2D vector v — dot products of nearby vectors drive high predicted ratings)</div>
            <svg id="fm-latent" viewBox="0 0 ${W2} ${H2}" style="width:100%;height:${H2}px"></svg>
          </div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>rating to paint</h3>
            <div class="btn-row" id="fm-rating-buttons">${[1, 2, 3, 4, 5].map((r) => `<button data-r="${r}">${r}</button>`).join("")}</div>
          </div>
          <div class="control-card">
            <h3>hyperparameters</h3>
            <div class="field"><label>learning rate <span class="val" id="fm-lr-val">0.05</span></label>
              <input type="range" id="fm-lr" min="1" max="20" step="1" value="10" /></div>
            <div class="field"><label>L2 regularization <span class="val" id="fm-reg-val">0.02</span></label>
              <input type="range" id="fm-reg" min="0" max="20" step="1" value="4" /></div>
          </div>
          <div class="control-card">
            <h3>fit</h3>
            <div class="readout" id="fm-readout">–</div>
            <div class="note">Factorization Machine with only user/item one-hot features reduces to biased matrix factorization: prediction = w<sub>0</sub> + w<sub>u</sub> + w<sub>i</sub> + &lang;v<sub>u</sub>,v<sub>i</sub>&rang;, trained by SGD — the pairwise-interaction term from <code>mla/fm.py</code>, here with a 2D latent space so it's directly plottable.</div>
          </div>
        </div>
      </div>
    `;

    let ratings = Array.from({ length: NU }, () => new Array(NI).fill(null));
    function seedRatings() {
      const density = 0.4;
      for (let u = 0; u < NU; u++) for (let i = 0; i < NI; i++) ratings[u][i] = MLU.rng() < density ? MLU.randInt(5) + 1 : null;
    }
    seedRatings();
    let params = initParams();
    let currentRating = 5;
    let lastRmse = null;

    function lr() { return +document.getElementById("fm-lr").value / 200; }
    function reg() { return +document.getElementById("fm-reg").value / 200; }

    function renderTable() {
      const table = document.getElementById("fm-table");
      table.innerHTML = "";
      const header = document.createElement("tr");
      header.appendChild(document.createElement("td"));
      for (let i = 0; i < NI; i++) { const td = document.createElement("td"); td.textContent = "item " + i; td.style.cssText = "padding:4px 8px;color:var(--text-faint)"; header.appendChild(td); }
      table.appendChild(header);
      for (let u = 0; u < NU; u++) {
        const row = document.createElement("tr");
        const label = document.createElement("td"); label.textContent = "user " + u; label.style.cssText = "padding:4px 8px;color:var(--text-faint)"; row.appendChild(label);
        for (let i = 0; i < NI; i++) {
          const td = document.createElement("td");
          const known = ratings[u][i] != null;
          const val = known ? ratings[u][i] : predict(params, u, i);
          td.textContent = val.toFixed(known ? 0 : 2);
          td.style.cssText = `padding:6px 10px;text-align:center;border:1px solid var(--border);cursor:pointer;background:${known ? "var(--bg-inset)" : "transparent"};color:${known ? "var(--text)" : "var(--text-faint)"};font-weight:${known ? "700" : "400"};`;
          td.addEventListener("click", () => { ratings[u][i] = currentRating; renderTable(); });
          td.addEventListener("contextmenu", (e) => { e.preventDefault(); ratings[u][i] = null; renderTable(); });
          row.appendChild(td);
        }
        table.appendChild(row);
      }
    }

    const svg = d3.select("#fm-latent");
    function renderLatent() {
      const all = [...params.vu, ...params.vi].flat();
      const ext = Math.max(0.5, ...all.map(Math.abs)) * 1.3;
      const x = d3.scaleLinear().domain([-ext, ext]).range([PAD, W2 - PAD]);
      const y = d3.scaleLinear().domain([-ext, ext]).range([H2 - PAD, PAD]);
      svg.selectAll("*").remove();
      svg.append("rect").attr("x", PAD).attr("y", PAD).attr("width", W2 - 2 * PAD).attr("height", H2 - 2 * PAD).attr("fill", "none").attr("stroke", "var(--border-soft)");
      const uData = params.vu.map((v, i) => ({ x: v[0], y: v[1], label: "u" + i }));
      const iData = params.vi.map((v, i) => ({ x: v[0], y: v[1], label: "i" + i }));
      const g1 = svg.append("g");
      g1.selectAll("circle").data(uData).enter().append("circle").attr("r", 6).attr("fill", MLU.palette[0])
        .attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y));
      g1.selectAll("text").data(uData).enter().append("text").attr("x", (d) => x(d.x) + 8).attr("y", (d) => y(d.y) + 3)
        .text((d) => d.label).attr("font-size", 10).attr("fill", "var(--text-dim)");
      const g2 = svg.append("g");
      g2.selectAll("rect.pt").data(iData).enter().append("rect").attr("class", "pt").attr("width", 9).attr("height", 9)
        .attr("fill", MLU.palette[1]).attr("x", (d) => x(d.x) - 4.5).attr("y", (d) => y(d.y) - 4.5);
      g2.selectAll("text").data(iData).enter().append("text").attr("x", (d) => x(d.x) + 8).attr("y", (d) => y(d.y) + 3)
        .text((d) => d.label).attr("font-size", 10).attr("fill", "var(--text-dim)");
    }

    function renderAll() {
      document.getElementById("fm-lr-val").textContent = lr().toFixed(3);
      document.getElementById("fm-reg-val").textContent = reg().toFixed(3);
      renderTable();
      renderLatent();
      document.getElementById("fm-readout").innerHTML = lastRmse != null
        ? `train RMSE: <b class="num">${lastRmse.toFixed(3)}</b>`
        : "click train to fit";
    }

    document.getElementById("fm-rating-buttons").addEventListener("click", (e) => {
      const btn = e.target.closest("button"); if (!btn) return;
      currentRating = +btn.dataset.r;
      document.querySelectorAll("#fm-rating-buttons button").forEach((b) => b.classList.toggle("primary", +b.dataset.r === currentRating));
    });
    document.querySelector('#fm-rating-buttons button[data-r="5"]').classList.add("primary");
    document.getElementById("fm-train").addEventListener("click", () => { lastRmse = trainEpochs(params, ratings, 500, lr(), reg()); renderAll(); });
    document.getElementById("fm-reset").addEventListener("click", () => { params = initParams(); lastRmse = null; renderAll(); });
    document.getElementById("fm-randdata").addEventListener("click", () => { seedRatings(); renderAll(); });
    ["fm-lr", "fm-reg"].forEach((id) => document.getElementById(id).addEventListener("input", renderAll));

    renderAll();
    return () => {};
  }

  MLApp.register({
    id: "factorization-machines",
    name: "Factorization Machines",
    category: "Supervised — Regression",
    tagline: "latent-vector pairwise interactions",
    description: "Predicts ratings in a sparse user×item matrix by learning a low-rank latent vector per user and item, so unseen pairs get a prediction from the dot product of their vectors. Edit the matrix and retrain to see the latent space reorganize.",
    sourceFile: "mla/fm.py",
    info: {
      type: "Supervised — Regression/ranking. Factorized second-order feature-interaction model.",
      scenario: "Sparse, high-cardinality categorical data (user IDs × item IDs, ad click prediction with many categorical fields) where you want pairwise feature interactions without an explosion of parameters — a generalization of matrix factorization.",
      inputs: "A (typically sparse, one-hot-encoded) feature vector x — here a user indicator and an item indicator — and a target rating.",
      decisionFunction: {
        text: "ŷ(x) = w₀ + Σᵢ wᵢxᵢ + Σ_{i<j} ⟨vᵢ,vⱼ⟩xᵢxⱼ",
        mechanism: "Instead of a separate weight per feature pair (O(n²) parameters, most never observed), each feature gets a small latent vector v; the interaction weight for any pair is just the dot product of their vectors — so interactions generalize even to feature pairs never co-observed in training.",
      },
      lossFunction: {
        text: "L = Σᵢ (yᵢ − ŷᵢ)² + regularization on w and v",
        mechanism: "Minimized by stochastic gradient descent. For the one-hot user/item case here it reduces to exactly the classic biased matrix-factorization update, adjusting w₀, the user/item biases, and the latent vectors after each observed rating.",
      },
      output: "A predicted continuous rating/score for any (user, item) pair — including pairs never observed during training.",
      parameters: [
        { name: "latent dimension k", effect: "How much interaction structure can be captured (fixed at 2 here so the latent space is directly plottable)." },
        { name: "learning rate", effect: "SGD step size for updating biases and latent vectors." },
        { name: "L2 regularization", effect: "Shrinks weights/vectors toward 0, important on sparse data where many features have few observations." },
      ],
      metrics: ["RMSE / MAE (rating prediction)", "AUC / log-loss (binary click-through variants)", "Precision@k (ranking/recommendation)"],
      typicalUses: ["Recommender systems", "Click-through-rate prediction in ad systems", "Tabular problems with many sparse categorical features and meaningful interactions"],
      workedExample: {
        setup: "w0=3, w_u=0.2 (user), w_i=−0.1 (item), v_u=[0.5, 0.2], v_i=[0.4, 0.1]. Predict the rating.",
        steps: [
          "Dot product ⟨v_u, v_i⟩ = 0.5×0.4 + 0.2×0.1 = 0.20 + 0.02 = 0.22.",
          "ŷ = w0 + w_u + w_i + ⟨v_u,v_i⟩ = 3 + 0.2 + (−0.1) + 0.22.",
        ],
        result: "ŷ = 3.32 — the bias terms set a baseline (~3.1) and the latent-vector interaction adds +0.22 for this specific user/item pair",
      },
    },
    mount,
  });
})();
