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
            <div class="stage-hint" style="margin-bottom:6px">learned latent space (each user/item is a 2D vector v - dot products of nearby vectors drive high predicted ratings)</div>
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
            <div class="note">Factorization Machine with only user/item one-hot features reduces to biased matrix factorization: prediction = w<sub>0</sub> + w<sub>u</sub> + w<sub>i</sub> + &lang;v<sub>u</sub>,v<sub>i</sub>&rang;, trained by SGD - the pairwise-interaction term, here with a 2D latent space so it's directly plottable.</div>
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
    category: "Supervised - Regression",
    tagline: "latent-vector pairwise interactions",
    description: "Predicts ratings in a sparse user×item matrix by learning a low-rank latent vector per user and item, so unseen pairs get a prediction from the dot product of their vectors. Edit the matrix and retrain to see the latent space reorganize.",
    info: {
      type: "Supervised - Regression/ranking. Factorized second-order feature-interaction model.",
      scenario: "Sparse, high-cardinality categorical data (user IDs × item IDs, ad click prediction with many categorical fields) where you want pairwise feature interactions without an explosion of parameters - a generalization of matrix factorization.",
      inputs: "A (typically sparse, one-hot-encoded) feature vector x - here a user indicator and an item indicator - and a target rating.",
      intuition: {
        definition: "A linear model cannot express 'this user likes this genre'. Adding a weight per feature pair would need O(n²) parameters, almost all of which are never observed. Factorization machines instead give every feature a small <b>latent vector</b>, and define the interaction weight for any pair as the dot product of their vectors.",
        steps: [
          "Start from a linear model with a global bias and per-feature weights.",
          "Add pairwise interaction terms.",
          "Factorize each pair's weight as ⟨vᵢ, vⱼ⟩ instead of learning it directly.",
          "Now every observation of feature i improves its vector for all pairings.",
        ],
        applications: [
          "Recommender systems with user and item identifiers",
          "Click-through-rate prediction across many categorical ad fields",
          "Cold-start ranking using side features such as age or category",
          "Any sparse categorical problem where interactions matter",
          "A generalization that subsumes matrix factorization, SVD++, and more",
        ],
      },
      math: [
        { title: "Model equation", formula: "ŷ(x) = w₀ + Σᵢ wᵢxᵢ + Σ_{i<j} ⟨vᵢ, vⱼ⟩·xᵢxⱼ", note: "The first two terms are ordinary linear regression. The third adds every pairwise interaction, but with factorized rather than free weights." },
        { title: "Factorized interaction", formula: "ŵᵢⱼ = ⟨vᵢ, vⱼ⟩ = Σ_{f=1}^{k} vᵢ,f · vⱼ,f", note: "Parameters drop from O(n²) to O(nk). More importantly, the weights become dependent, so information shares across pairs." },
        { title: "Linear-time computation", formula: "Σ_{i<j}⟨vᵢ,vⱼ⟩xᵢxⱼ = ½ Σ_f [ (Σᵢ vᵢ,f xᵢ)² − Σᵢ vᵢ,f² xᵢ² ]", note: "The key identity. It turns a naive O(n²k) sum into O(nk), and with sparse x only the non-zero entries contribute. This is what makes FMs practical." },
        { title: "Gradients", formula: "∂ŷ/∂w₀ = 1,  ∂ŷ/∂wᵢ = xᵢ,  ∂ŷ/∂vᵢ,f = xᵢ·(Σⱼ vⱼ,f xⱼ) − vᵢ,f·xᵢ²", note: "The inner sum is shared across all i, so a full gradient step costs the same order as a prediction." },
        { title: "Reduction to matrix factorization", formula: "with only user and item one-hots: ŷ = w₀ + w_u + w_i + ⟨v_u, v_i⟩", note: "Exactly biased matrix factorization. FMs are strictly more general, because you can append any other feature to the same vector." },
        { title: "Higher-order FMs", formula: "add Σ_{i<j<l} ⟨vᵢ,vⱼ,v_l⟩ terms", note: "Degree-3 and above are possible but rarely worth the cost. Second order captures most of the useful signal." },
      ],
      pipeline: [
        { label: "Sparse x", note: "one-hot fields" },
        { label: "Linear part", note: "w₀ + Σwᵢxᵢ" },
        { label: "Latent vectors", note: "vᵢ per feature" },
        { label: "Pairwise ⟨vᵢ,vⱼ⟩", note: "O(nk) trick" },
        { label: "Prediction ŷ", note: "score or rating", accent: "green" },
      ],
      decisionFunction: {
        text: "ŷ(x) = w₀ + Σᵢ wᵢxᵢ + Σ_{i<j} ⟨vᵢ,vⱼ⟩xᵢxⱼ",
        mechanism: "Instead of a separate weight per feature pair (O(n²) parameters, most never observed), each feature gets a small latent vector v; the interaction weight for any pair is just the dot product of their vectors - so interactions generalize even to feature pairs never co-observed in training.",
      },
      lossFunction: {
        text: "L = Σᵢ (yᵢ − ŷᵢ)² + regularization on w and v",
        mechanism: "Minimized by stochastic gradient descent. For the one-hot user/item case here it reduces to exactly the classic biased matrix-factorization update, adjusting w₀, the user/item biases, and the latent vectors after each observed rating.",
      },
      optimization: [
        { title: "SGD", formula: "θ := θ − η(∂L/∂θ + λθ)", note: "The usual choice. Cheap per step, handles streaming data, and works naturally with sparse updates that touch only the active features." },
        { title: "ALS / coordinate descent", formula: "solve one parameter at a time in closed form", note: "No learning rate to tune, and it converges reliably. Requires the full dataset in memory." },
        { title: "MCMC / Bayesian FM", formula: "sample parameters, integrate out regularization", note: "Removes hyperparameter tuning almost entirely by placing priors on the regularization terms. Often the best out-of-the-box accuracy." },
        { title: "Sparse update cost", formula: "O(k · nnz(x)) per example", note: "Only non-zero features are touched. With one-hot fields that is a handful of vectors per row regardless of vocabulary size." },
        { title: "Initialisation", formula: "v ~ N(0, σ²) with small σ, w = 0", note: "Latent vectors must be random. Initialising them to zero makes every dot product zero and every gradient zero, so they never move." },
      ],
      output: "A predicted continuous rating/score for any (user, item) pair - including pairs never observed during training.",
      assumptions: [
        { name: "Interactions are low-rank", why: "The whole method assumes the pairwise weight matrix can be approximated by a rank-k factorization.", check: "If accuracy plateaus well below a full interaction model, the structure may not be low-rank." },
        { name: "Data is sparse and categorical", why: "FMs shine when most pairs are unobserved. On dense numeric data a tree ensemble usually wins.", check: "Count the non-zero fraction. Very dense data is the wrong regime." },
        { name: "Second-order suffices", why: "Standard FMs model only pairwise interactions.", check: "If three-way effects matter, use higher-order FMs, DeepFM, or gradient boosting." },
        { name: "Features are properly encoded", why: "Interactions are between encoded columns, so a badly encoded categorical destroys the structure.", check: "One-hot encode categoricals; do not label-encode them into a single numeric column." },
        { name: "Enough observations per feature", why: "A latent vector needs data to be estimated.", check: "Features appearing once or twice need strong regularization or should be bucketed." },
      ],
      regularization: [
        { name: "L2 on weights", formula: "+ λ_w Σ wᵢ²", note: "Standard shrinkage on the linear part." },
        { name: "L2 on latent vectors", formula: "+ λ_v Σ ‖vᵢ‖²", note: "Critical on sparse data. Rare features would otherwise get large, unreliable vectors fitted to a handful of rows." },
        { name: "Per-group regularization", formula: "separate λ per feature field", note: "User IDs, item IDs, and side features have very different observation counts and deserve different shrinkage." },
        { name: "Latent dimension k", formula: "k controls rank", note: "k is itself a capacity control. Small k is a strong structural regularizer." },
        { name: "Early stopping", formula: "halt on validation RMSE", note: "SGD on sparse data overfits rare features quickly." },
        { name: "Bayesian priors", formula: "place hyperpriors on λ", note: "Lets the model infer regularization strength instead of requiring a grid search." },
      ],
      hyperparameters: [
        { name: "latent dimension k", range: "2 - 200", increasing: "Richer interaction structure, more parameters, more overfitting on sparse data.", strategy: "Start at 8 to 16. Rating prediction rarely needs more than 50. Larger k demands stronger L2." },
        { name: "learning rate η", range: "1e-4 - 0.1", increasing: "Faster convergence, then divergence.", strategy: "0.01 is a reasonable start with SGD. Use a decay schedule or switch to Adagrad, which suits sparse features well." },
        { name: "λ_v (latent L2)", range: "1e-5 - 1", increasing: "Vectors shrink toward zero, interactions weaken toward a purely linear model.", strategy: "The most important regularization knob. Log-scale search; rare features need more." },
        { name: "λ_w (linear L2)", range: "1e-5 - 1", increasing: "Bias terms shrink.", strategy: "Usually smaller than λ_v, since biases are estimated from more data." },
        { name: "epochs", range: "10 - 200", increasing: "Better fit, then overfitting of rare features.", strategy: "Early-stop on a validation split." },
        { name: "init stdev", range: "0.001 - 0.1", increasing: "Larger initial interactions, faster early movement, less stability.", strategy: "0.01 is standard. Must be non-zero." },
        { name: "optimizer", range: "SGD / ALS / MCMC", increasing: "Not applicable", strategy: "SGD for large or streaming data, ALS for reliability, MCMC when you want to skip tuning." },
      ],
      metrics: ["RMSE / MAE (rating prediction)", "AUC / log-loss (binary click-through variants)", "Precision@k / NDCG (ranking/recommendation)", "Coverage and cold-start performance"],
      typicalUses: ["Recommender systems", "Click-through-rate prediction in ad systems", "Tabular problems with many sparse categorical features and meaningful interactions"],
      diagnostics: [
        "Compare against a linear model with no interaction term. If FM does not beat it, the interactions are not carrying signal or λ_v is too high.",
        "Sweep k and watch validation error. A curve that keeps improving means you are under-parameterised; one that worsens means overfitting.",
        "Plot the latent vectors when k is 2 or after PCA. Similar users or items should cluster together.",
        "Check performance separately on rare versus frequent features. Rare-feature degradation points to insufficient regularization.",
        "Evaluate cold-start cases explicitly, since generalising to unseen pairs is the main reason to choose an FM.",
      ],
      advantages: [
        "Models pairwise interactions with O(nk) parameters instead of O(n²).",
        "Generalises to feature pairs never seen together in training, which plain interaction weights cannot do.",
        "Prediction and training are linear time thanks to the reformulation identity.",
        "Works directly on very sparse one-hot data where most models struggle.",
        "Strictly generalises matrix factorization, so side features slot in without changing the model.",
        "Handles cold start far better than pure collaborative filtering.",
      ],
      limitations: [
        { name: "Only second-order interactions", note: "three-way effects are not captured", fix: "higher-order FMs, DeepFM, or gradient boosting." },
        { name: "Shared latent space across all pairings", note: "one vector per feature must serve every interaction it participates in", fix: "field-aware FMs, which give each feature a separate vector per interacting field." },
        { name: "Assumes low-rank structure", note: "genuinely idiosyncratic interactions are not representable", fix: "add explicit crosses, or use a tree model." },
        { name: "Sensitive to regularization", note: "sparse features overfit readily", fix: "per-field λ, or a Bayesian FM." },
        { name: "Weaker on dense numeric data", note: "boosting typically wins outside the sparse categorical regime", fix: "benchmark both." },
        { name: "Latent vectors are not interpretable", note: "dimensions have no inherent meaning", fix: "inspect nearest neighbours in latent space instead." },
      ],
      alternatives: [
        { name: "Field-aware FM (FFM)", when: "Click-through prediction with many distinct fields. More parameters, usually more accurate." },
        { name: "DeepFM / xDeepFM", when: "You want FM's low-order interactions plus a neural network's higher-order ones." },
        { name: "Gradient boosting", when: "Dense tabular features, or interactions that are not low-rank." },
        { name: "Matrix factorization", when: "Only user and item identifiers, no side features. Simpler and equivalent in that case." },
        { name: "Two-tower neural retrieval", when: "Large-scale recommendation with rich content features." },
      ],
      pitfalls: [
        { problem: "Latent vectors never move from zero", solution: "They were initialised to zero, so every dot product and gradient is zero. Initialise randomly." },
        { problem: "Severe overfitting on rare features", solution: "Raise λ_v, lower k, or bucket rare categories into an 'other' bin." },
        { problem: "No improvement over linear regression", solution: "Either interactions carry no signal, or λ_v has shrunk them away. Try a lower λ_v and larger k first." },
        { problem: "Training is unexpectedly slow", solution: "You may be computing interactions naively. Use the O(nk) reformulation." },
        { problem: "Label-encoding categoricals", solution: "That imposes a false ordering and destroys the interaction structure. One-hot encode instead." },
        { problem: "Poor cold-start performance", solution: "Add side features such as category or demographics so a new ID still has informative vectors." },
      ],
      quickRef: [
        { name: "Model", formula: "ŷ = w₀ + Σwᵢxᵢ + Σ_{i<j}⟨vᵢ,vⱼ⟩xᵢxⱼ" },
        { name: "Interaction weight", formula: "ŵᵢⱼ = ⟨vᵢ, vⱼ⟩" },
        { name: "Linear-time form", formula: "½Σ_f[(Σᵢvᵢ,f xᵢ)² − Σᵢvᵢ,f²xᵢ²]" },
        { name: "Parameters", formula: "O(nk) instead of O(n²)" },
        { name: "Prediction cost", formula: "O(k · nnz(x))" },
        { name: "Latent gradient", formula: "xᵢ(Σⱼvⱼ,f xⱼ) − vᵢ,f xᵢ²" },
        { name: "MF special case", formula: "w₀ + w_u + w_i + ⟨v_u,v_i⟩" },
        { name: "Init", formula: "v ~ N(0, σ²), never zero" },
      ],
      code: `# pip install fastFM  (or use xlearn / LightFM for larger scale)
from fastFM import als
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp

# One-hot encode every categorical field. Do NOT label-encode:
# that invents an ordering and destroys the interaction structure.
enc = OneHotEncoder(handle_unknown="ignore")
X_tr = enc.fit_transform(df_train[["user_id", "item_id", "genre", "hour"]])
X_te = enc.transform(df_test[["user_id", "item_id", "genre", "hour"]])

fm = als.FMRegression(
    n_iter=200,
    rank=16,          # latent dimension k
    l2_reg_w=0.1,     # linear terms
    l2_reg_V=0.5,     # latent vectors: the key regularizer on sparse data
    init_stdev=0.01,  # must be non-zero or the vectors never move
)
fm.fit(X_tr, y_train)

pred = fm.predict(X_te)
print("RMSE:", mean_squared_error(y_test, pred, squared=False))

# Always compare against a purely linear baseline. If FM does not beat it,
# the interactions carry no signal or l2_reg_V is too strong.`,
      whyChain: [
        { q: "Why not just learn a separate weight for each feature pair?", a: "Two problems. There are O(n²) of them, which is unmanageable when n is millions of one-hot columns. And each weight would be trained only on rows where both features are non-zero, which for sparse data is usually zero rows, so most weights would never be estimated at all." },
        { q: "How does factorizing fix the second problem?", a: "It makes the weights dependent. Every observation involving feature i updates vᵢ, which changes its dot product with every other feature's vector. So the model can predict an interaction for a pair it has never seen together, by combining what it learned about each one separately." },
        { q: "Why is the O(nk) reformulation the crucial detail?", a: "Without it, evaluating the pairwise sum costs O(n²k), which defeats the purpose. The identity rewrites the sum of products as the difference between the square of a sum and a sum of squares, both of which are single passes. That is what makes FMs usable at web scale." },
        { q: "How exactly is this a generalization of matrix factorization?", a: "Give the model only two one-hot fields, user and item. The linear terms become user and item biases, and the single interaction term becomes ⟨v_u, v_i⟩. That is biased matrix factorization exactly. FMs let you append arbitrary extra fields to the same vector, which MF cannot." },
        { q: "Why must latent vectors be initialised randomly?", a: "The interaction term is a dot product of vectors. If all vectors are zero, every dot product is zero and the gradient with respect to each vector is also zero, so they can never move. Random initialisation breaks that degenerate symmetry." },
        { q: "What does a field-aware FM add?", a: "In a standard FM each feature has one vector serving every interaction. FFM gives each feature a separate vector per field it interacts with, so a user's vector when pairing with items differs from its vector when pairing with time-of-day. More parameters, but consistently better on click-through data." },
        { q: "Why do FMs handle cold start better than collaborative filtering?", a: "Pure CF only knows user and item identifiers, so a new user has no history and no prediction. An FM can include side features such as age or category in the same vector, so a new user still has informative components and receives a sensible score immediately." },
        { q: "When would you use boosting instead?", a: "When features are dense and numeric, or when interactions are idiosyncratic rather than low-rank. FMs assume the interaction matrix factorizes; a tree ensemble makes no such assumption and can carve arbitrary interactions, but needs enough data per region to do it." },
      ],
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
        result: "ŷ = 3.32 - the bias terms set a baseline (~3.1) and the latent-vector interaction adds +0.22 for this specific user/item pair",
      },
    },
    mount,
  });
})();
