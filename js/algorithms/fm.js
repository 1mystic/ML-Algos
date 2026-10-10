(() => {
  const { ui, col, fmt } = Lab;
  const NU = 8, NI = 8;
  const USERS = ["Ana", "Ben", "Cai", "Dev", "Eli", "Fay", "Gus", "Hui"];
  const ITEMS = ["Alien", "Brave", "Coco", "Dune", "Elf", "Frozen", "Gravity", "Heat"];

  function makeData(kind) {
    MLU.seedRng(kind === "two" ? 5 : kind === "three" ? 9 : 2);
    const R = Array.from({ length: NU }, () => new Array(NI).fill(null));
    const split = Array.from({ length: NU }, () => new Array(NI).fill(null));
    let U, V;
    if (kind === "two") { U = USERS.map((_, i) => (i % 2 ? [1, -0.2] : [-1, 0.2]).map((v) => v + MLU.randn() * 0.2)); V = ITEMS.map((_, j) => (j < 4 ? [1, 0] : [-1, 0]).map((v) => v + MLU.randn() * 0.25)); }
    else if (kind === "three") { const g = [[1, 0], [-0.5, 0.87], [-0.5, -0.87]]; U = USERS.map((_, i) => g[i % 3].map((v) => v + MLU.randn() * 0.2)); V = ITEMS.map((_, j) => g[j % 3].map((v) => v + MLU.randn() * 0.2)); }
    else { U = USERS.map(() => [MLU.randn() * 0.1, MLU.randn() * 0.1]); V = ITEMS.map(() => [MLU.randn() * 0.1, MLU.randn() * 0.1]); }
    for (let u = 0; u < NU; u++) for (let i = 0; i < NI; i++) {
      const raw = 3 + 1.6 * (U[u][0] * V[i][0] + U[u][1] * V[i][1]) + (kind === "random" ? MLU.randn() * 1.3 : MLU.randn() * 0.35);
      const r = Math.max(1, Math.min(5, Math.round(raw)));
      const z = MLU.rng();
      if (z < 0.6) { R[u][i] = r; split[u][i] = "train"; } else if (z < 0.78) { R[u][i] = r; split[u][i] = "test"; }
    }
    return { R, split };
  }

  function train(D, P) {
    MLU.seedRng(17);
    const k = P.k;
    const p = { w0: 0, wu: new Array(NU).fill(0), wi: new Array(NI).fill(0), vu: Array.from({ length: NU }, () => Array.from({ length: k }, () => MLU.randn() * 0.1)), vi: Array.from({ length: NI }, () => Array.from({ length: k }, () => MLU.randn() * 0.1)) };
    const entries = [], tests = [];
    for (let u = 0; u < NU; u++) for (let i = 0; i < NI; i++) if (D.R[u][i] != null) (D.split[u][i] === "train" ? entries : tests).push([u, i, D.R[u][i]]);
    const pred = (u, i) => p.w0 + p.wu[u] + p.wi[i] + p.vu[u].reduce((s, v, f) => s + v * p.vi[i][f], 0);
    const rmse = (set) => (set.length ? Math.sqrt(MLU.mean(set.map(([u, i, r]) => (pred(u, i) - r) ** 2))) : null);
    p.w0 = entries.length ? MLU.mean(entries.map((e) => e[2])) : 3;
    const snap = () => ({ w0: p.w0, wu: p.wu.slice(), wi: p.wi.slice(), vu: p.vu.map((v) => v.slice()), vi: p.vi.map((v) => v.slice()) });
    const frames = [{ phase: "init", major: true, arch: ["x", "lin", "pair"], title: `Start: global mean ${p.w0.toFixed(2)}, tiny random latent vectors`, text: "Each user and each item gets a bias and a k-dimensional latent vector. The model predicts a rating as the global mean plus both biases plus the dot product of the two vectors: users and items that point the same way get high ratings.", tex: "\\hat y(x)=w_0+\\sum_i w_ix_i+\\sum_{i<j}\\langle v_i,v_j\\rangle x_ix_j\\;\\overset{\\text{one-hot}}{=}\\;w_0+w_u+w_i+\\langle v_u,v_i\\rangle", st: snap(), m: { train: rmse(entries), test: rmse(tests) } }];
    const keep = new Set(Lab.checkpoints(P.epochs, 60));
    for (let e = 1; e <= P.epochs; e++) {
      for (const [u, i, r] of MLU.shuffle(entries)) {
        const err = pred(u, i) - r;
        p.w0 -= P.lr * err;
        p.wu[u] -= P.lr * (err + P.reg * p.wu[u]);
        p.wi[i] -= P.lr * (err + P.reg * p.wi[i]);
        for (let f = 0; f < k; f++) { const a = p.vu[u][f], b = p.vi[i][f]; p.vu[u][f] -= P.lr * (err * b + P.reg * a); p.vi[i][f] -= P.lr * (err * a + P.reg * b); }
      }
      if (keep.has(e)) frames.push({ phase: "sgd", major: e === 1, arch: ["pair", "loss", "sgd"], title: `Epoch ${e}`, text: `Train RMSE <b>${rmse(entries).toFixed(3)}</b>, held-out RMSE <b>${(rmse(tests) ?? 0).toFixed(3)}</b>. One SGD pass over the observed ratings. In the latent map, users drift toward the items they rate highly and away from those they dislike.`, tex: "v_u\\leftarrow v_u-\\eta\\big[(\\hat y-y)\\,v_i+\\lambda v_u\\big],\\qquad v_i\\leftarrow v_i-\\eta\\big[(\\hat y-y)\\,v_u+\\lambda v_i\\big]", st: snap(), m: { train: rmse(entries), test: rmse(tests) } });
    }
    frames.push({ ...frames[frames.length - 1], phase: "done", major: true, arch: ["out"], title: "Fill in the blanks", text: "Every empty cell now has a predicted rating (faded numbers): that is the recommendation. The general FM handles any extra features (time, device, item genre) the same way, with pairwise interactions computed in O(k·n) using the identity below.", tex: "\\sum_{i<j}\\langle v_i,v_j\\rangle x_ix_j=\\tfrac12\\sum_{f=1}^{k}\\Big[\\Big(\\sum_i v_{if}x_i\\Big)^2-\\sum_i v_{if}^2x_i^2\\Big]" });
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "fm", fps: 6, plane: false,
      stageHTML: `<div class="html-stage" style="display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:14px"><div><div class="sub-cap">ratings · bold = train · outlined = held-out test · faded = prediction · click to edit</div><div data-r="mat" class="hscroll"></div></div><div><div class="sub-cap">latent space (first two dimensions): ● users ■ items</div><svg data-r="lat" viewBox="0 0 340 300" class="arch-svg"></svg></div></div>`,
      legend: ui.legend([["var(--series-1)", "high rating"], ["var(--series-2)", "low rating"]]),
      tools: ui.btn("retrain", "retrain", true),
      side: [
        { title: "data", html: ui.select("ds", "ratings", [["two", "two taste groups"], ["three", "three genres"], ["random", "random (no structure)"]], "two") + ui.range("paint", "rating to paint", 1, 5, 1, 5) },
        { title: "model", html: ui.range("k", "latent dimensions k", 1, 4, 1, 2) + ui.range("lr", "learning rate η", 0.005, 0.1, 0.005, 0.03) + ui.range("reg", "regularisation λ", 0, 0.3, 0.01, 0.05) + ui.range("ep", "epochs", 20, 600, 20, 200) },
      ],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "x", label: "x = [user | item]", sub: "one-hot features", x: 80, y: 40, kind: "data" },
          { id: "lin", label: "w₀ + Σ wᵢxᵢ", sub: "biases", x: 80, y: 112, kind: "op" },
          { id: "pair", label: "Σ ⟨vᵢ,vⱼ⟩xᵢxⱼ", sub: "pairwise, O(kn)", x: 230, y: 40, kind: "op" },
          { id: "out", label: "ŷ", sub: "rating", x: 230, y: 112, w: 56, kind: "out" },
          { id: "loss", label: "(ŷ − y)² + λ‖θ‖²", sub: "loss", x: 230, y: 170, kind: "loss" },
          { id: "sgd", label: "SGD", sub: "per rating", x: 80, y: 170, w: 64, kind: "param" },
        ],
        edges: [{ from: "x", to: "lin" }, { from: "x", to: "pair" }, { from: "lin", to: "out" }, { from: "pair", to: "out" }, { from: "out", to: "loss" }, { from: "loss", to: "sgd" }, { from: "sgd", to: "lin", dash: true }],
      },
      phases: { init: { label: "init", color: 3 }, sgd: { label: "SGD epoch", color: 0 }, done: { label: "recommend", color: 2 } },
      charts: [{ id: "rmse", title: "RMSE: train vs held-out", series: [{ key: "train", label: "train", color: "var(--series-1)" }, { key: "test", label: "held-out", color: "var(--series-2)", dash: "4 3" }] }],
      onFrame: render,
    });
    let D = makeData("two");
    const P = () => ({ k: lab.val("k"), lr: lab.val("lr"), reg: lab.val("reg"), epochs: lab.val("ep") });
    function retrain({ at = "start" } = {}) { lab.load(train(D, P()), { at, play: at === "start" }); }
    const latSvg = d3.select(lab.R("lat"));
    function render(f) {
      const p = f.st, k = p.vu[0].length;
      const pred = (u, i) => p.w0 + p.wu[u] + p.wi[i] + p.vu[u].reduce((s, v, q) => s + v * p.vi[i][q], 0);
      const color = (r) => { const t = (Math.max(1, Math.min(5, r)) - 3) / 2; return t >= 0 ? `color-mix(in srgb, var(--series-1) ${Math.round(t * 70 + 8)}%, var(--bg-raised))` : `color-mix(in srgb, var(--series-2) ${Math.round(-t * 70 + 8)}%, var(--bg-raised))`; };
      lab.R("mat").innerHTML = `<table class="mini-table" style="min-width:330px;text-align:center"><tr><th></th>${ITEMS.map((t) => `<th style="text-align:center">${t.slice(0, 5)}</th>`).join("")}</tr>${USERS.map((u, ui2) => `<tr><td style="color:var(--text-dim)">${u}</td>${ITEMS.map((_, i) => {
        const r = D.R[ui2][i], sp = D.split[ui2][i], pr = pred(ui2, i);
        const shown = sp === "train" ? r : pr;
        return `<td data-c="${ui2},${i}" style="cursor:pointer;background:${color(shown)};${sp === "test" ? "outline:2px solid var(--series-4);outline-offset:-3px;" : ""}font-weight:${sp === "train" ? 700 : 400};color:${sp === "train" ? "var(--text)" : "var(--text-dim)"}" title="${sp === "train" ? "observed " + r : sp === "test" ? `held-out ${r}, predicted ${pr.toFixed(2)}` : "predicted " + pr.toFixed(2)}">${sp === "train" ? r : pr.toFixed(1)}</td>`;
      }).join("")}</tr>`).join("")}</table>`;
      lab.R("mat").querySelectorAll("td[data-c]").forEach((td) => td.onclick = () => {
        const [u, i] = td.dataset.c.split(",").map(Number);
        if (D.split[u][i] === "train" && D.R[u][i] === lab.val("paint")) { D.R[u][i] = null; D.split[u][i] = null; }
        else { D.R[u][i] = lab.val("paint"); D.split[u][i] = "train"; }
        retrain({ at: "end" });
      });
      // latent map
      latSvg.selectAll("*").remove();
      const all = p.vu.concat(p.vi).map((v) => [v[0], k > 1 ? v[1] : 0]);
      const ext = Math.max(0.3, ...all.flat().map(Math.abs)) * 1.15;
      const sx = d3.scaleLinear().domain([-ext, ext]).range([20, 320]), sy = d3.scaleLinear().domain([-ext, ext]).range([280, 20]);
      latSvg.append("line").attr("x1", 20).attr("x2", 320).attr("y1", sy(0)).attr("y2", sy(0)).attr("stroke", "var(--border)").attr("stroke-dasharray", "3 3");
      latSvg.append("line").attr("y1", 20).attr("y2", 280).attr("x1", sx(0)).attr("x2", sx(0)).attr("stroke", "var(--border)").attr("stroke-dasharray", "3 3");
      p.vi.forEach((v, i) => { const x = sx(v[0]), y = sy(k > 1 ? v[1] : 0); latSvg.append("rect").attr("x", x - 5).attr("y", y - 5).attr("width", 10).attr("height", 10).attr("fill", col(4)).attr("rx", 2); latSvg.append("text").attr("x", x + 8).attr("y", y + 4).style("font-size", "9px").attr("fill", "var(--text-dim)").text(ITEMS[i]); });
      p.vu.forEach((v, u) => { const x = sx(v[0]), y = sy(k > 1 ? v[1] : 0); latSvg.append("circle").attr("cx", x).attr("cy", y).attr("r", 5.5).attr("fill", col(6)); latSvg.append("text").attr("x", x + 8).attr("y", y + 4).style("font-size", "9px").attr("fill", "var(--text)").text(USERS[u]); });
      const m = f.m || {};
      lab.readout(`<div class="kv"><span>users × items</span><b>${NU} × ${NI}</b><span>parameters</span><b>${1 + NU + NI + (NU + NI) * k}</b><span>train RMSE</span><b>${fmt(m.train)}</b><span>held-out RMSE</span><b>${fmt(m.test)}</b></div>`);
    }
    lab.watch(["ds"], () => { D = makeData(lab.val("ds")); retrain(); });
    lab.watch(["k", "lr", "reg", "ep"], () => retrain());
    lab.q('[data-b="retrain"]').onclick = () => retrain();
    retrain();
    return lab.destroy;
  }

  MLApp.register({
    id: "factorization-machines",
    name: "Factorization Machines",
    category: "Supervised - Regression",
    tagline: "latent-factor recommender, SGD",
    description: "Pairwise feature interactions through learned latent vectors, shown as a recommender: watch user and item vectors move in latent space as SGD fits observed ratings, and the empty cells fill with predictions while held-out error tracks generalisation.",
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
