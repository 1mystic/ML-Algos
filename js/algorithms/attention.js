(() => {
  const { ui, col, fmt, esc } = Lab;
  const DM = 6;
  function hash(s) { let h = 2166136261; for (const ch of s) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
  function embed(w) { MLU.seedRng(hash(w.toLowerCase()) % 100000 + 1); return Array.from({ length: DM }, () => +(MLU.randn() * 0.9).toFixed(3)); }
  function pe(t) { return Array.from({ length: DM }, (_, i) => { const k = Math.floor(i / 2); const f = 1 / Math.pow(100, (2 * k) / DM); return +(i % 2 ? Math.cos(t * f) : Math.sin(t * f)).toFixed(3); }); }
  const matmul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
  const T_ = (A) => A[0].map((_, j) => A.map((r) => r[j]));
  const softmaxRow = (r, temp) => { const m = Math.max(...r.filter(isFinite)); const e = r.map((v) => (isFinite(v) ? Math.exp((v - m) / temp) : 0)); const s = e.reduce((a, b) => a + b, 0) || 1; return e.map((v) => v / s); };
  function randW(seed, r, c) { MLU.seedRng(seed); return Array.from({ length: r }, () => Array.from({ length: c }, () => +(MLU.randn() / Math.sqrt(r)).toFixed(3))); }

  function compute(tokens, P) {
    const T = tokens.length;
    const E = tokens.map(embed), PE = tokens.map((_, t) => pe(t));
    const X = E.map((e, t) => e.map((v, i) => v + (P.usePE ? PE[t][i] : 0)));
    let Q, K, V, dk = P.dk;
    if (P.head === "content") { Q = E.map((r) => r.slice(0, dk).concat(new Array(Math.max(0, dk - DM)).fill(0))); K = Q.map((r) => r.slice()); V = matmul(X, randW(7, DM, dk)); Q = Q.map((r) => r.map((v) => v * 1.6)); }
    else if (P.head === "prev") { dk = T; Q = tokens.map((_, i) => tokens.map((__, j) => (j === i - 1 ? 3 * Math.sqrt(T) : 0))); K = tokens.map((_, i) => tokens.map((__, j) => (j === i ? 1 : 0))); V = matmul(X, randW(7, DM, P.dk)); }
    else { Q = matmul(X, randW(1, DM, dk)); K = matmul(X, randW(2, DM, dk)); V = matmul(X, randW(3, DM, dk)); }
    const S = matmul(Q, T_(K));
    const Ss = S.map((r) => r.map((v) => v / Math.sqrt(dk)));
    const M = Ss.map((r, i) => r.map((v, j) => (P.causal && j > i ? -Infinity : v)));
    const A = M.map((r) => softmaxRow(r, P.temp));
    const O = matmul(A, V);
    return { E, PE, X, Q, K, V, S, Ss, M, A, O, dk };
  }

  function buildFrames(tokens, P, C) {
    const T = tokens.length;
    const frames = [];
    frames.push({ phase: "embed", major: true, arch: ["tok", "emb"], title: `Embed ${T} tokens as ${DM}-dimensional vectors`, text: "Each token is looked up in an embedding table (here: a fixed pseudo-random vector per word, so repeated words share a vector). Attention will let every token mix in information from the others.", tex: "X=\\big[e(w_1);\\dots;e(w_T)\\big]\\in\\mathbb R^{T\\times d}", st: { show: "E" } });
    if (P.usePE) frames.push({ phase: "pe", arch: ["emb", "pe"], title: "Add sinusoidal positional encodings", text: "Attention itself is permutation-invariant: without position information 'cat sat' and 'sat cat' look identical. Sines and cosines of different frequencies give every position a unique signature.", tex: "PE_{t,2k}=\\sin\\!\\big(t/10000^{2k/d}\\big),\\quad PE_{t,2k+1}=\\cos\\!\\big(t/10000^{2k/d}\\big)", st: { show: "X" } });
    frames.push({ phase: "qkv", major: true, arch: ["pe", "qkv"], title: "Project into queries, keys and values", text: P.head === "prev" ? "This hand-built head uses one-hot position codes so each query matches exactly the key one position back: a 'previous-token head', a pattern trained transformers really learn." : P.head === "content" ? "This head uses the raw word embeddings as both query and key, so tokens attend to identical or similar words: a 'content' head." : "Three learned linear maps. A query asks 'what am I looking for?', a key advertises 'what do I contain?', a value is 'what I pass on if selected'.", tex: "Q=XW_Q,\\quad K=XW_K,\\quad V=XW_V", st: { show: "QKV" } });
    for (let i = 0; i < T; i++) frames.push({ phase: "scores", arch: ["qkv", "dot"], title: `Scores for "${tokens[i]}": its query against every key`, text: `Row ${i + 1} of QKᵀ. A high dot product means the key matches what "${esc(tokens[i])}" is looking for.`, tex: "s_{ij}=q_i\\cdot k_j", st: { show: "S", row: i } });
    frames.push({ phase: "scale", arch: ["dot", "scale"], title: `Scale by 1/√d_k = 1/√${C.dk}`, text: "Dot products of d-dimensional vectors grow like √d. Without scaling, softmax saturates into near one-hot weights with vanishing gradients.", tex: "\\tilde s_{ij}=\\frac{q_i\\cdot k_j}{\\sqrt{d_k}}", st: { show: "Ss" } });
    if (P.causal) frames.push({ phase: "mask", arch: ["scale", "mask"], title: "Causal mask: no peeking at future tokens", text: "For language generation each position may only attend to itself and earlier positions. Masked scores are set to −∞, so they get zero weight after softmax.", tex: "\\tilde s_{ij}=-\\infty\\quad\\text{for } j>i", st: { show: "M" } });
    for (let i = 0; i < T; i += Math.max(1, Math.ceil(T / 5))) frames.push({ phase: "softmax", arch: ["mask", "soft"], title: `Softmax row by row (temperature ${P.temp})`, text: `Each row becomes a probability distribution over the tokens. Row "${esc(tokens[i])}" puts most weight on "${esc(tokens[C.A[i].indexOf(Math.max(...C.A[i]))])}".`, tex: "a_{ij}=\\frac{\\exp(\\tilde s_{ij}/\\tau)}{\\sum_{j'}\\exp(\\tilde s_{ij'}/\\tau)}", st: { show: "A", upto: i } });
    frames.push({ phase: "mix", major: true, arch: ["soft", "wv", "out"], title: "Output: each token becomes a weighted average of values", text: "The attention weights mix the value vectors. Hover a token to see where it looks (arcs). Stacking many such heads and layers, with feed-forward blocks in between, gives a transformer.", tex: "\\operatorname{Attention}(Q,K,V)=\\operatorname{softmax}\\!\\Big(\\frac{QK^\\top}{\\sqrt{d_k}}\\Big)V", st: { show: "A", upto: T - 1, out: true } });
    return frames;
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "attn", fps: 2, plane: false,
      stageHTML: `<div class="html-stage"><div class="sub-cap">tokens · hover one to see what it attends to</div><svg data-r="arcs" viewBox="0 0 720 110" class="arch-svg" style="max-height:120px"></svg><div data-r="toks" class="chip-row" style="justify-content:space-around;margin:2px 0 14px"></div><div class="sub-cap" data-r="mcap">matrix</div><div data-r="mat" class="hscroll"></div></div>`,
      legend: ui.legend([["var(--series-1)", "positive / weight"], ["var(--series-2)", "negative"]]),
      tools: ui.btn("replay", "replay", true),
      side: [
        { title: "input", html: `<div class="field"><label>sentence (≤ 10 tokens)</label><input type="text" data-k="txt" value="the cat sat on the mat because it was tired" /></div>` },
        { title: "attention head", html: ui.select("head", "head type", [["random", "random projections (untrained)"], ["content", "content head (same / similar words)"], ["prev", "previous-token head"]], "content") + ui.range("dk", "d_k (key dimension)", 2, 6, 1, 4) + ui.range("temp", "softmax temperature τ", 0.1, 3, 0.1, 1) + ui.check("causal", "causal mask (decoder)", false) + ui.check("pe", "add positional encoding", true) },
      ],
      arch: {
        w: 380, h: 200,
        nodes: [
          { id: "tok", label: "tokens", x: 38, y: 34, w: 60, kind: "data" },
          { id: "emb", label: "embed", sub: "lookup", x: 38, y: 100, w: 62, kind: "op" },
          { id: "pe", label: "+ PE", sub: "position", x: 38, y: 166, w: 62, kind: "param" },
          { id: "qkv", label: "Q, K, V", sub: "linear maps", x: 140, y: 100, kind: "op" },
          { id: "dot", label: "QKᵀ", sub: "similarity", x: 230, y: 34, w: 64, kind: "op" },
          { id: "scale", label: "÷ √d_k", x: 316, y: 34, w: 60, kind: "param" },
          { id: "mask", label: "mask", sub: "optional", x: 316, y: 100, w: 60, kind: "param" },
          { id: "soft", label: "softmax", sub: "rows sum to 1", x: 316, y: 166, kind: "loss" },
          { id: "wv", label: "× V", sub: "weighted sum", x: 214, y: 166, w: 70, kind: "op" },
          { id: "out", label: "output", sub: "T × d_v", x: 214, y: 100, w: 64, kind: "out" },
        ],
        edges: [{ from: "tok", to: "emb" }, { from: "emb", to: "pe" }, { from: "pe", to: "qkv" }, { from: "qkv", to: "dot" }, { from: "dot", to: "scale" }, { from: "scale", to: "mask" }, { from: "mask", to: "soft" }, { from: "soft", to: "wv" }, { from: "qkv", to: "wv", label: "V" }, { from: "wv", to: "out" }],
      },
      phases: { embed: { label: "embed", color: 3 }, pe: { label: "positions", color: 3 }, qkv: { label: "Q · K · V", color: 4 }, scores: { label: "scores", color: 0 }, scale: { label: "scale", color: 0 }, mask: { label: "mask", color: 1 }, softmax: { label: "softmax", color: 2 }, mix: { label: "output", color: 2 } },
      panels: [{ id: "qkv", title: "queries · keys · values", wide: true }],
      onFrame: render,
    });
    let tokens = [], C = null, hover = null;
    const P = () => ({ head: lab.val("head"), dk: lab.val("dk"), temp: lab.val("temp"), causal: lab.val("causal"), usePE: lab.val("pe") });
    function rebuild({ at = "start" } = {}) {
      tokens = String(lab.ctl("txt").value || "").trim().split(/\s+/).filter(Boolean).slice(0, 10);
      if (!tokens.length) tokens = ["hello"];
      C = compute(tokens, P());
      lab.load(buildFrames(tokens, P(), C), { at, play: at === "start" });
    }
    const mat = Lab.matrix(lab.R("mat"), { cell: 46, labelW: 64, labelH: 22, maxW: 720 });
    function render(f) {
      const st = f.st, T = tokens.length;
      // token chips
      const tk = lab.R("toks");
      tk.innerHTML = tokens.map((t, i) => `<span class="tok${hover === i ? " on" : ""}" data-i="${i}">${esc(t)}</span>`).join("");
      tk.querySelectorAll(".tok").forEach((el) => { el.onmouseenter = () => { hover = +el.dataset.i; lab.player.redraw(); }; });
      // arcs from the hovered token
      const arcs = d3.select(lab.R("arcs")); arcs.selectAll("*").remove();
      const xs = (i) => 360 + (i - (T - 1) / 2) * Math.min(72, 700 / T);
      const qi = hover != null && hover < T ? hover : null;
      if (qi != null && (st.show === "A")) {
        C.A[qi].forEach((w, j) => {
          if (w < 0.01) return;
          const x1 = xs(qi), x2 = xs(j), h = Math.min(95, 18 + Math.abs(x2 - x1) * 0.35);
          arcs.append("path").attr("d", j === qi ? `M${x1 - 6},106 C${x1 - 22},70 ${x1 + 22},70 ${x1 + 6},106` : `M${x1},106 Q${(x1 + x2) / 2},${106 - h * 1.6} ${x2},106`).attr("fill", "none").attr("stroke", "var(--series-1)").attr("stroke-width", 1 + 9 * w).attr("stroke-opacity", 0.25 + 0.7 * w);
          arcs.append("text").attr("x", x2).attr("y", 100 - (j === qi ? 28 : Math.min(60, h))).attr("text-anchor", "middle").style("font-size", "9px").attr("fill", "var(--text-dim)").text((w * 100).toFixed(0) + "%");
        });
      } else arcs.append("text").attr("x", 360).attr("y", 60).attr("text-anchor", "middle").attr("class", "lbl").text(st.show === "A" ? "hover a token" : "arcs appear after softmax");
      // main matrix
      const cap = lab.R("mcap");
      const labelsD = Array.from({ length: DM }, (_, i) => "d" + (i + 1));
      if (st.show === "E" || st.show === "X") { cap.textContent = st.show === "E" ? "embedding matrix X (rows = tokens)" : "X + positional encoding"; mat.set(st.show === "E" ? C.E : C.X, { rowL: tokens, colL: labelsD }); }
      else if (st.show === "QKV") { cap.textContent = "queries Q (rows = tokens)"; mat.set(C.Q, { rowL: tokens, colL: C.Q[0].map((_, i) => "q" + (i + 1)) }); }
      else {
        const src = st.show === "S" ? C.S : st.show === "Ss" ? C.Ss : st.show === "M" ? C.M.map((r) => r.map((v) => (isFinite(v) ? v : NaN))) : C.A;
        const rows = src.map((r, i) => (st.show === "S" && i > st.row) || (st.show === "A" && st.upto != null && i > st.upto && !st.out) ? r.map(() => NaN) : r);
        cap.textContent = { S: "raw scores QKᵀ (row = query, column = key)", Ss: "scaled scores", M: "masked scores (blank = −∞)", A: "attention weights (rows sum to 1)" }[st.show];
        mat.set(rows.map((r) => r.map((v) => (isFinite(v) ? v : NaN))), { rowL: tokens, colL: tokens, hl: st.show === "S" ? { r: st.row } : qi != null && st.show === "A" ? { r: qi } : null, range: st.show === "A" ? [0, 1] : null });
      }
      const qk = lab.R("panel-qkv"); qk.innerHTML = `<div style="display:grid;grid-template-columns:repeat(${st.out ? 4 : 3},minmax(0,1fr));gap:10px"></div>`;
      const g = qk.firstChild;
      const blocks = [["Q", C.Q], ["K", C.K], ["V", C.V]].concat(st.out ? [["output", C.O]] : []);
      for (const [name, M] of blocks) { const d = document.createElement("div"); d.innerHTML = `<div class="sub-cap">${name}</div>`; g.appendChild(d); const holder = document.createElement("div"); d.appendChild(holder); Lab.matrix(holder, { cell: 16, showValues: false, labelW: 40 }).set(M.map((r) => r.slice(0, 8)), { rowL: tokens.map((t) => t.slice(0, 6)) }); }
      const ent = C.A.map((r) => -r.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0));
      lab.readout(`<div class="kv"><span>tokens</span><b>${T}</b><span>d_model · d_k</span><b>${DM} · ${C.dk}</b><span>mean attention entropy</span><b>${MLU.mean(ent).toFixed(2)} nats</b><span>score matrix</span><b>${T} × ${T}</b></div><div class="note">Cost is O(T²·d): every token attends to every token. Weights are illustrative, not trained.</div>`);
    }
    lab.watch(["head", "dk", "temp", "causal", "pe"], () => rebuild({ at: "end" }), { live: false });
    lab.ctl("txt").addEventListener("change", () => rebuild());
    lab.q('[data-b="replay"]').onclick = () => rebuild();
    rebuild();
    return lab.destroy;
  }

  MLApp.register({
    id: "attention",
    name: "Self-Attention (Transformer)",
    category: "Deep Learning",
    tagline: "Q·Kᵀ/√d → softmax → ·V",
    isNew: true,
    description: "The core of every transformer, one matrix at a time. Type a sentence, then step through embeddings, positional encodings, query/key/value projections, the score matrix, scaling, causal masking and the softmax, and hover any token to see where it attends.",
    info: {
      type: "Deep learning - Sequence modelling building block (scaled dot-product attention). Differentiable, content-based routing of information.",
      scenario: "Any sequence or set where elements must exchange information depending on content: language models, translation, vision transformers, speech, protein models.",
      inputs: "A sequence of T token vectors X ∈ ℝ^{T×d}; learned matrices W_Q, W_K, W_V.",
      output: "A new sequence of T vectors, each a context-dependent mixture of the value vectors.",
      intuition: {
        definition: "Every token sends out a query describing what it needs, and every token advertises a key describing what it holds. Matching queries against keys gives a relevance score for every pair; softmax turns each token's scores into weights, and the token's new representation is the weighted average of everyone's values.",
        steps: ["Project tokens into queries, keys and values.", "Score every query against every key with a dot product.", "Scale by 1/√d_k and optionally mask the future.", "Softmax each row into attention weights.", "Mix the value vectors with those weights."],
        applications: ["Large language models (GPT, Claude, Llama)", "Machine translation (the original 2017 transformer)", "Vision transformers on image patches", "AlphaFold-style protein structure models"],
      },
      math: [
        { title: "Scaled dot-product attention", formula: "Attention(Q,K,V) = softmax(QKᵀ / √d_k) V", note: "One matrix multiply for scores, a row-wise softmax, another multiply for the output." },
        { title: "Why √d_k", formula: "Var(q·k) = d_k for unit-variance entries", note: "Scaling keeps scores O(1) so softmax does not saturate and gradients survive." },
        { title: "Causal mask", formula: "s_{ij} = −∞ for j > i", note: "Makes position i depend only on positions ≤ i, required for autoregressive generation." },
        { title: "Multi-head attention", formula: "MHA(X) = Concat(head₁..head_h) W_O,  headᵢ = Attention(XW_Q^i, XW_K^i, XW_V^i)", note: "Several smaller heads attend to different relationships in parallel." },
        { title: "Positional encoding", formula: "PE(t, 2k) = sin(t/10000^{2k/d}),  PE(t, 2k+1) = cos(...)", note: "Attention is permutation-equivariant; positions must be injected (sinusoidal, learned, or rotary RoPE)." },
        { title: "Complexity", formula: "O(T² d) time, O(T²) memory", note: "The quadratic cost in sequence length motivates FlashAttention, sparse and linear attention." },
      ],
      pipeline: [{ label: "Embed + position" }, { label: "Q, K, V" }, { label: "QKᵀ/√d" }, { label: "Mask, softmax" }, { label: "× V", accent: "green" }],
      decisionFunction: { text: "outᵢ = Σⱼ softmaxⱼ(qᵢ·kⱼ / √d_k) · vⱼ", mechanism: "A soft, differentiable lookup: instead of retrieving one value by exact key match, every value is retrieved with a weight proportional to the exponentiated similarity." },
      lossFunction: { text: "Attention has no loss of its own; it is trained end to end inside a model (e.g. next-token cross-entropy)", mechanism: "Gradients flow through the softmax into W_Q, W_K and W_V, so the projections learn whichever relationships reduce the model's overall loss." },
      optimization: [{ title: "Adam(W) + warmup", formula: "lr warmup then decay", note: "Transformers are sensitive to early large updates." }, { title: "Layer norm + residuals", formula: "x + Attn(LN(x))", note: "Keeps deep stacks trainable." }, { title: "FlashAttention", formula: "tiling, no T×T matrix in memory", note: "Exact attention with far less memory traffic." }],
      assumptions: [{ name: "Relationships are content-dependent", why: "Attention routes by similarity of learned projections.", check: "For fixed local structure a convolution may be cheaper." }, { name: "Sequence fits in memory", why: "O(T²) scores.", check: "Use efficient attention or chunking for very long inputs." }],
      hyperparameters: [{ name: "d_model", range: "64 - 8192", increasing: "More capacity per token.", strategy: "Scale with data and compute." }, { name: "heads", range: "1 - 128", increasing: "More parallel relationship types, smaller d_k each.", strategy: "d_k = d_model / heads ≈ 64 is common." }, { name: "context length T", range: "128 - 1M+", increasing: "More context, quadratic cost.", strategy: "As long as the task needs." }, { name: "dropout on weights", range: "0 - 0.1", increasing: "Regularises attention patterns.", strategy: "Small values for large data." }],
      metrics: ["Task loss (perplexity for language models)", "Attention entropy per head", "Probing / attribution analyses"],
      typicalUses: ["Language modelling", "Translation and summarisation", "Vision and multimodal models"],
      diagnostics: ["Visualise attention maps per head to find positional, syntactic or copying heads.", "Watch for attention collapsing onto a single token (entropy near 0).", "Check for exploding scores (missing scaling or normalisation)."],
      advantages: ["Any token can reach any other in one step (no recurrence).", "Fully parallel over the sequence on GPUs.", "Interpretable-ish attention maps.", "Scales remarkably well with data and parameters."],
      limitations: [{ name: "Quadratic cost in length", note: "T² time and memory", fix: "FlashAttention, sparse / linear / sliding-window attention." }, { name: "No built-in order", note: "needs positional encodings", fix: "sinusoidal, learned or rotary encodings." }, { name: "Data hungry", note: "weak inductive bias", fix: "pre-training and transfer." }],
      alternatives: [{ name: "RNN / LSTM", when: "Streaming with constant memory per step." }, { name: "CNN", when: "Local structure dominates (images, audio) and compute is tight." }, { name: "State-space models (Mamba)", when: "Very long sequences with linear-time processing." }],
      pitfalls: [{ problem: "Training diverges early", solution: "Use learning-rate warmup and pre-layer-norm." }, { problem: "Model ignores word order", solution: "Positional encodings missing or broken." }, { problem: "Future tokens leak into training", solution: "Apply the causal mask in decoder self-attention." }],
      quickRef: [{ name: "Attention", formula: "softmax(QKᵀ/√d_k)V" }, { name: "Mask", formula: "−∞ above diagonal" }, { name: "Multi-head", formula: "Concat(heads)W_O" }, { name: "Cost", formula: "O(T²d)" }],
      code: `import torch, torch.nn.functional as F

def attention(Q, K, V, causal=False):
    d = Q.size(-1)
    s = Q @ K.transpose(-2, -1) / d ** 0.5          # (T, T) scores
    if causal:
        T = s.size(-1)
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=bool), 1), float("-inf"))
    return F.softmax(s, dim=-1) @ V

# or the fused, memory-efficient kernel:
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)`,
      whyChain: [
        { q: "Why divide by √d_k?", a: "If query and key entries have unit variance, their dot product has variance d_k. Large scores push softmax into saturation where one weight is ~1 and gradients to the rest vanish. Dividing by √d_k restores unit variance." },
        { q: "Why separate queries, keys and values?", a: "What a token looks for, what it can be found by, and what it passes on are different roles. Separate projections let a token attend based on one property while transmitting another." },
        { q: "Why does attention need positional encodings?", a: "Softmax over a set does not depend on order: permuting the tokens just permutes the output. Adding position-dependent vectors breaks that symmetry so order can matter." },
        { q: "Why multiple heads?", a: "One softmax produces one weighted average per token, which blurs different relationships together. Several heads can track syntax, coreference and position separately and the output projection combines them." },
      ],
      workedExample: {
        setup: "Two tokens, d_k = 1: q₁ = 2, k₁ = 1, k₂ = 0, values v₁ = 10, v₂ = 0.",
        steps: ["Scores for token 1: q₁k₁ = 2, q₁k₂ = 0.", "Softmax: e²/(e² + 1) ≈ 0.881 and 0.119.", "Output: 0.881·10 + 0.119·0 = 8.81."],
        result: "Token 1's output is 8.81: mostly the value of the key it matched",
      },
    },
    mount,
  });
})();
