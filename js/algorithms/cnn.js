(() => {
  const { ui, col, fmt } = Lab;
  const N = 16;
  const KERNELS = {
    sobelx: { name: "vertical edges (Sobel x)", k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
    sobely: { name: "horizontal edges (Sobel y)", k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] },
    blur: { name: "box blur", k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => +(v / 9).toFixed(3))) },
    sharpen: { name: "sharpen", k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
    laplace: { name: "Laplacian (blobs / corners)", k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] },
    diag: { name: "diagonal ↘", k: [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]] },
    emboss: { name: "emboss", k: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]] },
    identity: { name: "identity", k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
  };
  function image(kind) {
    const g = Array.from({ length: N }, () => new Array(N).fill(0));
    const set = (r, c, v = 1) => { if (r >= 0 && r < N && c >= 0 && c < N) g[r][c] = v; };
    if (kind === "seven") { for (let c = 3; c < 13; c++) { set(3, c); set(4, c); } for (let r = 5; r < 14; r++) { const c = Math.round(12 - (r - 5) * 0.6); set(r, c); set(r, c - 1); } }
    else if (kind === "x") { for (let i = 2; i < 14; i++) { set(i, i); set(i, i + 1); set(i, 15 - i); set(i, 14 - i); } }
    else if (kind === "circle") { for (let a = 0; a < 64; a++) { const t = (a / 64) * Math.PI * 2; set(Math.round(7.5 + 5.5 * Math.sin(t)), Math.round(7.5 + 5.5 * Math.cos(t))); set(Math.round(7.5 + 4.6 * Math.sin(t)), Math.round(7.5 + 4.6 * Math.cos(t))); } }
    else if (kind === "bars") { for (let r = 2; r < 14; r++) for (const c of [3, 4, 9, 10]) set(r, c); }
    else if (kind === "square") { for (let r = 4; r < 12; r++) for (let c = 4; c < 12; c++) set(r, c); }
    else if (kind === "checker") { for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if ((Math.floor(r / 4) + Math.floor(c / 4)) % 2) set(r, c); }
    else if (kind === "noise") { for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) set(r, c, MLU.rng() < 0.3 ? 1 : 0); }
    return g;
  }
  function conv(img, k, stride, pad, bias) {
    const P = pad === "same" ? 1 : 0;
    const out = [], H = Math.floor((N + 2 * P - 3) / stride) + 1;
    for (let i = 0; i < H; i++) {
      const row = [];
      for (let j = 0; j < H; j++) {
        let s = bias;
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { const r = i * stride + a - P, c = j * stride + b - P; s += (r >= 0 && r < N && c >= 0 && c < N ? img[r][c] : 0) * k[a][b]; }
        row.push(s);
      }
      out.push(row);
    }
    return out;
  }
  function pool(M, kind) {
    const H = Math.floor(M.length / 2), out = [];
    for (let i = 0; i < H; i++) { const row = []; for (let j = 0; j < H; j++) { const v = [M[2 * i][2 * j], M[2 * i][2 * j + 1], M[2 * i + 1][2 * j], M[2 * i + 1][2 * j + 1]]; row.push(kind === "max" ? Math.max(...v) : MLU.mean(v)); } out.push(row); }
    return out;
  }

  function buildFrames(img, k, P) {
    const full = conv(img, k, P.stride, P.pad, P.bias);
    const H = full.length;
    const relu = full.map((r) => r.map((v) => Math.max(0, v)));
    const pooled = pool(relu, P.pool);
    const frames = [{ phase: "input", major: true, arch: ["in"], title: `Input image: ${N} × ${N} pixels`, text: "A convolutional layer does not look at the whole image at once. It slides a small kernel (the 3 × 3 grid of weights) across it, computing one weighted sum per position. Draw on the input to change it.", st: { done: 0 } }];
    const every = 1;
    let done = 0;
    for (let i = 0; i < H; i++) for (let j = 0; j < H; j++) {
      done++;
      if (done % every && done !== H * H) continue;
      const pad = P.pad === "same" ? 1 : 0;
      const terms = [];
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { const r = i * P.stride + a - pad, c = j * P.stride + b - pad; const x = r >= 0 && r < N && c >= 0 && c < N ? img[r][c] : 0; if (x && k[a][b]) terms.push(`${fmt(k[a][b], 2)}·${x}`); }
      frames.push({ phase: "conv", major: done === 1, arch: ["in", "k", "conv"], title: `Output (${i + 1}, ${j + 1}): slide the kernel and take a dot product`, text: `${terms.length ? terms.slice(0, 9).join(" + ") : "every overlapping input pixel is 0"}${P.bias ? ` + b(${P.bias})` : ""} = <b>${fmt(full[i][j], 2)}</b>. The same 9 weights are reused at every position: that weight sharing is why a conv layer has so few parameters and detects a pattern wherever it appears.`, tex: "y_{ij}=b+\\sum_{a=0}^{2}\\sum_{c=0}^{2}K_{ac}\\;x_{\\,i s+a,\\;j s+c}", st: { done, win: [i, j] } });
    }
    frames.push({ phase: "relu", major: true, arch: ["conv", "relu"], title: "Apply ReLU: keep positive responses, zero the rest", text: `${full.flat().filter((v) => v < 0).length} negative responses become 0. What survives is a map of where the kernel's pattern was found.`, tex: "\\mathrm{ReLU}(y)=\\max(0,y)", st: { done: H * H, relu: true } });
    const PH = pooled.length;
    for (let i = 0; i < PH; i++) for (let j = 0; j < PH; j++) frames.push({ phase: "pool", major: i === 0 && j === 0, arch: ["relu", "pool"], title: `${P.pool === "max" ? "Max" : "Average"}-pool 2 × 2 → cell (${i + 1}, ${j + 1})`, text: `Each 2 × 2 block collapses to its ${P.pool === "max" ? "maximum" : "mean"} (${fmt(pooled[i][j], 2)}). Pooling halves the resolution, makes the response tolerant to small shifts, and widens the receptive field of later layers.`, tex: P.pool === "max" ? "p_{ij}=\\max_{(a,c)\\in\\{0,1\\}^2} r_{2i+a,\\,2j+c}" : "p_{ij}=\\tfrac14\\sum_{(a,c)} r_{2i+a,\\,2j+c}", st: { done: H * H, relu: true, pdone: i * PH + j + 1, pwin: [i, j] } });
    frames.push({ phase: "done", major: true, arch: ["pool", "flat", "dense"], title: `Feature map ${PH} × ${PH} ready for the next layer`, text: `A real CNN learns dozens of kernels per layer (see the filter bank below), stacks several conv → ReLU → pool blocks, then flattens into dense layers for classification. Parameters in this layer: 3 × 3 + 1 = <b>10</b>, versus ${N * N * H * H} for a dense layer of the same output size.`, st: { done: H * H, relu: true, pdone: PH * PH } });
    return { frames, full, relu, pooled };
  }

  function mount(root) {
    const lab = Lab.studio(root, {
      id: "cnn", fps: 24, plane: false,
      stageHTML: `<svg data-r="cnn" class="lab-svg" viewBox="0 0 780 290"></svg>`,
      legend: ui.legend([["var(--series-1)", "positive response"], ["var(--series-2)", "negative response"], ["var(--series-4)", "receptive window", "ring"]]),
      tools: ui.btn("replay", "replay", true) + ui.btn("clear", "clear input"),
      hint: "click or drag on the input to paint pixels · edit kernel weights in the sidebar",
      side: [
        { title: "input & kernel", html: ui.select("img", "input image", [["seven", "digit 7"], ["x", "letter X"], ["circle", "ring"], ["bars", "vertical bars"], ["square", "filled square"], ["checker", "checkerboard"], ["noise", "random noise"]], "seven") + ui.select("ker", "kernel preset", Object.entries(KERNELS).map(([k, v]) => [k, v.name]), "sobelx") + `<div class="field"><label>kernel weights</label><div data-r="kgrid" style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px"></div></div>` },
        { title: "layer", html: ui.select("stride", "stride", [["1", "1"], ["2", "2"]], "1") + ui.select("pad", "padding", [["same", "same (zero-pad 1)"], ["valid", "valid (no padding)"]], "same") + ui.select("pool", "pooling", [["max", "max-pool 2 × 2"], ["avg", "average-pool 2 × 2"]], "max") + ui.range("bias", "bias b", -3, 3, 0.5, 0) },
      ],
      arch: {
        w: 380, h: 190,
        nodes: [
          { id: "in", label: "input", sub: "16 × 16 × 1", x: 50, y: 44, kind: "data" },
          { id: "k", label: "kernel K", sub: "3 × 3 + bias", x: 50, y: 130, kind: "param" },
          { id: "conv", label: "convolve", sub: "slide, dot product", x: 158, y: 86, kind: "op" },
          { id: "relu", label: "ReLU", x: 262, y: 44, w: 60, kind: "op" },
          { id: "pool", label: "pool 2×2", sub: "downsample", x: 262, y: 130, kind: "op" },
          { id: "flat", label: "flatten", x: 344, y: 86, w: 58, kind: "op" },
          { id: "dense", label: "dense + softmax", sub: "classifier", x: 300, y: 172, kind: "out" },
        ],
        edges: [{ from: "in", to: "conv" }, { from: "k", to: "conv" }, { from: "conv", to: "relu" }, { from: "relu", to: "pool" }, { from: "pool", to: "flat" }, { from: "flat", to: "dense" }],
      },
      phases: { input: { label: "input", color: 3 }, conv: { label: "convolution", color: 0 }, relu: { label: "ReLU", color: 2 }, pool: { label: "pooling", color: 4 }, done: { label: "feature map", color: 2 } },
      panels: [{ id: "bank", title: "a filter bank: every preset kernel applied to this input", sub: "after ReLU", wide: true }],
      onFrame: render,
    });
    let img = image("seven"), K = KERNELS.sobelx.k.map((r) => r.slice()), R = null;
    const P = () => ({ stride: +lab.val("stride"), pad: lab.val("pad"), pool: lab.val("pool"), bias: lab.val("bias") });
    const kgrid = lab.R("kgrid");
    function drawKInputs() {
      kgrid.innerHTML = K.flat().map((v, i) => `<input type="number" step="0.5" value="${v}" data-ki="${i}" style="padding:4px;text-align:center" aria-label="kernel weight ${i + 1}" />`).join("");
      kgrid.querySelectorAll("input").forEach((inp) => inp.onchange = () => { const i = +inp.dataset.ki; K[Math.floor(i / 3)][i % 3] = +inp.value || 0; replay({ at: "end" }); });
    }
    function replay({ at = "start" } = {}) { R = buildFrames(img, K, P()); drawBank(); lab.load(R.frames, { at, play: at === "start" }); }
    function drawBank() {
      const el = lab.R("panel-bank"); el.innerHTML = "";
      const wrap = d3.select(el).append("div").style("display", "grid").style("grid-template-columns", "repeat(auto-fill, minmax(96px, 1fr))").style("gap", "10px");
      for (const [key, v] of Object.entries(KERNELS)) {
        const out = conv(img, v.k, 1, "same", 0).map((r) => r.map((x) => Math.max(0, x)));
        const mx = Math.max(1e-6, ...out.flat());
        const d = wrap.append("div").style("cursor", "pointer").on("click", () => { lab.ctl("ker").value = key; lab.ctl("ker").dispatchEvent(new Event("change", { bubbles: true })); });
        const cv = d.append("canvas").attr("width", N).attr("height", N).style("width", "100%").style("image-rendering", "pixelated").style("border-radius", "4px").style("border", "1px solid var(--border-soft)").node();
        const ctx = cv.getContext("2d"), im = ctx.createImageData(N, N), c = Lab.palRGB()[0];
        out.flat().forEach((x, i) => { const t = x / mx; im.data[i * 4] = c[0]; im.data[i * 4 + 1] = c[1]; im.data[i * 4 + 2] = c[2]; im.data[i * 4 + 3] = Math.round(30 + t * 225); });
        ctx.putImageData(im, 0, 0);
        d.append("div").attr("class", "lab-card-note").style("margin-top", "3px").text(v.name.split(" (")[0]);
      }
    }
    const svg = d3.select(lab.R("cnn"));
    const CS = 12.5;
    function grid(g, M, x0, y0, cs, colorFn, upto = Infinity) {
      const H = M.length;
      M.forEach((row, i) => row.forEach((v, j) => {
        const k = i * H + j;
        g.append("rect").attr("x", x0 + j * cs).attr("y", y0 + i * cs).attr("width", cs - 1).attr("height", cs - 1).attr("rx", 1.5).attr("fill", k < upto ? colorFn(v) : "var(--bg-raised)");
      }));
    }
    const pos = (v, mx) => (v >= 0 ? `color-mix(in srgb, var(--series-1) ${Math.round(Math.min(1, v / mx) * 90 + 4)}%, var(--bg-raised))` : `color-mix(in srgb, var(--series-2) ${Math.round(Math.min(1, -v / mx) * 90 + 4)}%, var(--bg-raised))`);
    let painting = null;
    function render(f) {
      const st = f.st, p = P();
      svg.selectAll("*").remove();
      const H = R.full.length, PH = R.pooled.length;
      const pad = p.pad === "same" ? 1 : 0;
      const label = (x, y, t) => svg.append("text").attr("class", "lbl").attr("x", x).attr("y", y).text(t);
      // input
      const gi = svg.append("g");
      label(10, 14, "input 16×16");
      grid(gi, img, 10, 22, CS, (v) => (v ? "var(--text)" : "var(--bg-raised)"));
      gi.selectAll("rect").style("cursor", "crosshair").each(function (_, idx) {
        const r = Math.floor(idx / N), c = idx % N;
        d3.select(this).on("mousedown", (ev) => { ev.preventDefault(); painting = img[r][c] ? 0 : 1; img[r][c] = painting; replay({ at: "end" }); })
          .on("mouseenter", () => { if (painting != null && img[r][c] !== painting) { img[r][c] = painting; replay({ at: "end" }); } });
      });
      // kernel
      const kx = 236, ky = 90;
      label(kx, ky - 8, "kernel");
      const km = Math.max(1e-6, ...K.flat().map(Math.abs));
      K.forEach((row, a) => row.forEach((v, b) => {
        svg.append("rect").attr("x", kx + b * 28).attr("y", ky + a * 28).attr("width", 26).attr("height", 26).attr("rx", 3).attr("fill", pos(v, km)).attr("stroke", "var(--border)");
        svg.append("text").attr("x", kx + b * 28 + 13).attr("y", ky + a * 28 + 16).attr("text-anchor", "middle").style("font-size", "9px").attr("fill", "var(--text)").text(fmt(v, 2).replace(/\.?0+$/, ""));
      }));
      // conv output (or relu)
      const ox = 350, oy = 22, ocs = (CS * N) / H;
      const M = st.relu ? R.relu : R.full;
      const mx = Math.max(1e-6, ...R.full.flat().map(Math.abs));
      label(ox, 14, st.relu ? `after ReLU ${H}×${H}` : `conv output ${H}×${H}`);
      grid(svg.append("g"), M, ox, oy, ocs, (v) => pos(v, mx), st.done);
      // pooled
      const px = 588;
      label(px, 14, `pooled ${PH}×${PH}`);
      grid(svg.append("g"), R.pooled, px, oy, Math.min(26, (CS * N) / PH * 0.62), (v) => pos(v, mx), st.pdone || 0);
      // windows
      if (st.win) {
        const [i, j] = st.win;
        const r0 = i * p.stride - pad, c0 = j * p.stride - pad;
        svg.append("rect").attr("x", 10 + c0 * CS - 1).attr("y", 22 + r0 * CS - 1).attr("width", 3 * CS + 1).attr("height", 3 * CS + 1).attr("fill", "var(--series-4)").attr("fill-opacity", 0.12).attr("stroke", "var(--series-4)").attr("stroke-width", 2.4).attr("pointer-events", "none");
        svg.append("rect").attr("x", ox + j * ocs - 1).attr("y", oy + i * ocs - 1).attr("width", ocs + 1).attr("height", ocs + 1).attr("fill", "none").attr("stroke", "var(--series-4)").attr("stroke-width", 2.4);
        svg.append("line").attr("x1", 10 + (c0 + 3) * CS).attr("y1", 22 + (r0 + 1.5) * CS).attr("x2", ox + j * ocs).attr("y2", oy + (i + 0.5) * ocs).attr("stroke", "var(--series-4)").attr("stroke-dasharray", "3 3").attr("opacity", 0.7);
      }
      if (st.pwin) {
        const [i, j] = st.pwin, pc = Math.min(26, (CS * N) / PH * 0.62);
        svg.append("rect").attr("x", ox + 2 * j * ocs - 1).attr("y", oy + 2 * i * ocs - 1).attr("width", 2 * ocs + 1).attr("height", 2 * ocs + 1).attr("fill", "none").attr("stroke", "var(--series-4)").attr("stroke-width", 2.4);
        svg.append("rect").attr("x", px + j * pc - 1).attr("y", oy + i * pc - 1).attr("width", pc + 1).attr("height", pc + 1).attr("fill", "none").attr("stroke", "var(--series-4)").attr("stroke-width", 2.4);
      }
      const nIn = N * N;
      lab.readout(`<div class="kv"><span>output size</span><b>${H} × ${H}</b><span>after pooling</span><b>${PH} × ${PH}</b><span>parameters</span><b>10</b><span>dense equivalent</span><b>${(nIn * H * H).toLocaleString()}</b></div><div class="note">output = ⌊(n + 2p − k) / s⌋ + 1 = ⌊(16 + ${2 * pad} − 3) / ${p.stride}⌋ + 1 = ${H}</div>`);
    }
    const stopPaint = () => (painting = null);
    window.addEventListener("mouseup", stopPaint);
    lab.cleanup.push(() => window.removeEventListener("mouseup", stopPaint));
    lab.watch(["img"], () => { img = image(lab.val("img")); replay(); });
    lab.watch(["ker"], () => { K = KERNELS[lab.val("ker")].k.map((r) => r.slice()); drawKInputs(); replay(); });
    lab.watch(["stride", "pad", "pool", "bias"], () => replay());
    lab.q('[data-b="replay"]').onclick = () => replay();
    lab.q('[data-b="clear"]').onclick = () => { img = image("none"); replay({ at: "end" }); };
    drawKInputs();
    replay();
    return lab.destroy;
  }

  MLApp.register({
    id: "cnn",
    name: "Convolutional Layer",
    category: "Deep Learning",
    tagline: "sliding kernel, ReLU, pooling, filter bank",
    description: "Watch a 3 × 3 kernel slide across a drawable 16 × 16 image, computing every output cell as a dot product, then ReLU and 2 × 2 pooling shrink it into a feature map. Edit kernel weights, stride and padding, and compare a whole bank of filters at once.",
    info: {
      type: "Building block of supervised deep nets - a parametric, weight-sharing feature-extraction layer (convolution).",
      scenario: "Image (or other grid-structured, e.g. spectrogram) data where spatially-local patterns like edges and textures matter, and you want far fewer parameters than a fully-connected layer by reusing one small filter everywhere.",
      inputs: "A 2D grid of pixel/feature values, and one or more small kernels (filters).",
      intuition: {
        definition: "A dense layer would give every pixel its own weight, which ignores the fact that an edge is an edge wherever it appears. A convolution instead <b>slides one small filter across every position</b>, so the same pattern detector is reused everywhere, cutting parameters by orders of magnitude and building in translation equivariance.",
        steps: [
          "Slide a small kernel over every position of the input.",
          "At each position take a weighted sum of the local patch.",
          "Apply ReLU so only present patterns pass through.",
          "Pool to shrink the map and gain small-shift tolerance.",
        ],
        applications: [
          "Image classification and object detection",
          "Semantic segmentation in medical and satellite imaging",
          "Audio via spectrograms, which are grids too",
          "Video, using 3D convolutions across time",
          "Any signal where local spatial patterns matter",
        ],
      },
      math: [
        { title: "Convolution (cross-correlation)", formula: "S[i,j] = Σ_u Σ_v K[u,v] · I[i+u, j+v] + b", note: "Frameworks implement cross-correlation and call it convolution. True convolution flips the kernel, which is irrelevant when the kernel is learned anyway." },
        { title: "Output size", formula: "O = ⌊(W − F + 2P)/S⌋ + 1", note: "W input size, F filter size, P padding, S stride. Worth memorising: it is the most common source of shape errors." },
        { title: "Padding modes", formula: "'valid': P = 0, shrinks    'same': P = (F−1)/2, preserves size", note: "Same padding is what allows very deep stacks without the map vanishing after a few layers." },
        { title: "Parameter count", formula: "(F · F · C_in + 1) · C_out", note: "Independent of input resolution. A 3×3 kernel over 64 channels producing 64 channels needs about 37k weights, whereas a dense layer on a 224×224×64 input would need billions." },
        { title: "Pooling", formula: "max-pool: max over the window;  avg-pool: mean over the window", note: "Downsamples and grants small-translation invariance. Max-pool keeps the strongest response, which suits 'was this pattern present anywhere here'." },
        { title: "Receptive field", formula: "RF_l = RF_{l−1} + (F_l − 1)·Π_{i<l} S_i", note: "How much of the original image one deep unit sees. Stacking two 3×3 layers gives a 5×5 receptive field with fewer parameters than one 5×5 layer, and adds a non-linearity." },
        { title: "Backprop through convolution", formula: "∂L/∂K = conv(input, δ),   ∂L/∂input = full-conv(δ, K_rotated)", note: "The gradient of a convolution is another convolution, which is why the whole stack trains with the same machinery as a dense network." },
      ],
      pipeline: [
        { label: "Input grid", note: "H × W × C" },
        { label: "Convolve", note: "shared kernel" },
        { label: "ReLU", note: "keep positives" },
        { label: "Pool", note: "downsample" },
        { label: "Deeper features", note: "edges → objects", accent: "green" },
      ],
      decisionFunction: {
        text: "feature_map[i,j] = Σ_{ki,kj} kernel[ki,kj]·input[i+ki, j+kj]  (+ bias), then usually ReLU + max-pool",
        mechanism: "The same small kernel slides across every position of the input, computing a local weighted sum at each - this is what makes the layer translation-equivariant and far cheaper than a dense layer over the whole image.",
      },
      lossFunction: {
        text: "(in a trained CNN) the network's overall loss, e.g. cross-entropy on the final classification, backpropagated through pooling and convolution",
        mechanism: "Here the kernel is fixed/hand-set rather than learned, so you can see exactly what one filter does to an image before any training happens - in practice, gradients from the final loss update every kernel value via backprop.",
      },
      optimization: [
        { title: "Weight sharing", formula: "one kernel reused at every position", note: "The core idea. It slashes the parameter count and encodes the prior that a useful pattern is useful anywhere in the image." },
        { title: "Translation equivariance", formula: "conv(shift(x)) = shift(conv(x))", note: "Shifting the input shifts the feature map identically. Pooling then converts equivariance into approximate invariance." },
        { title: "im2col", formula: "unfold patches into a matrix, then one big GEMM", note: "How convolution actually runs fast: it is turned into a dense matrix multiply so it can use highly optimised BLAS and tensor cores." },
        { title: "Residual connections", formula: "y = F(x) + x", note: "Gives the gradient an identity path straight back through the network, which is what made hundred-layer CNNs trainable." },
        { title: "Transfer learning", formula: "freeze early layers, retrain the head", note: "Early layers learn generic edges and textures that transfer across almost any image task. This is why you rarely train a CNN from scratch." },
      ],
      output: "A transformed 'feature map' (same or smaller size than the input), then a further downsampled pooled map.",
      assumptions: [
        { name: "Input has grid structure", why: "Convolution assumes neighbouring positions are related. Shuffling the columns of tabular data changes nothing semantically, but destroys a CNN.", check: "Only use CNNs where adjacency is meaningful: images, audio spectrograms, time series." },
        { name: "Locality matters", why: "Small kernels can only see local patches, building global understanding through depth.", check: "If the task needs long-range relationships immediately, consider attention instead." },
        { name: "Translation invariance is desirable", why: "The architecture deliberately discards absolute position.", check: "If position itself is meaningful, add coordinate channels or positional encoding." },
        { name: "Inputs are normalised", why: "Unnormalised pixel ranges destabilise training.", check: "Scale to [0,1] or standardize per channel with the dataset statistics." },
        { name: "Enough data or a pretrained backbone", why: "CNNs have many parameters despite weight sharing.", check: "Below roughly ten thousand images, fine-tune a pretrained model rather than training from scratch." },
      ],
      regularization: [
        { name: "Data augmentation", formula: "random flips, crops, rotations, colour jitter", note: "By far the most effective regularizer for images. It teaches the invariances you actually want." },
        { name: "Batch normalisation", formula: "normalise per channel across the batch", note: "Stabilises training, permits higher learning rates, and regularizes mildly through batch noise." },
        { name: "Dropout", formula: "usually only in the dense head", note: "Standard dropout works poorly between conv layers because neighbouring activations are correlated. Use DropBlock or spatial dropout if needed." },
        { name: "Weight decay", formula: "L + λ‖W‖²", note: "Applied to kernel weights. Typically 1e-4 for image models." },
        { name: "Global average pooling", formula: "average each channel to a single value", note: "Replaces a huge flatten-plus-dense head, removing most of the network's parameters and a large chunk of its overfitting capacity." },
        { name: "Label smoothing / mixup", formula: "soften targets, or blend image pairs", note: "Improves calibration and robustness, standard in modern image training recipes." },
      ],
      hyperparameters: [
        { name: "kernel size F", range: "1, 3, 5, 7", increasing: "Larger receptive field per layer, quadratically more parameters.", strategy: "3×3 is near-universal. Two stacked 3×3 layers beat one 5×5: fewer parameters and an extra non-linearity. 1×1 mixes channels without touching space." },
        { name: "number of filters C_out", range: "16 - 512", increasing: "More patterns detectable per layer, more compute and memory.", strategy: "Conventionally doubles as spatial resolution halves, keeping compute roughly constant per stage." },
        { name: "stride S", range: "1 - 2", increasing: "Downsamples faster, cheaper, loses spatial detail.", strategy: "Stride 1 with pooling is classic; stride 2 convolution is the modern replacement for pooling." },
        { name: "padding", range: "valid / same", increasing: "Not applicable", strategy: "Same padding for deep stacks, otherwise the map shrinks away after a few layers." },
        { name: "pool size", range: "2 - 3", increasing: "More aggressive downsampling, more detail discarded.", strategy: "2×2 max-pool is standard. Many modern architectures drop pooling for strided convolution." },
        { name: "depth", range: "5 - 150+", increasing: "Larger receptive field and more abstraction, harder to train without residuals.", strategy: "Use a proven backbone such as ResNet or EfficientNet rather than designing depth yourself." },
        { name: "learning rate", range: "1e-4 - 1e-1", increasing: "Faster progress, then divergence.", strategy: "Still the dominant hyperparameter. Use warmup plus cosine decay for image models." },
      ],
      metrics: ["Classification accuracy / top-k accuracy (full network)", "IoU (segmentation/detection)", "mAP (object detection)", "Qualitative inspection of feature maps (this layer alone)"],
      typicalUses: ["Image classification", "Object detection / segmentation", "Any grid-structured signal (e.g. spectrograms) where local patterns matter"],
      diagnostics: [
        "Visualise the first-layer kernels. On a healthy trained network they look like edge and colour-blob detectors. Noise means training failed.",
        "Visualise intermediate feature maps. Maps that are entirely zero indicate dead filters.",
        "Compute the receptive field of your final conv layer. If it is smaller than the objects you care about, the network physically cannot see them whole.",
        "Use Grad-CAM to check the network attends to the object rather than the background. Models often latch onto spurious context.",
        "Track shape at every layer. Silent mismatches from padding and stride are the most common CNN bug.",
      ],
      advantages: [
        "Weight sharing gives orders of magnitude fewer parameters than a dense layer on the same input.",
        "Translation equivariance is built into the architecture rather than learned from data.",
        "Learns a natural hierarchy: edges, then textures, then parts, then objects.",
        "Parameter count is independent of input resolution, so one architecture handles many image sizes.",
        "Maps extremely well onto GPU hardware.",
        "Pretrained backbones transfer to new tasks with very little data.",
      ],
      limitations: [
        { name: "Only for grid-structured data", note: "meaningless on tabular features with no adjacency", fix: "use gradient boosting or a dense network." },
        { name: "Not rotation or scale invariant", note: "only translation is handled architecturally", fix: "data augmentation, or explicitly equivariant architectures." },
        { name: "Limited receptive field per layer", note: "long-range dependencies need considerable depth", fix: "dilated convolutions, or attention layers." },
        { name: "Pooling discards spatial precision", note: "harmful for segmentation and localisation", fix: "skip connections as in U-Net, or dilated convolutions." },
        { name: "Data and compute hungry", note: "training from scratch needs large labelled datasets", fix: "transfer learning from a pretrained backbone." },
        { name: "Vulnerable to adversarial and texture bias", note: "imperceptible perturbations flip predictions; models over-rely on texture", fix: "adversarial training, stronger augmentation." },
      ],
      alternatives: [
        { name: "Vision transformer", when: "Large datasets, and you want global context from the first layer. Needs more data than a CNN." },
        { name: "U-Net", when: "Segmentation, where skip connections restore the spatial detail pooling removed." },
        { name: "Pretrained backbone plus fine-tuning", when: "Almost always for real image tasks. Rarely worth training from scratch." },
        { name: "Dense network or boosting", when: "The data has no spatial structure." },
      ],
      pitfalls: [
        { problem: "Shape mismatch errors", solution: "Recompute O = (W − F + 2P)/S + 1 at every layer. Padding and stride interactions are the usual culprit." },
        { problem: "Feature map vanishes to 1×1 too early", solution: "Use same padding, or reduce the number of downsampling stages." },
        { problem: "Model overfits a small image dataset", solution: "Augment aggressively and fine-tune a pretrained backbone instead of training from scratch." },
        { problem: "Dropout between conv layers hurts", solution: "Neighbouring activations are correlated, so standard dropout is weak here. Use spatial dropout or rely on batch norm." },
        { problem: "Objects at the image edge are missed", solution: "Valid padding discards border information. Switch to same padding." },
        { problem: "Model keys on the background", solution: "Confirm with Grad-CAM, then augment with random cropping and background variation." },
      ],
      quickRef: [
        { name: "Convolution", formula: "S[i,j] = ΣΣ K[u,v]·I[i+u,j+v] + b" },
        { name: "Output size", formula: "O = ⌊(W − F + 2P)/S⌋ + 1" },
        { name: "Same padding", formula: "P = (F − 1)/2" },
        { name: "Parameters", formula: "(F·F·C_in + 1)·C_out" },
        { name: "Receptive field", formula: "RF += (F−1)·Π strides" },
        { name: "Two 3×3 vs one 5×5", formula: "same RF, fewer params, extra ReLU" },
        { name: "1×1 conv", formula: "channel mixing, no spatial change" },
        { name: "Residual block", formula: "y = F(x) + x" },
      ],
      code: `import torch, torch.nn as nn
from torchvision import models, transforms

# Train from scratch only with plenty of data.
block = lambda c_in, c_out: nn.Sequential(
    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),  # 'same' padding
    nn.BatchNorm2d(c_out),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),                                   # halves H and W
)
net = nn.Sequential(
    block(3, 32), block(32, 64), block(64, 128),
    nn.AdaptiveAvgPool2d(1),      # global average pooling, tiny head
    nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes),
)

# In practice: fine-tune a pretrained backbone instead.
backbone = models.resnet50(weights="IMAGENET1K_V2")
for p in backbone.parameters():
    p.requires_grad = False                    # freeze generic features
backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)

# Augmentation is the highest-leverage regularizer for images.
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])`,
      whyChain: [
        { q: "Why use convolution instead of a dense layer on images?", a: "Two reasons. Parameter count: a dense layer on a 224×224×3 image with 1000 units needs 150 million weights, while a 3×3 convolution needs a few hundred. And prior knowledge: sharing one kernel across all positions encodes the fact that a useful pattern is useful wherever it appears." },
        { q: "What is the difference between equivariance and invariance here?", a: "Convolution is equivariant: shift the input and the feature map shifts identically. Pooling then makes the representation approximately invariant, because the maximum over a window is unchanged by small shifts within it. Equivariance is the layer property, invariance is what pooling buys." },
        { q: "Why are two stacked 3×3 convolutions preferred to one 5×5?", a: "They cover the same 5×5 receptive field, but use 18 weights per channel pair instead of 25, and insert an extra non-linearity between them. More expressive and cheaper at once, which is why VGG established 3×3 as the default." },
        { q: "What does a 1×1 convolution do if it sees no spatial neighbourhood?", a: "It mixes channels. At each pixel it computes a learned linear combination across the channel dimension, so it changes channel depth cheaply. That is how bottleneck blocks in ResNet reduce cost before an expensive 3×3." },
        { q: "Why does receptive field matter in practice?", a: "A unit can only respond to what lies inside its receptive field. If your final conv layer sees a 40-pixel window but the objects are 150 pixels across, no amount of training lets it recognise them whole. You need more depth, larger strides, or dilation." },
        { q: "Why did residual connections make very deep CNNs possible?", a: "The identity path gives gradients a route that does not pass through weight layers, so they reach early layers undiminished. It also means a block only has to learn the residual correction rather than the whole mapping, and learning a near-identity is much easier." },
        { q: "Why is standard dropout ineffective between conv layers?", a: "Neighbouring activations in a feature map are highly correlated, so zeroing individual ones leaves the information available next door. Spatial dropout drops whole channels, and DropBlock drops contiguous regions, which actually removes information." },
        { q: "Why does transfer learning work so well for images?", a: "Early layers learn edges, corners, and textures that are generic to essentially all natural images. Only the later, task-specific layers need retraining, so a pretrained backbone gives you most of the representation for free and cuts the data requirement enormously." },
      ],
      parameters: [
        { name: "kernel weights", effect: "What pattern the filter detects (edges, blur, sharpen here - learned automatically during real training)." },
        { name: "ReLU on/off", effect: "Introduces non-linearity and zeroes out negative responses, so only 'this pattern is present' signals pass through positively." },
        { name: "pooling window/stride", effect: "How aggressively spatial resolution is reduced after convolution - larger pooling discards more fine spatial detail." },
      ],
      metrics: ["Classification accuracy / top-k accuracy (full network)", "IoU (segmentation/detection)", "Qualitative inspection of feature maps (this layer alone)"],
      typicalUses: ["Image classification", "Object detection / segmentation", "Any grid-structured signal (e.g. spectrograms) where local patterns matter"],
      workedExample: {
        setup: "3×3 vertical-edge kernel [[1,0,−1],[1,0,−1],[1,0,−1]] over a patch with a bright vertical stripe in the left column: [[1,0,0],[1,0,0],[1,0,0]].",
        steps: [
          "Row 1: 1×1 + 0×0 + 0×(−1) = 1.",
          "Row 2: 1×1 + 0×0 + 0×(−1) = 1.",
          "Row 3: 1×1 + 0×0 + 0×(−1) = 1.",
          "Sum all rows: 1 + 1 + 1 = 3.",
        ],
        result: "Output activation = 3 - a strong positive response, correctly flagging a left-edge under this vertical-edge filter",
      },
    },
    mount,
  });
})();
