(() => {
  const GRID = 18, CELL = 20; // input/feature-map cell rendering size
  const KERNELS = {
    "vertical edge": [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
    "horizontal edge": [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
    "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "blur": [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
    "identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  };

  function convolve(grid, k) {
    const n = grid.length;
    const out = MLU.zeros(n, n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let ki = -1; ki <= 1; ki++)
          for (let kj = -1; kj <= 1; kj++) {
            const gi = i + ki, gj = j + kj;
            if (gi >= 0 && gi < n && gj >= 0 && gj < n) s += grid[gi][gj] * k[ki + 1][kj + 1];
          }
        out[i][j] = s;
      }
    return out;
  }
  function relu(grid) { return grid.map((row) => row.map((v) => Math.max(0, v))); }
  function maxPool2(grid) {
    const n = grid.length, m = Math.floor(n / 2);
    const out = MLU.zeros(m, m);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++)
        out[i][j] = Math.max(grid[2 * i][2 * j], grid[2 * i + 1][2 * j], grid[2 * i][2 * j + 1], grid[2 * i + 1][2 * j + 1]);
    return out;
  }
  function toGray(v, lo, hi) {
    const t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
    const c = Math.round(Math.max(0, Math.min(1, t)) * 255);
    return `rgb(${c},${c},${c})`;
  }
  function drawGrid(svgSel, grid, cell, colorFn) {
    const n = grid.length;
    let flat = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) flat.push({ i, j, v: grid[i][j] });
    const lo = Math.min(...flat.map((d) => d.v)), hi = Math.max(...flat.map((d) => d.v));
    const sel = svgSel.selectAll("rect").data(flat);
    sel.enter().append("rect").attr("width", cell - 1).attr("height", cell - 1)
      .merge(sel)
      .attr("x", (d) => d.j * cell).attr("y", (d) => d.i * cell)
      .attr("fill", (d) => colorFn(d.v, lo, hi));
    sel.exit().remove();
  }

  function mount(root) {
    root.innerHTML = `
      <div class="panel">
        <div class="stage">
          <div class="stage-toolbar">
            <div class="legend"><span class="stage-hint">draw on the input grid - click and drag</span></div>
            <div class="btn-row">
              <button id="cnn-plus">draw +</button>
              <button id="cnn-diag">draw diagonal</button>
              <button id="cnn-square">draw square</button>
              <button id="cnn-clear">clear</button>
            </div>
          </div>
          <div class="hscroll" style="display:flex;gap:22px;flex-wrap:wrap;justify-content:center;padding:10px 0;flex:1;align-items:flex-start">
            <div style="text-align:center">
              <div class="stage-hint" style="margin-bottom:6px">input (${GRID}×${GRID})</div>
              <svg id="cnn-input" width="${GRID * CELL}" height="${GRID * CELL}"></svg>
            </div>
            <div style="text-align:center">
              <div class="stage-hint" style="margin-bottom:6px">after 3×3 convolution<span id="cnn-relu-tag"></span></div>
              <svg id="cnn-feature" width="${GRID * CELL}" height="${GRID * CELL}"></svg>
            </div>
            <div style="text-align:center">
              <div class="stage-hint" style="margin-bottom:6px">after 2×2 max-pool</div>
              <svg id="cnn-pool" width="${Math.floor(GRID / 2) * CELL * 2}" height="${Math.floor(GRID / 2) * CELL * 2}"></svg>
            </div>
          </div>
        </div>
        <div class="controls">
          <div class="control-card">
            <h3>kernel (filter)</h3>
            <div class="field"><label>preset</label>
              <select id="cnn-kernel">${Object.keys(KERNELS).map((k) => `<option value="${k}">${k}</option>`).join("")}<option value="custom">custom</option></select>
            </div>
            <div id="cnn-custom" style="display:none;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:8px"></div>
            <div class="field" style="margin-top:10px"><label><input type="checkbox" id="cnn-relu" style="width:auto;margin-right:6px" />apply ReLU</label></div>
          </div>
          <div class="control-card">
            <h3>what's happening</h3>
            <div class="note">A convolutional layer slides one small learned kernel across the image, computing a weighted sum at every position (the feature map). ReLU zeroes negative activations; max-pooling then downsamples by keeping the strongest response in each 2×2 block - the building blocks behind <code>mla/neuralnet</code>'s CNN layer, made tweakable here instead of learned.</div>
          </div>
        </div>
      </div>
    `;

    let grid = MLU.zeros(GRID, GRID);
    const inputSvg = d3.select("#cnn-input");
    const featSvg = d3.select("#cnn-feature");
    const poolSvg = d3.select("#cnn-pool");

    function currentKernel() {
      const sel = document.getElementById("cnn-kernel").value;
      if (sel === "custom") {
        const inputs = [...document.querySelectorAll("#cnn-custom input")];
        const k = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        inputs.forEach((inp, idx) => { k[Math.floor(idx / 3)][idx % 3] = +inp.value || 0; });
        return k;
      }
      return KERNELS[sel];
    }
    function buildCustomEditor() {
      const box = document.getElementById("cnn-custom");
      box.innerHTML = "";
      const flatId = KERNELS.identity;
      for (let i = 0; i < 9; i++) {
        const inp = document.createElement("input");
        inp.type = "number"; inp.step = "0.5"; inp.value = flatId[Math.floor(i / 3)][i % 3];
        inp.style.textAlign = "center";
        inp.addEventListener("input", render);
        box.appendChild(inp);
      }
    }
    buildCustomEditor();

    function render() {
      document.getElementById("cnn-custom").style.display = document.getElementById("cnn-kernel").value === "custom" ? "grid" : "none";
      const k = currentKernel();
      let feat = convolve(grid, k);
      const useRelu = document.getElementById("cnn-relu").checked;
      document.getElementById("cnn-relu-tag").textContent = useRelu ? " + ReLU" : "";
      if (useRelu) feat = relu(feat);
      const pooled = maxPool2(feat);

      drawGrid(inputSvg, grid, CELL, (v) => toGray(v, 0, 1));
      drawGrid(featSvg, feat, CELL, toGray);
      drawGrid(poolSvg, pooled, CELL * 2, toGray);
    }

    let painting = false, eraseMode = false;
    function paintAt(event) {
      const rect = inputSvg.node().getBoundingClientRect();
      const px = event.clientX - rect.left, py = event.clientY - rect.top;
      const j = Math.floor(px / CELL), i = Math.floor(py / CELL);
      if (i >= 0 && i < GRID && j >= 0 && j < GRID) { grid[i][j] = eraseMode ? 0 : 1; render(); }
    }
    inputSvg.on("mousedown", (event) => { painting = true; eraseMode = event.button === 2; paintAt(event); });
    inputSvg.on("mousemove", (event) => { if (painting) paintAt(event); });
    inputSvg.on("contextmenu", (event) => event.preventDefault());
    window.addEventListener("mouseup", () => { painting = false; });

    function stampShape(fn) {
      grid = MLU.zeros(GRID, GRID);
      for (let i = 0; i < GRID; i++) for (let j = 0; j < GRID; j++) if (fn(i, j)) grid[i][j] = 1;
      render();
    }
    document.getElementById("cnn-plus").addEventListener("click", () => stampShape((i, j) => Math.abs(i - GRID / 2) < 2 || Math.abs(j - GRID / 2) < 2));
    document.getElementById("cnn-diag").addEventListener("click", () => stampShape((i, j) => Math.abs(i - j) < 2));
    document.getElementById("cnn-square").addEventListener("click", () => stampShape((i, j) => i > 4 && i < GRID - 5 && j > 4 && j < GRID - 5 && (i < 7 || i > GRID - 8 || j < 7 || j > GRID - 8)));
    document.getElementById("cnn-clear").addEventListener("click", () => { grid = MLU.zeros(GRID, GRID); render(); });
    document.getElementById("cnn-kernel").addEventListener("change", render);
    document.getElementById("cnn-relu").addEventListener("change", render);

    stampShape((i, j) => Math.abs(i - GRID / 2) < 2 || Math.abs(j - GRID / 2) < 2);
    return () => { window.removeEventListener("mouseup", () => {}); };
  }

  MLApp.register({
    id: "cnn",
    name: "Convolutional Layer",
    category: "Deep Learning",
    tagline: "convolution → ReLU → max-pool",
    description: "Draw a small image and watch it pass through one convolutional layer: a 3×3 kernel slides across the input, an optional ReLU clips negatives, and max-pooling downsamples the result.",
    sourceFile: "mla/neuralnet",
    info: {
      type: "Building block of supervised deep nets - a parametric, weight-sharing feature-extraction layer (convolution).",
      scenario: "Image (or other grid-structured, e.g. spectrogram) data where spatially-local patterns like edges and textures matter, and you want far fewer parameters than a fully-connected layer by reusing one small filter everywhere.",
      inputs: "A 2D grid of pixel/feature values, and one or more small kernels (filters).",
      decisionFunction: {
        text: "feature_map[i,j] = Σ_{ki,kj} kernel[ki,kj]·input[i+ki, j+kj]  (+ bias), then usually ReLU + max-pool",
        mechanism: "The same small kernel slides across every position of the input, computing a local weighted sum at each - this is what makes the layer translation-equivariant and far cheaper than a dense layer over the whole image.",
      },
      lossFunction: {
        text: "(in a trained CNN) the network's overall loss, e.g. cross-entropy on the final classification, backpropagated through pooling and convolution",
        mechanism: "Here the kernel is fixed/hand-set rather than learned, so you can see exactly what one filter does to an image before any training happens - in practice, gradients from the final loss update every kernel value via backprop.",
      },
      output: "A transformed 'feature map' (same or smaller size than the input), then a further downsampled pooled map.",
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
