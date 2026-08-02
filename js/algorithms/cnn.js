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
