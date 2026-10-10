# ML Explorer Lab

An interactive machine-learning lab that runs entirely in the browser. 36 algorithms are implemented from scratch in JavaScript; each one loads example data, trains live, and records every step of training as a frame you can play, pause, scrub and step through.

## Run it

No build step. Open `index.html`, or serve the folder:

```
python3 -m http.server 8000
```

Libraries load from jsDelivr (pinned versions): D3 7.9.0 for every plot and diagram, TensorFlow.js 4.22.0 for the neural labs (MLP, autoencoder, GAN), KaTeX 0.16.11 for typeset maths. If the KaTeX stylesheet cannot load, equations fall back to native MathML automatically.

## What each lab has

- **Stage**: the main interactive view. Click to add points, drag to move them, double-click to delete; the run is re-recorded after every edit.
- **Player**: play / pause / step / scrub through recorded training frames, with chapter ticks at phase changes, speed control and loop. Keyboard: space, ← →, Home / End.
- **Narration**: what the algorithm is doing at this exact step, with the governing equation typeset.
- **Architecture diagram**: the algorithm's data flow; the active stage lights up as frames play. Hover a block for details, click it to jump to the next step that uses it.
- **Live charts**: loss, accuracy, regret, likelihood… with a cursor synced to the player.
- **Dataset card**: example datasets (blobs, moons, circles, XOR, spirals, anisotropic, outliers, sine, step, heteroscedastic…), sample size, noise, reseed, CSV upload (`x,y[,label]`) and CSV export.
- **Reference tab**: the full study sheet (concept, maths, loss, assumptions, hyperparameters, diagnostics, pitfalls, worked example, interview why-chain) plus typeset key equations, complexity, history & key papers, library entry points and related labs.

## Algorithms

| Family | Labs |
|---|---|
| Regression | Linear Regression (GD vs normal equation, loss landscape), Lasso & Elastic Net *(new)*, Gaussian Process Regression *(new)*, Factorization Machines |
| Classification | Perceptron *(new)*, Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Linear Discriminant Analysis *(new)*, Support Vector Machine (SMO) |
| Trees & ensembles | Decision Tree & Random Forest, AdaBoost *(new)*, Gradient Boosting, XGBoost, LightGBM |
| Clustering | K-Means, Gaussian Mixture Model, DBSCAN, Hierarchical *(new)*, Mean Shift *(new)*, Spectral *(new)* |
| Dimensionality reduction | PCA, t-SNE |
| Anomaly & association | Isolation Forest *(new)*, Apriori *(new)* |
| Probabilistic & sequential | Hidden Markov Model *(new)* |
| Deep learning | MLP (TF.js), Convolutional Layer, RNN / LSTM / GRU, Autoencoder *(new)*, Self-Attention *(new)*, GAN *(new)*, RBM |
| Optimization | Gradient Descent Optimisers race *(new)* |
| Reinforcement learning | Q-Learning / SARSA gridworld, Multi-Armed Bandits *(new)* |

Plus four practice guides: EDA, preprocessing, evaluation, and the ML pipeline.

## Shell

- Home launcher with learning paths, recently opened labs, and a sortable algorithm atlas (interpretability, scalability, non-linearity, scaling needs, probabilistic output).
- Search across names and every reference passage (`/` or `Ctrl K`), `[` `]` for previous / next algorithm, `L` / `R` for Lab / Reference, `?` for all shortcuts.
- Status bar showing the current lab, phase and step. Dark and light themes; phone layout.

## Structure

```
index.html            shell, pinned CDN scripts, load order
style.css             base theme (dark default, light toggle)
explorer.css          lab framework, launcher, status bar, reference additions
js/
  utils.js            seeded RNG, small linear algebra, generators
  app.js              registry, routing, home launcher, search, reference renderer
  lab.js              lab framework: player, plane, raster fields, contours, charts,
                      bar/matrix widgets, architecture & network diagrams, datasets
  extras.js           typeset equations, complexity, history, libraries, related, traits
  gbdt.js             shared second-order boosting core (XGBoost / LightGBM labs)
  algorithms/*.js     one file per algorithm: buildFrames() + mount() + reference info
  guides/*.js         supplementary study guides
```

Adding an algorithm: write a `buildFrames()` that returns an array of `{phase, title, text, tex, arch, st, m}` frames, call `Lab.studio(root, {...})` in `mount()`, render each frame in `onFrame`, and `MLApp.register({...})` with an `info` reference object. Add its extras entry in `js/extras.js`.
