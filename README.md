# ML Algorithms, Visual Implementation

A visual, interactive companion for a from-scratch collection of classical machine learning algorithms.

This project is a browser lab where you can click, drag, and step through 20 algorithms live.

## How to run it

No build step and no install required.

**Option 1.** Open `/index.html` directly in a browser.

**Option 2.** Serve it locally:

```
cd webapp
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

The page loads D3.js and TensorFlow.js from a CDN for rendering and the live neural network demo, so an internet connection is needed the first time it loads. Everything else, all the algorithm logic, is plain JavaScript with no dependencies and no server-side code.

## What is inside

Each algorithm has its own page with two tabs:

- **Playground**, an interactive demo. Click to add data points, drag them around, and watch the model refit live.
- **Reference**, a written explanation: what type of algorithm it is, when to use it, its inputs and output, its decision function and loss function (each with its mechanism explained), how to tune its parameters, what metrics to evaluate it with, typical real-world uses, and a small hand-solved numeric example.

### Algorithms covered

**Regression**
Linear Regression, Factorization Machines

**Classification**
Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Support Vector Machine

**Trees and ensembles**
Decision Tree and Random Forest, Gradient Boosting Trees, XGBoost, LightGBM

**Clustering**
K-Means, Gaussian Mixture Model, DBSCAN

**Dimensionality reduction**
Principal Component Analysis, t-SNE

**Deep learning**
Neural Network (MLP, trained live with TensorFlow.js), Convolutional layer, RNN and LSTM cell, Restricted Boltzmann Machine

**Reinforcement learning**
Q-Learning gridworld

Note: XGBoost, LightGBM, and DBSCAN are not part of the original MLAlgorithms repo. They were added here as commonly requested, widely used models the original collection did not cover. Their reference tabs say so directly.

## Project structure

```
./
    index.html           page shell, CDN script tags, script loading order
    style.css            theme (dark by default, light toggle available)
    js/
      utils.js           shared math helpers, data generators, plotting helper
      app.js             sidebar navigation, routing, theme toggle
      algorithms/
        <one file per algorithm, each self-registers into the app>
```

## Notes

- Dark and light themes are both supported. Use the toggle in the top right; the choice is remembered in the browser.
- On narrow screens the sidebar collapses into a menu button, and panels stack into a single column.
- Nothing here calls a backend. All computation, including the neural network training, runs in your browser.
