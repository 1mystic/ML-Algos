(() => {
  MLApp.register({
    id: "preprocessing",
    name: "Preprocessing Techniques",
    category: "Supplementary",
    tagline: "scaling, encoding, imputation, splits",
    kind: "guide",
    description: "How raw columns become model input: scaling, encoding categoricals, imputing gaps, transforming skew, handling imbalance, and splitting data. The recurring theme is that every step must be fitted on training data only.",
    guide: [
      {
        title: "The one rule that matters most",
        slug: "rule",
        body: [
          { p: "Almost every preprocessing bug is the same bug: <b>a transformation learned something from data the model should not have seen.</b> A scaler fitted on all rows knows the test set's mean. An encoder fitted before splitting knows which categories appear in the test set. Both inflate validation scores and both break in production." },
          {
            flow: [
              { label: "Split first", note: "train / val / test" },
              { label: "fit on train", note: "learn parameters" },
              { label: "transform train", note: "apply" },
              { label: "transform val/test", note: "apply only", accent: "green" },
              { label: "Wrap in a Pipeline", note: "so it cannot drift", accent: "green" },
            ],
            title: "Fit-transform discipline",
          },
          {
            callout: {
              tone: "red",
              title: "Never call fit on validation or test data",
              text: "Use fit_transform on training data and transform everywhere else. A scikit-learn Pipeline enforces this automatically inside cross-validation, which is the real reason to use one rather than transforming by hand.",
            },
          },
        ],
      },
      {
        title: "Scaling and normalisation",
        slug: "scaling",
        body: [
          { p: "Scaling matters when the algorithm measures distance or takes gradient steps. It is irrelevant when the algorithm only compares values within a single feature." },
          {
            table: {
              headers: ["Method", "Formula", "Result", "Use when"],
              rows: [
                ["Standardization", "<span class='inline-formula'>z = (x − μ) / σ</span>", "Mean 0, std 1, unbounded", "The default. Linear models, SVM, PCA, neural networks"],
                ["Min-max", "<span class='inline-formula'>(x − min) / (max − min)</span>", "Bounded to [0, 1]", "You need a fixed range, such as image pixels or a bounded activation"],
                ["Robust", "<span class='inline-formula'>(x − median) / IQR</span>", "Median 0, outlier-resistant", "Heavy outliers that you do not want to remove"],
                ["Max-abs", "<span class='inline-formula'>x / max|x|</span>", "[−1, 1], preserves zeros", "Sparse data, where centring would destroy sparsity"],
                ["L2 normalise", "<span class='inline-formula'>x / ‖x‖₂</span>", "Unit-length rows", "Text vectors and embeddings, where direction matters, not magnitude"],
              ],
            },
          },
          {
            branch: {
              q: "Does my model need feature scaling?",
              arms: [
                { when: "Distance-based", then: "Yes, essential", note: "KNN, K-Means, SVM, DBSCAN, PCA. Without it the largest-range feature dominates the metric.", accent: "red" },
                { when: "Gradient-based", then: "Yes, strongly recommended", note: "Linear and logistic regression, neural networks. Unscaled features make the loss surface elongated and slow to converge.", accent: "red" },
                { when: "Regularized linear", then: "Yes, required for correctness", note: "L1 and L2 penalties are scale-dependent, so unscaled features get penalised unequally.", accent: "red" },
                { when: "Tree-based", then: "No, it changes nothing", note: "Decision trees, random forests, XGBoost and LightGBM split on order, which monotone rescaling preserves.", accent: "green" },
              ],
            },
          },
          {
            callout: {
              tone: "red",
              title: "Sparse data needs care",
              text: "StandardScaler subtracts the mean, which turns every structural zero into a non-zero and can explode a sparse matrix into dense memory. Use MaxAbsScaler, or StandardScaler(with_mean=False).",
            },
          },
        ],
      },
      {
        title: "Encoding categorical features",
        slug: "encoding",
        body: [
          { p: "Models consume numbers. The encoding you choose depends almost entirely on <b>cardinality</b> and on whether the categories have a meaningful order." },
          {
            table: {
              headers: ["Encoding", "How it works", "Best for", "Watch out for"],
              rows: [
                ["One-hot", "One binary column per level", "Low cardinality (under ~15), no order", "Explodes dimensionality; drop one column for linear models to avoid collinearity"],
                ["Ordinal", "Map levels to 0, 1, 2, …", "Genuinely ordered levels: small &lt; medium &lt; large", "Never use on unordered categories with linear models: it invents a false ordering"],
                ["Label", "Same as ordinal, applied to the target", "Encoding y, not X", "Frequently misused on features; that is what creates the false ordering"],
                ["Target / mean", "Replace each level with its mean target", "High cardinality: zip code, product ID", "Leaks badly unless computed out-of-fold with smoothing"],
                ["Frequency / count", "Replace each level with how often it occurs", "High cardinality, tree models", "Two unrelated levels with equal counts collide"],
                ["Binary / hashing", "Encode into log₂(k) bits, or hash into fixed buckets", "Very high cardinality, streaming", "Hash collisions merge unrelated levels"],
                ["Embeddings", "Learned dense vector per level", "Very high cardinality in a neural network", "Needs a lot of data per level"],
              ],
            },
          },
          {
            callout: {
              tone: "red",
              title: "Target encoding leaks unless you are careful",
              text: "Computing a level's mean target using the row you are about to predict tells the model the answer. Always compute it out-of-fold, and smooth rare levels toward the global mean or they memorise single observations.",
            },
          },
          {
            code: `from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# handle_unknown="ignore" is essential: production will show you levels
# that never appeared during training.
categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
])
numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])

pre = ColumnTransformer([
    ("cat", categorical, cat_cols),
    ("num", numeric, num_cols),
], remainder="drop")

# Target encoding must be out-of-fold or it leaks.
from sklearn.preprocessing import TargetEncoder
te = TargetEncoder(smooth="auto", cv=5)   # internally cross-fitted`,
          },
        ],
      },
      {
        title: "Imputation",
        slug: "imputation",
        body: [
          { p: "Filling gaps is a modelling decision, not a cleanup chore. Every strategy makes an assumption, and the wrong one distorts the distribution you are trying to learn." },
          {
            table: {
              headers: ["Strategy", "What it assumes", "Cost"],
              rows: [
                ["Drop rows", "Missingness is random and rare", "Loses data; biased if missingness relates to the target"],
                ["Drop column", "The column carries little signal", "Loses a feature that may be informative through its absence"],
                ["Mean", "Roughly symmetric distribution", "Shrinks variance; badly distorted by skew and outliers"],
                ["Median", "Skew or outliers present", "Shrinks variance, but robustly. The sane default for numeric"],
                ["Mode / 'Unknown'", "Categorical column", "'Unknown' as an explicit level is usually better than the mode"],
                ["Forward / backward fill", "Time series with persistence", "Only valid for ordered data; never for cross-sectional rows"],
                ["KNN", "Similar rows have similar values", "Expensive; needs scaled features to work at all"],
                ["Iterative / MICE", "Features predict one another", "Most accurate under MAR; slowest and can overfit"],
              ],
            },
          },
          {
            callout: {
              tone: "green",
              title: "Add the indicator, almost always",
              text: "add_indicator=True appends a binary was-missing column. It costs one feature, and it lets the model use missingness as signal when it turns out to be informative. It is the single cheapest safeguard in preprocessing.",
            },
          },
          { p: "See the <b>Exploratory Data Analysis</b> guide for how to diagnose whether missingness is MCAR, MAR or MNAR, which is what should drive this choice." },
        ],
      },
      {
        title: "Transforming distributions",
        slug: "transforms",
        body: [
          { p: "Linear models assume roughly linear, homoscedastic relationships. When a feature is heavily skewed, a transform often buys more accuracy than any amount of hyperparameter tuning. Tree models are indifferent to all of this." },
          {
            steps: [
              { title: "Log transform", formula: "x' = log(1 + x)", note: "For right-skewed positive data such as income, counts and durations. The 1 + handles zeros. Turns multiplicative relationships into additive ones." },
              { title: "Box-Cox", formula: "x' = (x^λ − 1) / λ,  λ chosen by maximum likelihood", note: "Finds the best power transform automatically. Requires strictly positive values." },
              { title: "Yeo-Johnson", formula: "a Box-Cox variant defined for all reals", note: "Same idea but handles zero and negative values, so it is the safer default." },
              { title: "Quantile transform", formula: "map to a uniform or normal distribution by rank", note: "Forces any distribution into the target shape. Very robust, but non-linear and it discards the original spacing." },
              { title: "Binning", formula: "cut a continuous feature into ordered buckets", note: "Lets a linear model express a non-monotonic relationship. Loses within-bin resolution." },
            ],
          },
          {
            callout: {
              tone: "red",
              title: "Transform the target too, but invert your predictions",
              text: "If you train on log(y), the model predicts log-space values. Exponentiating back gives the median, not the mean, so a naive inverse under-predicts the average. Use TransformedTargetRegressor so the inversion is handled for you.",
            },
          },
        ],
      },
      {
        title: "Splitting data correctly",
        slug: "splitting",
        body: [
          { p: "The split defines what your evaluation actually means. Getting it wrong produces confident numbers that do not survive deployment." },
          {
            branch: {
              q: "How should I split this dataset?",
              arms: [
                { when: "Plain tabular, balanced", then: "Random split", note: "train_test_split with a fixed random_state. 70/15/15 or 80/20 with cross-validation.", accent: "green" },
                { when: "Imbalanced classes", then: "Stratified split", note: "Preserves the class ratio in every fold, otherwise a rare class may vanish from a fold entirely.", accent: "green" },
                { when: "Time matters", then: "Chronological split", note: "Train on the past, test on the future. A random split lets the model see the future and is the classic leak.", accent: "red" },
                { when: "Grouped rows", then: "Group-aware split", note: "Multiple rows per patient, user or session must stay together, or the same entity appears on both sides.", accent: "red" },
              ],
            },
          },
          {
            code: `from sklearn.model_selection import (
    train_test_split, StratifiedKFold, TimeSeriesSplit, GroupKFold)

# Imbalanced classification: stratify so every fold keeps the class ratio.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Time series: never shuffle. Each fold trains on the past only.
cv = TimeSeriesSplit(n_splits=5)

# Repeated measures: keep every row for a subject on the same side.
cv = GroupKFold(n_splits=5)     # pass groups=patient_id to cross_val_score`,
          },
          {
            callout: {
              tone: "green",
              title: "The pipeline is what makes cross-validation honest",
              text: "When preprocessing sits inside a Pipeline, scikit-learn refits it on each fold's training portion only. Transform the data by hand beforehand and every fold sees statistics from every other fold, which quietly inflates your score.",
            },
          },
        ],
      },
      {
        title: "Class imbalance",
        slug: "imbalance",
        body: [
          { p: "When one class is rare, a model that always predicts the majority scores well on accuracy and is useless. Fix the metric first, then consider changing the data." },
          {
            table: {
              headers: ["Approach", "How", "Trade-off"],
              rows: [
                ["Change the metric", "PR-AUC, F1, recall at fixed precision", "Costs nothing and is always correct. Do this first"],
                ["Class weights", "class_weight='balanced', or scale_pos_weight", "No data changes, no synthetic rows. Usually the best next step"],
                ["Threshold tuning", "Move the decision cutoff off 0.5", "Free, and directly targets the precision/recall trade-off you want"],
                ["Random undersampling", "Discard majority-class rows", "Fast, but throws away real information"],
                ["Random oversampling", "Duplicate minority rows", "No new information; overfits duplicated rows"],
                ["SMOTE", "Synthesise minority points between neighbours", "Can help, but invents data and may blur the boundary"],
              ],
            },
          },
          {
            callout: {
              tone: "red",
              title: "Resample inside the fold, never before splitting",
              text: "Oversampling before the split copies minority rows into both train and test, so the model is scored on rows it memorised. Use imblearn's Pipeline, which applies resampling to the training fold only.",
            },
          },
        ],
      },
      {
        title: "Common preprocessing mistakes",
        slug: "mistakes",
        body: [
          {
            table: {
              headers: ["Mistake", "Consequence", "Fix"],
              rows: [
                ["Scaling before splitting", "Test statistics leak into training", "Split first, fit the scaler on train only"],
                ["Ordinal-encoding unordered categories", "Invents a false ordering a linear model believes", "One-hot, or target encoding for high cardinality"],
                ["Target encoding in-fold", "Severe leakage, near-perfect validation, failure in production", "Cross-fitted target encoding with smoothing"],
                ["Forgetting handle_unknown", "Pipeline crashes on an unseen production category", "handle_unknown='ignore' on the encoder"],
                ["StandardScaler on sparse data", "Densifies the matrix and exhausts memory", "MaxAbsScaler, or with_mean=False"],
                ["Random split on time-series", "The model sees the future", "Chronological split or TimeSeriesSplit"],
                ["Oversampling before the split", "Duplicate rows on both sides", "Resample inside the pipeline, per fold"],
                ["Imputing without an indicator", "Loses informative missingness", "add_indicator=True"],
                ["Transforming y without inverting", "Predictions are in the wrong units", "TransformedTargetRegressor"],
              ],
            },
          },
        ],
      },
      {
        title: "Why-chain",
        slug: "whychain",
        body: [
          {
            qa: [
              { q: "Why must scalers be fitted on training data only?", a: "The scaler's mean and standard deviation are learned parameters. Fitting on all data means those parameters encode the test set's distribution, so your validation score reflects information the model would not have at serving time. The gap only appears in production, when it is expensive." },
              { q: "Why do tree models not need scaling when almost everything else does?", a: "A tree splits on whether a feature exceeds a threshold, which depends only on the ordering of values. Any monotone rescaling preserves that ordering, so the identical tree is learned. Distance-based and gradient-based models combine features numerically, so relative magnitudes change the answer." },
              { q: "Why is one-hot encoding wrong for high-cardinality features?", a: "It creates one column per level, so a zip-code field with 40,000 values adds 40,000 mostly-zero columns. That explodes memory, dilutes each feature's signal, and gives most levels too few observations to estimate anything. Target or frequency encoding keeps it to one column." },
              { q: "Why does target encoding leak so easily?", a: "The encoded value for a row is derived from the target column, including that row's own target. The model effectively receives a compressed copy of the answer. Computing the encoding out-of-fold, so a row's value comes only from other rows, is what makes it legitimate." },
              { q: "Why add a missing-indicator column instead of just imputing?", a: "Imputation replaces the gap with a plausible value and erases the fact that it was ever missing. When missingness is informative, which is common in real data, that fact is often more predictive than the imputed value. The indicator preserves it for one binary column." },
              { q: "Why is a random split wrong for time-series data?", a: "It puts future rows in the training set and past rows in the test set, so the model learns from information that would not exist at prediction time. Validation looks excellent because the task has become interpolation rather than forecasting." },
              { q: "Why resample inside cross-validation rather than beforehand?", a: "Oversampling duplicates or synthesises minority rows. Do it before splitting and copies of the same row land in both training and test, so the model is evaluated on rows it has already seen. The score measures memorisation." },
              { q: "Why does class_weight usually beat SMOTE?", a: "Class weighting reweights the loss so minority errors cost more, using only real data. SMOTE invents synthetic points by interpolating between neighbours, which can place them in regions where the minority class does not actually occur, blurring the boundary. Weighting is simpler and has no fabrication risk." },
            ],
          },
        ],
      },
    ],
  });
})();
