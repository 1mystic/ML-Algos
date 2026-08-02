(() => {
  MLApp.register({
    id: "ml-pipeline",
    name: "ML Pipeline & Deployment",
    category: "Supplementary",
    tagline: "problem framing to production",
    kind: "guide",
    description: "The full lifecycle around the model: framing the problem, building a pipeline that trains and serves identically, choosing a deployment pattern, and monitoring for the drift that eventually degrades every model in production.",
    guide: [
      {
        title: "The lifecycle",
        slug: "lifecycle",
        body: [
          { p: "Training a model is a small fraction of the work. Most projects fail on the parts either side of it: a problem framed so that no metric maps to business value, or a model that cannot be served, monitored, or retrained." },
          {
            lanes: [
              { name: "Frame", note: "Define success before touching data", items: ["What decision does this change?", "Classification, regression, or ranking?", "What is the cost of each error type?", "What is the baseline to beat?"], accent: "plum" },
              { name: "Data", note: "Collect, validate, label", items: ["Sources and freshness", "Schema and quality contracts", "Labelling process and its noise", "Split strategy decided now"], accent: "blue" },
              { name: "Explore", note: "See the data", items: ["Quality audit", "Distributions and relationships", "Leakage hunt", "Preprocessing plan"], accent: "blue" },
              { name: "Build", note: "Preprocess and train", items: ["Everything inside one pipeline", "Baseline first, then complexity", "Cross-validate honestly", "Tune with a held-out set"], accent: "green" },
              { name: "Evaluate", note: "Decide if it ships", items: ["Metric chosen up front", "Slice performance by segment", "Calibration and fairness checks", "Compare against the baseline"], accent: "amber" },
              { name: "Deploy", note: "Serve predictions", items: ["Batch, real-time, or streaming", "Version the model and the data", "Shadow, then canary", "Rollback plan"], accent: "red" },
              { name: "Monitor", note: "Catch decay", items: ["Data and prediction drift", "Live metrics as labels arrive", "Latency and error rates", "Retraining trigger"], accent: "red" },
            ],
          },
          {
            callout: {
              tone: "green",
              title: "Always ship the boring baseline first",
              text: "A logistic regression or a rules engine in production, being monitored, teaches you more than a gradient-boosted ensemble in a notebook. It establishes the plumbing, the metric, and the number every later model must beat.",
            },
          },
        ],
      },
      {
        title: "Framing the problem",
        slug: "framing",
        body: [
          { p: "The most expensive mistakes happen here, before any code. A technically excellent model solving the wrong problem is a total loss." },
          {
            table: {
              headers: ["Question", "Why it decides everything downstream"],
              rows: [
                ["What action follows a prediction?", "If no decision changes, the model has no value regardless of accuracy"],
                ["Is this really supervised?", "No labels means unsupervised, or an expensive labelling project you must plan for"],
                ["What does a false positive cost, in currency?", "Determines the metric and the operating threshold"],
                ["How fresh must predictions be?", "Milliseconds means real-time serving; daily means batch, which is far simpler"],
                ["How much data exists per class?", "Decides whether deep learning is even viable"],
                ["Will the features exist at prediction time?", "The single most common source of leakage in production systems"],
                ["Who is accountable if it is wrong?", "Drives interpretability, auditability and fairness requirements"],
              ],
            },
          },
          {
            callout: {
              tone: "red",
              title: "The feature availability trap",
              text: "A column present in your historical table may only be populated hours after the event you are predicting. It will look predictive in training and be null at serving time. For every feature, ask when its value actually becomes known.",
            },
          },
        ],
      },
      {
        title: "The training pipeline",
        slug: "pipeline",
        body: [
          { p: "A pipeline is a single object holding every transformation plus the estimator. It exists so that the exact same steps, fitted with the exact same parameters, run at training and at serving. <b>Transform data by hand and the two will eventually diverge.</b>" },
          {
            flow: [
              { label: "Raw input", note: "same schema both sides" },
              { label: "ColumnTransformer", note: "per-type preprocessing" },
              { label: "Feature engineering", note: "derived columns" },
              { label: "Estimator", note: "the model" },
              { label: "One artefact", note: "fit once, serve anywhere", accent: "green" },
            ],
            title: "One object, both paths",
          },
          {
            code: `from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pre = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Production will show you categories training never saw.
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ]), cat_cols),
])

pipe = Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier())])

# Because preprocessing is inside the pipeline, every CV fold refits it on
# that fold's training rows only. This is what makes the score honest.
search = GridSearchCV(
    pipe,
    {"model__learning_rate": [0.03, 0.1], "model__max_depth": [3, 6, None]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="average_precision", n_jobs=-1,
).fit(X_train, y_train)

import joblib
joblib.dump(search.best_estimator_, "model-v1.joblib")   # one artefact`,
          },
          {
            callout: {
              tone: "green",
              title: "The pipeline is a correctness tool, not a convenience",
              text: "Its real value is that cross-validation refits preprocessing per fold. Scale by hand before cross-validating and every fold sees statistics derived from every other fold, which silently inflates your score.",
            },
          },
        ],
      },
      {
        title: "Deployment patterns",
        slug: "deployment",
        body: [
          {
            branch: {
              q: "How should this model be served?",
              arms: [
                { when: "Predictions needed daily or hourly", then: "Batch scoring", note: "A scheduled job writes predictions to a table. Simplest to build, test and monitor. Choose this unless you cannot.", accent: "green" },
                { when: "Needed within a request", then: "Real-time API", note: "Model behind REST or gRPC. Adds latency budgets, autoscaling and warm-up concerns.", accent: "amber" },
                { when: "Continuous event flow", then: "Streaming", note: "Scoring inside Kafka or Flink. Highest complexity; justified by genuine event-driven needs.", accent: "red" },
                { when: "Offline or private", then: "Edge / on-device", note: "Export to ONNX, TFLite or Core ML. No network, but updates are hard and hardware is constrained.", accent: "plum" },
              ],
            },
          },
          {
            table: {
              headers: ["Rollout strategy", "How it works", "Protects against"],
              rows: [
                ["Shadow", "New model scores live traffic; predictions logged, not used", "Discovering serving bugs and skew with zero user risk"],
                ["Canary", "Route a small percentage of traffic to the new model", "Limiting blast radius of a regression"],
                ["A/B test", "Split traffic and compare business metrics", "Confirming the model actually improves the outcome, not just the offline metric"],
                ["Blue-green", "Two identical environments, switch over atomically", "Instant rollback"],
                ["Champion-challenger", "New model continuously scored against the incumbent", "Ongoing, automatic model selection"],
              ],
            },
          },
          {
            cards: [
              { name: "What to version", list: ["Model weights and the full pipeline artefact", "Training data snapshot or a query hash", "Feature definitions and their code", "Hyperparameters and random seeds", "Library versions: a pickle is not portable across them"] },
              { name: "Serving concerns", list: ["Latency budget, at p99 and not just the mean", "Throughput and autoscaling behaviour", "Cold-start and model load time", "Graceful degradation when the model fails", "Input validation, so bad requests do not become bad predictions"] },
            ],
          },
          {
            callout: {
              tone: "red",
              title: "Training-serving skew",
              text: "The most common production failure: preprocessing in the training notebook differs subtly from preprocessing in the serving code. Shipping one pipeline artefact used by both paths eliminates most of it. A feature store handles the rest when features are computed upstream.",
            },
          },
        ],
      },
      {
        title: "Monitoring and drift",
        slug: "monitoring",
        body: [
          { p: "Every deployed model degrades. The world changes, upstream systems change, and user behaviour adapts to the model itself. Monitoring is how you find out before your users do." },
          {
            steps: [
              { title: "Data drift (covariate shift)", formula: "P(X) changes, P(y | X) stays the same", note: "Input distribution moves: a new customer segment, a changed sensor, a new app version. Detect with PSI, KL divergence, or a Kolmogorov-Smirnov test per feature." },
              { title: "Concept drift", formula: "P(y | X) changes", note: "The relationship itself shifts: fraud tactics evolve, preferences change. Only detectable once labels arrive, which is why label latency matters so much." },
              { title: "Label drift", formula: "P(y) changes", note: "The base rate moves. Often breaks calibration even when the ranking is still fine." },
              { title: "Prediction drift", formula: "the output distribution moves", note: "The earliest available warning, because it needs no labels. A sudden shift in mean predicted probability is worth an alert on its own." },
              { title: "Training-serving skew", formula: "features differ between the two paths", note: "Not drift at all, but a bug. Log serving features and periodically compare their distribution against training." },
            ],
          },
          {
            table: {
              headers: ["Signal", "What it tells you", "Available when"],
              rows: [
                ["Feature distributions (PSI)", "Inputs have shifted", "Immediately, no labels needed"],
                ["Prediction distribution", "Model behaviour has shifted", "Immediately, no labels needed"],
                ["Live metric (PR-AUC, MAE)", "Actual performance decay", "Only once labels arrive"],
                ["Segment-level metrics", "Decay hidden inside an acceptable average", "Once labels arrive"],
                ["Latency, error rate, null rate", "Infrastructure and data-quality faults", "Immediately"],
              ],
            },
          },
          {
            callout: {
              tone: "green",
              title: "Monitor slices, not just the aggregate",
              text: "Overall accuracy can hold steady while performance collapses for a specific region, device or customer tier. Track your metric broken down by the segments that matter, because the aggregate hides exactly the failures people complain about.",
            },
          },
          {
            branch: {
              q: "When should the model be retrained?",
              arms: [
                { when: "Scheduled", then: "Fixed cadence", note: "Weekly or monthly. Simple and predictable; may retrain when nothing changed, or too late when it did.", accent: "green" },
                { when: "Triggered", then: "On drift or metric decay", note: "Retrain when PSI or the live metric crosses a threshold. Efficient, but needs reliable monitoring first.", accent: "green" },
                { when: "Continuous", then: "Online learning", note: "Model updates as data arrives. Powerful and genuinely risky: a bad hour of data corrupts the model.", accent: "red" },
                { when: "Manual", then: "Human in the loop", note: "Appropriate for high-stakes or regulated domains where every model version needs sign-off.", accent: "amber" },
              ],
            },
          },
        ],
      },
      {
        title: "Production checklist",
        slug: "checklist",
        body: [
          {
            cards: [
              { name: "Before training", list: ["Success metric agreed with stakeholders", "Baseline defined", "Split strategy matches the data (time, groups, stratification)", "Every feature confirmed available at prediction time"] },
              { name: "Before deploying", list: ["Preprocessing and model in one versioned artefact", "Test score from data used exactly once", "Performance checked per segment, not just overall", "Calibration checked if probabilities drive decisions", "Rollback path tested"] },
              { name: "After deploying", list: ["Feature and prediction distributions monitored", "Live metric tracked as labels arrive", "Alerting on drift and on infrastructure faults", "Retraining trigger defined and owned", "Prediction logs retained for debugging and future training"] },
              { name: "Frequent failure modes", list: ["Training-serving skew from duplicated preprocessing code", "A feature that is null at serving time", "Silent upstream schema change", "Unseen categorical level crashing the encoder", "Model file unpicklable after a library upgrade", "Nobody owns the model after the project ends"] },
            ],
          },
        ],
      },
      {
        title: "Why-chain",
        slug: "whychain",
        body: [
          {
            qa: [
              { q: "Why put preprocessing inside the pipeline instead of doing it first?", a: "Two reasons. Correctness: cross-validation refits the pipeline on each fold's training rows only, so no fold sees statistics from another. And parity: the same fitted object runs at serving time, which eliminates the skew you get from maintaining two copies of the transformation logic." },
              { q: "Why deploy a weak baseline before the good model?", a: "It proves the entire path works: data arrives, predictions are served, metrics are collected, rollback functions. Those are where projects actually fail. It also gives you the number the sophisticated model must beat to justify its complexity." },
              { q: "Why is training-serving skew so common?", a: "Training usually happens in a notebook with pandas, and serving in an application with different code. Any difference in null handling, category ordering, or default values produces subtly different features. The model sees inputs it was never trained on and degrades without any error being raised." },
              { q: "Why monitor prediction drift when you could monitor accuracy?", a: "Accuracy needs labels, and labels often arrive days or months later, if at all. The prediction distribution is available immediately. A sudden shift in the model's output distribution is an early warning you can act on long before the performance metric confirms it." },
              { q: "What is the difference between data drift and concept drift?", a: "Data drift means the inputs moved but the underlying relationship held, so retraining on recent data usually fixes it. Concept drift means the relationship itself changed, so the old labels are actively misleading and you may need new features, not just fresh rows." },
              { q: "Why shadow-deploy before a canary?", a: "Shadow mode runs the new model on real production traffic without anyone consuming its output. It surfaces serving bugs, latency problems and feature skew against genuine inputs at zero user risk. Canary limits damage; shadow avoids it entirely." },
              { q: "Why version the training data and not just the model?", a: "Without the data snapshot you cannot reproduce a model, explain a past prediction, or diagnose whether a regression came from a code change or a data change. In regulated settings you may be required to reproduce any historical decision exactly." },
              { q: "Why is online learning risky in production?", a: "The model updates continuously from live data, so there is no review step. A logging bug, an upstream outage, or an adversarial burst is absorbed directly into the weights, and there is no clean version to roll back to. Scheduled or triggered retraining keeps a human checkpoint." },
              { q: "Why check performance per segment rather than overall?", a: "Aggregate metrics are weighted by volume, so a large stable segment can mask total failure in a small one. That small segment is often a specific region, device or customer tier, and its users are the ones who notice and complain." },
            ],
          },
        ],
      },
    ],
  });
})();
