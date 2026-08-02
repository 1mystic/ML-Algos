(() => {
  MLApp.register({
    id: "evaluation",
    name: "Evaluation Metrics",
    category: "Supplementary",
    tagline: "measuring what actually matters",
    kind: "guide",
    description: "Choosing and reading the numbers that judge a model: the confusion matrix and everything derived from it, ranking and calibration measures, regression errors, clustering scores, and the validation schemes that make any of them trustworthy.",
    guide: [
      {
        title: "Pick the metric before you train",
        slug: "why",
        body: [
          { p: "A metric is a statement about which mistakes you care about. Accuracy says a false positive and a false negative are equally bad. Recall says missing a positive is unforgivable. Precision says a false alarm is. <b>These are business decisions, not technical ones</b>, and choosing after seeing results is how people talk themselves into a model that does not work." },
          {
            callout: {
              tone: "red",
              title: "Accuracy is the wrong default",
              text: "On a dataset with 1% positives, predicting 'negative' for everything scores 99% accuracy and catches nothing. Any time classes are imbalanced, accuracy measures the class ratio rather than the model.",
            },
          },
          {
            flow: [
              { label: "Define the cost", note: "FP vs FN" },
              { label: "Pick the metric", note: "before training" },
              { label: "Validate honestly", note: "correct split" },
              { label: "Tune the threshold", note: "not just the model" },
              { label: "Compare to baseline", note: "is it worth it?", accent: "green" },
            ],
            title: "Evaluation, in order",
          },
        ],
      },
      {
        title: "The confusion matrix",
        slug: "confusion",
        body: [
          { p: "Every classification metric is a ratio of these four counts. Learning to read the matrix directly is more useful than memorising the derived names." },
          {
            table: {
              headers: ["", "Predicted positive", "Predicted negative"],
              rows: [
                ["<b>Actually positive</b>", "True Positive (TP)", "False Negative (FN) - a miss, Type II error"],
                ["<b>Actually negative</b>", "False Positive (FP) - a false alarm, Type I error", "True Negative (TN)"],
              ],
            },
          },
          {
            steps: [
              { title: "Accuracy", formula: "(TP + TN) / (TP + TN + FP + FN)", note: "Fraction correct. Only meaningful when classes are roughly balanced and both error types cost the same." },
              { title: "Precision", formula: "TP / (TP + FP)", note: "Of everything flagged positive, how much really was. Optimise when a false alarm is expensive: spam filters, arrests, costly interventions." },
              { title: "Recall (sensitivity, TPR)", formula: "TP / (TP + FN)", note: "Of all real positives, how many you caught. Optimise when a miss is expensive: cancer screening, fraud, safety alerts." },
              { title: "Specificity (TNR)", formula: "TN / (TN + FP)", note: "Of all real negatives, how many you correctly cleared. The x-axis of the ROC curve is 1 − specificity." },
              { title: "F1", formula: "2 · P · R / (P + R)", note: "Harmonic mean of precision and recall, so it is low unless both are decent. The usual single-number summary for imbalanced problems." },
              { title: "Fβ", formula: "(1 + β²) · P · R / (β²·P + R)", note: "Weighted version. β > 1 favours recall, β < 1 favours precision. Use when you can state how many false alarms one miss is worth." },
              { title: "Matthews correlation", formula: "(TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))", note: "Balanced across all four cells and robust to imbalance. Ranges from −1 to 1. Arguably the best single-number classification metric." },
            ],
          },
          {
            branch: {
              q: "Which error hurts more in this problem?",
              arms: [
                { when: "Missing a positive", then: "Optimise recall", note: "Disease screening, fraud detection, predictive maintenance. A follow-up check is cheap; a miss is not.", accent: "red" },
                { when: "A false alarm", then: "Optimise precision", note: "Spam filtering, content takedown, customer-facing alerts. Crying wolf destroys trust in the system.", accent: "red" },
                { when: "Both equally", then: "F1 or MCC", note: "When you genuinely cannot rank the two error types.", accent: "green" },
                { when: "Costs are quantifiable", then: "Expected cost", note: "Assign a currency value to FP and FN and minimise total cost directly. Always the best option when possible.", accent: "green" },
              ],
            },
          },
        ],
      },
      {
        title: "Threshold-free metrics: ROC and PR",
        slug: "curves",
        body: [
          { p: "Precision, recall and F1 all depend on a decision threshold. Curve-based metrics sweep every threshold instead, which measures how well the model <b>ranks</b> examples independently of where you cut." },
          {
            table: {
              headers: ["Metric", "Axes", "Baseline", "Use when"],
              rows: [
                ["ROC-AUC", "TPR against FPR", "0.5 for random", "Classes are roughly balanced; you want a threshold-independent quality measure"],
                ["PR-AUC (average precision)", "Precision against recall", "The positive rate", "Classes are imbalanced. Far more informative than ROC here"],
                ["Log-loss", "Not a curve; penalises confident errors", "Depends on base rate", "You need calibrated probabilities, not just ranking"],
                ["Brier score", "Mean squared probability error", "Depends on base rate", "Calibration quality, decomposable into calibration and refinement"],
              ],
            },
          },
          {
            callout: {
              tone: "red",
              title: "ROC-AUC flatters imbalanced problems",
              text: "The false-positive rate divides by the large number of true negatives, so thousands of false alarms barely move it. On a 1% positive dataset a model can show ROC-AUC of 0.95 and still have precision near 0.1. Report PR-AUC alongside it, always.",
            },
          },
          { p: "An ROC-AUC of 0.80 has a clean interpretation: given one random positive and one random negative, the model scores the positive higher 80% of the time." },
        ],
      },
      {
        title: "Calibration",
        slug: "calibration",
        body: [
          { p: "A model is calibrated when its stated confidence matches reality: among predictions of 0.7, about 70% should be positive. Ranking metrics say nothing about this, and any decision involving expected value needs it." },
          {
            table: {
              headers: ["Model", "Typical calibration", "Fix"],
              rows: [
                ["Logistic regression", "Well calibrated by construction", "Usually nothing needed"],
                ["Naive Bayes", "Badly overconfident, pushed to 0 and 1", "Isotonic regression"],
                ["SVM", "No probabilities at all, only a signed distance", "Platt scaling"],
                ["Random forest", "Under-confident near 0 and 1", "Isotonic or Platt"],
                ["Gradient boosting", "Reasonable, drifts with many rounds", "Platt scaling"],
                ["Deep networks", "Strongly overconfident", "Temperature scaling, label smoothing"],
              ],
            },
          },
          {
            code: `from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

# Reliability diagram: perfectly calibrated sits on the diagonal.
frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)

# Platt (sigmoid) for small data; isotonic when you have plenty.
calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)

print("Brier:", brier_score_loss(y_test, y_prob))
print("Log-loss:", log_loss(y_test, y_prob))`,
          },
        ],
      },
      {
        title: "Regression metrics",
        slug: "regression",
        body: [
          {
            steps: [
              { title: "MAE", formula: "(1/m) Σ |y − ŷ|", note: "Mean absolute error, in the target's own units. Robust to outliers because errors are not squared. Optimising it predicts the conditional median." },
              { title: "MSE / RMSE", formula: "(1/m) Σ (y − ŷ)²,  RMSE = √MSE", note: "Squaring punishes large errors disproportionately. RMSE restores the original units. Optimising it predicts the conditional mean." },
              { title: "R²", formula: "1 − RSS / TSS", note: "Fraction of variance explained. 0 means no better than predicting the mean; negative means worse than that. Never decreases when features are added." },
              { title: "Adjusted R²", formula: "1 − (1−R²)(m−1)/(m−p−1)", note: "Penalises extra features, so it can fall when a useless one is added. Use it for model comparison." },
              { title: "MAPE", formula: "(100/m) Σ |y − ŷ| / |y|", note: "Percentage error, so it is scale-free and readable by non-specialists. Undefined at y = 0 and biased toward under-prediction." },
              { title: "Huber loss", formula: "quadratic within δ, linear beyond", note: "A compromise between MSE and MAE: differentiable near zero, robust in the tails." },
              { title: "Quantile (pinball) loss", formula: "asymmetric penalty at quantile τ", note: "For prediction intervals, and when over- and under-prediction cost differently." },
            ],
          },
          {
            branch: {
              q: "Which regression metric fits the problem?",
              arms: [
                { when: "Outliers are errors", then: "MAE or Huber", note: "Squared error lets a handful of bad rows dominate the fit.", accent: "green" },
                { when: "Large errors are genuinely worse", then: "RMSE", note: "Being wrong by 100 is more than ten times worse than being wrong by 10.", accent: "green" },
                { when: "Stakeholders want a percentage", then: "MAPE or sMAPE", note: "Readable, but never use it when the target can be zero or near zero.", accent: "red" },
                { when: "You need intervals", then: "Quantile loss", note: "Train separate models at, say, the 10th, 50th and 90th percentiles.", accent: "green" },
              ],
            },
          },
        ],
      },
      {
        title: "Clustering and ranking metrics",
        slug: "other",
        body: [
          {
            cards: [
              { name: "Clustering, no labels", list: ["<b>Silhouette</b>: (b − a) / max(a, b) per point. Near 1 is well separated, near 0 is on a boundary", "<b>Davies-Bouldin</b>: lower is better; average similarity to the nearest other cluster", "<b>Calinski-Harabasz</b>: higher is better; between- over within-cluster dispersion", "<b>Inertia</b>: only for elbow plots, never for choosing k directly"] },
              { name: "Clustering, with labels", list: ["<b>Adjusted Rand Index</b>: agreement corrected for chance, 0 is random, 1 is perfect", "<b>Normalized Mutual Information</b>: shared information, scaled to [0, 1]", "<b>Homogeneity / completeness / V-measure</b>: the clustering analogue of precision, recall and F1"] },
              { name: "Ranking and recommendation", list: ["<b>Precision@k / Recall@k</b>: quality of the top k results", "<b>MAP</b>: mean average precision across queries", "<b>NDCG</b>: discounted gain, rewards putting the best items highest", "<b>MRR</b>: mean reciprocal rank of the first relevant item"] },
              { name: "Always compare against", list: ["<b>Majority class</b> for classification", "<b>Mean or median</b> for regression", "<b>Last observed value</b> for time series, which is surprisingly hard to beat", "<b>The existing system</b>, if one is already in production"] },
            ],
          },
        ],
      },
      {
        title: "Validation strategy",
        slug: "validation",
        body: [
          { p: "The metric is only as trustworthy as the split it is computed on. A leaked split makes every number meaningless, however carefully chosen." },
          {
            lanes: [
              { name: "Hold-out", note: "One split, fast", items: ["Large datasets", "Quick iteration", "High variance on small data"], accent: "blue" },
              { name: "K-fold CV", note: "Every row used for both", items: ["The default for small and medium data", "5 or 10 folds is standard", "K times the training cost"], accent: "green" },
              { name: "Stratified K-fold", note: "Preserves class ratios", items: ["Any imbalanced classification", "Prevents empty classes in a fold", "The default for classification"], accent: "green" },
              { name: "Time series split", note: "Past predicts future", items: ["Any temporal data", "Expanding or rolling window", "Never shuffle"], accent: "amber" },
              { name: "Group K-fold", note: "Keeps entities together", items: ["Repeated measures per subject", "Multiple sessions per user", "Prevents identity leakage"], accent: "plum" },
              { name: "Nested CV", note: "Unbiased estimate", items: ["Inner loop tunes, outer loop scores", "The honest way to report after tuning", "Expensive"], accent: "red" },
            ],
          },
          {
            callout: {
              tone: "red",
              title: "Tuning on the test set is the most common self-deception",
              text: "Every time you check the test score and change something, you fit to it a little. Keep a validation set for decisions and touch the test set once, at the end. If you tuned extensively, nested cross-validation is the honest way to report.",
            },
          },
          {
            code: `from sklearn.model_selection import cross_validate, StratifiedKFold

# Track several metrics at once: a single number always hides something.
scores = cross_validate(
    pipeline, X_train, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring=["accuracy", "precision", "recall", "f1",
             "roc_auc", "average_precision", "neg_log_loss"],
    return_train_score=True,     # the train/val gap diagnoses over/underfitting
)

# Report spread, not just the mean: a low mean with high variance is unstable.
for k, v in scores.items():
    if k.startswith("test_"):
        print(f"{k:24s} {v.mean():.3f} +/- {v.std():.3f}")`,
          },
        ],
      },
      {
        title: "Common evaluation mistakes",
        slug: "mistakes",
        body: [
          {
            table: {
              headers: ["Mistake", "What goes wrong", "Fix"],
              rows: [
                ["Accuracy on imbalanced data", "Measures the class ratio, not the model", "PR-AUC, F1, MCC, or expected cost"],
                ["ROC-AUC on rare positives", "Hides thousands of false alarms", "Report PR-AUC alongside it"],
                ["Leaving the threshold at 0.5", "An arbitrary cut that rarely matches the cost profile", "Tune on the PR curve for your operating point"],
                ["Tuning against the test set", "Optimistic estimate that fails in production", "Separate validation set, or nested CV"],
                ["Reporting only the mean CV score", "Hides an unstable model", "Report the standard deviation too"],
                ["No baseline", "No idea whether the model adds value", "Compare to majority class, mean, or last value"],
                ["Comparing across different splits", "Differences are noise, not signal", "Fix the seed and the fold structure"],
                ["Ignoring calibration", "Probabilities cannot support decisions", "Reliability diagram, Brier score, calibrate"],
                ["Trusting a single metric", "Every metric is blind to something", "Track a small panel of complementary ones"],
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
              { q: "Why is accuracy misleading on imbalanced data?", a: "It is dominated by the majority class. With 1% positives, always predicting negative gives 99% accuracy while catching nothing. The number describes the class distribution rather than the model's ability, and it improves if you make the imbalance worse." },
              { q: "Why prefer PR-AUC to ROC-AUC when positives are rare?", a: "The false-positive rate has true negatives in its denominator. When negatives vastly outnumber positives, even thousands of false alarms barely change FPR, so the ROC curve looks excellent. Precision divides by predicted positives, so those false alarms hit it directly and the PR curve tells the truth." },
              { q: "Why is F1 a harmonic rather than arithmetic mean?", a: "The harmonic mean is dominated by the smaller value. A model with precision 1.0 and recall 0.0 has arithmetic mean 0.5, which looks acceptable, but F1 of 0. That is the correct verdict: a model that catches nothing is not half-good." },
              { q: "Why does optimising MSE predict the mean while MAE predicts the median?", a: "The constant minimising summed squared deviation is the mean; the constant minimising summed absolute deviation is the median. Since each leaf or region effectively predicts a constant, the loss you choose decides which central tendency you get, and therefore how outliers pull the prediction." },
              { q: "Why can R² be negative?", a: "R² is 1 − RSS/TSS, comparing your model against predicting the mean. If your model's squared error exceeds the variance of the target, the ratio exceeds 1 and R² goes negative. It means you would be better off ignoring the features entirely." },
              { q: "Why does a well-ranking model still need calibration?", a: "Ranking metrics only care about ordering. A model can order every example perfectly while stating 0.99 for cases that are actually 0.6. Any decision using expected value, such as pricing or triage thresholds, needs the number to mean what it says, not just to sort correctly." },
              { q: "Why is a threshold of 0.5 usually the wrong choice?", a: "It is the neutral point only if false positives and false negatives cost the same and the classes are balanced. Neither is typically true. The threshold is a free parameter you should set from the cost of each error, after training, using the PR curve." },
              { q: "Why does nested cross-validation matter after tuning?", a: "If you select hyperparameters using cross-validation and then report that same score, it is optimistic: the selection consumed the validation information. Nested CV puts tuning in an inner loop and scoring in an outer loop, so the reported number comes from data the tuning never touched." },
              { q: "Why report the standard deviation of cross-validation scores?", a: "A mean of 0.85 from folds ranging 0.60 to 0.98 is a very different model from one where every fold scores 0.84 to 0.86. High variance means the result depends heavily on which rows land where, which usually signals too little data or an unstable model." },
            ],
          },
        ],
      },
    ],
  });
})();
