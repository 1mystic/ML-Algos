(() => {
  MLApp.register({
    id: "eda",
    name: "Exploratory Data Analysis",
    category: "Supplementary",
    tagline: "look before you model",
    kind: "guide",
    description: "The pass you make over a dataset before choosing any algorithm: understand its shape, find what is missing or broken, see how features relate to the target, and decide what preprocessing the data actually needs.",
    guide: [
      {
        title: "Why EDA comes first",
        slug: "why",
        body: [
          { p: "Every modelling decision downstream is a bet on what the data looks like. Scaling assumes you know the ranges, imputation assumes you know why values are missing, and the choice between a linear model and a tree ensemble assumes you know whether relationships are linear. <b>EDA is where you find out instead of guessing.</b>" },
          { p: "It is also the cheapest place to catch problems. A leaked column, a duplicated join, or a target that is 99% one class costs minutes to spot here and days to debug after training." },
          {
            flow: [
              { label: "Raw data", note: "shape, dtypes" },
              { label: "Quality audit", note: "missing, dupes" },
              { label: "Univariate", note: "one feature at a time" },
              { label: "Bivariate", note: "feature vs target" },
              { label: "Preprocessing plan", note: "what to fix", accent: "green" },
            ],
            title: "The pass, end to end",
          },
          {
            callout: {
              tone: "red",
              title: "Split before you look too hard",
              text: "Summary statistics computed over the whole dataset leak test information into your decisions. Do the quality audit on everything, but hold out the test set before you start tuning choices against distributions.",
            },
          },
        ],
      },
      {
        title: "First contact: shape and types",
        slug: "shape",
        body: [
          { p: "Before any plot, establish what you are holding. These five checks take a minute and determine everything that follows." },
          {
            table: {
              headers: ["Check", "What you are looking for", "Red flag"],
              rows: [
                ["Rows and columns", "Is this the volume you expected?", "Far more rows than the source system has records usually means a join fanned out"],
                ["Data types", "Numeric columns stored as numbers, dates as dates", "A numeric column read as text usually means a stray symbol or thousands separator"],
                ["Memory footprint", "Whether the data fits comfortably in RAM", "Downcast int64 to int32 and object to category before optimising anything else"],
                ["Duplicate rows", "Exact and key-level duplicates", "Duplicates inflate your metrics and leak across a train/test split"],
                ["Target distribution", "Class balance, or the shape of a continuous target", "A 99/1 split means accuracy is meaningless from the outset"],
              ],
            },
          },
          {
            code: `import pandas as pd

df.shape                       # rows, columns
df.info(memory_usage="deep")   # dtypes and real memory
df.describe(include="all").T   # numeric + categorical summary
df.duplicated().sum()          # exact duplicates
df.duplicated(subset=key_cols).sum()   # duplicates on the business key
df[target].value_counts(normalize=True)

# A numeric column silently stored as text is the most common surprise.
obj_cols = df.select_dtypes("object").columns
for c in obj_cols:
    coerced = pd.to_numeric(df[c], errors="coerce")
    if coerced.notna().mean() > 0.9:
        print(c, "looks numeric but is stored as text")`,
          },
        ],
      },
      {
        title: "Missing data",
        slug: "missing",
        body: [
          { p: "The percentage missing matters far less than <b>why</b> it is missing. The mechanism decides whether imputation is harmless or actively introduces bias." },
          {
            steps: [
              { title: "MCAR: Missing Completely At Random", formula: "P(missing) is independent of everything", note: "A sensor dropped packets at random. Dropping or imputing these rows is unbiased. This is the rare, easy case." },
              { title: "MAR: Missing At Random", formula: "P(missing) depends on observed features", note: "Income is missing more often for younger respondents, and you observe age. Conditional imputation using the observed features recovers the truth." },
              { title: "MNAR: Missing Not At Random", formula: "P(missing) depends on the unobserved value itself", note: "High earners decline to state income. No imputation fixes this, because the reason for missingness is the thing you are trying to impute. Add a missingness indicator and treat it as signal." },
            ],
          },
          {
            branch: {
              q: "How should I handle a column with missing values?",
              arms: [
                { when: "Missing over ~60%", then: "Usually drop the column", note: "Unless the missingness itself is predictive, in which case keep only a binary indicator.", accent: "red" },
                { when: "Missing under ~5%, MCAR", then: "Simple imputation is fine", note: "Median for numeric, mode or an explicit 'Unknown' level for categorical." },
                { when: "MAR, moderate amount", then: "Model-based imputation", note: "KNN or iterative imputation using the correlated observed features.", accent: "green" },
                { when: "Suspected MNAR", then: "Add a missing-indicator column", note: "Impute anything reasonable, but let the model learn from the fact of absence.", accent: "red" },
              ],
            },
          },
          {
            callout: {
              tone: "green",
              title: "Always consider the indicator column",
              text: "Adding is_missing_x costs one binary feature and protects you when missingness turns out to be informative. Tree models exploit it immediately, and it costs almost nothing when it is noise.",
            },
          },
          {
            code: `# Where is data missing, and does it cluster?
miss = df.isna().mean().sort_values(ascending=False)
print(miss[miss > 0])

# Do two columns go missing together? That suggests a shared upstream cause.
df.isna().corr()

# Is missingness related to the target? If so it is informative, not noise.
for c in df.columns[df.isna().any()]:
    print(c, df.groupby(df[c].isna())[target].mean().to_dict())`,
          },
        ],
      },
      {
        title: "Univariate analysis",
        slug: "univariate",
        body: [
          { p: "One feature at a time: what is its range, its shape, and does it contain values that cannot be real?" },
          {
            cards: [
              { name: "Numeric features", list: ["Histogram for shape: symmetric, skewed, bimodal, or spiky", "Box plot for outliers and the interquartile range", "Skew and kurtosis as numbers, not just eyeballs", "Check for impossible values: negative ages, future dates, zeros that mean 'unknown'"] },
              { name: "Categorical features", list: ["Value counts and cardinality", "Rare levels that may need bucketing into 'Other'", "Near-constant columns that carry no information", "Inconsistent labels: 'UK', 'U.K.', 'United Kingdom'"] },
            ],
          },
          {
            table: {
              headers: ["Distribution shape", "What it suggests", "Typical response"],
              rows: [
                ["Heavily right-skewed", "Multiplicative process: income, counts, durations", "Log or Box-Cox transform for linear models; trees do not care"],
                ["Bimodal", "Two subpopulations mixed together", "Look for the grouping variable and consider modelling separately"],
                ["Spike at zero", "A mixture of 'did not happen' and 'amount'", "Two-part model, or a binary flag plus the amount"],
                ["Near-uniform", "Often an ID or a synthetic column", "Check it is not a row identifier leaking into the model"],
                ["Single dominant value", "Near-zero variance", "Usually safe to drop"],
              ],
            },
          },
          {
            callout: {
              tone: "red",
              title: "An outlier is not automatically an error",
              text: "Before removing anything, decide whether it is a data-entry mistake (a height of 700cm), a genuine rare event (a real high-value transaction), or the exact thing you are trying to predict (fraud). Deleting the third category destroys the problem.",
            },
          },
        ],
      },
      {
        title: "Bivariate and multivariate analysis",
        slug: "bivariate",
        body: [
          { p: "Now relationships: feature against target, and feature against feature. This is where you learn whether a linear model has a chance and which columns are redundant." },
          {
            table: {
              headers: ["Feature type", "Target type", "Tool", "Reads as"],
              rows: [
                ["Numeric", "Numeric", "Scatter plot, Pearson or Spearman", "Direction and linearity of the relationship"],
                ["Numeric", "Categorical", "Box or violin plot per class", "Whether the classes separate on this feature"],
                ["Categorical", "Numeric", "Grouped means with confidence intervals", "Which levels shift the target"],
                ["Categorical", "Categorical", "Crosstab, chi-square, Cramer's V", "Association strength between levels"],
                ["Many numeric", "Any", "Correlation heatmap, VIF", "Redundancy and multicollinearity"],
              ],
            },
          },
          { p: "Use <b>Pearson</b> for linear association and <b>Spearman</b> for monotonic association. A feature with Pearson near zero but Spearman near one is strongly related, just not linearly, which is a direct argument for a tree model or a transform." },
          {
            callout: {
              tone: "red",
              title: "The leakage check",
              text: "A single feature correlating above roughly 0.95 with the target is almost never good news. It usually means the column was computed after the outcome was known, or it is a proxy for the label. Trace where it comes from before celebrating.",
            },
          },
          {
            code: `import numpy as np

# Linear vs monotonic association tells you which model family fits.
pearson  = df.corr(numeric_only=True)[target]
spearman = df.corr(method="spearman", numeric_only=True)[target]
gap = (spearman.abs() - pearson.abs()).sort_values(ascending=False)
print(gap.head())    # large gap = non-linear but monotonic

# Redundant feature pairs, which destabilise linear coefficients.
corr = df.corr(numeric_only=True).abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
print([(a, b) for a in upper.index for b in upper.columns if upper.loc[a, b] > 0.9])

# Suspiciously predictive single features are usually leakage.
print(pearson.abs().sort_values(ascending=False).head())`,
          },
        ],
      },
      {
        title: "Turning findings into a plan",
        slug: "plan",
        body: [
          { p: "EDA is only useful if it ends in decisions. Every observation should map to an action taken in the preprocessing pipeline." },
          {
            table: {
              headers: ["What you found", "What you do about it"],
              rows: [
                ["Feature ranges differ by orders of magnitude", "Scale, if the model is distance- or gradient-based"],
                ["Heavy right skew in a numeric feature", "Log or Box-Cox transform for linear models"],
                ["High-cardinality categorical", "Target or frequency encoding rather than one-hot"],
                ["Two features correlated above 0.9", "Drop one, or combine them, or move to a regularized model"],
                ["Target classes are 95/5", "Stratified splits, class weights, and PR-AUC instead of accuracy"],
                ["Missingness correlates with the target", "Keep an explicit indicator column"],
                ["Clear non-linear relationship", "Tree ensemble, or add spline and interaction terms"],
                ["Timestamps present", "Split chronologically, never randomly"],
              ],
            },
          },
          { p: "See <b>Preprocessing Techniques</b> for how to implement each of these correctly, and <b>ML Pipeline &amp; Deployment</b> for where they belong so they run identically at training and serving time." },
        ],
      },
      {
        title: "Common EDA mistakes",
        slug: "mistakes",
        body: [
          {
            table: {
              headers: ["Mistake", "Why it hurts", "Do instead"],
              rows: [
                ["Exploring before splitting", "Your choices absorb test-set information", "Hold out the test set first; explore the training set"],
                ["Trusting summary statistics alone", "Anscombe's quartet: identical stats, completely different data", "Always plot as well as summarise"],
                ["Dropping every outlier", "Rare events are often the target", "Investigate the cause before removing anything"],
                ["Ignoring the target during EDA", "You learn about the data but not the problem", "Always plot each feature against the target"],
                ["Reading correlation as causation", "Confounders produce strong correlations", "Treat correlation as a modelling hint, not an explanation"],
                ["Skipping duplicate checks", "Duplicates leak across splits and inflate scores", "Check exact and key-level duplicates early"],
                ["Not checking time ordering", "Random splits on time-series data leak the future", "Confirm whether a time column exists before splitting"],
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
              { q: "Why should EDA happen after the train/test split rather than before?", a: "Because decisions made while looking at the full dataset encode test-set information into your pipeline. If you pick an imputation strategy or a transform because it suits the overall distribution, your test score is optimistic. Quality checks on everything are fine; tuning choices must use training data only." },
              { q: "Why does the missingness mechanism matter more than the missing percentage?", a: "Because it determines whether imputation is valid. Under MCAR or MAR, imputation recovers a reasonable estimate. Under MNAR the missingness depends on the unobserved value itself, so any imputation systematically biases the column in the direction you cannot see. A 5% MNAR column can be more dangerous than a 40% MCAR one." },
              { q: "Why compare Pearson and Spearman correlation?", a: "Pearson measures linear association, Spearman measures monotonic association on ranks. A feature with low Pearson and high Spearman is strongly predictive but not linearly so. That single comparison tells you whether a linear model needs a transform or whether you should reach for a tree ensemble." },
              { q: "Why is a very high feature-target correlation suspicious?", a: "Genuine predictive relationships in real data are rarely that clean. A correlation above roughly 0.95 usually means the feature was recorded after the outcome, is a restatement of the label, or is a proxy computed from it. It will look excellent in validation and fail completely in production." },
              { q: "Why plot the data when you already have summary statistics?", a: "Anscombe's quartet is four datasets with identical means, variances, correlations and regression lines that look entirely different when plotted. Summary statistics compress away exactly the structure, clustering and outliers you are looking for." },
              { q: "Why check duplicates before anything else?", a: "Duplicate rows inflate every metric and, worse, can land on both sides of a train/test split so the model is scored on rows it memorised. It looks like excellent generalisation and is the opposite." },
            ],
          },
        ],
      },
    ],
  });
})();
