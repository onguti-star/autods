# Explaining AutoDS Unsupervised Learning

Notes for walking someone through how the unsupervised module works, what each
number means, and how it behaves on large data.

Source of truth: `backend/unsupervised.py`. API routes: `backend/main.py`
(lines ~2377-2470). UI: `frontend/index.html` (~lines 5028-5430).

---

## 1. The one-sentence version

> "You give me a table with no answer column, I clean and standardise the
> numeric columns, and then I look for structure that is already there --
> groups of similar rows, rows that don't fit any group, a flat 2D map of the
> table, or categories that keep showing up together."

## 2. Why it is "unsupervised"

There is no target column. Nothing tells the algorithm the right answer, so it
cannot be *trained* in the usual sense and it cannot be *scored* against
ground truth. It can only be **fitted** (find structure) and then **evaluated**
(judge whether the structure looks reasonable).

This is the single most important thing to convey, because it explains every
design choice that follows:

| | Supervised | Unsupervised |
|---|---|---|
| Input | Features + target | Features only |
| Output | A predictor | A description of the data |
| "Correct?" | Measurable against held-out labels | **Not directly measurable** |
| Validation | Train/test split, cross-validation | Internal quality metrics + human judgement |
| Main risk | Overfitting to labels | **You get a plausible story that is not true** |

The last row is the honest caveat. Clustering will always return clusters if you
ask it to, even from pure noise. That is why the module scores its own output
and why `cluster_best` compares several candidates rather than trusting one.

---

## 3. The pipeline (same for every method)

```
raw DataFrame
   |
   +- 1. select numeric columns          _numeric_matrix()
   |     - drops generated columns (cluster_label) so re-running is idempotent
   |     - requires >= 2 numeric columns
   |
   +- 2. median-impute missing values     fillna(X.median())
   |     - median, not mean: a few extreme values must not drag the fill
   |
   +- 3. standardise each column          StandardScaler()
   |     - z = (x - mean) / std  ->  mean 0, std 1
   |     - makes "salary" and "age" equally weighted in a distance
   |
   +- 4. fit the algorithm (KMeans / DBSCAN / ...) on a bounded row sample
   |
   +- 5. score the result                _clustering_metrics()
   |
   +- 6. build a capped chart payload    _scatter_visualization()
```

### Why step 3 is not optional

Every distance-based method here compares rows by Euclidean distance, and that
distance is a sum over columns. Without scaling, one column dominates purely by
being numerically large:

```
AnnualIncome = 85,000      distance contribution ~ 85,000^2
Age          = 41          distance contribution ~     41^2
```

Two customers with identical purchasing behaviour but different incomes would
land in different clusters, purely because income carries more digits. After
`StandardScaler` both columns contribute on a 0-1-ish scale, so distance
reflects *behaviour* rather than *units*.

The UI states this in the hint text above the Unsupervised section, and
`preprocessing_summary()` returns the same description for the report.

### Why the `cluster_label` column is excluded

`_numeric_matrix()` drops `cluster_label` from the feature list. Without this,
running clustering twice would feed the first run's output back in as an input
feature, so the second run would be clustering on top of the first run's
clusters. The result would still "work" but would be meaningless and would drift
further on each click.


---

## 4. The four task families

### 4.1 Clustering -- "group similar rows"

**K-Means** (`cluster_kmeans`)
- You choose K up front. It picks K random starting points, assigns every row
  to the nearest one, moves each point to the mean of its rows, repeats.
- **Inertia** = total squared distance of each row to its own centre. Lower is
  more compact. It *always* falls as K rises, so it is useless on its own --
  that is what motivates the elbow below.
- Assumes round, similar-sized clusters. Sensitive to the starting points, so
  the code sets `random_state=42` and `n_init=10` (10 restarts, keep the best).

**DBSCAN** (`cluster_dbscan`)
- You choose `eps` (neighbourhood radius) and `min_samples` (how many
  neighbours make a "core point").
- Two things K-Means cannot do: it **finds K itself**, and it labels points in
  low-density regions as **noise** (`-1`), which makes it double as an outlier
  detector.
- `eps` is the hard one to tune. Too small and everything is noise; too large
  and everything collapses into one blob. That is why `cluster_best` sweeps
  `eps` over `(0.3, 0.5, 0.8, 1.2, 1.8, 2.5)`.

**Hierarchical / Agglomerative** (`cluster_hierarchical`)
- Builds a tree: every row starts alone, and the closest pairs merge repeatedly
  until K groups remain. `linkage="ward"` picks merges that keep group variance
  low.
- Best feature: you can look at the tree (a dendrogram) and *see* the natural
  cut. Also gives a deterministic answer -- no random restarts.
- Cost: it needs an n x n distance matrix, so memory grows with the **square**
  of row count. This is the most fragile method on large data (see section 6).

**Choosing K -- `suggest_clusters`**

Four independent criteria, then a vote:

| Criterion | Best value | Intuition |
|---|---|---|
| Inertia (elbow) | the bend | Where adding a cluster stops helping much |
| Silhouette | highest | Points well inside their own group, far from others |
| Davies-Bouldin | lowest | Groups don't overlap much; worst-case overlap is small |
| Calinski-Harabasz | highest | Separation relative to spread |

The code takes the **most common K across all four** rather than trusting one
metric, and breaks ties on silhouette. The reasoning to explain: each metric
fails differently, so agreement between them is evidence. If all four disagree,
there is probably no strong structure and the K is a judgement call, not a fact.


### 4.2 Anomaly detection -- "find rows that don't fit"

**Isolation Forest** (`detect_anomalies_isolation_forest`)
- The intuition is "outliers are easy to isolate." Build many random trees; for
  each row, measure how shallow a leaf it lands in. Anomalies isolate after
  very few splits (short average path), normal rows sink deep.
- `contamination` is your **guess** at what fraction is anomalous (default 10%).
  This is the key thing to explain: the model does not discover the rate, you
  tell it, and it flags the most isolated points up to that quota. Setting it
  to 0.5 forces 50% of the data to be called anomalous, which is why the
  anomaly-index list is capped for transport but the *count* is always exact.
- Runs on the full frame even when large: the forest trains on an internal
  subsample but can score any row.

**Local Outlier Factor** (`detect_anomalies_lof`)
- Asks a different question: "is this row denser or sparser than its
  neighbours?" A point in a tight cluster of oddities is *not* an outlier by LOF,
  even though Isolation Forest may well call it one.
- That difference is the reason to offer both. Use LOF for "is this row
  surrounded by other weird rows?", Isolation Forest for "is this row
  individually hard to describe?"

**The distinction to stress:** clustering outliers and anomalies are different
questions. DBSCAN's `-1` means "not near enough to anything to call". Isolation
Forest's `-1` means "isolated faster than expected". They often overlap; they do
not have to.

### 4.3 Dimensionality reduction -- "flatten the table to 2D for a picture"

**PCA** (`reduce_pca_advanced`)
- Finds the directions of greatest variance and projects onto the first few.
- **Explained variance** = the share of the original information each component
  retains. `PC1 (34.2%)` means the first axis alone carries a third of the
  table's variation.
- It is a *linear* projection, so it is stable and interpretable, but it can only
  find straight-line structure. Spherical or curved clusters in high dimensions
  may end up on top of each other in a PCA plot.
- Uses the randomized solver on large frames, which is much faster than the
  exact SVD at negligible cost in accuracy.

**t-SNE** (`reduce_tsne`)
- Preserves *local* neighbourhoods rather than global variance: it tries to keep
  each row near the rows it was near before, and does not care about the overall
  layout.
- `perplexity` is roughly "how many neighbours to care about" (default 30).
  Small values fragment the map; large values blur it.
- **The critical warning to give people:** t-SNE is for *eyeballing* clusters.
  The distances between the blobs are meaningless, the axes are meaningless, and
  the layout changes between runs or with a different `perplexity`. Never read
  "these two groups are far apart" off a t-SNE plot. Use PCA if you need the
  geometry to mean something.
- On wide frames it first compresses to 50 components with PCA. This is both a
  performance measure and a quality one -- t-SNE on 200 raw columns produces a
  hairball.

### 4.4 Association rules -- "which categories travel together"


---

## 5. The quality metrics, in plain language

For one row, the **silhouette** compares its distance to its own cluster (a) with
its distance to the nearest other cluster (b): `(b - a) / max(a, b)`.

- Near **+1** -- comfortably inside its own group, far from the rest. Good.
- Near **0** -- sitting on the boundary. Ambiguous.
- Near **-1** -- closer to another cluster than its own. Probably misassigned.

Averaging over all rows gives the silhouette score. Above ~0.5 is usually taken
as strong structure, below ~0.25 as weak.

**The honest framing for a non-technical audience:** silhouette cannot tell you
the clusters are *true*, only that they are *geometrically distinct*. A dataset
with three genuinely separate groups and a dataset with three arbitrary slices
through one continuous cloud can both score well. The metrics tell you whether
the picture is clean; only domain knowledge tells you whether it means anything.

`metrics.scored_on_rows` and `scored_on_sample` are reported for this reason --
a score computed on 5,000 sampled rows is an estimate, and the response says so.

---

## 6. Large datasets: why it used to hang, and what changed

This is the part worth being precise about, because "it hung" is almost never
one bug. It is an algorithm-cost problem plus a payload problem.

### 6.1 The cost problem

Every method here fits the whole matrix, and cost grows with row count:

| Method | Cost | Where it breaks |
|---|---|---|
| K-Means | ~O(n · k · d) per restart | Slow but survivable; `n_init=10` multiplies it 10x |
| Hierarchical | O(n^2) **memory** for the linkage matrix | 100k rows = 80 GB. Instant death. |
| DBSCAN | O(n log n) with a tree, O(n^2) worst case | Degrades badly in high dimension |
| LOF | ~O(n · d) to O(n^2) | Slow on wide frames |
| t-SNE | **O(n^2)** per gradient step | 50k rows is already minutes |
| PCA | O(n·d^2) for the exact SVD | Slow but linear-ish |
| Isolation Forest | ~O(n) | Fine at any size |
| Apriori | exponential in itemset length | One ID column and it never returns |

The budgets now in `unsupervised.py`:

```python
KMEANS_FULL_ROWS      = 20_000    # above this -> MiniBatchKMeans
KMEANS_MAX_FIT_ROWS   = 200_000   # hard cap on rows used to FIT
SUGGEST_MAX_ROWS      = 50_000    # "Analyze best K" fits once per K
AUTO_SEARCH_MAX_ROWS  = 20_000    # "Auto Best" fits ~26 models
DBSCAN_MAX_ROWS       = 20_000
LOF_MAX_ROWS          = 20_000
HIERARCHICAL_MAX_ROWS = 5_000     # hard limit, errors above it
TSNE_MAX_ROWS         = 5_000
TSNE_PRECOMPONENTS    = 50
METRICS_MAX_ROWS      = 5_000
ASSOCIATION_MAX_ROWS  = 100_000
ASSOCIATION_MAX_CARDINALITY = 50
ASSOCIATION_MAX_ITEMSET_LEN = 3
```

Sampling is deterministic (`_subsample_indices`, seed 42) and sorted, so:
- re-running on the same data gives the same answer, and
- the sample keeps the original row order, which matters for writing

### 6.3 The payload problem (the other half of "it hung")

Even with fast maths, the old code built one JSON object per row:

```python
"points": [{"x": ..., "y": ..., "cluster": ...} for i in range(len(X))]
```

At 1M rows that is a multi-megabyte response the browser must parse and a canvas
Chart.js must draw -- point by point. The request would finish server-side and
the page would still appear frozen. It looks identical to a compute hang from the
user's side, which is why both had to be fixed.

Caps now in place:

| Cap | Value | What it bounds |
|---|---|---|
| `MAX_SCATTER_POINTS` | 3,000 | Points drawn on any chart |
| `MAX_RETURNED_LABELS` | 200,000 | Label list in the response |
| `MAX_RETURNED_ANOMALY_INDICES` | 5,000 | Anomaly index list |

Random *subsets* rather than head slices, so the plotted cloud is not biased by
row order. The important property: **the analysis still used the full (or
sampled) matrix.** Only the display is capped. Scores, cluster sizes, and
variance figures are computed on the real data, and the response states the true
count alongside the truncated list:

- `n_rows_scored` -- exact, never capped
- `n_outliers` -- exact, never capped
- `scaling.rows_used` / `rows_total` / `sampled` / `note`
- `labels_complete` -- false when the label list was truncated
- `metrics.scored_on_sample` -- true when metrics came from a row sample

### 6.4 Why `cluster_label` is sometimes not added

`main.py` writes the column back only when the returned label list covers every
row:

```python
if len(result.get("labels", [])) == len(session.df):
    session.df["cluster_label"] = result["labels"]
```

Above 200k rows the list is truncated, so the column is skipped rather than
half-written. The UI explains this instead of leaving the user wondering where
the column went.

### 6.5 Other fixes worth knowing

- **`contamination` percentage measured correctly.** DBSCAN's outlier share is
  now relative to the rows actually clustered. Dividing a sample's outliers by
  the *full* row count made a sampled run look like it had found almost nothing.
- **LOF indices mapped back to real rows.** LOF labels a sampled matrix, so
  position 412 in the sample is not row 412 of the file. Without remapping, the
  UI would show row numbers that do not exist in the user's data.
- **High-cardinality columns dropped before one-hot encoding.** An ID or email
  column becomes one "item" per row; none can ever be frequent, and Apriori
  enumerates them all. Columns with more than 50 distinct values are now skipped
  and listed in the response.
- **`max_len=3` on Apriori.** Bounds the exponential itemset growth.
- **`n_jobs=-1`** on DBSCAN, LOF, and Isolation Forest to use all cores.
- **`_compact_unsupervised_result`** (in both `main.py` and `main_report.py`)
  now prefers the authoritative `n_rows_scored` over `len(labels)`, which would
  otherwise understate it once labels are capped.

### 6.6 Measured results

From `tests/test_unsupervised_large_dataset.py` (150k-200k row frames):

| Operation | Time |
|---|---|
| K-Means, 200k rows | ~2.4s |
| suggest_clusters, 150k rows, 9 K values | ~12.6s |

---

## 7. How to walk through it live

1. **Load a dataset.** Any table with 2+ numeric columns works.
2. **Hit "Suggest"** (`/api/unsupervised/suggest/{id}`). It reports the numeric
   and categorical column counts and which tasks apply. Good place to ask
   "which of these do we have data for?"
3. **Hit "Analyze best K"** (`/api/unsupervised/suggest_clusters/{id}`). This is
   the most instructive screen: the table shows all four metrics per K plus the
   vote count. Ask "why did it pick K=4 when the elbow looks like 3?" -- there is
   a real answer (the metrics disagree), and that is the honest state of affairs.
4. **Run K-Means with that K.** Labels land in `cluster_label` on the frame.
5. **Compare with DBSCAN.** Different K? More outliers? That comparison teaches
   more than either result alone.
6. **Run Isolation Forest, then LOF.** Find rows one flags and the other does
   not. Ask what that tells you about the shape of the data.
7. **Run PCA, then t-SNE on the same data.** Same information, very different
   pictures. Use this to make the "t-SNE is for eyeballing only" point
   concretely rather than as a warning to accept.
8. **Load a big file and repeat step 3.** Watch the "fitted on N of M rows"
   note appear. That is the scaling behaviour made visible.

---

## 8. Honest limitations

Worth stating plainly rather than discovering them later:

1. **No ground truth.** Every score is internal. Segments are not validated
   against anything external.
2. **Numeric columns only** (except association rules). Categorical features are
   ignored rather than encoded, so a dataset with one strong categorical signal
   will underperform.
3. **K-Means assumes spherical, similar-sized clusters.** On elongated or
   wildly unequal groups it will underperform, and DBSCAN will fit better.
   "Auto Best" exists partly to catch this automatically.
4. **Sampled results are estimates.** Above each budget the numbers describe a
   sample. For 200k+ rows the difference is usually immaterial, but it is not
   zero, and the response says when it applies.
5. **The 2D chart shows the first two features** (or the embedding), not the
   full clustering distance. Two clusters that look overlapping in the chart may
   be well separated in 10 dimensions. This is the most common misreading of a
   clustering plot.
6. **Cluster IDs are arbitrary.** "Cluster 0" has no inherent meaning and is not
   stable across runs with different K. Never use the number as a business label.
7. **t-SNE output is not stable** across `perplexity` values or data samples.
   Do not quote specific coordinates.
8. **`contamination` is an input, not a finding.** Isolation Forest returns
   exactly the quota you asked for. Choosing 0.1 does not mean 10% of your data
   is anomalous -- it means you asserted that, and the model found the most
   isolated 10%.

---

## 9. Where to change things

| Want | Change |
|---|---|
| Handle more rows exactly | Raise the `*_MAX_ROWS` constants in `backend/unsupervised.py` |
| Allow hierarchical on big data | Raise `HIERARCHICAL_MAX_ROWS` (memory is O(n^2) -- check RAM first) |
| Send bigger charts | Raise `MAX_SCATTER_POINTS` (browser performance is the limit) |
| Add a metric | `_clustering_metrics()` in `backend/unsupervised.py` |
| Add a method | Fit in `backend/unsupervised.py`, add a branch in `main.py`, add a `<select>` option in `frontend/index.html` |
| Change K selection | `suggest_clusters()` -- the vote is a `Counter` over four criteria |

Regression tests: `tests/test_unsupervised_large_dataset.py` (21 tests, ~2 min).
They pin the budgets, so raising a constant without a matching test update will
be visible in review.

| cluster_best, 150k rows, ~26 models | ~36.6s |
| t-SNE, 40k rows x 60 features (capped to 5k) | ~28.9s |
| Isolation Forest, 150k rows | ~5.0s |

`t-SNE` and `cluster_best` are the slow ones by design -- they are capped, and
the caps are the trade. If you need an exact t-SNE or an exhaustive 26-model
search on a million rows, raise `TSNE_MAX_ROWS` / `AUTO_SEARCH_MAX_ROWS` and
accept the wait, or run it offline.

  `cluster_label` back to the right rows.

### 6.2 The distinction that makes K-Means work at full size

This is the key idea, and it is worth explaining properly:

> **Fitting** a K-Means model is expensive. **Assigning** rows to an
> already-fitted model is cheap -- it is just "which of the K centroids is
> nearest?"

So the code fits on a bounded sample, then calls `model.predict(X)` on **all**
rows. Result: every row in a 5-million-row frame still gets a cluster label, at
a fraction of the cost. `MiniBatchKMeans` (used above 20k rows) keeps the same
interface but processes rows in batches of 4096.

**The methods that cannot do this** are DBSCAN, hierarchical, and LOF -- none
has an out-of-sample `predict`. For those, labels cover only the rows they were
fitted on, and the response says so explicitly rather than quietly returning
partial results. For hierarchical above 5,000 rows the API refuses outright and
points at K-Means, rather than starting a computation that cannot finish.

**Apriori** (`association_rules`)
- Only for categorical/boolean columns. Each distinct value becomes an "item",
  each row a "basket".
- **Support** = how often the itemset appears. *Gate*, not a quality measure.
- **Confidence** = of the rows containing A, how many also contain B. This is
  the directional one: A→B can be 95% confident while B→A is 40%.
- **Lift** = confidence ÷ support(B). **The number that actually matters.**
  Lift 1.0 means B appears just as often with A as it does on its own -- no
  relationship, however confident. Lift > 1 is a real positive association;
  lift < 1 is a real *negative* one. Rules are sorted by lift for this reason.

**Auto Best** (`cluster_best`)
Runs ~26 candidates (K-Means for each K, hierarchical for each K, DBSCAN for
each eps), ranks them by silhouette with Davies-Bouldin breaking close ties,
then **refits the winner** so the returned labels are a proper fit rather than a
leftover from the search.
