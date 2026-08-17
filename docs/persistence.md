# Saving and loading models

comprisk offers two ways to persist a fitted estimator. Both are lossless:
round-tripped predictions are bit-identical to the in-memory model, enforced
by the test suite.

## `save()` / `load()` — the recommended way to ship a model

```python
model = CompetingRiskForest(n_estimators=500, random_state=42).fit(X, time, event)
model.save("model.crm")

# on the receiving side
import comprisk
model = comprisk.load("model.crm")
```

The file is a zip container of JSON metadata plus raw `.npy` arrays. Two
properties make it the right format for sending models to collaborators:

- **No code execution on load.** Unlike pickle, `comprisk.load` never
  unpickles objects — it reads JSON and arrays
  (`np.load(allow_pickle=False)`). A model file from outside your trust
  boundary cannot run code on your machine.
- **Deterministic bytes.** Saving the same model twice in the same
  environment produces identical files (fixed zip timestamps), so a checksum
  recorded in a manifest stays valid across re-saves. Compressed bytes can
  differ across zlib builds, so verify checksums where the file was written.

`FineGrayRegression` supports the same `save()` / `comprisk.load()` pair.
`save()` covers the default flat-tree forest; reference-mode and
rfsrc-aligned forests (research and cross-library-parity configurations)
persist via pickle instead.

## Pickle / joblib

Standard pickling works as always and remains the right choice for
sklearn-ecosystem plumbing (`joblib` caching, `GridSearchCV`, model
registries that expect pickles). Old pickles from earlier comprisk versions
load unchanged.

## Sizes

Serialized forests are dominated by per-leaf competing-risk curves. comprisk
stores these sparsely and rebuilds the dense representation on load
(bit-identically), so files are far smaller than the in-memory footprint.
Measured on a 20,000-sample, 20-tree synthetic fit (default parameters):

| Serialization | Size | vs old format |
| --- | --- | --- |
| Plain pickle, comprisk ≤ 0.7 (dense state) | 202.8 MB | 1× |
| Plain pickle, compact state | 8.4 MB | 24× |
| Compact state + `joblib.dump(..., compress=("zlib", 3))` | 4.8 MB | 42× |
| `save()` (compact + deflate) | 4.6 MB | 44× |

The ratio grows with forest size. On a real clinical forest (500 trees,
`min_samples_leaf=3`, cohort size in the tens of thousands) whose
dense-state pickle is 33.3 GB, measured 2026-08-17: the compact pickle is 444 MB (75×) and
`save()` lands at 116 MB (287×) — written in 21 s, loaded in 40 s, with
bit-identical round-tripped predictions and unchanged held-out concordance.

Loading rebuilds each tree's leaf CIF table from its sparse counts; this is
a vectorized pass that adds seconds, not minutes, even for forests whose
dense state runs to gigabytes.
