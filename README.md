# ForestGeom
```text
     x_i ● ─────────────┐     ┌──────────── ● x_j
                        ▼     ▼
               ┌─────────────────────────┐
               │     FOREST ENSEMBLE     │
               └───────────┬─────────────┘
                           │
               ┌───────────┴───────────┐
               │                       │
               ▼                       ▼
      ┌─────────────────┐     ┌─────────────────┐
      │ same decision   │     │ divergent       │
      │ paths           │     │ decision paths  │
      │                 │     │                 │
      │        ●        │     │        ●        │
      │       / \       │     │       / \       │
      │      ●   ●      │     │      ●   ●      │
      │     /     \     │     │     /     \     │
      │    ●       ●    │     │    ●       ●    │
      │   / \     / \   │     │   / \     / \   │
      │  ●   ●   ●   ●  │     │  ●   ●   ●   ●  │
      │      ▲          │     │  ▲       ▲      │
      │     x_i         │     │ x_i     x_j     │
      │     x_j         │     │                 │
      └────────┬────────┘     └────────┬────────┘
               │                       │
               └───────────┬───────────┘
                           ▼     
              vectorizing forest geometry
                      x_i ↦ φ(x_i)
```

`forestgeom` implements the sparse leaf-incidence kernel framework developed in
“Revisiting Forest Proximities via Sparse Leaf-Incidence Kernels”
(https://arxiv.org/abs/2601.02735). The package treats a fitted tree ensemble as
a reusable geometric object: samples are encoded by the leaves they reach, and
forest proximities are represented through sparse linear maps rather than dense
pairwise matrices.

The current core API is `forestgeom.LeafEncoder`, which fits a supported
ensemble and encodes samples by the leaves they reach. This yields sparse
query-side `x_i -> phi_q(x_i)` and reference-side `x_i -> phi_w(x_i)` weighted
leaf-incidence maps that factorize the forest proximity:

```text
P = Q W^T
```

Here, `Q` and `W` are the query/reference-side representations of the training
samples, each with shape `n_train x n_leaves` for the fitted reference set. Both
maps are sparse, with at most one nonzero per tree per row, so downstream
methods can work directly with the factors instead of materializing pairwise
proximity matrices.

The implementation includes several proximity constructions within this
leaf-incidence view, including standard forest kernels, KeRF-style
leaf-size-normalized kernels, boosted tree-weighted kernels, and GAP/OOB
proximities from “Random Forest- Geometry- and Accuracy-Preserving Proximities”
(https://ieeexplore.ieee.org/document/10089875).

The project is intended to evolve beyond leaf-incidence maps into a broader
framework for forest-induced representation learning. Natural extensions include
path-based encoders, alternative forest geometries, additional base forest
families, and integrations with downstream tasks such as embedding, clustering,
imputation, uncertainty estimation, and semi-supervised learning. Contributions
in these directions are welcome.

# Installation

Install the latest released version from PyPI:

```bash
pip install forestgeom
```

Optional dependencies are grouped by feature:

```bash
# LightGBM and XGBoost adapters
pip install "forestgeom[boosted]"

# Visualization and embedding tools
pip install "forestgeom[viz]"

# Experiment dependencies
pip install "forestgeom[experiments]"

# Test dependencies
pip install "forestgeom[test]"

# Everything above
pip install "forestgeom[all]"
```

To try unreleased features from the GitHub repository, install directly from a
branch, tag, or commit:

```bash
# latest main branch
pip install git+https://github.com/JakeSRhodesLab/ForestGeom.git

# specific branch or tag
pip install git+https://github.com/JakeSRhodesLab/ForestGeom.git@main

# GitHub install with extras
pip install 'git+https://github.com/JakeSRhodesLab/ForestGeom.git@main#egg=forestgeom[boosted]'
```

For local development from a cloned checkout:

```bash
pip install -e ".[test]"
```

# Architecture

ForestGeom is organized around one estimator, `LeafEncoder`. The encoder turns a
tree ensemble into sparse leaf maps and then exposes those maps directly or uses
them to compute proximities and proximity-weighted predictions.

```text
RandomForest / ExtraTrees / GBT / LightGBM / XGBoost
                         |
                         v
X_train, y_train --> +-------------+
fit(...)             | LeafEncoder |
                     +-------------+
                         |
                         v
              fitted adapter + ForestCache
                         |
                         v
        +----------------+----------------+
        |                                 |
        v                                 v
+-----------------------+       +-----------------------+
| Q_train               |       | W_train               |
| training query map    |       | reference map         |
|                       |       |                       |
| training_query_map()  |       | reference_map()       |
| fit_transform(...)    |       +-----------------------+
+-----------------------+                 |
        |                                 |
        |                                 |
        |               +-----------------+
        |               |
        v               v
+-------------------------------+
| training geometry             |
| proximity()                   |
| Q_train @ W_train.T           |
+-------------------------------+

X_new --> transform(...) --> +---------------------------+
                             | Q_new                     |
                             | out-of-sample query map   |
                             +---------------------------+
                                        |
                                        v
                       +----------------+----------------+
                       |                                 |
                       v                                 v
        +------------------------------+   +------------------------------+
        | out-of-sample geometry       |   | proximity-weighted outputs   |
        | proximity_extend(X_new)      |   | proximity_predict(X_new)     |
        | Q_new @ W_train.T            |   | proximity_predict_proba(...) |
        +------------------------------+   +------------------------------+
```

The adapter layer hides backend-specific details such as leaf indexing,
bootstrap masks, in-bag counts, and boosted tree weights. The map-building layer
then uses those normalized quantities to construct `Q` and `W` for the selected
weighting scheme (`uniform`, `kerf`, `oob`, `gap`, or `boosted`).

# Usage

`LeafEncoder` wraps a tree ensemble estimator and clones/fits it during
`fit(...)`. It supports scikit-learn Random Forests, ExtraTrees, and Gradient
Boosting estimators, with optional adapters for LightGBM and XGBoost when those
packages are installed.

Supported base forest classes include:

- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.ensemble.ExtraTreesClassifier`
- `sklearn.ensemble.ExtraTreesRegressor`
- `sklearn.ensemble.GradientBoostingClassifier`
- `sklearn.ensemble.GradientBoostingRegressor`
- `lightgbm.LGBMClassifier` and `lightgbm.LGBMRegressor` with
  `forestgeom[boosted]`
- `xgboost.XGBClassifier` and `xgboost.XGBRegressor` with `forestgeom[boosted]`

Supported leaf-weighting schemes include:

- `uniform`: symmetric leaf co-occurrence factorization of the standard forest
  kernel.
- `kerf`: symmetric leaf-size-normalized factorization of the KeRF kernel.
- `oob`: separable OOB leaf-incidence factorization that approximates the
  off-diagonal Breiman OOB affinities.
- `gap`: asymmetric query/reference factorization that combines OOB-side query
  weights with in-bag reference weights to recover the GAP proximity definition.
- `boosted`: symmetric tree-weighted leaf kernel for supported boosted
  ensembles.

Not every estimator supports every weighting scheme. Random Forests and
ExtraTrees estimators support `uniform` and `kerf`; they support `oob` and
`gap` only when fitted with `bootstrap=True`. Boosted estimators support
`uniform`, `kerf`, and `boosted`.

Leaf maps follow the usual scikit-learn transformer flow: use `fit(...)` when
you want to keep the fitted encoder, `fit_transform(...)` when you want the
training query map immediately, and `transform(...)` for new samples. This makes
the query-side leaf representation easy to use in downstream estimators and
pipelines that consume sparse feature matrices.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from forestgeom import LeafEncoder

forest = RandomForestClassifier(
    n_estimators=500,
    bootstrap=True,
    random_state=0,
    n_jobs=-1,
)

encoder = LeafEncoder(forest=forest, weight_scheme="uniform")

# sklearn-style transformer flow.
Q_train = encoder.fit_transform(X_train, y_train)
Q_test = encoder.transform(X_test)

# The sparse leaf maps can feed downstream models directly.
clf = LogisticRegression(max_iter=1000)
clf.fit(Q_train, y_train)
pred = clf.predict(Q_test)

# The encoder can also be placed in a pipeline when only the query map is needed.
pipe = make_pipeline(
    LeafEncoder(forest=forest, weight_scheme="uniform"),
    LogisticRegression(max_iter=1000),
)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
```

For symmetric weighting schemes such as `uniform`, `kerf`, and `boosted`, the
training query map can usually be treated as the leaf-space feature matrix. For
asymmetric schemes such as `gap`, the geometry is defined by two maps: the
query-side map `Q` and the reference-side map `W`. In those cases, downstream
code that needs proximities should keep both maps or use `Q @ W.T` explicitly rather than
assuming a single symmetric feature representation.

```python
from sklearn.ensemble import RandomForestClassifier
from forestgeom import LeafEncoder

forest = RandomForestClassifier(
    n_estimators=500,
    bootstrap=True,
    random_state=0,
    n_jobs=-1,
)

encoder = LeafEncoder(forest=forest, weight_scheme="gap").fit(X_train, y_train)

# Sparse leaf maps for custom downstream work.
Q_train = encoder.training_query_map()
W_train = encoder.reference_map()
Q_test = encoder.transform(X_test)

# Explicit proximity matrices. These are sparse by default.
K_train = encoder.proximity()
K_test_train = encoder.proximity_extend(X_test)

# Dense output is available when needed.
K_train_dense = encoder.proximity(return_dense=True)
```

For proximity-weighted prediction, `LeafEncoder.proximity_predict(X)` and
`LeafEncoder.proximity_predict_proba(X)` provide matrix-free convenience wrappers that use
the fitted base forest task type: regression forests return weighted responses,
while classification forests return weighted class predictions/probabilities.
They avoid materializing the full proximity matrix `P` by multiplying the sparse
leaf factors against the training targets or class indicators directly.

For asymmetric weighting schemes such as `gap`, the fitted training query map
`Q_train` and reference map `W_train` differ. The train-train proximity is still
computed as `Q_train @ W_train.T`, and `proximity_extend(X)` returns
`Q(X) @ W_train.T` for out-of-sample data.

The sparse factors can be used directly in proximity methods, manifold learning,
dimensionality reduction, visualization, imputation, and other proximity-based
workflows.

# Citation

If you use this software in your research or experiments, please cite the following works:

```bibtex
@ARTICLE{10089875,
  author={Rhodes, Jake S. and Cutler, Adele and Moon, Kevin R.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Geometry- and Accuracy-Preserving Random Forest Proximities}, 
  year={2023},
  volume={45},
  number={9},
  pages={10947-10959},
  keywords={Random forests;Forestry;Geometry;Data visualization;Decision trees;Task analysis;Anomaly detection;Proximities;random forests;supervised learning},
  doi={10.1109/TPAMI.2023.3263774}}
```

```bibtex
@misc{aumon2026revisitingforestproximitiessparse,
      title={Revisiting Forest Proximities via Sparse Leaf-Incidence Kernels}, 
      author={Adrien Aumon and Guy Wolf and Kevin R. Moon and Jake S. Rhodes},
      year={2026},
      eprint={2601.02735},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.02735}, 
}
```
