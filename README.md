# ForestGeom

[![license: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-40cdbc)](LICENSE)
[![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-40cdbc)](pyproject.toml)
[![pkg: uv](https://img.shields.io/badge/pkg-uv-40cdbc)](https://docs.astral.sh/uv/)
[![PyPI: forestgeom](https://img.shields.io/badge/PyPI-forestgeom-40cdbc)](https://pypi.org/project/forestgeom/)

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
                    forest geometry
         unified - fast - sparse - vectorized
```

`forestgeom` implements the sparse leaf-incidence kernel framework developed in
“Revisiting Forest Proximities via Sparse Leaf-Incidence Kernels” [[1]](#ref-1). The
package treats a fitted tree ensemble as a reusable geometric object: samples
are encoded by the leaves they reach, and forest proximities are represented
through sparse linear maps rather than dense pairwise matrices.

Since their original formulation by Leo Breiman in the early 2000s, forest
proximities have given already-powerful decision forest models a geometric
perspective. This intuitive notion of semi-supervised relationships between
points, based on decision-path similarity across trees, has long been treated as
a fixed procedure whose direct computation is expensive in the number of
samples. `forestgeom` removes this burden with efficient sparse linear algebra
through sparse leaf-collision kernels; more importantly, it directly exposes
sparse forest-leaf-induced maps for matrix-free, forest-guided downstream
representation learning. See [[1]](#ref-1) for details.

Forest geometry has since been used in a wide range of applications that need a
graded, context-aware notion of similarity beyond class-conditional Euclidean
distances or black-box deep representation models. This makes it especially
useful for modality-agnostic pipelines, tabular data, and sparse, noisy,
high-dimensional settings such as single-cell analysis. Reference [[1]](#ref-1)
provides a comprehensive literature overview, and this package aims to
encourage further work in these directions by unifying a collection of forest
models and geometric perspectives behind a single API.

The implementation includes several proximity constructions within this
leaf-incidence view, including standard forest kernels, KeRF-style
leaf-size-normalized kernels, boosted tree-weighted kernels, and GAP/OOB
proximities from “Geometry- and Accuracy-Preserving Random Forest
Proximities” [[2]](#ref-2).

The project is intended to evolve beyond leaf-incidence maps into a broader
framework for forest-induced representation learning. Natural extensions include
path-based geometry, additional base forest
families, and GPU-accelerated pipelines.

# Installation

The recommended installer is [`uv`](https://docs.astral.sh/uv/). To install
into the active environment:

```bash
uv pip install forestgeom
```

Optional dependencies are grouped by feature:

```bash
# LightGBM and XGBoost adapters
uv pip install "forestgeom[boosted]"

# Visualization and embedding tools
uv pip install "forestgeom[viz]"

# Experiment dependencies
uv pip install "forestgeom[experiments]"

# Test dependencies
uv pip install "forestgeom[test]"

# Everything above
uv pip install "forestgeom[all]"
```

To try unreleased features from the GitHub repository, install directly from a
branch, tag, or commit:

```bash
# latest main branch
uv pip install git+https://github.com/JakeSRhodesLab/ForestGeom.git

# specific branch or tag
uv pip install git+https://github.com/JakeSRhodesLab/ForestGeom.git@main

# GitHub install with extras
uv pip install 'git+https://github.com/JakeSRhodesLab/ForestGeom.git@main#egg=forestgeom[boosted]'
```

If you are adding `forestgeom` to an existing uv-managed project, use `uv add`
instead:

```bash
uv add forestgeom
uv add "forestgeom[boosted]"
```

For local development from a cloned checkout:

```bash
uv sync --extra test
```

<details>
<summary>pip also works</summary>

```bash
pip install forestgeom
pip install "forestgeom[boosted]"
pip install "forestgeom[viz]"
pip install "forestgeom[experiments]"
pip install "forestgeom[test]"
pip install "forestgeom[all]"
pip install git+https://github.com/JakeSRhodesLab/ForestGeom.git
pip install -e ".[test]"
```

</details>

# Architecture

ForestGeom is organized around one central object, `ForestProximity`. The class
wraps a fitted tree ensemble and turns it into a reusable geometry object built
from sparse leaf-incidence maps.

```text
        RandomForest / ExtraTrees / GBT / LightGBM / XGBoost
                                    |
                                    v
   X_train, y_train --> +------------------------+
   fit(...)             |    ForestProximity     |
                        +------------------------+
                                    |
                                    v
                      fitted adapter + ForestCache
                                    |
                                    v
                  +------------------------------------+
                  | sparse forest representation       |
                  | leaf incidence + scheme weights    |
                  +------------------------------------+
                                    |
                      +-------------+-------------+
                      |                           |
                      v                           v
            +----------------------+    +------------------------+
            | separable schemes    |    | corrected schemes      |
            | P = Q W^T            |    | P = normalize(QQ^T)    |
            +----------------------+    +------------------------+
                      |                           |
                      v                           |
            +----------------------+              |
            | Q: query_map(X=None) |              |
            | W: reference_map()   |              |
            +----------------------+              |
                      |                           |
                      +-------------+-------------+
                                    |
                                    v
                          +-------------------+
                          | transform(X_new)  |
                          | P(X_new, X_train) |
                          +-------------------+
                                    |
                                    v
                      forest-induced proximity geometry
```

The adapter layer hides backend-specific details such as leaf indexing,
bootstrap masks, in-bag counts, and boosted tree weights. The map-building layer
then uses those quantities to construct the sparse geometry for the selected
weighting scheme (`uniform`, `kerf`, `oob`, `gap`, or `boosted`).

The important distinction is:

- Symmetric schemes such as `uniform`, `kerf`, and `boosted` use the same
  leaf-incidence geometry on both sides, so the resulting proximity is a kernel.
- Asymmetric schemes such as `gap` expose distinct query and reference maps,
  which induce a bilinear form `P(i, j) = <Q(i), W(j)>` rather than a kernel.
- The true Breiman OOB scheme is pairwise-normalized and is computed directly
  as a sparse proximity matrix; it does not factor into a single reusable `Q`/`W`
  pair.

# Usage

`ForestProximity` wraps a tree ensemble estimator and clones/fits it during
`fit(...)`. It supports a unified set of forest backends and weighting schemes:

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
- `oob`: pairwise-normalized Breiman OOB proximity computed directly in sparse
  form.
- `gap`: asymmetric query/reference factorization that combines OOB-side query
  weights with in-bag reference weights to recover the GAP proximity definition.
- `boosted`: symmetric tree-weighted leaf kernel for supported boosted
  ensembles.

Not every estimator supports every weighting scheme. Random Forests and
ExtraTrees estimators support `uniform` and `kerf`; they support `oob` and
`gap` only when fitted with `bootstrap=True`. Boosted estimators support
`uniform`, `kerf`, and `boosted`.

Use `fit(...)` when you want to train and keep the fitted geometry, and use
`fit_transform(...)` when you want the fitted train-train proximity matrix right
away. Use `query_map(...)` and `reference_map(...)` when you need the actual
leaf-incidence factors `Q` and `W` for matrix-free applications, and use `transform(...)`
for the proximity block from new samples to the fitted training set.

For schemes that are not symmetric kernels, **`fit_transform(...)` and
`fit(...).transform(...)` are not necessarily the same**. If you need the
training geometry, use `fit_transform(...)` directly or call
`training_proximity(...)` on the fitted estimator.

For symmetric weighting schemes such as `uniform`, `kerf`, and `boosted`, the
query map is typically the leaf-space feature matrix. For asymmetric schemes
such as `gap`, keep both `Q` and `W` if you want to work directly with the
geometry. For `oob`, use `training_proximity(...)` or `transform(...)` directly;
there is no separate query/reference factorization.

The sparse geometry can be used directly in proximity-based workflows such as
manifold learning, dimensionality reduction, visualization, imputation, and
custom downstream estimators.

# Quick Start

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC, SVC

from forestgeom import ForestProximity

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.25,
  stratify=y,
  random_state=0,
)

forest = RandomForestClassifier(
  n_estimators=200,
  bootstrap=True,
  random_state=0,
  n_jobs=-1,
)

geometry = ForestProximity(forest=forest, weight_scheme="uniform").fit(X_train, y_train)

# Query/reference maps define the symmetric geometry.
Q_train = geometry.query_map()
W_train = geometry.reference_map()
Q_test = geometry.query_map(X_test)

# The proximity matrix can be consumed directly in a pipeline.
pipe = make_pipeline(
  ForestProximity(forest=forest, weight_scheme="uniform"),
  SVC(kernel="precomputed"),
)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print(f"proximity-svm accuracy: {accuracy_score(y_test, pred):.3f}")

# Matrix-free variant using the query maps directly as sparse features.
svm = LinearSVC()
svm.fit(Q_train, y_train)
pred = svm.predict(Q_test)
print(f"query-map-svm accuracy: {accuracy_score(y_test, pred):.3f}")

# To run the boosted example, install optional dependencies first:
# uv pip install "forestgeom[boosted]"
from xgboost import XGBClassifier

forest = XGBClassifier(n_estimators=200, random_state=0)
boosted_geometry = ForestProximity(forest=forest, weight_scheme="boosted")
K_train = boosted_geometry.fit_transform(X_train, y_train)
K_test = boosted_geometry.transform(X_test)
```

# Demos and Experiments
The repository includes notebook demos for common workflows:

- `demos/demo_iris.ipynb`: general-purpose introduction on the Iris dataset.
- `demos/demo_leaf_pca.ipynb`: matrix-free supervised manifold learning with leaf PCA using the leaf-incidence maps in kernel proximities.
- `demos/demo_boosted.ipynb`: boosted-tree examples using the optional boosted
  adapters.

The `experiments/` directory contains Python scripts and notebooks used
to reproduce experiments and compile results from “Revisiting Forest Proximities via Sparse
Leaf-Incidence Kernels”.

# Citation

If you use this software in your research or experiments, please cite the
leaf-incidence kernel framework paper [[1]](#ref-1):

<a id="ref-1"></a>[1] Revisiting Forest Proximities via Sparse Leaf-Incidence
Kernels.

```bibtex
@misc{aumon2026revisitingforestproximitiessparse,
      title={Revisiting Forest Proximities via Sparse Leaf-Incidence Kernels}, 
      author={Adrien Aumon and Guy Wolf and Kevin R. Moon and Jake S. Rhodes},
      year={2026},
      eprint={2601.02735},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.02735}}
```

If you specifically use the `gap` weighting scheme, please also cite the GAP
proximity paper [[2]](#ref-2):

<a id="ref-2"></a>[2] Geometry- and Accuracy-Preserving Random Forest
Proximities.

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
