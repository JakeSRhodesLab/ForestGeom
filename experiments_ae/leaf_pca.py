"""Leaf PCA fitting and image reconstruction."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from forestgeom import ForestProximity
from mcr import train_decoder


@dataclass
class LeafPCAResult:
    """Artifacts needed to inspect or compare a Leaf PCA reconstruction."""

    encoder: ForestProximity
    pca: PCA
    Z_train: np.ndarray
    Z_test: np.ndarray
    predicted_leaves: np.ndarray
    X_test_hat: np.ndarray

    @property
    def forest(self):
        return self.encoder.forest_.estimator


def reconstruct_leaf_pca(
    X_train,
    y_train,
    X_test,
    image_shape,
    *,
    n_estimators=1000,
    min_samples_leaf=2,
    random_state=0,
    n_jobs=-1,
):
    """Fit a 2D KeRF Leaf PCA model and decode test points to image space."""
    encoder = ForestProximity(
        RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        weight_scheme="kerf",
    ).fit(X_train, y_train)

    leaf_train = encoder.query_map(return_dense=False)
    leaf_test = encoder.query_map(X_test, return_dense=False)
    pca = PCA(n_components=2, random_state=random_state)
    Z_train = pca.fit_transform(leaf_train)
    Z_test = pca.transform(leaf_test)
    leaf_test_hat = pca.inverse_transform(Z_test)

    # Select the highest-scoring terminal leaf inside each tree's coordinate block.
    forest = encoder.forest_.estimator
    predicted_leaves = np.column_stack(
        [
            ids[np.argmax(leaf_test_hat[:, offset + ids], axis=1)]
            for tree, offset in zip(forest.estimators_, encoder.cache_.leaf_offsets)
            for ids in [np.flatnonzero(tree.tree_.children_left == -1)]
        ]
    )

    columns = list(range(X_train.shape[1]))
    metadata = pd.DataFrame(
        {
            "variable": columns,
            "class": "numeric",
            "decimals": 0,
            "min": np.min(X_train, axis=0),
            "max": np.max(X_train, axis=0),
        }
    )
    emap = SimpleNamespace(
        leafIDs=predicted_leaves,
        meta={"metadata": metadata, "input_class": ["np.ndarray"]},
    )

    # mcr.py samples uniformly inside the intersected predicted leaf boxes.
    rng_state = np.random.get_state()
    np.random.seed(random_state)
    try:
        decoded = train_decoder(forest, emap, n_jobs=1).to_numpy()
    finally:
        np.random.set_state(rng_state)

    return LeafPCAResult(
        encoder=encoder,
        pca=pca,
        Z_train=Z_train,
        Z_test=Z_test,
        predicted_leaves=predicted_leaves,
        X_test_hat=decoded.reshape(-1, *image_shape),
    )
