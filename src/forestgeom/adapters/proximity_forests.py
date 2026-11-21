"""Adapter for the optional proximity-forests backend."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin, RegressorMixin

from proximity_forests import (
    ProximityForestClassifier,
    ProximityForestRegressor,
    ProximityIsolationForest,
)

from .base import EnsembleAdapter


def _is_pf_fitted(estimator):
    """Return fitted state using proximity-forests' native model check."""
    try:
        estimator._check_is_fitted()
    except RuntimeError:
        return False
    return True


class _ProximityForestProxyBase(BaseEstimator):
    """Private sklearn-compatible proxy around a proximity-forests estimator."""

    def __init__(self, estimator):
        self.estimator = estimator

    def __sklearn_clone__(self):
        """Clone the wrapped PF estimator without requiring sklearn inheritance."""
        params = self.estimator.get_params(deep=False)
        cloned = type(self.estimator)(**params)
        return type(self)(cloned)

    def __sklearn_is_fitted__(self):
        """Let sklearn.check_is_fitted use PF's native fitted-state check."""
        return _is_pf_fitted(self.estimator)

    def fit(self, X, y=None, **fit_kwargs):
        if y is None:
            self.estimator.fit(X, **fit_kwargs)
        else:
            self.estimator.fit(X, y, **fit_kwargs)
        # This conventional fitted attribute also supports older sklearn checks.
        self.estimator_ = self.estimator
        return self

    def apply(self, X):
        return self.estimator.apply(X)

    def get_oob_mask(self):
        return self.estimator.get_oob_mask()

    def get_in_bag_counts(self):
        return self.estimator.get_in_bag_counts()

    def __getattr__(self, name):
        # Avoid recursion during construction, cloning, and unpickling.
        if name == "estimator":
            raise AttributeError(name)
        estimator = self.__dict__.get("estimator")
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)


class _ProximityForestClassifierProxy(
    ClassifierMixin, _ProximityForestProxyBase
):
    """Classifier-role proxy recognized by sklearn.is_classifier."""


class _ProximityForestRegressorProxy(
    RegressorMixin, _ProximityForestProxyBase
):
    """Regressor-role proxy recognized by sklearn estimator tags."""


class _ProximityIsolationForestProxy(
    OutlierMixin, _ProximityForestProxyBase
):
    """Outlier-role proxy for the unsupervised isolation estimator."""


class ProximityForestsAdapter(EnsembleAdapter):
    """Adapt PF classifier, regressor, and isolation models to ForestGeom.

    The private proxy supplies sklearn cloning, fitted-state, and estimator-role
    conventions. The underlying proximity-forests estimator remains responsible
    for fitting, leaf assignment, and exact per-tree sampling metadata.
    """

    def __init__(self, estimator, weight_scheme=None):
        if isinstance(estimator, ProximityForestClassifier):
            proxy = _ProximityForestClassifierProxy(estimator)
        elif isinstance(estimator, ProximityForestRegressor):
            proxy = _ProximityForestRegressorProxy(estimator)
        elif isinstance(estimator, ProximityIsolationForest):
            proxy = _ProximityIsolationForestProxy(estimator)
        elif isinstance(estimator, _ProximityForestProxyBase):
            proxy = estimator
        else:
            raise TypeError(
                "Expected ProximityForestClassifier, ProximityForestRegressor, "
                "or ProximityIsolationForest."
            )
        super().__init__(proxy, weight_scheme=weight_scheme)

    @property
    def proximity_forest(self):
        """Return the underlying proximity-forests estimator."""
        return self.estimator.estimator

    def validate_weight_scheme(self, weight_scheme):
        super().validate_weight_scheme(weight_scheme)
        if (
            weight_scheme in {"oob", "gap"}
            and self.proximity_forest.sampling == "full"
        ):
            raise ValueError(
                f"weight_scheme='{weight_scheme}' requires proximity-forests "
                "sampling='bootstrap' or sampling='subsample'; sampling='full' "
                "has no out-of-bag observations."
            )
        return self

    def get_leaf_matrix(self, X):
        """Return stable terminal-node identifiers with shape (N, T)."""
        leaves = np.asarray(self.estimator.apply(X))
        if leaves.ndim != 2:
            raise ValueError(
                "proximity-forests apply(X) must return a two-dimensional "
                "sample-by-tree leaf matrix."
            )
        return leaves.astype(np.int32, copy=False)

    def get_oob_mask(self, X_train=None, sample_weight=None):
        """Return exact fitted OOB membership with shape (N_train, T)."""
        del X_train, sample_weight
        mask = np.asarray(self.estimator.get_oob_mask(), dtype=np.int8)
        self._validate_sampling_shape(mask, "OOB mask")
        return mask

    def get_in_bag_counts(self, X_train=None, sample_weight=None):
        """Return exact fitted multiplicities with shape (N_train, T)."""
        del X_train, sample_weight
        counts = np.asarray(
            self.estimator.get_in_bag_counts(), dtype=np.float32
        )
        self._validate_sampling_shape(counts, "in-bag count matrix")
        return counts

    def _validate_sampling_shape(self, values, name):
        if (
            values.ndim != 2
            or values.shape[1] != int(self.proximity_forest.n_estimators)
        ):
            raise ValueError(
                f"proximity-forests {name} must have shape "
                "(n_samples_fit, n_estimators)."
            )
