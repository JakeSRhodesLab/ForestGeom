import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state

from .base import EnsembleAdapter
from ..maps import (
    attach_bootstrap_stats,
    attach_inv_inbag_leaf_mass,
    build_Q_matrix,
    build_W_matrix,
    initialize_cache,
)

try:
    from sklearn.utils.validation import validate_data
except ImportError:  # pragma: no cover - sklearn<1.6 compatibility
    def validate_data(estimator, **kwargs):
        return estimator._validate_data(**kwargs)


def _suppress_aeon_validate_data_warning():
    return warnings.catch_warnings()


class AeonRotationForestAdapter(EnsembleAdapter):
    """
    Adapter for aeon RotationForest classifiers and regressors.

    aeon's RotationForest stores fitted trees in estimator-specific PCA feature
    spaces and does not expose a public ``apply`` method. For OOB/GAP schemes,
    the adapter follows GeneralForestProximities: it rebuilds bootstrap
    surrogate trees in those transformed spaces and uses their leaf assignments
    for training OOB/GAP maps. GAP out-of-sample extension keeps the original
    fitted RotationForest leaf space so proximity-weighted predictions match
    the fitted forest's predictions.
    """

    supported_weight_schemes = {"uniform", "kerf", "oob", "gap"}

    def __init__(self, estimator, weight_scheme=None):
        super().__init__(estimator, weight_scheme=weight_scheme)
        self._active_weight_scheme = weight_scheme
        self._oob_surrogate_estimators_ = None
        self._oob_surrogate_leaf_matrix_ = None
        self._oob_surrogate_n_nodes_per_tree_ = None
        self._oob_mask_ = None
        self._in_bag_counts_ = None
        self._rotation_original_gap_cache_ = None

    def fit(self, X, y=None, **fit_kwargs):
        self.estimator = clone(self.estimator)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*BaseEstimator\._validate_data.*",
                category=FutureWarning,
            )
            self.estimator.fit(X, y, **fit_kwargs)
        return self

    def validate_weight_scheme(self, weight_scheme):
        super().validate_weight_scheme(weight_scheme)

        if weight_scheme in {"oob", "gap"}:
            train_fn = self._get_train_estimate_fn(required=False)
            if train_fn is None:
                raise ValueError(
                    f"weight_scheme='{weight_scheme}' requires aeon "
                    "RotationForest internals `_train_probas_for_estimator` "
                    "or `_train_preds_for_estimator`."
                )

        return self

    def _check_rotation_forest_fitted(self):
        if not hasattr(self.estimator, "estimators_"):
            raise NotFittedError(
                f"{type(self.estimator).__name__} must be fitted before leaf "
                "assignments can be computed."
            )

    def _get_train_estimate_fn(self, required=True):
        train_fn = getattr(self.estimator, "_train_probas_for_estimator", None)
        if train_fn is None:
            train_fn = getattr(self.estimator, "_train_preds_for_estimator", None)

        if required and not callable(train_fn):
            raise RuntimeError(
                "aeon RotationForest OOB/GAP support requires either "
                "`_train_probas_for_estimator` or `_train_preds_for_estimator`."
            )

        return train_fn if callable(train_fn) else None

    def _check_required_internals(self):
        required = ("_pcas", "_groups", "_useful_atts", "_min", "_ptp", "_check_X")
        missing = [name for name in required if not hasattr(self.estimator, name)]
        if missing:
            raise RuntimeError(
                "Unsupported aeon RotationForest internals; missing "
                + ", ".join(missing)
                + "."
            )

    def _prepare_X(self, X):
        self._check_rotation_forest_fitted()
        self._check_required_internals()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*BaseEstimator\._validate_data.*",
                category=FutureWarning,
            )
            X_proc = self.estimator._check_X(X)
            X_proc = validate_data(
                self.estimator,
                X=X_proc,
                reset=False,
                accept_sparse=False,
            )

        X_proc = X_proc[:, self.estimator._useful_atts]
        return (X_proc - self.estimator._min) / self.estimator._ptp

    def _transformed_per_estimator(self, X):
        X_proc = self._prepare_X(X)
        transformed = []

        for pcas, groups in zip(self.estimator._pcas, self.estimator._groups):
            X_t = np.concatenate(
                [
                    pca.transform(X_proc[:, group])
                    for pca, group in zip(pcas, groups)
                ],
                axis=1,
            ).astype(np.float32)
            X_t = np.nan_to_num(
                X_t,
                copy=False,
                nan=0.0,
                posinf=float(np.finfo(np.float32).max),
                neginf=float(np.finfo(np.float32).min),
            )
            transformed.append(X_t)

        return transformed

    def _apply_original_leaf_matrix(self, X):
        X_t = self._transformed_per_estimator(X)
        leaves = [
            np.asarray(estimator.apply(X_tree))
            for estimator, X_tree in zip(self.estimator.estimators_, X_t)
        ]
        return np.column_stack(leaves).astype(np.int32)

    def _apply_surrogate_leaf_matrix(self, X):
        if self._oob_surrogate_estimators_ is None:
            raise RuntimeError(
                "RotationForest surrogate estimators are not initialized."
            )

        X_t = self._transformed_per_estimator(X)
        leaves = [
            np.asarray(surrogate.apply(X_tree))
            for surrogate, X_tree in zip(self._oob_surrogate_estimators_, X_t)
        ]
        return np.column_stack(leaves).astype(np.int32)

    def _node_counts(self, estimators):
        counts = []
        for estimator in estimators:
            tree = getattr(estimator, "tree_", None)
            if tree is None:
                raise RuntimeError(
                    "RotationForest base estimator does not expose `tree_`."
                )
            counts.append(int(tree.node_count))
        return counts

    def _fit_surrogate_bootstrap_trees(self, X, y):
        if y is None:
            raise ValueError(
                "aeon RotationForest OOB/GAP proximities require supervised "
                "targets `y`."
            )

        self._check_rotation_forest_fitted()
        train_estimate_fn = self._get_train_estimate_fn(required=True)
        X_t = self._transformed_per_estimator(X)
        y_arr = np.asarray(y)

        n_samples = int(X_t[0].shape[0])
        n_cases = int(getattr(self.estimator, "n_cases_", n_samples))
        if n_cases != n_samples:
            raise ValueError(
                "RotationForest n_cases_ does not match fit sample size."
            )

        n_trees = len(self.estimator.estimators_)
        rng = check_random_state(getattr(self.estimator, "random_state", None))
        max_int = np.iinfo(np.int32).max

        oob_mask = np.zeros((n_samples, n_trees), dtype=np.int8)
        in_bag_counts = np.zeros((n_samples, n_trees), dtype=np.float32)
        surrogate_estimators = []
        surrogate_leaves = []

        base_estimator = getattr(self.estimator, "_base_estimator", None)

        for t in range(n_trees):
            est_seed = rng.randint(max_int)
            est_rng = check_random_state(est_seed)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*BaseEstimator\._validate_data.*",
                    category=FutureWarning,
                )
                train_result = train_estimate_fn(
                    X_t,
                    y_arr,
                    t,
                    check_random_state(est_seed),
                )

            if (
                not isinstance(train_result, (tuple, list))
                or len(train_result) != 2
            ):
                raise RuntimeError(
                    "RotationForest train-estimate function returned an "
                    "unexpected result."
                )

            oob_idx = np.asarray(train_result[1], dtype=np.int64)
            sampled = est_rng.choice(n_cases, size=n_cases, replace=True)

            surrogate = clone(
                base_estimator
                if base_estimator is not None
                else self.estimator.estimators_[t]
            )
            if hasattr(surrogate, "random_state"):
                surrogate.random_state = est_rng
            surrogate.fit(X_t[t][sampled], y_arr[sampled])

            surrogate_estimators.append(surrogate)
            surrogate_leaves.append(np.asarray(surrogate.apply(X_t[t])))

            idx, cnt = np.unique(sampled, return_counts=True)
            in_bag_counts[idx, t] = cnt.astype(np.float32)
            oob_mask[oob_idx, t] = 1

        self._oob_surrogate_estimators_ = surrogate_estimators
        self._oob_surrogate_leaf_matrix_ = np.column_stack(surrogate_leaves).astype(
            np.int32
        )
        self._oob_surrogate_n_nodes_per_tree_ = self._node_counts(
            surrogate_estimators
        )
        self._oob_mask_ = oob_mask
        self._in_bag_counts_ = in_bag_counts

    def _prepare_original_gap_cache(self, X):
        original_leaves = self._apply_original_leaf_matrix(X)
        original_cache = initialize_cache(
            leaf_matrix=original_leaves,
            n_nodes_per_tree=self._node_counts(self.estimator.estimators_),
            n_samples=original_leaves.shape[0],
        )
        inbag_ones = np.ones(original_leaves.shape, dtype=np.float32)
        oob_ones = np.ones(original_leaves.shape, dtype=np.int8)
        attach_bootstrap_stats(
            original_cache,
            oob_mask=oob_ones,
            inbag_counts=inbag_ones,
        )
        attach_inv_inbag_leaf_mass(original_cache)
        original_cache.W_mat = build_W_matrix(original_cache, "gap")
        self._rotation_original_gap_cache_ = original_cache

    def prepare_proximity_cache(
        self,
        X,
        y=None,
        weight_scheme=None,
        sample_weight=None,
    ):
        self._active_weight_scheme = weight_scheme

        if weight_scheme not in {"oob", "gap"}:
            self._oob_surrogate_estimators_ = None
            self._oob_surrogate_leaf_matrix_ = None
            self._oob_surrogate_n_nodes_per_tree_ = None
            self._oob_mask_ = None
            self._in_bag_counts_ = None
            self._rotation_original_gap_cache_ = None
            return {}

        self._fit_surrogate_bootstrap_trees(X, y)
        if weight_scheme == "gap":
            self._prepare_original_gap_cache(X)

        prepared = {
            "leaf_matrix": self._oob_surrogate_leaf_matrix_,
            "n_nodes_per_tree": self._oob_surrogate_n_nodes_per_tree_,
            "oob_mask": self._oob_mask_,
        }
        if weight_scheme == "gap":
            prepared["inbag_counts"] = self._in_bag_counts_

        return prepared

    def get_leaf_matrix(self, X):
        if (
            self._active_weight_scheme in {"oob", "gap"}
            and self._oob_surrogate_estimators_ is not None
        ):
            return self._apply_surrogate_leaf_matrix(X)
        return self._apply_original_leaf_matrix(X)

    def get_n_nodes_per_tree(self):
        if (
            self._active_weight_scheme in {"oob", "gap"}
            and self._oob_surrogate_n_nodes_per_tree_ is not None
        ):
            return self._oob_surrogate_n_nodes_per_tree_
        self._check_rotation_forest_fitted()
        return self._node_counts(self.estimator.estimators_)

    def get_oob_mask(self, X_train=None, sample_weight=None):
        if self._oob_mask_ is None:
            raise RuntimeError(
                "RotationForest OOB mask has not been prepared. "
                "Call prepare_proximity_cache first."
            )
        return self._oob_mask_

    def get_in_bag_counts(self, X_train=None, sample_weight=None):
        if self._in_bag_counts_ is None:
            raise RuntimeError(
                "RotationForest in-bag counts have not been prepared. "
                "Call prepare_proximity_cache first."
            )
        return self._in_bag_counts_

    def transform_proximity(self, X, cache, weight_scheme):
        if weight_scheme != "gap" or self._rotation_original_gap_cache_ is None:
            return None

        original_cache = self._rotation_original_gap_cache_
        leaves = self._apply_original_leaf_matrix(X)
        Q = build_Q_matrix(
            original_cache,
            weight_scheme="gap",
            leaves=leaves,
            is_training=False,
        )
        return Q.dot(original_cache.W_mat.T)
