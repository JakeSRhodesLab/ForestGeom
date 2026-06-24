import numpy as np

from .base import EnsembleAdapter

try:
    from sklearn.ensemble._forest import (
        _generate_sample_indices as _skl_generate_sample_indices,
    )
    from sklearn.ensemble._forest import (
        _generate_unsampled_indices as _skl_generate_unsampled_indices,
    )
except Exception:  # pragma: no cover - sklearn private API availability
    _skl_generate_sample_indices = None
    _skl_generate_unsampled_indices = None


def _generate_sample_indices(
    random_state,
    n_samples,
    n_samples_bootstrap,
    sample_weight=None,
):
    if _skl_generate_sample_indices is None:
        raise RuntimeError("sklearn bootstrap helper is unavailable.")
    try:
        return _skl_generate_sample_indices(
            random_state,
            n_samples,
            n_samples_bootstrap,
            sample_weight,
        )
    except TypeError:
        return _skl_generate_sample_indices(
            random_state,
            n_samples,
            n_samples_bootstrap,
        )


def _generate_unsampled_indices(
    random_state,
    n_samples,
    n_samples_bootstrap,
    sample_weight=None,
):
    if _skl_generate_unsampled_indices is None:
        raise RuntimeError("sklearn bootstrap helper is unavailable.")
    try:
        return _skl_generate_unsampled_indices(
            random_state,
            n_samples,
            n_samples_bootstrap,
            sample_weight,
        )
    except TypeError:
        return _skl_generate_unsampled_indices(
            random_state,
            n_samples,
            n_samples_bootstrap,
        )


def patch_treeple_regressor_tags():
    """
    Patch treeple 0.10.x regressor tags across sklearn versions.
    """
    try:
        from treeple._lib.sklearn.ensemble._forest import ForestRegressor
    except Exception:
        return False

    if getattr(ForestRegressor, "_forestgeom_tags_patch", False):
        return True

    def _patched_sklearn_tags(self):
        tags = super(ForestRegressor, self).__sklearn_tags__()
        regressor_tags = getattr(tags, "regressor_tags", None)
        if regressor_tags is not None:
            try:
                regressor_tags.multi_label = True
            except AttributeError:
                pass
        return tags

    ForestRegressor.__sklearn_tags__ = _patched_sklearn_tags
    ForestRegressor._forestgeom_tags_patch = True
    return True


def patch_treeple_sample_weight_signature():
    """
    Patch treeple 0.10.x helper imports for sklearn's keyword-only dtype arg.
    """
    try:
        from sklearn.utils.validation import (
            _check_sample_weight as sklearn_check_sample_weight,
        )
        from treeple._lib.sklearn.ensemble import _forest as tree_forest
        from treeple._lib.sklearn.tree import _classes as tree_classes
    except Exception:
        return False

    if getattr(tree_classes, "_forestgeom_sample_weight_patch", False):
        return True

    def _compat_check_sample_weight(
        sample_weight,
        X,
        dtype=None,
        ensure_non_negative=False,
        copy=False,
    ):
        return sklearn_check_sample_weight(
            sample_weight,
            X,
            dtype=dtype,
            ensure_non_negative=ensure_non_negative,
            copy=copy,
        )

    tree_classes._check_sample_weight = _compat_check_sample_weight
    tree_forest._check_sample_weight = _compat_check_sample_weight
    tree_classes._forestgeom_sample_weight_patch = True
    tree_forest._forestgeom_sample_weight_patch = True
    return True


patch_treeple_regressor_tags()
patch_treeple_sample_weight_signature()


class TreepleForestAdapter(EnsembleAdapter):
    """
    Adapter for treeple forest-like estimators.
    """

    supported_weight_schemes = {"uniform", "kerf", "oob", "gap"}

    def validate_weight_scheme(self, weight_scheme):
        super().validate_weight_scheme(weight_scheme)

        if (
            weight_scheme in {"oob", "gap"}
            and hasattr(self.estimator, "bootstrap")
            and not getattr(self.estimator, "bootstrap")
        ):
            raise ValueError(
                f"weight_scheme='{weight_scheme}' requires bootstrap=True "
                f"for {type(self.estimator).__name__}."
            )

        return self

    def _get_tree_list(self):
        estimators = getattr(self.estimator, "estimators_", None)
        if estimators is None:
            raise RuntimeError(
                f"{type(self.estimator).__name__} must expose estimators_ "
                "after fitting."
            )
        return list(np.asarray(estimators, dtype=object).ravel())

    def _normalize_leaf_matrix(self, leaves):
        leaves = np.asarray(leaves, dtype=np.int32)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        elif leaves.ndim > 2:
            leaves = leaves.reshape(leaves.shape[0], -1)
        return leaves

    def get_leaf_matrix(self, X):
        apply_fn = getattr(self.estimator, "apply", None)
        if callable(apply_fn):
            return self._normalize_leaf_matrix(apply_fn(X))

        tree_list = self._get_tree_list()
        features_per_estimator = getattr(self.estimator, "estimators_features_", None)

        leaves = []
        if features_per_estimator is None:
            for tree in tree_list:
                leaves.append(np.asarray(tree.apply(X)))
        else:
            for tree, features in zip(tree_list, features_per_estimator):
                leaves.append(np.asarray(tree.apply(X[:, features])))

        return self._normalize_leaf_matrix(np.column_stack(leaves))

    def get_n_nodes_per_tree(self):
        counts = []
        for tree in self._get_tree_list():
            tree_struct = getattr(tree, "tree_", None)
            if tree_struct is None:
                raise RuntimeError(
                    f"{type(tree).__name__} does not expose tree_.node_count."
                )
            counts.append(int(tree_struct.node_count))
        return counts

    def _get_n_samples_bootstrap(self, n_samples):
        explicit = getattr(self.estimator, "_n_samples_bootstrap", None)
        if explicit is not None:
            return int(explicit)

        max_samples = getattr(self.estimator, "max_samples", None)
        if max_samples is None:
            return int(n_samples)
        if isinstance(max_samples, float):
            return max(1, int(round(n_samples * max_samples)))
        return int(max_samples)

    def _sample_indices_from_estimators_samples(self):
        try:
            estimators_samples = getattr(self.estimator, "estimators_samples_", None)
        except AttributeError:
            return None

        if estimators_samples is None:
            return None

        samples = []
        for sample_indices in estimators_samples:
            if isinstance(sample_indices, tuple) and len(sample_indices) == 2:
                sample_indices = sample_indices[1]
            samples.append(np.asarray(sample_indices, dtype=np.int64))
        return samples

    def _sample_indices_from_getter(self):
        get_indices = getattr(self.estimator, "_get_estimators_indices", None)
        if not callable(get_indices):
            return None

        try:
            generated = list(get_indices() or [])
        except AttributeError:
            return None

        samples = []
        for estimator_index in generated:
            sample_indices = estimator_index
            if isinstance(estimator_index, tuple) and len(estimator_index) == 2:
                sample_indices = estimator_index[1]
            samples.append(np.asarray(sample_indices, dtype=np.int64))
        return samples

    def _bootstrap_sample_indices(self, X_train=None, sample_weight=None):
        samples = self._sample_indices_from_estimators_samples()
        if samples is not None:
            return samples

        samples = self._sample_indices_from_getter()
        if samples is not None:
            return samples

        if X_train is None:
            raise RuntimeError("X_train is required to reconstruct bootstrap samples.")

        n_samples = X_train.shape[0]
        n_samples_bootstrap = self._get_n_samples_bootstrap(n_samples)

        reconstructed = []
        for tree in self._get_tree_list():
            if not hasattr(tree, "random_state"):
                raise RuntimeError(
                    "Cannot reconstruct treeple bootstrap samples because "
                    "a base estimator does not expose random_state."
                )
            reconstructed.append(
                _generate_sample_indices(
                    tree.random_state,
                    n_samples,
                    n_samples_bootstrap,
                    sample_weight,
                )
            )
        return reconstructed

    def get_oob_mask(self, X_train=None, sample_weight=None):
        if X_train is None:
            raise RuntimeError("get_oob_mask requires X_train.")

        n_samples = X_train.shape[0]
        sample_indices = self._bootstrap_sample_indices(
            X_train=X_train,
            sample_weight=sample_weight,
        )
        oob_mask = np.ones((n_samples, len(sample_indices)), dtype=np.int8)

        for t, sampled in enumerate(sample_indices):
            sampled = sampled[(sampled >= 0) & (sampled < n_samples)]
            oob_mask[sampled, t] = 0

        return oob_mask

    def get_in_bag_counts(self, X_train=None, sample_weight=None):
        if X_train is None:
            raise RuntimeError("get_in_bag_counts requires X_train.")

        n_samples = X_train.shape[0]
        sample_indices = self._bootstrap_sample_indices(
            X_train=X_train,
            sample_weight=sample_weight,
        )
        counts = np.zeros((n_samples, len(sample_indices)), dtype=np.float32)

        for t, sampled in enumerate(sample_indices):
            sampled = sampled[(sampled >= 0) & (sampled < n_samples)]
            binc = np.bincount(sampled, minlength=n_samples).astype(np.float32)
            counts[:, t] = binc

        return counts
