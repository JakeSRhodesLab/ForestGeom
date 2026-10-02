import numpy as np
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from forestgeom import Proximity
from forestgeom.adapters import make_adapter
from forestgeom.adapters.base import EnsembleAdapter
from forestgeom.adapters.rf_et_rte import RFETAdapter


class CustomForest(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=20):
        self.n_estimators = n_estimators

    def fit(self, X, y, sample_weight=None):
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=42
        ).fit(X, y, sample_weight=sample_weight)
        self.adapter_ = RFETAdapter(self.model_)
        self.classes_ = self.model_.classes_
        return self

    def get_leaf_matrix(self, X):
        return self.adapter_.get_leaf_matrix(X)

    def get_oob_mask(self, X_train=None, sample_weight=None):
        return self.adapter_.get_oob_mask(X_train, sample_weight=sample_weight)

    def get_in_bag_counts(self, X_train=None, sample_weight=None):
        return self.adapter_.get_in_bag_counts(X_train, sample_weight=sample_weight)

    def get_tree_weights(self, X_ref):
        return np.full(self.n_estimators, 1 / self.n_estimators, dtype=np.float32)


@pytest.mark.parametrize("scheme", ["uniform", "kerf", "oob", "gap", "boosted"])
def test_custom_forest_matches_builtin_outputs(scheme):
    X = np.arange(60, dtype=np.float32).reshape(30, 2)
    y = np.arange(30) % 2
    weights = np.linspace(1, 2, len(y))
    forest = CustomForest()
    custom = Proximity(forest, scheme).fit(X, y, sample_weight=weights)
    builtin = Proximity(
        custom.forest_.estimator.model_, "uniform" if scheme == "boosted" else scheme
    ).fit(X, y, sample_weight=weights)

    assert not hasattr(forest, "model_")  # Unfitted input is cloned.
    np.testing.assert_array_equal(custom.classes_, builtin.classes_)
    for operation in ("training_proximity", "transform"):
        args = () if operation == "training_proximity" else (X[:4],)
        actual = getattr(custom, operation)(*args)
        expected = getattr(builtin, operation)(*args)
        assert sparse.isspmatrix_csr(actual)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=1e-7)

    custom.set_weight_scheme("uniform")
    assert custom.query_map().shape == custom.reference_map().shape


@pytest.mark.parametrize("wrapped", [False, True])
def test_fitted_custom_forest_and_adapter_are_reused(wrapped, monkeypatch):
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20) % 2
    forest = CustomForest().fit(X, y)

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("fitted forest should not be refit")

    monkeypatch.setattr(forest, "fit", fail_if_refit)
    supplied = make_adapter(forest) if wrapped else forest
    proximity = Proximity(supplied).fit(X, y)
    assert proximity.forest_.estimator is forest
    if wrapped:
        assert proximity.forest_ is supplied


def test_leaf_only_forest_defaults_to_uniform_and_kerf():
    class LeafOnlyForest(BaseEstimator):
        def fit(self, X, y=None):
            self.fitted_ = True
            return self

        def get_leaf_matrix(self, X):
            return np.asarray(X, dtype=np.int32)

    X = np.array([[0, 0], [0, 1], [1, 1]])
    for scheme in ("uniform", "kerf"):
        proximity = Proximity(LeafOnlyForest(), scheme).fit(X)
        assert isinstance(proximity.forest_, EnsembleAdapter)
        assert proximity.joint_proximity(X[:1]).shape == (4, 4)
    with pytest.raises(ValueError, match="does not support"):
        Proximity(LeafOnlyForest(), "gap").fit(X)


def test_direct_adapter_detects_custom_forest_methods():
    adapter = EnsembleAdapter(CustomForest(), weight_scheme="gap")
    assert make_adapter(adapter, weight_scheme="boosted") is adapter


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "missing, scheme",
    [("get_leaf_matrix", "uniform"), ("get_oob_mask", "oob"),
     ("get_oob_mask", "gap"), ("get_in_bag_counts", "gap"),
     ("get_tree_weights", "boosted")],
)
def test_missing_methods_reject_schemes_before_fitting(wrapped, missing, scheme):
    forest = CustomForest()
    setattr(forest, missing, None)
    supplied = EnsembleAdapter(forest) if wrapped else forest
    with pytest.raises((TypeError, ValueError)):
        make_adapter(supplied, weight_scheme=scheme)
    assert not hasattr(forest, "model_")


def test_adapter_overrides_count_as_implementations():
    class LeafAdapter(EnsembleAdapter):
        def get_leaf_matrix(self, X):
            return np.asarray(X, dtype=np.int32)

    class OOBAdapter(LeafAdapter):
        def get_oob_mask(self, X_train=None, sample_weight=None):
            return np.ones_like(X_train, dtype=np.int8)

    adapter = OOBAdapter(BaseEstimator(), weight_scheme="oob")
    adapter.validate_weight_scheme("kerf")
    for scheme in ("gap", "boosted", "unknown"):
        with pytest.raises(ValueError, match="does not support"):
            adapter.validate_weight_scheme(scheme)
