import numpy as np
import pytest

proximity_forests = pytest.importorskip("proximity_forests")

from forestgeom import Proximity
from forestgeom.adapters import make_adapter
from forestgeom.adapters.proximity_forests import ProximityForestsAdapter


PFClassifier = proximity_forests.ProximityForestClassifier
PFRegressor = proximity_forests.ProximityForestRegressor
PFIsolation = proximity_forests.ProximityIsolationForest
SCHEMES = ("uniform", "kerf", "oob", "gap")


def data():
    rng = np.random.default_rng(9817)
    X = np.ascontiguousarray(rng.normal(size=(64, 6)), dtype=np.float64)
    query = np.ascontiguousarray(rng.normal(size=(9, 6)), dtype=np.float64)
    yc = (X[:, 0] + 0.2 * X[:, 1] > 0).astype(np.int64)
    yr = X[:, 0] - 0.5 * X[:, 1]
    return X, query, yc, yr


def estimator(kind, sampling="bootstrap"):
    common = dict(
        n_estimators=17,
        n_candidates=3,
        sampling=sampling,
        sample_size=None,
        random_state=37,
        n_jobs=1,
    )
    if kind == "classifier":
        return PFClassifier(**common)
    if kind == "regressor":
        return PFRegressor(**common)
    return PFIsolation(**common)


def fit(kind, scheme, sampling="bootstrap"):
    X, query, yc, yr = data()
    y = yc if kind == "classifier" else yr if kind == "regressor" else None
    model = Proximity(estimator(kind, sampling), scheme).fit(X, y)
    return model, X, query


@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
def test_dispatches_all_proximity_forest_families(kind):
    adapter = make_adapter(estimator(kind), weight_scheme="uniform")
    assert isinstance(adapter, ProximityForestsAdapter)


@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
@pytest.mark.parametrize("scheme", SCHEMES)
def test_all_nonboosted_schemes_fit_and_transform(kind, scheme):
    model, X, query = fit(kind, scheme)
    assert isinstance(model.forest_, ProximityForestsAdapter)
    assert model.cache_.leaf_matrix.shape == (X.shape[0], 17)
    train = model.training_proximity(return_dense=False)
    test = model.transform(query, return_dense=False)
    assert train.shape == (X.shape[0], X.shape[0])
    assert test.shape == (query.shape[0], X.shape[0])
    assert train.dtype == np.float32
    assert test.dtype == np.float32
    assert np.isfinite(train.data).all()
    assert np.isfinite(test.data).all()


@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
@pytest.mark.parametrize("scheme", ["oob", "gap"])
def test_subsample_supports_sampling_aware_schemes(kind, scheme):
    model, X, query = fit(kind, scheme, sampling="subsample")
    counts = model.cache_.inbag_counts if scheme == "gap" else None
    if counts is not None:
        assert np.all(counts <= 1)
    assert model.transform(query).shape == (query.shape[0], X.shape[0])


@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
@pytest.mark.parametrize("scheme", ["oob", "gap"])
def test_full_sampling_rejects_sampling_aware_schemes(kind, scheme):
    X, _, yc, yr = data()
    y = yc if kind == "classifier" else yr if kind == "regressor" else None
    with pytest.raises(ValueError, match="sampling='full'"):
        Proximity(estimator(kind, "full"), scheme).fit(X, y)


@pytest.mark.parametrize("kind", ["classifier", "regressor", "isolation"])
def test_boosted_is_not_supported(kind):
    with pytest.raises(ValueError, match="does not support"):
        make_adapter(estimator(kind), weight_scheme="boosted")


def test_gap_uses_exact_bootstrap_multiplicity_and_is_row_stochastic():
    model, X, query = fit("classifier", "gap")
    counts = model.cache_.inbag_counts
    assert np.any(counts > 1), "fixture must distinguish multiplicity from membership"
    np.testing.assert_array_equal(model.cache_.oob_mask, counts == 0)
    train = model.training_proximity(return_dense=True)
    test = model.transform(query, return_dense=True)
    np.testing.assert_allclose(train.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(test.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)


def test_adapter_matches_public_sampling_metadata_exactly():
    X, _, yc, _ = data()
    fitted = PFClassifier(
        n_estimators=13,
        sampling="bootstrap",
        random_state=71,
        n_jobs=1,
    ).fit(X, yc)
    adapter = ProximityForestsAdapter(fitted, weight_scheme="gap")
    np.testing.assert_array_equal(adapter.get_leaf_matrix(X), fitted.apply(X))
    np.testing.assert_array_equal(adapter.get_oob_mask(X), fitted.get_oob_mask())
    np.testing.assert_array_equal(
        adapter.get_in_bag_counts(X), fitted.get_in_bag_counts()
    )



def test_unfitted_backend_does_not_require_sklearn_tags():
    from sklearn.base import clone, is_classifier
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError

    raw = estimator("classifier")
    assert not hasattr(raw, "__sklearn_tags__")
    adapter = make_adapter(raw, weight_scheme="uniform")
    assert adapter.proximity_forest is raw
    assert is_classifier(adapter.estimator)
    with pytest.raises(NotFittedError):
        check_is_fitted(adapter.estimator)

    copied = clone(adapter.estimator)
    assert type(copied) is type(adapter.estimator)
    assert type(copied.estimator) is type(raw)
    assert copied.estimator is not raw
    assert copied.estimator.get_params(deep=False) == raw.get_params(deep=False)


def test_fitted_proximity_forest_is_reused_without_refitting(monkeypatch):
    X, _, yc, _ = data()
    fitted = estimator("classifier").fit(X, yc)

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("fitted proximity forest should not be refit")

    monkeypatch.setattr(fitted, "fit", fail_if_refit)
    model = Proximity(fitted, "uniform").fit(X, yc)
    assert model.forest_.proximity_forest is fitted
    np.testing.assert_array_equal(model.classes_, fitted.classes_)


def test_classifier_regressor_and_isolation_roles_are_preserved():
    X, _, yc, yr = data()
    classifier = Proximity(estimator("classifier"), "uniform").fit(X, yc)
    regressor = Proximity(estimator("regressor"), "uniform").fit(X, yr)
    isolation = Proximity(estimator("isolation"), "uniform").fit(X)
    np.testing.assert_array_equal(classifier.classes_, np.unique(yc))
    assert regressor.classes_ is None
    assert isolation.classes_ is None


def test_proxy_roles_match_pf_estimator_families():
    from sklearn.base import is_classifier, is_regressor

    classifier = make_adapter(estimator("classifier"), weight_scheme="uniform")
    regressor = make_adapter(estimator("regressor"), weight_scheme="uniform")
    isolation = make_adapter(estimator("isolation"), weight_scheme="uniform")

    assert is_classifier(classifier.estimator)
    assert is_regressor(regressor.estimator)
    assert not is_classifier(isolation.estimator)
    assert not is_regressor(isolation.estimator)
