import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from forestgeom import ForestProximity


class FakeEstimator:
    """A minimal unsupported estimator used to trigger make_adapter TypeError."""
    pass


def test_fit_forest_raises_typeerror_for_unsupported_estimator():
    X = np.zeros((10, 2), dtype=np.float32)
    y = np.arange(10)

    enc = ForestProximity(forest=FakeEstimator())

    with pytest.raises(TypeError) as exc:
        enc._fit_forest(X, y)

    assert "Unsupported forest estimator" in str(exc.value)


def test_fit_forest_reuses_fitted_estimator_without_refitting(monkeypatch):
    """
    ``monkeypatch`` is pytest's temporary patching fixture.

    Here it replaces ``forest.fit`` with a failing function so the test verifies
    that ForestProximity reuses an already fitted estimator without refitting it.
    """
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.tile([0, 1], 10)

    forest = RandomForestClassifier(
        n_estimators=5,
        bootstrap=True,
        random_state=0,
        n_jobs=1,
    ).fit(X, y)

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("fitted forest should not be refit")

    monkeypatch.setattr(forest, "fit", fail_if_refit)

    enc = ForestProximity(forest=forest, weight_scheme="uniform").fit(X, y)

    assert enc.forest_.estimator is forest
    assert enc.cache_ is not None
    np.testing.assert_array_equal(enc.classes_, forest.classes_)
