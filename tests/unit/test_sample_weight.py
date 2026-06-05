import numpy as np
from sklearn.base import clone
import pytest

from forestgeom import ForestProximity
from tests.prediction_helpers import predict_regression_from_proximity
from forestgeom.adapters import rf_et_rte


@pytest.mark.parametrize("forest_fixture", ["rf_regressor", "et_regressor"])
@pytest.mark.parametrize("data_fixture", ["regression_data"])
def test_gap_respects_sample_weight(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, y_test = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    # create deterministic non-uniform sample weights
    rng = np.random.RandomState(0)
    sample_weight = rng.rand(len(y_train)) + 0.1

    # Fit base forest with sample_weight
    base_forest = clone(forest)
    base_forest.fit(X_train, y_train, sample_weight=sample_weight)
    base_pred = base_forest.predict(X_test)

    # Fit ForestProximity with the same sample_weight and check it matches
    proximity_model = ForestProximity(forest=clone(forest), weight_scheme="gap")
    proximity_model.fit(X_train, y_train, sample_weight=sample_weight)
    P = proximity_model.transform(X_test)
    prox_pred = predict_regression_from_proximity(P, y_train)

    np.testing.assert_allclose(prox_pred, base_pred, rtol=1e-5, atol=1e-6)

    # Fit ForestProximity without sample_weight — predictions should differ
    proximity_unweighted = ForestProximity(forest=clone(forest), weight_scheme="gap")
    proximity_unweighted.fit(X_train, y_train)
    P2 = proximity_unweighted.transform(X_test)
    prox_pred2 = predict_regression_from_proximity(P2, y_train)

    assert not np.allclose(prox_pred2, base_pred, rtol=1e-5, atol=1e-6)

    # The check above (unweighted proximity differs from base forest fitted
    # with `sample_weight`) is the robust assertion that demonstrates the
    # importance of preserving `sample_weight` when building GAP proximities.
