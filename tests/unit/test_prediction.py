import numpy as np
from sklearn.base import clone
import pytest

from forestgeom import ForestProximity
from tests.prediction_helpers import (
    predict_classifier_from_proximity,
    predict_regression_from_proximity,
)


@pytest.mark.parametrize("forest_fixture", ["rf_classifier", "et_classifier"])
@pytest.mark.parametrize("data_fixture", ["classification_data"])
def test_gap_matches_base_classifier_predictions(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, y_test = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    base_forest = clone(forest)
    base_forest.fit(X_train, y_train)

    proximity_model = ForestProximity(forest=clone(forest), weight_scheme="gap")
    proximity_model.fit(X_train, y_train)
    P = proximity_model.transform(X_test)
    proximity_preds, proba = predict_classifier_from_proximity(
        P, y_train, classes=base_forest.classes_
    )

    np.testing.assert_array_equal(proximity_preds, base_forest.predict(X_test))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(proba.shape[0]), atol=1e-6)
    np.testing.assert_array_equal(proximity_model.classes_, base_forest.classes_)


@pytest.mark.parametrize("forest_fixture", ["rf_regressor", "et_regressor"])
@pytest.mark.parametrize("data_fixture", ["regression_data"])
def test_gap_matches_base_regressor_predictions(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, y_test = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    base_forest = clone(forest)
    base_forest.fit(X_train, y_train)

    proximity_model = ForestProximity(forest=clone(forest), weight_scheme="gap")
    proximity_model.fit(X_train, y_train)
    P = proximity_model.transform(X_test)
    proximity_preds = predict_regression_from_proximity(P, y_train)

    np.testing.assert_allclose(
        proximity_preds,
        base_forest.predict(X_test),
        rtol=1e-5,
        atol=1e-6,
    )
