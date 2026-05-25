import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import accuracy_score, mean_squared_error

from forestgeom import ForestProximity

from tests.fixtures.constants import RF_ET_WEIGHT_SCHEMES
from tests.prediction_helpers import (
    normalize_rows,
    predict_classifier_from_proximity,
    predict_regression_from_proximity,
)


@pytest.mark.integration
@pytest.mark.parametrize("forest_fixture", ["rf_classifier", "et_classifier"])
@pytest.mark.parametrize("data_fixture", ["classification_data_10k"])
def test_gap_is_closest_to_base_classifier_error(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, y_test = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    base_forest = clone(forest)
    base_forest.fit(X_train, y_train)
    base_error = 1.0 - accuracy_score(y_test, base_forest.predict(X_test))

    scheme_errors = {}
    for scheme in RF_ET_WEIGHT_SCHEMES:
        proximity_model = ForestProximity(forest=clone(forest), weight_scheme=scheme)
        proximity_model.fit(X_train, y_train)
        P = proximity_model.transform(X_test)
        if scheme not in {"gap", "kerf"}:
            P = normalize_rows(P)
        preds, _ = predict_classifier_from_proximity(P, y_train, base_forest.classes_)
        scheme_errors[scheme] = abs((1.0 - accuracy_score(y_test, preds)) - base_error)

    assert scheme_errors["gap"] <= min(v for k, v in scheme_errors.items() if k != "gap") + 1e-8


@pytest.mark.integration
@pytest.mark.parametrize("forest_fixture", ["rf_regressor", "et_regressor"])
@pytest.mark.parametrize("data_fixture", ["regression_data_10k"])
def test_gap_is_closest_to_base_regressor_mse(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, y_test = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    base_forest = clone(forest)
    base_forest.fit(X_train, y_train)
    base_mse = mean_squared_error(y_test, base_forest.predict(X_test))

    scheme_mses = {}
    for scheme in RF_ET_WEIGHT_SCHEMES:
        proximity_model = ForestProximity(forest=clone(forest), weight_scheme=scheme)
        proximity_model.fit(X_train, y_train)
        P = proximity_model.transform(X_test)
        if scheme not in {"gap", "kerf"}:
            P = normalize_rows(P)
        preds = predict_regression_from_proximity(P, y_train)
        scheme_mses[scheme] = abs(mean_squared_error(y_test, preds) - base_mse)

    assert scheme_mses["gap"] <= min(v for k, v in scheme_mses.items() if k != "gap") + 1e-8
