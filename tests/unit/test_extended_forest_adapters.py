import numpy as np
import pytest
from sklearn.base import clone

from forestgeom import ForestProximity
from tests.fixtures.constants import (
    AEON_FORESTS_AND_DATA,
    AEON_WEIGHT_SCHEMES,
    TREEPLE_SUPERVISED_FORESTS_AND_DATA,
    TREEPLE_UNSUPERVISED_FORESTS_AND_DATA,
    TREEPLE_WEIGHT_SCHEMES,
)
from tests.prediction_helpers import (
    predict_classifier_from_proximity,
    predict_regression_from_proximity,
)


def _assert_valid_proximity_blocks(enc, X_train, X_test, y_train=None):
    K_train = enc.training_proximity(return_dense=True)
    K_test = enc.transform(X_test, return_dense=True)

    assert K_train.shape == (X_train.shape[0], X_train.shape[0])
    assert K_test.shape == (X_test.shape[0], X_train.shape[0])
    assert np.isfinite(K_train).all()
    assert np.isfinite(K_test).all()
    assert K_train.min() >= -1e-8
    assert K_test.min() >= -1e-8


@pytest.mark.parametrize("forest_fixture,data_fixture", AEON_FORESTS_AND_DATA)
@pytest.mark.parametrize("weight_scheme", AEON_WEIGHT_SCHEMES)
def test_aeon_rotation_forest_proximities_are_valid(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(
        X_train,
        y_train,
    )

    _assert_valid_proximity_blocks(enc, X_train, X_test, y_train)

    if weight_scheme in {"uniform", "kerf", "oob"}:
        K_train = enc.training_proximity(return_dense=True)
        np.testing.assert_allclose(K_train, K_train.T, rtol=1e-5, atol=1e-6)

    if weight_scheme == "gap":
        train_rows = enc.training_proximity(return_dense=True).sum(axis=1)
        test_rows = enc.transform(X_test, return_dense=True).sum(axis=1)
        np.testing.assert_allclose(train_rows, 1.0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(test_rows, 1.0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "forest_fixture,data_fixture",
    TREEPLE_SUPERVISED_FORESTS_AND_DATA,
)
@pytest.mark.parametrize("weight_scheme", TREEPLE_WEIGHT_SCHEMES)
def test_treeple_supervised_forest_proximities_are_valid(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(
        X_train,
        y_train,
    )

    _assert_valid_proximity_blocks(enc, X_train, X_test, y_train)

    if weight_scheme in {"uniform", "kerf", "oob"}:
        K_train = enc.training_proximity(return_dense=True)
        np.testing.assert_allclose(K_train, K_train.T, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "forest_fixture,data_fixture",
    TREEPLE_UNSUPERVISED_FORESTS_AND_DATA,
)
@pytest.mark.parametrize("weight_scheme", ["uniform", "kerf"])
def test_treeple_unsupervised_forest_proximities_are_valid(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, X_test, _, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(X_train)

    _assert_valid_proximity_blocks(enc, X_train, X_test)


@pytest.mark.parametrize(
    "forest_fixture,data_fixture",
    [
        case
        for case in AEON_FORESTS_AND_DATA
        if case[1] == "classification_data"
    ],
)
def test_aeon_gap_extension_matches_rotation_forest_classifier_predictions(
    request,
    forest_fixture,
    data_fixture,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    base_forest = clone(forest).fit(X_train, y_train)
    enc = ForestProximity(
        forest=clone(forest),
        weight_scheme="gap",
    ).fit(X_train, y_train)

    P = enc.transform(X_test)
    preds, proba = predict_classifier_from_proximity(
        P,
        y_train,
        classes=base_forest.classes_,
    )

    np.testing.assert_array_equal(preds, base_forest.predict(X_test))
    np.testing.assert_allclose(
        proba,
        base_forest.predict_proba(X_test),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "forest_fixture,data_fixture",
    [
        case
        for case in AEON_FORESTS_AND_DATA
        if case[1] == "regression_data"
    ],
)
def test_aeon_gap_extension_matches_rotation_forest_regressor_predictions(
    request,
    forest_fixture,
    data_fixture,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    base_forest = clone(forest).fit(X_train, y_train)
    enc = ForestProximity(
        forest=clone(forest),
        weight_scheme="gap",
    ).fit(X_train, y_train)

    P = enc.transform(X_test)
    preds = predict_regression_from_proximity(P, y_train)

    np.testing.assert_allclose(
        preds,
        base_forest.predict(X_test),
        rtol=1e-4,
        atol=1e-4,
    )
