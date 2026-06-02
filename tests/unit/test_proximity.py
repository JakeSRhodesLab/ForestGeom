import numpy as np
import pytest

from forestgeom import ForestProximity

from tests.fixtures.constants import (
    RF_ET_FORESTS_AND_DATA,
    RF_ET_WEIGHT_SCHEMES,
    RTE_FORESTS_AND_DATA,
)


@pytest.mark.parametrize("weight_scheme", ["uniform", "kerf"])
def test_random_trees_embedding_supports_unsupervised_proximities(
    request,
    weight_scheme,
):
    X_train, X_test, _, _ = request.getfixturevalue("classification_data")
    forest = request.getfixturevalue("random_trees_embedding")

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(X_train)

    K_train = enc.training_proximity(return_dense=True)
    K_test = enc.transform(X_test, return_dense=True)

    assert enc.y_ is None
    assert enc.classes_ is None
    assert K_train.shape == (X_train.shape[0], X_train.shape[0])
    assert K_test.shape == (X_test.shape[0], X_train.shape[0])
    assert np.allclose(K_train, K_train.T, rtol=1e-6, atol=1e-8)


def test_random_trees_embedding_fit_with_y_matches_fit_without_y(request):
    X_train, X_test, y_train, _ = request.getfixturevalue("classification_data")
    forest = request.getfixturevalue("random_trees_embedding")

    enc_without_y = ForestProximity(
        forest=forest,
        weight_scheme="uniform",
    ).fit(X_train)
    enc_with_y = ForestProximity(
        forest=forest,
        weight_scheme="uniform",
    ).fit(X_train, y_train)

    np.testing.assert_array_equal(
        enc_with_y.cache_.leaf_matrix,
        enc_without_y.cache_.leaf_matrix,
    )
    np.testing.assert_allclose(
        enc_with_y.query_map(return_dense=True),
        enc_without_y.query_map(return_dense=True),
    )
    np.testing.assert_allclose(
        enc_with_y.query_map(X_test, return_dense=True),
        enc_without_y.query_map(X_test, return_dense=True),
    )
    np.testing.assert_allclose(
        enc_with_y.training_proximity(return_dense=True),
        enc_without_y.training_proximity(return_dense=True),
    )


@pytest.mark.parametrize("weight_scheme", ["oob", "gap"])
def test_random_trees_embedding_rejects_bootstrap_weight_schemes(
    request,
    weight_scheme,
):
    X_train, _, _, _ = request.getfixturevalue("classification_data")
    forest = request.getfixturevalue("random_trees_embedding")

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme)

    with pytest.raises(ValueError, match="requires bootstrap=True"):
        enc.fit(X_train)


@pytest.mark.parametrize("forest_fixture,data_fixture", RF_ET_FORESTS_AND_DATA)
def test_gap_train_and_test_rows_sum_to_one_exactly(request, forest_fixture, data_fixture):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme="gap").fit(X_train, y_train)

    K_train = enc.training_proximity(return_dense=False)
    K_test = enc.transform(X_test, return_dense=False)

    train_row_sums = np.asarray(K_train.sum(axis=1)).ravel()
    test_row_sums = np.asarray(K_test.sum(axis=1)).ravel()

    assert np.allclose(train_row_sums, 1.0, rtol=1e-5, atol=1e-6)
    assert np.allclose(test_row_sums, 1.0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "forest_fixture,data_fixture",
    RF_ET_FORESTS_AND_DATA + RTE_FORESTS_AND_DATA,
)
def test_kerf_train_kernel_is_doubly_stochastic_and_test_rows_sum_to_one(
    request,
    forest_fixture,
    data_fixture,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme="kerf").fit(X_train, y_train)

    K_train = enc.training_proximity(return_dense=False)
    K_test = enc.transform(X_test, return_dense=False)

    train_row_sums = np.asarray(K_train.sum(axis=1)).ravel()
    train_col_sums = np.asarray(K_train.sum(axis=0)).ravel()
    test_row_sums = np.asarray(K_test.sum(axis=1)).ravel()

    assert np.allclose(train_row_sums, 1.0, rtol=1e-5, atol=1e-6)
    assert np.allclose(train_col_sums, 1.0, rtol=1e-5, atol=1e-6)
    assert np.allclose(test_row_sums, 1.0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("forest_fixture,data_fixture,weight_scheme", [
    (f, d, s)
    for f, d in RF_ET_FORESTS_AND_DATA
    for s in RF_ET_WEIGHT_SCHEMES
    if s != "gap"
])
def test_training_kernel_is_symmetric(request, forest_fixture, data_fixture, weight_scheme):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    K = enc.training_proximity(return_dense=True)

    assert K.shape[0] == K.shape[1]
    assert np.allclose(K, K.T, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("forest_fixture,data_fixture", RF_ET_FORESTS_AND_DATA)
def test_gap_force_symmetric_kernel_is_symmetric(request, forest_fixture, data_fixture):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme="gap").fit(X_train, y_train)

    K = enc.training_proximity(force_symmetric=True, return_dense=True)

    assert K.shape[0] == K.shape[1]
    assert np.allclose(K, K.T, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("forest_fixture,data_fixture", RF_ET_FORESTS_AND_DATA)
def test_adjust_diagonal_matches_expected_gap_training_diagonal(
    request,
    forest_fixture,
    data_fixture,
):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme="gap").fit(X_train, y_train)

    K = enc.training_proximity(adjust_diagonal=True, return_dense=True)
    diag = np.diag(K)

    ref_mass = np.asarray(enc.reference_map(return_dense=False).sum(axis=1)).ravel()
    inbag_tree_counts = (enc.cache_.inbag_counts > 0).sum(axis=1).astype(np.float32)
    inbag_tree_counts[inbag_tree_counts == 0] = 1.0
    expected_diag = ref_mass / inbag_tree_counts

    assert np.allclose(diag, expected_diag, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "est_class,data_fixture",
    [
        ("RandomForestClassifier", "classification_data"),
        ("ExtraTreesClassifier", "classification_data"),
        ("RandomForestRegressor", "regression_data"),
        ("ExtraTreesRegressor", "regression_data"),
    ],
)
@pytest.mark.parametrize("weight_scheme", ["gap", "oob"])
def test_gap_oob_require_bootstrap_true(request, est_class, data_fixture, weight_scheme):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)

    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        RandomForestRegressor,
        ExtraTreesRegressor,
    )

    cls_map = {
        "RandomForestClassifier": RandomForestClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
    }

    est = cls_map[est_class](n_estimators=10, bootstrap=False, random_state=0, n_jobs=1)
    enc = ForestProximity(forest=est, weight_scheme=weight_scheme)

    with pytest.raises(ValueError):
        enc.fit(X_train, y_train)
