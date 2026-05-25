import numpy as np
import pytest

from forestgeom import ForestProximity

from tests.fixtures.constants import (
    ALL_SUPPORTED_FACTORIZABLE_CASES,
    RF_ET_CLASSIFICATION_CASES,
    RF_ET_CLASSIFICATION_INDUCTIVE_CASES,
)


def _max_nnz_per_row(matrix):
    return np.diff(matrix.tocsr().indptr).max()


def _assert_sparse_allclose(actual, expected):
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.toarray(), expected.toarray())


@pytest.mark.parametrize("forest_fixture,data_fixture,weight_scheme", RF_ET_CLASSIFICATION_CASES)
def test_fit_transform_matches_fit_then_training_proximity(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    fit_transform_encoder = ForestProximity(forest=forest, weight_scheme=weight_scheme)
    K_fit_transform = fit_transform_encoder.fit_transform(X_train, y_train, return_dense=False)

    fit_encoder = ForestProximity(forest=forest, weight_scheme=weight_scheme)
    fit_encoder.fit(X_train, y_train)
    K_training = fit_encoder.training_proximity(return_dense=False)

    _assert_sparse_allclose(K_fit_transform, K_training)


@pytest.mark.parametrize("forest_fixture,data_fixture,weight_scheme", RF_ET_CLASSIFICATION_INDUCTIVE_CASES)
def test_fit_transform_matches_fit_then_transform_for_inductive_classification_schemes(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    fit_transform_encoder = ForestProximity(forest=forest, weight_scheme=weight_scheme)
    Q_fit_transform = fit_transform_encoder.fit_transform(X_train, y_train, return_dense=False)

    fit_encoder = ForestProximity(forest=forest, weight_scheme=weight_scheme)
    fit_encoder.fit(X_train, y_train)
    Q_transform = fit_encoder.transform(X_train, return_dense=False)

    _assert_sparse_allclose(Q_fit_transform, Q_transform)


@pytest.mark.parametrize(
    "forest_fixture,data_fixture,weight_scheme",
    ALL_SUPPORTED_FACTORIZABLE_CASES,
)
def test_leaf_maps_have_consistent_shapes(request, forest_fixture, data_fixture, weight_scheme):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    Q_train = enc.query_map(return_dense=False)
    W = enc.reference_map(return_dense=False)
    Q_test = enc.query_map(X_test, return_dense=False)

    assert Q_train.shape[0] == X_train.shape[0]
    assert Q_test.shape[0] == X_test.shape[0]
    assert Q_train.shape[1] == W.shape[1]
    assert Q_test.shape[1] == W.shape[1]
    assert W.shape[0] == X_train.shape[0]


@pytest.mark.parametrize(
    "forest_fixture,data_fixture,weight_scheme",
    ALL_SUPPORTED_FACTORIZABLE_CASES,
)
def test_leaf_maps_have_at_most_one_nonzero_per_tree_per_row(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    Q_train = enc.query_map(return_dense=False)
    W = enc.reference_map(return_dense=False)
    Q_test = enc.query_map(X_test, return_dense=False)

    n_trees = enc.cache_.leaf_matrix.shape[1]

    assert _max_nnz_per_row(Q_train) <= n_trees
    assert _max_nnz_per_row(W) <= n_trees
    assert _max_nnz_per_row(Q_test) <= n_trees


@pytest.mark.parametrize(
    "forest_fixture,data_fixture,weight_scheme",
    ALL_SUPPORTED_FACTORIZABLE_CASES,
)
def test_proximity_matches_leaf_factorization(request, forest_fixture, data_fixture, weight_scheme):
    X_train, _, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    K = enc.training_proximity(return_dense=True)
    Q = enc.query_map(return_dense=False)
    W = enc.reference_map(return_dense=False)

    assert np.allclose(K, (Q @ W.T).toarray())


@pytest.mark.parametrize(
    "forest_fixture,data_fixture,weight_scheme",
    ALL_SUPPORTED_FACTORIZABLE_CASES,
)
def test_proximity_extend_matches_leaf_factorization(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    K_test = enc.transform(X_test, return_dense=True)
    Q_test = enc.query_map(X_test, return_dense=False)
    W = enc.reference_map(return_dense=False)

    assert np.allclose(K_test, (Q_test @ W.T).toarray())
