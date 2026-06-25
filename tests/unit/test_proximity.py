import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError

from forestgeom import ForestProximity

from tests.fixtures.constants import (
    JOINT_GAP_CASES,
    JOINT_OOB_CASES,
    JOINT_SYMMETRIC_CASES,
    RF_ET_FORESTS_AND_DATA,
    RF_ET_WEIGHT_SCHEMES,
    RTE_FORESTS_AND_DATA,
)


def _assert_sparse_allclose(actual, expected, **kwargs):
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), **kwargs)


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


@pytest.mark.parametrize(
    "forest_fixture,data_fixture,weight_scheme",
    JOINT_SYMMETRIC_CASES,
)
def test_joint_proximity_symmetric_schemes_match_stacked_query_maps(
    request,
    forest_fixture,
    data_fixture,
    weight_scheme,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(X_train, y_train)

    P_joint = enc.joint_proximity(X_test, return_dense=False)
    Q_train = enc.query_map(return_dense=False)
    Q_test = enc.query_map(X_test, return_dense=False)
    Q_all = sparse.vstack([Q_train, Q_test], format="csr")
    expected = Q_all @ Q_all.T

    assert sparse.isspmatrix_csr(P_joint)
    assert P_joint.shape == (
        X_train.shape[0] + X_test.shape[0],
        X_train.shape[0] + X_test.shape[0],
    )
    _assert_sparse_allclose(P_joint, expected)


@pytest.mark.parametrize("forest_fixture,data_fixture", JOINT_GAP_CASES)
def test_joint_proximity_rejects_gap_weight_scheme(
    request,
    forest_fixture,
    data_fixture,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)

    enc = ForestProximity(forest=forest, weight_scheme="gap").fit(X_train, y_train)

    with pytest.raises(ValueError, match="GAP is directional"):
        enc.joint_proximity(X_test)


@pytest.mark.parametrize("forest_fixture,data_fixture", JOINT_OOB_CASES)
def test_oob_joint_proximity_blocks_are_normalized(
    request,
    forest_fixture,
    data_fixture,
):
    X_train, X_test, y_train, _ = request.getfixturevalue(data_fixture)
    forest = request.getfixturevalue(forest_fixture)
    n_train = X_train.shape[0]

    enc = ForestProximity(forest=forest, weight_scheme="oob").fit(X_train, y_train)

    P_joint = enc.joint_proximity(X_test, return_dense=False)
    P_train_train = enc.training_proximity(return_dense=False)
    P_test_train = enc.transform(X_test, return_dense=False)
    test_leaves = enc.forest_.get_leaf_matrix(np.asarray(X_test))
    expected_test_test = (
        (test_leaves[:, None, :] == test_leaves[None, :, :]).sum(axis=2)
        / enc.cache_.n_trees
    ).astype(np.float32)
    expected_test_test = sparse.csr_matrix(expected_test_test)

    assert sparse.isspmatrix_csr(P_joint)
    assert P_joint.shape == (
        X_train.shape[0] + X_test.shape[0],
        X_train.shape[0] + X_test.shape[0],
    )
    np.testing.assert_allclose(
        P_joint.toarray(),
        P_joint.T.toarray(),
        rtol=1e-6,
        atol=1e-8,
    )
    _assert_sparse_allclose(P_joint[:n_train, :n_train], P_train_train)
    _assert_sparse_allclose(P_joint[n_train:, :n_train], P_test_train)
    _assert_sparse_allclose(P_joint[:n_train, n_train:], P_test_train.T)
    _assert_sparse_allclose(
        P_joint[n_train:, n_train:],
        expected_test_test,
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize("weight_scheme", ["uniform", "oob"])
def test_joint_proximity_dense_output_matches_sparse_output(request, weight_scheme):
    X_train, X_test, y_train, _ = request.getfixturevalue("classification_data")
    forest = request.getfixturevalue("rf_classifier")

    enc = ForestProximity(forest=forest, weight_scheme=weight_scheme).fit(
        X_train,
        y_train,
    )

    P_sparse = enc.joint_proximity(X_test, return_dense=False)
    P_dense = enc.joint_proximity(X_test, return_dense=True)

    assert isinstance(P_dense, np.ndarray)
    np.testing.assert_allclose(P_dense, P_sparse.toarray())


def test_joint_proximity_requires_fitted_estimator(request):
    X_train, X_test, y_train, _ = request.getfixturevalue("classification_data")
    forest = request.getfixturevalue("rf_classifier")

    enc = ForestProximity(forest=forest, weight_scheme="uniform")

    with pytest.raises(NotFittedError):
        enc.joint_proximity(X_test)


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
