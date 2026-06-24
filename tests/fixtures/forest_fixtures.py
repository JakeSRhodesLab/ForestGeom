import inspect

import pytest
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomTreesEmbedding,
    GradientBoostingClassifier,
)

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from aeon.classification.sklearn import RotationForestClassifier
    from aeon.regression.sklearn import RotationForestRegressor
except Exception:
    RotationForestClassifier = None
    RotationForestRegressor = None

try:
    import treeple as _treeple
except Exception:
    _treeple = None


def _make_treeple_estimator(class_name):
    if _treeple is None:
        pytest.importorskip("treeple")

    cls = getattr(_treeple, class_name, None)
    if cls is None:
        pytest.skip(f"treeple does not expose {class_name}")

    signature = inspect.signature(cls.__init__)
    kwargs = {}
    if "n_estimators" in signature.parameters:
        kwargs["n_estimators"] = 20
    if "random_state" in signature.parameters:
        kwargs["random_state"] = 0
    if "bootstrap" in signature.parameters:
        kwargs["bootstrap"] = True
    if "n_jobs" in signature.parameters:
        kwargs["n_jobs"] = 1

    return cls(**kwargs)


@pytest.fixture
def rf_classifier():
    return RandomForestClassifier(
        n_estimators=50,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )


@pytest.fixture
def et_classifier():
    return ExtraTreesClassifier(
        n_estimators=50,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )


@pytest.fixture
def rf_regressor():
    return RandomForestRegressor(
        n_estimators=50,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )


@pytest.fixture
def et_regressor():
    return ExtraTreesRegressor(
        n_estimators=50,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )


@pytest.fixture
def random_trees_embedding():
    return RandomTreesEmbedding(
        n_estimators=50,
        random_state=0,
        n_jobs=-1,
    )


@pytest.fixture
def gbt_classifier():
    return GradientBoostingClassifier(
        n_estimators=50,
        random_state=0,
    )


@pytest.fixture
def lgbm_classifier():
    if LGBMClassifier is None:
        pytest.importorskip("lightgbm")

    return LGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        random_state=0,
        verbose=-1,
        n_jobs=-1,
    )


@pytest.fixture
def xgb_classifier():
    if XGBClassifier is None:
        pytest.importorskip("xgboost")

    return XGBClassifier(
        n_estimators=50,
        learning_rate=0.1,
        random_state=0,
        verbosity=0,
        n_jobs=-1,
        eval_metric="logloss",
    )


@pytest.fixture
def rotation_forest_classifier():
    if RotationForestClassifier is None:
        pytest.importorskip("aeon")

    return RotationForestClassifier(
        n_estimators=12,
        random_state=0,
    )


@pytest.fixture
def rotation_forest_regressor():
    if RotationForestRegressor is None:
        pytest.importorskip("aeon")

    return RotationForestRegressor(
        n_estimators=12,
        random_state=0,
    )


@pytest.fixture
def treeple_oblique_classifier():
    return _make_treeple_estimator("ObliqueRandomForestClassifier")


@pytest.fixture
def treeple_patch_oblique_classifier():
    return _make_treeple_estimator("PatchObliqueRandomForestClassifier")


@pytest.fixture
def treeple_extra_oblique_classifier():
    return _make_treeple_estimator("ExtraObliqueRandomForestClassifier")


@pytest.fixture
def treeple_honest_classifier():
    return _make_treeple_estimator("HonestForestClassifier")


@pytest.fixture
def treeple_oblique_regressor():
    return _make_treeple_estimator("ObliqueRandomForestRegressor")


@pytest.fixture
def treeple_patch_oblique_regressor():
    return _make_treeple_estimator("PatchObliqueRandomForestRegressor")


@pytest.fixture
def treeple_extra_oblique_regressor():
    return _make_treeple_estimator("ExtraObliqueRandomForestRegressor")


@pytest.fixture
def treeple_unsupervised_random_forest():
    return _make_treeple_estimator("UnsupervisedRandomForest")


@pytest.fixture
def treeple_extended_isolation_forest():
    return _make_treeple_estimator("ExtendedIsolationForest")
