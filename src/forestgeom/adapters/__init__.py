from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomTreesEmbedding,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

from .rf_et_rte import RFETAdapter
from .gbt import GBTAdapter


try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from .lgbm import LightGBMAdapter

    _LGBM_CLASSES = (LGBMClassifier, LGBMRegressor)
except ImportError:
    LightGBMAdapter = None
    _LGBM_CLASSES = ()


try:
    from xgboost import XGBClassifier, XGBRegressor
    from .xgb import XGBoostAdapter

    _XGB_CLASSES = (XGBClassifier, XGBRegressor)
except ImportError:
    XGBoostAdapter = None
    _XGB_CLASSES = ()


try:
    from aeon.classification.sklearn import RotationForestClassifier
    from aeon.regression.sklearn import RotationForestRegressor
    from .aeon import AeonRotationForestAdapter

    _AEON_ROTATION_FOREST_CLASSES = (
        RotationForestClassifier,
        RotationForestRegressor,
    )
except Exception:
    AeonRotationForestAdapter = None
    _AEON_ROTATION_FOREST_CLASSES = ()


try:
    import treeple as _treeple
    from .treeple import TreepleForestAdapter

    _TREEPLE_CLASS_NAMES = (
        "ObliqueRandomForestClassifier",
        "ObliqueRandomForestRegressor",
        "PatchObliqueRandomForestClassifier",
        "PatchObliqueRandomForestRegressor",
        "ExtraObliqueRandomForestClassifier",
        "ExtraObliqueRandomForestRegressor",
        "HonestForestClassifier",
        "UnsupervisedRandomForest",
        "ExtendedIsolationForest",
    )
    _TREEPLE_CLASSES = tuple(
        getattr(_treeple, name)
        for name in _TREEPLE_CLASS_NAMES
        if hasattr(_treeple, name)
    )
except Exception:
    TreepleForestAdapter = None
    _TREEPLE_CLASSES = ()


_RF_ET_RTE_CLASSES = (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomTreesEmbedding,
)

_GBT_CLASSES = (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)


def make_adapter(estimator, weight_scheme=None):
    """
    Return the correct ensemble adapter for a supported estimator.

    If weight_scheme is provided, validate that the selected adapter supports
    this forest / weight_scheme combination.
    """
    if isinstance(estimator, _RF_ET_RTE_CLASSES):
        adapter = RFETAdapter(estimator)

    elif isinstance(estimator, _GBT_CLASSES):
        adapter = GBTAdapter(estimator)

    elif _LGBM_CLASSES and isinstance(estimator, _LGBM_CLASSES):
        adapter = LightGBMAdapter(estimator)

    elif _XGB_CLASSES and isinstance(estimator, _XGB_CLASSES):
        adapter = XGBoostAdapter(estimator)

    elif _AEON_ROTATION_FOREST_CLASSES and isinstance(
        estimator,
        _AEON_ROTATION_FOREST_CLASSES,
    ):
        adapter = AeonRotationForestAdapter(estimator)

    elif _TREEPLE_CLASSES and isinstance(estimator, _TREEPLE_CLASSES):
        adapter = TreepleForestAdapter(estimator)

    else:
        supported = [
            "RandomForestClassifier/Regressor",
            "ExtraTreesClassifier/Regressor",
            "RandomTreesEmbedding",
            "GradientBoostingClassifier/Regressor",
        ]

        if _LGBM_CLASSES:
            supported.append("LGBMClassifier/Regressor")

        if _XGB_CLASSES:
            supported.append("XGBClassifier/Regressor")

        if _AEON_ROTATION_FOREST_CLASSES:
            supported.append("aeon RotationForestClassifier/Regressor")

        if _TREEPLE_CLASSES:
            supported.append("treeple forest-like estimators")

        raise TypeError(
            "Unsupported forest estimator. Expected one of: "
            + ", ".join(supported)
            + "."
        )

    if weight_scheme is not None:
        adapter.validate_weight_scheme(weight_scheme)

    return adapter
