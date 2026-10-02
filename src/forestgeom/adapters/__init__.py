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
from .base import EnsembleAdapter


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
    Return an adapter for a built-in estimator or a custom forest.

    Custom forests expose the same get_* methods as EnsembleAdapter. Supported
    schemes are inferred from the methods implemented by the adapter or forest.

    If weight_scheme is provided, validate that the selected adapter supports
    this forest / weight_scheme combination.
    """
    if isinstance(estimator, EnsembleAdapter):
        adapter = estimator

    elif isinstance(estimator, _RF_ET_RTE_CLASSES):
        adapter = RFETAdapter(estimator)

    elif isinstance(estimator, _GBT_CLASSES):
        adapter = GBTAdapter(estimator)

    elif _LGBM_CLASSES and isinstance(estimator, _LGBM_CLASSES):
        adapter = LightGBMAdapter(estimator)

    elif _XGB_CLASSES and isinstance(estimator, _XGB_CLASSES):
        adapter = XGBoostAdapter(estimator)

    elif callable(getattr(estimator, "get_leaf_matrix", None)):
        adapter = EnsembleAdapter(estimator)

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

        raise TypeError(
            "Unsupported forest estimator. Expected one of: "
            + ", ".join(supported)
            + ", or a custom forest exposing get_leaf_matrix(X)."
        )

    if weight_scheme is not None:
        adapter.validate_weight_scheme(weight_scheme)

    return adapter
