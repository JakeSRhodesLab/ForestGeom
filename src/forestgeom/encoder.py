# forestgeom/encoder.py

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from .adapters import make_adapter
from .maps import (
    initialize_cache,
    attach_bootstrap_stats,
    attach_boosted_weights,
    attach_inv_sqrt_leaf_mass,
    attach_inv_inbag_leaf_mass,
    build_W_matrix,
    build_Q_matrix,
    augment_leaf_maps,
    block_symmetrize,
    format_output_matrix,
)
from .prediction import proximity_predict, PredictionDiagnostics

class LeafEncoder(TransformerMixin, BaseEstimator):
    """
    Sparse forest leaf encoder.

    The fitted encoder represents forest proximities in factored form:

        P = Q W^T,

    where ``Q`` is the query-side leaf map and ``W`` is the reference-side leaf map.
    Both maps are sparse, so the encoder can store and manipulate proximities
    without materializing the full dense matrix ``P``.

    Public methods expose the fitted leaf maps, inductive transforms, and
    proximity-based predictions. Use them when you want leaf-space features
    directly or when you need explicit train-train / train-test proximity blocks.

    The explicit proximity matrix is still available, but working with the sparse
    factors is usually the more scalable option.
    """

    def __init__(self, forest=None, weight_scheme="uniform"):
        """
        Create a leaf encoder around a tree ensemble.

        Parameters
        ----------
        forest : BaseEstimator, default=None
            The tree ensemble to wrap, such as a random forest or boosted tree model.
            It is cloned and fitted inside :meth:`fit`.

        weight_scheme : str, default="uniform"
            Leaf-weighting scheme used to build the query and reference maps.
            Supported values are ``"uniform"``, ``"oob"``, ``"gap"``, ``"kerf"``,
            and ``"boosted"``.
        """
        self.forest = forest
        self.weight_scheme = weight_scheme

    def _check_forest_fitted(self):
        check_is_fitted(self, attributes=["forest_"])
    
    def _check_fitted(self):
        check_is_fitted(self, attributes=["forest_", "cache_"])
    
        if self.cache_ is None:
            raise NotFittedError(
                "This LeafEncoder instance is not fitted yet. "
                "Call `fit(...)` first."
            )
    
    def _is_classifier(self, fitted=False):
        """
        Return whether the underlying forest is a classifier.
    
        If fitted=False, inspect the user-provided forest.
        If fitted=True, inspect the fitted adapter estimator.
    
        Non-classifier estimators are treated as regressors by default.
        """
        estimator = self.forest_.estimator if fitted else self.forest
        return is_classifier(estimator)

    def _format(self, matrix, return_dense=False):
        return format_output_matrix(matrix, return_dense=return_dense)

    def _fit_forest(self, X, y, **fit_kwargs):
        """
        Fit the wrapped forest and cache the training data needed to build maps.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
    
        if self.forest is None:
            raise ValueError("`forest` must be provided.")
    
        adapter = make_adapter(
            self.forest,
            weight_scheme=self.weight_scheme,
        )
    
        adapter.fit(X, y, **fit_kwargs)
    
        self.forest_ = adapter
        self.X_fit_ = X
        self.y_ = y
        self.classes_ = (
            getattr(adapter.estimator, "classes_", np.unique(y))
            if self._is_classifier(fitted=True)
            else None
        )
        self.cache_ = None
    
        return self

    def _build_cache(self):
        """
        Build the leaf-map cache and precompute the fitted ``Q`` and ``W`` matrices.
        """
        self._check_forest_fitted()

        X = self.X_fit_

        leaf_matrix = self.forest_.get_leaf_matrix(X)
        n_nodes_per_tree = self.forest_.get_n_nodes_per_tree()

        self.cache_ = initialize_cache(
            leaf_matrix=leaf_matrix,
            n_nodes_per_tree=n_nodes_per_tree,
            n_samples=X.shape[0],
        )

        if self.weight_scheme in ("oob", "gap"):
            oob_mask = self.forest_.get_oob_mask(X).astype(np.int8)

            inbag_counts = (
                self.forest_.get_in_bag_counts(X).astype(np.float32)
                if self.weight_scheme == "gap"
                else None
            )

            attach_bootstrap_stats(
                self.cache_,
                oob_mask=oob_mask,
                inbag_counts=inbag_counts,
            )

        if self.weight_scheme == "boosted":
            boosted_tree_weights = self.forest_.get_tree_weights(X)
            attach_boosted_weights(self.cache_, boosted_tree_weights)

        if self.weight_scheme == "kerf":
            attach_inv_sqrt_leaf_mass(self.cache_)

        if self.weight_scheme == "gap":
            attach_inv_inbag_leaf_mass(self.cache_)

        self.cache_.Q_mat = build_Q_matrix(
            self.cache_,
            weight_scheme=self.weight_scheme,
            leaves=self.cache_.leaf_matrix,
            is_training=True,
        )

        self.cache_.W_mat = build_W_matrix(
            self.cache_,
            weight_scheme=self.weight_scheme,
        )

        return self

    def set_weight_scheme(self, weight_scheme):
        """
        Parameters
        ----------
        weight_scheme : str
            New weighting scheme to activate.

        Returns
        -------
        self : LeafEncoder
            Fitted encoder with the updated weighting scheme.

        Notes
        -----
        Switch the active weighting scheme and rebuild cached maps if needed.

        The wrapped forest is not refit. If cache construction fails, the previous
        scheme and cache are restored.
        """
        self._check_forest_fitted()
    
        old_scheme = self.weight_scheme
        old_cache = getattr(self, "cache_", None)
    
        try:
            self.forest_.validate_weight_scheme(weight_scheme)
            self.weight_scheme = weight_scheme
    
            if weight_scheme != old_scheme or old_cache is None:
                self._build_cache()
    
        except Exception:
            self.weight_scheme = old_scheme
            self.cache_ = old_cache
            raise
    
        return self

    def fit(self, X, y, **fit_kwargs):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples used to fit the wrapped forest.
        y : array-like of shape (n_samples,)
            Training targets.
        **fit_kwargs
            Additional keyword arguments passed to the wrapped forest adapter.

        Returns
        -------
        self : LeafEncoder
            Fitted encoder.

        Notes
        -----
        Fit the forest, build the leaf-map cache, and mark the encoder as ready.
        """
        self._fit_forest(X, y, **fit_kwargs)
        self._build_cache()
    
        # Drop cached diagnostics because they depend on fitted state.
        if hasattr(self, "_diagnostics"):
            del self._diagnostics
    
        return self

    def fit_transform(self, X, y, return_dense=False, **fit_kwargs):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples used to fit the wrapped forest.
        y : array-like of shape (n_samples,)
            Training targets.
        return_dense : bool, default=False
            Return a dense array instead of a sparse matrix.
        **fit_kwargs
            Additional keyword arguments passed to the wrapped forest adapter.

        Returns
        -------
        Q : sparse matrix or ndarray of shape (n_samples, n_leaves)
            Training query-side leaf map.

        Notes
        -----
        Fit the encoder and return the training query map ``Q``.
        """
        self.fit(X, y, **fit_kwargs)
        return self.training_query_map(return_dense=return_dense)

    def training_query_map(self, return_dense=False):
        """
        Parameters
        ----------
        return_dense : bool, default=False
            Return a dense array instead of a sparse matrix.

        Returns
        -------
        Q : sparse matrix or ndarray of shape (n_samples, n_leaves)
            Fitted training query-side leaf map.

        Notes
        -----
        Return the fitted training query-side map ``Q``.

        For schemes with training-specific behavior, this can differ from
        ``transform(X_fit_)``.
        """
        self._check_fitted()
        return self._format(self.cache_.Q_mat, return_dense=return_dense)

    def reference_map(self, return_dense=False):
        """
        Parameters
        ----------
        return_dense : bool, default=False
            Return a dense array instead of a sparse matrix.

        Returns
        -------
        W : sparse matrix or ndarray of shape (n_samples, n_leaves)
            Fitted reference-side leaf map.

        Notes
        -----
        Return the fitted reference-side map ``W``.

        For symmetric proximities this matches :meth:`training_query_map`.
        For asymmetric schemes such as GAP, ``W`` can differ from ``Q``.
        """
        self._check_fitted()
        return self._format(self.cache_.W_mat, return_dense=return_dense)

    def transform(self, X, return_dense=False):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to encode.
        return_dense : bool, default=False
            Return a dense array instead of a sparse matrix.

        Returns
        -------
        Q : sparse matrix or ndarray of shape (n_samples, n_leaves)
            Query-side leaf map for ``X``.

        Notes
        -----
        Map new samples into query-space leaf features ``Q(X)``.
        """
        self._check_fitted()

        leaves = self.forest_.get_leaf_matrix(np.asarray(X))

        Q = build_Q_matrix(
            self.cache_,
            weight_scheme=self.weight_scheme,
            leaves=leaves,
            is_training=False,
        )

        return self._format(Q, return_dense=return_dense)

    def proximity(self, force_symmetric=False, adjust_diagonal=False, return_dense=False):
        """
        Parameters
        ----------
        force_symmetric : bool, default=False
            Symmetrize asymmetric proximity blocks such as GAP using block
            symmetrization. This is useful for downstream proximity-based
            applications, but it discards the asymmetric kernel factorization.
        adjust_diagonal : bool, default=False
            Apply weighting-scheme-specific diagonal corrections. For example,
            ``oob`` is a separable proxy for the true Breiman OOB proximity and
            inflates diagonal entries, so the diagonal is forced to ``1``. For
            GAP, the diagonal is zero by definition, so the extended
            self-similar GAP convention is used.
        return_dense : bool, default=False
            Return a dense array instead of a sparse matrix.

        Returns
        -------
        P : sparse matrix or ndarray of shape (n_samples, n_samples)
            Fitted train-train proximity matrix.

        Notes
        -----
        Return the fitted train-train proximity matrix ``P = Q W^T``.
        """
        self._check_fitted()

        Q = self.cache_.Q_mat
        W = self.cache_.W_mat

        Q, W = augment_leaf_maps(
            self.cache_,
            self.weight_scheme,
            Q,
            W,
            adjust_diagonal=adjust_diagonal,
            is_training=True,
        )

        if force_symmetric and self.weight_scheme in {"gap"}:
            P = block_symmetrize(Q, W)
        else:
            P = Q.dot(W.T)

        return self._format(P, return_dense=return_dense)

    def proximity_extend(self, X, return_dense=False):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New samples to compare against the fitted reference set.
        return_dense : bool, default=False
            Return a dense array instead of a sparse matrix.

        Returns
        -------
        P_new : sparse matrix or ndarray of shape (n_samples, n_train)
            Train-test proximity block.

        Notes
        -----
        Return the train-test proximity block for new samples.

        This computes the out-of-sample block

            P_new = Q(X) W_train^T,

        between new query samples and the fitted reference set.
        """
        self._check_fitted()

        Q = self.transform(X, return_dense=False)
        P = Q.dot(self.cache_.W_mat.T)

        return self._format(P, return_dense=return_dense)
    
    def proximity_predict(self, X):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray
            Predicted targets or classes, depending on the wrapped forest type.

        Notes
        -----
        Predict by weighting training labels in leaf space.

        The prediction mode is inferred from the fitted forest, so the same
        encoder works for classifier and regressor ensembles.
        """
        self._check_fitted()
    
        Q = self.transform(X, return_dense=False)
    
        return proximity_predict(
            Q,
            self.cache_.W_mat,
            self.y_,
            self.weight_scheme,
            is_classifier=self._is_classifier(fitted=True),
        )
    
    def proximity_predict_proba(self, X):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples for which to compute class probabilities.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class-probability predictions.

        Notes
        -----
        Return class probabilities for fitted classifier forests.

        Raises ``AttributeError`` when the fitted forest is not a classifier.
        """
        self._check_fitted()
    
        if not self._is_classifier(fitted=True):
            raise AttributeError(
                "proximity_predict_proba is only available when the fitted forest is a classifier."
            )
    
        Q = self.transform(X, return_dense=False)
    
        proba, _ = proximity_predict(
            Q,
            self.cache_.W_mat,
            self.y_,
            self.weight_scheme,
            is_classifier=True,
            return_proba=True,
        )
    
        return proba
    
    @property
    def diagnostics(self):
        """
        Returns
        -------
        PredictionDiagnostics
            Lazily constructed diagnostics object for the fitted encoder.

        Notes
        -----
        Lazily construct diagnostics for the fitted encoder.
        """
        if not hasattr(self, "_diagnostics"):
            self._diagnostics = PredictionDiagnostics(self)
    
        return self._diagnostics
