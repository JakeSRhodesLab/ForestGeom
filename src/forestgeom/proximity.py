# forestgeom/proximity.py

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

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
    normalize_oob_training_proximity,
    normalize_oob_oos_proximity,
    format_output_matrix,
)


class ForestProximity(TransformerMixin, BaseEstimator):
    """
    Sparse forest proximity estimator.

    ForestProximity computes forest-induced pairwise proximities while exposing
    efficient sparse, forest-based vector representations whenever available.

    Most proximity schemes admit a sparse factored representation of the form

        P = Q W^T,

    where Q is the query-side representation and W is the reference-side
    representation. Both are sparse weighted leaf-incidence maps, allowing
    proximity computation and storage without materializing dense pairwise
    matrices.

    This factorization defines a bilinear form:

        P(i, j) = <Q(i), W(j)>,

    enabling efficient proximity extension to new query samples and supporting
    downstream algorithms that can operate directly on the query/reference maps
    without explicitly constructing proximity matrices.

    SYMMETRIC PROXIMITIES
    For symmetric schemes, the same representation is used on both sides:

        P = Q Q^T.

    This defines a proper kernel with an explicit leaf-incidence feature map

        P(i, j) = <phi(i), phi(j)>.

    Under the additional constraint that phi is a nonnegative weighted
    leaf-incidence map defined over the fixed forest leaf coordinates,
    this reproducing feature map Q = W is unique.

    Examples include:

        - "uniform"
        - "kerf"
        - "boosted"

    ASYMMETRIC PROXIMITIES
    Some schemes admit a separable but asymmetric representation:

        P = Q W^T,
        Q != W.

    These proximities are not kernels and therefore do not admit a symmetric
    reproducing feature representation.

    Nevertheless, the bilinear factorization still enables efficient
    out-of-sample proximity computation and supports matrix-free downstream
    algorithms operating directly on Q and W.

    In general, Q and W are not unique, since invertible transformations of
    the latent leaf space preserve the same proximity matrix. However, under
    fixed forest leaf coordinates and the weighting constraints imposed by the
    selected scheme, a canonical query/reference representation is obtained.

    Examples include:

        - "gap"

    PAIRWISE-NORMALIZED LEAF-COLLISION PROXIMITIES
    Some schemes are not bilinear and therefore do not expose query/reference
    maps.

    The true Breiman OOB proximity is defined as

        P = (Q Q^T) / S,

    where Q is a sparse OOB leaf-incidence map and S contains pairwise shared
    OOB tree counts.

    Unlike bilinear schemes, the normalization depends on each sample pair and
    therefore cannot be absorbed into independent query and reference maps.

    Despite being non-separable, this representation remains efficient because
    S is evaluated only for the nonzero entries of Q Q^T, allowing sparse
    computation of proximity matrices without evaluating all pairwise OOB
    overlaps.

    For out-of-sample queries, all trees are treated as OOB for the query
    sample while the normalization remains defined with respect to the fitted
    training samples.

    Examples include:

        - "oob"

    References
    ----------
    Aumon et al. (2026), Revisiting Forest Proximities via Sparse
    Leaf-Incidence Kernels.
    """

    def __init__(self, forest=None, weight_scheme="uniform"):
        """
        Create a forest proximity estimator around a tree ensemble.

        Parameters
        ----------
        forest : BaseEstimator, default=None
            The tree ensemble to wrap, such as a random forest or boosted tree model.
            If unfitted, it is cloned and fitted inside :meth:`fit`. If already
            fitted, it is reused in-place and not refit.

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
                "This ForestProximity instance is not fitted yet. "
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

    def _fit_forest(self, X, y=None, **fit_kwargs):
        """
        Fit the wrapped forest and cache the training data needed to build maps.
        """
        X = np.asarray(X)
        y = None if y is None else np.asarray(y).ravel()

        if self.forest is None:
            raise ValueError("`forest` must be provided.")

        adapter = make_adapter(
            self.forest,
            weight_scheme=self.weight_scheme,
        )

        try:
            check_is_fitted(self.forest)
        except NotFittedError:
            adapter.fit(X, y, **fit_kwargs)

        self.forest_ = adapter
        self.X_fit_ = X
        self.y_ = y
        # Persist sample_weight passed at fit time so bootstrap statistics
        # (OOB mask, in-bag counts) can be reconstructed consistently.
        # Store the raw sample_weight (may be None).
        self.sample_weight_ = fit_kwargs.get("sample_weight", None)
        self.classes_ = (
            getattr(adapter.estimator, "classes_", np.unique(y))
            if y is not None and self._is_classifier(fitted=True)
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
            oob_mask = self.forest_.get_oob_mask(X, sample_weight=getattr(self, "sample_weight_", None)).astype(np.int8)

            inbag_counts = (
                self.forest_.get_in_bag_counts(X, sample_weight=getattr(self, "sample_weight_", None)).astype(np.float32)
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
        self : ForestProximity
            Fitted estimator with the updated weighting scheme.

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

    def fit(self, X, y=None, **fit_kwargs):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples used to fit the wrapped forest.
        y : array-like of shape (n_samples,), optional
            Training targets. May be omitted for unsupervised forests such as
            ``RandomTreesEmbedding``.
        **fit_kwargs
            Additional keyword arguments passed to the wrapped forest adapter.
            These are ignored when ``forest`` is already fitted, because the
            forest is reused without refitting.

            Note
            ----
            If a ``sample_weight`` keyword is provided it will be persisted on
            the fitted ``ForestProximity`` instance as the attribute
            ``sample_weight_`` and used to reconstruct bootstrap statistics
            (OOB mask and in-bag counts) when building the cache. This
            ensures that weighted sampling performed during fit is reflected
            in derived proximity statistics.

        Returns
        -------
        self : ForestProximity
            Fitted estimator.

        Notes
        -----
        Fit the forest, build the leaf-map cache, and mark the estimator as ready.
        """
        self._fit_forest(X, y, **fit_kwargs)
        self._build_cache()
        return self

    def fit_transform(
        self,
        X,
        y=None,
        return_dense=False,
        force_symmetric=False,
        adjust_diagonal=False,
        **fit_kwargs,
    ):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples used to fit the wrapped forest.
        y : array-like of shape (n_samples,), optional
            Training targets. May be omitted for unsupervised forests such as
            ``RandomTreesEmbedding``.
        return_dense : bool, default=False
            Return a dense array instead of a sparse matrix.
        force_symmetric : bool, default=False
            Symmetrize asymmetric proximity matrices such as GAP by averaging with their transpose.
        adjust_diagonal : bool, default=False
            Apply weighting-scheme-specific diagonal corrections where available.
        **fit_kwargs
            Additional keyword arguments passed to the wrapped forest adapter.

        Returns
        -------
        P : sparse matrix or ndarray of shape (n_samples, n_samples)
            Fitted train-train proximity matrix.
        
        Notes
        -----
        For schemes depending on training-specific bootstrap or OOB status,
        ``fit_transform(X, y)`` is not necessarily equivalent to
        ``fit(X, y).transform(X)``.
    
        ``fit_transform`` returns the fitted train-train proximity matrix, using
        the training-specific definition of the selected scheme. In contrast,
        ``transform`` applies the out-of-sample extension rule.
        """
        self.fit(X, y, **fit_kwargs)
        return self.training_proximity(
            force_symmetric=force_symmetric,
            adjust_diagonal=adjust_diagonal,
            return_dense=return_dense,
        )

    def query_map(self, X=None, return_dense=False):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Samples to encode. If omitted, return the fitted training query map.
        return_dense : bool, default=False
            Return a dense array instead of a sparse matrix.

        Returns
        -------
        Q : sparse matrix or ndarray of shape (n_samples, n_leaves)
            Fitted training query-side leaf map.

        Notes
        -----
        Return the query-side map ``Q``.

        For schemes with training-specific behavior, this can differ from
        the query map built for new samples.
        """
        self._check_fitted()

        if self.weight_scheme == "oob":
            raise AttributeError(
                "query_map is not available for weight_scheme='oob'."
            )

        if X is None:
            Q = self.cache_.Q_mat
        else:
            leaves = self.forest_.get_leaf_matrix(np.asarray(X))
            Q = build_Q_matrix(
                self.cache_,
                weight_scheme=self.weight_scheme,
                leaves=leaves,
                is_training=False,
            )

        return self._format(Q, return_dense=return_dense)

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

        For symmetric proximities this matches :meth:`query_map`.
        For asymmetric schemes such as GAP, ``W`` can differ from ``Q``.
        """
        self._check_fitted()

        if self.weight_scheme == "oob":
            raise AttributeError(
                "reference_map is not available for weight_scheme='oob'."
            )

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
        P_new : sparse matrix or ndarray of shape (n_samples, n_train)
            Proximity block between ``X`` and the fitted training set.

        Notes
        -----
        Return the out-of-sample proximity block between new query samples and
        the fitted training samples.
        """
        self._check_fitted()

        if self.weight_scheme == "oob":
            leaves = self.forest_.get_leaf_matrix(np.asarray(X))
            Q = build_Q_matrix(
                self.cache_,
                weight_scheme=self.weight_scheme,
                leaves=leaves,
                is_training=False,
            )
        else:
            Q = self.query_map(X, return_dense=False)

        P = Q.dot(self.cache_.W_mat.T)

        if self.weight_scheme == "oob":
            P = normalize_oob_oos_proximity(P, self.cache_.oob_mask)

        return self._format(P, return_dense=return_dense)

    def joint_proximity(
        self,
        X,
        return_dense=False,
        force_symmetric=False,
        adjust_diagonal=False,
    ):
        """
        Return a fitted train-plus-query proximity matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query samples to append after the fitted training samples.
        return_dense : bool, default=False
            Return a dense array instead of a sparse matrix.
        force_symmetric : bool, default=False
            For ``weight_scheme="gap"``, symmetrize the training block before
            assembling the joint matrix. Ignored for symmetric schemes.
        adjust_diagonal : bool, default=False
            For ``weight_scheme="gap"``, apply the existing training-block
            diagonal correction before assembling the joint matrix. Ignored for
            symmetric schemes.

        Returns
        -------
        P_joint : sparse matrix or ndarray of shape
            (n_train + n_samples, n_train + n_samples)
            Joint proximity matrix ordered as ``[X_train, X]``.

        Notes
        -----
        For ``"uniform"``, ``"kerf"``, and ``"boosted"``, the joint matrix is
        built from the stacked sparse query maps:

            ``P_joint = Q_all Q_all^T``.

        For ``"gap"``, the joint matrix is assembled as:

            ``[[P_train_train, P_test_train.T], [P_test_train, 0]]``.

        Here ``P_test_train`` is exactly the existing out-of-sample block
        returned by :meth:`transform`, and the test-test GAP block is explicitly
        zero-valued. The ``"oob"`` scheme is not supported because no canonical
        test-test OOB proximity is currently defined.
        """
        self._check_fitted()

        if self.weight_scheme == "oob":
            raise ValueError(
                "joint_proximity is not available for weight_scheme='oob' "
                "because no canonical test-test OOB proximity is defined."
            )

        if self.weight_scheme == "gap":
            P_train_train = self.training_proximity(
                force_symmetric=force_symmetric,
                adjust_diagonal=adjust_diagonal,
                return_dense=False,
            )
            P_test_train = self.transform(X, return_dense=False)
            n_query = P_test_train.shape[0]
            P_test_test = sparse.csr_matrix((n_query, n_query), dtype=np.float32)
            P_joint = sparse.bmat(
                [
                    [P_train_train, P_test_train.T],
                    [P_test_train, P_test_test],
                ],
                format="csr",
                dtype=np.float32,
            )
            return self._format(P_joint, return_dense=return_dense)

        Q_train = self.query_map(return_dense=False)
        Q_test = self.query_map(X, return_dense=False)
        Q_all = sparse.vstack([Q_train, Q_test], format="csr")
        P_joint = Q_all.dot(Q_all.T)

        return self._format(P_joint, return_dense=return_dense)

    def training_proximity(
        self,
        force_symmetric=False,
        adjust_diagonal=False,
        return_dense=False,
    ):
        """
        Return the fitted train-train proximity matrix.
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

        if self.weight_scheme == "oob":
            P = normalize_oob_training_proximity(P, self.cache_.oob_mask)

        return self._format(P, return_dense=return_dense)
