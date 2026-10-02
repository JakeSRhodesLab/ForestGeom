from sklearn.base import clone


class EnsembleAdapter:
    """
    Wrap an estimator and delegate methods not implemented by the adapter.

    Adapters or custom forests provide get_leaf_matrix(X), returning integer
    leaf IDs of shape (N, T). Optional methods enable additional schemes:
    get_oob_mask(X_train=None, sample_weight=None) returns a (N_train, T)
    mask with 1 for OOB samples; get_in_bag_counts with the same signature
    returns bootstrap multiplicities; get_tree_weights(X_ref) returns
    nonnegative weights of shape (T,) summing to one.

    get_n_nodes_per_tree() may also return per-tree node counts, but is not
    required for proximity construction. All outputs use consistent tree order.
    """
    def __init__(self, estimator, weight_scheme=None):
        self.estimator = estimator

        if weight_scheme is not None:
            self.validate_weight_scheme(weight_scheme)

    def validate_weight_scheme(self, weight_scheme):
        """
        Validate a scheme against implemented adapter or forest methods.
        """
        requirements = {
            "uniform": ("get_leaf_matrix",),
            "kerf": ("get_leaf_matrix",),
            "oob": ("get_leaf_matrix", "get_oob_mask"),
            "gap": ("get_leaf_matrix", "get_oob_mask", "get_in_bag_counts"),
            "boosted": ("get_leaf_matrix", "get_tree_weights"),
        }
        supported = [
            scheme for scheme, methods in requirements.items()
            if all(callable(getattr(self, method, None)) for method in methods)
        ]
        if weight_scheme not in supported:
            raise ValueError(
                f"{type(self).__name__} does not support "
                f"weight_scheme='{weight_scheme}'. "
                f"Supported schemes are {sorted(supported)}."
            )

        return self

    def fit(self, X, y=None, **fit_kwargs):
        """
        Clone and fit the wrapped estimator.

        Additional keyword arguments are passed to estimator.fit(...),
        e.g. sample_weight.
        """
        self.estimator = clone(self.estimator)
        self.estimator.fit(X, y, **fit_kwargs)
        return self

    def __getattr__(self, name):
        return getattr(self.estimator, name)
