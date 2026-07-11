"""Reconstruction Forest Autoencoder implementation from Vu's method."""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigs
import warnings

class EncodeResult:
    """Class to hold the result of the encode function."""
    def __init__(self, Z, A, V, lam, stepsize, leafIDs, sizes, meta):
        self.Z = Z
        self.A = A
        self.V = V
        self.lambda_ = lam
        self.stepsize = stepsize
        self.leafIDs = leafIDs
        self.sizes = sizes
        self.meta = meta

def encode(rf, x, k=5, stepsize=1):
    """
    Computes the diffusion map of a random forest kernel.
    Assumes x is already processed (numeric).
    """
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
        
    n_trees = len(rf.estimators_)
    n_samples = x.shape[0]
    
    if k >= n_samples:
        warnings.warn(f'Embedding k cannot exceed nrow(x)-1. Setting k to {n_samples - 1}.')
        k = n_samples - 1

    # Get terminal nodes
    leafIDs = rf.apply(x) 
    
    # Globalize leaf IDs
    max_leaf = leafIDs.max() + 1 
    offsets = np.arange(n_trees) * max_leaf
    leafIDs_global = leafIDs + offsets 
    
    # Construct Sparse Matrix M
    row_indices = np.tile(np.arange(n_samples), n_trees)
    col_indices = leafIDs_global.flatten(order='F') 
    data = np.ones(len(row_indices), dtype=int)
    
    M = sparse.csr_matrix((data, (row_indices, col_indices)), 
                          shape=(n_samples, int(leafIDs_global.max()) + 1))

    # Normalized Adjacency
    leaf_sizes = np.array(M.sum(axis=0)).flatten()
    with np.errstate(divide='ignore'):
        leaf_weights = 1.0 / leaf_sizes
    leaf_weights[~np.isfinite(leaf_weights)] = 0.0 
    
    D_weights = sparse.diags(leaf_weights)
    M_norm = M @ D_weights
    A = (M_norm @ M.T) / n_trees
    
    # Spectral decomposition
    vals, vecs = eigs(A, k=k+1, which='LM')
    idx = vals.argsort()[::-1] 
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    e_val = np.real(vals[1:k+1])
    e_vec = np.real(vecs[:, 1:k+1])
    
    # Diffusion map Z
    if k > 1:
        scale_factors = e_val ** stepsize
        Z = np.sqrt(n_samples) * (e_vec * scale_factors) 
    else:
        Z = np.sqrt(n_samples) * e_vec * (e_val ** stepsize)
        
    # Metadata Extraction (Simplified: No Levels)
    # We only care about tracking numeric precision for reconstruction bounds
    colnames_x = x.columns.tolist()
    meta_list = []
    
    for col in colnames_x:
        col_data = x[col]
        # Since x is preprocessed, everything is numeric (float/int).
        # We try to detect if it looks like a decimal or an integer for rounding purposes.
        decimals = 0
        is_float = False
        
        if pd.api.types.is_float_dtype(col_data):
            is_float = True
            # Check precision roughly
            strs = col_data.astype(str)
            if strs.str.contains(r'\.').any():
                lens = strs.apply(lambda s: len(s.split('.')[1]) if '.' in s else 0)
                decimals = lens.max()
        
        meta_list.append({
            'variable': col,
            'class': 'numeric' if is_float else 'integer',
            'decimals': decimals,
            'min': col_data.min(),
            'max': col_data.max()
        })

    metadata = pd.DataFrame(meta_list)
    
    # Leaf Sizes for Sparsify
    flat_leafs = leafIDs.flatten(order='F')
    flat_trees = np.tile(np.arange(1, n_trees+1), n_samples)
    sizes_df = pd.DataFrame({'tree': flat_trees, 'leaf': flat_leafs})
    sizes_df = sizes_df.groupby(['tree', 'leaf']).size().reset_index(name='leaf_size')
    
    meta = {'metadata': metadata, 'input_class': str(type(x))}

    return EncodeResult(Z, A, e_vec, e_val, stepsize, leafIDs, sizes_df, meta)


def predict_encode(emap, rf, x):
    """
    Project test data into the forest embedding space.
    """
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)

    n_trees = len(rf.estimators_)
    trn_n = emap.V.shape[0]
    tst_n = x.shape[0]
    
    leafIDs_train = emap.leafIDs
    leafIDs_test = rf.apply(x)
    
    max_leaf = max(leafIDs_train.max(), leafIDs_test.max()) + 1
    offsets = np.arange(n_trees) * max_leaf
    
    leafIDs_global_train = leafIDs_train + offsets
    leafIDs_global_test = leafIDs_test + offsets
    
    flat_train = leafIDs_global_train.flatten(order='F')
    flat_test = leafIDs_global_test.flatten(order='F')
    
    all_indices = np.union1d(flat_train, flat_test)
    leaf_id_map_train = np.searchsorted(all_indices, flat_train)
    leaf_id_map_test = np.searchsorted(all_indices, flat_test)
    num_cols = len(all_indices)
    
    row_ind_train = np.tile(np.arange(trn_n), n_trees)
    M_train = sparse.csr_matrix((np.ones(len(row_ind_train)), (row_ind_train, leaf_id_map_train)), 
                                shape=(trn_n, num_cols))
    
    row_ind_test = np.tile(np.arange(tst_n), n_trees)
    M_test = sparse.csr_matrix((np.ones(len(row_ind_test)), (row_ind_test, leaf_id_map_test)), 
                               shape=(tst_n, num_cols))
    
    leaf_sizes = np.array(M_train.sum(axis=0)).flatten()
    with np.errstate(divide='ignore'):
        leaf_weights = 1.0 / leaf_sizes
    leaf_weights[~np.isfinite(leaf_weights)] = 0.0
    
    M_test_norm = M_test @ sparse.diags(leaf_weights)
    A0 = (M_test_norm @ M_train.T) / n_trees
    
    inv_lambda = 1.0 / emap.lambda_
    inv_lambda[~np.isfinite(inv_lambda)] = 0.0
    
    Z0 = A0 @ emap.Z * inv_lambda 
    return Z0
