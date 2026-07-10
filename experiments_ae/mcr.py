import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from utils import post_x

def decode_knn(rf, emap, z, preprocessing_meta=None, x_tilde=None, k=5, round_vals=True, n_jobs=1):
    """
    Maps low-dimensional embedding back to input space via iterative KNN.
    """
    nbrs_model = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=n_jobs)
    nbrs_model.fit(emap.Z)
    dists, indices = nbrs_model.kneighbors(z)
    
    with np.errstate(divide='ignore'):
        weights = 1.0 / dists
    inf_mask = ~np.isfinite(weights)
    if inf_mask.any():
        weights[inf_mask] = 1.0
        weights[~inf_mask] = 0.0
    
    row_sums = weights.sum(axis=1)[:, np.newaxis]
    zero_sum_mask = (row_sums == 0).flatten()
    if zero_sum_mask.any():
        weights[zero_sum_mask, :] = 1.0
        row_sums[zero_sum_mask] = k # Sum becomes k
    weights /= row_sums
        
    unique_neighbor_ids = np.unique(indices)
    
    if x_tilde is None:
        x_tilde_df = train_decoder(rf, emap, neighbors=unique_neighbor_ids, n_jobs=n_jobs)
    else:
        x_tilde_df = x_tilde.iloc[unique_neighbor_ids].copy()
    
    id_map = {global_id: loc for loc, global_id in enumerate(unique_neighbor_ids)}
    vectorized_map = np.vectorize(id_map.get)
    local_indices = vectorized_map(indices)

    meta = emap.meta['metadata']
    col_names = meta['variable'].tolist()
    
    # Construct output in Encoded Space
    out_df = pd.DataFrame(index=range(z.shape[0]), columns=col_names)
    
    # We treat all columns as numeric for the weighted average
    # (Since x is processed, even categories are 0/1 or 0,1,2 integers)
    vals = x_tilde_df[col_names].values[local_indices]
    w_expanded = weights[:, :, np.newaxis]
    weighted_means = (vals * w_expanded).sum(axis=1)
    out_df[col_names] = weighted_means

    # Post-Process (Inverse Transform -> Original Space)
    x_hat = post_x(out_df, emap.meta, preprocessing_meta=preprocessing_meta, round_vals=round_vals)
            
    return {'x_hat': x_hat, 'x_tilde': x_tilde_df if x_tilde is None else x_tilde}


def train_decoder(rf, emap, neighbors=None, null_value=None, n_jobs=1):
    """
    Rebuilds training data using eForest. Returns encoded-space data.
    """
    n_samples_total = len(emap.leafIDs)
    if neighbors is None: neighbors = np.arange(n_samples_total)
    else: neighbors = np.unique(neighbors)
        
    n_neighbors = len(neighbors)
    n_features = rf.n_features_in_
    col_names = emap.meta['metadata']['variable'].tolist()
    
    final_lb = np.full((n_neighbors, n_features), -np.inf)
    final_ub = np.full((n_neighbors, n_features), np.inf)
    
    def process_tree_batch(tree_indices):
        batch_lb = np.full((n_neighbors, n_features), -np.inf)
        batch_ub = np.full((n_neighbors, n_features), np.inf)
        
        for t_idx in tree_indices:
            tree = rf.estimators_[t_idx].tree_
            leaf_bounds = _get_tree_leaf_bounds(tree, n_features)
            node_ids = emap.leafIDs[neighbors, t_idx]
            
            # Lookup tables
            max_node = tree.node_count
            lb_lookup = np.zeros((max_node, n_features))
            ub_lookup = np.zeros((max_node, n_features))
            
            for lid, (l, u) in leaf_bounds.items():
                lb_lookup[lid] = l
                ub_lookup[lid] = u
                
            batch_lb = np.maximum(batch_lb, lb_lookup[node_ids])
            batch_ub = np.minimum(batch_ub, ub_lookup[node_ids])
            
        return batch_lb, batch_ub

    n_trees = len(rf.estimators_)
    chunk_size = max(1, n_trees // n_jobs)
    chunks = [range(i, min(i + chunk_size, n_trees)) for i in range(0, n_trees, chunk_size)]
    results = Parallel(n_jobs=n_jobs)(delayed(process_tree_batch)(chunk) for chunk in chunks)
    
    for b_lb, b_ub in results:
        final_lb = np.maximum(final_lb, b_lb)
        final_ub = np.minimum(final_ub, b_ub)
        
    # Handle bounds
    meta_df = emap.meta['metadata'].set_index('variable').reindex(col_names)
    final_lb = np.where(np.isneginf(final_lb), meta_df['min'].values, final_lb)
    final_ub = np.where(np.isposinf(final_ub), meta_df['max'].values, final_ub)
    
    # Uniform sampling
    reconstructed = np.random.uniform(final_lb, final_ub)
    return pd.DataFrame(reconstructed, columns=col_names)


def _get_tree_leaf_bounds(tree, n_features):
    left = tree.children_left
    right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    
    leaf_bounds = {}
    stack = [(0, np.full(n_features, -np.inf), np.full(n_features, np.inf))]
    
    while stack:
        node_id, lb, ub = stack.pop()
        
        if left[node_id] != -1:
            f = feature[node_id]
            th = threshold[node_id]
            
            ub_left = ub.copy()
            ub_left[f] = min(ub_left[f], th)
            stack.append((left[node_id], lb.copy(), ub_left))
            
            lb_right = lb.copy()
            lb_right[f] = max(lb_right[f], th)
            stack.append((right[node_id], lb_right, ub.copy()))
        else:
            leaf_bounds[node_id] = (lb, ub)
            
    return leaf_bounds
