import numpy as np
from scipy import sparse


def block_symmetrize(Q, W):
    """
    Computes symmetric 'kernel' P using optimized sparse strategies.

    P = 0.5 * (Q W^T + W Q^T)
      = 0.5 * [Q, W] [W^T; Q^T]

    Uses the block matrix trick to avoid explicitly materializing both
    asymmetric products separately.
    """
    left_block = sparse.hstack([Q, W], format="csr", dtype=np.float32)
    right_block_T = sparse.vstack([W.T, Q.T], format="csc", dtype=np.float32)
    P = 0.5 * left_block.dot(right_block_T)
    del left_block, right_block_T
    return P


def normalize_oob_training_proximity(P, oob_mask):
    """
    Normalize sparse OOB collision counts by pairwise shared-OOB counts in place.

    Complexity:
        O(nnz(P) * T)

    where T is the number of trees.

    This is a row-wise algorithm, so peak extra memory is only the temporary
    boolean overlap vector for the current row, i.e. O(nnz(row) * T) in the
    worst case, not O(nnz(P) * T).
    """
    if not sparse.isspmatrix_csr(P):
        raise TypeError("normalize_oob_training_proximity requires a CSR input matrix.")

    oob = np.asarray(oob_mask, dtype=np.bool_)
    data = P.data
    indices = P.indices
    indptr = P.indptr

    for i in range(P.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue

        cols = indices[start:end]

        shared = np.count_nonzero(oob[cols] & oob[i], axis=1).astype(np.float32)

        keep = shared > 0
        segment = data[start:end]
        segment[keep] /= shared[keep]
        segment[~keep] = 0.0

    P.eliminate_zeros()
    return P


def normalize_oob_oos_proximity(P, oob_mask):
    """
    Normalize out-of-sample OOB proximity blocks by the number of OOB trees
    for each training reference sample.

    Complexity:
        O(nnz(P)) time for the sparse-diagonal scaling
        O(n_samples) extra memory for the diagonal weights
    """
    oob_counts = np.asarray(oob_mask, dtype=np.float32).sum(axis=1)
    oob_counts[oob_counts == 0] = 1.0
    return P @ sparse.diags(1.0 / oob_counts, dtype=np.float32)


def format_output_matrix(M, return_dense=False):
    """
    Format matrix-like outputs as sparse by default, or dense on demand.
    """
    if return_dense and hasattr(M, "toarray"):
        return M.toarray()
    return M
