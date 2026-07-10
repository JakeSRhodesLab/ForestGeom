import pandas as pd
import numpy as np

def reconstruction_error(Xhat, X):
    """
    Mixed-type Reconstruction Error
    
    Parameters:
    -----------
    Xhat : pd.DataFrame
        Reconstructed dataset
    X : pd.DataFrame
        Ground truth dataset
        
    Returns:
    --------
    dict
        A dictionary containing column-wise reconstruction accuracy, and the average
        reconstruction error for categorical and numeric variables. Values lie
        between 0-1, where 1 represents perfect reconstruction, and 0 represents
        no reconstruction (maximum distortion).
    """
    num_error = {}
    cat_error = {}
    
    # Ensure inputs are pandas DataFrames
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(Xhat, pd.DataFrame):
        Xhat = pd.DataFrame(Xhat)
        
    # Variables to track sums for overall error
    total_num_sum = 0
    total_cat_sum = 0
    
    for col in X.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(X[col]):
            # Calculate RSS and TSS
            # Note: We align indices to ensure element-wise subtraction works correctly
            rss = ((X[col] - Xhat[col]) ** 2).sum()
            tss = ((X[col] - X[col].mean()) ** 2).sum()
            
            # Calculate R^2 clamped at 0
            # If TSS is 0 (constant variable), we avoid division by zero errors
            if tss == 0:
                # If both are constant and equal, score is 1, else 0
                score = 1.0 if rss == 0 else 0.0
            else:
                score = max(1 - (rss / tss), 0)
                
            num_error[col] = score
            total_num_sum += score
            
        else:
            # Categorical: Accuracy
            # Convert to string to ensure strictly categorical comparison
            yhat = Xhat[col].astype(str)
            y = X[col].astype(str)
            
            # Accuracy: sum of matches / total rows
            score = (yhat == y).mean()
            
            cat_error[col] = score
            total_cat_sum += score

    # Calculate Numeric Average
    if num_error:
        num_avg = np.mean(list(num_error.values()))
    else:
        num_avg = 'No variables'

    # Calculate Categorical Average
    if cat_error:
        cat_avg = np.mean(list(cat_error.values()))
    else:
        cat_avg = 'No variables'

    # Calculate Overall Error (Average across all columns)
    # The R code sums all individual scores and divides by total columns
    ovr_error = (total_num_sum + total_cat_sum) / len(X.columns)

    return {
        'num_error': num_error,
        'cat_error': cat_error,
        'num_avg': num_avg,
        'cat_avg': cat_avg,
        'ovr_error': ovr_error
    }