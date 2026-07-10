import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

def prep_x(x, default=5,  preprocessing_meta = None):
    """
    Preprocess input data.
    
    Parameters
    ----------
    x : pd.DataFrame
        Input dataframe.
    to_numeric : list, optional
        List of column names to force convert to numeric, if None then these are selected according to default.
    to_categorical : list, optional
        List of column names to force convert to categorical, if None then these are selected according to default.
    default : int
        Threshold for unique values to decide between numeric and factor for integers - 
        equal or less values than this default will cause the integer to be converted into a category
        
    Returns
    -------
    dict
        Contains 'x' (processed DataFrame), 'to_numeric' (list), and 'to_categorical' (list).
    """
    x = x.copy() if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
    idx_char = x.select_dtypes(include = ['object', 'string']).columns
    idx_numeric = x.select_dtypes(include = 'integer').columns
    idx_bool = x.select_dtypes(include = 'bool').columns

    if len(idx_char) > 0:
        for col in idx_char:
            x[col] = x[col].astype('category')
    
    if len(idx_bool) > 0:
        for col in idx_bool:
            x[col] = x[col].astype('category')

    unsupported = x.columns.difference(idx_char.union(idx_numeric).union(idx_bool))
    if not unsupported.empty:
        raise TypeError(f"Unhandled columns: {unsupported}. prep_x can only handle columns of type object, string, number and bool (in particular, no datetime).")
    
    is_test = preprocessing_meta is not None
    to_numeric = []
    to_categorical = []

    if is_test:
        to_numeric = preprocessing_meta['to_numeric']
        to_categorical = preprocessing_meta['to_categorical']    
    else:
        if len(idx_numeric) > 0:
            to_numeric = idx_numeric[x[idx_numeric].nunique() >= default].tolist()
            to_categorical = idx_numeric[x[idx_numeric].nunique() < default].tolist()
        
    if to_numeric:
        if not is_test:
            warnings.warn(f"Recoding integers with more than {default} unique values as numeric. "
                          'To override, explicitly code these variables as categories.')
        for col in to_numeric:
            x[col] = pd.to_numeric(x[col])

    
    if to_categorical:
        if is_test:
            ordinal_encoder = preprocessing_meta['ordinal_encoder']
            x[to_categorical] = ordinal_encoder.transform(x[to_categorical])
        else:
            ordinal_encoder = OrdinalEncoder(handle_unknown='error')
            x[to_categorical] = ordinal_encoder.fit_transform(x[to_categorical])
    else:
        ordinal_encoder = None

    cols_to_onehot = idx_char.union(idx_bool)
    cols_to_onehot = [c for c in cols_to_onehot if c in x.columns]

    if cols_to_onehot:
        if is_test:
            onehot_encoder = preprocessing_meta['onehot_encoder']
            ohe_data = onehot_encoder.transform(x[cols_to_onehot])
        else:
            onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe_data = onehot_encoder.fit_transform(x[cols_to_onehot])
        feature_names = onehot_encoder.get_feature_names_out(cols_to_onehot)
        ohe_df = pd.DataFrame(ohe_data, columns = feature_names, index = x.index)

        x = x.drop(columns=cols_to_onehot)
        x = pd.concat([x, ohe_df], axis = 1)
    else:
        onehot_encoder = None
    preprocessing_meta = {
        'to_numeric': to_numeric,
        'to_categorical': to_categorical,
        'ordinal_encoder': ordinal_encoder,
        'onehot_encoder': onehot_encoder
    }
    return x, preprocessing_meta


def post_x(x, meta, preprocessing_meta, round_vals = True):
    """
    Post-process data to restore original structure.
    1. Reverses One-Hot and Ordinal Encodings (if preprocessing_meta provided).
    2. Applies rounding and type casting based on meta (from encode).
    """
    
    if not isinstance(x, pd.DataFrame):
        cols = meta['metadata']['variable'] if 'metadata' in meta else None
        x = pd.DataFrame(x, columns=cols)

    onehot_encoder = preprocessing_meta['onehot_encoder']
    # indicates there are categorial columns
    if onehot_encoder is not None:
        ohe_cols = onehot_encoder.get_feature_names_out()

        x_restored = onehot_encoder.inverse_transform(x[ohe_cols])
        original_cols = onehot_encoder.feature_names_in_
        x_restored = pd.DataFrame(x_restored, columns=original_cols, index=x.index)

        # Check for boolean type
        for i, col_name in enumerate(original_cols):
            cats = onehot_encoder.categories_[i]
            # Check if categories are exactly [False, True] or [0, 1] treating as bools
            if len(cats) == 2 and {True, False}.issubset(set(cats)):
                x_restored[col_name] = x_restored[col_name].astype(bool)
            
        x = x.drop(columns=ohe_cols)
        x = pd.concat([x, x_restored], axis = 1)

    ordinal_encoder = preprocessing_meta['ordinal_encoder']
    to_categorical = preprocessing_meta['to_categorical']
    # indicates there are categorical columns
    if ordinal_encoder is not None and to_categorical:
        x[to_categorical] = ordinal_encoder.inverse_transform(x[to_categorical].round())
            
    input_class = meta.get('input_class', ['pd.DataFrame'])
    meta_df = meta['metadata'].set_index('variable')
    
    idx_numeric = meta_df.index[meta_df['class'] == 'numeric']
    idx_integer = meta_df.index[meta_df['class'] == 'integer']

    # Numeric (Rounding)
    if round_vals:
        for col in idx_numeric:
            if col in x.columns:  # <--- Prevents KeyError on 'Department_HR'
                decimals = meta_df.loc[col, 'decimals']
                x[col] = x[col].astype(float).round(int(decimals))

    # Integers
    for col in idx_integer:
        if col in x.columns:  # <--- Prevents KeyError
            x[col] = pd.to_numeric(x[col], errors='coerce').fillna(0)
            if round_vals:
                x[col] = x[col].round().astype(int)
            else:
                x[col] = x[col].astype(int)
    # Note: Categoricals/Factors are already restored by inverse_transform above.
    # We can optionally cast them to 'category' dtype here if desired, 
    # but they are already back to their original String/Int representation.
    
    # Export
    if 'pd.DataFrame' in input_class or 'tbl_df' in input_class:
        return x
    elif 'matrix' in input_class or 'np.ndarray' in input_class:
        return x.to_numpy()
    
    return x

