import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, quantile_transform



def extract_columns_types(df: pd.DataFrame) -> Tuple[List[str], List[str], str]:
    """
    Extract column groupings from a DataFrame.

    Returns:
      - categorical columns (dtype object/category)
      - numerical columns (int/float), *excluding* the target
      - the target column name (first column matching two digits in its name)
    """
    # extract object or categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # extract int64 or float64 columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # find first column name with at least two digits
    target = [
        col for col in df.columns
        if isinstance(col, str) and re.search(r'\d.*\d', col)
    ]
    
    if len(target) !=1:
        raise ValueError(
            "Expected exactly one target column but got: {len(target)}:{target}  "
        )

    
    target_col = target[0]

    # if the target was detected as numeric, remove it
    if target_col in num_cols:
        num_cols.remove(target_col)

    return cat_cols, num_cols, target_col



def output_transform(
    target_col: pd.Series,
    transform_type: Optional[str] = None
) -> pd.Series:
    """
    Transform the target series.

    Args:
      target_col: pandas Series
      transform_type: 'none', 'log', or 'quantile'

    Returns:
      Transformed Series.
    """
    if not isinstance(target_col, pd.Series):
        raise TypeError("target_col must be a pandas Series")

    if transform_type in (None, 'none'):
        return target_col
    elif transform_type == 'log':
        return np.log10(target_col)
    elif transform_type == 'quantile':
        arr = target_col.values.reshape(-1, 1)
        qt = quantile_transform(
            arr,
            n_quantiles=min(1000, len(arr)),
            output_distribution='uniform',
            copy=True
        )
        return pd.Series(qt.flatten(), index=target_col.index)
    else:
        raise ValueError(
            f"Unsupported transform_type '{transform_type}'. "
            "Choose 'none', 'log', or 'quantile'."
        )



def fill_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaNs based on column name patterns:
      - columns containing 'crop' → 'No_crop'
      - columns containing 'organic' or 'mineral' → 0
    """
    out = df.copy()
    for col in out.columns:
        if 'crop' in col:
            out[col] = out[col].fillna("No_crop")
        elif 'organic' in col or 'mineral' in col:
            out[col] = out[col].fillna(0)
    return out


def select_columns(cols: List[str]) -> FunctionTransformer:
    """
    Returns a transformer that selects exactly the given columns from a DataFrame.
    """
    return FunctionTransformer(lambda X: X[cols], validate=False)


