"""
Parser / preprocessing utilities for Work 3 (clustering + PCA).

This script handles all data loading and preprocessing for the .arff files
provided for the assignment. It includes functions for:

- Loading a .arff file into a pandas DataFrame.
- Identifying numeric vs. categorical feature columns.
- Imputing missing values (median for numeric, mode for categorical).
- Label-encoding categorical feature columns (and optionally the class).
- Normalizing numeric features to a [0, 1] range with MinMaxScaler.
- A main helper to preprocess a SINGLE .arff file for clustering / PCA:
  preprocess_single_arff(...) -> X, y, info
"""

from scipy.io import arff
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import List, Tuple, Dict, Any, Optional


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def load_arff(filepath: str) -> pd.DataFrame:
    """
    Loads an .arff file from the given path into a pandas DataFrame.
    Uses scipy.io.arff.loadarff and decodes byte-string columns to str.
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Decode byte strings (loaded by scipy) to Python strings for categoricals.
    for col in df.columns:
        if df[col].dtype == object:
            # when dtype is 'object', values are often bytes
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
            )

    return df


def get_class_column_name(df: pd.DataFrame) -> str:
    """
    Finds the name of the class/label column.
    By default we assume the class is the LAST column in the DataFrame.
    """
    return df.columns[-1]


def identify_column_types(
    df: pd.DataFrame,
    class_column: str
) -> Tuple[List[str], List[str]]:
    """
    Separates feature columns (excluding the class column) into:
      - numeric_cols: float64 / int64
      - categorical_cols: everything else (object, bool, etc.)
    """
    feature_columns = [col for col in df.columns if col != class_column]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_columns:
        if df[col].dtype in ("float64", "int64", "float32", "int32"):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    """
    Imputes missing values in the DataFrame:
      - Numeric columns: median
      - Categorical columns: mode (most frequent value)
    """
    df = df.copy()

    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    return df


# ---------------------------------------------------------------------
# Single-file preprocessing
# ---------------------------------------------------------------------

def preprocess_single_arff(
    filepath: str,
    class_column: Optional[str] = None,
    drop_class: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Preprocesses a SINGLE .arff file for clustering / PCA experiments.

    Steps:
      1. Load full dataset (no folds).
      2. Identify class column (default: last column).
      3. Detect numeric and categorical feature columns (excluding class).
      4. Impute missing values.
      5. Label-encode categorical feature columns (NOT the class).
      6. Optionally label-encode the class column and return y.
      7. Scale numeric feature columns to [0, 1] with MinMaxScaler.
      8. Build feature matrix X = all features except class.

    Args:
        filepath:
            Path to the .arff file (full dataset, no train/test splits).
        class_column:
            Name of the class column. If None, the last column of the DataFrame
            is used.
        drop_class:
            If True, the function returns y = None and does not encode the
            class column. If False, y is returned as encoded integers.

    Returns:
        X:
            Numpy array of shape (n_samples, n_features) with preprocessed
            feature values (numeric + encoded categoricals).
        y:
            Numpy array of shape (n_samples,) with encoded class labels,
            or None if drop_class=True.
        info:
            Dictionary containing metadata:
              - "class_column": name of the class column
              - "numeric_cols": list of numeric feature column names
              - "categorical_cols": list of categorical feature column names
              - "feature_encoders": dict[col_name] -> LabelEncoder for features
              - "class_encoder": LabelEncoder for the class (or None)
              - "scaler": MinMaxScaler used for numeric features (or None)
              - "feature_names": list of columns used as features in X
    """
    # 1) Load full dataset
    df = load_arff(filepath)

    # 2) Identify class column
    if class_column is None:
        class_column = get_class_column_name(df)

    # 3) Identify numeric vs. categorical feature columns
    numeric_cols, categorical_cols = identify_column_types(df, class_column)

    # 4) Handle missing values
    df = handle_missing_values(df, numeric_cols, categorical_cols)

    # 5) Encode categorical feature columns (NOT the class)
    feature_encoders: Dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        if col == class_column:
            # class is handled separately below
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        feature_encoders[col] = le

    # 6) Optionally encode the class column
    if drop_class:
        y = None
        class_encoder: Optional[LabelEncoder] = None
    else:
        class_encoder = LabelEncoder()
        df[class_column] = class_encoder.fit_transform(df[class_column])
        y = df[class_column].values

    # 7) Normalize numeric feature columns to [0, 1]
    if numeric_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        scaler = None

    # 8) Build X (all features except the class column)
    feature_names = [col for col in df.columns if col != class_column]
    X = df[feature_names].values

    info: Dict[str, Any] = {
        "class_column": class_column,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_encoders": feature_encoders,
        "class_encoder": class_encoder,
        "scaler": scaler,
        "feature_names": feature_names,
    }

    return X, y, info
