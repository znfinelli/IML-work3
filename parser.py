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
      - numeric columns: median
      - categorical columns: mode (most frequent value)
    """
    df = df.copy()

    # 1. Replace '?' (common in UCI datasets) and empty strings with NaN
    # regex=False ensures we match these characters literally, not as patterns.
    df.replace('?', np.nan, inplace=True, regex=False)
    df.replace('', np.nan, inplace=True, regex=False)

    # 2. Handle Numeric Columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # 3. Handle Categorical Columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_series = df[col].mode()
            if not mode_series.empty:
                mode_value = mode_series[0]
                df[col] = df[col].fillna(mode_value)
            else:
                df[col] = df[col].fillna("Missing")

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
    """
    # 1) Load full dataset
    df = load_arff(filepath)

    # 2) Identify class column
    if class_column is None:
        class_column = get_class_column_name(df)

    # 3) Identify numeric vs. categorical feature columns
    numeric_cols, categorical_cols = identify_column_types(df, class_column)

    # 4) Handle missing values (includes replacing '?' with NaN)
    df = handle_missing_values(df, numeric_cols, categorical_cols)

    # 5) Encode categorical feature columns (NOT the class)
    feature_encoders: Dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        if col == class_column:
            continue
        le = LabelEncoder()
        # Convert to string to handle mixed types safely
        df[col] = le.fit_transform(df[col].astype(str))
        feature_encoders[col] = le

    # 6) Optionally encode the class column
    if drop_class:
        y = None
        class_encoder: Optional[LabelEncoder] = None
    else:
        class_encoder = LabelEncoder()
        df[class_column] = class_encoder.fit_transform(df[class_column].astype(str))
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