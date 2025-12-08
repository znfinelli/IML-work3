"""
Parser / preprocessing utilities for Work 3 (Clustering + PCA).

This script handles all data loading and preprocessing for the .arff files
provided for the assignment. It facilitates the first task of the project:
reading ARFF files, handling mixed attribute types (numerical and categorical),
and managing missing values as per the project specifications.

References
----------
[1] Work 3 Description, UB, 2025, "1.1.1 Tasks", pp. 2.
[2] Work 3 Description, UB, 2025, "Code and Packages", pp. 10 (Session 1 Slides).
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

    This implements Task 1.1.1.1: "Implement your code for reading the arff file
    in Python and store the information in memory" [Work 3 Description, UB, 2025].
    It utilizes scipy.io.arff as permitted by the assignment guidelines.

    Parameters
    ----------
    filepath : str
        The relative or absolute path to the .arff file.

    Returns
    -------
    df : pd.DataFrame
        The loaded data containing features and class labels.
    """
    # Load data using scipy as recommended in [Work 3 Description, UB, 2025]
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Decode byte strings (loaded by scipy) to Python strings for categorical columns
    for col in df.columns:
        if df[col].dtype == object:
            # when dtype is 'object', values are often bytes in arff loads
            df[col] = df[col].apply(
                lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
            )

    return df


def get_class_column_name(df: pd.DataFrame) -> str:
    """
    Finds the name of the class/label column.

    By default, we assume the class is the LAST column in the DataFrame,
    which is standard for UCI repositories used in this coursework.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    str
        The name of the last column.
    """
    return df.columns[-1]


def identify_column_types(
    df: pd.DataFrame,
    class_column: str
) -> Tuple[List[str], List[str]]:
    """
    Separates feature columns into numeric and categorical lists.

    This assists in handling "numerical and categorical data" as required
    by Task 1.1.1.1 [Work 3 Description, UB, 2025].

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    class_column : str
        The name of the target column to exclude from feature lists.

    Returns
    -------
    numeric_cols : List[str]
        List of column names containing float/int data.
    categorical_cols : List[str]
        List of column names containing object/string/bool data.
    """
    feature_columns = [col for col in df.columns if col != class_column]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_columns:
        # Check against standard pandas numeric types
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
    Imputes missing values in the DataFrame.

    Implements the requirement: "some of the data sets ... may also contain
    missing values" [Work 3 Description, UB, 2025].
    - Numeric columns: Imputed with Median.
    - Categorical columns: Imputed with Mode (Most Frequent).

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataframe.
    numeric_cols : List[str]
        List of numeric column names.
    categorical_cols : List[str]
        List of categorical column names.

    Returns
    -------
    df : pd.DataFrame
        The cleaned dataframe with no NaN values.
    """
    df = df.copy()

    # 1. Replace common missing markers (',', '?', '') with NaN
    # Note: Regex=False ensures we treat '?' as a literal string.
    df = df.replace(['?', '.', ''], np.nan, regex=False)

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
                # Fallback if the column is entirely empty
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

    This function orchestrates loading, cleaning, normalization, and encoding.
    It prepares the data for algorithms like K-Means and Agglomerative Clustering
    which require numerical input (One-Hot Encoding for categoricals).

    Parameters
    ----------
    filepath : str
        Path to the .arff file.
    class_column : str, optional
        Name of the target column. If None, it is inferred.
    drop_class : bool, default=False
        If True, y is returned as None.

    Returns
    -------
    X : np.ndarray
        The feature matrix (normalized, one-hot encoded).
    y : np.ndarray or None
        The label encoded target vector (for validation metrics).
    info : Dict[str, Any]
        Metadata about the preprocessing steps (encoders, scalers, feature names).
    """
    # 1) Load full dataset
    df = load_arff(filepath)

    # 2) Identify class column
    if class_column is None:
        class_column = get_class_column_name(df)

    # Attempt to convert ALL feature columns to numeric first to handle "dirty" numbers.
    for col in df.columns:
        if col == class_column:
            continue

        # 'coerce' turns non-numeric strings into NaN.
        # We only apply this transformation if the column is genuinely numeric.
        converted = pd.to_numeric(df[col], errors='coerce')

        # Heuristic: If > 50% of the column is valid numbers, treat it as numeric.
        if converted.notna().mean() > 0.5:
            df[col] = converted

    # 3) Identify numeric vs. categorical feature columns
    numeric_cols, categorical_cols = identify_column_types(df, class_column)

    # 4) Handle missing values
    df = handle_missing_values(df, numeric_cols, categorical_cols)

    # 5) Handle Class Column (for Validation Metrics)
    y = None
    class_encoder = None
    if not drop_class:
        class_encoder = LabelEncoder()
        # Ensure class column is string before encoding to handle mixed types
        y = class_encoder.fit_transform(df[class_column].astype(str))

    df_features = df.drop(columns=[class_column])

    # 6) Normalize numeric features
    # Standard practice for distance-based algorithms (K-Means, Agglomerative)
    if numeric_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_features[numeric_cols] = scaler.fit_transform(df_features[numeric_cols])
    else:
        scaler = None

    # 7) One-Hot Encoding for Categorical Features
    # Essential for computing Euclidean distances in clustering
    if categorical_cols:
        df_features = pd.get_dummies(
            df_features,
            columns=categorical_cols,
            prefix=categorical_cols,
            dtype=int
        )

    # 8) Build X matrix
    X = df_features.values

    info: Dict[str, Any] = {
        "class_column": class_column,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_encoders": "OneHot (pd.get_dummies)",
        "class_encoder": class_encoder,
        "scaler": scaler,
        "feature_names": list(df_features.columns),
    }

    return X, y, info