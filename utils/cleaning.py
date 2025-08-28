import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def apply_schema(df: pd.DataFrame, schema_overrides: Dict[str, str]) -> pd.DataFrame:
    """
    Apply data type schema overrides to dataframe
    
    Args:
        df: Input dataframe
        schema_overrides: Dictionary mapping column names to desired types
        
    Returns:
        pd.DataFrame: DataFrame with updated types
    """
    df_copy = df.copy()
    
    for col, dtype in schema_overrides.items():
        if col not in df_copy.columns:
            continue
            
        try:
            if dtype == 'categorical':
                df_copy[col] = df_copy[col].astype('category')
            elif dtype == 'numeric':
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            elif dtype == 'string':
                df_copy[col] = df_copy[col].astype('string')
            elif dtype == 'datetime':
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            elif dtype == 'boolean':
                df_copy[col] = df_copy[col].astype('boolean')
        except Exception as e:
            print(f"Warning: Could not convert {col} to {dtype}: {str(e)}")
    
    return df_copy

def reverse_code_items(df: pd.DataFrame, items_to_reverse: List[str], 
                      scale_range: Tuple[int, int]) -> pd.DataFrame:
    """
    Reverse code specified items (common in psychology)
    
    Args:
        df: Input dataframe
        items_to_reverse: List of column names to reverse code
        scale_range: Tuple of (min_value, max_value) for the scale
        
    Returns:
        pd.DataFrame: DataFrame with reverse-coded items
    """
    df_copy = df.copy()
    min_val, max_val = scale_range
    
    for item in items_to_reverse:
        if item in df_copy.columns:
            # Formula: new_value = (max + min) - old_value
            df_copy[f"{item}_reversed"] = (max_val + min_val) - df_copy[item]
            # Optionally replace original
            df_copy[item] = df_copy[f"{item}_reversed"]
            df_copy.drop(f"{item}_reversed", axis=1, inplace=True)
    
    return df_copy

def handle_missing_values(df: pd.DataFrame, policy: str, 
                         columns: List[str] = None, 
                         fill_value: Any = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle missing values according to specified policy
    
    Args:
        df: Input dataframe
        policy: 'drop_rows', 'drop_columns', 'impute_mean', 'impute_median', 
               'impute_mode', 'impute_constant'
        columns: Specific columns to apply policy to (None = all columns)
        fill_value: Value to use for constant imputation
        
    Returns:
        Tuple of (cleaned_dataframe, summary_dict)
    """
    df_copy = df.copy()
    original_shape = df_copy.shape
    
    if columns is None:
        columns = df_copy.columns.tolist()
    
    summary = {
        'policy': policy,
        'original_shape': original_shape,
        'columns_processed': columns,
        'rows_dropped': 0,
        'columns_dropped': 0,
        'values_imputed': 0
    }
    
    if policy == 'drop_rows':
        # Drop rows with any missing values in specified columns
        initial_rows = len(df_copy)
        df_copy = df_copy.dropna(subset=columns)
        summary['rows_dropped'] = initial_rows - len(df_copy)
        
    elif policy == 'drop_columns':
        # Drop columns with any missing values
        initial_cols = len(df_copy.columns)
        df_copy = df_copy.dropna(axis=1, subset=df_copy.index)
        summary['columns_dropped'] = initial_cols - len(df_copy.columns)
        
    elif policy in ['impute_mean', 'impute_median', 'impute_mode', 'impute_constant']:
        for col in columns:
            if col not in df_copy.columns:
                continue
                
            missing_count = df_copy[col].isnull().sum()
            if missing_count == 0:
                continue
                
            if policy == 'impute_mean':
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    fill_val = df_copy[col].mean()
                    df_copy[col].fillna(fill_val, inplace=True)
                    summary['values_imputed'] += missing_count
                    
            elif policy == 'impute_median':
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    fill_val = df_copy[col].median()
                    df_copy[col].fillna(fill_val, inplace=True)
                    summary['values_imputed'] += missing_count
                    
            elif policy == 'impute_mode':
                mode_val = df_copy[col].mode()
                if len(mode_val) > 0:
                    df_copy[col].fillna(mode_val[0], inplace=True)
                    summary['values_imputed'] += missing_count
                    
            elif policy == 'impute_constant':
                df_copy[col].fillna(fill_value, inplace=True)
                summary['values_imputed'] += missing_count
    
    summary['final_shape'] = df_copy.shape
    return df_copy, summary

def encode_categorical_variables(df: pd.DataFrame, categorical_cols: List[str], 
                               method: str = 'one_hot', drop_first: bool = True,
                               ordinal_orders: Dict[str, List] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names
        method: 'one_hot', 'label', or 'ordinal'
        drop_first: Whether to drop first category in one-hot encoding
        ordinal_orders: Dict mapping column names to ordered category lists
        
    Returns:
        Tuple of (encoded_dataframe, encoding_info)
    """
    df_copy = df.copy()
    encoding_info = {
        'method': method,
        'categorical_cols': categorical_cols,
        'new_columns': [],
        'dropped_columns': [],
        'encoders': {}
    }
    
    if ordinal_orders is None:
        ordinal_orders = {}
    
    for col in categorical_cols:
        if col not in df_copy.columns:
            continue
            
        try:
            if method == 'one_hot':
                # One-hot encoding
                dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=drop_first)
                df_copy = pd.concat([df_copy, dummies], axis=1)
                encoding_info['new_columns'].extend(dummies.columns.tolist())
                encoding_info['dropped_columns'].append(col)
                df_copy.drop(col, axis=1, inplace=True)
                
            elif method == 'label':
                # Label encoding
                le = LabelEncoder()
                df_copy[f"{col}_encoded"] = le.fit_transform(df_copy[col].astype(str))
                encoding_info['encoders'][col] = le
                encoding_info['new_columns'].append(f"{col}_encoded")
                encoding_info['dropped_columns'].append(col)
                df_copy.drop(col, axis=1, inplace=True)
                
            elif method == 'ordinal':
                # Ordinal encoding with specified order
                if col in ordinal_orders:
                    order = ordinal_orders[col]
                    df_copy[f"{col}_ordinal"] = df_copy[col].map({v: i for i, v in enumerate(order)})
                    encoding_info['new_columns'].append(f"{col}_ordinal")
                    encoding_info['dropped_columns'].append(col)
                    df_copy.drop(col, axis=1, inplace=True)
                else:
                    # Default to label encoding if no order specified
                    le = LabelEncoder()
                    df_copy[f"{col}_encoded"] = le.fit_transform(df_copy[col].astype(str))
                    encoding_info['encoders'][col] = le
                    encoding_info['new_columns'].append(f"{col}_encoded")
                    encoding_info['dropped_columns'].append(col)
                    df_copy.drop(col, axis=1, inplace=True)
                    
        except Exception as e:
            print(f"Warning: Could not encode {col}: {str(e)}")
    
    return df_copy, encoding_info

def handle_outliers(df: pd.DataFrame, numeric_cols: List[str], 
                   method: str = 'none', threshold: float = 3.0,
                   percentiles: Tuple[float, float] = (1, 99)) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle outliers in numeric columns
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric column names
        method: 'none', 'remove', 'winsorize', 'cap'
        threshold: Z-score threshold for outlier detection
        percentiles: Percentile bounds for winsorizing
        
    Returns:
        Tuple of (processed_dataframe, outlier_info)
    """
    df_copy = df.copy()
    outlier_info = {
        'method': method,
        'threshold': threshold,
        'percentiles': percentiles,
        'outliers_detected': {},
        'rows_removed': 0,
        'values_modified': 0
    }
    
    if method == 'none':
        return df_copy, outlier_info
    
    for col in numeric_cols:
        if col not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
            
        # Detect outliers using z-score
        z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
        outliers_mask = z_scores > threshold
        outlier_count = outliers_mask.sum()
        outlier_info['outliers_detected'][col] = outlier_count
        
        if outlier_count == 0:
            continue
            
        if method == 'remove':
            # Remove rows with outliers
            df_copy = df_copy[~outliers_mask]
            outlier_info['rows_removed'] += outlier_count
            
        elif method == 'winsorize':
            # Winsorize to percentiles
            lower_pct, upper_pct = percentiles
            lower_bound = df_copy[col].quantile(lower_pct / 100)
            upper_bound = df_copy[col].quantile(upper_pct / 100)
            
            modified = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).sum()
            df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)
            outlier_info['values_modified'] += modified
            
        elif method == 'cap':
            # Cap at mean ± threshold * std
            mean_val = df_copy[col].mean()
            std_val = df_copy[col].std()
            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val
            
            modified = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).sum()
            df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)
            outlier_info['values_modified'] += modified
    
    return df_copy, outlier_info

def standardize_variables(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Standardize numeric variables (z-score normalization)
    
    Args:
        df: Input dataframe
        columns: List of column names to standardize
        
    Returns:
        Tuple of (standardized_dataframe, standardization_info)
    """
    df_copy = df.copy()
    scaler = StandardScaler()
    
    standardization_info = {
        'columns': [],
        'means': {},
        'stds': {},
        'scaler': scaler
    }
    
    valid_cols = [col for col in columns if col in df_copy.columns and 
                  pd.api.types.is_numeric_dtype(df_copy[col])]
    
    if valid_cols:
        df_copy[valid_cols] = scaler.fit_transform(df_copy[valid_cols])
        
        # Store info for later reference
        standardization_info['columns'] = valid_cols
        for i, col in enumerate(valid_cols):
            standardization_info['means'][col] = scaler.mean_[i]
            standardization_info['stds'][col] = scaler.scale_[i]
    
    return df_copy, standardization_info

def create_interaction_terms(df: pd.DataFrame, interactions: List[Tuple[str, str]], 
                           max_interactions: int = 10) -> pd.DataFrame:
    """
    Create interaction terms between specified variable pairs
    
    Args:
        df: Input dataframe
        interactions: List of (var1, var2) tuples
        max_interactions: Maximum number of interaction terms to create
        
    Returns:
        pd.DataFrame: DataFrame with interaction terms added
    """
    df_copy = df.copy()
    
    # Limit number of interactions
    interactions = interactions[:max_interactions]
    
    for var1, var2 in interactions:
        if var1 in df_copy.columns and var2 in df_copy.columns:
            # Check if both variables are numeric
            if (pd.api.types.is_numeric_dtype(df_copy[var1]) and 
                pd.api.types.is_numeric_dtype(df_copy[var2])):
                
                interaction_name = f"{var1}_x_{var2}"
                df_copy[interaction_name] = df_copy[var1] * df_copy[var2]
    
    return df_copy

def get_cleaning_summary(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
    """
    Generate summary of cleaning operations performed
    
    Args:
        original_df: Original dataframe before cleaning
        cleaned_df: Dataframe after cleaning operations
        
    Returns:
        Dict: Summary of changes made
    """
    summary = {
        'original_shape': original_df.shape,
        'final_shape': cleaned_df.shape,
        'rows_changed': original_df.shape[0] - cleaned_df.shape[0],
        'columns_changed': cleaned_df.shape[1] - original_df.shape[1],
        'original_missing': original_df.isnull().sum().sum(),
        'final_missing': cleaned_df.isnull().sum().sum(),
        'missing_reduction': original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum(),
        'new_columns': [col for col in cleaned_df.columns if col not in original_df.columns],
        'removed_columns': [col for col in original_df.columns if col not in cleaned_df.columns],
        'memory_change_mb': (cleaned_df.memory_usage(deep=True).sum() - 
                            original_df.memory_usage(deep=True).sum()) / (1024**2)
    }
    
    return summary