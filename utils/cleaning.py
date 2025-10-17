import pandas as pd
import numpy as np
from typing import Dict, List, Optional
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




def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize all string columns: lowercase, strip whitespace, normalize common variations
    
    This preprocessing happens BEFORE any encoding or detection to ensure consistency.
    Args:
        df: Input dataframe
    Returns:
        DataFrame with cleaned text columns
    """
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=['object']).columns:
        # Convert to string, lowercase, and strip whitespace
        df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()  
        # Normalize common variations
        df_clean[col] = df_clean[col].replace({
            'yes': 'yes', 'y': 'yes', 'yeah': 'yes',
            'no': 'no', 'n': 'no', 'nope': 'no',
            'maybe': 'neutral', 'unsure': 'neutral', 'uncertain': 'neutral'
        })
        # Handle NaN and empty strings
        df_clean[col] = df_clean[col].replace({'nan': np.nan, '': np.nan})
    return df_clean

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
# ============================================================
# PREDEFINED ORDINAL MAPPINGS
# ============================================================

LIKERT_MAPPINGS = {
    # 5-point agreement scale
    'agreement_5': {
        'strongly disagree': 1,
        'disagree': 2,
        'neutral': 3,
        'agree': 4,
        'strongly agree': 5
    },
    # 5-point frequency scale
    'frequency_5': {
        'never': 1,
        'rarely': 2,
        'sometimes': 3,
        'often': 4,
        'always': 5
    },
    
    # 5-point satisfaction scale
    'satisfaction_5': {
        'very dissatisfied': 1,
        'dissatisfied': 2,
        'neutral': 3,
        'satisfied': 4,
        'very satisfied': 5
    },
    
    # 5-point confidence scale
    'confidence_5': {
        'not confident at all': 1,
        'slightly confident': 2,
        'somewhat confident': 3,
        'very confident': 4,
        'extremely confident': 5
    },
    
    # 5-point fear/anxiety scale
    'fear_5': {
        'not fearful at all': 1,
        'slightly fearful': 2,
        'somewhat fearful': 3,
        'very fearful': 4,
        'extremely fearful': 5
    },
    
    # 5-point helpfulness scale
    'helpfulness_5': {
        'not at all helpful': 1,
        'slightly helpful': 2,
        'somewhat helpful': 3,
        'very helpful': 4,
        'extremely helpful': 5
    },
    
    # 5-point efficiency scale
    'efficiency_5': {
        'not at all efficient': 1,
        'slightly efficient': 2,
        'somewhat efficient': 3,
        'very efficient': 4,
        'extremely efficient': 5
    },
    
    # 5-point comfort scale
    'comfort_5': {
        'much less comfortable': 1,
        'a little less comfortable': 2,
        'no change': 3,
        'a little more comfortable': 4,
        'much more comfortable': 5
    },
    
    # 5-point change scale (for anxiety)
    'change_5': {
        'increased very much': 1,
        'increased a little': 2,
        'no change': 3,
        'decreased a little': 4,
        'decreased very much': 5
    },
    
    # 4-point scale (no neutral)
    'degree_4': {
        'not at all': 1,
        'a little': 2,
        'quite a bit': 3,
        'very much': 4
    },
    
    # Information preference scale
    'information_5': {
        'i wanted a lot less information': 1,
        'i wanted a little less information': 2,
        'neutral': 3,
        'i wanted a little more information': 4,
        'i wanted a lot more information': 5
    },
    
    # Education level (ordinal)
    'education': {
        'no formal schooling': 1,
        'primary school': 2,
        'matric / o-levels': 3,
        'intermediate / a-levels': 4,
        'graduate (ba/bsc/bcom)': 5,
        'postgraduate (ma/msc/mbbs etc.)': 6
    },
    
    # Anxiety level categories
    'anxiety_level': {
        'low level': 1,
        'intermediate level': 2,
        'intermidate level': 2,  # Handle typo
        'high level': 3
    },
    'agreement_5': {
        'strongly disagree': 1, 'disagree': 2, 'neutral': 3, 
        'agree': 4, 'strongly agree': 5
    },
    'frequency_5': {
        'never': 1, 'rarely': 2, 'sometimes': 3, 
        'often': 4, 'always': 5
    },
    'satisfaction_5': {
        'very dissatisfied': 1, 'dissatisfied': 2, 'neutral': 3, 
        'satisfied': 4, 'very satisfied': 5
    },
    'confidence_5': {
        'not confident at all': 1, 'slightly confident': 2, 'somewhat confident': 3, 
        'very confident': 4, 'extremely confident': 5
    },
    'fear_5': {
        'not fearful at all': 1, 'slightly fearful': 2, 'somewhat fearful': 3, 
        'very fearful': 4, 'extremely fearful': 5
    },
    'helpfulness_5': {
        'not at all helpful': 1, 'slightly helpful': 2, 'somewhat helpful': 3, 
        'very helpful': 4, 'extremely helpful': 5
    },
    'efficiency_5': {
        'not at all efficient': 1, 'slightly efficient': 2, 'somewhat efficient': 3, 
        'very efficient': 4, 'extremely efficient': 5
    },
    'comfort_5': {
        'much less comfortable': 1, 'a little less comfortable': 2, 'no change': 3, 
        'a little more comfortable': 4, 'much more comfortable': 5
    },
    'change_5': {
        'increased very much': 1, 'increased a little': 2, 'no change': 3, 
        'decreased a little': 4, 'decreased very much': 5
    },
    'degree_4': {
        'not at all': 1, 'a little': 2, 'quite a bit': 3, 'very much': 4
    },
    'information_5': {
        'i wanted a lot less information': 1, 'i wanted a little less information': 2, 
        'neutral': 3, 'i wanted a little more information': 4, 
        'i wanted a lot more information': 5
    },
    'education': {
        'no formal schooling': 1, 'primary school': 2, 'matric / o-levels': 3, 
        'intermediate / a-levels': 4, 'graduate (ba/bsc/bcom)': 5, 
        'postgraduate (ma/msc/mbbs etc.)': 6
    },
    'anxiety_level': {
        'low level': 1, 'intermediate level': 2, 'intermidate level': 2, 'high level': 3
    }
}

def detect_ordinal_columns(df: pd.DataFrame, threshold: float = 0.6) -> Dict[str, str]:
    """
    Auto-detect ordinal text columns with relaxed matching
    
    FIXED: Threshold reduced from 0.8 to 0.6 to catch near-matches
    
    Args:
        df: Input dataframe
        threshold: Minimum proportion of values that must match (default 60%)
        
    Returns:
        Dict mapping column names to suggested scale types
    """
    detected = {}
    
    for col in df.columns:
        if df[col].dtype != 'object':
            continue
            
        # Clean text first
        unique_vals = df[col].astype(str).str.lower().str.strip().unique()
        unique_vals = [v for v in unique_vals if v not in ['nan', '', 'none']]
        
        if len(unique_vals) == 0:
            continue
        
        # Try to match against known patterns
        for scale_name, mapping in LIKERT_MAPPINGS.items():
            matches = sum(1 for val in unique_vals if val in mapping)
            match_ratio = matches / len(unique_vals)
            
            if match_ratio >= threshold:
                detected[col] = scale_name
                break
    
    return detected

def encode_ordinal_columns(df: pd.DataFrame, 
                          column_mappings: Optional[Dict[str, str]] = None,
                          auto_detect: bool = True,
                          replace_original: bool = False) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Encode ordinal text columns to numeric values with comprehensive validation
    
    FIXED: 
    - Handles mixed data types and nulls
    - Logs unmapped values explicitly
    - Optional column replacement
    - Robust error handling
    
    Args:
        df: Input dataframe
        column_mappings: Dict mapping column names to scale types
        auto_detect: Whether to automatically detect ordinal columns
        replace_original: If True, replace original columns; if False, keep both
        
    Returns:
        Tuple of (encoded_dataframe, encoding_log)
    """
    
    # STEP 1: Clean text first (FIX #3)
    df_work = clean_text_columns(df)
    df_encoded = df_work.copy()
    
    if column_mappings is None:
        column_mappings = {}
    
    # STEP 2: Auto-detect if requested (FIX #2)
    if auto_detect:
        detected = detect_ordinal_columns(df_work)
        detected.update(column_mappings)  # Manual mappings override auto-detected
        column_mappings = detected
    
    encoding_log = []
    
    for col, scale_type in column_mappings.items():
        if col not in df_encoded.columns:
            continue
        
        if scale_type not in LIKERT_MAPPINGS:
            encoding_log.append({
                'column': col,
                'status': 'error',
                'message': f"Unknown scale type '{scale_type}'"
            })
            continue
        
        mapping = LIKERT_MAPPINGS[scale_type]
        
        # STEP 3: Apply text cleaning to this column again (FIX #3)
        col_clean = (df_encoded[col]
                     .astype(str)
                     .str.lower()
                     .str.strip())
        
        # STEP 4: Map values
        new_col_name = col if replace_original else f"{col}_encoded"
        df_encoded[new_col_name] = col_clean.map(mapping)
        
        # STEP 5: Log unmapped values (FIX #7)
        unmapped_mask = col_clean.notna() & df_encoded[new_col_name].isna()
        unmapped_values = col_clean[unmapped_mask].unique().tolist()
        
        n_mapped = df_encoded[new_col_name].notna().sum()
        n_total = col_clean.notna().sum()
        
        log_entry = {
            'column': col,
            'scale_type': scale_type,
            'new_column': new_col_name,
            'mapped': int(n_mapped),
            'total': int(n_total),
            'success_rate': float(n_mapped / n_total) if n_total > 0 else 0,
            'status': 'success' if len(unmapped_values) == 0 else 'warning',
            'unmapped_values': unmapped_values if len(unmapped_values) > 0 else None
        }
        
        encoding_log.append(log_entry)
        
        # STEP 6: Replace original column if requested (FIX #5)
        if replace_original and new_col_name != col:
            df_encoded.drop(col, axis=1, inplace=True)
    
    return df_encoded, encoding_log

def create_custom_ordinal_mapping(categories: List[str], 
                                 start_value: int = 1) -> Dict[str, int]:
    """
    Create a custom ordinal mapping from a list of categories in order
    
    Args:
        categories: List of categories in order (low to high)
        start_value: Starting numeric value
        
    Returns:
        Dict mapping categories to numbers
    """
    return {cat.lower().strip(): start_value + i 
            for i, cat in enumerate(categories)}


def show_ordinal_encoding_ui(df: pd.DataFrame):
    """
    Streamlit UI component for ordinal encoding
    
    FIXED: Returns df_encoded regardless of button clicks
    """
    import streamlit as st
    
    st.subheader("🔢 Ordinal Encoding")
    
    # First, clean the text
    df_cleaned = clean_text_columns(df)
    
    # Auto-detect ordinal columns
    detected = detect_ordinal_columns(df_cleaned)
    
    if detected:
        st.success(f"Auto-detected {len(detected)} ordinal columns")
        for col, scale in detected.items():
            st.write(f"  • {col} → {scale}")
        
        if st.button("Apply Auto-Detected Encoding"):
            df_encoded, log = encode_ordinal_columns(
                df_cleaned,
                auto_detect=True,
                replace_original=False
            )
            
            st.success("Encoding complete!")
            
            # Show results table
            log_df = pd.DataFrame([
                {
                    'Column': item['column'],
                    'Scale': item['scale_type'],
                    'Mapped': item['mapped'],
                    'Total': item['total'],
                    'Success': f"{item['success_rate']:.0%}",
                    'Status': item['status']
                }
                for item in log
            ])
            st.dataframe(log_df, use_container_width=True)
            
            # Show unmapped values if any
            unmapped = [item for item in log if item.get('unmapped_values')]
            if unmapped:
                st.warning("⚠️ Some values could not be mapped:")
                for item in unmapped:
                    st.write(f"  {item['column']}: {item['unmapped_values']}")
            
            return df_encoded
    else:
        st.info("No ordinal columns auto-detected")
    
    # Manual encoding option
    st.write("**Manual Encoding**")
    text_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
    if text_cols:
        col_to_encode = st.selectbox("Select column to encode", 
                                     options=[None] + text_cols)
        
        if col_to_encode:
            unique_vals = df_cleaned[col_to_encode].dropna().unique()
            st.write(f"Unique values: {list(unique_vals)}")
            
            scale_type = st.selectbox("Select scale type",
                                     options=list(LIKERT_MAPPINGS.keys()))
            
            st.write("Mapping preview:")
            st.json(LIKERT_MAPPINGS[scale_type])
            
            if st.button("Apply Manual Encoding"):
                df_encoded, log = encode_ordinal_columns(
                    df_cleaned,
                    column_mappings={col_to_encode: scale_type},
                    auto_detect=False,
                    replace_original=False
                )
                
                st.success("Encoding complete!")
                st.dataframe(pd.DataFrame(log), use_container_width=True)
                
                return df_encoded
    
    # FIXED: Return df_cleaned even if no encoding applied (FIX #4)
    return df_cleaned


# ============================================================
# INTEGRATION WITH EXISTING MODELING
# ============================================================

def prepare_data_for_analysis(df: pd.DataFrame, 
                              target_col: str,
                              feature_cols: List[str],
                              auto_encode: bool = True) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    """
    Prepare data for statistical analysis with automatic ordinal encoding
    
    FIXED: Integrated with cleaned encoding pipeline
    
    Args:
        df: Input dataframe
        target_col: Target variable column name
        feature_cols: List of feature column names
        auto_encode: Whether to auto-encode ordinal variables
        
    Returns:
        Tuple of (y, X, encoding_info)
    """
    df_working = df[[target_col] + feature_cols].copy()
    
    encoding_info = {
        'columns_encoded': [],
        'encoding_log': []
    }
    
    if auto_encode:
        df_encoded, log = encode_ordinal_columns(
            df_working,
            auto_detect=True,
            replace_original=True
        )
        encoding_info['encoding_log'] = log
        encoding_info['columns_encoded'] = [item['column'] for item in log if item['status'] == 'success']
        df_working = df_encoded
    
    y = df_working[target_col]
    X = df_working[feature_cols]
    
    return y, X, encoding_info

# Add these functions to your utils/cleaning.py file

# ============================================================
# BINARY ENCODING (for 2-category variables)
# ============================================================

def encode_binary_columns(df: pd.DataFrame,
                         column_mappings: Optional[Dict[str, Dict]] = None) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Encode binary categorical columns (2 unique values) to 0/1
    
    Examples:
    - Male/Female → 0/1
    - Yes/No → 0/1
    - Treatment A/Treatment B → 0/1
    
    Args:
        df: Input dataframe
        column_mappings: Dict mapping column names to custom mappings
            Example: {'Gender': {'Male': 0, 'Female': 1}}
        
    Returns:
        Tuple of (encoded_dataframe, encoding_log)
    """
    
    df_encoded = df.copy()
    df_encoded = clean_text_columns(df_encoded)
    encoding_log = []
    
    if column_mappings is None:
        column_mappings = {}
    
    # Find binary columns if not manually specified
    binary_cols = {}
    for col in df_encoded.select_dtypes(include=['object']).columns:
        if col not in column_mappings:
            n_unique = df_encoded[col].nunique()
            if n_unique == 2:
                # Auto-detect: use sorted order (alphabetical)
                values = sorted(df_encoded[col].dropna().unique())
                binary_cols[col] = {values[0]: 0, values[1]: 1}
    
    # Merge with manual mappings (manual takes precedence)
    binary_cols.update(column_mappings)
    
    # Apply encoding
    for col, mapping in binary_cols.items():
        if col not in df_encoded.columns:
            continue
        
        new_col_name = f"{col}_encoded"
        
        # Map values
        df_encoded[new_col_name] = df_encoded[col].map(mapping)
        
        # Log results
        n_mapped = df_encoded[new_col_name].notna().sum()
        n_total = df_encoded[col].notna().sum()
        unmapped = df_encoded[col][df_encoded[new_col_name].isna()].unique()
        
        encoding_log.append({
            'column': col,
            'type': 'binary',
            'new_column': new_col_name,
            'mapping': mapping,
            'mapped': int(n_mapped),
            'total': int(n_total),
            'success_rate': float(n_mapped / n_total) if n_total > 0 else 0,
            'unmapped_values': list(unmapped) if len(unmapped) > 0 else None
        })
        
        # Drop original (optional: set replace_original=True to remove)
        # df_encoded.drop(col, axis=1, inplace=True)
    
    return df_encoded, encoding_log


# ============================================================
# ONE-HOT ENCODING (for multi-category variables)
# ============================================================

def encode_onehot_columns(df: pd.DataFrame,
                         columns: Optional[List[str]] = None,
                         drop_first: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    One-hot encode categorical columns (2+ unique values)
    
    Examples:
    - Tumor Type: Head & Neck, Breast, Pelvis, Brain
      → Creates: Tumor_Type_Breast, Tumor_Type_Head & Neck, Tumor_Type_Pelvis
      (Brain is dropped if drop_first=True)
    
    Args:
        df: Input dataframe
        columns: List of column names to encode (if None, auto-detect)
        drop_first: If True, drop first category to avoid multicollinearity
        
    Returns:
        Tuple of (encoded_dataframe, encoding_log)
    """
    
    df_encoded = df.copy()
    df_encoded = clean_text_columns(df_encoded)
    encoding_log = []
    
    # Auto-detect categorical columns if not specified
    if columns is None:
        columns = []
        for col in df_encoded.select_dtypes(include=['object']).columns:
            n_unique = df_encoded[col].nunique()
            # One-hot for columns with 3+ categories
            if n_unique >= 3:
                columns.append(col)
    
    # Apply one-hot encoding
    for col in columns:
        if col not in df_encoded.columns:
            continue
        
        # Get dummy variables
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first, dtype=int)
        
        new_columns = dummies.columns.tolist()
        
        # Add dummies to dataframe
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Log results
        encoding_log.append({
            'column': col,
            'type': 'one_hot',
            'n_categories': df_encoded[col].nunique(),
            'n_new_columns': len(new_columns),
            'new_columns': new_columns,
            'drop_first': drop_first
        })
        
        # Drop original column
        df_encoded.drop(col, axis=1, inplace=True)
    
    return df_encoded, encoding_log


# ============================================================
# AUTO-DETECT AND ENCODE ALL CATEGORICAL TYPES
# ============================================================

def encode_all_categorical(df: pd.DataFrame,
                          ordinal_cols: Optional[Dict[str, str]] = None,
                          binary_cols: Optional[Dict[str, Dict]] = None,
                          onehot_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Master function to encode all categorical variable types in one go
    
    Args:
        df: Input dataframe
        ordinal_cols: Dict mapping ordinal column names to scale types
        binary_cols: Dict mapping binary column names to value mappings
        onehot_cols: List of column names for one-hot encoding
        
    Returns:
        Tuple of (encoded_dataframe, comprehensive_log)
    """
    
    df_encoded = df.copy()
    all_logs = {
        'ordinal': [],
        'binary': [],
        'onehot': [],
        'summary': {}
    }
    
    # 1. Ordinal encoding
    if ordinal_cols or True:  # Always try auto-detect for ordinal
        df_encoded, log = encode_ordinal_columns(
            df_encoded,
            column_mappings=ordinal_cols,
            auto_detect=True,
            replace_original=False
        )
        all_logs['ordinal'] = log
    
    # 2. Binary encoding
    if binary_cols or True:  # Always try auto-detect for binary
        df_encoded, log = encode_binary_columns(df_encoded, column_mappings=binary_cols)
        all_logs['binary'] = log
    
    # 3. One-hot encoding
    if onehot_cols or True:  # Always try auto-detect for one-hot
        df_encoded, log = encode_onehot_columns(df_encoded, columns=onehot_cols, drop_first=True)
        all_logs['onehot'] = log
    
    # Summary
    all_logs['summary'] = {
        'ordinal_encoded': len(all_logs['ordinal']),
        'binary_encoded': len(all_logs['binary']),
        'onehot_encoded': len(all_logs['onehot']),
        'original_shape': df.shape,
        'final_shape': df_encoded.shape
    }
    
    return df_encoded, all_logs


# ============================================================
# STREAMLIT UI FOR CATEGORICAL ENCODING
# ============================================================

def show_categorical_encoding_ui(df: pd.DataFrame):
    """
    Streamlit UI for all types of categorical encoding
    """
    import streamlit as st
    
    st.subheader("Categorical Encoding")
    
    df_clean = clean_text_columns(df)
    
    # Get text columns
    text_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    if not text_cols:
        st.info("No categorical columns found")
        return df
    
    # Categorize by unique values
    binary_candidates = []
    multicat_candidates = []
    
    for col in text_cols:
        n_unique = df_clean[col].nunique()
        if n_unique == 2:
            binary_candidates.append(col)
        elif n_unique >= 3:
            multicat_candidates.append(col)
    
    # Tabs for different encoding types
    tab1, tab2 = st.tabs(["Binary Encoding", "One-Hot Encoding"])
    
    # TAB 1: Binary Encoding
    with tab1:
        st.write("Binary columns detected (2 unique values):")
        
        if binary_candidates:
            for col in binary_candidates:
                values = sorted(df_clean[col].dropna().unique())
                st.write(f"**{col}:** {values[0]} → 0, {values[1]} → 1")
            
            if st.button("Apply Binary Encoding"):
                df_encoded, log = encode_binary_columns(df_clean)
                
                st.success("Binary encoding applied!")
                for item in log:
                    st.write(f"✓ {item['column']} → {item['new_column']}")
                    st.write(f"  Mapping: {item['mapping']}")
                
                return df_encoded
        else:
            st.info("No binary columns detected")
    
    # TAB 2: One-Hot Encoding
    with tab2:
        st.write("Multi-category columns detected (3+ unique values):")
        
        if multicat_candidates:
            selected_cols = st.multiselect(
                "Select columns to one-hot encode",
                options=multicat_candidates,
                default=multicat_candidates
            )
            
            drop_first = st.checkbox("Drop first category (recommended)", value=True)
            
            if st.button("Apply One-Hot Encoding"):
                df_encoded, log = encode_onehot_columns(
                    df_clean,
                    columns=selected_cols,
                    drop_first=drop_first
                )
                
                st.success("One-hot encoding applied!")
                for item in log:
                    st.write(f"✓ {item['column']} → {item['n_new_columns']} columns")
                    st.write(f"  New columns: {', '.join(item['new_columns'])}")
                
                return df_encoded
        else:
            st.info("No multi-category columns detected")
    
    return df


# ============================================================
# HELPER: Auto-categorize all columns
# ============================================================

def categorize_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Auto-categorize all text columns by encoding type needed
    
    Returns:
        Dict with keys: ordinal, binary, onehot, unknown
    """
    df_clean = clean_text_columns(df)
    
    categorized = {
        'ordinal': [],
        'binary': [],
        'onehot': [],
        'unknown': []
    }
    
    # Detect ordinal
    ordinal_detected = detect_ordinal_columns(df_clean)
    categorized['ordinal'] = list(ordinal_detected.keys())
    
    # Detect binary and one-hot
    for col in df_clean.select_dtypes(include=['object']).columns:
        if col in categorized['ordinal']:
            continue
        
        n_unique = df_clean[col].nunique()
        if n_unique == 2:
            categorized['binary'].append(col)
        elif n_unique >= 3:
            categorized['onehot'].append(col)
        else:
            categorized['unknown'].append(col)
    
    return categorized