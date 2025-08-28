import pandas as pd
import numpy as np
import io
from typing import Optional, List, Union

def read_file(file, sheet_name: Optional[str] = None, header_row: int = 0, 
              skip_rows: int = 0, na_markers: List[str] = None) -> pd.DataFrame:
    """
    Read file with various options for headers and missing value markers
    
    Args:
        file: Uploaded file object or file path
        sheet_name: Excel sheet name (if applicable)
        header_row: Row index to use as column headers
        skip_rows: Number of rows to skip from top
        na_markers: List of strings to treat as NaN
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if na_markers is None:
        na_markers = ['', 'NA', 'N/A', 'na', 'n/a', 'NULL', 'null', 'None', 
                      'none', '-', '--', '99', '999', '9999', 'Missing', 'missing']
    
    try:
        if hasattr(file, 'name') and file.name.endswith('.csv'):
            # CSV file
            df = pd.read_csv(
                file,
                header=header_row,
                skiprows=skip_rows,
                na_values=na_markers,
                encoding='utf-8'
            )
        else:
            # Excel file
            df = pd.read_excel(
                file,
                sheet_name=sheet_name,
                header=header_row,
                skiprows=skip_rows,
                na_values=na_markers
            )
        
        # Clean column names - remove leading/trailing whitespace
        df.columns = df.columns.astype(str).str.strip()
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0)  # Remove empty rows
        df = df.dropna(how='all', axis=1)  # Remove empty columns
        
        return df
        
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def export_to_excel(dataframe: pd.DataFrame, filename: str = "export.xlsx", 
                   sheet_name: str = "Data") -> bytes:
    """
    Export dataframe to Excel format
    
    Args:
        dataframe: DataFrame to export
        filename: Output filename
        sheet_name: Excel sheet name
    
    Returns:
        bytes: Excel file as bytes
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return output.getvalue()

def export_multiple_sheets(data_dict: dict, filename: str = "export.xlsx") -> bytes:
    """
    Export multiple dataframes to different Excel sheets
    
    Args:
        data_dict: Dictionary with sheet_name: dataframe pairs
        filename: Output filename
    
    Returns:
        bytes: Excel file as bytes
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return output.getvalue()

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate dataframe and return summary information
    
    Args:
        df: DataFrame to validate
        
    Returns:
        dict: Validation summary
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'info': {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'missing_cells': df.isnull().sum().sum(),
            'missing_pct': (df.isnull().sum().sum() / df.size) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_cols': len(df.select_dtypes(include=['datetime64']).columns)
        }
    }
    
    # Check for critical issues
    if len(df) == 0:
        validation['is_valid'] = False
        validation['errors'].append("DataFrame is empty (no rows)")
    
    if len(df.columns) == 0:
        validation['is_valid'] = False
        validation['errors'].append("DataFrame has no columns")
    
    # Check for warnings
    if validation['info']['missing_pct'] > 50:
        validation['warnings'].append("More than 50% of data is missing")
    
    if validation['info']['duplicate_rows'] > 0:
        validation['warnings'].append(f"Found {validation['info']['duplicate_rows']} duplicate rows")
    
    if validation['info']['memory_mb'] > 500:
        validation['warnings'].append("Large dataset (>500MB) - processing may be slow")
    
    # Check for duplicate column names
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        validation['is_valid'] = False
        validation['errors'].append(f"Duplicate column names found: {duplicate_cols}")
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        validation['warnings'].append(f"Completely empty columns: {empty_cols}")
    
    return validation

def infer_column_types(df: pd.DataFrame) -> dict:
    """
    Infer appropriate types for columns
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        dict: Column type recommendations
    """
    recommendations = {}
    
    for col in df.columns:
        series = df[col].dropna()
        
        if len(series) == 0:
            recommendations[col] = 'exclude'
            continue
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            unique_vals = series.nunique()
            
            # Check if it might be categorical (few unique integer values)
            if unique_vals <= 10 and series.dtype in ['int64', 'int32']:
                # Check if it looks like Likert scale (consecutive integers)
                vals = sorted(series.unique())
                if len(vals) >= 3 and vals == list(range(min(vals), max(vals) + 1)):
                    recommendations[col] = 'likert'
                else:
                    recommendations[col] = 'categorical_numeric'
            else:
                recommendations[col] = 'continuous'
        
        # String/object columns
        elif pd.api.types.is_object_dtype(series):
            unique_vals = series.nunique()
            total_vals = len(series)
            
            # Check if it might be an ID column
            if unique_vals == total_vals:
                recommendations[col] = 'id'
            # Check if categorical
            elif unique_vals <= min(20, total_vals * 0.5):
                recommendations[col] = 'categorical'
            else:
                recommendations[col] = 'text'
        
        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(series):
            recommendations[col] = 'datetime'
        
        # Boolean columns
        elif pd.api.types.is_bool_dtype(series):
            recommendations[col] = 'boolean'
        
        else:
            recommendations[col] = 'other'
    
    return recommendations

def detect_likert_scales(df: pd.DataFrame) -> dict:
    """
    Detect potential Likert scale variables
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        dict: Detected Likert scales with their properties
    """
    likert_info = {}
    
    for col in df.columns:
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
            
        # Check if numeric and has limited range
        if pd.api.types.is_numeric_dtype(series):
            unique_vals = sorted(series.unique())
            
            # Potential Likert if:
            # - Between 3-10 unique values
            # - Values are consecutive integers
            # - Range starts from 1 or 0
            if (3 <= len(unique_vals) <= 10 and 
                all(isinstance(x, (int, np.integer)) for x in unique_vals) and
                unique_vals == list(range(min(unique_vals), max(unique_vals) + 1)) and
                min(unique_vals) in [0, 1]):
                
                likert_info[col] = {
                    'scale_range': f"{min(unique_vals)}-{max(unique_vals)}",
                    'n_points': len(unique_vals),
                    'values': unique_vals,
                    'might_need_reverse': False  # We'll let user decide
                }
    
    return likert_info

def read_file_bytes(path: str) -> bytes:
    """
    Read a file in binary mode and return its contents as bytes.
    
    Args:
        path: File path to read.
    
    Returns:
        bytes: File content.
    """
    with open(path, "rb") as f:
        return f.read()
