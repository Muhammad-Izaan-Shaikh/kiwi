import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def compute_correlation_with_pvalues(df: pd.DataFrame, method: str = 'pearson', 
                                   columns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrix with corresponding p-values
    
    Args:
        df: Input dataframe
        method: 'pearson', 'spearman', or 'kendall'
        columns: Specific columns to include (None = all numeric columns)
        
    Returns:
        Tuple of (correlation_matrix, pvalue_matrix)
    """
    if columns is None:
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df.columns and 
                       pd.api.types.is_numeric_dtype(df[col])]
    
    df_numeric = df[numeric_cols].dropna()
    n_vars = len(numeric_cols)
    
    # Initialize matrices
    corr_matrix = pd.DataFrame(np.eye(n_vars), columns=numeric_cols, index=numeric_cols)
    pval_matrix = pd.DataFrame(np.zeros((n_vars, n_vars)), columns=numeric_cols, index=numeric_cols)
    
    # Compute correlations and p-values
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i <= j:  # Only compute upper triangle + diagonal
                if i == j:
                    corr_matrix.iloc[i, j] = 1.0
                    pval_matrix.iloc[i, j] = 0.0
                else:
                    # Remove rows where either variable is NaN
                    mask = df_numeric[[col1, col2]].notna().all(axis=1)
                    x = df_numeric.loc[mask, col1]
                    y = df_numeric.loc[mask, col2]
                    
                    if len(x) < 3:  # Need at least 3 observations
                        corr_val, p_val = np.nan, np.nan
                    else:
                        try:
                            if method == 'pearson':
                                corr_val, p_val = pearsonr(x, y)
                            elif method == 'spearman':
                                corr_val, p_val = spearmanr(x, y)
                            elif method == 'kendall':
                                corr_val, p_val = kendalltau(x, y)
                            else:
                                raise ValueError(f"Unknown correlation method: {method}")
                        except:
                            corr_val, p_val = np.nan, np.nan
                    
                    corr_matrix.iloc[i, j] = corr_val
                    corr_matrix.iloc[j, i] = corr_val  # Fill lower triangle
                    pval_matrix.iloc[i, j] = p_val
                    pval_matrix.iloc[j, i] = p_val  # Fill lower triangle
    
    return corr_matrix, pval_matrix

def get_correlation_summary(corr_matrix: pd.DataFrame, pval_matrix: pd.DataFrame, 
                          alpha: float = 0.05, top_n: int = 10) -> Dict:
    """
    Generate summary statistics for correlation analysis
    
    Args:
        corr_matrix: Correlation coefficient matrix
        pval_matrix: P-value matrix
        alpha: Significance level
        top_n: Number of top correlations to return
        
    Returns:
        Dict: Summary information
    """
    # Get upper triangle indices (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    
    # Extract upper triangle values
    corr_values = corr_matrix.values[mask]
    pval_values = pval_matrix.values[mask]
    
    # Get variable pairs
    var_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            var_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'Variable_1': [pair[0] for pair in var_pairs],
        'Variable_2': [pair[1] for pair in var_pairs],
        'Correlation': corr_values,
        'P_Value': pval_values,
        'Abs_Correlation': np.abs(corr_values),
        'Significant': pval_values < alpha
    }).dropna()
    
    # Sort by absolute correlation
    summary_df = summary_df.sort_values('Abs_Correlation', ascending=False)
    
    summary = {
        'total_pairs': len(summary_df),
        'significant_pairs': (summary_df['P_Value'] < alpha).sum(),
        'mean_correlation': summary_df['Abs_Correlation'].mean(),
        'max_correlation': summary_df['Abs_Correlation'].max(),
        'top_correlations': summary_df.head(top_n),
        'strongest_positive': summary_df[summary_df['Correlation'] > 0].head(5) if len(summary_df[summary_df['Correlation'] > 0]) > 0 else pd.DataFrame(),
        'strongest_negative': summary_df[summary_df['Correlation'] < 0].head(5) if len(summary_df[summary_df['Correlation'] < 0]) > 0 else pd.DataFrame(),
        'distribution_stats': {
            'min': corr_values.min(),
            'max': corr_values.max(),
            'mean': corr_values.mean(),
            'std': corr_values.std(),
            'median': np.median(corr_values)
        }
    }
    
    return summary

def generate_descriptive_stats(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Generate descriptive statistics for specified columns
    
    Args:
        df: Input dataframe
        columns: Columns to analyze (None = all numeric columns)
        
    Returns:
        pd.DataFrame: Descriptive statistics
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        columns = [col for col in columns if col in df.columns]
    
    stats_dict = {}
    
    for col in columns:
        series = df[col].dropna()
        
        if pd.api.types.is_numeric_dtype(df[col]):
            stats_dict[col] = {
                'Count': len(series),
                'Missing': df[col].isnull().sum(),
                'Missing_%': (df[col].isnull().sum() / len(df)) * 100,
                'Mean': series.mean(),
                'Std': series.std(),
                'Min': series.min(),
                'Q1': series.quantile(0.25),
                'Median': series.median(),
                'Q3': series.quantile(0.75),
                'Max': series.max(),
                'Range': series.max() - series.min(),
                'IQR': series.quantile(0.75) - series.quantile(0.25),
                'Skewness': stats.skew(series),
                'Kurtosis': stats.kurtosis(series),
                'Unique': series.nunique()
            }
        else:
            # Categorical variable
            stats_dict[col] = {
                'Count': len(series),
                'Missing': df[col].isnull().sum(),
                'Missing_%': (df[col].isnull().sum() / len(df)) * 100,
                'Unique': series.nunique(),
                'Mode': series.mode().iloc[0] if len(series.mode()) > 0 else None,
                'Mode_Freq': (series == series.mode().iloc[0]).sum() if len(series.mode()) > 0 else 0,
                'Mode_%': ((series == series.mode().iloc[0]).sum() / len(series) * 100) if len(series.mode()) > 0 else 0
            }
    
    return pd.DataFrame(stats_dict).T

def detect_distribution_normality(df: pd.DataFrame, columns: List[str] = None, 
                                alpha: float = 0.05) -> Dict:
    """
    Test for normality in numeric columns
    
    Args:
        df: Input dataframe
        columns: Columns to test (None = all numeric columns)
        alpha: Significance level
        
    Returns:
        Dict: Normality test results
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    normality_results = {}
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        series = df[col].dropna()
        
        if len(series) < 3:
            normality_results[col] = {
                'test': 'insufficient_data',
                'statistic': np.nan,
                'p_value': np.nan,
                'is_normal': False,
                'n_obs': len(series)
            }
            continue
        
        # Choose appropriate test based on sample size
        if len(series) <= 5000:
            # Shapiro-Wilk test for smaller samples
            try:
                stat, p_val = stats.shapiro(series)
                test_name = 'shapiro_wilk'
            except:
                stat, p_val = np.nan, np.nan
                test_name = 'failed'
        else:
            # D'Agostino test for larger samples
            try:
                stat, p_val = stats.normaltest(series)
                test_name = 'dagostino'
            except:
                stat, p_val = np.nan, np.nan
                test_name = 'failed'
        
        normality_results[col] = {
            'test': test_name,
            'statistic': stat,
            'p_value': p_val,
            'is_normal': p_val > alpha if not np.isnan(p_val) else False,
            'n_obs': len(series)
        }
    
    return normality_results

def identify_potential_scales(df: pd.DataFrame, prefix_groups: List[str] = None) -> Dict:
    """
    Identify potential scale/subscale groupings based on column name prefixes
    
    Args:
        df: Input dataframe
        prefix_groups: List of prefixes to look for (e.g., ['anxiety_', 'wellbeing_'])
        
    Returns:
        Dict: Potential scale groupings
    """
    if prefix_groups is None:
        # Auto-detect common prefixes
        prefixes = {}
        for col in df.columns:
            if '_' in col:
                prefix = col.split('_')[0]
                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(col)
        
        # Keep only prefixes with multiple items
        prefix_groups = {k: v for k, v in prefixes.items() if len(v) >= 2}
    else:
        # Use provided prefixes
        prefix_groups = {}
        for prefix in prefix_groups:
            matching_cols = [col for col in df.columns if col.startswith(prefix)]
            if len(matching_cols) >= 2:
                prefix_groups[prefix] = matching_cols
    
    # Analyze each potential scale
    scale_analysis = {}
    for scale_name, columns in prefix_groups.items():
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) >= 2:
            # Compute inter-item correlations
            scale_df = df[numeric_cols].dropna()
            if len(scale_df) > 0:
                corr_matrix = scale_df.corr()
                
                # Get mean inter-item correlation (excluding diagonal)
                mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                inter_item_corrs = corr_matrix.values[mask]
                mean_inter_item = np.nanmean(inter_item_corrs)
                
                scale_analysis[scale_name] = {
                    'items': numeric_cols,
                    'n_items': len(numeric_cols),
                    'mean_inter_item_correlation': mean_inter_item,
                    'correlation_range': (np.nanmin(inter_item_corrs), np.nanmax(inter_item_corrs)),
                    'recommended_for_scale': mean_inter_item >= 0.3  # Common threshold
                }
    
    return scale_analysis

def create_correlation_significance_stars(pval_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Create significance star annotations for correlation matrix
    
    Args:
        pval_matrix: Matrix of p-values
        
    Returns:
        pd.DataFrame: Matrix with significance stars
    """
    stars_matrix = pval_matrix.copy()
    
    # Apply star ratings
    stars_matrix = stars_matrix.applymap(lambda x: 
        '***' if x < 0.001 else 
        '**' if x < 0.01 else 
        '*' if x < 0.05 else 
        '' if not np.isnan(x) else ''
    )
    
    return stars_matrix