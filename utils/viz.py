import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import io
import base64

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_correlation_heatmap(corr_matrix: pd.DataFrame, pval_matrix: pd.DataFrame = None,
                              mask_upper: bool = True, figsize: Tuple[int, int] = (10, 8),
                              title: str = "Correlation Matrix") -> plt.Figure:
    """
    Create correlation heatmap with significance annotations
    
    Args:
        corr_matrix: Correlation coefficient matrix
        pval_matrix: P-value matrix for significance stars
        mask_upper: Whether to mask upper triangle
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask if requested
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    else:
        mask = None
    
    # Create annotations with significance stars
    if pval_matrix is not None:
        annot_matrix = corr_matrix.round(3).astype(str)
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix.columns)):
                p_val = pval_matrix.iloc[i, j]
                if p_val < 0.001:
                    annot_matrix.iloc[i, j] += '***'
                elif p_val < 0.01:
                    annot_matrix.iloc[i, j] += '**'
                elif p_val < 0.05:
                    annot_matrix.iloc[i, j] += '*'
        annot = annot_matrix
    else:
        annot = True
    
    # Create heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='s' if pval_matrix is not None else '.3f',
                center=0, cmap='RdBu_r', square=True, linewidths=0.5, 
                cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig

def create_residual_plots(fitted_values: np.ndarray, residuals: np.ndarray,
                         studentized_resid: np.ndarray = None) -> plt.Figure:
    """
    Create residual diagnostic plots
    
    Args:
        fitted_values: Model fitted values
        residuals: Raw residuals
        studentized_resid: Studentized residuals
        
    Returns:
        matplotlib Figure with subplots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Residual Diagnostic Plots', fontsize=16, fontweight='bold')
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.6, color='steelblue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # Add smooth line
    try:
        z = np.polyfit(fitted_values, residuals, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(fitted_values.min(), fitted_values.max(), 100)
        axes[0, 0].plot(x_smooth, p(x_smooth), color='red', alpha=0.8)
    except:
        pass
    
    # 2. Normal Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    axes[0, 1].get_lines()[0].set_markerfacecolor('steelblue')
    axes[0, 1].get_lines()[0].set_markersize(4)
    axes[0, 1].get_lines()[1].set_color('red')
    
    # 3. Scale-Location Plot
    sqrt_abs_resid = np.sqrt(np.abs(residuals))
    axes[1, 0].scatter(fitted_values, sqrt_abs_resid, alpha=0.6, color='steelblue')
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Residuals|')
    axes[1, 0].set_title('Scale-Location Plot')
    
    # Add smooth line
    try:
        z = np.polyfit(fitted_values, sqrt_abs_resid, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(fitted_values.min(), fitted_values.max(), 100)
        axes[1, 0].plot(x_smooth, p(x_smooth), color='red', alpha=0.8)
    except:
        pass
    
    # 4. Histogram of Residuals
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Residuals')
    
    # Add normal distribution overlay
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
    y_norm = y_norm * len(residuals) * (residuals.max() - residuals.min()) / 30  # Scale to match histogram
    axes[1, 1].plot(x_norm, y_norm, color='red', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    return fig

def create_leverage_plot(leverage: np.ndarray, studentized_resid: np.ndarray,
                        cooks_distance: np.ndarray, n_obs: int, n_predictors: int) -> plt.Figure:
    """
    Create leverage vs residuals plot with Cook's distance contours
    
    Args:
        leverage: Hat matrix diagonal (leverage values)
        studentized_resid: Studentized residuals
        cooks_distance: Cook's distance values
        n_obs: Number of observations
        n_predictors: Number of predictors
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(leverage, studentized_resid, c=cooks_distance, 
                        cmap='Reds', alpha=0.7, s=50)
    
    # Add Cook's distance contours
    h_range = np.linspace(0, max(leverage) * 1.1, 100)
    for d in [0.5, 1.0]:  # Cook's distance contour lines
        y_pos = np.sqrt(d * (n_predictors + 1) * (1 - h_range) / h_range)
        y_neg = -y_pos
        ax.plot(h_range, y_pos, '--', color='red', alpha=0.6, 
               label=f'Cook\'s D = {d}' if d == 0.5 else '')
        ax.plot(h_range, y_neg, '--', color='red', alpha=0.6)
    
    # Add threshold lines
    leverage_threshold = 2 * (n_predictors + 1) / n_obs
    ax.axvline(x=leverage_threshold, color='blue', linestyle=':', alpha=0.7,
               label=f'Leverage threshold = {leverage_threshold:.3f}')
    
    ax.axhline(y=2, color='green', linestyle=':', alpha=0.7, label='|Residual| = 2')
    ax.axhline(y=-2, color='green', linestyle=':', alpha=0.7)
    
    # Labels and title
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Studentized Residuals')
    ax.set_title('Leverage vs Studentized Residuals\n(Cook\'s Distance in Color)')
    ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cook\'s Distance')
    
    plt.tight_layout()
    return fig

def create_distribution_plots(df: pd.DataFrame, columns: List[str], 
                             ncols: int = 3) -> plt.Figure:
    """
    Create distribution plots for multiple variables
    
    Args:
        df: Input dataframe
        columns: List of column names to plot
        ncols: Number of columns in subplot grid
        
    Returns:
        matplotlib Figure
    """
    n_vars = len(columns)
    nrows = (n_vars + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    fig.suptitle('Variable Distributions', fontsize=16, fontweight='bold')
    
    if nrows == 1:
        axes = [axes] if n_vars == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if col not in df.columns:
            continue
            
        ax = axes[i]
        data = df[col].dropna()
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Histogram with KDE
            ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black', density=True)
            
            # Add KDE if enough data points
            if len(data) > 10:
                try:
                    x_kde = np.linspace(data.min(), data.max(), 100)
                    kde = stats.gaussian_kde(data)
                    ax.plot(x_kde, kde(x_kde), color='red', linewidth=2)
                except:
                    pass
            
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            
        else:
            # Bar plot for categorical
            value_counts = data.value_counts().head(10)  # Top 10 categories
            ax.bar(range(len(value_counts)), value_counts.values, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
        
        ax.set_title(f'{col}\n(n={len(data)})')
    
    # Hide empty subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_scatter_matrix(df: pd.DataFrame, variables: List[str], 
                         target: str = None) -> plt.Figure:
    """
    Create scatter plot matrix
    
    Args:
        df: Input dataframe
        variables: List of variables to include
        target: Target variable for color coding
        
    Returns:
        matplotlib Figure
    """
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(2*n_vars, 2*n_vars))
    fig.suptitle('Scatter Plot Matrix', fontsize=16, fontweight='bold')
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                data = df[var1].dropna()
                ax.hist(data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                ax.set_title(var1)
            else:
                # Off-diagonal: scatter plot
                data = df[[var1, var2]].dropna()
                if len(data) > 0:
                    if target and target in df.columns:
                        # Color by target variable
                        target_data = df.loc[data.index, target]
                        scatter = ax.scatter(data[var2], data[var1], c=target_data, 
                                           alpha=0.6, s=20, cmap='viridis')
                    else:
                        ax.scatter(data[var2], data[var1], alpha=0.6, s=20, color='steelblue')
                    
                    # Add correlation coefficient
                    corr_coef = data[var1].corr(data[var2])
                    ax.text(0.05, 0.95, f'r={corr_coef:.3f}', transform=ax.transAxes, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Labels only on edges
            if i == n_vars - 1:
                ax.set_xlabel(var2)
            if j == 0:
                ax.set_ylabel(var1)
            
            # Remove tick labels from interior plots
            if i != n_vars - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
    
    plt.tight_layout()
    return fig

def create_model_comparison_plot(comparison_df: pd.DataFrame) -> plt.Figure:
    """
    Create model comparison visualization
    
    Args:
        comparison_df: DataFrame with model comparison metrics
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    models = comparison_df['Model']
    
    # R-squared comparison
    axes[0, 0].bar(models, comparison_df['R_Squared'], color='steelblue', alpha=0.7)
    axes[0, 0].set_title('R-squared')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Adjusted R-squared comparison
    axes[0, 1].bar(models, comparison_df['Adj_R_Squared'], color='darkgreen', alpha=0.7)
    axes[0, 1].set_title('Adjusted R-squared')
    axes[0, 1].set_ylabel('Adj R²')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # AIC comparison (lower is better)
    axes[1, 0].bar(models, comparison_df['AIC'], color='coral', alpha=0.7)
    axes[1, 0].set_title('AIC (lower is better)')
    axes[1, 0].set_ylabel('AIC')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # BIC comparison (lower is better)
    axes[1, 1].bar(models, comparison_df['BIC'], color='gold', alpha=0.7)
    axes[1, 1].set_title('BIC (lower is better)')
    axes[1, 1].set_ylabel('BIC')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_coefficient_plot(coef_table: pd.DataFrame, exclude_intercept: bool = True) -> plt.Figure:
    """
    Create coefficient plot with confidence intervals
    
    Args:
        coef_table: DataFrame with coefficient information
        exclude_intercept: Whether to exclude intercept from plot
        
    Returns:
        matplotlib Figure
    """
    plot_data = coef_table.copy()
    
    if exclude_intercept:
        plot_data = plot_data[plot_data['Variable'] != 'const']
    
    if len(plot_data) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No variables to plot', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.5)))
    
    y_pos = range(len(plot_data))
    
    # Plot coefficients with error bars
    ax.errorbar(plot_data['Coefficient'], y_pos, 
               xerr=[plot_data['Coefficient'] - plot_data['CI_Lower'],
                     plot_data['CI_Upper'] - plot_data['Coefficient']],
               fmt='o', color='steelblue', alpha=0.7)
    
    # Add variable names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data['Variable'])
    
    ax.set_xlabel('Coefficient')
    ax.set_title('Coefficient Plot')
    
    plt.tight_layout()
    return fig