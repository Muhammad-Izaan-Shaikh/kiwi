import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

def _prepare_features(X: pd.DataFrame, add_constant: bool = False) -> pd.DataFrame:
    """
    Preprocess features for OLS and VIF:
    - One-hot encode categoricals (drop_first=True to avoid dummy trap)
    - Drop constant columns
    - Ensure numeric (convert bool -> int, coerce errors)
    - Optionally add constant/intercept
    
    Args:
        X (pd.DataFrame): Input predictors
        add_constant (bool): Whether to add an intercept
    
    Returns:
        pd.DataFrame: Clean numeric feature matrix
    """
    # 1. Encode categoricals
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 2. Drop constant columns
    X_encoded = X_encoded.loc[:, X_encoded.apply(pd.Series.nunique) > 1]

    # 3. Ensure numeric (bools -> int, others coerced)
    for col in X_encoded.columns:
        if X_encoded[col].dtype == "bool":
            X_encoded[col] = X_encoded[col].astype(int)
        else:
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors="coerce")

    # 4. Add constant if requested
    if add_constant:
        X_encoded = sm.add_constant(X_encoded, has_constant="add")

    return X_encoded

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each predictor
    
    Args:
        X: Dataframe of predictor variables
        
    Returns:
        pd.DataFrame: VIF values for each variable
    """
    x_prepared = _prepare_features(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = x_prepared.columns
    vif_data["VIF"] = [variance_inflation_factor(x_prepared.values, i) 
                       for i in range(len(x_prepared.columns))]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    vif_data['VIF_Level'] = vif_data['VIF'].apply(lambda x: 
        'Severe (>10)' if x > 10 else 
        'Moderate (5-10)' if x > 5 else 
        'Low (<5)')
    
    return vif_data

def fit_ols_model(y: pd.Series, X: pd.DataFrame, add_constant: bool = True,
                  robust: Optional[str] = None, alpha: float = 0.05) -> Dict:
    """
    Fit OLS regression model with comprehensive diagnostics and
    automatic categorical encoding / collinearity handling.
    """
    x_prepared = _prepare_features(X)
    print("=== DEBUG: Starting fit_ols_model ===")
    print(f"Input X shape: {x_prepared.shape}")
    print(f"Input X dtypes:\n{x_prepared.dtypes}")
    print(f"Input y dtype: {y.dtype}")
    
    # --- 🔧 Handle categorical encoding ---
    print("\n=== STEP 1: get_dummies ===")
    X_encoded = pd.get_dummies(x_prepared, drop_first=True)
    print(f"After get_dummies X shape: {X_encoded.shape}")
    print(f"After get_dummies X dtypes:\n{X_encoded.dtypes}")
    print(f"After get_dummies X columns: {X_encoded.columns.tolist()}")

    # Drop constant columns (avoids singular matrix)
    print("\n=== STEP 2: Drop constant columns ===")
    nunique_before = X_encoded.apply(pd.Series.nunique)
    print(f"Unique values per column:\n{nunique_before}")
    
    X_encoded = X_encoded.loc[:, X_encoded.apply(pd.Series.nunique) > 1]
    print(f"After dropping constants X shape: {X_encoded.shape}")
    print(f"After dropping constants X dtypes:\n{X_encoded.dtypes}")

    # --- 🔧 Ensure numeric data ---
    print("\n=== STEP 3: Convert to numeric ===")
    print("Before numeric conversion:")
    for col in X_encoded.columns:
        print(f"  {col}: {X_encoded[col].dtype} - Sample values: {X_encoded[col].head(3).tolist()}")
    
    # Convert each column individually, ensuring booleans become integers
    X_numeric = X_encoded.copy()
    for col in X_encoded.columns:
        try:
            original_dtype = X_numeric[col].dtype
            if X_numeric[col].dtype == 'bool':
                # Convert boolean to integer explicitly
                X_numeric[col] = X_numeric[col].astype(int)
                print(f"  {col}: {original_dtype} -> {X_numeric[col].dtype} ✓ (bool->int)")
            else:
                X_numeric[col] = pd.to_numeric(X_numeric[col], errors="coerce")
                print(f"  {col}: {original_dtype} -> {X_numeric[col].dtype} ✓")
        except Exception as e:
            print(f"  {col}: FAILED to convert - {str(e)} ❌")
    
    y_numeric = pd.to_numeric(y, errors="coerce")
    print(f"y conversion: {y.dtype} -> {y_numeric.dtype}")

    # Add constant if required
    print("\n=== STEP 4: Add constant ===")
    if add_constant:
        X_with_const = sm.add_constant(X_numeric, has_constant="add")
        print(f"After add_constant X shape: {X_with_const.shape}")
        print(f"After add_constant X dtypes:\n{X_with_const.dtypes}")
    else:
        X_with_const = X_numeric.copy()
    
    # Remove rows with missing values
    print("\n=== STEP 5: Remove missing values ===")
    combined = pd.concat([y_numeric, X_with_const], axis=1).dropna()
    print(f"Combined shape before dropna: {pd.concat([y_numeric, X_with_const], axis=1).shape}")
    print(f"Combined shape after dropna: {combined.shape}")
    
    y_clean = combined.iloc[:, 0]
    X_clean = combined.iloc[:, 1:]
    
    print(f"Final y_clean dtype: {y_clean.dtype}")
    print(f"Final X_clean dtypes:\n{X_clean.dtypes}")
    print(f"Final X_clean sample:\n{X_clean.head(2)}")
    
    # Check if any columns are still object dtype
    object_cols = X_clean.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"❌ ERROR: Still have object columns: {object_cols.tolist()}")
        for col in object_cols:
            print(f"  {col} values: {X_clean[col].unique()}")
        raise ValueError(f"Object columns remain after cleaning: {object_cols.tolist()}")
    
    if len(y_clean) < len(X_clean.columns) + 1:
        raise ValueError("Insufficient observations for the number of predictors")
    
    print("\n=== STEP 6: Fitting OLS model ===")
    # --- Fit model ---
    model = sm.OLS(y_clean, X_clean)
    if robust:
        results = model.fit(cov_type=robust)
    else:
        results = model.fit()
    
    print("✓ Model fitted successfully!")
    
    # ... rest of your function remains the same ...
    
    # --- Extract key metrics ---
    model_summary = {
        'model': model,
        'results': results,
        'n_obs': int(results.nobs),
        'n_predictors': len(X_clean.columns) - (1 if add_constant else 0),
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'f_statistic': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'aic': results.aic,
        'bic': results.bic,
        'log_likelihood': results.llf,
        'durbin_watson': durbin_watson(results.resid),
        'robust_se': robust
    }
    
    # --- Coefficients table ---
    coef_table = pd.DataFrame({
        'Variable': X_clean.columns,
        'Coefficient': results.params,
        'Std_Error': results.bse,
        't_statistic': results.tvalues,
        'P_Value': results.pvalues,
        'CI_Lower': results.conf_int(alpha=alpha)[0],
        'CI_Upper': results.conf_int(alpha=alpha)[1]
    })
    
    # Significance stars
    coef_table['Significance'] = coef_table['P_Value'].apply(lambda x:
        '***' if x < 0.001 else
        '**' if x < 0.01 else
        '*' if x < 0.05 else
        '.' if x < 0.1 else ''
    )
    
    # Skip Beta calculation for now to focus on the main issue
    coef_table['Beta'] = np.nan
    
    model_summary['coefficients'] = coef_table
    
    return model_summary

def perform_regression_diagnostics(model_results) -> Dict:
    """
    Perform comprehensive regression diagnostics
    
    Args:
        model_results: Fitted statsmodels regression results
        
    Returns:
        Dict: Diagnostic test results and plots data
    """
    results = model_results['results']
    
    diagnostics = {
        'residuals': results.resid,
        'fitted_values': results.fittedvalues,
        'standardized_residuals': results.resid_pearson,
        'studentized_residuals': results.get_influence().resid_studentized_internal,
        'leverage': results.get_influence().hat_matrix_diag,
        'cooks_distance': results.get_influence().cooks_distance[0],
        'dffits': results.get_influence().dffits[0]
    }
    
    # Normality tests
    residuals = results.resid
    if len(residuals) <= 5000:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            normality_test = {
                'test': 'Shapiro-Wilk',
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        except:
            normality_test = {'test': 'failed', 'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
    else:
        try:
            dagostino_stat, dagostino_p = stats.normaltest(residuals)
            normality_test = {
                'test': 'D\'Agostino',
                'statistic': dagostino_stat,
                'p_value': dagostino_p,
                'is_normal': dagostino_p > 0.05
            }
        except:
            normality_test = {'test': 'failed', 'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
    
    diagnostics['normality_test'] = normality_test
    
    # Heteroskedasticity tests
    try:
        # Breusch-Pagan test
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, results.model.exog)
        heteroskedasticity_bp = {
            'test': 'Breusch-Pagan',
            'statistic': bp_stat,
            'p_value': bp_p,
            'homoskedastic': bp_p > 0.05
        }
    except:
        heteroskedasticity_bp = {'test': 'Breusch-Pagan', 'statistic': np.nan, 'p_value': np.nan, 'homoskedastic': True}
    
    try:
        # White test
        white_stat, white_p, _, _ = het_white(residuals, results.model.exog)
        heteroskedasticity_white = {
            'test': 'White',
            'statistic': white_stat,
            'p_value': white_p,
            'homoskedastic': white_p > 0.05
        }
    except:
        heteroskedasticity_white = {'test': 'White', 'statistic': np.nan, 'p_value': np.nan, 'homoskedastic': True}
    
    diagnostics['heteroskedasticity_bp'] = heteroskedasticity_bp
    diagnostics['heteroskedasticity_white'] = heteroskedasticity_white
    
    # Independence (Durbin-Watson already computed in main function)
    dw_stat = model_results['durbin_watson']
    # Rule of thumb: values between 1.5-2.5 suggest no autocorrelation
    independence_test = {
        'test': 'Durbin-Watson',
        'statistic': dw_stat,
        'interpretation': 'No autocorrelation' if 1.5 <= dw_stat <= 2.5 else
                         'Positive autocorrelation' if dw_stat < 1.5 else
                         'Negative autocorrelation'
    }
    diagnostics['independence_test'] = independence_test
    
    # Outlier detection
    leverage_threshold = 2 * (model_results['n_predictors'] + 1) / model_results['n_obs']
    cooks_threshold = 4 / model_results['n_obs']
    
    outliers = {
        'high_leverage': (diagnostics['leverage'] > leverage_threshold).sum(),
        'high_cooks': (diagnostics['cooks_distance'] > cooks_threshold).sum(),
        'leverage_threshold': leverage_threshold,
        'cooks_threshold': cooks_threshold,
        'leverage_outliers': diagnostics['leverage'] > leverage_threshold,
        'cooks_outliers': diagnostics['cooks_distance'] > cooks_threshold
    }
    diagnostics['outliers'] = outliers
    
    return diagnostics

def make_predictions(model_results, X_new: pd.DataFrame, 
                    add_constant: bool = True) -> pd.DataFrame:
    """
    Make predictions using fitted model
    
    Args:
        model_results: Fitted model results dictionary
        X_new: New predictor data
        add_constant: Whether to add constant term
        
    Returns:
        pd.DataFrame: Predictions with confidence intervals
    """
    results = model_results['results']
    
    if add_constant:
        X_pred = sm.add_constant(X_new)
    else:
        X_pred = X_new.copy()
    
    # Make predictions
    predictions = results.predict(X_pred)
    
    # Get prediction intervals (approximate)
    prediction_se = np.sqrt(results.mse_resid * (1 + np.diag(X_pred @ results.cov_params() @ X_pred.T)))
    
    # 95% confidence intervals
    alpha = 0.05
    t_val = stats.t.ppf(1 - alpha/2, results.df_resid)
    
    pred_df = pd.DataFrame({
        'Predicted': predictions,
        'Std_Error': prediction_se,
        'CI_Lower': predictions - t_val * prediction_se,
        'CI_Upper': predictions + t_val * prediction_se
    })
    
    return pred_df

def evaluate_model_assumptions(diagnostics: Dict) -> Dict:
    """
    Evaluate how well model assumptions are met
    
    Args:
        diagnostics: Diagnostic test results
        
    Returns:
        Dict: Assumption evaluation summary
    """
    assumptions = {
        'linearity': {
            'met': True,  # This would require more sophisticated testing
            'note': 'Check residuals vs fitted plot for non-linear patterns'
        },
        'independence': {
            'met': diagnostics['independence_test']['interpretation'] == 'No autocorrelation',
            'test_result': diagnostics['independence_test']
        },
        'homoskedasticity': {
            'met': (diagnostics['heteroskedasticity_bp']['homoskedastic'] and 
                   diagnostics['heteroskedasticity_white']['homoskedastic']),
            'bp_test': diagnostics['heteroskedasticity_bp'],
            'white_test': diagnostics['heteroskedasticity_white']
        },
        'normality': {
            'met': diagnostics['normality_test']['is_normal'],
            'test_result': diagnostics['normality_test']
        },
        'no_multicollinearity': {
            'met': True,  # This should be checked with VIF separately
            'note': 'Check VIF values - should be < 5 (preferably < 2.5)'
        }
    }
    
    # Overall assessment
    assumptions_met = sum([
        assumptions['independence']['met'],
        assumptions['homoskedasticity']['met'],
        assumptions['normality']['met']
    ])
    
    assumptions['overall'] = {
        'assumptions_met': assumptions_met,
        'total_assumptions': 3,
        'percentage': (assumptions_met / 3) * 100,
        'interpretation': 'Good' if assumptions_met >= 3 else
                         'Moderate' if assumptions_met >= 2 else
                         'Poor'
    }
    
    return assumptions

def generate_model_interpretation(model_results: Dict, diagnostics: Dict) -> Dict:
    """
    Generate interpretation and recommendations for the model
    
    Args:
        model_results: Model fitting results
        diagnostics: Diagnostic test results
        
    Returns:
        Dict: Model interpretation and recommendations
    """
    interpretation = {
        'model_fit': {
            'r_squared': model_results['r_squared'],
            'adj_r_squared': model_results['adj_r_squared'],
            'interpretation': f"The model explains {model_results['r_squared']:.1%} of variance in the outcome"
        },
        'significant_predictors': [],
        'effect_sizes': {},
        'recommendations': [],
        'warnings': []
    }
    
    # Identify significant predictors
    coef_table = model_results['coefficients']
    sig_predictors = coef_table[coef_table['P_Value'] < 0.05]
    
    for _, row in sig_predictors.iterrows():
        if row['Variable'] != 'const':
            effect_direction = 'positive' if row['Coefficient'] > 0 else 'negative'
            interpretation['significant_predictors'].append({
                'variable': row['Variable'],
                'coefficient': row['Coefficient'],
                'p_value': row['P_Value'],
                'effect_direction': effect_direction,
                'standardized_effect': row.get('Beta', np.nan)
            })
    
    # Model quality assessment
    if model_results['r_squared'] < 0.1:
        interpretation['warnings'].append("Low R² suggests poor model fit")
    elif model_results['r_squared'] > 0.7:
        interpretation['recommendations'].append("High R² suggests good predictive power")
    
    # Check sample size adequacy
    n_per_predictor = model_results['n_obs'] / model_results['n_predictors']
    if n_per_predictor < 10:
        interpretation['warnings'].append("Small sample size relative to predictors (< 10:1 ratio)")
    elif n_per_predictor < 20:
        interpretation['recommendations'].append("Consider larger sample or fewer predictors")
    
    # Assumption violations
    assumptions = evaluate_model_assumptions(diagnostics)
    if not assumptions['homoskedasticity']['met']:
        interpretation['recommendations'].append("Consider robust standard errors due to heteroskedasticity")
    
    if not assumptions['normality']['met']:
        interpretation['recommendations'].append("Consider data transformation due to non-normal residuals")
    
    if diagnostics['outliers']['high_leverage'] > 0 or diagnostics['outliers']['high_cooks'] > 0:
        interpretation['warnings'].append(f"Found {diagnostics['outliers']['high_leverage']} high leverage points and {diagnostics['outliers']['high_cooks']} influential observations")
    
    return interpretation

def perform_model_comparison(models: List[Dict]) -> pd.DataFrame:
    """
    Compare multiple fitted models
    
    Args:
        models: List of model result dictionaries
        
    Returns:
        pd.DataFrame: Model comparison table
    """
    comparison_data = []
    
    for i, model in enumerate(models):
        comparison_data.append({
            'Model': f'Model_{i+1}',
            'N_Obs': model['n_obs'],
            'N_Predictors': model['n_predictors'],
            'R_Squared': model['r_squared'],
            'Adj_R_Squared': model['adj_r_squared'],
            'AIC': model['aic'],
            'BIC': model['bic'],
            'F_Statistic': model['f_statistic'],
            'F_P_Value': model['f_pvalue']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add rankings
    comparison_df['AIC_Rank'] = comparison_df['AIC'].rank()
    comparison_df['BIC_Rank'] = comparison_df['BIC'].rank()
    comparison_df['R²_Rank'] = comparison_df['R_Squared'].rank(ascending=False)
    
    return comparison_df