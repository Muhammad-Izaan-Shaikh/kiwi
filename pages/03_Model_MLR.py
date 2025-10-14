# === pages/3_Model_MLR.py ===
"""
Modeling page: Multiple Linear Regression (MLR)
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils import modeling, fileio, viz
from utils.translations import get_text, init_language

# Initialize language system
init_language()

st.set_page_config(page_title="3 - Model MLR", layout="wide")
st.title(f"📈 {get_text('model_mlr_title')}")

# Load cleaned data
df = None
if "df" in st.session_state and st.session_state["df"] is not None:
    df = st.session_state["df"]
elif "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
    df = st.session_state["raw_df"]

if df is None:
    st.warning(get_text('no_data'))
    st.stop()

st.write(get_text('dataset_info_short', rows=df.shape[0], cols=df.shape[1]))

# ===============================================================
# Select Y and X
# ===============================================================
all_cols = df.columns.tolist()
target = st.selectbox(get_text('choose_dependent_variable'), [None] + all_cols)
if not target:
    st.info(get_text('please_pick_dependent_variable'))
    st.stop()

predictors = st.multiselect(get_text('choose_predictors'), [c for c in all_cols if c != target])
if not predictors:
    st.info(get_text('select_at_least_one_predictor'))
    st.stop()

# ===============================================================
# Sidebar options
# ===============================================================
st.sidebar.subheader(get_text('model_options'))
add_intercept = st.sidebar.checkbox(get_text('add_intercept'), value=True)
robust_se = st.sidebar.checkbox(get_text('use_robust_se'), value=False)

# ===============================================================
# Fit model
# ===============================================================
if st.button(get_text('fit_mlr')):
    X = df[predictors].copy()
    y = df[target].copy()
    
    # === DATA TYPE CLEANING ===
    # Clean target variable
    y_clean = pd.to_numeric(y, errors='coerce')
    
    # Clean predictor variables
    X_clean = X.copy()
    for col in X_clean.columns:
        original_dtype = X_clean[col].dtype
        if col not in ['gender', 'education']:  # Keep categorical as-is for get_dummies
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
    
    # Check for missing values after conversion
    missing_y = y_clean.isnull().sum()
    missing_X = X_clean.isnull().sum().sum()
    
    if missing_y > 0 or missing_X > 0:
        st.warning(get_text('missing_values_after_cleaning', y_missing=missing_y, x_missing=missing_X))
        st.write(get_text('missing_values_by_column'))
        st.write(X_clean.isnull().sum())
    else:
        st.success(get_text('no_missing_after_cleaning'))
        
        with st.spinner(get_text('fitting_model')):
            try:
                results = modeling.fit_ols_model(
                    y_clean,  # Use cleaned data
                    X_clean,  # Use cleaned data
                    add_constant=add_intercept, 
                    robust="HC3" if robust_se else None
                )
                st.session_state["mlr_results"] = results
                st.success(get_text('model_fitted_successfully'))
            except Exception as e:
                st.error(get_text('model_fitting_failed', error=str(e)))
                st.write(get_text('error_details'))
                st.write(f"y_clean shape: {y_clean.shape}, dtype: {y_clean.dtype}")
                st.write(f"X_clean shape: {X_clean.shape}")
                st.write(f"X_clean dtypes:\n{X_clean.dtypes}")

# ===============================================================
# Show results
# ===============================================================
if "mlr_results" in st.session_state:
    res = st.session_state["mlr_results"]

    st.subheader(get_text('model_summary'))
    st.write(get_text('model_summary_stats', 
                     n_obs=res['n_obs'], 
                     n_predictors=res['n_predictors'],
                     r_squared=res['r_squared'],
                     adj_r_squared=res['adj_r_squared'],
                     aic=res['aic'],
                     bic=res['bic']))

    # Coefficients
    st.subheader(get_text('coefficients'))
    coef_table = res["coefficients"].copy()
    st.dataframe(coef_table, use_container_width=True)

    # Download coefficients
    st.download_button(
        f"⬇️ {get_text('download_coefficients')}",
        data=fileio.export_to_excel(coef_table, filename="mlr_coefficients.xlsx"),
        file_name="mlr_coefficients.xlsx",
    )

    # ===============================================================
    # VIF
    # ===============================================================
    st.subheader(get_text('vif_title'))
    vif_df = modeling.compute_vif(df[predictors])
    st.dataframe(vif_df)

    # ===============================================================
    # Diagnostics
    # ===============================================================
    st.subheader(get_text('diagnostics'))
    diagnostics = modeling.perform_regression_diagnostics(res)

    # Clean, readable diagnostic output
    norm_test = diagnostics["normality_test"]
    st.write(f"**{get_text('normality_test')}:** Shapiro-Wilk = {norm_test['statistic']:.4f} (p = {norm_test['p_value']:.4f}) " +
             (f"✅ {get_text('normal')}" if bool(norm_test['is_normal']) else f"❌ {get_text('not_normal')}"))
    
    bp_test = diagnostics["heteroskedasticity_bp"]
    st.write(f"**{get_text('breusch_pagan_test')}:** {bp_test['statistic']:.3f} (p = {bp_test['p_value']:.4f}) " +
             (f"✅ {get_text('homoskedastic')}" if bool(bp_test['homoskedastic']) else f"❌ {get_text('heteroskedastic')}"))
    
    white_test = diagnostics["heteroskedasticity_white"]
    st.write(f"**{get_text('white_test')}:** {white_test['statistic']:.3f} (p = {white_test['p_value']:.4f}) " +
             (f"✅ {get_text('homoskedastic')}" if bool(white_test['homoskedastic']) else f"❌ {get_text('heteroskedastic')}"))
    
    indep_test = diagnostics["independence_test"]
    st.write(f"**{get_text('durbin_watson')}:** {indep_test['statistic']:.4f} ({indep_test['interpretation']})")

    outliers = diagnostics["outliers"]
    st.write(f"**{get_text('outliers')}:** {get_text('high_leverage')} = {int(outliers['high_leverage'])} | {get_text('high_cooks')} = {int(outliers['high_cooks'])}")

    # ===============================================================
    # Plots
    # ===============================================================
    st.subheader(get_text('diagnostic_plots'))

    fitted = diagnostics["fitted_values"]
    resid = diagnostics["residuals"]
    studentized = diagnostics["studentized_residuals"]
    cooks = diagnostics["cooks_distance"]
    leverage = diagnostics["leverage"]

    # Residual plots
    try:
        fig_diag = viz.create_residual_plots(fitted, resid, studentized)
        st.pyplot(fig_diag)
    except Exception as e:
        st.warning(get_text('residual_plots_unavailable', error=str(e)))

    # Leverage/Cook's plot
    try:
        fig_leverage = viz.create_leverage_plot(
            leverage=leverage,
            studentized_resid=studentized,
            cooks_distance=cooks,
            n_obs=res["n_obs"],
            n_predictors=res["n_predictors"],
        )
        st.pyplot(fig_leverage)
    except Exception as e:
        st.warning(get_text('leverage_plot_unavailable', error=str(e)))

    # ===============================================================
    # Interpretation
    # ===============================================================    
    st.subheader(get_text('model_interpretation'))
    interpretation = modeling.generate_model_interpretation(res, diagnostics)
    
    # Model Fit Section
    st.write(f"**{get_text('model_fit_quality')}**")
    fit_info = interpretation['model_fit']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(get_text('r_squared'), f"{fit_info['r_squared']:.3f}")
        
    with col2:
        st.metric(get_text('adj_r_squared'), f"{fit_info['adj_r_squared']:.3f}")
        
    with col3:
        r_sq_pct = f"{fit_info['r_squared']:.1%}"
        st.metric(get_text('variance_explained'), r_sq_pct)
    
    # Significant Predictors
    st.write(f"**{get_text('significant_predictors')}**")
    sig_predictors = interpretation['significant_predictors']
    
    if sig_predictors:
        for predictor in sig_predictors:
            coef = predictor['coefficient']
            p_val = predictor['p_value']
            direction = predictor['effect_direction']
            var_name = predictor['variable']
            
            # Color code by effect direction
            if direction == 'positive':
                st.success(f"{var_name}: +{coef:.4f} (p = {p_val:.4f}) - {get_text('positive_effect')}")
            else:
                st.error(f"{var_name}: {coef:.4f} (p = {p_val:.4f}) - {get_text('negative_effect')}")
    else:
        st.info(get_text('no_significant_predictors'))
    
    # Warnings and Recommendations
    warnings = interpretation.get('warnings', [])
    recommendations = interpretation.get('recommendations', [])
    
    if warnings:
        st.write(f"**{get_text('warnings')}**")
        for warning in warnings:
            st.warning(warning)
    
    if recommendations:
        st.write(f"**{get_text('recommendations')}**")
        for rec in recommendations:
            st.info(rec)
    
    # Overall Model Assessment
    st.write(f"**{get_text('overall_assessment')}**")
    r_squared = fit_info['r_squared']
    
    if r_squared >= 0.7:
        st.success(get_text('strong_model'))
    elif r_squared >= 0.5:
        st.info(get_text('moderate_model'))
    elif r_squared >= 0.3:
        st.warning(get_text('weak_model'))
    else:
        st.error(get_text('very_weak_model'))
    
    # Sample size assessment
    n_obs = res['n_obs']
    n_pred = res['n_predictors']
    ratio = n_obs / n_pred
    
    st.write(get_text('sample_size_info', n_obs=n_obs, n_pred=n_pred, ratio=ratio))
    if ratio >= 20:
        st.success(get_text('adequate_sample_size'))
    elif ratio >= 10:
        st.warning(get_text('marginal_sample_size'))
    else:
        st.error(get_text('insufficient_sample_size'))

    # Download model summary (text)
    st.download_button(
        f"⬇️ {get_text('download_model_summary')}",
        data=str(res["results"].summary()),
        file_name="mlr_summary.txt",
    )