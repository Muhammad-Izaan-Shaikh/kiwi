# === pages/3_Model_MLR.py ===
"""
Modeling page: Multiple Linear Regression (MLR)
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils import modeling, fileio, viz

st.set_page_config(page_title="3 - Model MLR", layout="wide")
st.title("📈 Multiple Linear Regression (MLR)")

# Load cleaned data
df = st.session_state.get("df") or st.session_state.get("raw_df")
if df is None:
    st.warning("No dataset available. Please upload data on the Import page.")
    st.stop()

st.write(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

# ===============================================================
# Select Y and X
# ===============================================================
all_cols = df.columns.tolist()
target = st.selectbox("Choose dependent variable (Y)", [None] + all_cols)
if not target:
    st.info("Please pick a dependent variable to proceed")
    st.stop()

predictors = st.multiselect("Choose predictors (X)", [c for c in all_cols if c != target])
if not predictors:
    st.info("Select at least one predictor")
    st.stop()

# ===============================================================
# Sidebar options
# ===============================================================
st.sidebar.subheader("Model options")
add_intercept = st.sidebar.checkbox("Add intercept", value=True)
robust_se = st.sidebar.checkbox("Use robust (HC3) standard errors", value=False)

# ===============================================================
# Fit model
# ===============================================================
# Replace this section in your pages/3_Model_MLR.py

if st.button("Fit MLR"):
    X = df[predictors].copy()
    y = df[target].copy()
    
    # === DATA TYPE CLEANING (ADD THIS PART) ===
    st.write("### Data Type Diagnostics")
    st.write(f"Original y dtype: {y.dtype}")
    st.write(f"Original X dtypes:\n{X.dtypes}")
    
    # Clean target variable
    y_clean = pd.to_numeric(y, errors='coerce')
    st.write(f"After cleaning y dtype: {y_clean.dtype}")
    
    # Clean predictor variables
    X_clean = X.copy()
    for col in X_clean.columns:
        original_dtype = X_clean[col].dtype
        if col not in ['gender', 'education']:  # Keep categorical as-is for get_dummies
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
            st.write(f"{col}: {original_dtype} -> {X_clean[col].dtype}")
        else:
            st.write(f"{col}: Keeping as categorical ({original_dtype})")
    
    # Check for missing values after conversion
    missing_y = y_clean.isnull().sum()
    missing_X = X_clean.isnull().sum().sum()
    
    if missing_y > 0 or missing_X > 0:
        st.warning(f"Missing values after cleaning - Y: {missing_y}, X: {missing_X}")
        st.write("Missing values by column:")
        st.write(X_clean.isnull().sum())
    else:
        st.success("No missing values after data cleaning!")
        
        with st.spinner("Fitting model..."):
            try:
                results = modeling.fit_ols_model(
                    y_clean,  # Use cleaned data
                    X_clean,  # Use cleaned data
                    add_constant=add_intercept, 
                    robust="HC3" if robust_se else None
                )
                st.session_state["mlr_results"] = results
                st.success("Model fitted successfully!")
            except Exception as e:
                st.error(f"Model fitting failed: {str(e)}")
                st.write("Error details:")
                st.write(f"y_clean shape: {y_clean.shape}, dtype: {y_clean.dtype}")
                st.write(f"X_clean shape: {X_clean.shape}")
                st.write(f"X_clean dtypes:\n{X_clean.dtypes}")

# ===============================================================
# Show results
# ===============================================================
if "mlr_results" in st.session_state:
    res = st.session_state["mlr_results"]

    st.subheader("Model summary")
    st.write(f"**Observations:** {res['n_obs']} — **Predictors:** {res['n_predictors']}")
    st.write(f"**R²:** {res['r_squared']:.3f} — **Adj. R²:** {res['adj_r_squared']:.3f}")
    st.write(f"**AIC:** {res['aic']:.2f} — **BIC:** {res['bic']:.2f}")

    # Coefficients
    st.subheader("Coefficients")
    coef_table = res["coefficients"].copy()
    st.dataframe(coef_table, use_container_width=True)

    # Download coefficients
    st.download_button(
        "⬇️ Download coefficients (xlsx)",
        data=fileio.export_to_excel(coef_table, filename="mlr_coefficients.xlsx"),
        file_name="mlr_coefficients.xlsx",
    )

    # ===============================================================
    # VIF
    # ===============================================================
    st.subheader("Variance Inflation Factor (VIF)")
    vif_df = modeling.compute_vif(df[predictors])
    st.dataframe(vif_df)

    # ===============================================================
    # Diagnostics
    # ===============================================================
    st.subheader("Diagnostics")
    diagnostics = modeling.perform_regression_diagnostics(res)

    st.write("**Normality test:**", diagnostics["normality_test"])
    st.write("**Heteroskedasticity (BP):**", diagnostics["heteroskedasticity_bp"])
    st.write("**Heteroskedasticity (White):**", diagnostics["heteroskedasticity_white"])
    st.write("**Independence test:**", diagnostics["independence_test"])
    st.write("**Outliers:**", {
        "High leverage": diagnostics["outliers"]["high_leverage"],
        "High Cook’s distance": diagnostics["outliers"]["high_cooks"],
    })

    # ===============================================================
    # Plots
    # ===============================================================
    st.subheader("Diagnostic Plots")

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
        st.warning(f"Residual plots not available: {e}")

    # Leverage/Cook’s plot
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
        st.warning(f"Leverage plot not available: {e}")

    # ===============================================================
    # Interpretation
    # ===============================================================
    st.subheader("Interpretation")
    interpretation = modeling.generate_model_interpretation(res, diagnostics)
    st.json(interpretation)

    # Download model summary (text)
    st.download_button(
        "⬇️ Download model summary (txt)",
        data=str(res["results"].summary()),
        file_name="mlr_summary.txt",
    )
