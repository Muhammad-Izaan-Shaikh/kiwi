# === pages/4_Visualize_and_Export.py ===
"""
Visualization hub: show saved plots and custom plotting
Save as: pages/4_Visualize_and_Export.py
"""
import streamlit as st
import pandas as pd
from utils import viz, fileio, modeling

st.set_page_config(page_title="4 - Visualize and Export", layout="wide")
st.title("📉 Visualization")

# Load data
df = None
if "df" in st.session_state and st.session_state["df"] is not None:
    df = st.session_state["df"]
elif "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
    df = st.session_state["raw_df"]
    
if df is None:
    st.warning("No data found. Upload a dataset on the Import page.")
    st.stop()

# ===============================================================
# Correlation heatmap
# ===============================================================
st.subheader("Correlation heatmap")
if "corr_df" in st.session_state:
    corr_df = st.session_state["corr_df"]
    pval_df = st.session_state.get("pval_df")
    fig = viz.create_correlation_heatmap(corr_df, pval_df)
    st.pyplot(fig)
else:
    st.info("Run correlation on the EDA page first to see a heatmap.")

# ===============================================================
# Coefficient plot
# ===============================================================
st.subheader("Coefficient plot")
if "mlr_results" in st.session_state:
    res = st.session_state["mlr_results"]
    coef_table = res.get("coefficients")  # already included in your modeling.py
    if coef_table is not None and not coef_table.empty:
        fig = viz.create_coefficient_plot(coef_table)
        st.pyplot(fig)
    else:
        st.info("No coefficients found. Fit a model in the Modeling page first.")
else:
    st.info("Fit a model in the Modeling page to generate coefficient plots.")

# ===============================================================
# Custom scatter
# ===============================================================
st.subheader("Custom scatter plot with regression line")

# Only show numeric columns for selection
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

if len(numeric_cols) < 2:
    st.warning("Need at least 2 numeric columns to create a scatter plot.")
else:
    x = st.selectbox("X variable", [None] + numeric_cols, index=0)
    y = st.selectbox("Y variable", [None] + numeric_cols, index=0)
    
    if x and y and x != y:
        try:
            # Ensure we have valid numeric data
            valid_data = df[[x, y]].dropna()
            
            if len(valid_data) < 2:
                st.error(f"Not enough valid data points. Found {len(valid_data)} valid pairs, need at least 2.")
            else:
                # Check if data is actually numeric
                if not (pd.api.types.is_numeric_dtype(valid_data[x]) and pd.api.types.is_numeric_dtype(valid_data[y])):
                    st.error("Selected variables must be numeric for scatter plot.")
                else:
                    fig = viz.create_scatter_matrix(df, [x, y])
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Error creating scatter plot: {str(e)}")
            st.info("Please ensure both variables are numeric and contain valid data.")
    elif x and y and x == y:
        st.warning("Please select different variables for X and Y axes.")

# ===============================================================
# Bundle figures as zip (if any were saved in session)
# ===============================================================
st.markdown("---")
if st.button("Bundle figures and download (zip)"):
    figs = st.session_state.get("report_figures", {})
    if not figs:
        st.warning("No figures collected in session. Plots are generated on-the-fly for download from their pages.")
    else:
        buf = fileio.figures_to_zip_bytes(figs)
        st.download_button("Download figures bundle", data=buf, file_name="figures_bundle.zip")
