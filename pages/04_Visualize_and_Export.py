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
cols = df.columns.tolist()
x = st.selectbox("X variable", [None] + cols, index=0)
y = st.selectbox("Y variable", [None] + cols, index=0)

if x and y:
    fig = viz.create_scatter_matrix(df, [x, y])
    st.pyplot(fig)

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
