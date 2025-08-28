# === pages/2_Explore_and_Correlate.py ===
"""
Exploratory Data Analysis page
Uses utils/eda.py functions
"""

import streamlit as st
import pandas as pd
from utils import eda, fileio

st.set_page_config(page_title="2 - Explore and Correlate", layout="wide")
st.title("📊 Exploratory Data Analysis")

# Load data from session (clean preferred, fallback to raw)
df = st.session_state.get("df") or st.session_state.get("raw_df")
if df is None:
    st.warning("No dataset found. Please upload and clean a file on the Import page.")
    st.stop()

st.write(f"**Rows:** {df.shape[0]} — **Columns:** {df.shape[1]}")

with st.expander("🔍 Preview Data (first 100 rows)"):
    st.dataframe(df.head(100), use_container_width=True)

# =========================================================
# 1. Descriptive statistics
# =========================================================
st.subheader("📈 Descriptive Statistics")

desc_cols = st.multiselect(
    "Select variables for descriptive stats",
    df.columns.tolist(),
    default=df.select_dtypes(include=["number"]).columns.tolist()[:5]
)

if desc_cols:
    desc_stats = eda.generate_descriptive_stats(df, desc_cols)
    st.dataframe(desc_stats, use_container_width=True)

    st.download_button(
        "⬇️ Download descriptive stats",
        fileio.export_to_excel(desc_stats, filename="descriptive_stats.xlsx"),
        file_name="descriptive_stats.xlsx",
    )

# =========================================================
# 2. Normality Tests
# =========================================================
st.subheader("📊 Normality Tests")

norm_cols = st.multiselect(
    "Select numeric columns for normality testing",
    df.select_dtypes(include=["number"]).columns.tolist(),
    default=df.select_dtypes(include=["number"]).columns.tolist()[:5],
)

if norm_cols:
    normality = eda.detect_distribution_normality(df, norm_cols)
    st.json(normality)

# =========================================================
# 3. Correlation Analysis
# =========================================================
st.subheader("🔗 Correlation Analysis")

method = st.selectbox("Select correlation method", ["pearson", "spearman", "kendall"], index=0)

if st.button("Compute correlation"):
    with st.spinner("Computing correlation and p-values..."):
        corr_df, pval_df = eda.compute_correlation_with_pvalues(df, method=method)
        st.session_state["corr_df"] = corr_df
        st.session_state["pval_df"] = pval_df
    st.success("Correlation computed!")

if "corr_df" in st.session_state:
    corr_df = st.session_state["corr_df"]
    pval_df = st.session_state["pval_df"]

    st.markdown("**Correlation Matrix (rounded)**")
    st.dataframe(corr_df.round(3), use_container_width=True)

    st.markdown("**P-values Matrix (rounded)**")
    st.dataframe(pval_df.round(3), use_container_width=True)

    # Summary
    summary = eda.get_correlation_summary(corr_df, pval_df, alpha=0.05, top_n=10)
    st.subheader("📌 Correlation Summary")
    st.json({
        "Total pairs": summary["total_pairs"],
        "Significant pairs": summary["significant_pairs"],
        "Mean correlation": summary["mean_correlation"],
        "Max correlation": summary["max_correlation"],
    })

    st.write("**Top correlated pairs**")
    st.dataframe(summary["top_correlations"], use_container_width=True)

    # Downloads
    st.download_button(
        "⬇️ Download correlation matrix",
        fileio.export_to_excel(corr_df, filename="correlation_matrix.xlsx"),
        file_name="correlation_matrix.xlsx",
    )
    st.download_button(
        "⬇️ Download p-values matrix",
        fileio.export_to_excel(pval_df, filename="pvalues_matrix.xlsx"),
        file_name="pvalues_matrix.xlsx",
    )
