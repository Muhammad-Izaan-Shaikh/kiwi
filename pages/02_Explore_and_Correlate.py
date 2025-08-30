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
df = None
if "df" in st.session_state and st.session_state["df"] is not None:
    df = st.session_state["df"]
elif "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
    df = st.session_state["raw_df"]

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
    
    # Display normality results in a professional table format
    normality_results = []
    for col, result in normality.items():
        is_normal = str(result.get('is_normal', False)).replace('np.False_', 'No').replace('np.True_', 'Yes')
        normality_results.append({
            "Column": col,
            "Test": result.get('test', 'N/A'),
            "Statistic": f"{result.get('statistic', 0):.4f}",
            "P-value": f"{result.get('p_value', 0):.6f}",
            "Is Normal": is_normal,
            "Sample Size": result.get('n_obs', 'N/A')
        })
    
    if normality_results:
        normality_df = pd.DataFrame(normality_results)
        st.dataframe(normality_df, use_container_width=True, hide_index=True)
        
        # Summary
        normal_count = sum(1 for result in normality_results if result["Is Normal"] == "Yes")
        total_count = len(normality_results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Normal Distributions", f"{normal_count}/{total_count}")
        with col2:
            st.metric("Non-Normal Distributions", f"{total_count - normal_count}/{total_count}")

# =========================================================
# 3. Correlation Analysis
# =========================================================
st.subheader("🔗 Correlation Analysis")

# Let users select variables for correlation analysis
corr_cols = st.multiselect(
    "Select variables for correlation analysis",
    df.select_dtypes(include=["number"]).columns.tolist(),
    default=df.select_dtypes(include=["number"]).columns.tolist()
)

method = st.selectbox("Select correlation method", ["pearson", "spearman", "kendall"], index=0)

if corr_cols and len(corr_cols) >= 2:
    if st.button("Compute correlation"):
        with st.spinner("Computing correlation and p-values..."):
            # Only use selected columns for correlation
            selected_df = df[corr_cols]
            corr_df, pval_df = eda.compute_correlation_with_pvalues(selected_df, method=method)
            st.session_state["corr_df"] = corr_df
            st.session_state["pval_df"] = pval_df
        st.success("Correlation computed!")
elif corr_cols and len(corr_cols) < 2:
    st.warning("Please select at least 2 variables for correlation analysis.")
else:
    st.info("Please select variables for correlation analysis.")

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
    
    # Display summary statistics in a clean format
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pairs = summary.get("total_pairs", 0)
        st.metric("Total Pairs", total_pairs)
    
    with col2:
        sig_pairs = summary.get("significant_pairs", 0)
        # Handle numpy int64 conversion
        sig_pairs_clean = str(sig_pairs).replace('np.int64(', '').replace(')', '') if 'np.int64' in str(sig_pairs) else sig_pairs
        st.metric("Significant Pairs", sig_pairs_clean)
    
    with col3:
        mean_corr = summary.get("mean_correlation", 0)
        st.metric("Mean Correlation", f"{mean_corr:.3f}")
    
    with col4:
        max_corr = summary.get("max_correlation", 0)
        st.metric("Max Correlation", f"{max_corr:.3f}")
    
    # Top correlations table
    if "top_correlations" in summary:
        st.write("**Top Correlated Pairs:**")
        top_corr_df = summary["top_correlations"]
        if isinstance(top_corr_df, pd.DataFrame) and not top_corr_df.empty:
            # Format the correlation values for better display
            display_df = top_corr_df.copy()
            if 'correlation' in display_df.columns:
                display_df['correlation'] = display_df['correlation'].round(4)
            if 'p_value' in display_df.columns:
                display_df['p_value'] = display_df['p_value'].round(6)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant correlations found.")
    
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
