# === pages/2_Explore_and_Correlate.py ===
"""
Exploratory Data Analysis page
Uses utils/eda.py functions
"""
import streamlit as st
import pandas as pd
from utils import eda, fileio
from utils.translations import get_text, init_language

# Initialize language system
init_language()

st.set_page_config(page_title="2 - Explore and Correlate", layout="wide")
st.title(f"📊 {get_text('explore_correlate')}")

# Load data from session (clean preferred, fallback to raw)
df = None
if "df" in st.session_state and st.session_state["df"] is not None:
    df = st.session_state["df"]
elif "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
    df = st.session_state["raw_df"]

if df is None:
    st.warning(get_text('no_data'))
    st.stop()

st.write(f"**{get_text('rows')}:** {df.shape[0]} — **{get_text('columns')}:** {df.shape[1]}")

with st.expander(f"🔍 {get_text('preview_data')} (first 100 rows)"):
    st.dataframe(df.head(100), use_container_width=True)

# =========================================================
# 1. Descriptive statistics
# =========================================================
st.subheader(f"📈 {get_text('descriptive_stats')}")
desc_cols = st.multiselect(
    get_text('select_variables_desc'),
    df.columns.tolist(),
    default=df.select_dtypes(include=["number"]).columns.tolist()[:5]
)
if desc_cols:
    desc_stats = eda.generate_descriptive_stats(df, desc_cols)
    st.dataframe(desc_stats, use_container_width=True)
    st.download_button(
        f"⬇️ {get_text('download_desc_stats')}",
        fileio.export_to_excel(desc_stats, filename="descriptive_stats.xlsx"),
        file_name="descriptive_stats.xlsx",
    )

# =========================================================
# 2. Normality Tests
# =========================================================
st.subheader(f"📊 {get_text('normality_tests')}")
norm_cols = st.multiselect(
    get_text('select_numeric_normality'),
    df.select_dtypes(include=["number"]).columns.tolist(),
    default=df.select_dtypes(include=["number"]).columns.tolist()[:5],
)
if norm_cols:
    normality = eda.detect_distribution_normality(df, norm_cols)
    
    # Display normality results in a professional table format
    normality_results = []
    for col, result in normality.items():
        is_normal = str(result.get('is_normal', False)).replace('np.False_', get_text('no')).replace('np.True_', get_text('yes'))
        normality_results.append({
            get_text('column'): col,
            get_text('test'): result.get('test', 'N/A'),
            get_text('statistic'): f"{result.get('statistic', 0):.4f}",
            get_text('p_value'): f"{result.get('p_value', 0):.6f}",
            get_text('is_normal'): is_normal,
            get_text('sample_size'): result.get('n_obs', 'N/A')
        })
    
    if normality_results:
        normality_df = pd.DataFrame(normality_results)
        st.dataframe(normality_df, use_container_width=True, hide_index=True)
        
        # Summary
        normal_count = sum(1 for result in normality_results if result[get_text('is_normal')] == get_text('yes'))
        total_count = len(normality_results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(get_text('normal_distributions'), f"{normal_count}/{total_count}")
        with col2:
            st.metric(get_text('non_normal_distributions'), f"{total_count - normal_count}/{total_count}")

# =========================================================
# 3. Correlation Analysis
# =========================================================
st.subheader(f"🔗 {get_text('correlation_analysis')}")

# Let users select variables for correlation analysis
corr_cols = st.multiselect(
    get_text('select_variables_corr'),
    df.select_dtypes(include=["number"]).columns.tolist(),
    default=df.select_dtypes(include=["number"]).columns.tolist()
)

method = st.selectbox(get_text('correlation_method'), ["pearson", "spearman", "kendall"], index=0)

if corr_cols and len(corr_cols) >= 2:
    if st.button(get_text('compute_correlation')):
        with st.spinner(f"{get_text('compute_correlation')}..."):
            # Only use selected columns for correlation
            selected_df = df[corr_cols]
            corr_df, pval_df = eda.compute_correlation_with_pvalues(selected_df, method=method)
            st.session_state["corr_df"] = corr_df
            st.session_state["pval_df"] = pval_df
        st.success(f"{get_text('correlation')} computed!")
elif corr_cols and len(corr_cols) < 2:
    st.warning(get_text('select_at_least_two_variables'))
else:
    st.info(get_text('please_select_variables'))

if "corr_df" in st.session_state:
    corr_df = st.session_state["corr_df"]
    pval_df = st.session_state["pval_df"]
    
    st.markdown(f"**{get_text('correlation_matrix')}**")
    st.dataframe(corr_df.round(3), use_container_width=True)
    
    st.markdown(f"**{get_text('pvalues_matrix')}**")
    st.dataframe(pval_df.round(3), use_container_width=True)
    
    # Summary
    summary = eda.get_correlation_summary(corr_df, pval_df, alpha=0.05, top_n=10)
    st.subheader(f"📌 {get_text('correlation_summary')}")
    
    # Display summary statistics in a clean format
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pairs = summary.get("total_pairs", 0)
        st.metric(get_text('total_pairs'), total_pairs)
    
    with col2:
        sig_pairs = summary.get("significant_pairs", 0)
        # Handle numpy int64 conversion
        sig_pairs_clean = str(sig_pairs).replace('np.int64(', '').replace(')', '') if 'np.int64' in str(sig_pairs) else sig_pairs
        st.metric(get_text('significant_pairs'), sig_pairs_clean)
    
    with col3:
        mean_corr = summary.get("mean_correlation", 0)
        st.metric(get_text('mean_correlation'), f"{mean_corr:.3f}")
    
    with col4:
        max_corr = summary.get("max_correlation", 0)
        st.metric(get_text('max_correlation'), f"{max_corr:.3f}")
    
    # Top correlations table
    if "top_correlations" in summary:
        st.write(f"**{get_text('top_correlated_pairs')}**")
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
            st.info(get_text('no_significant_correlations'))
    
    # Downloads
    st.download_button(
        f"⬇️ {get_text('download_correlation')}",
        fileio.export_to_excel(corr_df, filename="correlation_matrix.xlsx"),
        file_name="correlation_matrix.xlsx",
    )
    st.download_button(
        f"⬇️ {get_text('download_pvalues')}",
        fileio.export_to_excel(pval_df, filename="pvalues_matrix.xlsx"),
        file_name="pvalues_matrix.xlsx",
    )