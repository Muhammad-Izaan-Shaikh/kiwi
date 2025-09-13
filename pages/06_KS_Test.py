# pages/06_KS_Test.py
import streamlit as st
import pandas as pd
from utils.ks_test import run_ks_test, export_ks_to_pdf
from utils.translations import get_text, language_selector, init_language

# Initialize language system
init_language()

st.set_page_config(page_title="6 - KS Test", layout="wide")

# Add language selector to sidebar
language_selector()

st.title(f"📊 {get_text('ks_test_title')}")

# Load data from session (clean preferred, fallback to raw) - same as other pages
df = None
if "df" in st.session_state and st.session_state["df"] is not None:
    df = st.session_state["df"]
elif "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
    df = st.session_state["raw_df"]

if df is None:
    st.warning(get_text('no_data'))
    st.stop()

st.write(get_text('dataset_info', rows=df.shape[0], cols=df.shape[1]))

# Select a numeric column
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.error(f"{get_text('error')}: No numeric columns found in dataset.")
    st.stop()

st.subheader(f"📋 {get_text('test_configuration')}")
column = st.selectbox(get_text('select_column_ks'), numeric_cols)

# Select theoretical distribution
distribution = st.selectbox(get_text('distribution_test'), ["norm"])  # can expand later

# Show some info about selected column
if column:
    col_data = df[column].dropna()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(get_text('sample_size'), len(col_data))
    with col2:
        st.metric(get_text('mean'), f"{col_data.mean():.4f}")
    with col3:
        st.metric(get_text('std_dev'), f"{col_data.std():.4f}")

# Run test
st.subheader(f"🧪 {get_text('run_test')}")
if st.button(get_text('run_ks_test')):
    with st.spinner(f"{get_text('run_ks_test')}..."):
        result = run_ks_test(df, column, distribution)
    
    if "error" in result:
        st.error(result["error"])
    else:
        st.success(f"✅ {get_text('ks_completed')}")
        
        # Display results in a professional format
        st.subheader(f"📊 {get_text('test_results')}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(get_text('ks_statistic'), f"{result['ks_statistic']:.6f}")
        with col2:
            st.metric(get_text('p_value'), f"{result['p_value']:.6f}")
        
        # Show conclusion with appropriate styling
        conclusion = result['conclusion']
        if "Fail to reject" in conclusion:
            st.success(f"**{get_text('conclusion')}:** {conclusion}")
        else:
            st.warning(f"**{get_text('conclusion')}:** {conclusion}")
        
        # Additional interpretation
        st.info(f"""
        **{get_text('interpretation')}:**
        - **{get_text('null_hypothesis', dist=distribution)}**
        - **{get_text('alt_hypothesis', dist=distribution)}**
        - **{get_text('significance_level')}**
        - **Decision:** {'Accept H₀' if result['p_value'] > 0.05 else 'Reject H₀'}
        """)
        
        # Store result in session for potential batch export
        if "ks_results" not in st.session_state:
            st.session_state["ks_results"] = []
        st.session_state["ks_results"].append(result)
        
        # Export option
        st.subheader(f"📥 {get_text('export_results')}")
        pdf_buffer = export_ks_to_pdf([result])
        st.download_button(
            label=f"📄 {get_text('download_pdf')}",
            data=pdf_buffer,
            file_name=f"ks_test_{column}_{distribution}.pdf",
            mime="application/pdf",
        )

# Batch testing option
st.markdown("---")
st.subheader(f"🔄 {get_text('batch_testing')}")
if st.checkbox(get_text('enable_batch')):
    batch_cols = st.multiselect(
        get_text('select_batch_cols'),
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )
    
    if batch_cols and st.button(get_text('run_batch_tests')):
        batch_results = []
        progress_bar = st.progress(0)
        
        for i, col in enumerate(batch_cols):
            with st.spinner(f"Testing {col}..."):
                result = run_ks_test(df, col, distribution)
                batch_results.append(result)
            progress_bar.progress((i + 1) / len(batch_cols))
        
        st.success(get_text('batch_completed', count=len(batch_cols)))
        
        # Display batch results summary
        st.subheader(f"📊 {get_text('batch_summary')}")
        summary_data = []
        for res in batch_results:
            if "error" not in res:
                summary_data.append({
                    get_text('column'): res["column"],
                    get_text('ks_statistic'): f"{res['ks_statistic']:.6f}",
                    get_text('p_value'): f"{res['p_value']:.6f}",
                    get_text('follows_distribution'): get_text('yes') if res['p_value'] > 0.05 else get_text('no')
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Export batch results
            pdf_buffer = export_ks_to_pdf(batch_results)
            st.download_button(
                label=f"📄 {get_text('download_batch_pdf')}",
                data=pdf_buffer,
                file_name=f"ks_test_batch_{distribution}.pdf",
                mime="application/pdf",
            )