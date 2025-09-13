# pages/06_KS_Test.py
import streamlit as st
import pandas as pd
from utils.ks_test import run_ks_test, export_ks_to_pdf

st.set_page_config(page_title="6 - KS Test", layout="wide")
st.title("📊 Kolmogorov–Smirnov Test")

# Load data from session (clean preferred, fallback to raw) - same as other pages
df = None
if "df" in st.session_state and st.session_state["df"] is not None:
    df = st.session_state["df"]
elif "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
    df = st.session_state["raw_df"]

if df is None:
    st.warning("No dataset found. Please upload and clean a file on the Import page.")
    st.stop()

st.write(f"**Dataset Info:** {df.shape[0]} rows, {df.shape[1]} columns")

# Select a numeric column
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in dataset.")
    st.stop()

st.subheader("📋 Test Configuration")
column = st.selectbox("Select a column for KS Test:", numeric_cols)

# Select theoretical distribution
distribution = st.selectbox("Distribution to test against:", ["norm"])  # can expand later

# Show some info about selected column
if column:
    col_data = df[column].dropna()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sample Size", len(col_data))
    with col2:
        st.metric("Mean", f"{col_data.mean():.4f}")
    with col3:
        st.metric("Std Dev", f"{col_data.std():.4f}")

# Run test
st.subheader("🧪 Run Test")
if st.button("Run KS Test"):
    with st.spinner("Running Kolmogorov-Smirnov test..."):
        result = run_ks_test(df, column, distribution)
    
    if "error" in result:
        st.error(result["error"])
    else:
        st.success("✅ KS Test completed")
        
        # Display results in a professional format
        st.subheader("📊 Test Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("KS Statistic", f"{result['ks_statistic']:.6f}")
        with col2:
            st.metric("P-value", f"{result['p_value']:.6f}")
        
        # Show conclusion with appropriate styling
        conclusion = result['conclusion']
        if "Fail to reject" in conclusion:
            st.success(f"**Conclusion:** {conclusion}")
        else:
            st.warning(f"**Conclusion:** {conclusion}")
        
        # Additional interpretation
        st.info(f"""
        **Interpretation:**
        - **H₀ (Null Hypothesis):** The sample follows a {distribution} distribution
        - **H₁ (Alternative Hypothesis):** The sample does not follow a {distribution} distribution
        - **Significance Level:** α = 0.05
        - **Decision:** {'Accept H₀' if result['p_value'] > 0.05 else 'Reject H₀'}
        """)
        
        # Store result in session for potential batch export
        if "ks_results" not in st.session_state:
            st.session_state["ks_results"] = []
        st.session_state["ks_results"].append(result)
        
        # Export option
        st.subheader("📥 Export Results")
        pdf_buffer = export_ks_to_pdf([result])
        st.download_button(
            label="📄 Download Results as PDF",
            data=pdf_buffer,
            file_name=f"ks_test_{column}_{distribution}.pdf",
            mime="application/pdf",
        )

# Batch testing option
st.markdown("---")
st.subheader("🔄 Batch Testing")
if st.checkbox("Enable batch testing for multiple columns"):
    batch_cols = st.multiselect(
        "Select columns for batch KS testing:",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )
    
    if batch_cols and st.button("Run Batch KS Tests"):
        batch_results = []
        progress_bar = st.progress(0)
        
        for i, col in enumerate(batch_cols):
            with st.spinner(f"Testing {col}..."):
                result = run_ks_test(df, col, distribution)
                batch_results.append(result)
            progress_bar.progress((i + 1) / len(batch_cols))
        
        st.success(f"✅ Batch testing completed for {len(batch_cols)} columns")
        
        # Display batch results summary
        st.subheader("📊 Batch Results Summary")
        summary_data = []
        for res in batch_results:
            if "error" not in res:
                summary_data.append({
                    "Column": res["column"],
                    "KS Statistic": f"{res['ks_statistic']:.6f}",
                    "P-value": f"{res['p_value']:.6f}",
                    "Follows Distribution": "Yes" if res['p_value'] > 0.05 else "No"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Export batch results
            pdf_buffer = export_ks_to_pdf(batch_results)
            st.download_button(
                label="📄 Download Batch Results as PDF",
                data=pdf_buffer,
                file_name=f"ks_test_batch_{distribution}.pdf",
                mime="application/pdf",
            )