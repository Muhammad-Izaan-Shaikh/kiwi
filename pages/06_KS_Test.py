# pages/06_KS_Test.py

import streamlit as st
import pandas as pd
from utils.ks_test import run_ks_test, export_ks_to_pdf

st.title("📊 Kolmogorov–Smirnov Test")

# Load dataset (assume df is already uploaded elsewhere in session state)
if "data" not in st.session_state:
    st.warning("Please upload data first on the main page.")
    st.stop()

df = st.session_state["data"]

# Select a numeric column
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in dataset.")
    st.stop()

column = st.selectbox("Select a column for KS Test:", numeric_cols)

# Select theoretical distribution
distribution = st.selectbox("Distribution to test against:", ["norm"])  # can expand later

# Run test
if st.button("Run KS Test"):
    result = run_ks_test(df, column, distribution)

    if "error" in result:
        st.error(result["error"])
    else:
        st.success("✅ KS Test completed")
        st.json(result)

        # Export option
        pdf_buffer = export_ks_to_pdf([result])
        st.download_button(
            label="📥 Download Results as PDF",
            data=pdf_buffer,
            file_name="ks_test_results.pdf",
            mime="application/pdf",
        )