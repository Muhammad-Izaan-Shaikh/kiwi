# pages/1_Import_and_Clean.py
import streamlit as st
import pandas as pd
from utils import fileio
def app():
    st.title("📂 Import & Clean Data")
    # Upload file
    uploaded_file = st.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            # Read file using utils/io.py
            df = fileio.read_file(uploaded_file)
            # Store in session for later use
            st.session_state["raw_df"] = df.copy()
            st.subheader("🔎 Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            # Validate the data
            st.subheader("✅ Data Validation")
            validation = fileio.validate_dataframe(df)
            st.json(validation)
            # Cleaning options
            st.subheader("🧹 Data Cleaning Options")
            if st.checkbox("Drop duplicate rows"):
                df = df.drop_duplicates()
            missing_option = st.selectbox(
                "Handle missing values",
                ["Do nothing", "Drop rows with missing values", "Drop columns with missing values", "Fill with mean", "Fill with median", "Fill with mode"],
            )
            if missing_option == "Drop rows with missing values":
                df = df.dropna()
            elif missing_option == "Drop columns with missing values":
                df = df.dropna(axis=1)
            elif missing_option == "Fill with mean":
                df = df.fillna(df.mean(numeric_only=True))
            elif missing_option == "Fill with median":
                df = df.fillna(df.median(numeric_only=True))
            elif missing_option == "Fill with mode":
                df = df.fillna(df.mode().iloc[0])
            # Save cleaned df into session
            st.session_state["df"] = df
            st.subheader("📊 Cleaned Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            st.success("Data imported and cleaned successfully! You can now proceed to EDA.")
        except Exception as e:
            st.error(f"Error while importing file: {e}")
    else:
        st.info("Please upload a CSV or Excel file to get started.")
app()
