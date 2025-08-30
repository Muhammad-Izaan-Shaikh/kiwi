# pages/1_Import_and_Clean.py
import streamlit as st
import pandas as pd
from utils import fileio

def display_validation_results(validation):
    """Display validation results in a professional, readable format"""
    
    # Extract info section from validation
    info = validation.get('info', {})
    errors = validation.get('errors', [])
    warnings = validation.get('warnings', [])
    is_valid = validation.get('is_valid', True)
    
    # Overall Status
    if is_valid:
        if warnings:
            st.warning("⚠️ Data validation passed with warnings")
        else:
            st.success("✅ Data validation passed successfully")
    else:
        st.error("❌ Data validation failed")
    
    # Display errors if any
    if errors:
        st.subheader("🚨 Errors")
        for error in errors:
            st.error(f"• {error}")
    
    # Display warnings if any
    if warnings:
        st.subheader("⚠️ Warnings")
        for warning in warnings:
            st.warning(f"• {warning}")
    
    # Dataset Overview
    st.write("**Dataset Overview:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", info.get('n_rows', 'N/A'))
    with col2:
        st.metric("Total Columns", info.get('n_cols', 'N/A'))
    with col3:
        memory_mb = info.get('memory_mb', 0)
        st.metric("Memory Usage", f"{memory_mb:.3f} MB" if memory_mb else 'N/A')
    with col4:
        missing_pct = info.get('missing_pct', 0)
        st.metric("Missing Data", f"{missing_pct:.1f}%" if missing_pct is not None else 'N/A')
    
    # Data Quality Section
    st.write("**Data Quality:**")
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing cells
        missing_cells = info.get('missing_cells', 0)
        if str(missing_cells) != '0':
            st.warning(f"⚠️ Missing cells: {missing_cells}")
        else:
            st.success("✅ No missing values")
    
    with col2:
        # Duplicate rows
        duplicate_rows = info.get('duplicate_rows', 0)
        if str(duplicate_rows) != '0':
            st.warning(f"⚠️ Duplicate rows: {duplicate_rows}")
        else:
            st.success("✅ No duplicate rows")
    
    # Column Type Breakdown
    st.write("**Column Types:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        numeric_cols = info.get('numeric_cols', 0)
        st.info(f"🔢 Numeric: {numeric_cols}")
    with col2:
        categorical_cols = info.get('categorical_cols', 0)
        st.info(f"📝 Categorical: {categorical_cols}")
    with col3:
        datetime_cols = info.get('datetime_cols', 0)
        st.info(f"📅 DateTime: {datetime_cols}")

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
            
            # Display validation results professionally
            display_validation_results(validation)
            
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
