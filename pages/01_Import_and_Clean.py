# pages/1_Import_and_Clean.py
import streamlit as st
import pandas as pd
from utils import fileio

def display_validation_results(validation):
    """Display validation results in a professional, readable format"""
    
    # Dataset Overview
    st.write("**Dataset Overview:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", validation.get('total_rows', 'N/A'))
    with col2:
        st.metric("Total Columns", validation.get('total_columns', 'N/A'))
    with col3:
        st.metric("Memory Usage", f"{validation.get('memory_usage_mb', 0):.2f} MB" if validation.get('memory_usage_mb') else 'N/A')
    
    # Data Quality Metrics
    if 'missing_values' in validation:
        st.write("**Data Quality:**")
        missing_values = validation['missing_values']
        
        if isinstance(missing_values, dict) and missing_values:
            # Show columns with missing values
            missing_df = pd.DataFrame([
                {"Column": col, "Missing Values": count, "Percentage": f"{(count/validation.get('total_rows', 1)*100):.1f}%"}
                for col, count in missing_values.items() if count > 0
            ])
            
            if not missing_df.empty:
                st.warning(f"⚠️ Found missing values in {len(missing_df)} column(s)")
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No missing values found")
        else:
            st.success("✅ No missing values found")
    
    # Duplicate rows
    if 'duplicate_rows' in validation:
        duplicate_count = validation['duplicate_rows']
        if duplicate_count > 0:
            st.warning(f"⚠️ Found {duplicate_count} duplicate row(s)")
        else:
            st.success("✅ No duplicate rows found")
    
    # Data types
    if 'data_types' in validation:
        st.write("**Column Data Types:**")
        dtypes = validation['data_types']
        
        if isinstance(dtypes, dict):
            # Group columns by data type for cleaner display
            type_groups = {}
            for col, dtype in dtypes.items():
                if dtype not in type_groups:
                    type_groups[dtype] = []
                type_groups[dtype].append(col)
            
            for dtype, columns in type_groups.items():
                with st.expander(f"{dtype} ({len(columns)} columns)", expanded=False):
                    st.write(", ".join(columns))
    
    # Additional statistics
    if 'numeric_columns' in validation:
        numeric_cols = validation['numeric_columns']
        if numeric_cols:
            st.write(f"**Numeric Columns:** {len(numeric_cols)} found")
            
    if 'categorical_columns' in validation:
        categorical_cols = validation['categorical_columns']
        if categorical_cols:
            st.write(f"**Categorical Columns:** {len(categorical_cols)} found")

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
