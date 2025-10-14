# pages/04_Statistical_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
from utils import statistical_tests  # Import the analysis functions

st.set_page_config(page_title="4 - Statistical Analysis", layout="wide")
st.title("🔬 Statistical Analysis Portal")

# ============================================================
# STEP 1: Upload Data
# ============================================================
st.header("Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.session_state['analysis_df'] = df
    
    st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())
    
    # ============================================================
    # STEP 2: Detect Data Structure
    # ============================================================
    st.header("Step 2: Understand Your Data")
    
    # Auto-detect column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.write(f"**Numeric columns:** {len(numeric_cols)}")
    st.write(f"**Categorical columns:** {len(categorical_cols)}")
    
    # Detect potential paired data
    pre_cols = [col for col in df.columns if 'pre' in col.lower() or 'before' in col.lower()]
    post_cols = [col for col in df.columns if 'post' in col.lower() or 'after' in col.lower()]
    
    has_paired_data = len(pre_cols) > 0 and len(post_cols) > 0
    
    # Detect potential grouping variables
    potential_groups = []
    for col in categorical_cols:
        if df[col].nunique() == 2:
            potential_groups.append(col)
    
    # Show data structure summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Paired Variables Detected", "Yes" if has_paired_data else "No")
    with col2:
        st.metric("Potential Group Variables", len(potential_groups))
    with col3:
        st.metric("Total Variables", len(df.columns))
    
    if has_paired_data:
        st.info(f"Found potential pre-post pairs: {len(pre_cols)} pre variables, {len(post_cols)} post variables")
    
    if potential_groups:
        st.info(f"Found potential grouping variables: {', '.join(potential_groups)}")
    
    # ============================================================
    # STEP 3: Choose Analysis Type
    # ============================================================
    st.header("Step 3: What Do You Want to Analyze?")
    
    analysis_type = st.radio(
        "Select your research question:",
        [
            "Compare two groups (e.g., Treatment A vs B)",
            "Compare before and after (paired data)",
            "Compare change between groups (pre-post + groups)",
            "Analyze survey/Likert scale responses",
            "Explore correlations and relationships",
            "Perform subgroup analysis"
        ]
    )
    
    # ============================================================
    # STEP 4: Variable Selection (Dynamic Based on Analysis Type)
    # ============================================================
    st.header("Step 4: Select Variables")
    
    if analysis_type == "Compare two groups (e.g., Treatment A vs B)":
        st.write("**Independent Groups Comparison (t-test, Mann-Whitney U)**")
        
        group_var = st.selectbox("Select grouping variable (e.g., Treatment group)", 
                                 options=[None] + categorical_cols)
        
        if group_var:
            groups = df[group_var].unique()
            st.write(f"Groups found: {groups}")
            
            outcome_vars = st.multiselect("Select outcome variables to compare", 
                                         options=numeric_cols)
            
            if st.button("Run Analysis"):
                with st.spinner("Running analysis..."):
                    # Call function from utils
                    results = statistical_tests.independent_groups_comparison(
                        df, outcome_vars, group_var
                    )
                    st.session_state['stat_results'] = results
                st.success("Analysis complete!")
        
        # Display results if available
        if 'stat_results' in st.session_state:
            statistical_tests.display_independent_results(st.session_state['stat_results'])
    
    elif analysis_type == "Compare before and after (paired data)":
        st.write("**Paired Analysis (Paired t-test, Wilcoxon signed-rank)**")
        
        if not has_paired_data:
            st.warning("No pre/post columns detected. Please select manually:")
        
        st.write("Match your pre-post variable pairs:")
        
        num_pairs = st.number_input("How many pre-post pairs?", min_value=1, max_value=10, value=1)
        
        pairs = []
        for i in range(num_pairs):
            col1, col2 = st.columns(2)
            with col1:
                pre = st.selectbox(f"Pre variable {i+1}", options=[None] + numeric_cols, key=f"pre_{i}")
            with col2:
                post = st.selectbox(f"Post variable {i+1}", options=[None] + numeric_cols, key=f"post_{i}")
            
            if pre and post:
                pairs.append((pre, post))
        
        if st.button("Run Paired Analysis"):
            with st.spinner(f"Analyzing {len(pairs)} pre-post pairs..."):
                results = statistical_tests.paired_analysis(df, pairs)
                st.session_state['paired_results'] = results
            st.success("Analysis complete!")
        
        if 'paired_results' in st.session_state:
            statistical_tests.display_paired_results(st.session_state['paired_results'])
    
    elif analysis_type == "Compare change between groups (pre-post + groups)":
        st.write("**Change Score Analysis (Mixed Design)**")
        st.info("This analyzes whether the change from pre to post differs between groups")
        
        group_var = st.selectbox("Select grouping variable", options=[None] + categorical_cols)
        
        st.write("Select pre-post pairs:")
        num_pairs = st.number_input("How many outcomes?", min_value=1, max_value=5, value=1)
        
        pairs = []
        for i in range(num_pairs):
            col1, col2 = st.columns(2)
            with col1:
                pre = st.selectbox(f"Pre {i+1}", options=[None] + numeric_cols, key=f"chg_pre_{i}")
            with col2:
                post = st.selectbox(f"Post {i+1}", options=[None] + numeric_cols, key=f"chg_post_{i}")
            if pre and post:
                pairs.append((pre, post))
        
        if st.button("Run Change Analysis"):
            with st.spinner("Calculating change scores..."):
                results = statistical_tests.change_score_analysis(df, pairs, group_var)
                st.session_state['change_results'] = results
            st.success("Analysis complete!")
        
        if 'change_results' in st.session_state:
            statistical_tests.display_change_results(st.session_state['change_results'])
    
    elif analysis_type == "Analyze survey/Likert scale responses":
        st.write("**Likert Scale Analysis**")
        
        likert_vars = st.multiselect("Select Likert scale variables", options=numeric_cols)
        
        group_var = st.selectbox("(Optional) Group by:", options=[None] + categorical_cols)
        
        if st.button("Analyze Survey Data"):
            with st.spinner("Generating analysis..."):
                results = statistical_tests.likert_analysis(df, likert_vars, group_var)
                st.session_state['likert_results'] = results
            st.success("Analysis complete!")
        
        if 'likert_results' in st.session_state:
            statistical_tests.display_likert_results(st.session_state['likert_results'])
    
    elif analysis_type == "Explore correlations and relationships":
        st.write("**Correlation & Regression Analysis**")
        st.info("Use the existing EDA and MLR pages for this analysis")
        
    elif analysis_type == "Perform subgroup analysis":
        st.write("**Subgroup/Stratified Analysis**")
        
        stratify_by = st.selectbox("Stratify analysis by:", options=[None] + categorical_cols)
        
        main_analysis = st.selectbox("What analysis within each subgroup?", 
                                    options=["Compare groups", "Paired comparison", "Correlation"])
        
        if st.button("Run Subgroup Analysis"):
            with st.spinner("Running stratified analysis..."):
                results = statistical_tests.subgroup_analysis(
                    df, stratify_by, main_analysis
                )
                st.session_state['subgroup_results'] = results
            st.success("Analysis complete!")
        
        if 'subgroup_results' in st.session_state:
            statistical_tests.display_subgroup_results(st.session_state['subgroup_results'])

else:
    st.info("👆 Upload your dataset to begin analysis")
    
    # Show example data structure
    st.subheader("📋 Example Data Format")
    
    example_df = pd.DataFrame({
        'Patient_ID': ['P001', 'P002', 'P003', 'P004'],
        'Group': ['Video', 'Control', 'Video', 'Control'],
        'Age': [45, 52, 38, 61],
        'Gender': ['F', 'M', 'F', 'M'],
        'Pre_Anxiety': [8, 7, 9, 6],
        'Post_Anxiety': [3, 6, 4, 5],
        'Pre_Confidence': [2, 3, 2, 4],
        'Post_Confidence': [4, 3, 5, 4]
    })
    
    st.write("**Your data should look like this (one row per participant):**")
    st.dataframe(example_df)
    
    st.write("""
    **Key points:**
    - One row per participant/observation
    - Include a unique ID column
    - Group assignments in one column
    - Pre and post measurements in separate columns
    - All demographics and covariates included
    """)