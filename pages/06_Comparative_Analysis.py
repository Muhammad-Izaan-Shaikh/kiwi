# pages/04_Statistical_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
from utils import comparative

st.set_page_config(page_title="4 - Statistical Analysis", layout="wide")
st.title("🔬 Statistical Analysis Portal")

# ============================================================
# STEP 0: GET DATA FROM SESSION STATE (FIXED)
# ============================================================

# Try to get cleaned data from session state
if "df" in st.session_state and st.session_state["df"] is not None:
    df = st.session_state["df"]
    st.success(f"Using cleaned dataset from Import & Clean: {df.shape[0]} rows × {df.shape[1]} columns")
elif "raw_df" in st.session_state and st.session_state["raw_df"] is not None:
    df = st.session_state["raw_df"]
    st.warning(f"Using raw dataset (not yet cleaned): {df.shape[0]} rows × {df.shape[1]} columns")
else:
    # Allow upload only if no data in session state
    st.info("No dataset found in session. Please upload a file to get started.")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state['analysis_df'] = df
        st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        st.stop()

# ============================================================
# STEP 1: Display Data Overview
# ============================================================

st.header("Step 1: Data Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

with st.expander("View data preview"):
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================
# STEP 2: Detect Data Structure
# ============================================================

st.header("Step 2: Understand Your Data")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove ID columns from analysis (common pattern)
id_candidates = [col for col in numeric_cols if 'id' in col.lower() or 'patient' in col.lower()]
numeric_cols = [col for col in numeric_cols if col not in id_candidates]

st.write(f"**Numeric columns:** {len(numeric_cols)}")
st.write(f"**Categorical columns:** {len(categorical_cols)}")

# Detect potential paired data
pre_cols = [col for col in df.columns if 'pre' in col.lower() or 'before' in col.lower()]
post_cols = [col for col in df.columns if 'post' in col.lower() or 'after' in col.lower()]
has_paired_data = len(pre_cols) > 0 and len(post_cols) > 0

# Detect potential grouping variables
potential_groups = []
for col in categorical_cols:
    n_unique = df[col].nunique()
    if n_unique == 2:
        potential_groups.append(col)

# Show data structure summary
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Paired Variables", "Yes" if has_paired_data else "No")
with col2:
    st.metric("Group Variables", len(potential_groups))
with col3:
    st.metric("Total Variables", len(df.columns))

if has_paired_data:
    st.info(f"Pre-post pairs detected: {len(pre_cols)} pre, {len(post_cols)} post")

if potential_groups:
    st.info(f"Grouping variables: {', '.join(potential_groups)}")

# ============================================================
# STEP 3: Choose Analysis Type
# ============================================================

st.header("Step 3: Select Analysis Type")

analysis_type = st.radio(
    "What do you want to analyze?",
    [
        "Compare two groups (e.g., Treatment A vs B)",
        "Compare before and after (paired data)",
        "Compare change between groups (pre-post + groups)",
        "Analyze survey/Likert scale responses",
        "Perform subgroup analysis"
    ],
    horizontal=False
)

# ============================================================
# STEP 4: Variable Selection (Dynamic)
# ============================================================

st.header("Step 4: Configure Analysis")

if analysis_type == "Compare two groups (e.g., Treatment A vs B)":
    st.subheader("Independent Groups Comparison")
    st.caption("Compare outcomes between two groups using t-test or Mann-Whitney U")
    
    col1, col2 = st.columns(2)
    
    with col1:
        group_var = st.selectbox(
            "Select grouping variable",
            options=[None] + potential_groups,
            help="Column that defines the two groups"
        )
    
    with col2:
        if group_var:
            groups = df[group_var].unique()
            st.write(f"Groups: {', '.join(map(str, groups))}")
            st.write(f"Sample sizes: {[len(df[df[group_var] == g]) for g in groups]}")
    
    outcome_vars = st.multiselect(
        "Select outcome variables to compare",
        options=numeric_cols,
        help="Variables to compare between groups"
    )
    
    if st.button("Run Independent Groups Analysis", key="ind_groups"):
        if not group_var:
            st.error("Please select a grouping variable")
        elif not outcome_vars:
            st.error("Please select at least one outcome variable")
        else:
            with st.spinner("Running analysis..."):
                results = comparative.independent_groups_comparison(
                    df, outcome_vars, group_var
                )
                st.session_state['stat_results'] = results
            st.success("Analysis complete!")
    
    if 'stat_results' in st.session_state:
        comparative.display_independent_results(st.session_state['stat_results'])

# ============================================================

elif analysis_type == "Compare before and after (paired data)":
    st.subheader("Paired Analysis")
    st.caption("Compare pre and post measurements using paired t-test or Wilcoxon signed-rank")
    
    if not has_paired_data:
        st.warning("No pre/post columns detected. Select pairs manually below.")
    
    st.write("**Select pre-post variable pairs:**")
    
    num_pairs = st.number_input(
        "How many pairs?",
        min_value=1,
        max_value=min(10, len(numeric_cols)//2),
        value=1
    )
    
    pairs = []
    for i in range(num_pairs):
        col1, col2 = st.columns(2)
        with col1:
            pre = st.selectbox(
                f"Pre variable {i+1}",
                options=[None] + numeric_cols,
                key=f"pre_{i}"
            )
        with col2:
            post = st.selectbox(
                f"Post variable {i+1}",
                options=[None] + numeric_cols,
                key=f"post_{i}"
            )
        
        if pre and post and pre != post:
            pairs.append((pre, post))
    
    if st.button("Run Paired Analysis", key="paired"):
        if not pairs:
            st.error("Please select at least one pre-post pair")
        else:
            with st.spinner(f"Analyzing {len(pairs)} pairs..."):
                results = comparative.paired_analysis(df, pairs)
                st.session_state['paired_results'] = results
            st.success("Analysis complete!")
    
    if 'paired_results' in st.session_state:
        comparative.display_paired_results(st.session_state['paired_results'])

# ============================================================

elif analysis_type == "Compare change between groups (pre-post + groups)":
    st.subheader("Change Score Analysis")
    st.caption("Test if change from pre to post differs between groups")
    
    col1, col2 = st.columns(2)
    
    with col1:
        group_var = st.selectbox(
            "Select grouping variable",
            options=[None] + potential_groups,
            key="change_group"
        )
    
    with col2:
        if group_var:
            groups = df[group_var].unique()
            st.write(f"Groups: {', '.join(map(str, groups))}")
    
    st.write("**Select pre-post variable pairs:**")
    
    num_pairs = st.number_input(
        "How many outcomes?",
        min_value=1,
        max_value=min(5, len(numeric_cols)//2),
        value=1,
        key="change_pairs_count"
    )
    
    pairs = []
    for i in range(num_pairs):
        col1, col2 = st.columns(2)
        with col1:
            pre = st.selectbox(
                f"Pre {i+1}",
                options=[None] + numeric_cols,
                key=f"chg_pre_{i}"
            )
        with col2:
            post = st.selectbox(
                f"Post {i+1}",
                options=[None] + numeric_cols,
                key=f"chg_post_{i}"
            )
        if pre and post and pre != post:
            pairs.append((pre, post))
    
    if st.button("Run Change Score Analysis", key="change"):
        if not group_var:
            st.error("Please select a grouping variable")
        elif not pairs:
            st.error("Please select at least one pre-post pair")
        else:
            with st.spinner("Calculating change scores..."):
                results = comparative.change_score_analysis(df, pairs, group_var)
                st.session_state['change_results'] = results
            st.success("Analysis complete!")
    
    if 'change_results' in st.session_state:
        comparative.display_change_results(st.session_state['change_results'])

# ============================================================

elif analysis_type == "Analyze survey/Likert scale responses":
    st.subheader("Survey/Likert Analysis")
    st.caption("Analyze response distributions and means")
    
    col1, col2 = st.columns(2)
    
    with col1:
        likert_vars = st.multiselect(
            "Select Likert scale variables",
            options=numeric_cols
        )
    
    with col2:
        group_var = st.selectbox(
            "(Optional) Group by",
            options=[None] + categorical_cols,
            key="likert_group"
        )
    
    if st.button("Analyze Survey Data", key="likert"):
        if not likert_vars:
            st.error("Please select at least one variable")
        else:
            with st.spinner("Generating analysis..."):
                results = comparative.likert_analysis(df, likert_vars, group_var)
                st.session_state['likert_results'] = results
            st.success("Analysis complete!")
    
    if 'likert_results' in st.session_state:
        comparative.display_likert_results(st.session_state['likert_results'])

# ============================================================

elif analysis_type == "Perform subgroup analysis":
    st.subheader("Subgroup Analysis")
    st.caption("Stratify analysis by a grouping variable")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stratify_by = st.selectbox(
            "Stratify by",
            options=[None] + categorical_cols
        )
    
    with col2:
        analysis_within = st.selectbox(
            "Analysis type within subgroups",
            options=["Compare groups", "Paired comparison", "Descriptive"]
        )
    
    if st.button("Run Subgroup Analysis", key="subgroup"):
        if not stratify_by:
            st.error("Please select a stratification variable")
        else:
            with st.spinner("Running stratified analysis..."):
                results = comparative.subgroup_analysis(df, stratify_by, analysis_within)
                st.session_state['subgroup_results'] = results
            st.success("Analysis complete!")
    
    if 'subgroup_results' in st.session_state:
        comparative.display_subgroup_results(st.session_state['subgroup_results'])