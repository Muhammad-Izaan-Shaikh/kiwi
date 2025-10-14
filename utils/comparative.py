# utils/statistical_tests.py
"""
Statistical analysis functions for comparative and paired studies
"""

import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# INDEPENDENT GROUPS COMPARISON
# ============================================================

def independent_groups_comparison(df: pd.DataFrame, outcome_vars: List[str], 
                                 group_var: str) -> Dict:
    """
    Compare outcomes between two independent groups
    """
    results = {
        'group_var': group_var,
        'outcomes': {},
        'groups': df[group_var].unique().tolist()
    }
    
    for outcome in outcome_vars:
        # Get groups
        groups = df[group_var].unique()
        if len(groups) != 2:
            continue
        
        group1_data = df[df[group_var] == groups[0]][outcome].dropna()
        group2_data = df[df[group_var] == groups[1]][outcome].dropna()
        
        # Descriptive stats
        desc_stats = {
            'group1_name': groups[0],
            'group2_name': groups[1],
            'group1_n': len(group1_data),
            'group2_n': len(group2_data),
            'group1_mean': float(group1_data.mean()),
            'group2_mean': float(group2_data.mean()),
            'group1_sd': float(group1_data.std()),
            'group2_sd': float(group2_data.std()),
            'mean_diff': float(group1_data.mean() - group2_data.mean())
        }
        
        # Check normality
        if len(group1_data) >= 3 and len(group2_data) >= 3:
            _, norm1_p = stats.shapiro(group1_data) if len(group1_data) <= 5000 else (0, 1)
            _, norm2_p = stats.shapiro(group2_data) if len(group2_data) <= 5000 else (0, 1)
            is_normal = norm1_p > 0.05 and norm2_p > 0.05
        else:
            is_normal = False
        
        # Perform appropriate test
        if is_normal:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            test_name = "Independent t-test"
            test_stat = t_stat
            
            # Cohen's d effect size
            pooled_sd = np.sqrt(((len(group1_data)-1)*group1_data.std()**2 + 
                                 (len(group2_data)-1)*group2_data.std()**2) / 
                                (len(group1_data) + len(group2_data) - 2))
            cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_sd
            effect_size = cohens_d
        else:
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            test_name = "Mann-Whitney U"
            test_stat = u_stat
            
            # Rank-biserial correlation as effect size
            n1, n2 = len(group1_data), len(group2_data)
            effect_size = 1 - (2*u_stat) / (n1 * n2)
        
        results['outcomes'][outcome] = {
            **desc_stats,
            'test_name': test_name,
            'test_statistic': float(test_stat),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'is_significant': p_value < 0.05,
            'normality_assumption': is_normal
        }
    
    return results


def display_independent_results(results: Dict):
    """Display results of independent groups comparison"""
    st.subheader("📊 Independent Groups Comparison Results")
    
    st.write(f"**Comparing groups:** {results['groups'][0]} vs {results['groups'][1]}")
    
    for outcome, res in results['outcomes'].items():
        st.write(f"### {outcome}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{res['group1_name']} Mean", f"{res['group1_mean']:.2f}",
                     delta=f"SD: {res['group1_sd']:.2f}")
        
        with col2:
            st.metric(f"{res['group2_name']} Mean", f"{res['group2_mean']:.2f}",
                     delta=f"SD: {res['group2_sd']:.2f}")
        
        with col3:
            st.metric("Difference", f"{res['mean_diff']:.2f}")
        
        # Statistical test results
        st.write(f"**Test:** {res['test_name']}")
        st.write(f"**Test statistic:** {res['test_statistic']:.3f}")
        st.write(f"**P-value:** {res['p_value']:.4f}")
        st.write(f"**Effect size:** {res['effect_size']:.3f}")
        
        if res['is_significant']:
            st.success("✅ Statistically significant difference (p < 0.05)")
        else:
            st.info("No statistically significant difference (p ≥ 0.05)")
        
        st.divider()


# ============================================================
# PAIRED ANALYSIS
# ============================================================

def paired_analysis(df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> Dict:
    """
    Perform paired analysis for pre-post comparisons
    """
    results = {'pairs': {}}
    
    for pre_var, post_var in pairs:
        # Remove missing values pairwise
        valid_idx = df[[pre_var, post_var]].dropna().index
        pre_data = df.loc[valid_idx, pre_var]
        post_data = df.loc[valid_idx, post_var]
        
        if len(pre_data) < 3:
            continue
        
        # Descriptive stats
        desc_stats = {
            'n': len(pre_data),
            'pre_mean': float(pre_data.mean()),
            'post_mean': float(post_data.mean()),
            'pre_sd': float(pre_data.std()),
            'post_sd': float(post_data.std()),
            'mean_change': float(post_data.mean() - pre_data.mean())
        }
        
        # Check normality of differences
        differences = post_data - pre_data
        if len(differences) <= 5000:
            _, norm_p = stats.shapiro(differences)
            is_normal = norm_p > 0.05
        else:
            is_normal = False
        
        # Perform appropriate test
        if is_normal:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(post_data, pre_data)
            test_name = "Paired t-test"
            test_stat = t_stat
            
            # Cohen's d for paired samples
            cohens_d = (post_data.mean() - pre_data.mean()) / differences.std()
            effect_size = cohens_d
        else:
            # Wilcoxon signed-rank test
            w_stat, p_value = stats.wilcoxon(post_data, pre_data)
            test_name = "Wilcoxon signed-rank"
            test_stat = w_stat
            
            # Effect size r = Z / sqrt(N)
            z_stat = stats.norm.ppf(p_value/2)
            effect_size = abs(z_stat) / np.sqrt(len(differences))
        
        pair_name = f"{pre_var} → {post_var}"
        results['pairs'][pair_name] = {
            **desc_stats,
            'test_name': test_name,
            'test_statistic': float(test_stat),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'is_significant': p_value < 0.05,
            'normality_assumption': is_normal
        }
    
    return results


def display_paired_results(results: Dict):
    """Display results of paired analysis"""
    st.subheader("📊 Paired Analysis Results")
    
    for pair_name, res in results['pairs'].items():
        st.write(f"### {pair_name}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Pre Mean", f"{res['pre_mean']:.2f}",
                     delta=f"SD: {res['pre_sd']:.2f}")
        
        with col2:
            st.metric("Post Mean", f"{res['post_mean']:.2f}",
                     delta=f"SD: {res['post_sd']:.2f}")
        
        with col3:
            st.metric("Mean Change", f"{res['mean_change']:.2f}")
        
        st.write(f"**Sample size:** {res['n']} paired observations")
        st.write(f"**Test:** {res['test_name']}")
        st.write(f"**Test statistic:** {res['test_statistic']:.3f}")
        st.write(f"**P-value:** {res['p_value']:.4f}")
        st.write(f"**Effect size:** {res['effect_size']:.3f}")
        
        if res['is_significant']:
            st.success("✅ Statistically significant change (p < 0.05)")
        else:
            st.info("No statistically significant change (p ≥ 0.05)")
        
        st.divider()


# ============================================================
# CHANGE SCORE ANALYSIS
# ============================================================

def change_score_analysis(df: pd.DataFrame, pairs: List[Tuple[str, str]], 
                         group_var: str) -> Dict:
    """
    Compare change scores between groups
    """
    results = {'pairs': {}, 'group_var': group_var}
    
    for pre_var, post_var in pairs:
        # Calculate change scores
        df_temp = df[[group_var, pre_var, post_var]].dropna()
        df_temp['change'] = df_temp[post_var] - df_temp[pre_var]
        
        # Get groups
        groups = df_temp[group_var].unique()
        if len(groups) != 2:
            continue
        
        group1_change = df_temp[df_temp[group_var] == groups[0]]['change']
        group2_change = df_temp[df_temp[group_var] == groups[1]]['change']
        
        # Compare change scores between groups
        t_stat, p_value = stats.ttest_ind(group1_change, group2_change)
        
        pair_name = f"{pre_var} → {post_var}"
        results['pairs'][pair_name] = {
            'group1_name': groups[0],
            'group2_name': groups[1],
            'group1_change_mean': float(group1_change.mean()),
            'group2_change_mean': float(group2_change.mean()),
            'group1_change_sd': float(group1_change.std()),
            'group2_change_sd': float(group2_change.std()),
            'change_diff': float(group1_change.mean() - group2_change.mean()),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05
        }
    
    return results


def display_change_results(results: Dict):
    """Display change score comparison results"""
    st.subheader("📊 Change Score Analysis Results")
    
    for pair_name, res in results['pairs'].items():
        st.write(f"### {pair_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(f"{res['group1_name']} Change", 
                     f"{res['group1_change_mean']:.2f}",
                     delta=f"SD: {res['group1_change_sd']:.2f}")
        
        with col2:
            st.metric(f"{res['group2_name']} Change", 
                     f"{res['group2_change_mean']:.2f}",
                     delta=f"SD: {res['group2_change_sd']:.2f}")
        
        st.write(f"**Difference in change:** {res['change_diff']:.2f}")
        st.write(f"**t-statistic:** {res['t_statistic']:.3f}")
        st.write(f"**P-value:** {res['p_value']:.4f}")
        
        if res['is_significant']:
            st.success("✅ Groups differ significantly in their change (p < 0.05)")
        else:
            st.info("No significant difference in change between groups (p ≥ 0.05)")
        
        st.divider()


# ============================================================
# LIKERT SCALE ANALYSIS
# ============================================================

def likert_analysis(df: pd.DataFrame, likert_vars: List[str], 
                   group_var: Optional[str] = None) -> Dict:
    """
    Analyze Likert scale responses
    """
    results = {'variables': {}}
    
    for var in likert_vars:
        var_results = {
            'frequencies': df[var].value_counts().sort_index().to_dict(),
            'mean': float(df[var].mean()),
            'median': float(df[var].median()),
            'mode': float(df[var].mode()[0]) if len(df[var].mode()) > 0 else None,
            'sd': float(df[var].std())
        }
        
        if group_var:
            group_stats = df.groupby(group_var)[var].agg(['mean', 'median', 'count'])
            var_results['by_group'] = group_stats.to_dict()
        
        results['variables'][var] = var_results
    
    return results


def display_likert_results(results: Dict):
    """Display Likert scale analysis results"""
    st.subheader("📊 Likert Scale Analysis Results")
    
    for var, res in results['variables'].items():
        st.write(f"### {var}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{res['mean']:.2f}")
        with col2:
            st.metric("Median", f"{res['median']:.1f}")
        with col3:
            st.metric("SD", f"{res['sd']:.2f}")
        
        # Frequency distribution
        st.write("**Response Distribution:**")
        freq_df = pd.DataFrame(list(res['frequencies'].items()), 
                              columns=['Response', 'Count'])
        st.bar_chart(freq_df.set_index('Response'))
        
        st.divider()


# ============================================================
# SUBGROUP ANALYSIS
# ============================================================

def subgroup_analysis(df: pd.DataFrame, stratify_by: str, 
                     analysis_type: str) -> Dict:
    """
    Perform stratified analysis
    """
    results = {'stratify_by': stratify_by, 'subgroups': {}}
    
    for subgroup in df[stratify_by].unique():
        subgroup_df = df[df[stratify_by] == subgroup]
        results['subgroups'][str(subgroup)] = {
            'n': len(subgroup_df),
            'description': f"Analysis for {stratify_by} = {subgroup}"
        }
    
    return results


def display_subgroup_results(results: Dict):
    """Display subgroup analysis results"""
    st.subheader("📊 Subgroup Analysis Results")
    
    st.write(f"**Stratified by:** {results['stratify_by']}")
    
    for subgroup, res in results['subgroups'].items():
        st.write(f"### {subgroup}")
        st.write(f"Sample size: {res['n']}")
        st.divider()