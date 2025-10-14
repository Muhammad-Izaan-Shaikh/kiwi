import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import io
import textwrap
from utils.translations import get_text, init_language
# Hide Streamlit branding and menu
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}  /* hides the hamburger menu */
    footer {visibility: hidden;}    /* hides the "Made with Streamlit" footer */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title="Psychology Analytics Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-card {
        background-color: #808080;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .success-banner {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-banner {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-banner {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    # Initialize language first
    if 'language' not in st.session_state:
        st.session_state['language'] = 'en'

    """Initialize all session state variables"""
    defaults = {
        # Data
        'raw_df': None,
        'sheet_name': None,
        'schema': None,
        'clean_df': None,
        
        # Variables
        'target': None,
        'predictors': [],
        'categoricals': [],
        'id_cols': [],
        'exclude_cols': [],
        
        # Cleaning options
        'encoding': 'one_hot',
        'missing_policy': 'drop_rows',
        'outlier_policy': 'none',
        'standardize_predictors': False,
        'drop_first': True,
        
        # Analysis
        'corr_method': 'pearson',
        'corr_matrix': None,
        'pval_matrix': None,
        'model': None,
        'model_results': None,
        'vif': None,
        'diagnostics': None,
        
        # UI
        'guided_mode': True,
        'current_step': 1,
        'report_artifacts': {},
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_sample_dataset():
    """Create a synthetic psychology dataset for demo"""
    np.random.seed(42)
    n = 200
    
    # Demographics
    age = np.random.normal(25, 5, n).astype(int)
    age = np.clip(age, 18, 65)
    
    gender = np.random.choice(['Male', 'Female', 'Other'], n, p=[0.45, 0.50, 0.05])
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.3, 0.4, 0.25, 0.05])
    
    # Likert scales (1-7)
    anxiety_base = np.random.normal(4, 1.2, n)
    stress_items = []
    for i in range(5):
        item = anxiety_base + np.random.normal(0, 0.5, n)
        stress_items.append(np.clip(np.round(item), 1, 7).astype(int))
    
    wellbeing_base = 8 - anxiety_base + np.random.normal(0, 0.8, n)
    wellbeing_items = []
    for i in range(4):
        item = wellbeing_base + np.random.normal(0, 0.6, n)
        wellbeing_items.append(np.clip(np.round(item), 1, 7).astype(int))
    
    # Sleep and performance (continuous)
    sleep_hours = np.clip(np.random.normal(7.5, 1.2, n), 4, 12)
    performance_score = 70 + (sleep_hours - 7.5) * 3 - anxiety_base * 2 + np.random.normal(0, 5, n)
    performance_score = np.clip(performance_score, 0, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'participant_id': range(1, n+1),
        'age': age,
        'gender': gender,
        'education': education,
        'stress_1': stress_items[0],
        'stress_2': stress_items[1],
        'stress_3': stress_items[2],
        'stress_4': stress_items[3],
        'stress_5': stress_items[4],
        'wellbeing_1': wellbeing_items[0],
        'wellbeing_2': wellbeing_items[1],
        'wellbeing_3': wellbeing_items[2],
        'wellbeing_4': wellbeing_items[3],
        'sleep_hours': np.round(sleep_hours, 1),
        'performance_score': np.round(performance_score, 1)
    })
    
    return df

def save_analysis_config():
    """Save current analysis configuration to JSON"""
    config = {
        'data_info': {
            'sheet_name': st.session_state.get('sheet_name'),
            'n_rows': len(st.session_state.clean_df) if st.session_state.clean_df is not None else 0,
            'n_cols': len(st.session_state.clean_df.columns) if st.session_state.clean_df is not None else 0,
        },
        'variables': {
            'target': st.session_state.target,
            'predictors': st.session_state.predictors,
            'categoricals': st.session_state.categoricals,
            'id_cols': st.session_state.id_cols,
            'exclude_cols': st.session_state.exclude_cols,
        },
        'cleaning': {
            'encoding': st.session_state.encoding,
            'missing_policy': st.session_state.missing_policy,
            'outlier_policy': st.session_state.outlier_policy,
            'standardize_predictors': st.session_state.standardize_predictors,
            'drop_first': st.session_state.drop_first,
        },
        'analysis': {
            'corr_method': st.session_state.corr_method,
        }
    }
    
    return json.dumps(config, indent=2)

def load_analysis_config(config_json):
    """Load analysis configuration from JSON"""
    try:
        config = json.loads(config_json)
        
        # Update session state
        for category, settings in config.items():
            if category == 'variables':
                for key, value in settings.items():
                    st.session_state[key] = value
            elif category == 'cleaning':
                for key, value in settings.items():
                    st.session_state[key] = value
            elif category == 'analysis':
                for key, value in settings.items():
                    st.session_state[key] = value
        
        return True, "Configuration loaded successfully!"
    except Exception as e:
        return False, f"Error loading configuration: {str(e)}"

# Initialize session state
init_session_state()

# Sidebar
with st.sidebar:
    st.markdown("# 🧠 Psychology Analytics")
    st.markdown("---")
    
    # Mode toggle
    st.session_state.guided_mode = st.toggle(
        "Guided Mode", 
        value=st.session_state.guided_mode,
        help="Show explanations and tips for each step"
    )
    
    # Data upload
    st.markdown("### 📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Choose file", 
        type=['csv', 'xlsx'],
        help="Upload CSV or Excel file"
    )
    
    # Sample dataset
    if st.button("📊 Load Sample Dataset"):
        st.session_state.raw_df = create_sample_dataset()
        st.session_state.sheet_name = "sample_data"
        st.success("Sample dataset loaded!")
        st.rerun()
    
    # Config save/load
    st.markdown("### ⚙️ Analysis Config")
    
    if st.session_state.raw_df is not None:
        config_json = save_analysis_config()
        st.download_button(
            "💾 Save Config",
            config_json,
            file_name="analysis_config.json",
            mime="application/json"
        )
    
    config_file = st.file_uploader(
        "📂 Load Config", 
        type=['json'],
        help="Load previously saved analysis configuration"
    )
    
    if config_file is not None:
        config_content = config_file.read().decode('utf-8')
        success, message = load_analysis_config(config_content)
        if success:
            st.success(message)
        else:
            st.error(message)
    
    # App actions
    st.markdown("### 🔧 App Actions")
    
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    

    # Global Language Selector
    st.markdown("### 🌐 Language")
    
    languages = {
        'en': '🇺🇸 English',
        'ru': '🇷🇺 Русский'
    }
    
    current_lang = st.session_state.get('language', 'en')
    
    selected_lang = st.selectbox(
        "Select Language:",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(current_lang),
        key="global_language_selector"
    )
    
    # Update language in session state if changed
    if selected_lang != current_lang:
        st.session_state['language'] = selected_lang
        st.rerun()
    
    # Dark/Light theme toggle would be here if supported by Streamlit
    
    # Progress indicator
    if st.session_state.raw_df is not None:
        st.markdown("### 📋 Progress")
        steps = [
            "Import & Clean",
            "Explore & Correlate", 
            "Model MLR",
            "Visualize & Export"
        ]
        
        for i, step in enumerate(steps, 1):
            if i <= st.session_state.current_step:
                st.markdown(f"✅ Step {i}: {step}")
            else:
                st.markdown(f"⏳ Step {i}: {step}")

# Main content
if uploaded_file is not None:
    # Handle file upload
    try:
        if uploaded_file.type == "text/csv":
            st.session_state.raw_df = pd.read_csv(uploaded_file)
            st.session_state.sheet_name = uploaded_file.name
        else:  # Excel file
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox("Select Excel sheet:", sheet_names)
                st.session_state.raw_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                st.session_state.sheet_name = selected_sheet
            else:
                st.session_state.raw_df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
                st.session_state.sheet_name = sheet_names[0]
        
        st.success(f"File uploaded successfully! Shape: {st.session_state.raw_df.shape}")
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# Main page content
st.markdown('<div class="main-header">🧠 Psychology Analytics Suite</div>', unsafe_allow_html=True)

if st.session_state.raw_df is None:
    st.markdown(
        '<div class="step-card">\n'
        '    <h3>Welcome to Psychology Analytics Suite!</h3>\n'
        '    <p>This application provides SPSS-like statistical analysis capabilities in a user-friendly web interface.</p>\n'
        '    <h4>🚀 Getting Started:</h4>\n'
        '    <ol>\n'
        '        <li><strong>Upload your data</strong> using the sidebar (CSV or Excel)</li>\n'
        '        <li><strong>Or try the sample dataset</strong> to explore features</li>\n'
        '        <li><strong>Follow the 4-step process:</strong>\n'
        '            <ul>\n'
        '                <li>Step 1: Import & Clean Data</li>\n'
        '                <li>Step 2: Explore & Correlate Variables</li>\n'
        '                <li>Step 3: Build Multiple Linear Regression Model</li>\n'
        '                <li>Step 4: Visualize Results & Export Reports</li>\n'
        '            </ul>\n'
        '        </li>\n'
        '    </ol>\n'
        '    <h4>🎯 Features:</h4>\n'
        '    <ul>\n'
        '        <li>Clean and prepare psychological research data</li>\n'
        '        <li>Handle missing values, outliers, and categorical encoding</li>\n'
        '        <li>Comprehensive correlation analysis</li>\n'
        '        <li>Multiple linear regression with diagnostics</li>\n'
        '        <li>Professional reports and visualizations</li>\n'
        '        <li>Export results in multiple formats</li>\n'
        '    </ul>\n'
        '</div>',
        unsafe_allow_html=True
    )
    
else:
    # Show navigation to steps
    st.markdown(textwrap.dedent("""
        <div class="success-banner">
            ✅ Data loaded successfully! Navigate through the analysis steps using the pages in the sidebar.
        </div>
    """), unsafe_allow_html=True)
    
    # Quick data preview
    st.markdown('### 📊 Data Preview', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(st.session_state.raw_df))
    with col2:
        st.metric("Columns", len(st.session_state.raw_df.columns))
    with col3:
        st.metric("Memory", f"{st.session_state.raw_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        missing_pct = (st.session_state.raw_df.isnull().sum().sum() / st.session_state.raw_df.size * 100)
        st.metric("Missing %", f"{missing_pct:.1f}%")
    
    # Show first few rows
    st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)
    
    # Data types summary
    with st.expander("📋 Column Information"):
        col_info = pd.DataFrame({
            'Column': st.session_state.raw_df.columns,
            'Type': st.session_state.raw_df.dtypes.astype(str),
            'Non-Null Count': st.session_state.raw_df.count(),
            'Missing Count': st.session_state.raw_df.isnull().sum(),
            'Missing %': (st.session_state.raw_df.isnull().sum() / len(st.session_state.raw_df) * 100).round(1),
            'Unique Values': st.session_state.raw_df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
    
    # Quick start buttons
    st.markdown("### 🚀 Quick Start", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("➡️ Start Step 1: Import & Clean", type="primary"):
            st.session_state.current_step = 1
            st.switch_page("pages/01_Import_and_Clean.py")
    
    with col2:
        if st.button("🔍 Jump to Step 2: Explore"):
            st.session_state.current_step = 2
            st.switch_page("pages/02_Explore_and_Correlate.py")
    
    with col3:
        if st.button("📈 Jump to Step 3: Model"):
            st.session_state.current_step = 3
            st.switch_page("pages/03_Model_MLR.py")
    
    with col4:
        if st.button("📋 Jump to Step 4: Report"):
            st.session_state.current_step = 4
            st.switch_page("pages/04_Visualize_and_Export.py")