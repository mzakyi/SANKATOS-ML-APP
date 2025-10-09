import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table as RLTable, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import base64
from ydata_profiling import ProfileReport
import tempfile
import os
import datetime
import joblib




# Cache data loading for performance
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Cached KPI computation
@st.cache_data
def compute_kpi_base(df, kpi_def_without_filter):
    kpi_df = df.copy()
    return kpi_df

# Function to create PDF report with styled table
def create_pdf_report(kpi_results, filename="kpi_report.pdf"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    data = [['KPI Label', 'Value', 'Group']]
    for kpi in kpi_results:
        data.append([kpi['label'], kpi['value'], kpi['group'] or 'N/A'])
    t = RLTable(data, colWidths=[2*inch, 2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 14),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BOX', (0,0), (-1,-1), 2, colors.darkblue),
    ]))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.title("Dataset Cleaner, Insights & ML Predictions App")

# Initialize session state
def initialize_session_state():
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    if 'selected_insight_options' not in st.session_state:
        st.session_state.selected_insight_options = []
    if 'selected_clean_options' not in st.session_state:
        st.session_state.selected_clean_options = []
    if 'selected_change_types' not in st.session_state:
        st.session_state.selected_change_types = []
    if 'selected_delete_cols' not in st.session_state:
        st.session_state.selected_delete_cols = []
    if 'rename_cols' not in st.session_state:
        st.session_state.rename_cols = {}
    if 'new_columns' not in st.session_state:
        st.session_state.new_columns = []
    if 'kpi_definitions' not in st.session_state:
        st.session_state.kpi_definitions = []
    if 'selected_viz_cols' not in st.session_state:
        st.session_state.selected_viz_cols = []
    if 'selected_viz_type' not in st.session_state:
        st.session_state.selected_viz_type = "Choose plot type"
    if 'model_pipeline' not in st.session_state:
        st.session_state.model_pipeline = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    if 'task' not in st.session_state:
        st.session_state.task = None
    if 'selected_feature_cols' not in st.session_state:
        st.session_state.selected_feature_cols = []
    if 'num_cols' not in st.session_state:
        st.session_state.num_cols = []
    if 'cat_cols' not in st.session_state:
        st.session_state.cat_cols = []
    if 'selected_target_col' not in st.session_state:
        st.session_state.selected_target_col = None

initialize_session_state()

# Ensure rename_cols is a dictionary
if not isinstance(st.session_state.rename_cols, dict):
    st.warning("Session state 'rename_cols' was not a dictionary. Resetting to empty dictionary.")
    st.session_state.rename_cols = {}

# Description Section
st.markdown("""
### Welcome to the Sankatos App
The Sankatos App is a powerful tool for data analysis and machine learning. Upload your CSV dataset to clean and preprocess data, generate detailed profile reports, explore insights with interactive visualizations, define custom KPIs, and build predictive models with automated or manual ML options. Perfect for data enthusiasts and professionals seeking actionable insights.
""")

# Logo
st.markdown("""
<style>
.logo-container {
position: fixed;
top: 80px;
left: 40px;
width: 120px;
height: 120px;
z-index: 999;
}
</style>
<div class="logo-container">
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
 <defs>
   <linearGradient id="purpleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
     <stop offset="0%" style="stop-color:#8B5CF6;stop-opacity:1" />
     <stop offset="100%" style="stop-color:#6366F1;stop-opacity:1" />
   </linearGradient>
   <filter id="glow">
     <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
     <feMerge>
       <feMergeNode in="coloredBlur"/>
       <feMergeNode in="SourceGraphic"/>
     </feMerge>
   </filter>
 </defs>
 <circle cx="100" cy="100" r="90" fill="url(#purpleGradient)" filter="url(#glow)"/>
 <text x="100" y="145" font-family="Arial, sans-serif" font-size="120" font-weight="bold" fill="white" text-anchor="middle">S</text>
 <g transform="translate(150, 50)">
   <path d="M 0 -6 L 1 -1.5 L 6 0 L 1 1.5 L 0 6 L -1 1.5 L -6 0 L -1 -1.5 Z" fill="white" opacity="0.9"/>
 </g>
 <g transform="translate(55, 155)">
   <path d="M 0 -4 L 0.7 -1 L 4 0 L 0.7 1 L 0 4 L -0.7 1 L -4 0 L -0.7 -1 Z" fill="white" opacity="0.8"/>
 </g>
</svg>
</div>
""", unsafe_allow_html=True)

# About Section
st.markdown("""
<style>
.about-container {
    position: fixed;
    top: 80px;
    right: 40px;
    width: 200px;
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    z-index: 999;
    font-family: Arial, sans-serif;
}
.about-title {
    font-size: 1.2rem;
    font-weight: bold;
    color: #333;
    margin-bottom: 5px;
}
.about-content {
    font-size: 0.9rem;
    color: #555;
}
</style>
<div class="about-container">
    <div class="about-title">Owner/Creator</div>
    <div class="about-content">My name is Mohammed Zakyi and I am a data scientist specializing in machine learning and data analytics. I created this app to help users unlock insights from their data with ease. Feel free to contact me for collaborations or inquiries!
</div>
""", unsafe_allow_html=True)

# Step 1: Data Loading
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Reset session state when a new file is uploaded
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.cleaned_df = load_data(uploaded_file)
        st.session_state.history = [st.session_state.cleaned_df.copy()]
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.model_pipeline = None
        st.session_state.model_name = None
        st.session_state.task = None
        st.session_state.selected_feature_cols = []
        st.session_state.num_cols = []
        st.session_state.cat_cols = []
        st.session_state.selected_target_col = None
        st.session_state.selected_insight_options = []
        st.session_state.selected_clean_options = []
        st.session_state.selected_change_types = []
        st.session_state.selected_delete_cols = []
        st.session_state.rename_cols = {}
        st.session_state.new_columns = []
        st.session_state.kpi_definitions = []
        st.session_state.selected_viz_cols = []
        st.session_state.selected_viz_type = "Choose plot type"

if st.session_state.cleaned_df is not None:
    df = st.session_state.cleaned_df
    st.write("Original Dataset Preview:")
    st.dataframe(df.head())

    # Undo Last Action Button
    if len(st.session_state.history) > 1:
        if st.button("Undo Last Action"):
            st.session_state.history.pop()
            st.session_state.cleaned_df = st.session_state.history[-1].copy()
            # Clean up session state for deleted or renamed columns
            if not isinstance(st.session_state.rename_cols, dict):
                st.session_state.rename_cols = {}
            st.session_state.rename_cols = {k: v for k, v in st.session_state.rename_cols.items() if k in st.session_state.cleaned_df.columns}
            st.session_state.selected_change_types = [col for col in st.session_state.selected_change_types if col in st.session_state.cleaned_df.columns]
            st.session_state.selected_delete_cols = [col for col in st.session_state.selected_delete_cols if col in st.session_state.cleaned_df.columns]
            st.session_state.selected_viz_cols = [col for col in st.session_state.selected_viz_cols if col in st.session_state.cleaned_df.columns]
            st.session_state.selected_feature_cols = [col for col in st.session_state.selected_feature_cols if col in st.session_state.cleaned_df.columns]
            if st.session_state.selected_target_col not in st.session_state.cleaned_df.columns:
                st.session_state.selected_target_col = None
            st.success("Last action undone. Dataset restored to previous state.")
            st.rerun()

    # Step 2: Dataset Profiling
    st.header("Dataset Profiling")
    st.write("Generate profile reports to analyze the dataset before and after cleaning.")
    
    if len(df) > 10000 or len(df.columns) > 50:
        st.warning("Large dataset detected (>10,000 rows or >50 columns). Profile report generation may take time or cause memory issues.")
    
    if st.button("Generate Initial Dataset Profile Report"):
        with st.spinner("Generating initial profile report..."):
            try:
                profile = ProfileReport(df, title="Initial Dataset Profile Report", minimal=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                    profile.to_file(tmp_file.name)
                    with open(tmp_file.name, "r", encoding="utf-8") as f:
                        html_content = f.read()
                st.subheader("Initial Dataset Profile Report")
                st.components.v1.html(html_content, height=1000, scrolling=True)
                b64 = base64.b64encode(html_content.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="initial_profile_report.html">Download Initial Profile Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                os.unlink(tmp_file.name)
            except Exception as e:
                st.error(f"Error generating initial profile report: {str(e)}")

    # Step 3: Data Cleaning Options
    st.header("Clean the Dataset")
    
    def update_clean_options():
        st.session_state.selected_clean_options = st.session_state.clean_options
    
    clean_options = st.multiselect(
        "Select cleaning steps:",
        [
            "Remove duplicates",
            "Fill missing values (mean for numerics, mode for categoricals)",
            "Drop rows with missing values",
            "Convert columns to numeric (if possible)",
            "Remove outliers (IQR method for numeric columns)"
        ],
        default=st.session_state.selected_clean_options,
        key="clean_options",
        on_change=update_clean_options
    )

    if clean_options and st.button("Apply Cleaning"):
        cleaned_df = st.session_state.cleaned_df.copy()
        if "Remove duplicates" in clean_options:
            cleaned_df = cleaned_df.drop_duplicates()
        if "Fill missing values (mean for numerics, mode for categoricals)" in clean_options:
            for col in cleaned_df.columns:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                else:
                    mode_val = cleaned_df[col].mode()
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
        if "Drop rows with missing values" in clean_options:
            cleaned_df = cleaned_df.dropna()
        if "Convert columns to numeric (if possible)" in clean_options:
            for col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
        if "Remove outliers (IQR method for numeric columns)" in clean_options:
            num_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
            for col in num_cols:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
        st.session_state.cleaned_df = cleaned_df
        st.session_state.history.append(cleaned_df.copy())
        st.session_state.selected_clean_options = clean_options
        st.success("Cleaning applied successfully!")
        st.write("Cleaned Dataset Preview:")
        st.dataframe(cleaned_df.head())
        st.download_button("Download Cleaned CSV", cleaned_df.to_csv(index=False), file_name="cleaned_dataset.csv")
    else:
        st.write("Current Dataset Preview:")
        st.dataframe(st.session_state.cleaned_df.head())
        st.download_button("Download Current CSV", st.session_state.cleaned_df.to_csv(index=False), file_name="current_dataset.csv")

    if len(st.session_state.cleaned_df) > 10000 or len(st.session_state.cleaned_df.columns) > 50:
        st.warning("Large dataset detected (>10,000 rows or >50 columns). Profile report generation may take time or cause memory issues.")
    
    if st.button("Generate Cleaned Dataset Profile Report"):
        with st.spinner("Generating cleaned profile report..."):
            try:
                profile = ProfileReport(st.session_state.cleaned_df, title="Cleaned Dataset Profile Report", minimal=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                    profile.to_file(tmp_file.name)
                    with open(tmp_file.name, "r", encoding="utf-8") as f:
                        html_content = f.read()
                st.subheader("Cleaned Dataset Profile Report")
                st.components.v1.html(html_content, height=1000, scrolling=True)
                b64 = base64.b64encode(html_content.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="cleaned_profile_report.html">Download Cleaned Profile Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                os.unlink(tmp_file.name)
            except Exception as e:
                st.error(f"Error generating cleaned profile report: {str(e)}")

    # Step 4: Exploring The Dataset
    st.header("Exploring The Dataset")
    
    def update_insight_options():
        st.session_state.selected_insight_options = st.session_state.insight_options
    
    insight_options = st.multiselect(
        "Choose insights to display:",
        ["Summary Statistics", "Value Counts (for categorical columns)", "Unique Values & Counts by Column", 
         "Correlation Heatmap", "Histogram for Numeric Columns", "Box Plot for Numeric Columns", 
         "Strong Correlations", "Data Types", "Describe", "Info"],
        default=st.session_state.selected_insight_options,
        key="insight_options",
        on_change=update_insight_options
    )

    if "Summary Statistics" in insight_options:
        st.subheader("Summary Statistics")
        st.dataframe(st.session_state.cleaned_df.describe())

    if "Value Counts (for categorical columns)" in insight_options:
        st.subheader("Value Counts")
        cat_cols = st.session_state.cleaned_df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            st.write(f"Value Counts for {col}:")
            st.dataframe(st.session_state.cleaned_df[col].value_counts().reset_index())

    if "Unique Values & Counts by Column" in insight_options:
        st.subheader("Unique Values & Counts by Column")
        selected_cols_for_unique = st.multiselect(
            "Select columns to view unique values and counts:",
            st.session_state.cleaned_df.columns.tolist(),
            key="unique_cols_select"
        )
        if selected_cols_for_unique:
            for col in selected_cols_for_unique:
                with st.expander(f"📊 {col} - Unique Values & Counts"):
                    unique_count = st.session_state.cleaned_df[col].nunique()
                    total_count = len(st.session_state.cleaned_df[col])
                    null_count = st.session_state.cleaned_df[col].isnull().sum()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Unique Values", unique_count)
                    with col2:
                        st.metric("Total Records", total_count)
                    with col3:
                        st.metric("Null Values", null_count)
                    st.write("**Value Counts:**")
                    value_counts_df = st.session_state.cleaned_df[col].value_counts().reset_index()
                    value_counts_df.columns = [col, 'Count']
                    value_counts_df['Percentage'] = (value_counts_df['Count'] / total_count * 100).round(2)
                    st.dataframe(value_counts_df, use_container_width=True)
                    if len(value_counts_df) > 0:
                        top_n = min(10, len(value_counts_df))
                        fig = px.bar(
                            value_counts_df.head(top_n), 
                            x=col, 
                            y='Count',
                            title=f"Top {top_n} Values in {col}",
                            labels={col: col, 'Count': 'Frequency'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

    if "Correlation Heatmap" in insight_options:
        st.subheader("Correlation Heatmap")
        num_cols = st.session_state.cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 1:
            corr = st.session_state.cleaned_df[num_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.write("Need at least 2 numeric columns for correlation.")

    if "Histogram for Numeric Columns" in insight_options:
        st.subheader("Histograms")
        num_cols = st.session_state.cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            fig, ax = plt.subplots()
            st.session_state.cleaned_df[col].hist(ax=ax)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)

    if "Box Plot for Numeric Columns" in insight_options:
        st.subheader("Box Plots")
        num_cols = st.session_state.cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=st.session_state.cleaned_df[col], ax=ax)
            ax.set_title(f"Box Plot of {col}")
            st.pyplot(fig)

    if "Strong Correlations" in insight_options:
        st.subheader("Strong Correlations")
        num_cols = st.session_state.cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 1:
            corr_matrix = st.session_state.cleaned_df[num_cols].corr()
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strong_corrs.append((col1, col2, corr_value))
            if strong_corrs:
                st.write("Column pairs with strong correlation (|r| > 0.7):")
                corr_df = pd.DataFrame(strong_corrs, columns=['Column 1', 'Column 2', 'Correlation Score'])
                st.dataframe(corr_df)
            else:
                st.write("No strong correlations (|r| > 0.7) found among numeric columns.")

    if "Data Types" in insight_options:
        st.subheader("Data Types")
        st.write(st.session_state.cleaned_df.dtypes)

    if "Describe" in insight_options:
        st.subheader("Describe")
        st.dataframe(st.session_state.cleaned_df.describe(include='all'))

    if "Info" in insight_options:
        st.subheader("Info")
        buffer = io.StringIO()
        st.session_state.cleaned_df.info(buf=buffer)
        st.text(buffer.getvalue())

    # Step 5: Change Data Types
    st.header("Change Data Types")
    
    def update_change_types():
        st.session_state.selected_change_types = st.session_state.change_types
    
    change_types = st.multiselect(
        "Select columns to change data type",
        st.session_state.cleaned_df.columns,
        default=[col for col in st.session_state.selected_change_types if col in st.session_state.cleaned_df.columns],
        key="change_types",
        on_change=update_change_types
    )
    
    if change_types:
        type_changes = {}
        for col in change_types:
            current_type = str(st.session_state.cleaned_df[col].dtype)
            default_index = ["int64", "float64", "object", "category"].index(current_type) if current_type in ["int64", "float64", "object", "category"] else 2
            new_type = st.selectbox(f"New data type for {col}", ["int64", "float64", "object", "category"], key=f"type_{col}", index=default_index)
            type_changes[col] = new_type
        if type_changes and st.button("Apply Data Type Changes"):
            cleaned_df = st.session_state.cleaned_df.copy()
            for col, dtype in type_changes.items():
                try:
                    cleaned_df[col] = cleaned_df[col].astype(dtype)
                except Exception as e:
                    st.error(f"Error converting {col} to {dtype}: {str(e)}")
            st.session_state.cleaned_df = cleaned_df
            st.session_state.history.append(cleaned_df.copy())
            st.session_state.selected_change_types = change_types
            st.success("Data type changes applied successfully!")
            st.write("Updated Dataset Preview:")
            st.dataframe(cleaned_df.head())

    # Step 6: Delete Columns
    st.header("Delete Columns")
    
    def update_delete_cols():
        st.session_state.selected_delete_cols = st.session_state.delete_cols
    
    delete_cols = st.multiselect(
        "Select columns to delete",
        st.session_state.cleaned_df.columns,
        default=[col for col in st.session_state.selected_delete_cols if col in st.session_state.cleaned_df.columns],
        key="delete_cols",
        on_change=update_delete_cols
    )
    
    if delete_cols and st.button("Delete Selected Columns"):
        cleaned_df = st.session_state.cleaned_df.drop(columns=delete_cols)
        st.session_state.cleaned_df = cleaned_df
        st.session_state.history.append(cleaned_df.copy())
        st.session_state.selected_delete_cols = []
        # Clean up other session state variables
        if not isinstance(st.session_state.rename_cols, dict):
            st.session_state.rename_cols = {}
        st.session_state.rename_cols = {k: v for k, v in st.session_state.rename_cols.items() if k not in delete_cols}
        st.session_state.selected_change_types = [col for col in st.session_state.selected_change_types if col not in delete_cols]
        st.session_state.selected_viz_cols = [col for col in st.session_state.selected_viz_cols if col not in delete_cols]
        st.session_state.selected_feature_cols = [col for col in st.session_state.selected_feature_cols if col not in delete_cols]
        if st.session_state.selected_target_col in delete_cols:
            st.session_state.selected_target_col = None
        st.success("Selected columns deleted successfully!")
        st.write("Updated Dataset Preview:")
        st.dataframe(cleaned_df.head())

    # Step 7: Rename Columns - FIXED SECTION
    st.header("Rename Columns")
    if not isinstance(st.session_state.rename_cols, dict):
        st.warning("Session state 'rename_cols' was not a dictionary. Resetting to empty dictionary.")
        st.session_state.rename_cols = {}
    
    # Get valid columns that exist in the current dataframe
    valid_rename_cols = [col for col in st.session_state.rename_cols.keys() if col in st.session_state.cleaned_df.columns]
    
    # Use empty list as default if valid_rename_cols is empty
    rename_cols = st.multiselect(
        "Select columns to rename",
        st.session_state.cleaned_df.columns,
        default=valid_rename_cols if valid_rename_cols else [],  # FIX: Use empty list when no valid columns
        key="rename_cols_select"  # Changed key name to avoid conflict
    )
    
    new_names = {}
    for col in rename_cols:
        default_name = st.session_state.rename_cols.get(col, col)
        new_name = st.text_input(f"New name for {col}", value=default_name, key=f"rename_{col}")
        if new_name and new_name != col and new_name not in st.session_state.cleaned_df.columns:
            new_names[col] = new_name
    if new_names and st.button("Apply Renames"):
        cleaned_df = st.session_state.cleaned_df.rename(columns=new_names)
        st.session_state.cleaned_df = cleaned_df
        st.session_state.history.append(cleaned_df.copy())
        # Update rename_cols: remove old mappings, add new ones
        if not isinstance(st.session_state.rename_cols, dict):
            st.session_state.rename_cols = {}
        st.session_state.rename_cols = {k: v for k, v in st.session_state.rename_cols.items() if k not in new_names}
        st.session_state.rename_cols.update(new_names)
        # Update other session state variables
        st.session_state.selected_change_types = [new_names.get(col, col) for col in st.session_state.selected_change_types if col in st.session_state.cleaned_df.columns or col in new_names]
        st.session_state.selected_delete_cols = [new_names.get(col, col) for col in st.session_state.selected_delete_cols if col in st.session_state.cleaned_df.columns or col in new_names]
        st.session_state.selected_viz_cols = [new_names.get(col, col) for col in st.session_state.selected_viz_cols if col in st.session_state.cleaned_df.columns or col in new_names]
        st.session_state.selected_feature_cols = [new_names.get(col, col) for col in st.session_state.selected_feature_cols if col in st.session_state.cleaned_df.columns or col in new_names]
        if st.session_state.selected_target_col in new_names:
            st.session_state.selected_target_col = new_names[st.session_state.selected_target_col]
        st.success("Columns renamed successfully!")
        st.write("Updated Dataset Preview:")
        st.dataframe(cleaned_df.head())

    # Step 8: Add New Columns
    st.header("Add New Columns with Conditions or Arithmetic")
    add_columns = st.checkbox("Add new columns based on conditions or arithmetic")
    if add_columns:
        num_columns = st.number_input("Number of columns to add", min_value=1, max_value=10, value=1, step=1)
        column_definitions = []
        for i in range(num_columns):
            st.write(f"### Column {i+1}")
            new_col_name = st.text_input(f"New column name for Column {i+1}", key=f"new_col_name_{i}")
            if new_col_name:
                operation_type = st.radio(f"Select operation type for Column {i+1}", ["If-Else Condition", "Arithmetic Calculation"], key=f"operation_type_{i}")
                if operation_type == "If-Else Condition":
                    condition_col = st.selectbox(f"Select column for condition for Column {i+1}", st.session_state.cleaned_df.columns, key=f"condition_col_{i}")
                    condition_type = st.selectbox(f"Condition type for Column {i+1}", ["=", ">", "<", ">=", "<=", "!="], key=f"condition_type_{i}")
                    condition_value = st.text_input(f"Condition value for Column {i+1}", key=f"condition_value_{i}")
                    true_value = st.text_input(f"Value if true for Column {i+1}", key=f"true_value_{i}")
                    false_value = st.text_input(f"Value if false for Column {i+1}", key=f"false_value_{i}")
                    column_definitions.append({
                        'name': new_col_name, 'type': 'if-else',
                        'condition_col': condition_col, 'condition_type': condition_type,
                        'condition_value': condition_value, 'true_value': true_value, 'false_value': false_value
                    })
                elif operation_type == "Arithmetic Calculation":
                    col1 = st.selectbox(f"Select first column or enter value for Column {i+1}", [""] + st.session_state.cleaned_df.columns.tolist() + ["Custom Value"], key=f"col1_arithmetic_{i}")
                    val1 = st.text_input(f"Value for Col1 (if Custom Value selected) for Column {i+1}", "0", key=f"val1_{i}") if col1 == "Custom Value" else None
                    col2 = st.selectbox(f"Select second column or enter value for Column {i+1}", [""] + st.session_state.cleaned_df.columns.tolist() + ["Custom Value"], key=f"col2_arithmetic_{i}")
                    val2 = st.text_input(f"Value for Col2 (if Custom Value selected) for Column {i+1}", "1", key=f"val2_{i}") if col2 == "Custom Value" else None
                    operation = st.selectbox(f"Select operation for Column {i+1}", ["+", "-", "*", "/", "%", "**"], key=f"operation_{i}")
                    column_definitions.append({
                        'name': new_col_name, 'type': 'arithmetic',
                        'col1': col1, 'val1': val1, 'col2': col2, 'val2': val2, 'operation': operation
                    })
        if st.button("Apply Columns"):
            cleaned_df = st.session_state.cleaned_df.copy()
            for col_def in column_definitions:
                try:
                    if col_def['type'] == 'if-else':
                        condition_value = pd.to_numeric(col_def['condition_value'], errors='coerce') if col_def['condition_type'] in [">", "<", ">=", "<="] else col_def['condition_value']
                        if col_def['condition_type'] == "=":
                            cleaned_df[col_def['name']] = np.where(cleaned_df[col_def['condition_col']] == condition_value, col_def['true_value'], col_def['false_value'])
                        elif col_def['condition_type'] == ">":
                            cleaned_df[col_def['name']] = np.where(cleaned_df[col_def['condition_col']] > condition_value, col_def['true_value'], col_def['false_value'])
                        elif col_def['condition_type'] == "<":
                            cleaned_df[col_def['name']] = np.where(cleaned_df[col_def['condition_col']] < condition_value, col_def['true_value'], col_def['false_value'])
                        elif col_def['condition_type'] == ">=":
                            cleaned_df[col_def['name']] = np.where(cleaned_df[col_def['condition_col']] >= condition_value, col_def['true_value'], col_def['false_value'])
                        elif col_def['condition_type'] == "<=":
                            cleaned_df[col_def['name']] = np.where(cleaned_df[col_def['condition_col']] <= condition_value, col_def['true_value'], col_def['false_value'])
                        elif col_def['condition_type'] == "!=":
                            cleaned_df[col_def['name']] = np.where(cleaned_df[col_def['condition_col']] != condition_value, col_def['true_value'], col_def['false_value'])
                    elif col_def['type'] == 'arithmetic':
                        if col_def['col1'] == "Custom Value" and col_def['val1']:
                            val1_num = pd.to_numeric(col_def['val1'], errors='coerce')
                            if pd.isna(val1_num):
                                raise ValueError("Invalid numeric value for Col1")
                            series1 = pd.Series([val1_num] * len(cleaned_df), index=cleaned_df.index)
                        else:
                            series1 = cleaned_df[col_def['col1']] if col_def['col1'] else pd.Series([0] * len(cleaned_df), index=cleaned_df.index)
                        if col_def['col2'] == "Custom Value" and col_def['val2']:
                            val2_num = pd.to_numeric(col_def['val2'], errors='coerce')
                            if pd.isna(val2_num):
                                raise ValueError("Invalid numeric value for Col2")
                            series2 = pd.Series([val2_num] * len(cleaned_df), index=cleaned_df.index)
                        else:
                            series2 = cleaned_df[col_def['col2']] if col_def['col2'] else pd.Series([1] * len(cleaned_df), index=cleaned_df.index)
                        series1 = pd.to_numeric(series1, errors='coerce')
                        series2 = pd.to_numeric(series2, errors='coerce')
                        if series1.isna().all() or series2.isna().all():
                            raise ValueError("Columns or values must be numeric")
                        if col_def['operation'] == "+":
                            cleaned_df[col_def['name']] = series1 + series2
                        elif col_def['operation'] == "-":
                            cleaned_df[col_def['name']] = series1 - series2
                        elif col_def['operation'] == "*":
                            cleaned_df[col_def['name']] = series1 * series2
                        elif col_def['operation'] == "/":
                            cleaned_df[col_def['name']] = series1 / series2
                            cleaned_df[col_def['name']] = cleaned_df[col_def['name']].replace([np.inf, -np.inf], np.nan).fillna(0)
                        elif col_def['operation'] == "%":
                            cleaned_df[col_def['name']] = series1 % series2
                            cleaned_df[col_def['name']] = cleaned_df[col_def['name']].replace(np.nan, 0)
                        elif col_def['operation'] == "**":
                            cleaned_df[col_def['name']] = series1 ** series2
                except Exception as e:
                    st.error(f"Error applying column {col_def['name']}: {str(e)}")
            st.session_state.cleaned_df = cleaned_df
            st.session_state.history.append(cleaned_df.copy())
            st.session_state.new_columns.append(column_definitions)
            st.success("New columns added successfully!")
            st.write("Updated Dataset Preview:")
            st.dataframe(cleaned_df.head())


    # Step 10: Visualization Section
    st.header("Visualization")
    st.write("Create custom visualizations based on your data.")
    
    def update_viz_cols():
        st.session_state.selected_viz_cols = st.session_state.viz_cols
    
    def update_viz_type():
        st.session_state.selected_viz_type = st.session_state.viz_type
    
    available_cols = st.session_state.cleaned_df.columns.tolist()
    viz_cols = st.multiselect(
        "Select columns for visualization",
        available_cols,
        default=[col for col in st.session_state.selected_viz_cols if col in available_cols],
        key="viz_cols",
        on_change=update_viz_cols
    )
    
    if viz_cols:
        plot_types = ["Choose plot type"]
        num_cols = st.session_state.cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = st.session_state.cleaned_df.select_dtypes(include=['object']).columns
        if len(viz_cols) >= 1:
            plot_types.extend(["Histogram", "Boxplot", "Pie Chart"])
        if len(viz_cols) >= 2:
            plot_types.extend(["Line Plot", "Scatter Plot", "Bar Graph", "Area Chart"])
        if len(viz_cols) >= 2 and all(col in num_cols for col in viz_cols[:2]):
            plot_types.append("Pairplot")
        if len(num_cols) > 1 and any(col in viz_cols for col in num_cols):
            plot_types.append("Heatmap")
        if len(viz_cols) >= 2 and any(col in cat_cols for col in viz_cols):
            plot_types.append("Grouped Bar Plot")
        viz_type = st.selectbox(
            "Select plot type",
            plot_types,
            index=plot_types.index(st.session_state.selected_viz_type) if st.session_state.selected_viz_type in plot_types else 0,
            key="viz_type",
            on_change=update_viz_type
        )
        if viz_type != "Choose plot type":
            if viz_type == "Line Plot":
                x_col = st.selectbox("Select X-axis", viz_cols, key="line_x")
                y_cols = [col for col in viz_cols if col != x_col]
                if y_cols:
                    y_col = st.selectbox("Select Y-axis", y_cols, key="line_y")
                    fig = px.line(st.session_state.cleaned_df, x=x_col, y=y_col, title=f"{viz_type}: {y_col} vs {x_col}")
                else:
                    fig = px.line(st.session_state.cleaned_df, x=st.session_state.cleaned_df.index, y=x_col, title=f"{viz_type}: {x_col} vs Index")
                st.plotly_chart(fig)
            elif viz_type == "Scatter Plot" and any(col in num_cols for col in viz_cols):
                x_col = st.selectbox("Select X-axis", [col for col in viz_cols if col in num_cols], key="scatter_x")
                y_col = st.selectbox("Select Y-axis", [col for col in viz_cols if col in num_cols and col != x_col], key="scatter_y")
                if x_col and y_col:
                    color_col = st.selectbox("Select color/group (optional)", ["None"] + [col for col in viz_cols if col not in [x_col, y_col]], key="scatter_color")
                    if color_col == "None":
                        color_col = None
                    fig = px.scatter(st.session_state.cleaned_df, x=x_col, y=y_col, color=color_col, title=f"{viz_type}: {y_col} vs {x_col}")
                    st.plotly_chart(fig)
            elif viz_type == "Pairplot" and all(col in num_cols for col in viz_cols[:2]):
                fig = sns.pairplot(st.session_state.cleaned_df[viz_cols])
                st.pyplot(fig)
            elif viz_type == "Heatmap" and any(col in viz_cols for col in num_cols):
                corr_cols = [col for col in viz_cols if col in num_cols]
                if len(corr_cols) > 1:
                    corr = st.session_state.cleaned_df[corr_cols].corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.write("Need at least 2 numeric columns for heatmap.")
            elif viz_type == "Histogram":
                for col in viz_cols:
                    fig, ax = plt.subplots()
                    st.session_state.cleaned_df[col].hist(ax=ax)
                    ax.set_title(f"{viz_type} of {col}")
                    st.pyplot(fig)
            elif viz_type == "Boxplot":
                for col in viz_cols:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=st.session_state.cleaned_df[col], ax=ax)
                    ax.set_title(f"{viz_type} of {col}")
                    st.pyplot(fig)
            elif viz_type == "Pie Chart":
                if any(col in cat_cols for col in viz_cols):
                    names_col = st.selectbox("Select category column", [col for col in viz_cols if col in cat_cols], key="pie_names")
                    if any(col in num_cols for col in viz_cols):
                        values_col = st.selectbox("Select values column", [col for col in viz_cols if col in num_cols and col != names_col], key="pie_values")
                        fig = px.pie(st.session_state.cleaned_df, names=names_col, values=values_col, title=f"{viz_type}: {names_col} vs {values_col}")
                    else:
                        fig = px.pie(st.session_state.cleaned_df, names=names_col, values=None, title=f"{viz_type}: {names_col}")
                    st.plotly_chart(fig)
                else:
                    st.write("Need at least one categorical column for pie chart.")
            elif viz_type == "Bar Graph" and any(col in num_cols for col in viz_cols):
                x_col = st.selectbox("Select X-axis", [col for col in viz_cols if col in cat_cols or col in num_cols], key="bar_x")
                y_col = st.selectbox("Select Y-axis", [col for col in viz_cols if col in num_cols and col != x_col], key="bar_y")
                if x_col and y_col:
                    fig = px.bar(st.session_state.cleaned_df, x=x_col, y=y_col, title=f"{viz_type}: {y_col} vs {x_col}")
                    st.plotly_chart(fig)
            elif viz_type == "Area Chart" and any(col in num_cols for col in viz_cols):
                x_col = st.selectbox("Select X-axis", viz_cols, key="area_x")
                y_col = st.selectbox("Select Y-axis", [col for col in viz_cols if col in num_cols and col != x_col], key="area_y")
                if x_col and y_col:
                    fig = px.area(st.session_state.cleaned_df, x=x_col, y=y_col, title=f"{viz_type}: {y_col} vs {x_col}")
                    st.plotly_chart(fig)
            elif viz_type == "Grouped Bar Plot" and any(col in cat_cols for col in viz_cols):
                x_col = st.selectbox("Select X-axis", [col for col in viz_cols if col in cat_cols], key="group_bar_x")
                y_col = st.selectbox("Select Y-axis", [col for col in viz_cols if col in num_cols and col != x_col], key="group_bar_y")
                color_col = st.selectbox("Select group column", [col for col in viz_cols if col in cat_cols and col != x_col], key="group_bar_color")
                if x_col and y_col and color_col:
                    fig = px.bar(st.session_state.cleaned_df, x=x_col, y=y_col, color=color_col, title=f"{viz_type}: {y_col} by {x_col} and {color_col}")
                    st.plotly_chart(fig)

    # Step 11: Machine Learning Predictions with Hyperparameter Tuning
    st.header("Machine Learning Predictions")
    if len(st.session_state.cleaned_df) < 10:
        st.warning("Dataset is too small for meaningful ML training. Need at least 10 rows.")
    else:
        def update_target_col():
            st.session_state.selected_target_col = st.session_state.target_col
        
        target_col = st.selectbox(
            "Select target column for prediction",
            st.session_state.cleaned_df.columns,
            index=st.session_state.cleaned_df.columns.get_loc(st.session_state.selected_target_col) if st.session_state.selected_target_col in st.session_state.cleaned_df.columns else 0,
            key="target_col",
            on_change=update_target_col
        )
        
        if target_col:
            unique_vals = st.session_state.cleaned_df[target_col].nunique()
            task = 'classification' if pd.api.types.is_object_dtype(st.session_state.cleaned_df[target_col]) or unique_vals <= 20 else 'regression'
            st.write(f"Detected task: {task.capitalize()} (based on target type and uniqueness)")
            
            def update_feature_cols():
                st.session_state.selected_feature_cols = st.session_state.feature_cols
            
            feature_cols = st.multiselect(
                "Select feature columns",
                [col for col in st.session_state.cleaned_df.columns if col != target_col],
                default=[col for col in st.session_state.selected_feature_cols if col in st.session_state.cleaned_df.columns and col != target_col],
                key="feature_cols",
                on_change=update_feature_cols
            )
            
            model_mode = st.radio("Model Selection Mode", ["Manual (select one)", "Auto-select best", "Compare top two"], key="model_mode")
            selected_model = None
            if model_mode == "Manual (select one)":
                model_options = ['RandomForest', 'XGBoost', 'LogisticRegression', 'SVM'] if task == 'classification' else ['RandomForest', 'XGBoost', 'LinearRegression', 'SVM']
                selected_model = st.selectbox("Select ML model", model_options, key="selected_model")
            
            # Add hyperparameter tuning option
            use_tuning = st.checkbox("Use Hyperparameter Tuning (slower but better results)", value=True)
            use_smote = st.checkbox("Handle imbalanced classes with SMOTE") if task == 'classification' else False
            
            if feature_cols and st.button("Run ML Model"):
                with st.spinner("Training model(s) with hyperparameter tuning..." if use_tuning else "Training model(s)..."):
                    try:
                        X = st.session_state.cleaned_df[feature_cols]
                        y = st.session_state.cleaned_df[target_col]
                        if task == 'classification' and pd.api.types.is_object_dtype(y):
                            y = y.astype('category').cat.codes
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        if X_train.isna().any().any() or X_test.isna().any().any():
                            st.error("Input data contains missing values. Please clean the data first.")
                            st.stop()
                        num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
                        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                        st.session_state.num_cols = num_cols
                        st.session_state.cat_cols = cat_cols
                        st.session_state.selected_target_col = target_col
                        transformers = []
                        if num_cols:
                            transformers.append(('num', StandardScaler(), num_cols))
                        if cat_cols:
                            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
                        if not transformers:
                            st.error("No valid numeric or categorical columns selected for preprocessing.")
                            st.stop()
                        preprocessor = ColumnTransformer(transformers=transformers)
                        
                        # Define base models
                        models = {
                            'RandomForest': RandomForestClassifier(random_state=42) if task == 'classification' else RandomForestRegressor(random_state=42),
                            'XGBoost': XGBClassifier(random_state=42, enable_categorical=True) if task == 'classification' else XGBRegressor(random_state=42, enable_categorical=True),
                            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000) if task == 'classification' else LinearRegression(),
                            'SVM': SVC(random_state=42, probability=True) if task == 'classification' else SVR()
                        }
                        
                        # Define hyperparameter grids for tuning
                        if task == 'classification':
                            param_grids = {
                                'RandomForest': {
                                    'model__n_estimators': [100, 200, 300],
                                    'model__max_depth': [10, 20, 30, None],
                                    'model__min_samples_split': [2, 5, 10],
                                    'model__min_samples_leaf': [1, 2, 4]
                                },
                                'XGBoost': {
                                    'model__n_estimators': [100, 200, 300],
                                    'model__learning_rate': [0.01, 0.1, 0.3],
                                    'model__max_depth': [3, 5, 7],
                                    'model__subsample': [0.8, 0.9, 1.0]
                                },
                                'LogisticRegression': {
                                    'model__C': [0.01, 0.1, 1, 10, 100],
                                    'model__penalty': ['l2'],
                                    'model__solver': ['lbfgs', 'saga']
                                },
                                'SVM': {
                                    'model__C': [0.1, 1, 10],
                                    'model__kernel': ['rbf', 'linear'],
                                    'model__gamma': ['scale', 'auto']
                                }
                            }
                        else:  # regression
                            param_grids = {
                                'RandomForest': {
                                    'model__n_estimators': [100, 200, 300],
                                    'model__max_depth': [10, 20, 30, None],
                                    'model__min_samples_split': [2, 5, 10],
                                    'model__min_samples_leaf': [1, 2, 4]
                                },
                                'XGBoost': {
                                    'model__n_estimators': [100, 200, 300],
                                    'model__learning_rate': [0.01, 0.1, 0.3],
                                    'model__max_depth': [3, 5, 7],
                                    'model__subsample': [0.8, 0.9, 1.0]
                                },
                                'LinearRegression': {},  # No hyperparameters to tune
                                'SVM': {
                                    'model__C': [0.1, 1, 10],
                                    'model__kernel': ['rbf', 'linear'],
                                    'model__gamma': ['scale', 'auto']
                                }
                            }
                        
                        def build_pipeline(model):
                            if use_smote and task == 'classification':
                                return ImbPipeline(steps=[('preprocessor', preprocessor), ('smote', SMOTE(random_state=42)), ('model', model)])
                            return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                        
                        def train_with_tuning(name, model, param_grid):
                            """Train model with hyperparameter tuning using GridSearchCV"""
                            pipeline = build_pipeline(model)
                            
                            if use_tuning and param_grid:
                                # Use GridSearchCV for hyperparameter tuning
                                grid_search = GridSearchCV(
                                    pipeline,
                                    param_grid,
                                    cv=5,  # 5-fold cross-validation
                                    scoring='accuracy' if task == 'classification' else 'r2',
                                    n_jobs=-1,  # Use all CPU cores
                                    verbose=0
                                )
                                grid_search.fit(X_train, y_train)
                                best_pipeline = grid_search.best_estimator_
                                best_params = grid_search.best_params_
                                return best_pipeline, best_params
                            else:
                                # Train with default parameters
                                pipeline.fit(X_train, y_train)
                                return pipeline, {}
                        
                        results = {}
                        if model_mode != "Manual (select one)":
                            # Train all models and find the best one
                            for name, model in models.items():
                                st.write(f"Training {name}...")
                                best_pipeline, best_params = train_with_tuning(name, model, param_grids.get(name, {}))
                                y_pred = best_pipeline.predict(X_test)
                                metric_value = accuracy_score(y_test, y_pred) if task == 'classification' else r2_score(y_test, y_pred)
                                results[name] = {
                                    'pipeline': best_pipeline,
                                    'y_pred': y_pred,
                                    'metric_value': metric_value,
                                    'best_params': best_params
                                }
                            sorted_results = sorted(results.items(), key=lambda x: x[1]['metric_value'], reverse=True)
                        
                        def display_model_results(name, pipeline, y_pred, best_params=None):
                            """Display model results including metrics, confusion matrix, and feature importance"""
                            metric_name = "Accuracy" if task == 'classification' else "R² Score"
                            metric_value = accuracy_score(y_test, y_pred) if task == 'classification' else r2_score(y_test, y_pred)
                            
                            st.subheader(f"Model: {name} ({metric_name}: {metric_value:.4f})")
                            
                            # Display best hyperparameters if tuning was used
                            if best_params:
                                st.write("**Best Hyperparameters:**")
                                params_display = {k.replace('model__', ''): v for k, v in best_params.items()}
                                st.json(params_display)
                            
                            # Display metrics
                            if task == 'classification':
                                metrics_text = f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n" + classification_report(y_test, y_pred)
                            else:
                                mse = mean_squared_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)
                                metrics_text = f"Mean Squared Error: {mse:.4f}\nR² Score: {r2:.4f}"
                            st.text(metrics_text)
                            
                            # Actual vs Predicted
                            st.subheader("Actual vs. Predicted")
                            pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index()
                            pred_df.rename(columns={'index': 'Index'}, inplace=True)
                            st.dataframe(pred_df, hide_index=False)
                            
                            # Confusion Matrix or Scatter Plot
                            if task == 'classification':
                                st.subheader("Confusion Matrix")
                                cm = confusion_matrix(y_test, y_pred)
                                fig, ax = plt.subplots()
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                st.pyplot(fig)
                            else:
                                st.subheader("Actual vs. Predicted Scatter Plot")
                                fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, 
                                               trendline='ols', title='Actual vs. Predicted')
                                st.plotly_chart(fig)
                            
                            # Feature Importances
                            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                                st.subheader("Feature Importances")
                                importances = pipeline.named_steps['model'].feature_importances_
                                feature_names = num_cols + (list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)) if cat_cols else [])
                                imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
                                st.dataframe(imp_df)
                        
                        # Display results based on model mode
                        if model_mode == "Manual (select one)":
                            st.write(f"Training {selected_model}...")
                            model = models[selected_model]
                            best_pipeline, best_params = train_with_tuning(selected_model, model, param_grids.get(selected_model, {}))
                            y_pred = best_pipeline.predict(X_test)
                            display_model_results(selected_model, best_pipeline, y_pred, best_params)
                            st.session_state.model_pipeline = best_pipeline
                            st.session_state.model_name = selected_model
                            st.session_state.task = task
                            st.session_state.best_params = best_params
                            
                        elif model_mode == "Auto-select best":
                            best_name, best_result = sorted_results[0]
                            st.success(f"🏆 Best Model: {best_name}")
                            display_model_results(best_name, best_result['pipeline'], best_result['y_pred'], best_result['best_params'])
                            st.session_state.model_pipeline = best_result['pipeline']
                            st.session_state.model_name = best_name
                            st.session_state.task = task
                            st.session_state.best_params = best_result['best_params']
                            
                        elif model_mode == "Compare top two":
                            col1, col2 = st.columns(2)
                            for i, (name, result) in enumerate(sorted_results[:2]):
                                with [col1, col2][i]:
                                    rank = "🥇 Best" if i == 0 else "🥈 Second Best"
                                    st.info(rank)
                                    display_model_results(name, result['pipeline'], result['y_pred'], result['best_params'])
                            
                            # Save the best model
                            best_name, best_result = sorted_results[0]
                            st.session_state.model_pipeline = best_result['pipeline']
                            st.session_state.model_name = best_name
                            st.session_state.task = task
                            st.session_state.best_params = best_result['best_params']
                        
                        st.success("✅ Model training complete! The best model has been saved and can now be used to make new predictions.")
                        
                    except Exception as e:
                        st.error(f"Error running model: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())

        # Prediction Input Section
        if st.session_state.model_pipeline is not None and st.session_state.selected_feature_cols:
            st.subheader("Make New Predictions with Trained Model")
            prediction_mode = st.radio("Prediction Method", ["Manual Input", "Upload CSV"], key="prediction_mode")
            if prediction_mode == "Manual Input":
                st.write(f"Enter values for the selected features to get a prediction from the trained {st.session_state.model_name} model.")
                input_data = {}
                for col in st.session_state.selected_feature_cols:
                    if col in st.session_state.num_cols:
                        input_data[col] = st.number_input(f"Enter value for {col}", value=0.0, key=f"input_{col}")
                    else:
                        unique_vals = st.session_state.cleaned_df[col].dropna().unique().tolist()
                        input_data[col] = st.selectbox(f"Select value for {col}", unique_vals, key=f"input_{col}")
                if st.button("Predict"):
                    try:
                        input_df = pd.DataFrame([input_data], columns=st.session_state.selected_feature_cols)
                        prediction = st.session_state.model_pipeline.predict(input_df)[0]
                        if st.session_state.task == 'classification' and pd.api.types.is_object_dtype(st.session_state.cleaned_df[st.session_state.selected_target_col]):
                            unique_labels = st.session_state.cleaned_df[st.session_state.selected_target_col].astype('category').cat.categories
                            pred_label = unique_labels[int(prediction)]
                            st.success(f"Prediction from {st.session_state.model_name}: **{pred_label}**")
                        else:
                            st.success(f"Prediction from {st.session_state.model_name}: **{prediction:.2f}**")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            elif prediction_mode == "Upload CSV":
                st.write(f"Upload a new CSV file with the same feature columns to get batch predictions from the trained {st.session_state.model_name} model.")
                new_uploaded_file = st.file_uploader("Upload new CSV for predictions", type=["csv"], key="new_pred_file")
                if new_uploaded_file is not None:
                    try:
                        new_df = pd.read_csv(new_uploaded_file)
                        missing_cols = set(st.session_state.selected_feature_cols) - set(new_df.columns)
                        if missing_cols:
                            st.error(f"Missing required feature columns in uploaded CSV: {missing_cols}")
                            st.stop()
                        predictions = st.session_state.model_pipeline.predict(new_df[st.session_state.selected_feature_cols])
                        if st.session_state.task == 'classification' and pd.api.types.is_object_dtype(st.session_state.cleaned_df[st.session_state.selected_target_col]):
                            unique_labels = st.session_state.cleaned_df[st.session_state.selected_target_col].astype('category').cat.categories
                            predictions = [unique_labels[int(pred)] for pred in predictions]
                        new_df['Predicted'] = predictions
                        st.subheader("Predictions on Uploaded Data")
                        st.dataframe(new_df)
                        csv = new_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Error processing uploaded CSV: {str(e)}")
else:
    st.info("Please upload a CSV dataset to begin analysis.")

# Footer
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
<div style='text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f0f0f0; border-radius: 10px;'>
    <div style='font-size: 1.2rem; color: #000000; font-weight: bold;'>✨Sankatos APP✨</div>
    <div style='font-size: 0.9rem; color: #000000; margin-top: 0.2rem;'>Owner/Creator: Mohammed Zakyi</div>
    <div style='font-size: 0.9rem; color: #333; margin-top: 0.5rem;'>
        <div style='margin-bottom: 0.3rem;'>
            <i class="fab fa-linkedin" style='margin-right: 8px;'></i>
            <a href="https://www.linkedin.com/in/mohammed-zakyi-399b2114a/" target="_blank">LinkedIn Profile</a>
        </div>
        <div style='margin-bottom: 0.3rem;'>
            <i class="fas fa-envelope" style='margin-right: 8px;'></i>
            <a href="mailto:mzakyi06240@ucumberlands.edu">mzakyi06240@ucumberlands.edu</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
