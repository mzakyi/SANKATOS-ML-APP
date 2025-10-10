# Standard Library Imports
import base64
import datetime
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

# Third-Party Utility Imports
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from ydata_profiling import ProfileReport

# ReportLab Imports (PDF generation)
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table as RLTable, TableStyle

# Machine Learning & Imblearn Imports
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    mean_squared_error, r2_score
)
from sklearn.model_selection import (
    GridSearchCV, train_test_split
)
from sklearn.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor
)
from xgboost import XGBClassifier, XGBRegressor



# ============= AUTHENTICATION SYSTEM WITH ADMIN APPROVAL =============
USERS_FILE = "users.json"
ADMIN_USERNAME = "Mzakyi"
ADMIN_PASSWORD = "Neymar123!@#"

def initialize_users_file():
    """Initialize users file with admin account if it doesn't exist"""
    if not os.path.exists(USERS_FILE) or os.path.getsize(USERS_FILE) == 0:
        admin_user = {
            ADMIN_USERNAME: {
                "password": hash_password(ADMIN_PASSWORD),
                "approved": True,
                "is_admin": True,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        save_users(admin_user)

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE) and os.path.getsize(USERS_FILE) > 0:
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    """Hash password for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def admin_dashboard():
    """Admin dashboard to manage user approvals"""
    st.title("👨‍💼 Admin Dashboard")
    
    users = load_users()
    
    # Statistics
    non_admin_users = {u: data for u, data in users.items() if not data.get("is_admin", False)}
    total_users = len(non_admin_users) 
    pending_users = sum(1 for data in non_admin_users.values() if not data.get("approved", False))
    approved_users = total_users - pending_users
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", total_users)
    with col2:
        st.metric("Pending Approval", pending_users)
    with col3:
        st.metric("Approved Users", approved_users)
    
    st.markdown("---")
    
    # Pending Approvals
    st.subheader("🕐 Pending User Approvals")
    pending_list = [(u, data) for u, data in users.items() if not data.get("approved", False) and not data.get("is_admin", False)]
    
    if pending_list:
        for username, user_data in pending_list:
            with st.expander(f"👤 {username} - Registered on {user_data.get('created_at', 'Unknown')}"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Approve", key=f"approve_{username}", use_container_width=True):
                        users[username]["approved"] = True
                        save_users(users)
                        st.success(f"User {username} approved!")
                        st.rerun()
                with col2:
                    if st.button(f"❌ Reject", key=f"reject_{username}", use_container_width=True):
                        del users[username]
                        save_users(users)
                        st.warning(f"User {username} rejected and removed.")
                        st.rerun()
    else:
        st.info("No pending approvals")
    
    st.markdown("---")
    
    # Approved Users List
    st.subheader("✅ Approved Users")
    approved_list = [(u, data) for u, data in users.items() if data.get("approved", False) and not data.get("is_admin", False)]
    
    if approved_list:
        for username, user_data in approved_list:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"👤 **{username}** - Joined: {user_data.get('created_at', 'Unknown')}")
            with col2:
                if st.button(f"🗑️ Remove", key=f"remove_{username}"):
                    del users[username]
                    save_users(users)
                    st.warning(f"User {username} removed.")
                    st.rerun()
    else:
        st.info("No approved users yet")

def authenticate():
    """Handle login and signup with approval system"""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    if not st.session_state["logged_in"]:
        # Initialize users file with admin
        initialize_users_file()
        
        # Show logo at top
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" width="150" height="150">
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
        
        st.title("🔐 Sankatos Data Analyzer & ML Predictor")
        st.markdown("### Please login or create an account to continue")
        
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", type="primary", use_container_width=True):
                users = load_users()
                hashed_pw = hash_password(password)
                
                if username in users and users[username]["password"] == hashed_pw:
                    # Check if user is approved
                    if users[username].get("approved", False):
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.session_state["is_admin"] = users[username].get("is_admin", False)
                        
                        # === NEW: Set initial view based on role ===
                        if st.session_state["is_admin"]:
                            st.session_state["view"] = 'admin_dashboard' # Admin lands on dashboard
                        else:
                            st.session_state["view"] = 'main_app' # Regular user lands on main app
                        # ==========================================

                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.warning("⏳ Your account is pending approval. Please wait for admin approval.")
                else:
                    st.error("❌ Invalid username or password")
        
        with tab2:
            st.subheader("Create New Account")
            st.info("ℹ️ After signing up, your account will need to be approved by an administrator before you can access the app.")
            
            new_username = st.text_input("Choose Username", key="signup_user")
            new_password = st.text_input("Choose Password", type="password", key="signup_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_pass")
            
            if st.button("Sign Up", type="primary", use_container_width=True):
                users = load_users()
                
                # Validation
                if not new_username or not new_password:
                    st.error("❌ Please fill in all fields")
                elif new_username in users:
                    st.error("❌ Username already exists. Please choose another.")
                elif new_password != confirm_password:
                    st.error("❌ Passwords don't match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    # Create account (pending approval)
                    users[new_username] = {
                        "password": hash_password(new_password),
                        "approved": False,
                        "is_admin": False,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_users(users)
                    st.success("✅ Account created successfully! Your account is pending admin approval. You'll be able to login once approved.")
        
        # Footer for auth page
        st.markdown("""
        <div style='text-align: center; margin-top: 3rem; padding: 1rem; color: #666;'>
            <p>Created by Mohammed Zakyi | Data Scientist</p>
        </div>
        """, unsafe_allow_html=True)
        
        return False
    
    return True
# ============= END AUTHENTICATION SYSTEM =============


# --- Helper Functions (Existing) ---
def initialize_session_state():
    """Initializes all necessary session state variables."""
    if 'last_data_source' not in st.session_state:
        st.session_state.last_data_source = None
        
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
        st.session_state.history = []
        st.session_state.model_pipeline = None
        st.session_state.model_name = None
        st.session_state.task = None
        st.session_state.selected_feature_cols = []
        st.session_state.num_cols = []
        st.session_state.cat_cols = []
        st.session_state.selected_target_col = None
        st.session_state.selected_insight_options = []
        st.session_state.selected_clean_options = []
        st.session_state.selected_change_types = {} 
        st.session_state.selected_delete_cols = []
        st.session_state.rename_cols = {}
        st.session_state.new_columns_definitions = [] 
        st.session_state.selected_viz_cols = []
        st.session_state.selected_viz_type = "Choose plot type"
        st.session_state.best_params = {}
        st.session_state.target_col = None
        st.session_state.feature_cols = []
        st.session_state.remove_duplicates = False 
        st.session_state.coerce_errors = False 
        st.session_state.initial_report_html = None
        st.session_state.cleaned_report_html = None
        st.session_state.cleaning_iteration = 0 

    # === NEW: View State ===
    if 'view' not in st.session_state:
        st.session_state.view = 'main_app' 
    # =======================

def reset_session_state(df, data_source_key):
    """Reresets all data-dependent session state variables upon new data load."""
    st.session_state.cleaned_df = df
    st.session_state.history = [st.session_state.cleaned_df.copy()]
    st.session_state.last_data_source = data_source_key
    
    # ML/Prediction Reset
    st.session_state.model_pipeline = None
    st.session_state.model_name = None
    st.session_state.task = None
    st.session_state.selected_feature_cols = []
    st.session_state.selected_target_col = None
    st.session_state.best_params = {}
    st.session_state.target_col = None
    st.session_state.feature_cols = []

    # Column Type Reset
    st.session_state.num_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.session_state.cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Cleaning/Insight Reset
    st.session_state.selected_insight_options = []
    st.session_state.selected_clean_options = []
    st.session_state.selected_change_types = {}
    st.session_state.selected_delete_cols = [] 
    st.session_state.rename_cols = {}
    st.session_state.new_columns_definitions = []
    st.session_state.selected_viz_cols = []
    st.session_state.selected_viz_type = "Choose plot type"
    st.session_state.remove_duplicates = False
    st.session_state.coerce_errors = False
    
    # Report Reset
    st.session_state.initial_report_html = None
    st.session_state.cleaned_report_html = None
    st.session_state.cleaning_iteration = 0 
def load_data(uploaded_file):
    """Loads data from a CSV file."""
    try:
        if uploaded_file.size > 500 * 1024 * 1024:
            st.error("File size exceeds 500MB limit.")
            return None
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

def load_data_from_db(db_type, host, port, user, password, database, sql_query):
    """Mocks loading data from a database connection."""
    st.info(f"Attempting to connect to {db_type} at {host}:{port}...")
    # --- MOCK CONNECTION LOGIC ---
    data = {
        'id': range(1, 101),
        'Age': np.random.randint(20, 60, 100),
        'Income_USD': np.random.randint(30000, 150000, 100),
        'City': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 100),
        'Target_Category': np.random.choice(['High', 'Low'], 100, p=[0.7, 0.3]),
        'Score': np.random.uniform(0.5, 9.9, 100).round(2),
        'JoinDate': pd.to_datetime(pd.date_range(start='2020-01-01', periods=100, freq='D')),
    }
    df = pd.DataFrame(data)
    
    # Add some missing values for testing cleaning features
    df.loc[df.sample(frac=0.05).index, 'Income_USD'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'City'] = np.nan
    
    # Add duplicates for testing
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    # Add a data error for testing coercion
    df.loc[5, 'Income_USD'] = 'ERROR_TEXT'
    
    return df
    # --- END MOCK CONNECTION LOGIC ---

# === NEW: Sidebar Navigation Function ===




def navigation_sidebar():
    """Handles the sidebar navigation for logged-in users."""
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.get('username', 'User')}!")
        st.markdown("---")

        # Admin navigation options
        if st.session_state.get("is_admin"):
            st.subheader("Navigation")
            
            if st.session_state.view == 'admin_dashboard':
                if st.button("📊 Go to Main Data App", use_container_width=True, type='primary', key="switch_to_main"):
                    st.session_state.view = 'main_app'
                    st.rerun()
            elif st.session_state.view == 'main_app':
                if st.button("👨‍💼 Go to Admin Dashboard", use_container_width=True, type='primary', key="switch_to_admin"):
                    st.session_state.view = 'admin_dashboard'
                    st.rerun()
            st.markdown("---")
        
        # Logout option
        if st.button("🔒 Logout", use_container_width=True, key="logout_btn_sidebar"):
            for key in ["logged_in", "username", "is_admin", "view", "cleaned_df", "history"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
# ========================================


# ----------------- APP START -----------------
st.set_page_config(
    page_title="Sankatos Data Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize data-related session state BEFORE running the main app content
initialize_session_state()

# Wrap the entire main application content into a function
def run_main_app_content():
    
    st.title("Sankatos Data Analyzer & ML Predictor")
    st.subheader("Data Cleaning, Insights & ML Predictions App")
    st.markdown("""
    The Sankatos App is a powerful tool for data analysis and machine learning. 
    **Upload your CSV dataset or connect to a database** to clean and preprocess data, 
    generate detailed profile reports, explore insights with interactive visualizations, 
    and build predictive models with automated or manual ML options. 
    """)

    # --- Step 1: Data Source Selection ---
    st.subheader("Select Data Source")
    col1, col2 = st.columns(2)

    cleaning_key = st.session_state.cleaning_iteration

    with col1:
        data_source = st.radio(
            "Choose your data input method:",
            ('Upload CSV File', 'Connect to Database'),
            key='data_source_select',
            horizontal=True
        )

    if data_source == 'Upload CSV File':
        with col2:
            uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"], key="csv_uploader")

        if uploaded_file is not None:
            if st.session_state.last_data_source != uploaded_file.name:
                df_loaded = load_data(uploaded_file)
                if df_loaded is not None:
                    reset_session_state(df_loaded, uploaded_file.name)
                

    elif data_source == 'Connect to Database':
        st.markdown("---")
        with st.expander("Database Connection Details", expanded=True):
            db_type = st.selectbox("Database Type", ["PostgreSQL (Mock)", "MySQL (Mock)", "SQL Server (Mock)", "SQLite (Mock)"], key="db_type")
            host = st.text_input("Host", value="localhost", key="db_host")
            col_db1, col_db2 = st.columns(2)
            with col_db1:
                port = st.text_input("Port", value="5432", key="db_port")
                user = st.text_input("User", key="db_user")
            with col_db2:
                database = st.text_input("Database Name", key="db_name")
                password = st.text_input("Password", type="password", key="db_password")
                
            sql_query = st.text_area("SQL Query (Optional - defaults to mock data if empty)", "SELECT * FROM my_table LIMIT 1000", key="sql_query")

            if st.button("Connect and Load Data", key="db_load_button"):
                with st.spinner(f"Connecting to {db_type} and loading data..."):
                    try:
                        df_loaded = load_data_from_db(db_type, host, port, user, password, database, sql_query) 
                        
                        if df_loaded is not None:
                            reset_session_state(df_loaded, 'Database_Query_Loaded') 
                            st.success(f"Successfully connected to {db_type} and loaded data ({len(df_loaded)} rows of mock data).")
                            
                    except Exception as e:
                        st.error(f"Database connection or query failed (Mocked Error): {e}")

    # Check if data is available for processing
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df.copy() 

        st.markdown("---")
        # Current Dataset State
        st.subheader("Current Dataset State")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.write("Preview of the current dataset:")
        st.dataframe(df.head())

        # Undo Last Action Button
        if len(st.session_state.history) > 1:
            if st.button("Undo Last Action", key="undo_btn"):
                st.session_state.history.pop()
                st.session_state.cleaned_df = st.session_state.history[-1].copy()
                st.success("Last action undone. Dataset restored to previous state.")
                st.session_state.initial_report_html = None
                st.session_state.cleaned_report_html = None
                st.session_state.cleaning_iteration += 1 
                st.rerun()

        # --- Step 2: Dataset Profiling (Initial Only) ---
        st.header("2. Dataset Profiling")
        st.write("Generate a profile report to analyze the dataset before cleaning.")
        
        col_p1 = st.columns(1)[0]
        with col_p1:
            if st.button("Generate Initial Profile Report", key="profile_initial_btn"):
                with st.spinner("Generating initial profile report..."):
                    try:
                        profile = ProfileReport(st.session_state.history[0], title="Initial Dataset Profile Report", minimal=True)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                            profile.to_file(tmp_file.name)
                            with open(tmp_file.name, "r", encoding="utf-8") as f:
                                html_content = f.read()
                        
                        st.session_state.initial_report_html = html_content

                    except Exception as e:
                        st.error(f"Error generating initial profile report: {str(e)}")

        if st.session_state.initial_report_html:
            report_title = "Initial Dataset Profile Report"
            b64 = base64.b64encode(st.session_state.initial_report_html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="initial_profile_report.html">Download Initial Profile Report</a>'
            
            with st.expander(f"**{report_title}** (Click to View)", expanded=False):
                st.components.v1.html(st.session_state.initial_report_html, height=1000, scrolling=True)
                st.markdown(href, unsafe_allow_html=True)


        # --- Step 3: Data Preparation: Cleaning and Feature Engineering ---
        st.header("3. Data Preparation: Cleaning and Feature Engineering")
        st.info("Configure your data preparation steps below. Click 'Apply' once you are ready to process the changes.")

        # --- 3.1 Rename & Delete Columns ---
        with st.expander("3.1 Rename & Delete Columns (Schema Definition)", expanded=True):
            st.subheader("3.1: Rename & Delete Columns")
            
            # 3.1.1 Deletion
            st.markdown("##### 🗑️ Columns to Delete")
            current_cols_list = df.columns.tolist()
            safe_default_delete_cols = [
                col for col in st.session_state.selected_delete_cols 
                if col in current_cols_list
            ]
            
            st.session_state.selected_delete_cols = st.multiselect(
                "Select columns you wish to permanently remove from the dataset:",
                current_cols_list,
                default=safe_default_delete_cols,
                key=f"delete_cols_multiselect_{cleaning_key}"
            )
            
            # 3.1.2 Renaming
            st.markdown("##### ✏️ Rename Columns")
            st.write("Provide a new name for any column. The change will be applied only if the new name is different.")
            st.session_state.rename_cols = {}
            cols_to_rename = [col for col in df.columns if col not in st.session_state.selected_delete_cols]
            
            rename_cols = st.columns(3)
            for i, col in enumerate(cols_to_rename):
                with rename_cols[i % 3]:
                    new_name = st.text_input(f"Rename `{col}` to:", value=col, key=f"rename_{col}_clean_{cleaning_key}")
                    if new_name and new_name != col:
                        st.session_state.rename_cols[col] = new_name


        # --- 3.2 Handle Duplicates (NEW SECTION) ---
        with st.expander("3.2 Handle Duplicates", expanded=False):
            st.subheader("3.2: Remove Duplicate Rows")
            st.session_state.remove_duplicates = st.checkbox(
                "Check to remove all exact duplicate rows from the dataset.",
                value=st.session_state.remove_duplicates,
                key=f"remove_duplicates_checkbox_{cleaning_key}"
            )
            if st.session_state.remove_duplicates:
                st.info("Duplicate rows will be removed when you click 'Apply All Data Preparation Changes'.")


        # --- 3.3 Convert Column Data Types & Handle Errors ---
        with st.expander("3.3 Convert Data Types & Coerce Errors", expanded=False):
            st.subheader("3.3: Convert Column Data Types")
            st.session_state.coerce_errors = st.checkbox(
                "**Coerce Errors to NaN**: Convert non-numeric/non-date values to missing values (NaN) so they can be imputed in the next step.",
                value=st.session_state.coerce_errors,
                key=f"coerce_errors_checkbox_{cleaning_key}"
            )
            st.markdown("---")

            cols_to_convert = [col for col in df.columns.tolist() if col not in st.session_state.selected_delete_cols]
            st.session_state.selected_change_types = {
                k: v for k, v in st.session_state.selected_change_types.items() 
                if k in cols_to_convert
            }

            for col in cols_to_convert:
                current_type = str(df[col].dtype)
                possible_types = ["object", "int64", "float64", "datetime64[ns]", "bool"]
                
                saved_type = st.session_state.selected_change_types.get(col)
                default_label = saved_type if saved_type else "(Do not change - " + current_type + ")"
                
                type_options = ["(Do not change - " + current_type + ")"] + [t for t in possible_types if t != current_type]
                
                try:
                    default_index = type_options.index(default_label)
                except ValueError:
                    default_index = 0

                new_type_selection = st.selectbox(
                    f"Convert **`{col}`**:",
                    type_options,
                    index=default_index,
                    key=f"convert_{col}_{cleaning_key}"
                )
                if not new_type_selection.startswith("(Do not change"):
                    st.session_state.selected_change_types[col] = new_type_selection
                elif col in st.session_state.selected_change_types:
                    del st.session_state.selected_change_types[col]


        # --- 3.4 Handle Missing Values (Imputation/Deletion) ---
        with st.expander("3.4 Handle Missing Values (Imputation/Deletion)", expanded=False):
            st.subheader("3.4: Handle Missing Values")
            st.write("This step applies after type conversion, meaning any coerced errors will now show up as missing values here.")
            
            missing_counts = df.isnull().sum()
            st.session_state.missing_value_actions = []

            active_missing_cols = [col for col in df.columns.tolist() if df[col].isnull().any() and col not in st.session_state.selected_delete_cols]

            if active_missing_cols:
                st.info(f"Columns currently showing missing values (`NaN`): {', '.join(active_missing_cols)}")
                
                for col in active_missing_cols:
                    col_type = df[col].dtype
                    na_count = missing_counts[col]
                    st.markdown(f"**Column: `{col}`** ({col_type}) - {na_count} missing values")
                    
                    strategy_options = ["(Do nothing)", "Drop Rows", "Impute (Mean/Median/Mode)"]
                    strategy = st.radio(f"Select strategy for `{col}`:", strategy_options, key=f"strategy_{col}_{cleaning_key}", horizontal=True)
                    
                    if strategy == "Drop Rows":
                        st.session_state.missing_value_actions.append((col, 'drop_na', None))
                    elif strategy == "Impute (Mean/Median/Mode)":
                        is_numeric = pd.api.types.is_numeric_dtype(df[col]) or all(pd.to_numeric(df[col].dropna(), errors='coerce').notna())
                        
                        if is_numeric:
                            impute_method = st.selectbox(f"Numeric imputation method for `{col}`:", ["Mean", "Median"], key=f"impute_num_{col}_{cleaning_key}")
                            st.session_state.missing_value_actions.append((col, 'impute_num', impute_method))
                        else:
                            impute_method = st.selectbox(f"Categorical imputation method for `{col}`:", ["Mode", "Constant (e.g., 'Missing')"], key=f"impute_cat_{col}_{cleaning_key}")
                            st.session_state.missing_value_actions.append((col, 'impute_cat', impute_method))
            else:
                st.success("No missing values detected in the current dataset state.")


        # --- 3.5 Create New Features (Feature Engineering) ---
        with st.expander("3.5 Create New Features (Feature Engineering)", expanded=False):
            st.subheader("3.5: Create New Features / Add Column")
            
            # Display existing definitions
            if st.session_state.new_columns_definitions:
                st.markdown("##### Currently Defined Features")
                for i, definition in enumerate(st.session_state.new_columns_definitions):
                    col_key = f"new_col_{i}_{cleaning_key}" 
                    
                    if definition['type'] == 'conditional':
                        threshold_display = f"'{definition['threshold']}'" if not str(definition['threshold']).replace('.', '', 1).isdigit() else definition['threshold']
                        st.code(f"New Column: {definition['name']} | Logic: IF df['{definition['source_col']}'] {definition['condition']} {threshold_display} THEN '{definition['true_value']}' ELSE '{definition['false_value']}'")
                    else: # custom
                        st.code(f"New Column: {definition['name']} | Expression: {definition['expression']}")
                    
                    if st.button(f"Remove {definition['name']}", key=f"remove_new_col_{col_key}"):
                        st.session_state.new_columns_definitions.pop(i)
                        st.success(f"Feature {definition['name']} removed. Click 'Apply' to finalize.")
                        st.session_state.cleaning_iteration += 1 
                        st.rerun()

            st.markdown("---")
            st.markdown("##### Add New Feature")
            
            new_col_name = st.text_input("New Column Name", key=f"new_feature_name_input_{cleaning_key}")
            creation_type = st.radio(
                "Select Creation Method:",
                ['Conditional (If/Else)', 'Custom Python Expression'],
                key=f'creation_type_select_{cleaning_key}',
                horizontal=True
            )

            if creation_type == 'Conditional (If/Else)':
                col_cond1, col_cond2 = st.columns(2)
                with col_cond1:
                    available_cols = [col for col in df.columns if col not in st.session_state.selected_delete_cols]
                    source_col = st.selectbox("Source Column", available_cols, key=f"cond_source_col_{cleaning_key}")
                    condition = st.selectbox("Condition", ['>', '<', '==', '>=', '<=', '!='], key=f"cond_operator_{cleaning_key}")
                    threshold = st.text_input("Threshold Value (e.g., 100 or 'NYC')", key=f"cond_threshold_{cleaning_key}")
                
                with col_cond2:
                    true_value = st.text_input("Value if True", key=f"cond_true_val_{cleaning_key}")
                    false_value = st.text_input("Value if False", key=f"cond_false_val_{cleaning_key}")
                
                if st.button("Add Conditional Feature", key=f"add_conditional_btn_{cleaning_key}"):
                    if new_col_name and source_col and condition and threshold and true_value and false_value:
                        if new_col_name in df.columns:
                            st.error(f"Column '{new_col_name}' already exists. Please choose a different name.")
                        else:
                            definition = {
                                'name': new_col_name,
                                'type': 'conditional',
                                'source_col': source_col,
                                'condition': condition,
                                'threshold': threshold,
                                'true_value': true_value,
                                'false_value': false_value
                            }
                            st.session_state.new_columns_definitions.append(definition)
                            st.success(f"Conditional feature '{new_col_name}' defined. Click 'Apply' to execute.")
                            st.rerun()
                    else:
                        st.warning("Please fill all fields for the conditional feature.")

            elif creation_type == 'Custom Python Expression':
                expression = st.text_area(
                    "Python Expression (e.g., `np.log(df['colA']) * 100`)", 
                    key=f"custom_expression_input_{cleaning_key}"
                )
                st.markdown("*(You must reference the current DataFrame as `df`)*")

                if st.button("Add Custom Feature", key=f"add_custom_btn_{cleaning_key}"):
                    if new_col_name and expression:
                        if new_col_name in df.columns:
                            st.error(f"Column '{new_col_name}' already exists. Please choose a different name.")
                        else:
                            definition = {
                                'name': new_col_name,
                                'type': 'custom',
                                'expression': expression
                            }
                            st.session_state.new_columns_definitions.append(definition)
                            st.success(f"Custom feature '{new_col_name}' defined. Click 'Apply' to execute.")
                            st.rerun()
                    else:
                        st.warning("Please fill the name and expression for the custom feature.")


        # --- APPLY BUTTON ---
        if st.button("✅ Apply All Data Preparation Changes", key=f"apply_cleaning_btn_{cleaning_key}"):
            
            new_df = df.copy() 
            
            try:
                st.markdown("### ⚙️ Executing Data Preparation Steps...")

                # 1. Rename/Delete Columns
                st.markdown("##### 1. Applying Column Renaming and Deletion...")
                if st.session_state.selected_delete_cols:
                    new_df.drop(columns=st.session_state.selected_delete_cols, inplace=True, errors='ignore')
                    st.write(f"  - Deleted columns: {', '.join(st.session_state.selected_delete_cols)}")

                if st.session_state.rename_cols:
                    valid_renames = {k: v for k, v in st.session_state.rename_cols.items() if k in new_df.columns}
                    new_df.rename(columns=valid_renames, inplace=True)
                    st.write(f"  - Renamed columns: {valid_renames}")

                # 2. Handle Duplicates
                st.markdown("##### 2. Handling Duplicates...")
                if st.session_state.remove_duplicates:
                    initial_rows = len(new_df)
                    new_df.drop_duplicates(inplace=True)
                    st.write(f"  - Removed {initial_rows - len(new_df)} duplicate rows.")
                else:
                    st.write(f"  - Skipped duplicate removal.")


               # 3. Convert Data Types & Coerce Errors
                st.markdown("##### 3. Applying Data Type Conversions & Error Coercion...")
                for col, new_type in st.session_state.selected_change_types.items():
                    if col in new_df.columns:
                        
                        if new_type == "datetime64[ns]":
                            errors_param = 'coerce' if st.session_state.coerce_errors else 'ignore'
                            new_df[col] = pd.to_datetime(new_df[col], errors=errors_param)
                            st.write(f"  - Converted `{col}` to {new_type} (Errors handled: {errors_param}).")
                        
                        elif new_type == "bool":
                            new_df[col] = new_df[col].astype(bool)
                            st.write(f"  - Converted `{col}` to {new_type}.")
                        
                        elif new_type in ["float64", "int64"]:
                            # Use pd.to_numeric for numeric conversions (supports 'coerce')
                            if st.session_state.coerce_errors:
                                new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                                if new_type == "int64":
                                    # Convert to int, filling NaN values with 0 first
                                    new_df[col] = new_df[col].fillna(0).astype('int64')
                                st.write(f"  - Converted `{col}` to {new_type} (Errors handled: coerce).")
                            else:
                                new_df[col] = pd.to_numeric(new_df[col], errors='ignore')
                                st.write(f"  - Converted `{col}` to {new_type} (Errors handled: ignore).")
                        
                        else:
                            # For other types, use astype with 'ignore' only
                            new_df[col] = new_df[col].astype(new_type, errors='ignore')
                            st.write(f"  - Converted `{col}` to {new_type}.")

                if st.session_state.coerce_errors:
                    st.info("  - **Note:** Any data errors converted to `NaN` will be addressed in the next step (Handling Missing Values).")

                # 4. Handle Missing Values
                st.markdown("##### 4. Applying Missing Value Handling...")
                for col, operation, method in st.session_state.missing_value_actions:
                    if col not in new_df.columns: continue 
                    
                    if operation == 'drop_na':
                        initial_rows = len(new_df)
                        new_df.dropna(subset=[col], inplace=True)
                        st.write(f"  - Dropped {initial_rows - len(new_df)} rows where `{col}` was missing.")
                    elif operation == 'impute_num':
                        if method == 'Mean':
                            impute_val = new_df[col].mean()
                            new_df[col].fillna(impute_val, inplace=True)
                        else: # Median
                            impute_val = new_df[col].median()
                            new_df[col].fillna(impute_val, inplace=True)
                        st.write(f"  - Imputed `{col}` with its {method.lower()} ({impute_val:.2f}).")
                    elif operation == 'impute_cat':
                        if method == 'Mode':
                            impute_val = new_df[col].mode()[0]
                            new_df[col].fillna(impute_val, inplace=True)
                            st.write(f"  - Imputed `{col}` with its mode ({impute_val}).")
                        elif method.startswith('Constant'):
                            new_df[col].fillna("Missing", inplace=True)
                            st.write(f"  - Imputed `{col}` with constant value 'Missing'.")
                
                # 5. Create New Features
                st.markdown("##### 5. Applying New Feature Creation...")
                for definition in st.session_state.new_columns_definitions:
                    col_name = definition['name']
                    
                    try:
                        if definition['type'] == 'custom':
                            expression = definition['expression']
                            temp_df = new_df 
                            new_col_series = eval(expression, {'np': np, 'df': temp_df})
                            new_df[col_name] = new_col_series
                            st.write(f"  - Created custom column: `{col_name}` using expression.")
                        
                        elif definition['type'] == 'conditional':
                            source_col = definition['source_col']
                            condition = definition['condition']
                            threshold_str = definition['threshold']
                            true_value = definition['true_value']
                            false_value = definition['false_value']
                            
                            is_numeric = pd.api.types.is_numeric_dtype(new_df[source_col].dtype)
                            
                            if is_numeric:
                                try:
                                    threshold = float(threshold_str)
                                    condition_eval = f"(new_df['{source_col}'] {condition} {threshold})"
                                except ValueError:
                                    st.error(f"Error: Threshold '{threshold_str}' is not a valid number for numeric column '{source_col}'. Skipping feature '{col_name}'.")
                                    continue
                            else:
                                condition_eval = f"(new_df['{source_col}'] {condition} '{threshold_str}')"

                            condition_series = eval(condition_eval)
                            new_df[col_name] = np.where(condition_series, true_value, false_value)
                            
                            st.write(f"  - Created conditional column: `{col_name}` based on `{source_col}`.")
                            
                    except Exception as expr_e:
                        st.error(f"Error creating feature `{col_name}`: {expr_e}. Check your expression/condition.")
                        import traceback
                        st.code(traceback.format_exc())
                        
                # --- FINAL UPDATE STEP ---
                st.session_state.cleaned_df = new_df
                st.session_state.history.append(new_df.copy())
                st.session_state.num_cols = new_df.select_dtypes(include=np.number).columns.tolist()
                st.session_state.cat_cols = new_df.select_dtypes(include=['object', 'category']).columns.tolist()
                st.success("Cleaning and transformation applied successfully! Check the final data preview below.")
                
                st.session_state.cleaning_iteration += 1 
                
                st.session_state.cleaned_report_html = None 
                st.rerun() 
                
            except Exception as e:
                st.error(f"An error occurred during cleaning: {e}")
                import traceback
                st.code(traceback.format_exc())


        # --- Step 4: Download Cleaned Dataset & Profile ---
        st.header("4. Download Cleaned Dataset & Profile")
        st.markdown("Download the current state of your processed and cleaned data or generate a final profile report.")
        
        col_down_1, col_down_2 = st.columns(2)
        
        with col_down_1:
            if st.button("Generate Cleaned Dataset Profile Report", key="profile_cleaned_btn"):
                with st.spinner("Generating cleaned profile report..."):
                    try:
                        profile = ProfileReport(df, title="Cleaned Dataset Profile Report", minimal=True)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                            profile.to_file(tmp_file.name)
                            with open(tmp_file.name, "r", encoding="utf-8") as f:
                                html_content = f.read()
                        
                        st.session_state.cleaned_report_html = html_content

                    except Exception as e:
                        st.error(f"Error generating cleaned profile report: {str(e)}")

        if st.session_state.cleaned_report_html:
            report_title = "Cleaned Dataset Profile Report"
            b64 = base64.b64encode(st.session_state.cleaned_report_html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="cleaned_profile_report.html">Download Cleaned Profile Report</a>'
            
            with st.expander(f"**{report_title}** (Click to View)", expanded=False):
                st.components.v1.html(st.session_state.cleaned_report_html, height=1000, scrolling=True)
                st.markdown(href, unsafe_allow_html=True)
                
        with col_down_2:
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(df)
            
            st.download_button(
                label="Download Cleaned CSV",
                data=csv_data,
                file_name='cleaned_data.csv',
                mime='text/csv',
            )
        # --- Step 5: Interactive Data Visualization ---
        st.header("5. Interactive Data Visualization")
        st.write("Explore relationships and distributions in your dataset with Plotly.")

        col_viz_1, col_viz_2 = st.columns(2)
        with col_viz_1:
            st.session_state.selected_viz_type = st.selectbox(
                "Select Plot Type",
                ["Choose plot type", "Histogram", "Scatter Plot", "Box Plot", "Bar Chart", "Pairplot"],
                key="viz_type"
            )

        if st.session_state.selected_viz_type != "Choose plot type":
            
            all_cols = st.session_state.cleaned_df.columns.tolist()
            
            if st.session_state.selected_viz_type in ["Histogram", "Box Plot"]:
                with col_viz_2:
                    x_col = st.selectbox("Select Column", all_cols, key="viz_x")
                
                if st.session_state.selected_viz_type == "Histogram":
                    fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
                elif st.session_state.selected_viz_type == "Box Plot":
                    fig = px.box(df, y=x_col, title=f"Box Plot of {x_col}")
            
            elif st.session_state.selected_viz_type == "Scatter Plot":
                with col_viz_1:
                    x_col = st.selectbox("Select X-axis Column", all_cols, key="viz_scatter_x")
                with col_viz_2:
                    y_col = st.selectbox("Select Y-axis Column", [c for c in all_cols if c != x_col], key="viz_scatter_y")
                
                color_col = st.selectbox("Select Color/Group Column (Optional)", ["None"] + [c for c in all_cols if c not in [x_col, y_col]], key="viz_scatter_color")
                
                fig = px.scatter(
                    df, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col if color_col != "None" else None,
                    title=f"Scatter Plot: {y_col} vs {x_col}"
                )

            elif st.session_state.selected_viz_type == "Bar Chart":
                with col_viz_1:
                    x_col = st.selectbox("Select Category Column (X-axis)", st.session_state.cat_cols, key="viz_bar_x")
                with col_viz_2:
                    y_col = st.selectbox("Select Value Column (Y-axis - optional)", ["Count"] + st.session_state.num_cols, key="viz_bar_y")

                if y_col == "Count":
                    agg_df = df[x_col].value_counts().reset_index(name='Count')
                    fig = px.bar(
                        agg_df, 
                        x=x_col, 
                        y='Count', 
                        title=f"Count of {x_col}",
                    )
                else:
                    agg_func = st.radio("Aggregation Function", ["mean", "sum"], horizontal=True, key="bar_agg_func")
                    agg_df = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                    fig = px.bar(
                        agg_df, 
                        x=x_col, 
                        y=y_col, 
                        title=f"{agg_func.capitalize()} of {y_col} by {x_col}"
                    )
            
            elif st.session_state.selected_viz_type == "Pairplot":
                st.write("**Pairplot**: Shows relationships between all numeric columns")
                
                # Let user select which columns to include
                numeric_cols = st.session_state.cleaned_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns for a pairplot.")
                else:
                    selected_cols = st.multiselect(
                        "Select columns for pairplot (leave empty for all numeric columns)",
                        numeric_cols,
                        default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols,
                        key="pairplot_cols"
                    )
                    
                    color_col = st.selectbox(
                        "Select Color/Group Column (Optional)", 
                        ["None"] + all_cols, 
                        key="pairplot_color"
                    )
                    
                    if selected_cols and len(selected_cols) >= 2:
                        fig = px.scatter_matrix(
                            df,
                            dimensions=selected_cols,
                            color=color_col if color_col != "None" else None,
                            title="Pairplot of Selected Variables",
                            height=200 * len(selected_cols)  # Dynamic height based on number of columns
                        )
                        fig.update_traces(diagonal_visible=False, showupperhalf=False)
                        
                        # Make labels more readable
                        fig.update_layout(
                            font=dict(size=10),
                            margin=dict(t=50, l=100, r=100, b=100),
                        )
                    else:
                        st.warning("Please select at least 2 columns for the pairplot.")

            if 'fig' in locals():
                # Use different layout for pairplot vs other plots
                if st.session_state.selected_viz_type == "Pairplot":
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig.update_layout(height=500, margin=dict(t=50, l=10, r=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

        # --- Step 7: Machine Learning Predictions ---
        st.header("7. Machine Learning Predictions")

        if len(st.session_state.cleaned_df) < 10:
            st.warning("Dataset is too small for meaningful ML training. Need at least 10 rows.")
        else:
            def update_target_col():
                st.session_state.selected_target_col = st.session_state.target_col
            
            target_col = st.selectbox(
                "Select target column for prediction",
                st.session_state.cleaned_df.columns,
                index=st.session_state.cleaned_df.columns.get_loc(st.session_state.selected_target_col) if st.session_state.selected_target_col in st.session_state.cleaned_df.columns and st.session_state.selected_target_col else 0,
                key="target_col",
                on_change=update_target_col
            )
            
            if target_col:
                unique_vals = st.session_state.cleaned_df[target_col].nunique()
                task = 'classification' if pd.api.types.is_object_dtype(st.session_state.cleaned_df[target_col]) or unique_vals <= 20 else 'regression'
                st.write(f"Detected task: **{task.capitalize()}** (based on target type and unique value count: {unique_vals})")
                
                def update_feature_cols():
                    st.session_state.selected_feature_cols = st.session_state.feature_cols
                
                feature_cols = st.multiselect(
                    "Select feature columns",
                    [col for col in st.session_state.cleaned_df.columns if col != target_col],
                    default=[col for col in st.session_state.selected_feature_cols if col in st.session_state.cleaned_df.columns and col != target_col],
                    key="feature_cols",
                    on_change=update_feature_cols
                )
                
                model_mode = st.radio("Model Selection Mode", ["Manual (select one)", "Auto-select best", "Compare top two"], key="model_mode", horizontal=True)
                selected_model = None
                if model_mode == "Manual (select one)":
                    model_options = ['RandomForest', 'XGBoost', 'LogisticRegression', 'SVM'] if task == 'classification' else ['RandomForest', 'XGBoost', 'LinearRegression', 'SVM']
                    selected_model = st.selectbox("Select ML model", model_options, key="selected_model")
                
                col_tune1, col_tune2 = st.columns(2)
                with col_tune1:
                    use_tuning = st.checkbox("Use Hyperparameter Tuning (slower but potentially better results)", value=True)
                with col_tune2:
                    use_smote = st.checkbox("Handle imbalanced classes with SMOTE (Requires 'imblearn' package)", disabled=True) if task == 'classification' else False
                
                if feature_cols and st.button("Run ML Model"):
                    with st.spinner("Training model(s) with hyperparameter tuning..." if use_tuning else "Training model(s)..."):
                        try:
                            X = st.session_state.cleaned_df[feature_cols].copy()
                            y = st.session_state.cleaned_df[target_col].copy()
                            
                            if task == 'classification' and pd.api.types.is_object_dtype(y):
                                y = y.astype('category').cat.codes
                            
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            # Clean categorical columns - convert mixed types to strings and handle NaN
                            cat_cols_to_check = X.select_dtypes(include=['object', 'category']).columns.tolist()
                            for col in cat_cols_to_check:
                                # Convert all values to strings and replace NaN with 'missing'
                                X_train[col] = X_train[col].fillna('missing').astype(str)
                                X_test[col] = X_test[col].fillna('missing').astype(str)

                            if X_train.isna().any().any() or X_test.isna().any().any():
                                st.error("Input data contains missing values. Please go back to Step 3 and clean the data first.")
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
                                
                            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
                            
                            models = {
                                'RandomForest': RandomForestClassifier(random_state=42) if task == 'classification' else RandomForestRegressor(random_state=42),
                                'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss') if task == 'classification' else XGBRegressor(random_state=42),
                                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000) if task == 'classification' else LinearRegression(),
                                'SVM': SVC(random_state=42, probability=True) if task == 'classification' else SVR()
                            }
                            
                            if task == 'classification':
                                param_grids = {
                                    'RandomForest': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20]},
                                    'XGBoost': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.1, 0.3]},
                                    'LogisticRegression': {'model__C': [1, 10], 'model__penalty': ['l2']},
                                    'SVM': {'model__C': [1, 10], 'model__kernel': ['rbf']}
                                }
                                scoring_metric = 'accuracy'
                            else:  # regression
                                param_grids = {
                                    'RandomForest': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20]},
                                    'XGBoost': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.1, 0.3]},
                                    'LinearRegression': {},
                                    'SVM': {'model__C': [1, 10], 'model__kernel': ['rbf']}
                                }
                                scoring_metric = 'r2'
                            
                            def build_pipeline(model):
                                return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                            
                            def train_with_tuning(name, model, param_grid):
                                pipeline = build_pipeline(model)
                                
                                if use_tuning and param_grid:
                                    grid_search = GridSearchCV(
                                        pipeline,
                                        param_grid,
                                        cv=5,
                                        scoring=scoring_metric,
                                        n_jobs=-1,
                                        verbose=0
                                    )
                                    grid_search.fit(X_train, y_train)
                                    best_pipeline = grid_search.best_estimator_
                                    best_params = grid_search.best_params_
                                    return best_pipeline, best_params
                                else:
                                    pipeline.fit(X_train, y_train)
                                    return pipeline, {}
                            
                            results = {}
                            if model_mode != "Manual (select one)":
                                for name, model in models.items():
                                    st.info(f"Training {name}...")
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
                                metric_name = "Accuracy" if task == 'classification' else "R² Score"
                                metric_value = accuracy_score(y_test, y_pred) if task == 'classification' else r2_score(y_test, y_pred)
                                
                                st.subheader(f"Model: {name} ({metric_name}: {metric_value:.4f})")
                                
                                if best_params:
                                    st.write("**Best Hyperparameters:**")
                                    params_display = {k.replace('model__', ''): v for k, v in best_params.items()}
                                    st.json(params_display)
                                
                                if task == 'classification':
                                    st.text("Classification Report:")
                                    st.text(classification_report(y_test, y_pred))
                                else:
                                    mse = mean_squared_error(y_test, y_pred)
                                    r2 = r2_score(y_test, y_pred)
                                    metrics_text = f"Mean Squared Error: {mse:.4f}\nR² Score: {r2:.4f}"
                                    st.text(metrics_text)

                                st.subheader("Actual vs. Predicted")
                                pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index()
                                pred_df.rename(columns={'index': 'Index'}, inplace=True)
                                st.dataframe(pred_df, hide_index=False)
                                
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
                                
                                if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                                    st.subheader("Feature Importances")
                                    importances = pipeline.named_steps['model'].feature_importances_
                                    
                                    preprocessor = pipeline.named_steps['preprocessor']
                                    
                                    feature_names = num_cols
                                    
                                    if 'cat' in preprocessor.named_transformers_:
                                        ohe_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
                                        feature_names.extend(ohe_features)
                                    
                                    if len(importances) == len(feature_names):
                                        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)
                                        st.dataframe(imp_df)
                                        
                                        fig_imp = px.bar(
                                            imp_df.head(10), 
                                            x='Importance', 
                                            y='Feature', 
                                            orientation='h',
                                            title='Top 10 Feature Importances'
                                        )
                                        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                                        st.plotly_chart(fig_imp)
                                    else:
                                        st.warning("Could not map feature importances to preprocessed feature names.")
                            

                            if model_mode == "Manual (select one)":
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
                                st.success(f"🏆 Best Model: **{best_name}**")
                                display_model_results(best_name, best_result['pipeline'], best_result['y_pred'], best_result['best_params'])
                                st.session_state.model_pipeline = best_result['pipeline']
                                st.session_state.model_name = best_name
                                st.session_state.task = task
                                st.session_state.best_params = best_result['best_params']
                                
                            elif model_mode == "Compare top two":
                                st.subheader("Comparison of Top 2 Models")
                                col1, col2 = st.columns(2)
                                for i, (name, result) in enumerate(sorted_results[:2]):
                                    with [col1, col2][i]:
                                        rank = "🥇 Best" if i == 0 else "🥈 Second Best"
                                        st.info(rank)
                                        display_model_results(name, result['pipeline'], result['y_pred'], result['best_params'])
                                
                                best_name, best_result = sorted_results[0]
                                st.session_state.model_pipeline = best_result['pipeline']
                                st.session_state.model_name = best_name
                                st.session_state.task = task
                                st.session_state.best_params = best_result['best_params']
                            
                            st.balloons()
                            st.success(f"✅ Model training complete! The **{st.session_state.model_name}** model has been saved and can now be used to make new predictions.")
                            
                        except Exception as e:
                            st.error(f"Error running model: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

                # --- Step 8: Prediction Input Section ---
                if st.session_state.model_pipeline is not None and st.session_state.selected_feature_cols:
                    st.markdown("---")
                    st.header("8. Make New Predictions")
                    st.info(f"Using trained model: **{st.session_state.model_name}** ({st.session_state.task.capitalize()})")

                    prediction_mode = st.radio("Prediction Method", ["Manual Input", "Upload CSV"], key="prediction_mode", horizontal=True)

                    if prediction_mode == "Manual Input":
                        st.write(f"Enter values for the selected features ({', '.join(st.session_state.selected_feature_cols)}) to get a prediction.")
                        input_data = {}
                        
                        cols = st.columns(min(3, len(st.session_state.selected_feature_cols)))
                        
                        for i, col in enumerate(st.session_state.selected_feature_cols):
                            with cols[i % len(cols)]:
                                if col in st.session_state.num_cols:
                                    default_val = st.session_state.cleaned_df[col].mean() if col in st.session_state.cleaned_df.columns else 0.0
                                    input_data[col] = st.number_input(f"Enter {col} (Numeric)", value=float(default_val), key=f"input_{col}")
                                else: 
                                    unique_vals = st.session_state.cleaned_df[col].dropna().unique().tolist()
                                    if unique_vals:
                                        unique_vals = [str(v) for v in unique_vals] 
                                        input_data[col] = st.selectbox(f"Select {col} (Category)", unique_vals, key=f"input_{col}")
                                    else:
                                        input_data[col] = st.text_input(f"Enter {col} (Category)", key=f"input_{col}_text")
                        
                        if st.button("Predict Single Input"):
                            try:
                                input_df = pd.DataFrame([input_data], columns=st.session_state.selected_feature_cols)
                                prediction = st.session_state.model_pipeline.predict(input_df)[0]
                                
                                if st.session_state.task == 'classification' and pd.api.types.is_object_dtype(st.session_state.cleaned_df[st.session_state.selected_target_col]):
                                    unique_labels = st.session_state.cleaned_df[st.session_state.selected_target_col].astype('category').cat.categories
                                    try:
                                        pred_label = unique_labels[int(prediction)]
                                    except IndexError:
                                        pred_label = f"Unknown Label (Code: {int(prediction)})"
                                    st.success(f"Prediction from {st.session_state.model_name}: **{pred_label}**")
                                else:
                                    st.success(f"Prediction from {st.session_state.model_name}: **{prediction:.4f}**")
                            except Exception as e:
                                st.error(f"Error making prediction: {str(e)}")

                    elif prediction_mode == "Upload CSV":
                        st.write(f"Upload a new CSV file containing the features: **{', '.join(st.session_state.selected_feature_cols)}**.")
                        new_uploaded_file = st.file_uploader("Upload new CSV for batch predictions", type=["csv"], key="new_pred_file")
                        
                        if new_uploaded_file is not None:
                            try:
                                new_df = pd.read_csv(new_uploaded_file)
                                missing_cols = set(st.session_state.selected_feature_cols) - set(new_df.columns)
                                if missing_cols:
                                    st.error(f"Missing required feature columns in uploaded CSV: {missing_cols}. Please upload a file with all required features.")
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

# ----------------- MAIN CONTROL FLOW -----------------
if authenticate():
    navigation_sidebar() # Always show the sidebar once authenticated
    
    # Control what view is displayed based on the 'view' session state variable
    if st.session_state.view == 'admin_dashboard' and st.session_state.get("is_admin"):
        admin_dashboard()
    else:
        # This executes if:
        # 1. The user is a regular user (view defaults to 'main_app')
        # 2. The user is an admin who has switched the view to 'main_app'
        run_main_app_content()




