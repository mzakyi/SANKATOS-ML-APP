# 🌟 Sankatos App: Comprehensive ML Model Builder and Data Analyzer
# Overview
The Sankatos App is a powerful, integrated tool built on Streamlit for end-to-end data analysis and machine learning. It provides a user-friendly environment for data enthusiasts and professionals to streamline the entire data pipeline—from connection and rigorous cleaning to advanced model building and prediction—all within a single web application.

# 🔐 Core Feature: Authentication & User Management
To ensure controlled access and collaboration, the application includes a full-featured authentication system:

* Secure Access: Users must Sign Up or Login to access the main data analysis features.

* Admin Approval: All new accounts are initially placed in a pending state and require approval from the administrator before gaining full access.

* Admin Dashboard: The designated administrator has a dedicated dashboard to view, approve, reject, or manage all registered users.

# Key Features
The application's capabilities are derived from the extensive suite of modern Python libraries used in its core logic:

# 📊 Data Analysis & Preparation
Flexible Data Input: Load data via CSV upload or mock Database connection.

# Rigorous Data Cleaning:

* Handle Duplicates: Dedicated function to easily remove all duplicate rows.

* Missing Values: Granular control to drop rows or fill nulls with Mean/Median (Numeric) or Mode/Constant (Categorical).

* Outlier Removal: Identify and remove extreme values using the Interquartile Range (IQR) method.

* Data Types: Convert column types and handle errors by coercing bad data into nulls for later imputation.

* Undo Functionality: Revert the last data cleaning or preprocessing step to maintain flexibility.

* Dataset Profiling: Generates comprehensive profile reports using ydata-profiling to analyze datasets before and after cleaning.


# 📈 Interactive Visualizations & Reporting
* Interactive Visualizations: Create rich, interactive charts and plots using Plotly Express, Seaborn, and Matplotlib for exploratory data analysis (EDA). Supports line plots, scatter plots, histograms, box plots, pie charts, bar graphs, heatmaps, and more.

* Export Options: Download cleaned datasets, KPI results (CSV/PDF), and profile reports (HTML).

# 🤖 Advanced Machine Learning Pipeline
* Model Variety: Supports a wide range of models for both Classification and Regression: Random Forests, XGBoost, Logistic/Linear Regression, SVM, K-Nearest Neighbors, and Decision Trees.

* Automatic Hyperparameter Optimization: Uses GridSearchCV with 5-fold cross-validation to find the best hyperparameters for each model, ensuring optimal performance (with a toggle option for speed vs. accuracy).

* Comprehensive Parameter Search: Tests multiple combinations for tree depth, learning rate, regularization strength, and kernel types specific to each model.

* Best Parameters Display: Clearly shows the optimal hyperparameters found.

* Model Selection: Features automated model selection, comparison tools to evaluate the top performing models, and feature importance analysis.

* Imbalanced Data Handling: Integrates SMOTE (Synthetic Minority Over-sampling Technique) via an imblearn Pipeline.

* Prediction Capabilities: Use trained models (with optimized hyperparameters) to make predictions on new data via manual input or CSV upload.

* Model Serialization: Uses joblib to save and load trained machine learning pipelines.

## Installation
## Prerequisites
## Python 3.8+

# Setup Instructions
Clone the repository and navigate to the directory:

* git clone [https://github.com/your-username/sankatos-app.git](https://github.com/your-username/sankatos-app.git)
cd sankatos-app

* Create and Activate a Virtual Environment (Recommended):

## On macOS/Linux:

* python3 -m venv venv
* source venv/bin/activate

$$ On Windows:

* py -m venv venv
* venv\Scripts\activate

## Install Dependencies:
The application relies on the following packages. You should create a requirements.txt file from this list:

* pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly ydata-profiling reportlab imbalanced-learn xgboost joblib

* (For exact versions used in development, refer to the detailed list below, though the command above is sufficient for most environments.)

$ Detailed requirements.txt:

streamlit
pandas
seaborn
matplotlib
scikit-learn
xgboost
imbalanced-learn
numpy
plotly
reportlab
ydata-profiling

## How to Run the Application
Once the dependencies are installed, you can start the Streamlit web application from your terminal:

streamlit run insights_app.py
This opens the app in your default browser (e.g., http://localhost:8501).

## Usage Workflow
* Login/Signup: Use the authentication panel. New sign-ups require Admin approval.

* Upload a Dataset: Use the file uploader to load a CSV file or connect to a mock database.

* Clean Data: Select cleaning options (e.g., remove duplicates, handle missing values, remove outliers) and apply them.

* Explore Insights: Choose visualization options (e.g., histograms, correlation heatmaps) to explore data patterns.

* Build ML Models: Select a target column, features, and model type. Enable Hyperparameter Tuning for optimal performance.

* Make Predictions: Use the trained model (with optimized hyperparameters) to predict on new data via manual input or uploaded CSV.

## Hyperparameter Tuning: A Deep Dive
The app includes advanced hyperparameter optimization to ensure the best possible model performance:

## How It Works
* GridSearchCV: Exhaustively searches through a predefined parameter grid for each model.

* Cross-Validation: Uses 5-fold cross-validation to evaluate each parameter combination, ensuring robust performance estimates.

* Automatic Selection: Identifies and uses the best hyperparameters based on validation scores.

## Benefits
* Improved Accuracy: Typically achieves better performance compared to default parameters.
* Optimized for Your Data: Parameters are tuned specifically for your dataset.
* Production-Ready: Models are optimized for real-world deployment.

## Trade-offs
* Speed: Hyperparameter tuning takes longer (minutes instead of seconds) due to testing multiple parameter combinations.

## Parameters Tuned
The app optimizes key parameters for each model type:

* RandomForest: Number of trees, tree depth, split criteria, leaf samples.

* XGBoost: Learning rate, tree depth, subsample ratio, number of estimators.

* Logistic Regression: Regularization strength (C), penalty type, solver.

* SVM: Kernel type, C value, gamma settings.

## Contact & Branding
The application features custom branding, including a unique logo and a detailed footer with contact information.

## Owner/Creator: Mohammed Zakyi

* LinkedIn: https://www.linkedin.com/in/mohammed-zakyi-399b2114a/
* Email: mzakyi06240@ucumberlands.edu

# License
This project is licensed under the MIT License. See the LICENSE file for details.
