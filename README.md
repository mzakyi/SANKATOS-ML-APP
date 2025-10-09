# Sankatos App

The Sankatos App is a powerful Streamlit-based tool for data analysis and machine learning. It allows users to upload CSV datasets, clean and preprocess data, generate detailed profile reports, explore insights with interactive visualizations, and build predictive models using automated or manual machine learning options with advanced hyperparameter tuning. Designed for data enthusiasts and professionals, the app provides a user-friendly interface to derive actionable insights from data.

## Features

- **Data Cleaning**: Remove duplicates, handle missing values (fill with mean/mode or drop), convert data types, and remove outliers using the IQR method.
- **Dataset Profiling**: Generate comprehensive profile reports using `ydata-profiling` to analyze datasets before and after cleaning.
- **Custom KPIs**: Define and calculate KPIs with standard aggregations (mean, sum, count, etc.) or custom formulas, with optional grouping and time-based filtering.
- **Interactive Visualizations**: Create line plots, scatter plots, histograms, box plots, pie charts, bar graphs, heatmaps, and more using Plotly and Seaborn.
- **Advanced Machine Learning with Hyperparameter Tuning**: 
  - Build classification or regression models with options for RandomForest, XGBoost, Logistic/Linear Regression, or SVM
  - **Automatic Hyperparameter Optimization**: Uses GridSearchCV with 5-fold cross-validation to find the best hyperparameters for each model, ensuring optimal performance
  - **Comprehensive Parameter Search**: Tests multiple combinations of parameters including:
    - Tree-based models: number of estimators, max depth, min samples split/leaf
    - XGBoost: learning rate, max depth, subsample ratio
    - Regularization models: C values, penalty types, solvers
    - SVM: kernel types, C values, gamma settings
  - **Best Parameters Display**: Shows the optimal hyperparameters found for each trained model
  - **Toggle Option**: Users can enable/disable hyperparameter tuning based on their needs (speed vs. accuracy)
  - Features automated model selection, SMOTE for imbalanced classes, and feature importance analysis
  - Model comparison tools to evaluate and select the best performing model
- **Prediction Capabilities**: Use trained models (with optimized hyperparameters) to make predictions on new data via manual input or CSV upload
- **Undo Functionality**: Revert data cleaning or preprocessing steps to maintain flexibility.
- **Export Options**: Download cleaned datasets, KPI results (CSV/PDF), and profile reports (HTML).
- **Branding**: Includes a custom logo and an about section with owner details, displayed in a professional UI with a footer containing contact information.

## Installation

To run the Sankatos App locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/sankatos-app.git
   cd sankatos-app
   ```

2. **Create and Activate a Virtual Environment**:
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     py -m venv venv
     venv\Scripts\activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   ```
   streamlit==1.39.0
   pandas==2.2.3
   seaborn==0.13.2
   matplotlib==3.9.2
   scikit-learn==1.5.2
   xgboost==2.1.1
   imbalanced-learn==0.12.3
   numpy==2.1.1
   plotly==5.24.1
   reportlab==4.2.2
   ydata-profiling==4.10.0
   ```

4. **Run the App**:
   ```bash
   streamlit run insights_app.py
   ```
   This opens the app in your default browser (e.g., `http://localhost:8501`).

## Usage

1. **Upload a Dataset**: Use the file uploader to load a CSV file.
2. **Clean Data**: Select cleaning options (e.g., remove duplicates, handle missing values) and apply them. Use the "Undo Last Action" button to revert changes.
3. **Profile Dataset**: Generate profile reports for the original or cleaned dataset to analyze data distributions and statistics.
4. **Explore Insights**: Choose visualization options (e.g., histograms, correlation heatmaps) to explore data patterns.
6. **Build ML Models**: 
   - Select a target column and features to train classification or regression models
   - Choose manual model selection, auto-select best, or compare top models
   - **Enable Hyperparameter Tuning** for optimal model performance (recommended for production models)
   - View best hyperparameters found for each model
   - Compare model performance metrics including accuracy, R² scores, confusion matrices, and feature importances
7. **Make Predictions**: Use the trained model (with optimized hyperparameters) to predict on new data via manual input or uploaded CSV.
8. **Export Results**: Download cleaned datasets, KPI reports, or profile reports as needed.

## Hyperparameter Tuning

The app includes advanced hyperparameter optimization to ensure the best possible model performance:

### How It Works
- **GridSearchCV**: Exhaustively searches through a predefined parameter grid for each model
- **Cross-Validation**: Uses 5-fold cross-validation to evaluate each parameter combination, ensuring robust performance estimates
- **Automatic Selection**: Identifies and uses the best hyperparameters based on validation scores

### Benefits
- **Improved Accuracy**: Typically achieves 5-20% better performance compared to default parameters
- **Optimized for Your Data**: Parameters are tuned specifically for your dataset
- **Production-Ready**: Models are optimized for real-world deployment

### Trade-offs
- **Speed**: Hyperparameter tuning takes longer (minutes instead of seconds) due to testing multiple parameter combinations
- **When to Use**: 
  - Enable for final production models and important predictions
  - Disable for quick exploratory analysis or when speed is critical

### Parameters Tuned
The app optimizes key parameters for each model type:
- **RandomForest**: Number of trees, tree depth, split criteria, leaf samples
- **XGBoost**: Learning rate, tree depth, subsample ratio, number of estimators
- **Logistic Regression**: Regularization strength (C), penalty type, solver
- **SVM**: Kernel type, C value, gamma settings

## Notes

- **Large Datasets**: For datasets with >10,000 rows or >50 columns, profiling may be slow. Use minimal mode (enabled by default) for efficiency.
- **Hyperparameter Tuning Performance**: With tuning enabled, model training may take several minutes depending on dataset size and selected models. Progress is displayed during training.
- **Private Repository**: This repository is private, accessible only to the owner and invited collaborators.
- **Deployment**: The app is deployed.
## Contact

- **Owner/Creator**: Mohammed Zakyi
- **LinkedIn**: [https://www.linkedin.com/in/mohammed-zakyi-399b2114a/](https://www.linkedin.com/in/mohammed-zakyi-399b2114a/)
- **Email**: [mzakyi06240@ucumberlands.edu](mailto:mzakyi06240@ucumberlands.edu)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
