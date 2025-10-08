# Sankatos App

The Sankatos App is a powerful Streamlit-based tool for data analysis and machine learning. It allows users to upload CSV datasets, clean and preprocess data, generate detailed profile reports, create custom Key Performance Indicators (KPIs), explore insights with interactive visualizations, and build predictive models using automated or manual machine learning options. Designed for data enthusiasts and professionals, the app provides a user-friendly interface to derive actionable insights from data.

## Features

- **Data Cleaning**: Remove duplicates, handle missing values (fill with mean/mode or drop), convert data types, and remove outliers using the IQR method.
- **Dataset Profiling**: Generate comprehensive profile reports using `ydata-profiling` to analyze datasets before and after cleaning.
- **Custom KPIs**: Define and calculate KPIs with standard aggregations (mean, sum, count, etc.) or custom formulas, with optional grouping and time-based filtering.
- **Interactive Visualizations**: Create line plots, scatter plots, histograms, box plots, pie charts, bar graphs, heatmaps, and more using Plotly and Seaborn.
- **Machine Learning**: Build classification or regression models with options for RandomForest, XGBoost, Logistic/Linear Regression, or SVM. Features automated model selection, SMOTE for imbalanced classes, and feature importance analysis.
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
   streamlit run Perfect_code.py
   ```
   This opens the app in your default browser (e.g., `http://localhost:8501`).

## Usage

1. **Upload a Dataset**: Use the file uploader to load a CSV file.
2. **Clean Data**: Select cleaning options (e.g., remove duplicates, handle missing values) and apply them. Use the "Undo Last Action" button to revert changes.
3. **Profile Dataset**: Generate profile reports for the original or cleaned dataset to analyze data distributions and statistics.
4. **Explore Insights**: Choose visualization options (e.g., histograms, correlation heatmaps) to explore data patterns.
5. **Generate KPIs**: Define custom KPIs with aggregations or formulas, optionally grouped by columns or filtered by time periods. Visualize KPIs as bar, pie, or line charts.
6. **Build ML Models**: Select a target column and features to train classification or regression models. Choose manual model selection, auto-select best, or compare top models.
7. **Make Predictions**: Use the trained model to predict on new data via manual input or uploaded CSV.
8. **Export Results**: Download cleaned datasets, KPI reports, or profile reports as needed.

## Notes

- **Large Datasets**: For datasets with >10,000 rows or >50 columns, profiling may be slow. Use minimal mode (enabled by default) for efficiency.
- **Private Repository**: This repository is private, accessible only to the owner and invited collaborators.
- **Future Deployment**: The app is ready for deployment to platforms like Streamlit Cloud, which can connect to this private repository.

## Contact

- **Owner/Creator**: Mohammed Zakyi
- **LinkedIn**: [https://www.linkedin.com/in/mohammed-zakyi-399b2114a/](https://www.linkedin.com/in/your-profile)
- **Email**: [mzakyi06240@ucumberlands.edu](mailto:your.email@example.com)

Replace the placeholders above with your actual contact details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
