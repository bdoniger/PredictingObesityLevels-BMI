from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np

df = pd.read_csv("obesity.csv")

target_col = "Obesity_Level"
predictor_cols = [c for c in df.columns if c != target_col]
numeric_predictors = [c for c in predictor_cols if pd.api.types.is_numeric_dtype(df[c])]

# ----- UI LAYOUT -----
app_ui = ui.page_navbar(
    # README TAB
    ui.nav_panel(
        "README",
        ui.h2("Project Overview: Obesity Level Predictions from Nutritional and Physical Characteristics"),
        ui.markdown("""
        ## Data Source
        Obesity dataset with nutritional, physical, and behavioral features.

        ## Target Variable
        Obesity Level (categorical)

        ## Predictor Variables
        - Nutritional, behavioral, and physical characteristics (age, height, weight, calories, activity, etc.)

        ## Goals
        - Determine which factors affect obesity level
        - Build predictive models for obesity classification

        ## Machine Learning Pipeline Implemented by This App
        1. Model Selection: KNN, K-Means, Multinomial Logistic Regression, Multiple Linear Regression
        2. Model Training: Fit models on train/test split
        3. Model Evaluation: Metrics & plots
        4. Prediction: Enter new values for predictors to estimate obesity level
        
        ## App Usage
        1. See the Data preview tab
        2. Choose a model from the Model tab
        3. Adjust train/test split and other required thresholds
        4. Click 'Run Model' to train and evaluate a prediction
        5. Enter new values for predictors to predict probability using selected 
        """)
    ),

    # DATA TAB
    ui.nav_panel(
        "Data",
        ui.h3("Dataset Preview"),
        ui.output_data_frame("table")
    ),

    # MODELING TAB
    ui.nav_panel(
        "Modeling",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Choose a Model"),
                ui.input_select(
                    "model_choice",
                    "Model Type:",
                    {
                        "knn": "K-Nearest Neighbors",
                        "kmeans": "K-Means Clustering",
                        "mlogit": "Multinomial Logistic Regression",
                        "linreg": "Multiple Linear Regression"
                    }
                ),

                ui.input_slider("split", "Train/Test Split (%)", min=50, max=90, value=80),

                ui.h4("Predictor Inputs (for supervised models)"),
                *[
                    ui.input_numeric(f"pred_{col}", col, float(df[col].mean()))
                    for col in numeric_predictors
                ],

                ui.h4("K-Means Settings"),
                ui.input_numeric("k_clusters", "Number of Clusters", 3),

                ui.hr(),
                ui.input_action_button("run_model", "Run Model", class_="btn-primary")
            ),

            ui.card(
                ui.h3("Model Output"),
                ui.output_text("result")
            )
        )
    )
)

# ----- SERVER -----
def server(input, output, session):

    # Data tab
    @output
    @render.data_frame
    def table():
        return df.head(10)  # Show top 10 rows

    # Model tab (placeholder)
    @output
    @render.text
    def result():
        return "Run a model to see output here."

# ----- CREATE APP -----
app = App(app_ui, server)
