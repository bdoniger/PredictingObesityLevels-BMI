from shiny import App, ui, render
import pandas as pd
import numpy as np
import json

# Load dataset
df = pd.read_csv("obesity_cleaned.csv")

# Add BMI column
df['BMI_kg_m2'] = round(df['Weight'] / (df['Height'] ** 2), 2)

# Rename height and weight columns
df = df.rename(columns={
    'Height': 'Height_meters',
    'Weight': 'Weight_kg'
})

target_col = "Obesity_Level"
predictor_cols = [c for c in df.columns if c != target_col]
numeric_predictors = [c for c in predictor_cols if pd.api.types.is_numeric_dtype(df[c])]

total_rows = df.shape[0]  

# ----- UI LAYOUT -----
app_ui = ui.page_fluid(  
    # Project title at the top, aligned with navbar tabs
    ui.div(
        ui.h1(
            "Obesity Level Prediction Dashboard",
            style="font-size: 28px; margin-top: 20px; margin-bottom: 20px; white-space: nowrap;"
        ),
        style="padding-left: 15px;"  # aligns with navbar tabs
    ),

    # Navbar with tabs
    ui.page_navbar(
        # README TAB
        ui.nav_panel(
            "README",
            ui.h2("Project Overview: Obesity Level Predictions from Nutritional and Physical Characteristics"),
            ui.markdown("""
                        
            ## Data Source
            The dataset used in this project is the Estimation of Obesity Levels based on Eating Habits and Physical Condition from the UCI Machine Learning Repository. It contains nutritional, physical, and behavioral features of individuals, along with a target variable representing their obesity level. The data was originally collected via an anonymous online survey, and after initial preprocessing and BMI calculation, synthetic data was generated to address class imbalance, resulting in a final, balanced dataset of 2,111 records. The official dataset article was consulted to understand each variable and its codebook definitions.
            
            The dataset was inspected and found to have no missing values. Column names were made descriptive, binary variables standardized to "Yes"/"No", and discrete features rounded and mapped to meaningful categories. Text inconsistencies were corrected, categorical features were converted to category dtype, and a BMI column was calculated by dividing weight by height squared. 
            

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

            ## Authors
            Natalie Seah, Erin Siedlecki, Emily Garman, Ben Doniger, Bela Barton
            """)
        ),

        # DATA TAB
        ui.nav_panel(
            "Data",
            ui.h3("Dataset Preview"),
            ui.p(f"Showing first 100 rows out of {total_rows} total rows."),
            ui.tags.div(
                ui.tags.table(id="datatable"),
                ui.tags.link(rel="stylesheet", href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"),
                ui.tags.script(src="https://code.jquery.com/jquery-3.7.0.min.js"),
                ui.tags.script(src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"),
                ui.tags.script(
                    f"""
                    $(document).ready(function() {{
                        $('#datatable').DataTable({{
                            data: {df.head(100).to_json(orient='records')},
                            columns: [
                                {', '.join([f"{{ data: '{col}', title: '{col}' }}" for col in df.columns])}
                            ],
                            pageLength: 25,
                            lengthMenu: [10, 25, 50, 100],
                            responsive: true,
                            scrollX: true,
                            dom: '<"top"f>rt<"bottom"lip><"clear">'
                        }});
                    }});
                    """
                )
            )
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
)


# ----- SERVER -----
def server(input, output, session):

    # Model tab (placeholder)
    @output
    @render.text
    def result():
        return "Run a model to see output here."

# ----- CREATE APP -----
app = App(app_ui, server)
