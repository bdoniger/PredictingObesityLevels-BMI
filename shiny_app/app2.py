from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("obesity_cleaned.csv")
df['BMI_kg_m2'] = round(df['Weight'] / (df['Height'] ** 2), 2)
df = df.rename(columns={'Height': 'Height_meters', 'Weight': 'Weight_kg'})

total_rows = df.shape[0]
target_col = "BMI_kg_m2"
predictor_cols = [c for c in df.columns if c != target_col]
numeric_predictors = [c for c in predictor_cols if pd.api.types.is_numeric_dtype(df[c])]

# ----- UI -----
app_ui = ui.page_fluid(
    ui.div(
        ui.h1(
            "Obesity Level Prediction Dashboard",
            style="font-size: 28px; margin-top: 20px; margin-bottom: 20px; white-space: nowrap;"
        ),
        style="padding-left: 15px;"
    ),
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
            BMI_kg_m2 (continuous)

            ## Predictor Variables
            - Nutritional, behavioral, and physical characteristics (age, height, weight, calories, activity, etc.)

            ## Goals
            - Determine which factors affect BMI
            - Build predictive models for BMI

            ## Machine Learning Pipeline Implemented by This App
            1. Model Selection: Multiple Linear Regression, Lasso Regression, Random Forest
            2. Model Training: Fit models on train/test split
            3. Model Evaluation: Metrics & plots
            4. Prediction: Enter new values for predictors to estimate BMI

            ## App Usage
            1. See the Data preview tab
            2. Choose a model from the Model tab
            3. Adjust train/test split
            4. Click 'Run Model' to train and evaluate a prediction
            5. Enter new values for predictors to predict BMI

            ## Authors
            Natalie Seah, Erin Siedlecki, Emily Garman, Ben Doniger, Bela Barton
            """)
        ),
        # DATA TAB
        ui.nav_panel(
            "Data",
            ui.h3("Dataset Preview"),
            ui.p(f"Showing first 100 rows out of {total_rows} total rows."),
            ui.output_table("datatable", bordered=True, striped=True, hover=True)
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
                            "linreg": "Multiple Linear Regression",
                            "lasso": "Lasso Regression",
                            "rf": "Random Forest"
                        }
                    ),
                    ui.h4("Train/Test Split"),
                    ui.input_slider("split", "Train/Test Split (%)", min=50, max=90, value=80),
                    ui.input_action_button("run_model", "Run Model", class_="btn-primary"),
                    ui.hr(),
                    ui.h4("Predict on New Data"),
                    *[ui.input_numeric(f"pred_{col}", col, float(df[col].mean()))
                      for col in numeric_predictors],
                    ui.input_action_button("predict_model", "Predict", class_="btn-success")
                ),
                ui.card(
                    ui.h4("Regression Coefficients / Feature Importance"),
                    ui.output_table("coef_table"),
                    ui.h4("Evaluation Metrics"),
                    ui.output_ui("eval_metrics"),
                    ui.h4("Predicted vs Actual BMI Plot"),
                    ui.output_plot("pred_plot")
                )
            )
        )
    )
)

# ----- SERVER -----
def server(input, output, session):

    # DATA TAB
    @output
    @render.table
    def datatable():
        return df.head(100)

    # MODELING TAB: reactive for Linear Regression, Lasso, and Random Forest
    @reactive.Calc
    def model_results():
        input.run_model()
        model_type = input.model_choice()
        df_filtered = df[df['Alcohol_Consumption_Frequency'] != 'Always'].copy()
        nums = ['Age']
        cats = [
            'Overweight_Family_History',
            'High_Calorie_Consumption_Often',
            'Vegetable_Consumption_Often',
            'Calories_Monitored_Daily',
            'Workout_Frequency',
            'Alcohol_Consumption_Frequency',
            'Means_of_Transportation'
        ]
        X = df_filtered[nums + cats]
        y = df_filtered['BMI_kg_m2']
        test_size = 1 - input.split()/100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        preprocess = ColumnTransformer(
            transformers=[
                ("encoder", OneHotEncoder(drop="first"), cats),
                ("numeric", "passthrough", nums)
            ]
        )

        # --- LINEAR REGRESSION ---
        if model_type == "linreg":
            X_train_processed = preprocess.fit_transform(X_train)
            X_test_processed = preprocess.transform(X_test)
            feature_names = preprocess.get_feature_names_out()
            X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
            X_train_sm = sm.add_constant(X_train_df)
            model = sm.OLS(y_train, X_train_sm).fit()
            X_test_sm = sm.add_constant(X_test_df)
            y_pred = model.predict(X_test_sm)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            return {"model": model, "X_test": X_test_sm, "y_test": y_test, "y_pred": y_pred,
                    "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae}, "type": "linreg"}

        # --- LASSO REGRESSION ---
        elif model_type == "lasso":
            lasso_pipeline = Pipeline([
                ('preprocessor', preprocess),
                ('scaler', StandardScaler()),
                ('lasso', LassoCV(cv=5, random_state=42))
            ])
            lasso_pipeline.fit(X_train, y_train)
            y_pred = lasso_pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            feature_names = preprocess.get_feature_names_out()
            coefs = pd.Series(lasso_pipeline.named_steps['lasso'].coef_, index=feature_names)
            non_zero_coefs = coefs[coefs != 0].sort_values(ascending=False)
            return {"model": lasso_pipeline, "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
                    "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae}, "type": "lasso", "coefs": non_zero_coefs}

        # --- RANDOM FOREST ---
        elif model_type == "rf":
            X_train_processed = preprocess.fit_transform(X_train)
            X_test_processed = preprocess.transform(X_test)
            feature_names = preprocess.get_feature_names_out()
            X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

            rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
            rf_model.fit(X_train_df, y_train)
            y_pred = rf_model.predict(X_test_df)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            rf_importances = pd.Series(rf_model.feature_importances_, index=X_train_df.columns).sort_values(ascending=False)

            return {"model": rf_model, "X_test": X_test_df, "y_test": y_test, "y_pred": y_pred,
                    "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae}, "type": "rf", "coefs": rf_importances}

    # --- Regression coefficients / feature importance table ---
    @output
    @render.table
    def coef_table():
        results = model_results()
        if results is None:
            return pd.DataFrame()
        if results["type"] in ["linreg"]:
            coefs = results["model"].summary2().tables[1].reset_index().rename(columns={'index':'Variable'})
            cols = ['Variable'] + [c for c in coefs.columns if c != 'Variable']
            return coefs[cols]
        elif results["type"] in ["lasso", "rf"]:
            coefs = results["coefs"].reset_index()
            coefs.columns = ["Variable", "Coefficient" if results['type']=='lasso' else "Importance"]
            return coefs

    # --- Evaluation metrics ---
    @output
    @render.ui
    def eval_metrics():
        results = model_results()
        if results is None:
            return ui.HTML("")
        m = results["metrics"]
        return ui.HTML(f"<pre>RMSE: {m['RMSE']:.3f}\nMAE: {m['MAE']:.3f}\nR²: {m['R2']:.3f}</pre>")



# ----- CREATE APP -----
app = App(app_ui, server)
