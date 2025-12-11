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

# Load and preprocess data
df = pd.read_csv("obesity_cleaned.csv")
df['BMI_kg_m2'] = round(df['Weight'] / (df['Height'] ** 2), 2)
df = df.rename(columns={'Height': 'Height_meters', 'Weight': 'Weight_kg'})

total_rows = df.shape[0]
target_col = "BMI_kg_m2"
predictor_cols = [c for c in df.columns if c != target_col]
numeric_predictors = [c for c in predictor_cols if pd.api.types.is_numeric_dtype(df[c])]
categorical_predictors = [c for c in predictor_cols if c not in numeric_predictors]

# UI
app_ui = ui.page_fluid(
    # Global CSS for table alignment
    ui.tags.style("""
        table th, table td {
            text-align: left !important;
        }
    """),

    # Title
    ui.div(
        ui.h1(
            "BMI Prediction Dashboard",
            style="font-size:32px; font-weight:bold; margin-top:40px; margin-bottom:25px; color:#2c3e50;"
        ),
        style="padding-left:20px;"
    ),

    # Navbar
    ui.page_navbar(
        # README tab
        ui.nav_panel(
            "README",
            ui.div(
                ui.h2("Project Overview: ", style="color:#2c3e50;"),
                ui.markdown("""
                This app uses nutritional, physical, and behavioral data to build predictive models for BMI.
                <br>
                ## Data Source
                The dataset used in this project is the Estimation of Obesity Levels based on Eating Habits and Physical Condition from the UCI Machine Learning Repository. It contains nutritional, physical, and behavioral features of individuals, along with a target variable representing their obesity level. The data was originally collected via an anonymous online survey, and after initial preprocessing and BMI calculation, synthetic data was generated to address class imbalance, resulting in a final, balanced dataset of 2,111 records.

                ## Target Variable
                - BMI (kg/m2)

                ## Predictor Variables
                - Age   
                - Overweight_Family_History
                - High_Calorie_Consumption_Often
                - Vegetable_Consumption_Often
                - Calories_Monitored_Daily
                - Workout_Frequency
                - Alcohol_Consumption_Frequency
                - Means_of_Transportation

                ## Goals
                - Determine factors affecting BMI
                - Build predictive models: Linear Regression, Lasso, Random Forest

                ## App Usage
                - View data in Data tab
                - Select model & train/test split in Modeling tab
                - Enter new values to predict BMI based on model selected

                ## Authors
                Natalie Seah, Erin Siedlecki, Emily Garman, Ben Doniger, Bela Barton
                """),
                style="padding:20px; background-color:#ecf0f1; border-radius:12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);"
            )
        ),

        # Data tab
        ui.nav_panel(
            "Data",
            ui.div(
                ui.p(f"Entries shown are out of {total_rows} total rows.", style="font-weight:bold; color:#34495e;"),
                ui.div(
                    ui.TagList(
                        "Show ",
                        ui.div(
                            ui.input_select(
                                "rows_to_show",
                                "",
                                {10: "10", 25: "25", 50: "50", 100: "100"},
                                selected=100
                            ),
                            style="width: 80px; display: inline-block;"
                        ),
                        " entries"
                    ),
                    style="margin-bottom: 15px; white-space: nowrap;"
                ),
                ui.output_table("datatable", bordered=True, striped=True, hover=True, style="margin-top:10px;")
            )
        ),

        # Modeling tab
        ui.nav_panel(
            "Modeling",
            ui.layout_sidebar(
                # Sidebar
                ui.sidebar(
                    ui.card(
                        ui.TagList(
                            ui.h4("Choose a Model", style="color:#2c3e50; font-weight:bold;"),
                            ui.input_select(
                                "model_choice",
                                "Model Type:",
                                {
                                    "linreg": "Multiple Linear Regression",
                                    "lasso": "Lasso Regression",
                                    "rf": "Random Forest"
                                }
                            ),
                            ui.h4("Train/Test Split", style="color:#2c3e50; margin-top:15px; font-weight:bold;"),
                            ui.input_slider("split", "Train/Test Split (%)", min=50, max=90, value=80),
                            ui.hr(),
                            ui.h4("Predict on New Data", style="color:#2c3e50; margin-top:15px; font-weight:bold;"),
                            ui.output_ui("prediction_inputs")
                        ),
                        style="padding:20px; border-radius:15px; box-shadow:0 6px 12px rgba(0,0,0,0.15); background-color:#f8f9fa; margin-bottom:20px;"
                    ),
                    width="350px"
                ),
                # Main panel
                ui.row(
                    ui.column(12,
                        # Regression Coefficients / Feature Importance
                        ui.card(
                            ui.TagList(
                                ui.h4("Regression Coefficients / Feature Importance", style="color:#2c3e50; font-weight:bold;"),
                                ui.div(
                                    ui.output_table("coef_table"),
                                    style="text-align: left;"
                                )
                            ),
                            style="padding:20px; border-radius:15px; box-shadow:0 6px 12px rgba(0,0,0,0.15); background-color:#f8f9fa; margin-bottom:20px;"
                        ),
                        # Evaluation Metrics
                        ui.card(
                            ui.TagList(
                                ui.h4("Evaluation Metrics", style="color:#2c3e50; font-weight:bold;"),
                                ui.output_ui("eval_metrics")
                            ),
                            style="padding:20px; border-radius:15px; box-shadow:0 6px 12px rgba(0,0,0,0.15); background-color:#f8f9fa; margin-bottom:20px;"
                        )
                    )
                )
            )
        )
    )  # closes page_navbar
)  # closes page_fluid

# Server
def server(input, output, session):
    # Data tab
    @output
    @render.table
    def datatable():
        n = int(input.rows_to_show())
        return df.head(n)

    # Modeling tab
    @reactive.Calc
    def model_results():
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
                ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"), cats),
                ("numeric", "passthrough", nums)
            ]
        )

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
            return {"model": model, "preprocess": preprocess, "X_train": X_train_sm, "X_test": X_test_sm,
                    "y_test": y_test, "y_pred": y_pred,
                    "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae}, "type": "linreg", "cats": cats, "nums": nums}

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
            return {"model": lasso_pipeline, "preprocess": preprocess, "X_train": X_train, "X_test": X_test,
                    "y_test": y_test, "y_pred": y_pred,
                    "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae}, "type": "lasso",
                    "coefs": non_zero_coefs, "cats": cats, "nums": nums}

        else:  # rf
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
            return {"model": rf_model, "preprocess": preprocess, "X_train": X_train_df, "X_test": X_test_df,
                    "y_test": y_test, "y_pred": y_pred,
                    "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae}, "type": "rf",
                    "coefs": rf_importances, "cats": cats, "nums": nums}

    # Coefficient table
    @output
    @render.table
    def coef_table():
        results = model_results()
        if results is None:
            return pd.DataFrame()
        if results["type"] == "linreg":
            coefs = results["model"].summary2().tables[1].reset_index().rename(columns={'index':'Variable'})
            cols = ['Variable'] + [c for c in coefs.columns if c != 'Variable']
            return coefs[cols]
        else:
            coefs = results["coefs"].reset_index()
            coefs.columns = ["Variable", "Coefficient" if results['type']=='lasso' else "Importance"]
            return coefs

    # Evaluation metrics
    @output
    @render.ui
    def eval_metrics():
        results = model_results()
        if results is None:
            return ui.HTML("")
        m = results["metrics"]
        return ui.HTML(f"<pre>RMSE: {m['RMSE']:.3f}\nMAE: {m['MAE']:.3f}\nR²: {m['R2']:.3f}</pre>")

    # Prediction inputs
    @output
    @render.ui
    def prediction_inputs():
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
        numeric_inputs = [ui.input_numeric(f"pred_{col}", col, int(round(df[col].mean()))) for col in nums]
        categorical_inputs = [ui.input_select(f"pred_{col}", col, {val: val for val in df[col].dropna().unique()}) for col in cats]
        return ui.TagList(*(numeric_inputs + categorical_inputs +
                            [ui.input_action_button("predict_model", "Predict", class_="btn-success"),
                             ui.output_text("predicted_bmi")]))

    # Predicted BMI
    @output
    @render.text
    @reactive.event(input.predict_model)
    def predicted_bmi():
        results = model_results()
        if results is None:
            return "No model trained yet."
        new_data = {col: input[f"pred_{col}"]() for col in results["nums"]}
        for col in results["cats"]:
            new_data[col] = input[f"pred_{col}"]()
        new_df = pd.DataFrame([new_data])
        preprocess = results["preprocess"]

        if results["type"] == "linreg":
            X_new = preprocess.transform(new_df)
            X_new_df = pd.DataFrame(X_new, columns=preprocess.get_feature_names_out())
            X_new_sm = sm.add_constant(X_new_df, has_constant='add')
            X_new_sm = X_new_sm.reindex(columns=results["X_train"].columns, fill_value=0)
            pred_bmi = results["model"].predict(X_new_sm)[0]
        elif results["type"] == "lasso":
            X_new = preprocess.transform(new_df)
            scaler = results["model"].named_steps["scaler"]
            X_new_scaled = scaler.transform(X_new)
            pred_bmi = results["model"].named_steps["lasso"].predict(X_new_scaled)[0]
        else:  # rf
            X_new = preprocess.transform(new_df)
            X_new_df = pd.DataFrame(X_new, columns=preprocess.get_feature_names_out())
            pred_bmi = results["model"].predict(X_new_df)[0]

        return f"Predicted BMI: {pred_bmi:.2f}"

# Create app
app = App(app_ui, server)
