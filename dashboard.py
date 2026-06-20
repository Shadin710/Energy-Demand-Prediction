import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import os

# Import your custom modules
import src.data_loader as dl
import src.feature_engineering as fe

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Energy Demand Forecaster",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡Energy Demand Forecaster")

st.markdown("""
Forecast electricity demand using machine learning and time-series analysis.
""")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    # Adjusted path to standard relative format
    model_path = os.path.join("models", "xgboost_forecaster.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.sidebar.error("⚠️ Saved model (`models/xgboost_forecaster.pkl`) not found in repository.")

# -----------------------------
# Sidebar
# -----------------------------
# uploaded_file = st.sidebar.file_uploader(
#     "Upload CSV for Analysis/Inference",
#     type=["csv"]
# )
uploaded_file = './data/electricity_demand.csv'
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Explorer",
        "Forecast",
        "Performance",
        "Feature Importance"
    ]
)

# -----------------------------
# Load Data if uploaded
# -----------------------------
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded successfully!")

# -----------------------------
# Overview Page
# -----------------------------
if page == "Overview":
    st.header("Project Overview")

    st.write("""
    This application forecasts electricity demand using
    machine learning models trained on Australian energy market data.
    """)

    if df is not None:
        st.subheader("Quick Dataset Preview")
        st.dataframe(df.head())
    else:
        st.info("💡 Upload a CSV file in the sidebar to get started with data exploration and forecasting.")

# -----------------------------
# Data Explorer
# -----------------------------
if page == "Data Explorer":
    st.header("Data Explorer")
    
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Shape")
            st.write(df.shape)
        with col2:
            st.subheader("Missing Values")
            st.dataframe(df.isnull().sum())

        if "Settlement Date" in df.columns and "Scheduled Demand (MW)" in df.columns:
            df["Settlement Date"] = pd.to_datetime(df["Settlement Date"], errors="coerce")

            fig = px.line(
                df,
                x="Settlement Date",
                y="Scheduled Demand (MW)",
                title="Demand Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload a CSV file via the sidebar to use the Data Explorer.")

# -----------------------------
# Forecast Page (Updated to use your saved model)
# -----------------------------
if page == "Forecast":
    st.header("Demand Forecast")

    if model is None:
        st.error("Cannot run forecast because the model file is missing.")
    elif df is None:
        st.warning("Please upload a CSV dataset in the sidebar containing features for inference.")
    else:
        st.info("Model loaded successfully from repository. Ready for inference.")
        
        run_button = st.button("Generate Forecasts")

        if run_button:
            with st.spinner("Engineering features and generating predictions..."):
                try:
                    # 1. Format Dates and Sort
                    df_prep = dl.convert_datetime(df.copy())
                    df_prep = dl.sort_by_date(df_prep)

                    # 2. Apply Feature Engineering
                    df_prep = fe.create_time_features(df_prep)
                    df_prep = fe.create_lag_features(df_prep)
                    df_prep = fe.create_rolling_features(df_prep)

                    # 3. Drop rows with NaNs (caused by lag_288 and rolling windows)
                    df_prep = fe.remove_null_rows(df_prep)

                    if df_prep.empty:
                        st.error("Dataset is too small! You need at least 289 rows to calculate the 'lag_288' feature.")
                        st.stop()

                    # 4. Isolate inference features
                    X_infer = df_prep.copy()
                    
                    # Save dates and actuals for plotting
                    dates = X_infer["Settlement Date"]
                    actuals = X_infer["Scheduled Demand (MW)"] if "Scheduled Demand (MW)" in X_infer.columns else None
                    
                    # Drop non-feature columns (Targets, Dates, and Strings like 'Type')
                    cols_to_drop = ["Settlement Date", "Scheduled Demand (MW)", "Type"]
                    X_infer = X_infer.drop(columns=[c for c in cols_to_drop if c in X_infer.columns])

                    # 5. Generate predictions safely
                    predictions = model.predict(X_infer)
                    
                    # 6. Build output DataFrame
                    results_df = pd.DataFrame({
                        "Date": dates,
                        "Forecast": predictions
                    })
                    
                    if actuals is not None:
                        results_df["Actual"] = actuals.values

                    # Display Plot
                    st.success("Forecast generated successfully!")
                    
                    y_series = ["Actual", "Forecast"] if "Actual" in results_df.columns else ["Forecast"]
                    fig = px.line(
                        results_df,
                        x="Date",
                        y=y_series,
                        title="Model Forecasting Results"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Preview Results
                    st.subheader("Forecasted Values")
                    st.dataframe(results_df.head(50))
                    
                except Exception as e:
                    st.error("Prediction failed during feature engineering or inference.")
                    st.exception(e)

# -----------------------------
# Performance Page
# -----------------------------
if page == "Performance":
    st.header("Model Performance")

    if os.path.exists("reports/model_metrics.csv"):
        metrics = pd.read_csv("reports/model_metrics.csv")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", round(metrics["MAE"][0], 2))
        col2.metric("RMSE", round(metrics["RMSE"][0], 2))
        col3.metric("R²", round(metrics["R2"][0], 3))
    else:
        # Fallback if metrics csv isn't present in git, fallback to standard mock placeholder or instructions
        st.info("To see historical validation performance metrics, ensure `reports/model_metrics.csv` is committed to your repository.")

# -----------------------------
# Feature Importance Page
# -----------------------------
if page == "Feature Importance":
    st.header("Feature Importance")

    if os.path.exists("reports/figures/feature_importance.png"):
        st.image("reports/figures/feature_importance.png")
    else:
        st.info("Feature importance visual lookup (`reports/figures/feature_importance.png`) not found.")