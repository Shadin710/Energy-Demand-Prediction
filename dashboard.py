import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import os
import subprocess
# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Infrastructure Demand Forecaster",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Infrastructure Resource Demand Forecaster")

st.markdown("""
Forecast electricity demand using machine learning and time-series analysis.
""")

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_forecaster.pkl")

try:
    model = load_model()
except:
    model = None
    st.sidebar.warning("Model not found. Run main.py first.")

# -----------------------------
# Sidebar
# -----------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

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
    st.sidebar.success("Dataset uploaded successfully")

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

# -----------------------------
# Data Explorer
# -----------------------------
if page == "Data Explorer" and df is not None:

    st.header("Dataset Preview")

    st.dataframe(df.head(20))

    st.subheader("Dataset Shape")
    st.write(df.shape)

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

# -----------------------------
# Forecast Page
# -----------------------------
if page == "Forecast":

    st.header("Demand Forecast")

    run_button = st.button("Run Forecast Model")

    if run_button:

        with st.spinner("Running pipeline..."):

            result = subprocess.run(
                ["python", "main.py"],
                capture_output=True,
                text=True
            )

        if result.returncode == 0:
            st.success("Model executed successfully!")

        else:
            st.error("Pipeline failed")
            st.code(result.stderr)

    # Now load results AFTER running
    if os.path.exists("reports/forecast_results.csv"):

        forecast_df = pd.read_csv("reports/forecast_results.csv")

        fig = px.line(
            forecast_df,
            y=["Actual", "Forecast"],
            title="Actual vs Forecast"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No forecast results yet. Click 'Run Forecast Model'")
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
        st.warning("No metrics found. Run main.py first.")

# -----------------------------
# Feature Importance Page
# -----------------------------
if page == "Feature Importance":

    st.header("Feature Importance")

    if os.path.exists("reports/figures/feature_importance.png"):
        st.image("reports/figures/feature_importance.png")
    else:
        st.warning("Feature importance plot not found. Run main.py first.")