"""
Wine Quality MLOps - Monitoring Dashboard

This Streamlit application provides real-time monitoring for:
- Model performance metrics
- Data drift detection
- Model drift detection
- Prediction distribution analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metrics import *

# Page configuration
st.set_page_config(
    page_title="Wine Quality Monitoring",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MLflow configuration
import os
# Get absolute path to mlruns directory
MLRUNS_PATH = Path(__file__).parent.parent / "mlruns"
MLFLOW_TRACKING_URI = f"file://{MLRUNS_PATH.absolute()}"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stAlert {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data():
    """Load wine quality dataset."""
    data_path = Path(__file__).parent.parent / "data" / "winequality-red.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    return None


@st.cache_data(ttl=300)
def get_mlflow_experiments():
    """Get all MLflow experiments and runs."""
    experiments = client.search_experiments()
    runs_data = []

    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        for run in runs:
            runs_data.append({
                'run_id': run.info.run_id,
                'experiment_name': exp.name,
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                'status': run.info.status,
                'r2_score': run.data.metrics.get('r2_score', np.nan),
                'rmse': run.data.metrics.get('rmse', np.nan),
                'mae': run.data.metrics.get('mae', np.nan),
                'mse': run.data.metrics.get('mse', np.nan),
            })

    return pd.DataFrame(runs_data)


@st.cache_data(ttl=300)
def get_model_versions():
    """Get registered model versions."""
    try:
        models = client.search_registered_models()
        versions_data = []

        for model in models:
            for version in client.search_model_versions(f"name='{model.name}'"):
                versions_data.append({
                    'model_name': model.name,
                    'version': version.version,
                    'stage': version.current_stage,
                    'run_id': version.run_id,
                    'creation_time': pd.to_datetime(version.creation_timestamp, unit='ms')
                })

        return pd.DataFrame(versions_data)
    except Exception as e:
        st.error(f"Error loading model versions: {e}")
        return pd.DataFrame()


def create_performance_chart(runs_df):
    """Create model performance comparison chart."""
    if runs_df.empty:
        return None

    # Filter valid runs
    valid_runs = runs_df.dropna(subset=['r2_score', 'rmse'])

    if valid_runs.empty:
        return None

    # Sort by time
    valid_runs = valid_runs.sort_values('start_time')

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('R² Score Over Time', 'RMSE Over Time'),
        vertical_spacing=0.15
    )

    # R² Score
    fig.add_trace(
        go.Scatter(
            x=valid_runs['start_time'],
            y=valid_runs['r2_score'],
            mode='lines+markers',
            name='R² Score',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )

    # RMSE
    fig.add_trace(
        go.Scatter(
            x=valid_runs['start_time'],
            y=valid_runs['rmse'],
            mode='lines+markers',
            name='RMSE',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8)
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="R² Score", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white'
    )

    return fig


def create_metrics_comparison(runs_df):
    """Create metrics comparison bar chart."""
    if runs_df.empty or len(runs_df) < 2:
        return None

    # Get last 5 runs
    recent_runs = runs_df.nlargest(5, 'start_time')

    fig = go.Figure()

    metrics = ['r2_score', 'rmse', 'mae']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=[f"Run {j+1}" for j in range(len(recent_runs))],
            y=recent_runs[metric].values,
            marker_color=colors[i]
        ))

    fig.update_layout(
        title="Recent Runs - Metrics Comparison",
        xaxis_title="Run",
        yaxis_title="Value",
        barmode='group',
        template='plotly_white',
        height=400
    )

    return fig


def analyze_data_drift(reference_data, current_data):
    """Analyze data drift using Evidently."""
    if reference_data is None or current_data is None:
        return None

    try:
        # Create drift report
        report = Report(metrics=[
            DataDriftPreset(),
        ])

        # Ensure same columns
        common_cols = list(set(reference_data.columns) & set(current_data.columns))
        ref_subset = reference_data[common_cols].copy()
        curr_subset = current_data[common_cols].copy()

        report.run(reference_data=ref_subset, current_data=curr_subset)

        return report
    except Exception as e:
        st.error(f"Error in drift analysis: {e}")
        return None


# Main application
def main():
    st.title("🍷 Wine Quality MLOps Monitoring Dashboard")
    st.markdown("Real-time monitoring for model performance, data drift, and predictions")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Model Performance", "Data Drift", "Model Registry"]
    )

    # Load data
    data = load_data()
    runs_df = get_mlflow_experiments()
    versions_df = get_model_versions()

    # Page routing
    if page == "Overview":
        show_overview(data, runs_df, versions_df)
    elif page == "Model Performance":
        show_model_performance(runs_df)
    elif page == "Data Drift":
        show_data_drift(data)
    elif page == "Model Registry":
        show_model_registry(versions_df, runs_df)


def show_overview(data, runs_df, versions_df):
    """Show overview page."""
    st.header("System Overview")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Runs", len(runs_df) if not runs_df.empty else 0)

    with col2:
        if not runs_df.empty and 'r2_score' in runs_df.columns:
            latest_r2 = runs_df.nlargest(1, 'start_time')['r2_score'].values[0]
            st.metric("Latest R² Score", f"{latest_r2:.4f}")
        else:
            st.metric("Latest R² Score", "N/A")

    with col3:
        prod_models = versions_df[versions_df['stage'] == 'Production'] if not versions_df.empty else pd.DataFrame()
        st.metric("Production Models", len(prod_models))

    with col4:
        if data is not None:
            st.metric("Dataset Size", f"{len(data):,}")
        else:
            st.metric("Dataset Size", "N/A")

    st.markdown("---")

    # Recent activity
    st.subheader("Recent Training Runs")
    if not runs_df.empty:
        recent = runs_df.nlargest(5, 'start_time')[['start_time', 'experiment_name', 'status', 'r2_score', 'rmse']]
        st.dataframe(recent, use_container_width=True)
    else:
        st.info("No training runs found.")

    # Performance chart
    st.subheader("Performance Trends")
    perf_chart = create_performance_chart(runs_df)
    if perf_chart:
        st.plotly_chart(perf_chart, use_container_width=True)
    else:
        st.info("No performance data available.")


def show_model_performance(runs_df):
    """Show model performance page."""
    st.header("Model Performance Analysis")

    if runs_df.empty:
        st.warning("No runs found in MLflow.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        exp_filter = st.multiselect(
            "Filter by Experiment",
            options=runs_df['experiment_name'].unique(),
            default=runs_df['experiment_name'].unique()
        )

    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(
                runs_df['start_time'].min().date(),
                runs_df['start_time'].max().date()
            )
        )

    # Apply filters
    filtered_df = runs_df[
        (runs_df['experiment_name'].isin(exp_filter)) &
        (runs_df['start_time'].dt.date >= date_range[0]) &
        (runs_df['start_time'].dt.date <= date_range[1])
    ]

    # Metrics comparison
    st.subheader("Metrics Comparison")
    comp_chart = create_metrics_comparison(filtered_df)
    if comp_chart:
        st.plotly_chart(comp_chart, use_container_width=True)

    # Performance over time
    st.subheader("Performance Over Time")
    perf_chart = create_performance_chart(filtered_df)
    if perf_chart:
        st.plotly_chart(perf_chart, use_container_width=True)

    # Detailed metrics table
    st.subheader("Detailed Metrics")
    st.dataframe(
        filtered_df[['start_time', 'experiment_name', 'r2_score', 'rmse', 'mae', 'mse']].sort_values('start_time', ascending=False),
        use_container_width=True
    )


def show_data_drift(data):
    """Show data drift analysis page."""
    st.header("Data Drift Detection")

    if data is None:
        st.warning("No data available for drift analysis.")
        return

    st.info("Comparing recent data (last 20%) with reference data (first 80%)")

    # Split data into reference and current
    split_point = int(len(data) * 0.8)
    reference_data = data.iloc[:split_point].copy()
    current_data = data.iloc[split_point:].copy()

    st.write(f"Reference data: {len(reference_data)} samples")
    st.write(f"Current data: {len(current_data)} samples")

    # Run drift analysis
    with st.spinner("Analyzing data drift..."):
        report = analyze_data_drift(reference_data, current_data)

        if report:
            # Save report
            report_path = Path(__file__).parent / "drift_report.html"
            report.save_html(str(report_path))

            st.success("Drift analysis complete!")

            # Display report
            with open(report_path, 'r') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=800, scrolling=True)
        else:
            st.error("Failed to generate drift report.")


def show_model_registry(versions_df, runs_df):
    """Show model registry page."""
    st.header("Model Registry")

    if versions_df.empty:
        st.warning("No registered models found.")
        return

    # Model versions table
    st.subheader("Registered Models")
    st.dataframe(
        versions_df[['model_name', 'version', 'stage', 'creation_time']].sort_values('creation_time', ascending=False),
        use_container_width=True
    )

    # Production models
    st.subheader("Production Models")
    prod_models = versions_df[versions_df['stage'] == 'Production']

    if not prod_models.empty:
        for _, model in prod_models.iterrows():
            with st.expander(f"{model['model_name']} - Version {model['version']}"):
                # Get run metrics
                run_metrics = runs_df[runs_df['run_id'] == model['run_id']]

                if not run_metrics.empty:
                    metrics = run_metrics.iloc[0]
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("R² Score", f"{metrics['r2_score']:.4f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['rmse']:.4f}")
                    with col3:
                        st.metric("MAE", f"{metrics['mae']:.4f}")

                st.write(f"**Created:** {model['creation_time']}")
                st.write(f"**Run ID:** {model['run_id']}")
    else:
        st.info("No models in production stage.")


if __name__ == "__main__":
    main()
