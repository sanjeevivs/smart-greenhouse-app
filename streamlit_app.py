"""
STREAMLIT DASHBOARD
===================

This module implements the Streamlit dashboard for the smart greenhouse system.
It provides:
- Real-time monitoring of environmental conditions from a persistent simulation.
- Visualization of ML predictions and forecasts.
- Manual control of the irrigation system.
- An interface for human-in-the-loop review of uncertain predictions.

REFACTORING NOTES:
- Centralized all initialization logic into an `initialize_system` function.
- Removed all low-level imports (e.g., GreenhouseEnvironment). The UI now correctly
  interacts only with the high-level simulator and ML system interfaces.
- Implemented a persistent historical data log in `st.session_state` to show
  *true* historical data from the running simulation, instead of generating
  fake data on each refresh.
- Removed hardcoded diagnostic logic from the UI. The dashboard now correctly
  displays the diagnosis provided by the `SystemHealthMonitor` model.
"""

# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# --- Core Application Imports ---
from config import config
from data_generation import RealTimeDataSimulator
from models import GreenhouseMLSystem

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Greenhouse Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* [CSS from the original file is unchanged and remains here] */
    .metric-card {
        background-color: white; border-radius: 10px; padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 15px;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fffbe6; border-left: 5px solid #faad14;
        padding: 10px; margin: 10px 0; border-radius: 5px;
    }
    .critical-box {
        background-color: #fff1f0; border-left: 5px solid #f5222d;
        padding: 10px; margin: 10px 0; border-radius: 5px;
    }
    .success-box {
        background-color: #f6ffed; border-left: 5px solid #52c41a;
        padding: 10px; margin: 10px 0; border-radius: 5px;
    }
    .info-box {
        background-color: #e6f7ff; border-left: 5px solid #1890ff;
        padding: 10px; margin: 10px 0; border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# --- System Initialization and State Management ---
def initialize_system():
    """
    Initializes the simulator, ML system, and historical data log in Streamlit's session state.
    This function is called only once at the start of the app.
    """
    if 'system_initialized' not in st.session_state:
        st.session_state.simulator = RealTimeDataSimulator()
        st.session_state.simulator.initialize()
        
        st.session_state.ml_system = GreenhouseMLSystem()
        # Set train_if_needed=True to generate data and train models on first run
        st.session_state.ml_system.initialize(train_if_needed=True)
        
        # FIX: Create a persistent log for true historical data
        st.session_state.historical_data_log = pd.DataFrame()
        
        # Other state variables
        st.session_state.manual_irrigation_requested = False
        st.session_state.manual_irrigation_duration = 10
        st.session_state.current_view = "dashboard"
        st.session_state.human_review_needed = False
        st.session_state.review_prediction = None
        
        st.session_state.system_initialized = True
        st.toast("System Initialized Successfully!", icon="🚀")

# Call initialization function at the start of the script
initialize_system()

# --- Data Fetching and Caching ---
@st.cache_data(ttl=5) # Cache data for 5 seconds to avoid redundant calls
def get_live_data():
    """
    Fetches all live data from the simulator and ML system.
    This is the single source of truth for the app's current state.
    """
    sensor_data = st.session_state.simulator.get_current_readings()
    prediction = st.session_state.ml_system.predict_irrigation(sensor_data)
    health = st.session_state.ml_system.monitor_system_health(sensor_data)
    disease_image_path = st.session_state.simulator.get_disease_image()
    disease_detection = st.session_state.ml_system.detect_disease(disease_image_path)
    
    return sensor_data, prediction, health, disease_detection

def update_historical_log(sensor_data):
    """Appends the latest sensor data to the historical log."""
    new_record = pd.DataFrame([sensor_data])
    st.session_state.historical_data_log = pd.concat(
        [st.session_state.historical_data_log, new_record],
        ignore_index=True
    )
    # Keep the log from getting too large (e.g., last 7 days)
    max_log_size = 7 * 24 * (60 // config.DATA_COLLECTION_INTERVAL)
    if len(st.session_state.historical_data_log) > max_log_size:
        st.session_state.historical_data_log = st.session_state.historical_data_log.tail(max_log_size)

# --- UI Component Functions (Plotting and Display) ---

def create_gauge(value, min_val, max_val, title, thresholds):
    """Creates a Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#2c3e50"},
            'steps': thresholds,
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_soil_moisture_forecast_chart(forecast):
    """Creates a Plotly chart for the soil moisture forecast."""
    hours = list(range(1, len(forecast) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=forecast, mode='lines+markers', name='Forecast'))
    fig.update_layout(
        title='Soil Moisture Forecast (Next 24 Hours)',
        xaxis_title='Hours from Now',
        yaxis_title='Soil Moisture (%)',
        height=300,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_range=[max(0, min(forecast)-10), min(100, max(forecast)+10)]
    )
    return fig

# --- Page View Functions ---

def show_dashboard(sensor_data, prediction, health, disease_detection):
    """Renders the main dashboard view."""
    st.header("Real-Time Greenhouse Dashboard")

    # Handle Human-in-the-Loop trigger
    if prediction['needs_human_review'] and not st.session_state.human_review_needed:
        st.session_state.human_review_needed = True
        st.session_state.review_prediction = prediction
        st.warning("Prediction uncertainty is high. Please review the recommendation.")
        st.rerun()

    if st.session_state.human_review_needed:
        show_human_review_interface(st.session_state.review_prediction, sensor_data)
        return

    # --- Metrics Row ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.plotly_chart(create_gauge(sensor_data.get('soil_moisture', 0), 0, 100, "Soil Moisture (%)",
            [{'range': [0, 40], 'color': "lightcoral"}, {'range': [40, 75], 'color': "lightgreen"}, {'range': [75, 100], 'color': "lightsalmon"}]),
            use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.plotly_chart(create_gauge(sensor_data.get('temperature', 0), 10, 40, "Temperature (°C)",
            [{'range': [10, 18], 'color': "lightblue"}, {'range': [18, 28], 'color': "lightgreen"}, {'range': [28, 40], 'color': "lightcoral"}]),
            use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # ... other gauges ...
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.plotly_chart(create_gauge(sensor_data.get('humidity', 0), 20, 100, "Humidity (%)",
            [{'range': [20, 50], 'color': "lightsalmon"}, {'range': [50, 80], 'color': "lightgreen"}, {'range': [80, 100], 'color': "lightcoral"}]),
            use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.plotly_chart(create_gauge(sensor_data.get('light', 0), 0, 1000, "Light (lux)",
            [{'range': [0, 300], 'color': "lightgray"}, {'range': [300, 800], 'color': "lightgreen"}, {'range': [800, 1000], 'color': "gold"}]),
            use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Charts and Controls Row ---
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Live Environmental Trends")
        # FIX: Use the true historical log for the 24-hour chart
        recent_data = st.session_state.historical_data_log.tail(288) # Last 24 hours
        if not recent_data.empty:
            st.line_chart(recent_data, x='timestamp', y=['temperature', 'humidity', 'soil_moisture'])
        else:
            st.info("Collecting data for historical trends chart...")

    with col2:
        st.subheader("AI Decision Support")
        st.plotly_chart(create_soil_moisture_forecast_chart(prediction['soil_moisture_forecast']), use_container_width=True)
        
        if prediction['irrigation_needed']:
            st.success(f"**Recommendation:** Irrigate for **{prediction['water_volume']/10:.1f} seconds** ({prediction['water_volume']:.1f}ml). Confidence: {prediction['probability']:.1%}")
        else:
            st.info(f"**Recommendation:** No irrigation needed. Confidence: {1-prediction['probability']:.1%}")

def show_human_review_interface(prediction, sensor_data):
    """Renders the human-in-the-loop review interface."""
    st.error("🚨 **Human Review Required** 🚨")
    st.markdown(f"<div class='warning-box'>**Reason:** {prediction['review_reason']}</div>", unsafe_allow_html=True)
    
    st.write("The AI is uncertain about its recommendation. Please review the data and make a final decision.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Data")
        st.json(sensor_data)
        st.plotly_chart(create_soil_moisture_forecast_chart(prediction['soil_moisture_forecast']))

    with col2:
        st.subheader("AI's Uncertain Recommendation")
        st.write(f"Irrigation Needed: **{prediction['irrigation_needed']}** (Confidence: {prediction['probability']:.1%})")
        st.write(f"Recommended Volume: **{prediction['water_volume']:.1f} ml**")
        st.write(f"Prediction Interval: **{prediction['volume_lower_bound']:.1f}ml - {prediction['volume_upper_bound']:.1f}ml**")

        st.subheader("Your Decision")
        decision = st.radio("Override Decision:", ("Approve Recommendation", "Deny Recommendation (Do Not Irrigate)", "Manual Override"))
        
        if decision == "Manual Override":
            duration = st.slider("Manual Irrigation Duration (seconds)", 5, 60, 15)

        if st.button("Confirm and Log Decision", type="primary"):
            # In a real system, this feedback would be logged for model retraining
            st.success("Your decision has been logged. The system will learn from this feedback.")
            st.session_state.human_review_needed = False
            # If manual action was taken, trigger it
            if decision == "Manual Override":
                st.session_state.manual_irrigation_requested = True
                st.session_state.manual_irrigation_duration = duration
            elif decision == "Approve Recommendation" and prediction['irrigation_needed']:
                st.session_state.manual_irrigation_requested = True
                st.session_state.manual_irrigation_duration = prediction['water_volume'] / 10 # Assuming 10ml/s
            
            time.sleep(2)
            st.rerun()

def show_system_health(health):
    """Renders the system health and diagnostics page."""
    st.header("System Health & Diagnostics")
    
    if health['is_healthy']:
        st.success(f"**System Status: OK** (Anomaly Score: {health['anomaly_score']:.3f})")
    else:
        st.error(f"**System Status: Anomaly Detected!** (Anomaly Score: {health['anomaly_score']:.3f})")
        # FIX: Display the diagnosis from the model, not from hardcoded UI logic.
        st.markdown(f"<div class='critical-box'>**Diagnosis:** {health['diagnosis']}</div>", unsafe_allow_html=True)
    
    st.subheader("Recent System Events")
    # This would be populated from a real event log
    events = [{"Time": datetime.now().strftime("%H:%M:%S"), "Event": health.get('diagnosis', 'System Check OK'), "Level": "Critical" if not health['is_healthy'] else "Info"}]
    st.dataframe(events, use_container_width=True)

# --- Main Application Logic ---
def main():
    """Main application function that orchestrates the UI."""
    
    # Fetch live data once per run
    sensor_data, prediction, health, disease_detection = get_live_data()
    
    # Update the historical data log with the latest reading
    update_historical_log(sensor_data)

    # --- Sidebar ---
    with st.sidebar:
        st.title(f"🌱 {config.SYSTEM_NAME}")
        st.markdown(f"`v{config.VERSION}`")

        # Navigation
        st.session_state.current_view = st.radio(
            "Navigation",
            ["Dashboard", "System Health", "Historical Data", "Settings"],
            key="navigation_choice"
        )
        st.divider()

        # Key Status Indicators
        st.subheader("Live Status")
        if health['is_healthy']:
            st.success("System Health: OK")
        else:
            st.error("System Health: Anomaly!")
        
        st.metric("Plant Health", f"{sensor_data.get('plant_health', 0)*100:.1f} %")
        st.metric("Water Reservoir", f"{sensor_data.get('water_level', 0):.0f} ml")
        st.progress(sensor_data.get('water_level', 0) / 1000)

        st.divider()
        # Manual Irrigation Controls
        st.subheader("Manual Controls")
        duration = st.slider("Irrigation Duration (s)", 5, 60, 10, key="manual_duration")
        if st.button("Start Manual Irrigation", use_container_width=True, disabled=sensor_data.get('irrigation_active', False)):
            st.session_state.manual_irrigation_requested = True
            st.session_state.manual_irrigation_duration = duration
            st.rerun()

    # --- Main Content Area ---
    view = st.session_state.current_view
    if view == "Dashboard":
        show_dashboard(sensor_data, prediction, health, disease_detection)
    elif view == "System Health":
        show_system_health(health)
    elif view == "Historical Data":
        st.header("Historical Data Viewer")
        st.dataframe(st.session_state.historical_data_log)
        st.line_chart(st.session_state.historical_data_log.set_index('timestamp'))
    elif view == "Settings":
        st.header("System Settings")
        st.info("Settings configuration page is under development.")

    # --- Handle Manual Irrigation Request ---
    if st.session_state.manual_irrigation_requested:
        duration = st.session_state.manual_irrigation_duration
        success = st.session_state.simulator.trigger_irrigation(duration)
        if success:
            st.toast(f"Manual irrigation started for {duration} seconds!", icon="💧")
        else:
            st.toast("Failed to start manual irrigation.", icon="❌")
        st.session_state.manual_irrigation_requested = False
        time.sleep(0.5) # Give time for the toast to appear before rerun
        st.rerun()

    # --- Auto-refresh Loop ---
    time.sleep(config.STREAMLIT_CONFIG['dashboard']['refresh_interval'])
    st.rerun()

if __name__ == "__main__":
    main()
