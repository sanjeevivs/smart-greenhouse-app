"""
UTILITIES MODULE
================

This module contains genuinely reusable utility functions and helper classes for the
greenhouse system. It serves as a central library for calculations and logic
shared across different modules, adhering to the Don't Repeat Yourself (DRY) principle.

REFACTORING NOTES:
- Removed `generate_synthetic_sensor_data` and `create_digital_twin_environment` as
  they were conflicting and redundant implementations of the simulation logic.
  The single source of truth for simulation is now the `GreenhouseEnvironment` class
  in `data_generation.py`.
- Removed `load_config` and `save_config` as they were unused. The system uses a
  direct import of the config object. A future version could use these to allow
  users to modify the config via the UI.
"""

# utils.py
import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('utils')


def get_plant_growth_stage(plant_type, days_since_planting):
    """
    Determine the plant's current growth stage based on days since planting.
    Reads parameters directly from the central config.

    Parameters:
    - plant_type (str): The type of plant (e.g., "tomato").
    - days_since_planting (int): The number of days since the plant was sowed.

    Returns:
    - tuple: (stage_index, stage_name, days_into_current_stage)
    """
    if plant_type not in config.PLANT_PARAMETERS:
        logger.warning(f"Unknown plant type: {plant_type}. Using default parameters.")
        # Fallback for unknown plant types
        return 0, "Unknown Stage", days_since_planting

    plant_params = config.PLANT_PARAMETERS[plant_type]
    growth_stages = plant_params['growth_stages']
    
    cumulative_days = 0
    for i, stage in enumerate(growth_stages):
        stage_duration = stage['duration']
        if days_since_planting < cumulative_days + stage_duration:
            days_in_stage = days_since_planting - cumulative_days
            return i, stage['name'], days_in_stage
        cumulative_days += stage_duration
    
    # If past all defined stages, remain in the last stage.
    last_stage = growth_stages[-1]
    days_in_stage = days_since_planting - (cumulative_days - last_stage['duration'])
    return len(growth_stages) - 1, last_stage['name'], days_in_stage


def calculate_optimal_moisture(plant_type, growth_stage_index):
    """
    Calculate the optimal soil moisture range for a plant at a specific growth stage.
    Adjusts the base optimal range using the stage-specific 'water_factor'.

    Parameters:
    - plant_type (str): The type of plant.
    - growth_stage_index (int): The index of the current growth stage.

    Returns:
    - tuple: (min_optimal_moisture, max_optimal_moisture)
    """
    if plant_type not in config.PLANT_PARAMETERS:
        return (60, 80)  # Return a safe default range.

    plant_params = config.PLANT_PARAMETERS[plant_type]
    growth_stages = plant_params['growth_stages']

    if growth_stage_index < len(growth_stages):
        water_factor = growth_stages[growth_stage_index]['water_factor']
    else:
        water_factor = growth_stages[-1]['water_factor'] # Use last stage if out of bounds
    
    base_min, base_max = plant_params['optimal_soil_moisture']
    
    # Adjust the range but keep it within reasonable bounds (e.g., 20% to 95%)
    adjusted_min = min(95, max(20, base_min * water_factor))
    adjusted_max = min(95, max(20, base_max * water_factor))
    
    return (adjusted_min, adjusted_max)


def calculate_evapotranspiration(temperature, humidity, wind_speed, solar_radiation):
    """
    Calculate reference evapotranspiration (ET0) using the FAO Penman-Monteith equation.
    This function provides a standardized measure of water loss to the atmosphere.

    Parameters:
    - temperature (float): Air temperature in °C.
    - humidity (float): Relative humidity in %.
    - wind_speed (float): Wind speed in m/s.
    - solar_radiation (float): Solar radiation in MJ/m²/day.

    Returns:
    - float: ET0 value in mm/day.
    """
    # Assuming standard atmospheric pressure at sea level
    atmospheric_pressure = 101.3  # kPa

    # Saturation vapor pressure
    es = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
    # Actual vapor pressure
    ea = es * (humidity / 100.0)
    
    # Slope of the saturation vapor pressure curve
    delta = (4098 * es) / (temperature + 237.3) ** 2
    
    # Psychrometric constant
    gamma = 0.000665 * atmospheric_pressure
    
    # Net radiation (simplified assumption for greenhouse)
    Rn = 0.77 * solar_radiation
    
    # Soil heat flux (negligible for daily calculations)
    G = 0
    
    numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (temperature + 273)) * wind_speed * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * wind_speed)
    
    et0 = numerator / denominator if denominator > 0 else 0
    return max(0, et0)


def detect_sensor_drift(sensor_data_series, window_hours=24, threshold=0.2):
    """
    A simple heuristic to detect potential sensor drift by comparing the mean of the
    most recent window of data to a previous window.

    Parameters:
    - sensor_data_series (pd.Series): A time-indexed series of sensor readings.
    - window_hours (int): The size of the comparison window in hours.
    - threshold (float): The fractional change required to flag a potential drift.

    Returns:
    - tuple: (drift_detected_bool, drift_amount_float)
    """
    points_per_hour = 60 // config.DATA_COLLECTION_INTERVAL
    window_points = window_hours * points_per_hour
    
    if len(sensor_data_series) < window_points * 2:
        # Not enough data to compare two full windows
        return False, 0.0
    
    recent_window = sensor_data_series.iloc[-window_points:]
    historical_window = sensor_data_series.iloc[-window_points*2:-window_points]
    
    recent_mean = recent_window.mean()
    historical_mean = historical_window.mean()
    
    historical_range = historical_window.max() - historical_window.min()
    if historical_range < 1e-5:  # Avoid division by zero for flat-lining sensors
        return abs(recent_mean - historical_mean) > 1.0, 1.0 # Drift if value changes at all
    
    drift_amount = abs(recent_mean - historical_mean) / historical_range
    return drift_amount > threshold, drift_amount


def generate_system_health_report(sensor_df):
    """
    Generates a system health report based on recent sensor data using simple heuristics.
    This is a diagnostic tool, not a replacement for the anomaly detection model.

    Parameters:
    - sensor_df (pd.DataFrame): A DataFrame of recent sensor data with a 'timestamp' column.

    Returns:
    - dict: A structured dictionary containing the health report.
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'sensor_health': {},
        'overall_status': 'OK',
        'recommendations': []
    }
    
    # Sensor health assessment
    for sensor in ['temperature', 'humidity', 'soil_moisture', 'light']:
        if sensor not in sensor_df.columns:
            continue
            
        series = sensor_df[sensor].dropna()
        if series.empty:
            continue

        # Heuristic 1: Check for stuck sensors (low variability)
        if series.std() < 0.1:
            report['sensor_health'][sensor] = 'WARNING: Low variability, sensor may be stuck.'
            report['recommendations'].append(f"Check {sensor} sensor for physical obstruction or failure.")
            report['overall_status'] = 'WARNING'
        # Heuristic 2: Check for out-of-bounds readings
        elif series.max() > 1000 and sensor != 'light': # Arbitrary large number
            report['sensor_health'][sensor] = 'CRITICAL: Sensor reading out of realistic range.'
            report['recommendations'].append(f"Calibrate or replace {sensor} sensor immediately.")
            report['overall_status'] = 'CRITICAL'
        else:
            report['sensor_health'][sensor] = 'OK'
            
    if not report['recommendations']:
        report['recommendations'].append("All systems appear to be operating within normal parameters.")

    return report


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Utility Functions ---")

    # Test plant growth stage calculation
    plant_type = "tomato"
    days_since_planting = 25
    stage_idx, stage_name, days_in_stage = get_plant_growth_stage(plant_type, days_since_planting)
    print(f"\nPlant: {plant_type}, Days since planting: {days_since_planting}")
    print(f"Result -> Growth Stage: '{stage_name}' (Index {stage_idx}), Day {days_in_stage} of stage.")
    assert stage_name == "Vegetative"

    # Test optimal moisture calculation
    optimal_moisture = calculate_optimal_moisture(plant_type, stage_idx)
    print(f"Optimal Soil Moisture for this stage: {optimal_moisture[0]:.1f}% - {optimal_moisture[1]:.1f}%")
    assert 60 <= optimal_moisture[0] <= 80

    # Test evapotranspiration calculation
    et0 = calculate_evapotranspiration(temperature=25, humidity=65, wind_speed=1.5, solar_radiation=15)
    print(f"\nReference Evapotranspiration (ET0): {et0:.2f} mm/day")
    assert 2.0 < et0 < 5.0

    # Test sensor drift detection
    # Create a series with no drift
    no_drift_series = pd.Series(np.random.normal(50, 1, 1000))
    drift_detected, drift_amount = detect_sensor_drift(no_drift_series)
    print(f"Drift Test (No Drift): Detected={drift_detected}, Amount={drift_amount:.4f}")
    assert not drift_detected

    # Create a series with clear drift
    drifted_series = pd.Series([50 + (i * 0.01) for i in range(1000)])
    drift_detected, drift_amount = detect_sensor_drift(drifted_series)
    print(f"Drift Test (With Drift): Detected={drift_detected}, Amount={drift_amount:.4f}")
    assert drift_detected
    
    print("\n--- All utility function tests passed! ---")
