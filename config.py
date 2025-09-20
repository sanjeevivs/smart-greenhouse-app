"""
SMART GREENHOUSE IRRIGATION SYSTEM WITH ADVANCED ML INTEGRATION
=============================================================

This file centralizes all configuration settings for the system.
By modifying parameters here, you can change the behavior of the entire
application without altering the core logic in other modules.
"""

# ======================
# CONFIGURATION SETTINGS
# ======================

# config.py
import os
from datetime import datetime, timedelta

# --- System configuration ---
class Config:
    """A single class to hold all system configuration settings."""

    # 1. Basic System Settings
    SYSTEM_NAME = "SmartGreenhouse v2.0"
    VERSION = "2.3.1"
    TIMEZONE = "UTC"

    # 2. Data & Simulation Settings
    DATA_COLLECTION_INTERVAL = 5  # minutes
    HISTORICAL_DATA_RETENTION = 365  # days
    SIMULATED_SENSOR_FAILURE_RATE = 0.005  # 0.5% chance of sensor failure per reading

    # 3. Model Selection
    # Allows for easy swapping of ML model types. Must match options in `models.py`.
    IRRIGATION_CLASSIFICATION_MODEL = "random_forest"
    IRRIGATION_REGRESSION_MODEL = "gradient_boosting"
    FORECASTING_MODEL = "transformer"
    DISEASE_DETECTION_MODEL = "efficientnet_b3" # This is simulated in models.py

    # 4. Machine Learning Hyperparameters
    # 4.1 Reinforcement Learning Settings
    RL_TRAINING_EPISODES = 1000
    RL_MAX_STEPS = 288  # A full day at 5-minute intervals (24 * 60 / 5)
    RL_DISCOUNT_FACTOR = 0.99
    RL_LEARNING_RATE = 0.001
    RL_REWARD_YIELD_FACTOR = 1.0    # Reward for maintaining high plant health
    RL_REWARD_WATER_FACTOR = -0.3   # Penalty for water usage
    RL_REWARD_ENERGY_FACTOR = -0.2  # Penalty for energy usage (e.g., running the pump)

    # 4.2 Anomaly Detection Settings
    ANOMALY_CONTAMINATION = 0.02  # Expected percentage of anomalies in the data (used by IsolationForest)
    ANOMALY_HISTORY_WINDOW = 288  # Look back at 24 hours of data for context

    # 4.3 Uncertainty Settings
    UNCERTAINTY_THRESHOLD = 0.30  # If uncertainty score is above this, flag for human review.
    # NOTE: The following parameters from the original file were unused and have been removed to avoid confusion.
    # - ANOMALY_THRESHOLD: The anomaly model calculates its own internal threshold based on `ANOMALY_CONTAMINATION`.
    # - SENSOR_DRIFT_THRESHOLD: This logic is not implemented in the models; drift is part of the simulation.
    # - HUMAN_IN_LOOP_MIN_CONFIDENCE: This is redundant; `UNCERTAINTY_THRESHOLD` serves this purpose.

    # 5. File System Paths
    # Define paths relative to this file's location for robustness.
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(_BASE_DIR, "data")
    MODEL_DIR = os.path.join(_BASE_DIR, "models")
    IMAGE_DIR = os.path.join(_BASE_DIR, "images")

    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 5.1 Specific File Paths
    SENSOR_DATA_PATH = os.path.join(DATA_DIR, "sensor_data.csv")
    IRRIGATION_LOG_PATH = os.path.join(DATA_DIR, "irrigation_log.csv")
    DISEASE_LOG_PATH = os.path.join(DATA_DIR, "disease_log.csv")
    SYSTEM_HEALTH_PATH = os.path.join(DATA_DIR, "system_health.csv")

    # 5.2 Model File Paths
    IRRIGATION_CLASSIFICATION_MODEL_PATH = os.path.join(MODEL_DIR, "irrigation_classifier.joblib")
    IRRIGATION_REGRESSION_MODEL_PATH = os.path.join(MODEL_DIR, "irrigation_regressor.joblib")
    FORECASTING_MODEL_PATH = os.path.join(MODEL_DIR, "soil_forecaster.h5")
    DISEASE_DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, "disease_detector.joblib")
    ANOMALY_DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_detector.joblib")
    RL_POLICY_MODEL_PATH = os.path.join(MODEL_DIR, "rl_policy_model.h5")

    # 6. Simulation & Digital Twin Parameters
    BASE_EVAPORATION_RATE = 0.05      # % moisture loss per hour at 25°C
    # FIX: Renamed for clarity. This factor determines how much 1ml of water increases soil moisture %.
    # The calculation is: moisture_increase = (water_used_ml / this_factor)
    SOIL_MOISTURE_INCREASE_FACTOR = 100.0 # e.g., 100ml water increases moisture by 1%

    # 7. Plant-Specific Profiles
    # This structure is now fully self-contained, allowing data_generation.py to dynamically handle any plant.
    PLANT_PARAMETERS = {
        "tomato": {
            "optimal_soil_moisture": (60, 80), # min/max %
            "optimal_temperature": (20, 28),   # min/max °C
            "optimal_humidity": (60, 80),      # min/max %
            "growth_stages": [
                # 'water_factor' modifies the plant's water consumption rate at each stage.
                {"name": "Seedling", "duration": 14, "water_factor": 0.7},
                {"name": "Vegetative", "duration": 21, "water_factor": 1.0},
                {"name": "Flowering", "duration": 14, "water_factor": 1.2},
                {"name": "Fruiting", "duration": 28, "water_factor": 1.1}
            ],
            # NEW: Self-contained disease simulation rules.
            "disease_conditions": {
                "early_blight":   {"temp_range": (22, 28), "humidity_threshold": 80, "susceptibility": 0.3},
                "late_blight":    {"temp_range": (18, 24), "humidity_threshold": 85, "susceptibility": 0.4},
                "powdery_mildew": {"temp_range": (20, 30), "humidity_threshold": 90, "susceptibility": 0.2}
            }
        },
        "lettuce": {
            "optimal_soil_moisture": (70, 90),
            "optimal_temperature": (15, 22),
            "optimal_humidity": (65, 85),
            "growth_stages": [
                {"name": "Seedling", "duration": 7, "water_factor": 0.8},
                {"name": "Young", "duration": 14, "water_factor": 1.0},
                {"name": "Mature", "duration": 21, "water_factor": 0.9}
            ],
            "disease_conditions": {
                "downy_mildew": {"temp_range": (10, 18), "humidity_threshold": 90, "susceptibility": 0.5},
                "botrytis":     {"temp_range": (15, 25), "humidity_threshold": 88, "susceptibility": 0.3},
                "root_rot":     {"temp_range": (18, 28), "humidity_threshold": 95, "susceptibility": 0.4} # Favored by overwatering
            }
        }
    }

    # 8. Current Greenhouse Setup
    CURRENT_PLANT_TYPE = "tomato"
    # FIX: Made the planting date dynamic for more realistic simulations.
    PLANTING_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # 9. Hardware & UI Configuration (Placeholders)
    HARDWARE_CONFIG = {
        "esp32": {"baud_rate": 115200, "wifi_ssid": "GreenhouseNetwork"},
        "sensors": {"dht11": {"pin": "GPIO4"}, "soil_moisture": {"pin": "A0"}},
        "actuators": {"water_pump": {"pin": "GPIO5"}}
    }

    STREAMLIT_CONFIG = {
        "dashboard": {"refresh_interval": 5, "history_days": 7}
    }

    # 10. Federated Learning Configuration
    FEDERATED_LEARNING = {
        "enabled": False, # Disabled by default for simplicity
        "server_url": "https://federated-greenhouse.example.com",
        "update_frequency": "daily"
    }

# Create a singleton instance for easy import across the project
config = Config()

# --- Verification Block ---
if __name__ == "__main__":
    print(f"--- Configuration Loaded for {config.SYSTEM_NAME} v{config.VERSION} ---")
    print(f"Data Directory: {config.DATA_DIR}")
    print(f"Model Directory: {config.MODEL_DIR}")
    print("-" * 20)
    print(f"Current Plant: {config.CURRENT_PLANT_TYPE}")
    print(f"Planting Date: {config.PLANTING_DATE}")
    print(f"Irrigation Classifier Model: {config.IRRIGATION_CLASSIFICATION_MODEL}")
    print(f"Soil Forecaster Model: {config.FORECASTING_MODEL}")
    print("-" * 20)
    # Verify that the current plant type exists in the parameters
    if config.CURRENT_PLANT_TYPE not in config.PLANT_PARAMETERS:
        print(f"\033[91mERROR: Current plant type '{config.CURRENT_PLANT_TYPE}' not found in PLANT_PARAMETERS!\033[0m")
    else:
        print("\033[92mConfiguration appears valid.\033[0m")
