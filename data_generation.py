"""
DATA GENERATION MODULE
======================

This module generates realistic simulated sensor data for the greenhouse system.
It includes:
- Realistic sensor behavior with noise and drift
- Plant growth simulation
- Environmental condition modeling (now fully dynamic based on config)
- Sensor failure simulation
- Disease progression modeling (now fully dynamic based on config)
"""

# data_generation.py
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import time
import os
from config import config
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(config.DATA_DIR, 'data_generation.log')
)
logger = logging.getLogger('data_generation')

class GreenhouseEnvironment:
    """Simulates the greenhouse environment with realistic dynamics"""

    def __init__(self, start_date=None, plant_type=config.CURRENT_PLANT_TYPE):
        self.start_date = start_date or datetime.now()
        self.current_date = self.start_date
        self.plant_type = plant_type

        # Check if the plant type from config exists, otherwise fallback.
        if plant_type not in config.PLANT_PARAMETERS:
            logger.error(f"Plant type '{plant_type}' not found in config. Halting initialization.")
            raise ValueError(f"Invalid plant type '{plant_type}' configured.")
        self.plant_parameters = config.PLANT_PARAMETERS[plant_type]

        # Initialize environmental conditions
        self.temperature = 25.0  # °C
        self.humidity = 65.0     # %
        self.light = 500.0       # lux
        self.co2 = 400.0         # ppm
        self.soil_moisture = 70.0  # %
        self.soil_ph = 6.5
        self.soil_ec = 1.2  # electrical conductivity

        # Internal state for simulation
        self._soil_moisture_history = [self.soil_moisture]
        self._temperature_history = [self.temperature]
        self._humidity_history = [self.humidity]
        self._light_history = [self.light]

        # FIX: Initialize disease factors using the new 'disease_conditions' config structure.
        # This resolves the KeyError.
        self._disease_factors = {disease: 0.0 for disease in self.plant_parameters['disease_conditions'].keys()}
        
        self._plant_health = 1.0  # 1.0 = perfect health
        self._growth_stage = 0
        self._days_in_stage = 0
        self._water_level = 1000.0  # ml in reservoir
        self._last_irrigation = None
        self._irrigation_history = []

        # Sensor calibration states
        self._sensor_drift = {
            'temperature': 0.0, 'humidity': 0.0, 'soil_moisture': 0.0,
            'light': 0.0, 'co2': 0.0, 'soil_ph': 0.0, 'soil_ec': 0.0
        }

        logger.info(f"Greenhouse environment initialized for {plant_type} starting from {self.start_date}")

    def _update_growth_stage(self):
        """Update the plant growth stage based on time elapsed"""
        total_days = (self.current_date - self.start_date).days
        cumulative_days = 0

        for i, stage in enumerate(self.plant_parameters['growth_stages']):
            cumulative_days += stage['duration']
            if total_days < cumulative_days:
                if i != self._growth_stage:
                    logger.info(f"Plant entered growth stage: {stage['name']} (Day {total_days})")
                    self._growth_stage = i
                    self._days_in_stage = total_days - (cumulative_days - stage['duration'])
                return

        last_stage_index = len(self.plant_parameters['growth_stages']) - 1
        if self._growth_stage != last_stage_index:
            last_stage_name = self.plant_parameters['growth_stages'][last_stage_index]['name']
            logger.info(f"Plant entered final growth stage: {last_stage_name}")
            self._growth_stage = last_stage_index
            self._days_in_stage = total_days - sum(s['duration'] for s in self.plant_parameters['growth_stages'][:-1])

    def _calculate_evaporation_rate(self):
        """Calculate evaporation rate based on current conditions"""
        base_rate = config.BASE_EVAPORATION_RATE
        temp_factor = 1.0 + max(0, (self.temperature - 25.0) * 0.05)
        humidity_factor = 1.0 - (self.humidity / 100.0) * 0.5
        light_factor = 1.0 + (self.light / 1000.0) * 0.3
        return base_rate * temp_factor * humidity_factor * light_factor

    def _calculate_water_absorption(self):
        """Calculate water absorption by plants based on growth stage"""
        stage = self.plant_parameters['growth_stages'][self._growth_stage]
        return stage['water_factor'] * 0.03

    def _update_disease_factors(self):
        """
        FIX: Update disease factors dynamically based on the 'disease_conditions' in the config.
        This logic is now generic and works for any plant profile.
        """
        disease_conditions = self.plant_parameters.get('disease_conditions', {})
        for disease, conditions in disease_conditions.items():
            in_temp_range = conditions['temp_range'][0] <= self.temperature <= conditions['temp_range'][1]
            above_humidity_thresh = self.humidity > conditions['humidity_threshold']
            
            # Increase disease factor if conditions are met
            if in_temp_range and above_humidity_thresh:
                self._disease_factors[disease] = min(1.0, self._disease_factors[disease] + 0.01)
        
        # Generic low light factor (increases susceptibility for all diseases)
        if self.light < 300:
            for disease in self._disease_factors:
                self._disease_factors[disease] = min(1.0, self._disease_factors[disease] + 0.005)

    def _update_plant_health(self):
        """Update plant health based on environmental conditions and disease factors"""
        health = 1.0
        params = self.plant_parameters

        min_moisture, max_moisture = params['optimal_soil_moisture']
        if self.soil_moisture < min_moisture: health -= ((min_moisture - self.soil_moisture) / min_moisture) * 0.5
        elif self.soil_moisture > max_moisture: health -= ((self.soil_moisture - max_moisture) / (100 - max_moisture)) * 0.3

        min_temp, max_temp = params['optimal_temperature']
        if self.temperature < min_temp: health -= ((min_temp - self.temperature) / min_temp) * 0.3
        elif self.temperature > max_temp: health -= ((self.temperature - max_temp) / (40 - max_temp)) * 0.4

        min_humid, max_humid = params['optimal_humidity']
        if not (min_humid <= self.humidity <= max_humid): health -= 0.1

        # Update based on new disease structure
        for disease, severity in self._disease_factors.items():
            susceptibility = params['disease_conditions'][disease]['susceptibility']
            health -= severity * susceptibility * 0.5

        self._plant_health = max(0.0, min(1.0, health))

    def _apply_sensor_drift(self, value, sensor_type):
        """Simulate sensor drift over time"""
        if sensor_type not in self._sensor_drift: return value
        drift_rate = 0.0001
        self._sensor_drift[sensor_type] += drift_rate * random.gauss(1.0, 0.3)
        drift_amount = min(0.1, self._sensor_drift[sensor_type])
        return value * (1.0 + drift_amount * random.choice([-1, 1]))

    def _add_sensor_noise(self, value, sensor_type):
        """Add realistic sensor noise"""
        noise_levels = {'temperature': 0.5, 'humidity': 2.0, 'soil_moisture': 3.0, 'light': 10.0, 'co2': 20.0, 'soil_ph': 0.1, 'soil_ec': 0.05}
        noise = random.gauss(0, noise_levels.get(sensor_type, 1.0))
        return value + noise

    def _simulate_sensor_failure(self, value, sensor_type):
        """Simulate occasional sensor failures"""
        if random.random() < config.SIMULATED_SENSOR_FAILURE_RATE:
            logger.warning(f"Simulated sensor failure for {sensor_type}")
            return None if random.random() < 0.5 else value * random.uniform(0.2, 2.0)
        return value

    def advance_time(self, minutes=5):
        """Advance the simulation by the specified number of minutes"""
        self.current_date += timedelta(minutes=minutes)
        self._days_in_stage += minutes / (24 * 60)
        self._update_growth_stage()

        moisture_loss = (self._calculate_evaporation_rate() + self._calculate_water_absorption()) * (minutes / 60.0)
        self.soil_moisture = max(0.0, self.soil_moisture - moisture_loss)

        hour = self.current_date.hour + self.current_date.minute / 60.0
        self.temperature = 22 + 5 * np.sin((hour - 8) * np.pi / 12) + random.uniform(-1.0, 1.0)
        self.humidity = max(30, min(95, 70 - (self.temperature - 22) * 2 + random.uniform(-3.0, 3.0)))
        self.light = (1000 * np.exp(-((hour - 13) ** 2) / 8) * random.uniform(0.7, 1.3)) if 6 <= hour <= 20 else 5 + random.uniform(0, 5)
        self.co2 = 400 + 50 * np.sin(hour * np.pi / 12) + random.uniform(-20, 20)

        self._update_disease_factors()
        self._update_plant_health()

    def get_sensor_readings(self):
        """Get realistic sensor readings with noise, drift, and possible failures"""
        base_readings = {
            'timestamp': self.current_date, 'temperature': self.temperature, 'humidity': self.humidity,
            'soil_moisture': self.soil_moisture, 'light': self.light, 'co2': self.co2, 'soil_ph': self.soil_ph,
            'soil_ec': self.soil_ec, 'plant_health': self._plant_health, 'growth_stage': self._growth_stage,
            'days_in_stage': self._days_in_stage, 'water_level': self._water_level, **self._disease_factors
        }
        
        final_readings = {}
        for sensor, value in base_readings.items():
            if not isinstance(value, (int, float)):
                final_readings[sensor] = value
                continue
            
            sensor_type = sensor # Default to key name
            if 'temperature' in sensor: sensor_type = 'temperature'
            elif 'humidity' in sensor: sensor_type = 'humidity'
            elif 'soil_moisture' in sensor: sensor_type = 'soil_moisture'
            elif 'soil_ph' in sensor: sensor_type = 'soil_ph'
            elif 'soil_ec' in sensor: sensor_type = 'soil_ec'
            elif 'light' in sensor: sensor_type = 'light'
            elif 'co2' in sensor: sensor_type = 'co2'
            else: # It's a derived value like plant_health, don't add noise/drift
                final_readings[sensor] = value
                continue

            value = self._apply_sensor_drift(value, sensor_type)
            value = self._add_sensor_noise(value, sensor_type)
            value = self._simulate_sensor_failure(value, sensor_type)
            final_readings[sensor] = value

        return final_readings

    def simulate_irrigation(self, duration_seconds):
        """Simulate irrigation event and its effects"""
        if self._water_level <= 0: return False
        flow_rate = 10.0  # ml/s
        water_used = min(self._water_level, flow_rate * duration_seconds)
        self._water_level -= water_used
        
        # Use the corrected parameter from config.py
        moisture_increase = water_used / config.SOIL_MOISTURE_INCREASE_FACTOR
        self.soil_moisture = min(100.0, self.soil_moisture + moisture_increase)
        
        logger.info(f"Irrigation: {duration_seconds}s, {water_used:.1f}ml used, soil moisture now {self.soil_moisture:.1f}%")
        return True

    def get_disease_image(self):
        """Generate a simulated disease image path"""
        if not self._disease_factors: return "simulated_healthy.jpg"
        max_disease = max(self._disease_factors, key=self._disease_factors.get)
        severity = self._disease_factors[max_disease]
        if severity > 0.3:
            disease_name = max_disease.replace('_', ' ').title().replace(' ', '_').lower()
            return f"simulated_{disease_name}_{int(severity*100)}.jpg"
        return "simulated_healthy.jpg"

# ... (The rest of the file - HistoricalDataGenerator and RealTimeDataSimulator -
#      does not need changes as it relies on the corrected GreenhouseEnvironment class)

class HistoricalDataGenerator:
    """Generates historical data for model training and testing"""

    def __init__(self, plant_type=config.CURRENT_PLANT_TYPE, days=365):
        self.plant_type = plant_type
        self.days = days
        self.environment = GreenhouseEnvironment(
            start_date=datetime.now() - timedelta(days=days),
            plant_type=plant_type
        )

    def generate_historical_data(self):
        """Generate historical sensor data for the specified period"""
        logger.info(f"Generating {self.days} days of historical data for {self.plant_type}")

        data = []
        total_minutes = self.days * 24 * 60
        interval = config.DATA_COLLECTION_INTERVAL

        for _ in range(0, total_minutes, interval):
            self.environment.advance_time(minutes=interval)
            readings = self.environment.get_sensor_readings()
            data.append(readings)

            if readings.get('soil_moisture') is not None and readings['soil_moisture'] < 50 and random.random() < 0.3:
                self.environment.simulate_irrigation(random.randint(10, 60))

        df = pd.DataFrame(data)
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        df['hour_of_day'] = df['timestamp'].dt.hour
        df['water_needed'] = df['soil_moisture'].shift(-1) < df['soil_moisture']
        df['water_needed'] = df['water_needed'].fillna(False).astype(bool)
        df['water_volume'] = df['soil_moisture'].apply(lambda x: max(0, 70 - x) * 0.5 if x < 70 else 0)

        logger.info(f"Generated {len(df)} historical data points")
        return df

class RealTimeDataSimulator:
    """Simulates real-time data streaming from IoT sensors"""

    def __init__(self, plant_type=config.CURRENT_PLANT_TYPE):
        self.environment = GreenhouseEnvironment(plant_type=plant_type)
        self.last_update_time = None
        self.irrigation_active = False
        self.irrigation_end_time = None

    def initialize(self):
        """Initialize the simulator with some historical context"""
        for _ in range(random.randint(100, 500)):
            self.environment.advance_time(minutes=5)
        self.last_update_time = time.time()
        logger.info("Real-time data simulator initialized")

    def get_current_readings(self):
        """Get current sensor readings, simulating real-time data"""
        current_time = time.time()
        if self.last_update_time:
            elapsed_seconds = current_time - self.last_update_time
            if elapsed_seconds > 0:
                self.environment.advance_time(minutes=elapsed_seconds / 60.0)
        self.last_update_time = current_time

        if self.irrigation_active and current_time >= self.irrigation_end_time:
            self.irrigation_active = False
            self.irrigation_end_time = None
            logger.info("Irrigation cycle completed")

        readings = self.environment.get_sensor_readings()
        readings['irrigation_active'] = self.irrigation_active
        readings['irrigation_remaining'] = max(0, self.irrigation_end_time - current_time) if self.irrigation_active else 0
        
        return readings

    def trigger_irrigation(self, duration_seconds):
        """Trigger an irrigation event (simulates system action)"""
        if self.irrigation_active: return False
        success = self.environment.simulate_irrigation(duration_seconds)
        if success:
            self.irrigation_active = True
            self.irrigation_end_time = time.time() + duration_seconds
            logger.info(f"Irrigation started for {duration_seconds} seconds")
        return success

    def get_disease_image(self):
        """Get a simulated disease image path"""
        return self.environment.get_disease_image()
