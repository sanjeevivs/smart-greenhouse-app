"""
DATA GENERATION MODULE
======================

This module generates realistic simulated sensor data for the greenhouse system.
It includes:
- Realistic sensor behavior with noise and drift
- Plant growth simulation
- Environmental condition modeling
- Sensor failure simulation
- Disease progression modeling
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
        self._disease_factors = {disease: 0.0 for disease in self.plant_parameters['disease_susceptibility']}
        self._plant_health = 1.0  # 1.0 = perfect health
        self._growth_stage = 0
        self._days_in_stage = 0
        self._water_level = 1000.0  # ml in reservoir
        self._last_irrigation = None
        self._irrigation_history = []

        # Sensor calibration states
        # FIX 1: Expanded the drift dictionary to include all relevant sensors to prevent KeyErrors.
        self._sensor_drift = {
            'temperature': 0.0,
            'humidity': 0.0,
            'soil_moisture': 0.0,
            'light': 0.0,
            'co2': 0.0,
            'soil_ph': 0.0,
            'soil_ec': 0.0
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

        # If past all stages, stay in last stage
        last_stage = len(self.plant_parameters['growth_stages']) - 1
        if self._growth_stage != last_stage:
            logger.info(f"Plant entered final growth stage: {self.plant_parameters['growth_stages'][last_stage]['name']}")
            self._growth_stage = last_stage
            self._days_in_stage = total_days - sum(s['duration'] for s in self.plant_parameters['growth_stages'][:-1])

    def _calculate_evaporation_rate(self):
        """Calculate evaporation rate based on current conditions"""
        base_rate = config.BASE_EVAPORATION_RATE

        # Temperature effect (higher temp = more evaporation)
        temp_factor = 1.0 + max(0, (self.temperature - 25.0) * 0.05)

        # Humidity effect (higher humidity = less evaporation)
        humidity_factor = 1.0 - (self.humidity / 100.0) * 0.5

        # Light effect (more light = more evaporation)
        light_factor = 1.0 + (self.light / 1000.0) * 0.3

        return base_rate * temp_factor * humidity_factor * light_factor

    def _calculate_water_absorption(self):
        """Calculate water absorption by plants based on growth stage"""
        stage = self.plant_parameters['growth_stages'][self._growth_stage]
        return stage['water_factor'] * 0.03  # base absorption rate

    def _update_disease_factors(self):
        """Update disease susceptibility factors based on environmental conditions"""
        # This logic is simplified and specific to the default 'tomato' plant
        susceptibility = self.plant_parameters['disease_susceptibility']

        if 'early_blight' in susceptibility and 22 <= self.temperature <= 28:
            self._disease_factors['early_blight'] = min(1.0, self._disease_factors['early_blight'] + 0.01)
        if 'late_blight' in susceptibility and 22 <= self.temperature <= 28:
             self._disease_factors['late_blight'] = min(1.0, self._disease_factors['late_blight'] + 0.01)

        # High humidity effect
        if self.humidity > 80:
            if 'powdery_mildew' in susceptibility:
                self._disease_factors['powdery_mildew'] = min(1.0, self._disease_factors['powdery_mildew'] + 0.02)
            if 'early_blight' in susceptibility:
                self._disease_factors['early_blight'] = min(1.0, self._disease_factors['early_blight'] + 0.01)

        # Light effect (low light = more disease)
        if self.light < 300:
            for disease in self._disease_factors:
                self._disease_factors[disease] = min(1.0, self._disease_factors[disease] + 0.01)

    def _update_plant_health(self):
        """Update plant health based on environmental conditions and disease factors"""
        health = 1.0

        # Soil moisture impact
        min_moisture, max_moisture = self.plant_parameters['optimal_soil_moisture']
        if self.soil_moisture < min_moisture:
            moisture_penalty = (min_moisture - self.soil_moisture) / min_moisture * 0.5
            health -= min(0.5, moisture_penalty)
        elif self.soil_moisture > max_moisture:
            moisture_penalty = (self.soil_moisture - max_moisture) / (100 - max_moisture) * 0.3
            health -= min(0.3, moisture_penalty)

        # Temperature impact
        min_temp, max_temp = self.plant_parameters['optimal_temperature']
        if self.temperature < min_temp:
            temp_penalty = (min_temp - self.temperature) / min_temp * 0.3
            health -= min(0.3, temp_penalty)
        elif self.temperature > max_temp:
            temp_penalty = (self.temperature - max_temp) / (40 - max_temp) * 0.4
            health -= min(0.4, temp_penalty)

        # Humidity impact
        min_humid, max_humid = self.plant_parameters['optimal_humidity']
        if self.humidity < min_humid or self.humidity > max_humid:
            humid_penalty = abs(self.humidity - (min_humid + max_humid) / 2) / 50 * 0.2
            health -= min(0.2, humid_penalty)

        # Disease impact
        for disease, severity in self._disease_factors.items():
            health -= severity * self.plant_parameters['disease_susceptibility'].get(disease, 0.0) * 0.5

        # Ensure health stays within [0, 1]
        self._plant_health = max(0.0, min(1.0, health))

    def _apply_sensor_drift(self, value, sensor_type):
        """Simulate sensor drift over time"""
        # IMPROVEMENT: Check if sensor_type is eligible for drift.
        if sensor_type not in self._sensor_drift:
            return value

        drift_rate = 0.0001  # 0.01% per minute

        # Update drift
        self._sensor_drift[sensor_type] += drift_rate * random.gauss(1.0, 0.3)

        # Apply drift (max 10% drift)
        drift_amount = min(0.1, self._sensor_drift[sensor_type])
        drifted_value = value * (1.0 + drift_amount * random.choice([-1, 1]))

        return drifted_value

    def _add_sensor_noise(self, value, sensor_type):
        """Add realistic sensor noise"""
        noise_levels = {
            'temperature': 0.5,    # ±0.5°C
            'humidity': 2.0,       # ±2%
            'soil_moisture': 3.0,  # ±3%
            'light': 10.0,         # ±10 lux
            'co2': 20.0,           # ±20 ppm
            'soil_ph': 0.1,        # ±0.1
            'soil_ec': 0.05        # ±0.05
        }

        noise = random.gauss(0, noise_levels.get(sensor_type, 1.0))
        return value + noise

    def _simulate_sensor_failure(self, value, sensor_type):
        """Simulate occasional sensor failures"""
        if random.random() < config.SIMULATED_SENSOR_FAILURE_RATE:
            logger.warning(f"Simulated sensor failure for {sensor_type}")
            # 50% chance of complete failure, 50% of erratic readings
            if random.random() < 0.5:
                return None  # Complete failure
            else:
                # Erratic readings
                return value * random.uniform(0.2, 2.0)
        return value

    def advance_time(self, minutes=5):
        """Advance the simulation by the specified number of minutes"""
        self.current_date += timedelta(minutes=minutes)
        self._days_in_stage += minutes / (24 * 60)

        # Update growth stage if needed
        self._update_growth_stage()

        # Calculate environmental changes
        evaporation_rate = self._calculate_evaporation_rate()
        water_absorption = self._calculate_water_absorption()

        # Update soil moisture (decrease due to evaporation and plant uptake)
        moisture_loss = (evaporation_rate + water_absorption) * (minutes / 60.0)
        self.soil_moisture = max(0.0, min(100.0, self.soil_moisture - moisture_loss))

        # Update other environmental factors with diurnal patterns
        hour = self.current_date.hour + self.current_date.minute / 60.0

        # Temperature follows a sine wave with daily cycle
        base_temp = 22 + 5 * np.sin((hour - 8) * np.pi / 12)
        self.temperature = base_temp + random.uniform(-1.0, 1.0)

        # Humidity is inversely related to temperature
        self.humidity = 70 - (self.temperature - 22) * 2 + random.uniform(-3.0, 3.0)
        self.humidity = max(30, min(95, self.humidity))

        # Light follows a bell curve during daylight hours
        if 6 <= hour <= 20:
            light_factor = np.exp(-((hour - 12) ** 2) / 10)
            self.light = 1000 * light_factor * (0.8 + random.uniform(0, 0.4))  # Cloud cover variation
        else:
            self.light = 5 + random.uniform(0, 5)  # Minimal night light

        # CO2 varies with plant respiration and ventilation
        self.co2 = 400 + 50 * np.sin(hour * np.pi / 12) + random.uniform(-20, 20)

        # Update disease factors
        self._update_disease_factors()

        # Update plant health
        self._update_plant_health()

        # Record history
        self._soil_moisture_history.append(self.soil_moisture)
        self._temperature_history.append(self.temperature)
        self._humidity_history.append(self.humidity)
        self._light_history.append(self.light)

        # Keep history at reasonable length
        max_history = 1440  # 24 hours at 1-minute intervals
        if len(self._soil_moisture_history) > max_history:
            self._soil_moisture_history.pop(0)
            self._temperature_history.pop(0)
            self._humidity_history.pop(0)
            self._light_history.pop(0)

    def get_sensor_readings(self):
        """Get realistic sensor readings with noise, drift, and possible failures"""
        readings = {
            'timestamp': self.current_date,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'soil_moisture': self.soil_moisture,
            'light': self.light,
            'co2': self.co2,
            'soil_ph': self.soil_ph,
            'soil_ec': self.soil_ec,
            'plant_health': self._plant_health,
            'growth_stage': self._growth_stage,
            'days_in_stage': self._days_in_stage,
            'water_level': self._water_level,
            'disease_early_blight': self._disease_factors.get('early_blight', 0.0),
            'disease_late_blight': self._disease_factors.get('late_blight', 0.0),
            'disease_powdery_mildew': self._disease_factors.get('powdery_mildew', 0.0)
        }

        # Apply sensor characteristics
        processed_readings = readings.copy()
        for sensor, value in processed_readings.items():
            # FIX 4: Only apply effects to numeric values and skip non-sensor data.
            if not isinstance(value, (int, float)):
                continue

            # FIX 3: Refined sensor type mapping for better accuracy.
            if 'temperature' in sensor:
                sensor_type = 'temperature'
            elif 'humidity' in sensor:
                sensor_type = 'humidity'
            elif 'soil_moisture' in sensor:
                sensor_type = 'soil_moisture'
            elif 'soil_ph' in sensor:
                sensor_type = 'soil_ph'
            elif 'soil_ec' in sensor:
                sensor_type = 'soil_ec'
            elif 'light' in sensor:
                sensor_type = 'light'
            elif 'co2' in sensor:
                sensor_type = 'co2'
            else:
                # These are derived values, not direct sensor readings, so we skip them.
                continue

            # Apply sensor drift
            value = self._apply_sensor_drift(value, sensor_type)

            # Add sensor noise
            value = self._add_sensor_noise(value, sensor_type)

            # Simulate possible sensor failure
            value = self._simulate_sensor_failure(value, sensor_type)

            processed_readings[sensor] = value

        return processed_readings

    def simulate_irrigation(self, duration_seconds):
        """Simulate irrigation event and its effects"""
        if self._water_level <= 0:
            logger.warning("Cannot irrigate: water reservoir empty")
            return False

        # Calculate water usage (ml/s)
        flow_rate = 10.0  # ml/s - adjust based on your system
        water_used = min(self._water_level, flow_rate * duration_seconds)

        # Update water level
        self._water_level -= water_used

        # Update soil moisture (simplified model)
        moisture_increase = water_used * config.WATER_ABSORPTION_RATE / 10.0
        self.soil_moisture = min(100.0, self.soil_moisture + moisture_increase)

        # Record irrigation event
        self._irrigation_history.append({
            'timestamp': self.current_date,
            'duration': duration_seconds,
            'water_used': water_used
        })
        self._last_irrigation = self.current_date

        logger.info(f"Irrigation event: {duration_seconds}s, used {water_used:.1f}ml, soil moisture now {self.soil_moisture:.1f}%")
        return True

    def get_disease_image(self):
        """Generate a simulated disease image path (in a real system, this would capture an actual image)"""
        # In a real system, this would trigger a camera capture
        # For simulation, we'll return a path to a pre-generated image based on disease levels

        # Determine dominant disease
        if not self._disease_factors:
            return "simulated_healthy.jpg"
            
        max_disease = max(self._disease_factors, key=self._disease_factors.get)
        severity = self._disease_factors[max_disease]

        if severity > 0.3:
            # Return path to disease image
            disease_type = max_disease.replace('_', ' ').title()
            return f"simulated_{disease_type.replace(' ', '_').lower()}_{int(severity*100)}.jpg"

        # Return path to healthy image
        return "simulated_healthy.jpg"

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
        # Generate data at 5-minute intervals
        total_minutes = self.days * 24 * 60
        interval = 5  # minutes

        for minute in range(0, total_minutes, interval):
            self.environment.advance_time(minutes=interval)
            readings = self.environment.get_sensor_readings()
            data.append(readings)

            # Simulate irrigation events based on soil moisture
            # FIX 2: Added a check for `None` to prevent TypeError on sensor failure.
            if readings['soil_moisture'] is not None and readings['soil_moisture'] < 50 and random.random() < 0.3:
                irrigation_duration = random.randint(10, 60)  # 10-60 seconds
                self.environment.simulate_irrigation(irrigation_duration)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Handle potential None values from sensor failures by forward-filling
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True) # Drop any remaining NaNs at the beginning

        # Add derived features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear

        # Add irrigation labels (simplified)
        df['water_needed'] = df['soil_moisture'].shift(-1) < df['soil_moisture']
        df['water_needed'] = df['water_needed'].fillna(False).astype(bool)

        # Add water volume (simplified)
        df['water_volume'] = df['soil_moisture'].apply(
            lambda x: max(0, 70 - x) * 0.5 if x < 70 else 0
        )

        logger.info(f"Generated {len(df)} historical data points")
        return df

    def generate_disease_data(self, num_samples=1000):
        """Generate simulated disease image metadata for training CV models"""
        logger.info(f"Generating {num_samples} disease image samples")

        samples = []
        environment = GreenhouseEnvironment(
            start_date=datetime.now() - timedelta(days=30),
            plant_type=self.plant_type
        )

        while len(samples) < num_samples:
            # Advance time randomly
            environment.advance_time(minutes=random.randint(1, 60))

            # Get sensor readings
            readings = environment.get_sensor_readings()
            
            # FIX 2 (bis): Check for None values before creating the sample to avoid errors.
            required_keys = ['timestamp', 'growth_stage', 'soil_moisture', 'temperature', 'humidity']
            if any(readings.get(key) is None for key in required_keys):
                continue # Skip this sample if critical data is missing

            # Determine if disease is present
            max_disease = max(environment._disease_factors, key=environment._disease_factors.get)
            severity = environment._disease_factors[max_disease]

            # Create sample
            sample = {
                'timestamp': readings['timestamp'],
                'image_path': environment.get_disease_image(),
                'plant_type': self.plant_type,
                'growth_stage': readings['growth_stage'],
                'soil_moisture': readings['soil_moisture'],
                'temperature': readings['temperature'],
                'humidity': readings['humidity'],
                'disease_type': max_disease if severity > 0.2 else 'healthy',
                'disease_severity': severity
            }
            samples.append(sample)

            # Simulate disease progression
            if random.random() < 0.2:
                environment._disease_factors[max_disease] = min(
                    1.0,
                    environment._disease_factors[max_disease] + random.uniform(0.05, 0.15)
                )

        logger.info(f"Generated {len(samples)} disease image samples")
        return pd.DataFrame(samples)

class RealTimeDataSimulator:
    """Simulates real-time data streaming from IoT sensors"""

    def __init__(self, plant_type=config.CURRENT_PLANT_TYPE):
        self.environment = GreenhouseEnvironment(plant_type=plant_type)
        self.last_update = None
        self.irrigation_active = False
        self.irrigation_start_time = None
        self.irrigation_duration = 0

    def initialize(self):
        """Initialize the simulator with some historical context"""
        # Advance to a point with some variation
        for _ in range(random.randint(100, 500)):
            self.environment.advance_time(minutes=5)

        self.last_update = time.time()
        logger.info("Real-time data simulator initialized")

    def get_current_readings(self):
        """Get current sensor readings, simulating real-time data"""
        current_time = time.time()

        # Advance simulation based on elapsed time
        if self.last_update:
            elapsed_seconds = current_time - self.last_update
            # Only advance if more than a second has passed to avoid excessive calls
            if elapsed_seconds > 1:
                minutes_to_advance = elapsed_seconds / 60.0
                self.environment.advance_time(minutes=minutes_to_advance)
                self.last_update = current_time

        # Check if irrigation should end
        if self.irrigation_active and self.irrigation_start_time:
            elapsed_irrigation = current_time - self.irrigation_start_time
            if elapsed_irrigation >= self.irrigation_duration:
                self.irrigation_active = False
                self.irrigation_start_time = None
                logger.info("Irrigation cycle completed")

        # Get sensor readings
        readings = self.environment.get_sensor_readings()

        # Add irrigation status
        readings['irrigation_active'] = self.irrigation_active
        irrigation_elapsed = (current_time - self.irrigation_start_time) if self.irrigation_active and self.irrigation_start_time else 0
        readings['irrigation_remaining'] = max(0, self.irrigation_duration - irrigation_elapsed)
        
        return readings

    def trigger_irrigation(self, duration_seconds):
        """Trigger an irrigation event (simulates system action)"""
        if self.irrigation_active:
            logger.warning("Cannot start new irrigation: one is already active.")
            return False
            
        if self.environment._water_level <= 0:
            logger.warning("Cannot irrigate: water reservoir empty")
            return False

        success = self.environment.simulate_irrigation(duration_seconds)
        if success:
            self.irrigation_active = True
            self.irrigation_start_time = time.time()
            self.irrigation_duration = duration_seconds
            logger.info(f"Irrigation started for {duration_seconds} seconds")
            return True
        return False

    def get_disease_image(self):
        """Get a simulated disease image path"""
        return self.environment.get_disease_image()

# Example usage
if __name__ == "__main__":
    # Generate historical data
    historical_generator = HistoricalDataGenerator(days=30)
    historical_df = historical_generator.generate_historical_data()
    historical_df.to_csv(config.SENSOR_DATA_PATH, index=False)
    print(f"Saved {len(historical_df)} historical data points to {config.SENSOR_DATA_PATH}")

    # Generate disease data
    disease_df = historical_generator.generate_disease_data(num_samples=500)
    disease_df.to_csv(os.path.join(config.DATA_DIR, "disease_data.csv"), index=False)
    print(f"Saved {len(disease_df)} disease samples to {os.path.join(config.DATA_DIR, 'disease_data.csv')}")

    # Test real-time simulator
    simulator = RealTimeDataSimulator()
    simulator.initialize()

    print("\nTesting real-time data simulator...")
    for i in range(5):
        readings = simulator.get_current_readings()
        print(f"Sample {i+1}: Soil moisture={readings.get('soil_moisture', 'N/A'):.1f}%, Temperature={readings.get('temperature', 'N/A'):.1f}°C")
        time.sleep(1)

    # Test irrigation
    print("\nTesting irrigation...")
    simulator.trigger_irrigation(10)
    for i in range(15):
        readings = simulator.get_current_readings()
        status = "Active" if readings.get('irrigation_active') else "Inactive"
        print(f"Second {i+1}: Irrigation={status}, Soil moisture={readings.get('soil_moisture', 'N/A'):.1f}%")
        time.sleep(1)
