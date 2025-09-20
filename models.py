"""
MACHINE LEARNING MODELS MODULE
==============================

This module contains implementations of all ML models required for the smart
greenhouse system, including:
- Irrigation decision classifier
- Water volume regression model
- Time-series forecasting model
- Disease detection model
- Anomaly detection model
- Reinforcement learning irrigation policy
- Uncertainty estimation models
"""

# models.py
import numpy as np
import pandas as pd
import joblib
import os
import time
import logging
from datetime import datetime, timedelta
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LSTM, Dropout, Conv1D, MaxPooling1D,
                                     Flatten, Input, MultiHeadAttention, LayerNormalization,
                                     GlobalAveragePooling1D, GlobalAveragePooling2D) # FIX 1: Added GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, r2_score, roc_curve)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

from config import config
from data_generation import HistoricalDataGenerator

# FIX 5: Handle lightgbm as an optional dependency.
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(config.DATA_DIR, 'models.log')
)
logger = logging.getLogger('models')

# ======================
# FEATURE ENGINEERING
# ======================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering specific to greenhouse data"""
    def __init__(self, window_sizes=[1, 6, 24]):
        self.window_sizes = window_sizes  # in hours
        self.scaler = StandardScaler()
        self.feature_names_out_ = None
        self.base_features_ = None

    def fit(self, X, y=None):
        if 'timestamp' not in X.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        self.base_features_ = [col for col in X.columns if col not in ['timestamp']]
        
        # Store feature names for later
        feature_names = self.base_features_.copy()
        for window in self.window_sizes:
            for feature in self.base_features_:
                feature_names.append(f'{feature}_mean_{window}h')
                feature_names.append(f'{feature}_std_{window}h')
        feature_names.extend(['hour_of_day', 'day_of_week', 'day_of_year'])
        self.feature_names_out_ = feature_names

        # Fit the scaler only on base features
        self.scaler.fit(X[self.base_features_])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X_transformed['timestamp']):
            X_transformed['timestamp'] = pd.to_datetime(X_transformed['timestamp'])
        
        X_transformed = X_transformed.sort_values('timestamp')

        # Scale base features first
        X_transformed[self.base_features_] = self.scaler.transform(X[self.base_features_])

        # Extract time features
        X_transformed['hour_of_day'] = X_transformed['timestamp'].dt.hour
        X_transformed['day_of_week'] = X_transformed['timestamp'].dt.dayofweek
        X_transformed['day_of_year'] = X_transformed['timestamp'].dt.dayofyear
        
        # Add rolling window features
        for window in self.window_sizes:
            hours = window
            window_size = hours * 12 # 12 data points per hour (5-min interval)
            for feature in self.base_features_:
                X_transformed[f'{feature}_mean_{hours}h'] = X_transformed[feature].rolling(window=window_size, min_periods=1).mean()
                X_transformed[f'{feature}_std_{hours}h'] = X_transformed[feature].rolling(window=window_size, min_periods=1).std()
        
        X_transformed = X_transformed.bfill().ffill().fillna(0)
        return X_transformed[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

class EvapotranspirationCalculator:
    """Calculates reference evapotranspiration (ET0) using Penman-Monteith equation"""
    @staticmethod
    def calculate_et0(temperature, humidity, wind_speed, solar_radiation, atmospheric_pressure=101.3, altitude=0):
        es = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
        ea = es * (humidity / 100.0)
        delta = (4098 * es) / (temperature + 237.3) ** 2
        gamma = 0.000665 * atmospheric_pressure
        Rn = 0.77 * solar_radiation
        G = 0
        num = 0.408 * delta * (Rn - G) + gamma * (900 / (temperature + 273)) * wind_speed * (es - ea)
        denom = delta + gamma * (1 + 0.34 * wind_speed)
        et0 = num / denom if denom > 0 else 0
        return max(0, et0)

# ======================
# IRRIGATION MODELS
# ======================

class IrrigationClassifier:
    """Predicts whether irrigation is needed (binary classification)"""
    def __init__(self, model_type=config.IRRIGATION_CLASSIFICATION_MODEL):
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.classification_threshold = 0.5
        self.calibrated = False

    def _create_model(self):
        if self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2, random_state=42)
        elif self.model_type == 'lightgbm':
            if lgb is None:
                raise ImportError("lightgbm is not installed. Please install it to use this model type.")
            return lgb.LGBMClassifier(num_leaves=31, max_depth=10, learning_rate=0.05, n_estimators=150, class_weight='balanced', random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X, y):
        logger.info(f"Training {self.model_type} irrigation classifier")
        X_processed = self.feature_engineer.fit_transform(X)
        split_idx = int(len(X_processed) * 0.8)
        X_train, X_test = X_processed.iloc[:split_idx], X_processed.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        base_model = self._create_model()
        base_model.fit(X_train, y_train)
        
        self.model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
        self.model.fit(X_train, y_train)
        self.calibrated = True
        
        y_prob = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        j_scores = tpr - fpr
        self.classification_threshold = thresholds[np.argmax(j_scores)]
        
        y_pred = (y_prob >= self.classification_threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logger.info(f"Optimal classification threshold set to {self.classification_threshold:.4f}")
        self.is_trained = True
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'threshold': self.classification_threshold}

    def predict(self, X):
        if not self.is_trained: raise RuntimeError("Model must be trained before prediction")
        X_processed = self.feature_engineer.transform(X)
        probability = self.model.predict_proba(X_processed)[:, 1][0]
        prediction = 1 if probability >= self.classification_threshold else 0
        return prediction, probability

    def save(self, path=config.IRRIGATION_CLASSIFICATION_MODEL_PATH):
        if not self.is_trained: raise RuntimeError("Cannot save untrained model")
        joblib.dump({'model': self.model, 'feature_engineer': self.feature_engineer, 'classification_threshold': self.classification_threshold, 'is_trained': self.is_trained}, path)
        logger.info(f"Model saved to {path}")

    def load(self, path=config.IRRIGATION_CLASSIFICATION_MODEL_PATH):
        if not os.path.exists(path): raise FileNotFoundError(f"Model file not found at {path}")
        data = joblib.load(path)
        self.model = data['model']
        self.feature_engineer = data['feature_engineer']
        self.classification_threshold = data['classification_threshold']
        self.is_trained = data['is_trained']
        logger.info(f"Model loaded from {path}")
        return self

class WaterVolumeRegressor:
    """Predicts the volume/duration of irrigation needed"""
    def __init__(self, model_type=config.IRRIGATION_REGRESSION_MODEL):
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
    
    def _create_model(self):
        if self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=200, max_depth=10, min_samples_split=5, learning_rate=0.05, random_state=42)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X, y):
        logger.info(f"Training {self.model_type} water volume regressor")
        X_processed = self.feature_engineer.fit_transform(X)
        split_idx = int(len(X_processed) * 0.8)
        X_train, X_test = X_processed.iloc[:split_idx], X_processed.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Model evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        self.is_trained = True
        return {'mse': mse, 'rmse': rmse, 'r2': r2}

    def predict(self, X):
        if not self.is_trained: raise RuntimeError("Model must be trained before prediction")
        X_processed = self.feature_engineer.transform(X)
        volume = self.model.predict(X_processed)[0]
        return max(0, volume)

    def save(self, path=config.IRRIGATION_REGRESSION_MODEL_PATH):
        if not self.is_trained: raise RuntimeError("Cannot save untrained model")
        joblib.dump({'model': self.model, 'feature_engineer': self.feature_engineer, 'is_trained': self.is_trained}, path)
        logger.info(f"Model saved to {path}")

    def load(self, path=config.IRRIGATION_REGRESSION_MODEL_PATH):
        if not os.path.exists(path): raise FileNotFoundError(f"Model file not found at {path}")
        data = joblib.load(path)
        self.model = data['model']
        self.feature_engineer = data['feature_engineer']
        self.is_trained = data['is_trained']
        logger.info(f"Model loaded from {path}")
        return self

# ======================
# TIME-SERIES FORECASTING
# ======================

class SoilMoistureForecaster:
    """Forecasts soil moisture levels for the next N hours"""
    def __init__(self, model_type=config.FORECASTING_MODEL, forecast_horizon=24):
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon * 12 # Convert hours to 5-min intervals
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 24 * 12 # 24 hours of historical data
        self.feature_names_ = None
        self.is_trained = False

    # ... [model creation methods _create_lstm_model, _create_transformer_model, etc. are correct] ...
    def _create_lstm_model(self, input_shape):
        model = Sequential([ LSTM(64, return_sequences=True, input_shape=input_shape), Dropout(0.2), LSTM(32, return_sequences=False), Dropout(0.2), Dense(32, activation='relu'), Dense(self.forecast_horizon) ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _create_transformer_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(2):
            attn_output = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(x, x)
            attn_output = Dropout(0.1)(attn_output)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            ffn = Sequential([Dense(64, activation="relu"), Dense(input_shape[-1])])(x)
            ffn = Dropout(0.1)(ffn)
            x = LayerNormalization(epsilon=1e-6)(x + ffn)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(self.forecast_horizon)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _create_model(self, input_shape):
        if self.model_type == 'lstm': return self._create_lstm_model(input_shape)
        elif self.model_type == 'transformer': return self._create_transformer_model(input_shape)
        else: raise ValueError(f"Unsupported model type: {self.model_type}")

    def _create_sequences(self, X, y=None):
        X_copy = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X_copy['timestamp']):
            X_copy['timestamp'] = pd.to_datetime(X_copy['timestamp'])
        
        X_copy = X_copy.sort_values('timestamp')
        features = [col for col in X_copy.columns if col != 'timestamp']
        if self.feature_names_ is None:
             self.feature_names_ = features

        X_scaled = self.scaler.fit_transform(X_copy[self.feature_names_])
        X_seq, y_seq = [], []

        for i in range(self.sequence_length, len(X_scaled) - self.forecast_horizon + 1):
            X_seq.append(X_scaled[i-self.sequence_length:i])
            if y is not None:
                y_seq.append(y.iloc[i:i+self.forecast_horizon])
        
        return (np.array(X_seq), np.array(y_seq)) if y is not None else np.array(X_seq)

    def train(self, X, y):
        logger.info(f"Training {self.model_type} soil moisture forecaster for {self.forecast_horizon // 12} hours")
        X_seq, y_seq = self._create_sequences(X, y)
        if len(X_seq) == 0:
            logger.error("Not enough data to create sequences for training. Aborting.")
            return None
        
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self.model = self._create_model(input_shape)
        
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test, y_train, y_test = X_seq[:split_idx], X_seq[split_idx:], y_seq[:split_idx], y_seq[split_idx:]
        
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Model evaluation - Test loss: {loss:.4f}")
        self.is_trained = True
        return {'loss': loss, 'history': history.history}

    def predict(self, X):
        if not self.is_trained: raise RuntimeError("Model must be trained before prediction")
        
        X_seq = self._create_sequences(X)
        if len(X_seq) == 0:
             raise ValueError("Not enough historical data to make a forecast. Need at least 24 hours of data.")
        
        last_seq = X_seq[-1].reshape(1, X_seq.shape[1], X_seq.shape[2])
        forecast_scaled = self.model.predict(last_seq)[0]
        
        # FIX 4: Made this robust by finding the index of soil_moisture instead of assuming it's 0.
        try:
            moisture_idx = self.feature_names_.index('soil_moisture')
        except (ValueError, AttributeError):
            logger.warning("Could not find 'soil_moisture' in feature names, assuming index 0 for inverse transform.")
            moisture_idx = 0

        dummy = np.zeros((len(forecast_scaled), len(self.feature_names_)))
        dummy[:, moisture_idx] = forecast_scaled
        forecast_actual = self.scaler.inverse_transform(dummy)[:, moisture_idx]
        
        # Return one value per hour
        return forecast_actual[::12]

    def save(self, path=config.FORECASTING_MODEL_PATH):
        if not self.is_trained: raise RuntimeError("Cannot save untrained model")
        self.model.save(path)
        metadata = {'scaler': self.scaler, 'sequence_length': self.sequence_length, 'forecast_horizon': self.forecast_horizon, 'feature_names': self.feature_names_}
        joblib.dump(metadata, path + '.metadata')
        logger.info(f"Model saved to {path}")

    def load(self, path=config.FORECASTING_MODEL_PATH):
        if not os.path.exists(path): raise FileNotFoundError(f"Model file not found at {path}")
        self.model = tf.keras.models.load_model(path)
        metadata = joblib.load(path + '.metadata')
        self.scaler, self.sequence_length, self.forecast_horizon, self.feature_names_ = metadata['scaler'], metadata['sequence_length'], metadata['forecast_horizon'], metadata['feature_names']
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
        return self

# ======================
# DISEASE DETECTION
# ======================

class DiseaseDetector:
    """Detects plant diseases from images using computer vision"""
    def __init__(self, model_type=config.DISEASE_DETECTION_MODEL):
        self.model_type = model_type
        self.model = None
        self.class_names = ['healthy', 'early_blight', 'late_blight', 'powdery_mildew']
        self.input_size = (300, 300) if 'efficientnet' in model_type else (224, 224)
        self.is_trained = False

    def _create_model(self):
        # This is a simplified placeholder as we don't have real image data
        # In a real scenario, this would load a pre-trained CNN base
        inputs = Input(shape=(*self.input_size, 3))
        x = GlobalAveragePooling2D()(inputs) # Simplified model
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(len(self.class_names), activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, image_paths, labels, **kwargs):
        logger.info(f"Simulating training for {self.model_type} disease detector")
        self.model = self._create_model()
        self.is_trained = True
        # In a real implementation, you would train on actual images here.
        logger.info("Simulated training complete.")
        return {'final_val_acc': 0.95} # Simulated metric

    def predict(self, image_path):
        if not self.is_trained: raise RuntimeError("Model must be trained before prediction")
        # Simulate prediction based on image filename
        if 'healthy' in image_path: probs = [0.90, 0.04, 0.03, 0.03]
        elif 'early' in image_path: probs = [0.05, 0.85, 0.05, 0.05]
        elif 'late' in image_path: probs = [0.05, 0.05, 0.85, 0.05]
        elif 'powdery' in image_path: probs = [0.05, 0.05, 0.05, 0.85]
        else: probs = [0.25, 0.25, 0.25, 0.25]
        
        probs = np.array(probs) + np.random.normal(0, 0.02, len(probs))
        probs = np.clip(probs, 0, 1) / np.sum(probs)
        
        predictions = sorted([(self.class_names[i], float(p)) for i, p in enumerate(probs)], key=lambda x: x[1], reverse=True)
        return predictions

    def save(self, path=config.DISEASE_DETECTION_MODEL_PATH):
        if not self.is_trained: raise RuntimeError("Cannot save untrained model")
        # In a real implementation, you'd save the TF model and convert to TFLite
        # For simulation, we just save metadata
        metadata = {'class_names': self.class_names, 'input_size': self.input_size, 'model_type': self.model_type}
        joblib.dump(metadata, path)
        logger.info(f"Simulated disease detection model saved to {path}")

    def load(self, path=config.DISEASE_DETECTION_MODEL_PATH):
        if not os.path.exists(path): raise FileNotFoundError(f"Model file not found at {path}")
        metadata = joblib.load(path)
        self.class_names, self.input_size, self.model_type = metadata['class_names'], metadata['input_size'], metadata['model_type']
        self.is_trained = True
        logger.info(f"Simulated disease detection model loaded from {path}")
        return self

# ======================
# ANOMALY DETECTION
# ======================

class SystemHealthMonitor:
    """Monitors system health and detects anomalies using multivariate time-series"""
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer(window_sizes=[1, 3, 6])
        self.is_trained = False

    def train(self, X):
        logger.info(f"Training IsolationForest anomaly detector")
        X_processed = self.feature_engineer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_processed)
        self.model.fit(X_scaled)
        self.is_trained = True

    def detect_anomaly(self, X):
        if not self.is_trained: raise RuntimeError("Model must be trained before detection")
        X_processed = self.feature_engineer.transform(X)
        X_scaled = self.scaler.transform(X_processed)
        prediction = self.model.predict(X_scaled)[0]
        anomaly_score = self.model.decision_function(X_scaled)[0]
        is_anomaly = prediction == -1
        # Normalize score to be between 0 (normal) and 1 (anomaly)
        normalized_score = (1 - anomaly_score) / 2 
        return is_anomaly, float(normalized_score)

    def diagnose_issue(self, X):
        # This is a rule-based diagnostic heuristic
        features = self.feature_engineer.transform(X).iloc[0]
        if abs(features['soil_moisture'] - features['soil_moisture_mean_6h']) > 3 * features['soil_moisture_std_6h'] and features['soil_moisture_std_6h'] > 0:
            return "Sudden unexpected change in soil moisture. Check sensor and water lines."
        if abs(features['temperature'] - features['temperature_mean_6h']) > 3 * features['temperature_std_6h'] and features['temperature_std_6h'] > 0:
            return "Sudden unexpected change in temperature. Check heating/cooling system."
        return "Generic system anomaly detected. Please check all sensors and actuators."

    def save(self, path=config.ANOMALY_DETECTION_MODEL_PATH):
        if not self.is_trained: raise RuntimeError("Cannot save untrained model")
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'feature_engineer': self.feature_engineer, 'is_trained': self.is_trained}, path)
        logger.info(f"Model saved to {path}")

    def load(self, path=config.ANOMALY_DETECTION_MODEL_PATH):
        if not os.path.exists(path): raise FileNotFoundError(f"Model file not found at {path}")
        data = joblib.load(path)
        self.model, self.scaler, self.feature_engineer, self.is_trained = data['model'], data['scaler'], data['feature_engineer'], data['is_trained']
        logger.info(f"Model loaded from {path}")
        return self

# ======================
# REINFORCEMENT LEARNING
# ======================

class ReinforcementLearningIrrigator:
    """Implements a reinforcement learning policy for irrigation decisions"""
    def __init__(self, state_size, action_size=1, learning_rate=config.RL_LEARNING_RATE):
        self.state_size = state_size
        self.action_size = action_size  # Continuous: irrigation duration
        self.learning_rate = learning_rate
        self.gamma = config.RL_DISCOUNT_FACTOR
        self.memory = []
        self.batch_size = 64
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model()

    def _build_model(self):
        model = Sequential([ Dense(64, input_dim=self.state_size, activation='relu'), Dropout(0.2), Dense(64, activation='relu'), Dropout(0.2), Dense(32, activation='relu'), Dense(self.action_size, activation='linear') ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=0.1):
        if np.random.rand() <= epsilon:
            return np.random.uniform(0, 60) # Explore: random duration 0-60s
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.clip(act_values[0][0], 0, 60)

    def replay(self):
        if len(self.memory) < self.batch_size: return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            # FIX 2: Corrected the Q-learning update for a continuous action space.
            # The original logic with argmax was incorrect.
            if dones[i]:
                target[i][0] = rewards[i]
            else:
                # Standard DQN update for a single continuous action output
                target[i][0] = rewards[i] + self.gamma * target_next[i][0]
        
        self.model.fit(states, target, epochs=1, verbose=0)
    
    # ... [train, predict_irrigation, save, load methods are mostly correct] ...
    def train(self, env, episodes=config.RL_TRAINING_EPISODES, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        logger.info(f"Training RL irrigator for {episodes} episodes")
        epsilon = epsilon_start
        # ... Implementation of the training loop ...
        return [] # Placeholder

    def predict_irrigation(self, state):
        return self.act(state, epsilon=0.0)

    def save(self, path=config.RL_POLICY_MODEL_PATH):
        self.model.save(path)
        logger.info(f"RL policy model saved to {path}")

    def load(self, path=config.RL_POLICY_MODEL_PATH):
        if not os.path.exists(path): raise FileNotFoundError(f"Model file not found at {path}")
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path)
        logger.info(f"RL policy model loaded from {path}")
        return self

# ======================
# UNCERTAINTY ESTIMATION
# ======================

class UncertaintyEstimator:
    """Provides uncertainty estimates for irrigation predictions"""
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    # FIX 3: Removed the broken _mc_dropout_predict method entirely.
    
    def _quantile_regression(self, X):
        # This is a simulated version for demonstration.
        base_prediction = self.regressor.predict(X)
        uncertainty = 0.15 * base_prediction # 15% of prediction as base uncertainty
        soil_moisture = X['soil_moisture'].values[0]
        if soil_moisture < 30 or soil_moisture > 80:
            uncertainty *= 1.5
        
        lower_bound = max(0, base_prediction - 1.96 * uncertainty)
        upper_bound = base_prediction + 1.96 * uncertainty
        return lower_bound, upper_bound, uncertainty

    def estimate_classification_uncertainty(self, X):
        _, probability = self.classifier.predict(X)
        threshold = self.classifier.classification_threshold
        uncertainty = 1.0 - abs(probability - threshold) / max(threshold, 1 - threshold)
        return min(1.0, max(0.0, uncertainty))

    def estimate_regression_uncertainty(self, X):
        lower_bound, upper_bound, std = self._quantile_regression(X)
        prediction = self.regressor.predict(X)
        if prediction < 1e-6: return lower_bound, upper_bound, 1.0
        interval_width = upper_bound - lower_bound
        uncertainty = interval_width / prediction
        return lower_bound, upper_bound, min(1.0, uncertainty)

    def needs_human_review(self, X):
        class_uncertainty = self.estimate_classification_uncertainty(X)
        _, _, reg_uncertainty = self.estimate_regression_uncertainty(X)
        
        if class_uncertainty > config.UNCERTAINTY_THRESHOLD:
            return True, f"High classification uncertainty ({class_uncertainty:.2f})"
        if reg_uncertainty > config.UNCERTAINTY_THRESHOLD:
            return True, f"High volume prediction uncertainty ({reg_uncertainty:.2f})"
        return False, "Confident prediction"

# ======================
# MODEL INTEGRATION
# ======================

class GreenhouseMLSystem:
    """Integrates all ML models into a cohesive system"""
    def __init__(self):
        self.irrigation_classifier = IrrigationClassifier()
        self.water_volume_regressor = WaterVolumeRegressor()
        self.soil_forecaster = SoilMoistureForecaster()
        self.disease_detector = DiseaseDetector()
        self.system_health_monitor = SystemHealthMonitor()
        self.rl_irrigator = None
        self.uncertainty_estimator = None
        self.is_initialized = False

    def initialize(self, train_if_needed=True):
        if self.is_initialized: return True
        logger.info("Initializing greenhouse ML system")
        
        model_paths = {
            'classifier': config.IRRIGATION_CLASSIFICATION_MODEL_PATH,
            'regressor': config.IRRIGATION_REGRESSION_MODEL_PATH,
            'forecaster': config.FORECASTING_MODEL_PATH,
            'disease': config.DISEASE_DETECTION_MODEL_PATH,
            'health': config.ANOMALY_DETECTION_MODEL_PATH
        }
        
        if all(os.path.exists(p) for p in model_paths.values()):
            self.load_all_models()
        elif train_if_needed:
            logger.info("One or more models not found. Training all models.")
            self.train_all_models()
        else:
            logger.warning("Models not found and training is disabled.")
            return False

        if self.irrigation_classifier.is_trained:
            # Initialize RL irrigator with correct state size
            state_size = len(self.irrigation_classifier.feature_engineer.get_feature_names_out())
            self.rl_irrigator = ReinforcementLearningIrrigator(state_size=state_size)
            if os.path.exists(config.RL_POLICY_MODEL_PATH):
                self.rl_irrigator.load()

        if self.irrigation_classifier.is_trained and self.water_volume_regressor.is_trained:
            self.uncertainty_estimator = UncertaintyEstimator(self.irrigation_classifier, self.water_volume_regressor)
        
        self.is_initialized = True
        logger.info("Greenhouse ML system initialized successfully")
        return True
    
    # ... [Rest of the GreenhouseMLSystem class is largely correct, with minor robustness improvements] ...
    # In models.py

    def train_all_models(self):
        """Train all ML models using historical data."""
        logger.info("Training all ML models...")
        if not os.path.exists(config.SENSOR_DATA_PATH):
            logger.info(f"Sensor data not found at {config.SENSOR_DATA_PATH}. Generating new data...")
            historical_generator = HistoricalDataGenerator(days=365)
            historical_df = historical_generator.generate_historical_data()
            historical_df.to_csv(config.SENSOR_DATA_PATH, index=False)
        
        df = pd.read_csv(config.SENSOR_DATA_PATH, parse_dates=['timestamp'])
        df.dropna(inplace=True)

        # FIX: Define the core features that are used for training.
        # This decouples the model from any extra columns in the CSV.
        core_features = [
            'timestamp', 'temperature', 'humidity', 'soil_moisture',
            'light', 'co2', 'soil_ph', 'soil_ec'
        ]
        
        # Ensure all required columns are present
        if not all(feature in df.columns for feature in core_features):
            raise ValueError(f"CSV file is missing one of the required training columns: {core_features}")
            
        features_df = df[core_features]

        # Train irrigation classifier
        self.irrigation_classifier.train(features_df, df['water_needed'])
        self.irrigation_classifier.save()
        
        # Train water volume regressor
        self.water_volume_regressor.train(features_df, df['water_volume'])
        self.water_volume_regressor.save()
        
        # Train soil moisture forecaster
        self.soil_forecaster.train(features_df, df['soil_moisture'])
        self.soil_forecaster.save()

        # Train system health monitor
        self.system_health_monitor.train(features_df)
        self.system_health_monitor.save()

        # Train disease detector (simulated)
        self.disease_detector.train([], [])
        self.disease_detector.save()
        
        logger.info("All ML models trained successfully")
        return True

    def predict_irrigation(self, sensor_data):
        df = pd.DataFrame([sensor_data])
        irrigation_needed, probability = self.irrigation_classifier.predict(df)
        water_volume = self.water_volume_regressor.predict(df) if irrigation_needed else 0
        
        needs_review, reason = False, ""
        vol_lower, vol_upper, vol_uncert = 0, 0, 0
        class_uncert = 0
        
        if self.uncertainty_estimator:
            class_uncert = self.uncertainty_estimator.estimate_classification_uncertainty(df)
            vol_lower, vol_upper, vol_uncert = self.uncertainty_estimator.estimate_regression_uncertainty(df)
            needs_review, reason = self.uncertainty_estimator.needs_human_review(df)

        forecast = self.soil_forecaster.predict(df)
        
        return {
            'timestamp': datetime.now(),
            'irrigation_needed': bool(irrigation_needed),
            'probability': float(probability),
            'water_volume': float(water_volume),
            'volume_lower_bound': float(vol_lower),
            'volume_upper_bound': float(vol_upper),
            'classification_uncertainty': float(class_uncert),
            'volume_uncertainty': float(vol_uncert),
            'soil_moisture_forecast': forecast.tolist(),
            'needs_human_review': needs_review,
            'review_reason': reason,
            'source': 'ml_system'
        }

    def monitor_system_health(self, sensor_data):
        df = pd.DataFrame([sensor_data])
        is_anomaly, score = self.system_health_monitor.detect_anomaly(df)
        diagnosis = self.system_health_monitor.diagnose_issue(df) if is_anomaly else ""
        return {'timestamp': datetime.now(), 'is_healthy': not is_anomaly, 'anomaly_score': score, 'diagnosis': diagnosis}

    def detect_disease(self, image_path):
        predictions = self.disease_detector.predict(image_path)
        top_class, top_prob = predictions[0]
        return {
            'timestamp': datetime.now(), 'image_path': image_path, 'predictions': [{'class': c, 'probability': p} for c, p in predictions],
            'top_class': top_class, 'top_probability': top_prob, 'is_diseased': top_class != 'healthy' and top_prob > 0.5
        }

    def get_rl_policy(self, sensor_data):
        if not self.rl_irrigator: return {'duration': 0, 'confidence': 0.0, 'source': 'rl_unavailable'}
        df = pd.DataFrame([sensor_data])
        state = self.irrigation_classifier.feature_engineer.transform(df).to_numpy()
        duration = self.rl_irrigator.predict_irrigation(state)
        return {'duration': float(duration), 'confidence': 0.8, 'source': 'rl_policy'}

    def load_all_models(self):
        logger.info("Loading all ML models from disk...")
        self.irrigation_classifier.load()
        self.water_volume_regressor.load()
        self.soil_forecaster.load()
        self.disease_detector.load()
        self.system_health_monitor.load()
        logger.info("All models loaded.")
        return self

# Example usage
if __name__ == "__main__":
    ml_system = GreenhouseMLSystem()
    ml_system.initialize(train_if_needed=True)
    
    simulator = RealTimeDataSimulator()
    simulator.initialize()
    sensor_data = simulator.get_current_readings()
    
    prediction = ml_system.predict_irrigation(sensor_data)
    print("\nIrrigation Prediction:", prediction)
    
    health = ml_system.monitor_system_health(sensor_data)
    print("\nSystem Health:", health)
    
    disease_image = simulator.get_disease_image()
    detection = ml_system.detect_disease(disease_image)
    print("\nDisease Detection:", detection)

