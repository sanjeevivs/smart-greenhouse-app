# train_models.py
from data_generation import HistoricalDataGenerator
from models import GreenhouseMLSystem
import os

print("Initializing Greenhouse ML System...")
ml_system = GreenhouseMLSystem()

print("Checking for existing models...")
if not os.path.exists('models/irrigation_classifier.pkl'):
    print("Training all ML models...")
    ml_system.train_all_models()
else:
    print("Models already exist. Skipping training.")

print("Model training complete!")
