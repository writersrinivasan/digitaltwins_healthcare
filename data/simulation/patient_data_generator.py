"""
Synthetic Patient Data Generator for Virtual Patient Monitor
Generates realistic physiological data for digital twin healthcare system
"""

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatientProfile:
    """Patient demographic and baseline characteristics"""
    patient_id: str
    age: int
    gender: str
    weight: float  # kg
    height: float  # cm
    medical_conditions: List[str]
    medications: List[str]
    baseline_vitals: Dict[str, float]
    risk_factors: List[str]

@dataclass
class VitalSigns:
    """Real-time vital signs measurement"""
    timestamp: datetime
    patient_id: str
    heart_rate: float          # bpm
    systolic_bp: float         # mmHg
    diastolic_bp: float        # mmHg
    oxygen_saturation: float   # %
    respiratory_rate: float    # breaths/min
    temperature: float         # °C
    glucose_level: float       # mg/dL
    pain_score: int           # 0-10 scale
    activity_level: str       # sedentary, light, moderate, high
    sleep_quality: float      # 0-1 scale

class PhysiologicalModel:
    """Mathematical models for generating realistic vital signs"""
    
    def __init__(self, patient_profile: PatientProfile):
        self.profile = patient_profile
        self.baseline = patient_profile.baseline_vitals
        self.time_counter = 0
        
        # Circadian rhythm parameters
        self.circadian_phase = random.uniform(0, 2 * np.pi)
        
        # Trend parameters for simulating health deterioration/improvement
        self.health_trend = 0.0  # -1 (deteriorating) to 1 (improving)
        self.stress_level = 0.5  # 0-1 scale
        
    def generate_heart_rate(self, base_time: datetime) -> float:
        """Generate realistic heart rate with circadian rhythm and variability"""
        base_hr = self.baseline.get('heart_rate', 70)
        
        # Age adjustment
        age_factor = 1.0 - (self.profile.age - 30) * 0.001
        
        # Circadian rhythm (lower at night, higher during day)
        hour = base_time.hour
        circadian_effect = 10 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Activity-based variation
        activity_effects = {
            'sedentary': -5,
            'light': 0,
            'moderate': 15,
            'high': 30
        }
        
        # Medical condition effects
        condition_effects = 0
        if 'hypertension' in self.profile.medical_conditions:
            condition_effects += 5
        if 'diabetes' in self.profile.medical_conditions:
            condition_effects += 3
        if 'heart_disease' in self.profile.medical_conditions:
            condition_effects += 8
            
        # Stress and health trend effects
        stress_effect = self.stress_level * 10
        trend_effect = -self.health_trend * 5  # Negative trend increases HR
        
        # Random physiological noise
        noise = np.random.normal(0, 3)
        
        hr = (base_hr * age_factor + 
              circadian_effect + 
              condition_effects + 
              stress_effect + 
              trend_effect + 
              noise)
        
        return max(40, min(200, hr))  # Physiological bounds
    
    def generate_blood_pressure(self, heart_rate: float, base_time: datetime) -> Tuple[float, float]:
        """Generate systolic and diastolic blood pressure"""
        base_sys = self.baseline.get('systolic_bp', 120)
        base_dia = self.baseline.get('diastolic_bp', 80)
        
        # Age-related increase
        age_factor_sys = (self.profile.age - 20) * 0.5
        age_factor_dia = (self.profile.age - 20) * 0.3
        
        # Heart rate correlation
        hr_effect_sys = (heart_rate - 70) * 0.3
        hr_effect_dia = (heart_rate - 70) * 0.1
        
        # Medical conditions
        condition_effect_sys = 0
        condition_effect_dia = 0
        if 'hypertension' in self.profile.medical_conditions:
            condition_effect_sys += 20
            condition_effect_dia += 10
        if 'diabetes' in self.profile.medical_conditions:
            condition_effect_sys += 10
            condition_effect_dia += 5
            
        # Stress effects
        stress_effect_sys = self.stress_level * 15
        stress_effect_dia = self.stress_level * 8
        
        # Health trend
        trend_effect_sys = -self.health_trend * 10
        trend_effect_dia = -self.health_trend * 5
        
        # Noise
        noise_sys = np.random.normal(0, 5)
        noise_dia = np.random.normal(0, 3)
        
        systolic = (base_sys + age_factor_sys + hr_effect_sys + 
                   condition_effect_sys + stress_effect_sys + 
                   trend_effect_sys + noise_sys)
        
        diastolic = (base_dia + age_factor_dia + hr_effect_dia + 
                    condition_effect_dia + stress_effect_dia + 
                    trend_effect_dia + noise_dia)
        
        # Ensure diastolic < systolic
        diastolic = min(diastolic, systolic - 20)
        
        return max(70, min(250, systolic)), max(40, min(150, diastolic))
    
    def generate_oxygen_saturation(self, base_time: datetime) -> float:
        """Generate oxygen saturation percentage"""
        base_spo2 = self.baseline.get('oxygen_saturation', 98)
        
        # Medical conditions effect
        condition_effect = 0
        if 'copd' in self.profile.medical_conditions:
            condition_effect -= 5
        if 'asthma' in self.profile.medical_conditions:
            condition_effect -= 2
        if 'pneumonia' in self.profile.medical_conditions:
            condition_effect -= 8
            
        # Health trend effect
        trend_effect = self.health_trend * 2
        
        # Random variation
        noise = np.random.normal(0, 0.5)
        
        spo2 = base_spo2 + condition_effect + trend_effect + noise
        
        return max(70, min(100, spo2))
    
    def generate_temperature(self, base_time: datetime) -> float:
        """Generate body temperature in Celsius"""
        base_temp = 36.5  # Normal body temperature
        
        # Circadian rhythm (lower in early morning, higher in evening)
        hour = base_time.hour
        circadian_effect = 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Medical conditions
        condition_effect = 0
        if 'infection' in self.profile.medical_conditions:
            condition_effect += random.uniform(1.0, 3.0)
        if 'fever' in self.profile.medical_conditions:
            condition_effect += random.uniform(1.5, 2.5)
            
        # Health trend
        trend_effect = -self.health_trend * 0.5
        
        # Noise
        noise = np.random.normal(0, 0.2)
        
        temp = base_temp + circadian_effect + condition_effect + trend_effect + noise
        
        return max(34.0, min(42.0, temp))
    
    def generate_glucose_level(self, base_time: datetime) -> float:
        """Generate blood glucose level in mg/dL"""
        base_glucose = 90  # Fasting glucose
        
        # Time since last meal simulation
        hour = base_time.hour
        if hour in [7, 8, 9]:  # Post-breakfast
            meal_effect = random.uniform(20, 40)
        elif hour in [12, 13, 14]:  # Post-lunch
            meal_effect = random.uniform(25, 45)
        elif hour in [18, 19, 20]:  # Post-dinner
            meal_effect = random.uniform(20, 40)
        else:
            meal_effect = random.uniform(-10, 10)
            
        # Diabetes effect
        diabetes_effect = 0
        if 'diabetes' in self.profile.medical_conditions:
            diabetes_effect = random.uniform(30, 80)
            
        # Health trend
        trend_effect = -self.health_trend * 10
        
        # Noise
        noise = np.random.normal(0, 5)
        
        glucose = base_glucose + meal_effect + diabetes_effect + trend_effect + noise
        
        return max(50, min(400, glucose))

class SyntheticDataGenerator:
    """Main class for generating synthetic patient data"""
    
    def __init__(self):
        self.patients = {}
        self.models = {}
        
    def create_patient_profile(self, patient_id: str = None) -> PatientProfile:
        """Create a realistic patient profile"""
        if patient_id is None:
            patient_id = f"PATIENT_{random.randint(1000, 9999)}"
            
        age = random.randint(18, 85)
        gender = random.choice(['Male', 'Female'])
        weight = random.uniform(50, 120)  # kg
        height = random.uniform(150, 190)  # cm
        
        # Medical conditions based on age and risk factors
        conditions = []
        if age > 40 and random.random() < 0.3:
            conditions.append('hypertension')
        if age > 50 and random.random() < 0.2:
            conditions.append('diabetes')
        if age > 60 and random.random() < 0.15:
            conditions.append('heart_disease')
        if random.random() < 0.1:
            conditions.append('asthma')
            
        # Baseline vitals (normal ranges)
        baseline_vitals = {
            'heart_rate': random.uniform(60, 80),
            'systolic_bp': random.uniform(110, 130),
            'diastolic_bp': random.uniform(70, 85),
            'oxygen_saturation': random.uniform(96, 99),
            'temperature': random.uniform(36.2, 36.8),
        }
        
        return PatientProfile(
            patient_id=patient_id,
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            medical_conditions=conditions,
            medications=[],  # Simplified for this demo
            baseline_vitals=baseline_vitals,
            risk_factors=[]
        )
    
    def add_patient(self, patient_profile: PatientProfile):
        """Add a patient to the monitoring system"""
        self.patients[patient_profile.patient_id] = patient_profile
        self.models[patient_profile.patient_id] = PhysiologicalModel(patient_profile)
        logger.info(f"Added patient {patient_profile.patient_id} to monitoring system")
    
    def generate_vital_signs(self, patient_id: str, timestamp: datetime = None) -> VitalSigns:
        """Generate a single set of vital signs for a patient"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if patient_id not in self.models:
            raise ValueError(f"Patient {patient_id} not found in system")
            
        model = self.models[patient_id]
        
        # Generate correlated vital signs
        heart_rate = model.generate_heart_rate(timestamp)
        systolic_bp, diastolic_bp = model.generate_blood_pressure(heart_rate, timestamp)
        oxygen_saturation = model.generate_oxygen_saturation(timestamp)
        temperature = model.generate_temperature(timestamp)
        glucose_level = model.generate_glucose_level(timestamp)
        
        # Simple derived measurements
        respiratory_rate = random.uniform(12, 20)
        pain_score = random.randint(0, 3)  # Most patients have low pain
        activity_level = random.choice(['sedentary', 'light', 'light', 'moderate'])
        sleep_quality = random.uniform(0.6, 0.9)
        
        return VitalSigns(
            timestamp=timestamp,
            patient_id=patient_id,
            heart_rate=round(heart_rate, 1),
            systolic_bp=round(systolic_bp, 1),
            diastolic_bp=round(diastolic_bp, 1),
            oxygen_saturation=round(oxygen_saturation, 1),
            respiratory_rate=round(respiratory_rate, 1),
            temperature=round(temperature, 1),
            glucose_level=round(glucose_level, 1),
            pain_score=pain_score,
            activity_level=activity_level,
            sleep_quality=round(sleep_quality, 2)
        )
    
    def generate_time_series(self, patient_id: str, start_time: datetime, 
                           duration_hours: int, frequency_minutes: int = 5) -> pd.DataFrame:
        """Generate a time series of vital signs for a patient"""
        data = []
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        
        while current_time <= end_time:
            vitals = self.generate_vital_signs(patient_id, current_time)
            data.append(asdict(vitals))
            current_time += timedelta(minutes=frequency_minutes)
            
        return pd.DataFrame(data)
    
    def simulate_health_event(self, patient_id: str, event_type: str, timestamp: datetime):
        """Simulate a health event that affects vital signs"""
        if patient_id not in self.models:
            return
            
        model = self.models[patient_id]
        
        if event_type == "stress":
            model.stress_level = min(1.0, model.stress_level + 0.3)
        elif event_type == "deterioration":
            model.health_trend = max(-1.0, model.health_trend - 0.2)
        elif event_type == "improvement":
            model.health_trend = min(1.0, model.health_trend + 0.2)
        elif event_type == "medication":
            model.stress_level = max(0.0, model.stress_level - 0.2)
            
        logger.info(f"Simulated {event_type} event for patient {patient_id} at {timestamp}")

def demo_data_generation():
    """Demonstration of synthetic data generation"""
    generator = SyntheticDataGenerator()
    
    # Create sample patients
    for i in range(3):
        profile = generator.create_patient_profile()
        generator.add_patient(profile)
        print(f"Created patient: {profile.patient_id}")
        print(f"  Age: {profile.age}, Conditions: {profile.medical_conditions}")
        print(f"  Baseline HR: {profile.baseline_vitals['heart_rate']:.1f} bpm")
        print()
    
    # Generate sample data
    patient_id = list(generator.patients.keys())[0]
    start_time = datetime.now() - timedelta(hours=24)
    
    # Generate 24 hours of data at 5-minute intervals
    df = generator.generate_time_series(patient_id, start_time, 24, 5)
    
    print(f"Generated {len(df)} vital sign measurements for patient {patient_id}")
    print(df.head())
    
    # Save sample data
    df.to_csv(f'sample_patient_data_{patient_id}.csv', index=False)
    print(f"Saved data to sample_patient_data_{patient_id}.csv")

if __name__ == "__main__":
    demo_data_generation()
