"""
Configuration settings for patient profiles and system parameters
"""

# Patient profiles for demo initialization
DEMO_PATIENT_PROFILES = [
    {
        "name": "John Smith",
        "age": 65,
        "gender": "Male",
        "weight": 75.5,
        "height": 175,
        "room": "ICU-101",
        "medical_conditions": ["hypertension", "diabetes"],
        "medications": ["lisinopril", "metformin"],
        "baseline_vitals": {
            "heart_rate": 72,
            "systolic_bp": 125,
            "diastolic_bp": 82,
            "oxygen_saturation": 97.5,
            "temperature": 36.6
        }
    },
    {
        "name": "Sarah Johnson", 
        "age": 45,
        "gender": "Female",
        "weight": 68.2,
        "height": 165,
        "room": "Ward-205",
        "medical_conditions": ["asthma"],
        "medications": ["albuterol"],
        "baseline_vitals": {
            "heart_rate": 68,
            "systolic_bp": 118,
            "diastolic_bp": 78,
            "oxygen_saturation": 98.2,
            "temperature": 36.4
        }
    },
    {
        "name": "Robert Wilson",
        "age": 78,
        "gender": "Male", 
        "weight": 82.1,
        "height": 172,
        "room": "ICU-103",
        "medical_conditions": ["heart_disease", "hypertension"],
        "medications": ["atenolol", "lisinopril", "aspirin"],
        "baseline_vitals": {
            "heart_rate": 65,
            "systolic_bp": 135,
            "diastolic_bp": 85,
            "oxygen_saturation": 96.8,
            "temperature": 36.5
        }
    }
]

# System configuration
SYSTEM_CONFIG = {
    "data_streaming": {
        "update_interval_seconds": 5,
        "max_history_points": 2000,
        "buffer_size": 10000
    },
    "dashboard": {
        "auto_refresh_seconds": 10,
        "default_time_range": "1h",
        "chart_update_interval": 2
    },
    "alerts": {
        "enable_audio": True,
        "auto_acknowledge_timeout": 300,  # 5 minutes
        "escalation_delay": 900  # 15 minutes
    },
    "ml_models": {
        "anomaly_detection_threshold": 0.7,
        "prediction_confidence_threshold": 0.6,
        "model_update_frequency": "daily"
    }
}
