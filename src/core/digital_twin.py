"""
Digital Twin Core Engine for Virtual Patient Monitor
Manages the virtual representation and synchronization with patient data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from dataclasses import dataclass, asdict
import threading
import time
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Patient health status levels"""
    CRITICAL = "critical"
    WARNING = "warning" 
    STABLE = "stable"
    OPTIMAL = "optimal"

@dataclass
class DigitalTwinState:
    """Complete state of the digital twin at a point in time"""
    timestamp: datetime
    patient_id: str
    
    # Current vital signs
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    oxygen_saturation: float
    temperature: float
    respiratory_rate: float
    glucose_level: float
    
    # Derived metrics
    pulse_pressure: float
    mean_arterial_pressure: float
    
    # Health indicators
    health_status: HealthStatus
    anomaly_score: float
    trend_direction: str  # "improving", "stable", "deteriorating"
    
    # Predictions
    predicted_status_6h: str
    prediction_confidence: float
    
    # Contextual information
    activity_level: str
    medication_effects: Dict[str, float]
    environmental_factors: Dict[str, Any]

class PhysiologicalModel:
    """
    Mathematical model representing patient physiology
    Based on established medical equations and relationships
    """
    
    def __init__(self, patient_profile):
        self.patient_profile = patient_profile
        self.baseline_parameters = self._initialize_baseline_parameters()
        self.current_parameters = self.baseline_parameters.copy()
        
        # Model state variables
        self.cardiac_output = 5.0  # L/min
        self.total_peripheral_resistance = 1000  # dyn⋅s⋅cm⁻⁵
        self.blood_volume = 5000  # mL
        self.metabolic_rate = 1.0  # relative to baseline
        
    def _initialize_baseline_parameters(self) -> Dict[str, float]:
        """Initialize patient-specific baseline physiological parameters"""
        age = self.patient_profile.age
        weight = self.patient_profile.weight
        height = self.patient_profile.height
        
        # Calculate body surface area (Mosteller formula)
        bsa = np.sqrt((height * weight) / 3600)
        
        # Age-adjusted baselines
        baseline_hr = 70 - (age - 30) * 0.1
        baseline_sv = 70 * bsa  # Stroke volume (mL)
        baseline_co = baseline_hr * baseline_sv / 1000  # Cardiac output (L/min)
        
        return {
            'baseline_heart_rate': baseline_hr,
            'baseline_stroke_volume': baseline_sv,
            'baseline_cardiac_output': baseline_co,
            'baseline_systolic_bp': 120 - (age - 30) * 0.2,
            'baseline_diastolic_bp': 80 - (age - 30) * 0.1,
            'baseline_temperature': 36.5,
            'baseline_oxygen_consumption': 250 * bsa,  # mL/min
            'body_surface_area': bsa
        }
    
    def calculate_blood_pressure(self, cardiac_output: float, resistance: float) -> Tuple[float, float]:
        """
        Calculate blood pressure using cardiovascular hemodynamics
        BP = CO × TPR (simplified)
        """
        # Mean arterial pressure
        map_pressure = cardiac_output * resistance / 80  # Convert units
        
        # Estimate systolic and diastolic from MAP
        # Pulse pressure depends on stroke volume and arterial compliance
        stroke_volume = cardiac_output * 1000 / self.current_parameters['baseline_heart_rate']
        
        # Age-related arterial stiffening
        age_factor = 1 + (self.patient_profile.age - 30) * 0.01
        pulse_pressure = stroke_volume * 0.5 * age_factor
        
        systolic = map_pressure + (2/3) * pulse_pressure
        diastolic = map_pressure - (1/3) * pulse_pressure
        
        return systolic, diastolic
    
    def calculate_oxygen_saturation(self, oxygen_delivery: float, oxygen_consumption: float) -> float:
        """
        Calculate oxygen saturation based on delivery and consumption
        """
        # Simplified oxygen saturation model
        base_saturation = 98.0
        
        # Oxygen delivery/consumption ratio
        ratio = oxygen_delivery / oxygen_consumption
        
        if ratio >= 4.0:  # Normal ratio
            saturation = base_saturation
        elif ratio >= 3.0:
            saturation = base_saturation - (4.0 - ratio) * 2
        elif ratio >= 2.0:
            saturation = base_saturation - 2 - (3.0 - ratio) * 5
        else:
            saturation = base_saturation - 7 - (2.0 - ratio) * 10
        
        # Apply pathological effects
        if 'copd' in self.patient_profile.medical_conditions:
            saturation -= 3
        if 'pneumonia' in self.patient_profile.medical_conditions:
            saturation -= 5
        
        return max(70, min(100, saturation))
    
    def update_parameters(self, external_factors: Dict[str, Any]):
        """Update model parameters based on external factors"""
        
        # Medication effects
        medications = external_factors.get('medications', {})
        
        if 'beta_blocker' in medications:
            self.current_parameters['baseline_heart_rate'] *= 0.8
            
        if 'ace_inhibitor' in medications:
            self.total_peripheral_resistance *= 0.9
            
        if 'diuretic' in medications:
            self.blood_volume *= 0.95
        
        # Activity effects
        activity = external_factors.get('activity_level', 'sedentary')
        activity_multipliers = {
            'sedentary': 1.0,
            'light': 1.2,
            'moderate': 1.5,
            'high': 2.0
        }
        self.metabolic_rate = activity_multipliers.get(activity, 1.0)
        
        # Stress effects
        stress_level = external_factors.get('stress_level', 0.5)
        stress_factor = 1 + stress_level * 0.3
        self.current_parameters['baseline_heart_rate'] *= stress_factor

class DigitalTwin:
    """
    Main Digital Twin class that maintains virtual patient representation
    """
    
    def __init__(self, patient_profile, anomaly_detector=None, health_predictor=None):
        self.patient_profile = patient_profile
        self.patient_id = patient_profile.patient_id
        
        # Initialize physiological model
        self.physiology_model = PhysiologicalModel(patient_profile)
        
        # ML models for analysis
        self.anomaly_detector = anomaly_detector
        self.health_predictor = health_predictor
        
        # State management
        self.current_state = None
        self.state_history = []
        self.max_history = 2000  # Keep last 2000 states
        
        # Real-time synchronization
        self.last_sync_time = None
        self.sync_interval = timedelta(seconds=30)  # Sync every 30 seconds
        
        # Alert thresholds (personalized based on patient)
        self.alert_thresholds = self._initialize_alert_thresholds()
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_expiry = timedelta(minutes=5)
        
        logger.info(f"Digital Twin initialized for patient {self.patient_id}")
    
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize personalized alert thresholds"""
        age = self.patient_profile.age
        conditions = self.patient_profile.medical_conditions
        
        # Base thresholds
        thresholds = {
            'heart_rate': {'low': 60, 'high': 100, 'critical_low': 40, 'critical_high': 150},
            'systolic_bp': {'low': 90, 'high': 140, 'critical_low': 70, 'critical_high': 180},
            'diastolic_bp': {'low': 60, 'high': 90, 'critical_low': 40, 'critical_high': 110},
            'oxygen_saturation': {'low': 95, 'high': 100, 'critical_low': 88, 'critical_high': 100},
            'temperature': {'low': 36.1, 'high': 37.2, 'critical_low': 35.0, 'critical_high': 39.0}
        }
        
        # Adjust for age
        if age > 65:
            thresholds['systolic_bp']['high'] = 150  # Higher tolerance for elderly
            thresholds['heart_rate']['low'] = 55
        
        # Adjust for medical conditions
        if 'hypertension' in conditions:
            thresholds['systolic_bp']['high'] = 160
            thresholds['diastolic_bp']['high'] = 100
            
        if 'heart_disease' in conditions:
            thresholds['heart_rate']['critical_high'] = 120
            
        if 'copd' in conditions:
            thresholds['oxygen_saturation']['low'] = 88
            
        return thresholds
    
    def sync_with_real_data(self, vital_signs_data):
        """
        Synchronize digital twin with real patient data
        
        Args:
            vital_signs_data: Latest vital signs measurement
        """
        current_time = datetime.now()
        
        # Update physiological model with current data
        external_factors = {
            'activity_level': getattr(vital_signs_data, 'activity_level', 'sedentary'),
            'medications': {},  # Would be populated from patient records
            'stress_level': 0.5  # Would be inferred from various indicators
        }
        
        self.physiology_model.update_parameters(external_factors)
        
        # Calculate derived metrics
        pulse_pressure = vital_signs_data.systolic_bp - vital_signs_data.diastolic_bp
        mean_arterial_pressure = (vital_signs_data.systolic_bp + 2 * vital_signs_data.diastolic_bp) / 3
        
        # Determine health status
        health_status = self._assess_health_status(vital_signs_data)
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(vital_signs_data)
        
        # Determine trend
        trend = self._calculate_trend()
        
        # Make predictions
        predictions = self._get_predictions()
        
        # Create new state
        new_state = DigitalTwinState(
            timestamp=current_time,
            patient_id=self.patient_id,
            heart_rate=vital_signs_data.heart_rate,
            systolic_bp=vital_signs_data.systolic_bp,
            diastolic_bp=vital_signs_data.diastolic_bp,
            oxygen_saturation=vital_signs_data.oxygen_saturation,
            temperature=vital_signs_data.temperature,
            respiratory_rate=vital_signs_data.respiratory_rate,
            glucose_level=vital_signs_data.glucose_level,
            pulse_pressure=pulse_pressure,
            mean_arterial_pressure=mean_arterial_pressure,
            health_status=health_status,
            anomaly_score=anomaly_score,
            trend_direction=trend,
            predicted_status_6h=predictions.get('status', 'unknown'),
            prediction_confidence=predictions.get('confidence', 0.0),
            activity_level=vital_signs_data.activity_level,
            medication_effects={},
            environmental_factors={}
        )
        
        # Update state
        self.current_state = new_state
        self.state_history.append(new_state)
        
        # Maintain history size
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]
        
        self.last_sync_time = current_time
        
        logger.debug(f"Digital Twin synchronized for patient {self.patient_id} at {current_time}")
        
        return new_state
    
    def _assess_health_status(self, vital_signs) -> HealthStatus:
        """Assess overall health status based on vital signs"""
        critical_violations = 0
        warning_violations = 0
        
        # Check each vital against thresholds
        vitals_to_check = {
            'heart_rate': vital_signs.heart_rate,
            'systolic_bp': vital_signs.systolic_bp,
            'diastolic_bp': vital_signs.diastolic_bp,
            'oxygen_saturation': vital_signs.oxygen_saturation,
            'temperature': vital_signs.temperature
        }
        
        for vital_name, value in vitals_to_check.items():
            thresholds = self.alert_thresholds[vital_name]
            
            if value <= thresholds['critical_low'] or value >= thresholds['critical_high']:
                critical_violations += 1
            elif value <= thresholds['low'] or value >= thresholds['high']:
                warning_violations += 1
        
        # Determine status
        if critical_violations > 0:
            return HealthStatus.CRITICAL
        elif warning_violations >= 2:
            return HealthStatus.WARNING
        elif warning_violations == 1:
            return HealthStatus.STABLE
        else:
            return HealthStatus.OPTIMAL
    
    def _calculate_anomaly_score(self, vital_signs) -> float:
        """Calculate anomaly score using ML model or rule-based approach"""
        if self.anomaly_detector and self.anomaly_detector.is_trained:
            # Use trained ML model
            data_dict = {
                'heart_rate': [vital_signs.heart_rate],
                'systolic_bp': [vital_signs.systolic_bp],
                'diastolic_bp': [vital_signs.diastolic_bp],
                'oxygen_saturation': [vital_signs.oxygen_saturation],
                'temperature': [vital_signs.temperature],
                'respiratory_rate': [vital_signs.respiratory_rate]
            }
            
            df = pd.DataFrame(data_dict)
            result = self.anomaly_detector.detect_anomalies(df)
            
            if result['anomaly_scores']:
                return abs(result['anomaly_scores'][0])  # Return absolute score
            
        # Fallback to rule-based scoring
        return self._rule_based_anomaly_score(vital_signs)
    
    def _rule_based_anomaly_score(self, vital_signs) -> float:
        """Calculate anomaly score using rule-based approach"""
        score = 0.0
        
        # Calculate deviation from normal ranges
        vitals_to_check = {
            'heart_rate': (vital_signs.heart_rate, 60, 100),
            'systolic_bp': (vital_signs.systolic_bp, 90, 140),
            'diastolic_bp': (vital_signs.diastolic_bp, 60, 90),
            'oxygen_saturation': (vital_signs.oxygen_saturation, 95, 100),
            'temperature': (vital_signs.temperature, 36.1, 37.2)
        }
        
        for vital_name, (value, min_normal, max_normal) in vitals_to_check.items():
            if value < min_normal:
                deviation = (min_normal - value) / min_normal
                score += deviation
            elif value > max_normal:
                deviation = (value - max_normal) / max_normal
                score += deviation
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_trend(self) -> str:
        """Calculate health trend based on recent history"""
        if len(self.state_history) < 6:  # Need at least 30 minutes of data
            return "stable"
        
        # Get recent anomaly scores
        recent_scores = [state.anomaly_score for state in self.state_history[-6:]]
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]
        
        if slope > 0.01:
            return "deteriorating"
        elif slope < -0.01:
            return "improving"
        else:
            return "stable"
    
    def _get_predictions(self) -> Dict[str, Any]:
        """Get health predictions from ML model"""
        current_time = datetime.now()
        
        # Check cache
        if self.prediction_cache and current_time - self.prediction_cache.get('timestamp', datetime.min) < self.cache_expiry:
            return self.prediction_cache
        
        if self.health_predictor and self.health_predictor.is_trained and len(self.state_history) >= 12:
            # Prepare recent data for prediction
            recent_data = pd.DataFrame([
                {
                    'timestamp': state.timestamp,
                    'heart_rate': state.heart_rate,
                    'systolic_bp': state.systolic_bp,
                    'diastolic_bp': state.diastolic_bp,
                    'oxygen_saturation': state.oxygen_saturation,
                    'temperature': state.temperature,
                    'respiratory_rate': state.respiratory_rate
                }
                for state in self.state_history[-12:]
            ])
            
            prediction_result = self.health_predictor.predict_health_trend(recent_data)
            
            # Cache result
            self.prediction_cache = {
                'timestamp': current_time,
                'status': prediction_result.get('status', 'unknown'),
                'confidence': prediction_result.get('confidence', 0.0),
                'recommendation': prediction_result.get('recommendation', '')
            }
            
            return self.prediction_cache
        
        # Default predictions
        return {
            'status': 'stable',
            'confidence': 0.5,
            'recommendation': 'Continue monitoring'
        }
    
    def get_current_state(self) -> Optional[DigitalTwinState]:
        """Get the current state of the digital twin"""
        return self.current_state
    
    def get_state_history(self, hours: int = 24) -> List[DigitalTwinState]:
        """Get state history for specified number of hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            state for state in self.state_history
            if state.timestamp >= cutoff_time
        ]
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts based on digital twin state"""
        if not self.current_state:
            return []
        
        alerts = []
        
        # Health status alerts
        if self.current_state.health_status == HealthStatus.CRITICAL:
            alerts.append({
                'type': 'critical',
                'message': 'Patient in critical condition - immediate attention required',
                'timestamp': self.current_state.timestamp,
                'priority': 1
            })
        elif self.current_state.health_status == HealthStatus.WARNING:
            alerts.append({
                'type': 'warning',
                'message': 'Patient condition requires attention',
                'timestamp': self.current_state.timestamp,
                'priority': 2
            })
        
        # Trend alerts
        if self.current_state.trend_direction == "deteriorating":
            alerts.append({
                'type': 'trend',
                'message': 'Patient condition showing deteriorating trend',
                'timestamp': self.current_state.timestamp,
                'priority': 2
            })
        
        # Anomaly alerts
        if self.current_state.anomaly_score > 0.7:
            alerts.append({
                'type': 'anomaly',
                'message': f'High anomaly score detected: {self.current_state.anomaly_score:.2f}',
                'timestamp': self.current_state.timestamp,
                'priority': 2
            })
        
        # Prediction alerts
        if 'deterioration' in self.current_state.predicted_status_6h.lower():
            alerts.append({
                'type': 'prediction',
                'message': f'Predicted deterioration risk: {self.current_state.prediction_confidence:.0%}',
                'timestamp': self.current_state.timestamp,
                'priority': 3
            })
        
        return sorted(alerts, key=lambda x: x['priority'])
    
    def export_state(self) -> Dict[str, Any]:
        """Export current digital twin state as dictionary"""
        if not self.current_state:
            return {}
        
        return {
            'patient_id': self.patient_id,
            'current_state': asdict(self.current_state),
            'alert_thresholds': self.alert_thresholds,
            'model_status': {
                'anomaly_detector_trained': self.anomaly_detector.is_trained if self.anomaly_detector else False,
                'health_predictor_trained': self.health_predictor.is_trained if self.health_predictor else False
            },
            'last_sync': self.last_sync_time.isoformat() if self.last_sync_time else None,
            'state_history_length': len(self.state_history)
        }

class DigitalTwinManager:
    """
    Manages multiple digital twins for different patients
    """
    
    def __init__(self, anomaly_detector=None, health_predictor=None):
        self.digital_twins: Dict[str, DigitalTwin] = {}
        self.anomaly_detector = anomaly_detector
        self.health_predictor = health_predictor
        
        # Background synchronization
        self.sync_thread = None
        self.sync_active = False
        
    def create_digital_twin(self, patient_profile) -> DigitalTwin:
        """Create a new digital twin for a patient"""
        twin = DigitalTwin(
            patient_profile, 
            self.anomaly_detector, 
            self.health_predictor
        )
        
        self.digital_twins[patient_profile.patient_id] = twin
        logger.info(f"Created digital twin for patient {patient_profile.patient_id}")
        
        return twin
    
    def get_digital_twin(self, patient_id: str) -> Optional[DigitalTwin]:
        """Get digital twin for a specific patient"""
        return self.digital_twins.get(patient_id)
    
    def sync_all_twins(self, vital_signs_data_dict: Dict[str, Any]):
        """Sync all digital twins with new data"""
        for patient_id, vital_signs in vital_signs_data_dict.items():
            if patient_id in self.digital_twins:
                self.digital_twins[patient_id].sync_with_real_data(vital_signs)
    
    def get_all_alerts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get alerts from all digital twins"""
        all_alerts = {}
        
        for patient_id, twin in self.digital_twins.items():
            alerts = twin.get_alerts()
            if alerts:
                all_alerts[patient_id] = alerts
        
        return all_alerts
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of all digital twins"""
        summary = {
            'total_patients': len(self.digital_twins),
            'patients_by_status': {},
            'total_alerts': 0,
            'critical_patients': [],
            'last_updated': datetime.now().isoformat()
        }
        
        status_counts = {}
        
        for patient_id, twin in self.digital_twins.items():
            current_state = twin.get_current_state()
            
            if current_state:
                status = current_state.health_status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                alerts = twin.get_alerts()
                summary['total_alerts'] += len(alerts)
                
                if current_state.health_status == HealthStatus.CRITICAL:
                    summary['critical_patients'].append(patient_id)
        
        summary['patients_by_status'] = status_counts
        
        return summary
