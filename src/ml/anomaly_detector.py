"""
Machine Learning Models for Anomaly Detection and Predictive Analytics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Advanced anomaly detection for patient vital signs
    Uses multiple algorithms for robust detection
    """
    
    def __init__(self):
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 
            'oxygen_saturation', 'temperature', 'respiratory_rate'
        ]
        self.is_trained = False
        self.patient_baselines = {}  # Store patient-specific baselines
        
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract engineered features from raw vital signs data"""
        features = data[self.feature_columns].copy()
        
        # Time-based features
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            features['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # Derived vital signs
        features['pulse_pressure'] = features['systolic_bp'] - features['diastolic_bp']
        features['mean_arterial_pressure'] = (features['systolic_bp'] + 2 * features['diastolic_bp']) / 3
        
        # Rate of change features (if we have time series data)
        if len(features) > 1:
            for col in self.feature_columns:
                features[f'{col}_change'] = features[col].diff()
                features[f'{col}_rolling_mean'] = features[col].rolling(window=5, min_periods=1).mean()
                features[f'{col}_rolling_std'] = features[col].rolling(window=5, min_periods=1).std()
        
        # Fill NaN values (from diff and rolling operations)
        features = features.fillna(method='bfill').fillna(0)
        
        return features
    
    def train_baseline_model(self, training_data: pd.DataFrame, contamination: float = 0.1):
        """
        Train the anomaly detection model on normal patient data
        
        Args:
            training_data: DataFrame with normal patient vital signs
            contamination: Expected proportion of anomalies in training data
        """
        logger.info("Training anomaly detection model...")
        
        # Extract features
        features = self.extract_features(training_data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.isolation_forest.fit(scaled_features)
        self.is_trained = True
        
        logger.info(f"Model trained on {len(training_data)} samples")
        
        # Evaluate on training data
        predictions = self.isolation_forest.predict(scaled_features)
        anomaly_rate = np.sum(predictions == -1) / len(predictions)
        logger.info(f"Training anomaly rate: {anomaly_rate:.2%}")
    
    def detect_anomalies(self, data: pd.DataFrame, patient_id: Optional[str] = None) -> Dict:
        """
        Detect anomalies in patient vital signs
        
        Args:
            data: DataFrame with vital signs
            patient_id: Optional patient identifier for personalized detection
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not self.is_trained:
            logger.warning("Model not trained. Using rule-based detection.")
            return self._rule_based_detection(data)
        
        # Extract features
        features = self.extract_features(data)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict anomalies
        predictions = self.isolation_forest.predict(scaled_features)
        anomaly_scores = self.isolation_forest.decision_function(scaled_features)
        
        # Convert to binary indicators (1 = normal, 0 = anomaly)
        is_normal = predictions == 1
        
        # Calculate severity based on anomaly scores
        severity_levels = []
        for score in anomaly_scores:
            if score > -0.1:
                severity_levels.append('Normal')
            elif score > -0.3:
                severity_levels.append('Mild')
            elif score > -0.5:
                severity_levels.append('Moderate')
            else:
                severity_levels.append('Severe')
        
        # Prepare results
        results = {
            'anomaly_detected': not all(is_normal),
            'anomaly_count': np.sum(~is_normal),
            'total_measurements': len(data),
            'anomaly_rate': np.sum(~is_normal) / len(data),
            'predictions': predictions.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'severity_levels': severity_levels,
            'timestamps': data.get('timestamp', pd.date_range('2024-01-01', periods=len(data))).tolist()
        }
        
        # Add specific anomalous measurements
        if results['anomaly_detected']:
            anomalous_indices = np.where(~is_normal)[0]
            results['anomalous_measurements'] = []
            
            for idx in anomalous_indices:
                measurement = {
                    'index': int(idx),
                    'timestamp': results['timestamps'][idx],
                    'severity': severity_levels[idx],
                    'score': float(anomaly_scores[idx]),
                    'vitals': {col: float(data.iloc[idx][col]) for col in self.feature_columns if col in data.columns}
                }
                results['anomalous_measurements'].append(measurement)
        
        return results
    
    def _rule_based_detection(self, data: pd.DataFrame) -> Dict:
        """
        Fallback rule-based anomaly detection using clinical thresholds
        """
        anomalies = []
        
        # Define normal ranges for vital signs
        normal_ranges = {
            'heart_rate': (60, 100),
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'oxygen_saturation': (95, 100),
            'temperature': (36.1, 37.2),
            'respiratory_rate': (12, 20)
        }
        
        for idx, row in data.iterrows():
            violations = []
            severity_score = 0
            
            for vital, (min_val, max_val) in normal_ranges.items():
                if vital in row:
                    value = row[vital]
                    if value < min_val or value > max_val:
                        violations.append(vital)
                        # Calculate severity based on how far outside normal range
                        if value < min_val:
                            deviation = (min_val - value) / min_val
                        else:
                            deviation = (value - max_val) / max_val
                        severity_score += deviation
            
            if violations:
                if severity_score > 0.5:
                    severity = 'Severe'
                elif severity_score > 0.2:
                    severity = 'Moderate'
                else:
                    severity = 'Mild'
                
                anomalies.append({
                    'index': idx,
                    'timestamp': row.get('timestamp', datetime.now()),
                    'severity': severity,
                    'score': -severity_score,  # Negative to match isolation forest
                    'violations': violations,
                    'vitals': {col: row[col] for col in self.feature_columns if col in row}
                })
        
        return {
            'anomaly_detected': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'total_measurements': len(data),
            'anomaly_rate': len(anomalies) / len(data) if len(data) > 0 else 0,
            'anomalous_measurements': anomalies,
            'method': 'rule_based'
        }
    
    def update_patient_baseline(self, patient_id: str, data: pd.DataFrame):
        """Update patient-specific baseline for personalized detection"""
        if patient_id not in self.patient_baselines:
            self.patient_baselines[patient_id] = {}
        
        # Calculate patient-specific normal ranges
        for col in self.feature_columns:
            if col in data.columns:
                values = data[col].dropna()
                if len(values) > 10:  # Need sufficient data
                    mean_val = values.mean()
                    std_val = values.std()
                    
                    # Define personal normal range as mean ± 2 standard deviations
                    self.patient_baselines[patient_id][col] = {
                        'mean': mean_val,
                        'std': std_val,
                        'range': (mean_val - 2*std_val, mean_val + 2*std_val)
                    }
        
        logger.info(f"Updated baseline for patient {patient_id}")

class HealthPredictor:
    """
    Predictive model for health deterioration and improvement trends
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 
            'oxygen_saturation', 'temperature', 'respiratory_rate'
        ]
    
    def prepare_training_data(self, data: pd.DataFrame, prediction_window: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with features and labels for health prediction
        
        Args:
            data: Time series data with vital signs
            prediction_window: Hours ahead to predict
            
        Returns:
            Features and labels for training
        """
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Extract features for each time window
        features = []
        labels = []
        
        window_size = 12  # Number of measurements to use as features (1 hour at 5-min intervals)
        
        for i in range(len(data) - window_size - prediction_window):
            # Features: current window of vital signs
            window_data = data.iloc[i:i+window_size]
            
            # Calculate statistical features for the window
            feature_vector = []
            for col in self.feature_columns:
                if col in window_data.columns:
                    values = window_data[col]
                    feature_vector.extend([
                        values.mean(),
                        values.std(),
                        values.min(),
                        values.max(),
                        values.iloc[-1],  # Latest value
                        (values.iloc[-1] - values.iloc[0]) / window_size  # Trend
                    ])
            
            # Label: health status in the future
            future_window = data.iloc[i+window_size:i+window_size+prediction_window]
            
            # Simple health score based on deviation from normal ranges
            health_score = self._calculate_health_score(future_window)
            
            # Binary classification: 0 = deteriorating, 1 = stable/improving
            label = 1 if health_score > 0.7 else 0
            
            features.append(feature_vector)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def _calculate_health_score(self, data: pd.DataFrame) -> float:
        """Calculate a health score based on vital signs"""
        if data.empty:
            return 0.5
        
        normal_ranges = {
            'heart_rate': (60, 100),
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'oxygen_saturation': (95, 100),
            'temperature': (36.1, 37.2),
            'respiratory_rate': (12, 20)
        }
        
        scores = []
        for _, row in data.iterrows():
            row_score = 0
            count = 0
            
            for vital, (min_val, max_val) in normal_ranges.items():
                if vital in row and pd.notna(row[vital]):
                    value = row[vital]
                    if min_val <= value <= max_val:
                        row_score += 1
                    count += 1
            
            if count > 0:
                scores.append(row_score / count)
        
        return np.mean(scores) if scores else 0.5
    
    def train(self, training_data: pd.DataFrame):
        """Train the health prediction model"""
        logger.info("Training health prediction model...")
        
        # Prepare training data
        X, y = self.prepare_training_data(training_data)
        
        if len(X) == 0:
            logger.error("Not enough data for training")
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        logger.info("Training completed. Model evaluation:")
        logger.info("\\n" + classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        logger.info(f"Feature importance summary: mean={np.mean(feature_importance):.3f}, std={np.std(feature_importance):.3f}")
    
    def predict_health_trend(self, recent_data: pd.DataFrame) -> Dict:
        """
        Predict health trend for next few hours
        
        Args:
            recent_data: Recent vital signs data (at least 1 hour)
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            logger.warning("Model not trained. Cannot make predictions.")
            return {'error': 'Model not trained'}
        
        if len(recent_data) < 12:  # Need at least 1 hour of data
            return {'error': 'Insufficient data for prediction'}
        
        # Use the most recent window
        window_data = recent_data.tail(12)
        
        # Extract features
        feature_vector = []
        for col in self.feature_columns:
            if col in window_data.columns:
                values = window_data[col]
                feature_vector.extend([
                    values.mean(),
                    values.std(),
                    values.min(),
                    values.max(),
                    values.iloc[-1],  # Latest value
                    (values.iloc[-1] - values.iloc[0]) / len(values)  # Trend
                ])
        
        # Scale features
        features_scaled = self.scaler.transform([feature_vector])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        
        # Interpret results
        status = "Stable/Improving" if prediction == 1 else "Risk of Deterioration"
        confidence = max(prediction_proba)
        
        return {
            'prediction': int(prediction),
            'status': status,
            'confidence': float(confidence),
            'probability_stable': float(prediction_proba[1]),
            'probability_deterioration': float(prediction_proba[0]),
            'recommendation': self._get_recommendation(prediction, confidence)
        }
    
    def _get_recommendation(self, prediction: int, confidence: float) -> str:
        """Generate clinical recommendations based on prediction"""
        if prediction == 0:  # Risk of deterioration
            if confidence > 0.8:
                return "HIGH PRIORITY: Immediate clinical assessment recommended"
            elif confidence > 0.6:
                return "MEDIUM PRIORITY: Increased monitoring recommended"
            else:
                return "LOW PRIORITY: Continue routine monitoring"
        else:  # Stable/improving
            if confidence > 0.8:
                return "Patient stable - continue current care plan"
            else:
                return "Patient appears stable - maintain observation"

def save_models(anomaly_detector: AnomalyDetector, health_predictor: HealthPredictor, 
                base_path: str = "models/"):
    """Save trained models to disk"""
    import os
    os.makedirs(base_path, exist_ok=True)
    
    # Save anomaly detector
    if anomaly_detector.is_trained:
        joblib.dump(anomaly_detector.isolation_forest, f"{base_path}anomaly_model.pkl")
        joblib.dump(anomaly_detector.scaler, f"{base_path}anomaly_scaler.pkl")
        logger.info("Anomaly detection model saved")
    
    # Save health predictor
    if health_predictor.is_trained:
        joblib.dump(health_predictor.model, f"{base_path}health_predictor.pkl")
        joblib.dump(health_predictor.scaler, f"{base_path}health_scaler.pkl")
        logger.info("Health prediction model saved")

def load_models(base_path: str = "models/") -> Tuple[AnomalyDetector, HealthPredictor]:
    """Load trained models from disk"""
    anomaly_detector = AnomalyDetector()
    health_predictor = HealthPredictor()
    
    try:
        # Load anomaly detector
        anomaly_detector.isolation_forest = joblib.load(f"{base_path}anomaly_model.pkl")
        anomaly_detector.scaler = joblib.load(f"{base_path}anomaly_scaler.pkl")
        anomaly_detector.is_trained = True
        logger.info("Anomaly detection model loaded")
        
        # Load health predictor
        health_predictor.model = joblib.load(f"{base_path}health_predictor.pkl")
        health_predictor.scaler = joblib.load(f"{base_path}health_scaler.pkl")
        health_predictor.is_trained = True
        logger.info("Health prediction model loaded")
        
    except FileNotFoundError as e:
        logger.warning(f"Model files not found: {e}")
    
    return anomaly_detector, health_predictor
