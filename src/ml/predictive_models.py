"""
Predictive Models for Healthcare Analytics
Advanced machine learning models for health prediction and risk assessment
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# Optional TensorFlow import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. LSTM models will be disabled.")
    TENSORFLOW_AVAILABLE = False

import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class HealthDeteriorationPredictor:
    """
    Predicts patient health deterioration using ensemble methods
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 
            'oxygen_saturation', 'temperature', 'respiratory_rate'
        ]
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for deterioration prediction"""
        features = data[self.feature_columns].copy()
        
        # Derived features
        features['pulse_pressure'] = data['systolic_bp'] - data['diastolic_bp']
        features['mean_arterial_pressure'] = (data['systolic_bp'] + 2 * data['diastolic_bp']) / 3
        features['shock_index'] = data['heart_rate'] / data['systolic_bp']
        
        # Time-based features if timestamp available
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            features['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            features['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype(int)
        
        # Rolling statistics
        for window in [3, 6, 12]:  # 15min, 30min, 1hour windows (assuming 5min intervals)
            for col in self.feature_columns:
                if col in features.columns:
                    features[f'{col}_mean_{window}'] = features[col].rolling(window, min_periods=1).mean()
                    features[f'{col}_std_{window}'] = features[col].rolling(window, min_periods=1).std()
                    features[f'{col}_trend_{window}'] = features[col].rolling(window, min_periods=2).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features
    
    def create_deterioration_labels(self, data: pd.DataFrame, 
                                  prediction_horizon: int = 6) -> np.ndarray:
        """
        Create labels for deterioration prediction
        
        Args:
            data: DataFrame with vital signs
            prediction_horizon: Hours ahead to predict
            
        Returns:
            Binary labels (1 = deterioration, 0 = stable)
        """
        labels = np.zeros(len(data))
        
        # Define deterioration criteria
        for i in range(len(data) - prediction_horizon):
            future_window = data.iloc[i+1:i+prediction_horizon+1]
            
            # Check for multiple vital sign abnormalities
            deterioration_score = 0
            
            # Heart rate criteria
            if (future_window['heart_rate'] > 120).any() or (future_window['heart_rate'] < 50).any():
                deterioration_score += 1
            
            # Blood pressure criteria
            if (future_window['systolic_bp'] > 160).any() or (future_window['systolic_bp'] < 90).any():
                deterioration_score += 1
            
            # Oxygen saturation criteria
            if (future_window['oxygen_saturation'] < 92).any():
                deterioration_score += 1
            
            # Temperature criteria
            if (future_window['temperature'] > 38.5).any() or (future_window['temperature'] < 35.5).any():
                deterioration_score += 1
            
            # Label as deterioration if multiple criteria met
            if deterioration_score >= 2:
                labels[i] = 1
        
        return labels
    
    def train(self, training_data: pd.DataFrame):
        """Train the deterioration prediction models"""
        logger.info("Training health deterioration prediction models...")
        
        # Prepare features and labels
        features = self.prepare_features(training_data)
        labels = self.create_deterioration_labels(training_data)
        
        # Remove samples without future data
        valid_indices = ~np.isnan(labels)
        features = features[valid_indices]
        labels = labels[valid_indices]
        
        if len(features) < 100:
            logger.error("Insufficient training data")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train multiple models
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        self.models['logistic_regression'] = LogisticRegression(
            random_state=42, class_weight='balanced', max_iter=1000
        )
        self.models['logistic_regression'].fit(X_train_scaled, y_train)
        
        # Evaluate models
        for name, model in self.models.items():
            if name == 'logistic_regression':
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            logger.info(f"\\n{name.title()} Performance:")
            logger.info("\\n" + classification_report(y_test, y_pred))
        
        self.is_trained = True
        logger.info("Training completed successfully")
    
    def predict_deterioration(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Predict health deterioration risk"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        # Prepare features
        features = self.prepare_features(data)
        
        if len(features) == 0:
            return {'error': 'No valid features'}
        
        # Get latest data point
        latest_features = features.iloc[-1:].values
        
        # Make predictions with ensemble
        predictions = {}
        
        # Random Forest prediction
        rf_pred = self.models['random_forest'].predict_proba(latest_features)[0]
        predictions['random_forest'] = rf_pred[1]  # Probability of deterioration
        
        # Logistic Regression prediction
        latest_scaled = self.scalers['standard'].transform(latest_features)
        lr_pred = self.models['logistic_regression'].predict_proba(latest_scaled)[0]
        predictions['logistic_regression'] = lr_pred[1]
        
        # Ensemble prediction (average)
        ensemble_prob = np.mean(list(predictions.values()))
        
        # Risk categorization
        if ensemble_prob > 0.7:
            risk_level = 'High'
        elif ensemble_prob > 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'deterioration_probability': ensemble_prob,
            'risk_level': risk_level,
            'model_predictions': predictions,
            'recommendation': self._generate_recommendation(ensemble_prob, risk_level),
            'timestamp': datetime.now()
        }
    
    def _generate_recommendation(self, probability: float, risk_level: str) -> str:
        """Generate clinical recommendations based on risk"""
        if risk_level == 'High':
            return "URGENT: Immediate clinical assessment recommended. Consider ICU consultation."
        elif risk_level == 'Medium':
            return "CAUTION: Increase monitoring frequency. Consider physician notification."
        else:
            return "ROUTINE: Continue standard monitoring protocols."

class VitalSignsForecaster:
    """
    LSTM-based forecasting model for vital signs prediction
    """
    
    def __init__(self, sequence_length: int = 12, forecast_horizon: int = 6):
        self.sequence_length = sequence_length  # Number of past measurements to use
        self.forecast_horizon = forecast_horizon  # Number of future points to predict
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.feature_columns = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'oxygen_saturation', 'temperature']
    
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        # Select and scale features
        features = data[self.feature_columns].values
        scaled_features = self.scaler.fit_transform(features)
        
        X, y = [], []
        
        for i in range(len(scaled_features) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X.append(scaled_features[i:(i + self.sequence_length)])
            # Target sequence
            y.append(scaled_features[(i + self.sequence_length):(i + self.sequence_length + self.forecast_horizon)])
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models. Please install tensorflow.")
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.forecast_horizon * len(self.feature_columns))
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, training_data: pd.DataFrame):
        """Train the LSTM forecasting model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping LSTM model training.")
            return
            
        logger.info("Training LSTM vital signs forecasting model...")
        
        # Prepare sequences
        X, y = self.prepare_sequences(training_data)
        
        if len(X) < 100:
            logger.error("Insufficient training data for LSTM")
            return
        
        # Reshape y for dense layer output
        y = y.reshape(y.shape[0], -1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and train model
        self.model = self.build_model()
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_trained = True
        
        # Evaluate model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        logger.info(f"Training loss: {train_loss[0]:.4f}")
        logger.info(f"Validation loss: {val_loss[0]:.4f}")
        logger.info("LSTM training completed")
    
    def forecast(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate vital signs forecast"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        if len(recent_data) < self.sequence_length:
            return {'error': f'Need at least {self.sequence_length} recent measurements'}
        
        # Prepare input sequence
        features = recent_data[self.feature_columns].iloc[-self.sequence_length:].values
        scaled_features = self.scaler.transform(features)
        input_sequence = scaled_features.reshape(1, self.sequence_length, len(self.feature_columns))
        
        # Generate forecast
        forecast_scaled = self.model.predict(input_sequence, verbose=0)
        forecast_scaled = forecast_scaled.reshape(self.forecast_horizon, len(self.feature_columns))
        
        # Inverse transform to original scale
        forecast = self.scaler.inverse_transform(forecast_scaled)
        
        # Create forecast DataFrame
        last_timestamp = pd.to_datetime(recent_data['timestamp'].iloc[-1])
        future_timestamps = [
            last_timestamp + timedelta(minutes=5*i) for i in range(1, self.forecast_horizon + 1)
        ]
        
        forecast_df = pd.DataFrame(
            forecast,
            columns=self.feature_columns,
            index=future_timestamps
        )
        
        return {
            'forecast': forecast_df.to_dict('records'),
            'forecast_timestamps': [ts.isoformat() for ts in future_timestamps],
            'confidence_interval': self._calculate_confidence_interval(forecast),
            'forecast_horizon_minutes': self.forecast_horizon * 5,
            'timestamp': datetime.now()
        }
    
    def _calculate_confidence_interval(self, forecast: np.ndarray, confidence: float = 0.95) -> Dict:
        """Calculate confidence intervals for forecast"""
        # Simple method: use historical forecast error
        # In practice, this would be based on model uncertainty
        error_margin = np.std(forecast, axis=0) * 1.96  # 95% CI approximation
        
        return {
            'lower_bound': (forecast - error_margin).tolist(),
            'upper_bound': (forecast + error_margin).tolist(),
            'confidence_level': confidence
        }

class SepsisRiskPredictor:
    """
    Specialized model for early sepsis detection
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_sepsis_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features relevant to sepsis detection"""
        features = pd.DataFrame()
        
        # SIRS (Systemic Inflammatory Response Syndrome) criteria
        features['temperature_abnormal'] = (
            (data['temperature'] > 38.0) | (data['temperature'] < 36.0)
        ).astype(int)
        
        features['heart_rate_high'] = (data['heart_rate'] > 90).astype(int)
        features['respiratory_rate_high'] = (data['respiratory_rate'] > 20).astype(int)
        
        # qSOFA criteria
        features['altered_mental_status'] = 0  # Would need additional data
        features['systolic_bp_low'] = (data['systolic_bp'] <= 100).astype(int)
        features['respiratory_rate_very_high'] = (data['respiratory_rate'] >= 22).astype(int)
        
        # Additional sepsis indicators
        features['temp_hr_interaction'] = data['temperature'] * data['heart_rate']
        features['shock_index'] = data['heart_rate'] / data['systolic_bp']
        features['pulse_pressure'] = data['systolic_bp'] - data['diastolic_bp']
        
        # Trends (if enough data)
        if len(data) >= 3:
            for vital in ['temperature', 'heart_rate', 'systolic_bp']:
                features[f'{vital}_trend'] = data[vital].rolling(3).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 3 else 0
                )
        
        features = features.fillna(0)
        return features
    
    def calculate_sirs_score(self, data: pd.DataFrame) -> int:
        """Calculate SIRS score"""
        score = 0
        
        latest = data.iloc[-1]
        
        # Temperature
        if latest['temperature'] > 38.0 or latest['temperature'] < 36.0:
            score += 1
        
        # Heart rate
        if latest['heart_rate'] > 90:
            score += 1
        
        # Respiratory rate
        if latest['respiratory_rate'] > 20:
            score += 1
        
        # Would need WBC count for full SIRS score
        
        return score
    
    def calculate_qsofa_score(self, data: pd.DataFrame) -> int:
        """Calculate qSOFA score"""
        score = 0
        
        latest = data.iloc[-1]
        
        # Respiratory rate ≥ 22
        if latest['respiratory_rate'] >= 22:
            score += 1
        
        # Altered mental status (would need GCS)
        # score += 0
        
        # Systolic BP ≤ 100
        if latest['systolic_bp'] <= 100:
            score += 1
        
        return score

def save_predictive_models(deterioration_model: HealthDeteriorationPredictor, 
                          forecaster: VitalSignsForecaster, 
                          sepsis_model: SepsisRiskPredictor,
                          base_path: str = "data/models/"):
    """Save all trained models"""
    import os
    os.makedirs(base_path, exist_ok=True)
    
    # Save deterioration model
    if deterioration_model.is_trained:
        joblib.dump(deterioration_model.models, f"{base_path}deterioration_models.pkl")
        joblib.dump(deterioration_model.scalers, f"{base_path}deterioration_scalers.pkl")
        logger.info("Deterioration prediction models saved")
    
    # Save LSTM forecaster
    if forecaster.is_trained and forecaster.model:
        forecaster.model.save(f"{base_path}lstm_forecaster.h5")
        joblib.dump(forecaster.scaler, f"{base_path}lstm_scaler.pkl")
        logger.info("LSTM forecasting model saved")
    
    # Save sepsis model
    if sepsis_model.is_trained and sepsis_model.model:
        joblib.dump(sepsis_model.model, f"{base_path}sepsis_model.pkl")
        joblib.dump(sepsis_model.scaler, f"{base_path}sepsis_scaler.pkl")
        logger.info("Sepsis prediction model saved")

def demo_predictive_models():
    """Demonstration of predictive models"""
    print("Predictive Models Demo")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='5min')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'heart_rate': np.random.normal(75, 10, 200),
        'systolic_bp': np.random.normal(120, 15, 200),
        'diastolic_bp': np.random.normal(80, 10, 200),
        'oxygen_saturation': np.random.normal(97, 2, 200),
        'temperature': np.random.normal(36.5, 0.5, 200),
        'respiratory_rate': np.random.normal(16, 3, 200)
    })
    
    # Add some deterioration patterns
    sample_data.loc[150:160, 'heart_rate'] += 30
    sample_data.loc[150:160, 'temperature'] += 2
    
    print(f"Generated {len(sample_data)} sample measurements")
    
    # Test deterioration predictor
    print("\\n1. Testing Health Deterioration Predictor...")
    deterioration_model = HealthDeteriorationPredictor()
    deterioration_model.train(sample_data)
    
    if deterioration_model.is_trained:
        recent_data = sample_data.tail(20)
        prediction = deterioration_model.predict_deterioration(recent_data)
        print(f"   Deterioration Risk: {prediction['risk_level']}")
        print(f"   Probability: {prediction['deterioration_probability']:.2%}")
    
    # Test LSTM forecaster
    print("\\n2. Testing LSTM Vital Signs Forecaster...")
    forecaster = VitalSignsForecaster(sequence_length=12, forecast_horizon=6)
    forecaster.train(sample_data)
    
    if forecaster.is_trained:
        recent_data = sample_data.tail(15)
        forecast = forecaster.forecast(recent_data)
        if 'forecast' in forecast:
            print(f"   Generated forecast for next {forecast['forecast_horizon_minutes']} minutes")
            print(f"   Forecasted heart rate: {forecast['forecast'][0]['heart_rate']:.1f} bpm")
    
    print("\\nPredictive models demonstration completed!")

if __name__ == "__main__":
    demo_predictive_models()
