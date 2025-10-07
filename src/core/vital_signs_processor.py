"""
Vital Signs Processing Module
Handles data validation, cleaning, and preprocessing of patient vital signs
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VitalSignType(Enum):
    """Enumeration of vital sign types"""
    HEART_RATE = "heart_rate"
    SYSTOLIC_BP = "systolic_bp"
    DIASTOLIC_BP = "diastolic_bp"
    OXYGEN_SATURATION = "oxygen_saturation"
    TEMPERATURE = "temperature"
    RESPIRATORY_RATE = "respiratory_rate"
    GLUCOSE_LEVEL = "glucose_level"

@dataclass
class ValidationResult:
    """Result of vital signs validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    processed_value: Optional[float] = None

class VitalSignsValidator:
    """Validates vital signs against clinical ranges and data quality checks"""
    
    def __init__(self):
        # Clinical normal ranges for adults
        self.normal_ranges = {
            VitalSignType.HEART_RATE: (40, 200),
            VitalSignType.SYSTOLIC_BP: (60, 250),
            VitalSignType.DIASTOLIC_BP: (30, 150),
            VitalSignType.OXYGEN_SATURATION: (70, 100),
            VitalSignType.TEMPERATURE: (32.0, 45.0),
            VitalSignType.RESPIRATORY_RATE: (8, 50),
            VitalSignType.GLUCOSE_LEVEL: (30, 600)
        }
        
        # Physiologically possible ranges (wider than normal)
        self.physiological_ranges = {
            VitalSignType.HEART_RATE: (20, 300),
            VitalSignType.SYSTOLIC_BP: (40, 300),
            VitalSignType.DIASTOLIC_BP: (20, 200),
            VitalSignType.OXYGEN_SATURATION: (50, 100),
            VitalSignType.TEMPERATURE: (25.0, 50.0),
            VitalSignType.RESPIRATORY_RATE: (5, 80),
            VitalSignType.GLUCOSE_LEVEL: (10, 1000)
        }
    
    def validate_vital_sign(self, vital_type: VitalSignType, value: float, 
                           patient_context: Optional[Dict] = None) -> ValidationResult:
        """
        Validate a single vital sign value
        
        Args:
            vital_type: Type of vital sign
            value: Measured value
            patient_context: Optional patient context for personalized validation
            
        Returns:
            ValidationResult with validation status and any issues
        """
        errors = []
        warnings = []
        processed_value = value
        
        # Check for missing or invalid data
        if value is None or np.isnan(value):
            errors.append(f"Missing value for {vital_type.value}")
            return ValidationResult(False, errors, warnings)
        
        # Check physiological possibility
        phys_min, phys_max = self.physiological_ranges[vital_type]
        if value < phys_min or value > phys_max:
            errors.append(f"{vital_type.value} value {value} outside physiological range ({phys_min}-{phys_max})")
            return ValidationResult(False, errors, warnings, processed_value)
        
        # Check normal clinical ranges
        norm_min, norm_max = self.normal_ranges[vital_type]
        if value < norm_min or value > norm_max:
            warnings.append(f"{vital_type.value} value {value} outside normal range ({norm_min}-{norm_max})")
        
        # Additional validations based on vital sign type
        if vital_type == VitalSignType.OXYGEN_SATURATION:
            if value > 100:
                errors.append("Oxygen saturation cannot exceed 100%")
                return ValidationResult(False, errors, warnings, processed_value)
        
        # Patient-specific validations
        if patient_context:
            context_warnings = self._validate_with_context(vital_type, value, patient_context)
            warnings.extend(context_warnings)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, processed_value)
    
    def _validate_with_context(self, vital_type: VitalSignType, value: float, 
                              context: Dict) -> List[str]:
        """Validate vital sign considering patient context"""
        warnings = []
        
        # Age-based adjustments
        age = context.get('age', 0)
        if age > 65:
            # Elderly patients may have different normal ranges
            if vital_type == VitalSignType.SYSTOLIC_BP and value > 150:
                warnings.append("Elevated blood pressure for elderly patient")
        
        # Medical condition considerations
        conditions = context.get('medical_conditions', [])
        if 'hypertension' in conditions and vital_type == VitalSignType.SYSTOLIC_BP:
            if value > 160:
                warnings.append("Significantly elevated BP for hypertensive patient")
        
        if 'copd' in conditions and vital_type == VitalSignType.OXYGEN_SATURATION:
            if value < 88:
                warnings.append("Low oxygen saturation for COPD patient")
        
        return warnings

class VitalSignsProcessor:
    """Processes and cleans vital signs data"""
    
    def __init__(self):
        self.validator = VitalSignsValidator()
        
    def process_vital_signs_batch(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a batch of vital signs data
        
        Args:
            data: DataFrame with vital signs measurements
            
        Returns:
            Dictionary with processed data and validation results
        """
        processed_data = data.copy()
        validation_results = {}
        
        # Process each vital sign column
        for column in data.columns:
            if column == 'timestamp' or column == 'patient_id':
                continue
                
            try:
                vital_type = VitalSignType(column)
                column_results = self._process_column(processed_data, column, vital_type)
                validation_results[column] = column_results
                
            except ValueError:
                # Column is not a recognized vital sign
                logger.warning(f"Unknown vital sign column: {column}")
        
        return {
            'processed_data': processed_data,
            'validation_results': validation_results,
            'summary': self._generate_summary(validation_results)
        }
    
    def _process_column(self, data: pd.DataFrame, column: str, 
                       vital_type: VitalSignType) -> Dict[str, Any]:
        """Process a single vital sign column"""
        results = {
            'total_values': len(data),
            'valid_values': 0,
            'invalid_values': 0,
            'warnings': 0,
            'errors': [],
            'outliers_removed': 0,
            'values_interpolated': 0
        }
        
        # Validate each value
        for idx, value in data[column].items():
            validation = self.validator.validate_vital_sign(vital_type, value)
            
            if validation.is_valid:
                results['valid_values'] += 1
                if validation.warnings:
                    results['warnings'] += len(validation.warnings)
            else:
                results['invalid_values'] += 1
                results['errors'].extend(validation.errors)
                
                # Try to clean/correct the value
                corrected_value = self._attempt_correction(value, vital_type)
                if corrected_value is not None:
                    data.loc[idx, column] = corrected_value
                    results['outliers_removed'] += 1
        
        # Handle missing values
        missing_count = data[column].isna().sum()
        if missing_count > 0:
            # Interpolate missing values
            data[column] = data[column].interpolate(method='linear', limit_direction='both')
            results['values_interpolated'] = missing_count
        
        # Smooth extreme outliers
        data[column] = self._smooth_outliers(data[column], vital_type)
        
        return results
    
    def _attempt_correction(self, value: float, vital_type: VitalSignType) -> Optional[float]:
        """Attempt to correct obviously incorrect values"""
        if np.isnan(value) or value is None:
            return None
        
        # Common sensor errors and corrections
        if vital_type == VitalSignType.HEART_RATE:
            # Sometimes heart rate is recorded as half or double
            if 20 <= value <= 40:
                return value * 2  # Likely half the actual rate
            elif 200 <= value <= 400:
                return value / 2  # Likely double the actual rate
        
        elif vital_type == VitalSignType.TEMPERATURE:
            # Temperature unit confusion (Fahrenheit vs Celsius)
            if 95 <= value <= 110:  # Likely Fahrenheit
                return (value - 32) * 5/9  # Convert to Celsius
        
        elif vital_type == VitalSignType.OXYGEN_SATURATION:
            # Oxygen saturation as decimal vs percentage
            if 0.8 <= value <= 1.0:
                return value * 100  # Convert decimal to percentage
        
        return None
    
    def _smooth_outliers(self, series: pd.Series, vital_type: VitalSignType, 
                        window: int = 5) -> pd.Series:
        """Smooth extreme outliers using rolling median"""
        if len(series) < window:
            return series
        
        # Calculate rolling median and MAD (Median Absolute Deviation)
        rolling_median = series.rolling(window=window, center=True).median()
        rolling_mad = series.rolling(window=window, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        
        # Define outlier threshold (3 MAD from median)
        threshold = 3
        
        # Identify outliers
        outliers = np.abs(series - rolling_median) > threshold * rolling_mad
        
        # Replace outliers with rolling median
        smoothed_series = series.copy()
        smoothed_series[outliers] = rolling_median[outliers]
        
        return smoothed_series
    
    def _generate_summary(self, validation_results: Dict) -> Dict[str, Any]:
        """Generate summary of validation results"""
        total_values = sum(r['total_values'] for r in validation_results.values())
        total_valid = sum(r['valid_values'] for r in validation_results.values())
        total_invalid = sum(r['invalid_values'] for r in validation_results.values())
        total_warnings = sum(r['warnings'] for r in validation_results.values())
        total_corrections = sum(r['outliers_removed'] for r in validation_results.values())
        total_interpolated = sum(r['values_interpolated'] for r in validation_results.values())
        
        return {
            'total_measurements': total_values,
            'valid_measurements': total_valid,
            'invalid_measurements': total_invalid,
            'warnings_generated': total_warnings,
            'values_corrected': total_corrections,
            'values_interpolated': total_interpolated,
            'data_quality_score': (total_valid / total_values) if total_values > 0 else 0,
            'processing_timestamp': datetime.now().isoformat()
        }

class RealTimeProcessor:
    """Real-time vital signs processing for streaming data"""
    
    def __init__(self, buffer_size: int = 100):
        self.processor = VitalSignsProcessor()
        self.patient_buffers = {}  # Buffer for each patient
        self.buffer_size = buffer_size
        
    def process_real_time_measurement(self, patient_id: str, measurement: Dict) -> Dict[str, Any]:
        """
        Process a single real-time measurement
        
        Args:
            patient_id: Patient identifier
            measurement: Dictionary with vital signs measurement
            
        Returns:
            Processing result with validation and any alerts
        """
        # Initialize buffer for new patients
        if patient_id not in self.patient_buffers:
            self.patient_buffers[patient_id] = []
        
        # Add measurement to buffer
        self.patient_buffers[patient_id].append(measurement)
        
        # Maintain buffer size
        if len(self.patient_buffers[patient_id]) > self.buffer_size:
            self.patient_buffers[patient_id] = self.patient_buffers[patient_id][-self.buffer_size:]
        
        # Process current measurement
        result = self._process_single_measurement(measurement)
        
        # Add trend analysis if we have enough history
        if len(self.patient_buffers[patient_id]) > 5:
            trend_analysis = self._analyze_trends(patient_id)
            result['trend_analysis'] = trend_analysis
        
        return result
    
    def _process_single_measurement(self, measurement: Dict) -> Dict[str, Any]:
        """Process a single measurement"""
        result = {
            'timestamp': measurement.get('timestamp', datetime.now()),
            'validation_results': {},
            'alerts': [],
            'quality_score': 1.0
        }
        
        total_vitals = 0
        valid_vitals = 0
        
        # Validate each vital sign
        for key, value in measurement.items():
            if key in ['timestamp', 'patient_id']:
                continue
            
            try:
                vital_type = VitalSignType(key)
                validation = self.processor.validator.validate_vital_sign(vital_type, value)
                
                result['validation_results'][key] = {
                    'is_valid': validation.is_valid,
                    'errors': validation.errors,
                    'warnings': validation.warnings,
                    'processed_value': validation.processed_value
                }
                
                total_vitals += 1
                if validation.is_valid:
                    valid_vitals += 1
                
                # Generate alerts for critical values
                if validation.errors:
                    result['alerts'].append({
                        'type': 'critical',
                        'vital_sign': key,
                        'message': f"Critical error in {key}: {'; '.join(validation.errors)}",
                        'timestamp': result['timestamp']
                    })
                elif validation.warnings:
                    result['alerts'].append({
                        'type': 'warning',
                        'vital_sign': key,
                        'message': f"Warning for {key}: {'; '.join(validation.warnings)}",
                        'timestamp': result['timestamp']
                    })
                
            except ValueError:
                # Unknown vital sign type
                continue
        
        # Calculate quality score
        if total_vitals > 0:
            result['quality_score'] = valid_vitals / total_vitals
        
        return result
    
    def _analyze_trends(self, patient_id: str) -> Dict[str, Any]:
        """Analyze trends in patient's vital signs"""
        buffer = self.patient_buffers[patient_id]
        
        if len(buffer) < 3:
            return {'trend_analysis': 'insufficient_data'}
        
        trends = {}
        
        # Analyze each vital sign
        for vital_sign in [VitalSignType.HEART_RATE, VitalSignType.SYSTOLIC_BP, 
                          VitalSignType.OXYGEN_SATURATION, VitalSignType.TEMPERATURE]:
            values = []
            timestamps = []
            
            for measurement in buffer:
                if vital_sign.value in measurement:
                    values.append(measurement[vital_sign.value])
                    timestamps.append(measurement.get('timestamp', datetime.now()))
            
            if len(values) >= 3:
                trend = self._calculate_trend(values)
                trends[vital_sign.value] = {
                    'direction': trend['direction'],
                    'magnitude': trend['magnitude'],
                    'confidence': trend['confidence']
                }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and magnitude"""
        if len(values) < 3:
            return {'direction': 'stable', 'magnitude': 0, 'confidence': 0}
        
        # Simple linear regression to determine trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Calculate correlation coefficient for confidence
        correlation = np.corrcoef(x, values)[0, 1]
        confidence = abs(correlation)
        
        # Determine trend direction
        if abs(slope) < 0.1:  # Threshold for considering trend significant
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        return {
            'direction': direction,
            'magnitude': abs(slope),
            'confidence': confidence
        }

def demo_vital_signs_processing():
    """Demonstration of vital signs processing"""
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01 10:00:00', periods=100, freq='5min'),
        'patient_id': ['PATIENT_001'] * 100,
        'heart_rate': np.random.normal(75, 10, 100),
        'systolic_bp': np.random.normal(120, 15, 100),
        'diastolic_bp': np.random.normal(80, 10, 100),
        'oxygen_saturation': np.random.normal(97, 2, 100),
        'temperature': np.random.normal(36.5, 0.5, 100)
    })
    
    # Add some outliers and missing values
    data.loc[10, 'heart_rate'] = 300  # Outlier
    data.loc[20, 'oxygen_saturation'] = np.nan  # Missing value
    data.loc[30, 'temperature'] = 98.6  # Fahrenheit instead of Celsius
    
    # Process the data
    processor = VitalSignsProcessor()
    result = processor.process_vital_signs_batch(data)
    
    print("Vital Signs Processing Demo")
    print("=" * 40)
    print(f"Total measurements: {result['summary']['total_measurements']}")
    print(f"Valid measurements: {result['summary']['valid_measurements']}")
    print(f"Invalid measurements: {result['summary']['invalid_measurements']}")
    print(f"Warnings generated: {result['summary']['warnings_generated']}")
    print(f"Values corrected: {result['summary']['values_corrected']}")
    print(f"Values interpolated: {result['summary']['values_interpolated']}")
    print(f"Data quality score: {result['summary']['data_quality_score']:.2%}")

if __name__ == "__main__":
    demo_vital_signs_processing()
