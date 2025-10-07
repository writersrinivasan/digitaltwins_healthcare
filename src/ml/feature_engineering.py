"""
Feature Engineering for Healthcare Analytics
Advanced feature extraction and transformation for medical data
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class VitalSignsFeatureEngineer:
    """
    Comprehensive feature engineering for vital signs data
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        self.baseline_stats = {}
        
    def extract_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract basic derived features from vital signs"""
        features = data.copy()
        
        # Cardiovascular derived metrics
        if 'systolic_bp' in data.columns and 'diastolic_bp' in data.columns:
            features['pulse_pressure'] = data['systolic_bp'] - data['diastolic_bp']
            features['mean_arterial_pressure'] = (data['systolic_bp'] + 2 * data['diastolic_bp']) / 3
            
        # Shock index (heart rate / systolic BP)
        if 'heart_rate' in data.columns and 'systolic_bp' in data.columns:
            features['shock_index'] = data['heart_rate'] / data['systolic_bp']
            
        # Rate-pressure product (cardiac workload indicator)
        if 'heart_rate' in data.columns and 'systolic_bp' in data.columns:
            features['rate_pressure_product'] = data['heart_rate'] * data['systolic_bp']
            
        # Perfusion index approximation
        if 'oxygen_saturation' in data.columns and 'heart_rate' in data.columns:
            features['perfusion_index'] = data['oxygen_saturation'] / data['heart_rate']
            
        return features
    
    def extract_temporal_features(self, data: pd.DataFrame, 
                                timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Extract time-based features"""
        features = pd.DataFrame(index=data.index)
        
        if timestamp_col in data.columns:
            timestamps = pd.to_datetime(data[timestamp_col])
            
            # Time of day features
            features['hour'] = timestamps.dt.hour
            features['minute'] = timestamps.dt.minute
            features['day_of_week'] = timestamps.dt.dayofweek
            features['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)
            
            # Circadian rhythm encoding
            features['hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
            
            # Work hours indicator
            features['is_work_hours'] = (
                (timestamps.dt.hour >= 8) & (timestamps.dt.hour <= 17) & 
                (timestamps.dt.dayofweek < 5)
            ).astype(int)
            
            # Night shift indicator
            features['is_night_shift'] = (
                (timestamps.dt.hour >= 22) | (timestamps.dt.hour <= 6)
            ).astype(int)
            
            # Time since midnight
            features['minutes_since_midnight'] = timestamps.dt.hour * 60 + timestamps.dt.minute
        
        return features
    
    def extract_statistical_features(self, data: pd.DataFrame, 
                                   windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """Extract rolling statistical features"""
        features = pd.DataFrame(index=data.index)
        
        vital_columns = [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 
            'oxygen_saturation', 'temperature', 'respiratory_rate'
        ]
        
        for window in windows:
            for col in vital_columns:
                if col in data.columns:
                    # Basic statistics
                    features[f'{col}_mean_{window}'] = data[col].rolling(window, min_periods=1).mean()
                    features[f'{col}_std_{window}'] = data[col].rolling(window, min_periods=1).std()
                    features[f'{col}_min_{window}'] = data[col].rolling(window, min_periods=1).min()
                    features[f'{col}_max_{window}'] = data[col].rolling(window, min_periods=1).max()
                    features[f'{col}_range_{window}'] = (
                        features[f'{col}_max_{window}'] - features[f'{col}_min_{window}']
                    )
                    
                    # Percentiles
                    features[f'{col}_q25_{window}'] = data[col].rolling(window, min_periods=1).quantile(0.25)
                    features[f'{col}_q75_{window}'] = data[col].rolling(window, min_periods=1).quantile(0.75)
                    features[f'{col}_median_{window}'] = data[col].rolling(window, min_periods=1).median()
                    
                    # Variability measures
                    features[f'{col}_cv_{window}'] = (
                        features[f'{col}_std_{window}'] / features[f'{col}_mean_{window}']
                    )
                    
                    # Skewness and kurtosis
                    features[f'{col}_skew_{window}'] = data[col].rolling(window, min_periods=3).skew()
                    features[f'{col}_kurt_{window}'] = data[col].rolling(window, min_periods=4).kurt()
        
        return features.fillna(method='ffill').fillna(0)
    
    def extract_trend_features(self, data: pd.DataFrame, 
                             windows: List[int] = [6, 12, 24]) -> pd.DataFrame:
        """Extract trend and change-based features"""
        features = pd.DataFrame(index=data.index)
        
        vital_columns = [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 
            'oxygen_saturation', 'temperature', 'respiratory_rate'
        ]
        
        for col in vital_columns:
            if col in data.columns:
                # Simple differences
                features[f'{col}_diff_1'] = data[col].diff(1)
                features[f'{col}_diff_2'] = data[col].diff(2)
                
                # Percentage changes
                features[f'{col}_pct_change_1'] = data[col].pct_change(1)
                features[f'{col}_pct_change_6'] = data[col].pct_change(6)
                
                # Trend slopes for different windows
                for window in windows:
                    features[f'{col}_trend_{window}'] = data[col].rolling(window, min_periods=2).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )
                    
                    # Trend acceleration (second derivative)
                    if window >= 3:
                        features[f'{col}_accel_{window}'] = (
                            features[f'{col}_trend_{window}'].diff(1)
                        )
                
                # Momentum indicators
                features[f'{col}_momentum_3'] = data[col] - data[col].shift(3)
                features[f'{col}_momentum_6'] = data[col] - data[col].shift(6)
                
                # Rate of change
                features[f'{col}_roc_3'] = (
                    (data[col] - data[col].shift(3)) / data[col].shift(3) * 100
                )
        
        return features.fillna(0)
    
    def extract_frequency_features(self, data: pd.DataFrame, 
                                 sampling_rate: float = 1/300) -> pd.DataFrame:  # 5-min intervals
        """Extract frequency domain features"""
        features = pd.DataFrame(index=data.index)
        
        vital_columns = [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 
            'oxygen_saturation', 'temperature'
        ]
        
        window_size = min(64, len(data))  # Ensure we have enough data for FFT
        
        if window_size < 8:
            logger.warning("Insufficient data for frequency analysis")
            return features
        
        for col in vital_columns:
            if col in data.columns:
                # Rolling FFT features
                for i in range(window_size - 1, len(data)):
                    window_data = data[col].iloc[i-window_size+1:i+1].values
                    
                    if len(window_data) == window_size and not np.any(np.isnan(window_data)):
                        # Compute FFT
                        fft = np.fft.fft(window_data)
                        freqs = np.fft.fftfreq(window_size, 1/sampling_rate)
                        
                        # Power spectral density
                        psd = np.abs(fft) ** 2
                        
                        # Frequency domain features
                        features.loc[data.index[i], f'{col}_dominant_freq'] = freqs[np.argmax(psd[1:window_size//2]) + 1]
                        features.loc[data.index[i], f'{col}_spectral_energy'] = np.sum(psd)
                        features.loc[data.index[i], f'{col}_spectral_centroid'] = (
                            np.sum(freqs[:window_size//2] * psd[:window_size//2]) / np.sum(psd[:window_size//2])
                        )
                        
                        # Band power ratios
                        low_freq_power = np.sum(psd[1:window_size//8])  # Low frequency
                        high_freq_power = np.sum(psd[window_size//8:window_size//2])  # High frequency
                        
                        if high_freq_power > 0:
                            features.loc[data.index[i], f'{col}_lf_hf_ratio'] = low_freq_power / high_freq_power
        
        return features.fillna(method='ffill').fillna(0)
    
    def extract_complexity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract complexity and entropy-based features"""
        features = pd.DataFrame(index=data.index)
        
        vital_columns = [
            'heart_rate', 'systolic_bp', 'diastolic_bp', 
            'oxygen_saturation', 'temperature'
        ]
        
        window_size = 20  # Window for complexity analysis
        
        for col in vital_columns:
            if col in data.columns:
                for i in range(window_size - 1, len(data)):
                    window_data = data[col].iloc[i-window_size+1:i+1].values
                    
                    if len(window_data) == window_size and not np.any(np.isnan(window_data)):
                        # Sample entropy
                        features.loc[data.index[i], f'{col}_sample_entropy'] = self._sample_entropy(window_data)
                        
                        # Approximate entropy
                        features.loc[data.index[i], f'{col}_approx_entropy'] = self._approximate_entropy(window_data)
                        
                        # Permutation entropy
                        features.loc[data.index[i], f'{col}_perm_entropy'] = self._permutation_entropy(window_data)
                        
                        # Detrended fluctuation analysis
                        features.loc[data.index[i], f'{col}_dfa_alpha'] = self._dfa_alpha(window_data)
        
        return features.fillna(method='ffill').fillna(0)
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate sample entropy"""
        if r is None:
            r = 0.2 * np.std(data)
        
        N = len(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i+m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j], m) <= r:
                        C[i] += 1.0
            
            C = C / (N - m + 1.0)
            phi = np.mean(np.log(C))
            return phi
        
        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0
    
    def _approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate approximate entropy"""
        if r is None:
            r = 0.2 * np.std(data)
        
        try:
            def _phi(m):
                N = len(data)
                patterns = np.array([data[i:i+m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template = patterns[i]
                    for j in range(N - m + 1):
                        if max(abs(template - patterns[j])) <= r:
                            C[i] += 1.0
                
                C = C / (N - m + 1.0)
                return np.mean(np.log(C))
            
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0
    
    def _permutation_entropy(self, data: np.ndarray, order: int = 3) -> float:
        """Calculate permutation entropy"""
        try:
            from itertools import permutations
            
            N = len(data)
            ordinal_patterns = {}
            
            for i in range(N - order + 1):
                sorted_index = sorted(range(order), key=lambda k: data[i + k])
                pattern = tuple(sorted_index)
                
                if pattern in ordinal_patterns:
                    ordinal_patterns[pattern] += 1
                else:
                    ordinal_patterns[pattern] = 1
            
            total_patterns = N - order + 1
            probabilities = [count / total_patterns for count in ordinal_patterns.values()]
            
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            max_entropy = np.log2(np.math.factorial(order))
            
            return entropy / max_entropy if max_entropy > 0 else 0.0
        except:
            return 0.0
    
    def _dfa_alpha(self, data: np.ndarray) -> float:
        """Calculate DFA (Detrended Fluctuation Analysis) alpha exponent"""
        try:
            N = len(data)
            if N < 10:
                return 0.5
            
            # Integrate the data
            y = np.cumsum(data - np.mean(data))
            
            # Define box sizes
            box_sizes = np.logspace(0.5, np.log10(N//4), 10).astype(int)
            box_sizes = np.unique(box_sizes)
            
            fluctuations = []
            
            for box_size in box_sizes:
                if box_size < 4:
                    continue
                    
                # Divide into boxes
                n_boxes = N // box_size
                if n_boxes < 1:
                    continue
                
                box_fluctuations = []
                
                for i in range(n_boxes):
                    start_idx = i * box_size
                    end_idx = start_idx + box_size
                    box_data = y[start_idx:end_idx]
                    
                    # Fit linear trend
                    x = np.arange(len(box_data))
                    coeffs = np.polyfit(x, box_data, 1)
                    trend = np.polyval(coeffs, x)
                    
                    # Calculate fluctuation
                    fluctuation = np.sqrt(np.mean((box_data - trend) ** 2))
                    box_fluctuations.append(fluctuation)
                
                if box_fluctuations:
                    fluctuations.append(np.mean(box_fluctuations))
            
            if len(fluctuations) < 3:
                return 0.5
            
            # Fit power law: F(n) ~ n^alpha
            log_box_sizes = np.log10(box_sizes[:len(fluctuations)])
            log_fluctuations = np.log10(fluctuations)
            
            alpha = np.polyfit(log_box_sizes, log_fluctuations, 1)[0]
            
            return max(0.0, min(2.0, alpha))  # Clamp between 0 and 2
        except:
            return 0.5
    
    def extract_clinical_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract clinical severity scores"""
        features = pd.DataFrame(index=data.index)
        
        # Modified Early Warning Score (MEWS)
        if all(col in data.columns for col in ['systolic_bp', 'heart_rate', 'respiratory_rate', 'temperature']):
            mews_scores = []
            
            for _, row in data.iterrows():
                score = 0
                
                # Systolic BP scoring
                if row['systolic_bp'] <= 70:
                    score += 3
                elif row['systolic_bp'] <= 80:
                    score += 2
                elif row['systolic_bp'] <= 100:
                    score += 1
                elif row['systolic_bp'] >= 200:
                    score += 2
                
                # Heart rate scoring
                if row['heart_rate'] <= 40:
                    score += 2
                elif row['heart_rate'] <= 50:
                    score += 1
                elif row['heart_rate'] >= 100:
                    score += 1
                elif row['heart_rate'] >= 110:
                    score += 2
                elif row['heart_rate'] >= 130:
                    score += 3
                
                # Respiratory rate scoring
                if row['respiratory_rate'] <= 8:
                    score += 2
                elif row['respiratory_rate'] >= 15:
                    score += 1
                elif row['respiratory_rate'] >= 21:
                    score += 2
                elif row['respiratory_rate'] >= 30:
                    score += 3
                
                # Temperature scoring
                if row['temperature'] <= 35:
                    score += 2
                elif row['temperature'] >= 38.5:
                    score += 2
                
                mews_scores.append(score)
            
            features['mews_score'] = mews_scores
        
        # SIRS (Systemic Inflammatory Response Syndrome) criteria
        if all(col in data.columns for col in ['temperature', 'heart_rate', 'respiratory_rate']):
            sirs_scores = []
            
            for _, row in data.iterrows():
                score = 0
                
                if row['temperature'] > 38 or row['temperature'] < 36:
                    score += 1
                if row['heart_rate'] > 90:
                    score += 1
                if row['respiratory_rate'] > 20:
                    score += 1
                # Would need WBC count for full SIRS
                
                sirs_scores.append(score)
            
            features['sirs_score'] = sirs_scores
        
        return features
    
    def create_feature_pipeline(self, data: pd.DataFrame, 
                              include_basic: bool = True,
                              include_temporal: bool = True,
                              include_statistical: bool = True,
                              include_trends: bool = True,
                              include_frequency: bool = False,
                              include_complexity: bool = False,
                              include_clinical: bool = True) -> pd.DataFrame:
        """Create comprehensive feature engineering pipeline"""
        
        logger.info("Starting feature engineering pipeline...")
        all_features = [data.copy()]
        
        if include_basic:
            logger.info("Extracting basic derived features...")
            basic_features = self.extract_basic_features(data)
            all_features.append(basic_features)
        
        if include_temporal:
            logger.info("Extracting temporal features...")
            temporal_features = self.extract_temporal_features(data)
            all_features.append(temporal_features)
        
        if include_statistical:
            logger.info("Extracting statistical features...")
            statistical_features = self.extract_statistical_features(data)
            all_features.append(statistical_features)
        
        if include_trends:
            logger.info("Extracting trend features...")
            trend_features = self.extract_trend_features(data)
            all_features.append(trend_features)
        
        if include_frequency:
            logger.info("Extracting frequency domain features...")
            frequency_features = self.extract_frequency_features(data)
            all_features.append(frequency_features)
        
        if include_complexity:
            logger.info("Extracting complexity features...")
            complexity_features = self.extract_complexity_features(data)
            all_features.append(complexity_features)
        
        if include_clinical:
            logger.info("Extracting clinical scores...")
            clinical_features = self.extract_clinical_scores(data)
            all_features.append(clinical_features)
        
        # Combine all features
        logger.info("Combining all features...")
        combined_features = pd.concat(all_features, axis=1)
        
        # Remove duplicate columns
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
        
        # Store feature names
        self.feature_names = combined_features.columns.tolist()
        
        logger.info(f"Feature engineering completed. Generated {len(self.feature_names)} features.")
        
        return combined_features

def demo_feature_engineering():
    """Demonstration of feature engineering capabilities"""
    print("Feature Engineering Demo")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01 08:00:00', periods=100, freq='5min')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'heart_rate': np.random.normal(75, 10, 100) + 5 * np.sin(np.arange(100) * 0.1),
        'systolic_bp': np.random.normal(120, 15, 100),
        'diastolic_bp': np.random.normal(80, 10, 100),
        'oxygen_saturation': np.random.normal(97, 2, 100),
        'temperature': np.random.normal(36.5, 0.5, 100),
        'respiratory_rate': np.random.normal(16, 3, 100)
    })
    
    print(f"Input data shape: {sample_data.shape}")
    
    # Initialize feature engineer
    engineer = VitalSignsFeatureEngineer()
    
    # Extract different types of features
    print("\\nExtracting features...")
    
    # Basic features
    basic_features = engineer.extract_basic_features(sample_data)
    print(f"Basic features: {basic_features.shape[1] - sample_data.shape[1]} new features")
    
    # Temporal features
    temporal_features = engineer.extract_temporal_features(sample_data)
    print(f"Temporal features: {temporal_features.shape[1]} new features")
    
    # Statistical features
    statistical_features = engineer.extract_statistical_features(sample_data, windows=[3, 6, 12])
    print(f"Statistical features: {statistical_features.shape[1]} new features")
    
    # Trend features
    trend_features = engineer.extract_trend_features(sample_data, windows=[6, 12])
    print(f"Trend features: {trend_features.shape[1]} new features")
    
    # Clinical scores
    clinical_features = engineer.extract_clinical_scores(sample_data)
    print(f"Clinical scores: {clinical_features.shape[1]} new features")
    
    # Full pipeline
    print("\\nRunning full feature engineering pipeline...")
    all_features = engineer.create_feature_pipeline(
        sample_data,
        include_frequency=False,  # Skip for demo speed
        include_complexity=False  # Skip for demo speed
    )
    
    print(f"Total features generated: {all_features.shape[1]}")
    print(f"Feature engineering completed successfully!")
    
    # Show sample of feature names
    print("\\nSample of generated features:")
    for i, feature in enumerate(engineer.feature_names[:20]):
        print(f"  {i+1:2d}. {feature}")
    
    if len(engineer.feature_names) > 20:
        print(f"  ... and {len(engineer.feature_names) - 20} more features")

if __name__ == "__main__":
    demo_feature_engineering()
