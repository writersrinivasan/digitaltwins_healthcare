#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os

def test_core_imports():
    """Test core module imports"""
    print("Testing core module imports...")
    
    try:
        # Test data generator import
        sys.path.append(os.path.join(os.path.dirname(__file__), 'data', 'simulation'))
        from patient_data_generator import SyntheticDataGenerator, VitalSigns
        print("✓ patient_data_generator import successful")
        
        # Test data streamer import
        from src.core.data_streamer import DataStreamer, MQTTStreamer, WebSocketStreamer
        print("✓ data_streamer import successful")
        
        # Test digital twin import
        from src.core.digital_twin import DigitalTwin
        print("✓ digital_twin import successful")
        
        # Test vital signs processor
        from src.core.vital_signs_processor import VitalSignsProcessor
        print("✓ vital_signs_processor import successful")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

def test_ml_imports():
    """Test ML module imports"""
    print("\nTesting ML module imports...")
    
    try:
        from src.ml.anomaly_detector import AnomalyDetector
        print("✓ anomaly_detector import successful")
        
        from src.ml.predictive_models import HealthDeteriorationPredictor, VitalSignsForecaster
        print("✓ predictive_models import successful")
        
        from src.ml.feature_engineering import VitalSignsFeatureEngineer
        print("✓ feature_engineering import successful")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

def test_visualization_imports():
    """Test visualization module imports"""
    print("\nTesting visualization module imports...")
    
    try:
        # These might fail if streamlit isn't installed, but we'll catch that
        from src.visualization.plotly_charts import MedicalChartGenerator
        print("✓ plotly_charts import successful")
        
        # Skip streamlit dashboard as it requires streamlit runtime
        print("! Skipping streamlit_dashboard (requires streamlit runtime)")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

def test_optional_dependencies():
    """Test optional dependencies"""
    print("\nTesting optional dependencies...")
    
    # Test MQTT
    try:
        import paho.mqtt.client as mqtt
        print("✓ paho-mqtt available")
    except ImportError:
        print("! paho-mqtt not available (optional)")
    
    # Test WebSockets
    try:
        import websockets
        print("✓ websockets available")
    except ImportError:
        print("! websockets not available (optional)")
    
    # Test schedule
    try:
        import schedule
        print("✓ schedule available")
    except ImportError:
        print("! schedule not available")

def main():
    """Run all import tests"""
    print("Virtual Patient Monitor - Import Test")
    print("=" * 50)
    
    core_ok = test_core_imports()
    ml_ok = test_ml_imports()
    viz_ok = test_visualization_imports()
    test_optional_dependencies()
    
    print("\n" + "=" * 50)
    if core_ok and ml_ok and viz_ok:
        print("✓ All critical imports working correctly!")
        return True
    else:
        print("✗ Some imports failed. Check error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
