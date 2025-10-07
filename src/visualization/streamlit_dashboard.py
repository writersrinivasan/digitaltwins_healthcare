"""
Streamlit Dashboard for Virtual Patient Monitor
Real-time visualization and monitoring interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import json
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'simulation'))

from data.simulation.patient_data_generator import SyntheticDataGenerator, PatientProfile
from src.core.data_streamer import DataStreamer
from src.ml.anomaly_detector import AnomalyDetector

# Page configuration
st.set_page_config(
    page_title="Virtual Patient Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DashboardState:
    """Manages dashboard state and data"""
    
    def __init__(self):
        if 'data_streamer' not in st.session_state:
            st.session_state.data_streamer = DataStreamer(update_interval=5)
            st.session_state.anomaly_detector = AnomalyDetector()
            st.session_state.patients = {}
            st.session_state.last_update = datetime.now()
            st.session_state.streaming_active = False
            st.session_state.alert_history = []
    
    def initialize_demo_patients(self):
        """Initialize demo patients for the dashboard"""
        generator = SyntheticDataGenerator()
        
        # Create diverse patient profiles
        patient_profiles = [
            {
                'age': 65, 'gender': 'Male', 'conditions': ['hypertension', 'diabetes'],
                'name': 'John Smith', 'room': 'ICU-101'
            },
            {
                'age': 45, 'gender': 'Female', 'conditions': ['asthma'],
                'name': 'Sarah Johnson', 'room': 'Ward-205'
            },
            {
                'age': 78, 'gender': 'Male', 'conditions': ['heart_disease', 'hypertension'],
                'name': 'Robert Wilson', 'room': 'ICU-103'
            }
        ]
        
        for profile_data in patient_profiles:
            profile = generator.create_patient_profile()
            profile.age = profile_data['age']
            profile.gender = profile_data['gender']
            profile.medical_conditions = profile_data['conditions']
            
            st.session_state.data_streamer.add_patient(profile)
            st.session_state.patients[profile.patient_id] = {
                'profile': profile,
                'name': profile_data['name'],
                'room': profile_data['room'],
                'status': 'Stable'
            }

def create_vital_signs_chart(patient_data: pd.DataFrame, vital_name: str, 
                           normal_range: tuple, title: str) -> go.Figure:
    """Create a time series chart for vital signs"""
    fig = go.Figure()
    
    if not patient_data.empty and vital_name in patient_data.columns:
        # Main vital signs line
        fig.add_trace(go.Scatter(
            x=patient_data['timestamp'],
            y=patient_data[vital_name],
            mode='lines+markers',
            name=title,
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        # Normal range shading
        if normal_range:
            fig.add_hrect(
                y0=normal_range[0], y1=normal_range[1],
                fillcolor="rgba(0,255,0,0.1)",
                layer="below",
                line_width=0,
                annotation_text="Normal Range",
                annotation_position="top left"
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=vital_name.replace('_', ' ').title(),
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_multi_vital_chart(patient_data: pd.DataFrame) -> go.Figure:
    """Create a comprehensive multi-vital chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Heart Rate', 'Blood Pressure', 'Temperature', 'Oxygen Saturation'),
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    if not patient_data.empty:
        # Heart Rate
        fig.add_trace(
            go.Scatter(x=patient_data['timestamp'], y=patient_data['heart_rate'],
                      mode='lines', name='Heart Rate', line=dict(color='red')),
            row=1, col=1
        )
        
        # Blood Pressure
        fig.add_trace(
            go.Scatter(x=patient_data['timestamp'], y=patient_data['systolic_bp'],
                      mode='lines', name='Systolic', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=patient_data['timestamp'], y=patient_data['diastolic_bp'],
                      mode='lines', name='Diastolic', line=dict(color='lightblue')),
            row=1, col=2
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=patient_data['timestamp'], y=patient_data['temperature'],
                      mode='lines', name='Temperature', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Oxygen Saturation
        fig.add_trace(
            go.Scatter(x=patient_data['timestamp'], y=patient_data['oxygen_saturation'],
                      mode='lines', name='SpO2', line=dict(color='green')),
            row=2, col=2
        )
    
    fig.update_layout(height=500, showlegend=False, margin=dict(l=0, r=0, t=50, b=0))
    
    # Update y-axis ranges for better visualization
    fig.update_yaxes(range=[40, 150], row=1, col=1)  # Heart rate
    fig.update_yaxes(range=[60, 200], row=1, col=2)  # Blood pressure
    fig.update_yaxes(range=[35, 40], row=2, col=1)   # Temperature
    fig.update_yaxes(range=[85, 100], row=2, col=2)  # Oxygen saturation
    
    return fig

def evaluate_vital_status(value: float, normal_range: tuple) -> str:
    """Evaluate if a vital sign is normal, warning, or critical"""
    if normal_range[0] <= value <= normal_range[1]:
        return "normal"
    elif normal_range[0] * 0.9 <= value <= normal_range[1] * 1.1:
        return "warning"
    else:
        return "critical"

def create_patient_overview():
    """Create patient overview section"""
    st.markdown("### 👥 Patient Overview")
    
    if not st.session_state.patients:
        st.warning("No patients in the system. Initialize demo patients to get started.")
        return
    
    cols = st.columns(len(st.session_state.patients))
    
    for idx, (patient_id, patient_info) in enumerate(st.session_state.patients.items()):
        with cols[idx]:
            # Get latest vital signs
            latest_data = st.session_state.data_streamer.get_latest_data(patient_id, 1)
            
            if latest_data:
                vitals = latest_data[0]
                
                # Determine overall status
                statuses = []
                statuses.append(evaluate_vital_status(vitals.heart_rate, (60, 100)))
                statuses.append(evaluate_vital_status(vitals.systolic_bp, (90, 140)))
                statuses.append(evaluate_vital_status(vitals.oxygen_saturation, (95, 100)))
                statuses.append(evaluate_vital_status(vitals.temperature, (36.1, 37.2)))
                
                if "critical" in statuses:
                    status_color = "🔴"
                    status_text = "Critical"
                elif "warning" in statuses:
                    status_color = "🟡"
                    status_text = "Warning"
                else:
                    status_color = "🟢"
                    status_text = "Stable"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{status_color} {patient_info['name']}</h4>
                    <p><strong>Room:</strong> {patient_info['room']}</p>
                    <p><strong>Status:</strong> {status_text}</p>
                    <p><strong>Last Update:</strong> {vitals.timestamp.strftime('%H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Key metrics
                st.metric("Heart Rate", f"{vitals.heart_rate:.0f} bpm")
                st.metric("Blood Pressure", f"{vitals.systolic_bp:.0f}/{vitals.diastolic_bp:.0f}")
                st.metric("SpO2", f"{vitals.oxygen_saturation:.1f}%")
                
            else:
                st.warning(f"No data available for {patient_info['name']}")

def create_detailed_patient_view(patient_id: str):
    """Create detailed view for a specific patient"""
    if patient_id not in st.session_state.patients:
        st.error("Patient not found")
        return
    
    patient_info = st.session_state.patients[patient_id]
    
    st.markdown(f"### 📊 Detailed View - {patient_info['name']}")
    
    # Time range selector
    col1, col2 = st.columns([3, 1])
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours"],
            index=0
        )
    
    with col2:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    
    # Convert time range to minutes
    time_mapping = {
        "Last 1 Hour": 60,
        "Last 6 Hours": 360,
        "Last 24 Hours": 1440
    }
    minutes = time_mapping[time_range]
    
    # Get patient data
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=minutes)
    
    patient_data = st.session_state.data_streamer.get_time_range_data(
        patient_id, start_time, end_time
    )
    
    if not patient_data:
        st.warning("No data available for the selected time range")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'timestamp': vs.timestamp,
            'heart_rate': vs.heart_rate,
            'systolic_bp': vs.systolic_bp,
            'diastolic_bp': vs.diastolic_bp,
            'oxygen_saturation': vs.oxygen_saturation,
            'temperature': vs.temperature,
            'respiratory_rate': vs.respiratory_rate,
            'glucose_level': vs.glucose_level
        }
        for vs in patient_data
    ])
    
    # Current vitals display
    if not df.empty:
        latest = df.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta_hr = latest['heart_rate'] - df.iloc[-2]['heart_rate'] if len(df) > 1 else 0
            st.metric("Heart Rate", f"{latest['heart_rate']:.0f} bpm", f"{delta_hr:+.1f}")
        
        with col2:
            st.metric("Blood Pressure", 
                     f"{latest['systolic_bp']:.0f}/{latest['diastolic_bp']:.0f} mmHg")
        
        with col3:
            delta_spo2 = latest['oxygen_saturation'] - df.iloc[-2]['oxygen_saturation'] if len(df) > 1 else 0
            st.metric("SpO2", f"{latest['oxygen_saturation']:.1f}%", f"{delta_spo2:+.1f}")
        
        with col4:
            delta_temp = latest['temperature'] - df.iloc[-2]['temperature'] if len(df) > 1 else 0
            st.metric("Temperature", f"{latest['temperature']:.1f}°C", f"{delta_temp:+.1f}")
    
    # Multi-vital chart
    st.plotly_chart(create_multi_vital_chart(df), use_container_width=True)
    
    # Individual vital charts
    col1, col2 = st.columns(2)
    
    with col1:
        hr_chart = create_vital_signs_chart(df, 'heart_rate', (60, 100), 'Heart Rate (bpm)')
        st.plotly_chart(hr_chart, use_container_width=True)
        
        temp_chart = create_vital_signs_chart(df, 'temperature', (36.1, 37.2), 'Temperature (°C)')
        st.plotly_chart(temp_chart, use_container_width=True)
    
    with col2:
        spo2_chart = create_vital_signs_chart(df, 'oxygen_saturation', (95, 100), 'Oxygen Saturation (%)')
        st.plotly_chart(spo2_chart, use_container_width=True)
        
        glucose_chart = create_vital_signs_chart(df, 'glucose_level', (70, 140), 'Glucose Level (mg/dL)')
        st.plotly_chart(glucose_chart, use_container_width=True)
    
    # Anomaly detection results
    st.markdown("### 🔍 Anomaly Detection")
    
    # Run anomaly detection on recent data
    if len(df) > 10:
        features = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'oxygen_saturation', 'temperature']
        feature_data = df[features].values
        
        # Simple anomaly detection (this would be more sophisticated in practice)
        anomalies = []
        for i, row in df.iterrows():
            score = 0
            if not (60 <= row['heart_rate'] <= 100):
                score += 1
            if not (90 <= row['systolic_bp'] <= 140):
                score += 1
            if not (95 <= row['oxygen_saturation'] <= 100):
                score += 1
            if not (36.1 <= row['temperature'] <= 37.2):
                score += 1
            
            if score >= 2:
                anomalies.append({
                    'timestamp': row['timestamp'],
                    'severity': 'High' if score >= 3 else 'Medium',
                    'description': f"Multiple vital signs abnormal (score: {score})"
                })
        
        if anomalies:
            st.warning(f"⚠️ {len(anomalies)} anomalies detected in the selected time range")
            for anomaly in anomalies[-5:]:  # Show last 5 anomalies
                severity_class = "alert-high" if anomaly['severity'] == 'High' else "alert-medium"
                st.markdown(f"""
                <div class="{severity_class}">
                    <strong>{anomaly['severity']} Alert</strong> - {anomaly['timestamp'].strftime('%H:%M:%S')}<br>
                    {anomaly['description']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No anomalies detected in the selected time range")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()

def main():
    """Main dashboard application"""
    # Initialize dashboard state
    dashboard = DashboardState()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Virtual Patient Monitor</h1>', unsafe_allow_html=True)
    st.markdown("*Real-time Digital Twin Healthcare Monitoring System*")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Initialize demo data
        if st.button("🚀 Initialize Demo Patients"):
            dashboard.initialize_demo_patients()
            st.success("Demo patients initialized!")
            st.rerun()
        
        # Streaming controls
        st.markdown("### Data Streaming")
        
        if not st.session_state.streaming_active:
            if st.button("▶️ Start Streaming"):
                st.session_state.data_streamer.start_streaming()
                st.session_state.streaming_active = True
                st.success("Streaming started!")
                st.rerun()
        else:
            if st.button("⏹️ Stop Streaming"):
                st.session_state.data_streamer.stop_streaming()
                st.session_state.streaming_active = False
                st.success("Streaming stopped!")
                st.rerun()
        
        # Streaming status
        status_color = "🟢" if st.session_state.streaming_active else "🔴"
        st.markdown(f"**Status:** {status_color} {'Active' if st.session_state.streaming_active else 'Inactive'}")
        
        # Patient selection
        if st.session_state.patients:
            st.markdown("### Patient Selection")
            selected_patient = st.selectbox(
                "Select Patient for Detailed View",
                options=list(st.session_state.patients.keys()),
                format_func=lambda x: st.session_state.patients[x]['name']
            )
        else:
            selected_patient = None
        
        # System info
        st.markdown("### System Information")
        st.info(f"""
        **Patients Monitored:** {len(st.session_state.patients)}
        **Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}
        **Streaming Interval:** 5 seconds
        """)
    
    # Main content area
    if not st.session_state.patients:
        st.info("👋 Welcome to the Virtual Patient Monitor! Click 'Initialize Demo Patients' in the sidebar to get started.")
    else:
        # Patient overview
        create_patient_overview()
        
        st.markdown("---")
        
        # Detailed patient view
        if selected_patient:
            create_detailed_patient_view(selected_patient)
        
        # Auto-refresh for overview
        if st.session_state.streaming_active:
            # Update last update time
            st.session_state.last_update = datetime.now()
            
            # Refresh every 10 seconds
            time.sleep(10)
            st.rerun()

if __name__ == "__main__":
    main()
