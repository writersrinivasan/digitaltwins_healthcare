"""
Advanced Plotly Charts for Healthcare Visualization
Interactive charts optimized for medical data presentation
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class MedicalChartGenerator:
    """
    Generator for medical-specific interactive charts
    """
    
    def __init__(self):
        # Medical color schemes
        self.colors = {
            'critical': '#FF4444',
            'warning': '#FF8800', 
            'normal': '#4CAF50',
            'excellent': '#2196F3',
            'heart_rate': '#FF6B6B',
            'blood_pressure': '#4ECDC4',
            'oxygen': '#45B7D1',
            'temperature': '#FFA07A',
            'respiratory': '#98D8C8'
        }
        
        # Medical reference ranges
        self.normal_ranges = {
            'heart_rate': (60, 100),
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'oxygen_saturation': (95, 100),
            'temperature': (36.1, 37.2),
            'respiratory_rate': (12, 20)
        }
    
    def create_vital_signs_timeline(self, data: pd.DataFrame, 
                                  patient_name: str = "Patient",
                                  highlight_anomalies: bool = True) -> go.Figure:
        """
        Create comprehensive vital signs timeline with all parameters
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Heart Rate (bpm)', 'Blood Pressure (mmHg)',
                'Oxygen Saturation (%)', 'Temperature (°C)',
                'Respiratory Rate', 'Glucose Level (mg/dL)'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        timestamps = pd.to_datetime(data['timestamp']) if 'timestamp' in data.columns else data.index
        
        # Heart Rate
        if 'heart_rate' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=data['heart_rate'],
                    mode='lines+markers',
                    name='Heart Rate',
                    line=dict(color=self.colors['heart_rate'], width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>Heart Rate</b><br>Time: %{x}<br>Value: %{y} bpm<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add normal range
            normal_range = self.normal_ranges['heart_rate']
            fig.add_hrect(
                y0=normal_range[0], y1=normal_range[1],
                fillcolor="rgba(76, 175, 80, 0.1)",
                layer="below", line_width=0,
                row=1, col=1
            )
        
        # Blood Pressure
        if 'systolic_bp' in data.columns and 'diastolic_bp' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=data['systolic_bp'],
                    mode='lines+markers',
                    name='Systolic BP',
                    line=dict(color=self.colors['blood_pressure'], width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>Systolic BP</b><br>Time: %{x}<br>Value: %{y} mmHg<extra></extra>'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=data['diastolic_bp'],
                    mode='lines+markers',
                    name='Diastolic BP',
                    line=dict(color=self.colors['blood_pressure'], width=2, dash='dash'),
                    marker=dict(size=4),
                    hovertemplate='<b>Diastolic BP</b><br>Time: %{x}<br>Value: %{y} mmHg<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Oxygen Saturation
        if 'oxygen_saturation' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=data['oxygen_saturation'],
                    mode='lines+markers',
                    name='SpO2',
                    line=dict(color=self.colors['oxygen'], width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>Oxygen Saturation</b><br>Time: %{x}<br>Value: %{y}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Critical threshold line
            fig.add_hline(
                y=95, line_dash="dash", line_color="orange",
                annotation_text="Critical Threshold",
                row=2, col=1
            )
        
        # Temperature
        if 'temperature' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=data['temperature'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color=self.colors['temperature'], width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>Temperature</b><br>Time: %{x}<br>Value: %{y}°C<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Normal range
            normal_range = self.normal_ranges['temperature']
            fig.add_hrect(
                y0=normal_range[0], y1=normal_range[1],
                fillcolor="rgba(76, 175, 80, 0.1)",
                layer="below", line_width=0,
                row=2, col=2
            )
        
        # Respiratory Rate
        if 'respiratory_rate' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=data['respiratory_rate'],
                    mode='lines+markers',
                    name='Respiratory Rate',
                    line=dict(color=self.colors['respiratory'], width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>Respiratory Rate</b><br>Time: %{x}<br>Value: %{y} breaths/min<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Glucose Level
        if 'glucose_level' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=data['glucose_level'],
                    mode='lines+markers',
                    name='Glucose',
                    line=dict(color='purple', width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>Glucose Level</b><br>Time: %{x}<br>Value: %{y} mg/dL<extra></extra>'
                ),
                row=3, col=2
            )
            
            # Target range for diabetics
            fig.add_hrect(
                y0=70, y1=140,
                fillcolor="rgba(76, 175, 80, 0.1)",
                layer="below", line_width=0,
                annotation_text="Target Range",
                row=3, col=2
            )
        
        # Highlight anomalies if requested
        if highlight_anomalies and 'anomaly_score' in data.columns:
            anomaly_timestamps = timestamps[data['anomaly_score'] > 0.5]
            
            for timestamp in anomaly_timestamps:
                fig.add_vline(
                    x=timestamp,
                    line_dash="dot",
                    line_color="red",
                    opacity=0.5,
                    annotation_text="Anomaly"
                )
        
        # Update layout
        fig.update_layout(
            title=f"Vital Signs Timeline - {patient_name}",
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="bpm", row=1, col=1)
        fig.update_yaxes(title_text="mmHg", row=1, col=2)
        fig.update_yaxes(title_text="%", row=2, col=1, range=[85, 100])
        fig.update_yaxes(title_text="°C", row=2, col=2)
        fig.update_yaxes(title_text="breaths/min", row=3, col=1)
        fig.update_yaxes(title_text="mg/dL", row=3, col=2)
        
        return fig
    
    def create_real_time_gauge(self, vital_name: str, current_value: float, 
                             normal_range: Tuple[float, float],
                             critical_range: Tuple[float, float] = None) -> go.Figure:
        """Create real-time gauge chart for a single vital sign"""
        
        # Determine gauge range
        gauge_min = min(normal_range[0] * 0.5, current_value * 0.8)
        gauge_max = max(normal_range[1] * 1.5, current_value * 1.2)
        
        # Determine color based on value
        if normal_range[0] <= current_value <= normal_range[1]:
            color = self.colors['normal']
            status = "Normal"
        elif critical_range and (current_value <= critical_range[0] or current_value >= critical_range[1]):
            color = self.colors['critical']
            status = "Critical"
        else:
            color = self.colors['warning']
            status = "Warning"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{vital_name}<br><span style='font-size:0.8em;color:gray'>{status}</span>"},
            delta={'reference': (normal_range[0] + normal_range[1]) / 2},
            gauge={
                'axis': {'range': [None, gauge_max]},
                'bar': {'color': color},
                'steps': [
                    {'range': [gauge_min, normal_range[0]], 'color': "lightgray"},
                    {'range': [normal_range[0], normal_range[1]], 'color': "lightgreen"},
                    {'range': [normal_range[1], gauge_max], 'color': "lightgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': current_value
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        return fig
    
    def create_patient_status_heatmap(self, patients_data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create heatmap showing status of multiple patients over time"""
        
        # Prepare data for heatmap
        patient_ids = list(patients_data.keys())
        all_timestamps = set()
        
        for data in patients_data.values():
            if 'timestamp' in data.columns:
                all_timestamps.update(pd.to_datetime(data['timestamp']))
        
        all_timestamps = sorted(list(all_timestamps))
        
        # Create status matrix
        status_matrix = np.zeros((len(patient_ids), len(all_timestamps)))
        
        for i, patient_id in enumerate(patient_ids):
            data = patients_data[patient_id]
            if 'timestamp' in data.columns:
                timestamps = pd.to_datetime(data['timestamp'])
                
                for j, ts in enumerate(all_timestamps):
                    # Find closest timestamp
                    closest_idx = np.argmin(np.abs(timestamps - ts))
                    
                    if abs((timestamps.iloc[closest_idx] - ts).total_seconds()) <= 300:  # Within 5 minutes
                        # Calculate status score
                        row = data.iloc[closest_idx]
                        score = self._calculate_status_score(row)
                        status_matrix[i, j] = score
                    else:
                        status_matrix[i, j] = np.nan
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=status_matrix,
            x=[ts.strftime('%H:%M') for ts in all_timestamps],
            y=[f"Patient {pid}" for pid in patient_ids],
            colorscale=[
                [0, self.colors['critical']],
                [0.33, self.colors['warning']],
                [0.66, self.colors['normal']],
                [1, self.colors['excellent']]
            ],
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Status Score: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Patient Status Heatmap",
            xaxis_title="Time",
            yaxis_title="Patients",
            height=400
        )
        
        return fig
    
    def _calculate_status_score(self, row: pd.Series) -> float:
        """Calculate overall status score for a patient measurement"""
        score = 0.0
        count = 0
        
        # Heart rate scoring
        if 'heart_rate' in row:
            hr = row['heart_rate']
            if 60 <= hr <= 100:
                score += 1.0
            elif 50 <= hr <= 120:
                score += 0.7
            elif 40 <= hr <= 150:
                score += 0.3
            else:
                score += 0.0
            count += 1
        
        # Blood pressure scoring
        if 'systolic_bp' in row:
            sbp = row['systolic_bp']
            if 90 <= sbp <= 140:
                score += 1.0
            elif 80 <= sbp <= 160:
                score += 0.7
            elif 70 <= sbp <= 180:
                score += 0.3
            else:
                score += 0.0
            count += 1
        
        # Oxygen saturation scoring
        if 'oxygen_saturation' in row:
            spo2 = row['oxygen_saturation']
            if spo2 >= 95:
                score += 1.0
            elif spo2 >= 90:
                score += 0.7
            elif spo2 >= 85:
                score += 0.3
            else:
                score += 0.0
            count += 1
        
        # Temperature scoring
        if 'temperature' in row:
            temp = row['temperature']
            if 36.1 <= temp <= 37.2:
                score += 1.0
            elif 35.5 <= temp <= 38.0:
                score += 0.7
            elif 35.0 <= temp <= 39.0:
                score += 0.3
            else:
                score += 0.0
            count += 1
        
        return score / count if count > 0 else 0.5
    
    def create_correlation_matrix(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation matrix for vital signs"""
        
        vital_columns = [
            'heart_rate', 'systolic_bp', 'diastolic_bp',
            'oxygen_saturation', 'temperature', 'respiratory_rate'
        ]
        
        # Filter available columns
        available_columns = [col for col in vital_columns if col in data.columns]
        
        if len(available_columns) < 2:
            # Return empty figure if insufficient data
            return go.Figure().add_annotation(
                text="Insufficient data for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Calculate correlation matrix
        corr_matrix = data[available_columns].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Vital Signs Correlation Matrix",
            xaxis_title="Vital Signs",
            yaxis_title="Vital Signs",
            height=500,
            width=500
        )
        
        return fig
    
    def create_anomaly_scatter(self, data: pd.DataFrame, 
                             x_col: str = 'heart_rate', 
                             y_col: str = 'systolic_bp') -> go.Figure:
        """Create scatter plot highlighting anomalies"""
        
        if x_col not in data.columns or y_col not in data.columns:
            return go.Figure().add_annotation(
                text=f"Required columns {x_col} or {y_col} not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Determine anomalies
        anomaly_col = 'anomaly_score' if 'anomaly_score' in data.columns else None
        
        if anomaly_col:
            normal_data = data[data[anomaly_col] <= 0.5]
            anomaly_data = data[data[anomaly_col] > 0.5]
        else:
            normal_data = data
            anomaly_data = pd.DataFrame()
        
        fig = go.Figure()
        
        # Normal data points
        if not normal_data.empty:
            fig.add_trace(go.Scatter(
                x=normal_data[x_col],
                y=normal_data[y_col],
                mode='markers',
                name='Normal',
                marker=dict(
                    color=self.colors['normal'],
                    size=8,
                    opacity=0.6
                ),
                hovertemplate=f'<b>Normal</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>'
            ))
        
        # Anomaly data points
        if not anomaly_data.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_data[x_col],
                y=anomaly_data[y_col],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    color=self.colors['critical'],
                    size=12,
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                hovertemplate=f'<b>Anomaly</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>'
            ))
        
        # Add normal range rectangles
        if x_col in self.normal_ranges and y_col in self.normal_ranges:
            x_range = self.normal_ranges[x_col]
            y_range = self.normal_ranges[y_col]
            
            fig.add_shape(
                type="rect",
                x0=x_range[0], y0=y_range[0],
                x1=x_range[1], y1=y_range[1],
                fillcolor="rgba(76, 175, 80, 0.1)",
                line=dict(color="green", width=2),
                layer="below"
            )
        
        fig.update_layout(
            title=f"Anomaly Detection: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_trend_prediction_chart(self, historical_data: pd.DataFrame,
                                    predicted_data: pd.DataFrame,
                                    vital_sign: str = 'heart_rate') -> go.Figure:
        """Create chart showing historical data with future predictions"""
        
        fig = go.Figure()
        
        # Historical data
        if vital_sign in historical_data.columns:
            timestamps = pd.to_datetime(historical_data['timestamp']) if 'timestamp' in historical_data.columns else historical_data.index
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=historical_data[vital_sign],
                mode='lines+markers',
                name='Historical',
                line=dict(color=self.colors['normal'], width=2),
                marker=dict(size=4)
            ))
        
        # Predicted data
        if vital_sign in predicted_data.columns:
            pred_timestamps = pd.to_datetime(predicted_data['timestamp']) if 'timestamp' in predicted_data.columns else predicted_data.index
            
            fig.add_trace(go.Scatter(
                x=pred_timestamps,
                y=predicted_data[vital_sign],
                mode='lines+markers',
                name='Predicted',
                line=dict(color=self.colors['warning'], width=2, dash='dash'),
                marker=dict(size=4, symbol='diamond')
            ))
            
            # Add confidence interval if available
            if 'lower_bound' in predicted_data.columns and 'upper_bound' in predicted_data.columns:
                fig.add_trace(go.Scatter(
                    x=pred_timestamps,
                    y=predicted_data['upper_bound'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=pred_timestamps,
                    y=predicted_data['lower_bound'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 136, 0, 0.2)',
                    name='Confidence Interval',
                    hoverinfo='skip'
                ))
        
        # Add normal range
        if vital_sign in self.normal_ranges:
            normal_range = self.normal_ranges[vital_sign]
            
            fig.add_hrect(
                y0=normal_range[0], y1=normal_range[1],
                fillcolor="rgba(76, 175, 80, 0.1)",
                layer="below", line_width=0,
                annotation_text="Normal Range"
            )
        
        fig.update_layout(
            title=f"Trend Prediction: {vital_sign.replace('_', ' ').title()}",
            xaxis_title="Time",
            yaxis_title=vital_sign.replace('_', ' ').title(),
            height=400,
            showlegend=True
        )
        
        return fig

def demo_plotly_charts():
    """Demonstration of medical chart capabilities"""
    print("Medical Charts Demo")
    print("=" * 30)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01 10:00:00', periods=100, freq='5min')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'heart_rate': np.random.normal(75, 10, 100),
        'systolic_bp': np.random.normal(120, 15, 100),
        'diastolic_bp': np.random.normal(80, 10, 100),
        'oxygen_saturation': np.random.normal(97, 2, 100),
        'temperature': np.random.normal(36.5, 0.5, 100),
        'respiratory_rate': np.random.normal(16, 3, 100),
        'glucose_level': np.random.normal(90, 20, 100)
    })
    
    # Add some anomalies
    sample_data.loc[80:85, 'heart_rate'] += 40
    sample_data['anomaly_score'] = 0.0
    sample_data.loc[80:85, 'anomaly_score'] = 0.8
    
    # Initialize chart generator
    chart_gen = MedicalChartGenerator()
    
    print(f"Generated sample data with {len(sample_data)} measurements")
    
    # Create different types of charts
    print("\\n1. Creating vital signs timeline...")
    timeline_fig = chart_gen.create_vital_signs_timeline(sample_data, "John Doe")
    print(f"   Timeline chart created with {len(timeline_fig.data)} traces")
    
    print("\\n2. Creating real-time gauge...")
    gauge_fig = chart_gen.create_real_time_gauge(
        "Heart Rate", 
        sample_data['heart_rate'].iloc[-1], 
        (60, 100), 
        (40, 150)
    )
    print("   Gauge chart created")
    
    print("\\n3. Creating correlation matrix...")
    corr_fig = chart_gen.create_correlation_matrix(sample_data)
    print("   Correlation matrix created")
    
    print("\\n4. Creating anomaly scatter plot...")
    scatter_fig = chart_gen.create_anomaly_scatter(
        sample_data, 
        'heart_rate', 
        'systolic_bp'
    )
    print("   Anomaly scatter plot created")
    
    print("\\nAll medical charts generated successfully!")
    print("Charts are ready for display in Streamlit or Jupyter notebooks")

if __name__ == "__main__":
    demo_plotly_charts()
