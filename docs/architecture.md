# System Architecture - Virtual Patient Monitor

## 🏗️ Overall Architecture

The Virtual Patient Monitor follows a **layered, event-driven architecture** designed for real-time healthcare monitoring and predictive analytics.

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard  │  Web APIs  │  Mobile Interface      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Alert & Notification Layer                │
├─────────────────────────────────────────────────────────────┤
│   Clinical Alerts   │   Predictive Warnings  │  Escalation  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Analytics & ML Layer                      │
├─────────────────────────────────────────────────────────────┤
│  Anomaly Detection  │  Health Prediction  │  Trend Analysis │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Digital Twin Engine                       │
├─────────────────────────────────────────────────────────────┤
│   Patient State Management   │   Physiological Modeling     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                     │
├─────────────────────────────────────────────────────────────┤
│  Stream Processing  │  Data Validation  │  Feature Extract  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources Layer                       │
├─────────────────────────────────────────────────────────────┤
│  IoT Sensors  │  Hospital Systems  │  Synthetic Data Gen   │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow Architecture

### 1. Data Ingestion Pipeline

```python
# Data Flow: Sensors → Streaming → Processing → Digital Twin → Analytics → UI

IoT Devices/Sensors 
    ↓ (MQTT/HTTP)
Data Streamer (src/core/data_streamer.py)
    ↓ (Real-time Queue)
Data Validation & Cleaning
    ↓ (Structured Data)
Digital Twin Engine (src/core/digital_twin.py)
    ↓ (State Updates)
ML Analytics (src/ml/anomaly_detector.py)
    ↓ (Insights & Predictions)
Alert System & Dashboard (src/visualization/)
```

### 2. Real-time Synchronization

- **Frequency**: Every 5 seconds for vital signs
- **Latency Target**: < 2 seconds end-to-end
- **Data Validation**: Real-time quality checks
- **Buffering**: Circular buffers for efficiency

### 3. Scalability Considerations

```yaml
Horizontal Scaling:
  - Microservices architecture
  - Container-based deployment
  - Load balancing for multiple patients
  
Vertical Scaling:
  - Optimized data structures
  - Efficient algorithms
  - Memory management
  
Data Storage:
  - Time-series database (InfluxDB/TimescaleDB)
  - Redis for real-time caching
  - PostgreSQL for patient records
```

## 🧠 Digital Twin Core Engine

### Patient State Management

The digital twin maintains a comprehensive virtual representation:

```python
@dataclass
class DigitalTwinState:
    # Real-time vitals
    timestamp: datetime
    heart_rate: float
    blood_pressure: Tuple[float, float]
    oxygen_saturation: float
    temperature: float
    
    # Derived metrics
    pulse_pressure: float
    cardiac_output: float
    
    # Health indicators
    health_status: HealthStatus
    anomaly_score: float
    trend_direction: str
    
    # Predictions
    predicted_status_6h: str
    prediction_confidence: float
```

### Physiological Modeling

Mathematical models based on established medical equations:

1. **Cardiovascular Model**:
   - Blood Pressure = Cardiac Output × Total Peripheral Resistance
   - Cardiac Output = Heart Rate × Stroke Volume
   - Mean Arterial Pressure = DBP + 1/3(SBP - DBP)

2. **Respiratory Model**:
   - Oxygen Saturation = f(O₂ delivery, O₂ consumption, lung function)
   - Respiratory Rate adjustments based on metabolic demand

3. **Metabolic Model**:
   - Temperature regulation based on metabolic rate
   - Glucose dynamics with meal timing and medication effects

## 🤖 Machine Learning Pipeline

### 1. Anomaly Detection System

```python
# Multi-algorithm approach for robust detection
algorithms = [
    IsolationForest(contamination=0.1),
    LocalOutlierFactor(n_neighbors=20),
    OneClassSVM(nu=0.1)
]

# Feature engineering
features = [
    'heart_rate', 'bp_systolic', 'bp_diastolic',
    'oxygen_saturation', 'temperature',
    'pulse_pressure', 'mean_arterial_pressure',
    'heart_rate_variability', 'trend_indicators'
]
```

### 2. Predictive Analytics

- **Time Series Forecasting**: LSTM networks for vital sign prediction
- **Health Deterioration Risk**: Random Forest classifier
- **Medication Response**: Regression models for dosage optimization
- **Recovery Prediction**: Survival analysis techniques

### 3. Model Training Pipeline

```yaml
Training Data Sources:
  - Historical patient data (anonymized)
  - Synthetic data generation
  - Public medical datasets
  - Simulated scenarios

Model Validation:
  - Cross-validation with temporal splits
  - Clinical validation with domain experts
  - A/B testing in controlled environments
  - Continuous monitoring of model performance
```

## 📊 Real-time Dashboard Architecture

### Frontend Architecture (Streamlit)

```python
# Component-based dashboard design
components = [
    PatientOverview(),      # High-level status
    VitalSignsCharts(),     # Real-time plotting
    AlertsPanel(),          # Active alerts
    PredictiveInsights(),   # ML-driven insights
    TrendAnalysis(),        # Historical patterns
    ClinicalRecommendations()  # Actionable advice
]
```

### Visualization Strategy

1. **Real-time Updates**: WebSocket connections for live data
2. **Interactive Charts**: Plotly for responsive visualization
3. **Color Coding**: Clinical standards (Red/Yellow/Green)
4. **Responsive Design**: Multi-device compatibility

### Performance Optimization

- **Data Sampling**: Intelligent downsampling for visualization
- **Caching**: Redis for frequently accessed data
- **Lazy Loading**: Progressive data loading
- **Compression**: Data compression for network efficiency

## 🔔 Alert and Notification System

### Alert Classification

```python
class AlertLevel(Enum):
    CRITICAL = 1    # Immediate intervention required
    HIGH = 2        # Attention needed within 15 minutes
    MEDIUM = 3      # Monitor closely
    LOW = 4         # Informational
```

### Alert Processing Pipeline

1. **Detection**: Real-time threshold monitoring
2. **Validation**: Multi-factor confirmation
3. **Prioritization**: Clinical risk scoring
4. **Escalation**: Automatic escalation rules
5. **Acknowledgment**: Healthcare provider response tracking

### Notification Channels

- **Dashboard**: Real-time visual alerts
- **Email**: Detailed alert summaries
- **SMS**: Critical alerts for immediate response
- **Mobile Push**: Mobile app notifications
- **Pager**: Legacy hospital systems integration

## 🔒 Security and Privacy Architecture

### Data Protection

```yaml
Encryption:
  - TLS 1.3 for data in transit
  - AES-256 for data at rest
  - Field-level encryption for PII

Access Control:
  - Role-based access control (RBAC)
  - Multi-factor authentication
  - Session management
  - Audit logging

Compliance:
  - HIPAA compliance
  - GDPR compliance
  - SOC 2 Type II
  - FDA guidance for medical software
```

### Privacy Preservation

- **Data Anonymization**: Advanced anonymization techniques
- **Differential Privacy**: Privacy-preserving analytics
- **Federated Learning**: Decentralized model training
- **Secure Enclaves**: Hardware-based data protection

## 🚀 Deployment Architecture

### Container-based Deployment

```dockerfile
# Example microservice structure
services:
  - data-streamer:      # Real-time data ingestion
  - digital-twin-engine: # Core twin logic
  - ml-analytics:       # Machine learning services
  - alert-manager:      # Alert processing
  - dashboard-frontend: # User interface
  - api-gateway:        # External integrations
```

### Cloud Architecture (Multi-cloud)

```yaml
Primary Cloud (AWS):
  - EKS for container orchestration
  - RDS for patient records
  - ElastiCache for real-time data
  - Lambda for serverless functions

Secondary Cloud (Azure):
  - Disaster recovery
  - Data backup
  - Analytics workloads
  - ML model training

Edge Computing:
  - IoT gateways for local processing
  - Reduced latency for critical alerts
  - Offline capability
```

### Monitoring and Observability

- **Application Monitoring**: Prometheus + Grafana
- **Log Management**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Distributed Tracing**: Jaeger for request tracing
- **Health Checks**: Automated system health monitoring
- **Performance Metrics**: Real-time performance dashboards

## 🔄 Integration Points

### Hospital System Integration

```python
# HL7 FHIR standard for healthcare interoperability
integrations = [
    'Electronic Health Records (EHR)',
    'Laboratory Information Systems (LIS)',
    'Picture Archiving Systems (PACS)',
    'Pharmacy Management Systems',
    'Nurse Call Systems',
    'Bed Management Systems'
]
```

### IoT Device Integration

- **Medical Devices**: Standard protocols (IEEE 11073, Continua)
- **Wearable Devices**: Bluetooth LE, WiFi
- **Environmental Sensors**: MQTT, CoAP
- **Mobile Devices**: REST APIs, SDK integration

### Third-party Services

- **Telemedicine Platforms**: Video consultation integration
- **Clinical Decision Support**: Integration with medical knowledge bases
- **Pharmacy Systems**: Medication management
- **Emergency Services**: Automated emergency response

## 📈 Performance and Reliability

### Performance Targets

```yaml
Latency:
  - Data ingestion: < 100ms
  - Alert generation: < 500ms
  - Dashboard updates: < 1s
  - Prediction generation: < 2s

Throughput:
  - 1000+ patients concurrent monitoring
  - 10,000+ measurements per second
  - 100+ simultaneous dashboard users

Availability:
  - 99.9% uptime requirement
  - < 4 hours downtime per year
  - Recovery time objective: < 15 minutes
```

### Reliability Mechanisms

- **Redundancy**: Multi-region deployment
- **Failover**: Automatic failover mechanisms
- **Data Backup**: Real-time data replication
- **Circuit Breakers**: Fault tolerance patterns
- **Health Checks**: Continuous system monitoring

This architecture ensures the Virtual Patient Monitor can scale to support thousands of patients while maintaining real-time performance and clinical-grade reliability.
