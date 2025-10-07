"""
Real-time Data Streaming Engine for Virtual Patient Monitor
Manages continuous data flow and real-time updates
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import pandas as pd
import logging
from dataclasses import asdict
import queue
import sys
import os

# Add data simulation path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'simulation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from patient_data_generator import SyntheticDataGenerator, VitalSigns
except ImportError:
    try:
        # Fallback for when running from different directories
        from data.simulation.patient_data_generator import SyntheticDataGenerator, VitalSigns
    except ImportError:
        # If still can't import, define minimal classes for testing
        class VitalSigns:
            pass
        class SyntheticDataGenerator:
            pass

logger = logging.getLogger(__name__)

class DataStreamer:
    """Manages real-time streaming of patient vital signs"""
    
    def __init__(self, update_interval: int = 5):
        """
        Initialize data streamer
        
        Args:
            update_interval: Seconds between updates
        """
        self.update_interval = update_interval
        self.generator = SyntheticDataGenerator()
        self.subscribers = []
        self.data_queue = queue.Queue()
        self.is_streaming = False
        self.stream_thread = None
        
        # Data storage for recent measurements
        self.recent_data: Dict[str, List[VitalSigns]] = {}
        self.max_history = 1000  # Keep last 1000 measurements per patient
        
    def add_patient(self, patient_profile):
        """Add a patient to the streaming system"""
        self.generator.add_patient(patient_profile)
        self.recent_data[patient_profile.patient_id] = []
        logger.info(f"Added patient {patient_profile.patient_id} to streaming system")
    
    def subscribe(self, callback: Callable[[VitalSigns], None]):
        """Subscribe to real-time data updates"""
        self.subscribers.append(callback)
        logger.info(f"Added subscriber: {callback.__name__}")
    
    def unsubscribe(self, callback: Callable[[VitalSigns], None]):
        """Unsubscribe from data updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Removed subscriber: {callback.__name__}")
    
    def _notify_subscribers(self, vital_signs: VitalSigns):
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                callback(vital_signs)
            except Exception as e:
                logger.error(f"Error notifying subscriber {callback.__name__}: {e}")
    
    def _stream_worker(self):
        """Background worker for continuous data generation"""
        while self.is_streaming:
            try:
                current_time = datetime.now()
                
                # Generate data for all patients
                for patient_id in self.generator.patients.keys():
                    vital_signs = self.generator.generate_vital_signs(patient_id, current_time)
                    
                    # Store in recent data
                    if patient_id not in self.recent_data:
                        self.recent_data[patient_id] = []
                    
                    self.recent_data[patient_id].append(vital_signs)
                    
                    # Maintain maximum history size
                    if len(self.recent_data[patient_id]) > self.max_history:
                        self.recent_data[patient_id] = self.recent_data[patient_id][-self.max_history:]
                    
                    # Add to queue for processing
                    self.data_queue.put(vital_signs)
                    
                    # Notify subscribers
                    self._notify_subscribers(vital_signs)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in stream worker: {e}")
                time.sleep(1)
    
    def start_streaming(self):
        """Start the real-time data stream"""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        logger.info("Started real-time data streaming")
    
    def stop_streaming(self):
        """Stop the real-time data stream"""
        if not self.is_streaming:
            logger.warning("Streaming not active")
            return
        
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        logger.info("Stopped real-time data streaming")
    
    def get_latest_data(self, patient_id: str, count: int = 1) -> List[VitalSigns]:
        """Get the most recent vital signs for a patient"""
        if patient_id not in self.recent_data:
            return []
        
        return self.recent_data[patient_id][-count:] if count > 0 else self.recent_data[patient_id]
    
    def get_time_range_data(self, patient_id: str, start_time: datetime, 
                           end_time: datetime) -> List[VitalSigns]:
        """Get vital signs within a specific time range"""
        if patient_id not in self.recent_data:
            return []
        
        filtered_data = [
            vs for vs in self.recent_data[patient_id]
            if start_time <= vs.timestamp <= end_time
        ]
        
        return filtered_data
    
    def get_data_summary(self, patient_id: str, minutes: int = 60) -> Dict:
        """Get statistical summary of recent data"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_vitals = [
            vs for vs in self.recent_data.get(patient_id, [])
            if vs.timestamp >= cutoff_time
        ]
        
        if not recent_vitals:
            return {}
        
        # Convert to DataFrame for easy statistics
        df = pd.DataFrame([asdict(vs) for vs in recent_vitals])
        
        numeric_columns = ['heart_rate', 'systolic_bp', 'diastolic_bp', 
                          'oxygen_saturation', 'respiratory_rate', 'temperature', 
                          'glucose_level']
        
        summary = {}
        for col in numeric_columns:
            if col in df.columns:
                summary[col] = {
                    'mean': df[col].mean(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'std': df[col].std(),
                    'latest': df[col].iloc[-1] if len(df) > 0 else None
                }
        
        summary['data_points'] = len(recent_vitals)
        summary['time_range'] = f"Last {minutes} minutes"
        
        return summary

class MQTTStreamer:
    """MQTT-based streaming for IoT integration"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client = None
        self.connected = False
        
    def connect(self):
        """Connect to MQTT broker"""
        try:
            import paho.mqtt.client as mqtt
            
            self.client = mqtt.Client()
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
        except ImportError:
            logger.error("paho-mqtt not installed. Install with: pip install paho-mqtt")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        else:
            logger.error(f"Failed to connect to MQTT broker, return code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        self.connected = False
        logger.info("Disconnected from MQTT broker")
    
    def publish_vital_signs(self, vital_signs: VitalSigns):
        """Publish vital signs to MQTT topic"""
        if not self.connected or not self.client:
            logger.warning("MQTT client not connected")
            return
        
        topic = f"patients/{vital_signs.patient_id}/vitals"
        payload = json.dumps(asdict(vital_signs), default=str)
        
        try:
            self.client.publish(topic, payload)
            logger.debug(f"Published vital signs for {vital_signs.patient_id}")
        except Exception as e:
            logger.error(f"Failed to publish to MQTT: {e}")

class WebSocketStreamer:
    """WebSocket streaming for real-time web applications"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.connected_clients = set()
        self.server = None
        
    async def register_client(self, websocket, path):
        """Register a new WebSocket client"""
        self.connected_clients.add(websocket)
        logger.info(f"WebSocket client connected from {websocket.remote_address}")
        
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected")
    
    async def broadcast_vital_signs(self, vital_signs: VitalSigns):
        """Broadcast vital signs to all connected clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps(asdict(vital_signs), default=str)
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    def start_server(self):
        """Start the WebSocket server"""
        try:
            import websockets
            
            start_server = websockets.serve(self.register_client, "localhost", self.port)
            logger.info(f"WebSocket server starting on port {self.port}")
            
            return start_server
            
        except ImportError:
            logger.error("websockets not installed. Install with: pip install websockets")
            return None

class DataBuffer:
    """Circular buffer for efficient data storage"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.data = []
        self.index = 0
        
    def add(self, item):
        """Add item to buffer"""
        if len(self.data) < self.max_size:
            self.data.append(item)
        else:
            self.data[self.index] = item
            self.index = (self.index + 1) % self.max_size
    
    def get_recent(self, count: int) -> List:
        """Get the most recent items"""
        if count >= len(self.data):
            return self.data.copy()
        
        if len(self.data) < self.max_size:
            return self.data[-count:]
        else:
            # Handle circular buffer
            if self.index >= count:
                return self.data[self.index - count:self.index]
            else:
                return self.data[self.max_size - (count - self.index):] + self.data[:self.index]

def demo_streaming():
    """Demonstration of real-time streaming"""
    # Create streamer
    streamer = DataStreamer(update_interval=2)  # Update every 2 seconds
    
    # Create and add patients
    generator = SyntheticDataGenerator()
    for i in range(2):
        profile = generator.create_patient_profile()
        streamer.add_patient(profile)
    
    # Subscribe to updates
    def print_update(vital_signs: VitalSigns):
        print(f"[{vital_signs.timestamp.strftime('%H:%M:%S')}] "
              f"Patient {vital_signs.patient_id}: "
              f"HR={vital_signs.heart_rate:.1f}, "
              f"BP={vital_signs.systolic_bp:.0f}/{vital_signs.diastolic_bp:.0f}, "
              f"SpO2={vital_signs.oxygen_saturation:.1f}%")
    
    streamer.subscribe(print_update)
    
    # Start streaming
    print("Starting real-time data streaming...")
    streamer.start_streaming()
    
    try:
        # Let it run for 30 seconds
        time.sleep(30)
    except KeyboardInterrupt:
        print("\\nStopping...")
    finally:
        streamer.stop_streaming()
        
        # Show summary
        for patient_id in streamer.generator.patients.keys():
            summary = streamer.get_data_summary(patient_id, minutes=5)
            print(f"\\nSummary for {patient_id}:")
            print(f"  Data points: {summary.get('data_points', 0)}")
            if 'heart_rate' in summary:
                hr_stats = summary['heart_rate']
                print(f"  Heart Rate: {hr_stats['mean']:.1f} ± {hr_stats['std']:.1f} bpm")

if __name__ == "__main__":
    demo_streaming()
