"""
NetraGuard Core Engine
Author: Supriya Mishra
Description:
    Central anomaly detection pipeline for the NetraGuard AI surveillance system.
    Handles video ingestion, object detection, behavioral analysis, and alert triggering.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import json
import threading
import queue
import time
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import math
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for storing alerts and system data"""
    
    def __init__(self, db_path: str = "surveillance_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox TEXT,
                frame_path TEXT,
                description TEXT,
                severity INTEGER DEFAULT 1
            )
        ''')
        
        # Create system_stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                fps REAL,
                detection_count INTEGER,
                alert_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_alert(self, alert_data: Dict):
        """Save an alert to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, anomaly_type, confidence, bbox, frame_path, description, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert_data['timestamp'],
            alert_data['anomaly_type'],
            alert_data['confidence'],
            json.dumps(alert_data['bbox']),
            alert_data.get('frame_path', ''),
            alert_data.get('description', ''),
            alert_data.get('severity', 1)
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Retrieve recent alerts from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM alerts 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in rows:
            alert = {
                'id': row[0],
                'timestamp': row[1],
                'anomaly_type': row[2],
                'confidence': row[3],
                'bbox': json.loads(row[4]) if row[4] else None,
                'frame_path': row[5],
                'description': row[6],
                'severity': row[7]
            }
            alerts.append(alert)
        
        return alerts

class PersonTracker:
    """Tracks persons across frames for behavioral analysis"""
    
    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_id] = {
            'centroid': centroid,
            'positions': deque([centroid], maxlen=50),
            'timestamps': deque([time.time()], maxlen=50),
            'first_seen': time.time()
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Initialize arrays for new detections
        if len(self.objects) == 0:
            for detection in detections:
                centroid = self.get_centroid(detection)
                self.register(centroid)
        else:
            # Match existing objects to new detections
            object_centroids = np.array([obj['centroid'] for obj in self.objects.values()])
            object_ids = list(self.objects.keys())
            
            detection_centroids = np.array([self.get_centroid(det) for det in detections])
            
            # Compute distances between existing objects and new detections
            D = np.linalg.norm(object_centroids[:, np.newaxis] - detection_centroids, axis=2)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] <= 100:  # Distance threshold
                    object_id = object_ids[row]
                    centroid = detection_centroids[col]
                    
                    # Update object
                    self.objects[object_id]['centroid'] = centroid
                    self.objects[object_id]['positions'].append(centroid)
                    self.objects[object_id]['timestamps'].append(time.time())
                    self.disappeared[object_id] = 0
                    
                    used_row_indices.add(row)
                    used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    centroid = detection_centroids[col]
                    self.register(centroid)
        
        return self.objects
    
    @staticmethod
    def get_centroid(detection):
        """Calculate centroid from detection bounding box"""
        x1, y1, x2, y2 = detection
        return np.array([(x1 + x2) // 2, (y1 + y2) // 2])

class AnomalyDetector:
    """Main anomaly detection engine"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_buffer = []
        self.training_features = []
        
    def extract_features(self, tracked_objects: Dict, frame_shape: Tuple) -> np.ndarray:
        """Extract features from tracked objects for anomaly detection"""
        features = []
        
        for obj_id, obj_data in tracked_objects.items():
            current_time = time.time()
            positions = list(obj_data['positions'])
            timestamps = list(obj_data['timestamps'])
            
            if len(positions) < 2:
                continue
                
            # Basic position features
            current_pos = positions[-1]
            x_norm = current_pos[0] / frame_shape[1]
            y_norm = current_pos[1] / frame_shape[0]
            
            # Motion features
            velocity = self.calculate_velocity(positions, timestamps)
            acceleration = self.calculate_acceleration(positions, timestamps)
            direction_change = self.calculate_direction_change(positions)
            
            # Temporal features
            dwell_time = current_time - obj_data['first_seen']
            recent_movement = self.calculate_recent_movement(positions[-10:])
            
            # Spatial features
            distance_from_center = np.linalg.norm(current_pos - np.array([frame_shape[1]/2, frame_shape[0]/2]))
            distance_from_center_norm = distance_from_center / (frame_shape[1] + frame_shape[0])
            
            # Compile feature vector
            feature_vector = [
                x_norm, y_norm,
                velocity, acceleration,
                direction_change,
                dwell_time,
                recent_movement,
                distance_from_center_norm
            ]
            
            features.append({
                'object_id': obj_id,
                'features': feature_vector,
                'position': current_pos,
                'dwell_time': dwell_time
            })
        
        return features
    
    @staticmethod
    def calculate_velocity(positions, timestamps):
        """Calculate velocity from position history"""
        if len(positions) < 2:
            return 0.0
        
        recent_pos = positions[-5:]
        recent_time = timestamps[-5:]
        
        if len(recent_pos) < 2:
            return 0.0
        
        distances = []
        time_diffs = []
        
        for i in range(1, len(recent_pos)):
            dist = np.linalg.norm(np.array(recent_pos[i]) - np.array(recent_pos[i-1]))
            time_diff = recent_time[i] - recent_time[i-1]
            if time_diff > 0:
                distances.append(dist)
                time_diffs.append(time_diff)
        
        if not distances:
            return 0.0
        
        return np.mean([d/t for d, t in zip(distances, time_diffs)])
    
    @staticmethod
    def calculate_acceleration(positions, timestamps):
        """Calculate acceleration from position history"""
        if len(positions) < 3:
            return 0.0
        
        velocities = []
        for i in range(1, len(positions)):
            if i < len(timestamps):
                time_diff = timestamps[i] - timestamps[i-1]
                if time_diff > 0:
                    dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
                    velocities.append(dist / time_diff)
        
        if len(velocities) < 2:
            return 0.0
        
        # Calculate acceleration as change in velocity
        acc = []
        for i in range(1, len(velocities)):
            acc.append(abs(velocities[i] - velocities[i-1]))
        
        return np.mean(acc) if acc else 0.0
    
    @staticmethod
    def calculate_direction_change(positions):
        """Calculate direction change metric"""
        if len(positions) < 3:
            return 0.0
        
        angles = []
        for i in range(2, len(positions)):
            v1 = np.array(positions[i-1]) - np.array(positions[i-2])
            v2 = np.array(positions[i]) - np.array(positions[i-1])
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        return np.mean(angles) if angles else 0.0
    
    @staticmethod
    def calculate_recent_movement(positions):
        """Calculate recent movement metric"""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0
        for i in range(1, len(positions)):
            total_distance += np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
        
        return total_distance
    
    def train_model(self, features_list: List):
        """Train the anomaly detection model"""
        if len(features_list) < 10:
            return False
        
        feature_vectors = [f['features'] for f in features_list]
        X = np.array(feature_vectors)
        
        # Handle NaN and infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.isolation_forest.fit(X_scaled)
            self.is_trained = True
            logger.info(f"Model trained with {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def detect_anomalies(self, features_list: List) -> List[Dict]:
        """Detect anomalies in the given features"""
        if not self.is_trained or not features_list:
            return []
        
        anomalies = []
        feature_vectors = [f['features'] for f in features_list]
        X = np.array(feature_vectors)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        try:
            X_scaled = self.scaler.transform(X)
            anomaly_scores = self.isolation_forest.decision_function(X_scaled)
            predictions = self.isolation_forest.predict(X_scaled)
            
            for i, (pred, score, feature_data) in enumerate(zip(predictions, anomaly_scores, features_list)):
                if pred == -1:  # Anomaly detected
                    anomaly_type = self.classify_anomaly(feature_data)
                    anomaly = {
                        'object_id': feature_data['object_id'],
                        'anomaly_type': anomaly_type,
                        'confidence': abs(score),
                        'position': feature_data['position'],
                        'features': feature_data['features']
                    }
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def classify_anomaly(self, feature_data: Dict) -> str:
        """Classify the type of anomaly based on features"""
        features = feature_data['features']
        dwell_time = feature_data['dwell_time']
        
        # Extract relevant features
        velocity = features[2] if len(features) > 2 else 0
        recent_movement = features[6] if len(features) > 6 else 0
        
        # Classification logic
        if dwell_time > 30 and velocity < 1.0 and recent_movement < 10:
            return "loitering"
        elif velocity > 50:
            return "unusual_movement"
        elif recent_movement > 100:
            return "erratic_behavior"
        else:
            return "suspicious_activity"

class SurveillanceSystem:
    """Main surveillance system coordinator"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        # Initialize components
        self.yolo_model = YOLO(model_path)
        self.tracker = PersonTracker()
        self.anomaly_detector = AnomalyDetector()
        self.db_manager = DatabaseManager()
        
        # System state
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=100)
        self.alert_queue = queue.Queue()
        self.stats = {
            'fps': 0,
            'detection_count': 0,
            'alert_count': 0,
            'processed_frames': 0
        }
        
        # Training data collection
        self.training_mode = True
        self.training_frames = 0
        self.max_training_frames = 1000
        
        logger.info("Surveillance system initialized")
    
    def detect_objects(self, frame: np.ndarray) -> List[Tuple]:
        """Detect persons in the frame using YOLO"""
        try:
            results = self.yolo_model(frame, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Only detect persons (class 0 in COCO dataset)
                        if box.cls == 0 and box.conf > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append((int(x1), int(y1), int(x2), int(y2)))
            
            return detections
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame and return annotated frame with alerts"""
        # Detect objects
        detections = self.detect_objects(frame)
        self.stats['detection_count'] += len(detections)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Extract features for anomaly detection
        features_list = self.anomaly_detector.extract_features(tracked_objects, frame.shape[:2])
        
        # Training phase
        if self.training_mode and self.training_frames < self.max_training_frames:
            self.anomaly_detector.training_features.extend(features_list)
            self.training_frames += len(features_list)
            
            if self.training_frames >= self.max_training_frames:
                logger.info("Starting model training...")
                if self.anomaly_detector.train_model(self.anomaly_detector.training_features):
                    self.training_mode = False
                    logger.info("Training completed. Anomaly detection active.")
        
        # Detect anomalies
        anomalies = []
        if not self.training_mode:
            anomalies = self.anomaly_detector.detect_anomalies(features_list)
        
        # Process alerts
        alerts = []
        for anomaly in anomalies:
            alert = self.create_alert(anomaly, frame.shape[:2])
            alerts.append(alert)
            self.alert_queue.put(alert)
            self.stats['alert_count'] += 1
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame, tracked_objects, anomalies)
        
        return annotated_frame, alerts
    
    def create_alert(self, anomaly: Dict, frame_shape: Tuple) -> Dict:
        """Create an alert from detected anomaly"""
        position = anomaly['position']
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'anomaly_type': anomaly['anomaly_type'],
            'confidence': float(anomaly['confidence']),
            'bbox': [
                max(0, int(position[0] - 50)),
                max(0, int(position[1] - 50)),
                min(frame_shape[1], int(position[0] + 50)),
                min(frame_shape[0], int(position[1] + 50))
            ],
            'description': f"{anomaly['anomaly_type']} detected with confidence {anomaly['confidence']:.2f}",
            'severity': self.get_severity(anomaly['anomaly_type'])
        }
        
        # Save to database
        self.db_manager.save_alert(alert)
        
        return alert
    
    @staticmethod
    def get_severity(anomaly_type: str) -> int:
        """Get severity level for anomaly type"""
        severity_map = {
            'loitering': 2,
            'unusual_movement': 3,
            'erratic_behavior': 2,
            'suspicious_activity': 1
        }
        return severity_map.get(anomaly_type, 1)
    
    def annotate_frame(self, frame: np.ndarray, tracked_objects: Dict, anomalies: List[Dict]) -> np.ndarray:
        """Annotate frame with detections and alerts"""
        annotated = frame.copy()
        
        # Draw tracked objects
        for obj_id, obj_data in tracked_objects.items():
            pos = obj_data['centroid']
            cv2.circle(annotated, tuple(pos.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(annotated, f"ID:{obj_id}", 
                       (pos[0] - 20, pos[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw trajectory
            positions = list(obj_data['positions'])
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    cv2.line(annotated, tuple(positions[i-1].astype(int)), 
                           tuple(positions[i].astype(int)), (0, 255, 0), 1)
        
        # Draw anomalies
        for anomaly in anomalies:
            pos = anomaly['position']
            cv2.circle(annotated, tuple(pos.astype(int)), 10, (0, 0, 255), 2)
            cv2.putText(annotated, f"ALERT: {anomaly['anomaly_type']}", 
                       (pos[0] - 50, pos[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add system info
        info_text = f"Training: {'ON' if self.training_mode else 'OFF'} | "
        info_text += f"Objects: {len(tracked_objects)} | Alerts: {len(anomalies)}"
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated
    
    def process_video(self, video_path: str, output_path: str = None):
        """Process video file for anomaly detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        logger.info(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, alerts = self.process_frame(frame)
            
            # Update stats
            frame_count += 1
            self.stats['processed_frames'] = frame_count
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                self.stats['fps'] = frame_count / elapsed_time
            
            # Write output frame
            if out:
                out.write(annotated_frame)
            
            # Log alerts
            for alert in alerts:
                logger.info(f"Alert: {alert['anomaly_type']} at {alert['timestamp']}")
            
            # Display progress
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames, FPS: {self.stats['fps']:.2f}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        logger.info(f"Video processing completed. Total frames: {frame_count}")
        logger.info(f"Total detections: {self.stats['detection_count']}")
        logger.info(f"Total alerts: {self.stats['alert_count']}")
    
    def process_realtime(self, camera_id: int = 0):
        """Process real-time camera feed"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera: {camera_id}")
            return
        
        logger.info("Starting real-time processing. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, alerts = self.process_frame(frame)
            
            # Display frame
            cv2.imshow('AI Surveillance System', annotated_frame)
            
            # Log alerts
            for alert in alerts:
                logger.info(f"REAL-TIME Alert: {alert['anomaly_type']} - {alert['description']}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        return self.stats.copy()
    
    def save_model(self, model_path: str):
        """Save the trained anomaly detection model"""
        if self.anomaly_detector.is_trained:
            model_data = {
                'isolation_forest': self.anomaly_detector.isolation_forest,
                'scaler': self.anomaly_detector.scaler
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.warning("No trained model to save")
    
    def load_model(self, model_path: str):
        """Load a pre-trained anomaly detection model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.anomaly_detector.isolation_forest = model_data['isolation_forest']
            self.anomaly_detector.scaler = model_data['scaler']
            self.anomaly_detector.is_trained = True
            self.training_mode = False
            
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

# Example usage and testing functions
def test_avenue_dataset(dataset_path: str):
    """Test the system with Avenue dataset"""
    system = SurveillanceSystem()
    
    # Process videos from Avenue dataset
    video_files = list(Path(dataset_path).glob("*.avi"))
    
    for video_file in video_files[:3]:  # Process first 3 videos for testing
        logger.info(f"Processing: {video_file}")
        output_path = f"output_{video_file.stem}.avi"
        system.process_video(str(video_file), output_path)
        
        # Save model after processing
        system.save_model(f"model_{video_file.stem}.pkl")

def demo_realtime():
    """Demo real-time surveillance"""
    system = SurveillanceSystem()
    system.process_realtime(0)  # Use default camera

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-Powered Surveillance System')
    parser.add_argument('--mode', choices=['video', 'realtime', 'test'], 
                       default='test', help='Operation mode')
    parser.add_argument('--input', type=str, help='Input video path or dataset path')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID for realtime mode')
    
    args = parser.parse_args()
    
    if args.mode == 'video':
        if args.input:
            system = SurveillanceSystem()
            system.process_video(args.input, args.output)
        else:
            print("Please provide input video path with --input")
    
    elif args.mode == 'realtime':
        system = SurveillanceSystem()
        system.process_realtime(args.camera)
    
    elif args.mode == 'test':
        if args.input:
            test_avenue_dataset(args.input)
        else:
            print("Please provide Avenue dataset path with --input")
    
    else:
        print("Invalid mode selected")