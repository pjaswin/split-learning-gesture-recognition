#!/usr/bin/env python3
"""
Split Learning Client with Resource Monitoring
"""

import os
import json
import socket
import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import logging
import psutil
import threading
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor CPU, Memory, and timing"""
    
    def __init__(self, component_name="Unknown"):
        self.component_name = component_name
        self.process = psutil.Process()
        self.operation_stats = {}
        
    def measure_operation(self, operation_name, operation_func, *args, **kwargs):
        """Measure resource usage of a specific operation"""
        # Before operation
        cpu_before = self.process.cpu_percent()
        memory_before = self.process.memory_info().rss / 1024 / 1024
        time_before = time.time()
        
        # Run operation
        result = operation_func(*args, **kwargs)
        
        # After operation
        time_after = time.time()
        cpu_after = self.process.cpu_percent()
        memory_after = self.process.memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        operation_time = time_after - time_before
        memory_delta = memory_after - memory_before
        
        measurement = {
            'operation': operation_name,
            'time_seconds': operation_time,
            'cpu_percent_avg': (cpu_before + cpu_after) / 2,
            'memory_delta_mb': memory_delta,
            'memory_after_mb': memory_after,
            'timestamp': datetime.now().isoformat()
        }
        
        self.operation_stats[operation_name] = measurement
        
        print(f"📊 {operation_name}: {operation_time:.3f}s | {measurement['cpu_percent_avg']:.1f}% CPU | {memory_delta:+.1f}MB")
        
        return result
    
    def print_breakdown(self):
        """Print detailed breakdown of operations"""
        print("\n" + "="*70)
        print("CLIENT RESOURCE BREAKDOWN")
        print("="*70)
        
        total_time = sum(stats['time_seconds'] for stats in self.operation_stats.values())
        
        print(f"{'Operation':<25} | {'Time (s)':<8} | {'% Total':<8} | {'CPU %':<8} | {'Memory Δ':<10}")
        print("-" * 70)
        
        for op_name, stats in self.operation_stats.items():
            percentage = (stats['time_seconds'] / total_time * 100) if total_time > 0 else 0
            print(f"{op_name:<25} | {stats['time_seconds']:<8.3f} | {percentage:<8.1f} | {stats['cpu_percent_avg']:<8.1f} | {stats['memory_delta_mb']:<+10.1f}")
        
        print("-" * 70)
        print(f"{'TOTAL':<25} | {total_time:<8.3f} | {'100.0':<8} | {'':<8} | {'':<10}")
        print()
        
        # Resource usage summary
        mediapipe_time = self.operation_stats.get('MediaPipe_Extraction', {}).get('time_seconds', 0)
        client_inference_time = self.operation_stats.get('Client_Model_Inference', {}).get('time_seconds', 0)
        communication_time = self.operation_stats.get('Server_Communication', {}).get('time_seconds', 0)
        
        print("🎯 KEY INSIGHTS:")
        print(f"   • MediaPipe (edge): {mediapipe_time:.3f}s ({mediapipe_time/total_time*100:.1f}% of total)")
        print(f"   • Client NN (edge): {client_inference_time:.3f}s ({client_inference_time/total_time*100:.1f}% of total)")
        print(f"   • Communication: {communication_time:.3f}s ({communication_time/total_time*100:.1f}% of total)")
        print(f"   • Edge device load: {(mediapipe_time + client_inference_time)/total_time*100:.1f}%")

class HandLandmarkExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands_detector = None
        
    def initialize_detector(self):
        """Initialize MediaPipe hands detector"""
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position and scale"""
        if len(landmarks) != 42:
            return None
            
        # Use wrist (landmark 0) as reference point
        base_x, base_y = landmarks[0], landmarks[1]
        
        # Normalize relative to wrist
        normalized = []
        for i in range(0, len(landmarks), 2):
            normalized.append(landmarks[i] - base_x)
            normalized.append(landmarks[i + 1] - base_y)
        
        # Scale by maximum distance from wrist
        distances = [abs(x) for x in normalized]
        max_distance = max(distances) if distances else 1
        
        if max_distance > 0:
            normalized = [x / max_distance for x in normalized]
        
        return normalized
    
    def extract_landmarks_from_image(self, image_path):
        """Extract hand landmarks from a single image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands_detector.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # Extract landmarks from first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Convert to list of coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])
                
                # Normalize landmarks
                normalized_landmarks = self.normalize_landmarks(landmarks)
                
                if normalized_landmarks and len(normalized_landmarks) == 42:
                    return normalized_landmarks
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting landmarks from {image_path}: {e}")
            return None
    
    def close(self):
        """Close MediaPipe detector"""
        if self.hands_detector:
            self.hands_detector.close()

class SplitLearningClient:
    """Split Learning Client with resource monitoring"""
    
    def __init__(self, server_host='localhost', server_port=8080):
        self.server_host = server_host
        self.server_port = server_port
        self.client_socket = None
        self.client_model = None
        self.landmark_extractor = None
        self.resource_monitor = ResourceMonitor("Split Learning Client")
        
        # Paths
        self.model_path = "models/client_model.keras"
        self.test_data_path = "test"
        
        # Statistics
        self.total_processed = 0
        self.successful_extractions = 0
        self.successful_predictions = 0
        self.session_results = []
        
    def load_client_model(self):
        """Load client model"""
        try:
            logger.info("Loading client model...")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Client model not found: {self.model_path}")
            
            self.client_model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Client model loaded: {self.client_model.input_shape} -> {self.client_model.output_shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load client model: {e}")
            return False
    
    def initialize_landmark_extractor(self):
        """Initialize MediaPipe landmark extractor"""
        try:
            self.landmark_extractor = HandLandmarkExtractor()
            self.landmark_extractor.initialize_detector()
            logger.info("MediaPipe landmark extractor initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize landmark extractor: {e}")
            return False
    
    def connect_to_server(self):
        """Connect to split learning server"""
        try:
            logger.info(f"Connecting to server at {self.server_host}:{self.server_port}")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.server_host, self.server_port))
            
            # Send ping to test connection
            ping_message = {'type': 'ping', 'timestamp': datetime.now().isoformat()}
            response = self.send_message(ping_message)
            
            if response and response.get('type') == 'pong':
                logger.info("✅ Successfully connected to server")
                return True
            else:
                logger.error("❌ Failed to get pong response from server")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            return False
    
    def send_message(self, message):
        """Send message to server and receive response"""
        try:
            # Send message
            message_data = json.dumps(message).encode('utf-8')
            message_length = len(message_data).to_bytes(4, byteorder='big')
            
            self.client_socket.send(message_length + message_data)
            
            # Receive response length
            length_data = self.client_socket.recv(4)
            if not length_data:
                return None
            
            response_length = int.from_bytes(length_data, byteorder='big')
            
            # Receive response data
            response_data = b''
            while len(response_data) < response_length:
                chunk = self.client_socket.recv(response_length - len(response_data))
                if not chunk:
                    break
                response_data += chunk
            
            return json.loads(response_data.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error communicating with server: {e}")
            return None
    
    def process_single_image(self, image_path):
        """Process a single image with resource monitoring"""
        image_name = os.path.basename(image_path)
        logger.info(f"\n🖼️ Processing: {image_name}")
        
        try:
            # Step 1: Extract landmarks (MONITORED)
            landmarks = self.resource_monitor.measure_operation(
                "MediaPipe_Extraction",
                self.landmark_extractor.extract_landmarks_from_image,
                image_path
            )
            
            if landmarks is None:
                logger.warning(f"❌ No landmarks detected in {image_name}")
                return {'image_name': image_name, 'success': False, 'error': 'No landmarks detected'}
            
            self.successful_extractions += 1
            
            # Step 2: Client model inference (MONITORED)
            landmarks_array = np.array(landmarks).reshape(1, -1)
            features = self.resource_monitor.measure_operation(
                "Client_Model_Inference",
                self.client_model.predict,
                landmarks_array,
                verbose=0
            )
            
            # Step 3: Send features to server (MONITORED)
            message = {
                'type': 'prediction',
                'image_name': image_name,
                'features': features.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            response = self.resource_monitor.measure_operation(
                "Server_Communication",
                self.send_message,
                message
            )
            
            if response and response.get('type') == 'prediction_result':
                result = response['result']
                
                if 'error' not in result:
                    self.successful_predictions += 1
                    logger.info(f"✅ Result: {result['predicted_class']} ({result['confidence']:.3f})")
                    
                    return {
                        'image_name': image_name,
                        'success': True,
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence'],
                        'server_inference_time': result['inference_time_ms'] / 1000,
                        'features_size_bytes': len(features.tobytes()),
                        'resource_stats': dict(self.resource_monitor.operation_stats)
                    }
                else:
                    logger.error(f"❌ Server error: {result['error']}")
                    return {'image_name': image_name, 'success': False, 'error': result['error']}
            else:
                logger.error(f"❌ Invalid server response")
                return {'image_name': image_name, 'success': False, 'error': 'Invalid server response'}
                
        except Exception as e:
            logger.error(f"❌ Error processing {image_name}: {e}")
            return {'image_name': image_name, 'success': False, 'error': str(e)}
    
    def process_test_images(self):
        """Process all test images with resource monitoring"""
        if not os.path.exists(self.test_data_path):
            logger.error(f"Test data path not found: {self.test_data_path}")
            return False
        
        logger.info("="*60)
        logger.info("PROCESSING TEST IMAGES WITH RESOURCE MONITORING")
        logger.info("="*60)
        
        # Get test images
        gesture_classes = ['call', 'fist', 'like', 'ok', 'palm', 'peace']
        all_images = []
        
        for gesture_class in gesture_classes:
            class_path = os.path.join(self.test_data_path, gesture_class)
            if os.path.exists(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(class_path, image_file)
                        all_images.append((image_path, gesture_class))
        
        if not all_images:
            logger.error("No test images found!")
            return False
        
        # Limit to 10 images for resource analysis
        all_images = all_images[:10]
        logger.info(f"Processing {len(all_images)} images for resource analysis...")
        
        # Process images
        class_results = {gesture: {'correct': 0, 'total': 0} for gesture in gesture_classes}
        
        for i, (image_path, true_class) in enumerate(all_images, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"[{i}/{len(all_images)}] {os.path.basename(image_path)} (True: {true_class})")
            logger.info('='*50)
            
            result = self.process_single_image(image_path)
            result['true_class'] = true_class
            
            self.session_results.append(result)
            self.total_processed += 1
            
            if result['success']:
                predicted_class = result['predicted_class']
                class_results[true_class]['total'] += 1
                
                if predicted_class == true_class:
                    class_results[true_class]['correct'] += 1
                    logger.info(f"  ✅ Correct prediction!")
                else:
                    logger.info(f"  ❌ Incorrect: predicted {predicted_class}")
            
            # Reset operation stats for next image
            self.resource_monitor.operation_stats = {}
        
        # Print final resource analysis
        self.print_resource_analysis()
        return True
    
    def print_resource_analysis(self):
        """Print comprehensive resource analysis"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE RESOURCE ANALYSIS")
        logger.info("="*80)
        
        if not self.session_results:
            return
        
        # Aggregate resource data
        successful_results = [r for r in self.session_results if r['success']]
        
        if not successful_results:
            logger.info("No successful results to analyze")
            return
        
        # Extract timing data
        mediapipe_times = []
        client_inference_times = []
        communication_times = []
        server_inference_times = []
        feature_sizes = []
        
        for result in successful_results:
            if 'resource_stats' in result:
                stats = result['resource_stats']
                mediapipe_times.append(stats.get('MediaPipe_Extraction', {}).get('time_seconds', 0))
                client_inference_times.append(stats.get('Client_Model_Inference', {}).get('time_seconds', 0))
                communication_times.append(stats.get('Server_Communication', {}).get('time_seconds', 0))
                server_inference_times.append(result.get('server_inference_time', 0))
                feature_sizes.append(result.get('features_size_bytes', 0))
        
        # Calculate averages
        def safe_mean(lst):
            return np.mean(lst) if lst else 0
        
        avg_mediapipe = safe_mean(mediapipe_times)
        avg_client = safe_mean(client_inference_times)
        avg_communication = safe_mean(communication_times)
        avg_server = safe_mean(server_inference_times)
        avg_feature_size = safe_mean(feature_sizes)
        
        total_avg = avg_mediapipe + avg_client + avg_communication
        
        print(f"📊 AVERAGE RESOURCE USAGE PER IMAGE:")
        print(f"   {'Component':<25} | {'Time (ms)':<10} | {'% of Total':<12} | {'Location':<10}")
        print(f"   {'-'*25} | {'-'*10} | {'-'*12} | {'-'*10}")
        print(f"   {'MediaPipe Processing':<25} | {avg_mediapipe*1000:<10.1f} | {avg_mediapipe/total_avg*100:<12.1f} | {'EDGE':<10}")
        print(f"   {'Client NN Inference':<25} | {avg_client*1000:<10.1f} | {avg_client/total_avg*100:<12.1f} | {'EDGE':<10}")
        print(f"   {'Server NN Inference':<25} | {avg_server*1000:<10.1f} | {avg_server/total_avg*100:<12.1f} | {'SERVER':<10}")
        print(f"   {'Network Communication':<25} | {avg_communication*1000:<10.1f} | {avg_communication/total_avg*100:<12.1f} | {'NETWORK':<10}")
        print(f"   {'-'*25} | {'-'*10} | {'-'*12} | {'-'*10}")
        print(f"   {'TOTAL':<25} | {total_avg*1000:<10.1f} | {'100.0':<12} | {'':<10}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        edge_percentage = (avg_mediapipe + avg_client) / total_avg * 100
        server_percentage = avg_server / total_avg * 100
        
        print(f"   • Edge device handles: {edge_percentage:.1f}% of computation")
        print(f"   • Server handles: {server_percentage:.1f}% of computation")
        print(f"   • MediaPipe is {avg_mediapipe/avg_client:.1f}x slower than client NN")
        print(f"   • Average feature transfer: {avg_feature_size:.0f} bytes")
        print(f"   • Processing rate: {1/total_avg:.1f} images/second")
        
        print(f"\n💡 OPTIMIZATION OPPORTUNITIES:")
        if avg_mediapipe > avg_client * 5:
            print(f"   🔴 MediaPipe is the bottleneck ({avg_mediapipe/total_avg*100:.1f}% of time)")
            print(f"      → Optimize MediaPipe settings")
            print(f"      → Use smaller input images")
            print(f"      → Reduce detection confidence")
        
        if edge_percentage > 80:
            print(f"   ⚠️ Edge device doing most work ({edge_percentage:.1f}%)")
            print(f"      → Consider moving more processing to server")
            print(f"      → Reduce feature extraction complexity")
        
        print(f"\n📈 CURRENT SPLIT LEARNING EFFICIENCY:")
        bandwidth_vs_image = avg_feature_size / (150 * 1024)  # Assume 150KB images
        print(f"   • Data reduction: {bandwidth_vs_image*100:.2f}% of raw image size")
        print(f"   • Privacy level: HIGH (only landmarks transmitted)")
        print(f"   • Edge resource usage: {edge_percentage:.1f}% (target: <50%)")
    
    def disconnect_from_server(self):
        """Disconnect from server"""
        if self.client_socket:
            try:
                disconnect_message = {'type': 'disconnect'}
                self.send_message(disconnect_message)
                self.client_socket.close()
                logger.info("Disconnected from server")
            except:
                pass
    
    def cleanup(self):
        """Cleanup resources"""
        self.disconnect_from_server()
        if self.landmark_extractor:
            self.landmark_extractor.close()

def main():
    """Main function with resource monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split Learning Client with Resource Monitoring')
    parser.add_argument('--server', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('--test-path', default='test', help='Path to test images (default: test)')
    
    args = parser.parse_args()
    
    # Create client
    client = SplitLearningClient(server_host=args.server, server_port=args.port)
    client.test_data_path = args.test_path
    
    try:
        # Initialize everything
        logger.info("Initializing split learning client with resource monitoring...")
        
        if not client.load_client_model():
            return False
        
        if not client.initialize_landmark_extractor():
            return False
        
        if not client.connect_to_server():
            return False
        
        # Process test images with monitoring
        success = client.process_test_images()
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Client interrupted by user")
    except Exception as e:
        logger.error(f"Client error: {e}")
        return False
    finally:
        client.cleanup()
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
