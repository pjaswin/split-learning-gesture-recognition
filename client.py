#!/usr/bin/env python3
"""
Split Learning Client for Gesture Recognition
Processes images with MediaPipe, runs client model, sends features to server
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
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        """
        Normalize landmarks relative to wrist position and scale
        Input: 42 coordinates [x1,y1,x2,y2,...,x21,y21]
        Output: Normalized 42 coordinates
        """
        if len(landmarks) != 42:
            return None
            
        # Use wrist (landmark 0) as reference point
        base_x, base_y = landmarks[0], landmarks[1]
        
        # Normalize relative to wrist
        normalized = []
        for i in range(0, len(landmarks), 2):
            normalized.append(landmarks[i] - base_x)      # x relative to wrist
            normalized.append(landmarks[i + 1] - base_y)  # y relative to wrist
        
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
    """Split Learning Client for gesture recognition"""
    
    def __init__(self, server_host='localhost', server_port=8080):
        self.server_host = server_host
        self.server_port = server_port
        self.client_socket = None
        self.client_model = None
        self.landmark_extractor = None
        
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
            ping_message = {
                'type': 'ping',
                'timestamp': datetime.now().isoformat()
            }
            
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
        """Process a single image through the split learning pipeline"""
        image_name = os.path.basename(image_path)
        logger.info(f"Processing: {image_name}")
        
        try:
            # Step 1: Extract landmarks
            start_time = time.time()
            landmarks = self.landmark_extractor.extract_landmarks_from_image(image_path)
            landmark_time = time.time() - start_time
            
            if landmarks is None:
                logger.warning(f"❌ No landmarks detected in {image_name}")
                return {
                    'image_name': image_name,
                    'success': False,
                    'error': 'No landmarks detected',
                    'landmark_extraction_time': landmark_time
                }
            
            self.successful_extractions += 1
            logger.debug(f"✅ Landmarks extracted from {image_name} in {landmark_time:.3f}s")
            
            # Step 2: Client model inference
            start_time = time.time()
            landmarks_array = np.array(landmarks).reshape(1, -1)
            features = self.client_model.predict(landmarks_array, verbose=0)
            client_inference_time = time.time() - start_time
            
            logger.debug(f"Client model output shape: {features.shape}")
            
            # Step 3: Send features to server
            start_time = time.time()
            message = {
                'type': 'prediction',
                'image_name': image_name,
                'features': features.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            response = self.send_message(message)
            communication_time = time.time() - start_time
            
            if response and response.get('type') == 'prediction_result':
                result = response['result']
                
                if 'error' not in result:
                    self.successful_predictions += 1
                    logger.info(f"✅ {image_name}: {result['predicted_class']} ({result['confidence']:.3f})")
                    
                    return {
                        'image_name': image_name,
                        'success': True,
                        'predicted_class': result['predicted_class'],
                        'confidence': result['confidence'],
                        'top_3_predictions': result['top_3_predictions'],
                        'landmark_extraction_time': landmark_time,
                        'client_inference_time': client_inference_time,
                        'server_inference_time': result['inference_time_ms'] / 1000,
                        'communication_time': communication_time,
                        'total_time': landmark_time + client_inference_time + communication_time,
                        'features_size_bytes': len(features.tobytes())
                    }
                else:
                    logger.error(f"❌ Server error for {image_name}: {result['error']}")
                    return {
                        'image_name': image_name,
                        'success': False,
                        'error': result['error'],
                        'landmark_extraction_time': landmark_time,
                        'client_inference_time': client_inference_time
                    }
            else:
                logger.error(f"❌ Invalid response from server for {image_name}")
                return {
                    'image_name': image_name,
                    'success': False,
                    'error': 'Invalid server response',
                    'landmark_extraction_time': landmark_time,
                    'client_inference_time': client_inference_time
                }
                
        except Exception as e:
            logger.error(f"❌ Error processing {image_name}: {e}")
            return {
                'image_name': image_name,
                'success': False,
                'error': str(e)
            }
    
    def process_test_images(self):
        """Process all test images"""
        if not os.path.exists(self.test_data_path):
            logger.error(f"Test data path not found: {self.test_data_path}")
            return False
        
        logger.info("="*60)
        logger.info("PROCESSING TEST IMAGES")
        logger.info("="*60)
        
        # Get all test images organized by class
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
        
        logger.info(f"Found {len(all_images)} test images")
        
        # Process each image
        class_results = {gesture: {'correct': 0, 'total': 0, 'predictions': []} for gesture in gesture_classes}
        
        for i, (image_path, true_class) in enumerate(all_images, 1):
            logger.info(f"\n[{i}/{len(all_images)}] Processing {os.path.basename(image_path)} (True: {true_class})")
            
            result = self.process_single_image(image_path)
            result['true_class'] = true_class
            
            self.session_results.append(result)
            self.total_processed += 1
            
            if result['success']:
                predicted_class = result['predicted_class']
                class_results[true_class]['total'] += 1
                class_results[true_class]['predictions'].append(predicted_class)
                
                if predicted_class == true_class:
                    class_results[true_class]['correct'] += 1
                    logger.info(f"  ✅ Correct prediction!")
                else:
                    logger.info(f"  ❌ Incorrect: predicted {predicted_class}")
            else:
                logger.info(f"  ❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Print results summary
        self.print_session_summary(class_results)
        return True
    
    def print_session_summary(self, class_results):
        """Print comprehensive session summary"""
        logger.info("\n" + "="*60)
        logger.info("SESSION SUMMARY")
        logger.info("="*60)
        
        # Overall statistics
        overall_accuracy = self.successful_predictions / self.total_processed if self.total_processed > 0 else 0
        extraction_rate = self.successful_extractions / self.total_processed if self.total_processed > 0 else 0
        
        logger.info(f"Total images processed: {self.total_processed}")
        logger.info(f"Successful landmark extractions: {self.successful_extractions} ({extraction_rate:.1%})")
        logger.info(f"Successful predictions: {self.successful_predictions} ({overall_accuracy:.1%})")
        
        # Per-class accuracy
        logger.info("\nPer-class results:")
        for gesture_class, results in class_results.items():
            if results['total'] > 0:
                accuracy = results['correct'] / results['total']
                logger.info(f"  {gesture_class}: {results['correct']}/{results['total']} ({accuracy:.1%})")
            else:
                logger.info(f"  {gesture_class}: No samples processed")
        
        # Timing statistics
        successful_results = [r for r in self.session_results if r['success']]
        if successful_results:
            avg_landmark_time = np.mean([r['landmark_extraction_time'] for r in successful_results])
            avg_client_time = np.mean([r['client_inference_time'] for r in successful_results])
            avg_server_time = np.mean([r['server_inference_time'] for r in successful_results])
            avg_comm_time = np.mean([r['communication_time'] for r in successful_results])
            avg_total_time = np.mean([r['total_time'] for r in successful_results])
            avg_feature_size = np.mean([r['features_size_bytes'] for r in successful_results])
            
            logger.info(f"\nTiming statistics (average):")
            logger.info(f"  Landmark extraction: {avg_landmark_time:.3f}s")
            logger.info(f"  Client inference: {avg_client_time:.3f}s")
            logger.info(f"  Server inference: {avg_server_time:.3f}s")
            logger.info(f"  Communication: {avg_comm_time:.3f}s")
            logger.info(f"  Total per image: {avg_total_time:.3f}s")
            logger.info(f"  Feature transfer size: {avg_feature_size:.0f} bytes")
    
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
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split Learning Client for Gesture Recognition')
    parser.add_argument('--server', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    parser.add_argument('--test-path', default='test', help='Path to test images (default: test)')
    
    args = parser.parse_args()
    
    # Create client
    client = SplitLearningClient(server_host=args.server, server_port=args.port)
    client.test_data_path = args.test_path
    
    try:
        # Initialize everything
        logger.info("Initializing split learning client...")
        
        if not client.load_client_model():
            return False
        
        if not client.initialize_landmark_extractor():
            return False
        
        if not client.connect_to_server():
            return False
        
        # Process test images
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
