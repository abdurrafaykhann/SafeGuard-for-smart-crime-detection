#!/usr/bin/env python3
"""
Enhanced Violence Detection System
Advanced AI-powered violence detection with multiple detection methods
"""

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedViolenceDetector:
    def __init__(self):
        """Initialize the enhanced violence detection system"""
        self.yolo_model = None
        self.motion_history = []
        self.color_history = []
        self.shape_history = []
        self.violence_confidence_history = []
        self.non_violence_confidence_history = []
        
        # Detection thresholds - Maximum accuracy optimized
        self.MOTION_THRESHOLD_LOW = 0.10
        self.MOTION_THRESHOLD_MEDIUM = 0.18
        self.MOTION_THRESHOLD_HIGH = 0.30
        
        self.RED_DOMINANCE_THRESHOLD = 0.45  # Maximum accuracy threshold
        self.BLOOD_RED_THRESHOLD = 0.55      # Maximum accuracy for blood detection
        
        self.PROXIMITY_THRESHOLD = 120
        self.SHAPE_VIOLENCE_THRESHOLD = 0.15
        
        # Confidence weights for ensemble detection
        self.weights = {
            'motion': 0.25,
            'color': 0.20,
            'shape': 0.15,
            'proximity': 0.20,
            'yolo': 0.20
        }
        
        self.load_models()
    
    def load_models(self):
        """Load YOLO and other detection models"""
        try:
            # Try to load custom violence detection model
            if os.path.exists("Yolo_nano_weights.pt"):
                self.yolo_model = YOLO("Yolo_nano_weights.pt")
                logger.info("✅ Custom YOLO violence detection model loaded")
                print(f"📋 Model classes: {self.yolo_model.names}")
            else:
                # Fallback to default YOLO model
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("✅ Default YOLO model loaded (violence detection limited)")
                print(f"📋 Model classes: {self.yolo_model.names}")
        except Exception as e:
            logger.error(f"❌ Error loading YOLO model: {e}")
            self.yolo_model = None
    
    def analyze_motion(self, frame, prev_frame):
        """Enhanced motion analysis for violence detection"""
        if prev_frame is None:
            return 0.0, "No motion"
        
        # Convert to grayscale for motion analysis
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(gray_current, gray_prev)
        
        # Apply Gaussian blur to reduce noise
        blurred_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)
        
        # Calculate motion metrics
        motion_score = np.mean(blurred_diff) / 255.0
        motion_variance = np.var(blurred_diff) / 255.0
        
        # Detect motion patterns
        motion_pattern = self.analyze_motion_pattern(motion_score, motion_variance)
        
        return motion_score, motion_pattern
    
    def analyze_motion_pattern(self, motion_score, motion_variance):
        """Analyze motion patterns for violence indicators"""
        # Store motion history
        self.motion_history.append({
            'score': motion_score,
            'variance': motion_variance,
            'timestamp': time.time()
        })
        
        # Keep only last 15 frames
        if len(self.motion_history) > 15:
            self.motion_history.pop(0)
        
        # Maximum accuracy pattern analysis
        if len(self.motion_history) >= 5:
            recent_motions = [m['score'] for m in self.motion_history[-5:]]
            recent_variances = [m['variance'] for m in self.motion_history[-5:]]
            
            # Sudden spike detection (violence indicator) - Maximum accuracy
            if motion_score > self.MOTION_THRESHOLD_HIGH and np.std(recent_motions) > 0.020:
                return "Sudden violence"
            
            # Consistent high motion (fighting) - Maximum accuracy
            if np.mean(recent_motions) > self.MOTION_THRESHOLD_MEDIUM and np.std(recent_motions) < 0.008:
                return "Consistent fighting"
            
            # Erratic motion (struggle) - Maximum accuracy
            if np.std(recent_motions) > 0.030 and motion_score > self.MOTION_THRESHOLD_LOW:
                return "Erratic struggle"
            
            # High variance motion (weapon use) - Maximum accuracy
            if np.mean(recent_variances) > 0.020 and motion_score > self.MOTION_THRESHOLD_LOW:
                return "Weapon motion"
        
        return "Normal motion"
    
    def analyze_color_violence(self, frame):
        """Enhanced color analysis for violence detection"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Extract color channels
        b, g, r = cv2.split(frame)
        h, s, v = cv2.split(hsv)
        l, a, b_lab = cv2.split(lab)
        
        # Blood detection (red dominance analysis)
        red_dominance = np.mean(r) / (np.mean(g) + np.mean(b) + 1)
        red_saturation = np.mean(s) / 255.0
        
        # Maximum accuracy red dominance normalization
        # Only consider it violence if red is significantly dominant
        if red_dominance > 0.68:  # Red is more than 68% of total color (maximum accuracy threshold)
            red_violence_score = red_dominance
        else:
            red_violence_score = 0.0
        
        # Store color history
        self.color_history.append({
            'red_dominance': red_dominance,
            'red_saturation': red_saturation,
            'timestamp': time.time()
        })
        
        if len(self.color_history) > 10:
            self.color_history.pop(0)
        
        # Analyze color patterns
        violence_indicators = []
        
        # Maximum accuracy blood detection
        if red_violence_score > self.BLOOD_RED_THRESHOLD and red_saturation > 0.6:
            violence_indicators.append(("Blood detected", 0.98))
        elif red_violence_score > self.RED_DOMINANCE_THRESHOLD and red_saturation > 0.5:
            violence_indicators.append(("Red violence", 0.85))
        
        # Maximum accuracy dark/violent content detection
        brightness = np.mean(v) / 255.0
        if brightness < 0.20 and red_violence_score > 0.35:  # Maximum accuracy dark and higher red
            violence_indicators.append(("Dark violence", 0.75))
        
        # Maximum accuracy high contrast violence detection
        contrast = np.std(frame) / 255.0
        if contrast > 0.35 and red_violence_score > 0.30:  # Maximum accuracy contrast and red
            violence_indicators.append(("High contrast violence", 0.65))
        
        return violence_indicators, red_dominance, red_violence_score
    
    def analyze_non_violence_indicators(self, frame, motion_score, red_violence_score, person_boxes):
        """Enhanced analysis for non-violence indicators"""
        non_violence_indicators = []
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Extract color channels
        b, g, r = cv2.split(frame)
        h, s, v = cv2.split(hsv)
        l, a, b_lab = cv2.split(lab)
        
        # 1. Calm color analysis (green/blue dominance)
        green_dominance = np.mean(g) / (np.mean(r) + np.mean(b) + 1)
        blue_dominance = np.mean(b) / (np.mean(r) + np.mean(g) + 1)
        
        if green_dominance > 0.4:  # Green is calming
            non_violence_indicators.append(("Calm green environment", 0.8))
        if blue_dominance > 0.4:  # Blue is peaceful
            non_violence_indicators.append(("Peaceful blue environment", 0.8))
        
        # 2. Brightness analysis (well-lit scenes are usually safe)
        brightness = np.mean(v) / 255.0
        if brightness > 0.6:  # Well-lit environment
            non_violence_indicators.append(("Well-lit environment", 0.7))
        elif brightness > 0.4:  # Adequate lighting
            non_violence_indicators.append(("Adequate lighting", 0.6))
        
        # 3. Low contrast analysis (smooth scenes are usually safe)
        contrast = np.std(frame) / 255.0
        if contrast < 0.15:  # Low contrast, smooth scene
            non_violence_indicators.append(("Smooth scene", 0.7))
        elif contrast < 0.25:  # Moderate contrast
            non_violence_indicators.append(("Moderate scene", 0.6))
        
        # 4. Motion analysis for calm behavior
        if motion_score < self.MOTION_THRESHOLD_LOW * 0.5:  # Very low motion
            non_violence_indicators.append(("Calm motion", 0.9))
        elif motion_score < self.MOTION_THRESHOLD_LOW:  # Low motion
            non_violence_indicators.append(("Low motion", 0.8))
        
        # 5. Person behavior analysis
        if len(person_boxes) == 1:  # Single person, usually safe
            non_violence_indicators.append(("Single person", 0.7))
        elif len(person_boxes) == 0:  # No people, definitely safe
            non_violence_indicators.append(("No people present", 0.95))
        
        # 6. Red absence analysis
        if red_violence_score < 0.01:  # Very low red
            non_violence_indicators.append(("No red violence", 0.9))
        elif red_violence_score < 0.05:  # Low red
            non_violence_indicators.append(("Low red presence", 0.8))
        
        # 7. Saturation analysis (low saturation often indicates calm)
        saturation = np.mean(s) / 255.0
        if saturation < 0.3:  # Low saturation, calm colors
            non_violence_indicators.append(("Calm colors", 0.7))
        
        # 8. Edge density analysis (few edges often means calm scene)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        if edge_density < 0.05:  # Very few edges, smooth scene
            non_violence_indicators.append(("Smooth scene", 0.8))
        elif edge_density < 0.1:  # Few edges
            non_violence_indicators.append(("Low edge density", 0.7))
        
        return non_violence_indicators
    
    def analyze_shape_violence(self, frame):
        """Analyze shapes and objects for violence indicators"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        violence_shapes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue
            
            # Analyze shape characteristics
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Shape complexity
            complexity = area / (perimeter * perimeter)
            
            # Detect weapon-like shapes (long, thin objects)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            
            # Weapon detection
            if aspect_ratio > 3 and area > 200:
                violence_shapes.append(("Weapon-like object", 0.8))
            
            # Sharp object detection
            if complexity < 0.01 and aspect_ratio > 2:
                violence_shapes.append(("Sharp object", 0.7))
            
            # Screen/device detection for video violence
            if len(cv2.approxPolyDP(contour, 0.02 * perimeter, True)) == 4:
                if area > 1000:  # Large rectangular object
                    violence_shapes.append(("Screen/device", 0.3))
        
        return violence_shapes
    
    def analyze_proximity_violence(self, person_boxes):
        """Analyze person proximity for violence indicators"""
        if len(person_boxes) < 2:
            return []
        
        proximity_violence = []
        
        for i in range(len(person_boxes)):
            for j in range(i + 1, len(person_boxes)):
                x1, y1, x2, y2 = person_boxes[i]
                x3, y3, x4, y4 = person_boxes[j]
                
                # Calculate centers
                center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
                center2 = ((x3 + x4) // 2, (y3 + y4) // 2)
                
                # Calculate distance
                distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                
                # Calculate overlap
                overlap_x = max(0, min(x2, x4) - max(x1, x3))
                overlap_y = max(0, min(y2, y4) - max(y1, y3))
                overlap_area = overlap_x * overlap_y
                
                # Maximum accuracy proximity violence detection
                if distance < self.PROXIMITY_THRESHOLD:
                    if overlap_area > 0:
                        proximity_violence.append(("Physical contact", 0.95))
                    elif distance < 60:
                        proximity_violence.append(("Close combat", 0.9))
                    elif distance < 80:
                        proximity_violence.append(("Close combat", 0.8))
                    else:
                        proximity_violence.append(("Close proximity", 0.7))
        
        return proximity_violence
    
    def run_yolo_detection(self, frame):
        """Run YOLO-based violence detection"""
        if self.yolo_model is None:
            return [], 0
        
        try:
            results = self.yolo_model(frame)[0]
            
            violence_detections = []
            person_boxes = []
            
            for i in range(len(results.boxes.cls)):
                class_id = int(results.boxes.cls[i])
                confidence = float(results.boxes.conf[i])
                box = results.boxes.xyxy[i]
                
                if class_id == 0:  # non_violence (person)
                    person_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                elif class_id == 1:  # violence
                    violence_detections.append(("YOLO violence", confidence))
            
            return violence_detections, person_boxes
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return [], []
    
    def ensemble_detection(self, frame, prev_frame):
        """Ensemble violence detection combining multiple methods"""
        violence_indicators = []
        non_violence_confidence = 0.0
        red_violence_score = 0.0  # Initialize red violence score
        
        # 1. Motion Analysis
        motion_score, motion_pattern = self.analyze_motion(frame, prev_frame)
        if "violence" in motion_pattern.lower() or "fighting" in motion_pattern.lower():
            violence_indicators.append((motion_pattern, motion_score * 0.8))
        
        # 2. Color Analysis
        color_violence, red_dominance, red_violence_score = self.analyze_color_violence(frame)
        violence_indicators.extend(color_violence)
        
        # 3. Shape Analysis
        shape_violence = self.analyze_shape_violence(frame)
        violence_indicators.extend(shape_violence)
        
        # 4. YOLO Detection
        yolo_violence, person_boxes = self.run_yolo_detection(frame)
        violence_indicators.extend(yolo_violence)
        
        # 5. Proximity Analysis
        proximity_violence = self.analyze_proximity_violence(person_boxes)
        violence_indicators.extend(proximity_violence)
        
        # 6. Enhanced Non-Violence Analysis
        non_violence_indicators = self.analyze_non_violence_indicators(frame, motion_score, red_violence_score, person_boxes)
        
        # Calculate ensemble confidence
        violence_confidence = 0.0
        if violence_indicators:
            # Weighted average of all violence indicators
            total_weight = 0
            weighted_sum = 0
            
            for indicator, confidence in violence_indicators:
                if "blood" in indicator.lower():
                    weight = 0.35  # Increased weight for blood detection
                elif "weapon" in indicator.lower():
                    weight = 0.30  # Increased weight for weapon detection
                elif "contact" in indicator.lower():
                    weight = 0.25  # Increased weight for physical contact
                elif "yolo" in indicator.lower():
                    weight = 0.20  # Increased weight for YOLO detection
                else:
                    weight = 0.15  # Increased base weight
                
                weighted_sum += confidence * weight
                total_weight += weight
            
            violence_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Enhanced non-violence confidence calculation with indicators
        non_violence_confidence = 0.0
        if non_violence_indicators:
            # Calculate non-violence confidence from indicators
            non_violence_weighted_sum = 0
            non_violence_total_weight = 0
            
            for indicator, confidence in non_violence_indicators:
                if "no people" in indicator.lower():
                    weight = 0.25  # High weight for no people
                elif "calm motion" in indicator.lower():
                    weight = 0.20  # High weight for calm motion
                elif "no red violence" in indicator.lower():
                    weight = 0.20  # High weight for no red
                elif "smooth scene" in indicator.lower():
                    weight = 0.15  # Medium weight for smooth scene
                elif "well-lit" in indicator.lower():
                    weight = 0.10  # Medium weight for lighting
                else:
                    weight = 0.10  # Base weight for other indicators
                
                non_violence_weighted_sum += confidence * weight
                non_violence_total_weight += weight
            
            if non_violence_total_weight > 0:
                non_violence_confidence = non_violence_weighted_sum / non_violence_total_weight
        
        # Enhanced maximum accuracy non-violence confidence calculation
        # Use the higher confidence between indicators and fallback logic
        fallback_confidence = 0.0
        
        if violence_confidence < 0.02:  # Ultra-low violence confidence
            fallback_confidence = 0.995
        elif len(person_boxes) > 0 and violence_confidence < 0.08:
            fallback_confidence = 0.985
        elif motion_score < self.MOTION_THRESHOLD_LOW and red_violence_score < 0.005:
            fallback_confidence = 0.995
        elif len(person_boxes) > 0 and violence_confidence < 0.15:
            fallback_confidence = 0.97
        elif motion_score < self.MOTION_THRESHOLD_MEDIUM and red_violence_score < 0.02:
            fallback_confidence = 0.98
        elif violence_confidence < 0.25:  # Enhanced low violence confidence
            fallback_confidence = 0.90
        elif violence_confidence < 0.35:  # Medium violence confidence
            fallback_confidence = 0.80
        else:
            fallback_confidence = 0.60  # Default for higher violence
        
        # Use the higher confidence between indicators and fallback
        non_violence_confidence = max(non_violence_confidence, fallback_confidence)
        
        # Store confidence history
        self.violence_confidence_history.append(violence_confidence)
        self.non_violence_confidence_history.append(non_violence_confidence)
        
        if len(self.violence_confidence_history) > 10:
            self.violence_confidence_history.pop(0)
            self.non_violence_confidence_history.pop(0)
        
        return {
            'violence_detected': violence_confidence > 0.72,  # Maximum accuracy threshold for violence detection
            'violence_confidence': violence_confidence,
            'non_violence_confidence': non_violence_confidence,
            'violence_type': self.get_primary_violence_type(violence_indicators),
            'non_violence_type': self.get_primary_non_violence_type(non_violence_indicators),
            'total_persons': len(person_boxes),
            'motion_score': motion_score,
            'red_dominance': red_dominance,
            'violence_indicators': violence_indicators,
            'non_violence_indicators': non_violence_indicators,
            'person_boxes': person_boxes
        }
    
    def get_primary_violence_type(self, violence_indicators):
        """Get the primary type of violence detected"""
        if not violence_indicators:
            return "None"
        
        # Sort by confidence
        sorted_indicators = sorted(violence_indicators, key=lambda x: x[1], reverse=True)
        primary_indicator = sorted_indicators[0][0]
        
        # Categorize violence types
        if "blood" in primary_indicator.lower():
            return "Blood violence"
        elif "weapon" in primary_indicator.lower():
            return "Weapon violence"
        elif "contact" in primary_indicator.lower():
            return "Physical violence"
        elif "fighting" in primary_indicator.lower():
            return "Fighting"
        elif "yolo" in primary_indicator.lower():
            return "AI detected violence"
        else:
            return primary_indicator
    
    def get_primary_non_violence_type(self, non_violence_indicators):
        """Get the primary type of non-violence detected"""
        if not non_violence_indicators:
            return "None"
        
        # Sort by confidence
        sorted_indicators = sorted(non_violence_indicators, key=lambda x: x[1], reverse=True)
        primary_indicator = sorted_indicators[0][0]
        
        # Categorize non-violence types
        if "no people" in primary_indicator.lower():
            return "Empty scene"
        elif "calm motion" in primary_indicator.lower():
            return "Calm behavior"
        elif "no red violence" in primary_indicator.lower():
            return "No red indicators"
        elif "smooth scene" in primary_indicator.lower():
            return "Smooth environment"
        elif "well-lit" in primary_indicator.lower():
            return "Good lighting"
        elif "green" in primary_indicator.lower():
            return "Calm environment"
        elif "blue" in primary_indicator.lower():
            return "Peaceful environment"
        else:
            return primary_indicator
    
    def get_detection_summary(self):
        """Get summary of recent detections"""
        if not self.violence_confidence_history:
            return "No detection history"
        
        avg_violence = np.mean(self.violence_confidence_history)
        avg_non_violence = np.mean(self.non_violence_confidence_history)
        
        return {
            'average_violence_confidence': avg_violence,
            'average_non_violence_confidence': avg_non_violence,
            'detection_trend': 'Violence' if avg_violence > avg_non_violence else 'Non-violence',
            'confidence_difference': abs(avg_violence - avg_non_violence)
        }

# Global instance
enhanced_detector = EnhancedViolenceDetector()

def get_enhanced_detector():
    """Get the global enhanced violence detector instance"""
    return enhanced_detector 