# Driver Drowsiness Detection System
# Uses CNN models to detect eye closure and mouth yawning
# Triggers alarm when driver shows signs of drowsiness

import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import pygame
import time
import os
import sys

# Import custom models and preprocessing function
from eye_cnn import EyeCNN
from mouth_cnn import MouthCNN
from utils_preprocess import preprocess_roi

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get the parent directory to access models and audio folders
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load Eye CNN Model (0=Closed, 1=Open)
eye_model_path = os.path.join(parent_dir, "models", "eye_cnn.pth")
if not os.path.exists(eye_model_path):
    print(f"ERROR: Eye model not found at {eye_model_path}")
    sys.exit(1)

eye_model = EyeCNN().to(device)
eye_model.load_state_dict(torch.load(eye_model_path, map_location=device, weights_only=True))
eye_model.eval()
print("Eye CNN model loaded successfully")

# Load Mouth CNN Model (0=Normal, 1=Yawning)
mouth_model_path = os.path.join(parent_dir, "models", "mouth_cnn.pth")
if not os.path.exists(mouth_model_path):
    print(f"ERROR: Mouth model not found at {mouth_model_path}")
    sys.exit(1)

mouth_model = MouthCNN().to(device)
mouth_model.load_state_dict(torch.load(mouth_model_path, map_location=device, weights_only=True))
mouth_model.eval()
print("Mouth CNN model loaded successfully")

# Initialize Pygame for alarm sound
pygame.mixer.init()
alarm_path = os.path.join(parent_dir, "audio", "alarm.wav")
if not os.path.exists(alarm_path):
    print(f"WARNING: Alarm sound not found at {alarm_path}")
    alarm_available = False
else:
    pygame.mixer.music.load(alarm_path)
    alarm_available = True
    print("Alarm sound loaded successfully")
# Initialize MediaPipe Face Mesh
# Detects 468 facial landmarks for precise eye and mouth region extraction
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,        # Optimized for video streaming
    max_num_faces=1,                 # Detect only driver's face
    refine_landmarks=True,           # Enhanced landmarks for eyes/lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot access webcam")
    sys.exit(1)

# Set camera resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Webcam initialized successfully")

# Drowsiness detection parameters
EYE_CLOSE_THRESHOLD = 2.0  # Eyes closed for 2 seconds = drowsy
YAWN_FRAME_THRESHOLD = 15   # Yawning for 15 frames = drowsy
eyes_closed_start_time = None
yawn_frame_counter = 0

# MediaPipe landmark indices for eye and mouth regions
LEFT_EYE_LANDMARKS = [33, 133, 160, 158, 157, 173]
RIGHT_EYE_LANDMARKS = [362, 263, 387, 385, 386, 380]
MOUTH_LANDMARKS = [61, 291, 0, 17, 78, 308]
def extract_roi(frame, face_landmarks, indices, scale=1.8):
    """
    Extract Region of Interest (ROI) from frame based on landmark indices
    
    Args:
        frame: Input video frame
        face_landmarks: MediaPipe face landmarks
        indices: List of landmark indices to extract
        scale: Scaling factor to expand ROI (default 1.8)
    
    Returns:
        roi: Extracted image region
        bbox: Bounding box coordinates (x1, y1, x2, y2)
    """
    h, w, _ = frame.shape
    
    # Get all landmark coordinates
    xs = [int(face_landmarks.landmark[i].x * w) for i in indices]
    ys = [int(face_landmarks.landmark[i].y * h) for i in indices]
    
    if not xs or not ys:
        return None, None
    
    # Calculate center and bounding box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2
    
    box_size = int(max(max_x - min_x, max_y - min_y) * scale)
    
    # Ensure box stays within frame boundaries
    x1 = max(cx - box_size // 2, 0)
    y1 = max(cy - box_size // 2, 0)
    x2 = min(cx + box_size // 2, w)
    y2 = min(cy + box_size // 2, h)
    
    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None, None
    
    return roi, (x1, y1, x2, y2)

print("\n" + "="*50)
print("DRIVER DROWSINESS DETECTION SYSTEM")
print("="*50)
print("Controls:")
print("  - Press 'q' to quit")
print("  - Ensure your face is clearly visible")
print("="*50 + "\n")

# Main detection loop
frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from webcam")
            break
        
        frame_count += 1
        h, w, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)
        
        # Default status
        eyes_closed = False
        yawning = False
        status_text = "Status: Alert"
        status_color = (0, 255, 0)  # Green
        #define a function to extract ROIs based on landmark indices
        # Process face landmarks if detected
        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            
            # ==================== EYE DETECTION ====================
            try:
                # Extract left and right eye ROIs
                left_eye_roi, left_box = extract_roi(frame, face_landmarks, LEFT_EYE_LANDMARKS, scale=1.8)
                right_eye_roi, right_box = extract_roi(frame, face_landmarks, RIGHT_EYE_LANDMARKS, scale=1.8)
                
                if left_eye_roi is not None and right_eye_roi is not None:
                    # Preprocess ROIs for CNN
                    left_tensor = preprocess_roi(left_eye_roi, device)
                    right_tensor = preprocess_roi(right_eye_roi, device)
                    
                    if left_tensor is not None and right_tensor is not None:
                        # Get predictions (0=Closed, 1=Open)
                        with torch.no_grad():
                            left_output = eye_model(left_tensor)
                            right_output = eye_model(right_tensor)
                            
                            left_probs = F.softmax(left_output, dim=1)
                            right_probs = F.softmax(right_output, dim=1)
                            
                            left_pred = torch.argmax(left_probs, dim=1).item()
                            right_pred = torch.argmax(right_probs, dim=1).item()
                        
                        # Draw bounding boxes and labels
                        left_label = "Open" if left_pred == 1 else "Closed"
                        right_label = "Open" if right_pred == 1 else "Closed"
                        
                        cv2.rectangle(frame, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"L: {left_label}", (left_box[0], left_box[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        cv2.rectangle(frame, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"R: {right_label}", (right_box[0], right_box[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Both eyes closed = drowsy
                        if left_pred == 0 and right_pred == 0:
                            eyes_closed = True
            
            except Exception as e:
                print(f"Eye detection error: {e}")
            
            # ==================== MOUTH DETECTION ====================
            try:
                # Extract mouth ROI
                mouth_roi, mouth_box = extract_roi(frame, face_landmarks, MOUTH_LANDMARKS, scale=1.5)
                
                if mouth_roi is not None:
                    # Preprocess for CNN
                    mouth_tensor = preprocess_roi(mouth_roi, device)
                    
                    if mouth_tensor is not None:
                        # Get prediction (0=Normal, 1=Yawning)
                        with torch.no_grad():
                            mouth_output = mouth_model(mouth_tensor)
                            mouth_probs = F.softmax(mouth_output, dim=1)
                            mouth_pred = torch.argmax(mouth_probs, dim=1).item()
                        
                        # Draw bounding box and label
                        mouth_label = "Yawning" if mouth_pred == 1 else "Normal"
                        mouth_color = (0, 0, 255) if mouth_pred == 1 else (0, 255, 0)
                        
                        cv2.rectangle(frame, (mouth_box[0], mouth_box[1]), (mouth_box[2], mouth_box[3]), mouth_color, 2)
                        cv2.putText(frame, f"Mouth: {mouth_label}", (mouth_box[0], mouth_box[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mouth_color, 2)
                        
                        # Track yawning frames
                        if mouth_pred == 1:
                            yawn_frame_counter += 1
                            if yawn_frame_counter >= YAWN_FRAME_THRESHOLD:
                                yawning = True
                        else:
                            yawn_frame_counter = 0
            
            except Exception as e:
                print(f"Mouth detection error: {e}")
        
        else:
            # No face detected
            cv2.putText(frame, "No Face Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            eyes_closed_start_time = None
            yawn_frame_counter = 0
        
        # ==================== DROWSINESS LOGIC ====================
        # Track eye closure duration
        if eyes_closed:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            elapsed = time.time() - eyes_closed_start_time
        else:
            eyes_closed_start_time = None
            elapsed = 0
        
        # Determine drowsiness state
        is_drowsy = False
        
        if eyes_closed and elapsed >= EYE_CLOSE_THRESHOLD:
            is_drowsy = True
            status_text = f"DROWSY: Eyes Closed {elapsed:.1f}s"
            status_color = (0, 0, 255)  # Red
        elif yawning:
            is_drowsy = True
            status_text = f"DROWSY: Yawning Detected"
            status_color = (0, 0, 255)  # Red
        elif eyes_closed:
            status_text = f"Warning: Eyes Closing {elapsed:.1f}s"
            status_color = (0, 165, 255)  # Orange
        else:
            status_text = "Status: Alert"
            status_color = (0, 255, 0)  # Green
        
        # ==================== ALARM SYSTEM ====================
        if is_drowsy:
            # Display alert message
            cv2.rectangle(frame, (10, 10), (w - 10, 100), (0, 0, 255), -1)
            cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Play alarm sound
            if alarm_available and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)  # Loop alarm
        else:
            # Stop alarm when alert
            if alarm_available:
                pygame.mixer.music.stop()
        
        # ==================== DISPLAY INFO ====================
        # Status bar at bottom
        cv2.rectangle(frame, (0, h - 60), (w, h), (50, 50, 50), -1)
        cv2.putText(frame, status_text, (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (w - 180, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow("Driver Drowsiness Detection System", frame)
        
        # Exit on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nExiting system...")
            break

except KeyboardInterrupt:
    print("\nSystem interrupted by user")
except Exception as e:
    print(f"\nERROR: {e}")
finally:
    # Cleanup
    print("Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()
    if alarm_available:
        pygame.mixer.music.stop()
    face_mesh.close()
    print("System shut down successfully")