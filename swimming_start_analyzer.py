import cv2
import mediapipe as mp
import time
import google.generativeai as genai
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import os

# gemini api key setup
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

# YOLOv8 for person detection (removing ball detection for swimming)
yolo_model = YOLO("yolov8n.pt")

def detect_person(frame):
    results = yolo_model.predict(source=frame, verbose=False)[0]
    if results.boxes is None:
        return None
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        if "person" in label.lower():
            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            return (cx, cy)
    return None

# gemini feedback generation for swimming starts
def generate_feedback(pose_sequence):
    pose_text = ""
    for i, pose in enumerate(pose_sequence):
        pose_text += f"Frame {i+1}: " + ", ".join([
            f"{name}({round(x,2)}, {round(y,2)})" for name, (x, y) in pose.items()
        ]) + "\n"

    prompt = f"""
You are a swimming coach. Analyze this sequence of joint positions from a swimmer's racing start motion.

- Each line contains normalized (x, y) joint coordinates from a swimmer preparing for and executing a racing start.
- Focus on body position, angle of entry, start mechanics, and streamline technique.
- Key aspects: starting position, arm positioning, dive angle, streamline.

Sequence:
{pose_text}

Give specific, actionable feedback for improving the swimming start technique. Respond with exactly 2 sentences only, Max of 25 words. 
"""

    response = model.generate_content(prompt)
    feedback_text = response.text.strip()
    
    # More robust sentence splitting - look for sentence-ending patterns
    import re
    sentences = re.split(r'[.!?]+\s+', feedback_text)
    # Remove empty sentences and clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Ensure exactly 2 sentences
    if len(sentences) > 2:
        sentences = sentences[:2]
    elif len(sentences) == 1:
        # If only one sentence, split it in half if it's too long
        if len(sentences[0]) > 100:
            words = sentences[0].split()
            mid = len(words) // 2
            sentences = [' '.join(words[:mid]) + '.', ' '.join(words[mid:])]
    
    # Rejoin sentences and ensure proper formatting
    result = '. '.join(sentences)
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
        
    return result

# player pose setup + extraction
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(landmarks):
    important_joints = {
        "nose": mp_pose.PoseLandmark.NOSE,
        "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
        "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
        "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
        "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
        "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
        "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
        "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
        "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
        "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
        "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE
    }

    keypoints = {}
    for name, index in important_joints.items():
        landmark = landmarks[index]
        keypoints[name] = (landmark.x, landmark.y)
    return keypoints

def check_full_body_visible(landmarks):
    """Check if essential body parts are visible for swimming analysis"""
    # Check if at least one shoulder is visible
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    if not (is_joint_visible(left_shoulder) or is_joint_visible(right_shoulder)):
        return False
    
    # Check if at least one hip is visible
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    if not (is_joint_visible(left_hip) or is_joint_visible(right_hip)):
        return False
    
    # Check if at least one knee is visible
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    if not (is_joint_visible(left_knee) or is_joint_visible(right_knee)):
        return False
    
    # Check if at least one ankle is visible
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    if not (is_joint_visible(left_ankle) or is_joint_visible(right_ankle)):
        return False
    
    # Check if at least one wrist is visible
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    if not (is_joint_visible(left_wrist) or is_joint_visible(right_wrist)):
        return False
    
    return True

def is_joint_visible(landmark):
    """Helper function to check if a single joint is visible and within frame"""
    return (landmark.visibility >= 0.5 and 
            landmark.x >= 0.05 and landmark.x <= 0.95 and 
            landmark.y >= 0.05 and landmark.y <= 0.95)

def is_start_motion(pose_seq):
    if len(pose_seq) < 15:
        return False
    
    # Check for forward lean indicating start preparation
    nose_y = [pose["nose"][1] for pose in pose_seq]
    hip_y = [pose["left_hip"][1] for pose in pose_seq]
    
    # Calculate body angle changes (forward lean)
    lean_angles = []
    for i in range(len(pose_seq)):
        if i < len(nose_y) and i < len(hip_y):
            # Simple approximation of forward lean
            lean = nose_y[i] - hip_y[i]
            lean_angles.append(lean)
    
    if len(lean_angles) < 10:
        return False
        
    # Check for significant forward movement/lean change
    initial_lean = np.mean(lean_angles[:5])
    final_lean = np.mean(lean_angles[-5:])
    lean_change = abs(final_lean - initial_lean)
    
    # Also check for explosive movement (position changes)
    hip_x = [pose["left_hip"][0] for pose in pose_seq]
    position_change = abs(hip_x[-1] - hip_x[0]) if len(hip_x) > 1 else 0
    
    return lean_change > 0.1 or position_change > 0.15

# Main capture loop
cap = cv2.VideoCapture(0)
pose_buffer = []
feedback_displayed = False

print("Starting Swimming Start Analyzer. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Resize frame to make video window bigger (2x original size to fill window better)
    height, width = frame.shape[:2]
    new_width = int(width * 2.0)
    new_height = int(height * 2.0)
    frame = cv2.resize(frame, (new_width, new_height))
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    h, w, _ = frame.shape

    if results.pose_landmarks:
        # Check if full body is visible
        full_body_visible = check_full_body_visible(results.pose_landmarks.landmark)
        
        if not full_body_visible:
            # Display warning message when full body is not visible (moved to bottom right)
            cv2.putText(frame, "FULL BODY NEEDS TO BE IN FRAME", (50, frame.shape[0] - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, "Step back to show head to feet", (50, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Don't clear pose buffer or feedback when full body not visible during feedback display
            if not feedback_displayed:
                pose_buffer = []
        else:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            keypoints = extract_keypoints(results.pose_landmarks.landmark)

            pose_buffer.append(keypoints)

            if len(pose_buffer) > 30:  # Longer buffer for swimming starts
                pose_buffer.pop(0)

            if is_start_motion(pose_buffer) and not feedback_displayed:
                print("Swimming start motion detected. Sending to Gemini.")
                feedback = generate_feedback(pose_buffer)
                print("Gemini Feedback:\n", feedback)
                feedback_displayed = True
                feedback_time = time.time()
                # Clear pose buffer after detecting start to prepare for next analysis
                pose_buffer = []

            # Draw keypoints with swimming-specific colors
            for name, (x, y) in keypoints.items():
                cx, cy = int(x * w), int(y * h)
                if name == "nose":
                    color = (255, 0, 0)  # Blue for head position
                elif "shoulder" in name:
                    color = (0, 255, 255)  # Yellow for shoulders
                elif "hip" in name:
                    color = (255, 255, 0)  # Cyan for hips
                elif "wrist" in name:
                    color = (0, 255, 255)  # Yellow for hands
                elif "ankle" in name:
                    color = (255, 0, 255)  # Magenta for feet
                else:
                    color = (0, 255, 0)  # Green for other joints
                cv2.circle(frame, (cx, cy), 6, color, -1)
    else:
        # No body detected at all
        cv2.putText(frame, "STEP IN FRAME", (200, frame.shape[0] - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        # Don't clear feedback when no body detected during feedback display
        if not feedback_displayed:
            pose_buffer = []

    def draw_feedback(frame, feedback_text, x=20, y=45, max_width=900, line_height=35):
        lines = []
        for line in feedback_text.split('\n'):
            words = line.split(' ')
            current_line = ""
            for w in words:
                test_line = current_line + w + " "
                (text_w, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                if text_w > max_width:
                    lines.append(current_line.strip())
                    current_line = w + " "
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line.strip())

        rect_height = line_height * len(lines) + 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (x-15, y - line_height), (x + max_width + 15, y - line_height + rect_height), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for i, line in enumerate(lines):
            y_pos = y + i * line_height
            cv2.putText(frame, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display feedback until next start is detected
    if feedback_displayed:
        draw_feedback(frame, feedback)
        
        # Reset feedback when a new start motion is detected
        if results.pose_landmarks and check_full_body_visible(results.pose_landmarks.landmark):
            if len(pose_buffer) >= 15 and is_start_motion(pose_buffer):
                feedback_displayed = False  # This will trigger new feedback generation above

    cv2.putText(frame, "Swimming Start Analyzer", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Swimming Start Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
