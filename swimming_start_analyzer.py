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
- Key aspects: starting position, forward lean, arm positioning, and dive angle.

Sequence:
{pose_text}

Give specific, actionable feedback for improving the swimming start technique (max 2 sentences).
"""

    response = model.generate_content(prompt)
    return response.text

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    h, w, _ = frame.shape

    if results.pose_landmarks:
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

    def draw_feedback(frame, feedback_text, x=15, y=30, max_width=620, line_height=25):
        lines = []
        for line in feedback_text.split('\n'):
            words = line.split(' ')
            current_line = ""
            for w in words:
                test_line = current_line + w + " "
                (text_w, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                if text_w > max_width:
                    lines.append(current_line.strip())
                    current_line = w + " "
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line.strip())

        rect_height = line_height * len(lines) + 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x-10, y - line_height), (x + max_width + 10, y - line_height + rect_height), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for i, line in enumerate(lines):
            y_pos = y + i * line_height
            cv2.putText(frame, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Display feedback for 7 seconds (longer for swimming starts)
    if feedback_displayed and time.time() - feedback_time < 7:
        draw_feedback(frame, feedback)

    elif feedback_displayed and time.time() - feedback_time >= 7:
        feedback_displayed = False
        pose_buffer = []

    cv2.putText(frame, "Swimming Start Analyzer", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Swimming Start Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
