import cv2
import mediapipe as mp
import time
import google.generativeai as genai
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import os
import math


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
You are a competitive swimming coach. Analyze this sequence of joint positions from a swimmer's racing start from the starting blocks.

- Each line contains normalized (x, y) joint coordinates from a swimmer preparing for and executing a competitive swimming start.
- Focus on: starting stance, crouch position, explosive leg drive, arm use, and entry angle into the pool.
- Key swimming start elements: block positioning, knee bend depth, forward lean, arm preparation, and water entry.

Sequence:
{pose_text}

Give specific, actionable feedback for improving the competitive swimming racing start technique. Focus on swimming pool starts, not platform diving. Respond with exactly 2 sentences only, Max of 30 words each.
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

import numpy as np
import math

def calculate_angle(p1, p2, p3):
    """
    Calculate angle between three points (p1-p2-p3)
    Returns angle in degrees
    """
    # Vector from p2 to p1
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    # Vector from p2 to p3
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Calculate angle
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norms == 0:
        return 0
    
    cos_angle = dot_product / norms
    cos_angle = np.clip(cos_angle, -1, 1)  # Prevent numerical errors
    angle = math.degrees(math.acos(cos_angle))
    
    return angle

def extract_joint_angles(pose):
    """
    Extract key joint angles from a single pose - simplified for available joints
    """
    angles = {}
    
    try:
        # Knee angles (should decrease significantly during start crouch)
        angles['left_knee'] = calculate_angle(
            pose['left_hip'], pose['left_knee'], pose['left_ankle']
        )
        angles['right_knee'] = calculate_angle(
            pose['right_hip'], pose['right_knee'], pose['right_ankle']
        )
        
        # Hip angles (torso to thigh angle)
        # Using shoulder as proxy for torso direction
        angles['left_hip_angle'] = calculate_angle(
            pose['left_shoulder'], pose['left_hip'], pose['left_knee']
        )
        angles['right_hip_angle'] = calculate_angle(
            pose['right_shoulder'], pose['right_hip'], pose['right_knee']
        )
        
        # Shoulder angles (arm position) - simplified
        angles['left_shoulder'] = calculate_angle(
            pose['left_elbow'], pose['left_shoulder'], pose['left_hip']
        )
        angles['right_shoulder'] = calculate_angle(
            pose['right_elbow'], pose['right_shoulder'], pose['right_hip']
        )
        
        # Torso angle (relative to vertical)
        # Calculate angle between shoulder-hip line and vertical
        shoulder_center = [
            (pose['left_shoulder'][0] + pose['right_shoulder'][0]) / 2,
            (pose['left_shoulder'][1] + pose['right_shoulder'][1]) / 2
        ]
        hip_center = [
            (pose['left_hip'][0] + pose['right_hip'][0]) / 2,
            (pose['left_hip'][1] + pose['right_hip'][1]) / 2
        ]
        
        # Create vertical reference point
        vertical_ref = [hip_center[0], hip_center[1] - 0.1]  # Smaller offset for normalized coords
        angles['torso_angle'] = calculate_angle(
            shoulder_center, hip_center, vertical_ref
        )
        
    except KeyError as e:
        print(f"Missing joint in pose: {e}")
        return None
    
    return angles

def is_start_motion(pose_seq, return_confidence=False):
    """
    Enhanced swimming start detection using joint angle analysis
    
    Args:
        pose_seq: Sequence of pose data
        return_confidence: If True, returns (is_start, confidence, details)
                          If False, returns just boolean
    """
    if len(pose_seq) < 15:
        if return_confidence:
            return False, 0.0, {"error": "Sequence too short"}
        return False
    
    # Extract angles for all poses
    angle_sequences = {}
    valid_poses = []
    
    for pose in pose_seq:
        angles = extract_joint_angles(pose)
        if angles is not None:
            valid_poses.append(angles)
    
    if len(valid_poses) < 10:
        if return_confidence:
            return False, 0.0, {"error": "Not enough valid poses"}
        return False
    
    # Convert to time series for each angle
    for angle_name in valid_poses[0].keys():
        angle_sequences[angle_name] = [pose[angle_name] for pose in valid_poses]
    
    # Analyze angle changes characteristic of swimming start
    start_indicators = 0
    total_checks = 0
    details = {}
    
    # 1. Knee angle analysis - should decrease significantly (deep crouch for swimming)
    knee_scores = []
    for knee in ['left_knee', 'right_knee']:
        if knee in angle_sequences:
            initial_knee = np.mean(angle_sequences[knee][:3])
            min_knee = np.min(angle_sequences[knee])
            knee_flexion = initial_knee - min_knee
            
            knee_passed = knee_flexion > 30  # Increased from 20 to 30 - requires deeper crouch
            knee_scores.append(knee_flexion)
            
            if knee_passed:
                start_indicators += 1
            total_checks += 1
    
    details['knee_flexion'] = {
        'scores': knee_scores,
        'passed': any(score > 30 for score in knee_scores),  # At least one knee must flex significantly
        'threshold': 30
    }
    
    # 2. Hip angle analysis - should show significant forward lean preparation
    hip_scores = []
    hip_indicators = 0
    for hip in ['left_hip_angle', 'right_hip_angle']:
        if hip in angle_sequences:
            initial_hip = np.mean(angle_sequences[hip][:3])
            final_hip = np.mean(angle_sequences[hip][-3:])
            hip_change = abs(final_hip - initial_hip)
            hip_scores.append(hip_change)
            
            if hip_change > 25:  # Increased from 15 to 25 - requires more significant hip movement
                start_indicators += 1
                hip_indicators += 1
            total_checks += 1
    
    details['hip_angle_change'] = {
        'scores': hip_scores,
        'passed': hip_indicators > 0,
        'threshold': 25
    }
    
    # 3. Torso angle analysis - forward lean (must be bent forward for swimming start)
    torso_indicators = 0
    is_upright = False
    
    if 'torso_angle' in angle_sequences:
        initial_torso = np.mean(angle_sequences['torso_angle'][:3])
        final_torso = np.mean(angle_sequences['torso_angle'][-3:])
        torso_lean = abs(final_torso - initial_torso)
        
        # Check if person is too upright (looking up instead of bent forward)
        # If torso angle is close to 0 (vertical), person is standing too upright
        avg_torso_angle = np.mean(angle_sequences['torso_angle'])
        is_upright = avg_torso_angle < 15  # Less than 15 degrees from vertical = too upright
        
        # Must have forward lean AND not be standing upright
        torso_passed = torso_lean > 10 and not is_upright
        if torso_passed:
            start_indicators += 1
            torso_indicators += 1
        total_checks += 1
        
        details['torso_lean'] = {
            'score': torso_lean,
            'avg_angle': avg_torso_angle,
            'is_upright': is_upright,
            'passed': torso_indicators > 0,
            'threshold': 10
        }
    
    # 4. Position stability check - if person is just standing still, reject
    position_variance = 0
    key_joints = ['left_hip', 'right_hip', 'left_knee', 'right_knee']
    
    for joint in key_joints:
        if joint in pose_seq[0] and joint in pose_seq[-1]:
            # Calculate variance in position over time
            x_positions = [pose[joint][0] for pose in pose_seq if joint in pose]
            y_positions = [pose[joint][1] for pose in pose_seq if joint in pose]
            
            if len(x_positions) > 5:
                x_var = np.var(x_positions)
                y_var = np.var(y_positions)
                joint_variance = x_var + y_var
                position_variance = max(position_variance, joint_variance)
    
    # If person is standing too still, it's not a start motion
    stillness_check = position_variance > 0.002  # Lowered from 0.01 to 0.002 based on test results
    if stillness_check:
        start_indicators += 1
    total_checks += 1
    
    details['position_variance'] = {
        'score': position_variance,
        'passed': stillness_check,
        'threshold': 0.002,
        'debug_info': f"Max variance among joints: {position_variance:.6f}"
    }
    
    # 5. Arm position analysis - arms should move (swing back or prepare)
    arm_scores = []
    arm_indicators = 0
    for shoulder in ['left_shoulder', 'right_shoulder']:
        if shoulder in angle_sequences:
            initial_arm = np.mean(angle_sequences[shoulder][:3])
            final_arm = np.mean(angle_sequences[shoulder][-3:])
            arm_movement = abs(final_arm - initial_arm)
            arm_scores.append(arm_movement)
            
            if arm_movement > 20:
                start_indicators += 1
                arm_indicators += 1
            total_checks += 1
    
    details['arm_movement'] = {
        'scores': arm_scores,
        'passed': arm_indicators > 0,
        'threshold': 20
    }
    
    # 6. Overall movement velocity - calculate joint position changes
    movement_velocity = 0
    frame_count = 0
    
    for i in range(1, len(pose_seq)):
        # Calculate total joint displacement for this frame
        total_displacement = 0
        joint_count = 0
        
        for joint in ['left_hip', 'right_hip', 'left_knee', 'right_knee']:
            if joint in pose_seq[i] and joint in pose_seq[i-1]:
                dx = pose_seq[i][joint][0] - pose_seq[i-1][joint][0]
                dy = pose_seq[i][joint][1] - pose_seq[i-1][joint][1]
                displacement = math.sqrt(dx*dx + dy*dy)
                total_displacement += displacement
                joint_count += 1
        
        if joint_count > 0:
            frame_displacement = total_displacement / joint_count
            movement_velocity += frame_displacement
            frame_count += 1
    
    avg_movement_velocity = movement_velocity / frame_count if frame_count > 0 else 0
    
    # More strict movement threshold to prevent false positives from standing still
    velocity_passed = avg_movement_velocity > 0.008  # Lowered from 0.02 to 0.008 based on test results
    if velocity_passed:
        start_indicators += 1
    total_checks += 1
    
    details['movement_velocity'] = {
        'score': avg_movement_velocity,
        'passed': velocity_passed,
        'threshold': 0.008,
        'debug_info': f"Avg velocity over {frame_count} frames: {avg_movement_velocity:.6f}"
    }
    
    # Calculate final confidence
    if total_checks > 0:
        confidence = start_indicators / total_checks
        
        # STRICTER REQUIREMENTS: Must pass multiple key swimming start indicators
        movement_required = details.get('movement_velocity', {}).get('passed', False)
        knee_flexion_required = details.get('knee_flexion', {}).get('passed', False)
        not_upright = not details.get('torso_lean', {}).get('is_upright', True)
        
        # For swimming starts, require proper body position (not standing upright)
        is_start = (confidence > 0.7 and  
                   movement_required and 
                   knee_flexion_required and
                   not_upright and
                   start_indicators >= 4)  # Must pass at least 4 out of 6 tests
        
        details['summary'] = {
            'indicators_passed': start_indicators,
            'total_checks': total_checks,
            'confidence': confidence,
            'movement_required': movement_required,
            'knee_flexion_required': knee_flexion_required,
            'not_upright': not_upright,
            'min_indicators_met': start_indicators >= 4,
            'is_start': is_start
        }
        
        if return_confidence:
            return is_start, confidence, details
        else:
            return is_start
    
    if return_confidence:
        return False, 0.0, {"error": "No valid checks completed"}
    return False

def analyze_start_phases(pose_seq):
    """
    Optional: Analyze different phases of the start for more detailed detection
    """
    if len(pose_seq) < 15:
        return None
    
    phases = {
        'preparation': pose_seq[:5],
        'crouch': pose_seq[5:10],
        'launch': pose_seq[10:]
    }
    
    phase_analysis = {}
    
    for phase_name, phase_poses in phases.items():
        angles_in_phase = []
        for pose in phase_poses:
            angles = extract_joint_angles(pose)
            if angles:
                angles_in_phase.append(angles)
        
        if angles_in_phase:
            # Calculate average angles for this phase
            avg_angles = {}
            for angle_name in angles_in_phase[0].keys():
                avg_angles[angle_name] = np.mean([pose[angle_name] for pose in angles_in_phase])
            
            phase_analysis[phase_name] = avg_angles
    
    return phase_analysis

# Usage examples:

# Simple boolean result:
# result = is_start_motion(your_pose_sequence)

# Get confidence and detailed breakdown:
# is_start, confidence, details = is_start_motion(your_pose_sequence, return_confidence=True)
# print(f"Is start: {is_start}")
# print(f"Confidence: {confidence:.2f}")
# print(f"Details: {details}")

# Example of printing detailed results:
def debug_pose_data(pose_seq):
    """
    Debug function to examine pose data structure and joint availability
    """
    print("=== Pose Data Debug ===")
    print(f"Sequence length: {len(pose_seq)}")
    
    if len(pose_seq) > 0:
        print(f"Available joints in first pose: {list(pose_seq[0].keys())}")
        print()
        
        # Check a few key joints
        sample_joints = ['left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle']
        
        for joint in sample_joints:
            if joint in pose_seq[0]:
                print(f"{joint}: {pose_seq[0][joint]}")
            else:
                print(f"{joint}: MISSING")
    
    print()

def print_start_analysis(pose_seq, debug=False):
    """
    Print detailed analysis of start detection
    """
    if debug:
        debug_pose_data(pose_seq)
    
    is_start, confidence, details = is_start_motion(pose_seq, return_confidence=True)
    
    print("=== Swimming Start Analysis ===")
    print(f"Result: {'START DETECTED' if is_start else 'NO START'}")
    print(f"Confidence: {confidence:.1%}")
    print()
    
    if 'error' in details:
        print(f"Error: {details['error']}")
        return is_start, confidence, details
    
    if 'summary' in details:
        summary = details['summary']
        print(f"Indicators passed: {summary['indicators_passed']}/{summary['total_checks']}")
        print(f"Movement required: {'✓' if summary.get('movement_required', False) else '✗'}")
        print(f"Knee flexion required: {'✓' if summary.get('knee_flexion_required', False) else '✗'}")
        print(f"Not standing upright: {'✓' if summary.get('not_upright', False) else '✗'}")
        print(f"Minimum indicators (4+): {'✓' if summary.get('min_indicators_met', False) else '✗'}")
        print()
    
    # Print individual test results
    test_results = [
        ('Knee Flexion', 'knee_flexion'),
        ('Hip Angle Change', 'hip_angle_change'), 
        ('Torso Lean', 'torso_lean'),
        ('Position Variance', 'position_variance'),
        ('Arm Movement', 'arm_movement'),
        ('Movement Velocity', 'movement_velocity')
    ]
    
    for test_name, key in test_results:
        if key in details:
            test_data = details[key]
            status = "✓ PASS" if test_data['passed'] else "✗ FAIL"
            
            if 'scores' in test_data:
                scores = test_data['scores']
                print(f"{test_name}: {status} (scores: {[f'{s:.1f}' for s in scores]}, threshold: {test_data['threshold']})")
            else:
                score = test_data['score']
                print(f"{test_name}: {status} (score: {score:.1f}, threshold: {test_data['threshold']})")
    
    return is_start, confidence, details

def is_start_motion_relaxed(pose_seq, return_confidence=False):
    """
    Version with much more relaxed thresholds for debugging
    """
    if len(pose_seq) < 10:  # Reduced from 15
        if return_confidence:
            return False, 0.0, {"error": "Sequence too short"}
        return False
    
    # Extract angles for all poses
    angle_sequences = {}
    valid_poses = []
    
    for pose in pose_seq:
        angles = extract_joint_angles(pose)
        if angles is not None:
            valid_poses.append(angles)
    
    if len(valid_poses) < 5:  # Reduced from 10
        if return_confidence:
            return False, 0.0, {"error": "Not enough valid poses"}
        return False
    
    # Convert to time series for each angle
    for angle_name in valid_poses[0].keys():
        angle_sequences[angle_name] = [pose[angle_name] for pose in valid_poses]
    
    # Analyze with MUCH more relaxed thresholds
    start_indicators = 0
    total_checks = 0
    details = {}
    
    # 1. Knee angle analysis - RELAXED threshold
    knee_scores = []
    for knee in ['left_knee', 'right_knee']:
        if knee in angle_sequences:
            initial_knee = np.mean(angle_sequences[knee][:3])
            min_knee = np.min(angle_sequences[knee])
            knee_flexion = initial_knee - min_knee
            
            knee_passed = knee_flexion > 5  # Was 20, now 5
            knee_scores.append(knee_flexion)
            
            if knee_passed:
                start_indicators += 1
            total_checks += 1
    
    details['knee_flexion'] = {
        'scores': knee_scores,
        'passed': len([s for s in knee_scores if s > 5]) > 0,
        'threshold': 5
    }
    
    # 2. Hip angle analysis - RELAXED
    hip_scores = []
    hip_indicators = 0
    for hip in ['left_hip_angle', 'right_hip_angle']:
        if hip in angle_sequences:
            initial_hip = np.mean(angle_sequences[hip][:3])
            final_hip = np.mean(angle_sequences[hip][-3:])
            hip_change = abs(final_hip - initial_hip)
            hip_scores.append(hip_change)
            
            if hip_change > 5:  # Was 15, now 5
                start_indicators += 1
                hip_indicators += 1
            total_checks += 1
    
    details['hip_angle_change'] = {
        'scores': hip_scores,
        'passed': hip_indicators > 0,
        'threshold': 5
    }
    
    # 3. Simple position change test (fallback)
    position_change = 0
    if len(pose_seq) > 1:
        for joint in ['left_hip', 'right_hip']:
            if joint in pose_seq[0] and joint in pose_seq[-1]:
                dx = pose_seq[-1][joint][0] - pose_seq[0][joint][0]
                dy = pose_seq[-1][joint][1] - pose_seq[0][joint][1]
                change = abs(dx) + abs(dy)
                position_change = max(position_change, change)
    
    if position_change > 0.02:  # Very low threshold
        start_indicators += 1
    total_checks += 1
    
    details['position_change'] = {
        'score': position_change,
        'passed': position_change > 0.02,
        'threshold': 0.02
    }
    
    # Calculate final confidence
    if total_checks > 0:
        confidence = start_indicators / total_checks
        is_start = confidence > 0.3  # Lower threshold (was 0.5)
        
        details['summary'] = {
            'indicators_passed': start_indicators,
            'total_checks': total_checks,
            'confidence': confidence,
            'is_start': is_start
        }
        
        if return_confidence:
            return is_start, confidence, details
        else:
            return is_start
    
    if return_confidence:
        return False, 0.0, {"error": "No valid checks completed"}
    return False

# Debugging usage:

# Step 1: Check your pose data structure
# print_start_analysis(your_pose_sequence, debug=True)

# Step 2: Try the relaxed version
# is_start, confidence, details = is_start_motion_relaxed(your_pose_sequence, return_confidence=True)
# print(f"Relaxed detection - Is start: {is_start}, Confidence: {confidence:.2f}")

# Step 3: Manual threshold testing
def test_with_custom_thresholds(pose_seq, knee_thresh=5, hip_thresh=5, pos_thresh=0.02):
    """
    Test detection with custom thresholds
    """
    print(f"\n=== Testing with custom thresholds ===")
    print(f"Knee: {knee_thresh}, Hip: {hip_thresh}, Position: {pos_thresh}")
    
    if len(pose_seq) < 10:
        print("Sequence too short")
        return False
    
    # Simple tests
    passed_tests = 0
    total_tests = 0
    
    # Test 1: Basic position change
    if len(pose_seq) > 1:
        for joint in ['left_hip', 'right_hip', 'left_knee', 'right_knee']:
            if joint in pose_seq[0] and joint in pose_seq[-1]:
                dx = pose_seq[-1][joint][0] - pose_seq[0][joint][0]
                dy = pose_seq[-1][joint][1] - pose_seq[0][joint][1]
                change = abs(dx) + abs(dy)
                
                print(f"{joint} total change: {change:.3f}")
                if change > pos_thresh:
                    passed_tests += 1
                total_tests += 1
    
    # Test 2: Height changes (crouch detection)
    if len(pose_seq) > 5:
        for joint in ['left_hip', 'right_hip']:
            if joint in pose_seq[0] and joint in pose_seq[len(pose_seq)//2]:
                y_start = pose_seq[0][joint][1]
                y_mid = pose_seq[len(pose_seq)//2][joint][1]
                y_change = abs(y_mid - y_start)
                
                print(f"{joint} height change: {y_change:.3f}")
                if y_change > pos_thresh:
                    passed_tests += 1
                total_tests += 1
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if total_tests > 0:
        confidence = passed_tests / total_tests
        is_start = confidence > 0.3
        print(f"Result: {is_start}, Confidence: {confidence:.2f}")
        return is_start
    
    return False
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
            
            # Debug: Print buffer length every 10 frames
            if len(pose_buffer) % 10 == 0:
                print(f"Pose buffer length: {len(pose_buffer)}")

            if is_start_motion(pose_buffer) and not feedback_displayed:
                print("Swimming start motion detected. Sending to Gemini.")
                
                # Add detailed start analysis for debugging
                print_start_analysis(pose_buffer, debug=True)
                
                feedback = generate_feedback(pose_buffer)
                print("Gemini Feedback:\n", feedback)
                feedback_displayed = True
                feedback_time = time.time()
                # Clear pose buffer after detecting start to prepare for next analysis
                pose_buffer = []
            
            # Fallback: Try simpler detection if main detection fails
            elif len(pose_buffer) >= 15 and not feedback_displayed:
                # Simple position-based detection
                start_pos = pose_buffer[0]
                end_pos = pose_buffer[-1]
                
                # Check for significant movement in key joints
                movement_detected = False
                for joint in ['left_hip', 'right_hip', 'left_knee', 'right_knee']:
                    if joint in start_pos and joint in end_pos:
                        dx = abs(end_pos[joint][0] - start_pos[joint][0])
                        dy = abs(end_pos[joint][1] - start_pos[joint][1])
                        total_movement = dx + dy
                        
                        if total_movement > 0.1:  # Threshold for movement
                            movement_detected = True
                            print(f"Simple detection: {joint} moved {total_movement:.3f}")
                            break
                
                if movement_detected:
                    print("Swimming start detected via simple movement detection!")
                    feedback = generate_feedback(pose_buffer)
                    print("Gemini Feedback:\n", feedback)
                    feedback_displayed = True
                    feedback_time = time.time()
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
