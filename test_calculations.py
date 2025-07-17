import numpy as np
import math

# Test the position variance and movement velocity calculations
def test_movement_calculations():
    print("Testing movement calculations...")
    
    # Create sample pose sequence with actual movement
    pose_seq = []
    for i in range(20):
        # Simulate movement - hip moving down and forward over time
        t = i / 19.0  # Progress from 0 to 1
        pose = {
            'left_hip': (0.3 + t * 0.1, 0.5 + t * 0.1),    # Moving right and down
            'right_hip': (0.7 + t * 0.1, 0.5 + t * 0.1),   # Moving right and down  
            'left_knee': (0.3 + t * 0.05, 0.7 + t * 0.15), # Moving down more
            'right_knee': (0.7 + t * 0.05, 0.7 + t * 0.15) # Moving down more
        }
        pose_seq.append(pose)
    
    # Test position variance calculation
    position_variance = 0
    key_joints = ['left_hip', 'right_hip', 'left_knee', 'right_knee']
    
    for joint in key_joints:
        if joint in pose_seq[0] and joint in pose_seq[-1]:
            x_positions = [pose[joint][0] for pose in pose_seq if joint in pose]
            y_positions = [pose[joint][1] for pose in pose_seq if joint in pose]
            
            if len(x_positions) > 5:
                x_var = np.var(x_positions)
                y_var = np.var(y_positions)
                joint_variance = x_var + y_var
                position_variance = max(position_variance, joint_variance)
                print(f"Joint {joint}: x_var={x_var:.6f}, y_var={y_var:.6f}, total_var={joint_variance:.6f}")
    
    print(f"Final position variance: {position_variance:.6f}")
    print(f"Position variance > 0.01: {position_variance > 0.01}")
    print()
    
    # Test movement velocity calculation
    movement_velocity = 0
    frame_count = 0
    
    for i in range(1, len(pose_seq)):
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
    print(f"Average movement velocity: {avg_movement_velocity:.6f}")
    print(f"Movement velocity > 0.02: {avg_movement_velocity > 0.02}")

if __name__ == "__main__":
    test_movement_calculations()
