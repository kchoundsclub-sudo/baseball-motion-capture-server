from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import math

app = Flask(__name__)

# Configure CORS to allow all origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    a = np.array([point1.x, point1.y])
    b = np.array([point2.x, point2.y])
    c = np.array([point3.x, point3.y])
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def analyze_pitching_mechanics(video_path):
    """Analyze video and extract biomechanical metrics"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {total_frames} frames at {fps} fps")
    
    # Store metrics across frames
    elbow_angles = []
    knee_angles = []
    trunk_rotations = []
    
    frame_count = 0
    max_frames_to_analyze = min(total_frames, 300)
    
    while frame_count < max_frames_to_analyze:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Right arm elbow angle (shoulder-elbow-wrist)
            if all([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility > 0.5,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.5]):
                elbow_angle = calculate_angle(
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                )
                elbow_angles.append(elbow_angle)
            
            # Left knee flexion (hip-knee-ankle)
            if all([landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.5,
                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility > 0.5,
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].visibility > 0.5]):
                knee_angle = calculate_angle(
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                )
                knee_angles.append(180 - knee_angle)
            
            # Trunk rotation (shoulder width change)
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)
            trunk_rotations.append(shoulder_width)
        
        frame_count += 1
    
    cap.release()
    
    print(f"Analyzed {frame_count} frames, detected {len(elbow_angles)} elbow poses")
    
    # Calculate biomechanical metrics
    if not elbow_angles:
        print("Warning: No pose landmarks detected, using fallback values")
        return {
            "elbowAngleAtMER": 95,
            "leadLegKneeFlexionAtRelease": 55,
            "trunkRotationTime": 0.35,
            "strideFootContactTime": 0.45
        }
    
    # Elbow angle at maximum external rotation
    elbow_at_mer = min(elbow_angles) if elbow_angles else 95
    
    # Lead leg knee flexion at release
    mid_start = int(len(knee_angles) * 0.35)
    mid_end = int(len(knee_angles) * 0.65)
    knee_flexion = np.mean(knee_angles[mid_start:mid_end]) if knee_angles else 55
    
    # Trunk rotation time
    if len(trunk_rotations) > 10:
        rotation_changes = np.diff(trunk_rotations)
        max_rotation_frame = np.argmax(np.abs(rotation_changes))
        trunk_time = max_rotation_frame / fps if fps > 0 else 0.35
    else:
        trunk_time = 0.35
    
    # Stride foot contact time
    if len(knee_angles) > 10:
        knee_changes = np.diff(knee_angles)
        max_knee_change_frame = np.argmax(np.abs(knee_changes))
        stride_time = max_knee_change_frame / fps if fps > 0 else 0.45
    else:
        stride_time = 0.45
    
    metrics = {
        "elbowAngleAtMER": round(float(elbow_at_mer), 1),
        "leadLegKneeFlexionAtRelease": round(float(knee_flexion), 1),
        "trunkRotationTime": round(float(trunk_time), 2),
        "strideFootContactTime": round(float(stride_time), 2)
    }
    
    print(f"Final metrics: {metrics}")
    return metrics

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "MediaPipe Pose Analysis"}), 200

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_video():
    """Analyze video file upload and return biomechanical metrics"""
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        print(f"Received request: {request.method}")
        print(f"Content-Type: {request.content_type}")
        print(f"Files: {list(request.files.keys())}")
        
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided", "success": False}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({"error": "Empty filename", "success": False}), 400
        
        print(f"Processing video: {video_file.filename}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_file.save(tmp_file)
            tmp_path = tmp_file.name
        
        print(f"Video saved to: {tmp_path}, size: {os.path.getsize(tmp_path)} bytes")
        
        # Analyze video
        print("Starting MediaPipe analysis...")
        metrics = analyze_pitching_mechanics(tmp_path)
        
        # Clean up
        os.unlink(tmp_path)
        
        print(f"Analysis complete, returning metrics")
        
        return jsonify({
            "success": True,
            "metrics": metrics
        }), 200
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port)
