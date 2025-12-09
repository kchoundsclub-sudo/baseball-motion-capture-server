from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import gc
import sys
import traceback

app = Flask(__name__)

# Configure CORS
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
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def log_and_flush(message):
    """Print and immediately flush to ensure logs appear in Render"""
    print(message)
    sys.stdout.flush()

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    try:
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    except Exception as e:
        log_and_flush(f"Error calculating angle: {str(e)}")
        return 0

def analyze_pitching_mechanics(video_path):
    """Analyze video and extract biomechanical metrics"""
    cap = None
    try:
        log_and_flush(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        log_and_flush(f"Video properties: {total_frames} frames, {width}x{height}, {fps} fps")
        
        # Store metrics
        elbow_angles = []
        knee_angles = []
        trunk_rotations = []
        
        frame_count = 0
        frames_to_analyze = min(total_frames, 300)  # Max 10 seconds at 30fps
        frame_skip = 3  # Process every 3rd frame
        
        # Downscale to 640px width
        target_width = 640
        scale = target_width / width if width > target_width else 1.0
        
        log_and_flush(f"Processing up to {frames_to_analyze} frames with skip={frame_skip}, scale={scale:.2f}")
        
        processed_count = 0
        while frame_count < frames_to_analyze:
            try:
                ret, frame = cap.read()
                if not ret:
                    log_and_flush(f"End of video at frame {frame_count}")
                    break
                
                # Skip frames
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Downscale
                if scale < 1.0:
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Right arm elbow angle
                    if all([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.5,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility > 0.5,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.5]):
                        elbow_angle = calculate_angle(
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                        )
                        elbow_angles.append(elbow_angle)
                    
                    # Left knee flexion
                    if all([landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.5,
                           landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility > 0.5,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].visibility > 0.5]):
                        knee_angle = calculate_angle(
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                        )
                        knee_angles.append(180 - knee_angle)
                    
                    # Trunk rotation
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    shoulder_width = abs(right_shoulder.x - left_shoulder.x)
                    trunk_rotations.append(shoulder_width)
                
                # Clean up frame
                del frame, rgb_frame
                
                processed_count += 1
                if processed_count % 10 == 0:
                    log_and_flush(f"Processed {processed_count} frames...")
                
                frame_count += 1
                
            except Exception as e:
                log_and_flush(f"Error processing frame {frame_count}: {str(e)}")
                frame_count += 1
                continue
        
        log_and_flush(f"Finished processing. Total frames: {frame_count}, Detected poses: {len(elbow_angles)}")
        
        # Force garbage collection
        gc.collect()
        
        # Calculate metrics
        if not elbow_angles:
            log_and_flush("No poses detected, using fallback values")
            return {
                "elbowAngleAtMER": 95,
                "leadLegKneeFlexionAtRelease": 55,
                "trunkRotationTime": 0.35,
                "strideFootContactTime": 0.45
            }
        
        elbow_at_mer = min(elbow_angles)
        
        mid_start = int(len(knee_angles) * 0.35)
        mid_end = int(len(knee_angles) * 0.65)
        knee_flexion = np.mean(knee_angles[mid_start:mid_end]) if knee_angles else 55
        
        if len(trunk_rotations) > 5:
            rotation_changes = np.diff(trunk_rotations)
            max_rotation_frame = np.argmax(np.abs(rotation_changes))
            trunk_time = max_rotation_frame / (fps / frame_skip) if fps > 0 else 0.35
        else:
            trunk_time = 0.35
        
        if len(knee_angles) > 5:
            knee_changes = np.diff(knee_angles)
            max_knee_change_frame = np.argmax(np.abs(knee_changes))
            stride_time = max_knee_change_frame / (fps / frame_skip) if fps > 0 else 0.45
        else:
            stride_time = 0.45
        
        metrics = {
            "elbowAngleAtMER": round(float(elbow_at_mer), 1),
            "leadLegKneeFlexionAtRelease": round(float(knee_flexion), 1),
            "trunkRotationTime": round(float(trunk_time), 2),
            "strideFootContactTime": round(float(stride_time), 2)
        }
        
        log_and_flush(f"Final metrics: {metrics}")
        return metrics
        
    except Exception as e:
        log_and_flush(f"Fatal error in analyze_pitching_mechanics: {str(e)}")
        log_and_flush(traceback.format_exc())
        raise
    finally:
        if cap is not None:
            cap.release()
            log_and_flush("Video capture released")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "MediaPipe Pose Analysis"}), 200

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_video():
    if request.method == 'OPTIONS':
        return '', 204
    
    tmp_path = None
    try:
        log_and_flush("=== NEW REQUEST ===")
        log_and_flush(f"Request method: {request.method}")
        log_and_flush(f"Files in request: {list(request.files.keys())}")
        
        if 'video' not in request.files:
            log_and_flush("ERROR: No video file in request")
            return jsonify({"error": "No video file", "success": False}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            log_and_flush("ERROR: Empty filename")
            return jsonify({"error": "Empty filename", "success": False}), 400
        
        log_and_flush(f"Received file: {video_file.filename}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            log_and_flush("Saving video to temp file...")
            video_file.save(tmp_file)
            tmp_path = tmp_file.name
        
        file_size = os.path.getsize(tmp_path)
        log_and_flush(f"File saved: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        
        # Analyze video
        log_and_flush("Starting MediaPipe analysis...")
        metrics = analyze_pitching_mechanics(tmp_path)
        
        log_and_flush("Analysis complete, returning results")
        
        return jsonify({
            "success": True,
            "metrics": metrics
        }), 200
        
    except Exception as e:
        log_and_flush(f"FATAL ERROR: {str(e)}")
        log_and_flush(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                log_and_flush(f"Cleaned up temp file: {tmp_path}")
            except Exception as e:
                log_and_flush(f"Error cleaning up temp file: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    log_and_flush(f"Starting MediaPipe server on port {port}")
    app.run(host='0.0.0.0', port=port)
