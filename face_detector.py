import cv2
import mediapipe as mp
import numpy as np
import os
import json
from math import hypot

# --- Configuration ---
FACE_INFO_DIR = 'face_info'
RECOGNITION_THRESHOLD = 0.06  # Slightly increased for more flexibility
FACE_LOST_THRESHOLD = 10      # Frames before resetting tracker
COLLECTION_FRAMES_TARGET = 50 # Number of samples to collect per 's' press

# --- Initialization ---
if not os.path.exists(FACE_INFO_DIR):
    os.makedirs(FACE_INFO_DIR)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

video_capture = cv2.VideoCapture(0)

# --- Core Functions (largely unchanged) ---
def get_facial_metrics(landmarks):
    """Calculates a dictionary of metrics from facial landmarks."""
    metrics = {}
    
    # Using normalized coordinates is more robust to camera distance changes
    def get_norm_dist(p1_idx, p2_idx):
        p1 = landmarks.landmark[p1_idx]
        p2 = landmarks.landmark[p2_idx]
        return hypot(p1.x - p2.x, p1.y - p2.y)

    # Eye Aspect Ratio (EAR)
    v_dist_l = get_norm_dist(386, 374)
    v_dist_r = get_norm_dist(159, 145)
    h_dist_l = get_norm_dist(362, 263)
    h_dist_r = get_norm_dist(133, 33)
    metrics['left_ear'] = v_dist_l / h_dist_l if h_dist_l != 0 else 0
    metrics['right_ear'] = v_dist_r / h_dist_r if h_dist_r != 0 else 0

    # Mouth Aspect Ratio (MAR)
    v_dist_m = get_norm_dist(13, 14)
    h_dist_m = get_norm_dist(61, 291)
    metrics['mar'] = v_dist_m / h_dist_m if h_dist_m != 0 else 0

    # Other distances
    metrics['eyebrow_eye_dist_l'] = get_norm_dist(285, 386)
    metrics['eyebrow_eye_dist_r'] = get_norm_dist(55, 159)
    metrics['nose_to_chin_dist'] = get_norm_dist(1, 152)
    
    return metrics

def load_known_faces(directory):
    """Loads all known face metrics from the specified directory."""
    known_faces = []
    if not os.path.exists(directory): return known_faces
    for name in os.listdir(directory):
        person_dir = os.path.join(directory, name)
        if os.path.isdir(person_dir):
            person_metrics = []
            for filename in os.listdir(person_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(person_dir, filename)
                    with open(filepath, 'r') as f:
                        person_metrics.append(list(json.load(f).values()))
            if person_metrics:
                # Average the metrics for a more stable profile
                avg_metrics = np.mean(np.array(person_metrics), axis=0)
                known_faces.append({'name': name, 'metrics_vector': avg_metrics})
    print(f"Loaded and averaged data for {len(known_faces)} known faces.")
    return known_faces

def find_identity(current_metrics_vector, known_faces, threshold):
    """Compares current face metrics to known faces and returns the name of the best match."""
    if not known_faces: return "Unknown"
    best_match_name = "Unknown"
    min_diff = float('inf')

    for face in known_faces:
        diff = np.linalg.norm(current_metrics_vector - face['metrics_vector'])
        if diff < min_diff:
            min_diff = diff
            if diff < threshold:
                best_match_name = face['name']
    return best_match_name

# --- Main Application Logic ---
known_faces_db = load_known_faces(FACE_INFO_DIR)

# State variables for tracking and data collection
tracked_identity = None
face_lost_counter = 0
is_collecting_data = False
collection_name = ""
collection_frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    display_text = ""

    if results.multi_face_landmarks:
        face_lost_counter = 0 # Reset counter since we see a face
        face_landmarks = results.multi_face_landmarks[0]
        current_metrics = get_facial_metrics(face_landmarks)
        current_metrics_vector = np.array(list(current_metrics.values()))

        # --- Data Collection Mode ---
        if is_collecting_data:
            collection_frame_count += 1
            
            # Save the metrics
            person_dir = os.path.join(FACE_INFO_DIR, collection_name)
            count = 1
            while os.path.exists(os.path.join(person_dir, f"{count}.json")): count += 1
            metrics_path = os.path.join(person_dir, f"{count}.json")
            with open(metrics_path, 'w') as f: json.dump(current_metrics, f, indent=4)

            # Update display and check if done
            progress = int((collection_frame_count / COLLECTION_FRAMES_TARGET) * 100)
            display_text = f"Collecting for {collection_name}: {progress}%"
            if collection_frame_count >= COLLECTION_FRAMES_TARGET:
                print(f"Finished collecting data for {collection_name}.")
                is_collecting_data = False
                known_faces_db = load_known_faces(FACE_INFO_DIR) # Reload DB with new data
                tracked_identity = collection_name # Lock on to the new person

        # --- Recognition Mode ---
        else:
            if tracked_identity is None: # If not tracking anyone, try to identify
                tracked_identity = find_identity(current_metrics_vector, known_faces_db, RECOGNITION_THRESHOLD)
            display_text = f"Identity: {tracked_identity}"

        # Draw mesh and display text
        mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if not is_collecting_data:
            cv2.putText(frame, "Press 's' to save/update face.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else: # No face detected
        face_lost_counter += 1
        if face_lost_counter > FACE_LOST_THRESHOLD:
            tracked_identity = None # Reset tracker
            if is_collecting_data: # Cancel collection if face is lost
                print("Face lost. Cancelling data collection.")
                is_collecting_data = False
                known_faces_db = load_known_faces(FACE_INFO_DIR)

    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and results.multi_face_landmarks and not is_collecting_data:
        name = input("Enter name to save/update: ")
        if name:
            collection_name = name
            person_dir = os.path.join(FACE_INFO_DIR, collection_name)
            if not os.path.exists(person_dir): os.makedirs(person_dir)
            is_collecting_data = True
            collection_frame_count = 0
            print(f"Starting data collection for {name}. Please turn your head slowly.")

    if key == ord('q'): break

video_capture.release()
cv2.destroyAllWindows()
