import cv2
import mediapipe as mp
import time
import math
import winsound

# initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# eye landmarks indices
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def calculate_distance(p1, p2):
    # calculate euclidean distance
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_eye_aspect_ratio(landmarks, eye_indices):
    # calculate eye aspect ratio
    # vertical landmarks
    p2 = landmarks[eye_indices[12]] # top
    p6 = landmarks[eye_indices[4]]  # bottom
    # horizontal landmarks
    p1 = landmarks[eye_indices[0]]  # left
    p4 = landmarks[eye_indices[8]]  # right

    vertical_dist = calculate_distance(p2, p6)
    horizontal_dist = calculate_distance(p1, p4)

    if horizontal_dist == 0:
        return 0.0

    ear = vertical_dist / horizontal_dist
    return ear

# webcam setup
cap = cv2.VideoCapture(0)

# drowsiness detection variables
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 20 # consecutive frames for drowsiness
COUNTER = 0
ALARM_ON = False
last_wink_time = 0
wink_counter = 0
wink_threshold = 2 # winks to trigger
wink_time_window = 2 # seconds

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("ignoring empty camera frame.")
        continue

    # flip image and convert from bgr to rgb
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    # draw face mesh annotations
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_ear = get_eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
            right_ear = get_eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)

            ear = (left_ear + right_ear) / 2.0

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        # play a beep sound
                        winsound.Beep(440, 1000) # frequency, duration
                        cv2.putText(image, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False

            cv2.putText(image, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # draw landmarks for eyes for visualization
            for index in LEFT_EYE_INDICES:
                x = int(landmarks[index].x * image.shape[1])
                y = int(landmarks[index].y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            for index in RIGHT_EYE_INDICES:
                x = int(landmarks[index].x * image.shape[1])
                y = int(landmarks[index].y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)


    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
