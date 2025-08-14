import cv2
import mediapipe as mp
import numpy as np
import math
from pynput.keyboard import Controller, Key

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils
keyboard = Controller()

# Keyboard layout
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "<-"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "Enter"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["Space"]
]

def draw_keyboard(img, key_positions):
    for row_keys in keys:
        for key in row_keys:
            x, y = key_positions[key]
            if key == "Space":
                cv2.rectangle(img, (x, y), (x + 400, y + 80), (255, 0, 0), 2)
            else:
                cv2.rectangle(img, (x, y), (x + 80, y + 80), (255, 0, 0), 2)
            cv2.putText(img, key, (x + 10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return img

def get_key_positions(img_width):
    positions = {}
    for i, row_keys in enumerate(keys):
        for j, key in enumerate(row_keys):
            if key == "Space":
                positions[key] = (j * 100 + 400, i * 100 + 100)
            else:
                positions[key] = (j * 100 + 50, i * 100 + 100)
    return positions

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(3, 1280)
    cap.set(4, 720)

    key_positions = get_key_positions(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    text = ""

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        img = draw_keyboard(img, key_positions)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of thumb and index finger tips
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                h, w, c = img.shape
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                # Calculate distance between thumb and index finger
                distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)

                # Print finger angles/status
                cv2.putText(img, f"Pinch Distance: {int(distance)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Check for pinch gesture
                if distance < 50:
                    for key, (kx, ky) in key_positions.items():
                        key_width = 400 if key == "Space" else 80
                        if kx < index_x < kx + key_width and ky < index_y < ky + 80:
                            if key == "<-":
                                keyboard.press(Key.backspace)
                                keyboard.release(Key.backspace)
                            elif key == "Enter":
                                keyboard.press(Key.enter)
                                keyboard.release(Key.enter)
                            elif key == "Space":
                                keyboard.press(Key.space)
                                keyboard.release(Key.space)
                            else:
                                keyboard.press(key)
                                keyboard.release(key)
                            cv2.rectangle(img, (kx, ky), (kx + key_width, ky + 80), (0, 255, 0), -1)
                            cv2.putText(img, key, (kx + 10, ky + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        cv2.imshow("Virtual Keyboard", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
