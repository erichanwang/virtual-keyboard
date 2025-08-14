import cv2
import mediapipe as mp
import math
from pynput.keyboard import Controller, Key
import time

# mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# keyboard
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "backspace"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "Enter"],
    ["Caps Lock", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "Shift"],
    ["Space"]
]

# create keyboard
def draw_keyboard(img, key_positions):
    for row_keys in keys:
        for key in row_keys:
            x, y = key_positions[key]
            width = 60
            if key == "Space":
                width = 320
            elif key in ["backspace", "Enter", "Shift", "Caps Lock"]:
                width = 130
            
            cv2.rectangle(img, (x, y), (x + width, y + 60), (255, 0, 0), 2)
            cv2.putText(img, key, (x + 5, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

def get_key_positions(img_width):
    positions = {}
    y_offset = 200  # move down
    key_width = 60
    key_height = 60
    key_spacing = 10
    for i, row_keys in enumerate(keys):
        # Center each row
        row_width = sum(320 if k == "Space" else (130 if k in ["backspace", "Enter", "Shift", "Caps Lock"] else key_width) for k in row_keys) + (len(row_keys) - 1) * key_spacing
        x_offset = (img_width - row_width) // 2
        
        current_x = x_offset
        for key in row_keys:
            positions[key] = (current_x, i * (key_height + key_spacing) + y_offset)
            if key == "Space":
                current_x += 320 + key_spacing
            elif key in ["backspace", "Enter", "Shift", "Caps Lock"]:
                current_x += 130 + key_spacing
            else:
                current_x += key_width + key_spacing
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
    shift = False
    caps_lock = False
    hand_closed = False
    last_press_time = 0
    cooldown = 0.5  # 500ms cooldown

    keyboard = Controller()

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

                # fingertips
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                h, w, c = img.shape
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                # fingertip distance using distance formula
                distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)

                # print on screen
                cv2.putText(img, f"Pinch Distance: {int(distance)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # check if user is pinching
                current_time = time.time()
                if distance < 50:
                    if not hand_closed and (current_time - last_press_time > cooldown):
                        for key, (kx, ky) in key_positions.items():
                            key_height = 60
                            key_width = 60
                            if key == "Space":
                                key_width = 320
                            elif key in ["backspace", "Enter", "Shift", "Caps Lock"]:
                                key_width = 130

                            if kx < index_x < kx + key_width and ky < index_y < ky + key_height:
                                try:
                                    if key == "backspace":
                                        text = text[:-1]
                                        keyboard.press(Key.backspace)
                                        keyboard.release(Key.backspace)
                                    elif key == "Enter":
                                        text += "\n"
                                        keyboard.press(Key.enter)
                                        keyboard.release(Key.enter)
                                    elif key == "Space":
                                        text += " "
                                        keyboard.press(Key.space)
                                        keyboard.release(Key.space)
                                    elif key == "Shift":
                                        shift = not shift
                                    elif key == "Caps Lock":
                                        caps_lock = not caps_lock
                                    else:
                                        if shift or caps_lock:
                                            text += key.upper()
                                            keyboard.press(key.upper())
                                            keyboard.release(key.upper())
                                            shift = False
                                        else:
                                            text += key.lower()
                                            keyboard.press(key.lower())
                                            keyboard.release(key.lower())
                                    cv2.rectangle(img, (kx, ky), (kx + key_width, ky + key_height), (0, 255, 0), -1)
                                    cv2.putText(img, key, (kx + 5, ky + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                                    hand_closed = True
                                    last_press_time = current_time
                                except Exception as e:
                                    print(f"Error pressing key: {e}")
                else:
                    hand_closed = False
        
        # show the typed text
        cv2.rectangle(img, (50, 550), (1230, 650), (0, 0, 0), -1)
        cv2.putText(img, text, (60, 620), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        cv2.imshow("Virtual Keyboard", img)

        key = cv2.waitKey(1) & 0xFF

        # close window
        try:
            if cv2.getWindowProperty('Virtual Keyboard', cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
